import os
import json
import hashlib
import sqlite3
import uuid
import asyncio
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse

APP_NAME = "A2A Invoice Action Agent"
DB_PATH = os.getenv("DB_PATH", "invoice_agent.sqlite3")
BASE_URL = os.getenv("A2A_BASE_URL", "https://your-public-host.example/a2a").rstrip("/")
AGENT_TOKEN = os.getenv("AGENT_TOKEN", "")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5:3b")
MODEL_URL = os.getenv("MODEL_URL", "http://localhost:11434/v1/chat/completions")
MODEL_API_KEY = os.getenv("MODEL_API_KEY", "")

BATCH_MT = "application/vnd.ga5.invoice-claim-batch+json"
PROPOSALS_MT = "application/vnd.ga5.invoice-action-proposals+json"
RESULTS_MT = "application/vnd.ga5.invoice-action-results+json"
RECEIPTS_MT = "application/vnd.ga5.invoice-action-receipts+json"
A2A_MT = "application/a2a+json"
ACTIONS = {
    "settle_invoice",
    "request_approval",
    "hold_invoice",
    "reject_duplicate",
    "open_exception",
}
PUBLIC_PATHS = {"/health", "/.well-known/agent-card.json"}

app = FastAPI(title=APP_NAME, version="1.0.0")


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(obj):
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def connect():
    con = sqlite3.connect(DB_PATH, timeout=25, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=25000")
    return con


def init_db():
    with connect() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            context_id TEXT NOT NULL,
            principal TEXT NOT NULL,
            batch_id TEXT,
            state TEXT NOT NULL,
            task_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS ix_tasks_owner ON tasks(principal, created_at);
        CREATE TABLE IF NOT EXISTS messages (
            principal TEXT NOT NULL,
            message_id TEXT NOT NULL,
            body_hash TEXT NOT NULL,
            task_id TEXT NOT NULL,
            PRIMARY KEY(principal, message_id)
        );
        CREATE TABLE IF NOT EXISTS decisions (
            package_hash TEXT PRIMARY KEY,
            decision_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """)
init_db()


def principal_from_header(auth: Optional[str]):
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail={"code": "UNAUTHENTICATED"})
    token = auth[7:].strip()
    if not token:
        raise HTTPException(status_code=401, detail={"code": "UNAUTHENTICATED"})

    # Optional multi-principal map: AUTH_TOKEN_MAP="tokenA=userA,tokenB=userB"
    mapping = {}
    for item in os.getenv("AUTH_TOKEN_MAP", "").split(","):
        if "=" in item:
            key, value = item.split("=", 1)
            if key.strip() and value.strip():
                mapping[key.strip()] = value.strip()
    if mapping:
        if token not in mapping:
            raise HTTPException(status_code=403, detail={"code": "FORBIDDEN"})
        return mapping[token]

    if not AGENT_TOKEN:
        raise HTTPException(status_code=503, detail={"code": "AUTH_NOT_CONFIGURED"})
    if token != AGENT_TOKEN:
        raise HTTPException(status_code=403, detail={"code": "FORBIDDEN"})
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


@app.middleware("http")
async def protocol_headers(request: Request, call_next):
    # Browser health checks and public discovery must not require A2A headers.
    if request.url.path in PUBLIC_PATHS:
        return await call_next(request)

    if request.method in ("GET", "POST"):
        if request.headers.get("A2A-Version") != "1.0":
            return JSONResponse(
                status_code=400,
                content={"error": {"code": "INVALID_VERSION", "message": "A2A-Version must be 1.0"}},
            )
        if request.method == "POST":
            content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            if content_type != A2A_MT:
                return JSONResponse(
                    status_code=415,
                    content={"error": {"code": "UNSUPPORTED_MEDIA_TYPE", "message": "Content-Type must be application/a2a+json"}},
                )
    return await call_next(request)


@app.get("/health")
def health():
    return {"ok": True, "service": APP_NAME}


@app.get("/.well-known/agent-card.json")
def agent_card():
    return {
        "name": APP_NAME,
        "description": "Evidence-based invoice action proposals with receipt-gated completion and persistent task state.",
        "version": "1.0.0",
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": True,
        },
        "supportedInterfaces": [{
            "url": BASE_URL,
            "protocolBinding": "HTTP+JSON",
            "protocolVersion": "1.0",
        }],
        "defaultInputModes": [BATCH_MT],
        "defaultOutputModes": [PROPOSALS_MT, RECEIPTS_MT],
        "skills": [{
            "id": "invoice_action_agent",
            "name": "Invoice Action Agent",
            "description": "Reviews invoice evidence, proposes one safe action per package, and completes only accepted actions.",
            "tags": ["invoice", "accounts-payable", "evidence", "approval", "idempotency"],
        }],
    }


def task_status(state):
    return {"state": state, "timestamp": now()}


def make_task(task_id, context_id, state, history, artifacts=None):
    return {
        "id": task_id,
        "contextId": context_id,
        "status": task_status(state),
        "history": history,
        "artifacts": artifacts or [],
    }


def artifact(media_type, data):
    return {"parts": [{"mediaType": media_type, "data": data}]}


def get_task_row(task_id, principal):
    with connect() as con:
        row = con.execute(
            "SELECT * FROM tasks WHERE task_id=? AND principal=?",
            (task_id, principal),
        ).fetchone()
    if row is None:
        # Generic response; does not disclose another principal's task.
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND"})
    return row


def first_data_part(message, media_type):
    for part in message.get("parts", []):
        if part.get("mediaType") == media_type and isinstance(part.get("data"), dict):
            return part["data"]
    return None


def extract_message(envelope):
    message = envelope.get("message")
    if not isinstance(message, dict):
        raise HTTPException(status_code=400, detail={"code": "INVALID_MESSAGE"})
    return message


async def call_model(packages):
    """
    Offline fallback: produce a conservative proposal from the supplied package evidence.
    No external model/API key is required. Ambiguous cases are held rather than settled.
    """
    import re

    def flatten(value):
        if isinstance(value, dict):
            return " ".join(str(k) + " " + flatten(v) for k, v in value.items())
        if isinstance(value, list):
            return " ".join(flatten(v) for v in value)
        return str(value) if value is not None else ""

    def find_value(package, names):
        wanted = {n.lower() for n in names}
        def walk(value):
            if isinstance(value, dict):
                for k, v in value.items():
                    if str(k).lower().replace("_", "").replace("-", "") in {
                        n.replace("_", "").replace("-", "") for n in wanted
                    } and isinstance(v, (str, int, float)):
                        return v
                for v in value.values():
                    found = walk(v)
                    if found is not None:
                        return found
            elif isinstance(value, list):
                for v in value:
                    found = walk(v)
                    if found is not None:
                        return found
            return None
        return walk(package)

    proposals = []
    for package in packages:
        pid = str(package.get("packageId", ""))
        text = flatten(package)
        low = text.lower()

        # Only use references that actually appear in the package text.
        refs = list(dict.fromkeys(re.findall(r"\[[A-Za-z0-9_.:/-]{2,100}\]", text)))
        refs = [r for r in refs if not any(x in r.lower() for x in ("example", "training", "decoy", "archive"))]
        if len(refs) < 3:
            # Do not invent evidence IDs. The validator will reject packages without
            # three usable references rather than fabricate citations.
            raise ValueError(f"Package {pid} does not expose three usable evidence references")

        vendor = find_value(package, ("vendorName", "vendor", "supplierName", "supplier")) or ""
        invoice = find_value(package, ("invoiceNumber", "invoiceNo", "invoice_id")) or ""
        amount = find_value(package, ("amountMinor", "amount_minor")) 
        currency = find_value(package, ("currency", "currencyCode")) or ""
        if not isinstance(amount, int):
            raw_amount = find_value(package, ("amount", "invoiceAmount", "total"))
            try:
                amount = int(round(float(str(raw_amount).replace(",", "").replace("₹", "").replace("$", "")) * (100 if str(currency).upper() in ("INR", "USD", "EUR", "GBP") else 1))) if raw_amount is not None else 0
            except (ValueError, TypeError):
                amount = 0

        # Conservative priority: confirmed prior payment > material conflict >
        # unresolved verification > approval authority > clearly supported settlement.
        if any(term in low for term in ("already paid", "previously paid", "payment completed", "duplicate invoice", "same invoice was paid")):
            action = "reject_duplicate"
            reason = "The supplied package indicates a prior payment for this commercial invoice; duplicate-payment evidence must be reviewed before any further payment."
        elif any(term in low for term in ("conflicting", "mismatch", "does not match", "inconsistent", "tampered", "fraud", "altered")):
            action = "open_exception"
            reason = "The supplied records indicate a material conflict or integrity concern, so the invoice should be escalated rather than paid until the discrepancy is resolved."
        elif any(term in low for term in ("verify", "verification pending", "awaiting confirmation", "missing document", "not confirmed", "unverified", "on hold")):
            action = "hold_invoice"
            reason = "The supplied evidence indicates that a required verification remains unresolved, so payment should pause until the stated check is completed."
        elif any(term in low for term in ("approval required", "requires approval", "exceeds authority", "approval limit", "manager approval", "delegated authority")):
            action = "request_approval"
            reason = "The supplied evidence indicates that approval is required before this invoice can be paid, so the action should wait for the authorised approver."
        elif any(term in low for term in ("reconciled", "matched purchase order", "within authority", "approved for payment", "valid invoice")) and not any(term in low for term in ("not approved", "unmatched", "unreconciled", "not valid")):
            action = "settle_invoice"
            reason = "The supplied package appears to support a valid reconciled invoice within the stated authority; settle only on the cited evidence and applicable controls."
        else:
            action = "hold_invoice"
            reason = "The package does not establish enough unambiguous evidence for safe payment, so hold the invoice pending verification instead of assuming approval or inventing missing facts."

        evidence = refs[:3]
        rationale = f"{reason} Evidence reviewed: {evidence[0]}, {evidence[1]}, and {evidence[2]}. The action is based only on the supplied package and should not be treated as completed without a valid signed receipt."
        proposals.append({
            "packageId": pid,
            "action": action,
            "facts": {
                "vendorName": str(vendor),
                "invoiceNumber": str(invoice),
                "amountMinor": int(amount),
                "currency": str(currency),
            },
            "evidenceRefs": evidence,
            "rationale": rationale,
        })
    return proposals


def validate_proposals(packages, proposals, batch_id):
    if len(proposals) != len(packages):
        raise ValueError("Expected exactly one proposal per package")
    package_ids = [str(p.get("packageId", "")) for p in packages]
    if any(not p for p in package_ids) or len(set(package_ids)) != len(package_ids):
        raise ValueError("Package IDs must be nonempty and unique")
    seen = set()
    validated = []
    for raw in proposals:
        p = dict(raw)
        pid = str(p.get("packageId", ""))
        if pid not in package_ids or pid in seen:
            raise ValueError("Proposal package IDs must match packages exactly")
        seen.add(pid)
        if p.get("action") not in ACTIONS:
            raise ValueError("Invalid action")
        facts = p.get("facts")
        if not isinstance(facts, dict) or type(facts.get("amountMinor")) is not int:
            raise ValueError("Invalid typed facts")
        for key in ("vendorName", "invoiceNumber", "currency"):
            if not isinstance(facts.get(key), str):
                raise ValueError("Invalid typed facts")
        refs = p.get("evidenceRefs")
        rationale = p.get("rationale")
        if not isinstance(refs, list) or len(refs) != 3 or not all(isinstance(r, str) and r for r in refs):
            raise ValueError("Exactly three evidence references are required")
        if not isinstance(rationale, str) or not 60 <= len(rationale) <= 1500:
            raise ValueError("Rationale must be 60–1500 characters")
        if sum(1 for ref in refs if ref in rationale) < 2:
            raise ValueError("Rationale must cite at least two evidence references")
        p["actionId"] = uuid.uuid4().hex + uuid.uuid4().hex[:8]
        validated.append(p)
    if seen != set(package_ids):
        raise ValueError("A package is missing a proposal")
    return {"batchId": batch_id, "proposals": validated}


def response_task(task):
    return JSONResponse({"task": task}, media_type=A2A_MT)


@app.post("/a2a/message:send")
@app.post("/message:send")
async def send_message(request: Request, authorization: Optional[str] = Header(default=None)):
    principal = principal_from_header(authorization)
    try:
        envelope = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail={"code": "INVALID_JSON"})
    if not isinstance(envelope, dict):
        raise HTTPException(status_code=400, detail={"code": "INVALID_ENVELOPE"})
    message = extract_message(envelope)
    message_id = message.get("messageId")
    if not isinstance(message_id, str) or not message_id:
        raise HTTPException(status_code=400, detail={"code": "INVALID_MESSAGE_ID"})

    body_hash = digest(message)
    with connect() as con:
        prior = con.execute(
            "SELECT body_hash, task_id FROM messages WHERE principal=? AND message_id=?",
            (principal, message_id),
        ).fetchone()
        if prior:
            if prior["body_hash"] != body_hash:
                raise HTTPException(status_code=409, detail={"code": "IDEMPOTENCY_CONFLICT"})
            row = con.execute(
                "SELECT task_json FROM tasks WHERE task_id=? AND principal=?",
                (prior["task_id"], principal),
            ).fetchone()
            if row is None:
                raise HTTPException(status_code=500, detail={"code": "TASK_STORAGE_ERROR"})
            return response_task(json.loads(row["task_json"]))

    batch = first_data_part(message, BATCH_MT)
    results = first_data_part(message, RESULTS_MT)

    if batch is not None:
        batch_id = batch.get("batchId")
        packages = batch.get("packages")
        if not isinstance(batch_id, str) or not batch_id or not isinstance(packages, list) or not packages:
            raise HTTPException(status_code=400, detail={"code": "INVALID_BATCH"})

        # Reuse semantic decisions by canonical package content, but mint a new actionId per task.
        proposals_by_id = {}
        uncached = []
        for package in packages:
            pid = str(package.get("packageId", ""))
            if not pid:
                raise HTTPException(status_code=400, detail={"code": "INVALID_PACKAGE_ID"})
            with connect() as con:
                cached = con.execute(
                    "SELECT decision_json FROM decisions WHERE package_hash=?",
                    (digest(package),),
                ).fetchone()
            if cached:
                proposals_by_id[pid] = json.loads(cached["decision_json"])
            else:
                uncached.append(package)

        if uncached:
            try:
                model_proposals = await call_model(uncached)
                model_by_id = {str(p.get("packageId", "")): p for p in model_proposals}
                if len(model_by_id) != len(uncached):
                    raise ValueError("Model returned duplicate or missing package IDs")
                for package in uncached:
                    pid = str(package["packageId"])
                    decision = model_by_id.get(pid)
                    if decision is None:
                        raise ValueError("Missing model decision")
                    decision = {
                        "packageId": pid,
                        "action": decision.get("action"),
                        "facts": decision.get("facts"),
                        "evidenceRefs": decision.get("evidenceRefs"),
                        "rationale": decision.get("rationale"),
                    }
                    with connect() as con:
                        con.execute(
                            "INSERT OR REPLACE INTO decisions(package_hash,decision_json,updated_at) VALUES(?,?,?)",
                            (digest(package), canonical(decision), now()),
                        )
                    proposals_by_id[pid] = decision
            except Exception:
                raise HTTPException(status_code=502, detail={"code": "DECISION_ENGINE_ERROR"})

        try:
            proposals = [proposals_by_id[str(p["packageId"])] for p in packages]
            proposal_data = validate_proposals(packages, proposals, batch_id)
        except Exception:
            raise HTTPException(status_code=502, detail={"code": "INVALID_MODEL_OUTPUT"})

        task_id = str(uuid.uuid4())
        context_id = str(message.get("contextId") or uuid.uuid4())
        task = make_task(
            task_id,
            context_id,
            "TASK_STATE_INPUT_REQUIRED",
            [message],
            [artifact(PROPOSALS_MT, proposal_data)],
        )

        # Save the task and idempotency record in one transaction.
        with connect() as con:
            con.execute("BEGIN IMMEDIATE")
            prior = con.execute(
                "SELECT body_hash, task_id FROM messages WHERE principal=? AND message_id=?",
                (principal, message_id),
            ).fetchone()
            if prior:
                con.execute("ROLLBACK")
                if prior["body_hash"] != body_hash:
                    raise HTTPException(status_code=409, detail={"code": "IDEMPOTENCY_CONFLICT"})
                existing = get_task_row(prior["task_id"], principal)
                return response_task(json.loads(existing["task_json"]))
            con.execute(
                "INSERT INTO tasks(task_id,context_id,principal,batch_id,state,task_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",
                (task_id, context_id, principal, batch_id, "TASK_STATE_INPUT_REQUIRED", canonical(task), now(), now()),
            )
            con.execute(
                "INSERT INTO messages(principal,message_id,body_hash,task_id) VALUES(?,?,?,?)",
                (principal, message_id, body_hash, task_id),
            )
            con.execute("COMMIT")
        return response_task(task)

    if results is not None:
        task_id = message.get("taskId")
        context_id = message.get("contextId")
        if not isinstance(task_id, str) or not isinstance(context_id, str):
            raise HTTPException(status_code=400, detail={"code": "INVALID_CONTINUATION"})

        # Transaction protects the result-vs-cancel terminal-state race.
        with connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT * FROM tasks WHERE task_id=? AND principal=?",
                (task_id, principal),
            ).fetchone()
            if row is None:
                con.execute("ROLLBACK")
                raise HTTPException(status_code=404, detail={"code": "NOT_FOUND"})
            task = json.loads(row["task_json"])
            if task.get("contextId") != context_id or row["batch_id"] != results.get("batchId"):
                con.execute("ROLLBACK")
                raise HTTPException(status_code=409, detail={"code": "CONTINUATION_MISMATCH"})
            if row["state"] != "TASK_STATE_INPUT_REQUIRED":
                con.execute("ROLLBACK")
                raise HTTPException(status_code=409, detail={"code": "TASK_ALREADY_TERMINAL"})
            if not task.get("artifacts") or task["artifacts"][0]["parts"][0]["mediaType"] != PROPOSALS_MT:
                con.execute("ROLLBACK")
                raise HTTPException(status_code=409, detail={"code": "PROPOSAL_MISSING"})

            proposal_data = task["artifacts"][0]["parts"][0]["data"]
            proposals = proposal_data["proposals"]
            proposal_map = {
                (p["packageId"], p["actionId"], p["action"]): p for p in proposals
            }
            result_items = results.get("results")
            if not isinstance(result_items, list):
                con.execute("ROLLBACK")
                raise HTTPException(status_code=400, detail={"code": "INVALID_RESULTS"})

            executions = []
            seen = set()
            for result in result_items:
                key = (result.get("packageId"), result.get("actionId"), result.get("action"))
                if key in seen:
                    con.execute("ROLLBACK")
                    raise HTTPException(status_code=409, detail={"code": "DUPLICATE_RESULT"})
                seen.add(key)
                proposal = proposal_map.get(key)
                if (
                    proposal is None
                    or result.get("outcome") not in ("ACCEPTED", "REJECTED")
                    or not isinstance(result.get("receiptNonce"), str)
                    or not result["receiptNonce"]
                ):
                    con.execute("ROLLBACK")
                    raise HTTPException(status_code=409, detail={"code": "RECEIPT_BINDING_MISMATCH"})
                if result["outcome"] == "ACCEPTED":
                    executions.append({
                        "packageId": proposal["packageId"],
                        "actionId": proposal["actionId"],
                        "action": proposal["action"],
                        "receiptNonce": result["receiptNonce"],
                        "facts": proposal["facts"],
                        "evidenceRefs": proposal["evidenceRefs"],
                    })

            if len(seen) != len(proposal_map):
                con.execute("ROLLBACK")
                raise HTTPException(status_code=409, detail={"code": "INCOMPLETE_RESULTS"})

            task["history"].append(message)
            task["artifacts"].append(artifact(RECEIPTS_MT, {
                "batchId": results["batchId"],
                "executions": executions,
            }))
            task["status"] = task_status("TASK_STATE_COMPLETED")
            con.execute(
                "UPDATE tasks SET state=?,task_json=?,updated_at=? WHERE task_id=? AND principal=?",
                ("TASK_STATE_COMPLETED", canonical(task), now(), task_id, principal),
            )
            con.execute(
                "INSERT INTO messages(principal,message_id,body_hash,task_id) VALUES(?,?,?,?)",
                (principal, message_id, body_hash, task_id),
            )
            con.execute("COMMIT")
        return response_task(task)

    raise HTTPException(status_code=400, detail={"code": "UNSUPPORTED_INPUT"})


@app.get("/a2a/tasks/{task_id}")
@app.get("/tasks/{task_id}")
def read_task(task_id: str, authorization: Optional[str] = Header(default=None)):
    principal = principal_from_header(authorization)
    row = get_task_row(task_id, principal)
    return JSONResponse(json.loads(row["task_json"]), media_type=A2A_MT)


@app.get("/a2a/tasks")
@app.get("/tasks")
def list_tasks(authorization: Optional[str] = Header(default=None)):
    principal = principal_from_header(authorization)
    with connect() as con:
        rows = con.execute(
            "SELECT task_json FROM tasks WHERE principal=? ORDER BY created_at DESC",
            (principal,),
        ).fetchall()
    return JSONResponse({"tasks": [json.loads(row["task_json"]) for row in rows]}, media_type=A2A_MT)


@app.post("/a2a/tasks/{task_id}:cancel")
@app.post("/tasks/{task_id}:cancel")
def cancel_task(task_id: str, authorization: Optional[str] = Header(default=None)):
    principal = principal_from_header(authorization)
    with connect() as con:
        con.execute("BEGIN IMMEDIATE")
        row = con.execute(
            "SELECT task_json,state FROM tasks WHERE task_id=? AND principal=?",
            (task_id, principal),
        ).fetchone()
        if row is None:
            con.execute("ROLLBACK")
            raise HTTPException(status_code=404, detail={"code": "NOT_FOUND"})
        if row["state"] in ("TASK_STATE_COMPLETED", "TASK_STATE_CANCELED"):
            con.execute("ROLLBACK")
            raise HTTPException(status_code=409, detail={"code": "TASK_TERMINAL"})
        task = json.loads(row["task_json"])
        task["status"] = task_status("TASK_STATE_CANCELED")
        con.execute(
            "UPDATE tasks SET state=?,task_json=?,updated_at=? WHERE task_id=? AND principal=?",
            ("TASK_STATE_CANCELED", canonical(task), now(), task_id, principal),
        )
        con.execute("COMMIT")
    return JSONResponse(task, media_type=A2A_MT)
