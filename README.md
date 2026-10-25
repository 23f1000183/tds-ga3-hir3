# A2A Invoice Agent — starter implementation

This is a local-first implementation of the invoice-agent assignment. **Do not use a grader Check/Save until you have completed the local test checklist and deployed a public HTTPS base URL.**

## 1. Requirements

- Windows with Python 3.11+ recommended
- A model available through an OpenAI-compatible Chat Completions endpoint (Ollama works locally)
- For final verification: public HTTPS hosting and a stable persistent disk/volume for SQLite

## 2. Install (PowerShell)

```powershell
mkdir a2a-invoice-agent
cd a2a-invoice-agent
# Extract the ZIP contents into this folder first.
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set a strong `AGENT_TOKEN`. Never share or commit the token.

Set environment variables in the current PowerShell session (Windows PowerShell does not automatically load .env):

```powershell
$env:AGENT_TOKEN = "paste-a-long-random-secret-here"
$env:A2A_BASE_URL = "https://your-public-host.example/a2a"
$env:MODEL_NAME = "qwen2.5:3b"
$env:MODEL_URL = "http://localhost:11434/v1/chat/completions"
$env:DB_PATH = "invoice_agent.sqlite3"
```

Install/start Ollama separately, then pull a model if you choose the local option:

```powershell
ollama pull qwen2.5:3b
ollama serve
```

Keep Ollama running in a separate terminal. If you use a hosted provider, set `MODEL_URL`, `MODEL_NAME`, and `MODEL_API_KEY` appropriately.

## 3. Run locally

```powershell
uvicorn app:app --host 127.0.0.1 --port 8000
```

Check health in a browser: `http://127.0.0.1:8000/health`. OpenAPI docs: `http://127.0.0.1:8000/docs`.

The local URL is not a valid assignment submission URL. It must be public HTTPS with no redirect, credentials, query, or fragment.

## 4. Protocol notes

- Public Agent Card: `/.well-known/agent-card.json`
- Protected routes: `/message:send`, `/tasks`, `/tasks/{id}`, `/tasks/{id}:cancel`
- Protected routes require `Authorization: Bearer ...`, `A2A-Version: 1.0`; POST requires `Content-Type: application/a2a+json`.
- Initial messages use `application/vnd.ga5.invoice-claim-batch+json`.
- Result continuations use `application/vnd.ga5.invoice-action-results+json`.
- Proposals remain proposals until accepted result receipts arrive.

## 5. Before submission: important hardening still required

This starter is intended to get the project running and provide the protocol skeleton. It is **not yet verified against the grader**. In particular:
1. Confirm the exact A2A Task schema/envelope expected by the grader.
2. Add robust per-token multi-principal auth for the hosting setup and ensure the same persistent DB volume is used across restarts.
3. Test model output against real synthetic package structure and validate that every evidence reference actually occurs in the decisive paragraph.
4. Check the receipt continuation contract against the grader: accepted/rejected results, nonce rules, and exact state names.
5. Test concurrent same-message requests, message-ID conflicts, task ownership, cancel-vs-result races, terminal replay, and request time limits.
6. Implement atomic handling for every race path and verify that no model call happens for replay/list/poll/cancel/continuation requests.
7. Check the 512 KiB response limit and all media-type details.

This code does not move money or call a payment system. It only proposes typed actions and records grader-accepted results.
