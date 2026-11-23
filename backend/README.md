# AARYX Backend

FastAPI backend for the AARYX personal AI companion. Local-first: it talks to a local Ollama model by default and only uses a cloud provider if you configure one.

Install, run, back up and troubleshoot: see [`../docs/SETUP.md`](../docs/SETUP.md).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
cp .env.example .env
python -m app
pytest
```

Run everything from this `backend/` folder. API docs are served at `http://127.0.0.1:8000/docs`.

## API

| Method and path | Purpose |
|---|---|
| `GET /health` | status of the app and the model provider (`ok` or `degraded`) |
| `GET /api/models` | models available on the active provider |
| `POST /api/chat` | send a message, get a full reply |
| `POST /api/chat/stream` | same, streamed as server-sent events |
| `POST/GET /api/conversations`, `GET/DELETE /api/conversations/{id}` | conversation history |
| `POST/GET /api/memories`, `GET/PATCH/DELETE /api/memories/{id}` | long-term memory |
| `POST /api/memories/{id}/approve`, `.../reject` | approve or reject a memory suggestion |

## Layout

```text
app/
├── api/         HTTP routes
├── brain/       prompt assembly, orchestration, reasoning-tag filtering
├── core/        settings and logging
├── memory/      conversations, long-term memory, suggestions, extraction
├── providers/   Ollama, OpenAI, Anthropic, Gemini adapters, router and fallback
├── voice/       not started
├── emotion/     not started
└── tools/       not started (permissioned desktop actions)
tests/           ~1000 tests, no Ollama or internet needed
```

## Rules the code enforces

- Memories you ask for are saved; memories AARYX infers are suggestions until you approve them, and only approved memories reach a prompt.
- Memory extraction uses the local model only.
- Passwords, API keys, card and ID numbers are refused as memories; keys are redacted from logs.
- A failure in memory never breaks a chat reply.
- The API answers only to localhost host names, can require an `API_TOKEN`, and `python -m app` refuses to listen beyond this computer without one.
- No desktop automation exists yet. None will be added until the tool permission model is in place (see `../docs/06-EXECUTION-PLAN.md`, Phase 4).
