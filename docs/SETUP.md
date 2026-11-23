# AARYX — Setup, run, and recovery guide

This guide takes you from a fresh computer to a running AARYX backend. It covers the backend only; the desktop app is not built yet (see `06-EXECUTION-PLAN.md`).

## 1. What you need

| Tool | Why | Check |
|---|---|---|
| Python 3.11 or newer | runs the backend | `python --version` |
| Git | get the code | `git --version` |
| [Ollama](https://ollama.com) | runs the local AI model (optional if you only use a cloud provider) | `ollama --version` |

Everything stays on your computer by default. Nothing is sent to a cloud provider unless you configure one and allow it.

## 2. Get the code and install

Windows (PowerShell):

```powershell
git clone https://github.com/Dipanshup66/aaryx.git
cd aaryx\backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --require-hashes -r requirements-dev.txt
pip install --no-deps -e .
```

macOS / Linux:

```bash
git clone https://github.com/Dipanshup66/aaryx.git
cd aaryx/backend
python3 -m venv .venv
source .venv/bin/activate
pip install --require-hashes -r requirements-dev.txt
pip install --no-deps -e .
```

The first command installs the exact package versions this project is tested with (pinned, with hashes, so a tampered or different download is refused). The second installs AARYX itself. To run without the test tools use `requirements.txt` instead of `requirements-dev.txt`.

> Always run the backend and the tests from the `backend/` folder. The default database path (`./data/aaryx.db`) is relative to the folder you start the server from.

## 3. Configure

```bash
cp .env.example .env        # Windows PowerShell: Copy-Item .env.example .env
```

An empty or unchanged `.env` works for local Ollama use. `.env.example` explains every option. `.env` is git-ignored: never commit it or paste API keys anywhere else.

The settings you are most likely to change:

| Setting | Meaning |
|---|---|
| `OLLAMA_MODEL` | the local model to chat with (default `qwen3.5:4b`) |
| `MEMORY_EXTRACTION_MODEL` | the local model that suggests memories (default `qwen3.5:4b`) |
| `MEMORY_SUGGESTIONS_ENABLED` | `true`: AARYX suggests memories and you approve them. `false`: only "remember that ..." creates memories |
| `MEMORY_MAX_CONTEXT_ITEMS` | how many approved memories go into each prompt (`0` = none) |
| `MEMORY_ENABLED` | `false` turns off all history and memory |
| `API_HOST`, `API_PORT` | where the server listens (default `127.0.0.1:8000`) |
| `API_TOKEN` | optional password for the API; required to listen beyond this computer (section 5) |
| `AI_PROVIDER` | `ollama` (local), or `openai`, `anthropic`, `gemini` (cloud; needs the matching `*_API_KEY`) |

## 4. Prepare the local model (Ollama)

```bash
ollama serve                 # skip if the Ollama app is already running
ollama pull qwen3.5:4b       # the project default
ollama list                  # shows the models you have
```

If that model name is not available in your Ollama version, pick any model from `ollama list` and set both `OLLAMA_MODEL` and `MEMORY_EXTRACTION_MODEL` in `.env`.

## 5. Run the backend

From the `backend/` folder with the virtual environment active:

```bash
python -m app
```

Add `--reload` while developing. The address comes from `API_HOST` and `API_PORT` in `.env` (default `127.0.0.1:8000`, reachable only from this computer). `python -m app` refuses to listen on any other address unless `API_TOKEN` is set. Avoid starting `uvicorn` by hand with `--host 0.0.0.0`: that skips this safety check.

Check it works:

- Open <http://127.0.0.1:8000/docs> for the interactive API page.
- `GET /health` returns `"status": "ok"` when the model provider is reachable and `"degraded"` when it is not (for example Ollama is not running).

## 6. Try it

Windows PowerShell:

```powershell
$r = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/chat -ContentType "application/json" -Body '{"message": "Hello AARYX"}'
$r.reply
$r.conversation_id
```

macOS / Linux:

```bash
curl -s http://127.0.0.1:8000/api/chat -H "content-type: application/json" -d '{"message": "Hello AARYX"}'
```

Send `"conversation_id": "<id from the first reply>"` to continue the same conversation.

### Memory in two minutes

1. Tell AARYX something to keep: `"Remember that I live in Pune."` (Hinglish works too: `"yaad rakho ki main Pune mein rehta hoon"`). It is saved right away.
2. See it: `GET /api/memories`. Later chats get the relevant approved memories automatically.
3. When AARYX infers something on its own (for example "I really like green tea"), it only becomes a *suggestion*. List them with `GET /api/memories?status=candidate`, then `POST /api/memories/{id}/approve` (optionally with `{"content": "corrected text"}`) or `POST /api/memories/{id}/reject`. Suggestions are never used until you approve them.
4. Delete any memory with `DELETE /api/memories/{id}`.

Passwords, API keys, card numbers and national ID numbers are never stored as memories.

### Protecting the API (optional)

Your memories are personal data, so the API defends itself:

- It answers only to the host names in `API_ALLOWED_HOSTS` (default `localhost`, `127.0.0.1`, `[::1]`). This stops a web page from reaching your local API through DNS rebinding.
- Browsers may call it only from the desktop app's origins (`API_CORS_ORIGINS`, default `tauri://localhost,http://tauri.localhost`). While developing the UI, add its dev server, for example `API_CORS_ORIGINS=tauri://localhost,http://tauri.localhost,http://localhost:1420`.
- To require a password on every `/api` call, set a token (at least 16 characters):

  ```bash
  python -c "import secrets; print(secrets.token_urlsafe(32))"   # prints a good token
  ```

  Put it in `.env` as `API_TOKEN=...`, restart, and send `Authorization: Bearer <token>` (or `X-API-Key: <token>`) with each request. `/health` and `/docs` stay open.
- To let another computer reach the API you must set `API_TOKEN`, set `API_HOST` to the address to listen on, and add the host name clients use to `API_ALLOWED_HOSTS`. Never expose it to the public internet.

## 7. Using a cloud model (optional)

Put the key in `.env` (never in code or chat):

```
AI_PROVIDER=openai          # or anthropic, gemini
OPENAI_API_KEY=...
```

Cloud models receive the conversation and the approved memories selected for that turn. To let a local-first setup fall back to the cloud only when Ollama fails, set `AI_FALLBACK_ENABLED=true`, `AI_FALLBACK_PROVIDER=...` and the explicit consent flag `AI_ALLOW_CLOUD_FALLBACK=true`. Leave the consent flag `false` to keep everything on this computer.

## 8. Run the tests

From `backend/`:

```bash
pytest                                              # no Ollama or internet needed
ruff check app tests
ruff format --check app tests
```

`pytest -m integration` runs the few tests that talk to a real local Ollama.

## 9. Update the pinned dependencies

Versions live in `backend/requirements.txt` and `backend/requirements-dev.txt` (generated, do not edit by hand). After you change dependencies in `backend/pyproject.toml`, or when you want newer versions:

```bash
python -m pip install uv                 # once
python scripts/update_lock.py            # keep current pins, apply pyproject changes
python scripts/update_lock.py --upgrade  # move everything to the newest allowed versions
```

Then reinstall (the two `pip install` commands in section 2), run `pytest`, and commit both files together with the change. `pytest` fails if the files no longer match `pyproject.toml`.

## 10. Back up and restore

Your conversations and memories live in one SQLite database, `backend/data/aaryx.db` by default (see `MEMORY_DATABASE_URL`). The database uses WAL mode, so:

1. **Stop the server first.**
2. Copy `aaryx.db` and, if present, `aaryx.db-wal` and `aaryx.db-shm` together.

To restore, stop the server, replace those files with your backup, and start it again.

Starting fresh: stop the server and delete the `data/` files; an empty database is created on the next start.

## 11. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| The `pip install` commands fail | run them inside `backend/`, with Python 3.11+, and update pip first: `python -m pip install -U pip` |
| pip says `THESE PACKAGES DO NOT MATCH THE HASHES` | the download differs from the pinned one: retry once; if it repeats, do not bypass the check, report it |
| pip finds no matching file for a package on your system (a very new Python or an unusual platform) | as a fallback only, `pip install -e ".[dev]"` installs the newest versions allowed by `pyproject.toml`; then tell us which package failed |
| `/health` says `degraded`, chat returns `503 Provider connection failed` | Ollama is not running: start it with `ollama serve` (or open the app), and check `OLLAMA_BASE_URL` |
| Chat fails with a provider error right after you changed the model | the model is probably not pulled: compare `OLLAMA_MODEL` with `ollama list` (exact name), and run `ollama pull <model>` if it is missing. A clearer message for this case is planned (task P1-11) |
| Chat times out | local models can be slow on CPU: raise `OLLAMA_TIMEOUT` or use a smaller model |
| Server will not start with a message like "OpenAI configured as fallback provider, but OPENAI_API_KEY is not set" | you enabled a cloud fallback without its `*_API_KEY`: add the key, or set `AI_FALLBACK_ENABLED=false` |
| `Address already in use` | another program uses the port: set `API_PORT=8001` in `.env` |
| `python -m app` prints "Refusing to listen on ..." | `API_HOST` is not a loopback address and `API_TOKEN` is empty: use `127.0.0.1`, or set a token |
| Startup says `API_TOKEN must be at least 16 characters` | generate a longer one (command in "Protecting the API") |
| Every call returns `401 Unauthorized` | `API_TOKEN` is set: send `Authorization: Bearer <token>` |
| Every call returns `400 Invalid host` | the request used a host name that is not in `API_ALLOWED_HOSTS` (for example your computer's name instead of `localhost`) |
| The desktop app or a browser page gets a CORS error | add its origin to `API_CORS_ORIGINS` |
| My memories disappeared / a new empty database appeared | you started the server from a different folder (the default path is relative). Start it from `backend/`, or set an absolute `MEMORY_DATABASE_URL` |
| No memory suggestions appear | `MEMORY_SUGGESTIONS_ENABLED` is `false`, or Ollama (the extraction model) is not running. "Remember that ..." works without the extraction model |
| Tests fail on a fresh checkout | reinstall with the two `pip install` commands from section 2, make sure you are in `backend/`, and report the failing test name |
