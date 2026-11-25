# AARYX — Low-Level Technical Design (LLD)

**Version:** 1.1 (aligned with the code on 2026-11-24)  
**Purpose:** Define module boundaries, APIs, data entities, and implementation rules for the personal MVP.  
**How to read this:** sections describe what is **implemented**. Anything not built yet is marked **(planned)** with its task id from `06-EXECUTION-PLAN.md`.

## 1. Repository structure

```text
aaryx/
├── .github/workflows/ci.yml        # lint, format check, tests
├── backend/
│   ├── app/
│   │   ├── __init__.py             # __version__
│   │   ├── __main__.py             # `python -m app` launcher (bind-address safety)
│   │   ├── main.py                 # FastAPI app factory, lifespan, error handlers, middleware
│   │   ├── api/                    # routes_chat, routes_conversations, routes_memories,
│   │   │                           # routes_models, routes_health, deps
│   │   ├── core/                   # config.py (Settings), logging.py, security.py
│   │   ├── brain/                  # orchestrator.py, prompts.py, schemas.py
│   │   ├── providers/              # base.py, ollama.py, openai.py, anthropic.py, gemini.py,
│   │   │                           # factory.py, router.py
│   │   ├── memory/                 # models.py, database.py, repository.py, service.py,
│   │   │                           # extractor.py, schemas.py, exceptions.py
│   │   ├── schemas/                # health.py
│   │   ├── voice/                  # not started
│   │   ├── emotion/                # not started
│   │   └── tools/                  # not started
│   ├── tests/
│   ├── pyproject.toml              # supported dependency ranges and tool config
│   ├── requirements.txt            # generated: exact runtime versions with hashes
│   ├── requirements-dev.txt        # generated: runtime + dev tools
│   └── .env.example
├── frontend/                       # Tauri + React client (not started; replaces `apps/desktop`)
├── desktop-agent/                  # common / macos / windows (not started)
├── infrastructure/                 # placeholder
├── scripts/update_lock.py          # regenerates the pinned requirements files
└── docs/
```

Differences from LLD 1.0: `frontend/` instead of `apps/desktop/`; `pyproject.toml` holds the supported ranges and `requirements*.txt` are generated lock files with exact versions and hashes; the router lives in `providers/router.py` (not `brain/router.py`); there are four provider adapters (`openai.py`, `anthropic.py`, `gemini.py`) instead of one `cloud_openai_compatible.py`; memory routes are in `routes_memories.py`.

## 2. API contract

### 2.1 Conventions

- JSON over HTTP on `http://127.0.0.1:8000`. Interactive docs at `/docs`.
- Streaming uses Server-Sent Events (decision D-008, see `05-ROADMAP`).
- All `/api/*` routes need `Authorization: Bearer <token>` (or `X-API-Key`) when `API_TOKEN` is set (section 7).
- Errors: see section 6.
- When memory is disabled (`MEMORY_ENABLED=false`) conversation and memory routes return `503`, and chat works without history.

### 2.2 `GET /health`

Returns `200` for every accepted request (a refused Host header gets `400` first). `status` is `ok` when the primary provider (or an allowed fallback) is reachable, otherwise `degraded`.

```json
{
  "status": "degraded",
  "app": "AARYX",
  "environment": "development",
  "provider": {
    "healthy": false,
    "provider": "router",
    "details": {
      "routing": {
        "primary_provider": "ollama",
        "fallback_provider": null,
        "fallback_enabled": false,
        "max_retries": 1,
        "allow_cloud_fallback": false
      },
      "primary": {
        "healthy": false,
        "provider": "ollama",
        "details": null,
        "error": "Unreachable at http://127.0.0.1:11434 (ConnectError): All connection attempts failed"
      },
      "fallback": null,
      "failure_counts": {},
      "last_failure": null
    },
    "error": "Unreachable at http://127.0.0.1:11434 (ConnectError): All connection attempts failed"
  }
}
```

When Ollama is reachable, `primary.details` is `{"base_url", "default_model", "model_count", "available_models": [...]}` and `provider.error` is `null`. Error text never contains secrets. Reachable without a token.

### 2.3 `GET /api/models`

Returns the models of the active provider: `[{"name": "qwen3.5:4b", "size": 123, "modified_at": "...", "details": {...}}]`.

### 2.4 Chat

`POST /api/chat`

```json
{
  "conversation_id": "optional existing id",
  "message": "Hello AARYX",
  "history": [],
  "model": "optional model override",
  "temperature": 0.7,
  "max_tokens": 512
}
```

- `message` is required, trimmed, and cannot be empty. `temperature` is 0–2, `max_tokens` at least 1.
- With memory enabled, history comes from storage and any `history` sent by the client is ignored; without `conversation_id` a new conversation is created. An unknown `conversation_id` returns `404`.
- With memory disabled, the client's `history` is used and `conversation_id` in the reply is `null`.
- There is no per-request local/cloud selector yet **(planned, P1-9)**; the provider comes from settings.

Response:

```json
{
  "reply": "Hello! How can I help?",
  "model": "qwen3.5:4b",
  "finish_reason": "stop",
  "usage": {"prompt_tokens": 12, "completion_tokens": 9, "total_tokens": 21},
  "thinking": null,
  "provider": "ollama",
  "latency_ms": 1840,
  "conversation_id": "uuid"
}
```

`thinking` is always `null`: reasoning text never reaches the client. `latency_ms` is the whole number of milliseconds from sending the request to the model until the reply was ready; it includes retries and fallback, and is also written to the log (never together with message text). It is absent from error responses.

`POST /api/chat/stream` takes the same body and answers with `text/event-stream`. Each event is `data: {chunk}\n\n` where a chunk has `delta`, `model`, `finish_reason`, `usage`, `thinking` (only the placeholder `"AARYX is thinking..."`), `provider`, `latency_ms`, `conversation_id`. `latency_ms` is set only on the last chunk (the one with `finish_reason`) and is the total time, not the time to the first token. A normal stream ends with `data: [DONE]`. If generation or saving fails, the stream ends without `[DONE]`, and no partial assistant message is stored.

Turn rules: the user message is stored before inference and kept if the turn fails; an assistant message is stored only after a complete, non-empty reply; unanswered user messages from failed turns are merged into the next prompt so roles keep alternating.

### 2.5 Conversations

| Route | Notes |
|---|---|
| `POST /api/conversations` | body `{"title"?: str, "metadata"?: object}` → `201` conversation |
| `GET /api/conversations?limit=50&offset=0` | `limit` 1–100; returns `{conversations, total, limit, offset}` ordered by recent activity |
| `GET /api/conversations/{id}` | conversation plus its `messages` in order; `404` if unknown |
| `DELETE /api/conversations/{id}` | deletes messages too; returns `{"message": "...", "id": "..."}`; memories learned in it are kept |

Conversation: `id, title, created_at, updated_at, metadata_json, message_count`. Message: `id, conversation_id, role, content, model, provider, token_usage, created_at`.

### 2.6 Memories

| Route | Notes |
|---|---|
| `POST /api/memories` | create (see fields below) → `201`; refused for passwords, keys, ID numbers, prompt-override text |
| `GET /api/memories` | filters `status`, `category`, `include_deleted`, `limit` (1–100, default 50), `offset` |
| `GET /api/memories/{id}` | one memory |
| `PATCH /api/memories/{id}` | edit content, category, confidence, importance, metadata. Cannot activate an inferred memory or set `rejected` |
| `DELETE /api/memories/{id}?hard=false` | soft delete by default; `hard=true` removes the row |
| `POST /api/memories/{id}/approve` | pending suggestion → `active`; optional body `{"content": "..."}` corrects the text first |
| `POST /api/memories/{id}/reject` | pending suggestion → `rejected` |

Create fields: `content` (1–5000 chars), `category` (`preference`, `fact`, `project`, `biography`, `instruction`), `origin` (`explicit` default, or `inferred`), `confidence` (0–1; explicit needs ≥ 0.90), `importance` (1–5), `status` (default `active`; an inferred memory can only be created as `candidate`), `source_conversation_id`, `metadata`.

Read fields: `id, content, category, origin, confidence, importance, status, superseded_by_id, source_conversation_id, created_at, updated_at, last_accessed_at, access_count, metadata_json`.

Suggestion actions return `400` if the memory is not a pending suggestion and `404` if it does not exist. `MEMORY_SUGGESTIONS_ENABLED=true|false` (default `true`) turns inferred extraction on or off; explicit "remember this" always works. Identical pending suggestions are not stored twice.

### 2.7 Memory in the chat flow

- Before inference, chat selects `active` memories relevant to the user message and appends them to the system prompt as information about the user (not as instructions). Clients cannot supply this block; only the server does.
- Selection: core memories (category `instruction`, or importance >= 4, which covers everything the user explicitly asked to remember) plus memories sharing meaningful words with the message; ranked by overlap and importance; capped by `MEMORY_MAX_CONTEXT_ITEMS` (default 8; `0` disables injection). Selected memories get `access_count` and `last_accessed_at` updated.
- After a completed, persisted reply, a background task runs memory extraction with the original user message and the reply. Explicit "remember this" is stored `active`; inferred memories are stored as pending suggestions (if `MEMORY_SUGGESTIONS_ENABLED`). Failed turns, empty replies and interrupted streams trigger no extraction.
- A failure in memory lookup or extraction is logged (without message content) and never changes the chat response.
- "Remember this" is recognised in English ("remember that...", "don't forget...", "keep in mind...", "note that...") and Hinglish ("yaad rakho ki...", "dhyan rakho...", "mat bhoolna...", "note kar lo..."). Requests containing passwords, keys, ID numbers or prompt-override text are refused and nothing is stored.
- Matching a message to memories is case-insensitive, ignores punctuation, handles simple plurals/inflections ("garden"/"gardens", "run"/"running") and Devanagari text. Identical memories appear once in the prompt.
- At most 5 suggestions are queued per turn. Memory writes (extraction, approve, reject) are serialized, so parallel turns or double clicks cannot create duplicate suggestions or approve one suggestion twice.

### 2.8 Tools (planned, Phase 4)

- `POST /api/tools/propose` creates a proposed action. It must not execute the action.
- `POST /api/tools/{proposal_id}/approve` executes only a valid, unexpired, approved proposal.
- Also planned: reject, audit listing, scope management (P4-3, P4-14).

### 2.9 Planned API additions

| Addition | Task |
|---|---|
| `mode` / `provider` / `model` per chat request; `GET /api/providers` | P1-9, P1-10 |
| `GET /api/usage`, spend caps | P1-14 |
| `GET/PUT /api/settings` | P1-16 |
| Private (unsaved) chat; conversation search and export | P1-15, P1-17 |
| `provider_mode` stored on conversations | P1-9 |

## 3. Core data entities (SQLite, WAL mode)

### Conversation
`id` (UUID string), `title`, `created_at`, `updated_at`, `metadata_json`. Messages belong to a conversation and are deleted with it. (`provider_mode` from LLD 1.0 is not stored yet, see P1-9.)

### Message
`id`, `conversation_id`, `role` (`user` / `assistant` / `system`), `content`, `model`, `provider`, `token_usage` (JSON), `created_at`.

### Memory
`id`, `content`, `category`, `origin` (`explicit` / `inferred`), `confidence`, `importance`, `status` (`candidate` = pending suggestion, `active`, `superseded`, `rejected`, `deleted`), `superseded_by_id`, `source_conversation_id` (kept when the conversation is deleted), `created_at`, `updated_at`, `last_accessed_at`, `access_count`, `metadata_json` (holds `approved_by_user`, `conflicts_with`, rejection and approval times).

Rules (Decision D-001):
- An `inferred` memory is created only as `candidate`, whatever its confidence.
- `candidate` → `active` happens only through explicit user approval (recorded as `approved_by_user` in `metadata_json`; there is no separate `user_confirmed` column).
- Prompt retrieval reads `active` memories only.
- An `inferred` memory never supersedes an `explicit` one; an approved suggestion that conflicts with an active memory supersedes it.
- A memory cannot enter `superseded` without a valid `superseded_by_id`.
- Extraction uses the local model only.
- Compared with LLD 1.0, `source_message_id`, `user_confirmed` and `deleted_at` do not exist; soft delete uses `status = deleted`.

### ToolProposal (planned, P4-2)
`id`, `action_name`, `arguments` (validated JSON), `risk_level`, `status` (`pending`, `approved`, `rejected`, `expired`, `completed`, `failed`), `created_at`, `expires_at`, `approved_at`.

### AuditEvent (planned, P4-4)
`id`, `action`, `result`, `timestamp`, `proposal_id` (optional).

## 4. Provider interface

Every model provider subclasses `LLMProvider` (`providers/base.py`):

```python
class LLMProvider(ABC):
    async def generate(self, request: ChatRequest) -> ChatResponse: ...
    def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]: ...
    async def list_models(self) -> list[ModelInfo]: ...
    async def health_check(self) -> ProviderHealth: ...
    async def aclose(self) -> None: ...
```

- `ChatRequest`: `messages` (`role` + `content`), optional `model`, `temperature` (0–2), `max_tokens`.
- `ChatResponse`: `content`, `model`, `finish_reason`, `usage` (`prompt_tokens`, `completion_tokens`, `total_tokens`), `thinking`, `provider`. Latency is measured one layer up, by the brain orchestrator around the provider call, and exposed as `latency_ms` on the chat response and the final stream chunk.
- `ChatStreamChunk`: `delta`, `model`, `finish_reason`, `usage`, `thinking`, `provider`.
- `ProviderHealth`: `healthy`, `provider`, `details`, `error`.
- Provider-specific request and response formats stay inside each adapter.
- Errors derive from `ProviderError` (with a `safe_message` free of secrets): `ProviderConfigurationError`, `UnsupportedProviderError`, `ProviderConnectionError`, `ProviderTimeoutError`, `ProviderResponseError`.
- `ProviderRouter` (`providers/router.py`) is itself an `LLMProvider`: it picks the primary provider, retries transient failures (`AI_MAX_RETRIES`), and fails over to the fallback provider only when `AI_FALLBACK_ENABLED` is true and, for cloud fallbacks from a local primary, `AI_ALLOW_CLOUD_FALLBACK` is true.
- Memory extraction uses a local `OllamaProvider` only; cloud providers are rejected.

## 5. Configuration

`backend/.env.example` is the complete, tested reference (`tests/test_env_example.py` keeps it in sync with `Settings`). Groups:

| Group | Settings |
|---|---|
| Application and server | `APP_NAME`, `APP_ENV`, `LOG_LEVEL`, `API_HOST`, `API_PORT`, `API_TOKEN`, `API_CORS_ORIGINS`, `API_ALLOWED_HOSTS` |
| AI routing | `AI_MODE`, `AI_PROVIDER`, `AI_FALLBACK_ENABLED`, `AI_FALLBACK_PROVIDER`, `AI_MAX_RETRIES`, `AI_ALLOW_CLOUD_FALLBACK` |
| Providers | `OLLAMA_*`, `OPENAI_*`, `ANTHROPIC_*`, `GEMINI_*`, `CLOUD_*` (base URL, key, model, timeout) |
| Conversations and memory | `MEMORY_ENABLED`, `MEMORY_DATABASE_URL`, `MEMORY_MAX_HISTORY_MESSAGES`, `MEMORY_EXTRACTION_MODEL`, `MEMORY_MAX_CONTEXT_ITEMS`, `MEMORY_SUGGESTIONS_ENABLED`, `MEMORY_CANDIDATE_MIN_CONFIDENCE` |

Rules:
- Settings load from `backend/.env` or environment variables; every setting has a safe default.
- `create_app(settings)` keeps the settings it was given on `app.state.settings`; startup (provider router, logging, memory), the memory service and `/health` all use those, never a second copy read from the environment. Only an app built some other way falls back to the cached environment settings.
- `.env` is local-only and excluded from Git; `.env.example` holds placeholders only.
- API keys and the API token are secret values: never logged, never returned by the API, never echoed in a validation error.
- Invalid configuration fails at startup with a clear message (for example a fallback cloud provider without its key, a too-short `API_TOKEN`, a wildcard CORS origin).
- Production secrets will move to the OS keychain (P7-2).

## 6. Error handling

| Situation | Status | Body |
|---|---|---|
| Invalid request body or parameters | 422 | FastAPI default: `{"detail": [{"loc": ..., "msg": ...}]}` |
| Unknown conversation or memory | 404 | `{"detail": "..."}` or `{"error": "Not Found", "detail": "..."}` |
| Business-rule violation (empty id, unsafe memory text, wrong status) | 400 | `{"detail": "..."}` or `{"error": "Validation Error", "detail": "..."}` |
| Missing or wrong API token | 401 | `{"error": "Unauthorized", "detail": "A valid API token is required."}` plus `WWW-Authenticate: Bearer` |
| Disallowed Host header | 400 | `{"error": "Invalid host", "detail": "..."}` |
| Memory disabled | 503 | `{"detail": "Persistent conversation memory is disabled."}` |
| Provider unreachable | 503 | provider error body, `code` `provider_unreachable` |
| Provider timeout | 504 | provider error body, `code` `provider_timeout` |
| Model not available (provider answered 404) | 502 | provider error body, `code` `model_not_found`, `model` |
| Wrong address (provider answered 404 while listing models, or Ollama answered 404 without naming a model) | 502 | provider error body, `code` `endpoint_not_found` |
| Provider rejected the key / rate limit | 502 | provider error body, `code` `authentication_failed` / `rate_limited` |
| Provider returned another error | 502 | provider error body, `code` `provider_response_error` |
| Other provider error | 500 | provider error body, `code` `provider_error` |

**Provider error body** (every row above; the same object is sent as `{"error": {...}}` in a stream):

```json
{
  "error": "Model not available",
  "code": "model_not_found",
  "detail": "Ollama returned HTTP 404: {\"error\":\"model 'qwen3.5:4b' not found ...\"}",
  "provider": "ollama",
  "model": "qwen3.5:4b",
  "hint": "Run `ollama pull qwen3.5:4b` to install it (see installed models with `ollama list`), or set OLLAMA_MODEL to a model you have."
}
```

- `error` (title), `detail` and `provider` are unchanged from before; `code` is the stable value a UI should switch on; `model` appears only when the adapter knows it and it is a plain model id; `hint` is one sentence telling the user how to fix it, built from fixed text and setting names, and is absent when there is nothing useful to say.
- A 404 is read carefully: on a chat request it means an unknown model (`model_not_found`), but a 404 while listing models, or from an Ollama server whose answer does not mention a model (some other server, or a wrong `OLLAMA_BASE_URL`), is `endpoint_not_found` and the hint points at the base URL instead of telling the user to install a model.
- Hints: Ollama not running says to run `ollama serve` (or open the app) and check `OLLAMA_BASE_URL`; a missing Ollama model names the exact `ollama pull <model>` command; cloud providers point at `<PROVIDER>_MODEL`, `<PROVIDER>_API_KEY`, `<PROVIDER>_BASE_URL` or `<PROVIDER>_TIMEOUT`. A model name that is not a plain id (spaces, quotes, `;`, `$`, ...) is dropped and the hint shows `<model>`, so a request can never inject a command into a hint.
- Streams: the status line is already sent, so when a provider fails the stream carries a final event `data: {"error": {...}}` and then ends without `[DONE]`. Unexpected non-provider errors send no event and never reveal internals.

- Never expose stack traces or secret values to clients or logs.
- Tool denied (planned): return a denied status and do not attempt execution.

## 7. Local API security

Implemented in `core/security.py` and wired in `main.py`; middleware order is Host check → CORS → API token → routes.

- **Bind address:** start the server with `python -m app`, which uses `API_HOST` and `API_PORT` and refuses any non-loopback address unless `API_TOKEN` is set. With `--reload` the launcher keeps its socket open while every new worker re-reads `.env`, so the launcher records the address it bound in `AARYX_LAUNCH_HOST` and every worker checks it at startup: a worker whose settings have no `API_TOKEN` refuses to start when that address is not loopback. The recorded address, not the current `API_HOST`, decides. (Starting `uvicorn` directly bypasses these checks; do not use `--host 0.0.0.0` that way.)
- **Host guard:** only Host names in `API_ALLOWED_HOSTS` (default `localhost`, `127.0.0.1`, `[::1]`) are served. A web page cannot reach the API through DNS rebinding because its requests carry the attacker's Host name.
- **API token:** optional. When set (at least 16 characters), every route except `/health`, `/docs`, `/redoc` and `/openapi.json` needs it as `Authorization: Bearer <token>` or `X-API-Key`; it is compared in constant time, never accepted in the URL, never logged.
- **CORS:** only the origins in `API_CORS_ORIGINS` (default: the Tauri app origins `tauri://localhost` and `http://tauri.localhost`). Add the UI dev server (for example `http://localhost:1420`) while developing. No wildcards, no credentials. Preflight requests work without a token.

## 8. Testing strategy

- Unit tests for router and provider adapters using mocked HTTP.
- Integration tests against a local Ollama when available (`pytest -m integration`).
- API tests for validation, errors, security (Host, token, CORS, bind rule).
- Memory tests for save, retrieve, edit, delete, suggestions, duplicates, contradictions, supersession.
- Chat-memory integration tests and a user-scenario suite (`tests/test_user_scenarios.py`) that attacks each feature the way a user, a flaky model or a hostile input could.
- Guard tests: `.env.example` matches the settings; versions match across files; the pinned requirements files match `pyproject.toml`.
- CI runs ruff, format check and pytest on Linux, macOS (primary platform) and Windows.
- Permission tests proving denied actions do not execute (with Phase 4).
- Manual smoke test on the MacBook (primary OS) first, then the Windows laptop (P0-16, P7-1, P7-8).

## 9. Definition of done

A module is complete only when:
- Code is implemented.
- Input validation exists.
- Errors are handled.
- Tests cover expected and failure paths.
- Secrets are not committed.
- User-facing behavior is documented.
