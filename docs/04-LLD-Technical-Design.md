# AARYX — Low-Level Technical Design (LLD)

**Version:** 1.0  
**Purpose:** Define module boundaries, APIs, data entities, and implementation rules for the personal MVP.

## 1. Repository structure

```text
AARYX/
├── apps/
│   └── desktop/                 # Tauri + React client (later phase)
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application
│   │   ├── api/
│   │   │   ├── routes_chat.py
│   │   │   ├── routes_health.py
│   │   │   └── routes_memory.py
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   └── logging.py
│   │   ├── brain/
│   │   │   ├── orchestrator.py
│   │   │   ├── router.py
│   │   │   └── prompts.py
│   │   ├── providers/
│   │   │   ├── base.py
│   │   │   ├── ollama.py
│   │   │   └── cloud_openai_compatible.py
│   │   ├── memory/
│   │   │   ├── service.py
│   │   │   └── repository.py
│   │   ├── voice/
│   │   ├── emotion/
│   │   ├── tools/
│   │   │   ├── broker.py
│   │   │   └── permissions.py
│   │   └── schemas/
│   ├── tests/
│   ├── requirements.txt
│   └── .env.example
├── desktop-agent/
│   ├── common/
│   ├── macos/
│   └── windows/
├── docs/
└── README.md
```

## 2. API contract (initial)
### `GET /health`
Response:
```json
{"status": "ok", "app": "AARYX"}
```

### `POST /api/chat`
Request:
```json
{
  "session_id": "optional-session-id",
  "message": "Hello AARYX",
  "mode": "local"
}
```
Response:
```json
{
  "session_id": "session-id",
  "reply": "Hello! How can I help?",
  "provider": "ollama",
  "model": "qwen3.5:4b"
}
```

### `GET /api/memory`
Returns user-approved memory records.

### `POST /api/memory`
Creates a memory only after explicit user intent or confirmation.

### `DELETE /api/memory/{memory_id}`
Deletes the selected memory and removes it from retrieval.

### `POST /api/tools/propose`
Creates a proposed action. It must not execute the action.

### `POST /api/tools/{proposal_id}/approve`
Executes only a valid, unexpired, approved proposal.

## 3. Core data entities
### Conversation
- `id` (UUID)
- `created_at`
- `title`
- `provider_mode`

### Message
- `id` (UUID)
- `conversation_id`
- `role` (`user` / `assistant` / `system`)
- `content`
- `created_at`
- `provider`
- `model`

### Memory
- `id` (UUID)
- `content`
- `category`
- `created_at`
- `updated_at`
- `source_message_id` (optional)
- `user_confirmed` (boolean)
- `deleted_at` (optional)

### ToolProposal
- `id` (UUID)
- `action_name`
- `arguments` (validated JSON)
- `risk_level`
- `status` (`pending`, `approved`, `rejected`, `expired`, `completed`, `failed`)
- `created_at`
- `expires_at`
- `approved_at` (optional)

### AuditEvent
- `id` (UUID)
- `action`
- `result`
- `timestamp`
- `proposal_id` (optional)

## 4. Provider interface
All model providers should implement a common method:

```python
async def generate(messages: list[dict], model: str) -> AIResponse:
    ...
```

`AIResponse` should contain:
- `text`
- `provider`
- `model`
- `usage` (optional)
- `latency_ms` (optional)

Provider-specific request/response formats must stay inside the adapter.

## 5. Configuration
Environment configuration:
- `AI_MODE=local|cloud`
- `OLLAMA_BASE_URL`
- `OLLAMA_MODEL`
- `CLOUD_BASE_URL`
- `CLOUD_API_KEY`
- `CLOUD_MODEL`

Rules:
- `.env` is local-only and excluded from Git.
- `.env.example` contains placeholders only.
- Production secrets must use OS keychain or managed secret storage.
- Never log API keys or full authorization headers.

## 6. Error handling
- Provider unavailable: return a clear provider-unavailable error.
- Timeout: return timeout code and allow retry.
- Invalid configuration: fail startup or disable that provider with a clear warning.
- Invalid input: return HTTP 422.
- Tool denied: return a denied status; do not attempt execution.
- Never expose stack traces or secret values to the UI.

## 7. Testing strategy
- Unit tests for router and provider adapters using mocked HTTP.
- Integration test against local Ollama when available.
- API tests for request validation and errors.
- Memory tests for save, retrieve, edit, delete.
- Permission tests proving denied actions do not execute.
- Manual smoke test on macOS; later repeat on Windows.

## 8. Definition of done
A module is complete only when:
- Code is implemented.
- Input validation exists.
- Errors are handled.
- Tests cover expected and failure paths.
- Secrets are not committed.
- User-facing behavior is documented.
