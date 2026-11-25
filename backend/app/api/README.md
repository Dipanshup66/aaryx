# api

HTTP routes (FastAPI routers), mounted in `app/main.py`.

| Module | Endpoints |
|---|---|
| `routes_health.py` | `GET /health` |
| `routes_models.py` | `GET /api/models` |
| `routes_chat.py` | `POST /api/chat`, `POST /api/chat/stream` (SSE) |
| `routes_conversations.py` | `POST /api/conversations`, `GET /api/conversations`, `GET /api/conversations/{id}`, `DELETE /api/conversations/{id}` |
| `routes_memories.py` | `POST /api/memories`, `GET /api/memories`, `GET/PATCH/DELETE /api/memories/{id}`, `POST /api/memories/{id}/approve`, `POST /api/memories/{id}/reject` |

`deps.py` provides the injected provider, orchestrator and memory service. Tools routes are not built yet.
