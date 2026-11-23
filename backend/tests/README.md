# tests

Run from the `backend/` folder after installing the pinned dependencies (see `docs/SETUP.md`):

```bash
pytest                      # all tests (needs no Ollama and no internet)
pytest -m integration       # only tests that talk to a real local Ollama
ruff check app tests && ruff format --check app tests
```

Notable files:

- `test_user_scenarios.py`: attacks each feature the way a real user, a flaky model or a hostile input could (phrasing, API misuse, concurrency, settings).
- `test_chat_memory_integration.py`: memory in the chat flow (what reaches the prompt, post-turn suggestions).
- `test_memory_*.py`: memory rules (security, duplicates, contradictions, supersession, suggestions).
- `test_provider_*.py`, `test_*_provider.py`, `test_router_*.py`: adapters, routing, fallback, error handling.
- `test_env_example.py`: keeps `.env.example` in sync with the settings.

`conftest.py` blocks real Ollama calls from memory extraction so tests never send data to a local model.
