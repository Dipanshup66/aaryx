# providers

Model adapters behind one interface (`base.py`): `ollama.py` (local), `openai.py`, `anthropic.py`, `gemini.py`.

- `factory.py` builds a provider from settings.
- `router.py` picks the primary provider, retries transient failures, and falls back to a second provider only when the user has allowed it (cloud fallback needs explicit consent).
- Errors are classified (connection, timeout, response) and never expose API keys.
