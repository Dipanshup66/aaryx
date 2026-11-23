# Changelog

All notable changes to AARYX are recorded here, newest first. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Versioning rule

- Versions look like `0.MINOR.PATCH` until Personal v1, which is `1.0.0`.
- Raise **MINOR** for a new user-visible feature, or for any change that can break a client or an existing `.env` (renamed route, removed or renamed setting, changed response).
- Raise **PATCH** for bug fixes, security fixes that change no contract, documentation and tests.
- The version is defined in two places that must match: `backend/pyproject.toml` and `backend/app/__init__.py`. `tests/test_project_metadata.py` fails if they differ or if this file has no entry for the current version.
- Write entries for users (what changed and why it matters), not file lists. Put new work under **Unreleased** as you go.
- To release: move the **Unreleased** entries under a new `## [x.y.z] - YYYY-MM-DD` heading, bump both version files, commit as `release: vx.y.z`, and tag `vx.y.z`.

## [Unreleased]

### Added
- Long-term memory in the chat flow: relevant approved memories are added to each prompt, and AARYX proposes new memories after each completed reply.
- Memory suggestions: things AARYX infers are only suggestions until you approve them (`POST /api/memories/{id}/approve`) or reject them (`.../reject`). Rejected suggestions are never proposed again.
- "Remember this" requests are recognised in English and Hinglish ("yaad rakho ki ...", "dhyan rakho", "mat bhoolna", "note kar lo").
- Settings `MEMORY_SUGGESTIONS_ENABLED`, `MEMORY_MAX_CONTEXT_ITEMS`, `MEMORY_EXTRACTION_MODEL`.
- Local API protection: `python -m app` launcher that refuses non-loopback addresses without a token, Host header guard against DNS rebinding, optional API token (`API_TOKEN`), CORS limited to the desktop app (`API_CORS_ORIGINS`, `API_ALLOWED_HOSTS`).
- Pinned dependencies: `backend/requirements.txt` and `requirements-dev.txt` (exact versions with hashes, for Windows, macOS and Linux, Python 3.11 and newer) and `scripts/update_lock.py` to refresh them. Installs and CI now use them.
- `docs/SETUP.md`, `docs/06-EXECUTION-PLAN.md`, GitHub Actions CI (Linux, macOS, Windows), `.env.example` covering every setting (kept in sync by a test).
- About 1100 automated tests, including a user-scenario suite.

### Changed
- Inferred memories are never active on their own; the confidence-based auto-activation and `MEMORY_AUTO_STORE_MIN_CONFIDENCE` were removed.
- Install with `pip install --require-hashes -r requirements-dev.txt` then `pip install --no-deps -e .` (see `docs/SETUP.md`).
- Run the server with `python -m app` (it honours `API_HOST` and `API_PORT`). Settings now reject a rejected value in their error messages instead of echoing it.
- LLD rewritten to match the code; READMEs describe what exists today.

### Fixed
- Tests no longer depend on `API_*` settings exported in the shell or written in `backend/.env` (they used to fail with `400 Invalid host` or `401`).
- `pip install -e ".[dev]"` failed because package discovery picked up the `data/` folder.
- An app built with its own settings (`create_app(settings)`) ignored them at startup and used the cached environment settings instead, so for example `memory_enabled=False` still created the database and a memory service. Startup, the memory service, provider routing, logging and `/health` now all use the app's own settings.
- Parallel turns could store the same memory suggestion several times, and a double click could approve one suggestion twice.
- Plurals, inflections and Devanagari text now match memories ("gardens" finds "garden"); identical memories appear once in a prompt; a runaway model can queue at most 5 suggestions per turn.

### Security
- `python -m app --reload`: every restarted worker re-checks the address the launcher bound, so removing `API_TOKEN` from `.env` while the server listens on an external address can no longer leave the API open without a password (the worker refuses to start).
- The API only answers to its configured Host names, and can require a bearer token on every `/api` call.
- Settings validation errors never print API keys or the API token.

## [0.1.0] - 2026-11-13

First backend release (commit `1fa98ed`).

### Added
- FastAPI application with structured logging, configuration and health endpoint.
- Model providers for Ollama (local), OpenAI, Anthropic and Gemini, a provider factory, a router with retries and consent-based cloud fallback, and the AI Brain orchestrator.
- Persistent conversations in SQLite with atomic turn preparation and removal of reasoning tags from replies.
- Long-term memory foundation: CRUD API, explicit "remember this" parser, local inferred extraction with Ollama, protection against storing financial and national ID numbers, duplicate detection and contradiction resolution.
