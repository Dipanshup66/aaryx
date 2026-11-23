# AARYX — Personal AI Companion

**Tagline:** More Than AI. Your Intelligent Companion.

AARYX is a personal-first AI companion project designed for macOS and Windows. The initial architecture supports local AI through Ollama and optional cloud AI through provider adapters. Future modules will add voice, user-controlled memory, avatar modes, and permission-based desktop automation.

## Project status

**Stage:** Backend working (pre-alpha). The desktop app, voice, avatar and desktop tools are not built yet.  
**Current release:** none  
**Target platforms:** macOS and Windows (primary platform still to be decided, see D-002 in `docs/06-EXECUTION-PLAN.md`)

### What works today (backend)

- Chat API with streaming, on a local Ollama model or a cloud provider (OpenAI, Anthropic, Gemini), with retries and an optional, consent-based fallback.
- Conversation history saved locally in SQLite.
- Long-term memory you control: say "remember that ..." (English or Hinglish) and it is saved; things AARYX infers are only suggestions until you approve them; approved memories are added to later prompts; everything can be edited or deleted.
- Safety checks: passwords, keys and ID numbers are never stored as memories; API keys are redacted from logs.
- About 1000 automated tests.

### What is next

See `docs/06-EXECUTION-PLAN.md` for the full task list (settings API and local/cloud choice, emotional layer, desktop UI, permissioned desktop tools, voice, avatar, release).

## Repository map

```text
AARYX/
├── docs/                  # BRD, PRD, HLD, LLD and roadmap
├── backend/               # Python + FastAPI application
│   ├── app/
│   │   ├── api/           # HTTP routes
│   │   ├── core/          # Configuration, logging, shared setup
│   │   ├── brain/         # AI orchestration and routing
│   │   ├── providers/     # Ollama and cloud model adapters
│   │   ├── memory/        # Conversation and long-term memory
│   │   ├── voice/         # Speech-to-text and text-to-speech
│   │   ├── emotion/       # Emotion/context signals
│   │   └── tools/         # Permissioned tool execution
│   └── tests/             # Backend tests
├── frontend/              # React + TypeScript desktop UI (planned)
├── desktop-agent/
│   ├── common/            # Shared tool contracts
│   ├── macos/             # macOS-specific agent
│   └── windows/           # Windows-specific agent
├── infrastructure/        # Local development and deployment config
└── scripts/               # Developer helper scripts
```

## Read the documents first

1. `docs/01-BRD-Business-Requirements.md`
2. `docs/02-PRD-Product-Requirements.md`
3. `docs/03-HLD-System-Architecture.md`
4. `docs/04-LLD-Technical-Design.md`
5. `docs/05-ROADMAP-Backlog-and-Risks.md`
6. `docs/06-EXECUTION-PLAN.md` (the task list to work through)
7. `docs/SETUP.md` (install, run, back up, troubleshoot)

## Planned technology

- Backend: Python 3.11+ and FastAPI
- Local AI: Ollama
- Cloud AI: provider-neutral adapter
- Desktop shell: Tauri 2
- Frontend: React + TypeScript
- Initial local persistence: SQLite
- Future multi-user persistence: PostgreSQL + pgvector
- Voice: Whisper-compatible STT and selected TTS provider
- Avatar: Three.js

## Development principles

- Build and verify one milestone at a time.
- Local-first by default; clearly indicate when data is sent to a cloud provider.
- Never commit `.env`, API keys, tokens, or private user data.
- No unrestricted shell access.
- Desktop actions must use an allowlist and explicit approval for sensitive operations.
- Keep macOS- and Windows-specific execution behind separate adapters.

## Getting started

Quick start (details, Ollama setup and troubleshooting are in `docs/SETUP.md`):

```bash
cd backend
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
cp .env.example .env
python -m app
```

Then open <http://127.0.0.1:8000/docs>. Run the tests with `pytest` from the `backend/` folder.
