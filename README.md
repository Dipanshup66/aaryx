# AARYX — Personal AI Companion

**Tagline:** More Than AI. Your Intelligent Companion.

AARYX is a personal-first AI companion project designed for macOS and Windows. The initial architecture supports local AI through Ollama and optional cloud AI through provider adapters. Future modules will add voice, user-controlled memory, avatar modes, and permission-based desktop automation.

## Project status

**Stage:** Planning and repository foundation  
**Current release:** Not yet implemented  
**First development target:** macOS  
**Future target:** Windows compatibility

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

The application code is intentionally scaffolded, not implemented yet. Start with the documentation and then implement Milestone M0/M1 from the roadmap.

Backend setup instructions will be added with the first working backend milestone.
