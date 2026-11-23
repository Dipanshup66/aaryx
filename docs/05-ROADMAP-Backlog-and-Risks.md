# AARYX — Development Roadmap, Backlog & Risk Register

**Version:** 1.0  
**Delivery approach:** Incremental milestones; validate each milestone before expanding scope.

## 1. Delivery roadmap

| Milestone | Main deliverable | Exit criteria |
|---|---|---|
| M0 — Project setup | Repo, environment, docs, coding standards | Clean setup works on Mac |
| M1 — AI foundation | FastAPI, Ollama adapter, cloud adapter, chat endpoint | Local chat passes tests |
| M2 — Desktop UI | Tauri + React chat interface | User can chat from desktop |
| M3 — Conversation & memory | Sessions, history, explicit memory manager | User can review/delete memory |
| M4 — Voice | Push-to-talk STT and TTS | Voice conversation works with text fallback |
| M5 — Avatar | Human/Hologram switch and animation states | Both modes switch reliably |
| M6 — Safe tools | Tool proposals, permission broker, macOS agent | Approved allowlisted action executes |
| M7 — Windows port | Windows agent and packaging | Core features pass Windows smoke tests |
| M8 — Personal release | Installer, backup, settings, diagnostics | Fresh install and recovery documented |

## 2. Initial implementation backlog
### M0 — Setup
- [ ] Create Git repository.
- [ ] Add README, BRD, PRD, HLD, LLD.
- [ ] Add `.gitignore` including `.env`, `.venv`, caches, local databases.
- [ ] Create Python virtual environment.
- [ ] Pin dependencies and add test tooling.

### M1 — AI foundation
- [ ] Implement settings loader.
- [ ] Define common provider interface.
- [ ] Implement Ollama adapter.
- [ ] Implement cloud-compatible adapter.
- [ ] Implement Local/Cloud router.
- [ ] Add `/health`.
- [ ] Add `/api/chat`.
- [ ] Add tests for success, timeout, and missing provider.

### M2 — Desktop UI
- [ ] Initialize Tauri + React + TypeScript.
- [ ] Build chat screen.
- [ ] Add provider selector.
- [ ] Show provider/model on responses.
- [ ] Add loading, retry, and error states.

### M3 — Memory
- [ ] Add local database.
- [ ] Persist conversations and messages.
- [ ] Add explicit memory save flow.
- [ ] Add memory list/edit/delete.
- [ ] Add retrieval tests.
- [x] Retrieve relevant `active` memories into the system prompt on every chat turn (`/api/chat` and `/api/chat/stream`). Keyword-based and local; core memories (instructions, importance >= 4) are always eligible; capped by `MEMORY_MAX_CONTEXT_ITEMS` (default 8, 0 = off).
- [x] Run extraction after each completed turn in the background (local model only); inferred memories become `candidate` only (D-001), explicit "remember this" becomes `active`. A memory failure never breaks the chat reply.
- [x] User-scenario hardening suite (`tests/test_user_scenarios.py`, ~270 cases): remember-phrasing (English + Hinglish), hostile text, chat input abuse, approve/reject misuse, retrieval phrasing, model misbehaviour, failure recovery, concurrency, settings.
- [ ] Decide whether memories may be sent to a cloud provider (they are part of the prompt, so they are when the cloud model answers). Options: always, ask once, or local-only.
- [ ] Better relevance than keywords (embeddings / vector search) and avoid loading every active memory per turn once memories grow large.
- [x] Inferred memories can never be created or updated into `active` (D-001); the old confidence-based auto-activation and `MEMORY_AUTO_STORE_MIN_CONFIDENCE` were removed.
- [x] Approve / reject endpoints for suggestions and the `MEMORY_SUGGESTIONS_ENABLED` setting.
- [x] Tests: inferred never active, rejected never re-suggested, pending suggestions not duplicated, suggestions off creates no inferred memories (`tests/test_memory_suggestions.py`).
- [x] Rejected, pending, superseded and deleted memories are never injected into prompts (`tests/test_chat_memory_integration.py`).

### M4 — Voice
- [ ] Add push-to-talk.
- [ ] Add speech-to-text.
- [ ] Add text-to-speech.
- [ ] Add voice selection and mute controls.
- [ ] Do not enable always-on recording by default.

### M5 — Avatar
- [ ] Define avatar state model.
- [ ] Add Human mode.
- [ ] Add Hologram mode.
- [ ] Connect listening/thinking/speaking states.
- [ ] Verify asset and model licenses.

### M6 — Desktop tools
- [ ] Define allowlisted tool schema.
- [ ] Add proposal/approval flow.
- [ ] Implement low-risk read-only actions first.
- [ ] Add audit events.
- [ ] Add stop/cancel control.
- [ ] Block arbitrary shell execution in the MVP.

## 2a. Decision log
| ID | Date | Decision | Reason |
|---|---|---|---|
| D-001 | 2026-11-23 | Inferred memories are kept, but only as user-approved suggestions. They are never active without approval, extraction is local-only, and suggestions can be turned off. | Gives a companion feel while keeping memory user-owned (PRD principle 5, BR-03, BR-07). |

## 3. Risk register
| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Scope creep | High | High | Milestone gates; no feature added without priority |
| Cloud spend spikes | Medium | High | Usage caps, routing, per-model budget |
| Local model too slow | Medium | Medium | Benchmark several small models |
| Sensitive data exposure | Medium | High | Local-first, explicit cloud consent, secret storage |
| Unsafe computer actions | Medium | High | Allowlist, preview, approval, audit |
| Avatar licensing issue | Medium | Medium | Use assets with commercial-use rights |
| Cross-platform differences | High | Medium | OS abstraction and Windows test milestone |

## 4. Release gates
### Personal alpha
- Local chat and settings work.
- No arbitrary desktop execution.
- Error handling and setup guide complete.

### Personal beta
- Memory controls, voice, and avatar modes work.
- Desktop actions are permissioned and audited.
- Backup and restore are tested.

### Personal v1
- macOS package is installable.
- Windows compatibility has a defined tested scope.
- Privacy and security review complete.
- Known limitations documented.

## 5. Commercialization gate (future, not MVP)
Before accepting public customers, add:
- Account and tenant isolation.
- Billing and subscription lifecycle.
- Usage metering and abuse controls.
- Privacy policy, terms, data deletion/export.
- Support and incident response.
- Security testing and software update signing.
- Provider licensing and avatar/voice commercial rights.
- Unit economics and pricing validation.
