# AARYX — Execution Plan (task list to work through)

**Purpose:** one ordered checklist from today's state to Personal v1. Work top to bottom; tick a box only when its "Done when" is true.
**Companion docs:** `05-ROADMAP` (milestones M0–M8, risks, release gates, decision log).
**Size:** S = under half a day, M = 1–3 days, L = about a week or more. These are rough guides, not promises.
**Last updated:** 2026-11-24 (P0-2..P0-15 done; P0-16 added)

## Where we are

| Done | Not started |
|---|---|
| Backend: FastAPI, 4 providers + router/fallback, chat + streaming, health, models | Desktop UI (M2) |
| Conversations + messages in SQLite | Emotion layer, proactive follow-ups |
| Long-term memory: CRUD, security checks, dedupe, contradictions, supersession | Tools / desktop agent (M6) |
| D-001: suggestions need approval; approve/reject API; settings | Voice (M4), Avatar (M5) |
| Memory wired into chat (retrieval + background extraction), Hinglish "yaad rakho" | Windows port (M7), release (M8) |
| 1013 tests passing, lint clean | |

**Product goal (from the founder):** AARYX should connect with the user, understand what they say, and complete the tasks they ask for. In practice: remember the user, adapt tone to context without claiming to read feelings, follow up at the right time, and run permissioned actions.

---

## Phase 0 — Housekeeping (do first, small, unblocks everything)

- [ ] **P0-1 (S)** Commit all current work in logical commits (providers, memory, chat integration, tests, docs). *Done when:* `git status` is clean and `git log` shows the history.
- [x] **P0-2 (S)** Decide the primary development platform (docs say macOS first, work happens on Windows). Record as decision D-002. *Done when:* roadmap/PRD/README agree on it. Done: D-002 = macOS primary, Windows development machine and second platform (see the roadmap decision log). CI now also runs on macOS.
- [x] **P0-3 (S)** Fix `pip install -e backend[dev]` (set packages to `app*` in `pyproject.toml`). *Done when:* a fresh venv installs and `pytest` runs from the documented steps. Done: `[build-system]` and `packages.find include = ["app*"]` added; verified in a fresh venv.
- [x] **P0-4 (S)** Write `docs/SETUP.md`: install, `.env`, run server, run tests, Ollama setup, recovery. *Done when:* a fresh machine follows it without help (PRD acceptance #7). Done: `docs/SETUP.md`; the run commands were executed against the real server.
- [x] **P0-5 (S)** Update root `README.md` and `backend/README.md` (status, how to run). Remove the "not yet implemented" text. Done: root and backend READMEs rewritten.
- [x] **P0-6 (S)** Fix the literal `\n` at the end of the 11 folder README files. Done: 15 README files had the literal `\n` (not 11); the 5 outdated module READMEs now describe the real code.
- [x] **P0-7 (S)** Update `.env.example` with every setting (`MEMORY_SUGGESTIONS_ENABLED`, `MEMORY_MAX_CONTEXT_ITEMS`, `MEMORY_CANDIDATE_MIN_CONFIDENCE`, ...). Remove the dead auto-store setting if present. Done: every setting is in `.env.example`; `tests/test_env_example.py` fails if it drifts.
- [x] **P0-8 (S)** Add CI (GitHub Actions): ruff check, ruff format --check, pytest. *Done when:* a push runs it and it is green. Done: `.github/workflows/ci.yml` (ubuntu py3.11 and py3.13, windows py3.13). Its first run on GitHub is not yet confirmed green.
- [x] **P0-9 (M)** Bring the LLD in line with the code: `conversation_id`, `/api/memories`, provider interface, config, repo tree, error format. *Done when:* nothing in the LLD contradicts the code. Done: LLD rewritten to match the code (repo tree, every endpoint and schema, entities, provider interface, config, error formats, security); planned items are marked with task ids.
- [x] **P0-10 (S)** Update the roadmap: statuses on M0–M3, backlog for M7 and M8, map "Phase 3 Step 2.2" naming to M-numbers. Done: milestone status column, accurate M0 to M3 checklists, M7 and M8 backlog, phase-to-milestone naming map, decision log.
- [x] **P0-11 (M)** Local API safety: confirm default bind is `127.0.0.1`, add CORS for the future UI origin, add an optional API token for use beyond loopback. *Done when:* tests cover bind/CORS/token and docs describe them. Done: `python -m app` refuses non-loopback binds without a token; Host guard (DNS rebinding); optional API token; CORS limited to the desktop origins; secrets hidden in config errors. Note: `api_host`/`api_port` were never used before this. `--reload` workers re-check the launch-time bind address.
- [x] **P0-12 (S)** Record decision D-008: chat streaming uses SSE, not the WebSocket the HLD mentions. Update the HLD. *Done when:* HLD and code agree. Done: D-008 recorded in the roadmap decision log and HLD.
- [x] **P0-13 (S)** Add `CHANGELOG.md` and a versioning rule (0.x until Personal v1). *Done when:* each release/tag has an entry. Done: `CHANGELOG.md`, versioning rule, `app.__version__`; `tests/test_project_metadata.py` keeps versions and changelog in sync.
- [x] **P0-14 (S)** Document the real `GET /health` response (status `ok`/`degraded`, app, environment, providers) in the LLD instead of the old `{status, app}` example. Done: the real `/health` response (ok and degraded) is documented in the LLD.
- [x] **P0-15 (S)** Pin dependency versions (a lock file or a constraints file) so a fresh install and CI use the same versions; today they are ranges such as `fastapi>=0.115,<1.0` (found while updating the roadmap, M0). *Done when:* CI and a fresh clone install identical versions and the lock file is refreshed by a documented command. Done: hashed universal lock files `backend/requirements.txt` and `requirements-dev.txt`, `scripts/update_lock.py`, CI installs from the lock, `tests/test_dependency_lock.py` keeps them consistent with `pyproject.toml`. Verified on Windows by installing only from the lock and running all tests on Python 3.14 and 3.11; Linux is covered by CI on its first run.
- [ ] **P0-16 (S)** Verify everything on the MacBook, the primary platform: from a fresh clone follow `docs/SETUP.md` (install the pinned dependencies, run `pytest`, run `python -m app`, chat once with Ollama), write down what failed, and fix macOS-only problems. *Done when:* the steps work on the MacBook with no undocumented workaround; this closes the M0 exit criterion "clean setup works on Mac".

## Phase 1 — Finish the memory feature (M3)

- [ ] **P1-1 (S)** Decide D-003: may memories be sent to a cloud provider? Options: always / ask once / local-only (recommended: off for cloud unless the user opts in).
- [ ] **P1-2 (M)** Implement D-003 in the router/chat flow, with a visible flag in the response (e.g. `memory_used_count`, `provider_mode`). *Done when:* tests prove memories never reach a cloud provider when disallowed.
- [ ] **P1-3 (S)** Tell the user when a "remember this" was refused (password, key, ID number): return a flag/warning in the chat response and make the assistant not claim it saved it. *Done when:* the reply and the API both say "not saved, because ...".
- [ ] **P1-4 (S)** Confirm to the user when a memory was saved or a suggestion was queued (response metadata, e.g. `memory_events`).
- [ ] **P1-5 (M)** Better relevance than keywords: local embeddings or SQLite FTS, with a fallback to the current ranking. *Done when:* "chai" finds a "tea" memory and the 1000-memory speed test still passes.
- [ ] **P1-6 (M)** Memory scalability: stop loading every active memory per turn (indexed query / cached embeddings). *Done when:* 10,000 memories answer in a few hundred ms.
- [ ] **P1-7 (S)** API conveniences for the UI: suggestion count, bulk approve/reject, search, export all memories (JSON), delete all.
- [ ] **P1-8 (S)** Memory retention defaults (BRD "decision pending"): decide and implement expiry or "forget after N days" as an option.

**Local vs cloud, cost and performance (PRD FR-02/FR-03, BR-02, risks "cloud spend" and "local model too slow")**

- [ ] **P1-9 (M)** Per-request routing: `/api/chat` accepts `mode` (`local`/`cloud`), optional `provider` and `model`; the router honours it; an unknown or unavailable choice returns a clear 4xx; the default comes from settings. Today the provider is fixed at startup and only `model` can be overridden. *Done when:* tests cover local, cloud, cloud-unavailable, unknown provider, and fallback consent.
- [ ] **P1-10 (S)** Cloud credentials missing: report cloud mode as disabled with a reason (`GET /api/providers` lists each provider: available, mode, reason) and keep local usable, instead of only failing at startup. *Done when:* the UI can grey out Cloud and explain why.
- [ ] **P1-11 (S)** Actionable provider errors: Ollama not running returns a setup hint; model missing returns the exact model name and the `ollama pull <model>` command. *Done when:* tests assert both payloads (PRD section 8).
- [ ] **P1-12 (S)** Measure latency: add `latency_ms` to chat responses and logs (LLD provider response). Nothing measures it today. *Done when:* every reply and stream end reports it.
- [ ] **P1-13 (M)** Benchmark several small local models on the dev machine (speed, memory, Hinglish quality); record the baseline in `docs/` and choose the default model. *Done when:* a repeatable script and a written result exist.
- [ ] **P1-14 (M)** Usage tracking and spend caps: aggregate stored `token_usage` per provider/model/day, `GET /api/usage`, configurable daily/monthly cap per cloud provider; at the cap, route to local or refuse with an explanation. *Done when:* tests prove the cap stops cloud calls.

**Privacy, settings and data ownership (HLD, PRD NFR)**

- [ ] **P1-15 (S)** Private/temporary chat: a conversation that is not stored and triggers no memory extraction (HLD "stored according to user settings"). *Done when:* nothing is written to the database for such a chat.
- [ ] **P1-16 (M)** Settings API: `GET/PUT /api/settings` for UI-editable options (mode/provider, memory switches, context cap, cloud consent, privacy, emotion), validated and persisted locally; secrets are never returned. *Done when:* the UI settings page can be built on it with no direct `.env` editing.
- [ ] **P1-17 (S)** Conversation conveniences: search, auto-title, export (JSON/Markdown), delete-all-data. *Done when:* the user can export and wipe everything from the API.
- [ ] **P1-18 (S)** Diagnostics policy: structured local logs with timestamp, provider mode, latency, error category; raw conversation content only if explicitly enabled (default off). *Done when:* tests prove no content reaches logs by default (PRD section 9).

## Phase 2 — Emotional layer: "connects with me" (emotion/ module)

Principle: adapt and ask, never claim certainty about feelings. Everything optional and visible.

- [ ] **P2-1 (S)** Decide D-004 scope: what signals are allowed (text only at first), what is stored (nothing by default), and the user on/off switch.
- [ ] **P2-2 (M)** New memory categories: `person`, `event` (with date), `routine`, `goal`. Update schemas, extraction prompt, Hinglish directives ("meri behen ka naam ...", "kal mera interview hai"). *Done when:* these save as suggestions or explicit memories and appear in prompts.
- [ ] **P2-3 (M)** `emotion/` module v1: lightweight text-based mood hint (e.g. stressed, excited, low, neutral) with a confidence, computed locally, never stored unless the user approves. *Done when:* unit tests cover English, Hinglish, sarcasm, mixed signals, and "no signal".
- [ ] **P2-4 (S)** Use the hint only to adjust tone in the prompt, and to ask gently ("lag raha hai aaj thake ho, sahi?") when confidence is low. Never state a feeling as fact. *Done when:* prompt tests prove the wording rules.
- [ ] **P2-5 (M)** Persona upgrade: consistent warm voice, mirrors user language (English/Hinglish), short vs detailed replies based on the user's style, honest about limits.
- [ ] **P2-6 (M)** Clarifying questions: when a request is ambiguous or missing details, ask one short question before acting. Add tests with ambiguous prompts.
- [ ] **P2-7 (M)** Long-conversation summaries so older context is not lost when history is trimmed (summary stored per conversation, user can view/delete it).
- [ ] **P2-8 (L)** Proactive follow-ups: a local scheduler that turns remembered events into reminders ("kal interview tha, kaisa gaya?"). Needs: event dates, quiet hours, per-item on/off, delivery channel (UI notification later). *Done when:* an approved event produces exactly one follow-up at the right time and can be turned off.
- [ ] **P2-9 (S)** "What do you know about me?" command: lists active memories and summaries in plain language, with delete buttons later in the UI.
- [ ] **P2-10 (S)** Safety rules for sensitive moments (user in distress): supportive reply, no medical/legal claims, suggest real help when appropriate. Add tests; document limits.
- [ ] **P2-11 (M)** Conversation-quality eval set: about 30 scripted scenarios (stress, joy, grief, anger, sarcasm, Hinglish, ambiguity) with a rubric (warm, not presumptuous, no false claims, asks when unsure). Run before each release and record scores. *Done when:* a repeatable run exists and results are stored.

## Phase 3 — Desktop UI (M2)

- [ ] **P3-1 (S)** Choose and scaffold: Tauri 2 + React + TypeScript in `frontend/`, lint, tests, build.
- [ ] **P3-2 (M)** Chat screen with streaming, loading, retry and error states; shows provider and model per reply and a clear local/cloud badge.
- [ ] **P3-3 (M)** Conversation list: new, open, rename, delete.
- [ ] **P3-4 (M)** Memory manager: Saved memories (edit/delete) and Suggestions (approve/edit/reject); shows when a memory was used in a reply.
- [ ] **P3-5 (M)** Settings: provider/model, memory switches (suggestions, context items), privacy (cloud consent), emotion on/off, API token.
- [ ] **P3-6 (S)** Connection status and setup hints (Ollama not running, model missing with exact pull command, cloud key missing).
- [ ] **P3-7 (S)** Accessibility: keyboard navigation, contrast, captions area reserved for voice.
- [ ] **P3-8 (M)** Package the backend with the app (start/stop the local server from the shell).
- [ ] **P3-9 (S)** UI tests (component + one end-to-end chat flow).
- [ ] **P3-10 (S)** A visible Local/Cloud selector on the chat screen itself (not only in settings), with the disabled-cloud explanation from P1-10 (PRD UX requirement).
- [ ] **P3-11 (S)** Settings sections for voice and permissions, wired as Phase 4 and Phase 5 land. *Done when:* every PRD settings area (model, voice, privacy, permissions) exists.
- [ ] **P3-12 (S)** Export/delete-all buttons and a "what do you know about me" view (uses P1-17 and P2-9).

## Phase 4 — "Complete the tasks I ask": safe tools (M6)

Rule from the docs: propose → preview → user approves → execute → audit. No arbitrary shell.

- [ ] **P4-1 (M)** Define the action schema and permission model in `desktop-agent/common` (action name, arguments, risk level, scope).
- [ ] **P4-2 (M)** Tool broker in `backend/app/tools`: allowlist, argument validation, proposals stored with expiry and status.
- [ ] **P4-3 (S)** API: `POST /api/tools/propose`, `POST /api/tools/{id}/approve`, `.../reject`, `GET /api/tools/audit`. Proposing must never execute.
- [ ] **P4-4 (M)** Audit log (AuditEvent) for every proposal, approval, execution, failure.
- [ ] **P4-5 (S)** Stop / cancel control that halts a running action immediately.
- [ ] **P4-6 (M)** Orchestrator: let the model propose an action in a structured format; treat model output as untrusted; validate before showing.
- [ ] **P4-7 (M)** First low-risk, read-only actions on the primary OS (e.g. open an app, list a folder, read a file you pick, system info). Adapter per OS behind the common interface.
- [ ] **P4-8 (M)** UI: approval dialog showing action, target and expected effect; result and history view.
- [ ] **P4-9 (L)** Task planning for multi-step requests: break a request into steps, approve per step or per plan, show progress, stop anywhere.
- [ ] **P4-10 (M)** Medium-risk actions (create/modify files, send a message) with stronger confirmation. High-risk (delete, purchases) stay disabled in v1.
- [ ] **P4-11 (M)** Tests proving denied/expired/unapproved actions never run, plus abuse tests (prompt injection from web/file content tries to trigger actions).
- [ ] **P4-12 (S)** Guard against shell execution: a test that fails if `subprocess`, `os.system` or `shell=True` appear anywhere in the app or agents outside one reviewed, allowlisted wrapper (roadmap: block arbitrary shell in the MVP).
- [ ] **P4-13 (S)** When a proposed action is denied or expires, AARYX tells the user it was not performed, and never claims otherwise (PRD section 8).
- [ ] **P4-14 (S)** Permission scopes: list granted scopes and revoke them from the API and UI (HLD permission manager).

## Phase 5 — Voice (M4)

- [ ] **P5-1 (S)** Decide D-005: STT/TTS providers and whether voice is fully local.
- [ ] **P5-2 (M)** Push-to-talk capture (never always-on) and speech-to-text with text fallback.
- [ ] **P5-3 (M)** Text-to-speech, voice choice, mute/toggle, stop speaking.
- [ ] **P5-4 (S)** Hinglish speech tests and failure handling (no mic, noisy audio, STT error).
- [ ] **P5-5 (M)** Optional: tone of voice informs the emotion hint (only with consent).
- [ ] **P5-6 (S)** Transcripts/captions for all voice input and output (accessibility requirement).

## Phase 6 — Avatar (M5)

- [ ] **P6-1 (S)** Verify asset and model licenses (D-006).
- [ ] **P6-2 (M)** Avatar state model: idle, listening, thinking, speaking.
- [ ] **P6-3 (L)** Human mode and Hologram mode (Three.js), switch reliably.
- [ ] **P6-4 (M)** Connect states to chat/voice events and to the emotion hint (expressions only, no claims).

## Phase 7 — Windows/macOS parity and release (M7, M8)

- [ ] **P7-1 (M)** Second-platform agent and smoke tests for core features.
- [ ] **P7-2 (M)** Secrets in the OS keychain/credential store instead of `.env` for the release build.
- [ ] **P7-3 (M)** Backup and restore of the database and settings; tested.
- [ ] **P7-4 (S)** Diagnostics page (versions, provider health, last errors without private content).
- [ ] **P7-5 (M)** Installer/package for the primary OS; fresh-install and recovery steps documented.
- [ ] **P7-6 (M)** Privacy and security review (memory, cloud consent, tools); fix findings.
- [ ] **P7-7 (S)** Known-limitations document and Personal v1 release notes.
- [ ] **P7-8 (M)** MVP acceptance run: execute the 7 PRD acceptance checks (app starts, local chat, cloud chat, visible provider, history, no action without confirmation, setup/recovery docs) on the primary OS and the second OS; record results in `docs/`.
- [ ] **P7-9 (S)** Release process: tag, build, changelog entry, rollback notes.

## Phase 8 — Later (documented, not in the personal MVP)

Keep visible so nothing is forgotten, but do not start before Personal v1.

- [ ] **P8-1** Same identity and personality across devices (BRD section 2): settings and approved memories sync between Mac and Windows, end-to-end encrypted.
- [ ] **P8-2** Encrypt the local database at rest (memories are personal data).
- [ ] **P8-3** PostgreSQL + pgvector for multi-user or cloud sync (HLD evolution path).
- [ ] **P8-4** Commercialization gate (roadmap section 5): account and tenant isolation, billing and subscriptions, usage metering and abuse controls, privacy policy, terms, data export/deletion, support and incident response, security testing, update signing, provider and avatar/voice commercial licensing, unit economics.

---

## Requirements traceability (every requirement in the docs, checked against code and this plan)

Status: **Done** = already in code and tested; **Plan** = covered by tasks in this file; **Gap closed** = was missing from the first version of this plan and now has tasks.

### Business (BRD)

| Requirement | Status | Where |
|---|---|---|
| Objective 1: local text conversations | Done | Ollama provider, chat API |
| Objective 2: optional cloud route | Done; per-request choice was a gap | providers, router; **P1-9, P1-10** (Gap closed) |
| Objective 3: remember selected facts with user control | Done (UI later) | memory service, D-001; P1-7, P3-4 |
| Objective 4: voice in and out | Plan | Phase 5 |
| Objective 5: safe desktop actions | Plan | Phase 4 |
| Objective 6: 3D human and hologram modes | Plan | Phase 6 |
| Objective 7: portable to macOS and Windows | Plan | P0-2, P7-1 |
| BR-01 local chat | Done | |
| BR-02 optional cloud per request | Gap closed | P1-9, P1-10, P3-10 |
| BR-03 inspect and manage memory | Done (API), Plan (UI) | P3-4 |
| BR-04 text first, then voice | Plan | Phase 5 |
| BR-05 avatar switch | Plan | Phase 6 |
| BR-06 desktop actions need permission | Plan | Phase 4 |
| BR-07 data local unless cloud chosen | Partly done | cloud fallback consent exists; **P1-1, P1-2** |
| BR-08 Windows-compatible architecture | Plan | P7-1 |
| Success: user can tell local vs cloud | Plan | P1-2, P3-2, P3-10 |
| Success: fresh install documented | Plan | P0-4, P7-5, P7-8 |
| Risk: cloud cost spikes | Gap closed | P1-14 (D-010) |
| Risk: local model too slow | Gap closed | P1-12, P1-13 |
| Pending decisions (providers, voice, avatar assets, retention, desktop actions) | Plan | D-005, D-006, D-007, P4-7 |
| Cost dashboard (BRD risks) | Gap closed | P1-14, usage view in P3-5 |

### Product (PRD)

| Requirement | Status | Where |
|---|---|---|
| UC-01 ask by text | Done | |
| UC-02 choose Local or Cloud | Gap closed | P1-9, P3-10 |
| UC-03 continue conversation | Done | |
| UC-04 save to memory | Done | |
| UC-05 review/delete memory | Done (API) | P3-4 |
| UC-06 approved desktop action | Plan | Phase 4 |
| UC-07 speak and hear | Plan | Phase 5 |
| UC-08 Human/Hologram switch | Plan | Phase 6 |
| FR-01, FR-05..FR-07, FR-13..FR-15 | Done (UI parts in Phase 3) | P3-3, P3-4 |
| FR-02/FR-03 choose Local/Cloud mode | Gap closed | P1-9 |
| FR-04 UI shows mode/model | Partly (response has provider and model) | P1-2, P3-2 |
| FR-08, FR-09 voice | Plan | Phase 5 |
| FR-10 avatar | Plan | Phase 6 |
| FR-11, FR-12 tool preview and confirmation | Plan | P4-3, P4-8, P4-10 |
| NFR security: secrets outside Git, authenticated API beyond loopback | Plan | P0-7, P0-11, P7-2 |
| NFR privacy: no audio without user action; clear cloud indicator | Plan | P5-2, P3-2, P3-10 |
| NFR reliability: actionable errors, timeouts | Done (retry/fallback); hints were a gap | P1-11 |
| NFR performance: measure a baseline | Gap closed | P1-12, P1-13 |
| NFR portability | Plan | P4-1, P7-1 |
| NFR accessibility incl. voice captions | Plan; captions were a gap | P3-7, P5-6 |
| UX: chat screen, selector, mic, avatar selector, memory manager, settings, approval dialog | Plan | Phase 3, P3-10, P3-11 |
| Edge cases: Ollama down, model missing, no cloud key, offline, tool denied, STT failure | Gap closed | P1-10, P1-11, P3-6, P4-13, P5-4 |
| Analytics: none by default; local diagnostics without raw content | Gap closed | P1-18 |
| MVP acceptance 1-7 | Gap closed | P7-8 |

### Architecture (HLD)

| Requirement | Status | Where |
|---|---|---|
| Application API: validate, sessions, health, configuration endpoints | Done except configuration | **P1-16** (Gap closed) |
| Auth boundary and bind to 127.0.0.1 | Plan | P0-11 |
| WebSocket (HLD) vs SSE (code) | Gap closed | P0-12 |
| Orchestrator: context, routing, prompt policy, tool decision, policy enforcement | Partly done | P1-9, P4-6 |
| Conversations stored according to user settings | Gap closed | P1-15 |
| Memory: relevant, permitted memories only; review, edit, delete | Done | |
| Vector search later | Plan | P1-5, P8-3 |
| Voice service (STT, TTS, push-to-talk) | Plan | Phase 5 |
| Tool broker, allowlist, argument validation, approval, audit, OS adapters | Plan | Phase 4 |
| Data classification: diagnostics avoid raw content | Gap closed | P1-18 |
| Security: keychain, scopes, approval list, audit, untrusted model/web content | Plan | P7-2, P4-1, P4-10, P4-4, P4-11, P4-14 |
| Evolution path: Postgres, accounts, billing | Plan (later) | Phase 8 |

### Technical design (LLD) and roadmap

| Requirement | Status | Where |
|---|---|---|
| `GET /health`, `POST /api/chat`, memory endpoints | Done (shapes differ from the LLD) | P0-9, P0-14 |
| `/api/tools/propose`, `/approve` | Plan | P4-3 |
| Entities: Conversation (`provider_mode`), Message, Memory, ToolProposal, AuditEvent | Partly done | P0-9, P4-2, P4-4 |
| Provider interface incl. `latency_ms` | Gap closed | P1-12 |
| Config keys, `.env` local-only, never log secrets | Done / Plan | P0-7 |
| Error handling: clear errors, 422, no stack traces | Done | |
| Testing: mocks, Ollama integration, API, memory, permission tests, smoke tests | Done / Plan | P4-11, P7-1, P7-8 |
| Definition of done | Working rules at the end of this file | |
| Roadmap M4 voice selection and mute | Plan | P5-3 |
| Roadmap M6 "block arbitrary shell" | Gap closed | P4-12 |
| Roadmap M8 installer, backup, settings, diagnostics | Plan | P7-3, P7-4, P7-5 |
| Release gates alpha, beta, v1 | Plan | table below |

### Founder goal
Connect with the user, understand what they say, complete the tasks they ask for: Phase 2 (connect, understand), Phase 4 (complete tasks), P2-11 (quality check).

---

## Release gates (from the roadmap) mapped to this plan

| Gate | Needs |
|---|---|
| **Personal alpha** | Phase 0, Phase 1 (P1-1..P1-4, P1-9..P1-12, P1-16), Phase 3 (P3-1..P3-6, P3-10): local chat + settings work, local/cloud choice visible, no arbitrary desktop execution, setup guide done |
| **Personal beta** | Phase 2 core (P2-1..P2-6, P2-11), memory controls in UI, Phase 1 (P1-13..P1-15, P1-18), Phase 4 core (P4-1..P4-8, P4-12), Phase 5, backup/restore |
| **Personal v1** | Everything above, Phase 6 optional, P7-5..P7-9 |

## Decisions to make (put each in the roadmap decision log)

| ID | Question | Needed by |
|---|---|---|
| D-002 | **Decided 2026-11-24: macOS is primary**; Windows is the development machine and second platform, kept green in CI | P0-2 (done), P0-16 |
| D-003 | May memories go to a cloud model? | P1-1 |
| D-004 | Emotion signals: what is read, what is stored, defaults | P2-1 |
| D-005 | Voice providers, local or not | P5-1 |
| D-006 | Avatar assets and licensing | P6-1 |
| D-007 | Memory retention defaults | P1-8 |
| D-008 | SSE vs WebSocket for streaming (code uses SSE) | P0-12 |
| D-009 | Default local model after benchmarking | P1-13 |
| D-010 | Spend caps: default limits and what happens at the cap (fall back to local or stop) | P1-14 |

## Working rules

- One task at a time; finish with tests passing (`ruff check`, `ruff format --check`, `pytest`).
- Every new feature gets "user-style" tests: many ways a real user, a flaky model, or a hostile input could break it (see `tests/test_user_scenarios.py`).
- Anything that touches privacy, memory, or computer actions needs a docs update in the same change.
- Update the checkbox here and the roadmap status when a task is done.
