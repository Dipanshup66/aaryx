# AARYX — Product Requirements Document (PRD)

**Version:** 1.0  
**Product:** AARYX Personal AI Companion  
**Release target:** Personal MVP  
**Platform:** macOS first; Windows next  
**Primary experience:** Text-first AI companion with local/cloud model routing

## 1. Product summary
AARYX is a desktop AI companion with a consistent identity, conversational interface, configurable model backend, user-controlled memory, and permissioned tools. The product will evolve through staged releases.

## 2. Product principles
1. **Personal-first:** solve the founder's real daily workflow before adding commercial features.
2. **Local-first:** use local inference by default when practical.
3. **Transparent:** show whether a response uses local or cloud processing.
4. **Permissioned:** the AI may propose actions, but sensitive actions require user approval.
5. **User-owned memory:** memory must be reviewable, editable, and deletable.
6. **Modular:** voice, avatars, memory, and agents are separate modules.

## 3. Personas and primary use cases
### Persona A — Primary user
Wants to talk to AARYX, ask questions, get help with coding/research, and automate repetitive desktop tasks.

### Use cases
- UC-01: Ask AARYX a question using text.
- UC-02: Select Local or Cloud mode.
- UC-03: Continue a conversation with recent context.
- UC-04: Save a useful preference or fact to memory.
- UC-05: Review or delete saved memory.
- UC-06: Ask AARYX to perform an approved desktop action.
- UC-07: Speak to AARYX and hear a spoken reply.
- UC-08: Switch between Human and Hologram visual modes.

## 4. MVP scope and release boundaries
### MVP-1: AI foundation
- FastAPI backend.
- Ollama local model adapter.
- Provider-neutral cloud adapter.
- Chat API.
- Configuration and health endpoint.
- Basic logs and error handling.

### MVP-2: Desktop chat
- Desktop shell and chat UI.
- Conversation list and session handling.
- Local/cloud mode selector.
- Connection status and error messages.

### MVP-3: Memory
- Conversation persistence.
- Explicit memory save.
- Memory list, edit, and delete.
- Retrieval of relevant approved memories.

### MVP-4: Voice and visual identity
- Push-to-talk voice capture.
- Speech-to-text.
- Text-to-speech.
- Human/Hologram mode selector.
- Avatar animation states: idle, listening, thinking, speaking.

### MVP-5: Safe desktop tools
- Small allowlisted action set.
- Preview of proposed action.
- Confirmation before execution.
- Execution result and audit log.
- Emergency stop.

## 5. Functional requirements
| ID | Requirement | Priority | Acceptance criteria |
|---|---|---|---|
| FR-01 | User can send a text message. | Must | A valid request returns an assistant reply. |
| FR-02 | User can choose Local mode. | Must | Requests route to configured Ollama model. |
| FR-03 | User can choose Cloud mode. | Should | Requests route to configured compatible cloud endpoint. |
| FR-04 | UI shows active mode/model. | Must | Each response includes provider mode and model label. |
| FR-05 | Conversation history is persisted. | Should | User can reopen a prior session. |
| FR-06 | User can save a memory explicitly. | Should | Saved item appears in memory manager. |
| FR-07 | User can delete a memory. | Must | Deleted memory is not returned by retrieval. |
| FR-08 | Voice input is push-to-talk. | Should | Recording starts/stops only through user action. |
| FR-09 | Spoken response can be enabled/disabled. | Should | User has a voice output toggle. |
| FR-10 | Avatar mode can be switched. | Could | Human and Hologram modes are selectable. |
| FR-11 | Desktop tool action is previewed. | Must | User sees action details before approval. |
| FR-12 | Sensitive tool action requires confirmation. | Must | No action executes without confirmation. |

## 6. Non-functional requirements
- **Security:** secrets stored outside source control; least privilege; authenticated local API if exposed beyond loopback.
- **Privacy:** no audio recording without explicit user interaction; clear cloud-routing indicator.
- **Reliability:** backend returns actionable errors; provider timeout handling.
- **Performance:** local response latency depends on hardware/model; measure baseline rather than promise a fixed time.
- **Portability:** isolate OS-specific functions behind a desktop-agent interface.
- **Maintainability:** typed request/response schemas, tests, modular adapters, documented configuration.
- **Accessibility:** keyboard navigation, readable contrast, captions/transcripts for voice.

## 7. UX requirements
- Main chat screen with message history.
- Visible Local / Cloud selector.
- Visible microphone control; no hidden always-on recording.
- Avatar mode selector.
- Memory manager.
- Settings page for model, voice, privacy, and permissions.
- Tool approval dialog with action, target, and expected effect.

## 8. Error and edge cases
- Ollama not running: show connection issue and setup hint.
- Model missing: show exact model name and pull instruction.
- Cloud credentials missing: disable Cloud mode and explain configuration.
- Network unavailable: Local mode remains usable if Ollama is running.
- Tool denied: report that action was not performed.
- Speech recognition failure: allow text fallback.

## 9. Analytics for personal MVP
No third-party analytics by default. Local diagnostic logs may record timestamps, provider mode, latency, and error category, but should not store raw private conversation content unless explicitly enabled.

## 10. MVP release acceptance
The MVP is acceptable when:
1. App starts on the Mac.
2. Local chat works end-to-end.
3. Cloud mode works with a configured provider.
4. User can see the active provider.
5. Conversation history works.
6. No desktop action can run without confirmation.
7. Setup and recovery steps are documented.
