# AARYX — Business Requirements Document (BRD)

**Document version:** 1.0  
**Project:** AARYX Personal AI Companion  
**Status:** Initial baseline  
**Primary launch platform:** macOS first; Windows-compatible architecture  
**Product direction:** Personal-first, designed so it can later become a commercial product

## 1. Executive summary
AARYX is a personal AI companion intended to help one user communicate naturally with an AI, retain user-approved context, and perform useful digital tasks through a permission-controlled desktop agent. It will support local and cloud AI models, voice interaction, selectable visual identities (3D human avatar and holographic assistant), and desktop automation.

The first release is for personal use. Commercialization is a later decision and must not compromise the privacy, security, or reliability of the personal version.

## 2. Business problem and opportunity
Users currently switch between separate tools for chat, voice, memory, research, and computer automation. AARYX aims to provide one consistent assistant experience with:
- A persistent, user-controlled personal context.
- Local-first processing where practical.
- Optional cloud intelligence for tasks that benefit from it.
- A single interface for conversation and approved actions.
- A consistent identity and personality across devices.

## 3. Vision and objectives
### Vision
“More than AI. Your intelligent companion.”

### Initial objectives
1. Have useful text conversations through a local model.
2. Offer an optional cloud-model route.
3. Remember selected facts and conversation context with user control.
4. Support voice input and spoken responses.
5. Execute a small, safe set of desktop actions only after permission.
6. Provide switchable 3D human and holographic presentation modes.
7. Keep the design portable to macOS and Windows.

## 4. Stakeholders
- Product owner / primary user: project founder.
- Development: founder initially.
- Future beta users: invited testers only after personal version is stable.
- External providers: model, speech, hosting, and payment providers if/when required.

## 5. Target users
### Initial user
A single technically capable user who wants a personal AI companion for daily conversation, learning, coding assistance, research, and desktop productivity.

### Future users
Individuals who want a privacy-aware desktop AI assistant. This is a future hypothesis, not an approved launch scope.

## 6. Business scope
### In scope for personal v1
- macOS desktop-first experience.
- Local Ollama model connection.
- Optional cloud model adapter.
- Text chat.
- Basic conversation history.
- User-approved memory.
- Voice input/output as a later v1 milestone.
- Safe desktop actions with explicit confirmation.
- Avatar mode selection.

### Out of scope for initial release
- Public signup and subscriptions.
- Team accounts and enterprise administration.
- Unattended unrestricted computer control.
- Always-on recording.
- Mobile apps.
- Training a foundation model from scratch.
- Guaranteed human-level emotion understanding.

## 7. Business requirements
| ID | Requirement | Priority |
|---|---|---|
| BR-01 | User can chat with AARYX using a local model. | Must |
| BR-02 | User can optionally route a request to a configured cloud model. | Must |
| BR-03 | User can inspect and manage saved memory. | Must |
| BR-04 | User can use text first, then voice interaction. | Should |
| BR-05 | User can switch between human-avatar and hologram modes. | Should |
| BR-06 | Desktop actions require clear permissions and confirmation. | Must |
| BR-07 | Sensitive data remains local unless the user chooses cloud processing. | Must |
| BR-08 | Architecture supports later Windows compatibility. | Should |

## 8. Success measures
- Local chat works reliably on the primary Mac.
- User can tell whether a request is processed locally or in the cloud.
- No desktop action runs without the required permission.
- User can review, correct, and delete saved memory.
- Core conversation flow remains usable if cloud services are unavailable.
- A fresh installation can be set up using documented steps.

## 9. Assumptions and constraints
- Existing MacBook and Windows laptop are available.
- Ollama is already installed on the Mac.
- Initial development is founder-led.
- Cloud usage may incur variable API charges.
- Local model speed and quality depend on device hardware and selected model.
- Voice, avatar, and automation features will be phased rather than built simultaneously.

## 10. Key risks
| Risk | Mitigation |
|---|---|
| Cloud API cost grows unexpectedly | Usage limits, local routing, cost dashboard |
| Local model quality varies | Model selection and fallback options |
| Desktop automation causes damage | Allowlist actions, sandboxing, confirmation |
| Personal data leaks to cloud | Local-first defaults, clear data-flow disclosure |
| Scope becomes too large | Milestone-based delivery and acceptance criteria |

## 11. Decisions pending
- Final local and cloud model providers.
- Voice provider and whether voice is fully local.
- Avatar asset and licensing.
- Memory retention defaults.
- Exact desktop actions permitted in v1.
