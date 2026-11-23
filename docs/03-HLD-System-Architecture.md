# AARYX — High-Level Design (HLD) / System Architecture

**Version:** 1.0  
**Architecture style:** Modular monolith initially, with clear service boundaries  
**Deployment:** Local desktop application + local backend; optional cloud model APIs

## 1. Architecture goals
- Deliver a working personal assistant quickly.
- Keep local inference and personal data local by default.
- Support cloud models through replaceable adapters.
- Isolate OS-specific desktop capabilities.
- Avoid premature microservices; split services only when scale requires it.

## 2. Logical architecture

```text
┌──────────────────────────────────────────────────────┐
│                 AARYX Desktop Client                 │
│  React UI | Chat | Settings | Memory | Avatar Modes  │
└──────────────────────────┬───────────────────────────┘
                           │ localhost API / WebSocket
┌──────────────────────────▼───────────────────────────┐
│                 AARYX Application API                │
│          FastAPI | Auth boundary | Session API       │
└──────────────────────────┬───────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────┐
│                 AI Orchestrator                      │
│ Context assembly | Routing | Prompt policy | Tools   │
└───────────────┬──────────────────────────┬───────────┘
                │                          │
       ┌────────▼────────┐        ┌────────▼─────────┐
       │ Local Model     │        │ Cloud Model      │
       │ Ollama Adapter  │        │ Provider Adapter │
       └─────────────────┘        └──────────────────┘

┌──────────────────────────────────────────────────────┐
│ Supporting modules                                    │
│ Memory | Voice STT/TTS | Emotion/context | Tool broker│
│ Audit log | Permission manager | Avatar state        │
└──────────────────────────────────────────────────────┘
                           │
                 ┌─────────▼─────────┐
                 │ Desktop Agent     │
                 │ macOS / Windows   │
                 └───────────────────┘
```

## 3. Component responsibilities
### Desktop client
- Displays chat, settings, provider state, memory manager, and avatar.
- Sends user input to local API.
- Shows tool proposals and obtains approval.

### Application API
- Validates requests.
- Manages chat sessions.
- Provides health and configuration endpoints.
- Does not directly execute arbitrary shell commands.

### AI orchestrator
- Builds system and conversation context.
- Selects Local or Cloud route according to user setting.
- Calls the selected model adapter.
- Decides whether a tool proposal is needed.
- Enforces policy before any tool is invoked.

### Model adapters
- Local adapter communicates with Ollama.
- Cloud adapter uses a configured provider API.
- Both return a common internal response format.

### Memory service
- Stores conversations separately from long-term memories.
- Two memory origins: **explicit** (user said "remember this"; stored as active) and **inferred** (guessed by the extractor).
- Inferred memories are always stored as pending suggestions (`candidate`). They become `active` only after user approval (Decision D-001).
- Inferred extraction runs on the local model only; conversation content is never sent to a cloud provider for extraction, even when cloud fallback is enabled.
- An inferred memory never overwrites or supersedes an explicit memory.
- Retrieves only relevant `active` memories; suggestions, rejected and superseded memories are never used in prompts.
- Supports review, edit, approval/rejection of suggestions, and deletion.
- Suggestions can be disabled by a user setting.

### Voice service
- STT converts user-authorized audio to text.
- TTS converts assistant text to speech.
- Push-to-talk is the default interaction.

### Tool broker and desktop agent
- Exposes a small allowlist of actions.
- Validates action arguments.
- Requests approval for sensitive actions.
- Executes through OS-specific adapters.
- Writes an audit record.

## 4. Recommended technology choices
| Area | Initial choice |
|---|---|
| Desktop shell | Tauri 2 |
| UI | React + TypeScript |
| API | Python + FastAPI |
| Local LLM | Ollama |
| Cloud LLM | OpenAI-compatible adapter initially |
| Data store | SQLite for single-user MVP; PostgreSQL when multi-user/cloud sync is needed |
| Vector search | pgvector later; local vector store can be evaluated |
| Voice STT | Whisper-compatible local service |
| Voice TTS | Piper or selected commercial provider |
| Avatar | Three.js |
| Packaging | Tauri bundling |
| Tests | pytest + frontend test framework |

## 5. Data flow: text chat
1. User submits message in desktop client.
2. Client sends message and session ID to API.
3. API validates payload.
4. Orchestrator loads recent conversation and approved relevant memories.
5. Router selects Local or Cloud model.
6. Adapter sends prompt to provider.
7. Response is normalized and returned.
8. Conversation is stored according to user settings.
9. UI displays response and provider label.
10. If suggestions are enabled, the local extractor may propose memories from the turn; they are stored as pending suggestions and shown to the user for approval. They do not affect later replies until approved.

## 6. Data flow: desktop action
1. User asks for an action.
2. Orchestrator creates a structured action proposal.
3. Tool broker checks action against allowlist and permissions.
4. UI displays the exact action and target.
5. User approves or rejects.
6. Desktop agent executes approved action.
7. Result is returned and audit logged.
8. High-risk actions remain unavailable in MVP.

## 7. Data classification
- **Local private data:** conversations, memory, local file metadata, local settings.
- **Cloud-transmitted data:** only prompt/context sent to selected cloud provider. Approved long-term memories selected for a turn are part of that prompt, so they are sent to a cloud provider whenever it answers (open decision: see roadmap M3). Memory extraction itself never uses the cloud.
- **Secrets:** API keys and tokens; never put in frontend bundle or Git.
- **Diagnostics:** operational metadata; avoid raw conversation/audio by default.

## 8. Security boundaries
- Bind local API to `127.0.0.1` by default.
- Do not expose the local API to the public internet.
- Store secrets in OS keychain/credential store in the desktop release.
- Use explicit permission scopes for tools.
- Validate all tool arguments.
- Require approval for file modification, sending messages, purchases, deletion, or other consequential actions.
- Keep an audit trail for tool execution.
- Treat model output and web content as untrusted input.
- Memory extraction is local-only; inferred memories require user approval before use.

## 9. Evolution path
- Personal MVP: local FastAPI + SQLite + Ollama.
- Personal v1: Tauri desktop client, memory, voice, safe tools.
- Future commercial: account service, PostgreSQL, tenant isolation, billing, usage metering, cloud deployment, support and compliance processes.
- Do not split into microservices until real load or team boundaries justify it.
