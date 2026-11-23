# brain

AI orchestration.

- `orchestrator.py`: builds the prompt (persona, approved memories, recent history, the user message), calls the provider, strips `<think>` reasoning from replies and streams, and merges unanswered turns.
- `prompts.py`: the AARYX persona and the block that presents memories to the model as information, not instructions.
- `schemas.py`: request, response and stream-chunk models.

Memories are passed in by the server (never by the client request).
