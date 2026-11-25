# memory

Conversation storage and long-term memory, backed by local SQLite.

- `models.py`, `database.py`, `repository.py`: tables, engine and queries.
- `service.py`: conversations, memory CRUD, suggestion approve/reject, prompt memory selection, post-turn extraction. Memory writes are serialized.
- `extractor.py`: explicit "remember this" parser (English and Hinglish), local-only inferred suggestions, duplicate and contradiction detection.
- `schemas.py`: validation, including the security checks that refuse passwords, keys, ID numbers and prompt-override text.

Rules (decision D-001): a memory the user asked for explicitly is stored `active`; anything AARYX infers is only a `candidate` suggestion until the user approves it. Only `active` memories ever reach a prompt.
