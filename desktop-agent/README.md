# AARYX Desktop Agent

OS-specific integrations must be isolated here.

- `common/`: shared action contracts and permission model.
- `macos/`: macOS implementation.
- `windows/`: Windows implementation.

Safety rule: do not add unrestricted shell execution. Actions must be allowlisted, validated, permission-scoped, and audited.
