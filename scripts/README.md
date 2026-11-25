Developer scripts.

- `update_lock.py`: regenerates the pinned dependency files `backend/requirements.txt` and
  `backend/requirements-dev.txt` (`python scripts/update_lock.py`, add `--upgrade` to move to the newest
  allowed versions). Needs `uv` (`python -m pip install uv`).
- `claude_pretool_guard.py`: the Claude Code `PreToolUse` hook wired up in `.claude/settings.json`.
  It blocks Claude Code shell commands that add Claude attribution to commits or PRs, commit or push
  on `main`, force-push, or read `.env` files and the local databases. Standard library only, never
  runs a process or touches the network, and fails closed (exit code 2) when it cannot decide. It
  is a guard rail, not a sandbox. Tests: `backend/tests/test_claude_guard.py` and
  `backend/tests/test_claude_settings.py`. Changes to it need review like any executable code.

More setup, test and packaging scripts will be added here.

Planned work: see `docs/06-EXECUTION-PLAN.md`.
