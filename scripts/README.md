Developer scripts.

- `update_lock.py`: regenerates the pinned dependency files `backend/requirements.txt` and
  `backend/requirements-dev.txt` (`python scripts/update_lock.py`, add `--upgrade` to move to the newest
  allowed versions). Needs `uv` (`python -m pip install uv`).

More setup, test and packaging scripts will be added here.

Planned work: see `docs/06-EXECUTION-PLAN.md`.
