"""Regenerate the pinned dependency files for the backend.

    python scripts/update_lock.py             # keep current pins, apply pyproject.toml changes
    python scripts/update_lock.py --upgrade   # move every package to its newest allowed version

Writes backend/requirements.txt (runtime) and backend/requirements-dev.txt (runtime + dev tools).
Both are pinned to exact versions with hashes, and work on Windows, macOS and Linux
(Python 3.11 and newer). Needs `uv`: python -m pip install uv
"""

import argparse
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "backend"
MIN_PYTHON = "3.11"  # keep equal to requires-python in backend/pyproject.toml
JOBS = (("requirements.txt", ()), ("requirements-dev.txt", ("--extra", "dev")))


def uv_command() -> list[str] | None:
    """Return the command that runs uv, or None if it is not installed."""
    found = shutil.which("uv")
    if found:
        return [found]
    probe = subprocess.run(
        [sys.executable, "-m", "uv", "--version"], capture_output=True, check=False
    )
    return [sys.executable, "-m", "uv"] if probe.returncode == 0 else None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate backend/requirements*.txt")
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="move every package to the newest version pyproject.toml allows",
    )
    args = parser.parse_args(argv)

    uv = uv_command()
    if uv is None:
        print(
            "error: uv is required to refresh the lock files. Install it with: "
            "python -m pip install uv",
            file=sys.stderr,
        )
        return 2

    for output, extra in JOBS:
        command = [
            *uv,
            "pip",
            "compile",
            "pyproject.toml",
            *extra,
            "--universal",
            "--python-version",
            MIN_PYTHON,
            "--generate-hashes",
            "-o",
            output,
        ]
        if args.upgrade:
            command.append("--upgrade")
        print("+ uv", " ".join(command[len(uv) :]))
        result = subprocess.run(command, cwd=BACKEND, check=False)
        if result.returncode:
            return result.returncode

    print("Done. Review `git diff backend/requirements*.txt`, run the tests, then commit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
