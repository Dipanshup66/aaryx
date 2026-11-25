"""The Claude Code project settings (.claude/settings.json) and the hook command inside them.

The guard script has its own tests (test_claude_guard.py). These tests check the wiring: the file
has the right shape, and the real hook command, run through bash exactly as Claude Code runs it,
blocks with exit code 2 and fails closed when Python or the script is missing or broken.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SETTINGS_PATH = REPO / ".claude" / "settings.json"
SETTINGS = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
HOOK = SETTINGS["hooks"]["PreToolUse"][0]["hooks"][0]


def find_bash() -> str | None:
    """Git Bash on Windows (never the WSL launcher in System32), plain bash elsewhere."""
    if sys.platform == "win32":
        for candidate in (
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git" / "bin" / "bash.exe",
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git" / "usr" / "bin" / "bash.exe",
        ):  # fmt: skip
            if candidate.is_file():
                return str(candidate)
        return None
    return shutil.which("bash")


BASH = find_bash()
needs_bash = pytest.mark.skipif(BASH is None, reason="bash is not installed here")


# ------------------------------------------------------------------------------ the file


def test_settings_file_declares_its_schema_and_empty_attribution() -> None:
    assert SETTINGS["$schema"] == "https://json.schemastore.org/claude-code-settings.json"
    assert SETTINGS["attribution"] == {"commit": "", "pr": ""}


def test_there_is_exactly_one_pretooluse_hook_for_shell_tools() -> None:
    groups = SETTINGS["hooks"]["PreToolUse"]

    assert len(groups) == 1
    assert set(groups[0]["matcher"].split("|")) == {"Bash", "PowerShell"}
    assert len(groups[0]["hooks"]) == 1


def test_the_hook_fails_closed_and_cannot_hang() -> None:
    assert HOOK["type"] == "command"
    assert HOOK["onFailure"] == "block"  # a timeout or a missing command must not let it through
    assert HOOK["shell"] == "bash"  # one syntax on Windows (Git Bash) and macOS
    assert isinstance(HOOK["timeout"], int) and 1 <= HOOK["timeout"] <= 30


def test_the_hook_runs_the_guard_from_the_project_folder() -> None:
    assert "$CLAUDE_PROJECT_DIR/scripts/claude_pretool_guard.py" in HOOK["command"]
    assert (REPO / "scripts" / "claude_pretool_guard.py").is_file()


def test_the_hook_only_uses_the_project_python_guard_and_no_network_tools() -> None:
    command = HOOK["command"]

    for risky in ("curl", "wget", "nc ", "eval", "base64", "http://", "https://", "sudo", "rm "):
        assert risky not in command, risky


def test_secret_files_are_denied_but_the_example_file_is_not() -> None:
    deny = SETTINGS["permissions"]["deny"]

    assert all(rule.startswith("Read(") and rule.endswith(")") for rule in deny)
    for required in ("Read(.env)", "Read(.env.local)", "Read(.envrc)", "Read(**/data/**)",
                     "Read(*.db)", "Read(*.db-wal)", "Read(*.sqlite)", "Read(*.pem)", "Read(*.key)"):  # fmt: skip
        assert required in deny
    assert not any("example" in rule for rule in deny)
    assert len(deny) == len(set(deny))


def test_personal_overrides_are_git_ignored() -> None:
    lines = (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert ".claude/settings.local.json" in lines
    assert ".claude/settings.json" not in lines  # the shared file must be committed


# --------------------------------------------------------------- the real hook command


def make_repo(root: Path, branch: str) -> Path:
    (root / ".git").mkdir(parents=True)
    (root / ".git" / "HEAD").write_text(f"ref: refs/heads/{branch}\n", encoding="utf-8")
    return root


def run_hook(
    payload: object,
    *,
    command: str | None = None,
    project_dir: Path | None = REPO,
    raw: bytes | None = None,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env.pop("CLAUDE_PROJECT_DIR", None)
    if project_dir is not None:
        env["CLAUDE_PROJECT_DIR"] = str(project_dir)
    data = raw if raw is not None else json.dumps(payload).encode("utf-8")
    assert BASH is not None
    return subprocess.run(
        [BASH, "-c", command or HOOK["command"]],
        input=data,
        capture_output=True,
        env=env,
        timeout=60,
        check=False,
    )  # type: ignore[return-value]


def shell_payload(command: str, cwd: Path, tool: str = "Bash") -> dict:
    return {"tool_name": tool, "tool_input": {"command": command}, "cwd": str(cwd)}


@needs_bash
def test_the_hook_lets_an_ordinary_command_run(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "r", "dev")

    result = run_hook(shell_payload("git status", repo))

    assert result.returncode == 0 and result.stdout == b""


@needs_bash
@pytest.mark.parametrize("tool", ["Bash", "PowerShell"])
def test_the_hook_blocks_with_exit_code_two(tmp_path: Path, tool: str) -> None:
    repo = make_repo(tmp_path / "r", "main")

    result = run_hook(shell_payload('git commit -m "x"', repo, tool))

    assert result.returncode == 2
    assert b"no-commit-on-protected-branch" in result.stderr


@needs_bash
def test_the_hook_blocks_secret_reads(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "r", "dev")

    result = run_hook(shell_payload("cat backend/.env", repo))

    assert result.returncode == 2 and b"no-secret-read" in result.stderr


@needs_bash
def test_the_hook_ignores_other_tools(tmp_path: Path) -> None:
    result = run_hook({"tool_name": "Read", "tool_input": {"file_path": "x"}, "cwd": str(tmp_path)})

    assert result.returncode == 0


@needs_bash
@pytest.mark.parametrize("raw", [b"", b"not json", b"[]", b"{}"])
def test_the_hook_blocks_input_it_cannot_understand(raw: bytes) -> None:
    result = run_hook(None, raw=raw)

    assert result.returncode == 2


@needs_bash
def test_the_hook_blocks_when_no_python_is_found() -> None:
    command = HOOK["command"].replace("python3 python py", "no_such_python_a no_such_python_b")

    result = run_hook(
        {"tool_name": "Bash", "tool_input": {"command": "git status"}}, command=command
    )

    assert result.returncode == 2
    assert b"Python 3.11+ not found" in result.stderr


@needs_bash
def test_the_hook_blocks_when_the_script_is_missing(tmp_path: Path) -> None:
    payload = {"tool_name": "Bash", "tool_input": {"command": "git status"}}

    assert run_hook(payload, project_dir=tmp_path).returncode == 2
    assert run_hook(payload, project_dir=None).returncode == 2  # CLAUDE_PROJECT_DIR unset


@needs_bash
@pytest.mark.parametrize(
    "script",
    [
        "raise SystemExit(1)\n",  # Claude Code treats exit 1 as "allow", so the wrapper must map it
        "def broken(:\n",  # SyntaxError exits 1 as well
        "import sys\nsys.exit(3)\n",
        "raise RuntimeError('boom')\n",
    ],
)
def test_a_crashing_guard_blocks_instead_of_letting_the_command_through(
    tmp_path: Path, script: str
) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "claude_pretool_guard.py").write_text(script, encoding="utf-8")
    payload = {"tool_name": "Bash", "tool_input": {"command": "git status"}}

    result = run_hook(payload, project_dir=tmp_path)

    assert result.returncode == 2


@needs_bash
def test_a_guard_that_exits_zero_lets_the_command_through(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "claude_pretool_guard.py").write_text("pass\n", encoding="utf-8")
    payload = {"tool_name": "Bash", "tool_input": {"command": "git status"}}

    assert run_hook(payload, project_dir=tmp_path).returncode == 0
