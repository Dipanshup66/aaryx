#!/usr/bin/env python3
"""AARYX project guard for Claude Code: a PreToolUse hook for Bash and PowerShell commands.

It blocks a small, fixed set of commands that break this project's rules, and nothing else:

* ``no-claude-attribution``        a git commit / tag or ``gh pr`` text that carries a Claude
                                   co-author trailer or a "Generated with Claude" footer
* ``no-commit-on-protected-branch`` ``git commit`` while on main / master
* ``no-push-to-protected-branch``  ``git push`` that updates main / master
* ``no-destructive-push``          force pushes, remote branch deletes, mirror / all pushes
* ``no-secret-read``               printing, searching or sending ``.env`` files, the local
                                   SQLite databases (personal memories) or key files

How Claude Code uses it: the hook receives JSON on stdin; exit code 2 blocks the command and the
message on stderr is shown to the model. Any other exit code lets the command run, so this
script is FAIL-CLOSED: unparsable input, a missing command, or any unexpected error exits 2.

Security notes (this file is executable code that runs on every contributor's machine):

* standard library only; it never starts a process, opens a network connection, or writes a file
* the only file it reads is the ``HEAD`` file of the git repository around the working directory
* it is a guard rail against mistakes, not a sandbox: a determined shell command can still
  disguise a path. The permissions deny rules in ``.claude/settings.json`` and the OS sandbox are
  the other layers
* messages are plain ASCII so a Windows console code page can never make the guard itself fail
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, TextIO

PROTECTED_BRANCHES = frozenset({"main", "master"})
BLOCK_EXIT_CODE = 2
CHECKED_TOOLS = frozenset({"Bash", "PowerShell"})


class GuardError(Exception):
    """The guard could not decide safely; the command is blocked."""


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    rule: str = ""
    reason: str = ""


ALLOW = Verdict(True)


def _block(rule: str, reason: str) -> Verdict:
    return Verdict(False, rule, reason)


# --------------------------------------------------------------------------------- git HEAD


@dataclass(frozen=True)
class Head:
    """Where HEAD of the surrounding repository points."""

    kind: str  # "branch", "detached" or "not_repo"
    branch: str = ""


def read_head(start: Path) -> Head:
    """Read HEAD of the repository containing ``start`` without running git.

    Works for normal repositories and linked worktrees (``.git`` may be a file). Raises
    ``GuardError`` when a repository exists but HEAD cannot be understood.
    """
    start = start.resolve()
    for directory in (start, *start.parents):
        dot_git = directory / ".git"
        try:
            if dot_git.is_dir():
                git_dir = dot_git
            elif dot_git.is_file():
                text = dot_git.read_text(encoding="utf-8", errors="replace").strip()
                if not text.startswith("gitdir:"):
                    raise GuardError("unreadable .git file")
                target = Path(text[len("gitdir:") :].strip())
                git_dir = target if target.is_absolute() else (directory / target).resolve()
            else:
                continue
            head = (git_dir / "HEAD").read_text(encoding="utf-8", errors="replace").strip()
        except OSError as exc:
            raise GuardError(f"cannot read git HEAD ({type(exc).__name__})") from exc
        if head.startswith("ref: refs/heads/"):
            return Head("branch", head[len("ref: refs/heads/") :].strip())
        if re.fullmatch(r"[0-9a-fA-F]{40,64}", head):
            return Head("detached")
        raise GuardError("unrecognised git HEAD")
    return Head("not_repo")


HeadReader = Callable[[Path], Head]


# ----------------------------------------------------------------------------- shell parsing

_HEREDOC = re.compile(
    r"<<-?[ \t]*(['\"]?)(?P<tag>\w+)\1[^\n]*\n.*?\n[ \t]*(?P=tag)[ \t]*(?=\n|$)", re.DOTALL
)
_HERE_STRING = re.compile(r"@(['\"])[ \t]*\n.*?\n\1@", re.DOTALL)  # PowerShell @' ... '@


def strip_heredocs(command: str) -> str:
    """Remove heredoc / here-string bodies so message text is not mistaken for commands."""
    command = _HEREDOC.sub(lambda m: m.group(0).split("\n", 1)[0] + "\n", command)
    return _HERE_STRING.sub("''", command)


def split_segments(command: str) -> list[list[str]]:
    """Split a command line into simple commands, each a list of unquoted tokens.

    Separators outside quotes: ``&&``, ``||``, ``;``, ``|``, ``&`` and newlines. A backslash is
    an escape only inside double quotes, so Windows paths like ``backend\\data`` stay intact.
    """
    segments: list[list[str]] = []
    tokens: list[str] = []
    buf: list[str] = []
    in_token = False
    quote = ""
    i, n = 0, len(command)

    def end_token() -> None:
        nonlocal in_token
        if in_token:
            tokens.append("".join(buf))
            buf.clear()
            in_token = False

    def end_segment() -> None:
        end_token()
        if tokens:
            segments.append(tokens[:])
            tokens.clear()

    while i < n:
        ch = command[i]
        if quote:
            if quote == '"' and ch == "\\" and i + 1 < n and command[i + 1] in '"\\$`':
                buf.append(command[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = ""
            else:
                buf.append(ch)
            i += 1
            continue
        if ch in "'\"":
            quote = ch
            in_token = True
            i += 1
            continue
        if ch in "\n;|&":
            end_segment()
            i += 2 if command[i : i + 2] in ("&&", "||") else 1
            continue
        if ch in " \t\r":
            end_token()
            i += 1
            continue
        buf.append(ch)
        in_token = True
        i += 1
    end_segment()
    return segments


_WRAPPERS = frozenset({"sudo", "command", "exec", "time", "nohup", "env", "nice", "builtin"})


def _normalise_word(token: str) -> str:
    word = token.replace("\\", "/").rsplit("/", 1)[-1].lower()
    return word.removesuffix(".exe")


def command_and_args(tokens: Sequence[str]) -> tuple[str, list[str]]:
    """The command word of a simple command (without wrappers / VAR=x) and its arguments."""
    rest = list(tokens)
    while rest:
        first = rest[0]
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", first) or _normalise_word(first) in _WRAPPERS:
            rest.pop(0)
            continue
        break
    if not rest:
        return "", []
    return _normalise_word(rest[0]), rest[1:]


@dataclass(frozen=True)
class GitCall:
    subcommand: str
    args: list[str]
    directory: str  # value of -C, or ""


_GIT_OPTIONS_WITH_VALUE = frozenset(
    {"-c", "--git-dir", "--work-tree", "--namespace", "--exec-path"}
)


def git_call(tokens: Sequence[str]) -> GitCall | None:
    cmd, args = command_and_args(tokens)
    if cmd != "git":
        return None
    directory = ""
    i = 0
    while i < len(args):
        tok = args[i]
        if tok == "-C" and i + 1 < len(args):
            directory = args[i + 1]
            i += 2
        elif tok in _GIT_OPTIONS_WITH_VALUE and i + 1 < len(args):
            i += 2
        elif tok.startswith("-"):
            i += 1
        else:
            return GitCall(tok.lower(), args[i + 1 :], directory)
    return None


# --------------------------------------------------------------------- rule: attribution

_ATTRIBUTION_PATTERNS = (
    re.compile(r"co-authored-by\s*:[^\n]*(claude|anthropic)", re.IGNORECASE),
    re.compile(r"generated\s+with[^\n]{0,60}claude", re.IGNORECASE),
    re.compile(r"noreply@anthropic\.com", re.IGNORECASE),
    re.compile(r"claude\.com/claude-code", re.IGNORECASE),
)


def _publishes_text(tokens: Sequence[str]) -> bool:
    """True for git commit / tag and ``gh pr create|edit|comment|review`` style commands."""
    call = git_call(tokens)
    if call is not None:
        return call.subcommand in {"commit", "tag"}
    cmd, args = command_and_args(tokens)
    return (
        cmd == "gh"
        and len(args) >= 2
        and args[0] == "pr"
        and args[1] in {"create", "edit", "comment", "review"}
    )


def check_attribution(full_command: str, segments: list[list[str]]) -> Verdict:
    if not any(_publishes_text(tokens) for tokens in segments):
        return ALLOW
    if any(pattern.search(full_command) for pattern in _ATTRIBUTION_PATTERNS):
        return _block(
            "no-claude-attribution",
            "this project does not allow Claude attribution (Co-Authored-By trailer or "
            "'Generated with Claude Code') in commits, tags or pull requests. Remove those lines.",
        )
    return ALLOW


# ------------------------------------------------------------------------------ rule: git

_PUSH_OPTIONS_WITH_VALUE = frozenset(
    {
        "-o",
        "--push-option",
        "--repo",
        "--receive-pack",
        "--exec",
        "--signed",
        "--recurse-submodules",
    }
)


def _head_for(call: GitCall, cwd: Path, read: HeadReader) -> Head:
    return read(cwd / call.directory if call.directory else cwd)


def _is_force_flag(token: str) -> bool:
    if token in {"--force", "--force-if-includes"} or token.startswith("--force-with-lease"):
        return True
    return bool(re.fullmatch(r"-[A-Za-z]*f[A-Za-z]*", token))


def check_git(segments: list[list[str]], cwd: Path, read: HeadReader) -> Verdict:
    for tokens in segments:
        call = git_call(tokens)
        if call is None:
            continue
        if call.subcommand == "commit":
            verdict = _check_commit(call, cwd, read)
        elif call.subcommand == "push":
            verdict = _check_push(call, cwd, read)
        else:
            continue
        if not verdict.allowed:
            return verdict
    return ALLOW


def _check_commit(call: GitCall, cwd: Path, read: HeadReader) -> Verdict:
    if "--dry-run" in call.args or "--help" in call.args:
        return ALLOW
    head = _head_for(call, cwd, read)
    if head.kind == "branch" and head.branch in PROTECTED_BRANCHES:
        return _block(
            "no-commit-on-protected-branch",
            f"committing directly on '{head.branch}' is not allowed. Switch to the dev branch "
            "(git switch dev) and commit there.",
        )
    return ALLOW


def _check_push(call: GitCall, cwd: Path, read: HeadReader) -> Verdict:
    positional: list[str] = []
    i = 0
    while i < len(call.args):
        tok = call.args[i]
        if _is_force_flag(tok):
            return _block("no-destructive-push", "force pushes are not allowed.")
        if tok in {"--delete", "-d", "--mirror", "--prune"}:
            return _block(
                "no-destructive-push",
                "deleting remote branches, mirror pushes and prune pushes are not allowed.",
            )
        if tok == "--all":
            return _block(
                "no-push-to-protected-branch",
                "'git push --all' would also push main. Push the dev branch by name instead.",
            )
        if tok in _PUSH_OPTIONS_WITH_VALUE:
            i += 2
            continue
        if tok.startswith("-"):
            i += 1
            continue
        positional.append(tok)
        i += 1

    refspecs = positional[1:]  # positional[0] is the remote
    head: Head | None = None
    if not refspecs:
        head = _head_for(call, cwd, read)
        if head.kind == "branch" and head.branch in PROTECTED_BRANCHES:
            return _block(
                "no-push-to-protected-branch",
                f"pushing '{head.branch}' is not allowed. Work on dev and open a pull request.",
            )
        return ALLOW

    for spec in refspecs:
        if spec.startswith("+"):
            return _block("no-destructive-push", "force pushes are not allowed.")
        source, _, destination = spec.partition(":")
        if source == "" and ":" in spec:
            return _block("no-destructive-push", "deleting a remote branch is not allowed.")
        target = (destination or source).removeprefix("refs/heads/")
        if target in PROTECTED_BRANCHES:
            return _block(
                "no-push-to-protected-branch",
                f"pushing to '{target}' is not allowed. Push dev and open a pull request.",
            )
        if target == "HEAD":
            head = head or _head_for(call, cwd, read)
            if head.kind == "branch" and head.branch in PROTECTED_BRANCHES:
                return _block(
                    "no-push-to-protected-branch",
                    f"HEAD is '{head.branch}'; pushing it is not allowed.",
                )
    return ALLOW


# --------------------------------------------------------------------- rule: secret reads

_READERS = frozenset(
    {
        "cat", "type", "head", "tail", "more", "less", "sed", "awk", "gawk", "nl", "tac",
        "bat", "grep", "egrep", "fgrep", "rg", "ag", "ack", "findstr", "strings", "xxd", "od",
        "hexdump", "base64", "sqlite3", "get-content", "gc", "select-string", "sls",
    }
)  # fmt: skip
_SENDERS = frozenset(
    {"curl", "wget", "scp", "rsync", "nc", "ncat", "ftp", "sftp", "invoke-webrequest", "iwr",
     "invoke-restmethod", "irm"}
)  # fmt: skip
_COPIERS = frozenset({"cp", "mv", "copy", "move", "copy-item", "move-item", "cpi", "mi"})
_INTERPRETERS = frozenset(
    {"python", "python3", "py", "node", "nodejs", "ruby", "perl", "php", "pwsh", "powershell"}
)
_INLINE_CODE_FLAGS = frozenset({"-c", "-e", "-p", "-r", "-command", "-encodedcommand"})

_SENSITIVE_IN_TEXT = (
    re.compile(r"(?<![A-Za-z0-9_])\.env(?!\.example)(?![A-Za-z0-9_-])", re.IGNORECASE),
    re.compile(r"\.envrc", re.IGNORECASE),
    re.compile(r"\.(db|db-wal|db-shm|sqlite3?)(?![A-Za-z0-9_])", re.IGNORECASE),
    re.compile(r"\.(pem|key)(?![A-Za-z0-9_])", re.IGNORECASE),
)
_DB_NAME = re.compile(r"\.(db|sqlite3?)(-[a-z]+)?\*?$")


def sensitive_reason(token: str) -> str:
    """Why ``token`` names a protected file (a description), or "" when it does not."""
    text = token.strip().strip("'\"")
    if "=" in text:
        text = text.split("=", 1)[1]
    text = text.lstrip("@<>").replace("\\", "/").lower()
    parts = [p for p in text.split("/") if p and p != "."]
    if not parts:
        return ""
    base = parts[-1]
    if base == ".env.example":
        return ""
    if base in {".env", ".envrc"} or base.startswith((".env.", ".env*", ".env?", ".env[")):
        return "an environment file that holds secrets"
    if _DB_NAME.search(base):
        return "a local database (it holds personal memories)"
    if base.endswith((".pem", ".key")):
        return "a key file"
    if "data" in parts and (parts[0] == "data" or "backend" in parts):
        return "the local data folder (it holds personal memories)"
    return ""


def _code_mentions_sensitive(code: str) -> bool:
    return any(pattern.search(code) for pattern in _SENSITIVE_IN_TEXT)


def check_secret_reads(segments: list[list[str]]) -> Verdict:
    for tokens in segments:
        # input redirection, e.g.  sqlite3 x < .env
        for index, tok in enumerate(tokens):
            if tok == "<" and index + 1 < len(tokens):
                target = tokens[index + 1]
            elif tok.startswith("<") and not tok.startswith("<<"):
                target = tok[1:]
            else:
                continue
            if sensitive_reason(target):
                return _secret_block(sensitive_reason(target))
        cmd, args = command_and_args(tokens)
        if not cmd:
            continue
        if cmd in _READERS or cmd in _SENDERS:
            for arg in args:
                reason = sensitive_reason(arg)
                if reason:
                    return _secret_block(reason)
        elif cmd in _COPIERS:
            positional = [a for a in args if not a.startswith("-")]
            for arg in positional[:-1]:  # the last one is the destination
                reason = sensitive_reason(arg)
                if reason:
                    return _secret_block(reason)
        elif (
            cmd in _INTERPRETERS
            and any(a.lower() in _INLINE_CODE_FLAGS for a in args)
            and any(_code_mentions_sensitive(a) for a in args)
        ):
            return _secret_block("a protected file referenced from inline code")
    return ALLOW


def _secret_block(reason: str) -> Verdict:
    return _block(
        "no-secret-read",
        f"this command would read or send {reason}. Secrets and personal memories must never be "
        "shown to the assistant. Run it yourself in a terminal if you really need it.",
    )


# --------------------------------------------------------------------------------- driver


def evaluate(command: str, cwd: str = ".", read: HeadReader = read_head) -> Verdict:
    """Decide whether a shell command may run. Raises ``GuardError`` if it cannot be decided."""
    segments = split_segments(strip_heredocs(command))
    base = Path(cwd or ".")
    for verdict in (
        check_attribution(command, segments),  # the full text: the message is in the heredoc
        check_git(segments, base, read),
        check_secret_reads(segments),
    ):
        if not verdict.allowed:
            return verdict
    return ALLOW


def _say(stream: TextIO, text: str) -> None:
    stream.write(text.encode("ascii", "replace").decode("ascii") + "\n")


def main(stdin: BinaryIO | None = None, stderr: TextIO | None = None) -> int:
    """Hook entry point. 0 lets the command run, 2 blocks it (also when the guard itself fails)."""
    stream = stdin if stdin is not None else sys.stdin.buffer
    err = stderr if stderr is not None else sys.stderr
    try:
        payload = json.loads(stream.read().decode("utf-8", errors="replace"))
        if not isinstance(payload, dict):
            raise GuardError("hook input is not a JSON object")
        tool_name = payload.get("tool_name")
        if not isinstance(tool_name, str):
            raise GuardError("hook input does not name a tool")
        if tool_name not in CHECKED_TOOLS:
            return 0  # some other tool: not this guard's business
        tool_input = payload.get("tool_input")
        command = tool_input.get("command") if isinstance(tool_input, dict) else None
        if not isinstance(command, str):
            raise GuardError("hook input has no command to check")
        verdict = evaluate(command, str(payload.get("cwd") or "."))
    except Exception as exc:  # noqa: BLE001 - a guard that fails must block, never allow
        _say(err, f"AARYX guard could not check this command ({type(exc).__name__}); blocked.")
        return BLOCK_EXIT_CODE
    if verdict.allowed:
        return 0
    _say(err, f"AARYX guard blocked this command [{verdict.rule}]: {verdict.reason}")
    return BLOCK_EXIT_CODE


if __name__ == "__main__":
    sys.exit(main())
