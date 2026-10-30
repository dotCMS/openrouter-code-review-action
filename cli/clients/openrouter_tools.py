"""Tool definitions and executor for the OpenRouter agent loop.

The Codex SDK shipped its own sandboxed agent runtime; OpenRouter raw
chat completions do not. This module provides the small tool surface the
review and act workflows rely on:

* ``read_file`` — read a file inside the repository (line-bounded).
* ``write_file`` — create/overwrite a file inside the repository.
* ``run_command`` — run a shell command in the repository root.

Sandbox parity
--------------
``sandbox_mode="read-only"`` permits reading, and pinned panel reviews also
support listing/searching the reviewed commit. ``workspace-write`` and
``danger-full-access`` keep the original three editing tools.
Path escapes are rejected in every mode: file tools may only touch
paths that resolve inside the repository root.

Tool results are plain strings so they can be posted back verbatim as
``role: "tool"`` messages. Errors are reported *as tool output* rather
than raised, so the model can observe and recover from its own
mistakes instead of aborting the run.
"""

from __future__ import annotations

import json
import os
import subprocess  # nosec B404
from pathlib import Path
from typing import Any

_READ_ONLY_SANDBOX_MODES = frozenset({"read-only"})

MAX_TOOL_OUTPUT_CHARS = 24_000
MAX_READ_FILE_CHARS = 100_000
COMMAND_TIMEOUT_SECONDS = 120

READ_FILE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": (
            "Read a file from the repository. Returns the file content. "
            "Use offset (1-based line number) and limit to read slices of large files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Repository-relative file path"},
                "offset": {
                    "type": "integer",
                    "description": "1-based first line to read (optional)",
                },
                "limit": {"type": "integer", "description": "Max lines to read (optional)"},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
}

WRITE_FILE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Create or overwrite a file in the repository with the given content.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Repository-relative file path"},
                "content": {"type": "string", "description": "Full file content to write"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    },
}

RUN_COMMAND_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "run_command",
        "description": (
            "Run a shell command in the repository root (e.g. git diff, tests, "
            "grep). Returns stdout+stderr and the exit code."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"}
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}

LIST_FILES_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "list_files",
        "description": "List tracked repository files without running repository code.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
}
SEARCH_CODE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_code",
        "description": "Search tracked repository files for literal text. Returns path:line matches.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
    },
}
READ_ONLY_TOOL_SPECS = [READ_FILE_TOOL, LIST_FILES_TOOL, SEARCH_CODE_TOOL]
TOOL_SPECS: list[dict[str, Any]] = [READ_FILE_TOOL, WRITE_FILE_TOOL, RUN_COMMAND_TOOL]


def execute_tool(
    name: str,
    arguments_json: str,
    *,
    repo_root: Path,
    sandbox_mode: str,
    revision: str | None = None,
) -> str:
    """Execute one tool call and return its output as a string.

    Never raises for model-caused errors (bad arguments, path escapes,
    sandbox violations, failing commands): those come back as
    ``error: ...`` tool output. Only truly unexpected conditions
    (undecodable JSON) produce an error string as well — the agent loop
    must always be able to feed the result back to the model.
    """
    try:
        arguments = json.loads(arguments_json) if arguments_json.strip() else {}
    except json.JSONDecodeError as exc:
        return f"error: tool arguments were not valid JSON: {exc}"
    if not isinstance(arguments, dict):
        return "error: tool arguments must be a JSON object"

    read_only = sandbox_mode in _READ_ONLY_SANDBOX_MODES
    if read_only and revision is not None and name in {"read_file", "list_files", "search_code"}:
        return _tool_inspect_revision(name, arguments, repo_root=repo_root, revision=revision)
    if name == "read_file":
        if read_only:
            path = _resolve_repo_path(arguments.get("path"), repo_root)
            if isinstance(path, str):
                return path
            if not is_reviewable_path(path, repo_root):
                return "error: review access to repository metadata or credential files is denied"
        return _tool_read_file(arguments, repo_root=repo_root)
    if name == "write_file":
        if read_only:
            return "error: sandbox is read-only; write_file is not available"
        return _tool_write_file(arguments, repo_root=repo_root)
    if name == "run_command":
        if read_only:
            return "error: sandbox is read-only; run_command is not available"
        return _tool_run_command(arguments, repo_root=repo_root)
    return f"error: unknown tool '{name}'"


def is_reviewable_path(path: Path, repo_root: Path) -> bool:
    resolved = Path(os.path.abspath(path))
    if not resolved.is_relative_to(repo_root.resolve()):
        return False
    parts = resolved.relative_to(repo_root.resolve()).parts
    name, suffix = resolved.name.casefold(), resolved.suffix.casefold()
    return not (
        ".git" in parts
        or name == ".env"
        or (name.startswith(".env.") and suffix not in {".example", ".sample"})
        or suffix in {".pem", ".key"}
    )


def _render_pinned_search_matches(stdout: bytes, *, revision: str, repo_root: Path) -> str:
    """Filter complete NUL-delimited filenames before rendering any matched text."""
    root = repo_root.resolve()
    prefix = revision.encode("utf-8", errors="replace")
    rendered: list[str] = []
    pos = 0
    while pos < len(stdout):
        path_end = stdout.find(b"\0", pos)
        if path_end < 0:
            raise ValueError("Invalid pinned search filename field")
        head = stdout[pos:path_end]
        if (
            head[: len(prefix)].lower() != prefix.lower()
            or head[len(prefix) : len(prefix) + 1] != b":"
        ):
            raise ValueError("Invalid pinned search revision prefix")
        path = head[len(prefix) + 1 :].decode("utf-8", errors="replace")
        line_start = path_end + 1
        line_end = stdout.find(b"\0", line_start)
        if line_end < 0 or not stdout[line_start:line_end].isdigit():
            raise ValueError("Invalid pinned search line field")
        lineno = stdout[line_start:line_end].decode("ascii")
        content_start = line_end + 1
        content_end = stdout.find(b"\n", content_start)
        if content_end < 0:
            content_end = len(stdout)
        content = stdout[content_start:content_end].decode("utf-8", errors="replace")
        pos = content_end + 1
        if is_reviewable_path(root / path, root):
            rendered.append(f"{path}:{lineno}:{content}")
    return "\n".join(rendered)


def _pinned_blob_at(repo_root: Path, revision: str, relative: str) -> bool:
    """Probe the pinned tree so failed reads cannot masquerade as missing paths."""
    tree = subprocess.run(
        ["git", "--no-pager", "--literal-pathspecs", "ls-tree", "-z", revision, "--", relative],
        cwd=repo_root,
        capture_output=True,
        timeout=COMMAND_TIMEOUT_SECONDS,
        check=True,
    )
    for entry in tree.stdout.decode("utf-8", errors="replace").split("\0"):
        metadata, _, path = entry.partition("\t")
        fields = metadata.split()
        if len(fields) >= 2 and fields[1] == "blob" and path == relative:
            return True
    return False


def read_file_at_revision(repo_root: Path, revision: str, raw_path: str) -> str | None:
    """Read full pinned evidence for host checks; model output limits belong to tools.

    Returns ``None`` only when the path is absent, denied, or not a blob at the
    revision. Git probe/show failures (nonzero status, timeout, ``OSError``)
    propagate to the caller.
    """
    if len(revision) not in {40, 64} or any(
        char not in "0123456789abcdefABCDEF" for char in revision
    ):
        return None
    if not raw_path.strip() or "\0" in raw_path:
        return None
    # Git object paths must not follow symlinks in a different checkout.
    path = Path(os.path.abspath(repo_root.resolve() / raw_path))
    if not is_reviewable_path(path, repo_root):
        return None
    relative = path.relative_to(repo_root.resolve()).as_posix()
    if not _pinned_blob_at(repo_root, revision, relative):
        return None
    result = subprocess.run(
        ["git", "--no-pager", "show", "--no-textconv", f"{revision}:{relative}"],
        cwd=repo_root,
        capture_output=True,
        timeout=COMMAND_TIMEOUT_SECONDS,
        check=True,
    )
    return result.stdout.decode("utf-8", errors="replace")


def _tool_inspect_revision(
    name: str, arguments: dict[str, Any], *, repo_root: Path, revision: str
) -> str:
    if len(revision) not in {40, 64} or any(
        char not in "0123456789abcdefABCDEF" for char in revision
    ):
        return "error: review revision must be a commit SHA"
    try:
        if name == "read_file":
            raw_path = arguments.get("path")
            if not isinstance(raw_path, str):
                return "error: read_file requires a string path"
            content = read_file_at_revision(repo_root, revision, raw_path)
            if content is None:
                return "error: could not read reviewed file or path is denied"
            lines = content.splitlines(keepends=True)
            offset = _as_positive_int(arguments.get("offset"), default=1) or 1
            limit = _as_positive_int(arguments.get("limit"), default=None)
            content = "".join(
                lines[offset - 1 :] if limit is None else lines[offset - 1 : offset - 1 + limit]
            )
            return (
                content[:MAX_READ_FILE_CHARS] + "\n... (truncated)"
                if len(content) > MAX_READ_FILE_CHARS
                else content
            )
        if name == "list_files":
            args = ["ls-tree", "-r", "--name-only", "-z", revision]
        else:
            text = arguments.get("text")
            if not isinstance(text, str) or not text.strip():
                return "error: search_code requires non-empty literal text"
            args = ["grep", "-I", "-n", "-z", "-F", "--no-textconv", "-e", text, revision, "--"]
        result = subprocess.run(
            ["git", "--no-pager", *args],
            cwd=repo_root,
            capture_output=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
            check=False,
        )
        if result.returncode not in ({0, 1} if name == "search_code" else {0}):
            return "error: could not inspect reviewed commit"
        if name == "list_files":
            content = "\n".join(
                p
                for p in result.stdout.decode("utf-8", errors="replace").split("\0")
                if p and is_reviewable_path(repo_root.resolve() / p, repo_root)
            )
        else:
            content = _render_pinned_search_matches(
                result.stdout, revision=revision, repo_root=repo_root
            )
        return (
            content[:MAX_TOOL_OUTPUT_CHARS] + "\n... (truncated)"
            if len(content) > MAX_TOOL_OUTPUT_CHARS
            else content
        ) or "(no matches)"
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
        return "error: reviewed-commit inspection failed"


def _tool_read_file(arguments: dict[str, Any], *, repo_root: Path) -> str:
    path = _resolve_repo_path(arguments.get("path"), repo_root)
    if isinstance(path, str):
        return path
    if not path.is_file():
        return f"error: file not found: {path}"

    offset = _as_positive_int(arguments.get("offset"), default=1)
    limit = _as_positive_int(arguments.get("limit"), default=None)

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"error: failed to read {path}: {exc}"

    lines = text.splitlines(keepends=True)
    start_index = max((offset or 1) - 1, 0)
    selected = lines[start_index:] if limit is None else lines[start_index : start_index + limit]
    content = "".join(selected)
    if len(content) > MAX_READ_FILE_CHARS:
        content = content[:MAX_READ_FILE_CHARS] + "\n... (truncated)"
    return content


def _tool_write_file(arguments: dict[str, Any], *, repo_root: Path) -> str:
    path = _resolve_repo_path(arguments.get("path"), repo_root)
    if isinstance(path, str):
        return path
    content = arguments.get("content")
    if not isinstance(content, str):
        return "error: write_file requires a string 'content' argument"

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        return f"error: failed to write {path}: {exc}"
    return f"wrote {len(content)} chars to {path}"


def _tool_run_command(arguments: dict[str, Any], *, repo_root: Path) -> str:
    command = arguments.get("command")
    if not isinstance(command, str) or not command.strip():
        return "error: run_command requires a non-empty string 'command' argument"

    try:
        completed = subprocess.run(  # nosec B602 — the model is the operator; sandbox mode gates availability
            ["bash", "-lc", command],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return f"error: command timed out after {COMMAND_TIMEOUT_SECONDS}s: {command}"
    except FileNotFoundError:
        return "error: 'bash' is not available on this runner"

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    output = stdout
    if stderr:
        output = f"{output}\n{stderr}" if output else stderr
    if len(output) > MAX_TOOL_OUTPUT_CHARS:
        output = output[:MAX_TOOL_OUTPUT_CHARS] + "\n... (truncated)"
    return f"exit_code={completed.returncode}\n{output}"


def _resolve_repo_path(raw_path: Any, repo_root: Path) -> Path | str:
    """Resolve a model-supplied path inside the repo root, or an error string."""
    if not isinstance(raw_path, str) or not raw_path.strip():
        return "error: a non-empty string 'path' argument is required"

    candidate = Path(raw_path.strip())
    resolved_root = repo_root.resolve()
    resolved = (
        (resolved_root / candidate).resolve()
        if not candidate.is_absolute()
        else candidate.resolve()
    )
    if resolved != resolved_root and not resolved.is_relative_to(resolved_root):
        return f"error: path escapes the repository root: {raw_path}"
    return resolved


def _as_positive_int(value: Any, *, default: int | None) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    if isinstance(value, float) and not value.is_integer():
        return default
    parsed = int(value)
    return parsed if parsed > 0 else default
