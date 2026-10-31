from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from cli.clients.openrouter_client import OpenRouterClient
from cli.clients.openrouter_thread_store import OpenRouterThreadStore
from cli.clients.openrouter_tools import (
    MAX_READ_FILE_CHARS,
    READ_ONLY_TOOL_SPECS,
    execute_tool,
    read_file_at_revision,
)
from cli.core.config import ReviewConfig
from cli.review.dedupe import collect_prior_dotbot_review_comments
from cli.review.patch_parser import to_relative_path
from cli.workflows.review_workflow import ReviewWorkflow


@pytest.fixture
def reviewed_repo(tmp_path: Path) -> tuple[Path, str]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=tmp_path, capture_output=True, text=True, check=True
        ).stdout.strip()

    git("init")
    git("config", "user.email", "test@example.test")
    git("config", "user.name", "Test")
    (tmp_path / "caller.py").write_text("authorize(user)\nprivileged_operation()\n")
    (tmp_path / ".env").write_text("TOKEN=should-not-be-readable\n")
    git("add", ".")
    git("commit", "-m", "Reviewed revision")
    revision = git("rev-parse", "HEAD")
    # Reading the checkout would now produce the wrong review evidence.
    (tmp_path / "caller.py").write_text("different unreviewed checkout\n")
    return tmp_path, revision


def test_inspection_reads_and_searches_the_pinned_revision(reviewed_repo: tuple[Path, str]) -> None:
    root, revision = reviewed_repo

    def tool(name: str, args: dict[str, Any]) -> str:
        return execute_tool(
            name, json.dumps(args), repo_root=root, sandbox_mode="read-only", revision=revision
        )

    assert (
        tool("read_file", {"path": "caller.py", "offset": 2, "limit": 1})
        == "privileged_operation()\n"
    )
    assert tool("list_files", {}) == "caller.py"
    assert "caller.py:1:authorize(user)" in tool("search_code", {"text": "authorize"})
    assert "different unreviewed" not in tool("read_file", {"path": "caller.py"})
    assert "should-not-be-readable" not in tool("search_code", {"text": "TOKEN"})
    assert tool("search_code", {"text": "--exec=malicious"}) == "(no matches)"


@pytest.mark.parametrize("path", [".git/config", ".env", "../escape", "/etc/passwd"])
def test_review_cannot_read_credentials_or_escape(
    reviewed_repo: tuple[Path, str], path: str
) -> None:
    root, revision = reviewed_repo
    for pinned in (None, revision):
        assert execute_tool(
            "read_file",
            json.dumps({"path": path}),
            repo_root=root,
            sandbox_mode="read-only",
            revision=pinned,
        ).startswith("error:")


def test_pinned_review_still_rejects_all_mutating_tools(reviewed_repo: tuple[Path, str]) -> None:
    root, revision = reviewed_repo
    for name, args in (
        ("write_file", {"path": "created", "content": "no"}),
        ("run_command", {"command": "touch created"}),
    ):
        assert execute_tool(
            name, json.dumps(args), repo_root=root, sandbox_mode="read-only", revision=revision
        ).startswith("error:")
    assert not (root / "created").exists()


def test_revision_and_search_arguments_are_validated(reviewed_repo: tuple[Path, str]) -> None:
    root, revision = reviewed_repo
    assert execute_tool(
        "list_files", "{}", repo_root=root, sandbox_mode="read-only", revision="HEAD; touch created"
    ).startswith("error:")
    assert execute_tool(
        "search_code", '{"text": ""}', repo_root=root, sandbox_mode="read-only", revision=revision
    ).startswith("error:")
    assert not (root / "created").exists()


def test_openrouter_readonly_payload_has_no_mutating_tool_and_uses_pinned_content(
    reviewed_repo: tuple[Path, str],
) -> None:
    root, revision = reviewed_repo
    payloads: list[dict[str, Any]] = []

    def transport(payload: dict[str, Any]) -> dict[str, Any]:
        payloads.append(payload)
        if len(payloads) == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "read-1",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path": "caller.py"}',
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        return {"choices": [{"message": {"content": "Review complete"}, "finish_reason": "stop"}]}

    client = OpenRouterClient(
        ReviewConfig(
            github_token="token",
            repository="o/r",
            repo_root=root,
            openrouter_api_key="key",
            stream_output=False,
        ),
        transport=transport,
        review_head_sha=revision,
        thread_store=OpenRouterThreadStore(root / "threads"),
    )
    assert client.execute_text("Inspect the caller", sandbox_mode="read-only") == "Review complete"
    assert payloads[0]["tools"] == READ_ONLY_TOOL_SPECS
    tool_result = next(
        message["content"] for message in payloads[1]["messages"] if message["role"] == "tool"
    )
    assert "authorize(user)" in tool_result
    assert "different unreviewed" not in tool_result


def test_panel_diff_uses_exact_commits_and_never_runs_external_diff(
    reviewed_repo: tuple[Path, str],
) -> None:
    root, base = reviewed_repo

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()

    (root / "caller.py").write_text("reviewed change\n")
    git("add", "caller.py")
    git("commit", "-m", "PR change")
    head = git("rev-parse", "HEAD")
    (root / "caller.py").write_text("unreviewed checkout\n")
    git("config", "diff.external", "touch external-diff-executed")
    workflow = ReviewWorkflow(
        ReviewConfig(github_token="token", repository="o/r", repo_root=root),
        github_client=cast(Any, object()),
    )
    diff = workflow._panel_diff(cast(Any, SimpleNamespace(base=SimpleNamespace(sha=base))), head)
    assert diff is not None
    assert "+reviewed change" in diff
    assert "unreviewed checkout" not in diff
    assert not (root / "external-diff-executed").exists()


@pytest.mark.parametrize(
    "credential_path",
    [
        ".env",
        "private.pem",
        "nested/client.key",
        ".env.production",
        ".ENV.local",
        "PRIVATE.PEM",
        ".env.production.example",
        ".env.production.sample",
        ".ENV.PRODUCTION.EXAMPLE",
        ".pem",
        ".key",
        ".PEM",
        ".KEY",
    ],
)
def test_credential_file_changes_require_human_review_before_any_diff_contents(
    reviewed_repo: tuple[Path, str],
    credential_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, base = reviewed_repo
    path = root / credential_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("credential data must not enter a model context\n")
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Credential change"], cwd=root, check=True, capture_output=True
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    real_run = subprocess.run

    def metadata_only(args: list[str], **kwargs: Any) -> Any:
        assert "--name-only" in args, (
            "Credential scope must be checked before retrieving diff contents"
        )
        return real_run(args, **kwargs)

    monkeypatch.setattr("cli.workflows.review_workflow.subprocess.run", metadata_only)
    workflow = ReviewWorkflow(
        ReviewConfig(github_token="token", repository="o/r", repo_root=root),
        github_client=cast(Any, object()),
    )
    assert (
        workflow._panel_diff(cast(Any, SimpleNamespace(base=SimpleNamespace(sha=base))), head)
        is None
    )


def test_renamed_credentials_require_manual_review_but_templates_are_reviewable(
    reviewed_repo: tuple[Path, str],
) -> None:
    root, base = reviewed_repo
    subprocess.run(["git", "mv", ".env", "ordinary.txt"], cwd=root, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Rename credentials"], cwd=root, check=True, capture_output=True
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    workflow = ReviewWorkflow(
        ReviewConfig(github_token="token", repository="o/r", repo_root=root),
        github_client=cast(Any, object()),
    )
    assert (
        workflow._panel_diff(cast(Any, SimpleNamespace(base=SimpleNamespace(sha=base))), head)
        is None
    )
    (root / ".env.example").write_text("TOKEN=example\n")
    subprocess.run(["git", "add", ".env.example"], cwd=root, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Add public template"], cwd=root, check=True, capture_output=True
    )
    template_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    diff = workflow._panel_diff(
        cast(Any, SimpleNamespace(base=SimpleNamespace(sha=head))), template_head
    )
    assert diff is not None and "+TOKEN=example" in diff


@pytest.mark.parametrize("name", ["literal[1].py", ":(glob)*.py"])
def test_pinned_reads_treat_filename_pathspec_syntax_literally(
    reviewed_repo: tuple[Path, str],
    name: str,
) -> None:
    root, _ = reviewed_repo
    (root / name).write_text("literal path content\n")
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Add literal filename"], cwd=root, check=True, capture_output=True
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    assert read_file_at_revision(root, revision, name) == "literal path content\n"


def _commit_credential_and_ordinary_matches(root: Path, directory: str) -> str:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()

    credential_dir = root / directory
    credential_dir.mkdir()
    (credential_dir / ".env").write_text("TOKEN=denied-env-secret\n")
    (credential_dir / "private.pem").write_text("TOKEN=denied-pem-secret\n")
    (root / "src").mkdir()
    (root / "src" / "app.py").write_text("TOKEN=ordinary\n")
    (root / ".env.example").write_text("TOKEN=example\n")
    (root / "docs:readme.txt").write_text("TOKEN=colon-ok\n")
    (root / "nl\nfile.txt").write_text("TOKEN=newline-ok\n")
    git("add", ".")
    git("commit", "-m", "Add credential and ordinary matches")
    return git("rev-parse", "HEAD")


@pytest.mark.parametrize("directory", ["safe:dir", "safe\ndir"])
def test_pinned_search_denies_credentials_under_colon_and_newline_parents(
    reviewed_repo: tuple[Path, str],
    directory: str,
) -> None:
    root, _ = reviewed_repo
    revision = _commit_credential_and_ordinary_matches(root, directory)
    output = execute_tool(
        "search_code",
        json.dumps({"text": "TOKEN"}),
        repo_root=root,
        sandbox_mode="read-only",
        revision=revision,
    )
    assert "denied-env-secret" not in output
    assert "denied-pem-secret" not in output
    assert "should-not-be-readable" not in output
    assert "src/app.py:1:TOKEN=ordinary" in output
    assert ".env.example:1:TOKEN=example" in output
    assert "docs:readme.txt:1:TOKEN=colon-ok" in output
    assert "nl\nfile.txt:1:TOKEN=newline-ok" in output


def test_prior_comment_applicability_uses_reviewed_commit_not_checkout(
    reviewed_repo: tuple[Path, str],
) -> None:
    root, revision = reviewed_repo
    thread = SimpleNamespace(
        id="thread-1",
        is_resolved=False,
        comments=[
            SimpleNamespace(
                id="comment-1",
                prompt_line=1,
                author="bot",
                path="caller.py",
                body="**Current code:**\n```python\nauthorize(user)\n```",
            )
        ],
    )
    comments = cast(Any, [thread])
    assert not collect_prior_dotbot_review_comments(comments, {"bot"}, root)[
        0
    ].is_currently_applicable
    matched = collect_prior_dotbot_review_comments(
        comments,
        {"bot"},
        root,
        source_reader=lambda path: execute_tool(
            "read_file",
            json.dumps({"path": path}),
            repo_root=root,
            sandbox_mode="read-only",
            revision=revision,
        ),
    )
    assert matched[0].is_currently_applicable


def test_pinned_paths_never_follow_symlinks_in_the_live_checkout(
    reviewed_repo: tuple[Path, str],
) -> None:
    root, revision = reviewed_repo
    (root / "caller.py").unlink()
    (root / "caller.py").symlink_to(root / ".env")
    result = execute_tool(
        "read_file",
        '{"path": "caller.py"}',
        repo_root=root,
        sandbox_mode="read-only",
        revision=revision,
    )
    assert "authorize(user)" in result
    assert "should-not-be-readable" not in result
    assert to_relative_path(str(root / "caller.py"), root) == "caller.py"


def test_read_file_at_revision_returns_none_only_for_absent_denied_or_non_blob_paths(
    reviewed_repo: tuple[Path, str],
) -> None:
    root, revision = reviewed_repo
    assert read_file_at_revision(root, revision, "missing.py") is None
    assert read_file_at_revision(root, revision, ".env") is None
    assert read_file_at_revision(root, revision, "../escape") is None
    assert read_file_at_revision(root, revision, ".git/config") is None

    (root / "pkg").mkdir()
    (root / "pkg" / "mod.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "pkg/mod.py"], cwd=root, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Add package"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, text=True, capture_output=True
    ).stdout.strip()
    assert read_file_at_revision(root, revision, "pkg") is None
    assert read_file_at_revision(root, revision, "pkg/mod.py") == "x = 1\n"


@pytest.mark.parametrize("failing_stage", ["ls-tree", "show"])
def test_read_file_at_revision_propagates_nonzero_git_results(
    reviewed_repo: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
    failing_stage: str,
) -> None:
    root, revision = reviewed_repo
    real_run = subprocess.run

    def fail_at_failing_stage(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        if failing_stage in args:
            raise subprocess.CalledProcessError(1, args, output=b"", stderr=b"fatal")
        return real_run(args, **kwargs)

    monkeypatch.setattr("cli.clients.openrouter_tools.subprocess.run", fail_at_failing_stage)
    with pytest.raises(subprocess.CalledProcessError):
        read_file_at_revision(root, revision, "caller.py")
    assert execute_tool(
        "read_file",
        '{"path": "caller.py"}',
        repo_root=root,
        sandbox_mode="read-only",
        revision=revision,
    ).startswith("error:")


@pytest.mark.parametrize("error", [subprocess.TimeoutExpired("git", 120), OSError("boom")])
def test_read_file_at_revision_propagates_timeout_and_oserror(
    reviewed_repo: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
) -> None:
    root, revision = reviewed_repo

    def raise_error(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        raise error

    monkeypatch.setattr("cli.clients.openrouter_tools.subprocess.run", raise_error)
    with pytest.raises(type(error)):
        read_file_at_revision(root, revision, "caller.py")
    assert execute_tool(
        "read_file",
        '{"path": "caller.py"}',
        repo_root=root,
        sandbox_mode="read-only",
        revision=revision,
    ).startswith("error:")


def test_prior_evidence_reads_past_the_model_tool_output_limit(
    reviewed_repo: tuple[Path, str],
) -> None:
    root, _ = reviewed_repo
    source = "# padding\n" * (MAX_READ_FILE_CHARS // 10 + 100) + "authorize(user)\n"
    (root / "caller.py").write_text(source)
    subprocess.run(["git", "add", "caller.py"], cwd=root, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Large reviewed source"], cwd=root, check=True, capture_output=True
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, text=True, capture_output=True
    ).stdout.strip()
    assert read_file_at_revision(root, revision, "caller.py") == source
    bounded = execute_tool(
        "read_file",
        '{"path": "caller.py"}',
        repo_root=root,
        sandbox_mode="read-only",
        revision=revision,
    )
    assert "authorize(user)" not in bounded
    assert bounded.endswith("... (truncated)")
    thread = SimpleNamespace(
        id="thread",
        is_resolved=False,
        comments=[
            SimpleNamespace(
                id="comment",
                author="bot",
                path="caller.py",
                prompt_line=1,
                body="**Current code:**\n```python\nauthorize(user)\n```",
            )
        ],
    )
    prior = collect_prior_dotbot_review_comments(
        cast(Any, [thread]),
        {"bot"},
        root,
        source_reader=lambda path: read_file_at_revision(root, revision, path),
    )
    assert prior[0].is_currently_applicable
