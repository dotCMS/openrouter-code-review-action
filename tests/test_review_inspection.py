from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from cli.clients.openrouter_client import OpenRouterClient
from cli.clients.openrouter_thread_store import OpenRouterThreadStore
from cli.clients.openrouter_tools import READ_ONLY_TOOL_SPECS, execute_tool
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
        model_client=cast(Any, object()),
    )
    diff = workflow._panel_diff(cast(Any, SimpleNamespace(base=SimpleNamespace(sha=base))), head)
    assert "+reviewed change" in diff
    assert "unreviewed checkout" not in diff
    assert not (root / "external-diff-executed").exists()


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
