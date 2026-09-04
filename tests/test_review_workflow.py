from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest

from cli.clients.openrouter_thread_store import OpenRouterThreadStore
from cli.core.config import ReviewConfig
from cli.core.exceptions import DotBotExecutionError, ReviewContractError, ReviewResumeError
from cli.core.model_config import DEFAULT_REVIEW_MODEL
from cli.core.models import InlineCommentPayload, ReviewThreadComment, ReviewThreadSnapshot
from cli.review.artifacts import ReviewArtifacts
from cli.review.posting import ReviewPostingOutcome
from cli.review.resume_state import render_review_summary_metadata
from cli.workflows.review_workflow import SUMMARY_MARKER, ReviewSummary, ReviewWorkflow


@dataclass
class _FakeUser:
    login: str = "octocat"


@dataclass
class _FakeHead:
    sha: str = "head-sha"
    ref: str = "feature"
    label: str = "octocat:feature"


@dataclass
class _FakeRepoOwner:
    login: str = "owner"


@dataclass
class _FakeRepo:
    owner: _FakeRepoOwner = field(default_factory=_FakeRepoOwner)
    name: str = "repo"


@dataclass
class _FakeBase:
    ref: str = "main"
    sha: str = "base-sha"
    label: str = "owner:main"
    repo: _FakeRepo = field(default_factory=_FakeRepo)


@dataclass
class _FakeChangedFile:
    filename: str
    status: str = "modified"
    patch: str | None = "@@ -1 +1 @@\n-old\n+new\n"
    previous_filename: str | None = None


class _FakeReviewComment:
    def __init__(self, body: str, *, path: str = "src.py", line: int = 3) -> None:
        self.id = 1
        self.body = body
        self.path = path
        self.line = line
        self.original_line = line
        self.in_reply_to_id = None
        self.diff_hunk = "@@ -1 +1 @@"
        self.commit_id = "head-sha"
        self.user = _FakeUser("reviewer")
        self.created_at = "now"


def _structured_review_body(
    current_code: str,
    *,
    problem: str = "Still broken.",
    language: str = "python",
) -> str:
    return (
        f"**Current code:**\n```{language}\n{current_code}\n```\n\n"
        f"**Problem:** {problem}\n\n"
        f"**Fix:**\n```{language}\n{current_code}\n```\n\n---"
    )


class _FakeIssueComment:
    def __init__(
        self,
        body: str,
        *,
        comment_id: int = 1,
        fail_on_delete: bool = False,
        login: str = "commenter",
    ) -> None:
        self.body = body
        self.id = comment_id
        self.deleted = False
        self.fail_on_delete = fail_on_delete
        self.user = _FakeUser(login)
        self.created_at = "now"

    def delete(self) -> None:
        if self.fail_on_delete:
            raise RuntimeError("permission denied")
        self.deleted = True


class _FakeIssue:
    def __init__(self) -> None:
        self.created_comments: list[str] = []

    def create_comment(self, text: str) -> None:
        self.created_comments.append(text)


class _FakePR:
    def __init__(
        self,
        *,
        issue_comments: list[_FakeIssueComment] | None = None,
        review_comments: list[_FakeReviewComment] | None = None,
        review_threads: list[ReviewThreadSnapshot] | None = None,
        changed_files: list[_FakeChangedFile] | None = None,
    ) -> None:
        self.number = 7
        self.title = "Improve review flow"
        self.body = "PR body"
        self.html_url = "https://example.test/pr/7"
        self.state = "open"
        self.url = "https://api.example.test/repos/owner/repo/pulls/7"
        self.user = _FakeUser()
        self.head: _FakeHead | None = _FakeHead()
        self.base = _FakeBase()
        self._issue_comments = issue_comments or []
        self._review_comments = review_comments or []
        self._review_threads = review_threads or [
            ReviewThreadSnapshot(
                id=f"thread-{index}",
                is_resolved=False,
                comments=[
                    ReviewThreadComment(
                        id=f"comment-{index}",
                        body=comment.body,
                        path=comment.path,
                        line=comment.line,
                        original_line=comment.original_line,
                        author=comment.user.login,
                    )
                ],
            )
            for index, comment in enumerate(self._review_comments, start=1)
        ]
        self._reviews: list[Any] = []
        self._issue = _FakeIssue()
        self._changed_files = changed_files or [_FakeChangedFile("src.py")]

    def get_files(self) -> list[_FakeChangedFile]:
        return list(self._changed_files)

    def get_issue_comments(self) -> list[_FakeIssueComment]:
        return list(self._issue_comments)

    def get_review_comments(self) -> list[_FakeReviewComment]:
        return list(self._review_comments)

    def get_reviews(self) -> list[Any]:
        return list(self._reviews)

    def as_issue(self) -> _FakeIssue:
        return self._issue


class _FakeGitHubClient:
    def __init__(self, pr: _FakePR) -> None:
        self.pr = pr
        self.calls: list[int] = []
        # ``inline_comments`` keeps the same per-comment dict shape the
        # workflow tests originally asserted on, regardless of whether
        # the workflow took the batched-review path or the per-comment
        # fallback path. This lets the existing assertions keep working
        # after Sub-AC 3.3.3's batched submission was wired in.
        self.inline_comments: list[dict[str, Any]] = []
        self.batch_review_calls: list[dict[str, Any]] = []
        self.resolved_thread_ids: list[str] = []

    def get_pr(self, pr_number: int) -> _FakePR:
        self.calls.append(pr_number)
        return self.pr

    def get_review_threads(self, pr: _FakePR) -> list[ReviewThreadSnapshot]:
        assert pr is self.pr
        return list(self.pr._review_threads)

    def resolve_review_thread(self, pr: _FakePR, thread_id: str) -> None:
        assert pr is self.pr
        self.resolved_thread_ids.append(thread_id)

    def post_inline_comment(
        self,
        pr: _FakePR,
        payload: InlineCommentPayload,
        *,
        head_sha: str,
    ) -> None:
        assert pr is self.pr
        self.inline_comments.append(payload.to_request_payload(head_sha))

    def post_pull_request_review(
        self,
        pr: _FakePR,
        payloads: Any,
        *,
        head_sha: str,
    ) -> None:
        assert pr is self.pr
        materialised = list(payloads)
        self.batch_review_calls.append(
            {
                "head_sha": head_sha,
                "comments": [payload.to_review_comment_input() for payload in materialised],
            }
        )
        # Mirror each comment into the ``inline_comments`` log too so
        # tests that grew up around the per-comment endpoint don't have
        # to know whether the workflow took the batched or fallback path.
        for payload in materialised:
            self.inline_comments.append(payload.to_request_payload(head_sha))


class _FakeCodexClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def execute_structured(
        self,
        prompt: str,
        *,
        output_schema: dict[str, object],
        schema_prompt: str,
        sandbox_mode: str,
        model_name: str | None = None,
        resume_thread_id: str | None = None,
    ) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "output_schema": output_schema,
                "schema_prompt": schema_prompt,
                "sandbox_mode": sandbox_mode,
                "model_name": model_name,
                "resume_thread_id": resume_thread_id,
            }
        )
        return self.response


def _make_config(
    tmp_path: Path,
    *,
    dry_run: bool = False,
    model_provider: str = "openrouter",
    review_models: tuple[str, ...] = (),
    model_name: str = "gpt-codex-model",
    resolve_stale_threads: bool = False,
) -> ReviewConfig:
    return ReviewConfig(
        github_token="token",
        repository="owner/repo",
        pr_number=7,
        mode="review",
        model_provider=model_provider,
        dry_run=dry_run,
        model_name=model_name,
        review_models=review_models,
        repo_root=tmp_path,
        resolve_stale_threads=resolve_stale_threads,
    )


def test_process_review_posts_summary_and_passes_dedupe_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "src.py").write_text("value = 1\n", encoding="utf-8")
    prior_summary = _FakeIssueComment(
        f"{SUMMARY_MARKER}\nold summary",
        comment_id=10,
        login="reviewer",
    )
    existing_review_comment = _FakeReviewComment(_structured_review_body("value = 1"))
    pr = _FakePR(
        issue_comments=[prior_summary],
        review_comments=[existing_review_comment],
    )
    github_client = _FakeGitHubClient(pr)
    model_client = _FakeCodexClient(
        json.dumps(
            {
                "overall_correctness": "patch is incorrect",
                "overall_explanation": "Needs one fix.",
                "overall_confidence_score": 0.9,
                "carried_forward": [],
                "findings": [
                    {
                        "title": "Example finding",
                        "body": "Details",
                        "confidence_score": None,
                        "priority": None,
                        "code_location": {
                            "absolute_file_path": str((tmp_path / "src.py").resolve()),
                            "line_range": {"start": 1, "end": 1},
                        },
                    }
                ],
            }
        )
    )
    config = _make_config(tmp_path)
    config.additional_prompt = "Review only security and correctness issues."
    workflow = ReviewWorkflow(
        config,
        github_client=cast(Any, github_client),
        model_client=cast(Any, model_client),
    )

    context_writes: list[tuple[int, ReviewArtifacts, int, int]] = []
    post_calls: list[dict[str, Any]] = []

    def _capture_post_results(
        result, changed_files, current_pr, head_sha, rename_map
    ) -> ReviewPostingOutcome:
        post_calls.append(
            {
                "result": result,
                "changed_files": changed_files,
                "pr": current_pr,
                "head_sha": head_sha,
                "rename_map": rename_map,
            }
        )
        return ReviewPostingOutcome.empty(len(result.findings))

    monkeypatch.setattr(
        workflow.context_manager,
        "write_context_artifacts",
        lambda current_pr, artifacts, *, issue_comments, review_comments: context_writes.append(
            (current_pr.number, artifacts, len(issue_comments), len(review_comments))
        ),
    )
    monkeypatch.setattr(workflow, "_post_results", _capture_post_results)

    result = workflow.process_review(7)

    assert github_client.calls == [7]
    assert context_writes[0][0] == 7
    assert context_writes[0][1] == ReviewArtifacts(
        repo_root=tmp_path,
        context_dir_name=".dotbot-context",
    )
    assert context_writes[0][2:] == (1, 1)
    assert model_client.calls[0]["sandbox_mode"] == "danger-full-access"
    assert "<prior_dotbot_review_comments>" in model_client.calls[0]["schema_prompt"]
    assert '"id": "comment-1"' in model_client.calls[0]["schema_prompt"]
    assert '"path": "src.py"' in model_client.calls[0]["schema_prompt"]
    assert '"current_code": "value = 1"' in model_client.calls[0]["schema_prompt"]
    assert prior_summary.deleted is True
    assert len(pr.as_issue().created_comments) == 1
    assert SUMMARY_MARKER in pr.as_issue().created_comments[0]
    assert (
        render_review_summary_metadata(
            "head-sha", model=DEFAULT_REVIEW_MODEL, reasoning_effort="medium"
        )
        in pr.as_issue().created_comments[0]
    )
    assert "Needs one fix." in pr.as_issue().created_comments[0]
    assert post_calls[0]["head_sha"] == "head-sha"
    assert result.review.overall_correctness == "patch is incorrect"
    assert result.review.carried_forward_comment_ids == []
    assert result.summary == ReviewSummary(
        overall_correctness="patch is incorrect",
        current_findings_count=1,
        carried_forward_count=0,
        active_findings_count=1,
    )
    assert [finding.as_dict() for finding in result.review.findings] == [
        {
            "title": "Example finding",
            "body": "Details",
            "confidence_score": None,
            "priority": None,
            "code_location": {
                "absolute_file_path": str((tmp_path / "src.py").resolve()),
                "line_range": {"start": 1, "end": 1},
            },
            "side": None,
        },
    ]


def test_process_review_dry_run_skips_summary_comment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pr = _FakePR()
    workflow = ReviewWorkflow(
        _make_config(tmp_path, dry_run=True),
        github_client=cast(Any, _FakeGitHubClient(pr)),
        model_client=cast(
            Any,
            _FakeCodexClient(
                json.dumps(
                    {
                        "overall_correctness": "patch is correct",
                        "overall_explanation": "",
                        "overall_confidence_score": None,
                        "carried_forward": [],
                        "findings": [],
                    }
                )
            ),
        ),
    )

    post_calls: list[dict[str, Any]] = []

    def _capture_post_results(
        result, changed_files, current_pr, head_sha, rename_map, **kwargs
    ) -> ReviewPostingOutcome:
        post_calls.append(
            {
                "result": result,
                "changed_files": changed_files,
                "pr": current_pr,
                "head_sha": head_sha,
                "rename_map": rename_map,
                **kwargs,
            }
        )
        return ReviewPostingOutcome.empty(len(result.findings), dry_run=True)

    monkeypatch.setattr(
        workflow.context_manager,
        "write_context_artifacts",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(workflow, "_post_results", _capture_post_results)

    workflow.process_review(7)

    assert pr.as_issue().created_comments == []
    assert post_calls[0]["pr"] is pr
    assert post_calls[0]["head_sha"] == "head-sha"


def test_process_review_summary_counts_carried_forward_codex_comments(tmp_path: Path) -> None:
    (tmp_path / "src.py").write_text("value = 1\n", encoding="utf-8")
    pr = _FakePR(
        issue_comments=[
            _FakeIssueComment(
                f"{SUMMARY_MARKER}\nold summary",
                comment_id=10,
                login="reviewer",
            )
        ],
        review_threads=[
            ReviewThreadSnapshot(
                id="thread-1",
                is_resolved=False,
                comments=[
                    ReviewThreadComment(
                        id="comment-1",
                        body=_structured_review_body("value = 1"),
                        path="src.py",
                        line=2,
                        original_line=2,
                        author="reviewer",
                    )
                ],
            )
        ],
    )
    workflow = ReviewWorkflow(
        _make_config(tmp_path),
        github_client=cast(Any, _FakeGitHubClient(pr)),
        model_client=cast(
            Any,
            _FakeCodexClient(
                json.dumps(
                    {
                        "overall_correctness": "patch is correct",
                        "overall_explanation": "No additional non-redundant findings.",
                        "overall_confidence_score": 0.8,
                        "carried_forward": [
                            {
                                "comment_id": "comment-1",
                                "current_evidence": "value = 1",
                            }
                        ],
                        "findings": [],
                    }
                )
            ),
        ),
    )

    result = workflow.process_review(7)

    assert result.review.overall_correctness == "patch is correct"
    assert result.review.findings == []
    assert result.review.carried_forward_comment_ids == ["comment-1"]
    assert result.summary == ReviewSummary(
        overall_correctness="patch is incorrect",
        current_findings_count=0,
        carried_forward_count=1,
        active_findings_count=1,
    )
    assert "- New findings this run: 0" in pr.as_issue().created_comments[0]
    assert (
        "- Prior unresolved dotbot findings still relevant: 1" in pr.as_issue().created_comments[0]
    )
    assert "- Active findings total: 1" in pr.as_issue().created_comments[0]
    assert (
        "No new actionable bugs were found in the current changes, but 1 prior unresolved "
        "dotbot finding still applies, so the patch remains incorrect."
        in pr.as_issue().created_comments[0]
    )
    assert "No additional non-redundant findings." not in pr.as_issue().created_comments[0]
    assert (
        render_review_summary_metadata(
            "head-sha", model=DEFAULT_REVIEW_MODEL, reasoning_effort="medium"
        )
        in pr.as_issue().created_comments[0]
    )


def test_process_review_ignores_stale_prior_codex_thread(tmp_path: Path) -> None:
    (tmp_path / "src.py").write_text("value = 2\n", encoding="utf-8")
    pr = _FakePR(
        issue_comments=[
            _FakeIssueComment(
                f"{SUMMARY_MARKER}\nold summary",
                comment_id=10,
                login="reviewer",
            )
        ],
        review_threads=[
            ReviewThreadSnapshot(
                id="thread-1",
                is_resolved=False,
                comments=[
                    ReviewThreadComment(
                        id="comment-1",
                        body=_structured_review_body("value = 1"),
                        path="src.py",
                        line=2,
                        original_line=2,
                        author="reviewer",
                    )
                ],
            )
        ],
    )
    github_client = _FakeGitHubClient(pr)
    model_client = _FakeCodexClient(
        json.dumps(
            {
                "overall_correctness": "patch is correct",
                "overall_explanation": "No active issues remain.",
                "overall_confidence_score": 0.8,
                "carried_forward": [],
                "findings": [],
            }
        )
    )
    config = _make_config(tmp_path)
    config.additional_prompt = "Review only security and correctness issues."
    workflow = ReviewWorkflow(
        config,
        github_client=cast(Any, github_client),
        model_client=cast(Any, model_client),
    )

    result = workflow.process_review(7)

    assert "<prior_dotbot_review_comments>" not in model_client.calls[0]["schema_prompt"]
    assert result.summary == ReviewSummary(
        overall_correctness="patch is correct",
        current_findings_count=0,
        carried_forward_count=0,
        active_findings_count=0,
    )
    assert "- Active findings total: 0" in pr.as_issue().created_comments[0]


def test_process_review_dry_run_ignores_stale_prior_codex_thread(tmp_path: Path) -> None:
    (tmp_path / "src.py").write_text("value = 2\n", encoding="utf-8")
    pr = _FakePR(
        issue_comments=[
            _FakeIssueComment(
                f"{SUMMARY_MARKER}\nold summary",
                comment_id=10,
                login="reviewer",
            )
        ],
        review_threads=[
            ReviewThreadSnapshot(
                id="thread-1",
                is_resolved=False,
                comments=[
                    ReviewThreadComment(
                        id="comment-1",
                        body=_structured_review_body("value = 1"),
                        path="src.py",
                        line=2,
                        original_line=2,
                        author="reviewer",
                    )
                ],
            )
        ],
    )
    github_client = _FakeGitHubClient(pr)
    workflow = ReviewWorkflow(
        _make_config(tmp_path, dry_run=True),
        github_client=cast(Any, github_client),
        model_client=cast(
            Any,
            _FakeCodexClient(
                json.dumps(
                    {
                        "overall_correctness": "patch is correct",
                        "overall_explanation": "",
                        "overall_confidence_score": 0.8,
                        "carried_forward": [],
                        "findings": [],
                    }
                )
            ),
        ),
    )

    workflow.process_review(7)


def test_process_review_matches_github_bot_logins_across_issue_and_thread_apis(
    tmp_path: Path,
) -> None:
    (tmp_path / "action.yml").write_text(
        "name: action\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - name: Save review dotbot cache\n"
        "      if: ${{ inputs.mode == 'review' && steps.run_codex_cli.outcome == 'success' && steps.review_resume_state.outputs.current_cache_key != '' && !(steps.review_codex_cache.outputs.cache-hit == 'true' && steps.review_resume_state.outputs.restore_key == steps.review_resume_state.outputs.current_cache_key) }}\n"
        "      uses: actions/cache/save@v4\n",
        encoding="utf-8",
    )
    stale_body = (
        "**Current code:**\n```yaml\n"
        "    - name: Save review dotbot cache\n"
        "      if: ${{ inputs.mode == 'review' && steps.run_codex_cli.outcome == 'success' && steps.review_resume_state.outputs.current_cache_key != '' }}\n"
        "      uses: actions/cache/save@v4\n"
        "```\n\n"
        "**Problem:** stale.\n\n"
        "**Fix:**\n```yaml\n"
        "    - name: Save review dotbot cache\n"
        "      if: ${{ inputs.mode == 'review' && steps.run_codex_cli.outcome == 'success' && steps.review_resume_state.outputs.current_cache_key != '' && !(steps.review_codex_cache.outputs.cache-hit == 'true' && steps.review_resume_state.outputs.restore_key == steps.review_resume_state.outputs.current_cache_key) }}\n"
        "      uses: actions/cache/save@v4\n"
        "```\n\n---"
    )
    pr = _FakePR(
        issue_comments=[
            _FakeIssueComment(
                f"{SUMMARY_MARKER}\nold summary",
                comment_id=10,
                login="github-actions[bot]",
            )
        ],
        review_threads=[
            ReviewThreadSnapshot(
                id="thread-1",
                is_resolved=False,
                comments=[
                    ReviewThreadComment(
                        id="comment-1",
                        body=stale_body,
                        path="action.yml",
                        line=136,
                        original_line=136,
                        author="github-actions",
                    )
                ],
            )
        ],
    )
    github_client = _FakeGitHubClient(pr)
    model_client = _FakeCodexClient(
        json.dumps(
            {
                "overall_correctness": "patch is correct",
                "overall_explanation": "No active issues remain.",
                "overall_confidence_score": 0.8,
                "carried_forward": [],
                "findings": [],
            }
        )
    )
    config = _make_config(tmp_path)
    config.additional_prompt = "Review only security and correctness issues."
    workflow = ReviewWorkflow(
        config,
        github_client=cast(Any, github_client),
        model_client=cast(Any, model_client),
    )

    result = workflow.process_review(7)

    assert result.review.carried_forward_comment_ids == []


def test_process_review_drops_invalid_carried_forward_entries(tmp_path: Path) -> None:
    (tmp_path / "src.py").write_text("value = 1\n", encoding="utf-8")
    pr = _FakePR(
        issue_comments=[
            _FakeIssueComment(
                f"{SUMMARY_MARKER}\nold summary",
                comment_id=10,
                login="reviewer",
            )
        ],
        review_threads=[
            ReviewThreadSnapshot(
                id="thread-1",
                is_resolved=False,
                comments=[
                    ReviewThreadComment(
                        id="comment-1",
                        body=_structured_review_body("value = 1"),
                        path="src.py",
                        line=2,
                        original_line=2,
                        author="reviewer",
                    )
                ],
            )
        ],
    )
    github_client = _FakeGitHubClient(pr)
    workflow = ReviewWorkflow(
        _make_config(tmp_path),
        github_client=cast(Any, github_client),
        model_client=cast(
            Any,
            _FakeCodexClient(
                json.dumps(
                    {
                        "overall_correctness": "patch is incorrect",
                        "overall_explanation": "",
                        "overall_confidence_score": 0.8,
                        "carried_forward": [
                            {
                                "comment_id": "comment-1",
                                "current_evidence": "value = 1",
                            }
                        ]
                        + [
                            {
                                "comment_id": "comment-unknown",
                                "current_evidence": "value = 1",
                            }
                        ],
                        "findings": [],
                    }
                )
            ),
        ),
    )

    result = workflow.process_review(7)

    assert result.review.carried_forward_comment_ids == ["comment-1"]


def test_process_review_keeps_new_finding_when_stale_prior_thread_exists(tmp_path: Path) -> None:
    sample_file = tmp_path / "src.py"
    sample_file.write_text("value = 2\n", encoding="utf-8")
    pr = _FakePR(
        issue_comments=[
            _FakeIssueComment(
                f"{SUMMARY_MARKER}\nold summary",
                comment_id=10,
                login="reviewer",
            )
        ],
        review_threads=[
            ReviewThreadSnapshot(
                id="thread-1",
                is_resolved=False,
                comments=[
                    ReviewThreadComment(
                        id="comment-1",
                        body=_structured_review_body("value = 1"),
                        path="src.py",
                        line=2,
                        original_line=2,
                        author="reviewer",
                    )
                ],
            )
        ],
    )
    github_client = _FakeGitHubClient(pr)
    workflow = ReviewWorkflow(
        _make_config(tmp_path),
        github_client=cast(Any, github_client),
        model_client=cast(
            Any,
            _FakeCodexClient(
                json.dumps(
                    {
                        "overall_correctness": "patch is incorrect",
                        "overall_explanation": "Issue still exists in updated form.",
                        "overall_confidence_score": 0.9,
                        "carried_forward": [],
                        "findings": [
                            {
                                "title": "Updated finding",
                                "body": "Still broken nearby.",
                                "confidence_score": 0.9,
                                "priority": 1,
                                "code_location": {
                                    "absolute_file_path": str(sample_file.resolve()),
                                    "line_range": {"start": 1, "end": 2},
                                },
                            }
                        ],
                    }
                )
            ),
        ),
    )

    workflow.process_review(7)


def test_process_review_warns_when_prior_summary_delete_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prior_summary = _FakeIssueComment(
        f"{SUMMARY_MARKER}\nold summary",
        comment_id=99,
        fail_on_delete=True,
    )
    pr = _FakePR(issue_comments=[prior_summary])
    workflow = ReviewWorkflow(
        _make_config(tmp_path),
        github_client=cast(Any, _FakeGitHubClient(pr)),
        model_client=cast(
            Any,
            _FakeCodexClient(
                json.dumps(
                    {
                        "overall_correctness": "patch is correct",
                        "overall_explanation": "",
                        "overall_confidence_score": None,
                        "carried_forward": [],
                        "findings": [],
                    }
                )
            ),
        ),
    )

    monkeypatch.setattr(
        workflow.context_manager,
        "write_context_artifacts",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        workflow,
        "_post_results",
        lambda *args, **kwargs: ReviewPostingOutcome.empty(0),
    )

    workflow.process_review(7)

    err = capsys.readouterr().err
    assert "Failed to delete prior summary issue comment id=99: permission denied" in err
    assert len(pr.as_issue().created_comments) == 1


def test_process_review_resumes_prior_thread_with_inline_incremental_diff(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "src.py").write_text("value = 2\n", encoding="utf-8")
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    prior_summary = _FakeIssueComment(
        (f"{SUMMARY_MARKER}\n{render_review_summary_metadata('prev-sha')}\nold summary"),
        comment_id=10,
        login="reviewer",
    )
    pr = _FakePR(issue_comments=[prior_summary])
    github_client = _FakeGitHubClient(pr)
    model_client = _FakeCodexClient(
        json.dumps(
            {
                "overall_correctness": "patch is correct",
                "overall_explanation": "",
                "overall_confidence_score": None,
                "carried_forward": [],
                "findings": [],
            }
        )
    )
    config = _make_config(tmp_path, model_provider="openai")
    config.additional_prompt = "Review only security and correctness issues."
    workflow = ReviewWorkflow(
        config,
        github_client=cast(Any, github_client),
        model_client=cast(Any, model_client),
    )

    monkeypatch.setattr(
        workflow.context_manager,
        "write_context_artifacts",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "cli.workflows.review_workflow.compose_prompt",
        lambda *args, **kwargs: "FULL PR PROMPT",
    )
    monkeypatch.setattr(
        workflow,
        "_post_results",
        lambda *args, **kwargs: ReviewPostingOutcome.empty(0),
    )
    monkeypatch.setattr("cli.workflows.review_workflow.git_is_ancestor", lambda older, newer: True)
    monkeypatch.setattr(
        "cli.workflows.review_workflow.git_diff_text",
        lambda revision_range: "@@ -1 +1 @@\n-value = 1\n+value = 2\n",
    )
    monkeypatch.setattr(
        "cli.workflows.review_workflow.git_commit_shas",
        lambda revision_range: ["commit-1"],
    )
    monkeypatch.setattr(
        "cli.workflows.review_workflow.load_latest_thread_id",
        lambda codex_home, cwd: "thread-1",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setenv("DOTBOT_REVIEW_PREVIOUS_HEAD_SHA", "prev-sha")
    monkeypatch.setenv("DOTBOT_REVIEW_CACHE_HIT", "true")

    workflow.process_review(7)

    assert model_client.calls[0]["resume_thread_id"] == "thread-1"
    assert "<review_resume_context>" in model_client.calls[0]["prompt"]
    assert (
        "<previous_reviewed_head_sha>prev-sha</previous_reviewed_head_sha>"
        in model_client.calls[0]["prompt"]
    )
    assert "<current_head_sha>head-sha</current_head_sha>" in model_client.calls[0]["prompt"]
    assert "<incremental_diff>" in model_client.calls[0]["prompt"]
    assert "+value = 2" in model_client.calls[0]["prompt"]
    assert "Review only security and correctness issues." in model_client.calls[0]["prompt"]
    assert "FULL PR PROMPT" not in model_client.calls[0]["prompt"]


def test_process_review_resumes_openrouter_thread_from_thread_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "src.py").write_text("value = 2\n", encoding="utf-8")
    thread_dir = tmp_path / "openrouter-threads"
    store = OpenRouterThreadStore(thread_dir)
    store.save(
        "thread-or-1",
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "prior review prompt"},
        ],
        repository="owner/repo",
        pr_number=7,
        model="deepseek/deepseek-v4-pro-0813",
    )
    prior_summary = _FakeIssueComment(
        (f"{SUMMARY_MARKER}\n{render_review_summary_metadata('prev-sha')}\nold summary"),
        comment_id=10,
        login="reviewer",
    )
    pr = _FakePR(issue_comments=[prior_summary])
    github_client = _FakeGitHubClient(pr)
    model_client = _FakeCodexClient(
        json.dumps(
            {
                "overall_correctness": "patch is correct",
                "overall_explanation": "",
                "overall_confidence_score": None,
                "carried_forward": [],
                "findings": [],
            }
        )
    )
    workflow = ReviewWorkflow(
        _make_config(tmp_path),
        github_client=cast(Any, github_client),
        model_client=cast(Any, model_client),
    )

    monkeypatch.setattr(
        workflow.context_manager,
        "write_context_artifacts",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "cli.workflows.review_workflow.compose_prompt",
        lambda *args, **kwargs: "FULL PR PROMPT",
    )
    monkeypatch.setattr(
        workflow,
        "_post_results",
        lambda *args, **kwargs: ReviewPostingOutcome.empty(0),
    )
    monkeypatch.setattr("cli.workflows.review_workflow.git_is_ancestor", lambda older, newer: True)
    monkeypatch.setattr(
        "cli.workflows.review_workflow.git_diff_text",
        lambda revision_range: "@@ -1 +1 @@\n-value = 1\n+value = 2\n",
    )
    monkeypatch.setattr(
        "cli.workflows.review_workflow.git_commit_shas",
        lambda revision_range: ["commit-1"],
    )
    monkeypatch.setenv("OPENROUTER_THREAD_DIR", str(thread_dir))
    monkeypatch.setenv("DOTBOT_REVIEW_PREVIOUS_HEAD_SHA", "prev-sha")
    monkeypatch.setenv("DOTBOT_REVIEW_CACHE_HIT", "true")
    monkeypatch.delenv("CODEX_HOME", raising=False)

    workflow.process_review(7)

    assert model_client.calls[0]["resume_thread_id"] == "thread-or-1"
    assert "<review_resume_context>" in model_client.calls[0]["prompt"]
    assert "<incremental_diff>" in model_client.calls[0]["prompt"]
    assert "FULL PR PROMPT" not in model_client.calls[0]["prompt"]


def test_process_review_falls_back_to_fresh_review_when_openrouter_thread_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prior_summary = _FakeIssueComment(
        (f"{SUMMARY_MARKER}\n{render_review_summary_metadata('prev-sha')}\nold summary"),
        comment_id=10,
        login="reviewer",
    )
    pr = _FakePR(issue_comments=[prior_summary])
    github_client = _FakeGitHubClient(pr)
    model_client = _FakeCodexClient(
        json.dumps(
            {
                "overall_correctness": "patch is correct",
                "overall_explanation": "",
                "overall_confidence_score": None,
                "carried_forward": [],
                "findings": [],
            }
        )
    )
    workflow = ReviewWorkflow(
        _make_config(tmp_path),
        github_client=cast(Any, github_client),
        model_client=cast(Any, model_client),
    )

    monkeypatch.setattr(
        workflow.context_manager,
        "write_context_artifacts",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "cli.workflows.review_workflow.compose_prompt",
        lambda *args, **kwargs: "FULL PR PROMPT",
    )
    monkeypatch.setattr(
        workflow,
        "_post_results",
        lambda *args, **kwargs: ReviewPostingOutcome.empty(0),
    )
    monkeypatch.setattr("cli.workflows.review_workflow.git_is_ancestor", lambda older, newer: True)
    monkeypatch.setenv("OPENROUTER_THREAD_DIR", str(tmp_path / "empty-threads"))
    monkeypatch.setenv("DOTBOT_REVIEW_PREVIOUS_HEAD_SHA", "prev-sha")
    monkeypatch.setenv("DOTBOT_REVIEW_CACHE_HIT", "true")
    monkeypatch.delenv("CODEX_HOME", raising=False)

    workflow.process_review(7)

    assert model_client.calls[0]["resume_thread_id"] is None
    assert "<review_resume_context>" not in model_client.calls[0]["prompt"]
    assert "FULL PR PROMPT" in model_client.calls[0]["prompt"]


def test_process_review_falls_back_to_fresh_review_when_prior_sha_is_not_ancestor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prior_summary = _FakeIssueComment(
        (f"{SUMMARY_MARKER}\n{render_review_summary_metadata('prev-sha')}\nold summary"),
        comment_id=10,
        login="reviewer",
    )
    pr = _FakePR(issue_comments=[prior_summary])
    github_client = _FakeGitHubClient(pr)
    model_client = _FakeCodexClient(
        json.dumps(
            {
                "overall_correctness": "patch is correct",
                "overall_explanation": "",
                "overall_confidence_score": None,
                "carried_forward": [],
                "findings": [],
            }
        )
    )
    workflow = ReviewWorkflow(
        _make_config(tmp_path),
        github_client=cast(Any, github_client),
        model_client=cast(Any, model_client),
    )

    monkeypatch.setattr(
        workflow.context_manager,
        "write_context_artifacts",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        workflow,
        "_post_results",
        lambda *args, **kwargs: ReviewPostingOutcome.empty(0),
    )
    monkeypatch.setattr("cli.workflows.review_workflow.git_is_ancestor", lambda older, newer: False)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    monkeypatch.setenv("DOTBOT_REVIEW_PREVIOUS_HEAD_SHA", "prev-sha")
    monkeypatch.setenv("DOTBOT_REVIEW_CACHE_HIT", "true")

    workflow.process_review(7)

    assert model_client.calls[0]["resume_thread_id"] is None
    assert "<review_resume_context>" not in model_client.calls[0]["prompt"]


def test_process_review_raises_when_code_home_is_missing_after_cache_restore(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prior_summary = _FakeIssueComment(
        f"{SUMMARY_MARKER}\n{render_review_summary_metadata('prev-sha')}\nold summary",
    )
    workflow = ReviewWorkflow(
        _make_config(tmp_path, model_provider="openai"),
        github_client=cast(Any, _FakeGitHubClient(_FakePR(issue_comments=[prior_summary]))),
        model_client=cast(Any, _FakeCodexClient("{}")),
    )

    monkeypatch.setattr("cli.workflows.review_workflow.git_is_ancestor", lambda older, newer: True)
    monkeypatch.setenv("DOTBOT_REVIEW_PREVIOUS_HEAD_SHA", "prev-sha")
    monkeypatch.setenv("DOTBOT_REVIEW_CACHE_HIT", "true")
    monkeypatch.delenv("CODEX_HOME", raising=False)

    with pytest.raises(ReviewResumeError, match="CODEX_HOME is unset"):
        workflow.process_review(7)


def test_process_review_falls_back_to_fresh_review_when_cached_thread_lookup_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prior_summary = _FakeIssueComment(
        f"{SUMMARY_MARKER}\n{render_review_summary_metadata('prev-sha')}\nold summary",
    )
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    model_client = _FakeCodexClient(
        json.dumps(
            {
                "overall_correctness": "patch is correct",
                "overall_explanation": "",
                "overall_confidence_score": None,
                "carried_forward": [],
                "findings": [],
            }
        )
    )
    workflow = ReviewWorkflow(
        _make_config(tmp_path),
        github_client=cast(Any, _FakeGitHubClient(_FakePR(issue_comments=[prior_summary]))),
        model_client=cast(Any, model_client),
    )

    monkeypatch.setattr("cli.workflows.review_workflow.git_is_ancestor", lambda older, newer: True)
    monkeypatch.setattr(
        "cli.workflows.review_workflow.load_latest_thread_id",
        lambda codex_home, cwd: (_ for _ in ()).throw(ReviewResumeError("thread lookup failed")),
    )
    monkeypatch.setattr(
        workflow.context_manager,
        "write_context_artifacts",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        workflow,
        "_post_results",
        lambda *args, **kwargs: ReviewPostingOutcome.empty(0),
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setenv("DOTBOT_REVIEW_PREVIOUS_HEAD_SHA", "prev-sha")
    monkeypatch.setenv("DOTBOT_REVIEW_CACHE_HIT", "true")

    workflow.process_review(7)

    assert model_client.calls[0]["resume_thread_id"] is None
    assert "<review_resume_context>" not in model_client.calls[0]["prompt"]


def test_process_review_falls_back_to_fresh_review_when_resume_ancestry_check_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prior_summary = _FakeIssueComment(
        f"{SUMMARY_MARKER}\n{render_review_summary_metadata('prev-sha')}\nold summary",
    )
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    model_client = _FakeCodexClient(
        json.dumps(
            {
                "overall_correctness": "patch is correct",
                "overall_explanation": "",
                "overall_confidence_score": None,
                "carried_forward": [],
                "findings": [],
            }
        )
    )
    workflow = ReviewWorkflow(
        _make_config(tmp_path),
        github_client=cast(Any, _FakeGitHubClient(_FakePR(issue_comments=[prior_summary]))),
        model_client=cast(Any, model_client),
    )

    def _raise_git(*args: object, **kwargs: object) -> bool:
        raise subprocess.CalledProcessError(
            128,
            ["/usr/bin/git", "merge-base", "--is-ancestor", "prev-sha", "head-sha"],
            "",
            "fatal: Not a valid commit name prev-sha",
        )

    monkeypatch.setattr("cli.workflows.review_workflow.git_is_ancestor", _raise_git)
    monkeypatch.setattr(
        workflow.context_manager,
        "write_context_artifacts",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        workflow,
        "_post_results",
        lambda *args, **kwargs: ReviewPostingOutcome.empty(0),
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setenv("DOTBOT_REVIEW_PREVIOUS_HEAD_SHA", "prev-sha")
    monkeypatch.setenv("DOTBOT_REVIEW_CACHE_HIT", "true")

    workflow.process_review(7)

    assert model_client.calls[0]["resume_thread_id"] is None
    assert "<review_resume_context>" not in model_client.calls[0]["prompt"]


def test_process_review_raises_when_incremental_git_context_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prior_summary = _FakeIssueComment(
        f"{SUMMARY_MARKER}\n{render_review_summary_metadata('prev-sha')}\nold summary",
    )
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    workflow = ReviewWorkflow(
        _make_config(tmp_path, model_provider="openai"),
        github_client=cast(Any, _FakeGitHubClient(_FakePR(issue_comments=[prior_summary]))),
        model_client=cast(Any, _FakeCodexClient("{}")),
    )

    monkeypatch.setattr("cli.workflows.review_workflow.git_is_ancestor", lambda older, newer: True)
    monkeypatch.setattr(
        "cli.workflows.review_workflow.load_latest_thread_id",
        lambda codex_home, cwd: "thread-1",
    )

    def _raise_diff(revision_range: str) -> str:
        _ = revision_range
        raise subprocess.CalledProcessError(1, "git diff")

    monkeypatch.setattr("cli.workflows.review_workflow.git_diff_text", _raise_diff)
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setenv("DOTBOT_REVIEW_PREVIOUS_HEAD_SHA", "prev-sha")
    monkeypatch.setenv("DOTBOT_REVIEW_CACHE_HIT", "true")

    with pytest.raises(ReviewResumeError, match="Failed to compute incremental review context"):
        workflow.process_review(7)


def test_process_review_raises_for_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pr = _FakePR()
    workflow = ReviewWorkflow(
        _make_config(tmp_path),
        github_client=cast(Any, _FakeGitHubClient(pr)),
        model_client=cast(Any, _FakeCodexClient("not-json")),
    )

    monkeypatch.setattr(
        workflow.context_manager,
        "write_context_artifacts",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        workflow,
        "_post_results",
        lambda *args, **kwargs: pytest.fail("_post_results should not run on invalid JSON"),
    )

    with pytest.raises(DotBotExecutionError, match="JSON parsing error"):
        workflow.process_review(7)

    assert pr.as_issue().created_comments == []


def test_process_review_raises_for_invalid_structured_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pr = _FakePR()
    workflow = ReviewWorkflow(
        _make_config(tmp_path),
        github_client=cast(Any, _FakeGitHubClient(pr)),
        model_client=cast(
            Any,
            _FakeCodexClient(
                json.dumps(
                    {
                        "overall_correctness": "patch is incorrect",
                        "overall_explanation": "bad payload",
                        "findings": [],
                    }
                )
            ),
        ),
    )

    monkeypatch.setattr(
        workflow.context_manager,
        "write_context_artifacts",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        workflow,
        "_post_results",
        lambda *args, **kwargs: pytest.fail(
            "_post_results should not run on payload contract errors"
        ),
    )

    with pytest.raises(ReviewContractError, match="missing required fields"):
        workflow.process_review(7)


def test_process_review_raises_domain_error_for_missing_head_sha(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pr = _FakePR()
    pr.head = None
    workflow = ReviewWorkflow(
        _make_config(tmp_path),
        github_client=cast(Any, _FakeGitHubClient(pr)),
        model_client=cast(
            Any,
            _FakeCodexClient(
                json.dumps(
                    {
                        "overall_correctness": "patch is correct",
                        "overall_explanation": "",
                        "overall_confidence_score": None,
                        "carried_forward": [],
                        "findings": [],
                    }
                )
            ),
        ),
    )

    monkeypatch.setattr(
        workflow.context_manager,
        "write_context_artifacts",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(ReviewContractError, match="Missing PR head commit SHA"):
        workflow.process_review(7)


def test_process_review_raises_domain_error_for_issue_comment_snapshot_failure(
    tmp_path: Path,
) -> None:
    class _BrokenPR(_FakePR):
        def get_issue_comments(self) -> list[Any]:
            raise RuntimeError("issue fetch failed")

    pr = _BrokenPR()
    model_client = _FakeCodexClient(
        json.dumps(
            {
                "overall_correctness": "patch is correct",
                "overall_explanation": "",
                "overall_confidence_score": None,
                "carried_forward": [],
                "findings": [],
            }
        )
    )
    workflow = ReviewWorkflow(
        _make_config(tmp_path),
        github_client=cast(Any, _FakeGitHubClient(pr)),
        model_client=cast(Any, model_client),
    )

    with pytest.raises(ReviewContractError, match="Failed to retrieve issue comments"):
        workflow.process_review(7)

    assert model_client.calls == []
    assert pr.as_issue().created_comments == []


def test_process_review_wires_real_artifacts_and_inline_posting(tmp_path: Path) -> None:
    sample_file = tmp_path / "src.py"
    sample_file.write_text("old\nnew\n", encoding="utf-8")

    pr = _FakePR(
        changed_files=[
            _FakeChangedFile(
                "src.py",
                patch="@@ -1,1 +1,2 @@\n old\n+new\n",
            )
        ]
    )
    github_client = _FakeGitHubClient(pr)
    model_client = _FakeCodexClient(
        json.dumps(
            {
                "overall_correctness": "patch is incorrect",
                "overall_explanation": "Needs one focused fix.",
                "overall_confidence_score": 0.8,
                "carried_forward": [],
                "findings": [
                    {
                        "title": "Example finding",
                        "body": "Please adjust this line.",
                        "confidence_score": 0.9,
                        "priority": 1,
                        "code_location": {
                            "absolute_file_path": str(sample_file.resolve()),
                            "line_range": {"start": 2, "end": 2},
                        },
                    }
                ],
            }
        )
    )
    workflow = ReviewWorkflow(
        _make_config(tmp_path),
        github_client=cast(Any, github_client),
        model_client=cast(Any, model_client),
    )

    result = workflow.process_review(7)

    artifacts = ReviewArtifacts(repo_root=tmp_path, context_dir_name=".dotbot-context")
    assert artifacts.pr_metadata_path.exists()
    assert artifacts.review_comments_path.exists()
    assert artifacts.anchor_maps_path.exists()
    assert "PR #7: Improve review flow" in artifacts.pr_metadata_path.read_text(encoding="utf-8")
    assert len(pr.as_issue().created_comments) == 1
    assert "Needs one focused fix." in pr.as_issue().created_comments[0]
    assert github_client.inline_comments == [
        {
            "body": "Example finding\n\nPlease adjust this line.",
            "path": "src.py",
            "side": "RIGHT",
            "commit_id": "head-sha",
            "line": 2,
        }
    ]
    assert result.review.findings[0].code_location.as_dict() == {
        "absolute_file_path": str(sample_file.resolve()),
        "line_range": {"start": 2, "end": 2},
    }


def test_process_review_runs_each_model_in_roster_and_posts_per_model_summary(
    tmp_path: Path,
) -> None:
    sample_file = tmp_path / "src.py"
    sample_file.write_text("old\nnew\n", encoding="utf-8")

    pr = _FakePR(
        changed_files=[
            _FakeChangedFile(
                "src.py",
                patch="@@ -1,1 +1,2 @@\n old\n+new\n",
            )
        ]
    )
    github_client = _FakeGitHubClient(pr)
    model_client = _FakeCodexClient(
        json.dumps(
            {
                "overall_correctness": "patch is correct",
                "overall_explanation": "",
                "overall_confidence_score": 0.8,
                "carried_forward": [],
                "findings": [],
            }
        )
    )
    workflow = ReviewWorkflow(
        _make_config(
            tmp_path,
            review_models=("openai/gpt-5", "google/gemini-2.5-pro"),
        ),
        github_client=cast(Any, github_client),
        model_client=cast(Any, model_client),
    )

    result = workflow.process_review(7)

    # The roster is primary + 2 extras = 3 review passes.
    assert len(model_client.calls) == 3
    assert [call["model_name"] for call in model_client.calls] == [
        DEFAULT_REVIEW_MODEL,
        "openai/gpt-5",
        "google/gemini-2.5-pro",
    ]
    # One summary posted per reviewer, each tagged with its model.
    created = pr.as_issue().created_comments
    assert len(created) == 3
    assert f"- Reviewer: {DEFAULT_REVIEW_MODEL}" in created[0]
    assert "- Reviewer: openai/gpt-5" in created[1]
    assert "- Reviewer: google/gemini-2.5-pro" in created[2]
    assert result.summary.overall_correctness == "patch is correct"


def test_process_review_openai_provider_uses_config_model_name_not_roster(
    tmp_path: Path,
) -> None:
    pr = _FakePR()
    github_client = _FakeGitHubClient(pr)
    model_client = _FakeCodexClient(
        json.dumps(
            {
                "overall_correctness": "patch is correct",
                "overall_explanation": "",
                "overall_confidence_score": 0.8,
                "carried_forward": [],
                "findings": [],
            }
        )
    )
    workflow = ReviewWorkflow(
        # openai provider must keep using config.model_name (the Codex client
        # contract), not the review-model roster.
        _make_config(
            tmp_path,
            model_provider="openai",
            review_models=("openai/gpt-5",),
            model_name="gpt-codex-model",
        ),
        github_client=cast(Any, github_client),
        model_client=cast(Any, model_client),
    )

    result = workflow.process_review(7)

    assert [call["model_name"] for call in model_client.calls] == [
        "gpt-codex-model",
        "gpt-codex-model",
    ]
    assert "- Reviewer: gpt-codex-model" in pr.as_issue().created_comments[0]
    assert "- Reviewer: gpt-codex-model" in pr.as_issue().created_comments[1]
    assert result.summary.overall_correctness == "patch is correct"


def test_process_review_renamed_file_posts_current_findings_without_prefilter(
    tmp_path: Path,
) -> None:
    renamed_file = tmp_path / "new_name.py"
    renamed_file.write_text("one\ntwo\nthree\nfour\nfive\n", encoding="utf-8")

    pr = _FakePR(
        issue_comments=[
            _FakeIssueComment(
                f"{SUMMARY_MARKER}\nold summary",
                comment_id=21,
                login="reviewer",
            )
        ],
        review_comments=[_FakeReviewComment("existing", path="new_name.py", line=1)],
        changed_files=[
            _FakeChangedFile(
                "new_name.py",
                status="renamed",
                previous_filename="old_name.py",
                patch="@@ -0,0 +1,5 @@\n+one\n+two\n+three\n+four\n+five\n",
            )
        ],
    )
    github_client = _FakeGitHubClient(pr)
    model_client = _FakeCodexClient(
        json.dumps(
            {
                "overall_correctness": "patch is incorrect",
                "overall_explanation": "Needs one follow-up.",
                "overall_confidence_score": 0.8,
                "carried_forward": [],
                "findings": [
                    {
                        "title": "New finding",
                        "body": "Not covered.",
                        "confidence_score": 0.9,
                        "priority": 1,
                        "code_location": {
                            "absolute_file_path": str((tmp_path / "old_name.py").resolve()),
                            "line_range": {"start": 5, "end": 5},
                        },
                    },
                ],
            }
        )
    )
    workflow = ReviewWorkflow(
        _make_config(tmp_path),
        github_client=cast(Any, github_client),
        model_client=cast(Any, model_client),
    )

    result = workflow.process_review(7)

    assert len(github_client.inline_comments) == 1
    assert github_client.inline_comments[0]["path"] == "new_name.py"
    assert github_client.inline_comments[0]["line"] == 5

    assert result.posting_outcome.total_findings == 1
    assert result.posting_outcome.prefiltered_count == 0
    assert result.posting_outcome.published_count == 1


def test_process_review_ignores_non_codex_threads_in_rerun_context(tmp_path: Path) -> None:
    sample_file = tmp_path / "src.py"
    sample_file.write_text("old\nnew\n", encoding="utf-8")

    pr = _FakePR(
        issue_comments=[
            _FakeIssueComment(
                f"{SUMMARY_MARKER}\nold summary",
                comment_id=10,
                login="codex-bot",
            )
        ],
        review_threads=[
            ReviewThreadSnapshot(
                id="thread-1",
                is_resolved=True,
                comments=[
                    ReviewThreadComment(
                        id="comment-0",
                        body="Resolved dotbot finding",
                        path="src.py",
                        line=1,
                        original_line=1,
                        author="codex-bot",
                    )
                ],
            ),
            ReviewThreadSnapshot(
                id="thread-2",
                is_resolved=False,
                comments=[
                    ReviewThreadComment(
                        id="comment-1",
                        body="🔴 [P1] Existing finding",
                        path="src.py",
                        line=2,
                        original_line=2,
                        author="human-reviewer",
                    )
                ],
            ),
        ],
        changed_files=[
            _FakeChangedFile(
                "src.py",
                patch="@@ -1,1 +1,2 @@\n old\n+new\n",
            )
        ],
    )
    github_client = _FakeGitHubClient(pr)
    fake_codex = _FakeCodexClient(
        json.dumps(
            {
                "overall_correctness": "patch is incorrect",
                "overall_explanation": "Still broken.",
                "overall_confidence_score": 0.9,
                "carried_forward": [
                    {
                        "comment_id": "comment-1",
                        "current_evidence": "old",
                    }
                ],
                "findings": [
                    {
                        "title": "🔴 [P1] Existing finding",
                        "body": "Still broken.",
                        "confidence_score": 0.9,
                        "priority": 1,
                        "code_location": {
                            "absolute_file_path": str(sample_file.resolve()),
                            "line_range": {"start": 2, "end": 2},
                        },
                    }
                ],
            }
        )
    )
    workflow = ReviewWorkflow(
        _make_config(tmp_path),
        github_client=cast(Any, github_client),
        model_client=cast(Any, fake_codex),
    )

    result = workflow.process_review(7)

    assert "<prior_dotbot_review_comments>" not in fake_codex.calls[0]["schema_prompt"]
    assert result.posting_outcome.prefiltered_count == 0
    assert len(github_client.inline_comments) == 1
    assert result.review.carried_forward_comment_ids == []
    assert [finding.title for finding in result.review.findings] == ["🔴 [P1] Existing finding"]


def test_process_review_drops_invalid_carried_forward_comment_ids(tmp_path: Path) -> None:
    (tmp_path / "src.py").write_text("value = 1\n", encoding="utf-8")
    pr = _FakePR(
        issue_comments=[
            _FakeIssueComment(
                f"{SUMMARY_MARKER}\nold summary",
                comment_id=10,
                login="reviewer",
            )
        ],
        review_threads=[
            ReviewThreadSnapshot(
                id="thread-1",
                is_resolved=False,
                comments=[
                    ReviewThreadComment(
                        id="comment-1",
                        body=_structured_review_body("value = 1"),
                        path="src.py",
                        line=2,
                        original_line=2,
                        author="reviewer",
                    )
                ],
            )
        ],
    )
    workflow = ReviewWorkflow(
        _make_config(tmp_path),
        github_client=cast(Any, _FakeGitHubClient(pr)),
        model_client=cast(
            Any,
            _FakeCodexClient(
                json.dumps(
                    {
                        "overall_correctness": "patch is correct",
                        "overall_explanation": "",
                        "overall_confidence_score": 0.8,
                        "carried_forward": [
                            {
                                "comment_id": "comment-unknown",
                                "current_evidence": "value = 1",
                            },
                            {
                                "comment_id": "comment-1",
                                "current_evidence": "not the same snippet",
                            },
                            {
                                "comment_id": "comment-1",
                                "current_evidence": "value = 1",
                            },
                        ],
                        "findings": [],
                    }
                )
            ),
        ),
    )

    result = workflow.process_review(7)

    assert result.review.carried_forward_comment_ids == ["comment-1"]
    assert result.summary.carried_forward_count == 1


class _MultiResponseModelClient:
    """Model client that returns a canned response per roster slot."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def execute_structured(
        self,
        prompt: str,
        *,
        output_schema: dict[str, object],
        schema_prompt: str,
        sandbox_mode: str,
        model_name: str | None = None,
        resume_thread_id: str | None = None,
    ) -> str:
        self.calls.append({"model_name": model_name})
        return self.responses[len(self.calls) - 1]


def _correct_response() -> str:
    return json.dumps(
        {
            "overall_correctness": "patch is correct",
            "overall_explanation": "",
            "overall_confidence_score": 0.9,
            "carried_forward": [],
            "findings": [],
        }
    )


def _incorrect_response() -> str:
    return json.dumps(
        {
            "overall_correctness": "patch is incorrect",
            "overall_explanation": "Has issues.",
            "overall_confidence_score": 0.7,
            "carried_forward": [],
            "findings": [],
        }
    )


def _patch_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, Any]]:
    """Replace ``submit_pr_approval`` in the workflow namespace with a spy."""
    calls: list[dict[str, Any]] = []

    def _fake_submit_pr_approval(**kwargs: Any) -> Any:
        calls.append(kwargs)
        from cli.review.approval import ApprovalOutcome

        return ApprovalOutcome(submitted=True, login="dotCMS-Machine-User")

    monkeypatch.setattr(
        "cli.workflows.review_workflow.submit_pr_approval", _fake_submit_pr_approval
    )
    return calls


def test_process_review_approves_pr_when_all_models_correct(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "src.py").write_text("new\n", encoding="utf-8")
    pr = _FakePR(changed_files=[_FakeChangedFile("src.py", patch="@@ -1 +1 @@\n-old\n+new\n")])
    github_client = _FakeGitHubClient(pr)
    model_client = _MultiResponseModelClient([_correct_response(), _correct_response()])
    config = _make_config(tmp_path, review_models=("openai/gpt-5",))
    config.github_approval_token = "machine-user-pat"
    workflow = ReviewWorkflow(
        config,
        github_client=cast(Any, github_client),
        model_client=cast(Any, model_client),
    )
    approval_calls = _patch_approval(monkeypatch)

    workflow.process_review(7)

    assert len(approval_calls) == 1
    call = approval_calls[0]
    assert call["repository"] == "owner/repo"
    assert call["pr_number"] == 7
    assert call["head_sha"] == "head-sha"
    assert call["token"] == "machine-user-pat"
    assert DEFAULT_REVIEW_MODEL in call["body"]
    assert "openai/gpt-5" in call["body"]


def test_process_review_skips_approval_when_any_model_dissents(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "src.py").write_text("new\n", encoding="utf-8")
    pr = _FakePR(changed_files=[_FakeChangedFile("src.py", patch="@@ -1 +1 @@\n-old\n+new\n")])
    github_client = _FakeGitHubClient(pr)
    model_client = _MultiResponseModelClient([_correct_response(), _incorrect_response()])
    config = _make_config(tmp_path, review_models=("openai/gpt-5",))
    config.github_approval_token = "machine-user-pat"
    workflow = ReviewWorkflow(
        config,
        github_client=cast(Any, github_client),
        model_client=cast(Any, model_client),
    )
    approval_calls = _patch_approval(monkeypatch)

    workflow.process_review(7)

    assert approval_calls == []
    # Review summaries still posted normally.
    assert len(pr.as_issue().created_comments) == 2


def test_process_review_without_token_skips_approval_but_posts_comments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "src.py").write_text("new\n", encoding="utf-8")
    pr = _FakePR(changed_files=[_FakeChangedFile("src.py", patch="@@ -1 +1 @@\n-old\n+new\n")])
    github_client = _FakeGitHubClient(pr)
    model_client = _MultiResponseModelClient([_correct_response()])
    workflow = ReviewWorkflow(
        _make_config(tmp_path),
        github_client=cast(Any, github_client),
        model_client=cast(Any, model_client),
    )
    approval_calls = _patch_approval(monkeypatch)

    workflow.process_review(7)

    assert approval_calls == []
    assert len(pr.as_issue().created_comments) == 1


def test_process_review_dry_run_does_not_submit_approval(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    (tmp_path / "src.py").write_text("new\n", encoding="utf-8")
    pr = _FakePR(changed_files=[_FakeChangedFile("src.py", patch="@@ -1 +1 @@\n-old\n+new\n")])
    github_client = _FakeGitHubClient(pr)
    model_client = _MultiResponseModelClient([_correct_response()])
    config = _make_config(tmp_path, dry_run=True)
    config.github_approval_token = "machine-user-pat"
    workflow = ReviewWorkflow(
        config,
        github_client=cast(Any, github_client),
        model_client=cast(Any, model_client),
    )
    approval_calls = _patch_approval(monkeypatch)

    workflow.process_review(7)

    assert approval_calls == []
    assert "DRY_RUN: would approve PR" in capsys.readouterr().out


def test_process_review_approval_failure_is_nonfatal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "src.py").write_text("new\n", encoding="utf-8")
    pr = _FakePR(changed_files=[_FakeChangedFile("src.py", patch="@@ -1 +1 @@\n-old\n+new\n")])
    github_client = _FakeGitHubClient(pr)
    model_client = _MultiResponseModelClient([_correct_response()])

    def _boom(**kwargs: Any) -> Any:
        raise RuntimeError("approval exploded")

    monkeypatch.setattr("cli.workflows.review_workflow.submit_pr_approval", _boom)

    config = _make_config(tmp_path)
    config.github_approval_token = "machine-user-pat"
    workflow = ReviewWorkflow(
        config,
        github_client=cast(Any, github_client),
        model_client=cast(Any, model_client),
    )

    result = workflow.process_review(7)

    assert result.summary.overall_correctness == "patch is correct"
    assert len(pr.as_issue().created_comments) == 1


class _SlowModelClient:
    """Model client that takes `delay` seconds per structured call."""

    def __init__(self, response: str, delay: float) -> None:
        self.response = response
        self.delay = delay

    def execute_structured(
        self,
        prompt: str,
        *,
        output_schema: dict[str, object],
        schema_prompt: str,
        sandbox_mode: str,
        model_name: str | None = None,
        resume_thread_id: str | None = None,
    ) -> str:
        import time

        time.sleep(self.delay)
        return self.response


def test_process_review_model_timeout_aborts_slow_reviewer(tmp_path: Path) -> None:
    (tmp_path / "src.py").write_text("new\n", encoding="utf-8")
    pr = _FakePR(changed_files=[_FakeChangedFile("src.py", patch="@@ -1 +1 @@\n-old\n+new\n")])
    github_client = _FakeGitHubClient(pr)
    model_client = _SlowModelClient(_correct_response(), delay=2.0)
    config = _make_config(tmp_path)
    config.model_timeout_seconds = 1
    workflow = ReviewWorkflow(
        config,
        github_client=cast(Any, github_client),
        model_client=cast(Any, model_client),
    )

    with pytest.raises(DotBotExecutionError, match="exceeded the 1s wall-clock budget"):
        workflow.process_review(7)


def test_process_review_model_timeout_zero_disables_cap(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "src.py").write_text("new\n", encoding="utf-8")
    pr = _FakePR(changed_files=[_FakeChangedFile("src.py", patch="@@ -1 +1 @@\n-old\n+new\n")])
    github_client = _FakeGitHubClient(pr)
    model_client = _SlowModelClient(_correct_response(), delay=0.2)
    config = _make_config(tmp_path)
    config.model_timeout_seconds = 0
    workflow = ReviewWorkflow(
        config,
        github_client=cast(Any, github_client),
        model_client=cast(Any, model_client),
    )

    result = workflow.process_review(7)

    assert result.summary.overall_correctness == "patch is correct"
    assert f"Reviewer {DEFAULT_REVIEW_MODEL} finished in" in capsys.readouterr().out


def test_process_review_logs_reviewer_elapsed_time(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "src.py").write_text("new\n", encoding="utf-8")
    pr = _FakePR(changed_files=[_FakeChangedFile("src.py", patch="@@ -1 +1 @@\n-old\n+new\n")])
    github_client = _FakeGitHubClient(pr)
    model_client = _MultiResponseModelClient([_correct_response()])
    workflow = ReviewWorkflow(
        _make_config(tmp_path),
        github_client=cast(Any, github_client),
        model_client=cast(Any, model_client),
    )

    workflow.process_review(7)

    out = capsys.readouterr().out
    assert f"Reviewer {DEFAULT_REVIEW_MODEL} finished in 0." in out


def _prior_thread_review_response(carried_forward: list[dict[str, Any]]) -> str:
    return json.dumps(
        {
            "overall_correctness": "patch is correct",
            "overall_explanation": "No additional non-redundant findings.",
            "overall_confidence_score": 0.8,
            "carried_forward": carried_forward,
            "findings": [],
        }
    )


def _prior_thread_pr() -> _FakePR:
    return _FakePR(
        issue_comments=[
            _FakeIssueComment(
                f"{SUMMARY_MARKER}\nold summary",
                comment_id=10,
                login="reviewer",
            )
        ],
        review_threads=[
            ReviewThreadSnapshot(
                id="thread-1",
                is_resolved=False,
                comments=[
                    ReviewThreadComment(
                        id="comment-1",
                        body=_structured_review_body("value = 1"),
                        path="src.py",
                        line=2,
                        original_line=2,
                        author="reviewer",
                    )
                ],
            )
        ],
    )


def test_process_review_resolves_stale_prior_threads_when_enabled(tmp_path: Path) -> None:
    (tmp_path / "src.py").write_text("value = 1\n", encoding="utf-8")
    pr = _prior_thread_pr()
    github_client = _FakeGitHubClient(pr)
    workflow = ReviewWorkflow(
        _make_config(tmp_path, resolve_stale_threads=True),
        github_client=cast(Any, github_client),
        model_client=cast(Any, _FakeCodexClient(_prior_thread_review_response([]))),
    )

    workflow.process_review(7)

    assert github_client.resolved_thread_ids == ["thread-1"]


def test_process_review_keeps_carried_forward_thread_open(tmp_path: Path) -> None:
    (tmp_path / "src.py").write_text("value = 1\n", encoding="utf-8")
    pr = _prior_thread_pr()
    github_client = _FakeGitHubClient(pr)
    workflow = ReviewWorkflow(
        _make_config(tmp_path, resolve_stale_threads=True),
        github_client=cast(Any, github_client),
        model_client=cast(
            Any,
            _FakeCodexClient(
                _prior_thread_review_response(
                    [{"comment_id": "comment-1", "current_evidence": "value = 1"}]
                )
            ),
        ),
    )

    workflow.process_review(7)

    assert github_client.resolved_thread_ids == []


def test_process_review_skips_stale_thread_resolution_when_disabled(tmp_path: Path) -> None:
    (tmp_path / "src.py").write_text("value = 1\n", encoding="utf-8")
    pr = _prior_thread_pr()
    github_client = _FakeGitHubClient(pr)
    workflow = ReviewWorkflow(
        _make_config(tmp_path),
        github_client=cast(Any, github_client),
        model_client=cast(Any, _FakeCodexClient(_prior_thread_review_response([]))),
    )

    workflow.process_review(7)

    assert github_client.resolved_thread_ids == []


def test_process_review_dry_run_skips_stale_thread_resolution(tmp_path: Path) -> None:
    (tmp_path / "src.py").write_text("value = 1\n", encoding="utf-8")
    pr = _prior_thread_pr()
    github_client = _FakeGitHubClient(pr)
    workflow = ReviewWorkflow(
        _make_config(tmp_path, dry_run=True, resolve_stale_threads=True),
        github_client=cast(Any, github_client),
        model_client=cast(Any, _FakeCodexClient(_prior_thread_review_response([]))),
    )

    workflow.process_review(7)

    assert github_client.resolved_thread_ids == []


def test_process_review_leaves_unseen_stale_thread_open(tmp_path: Path) -> None:
    # The thread's current_code ("value = 1") no longer matches the file, so
    # the reviewer never saw it — it must stay open for a human.
    (tmp_path / "src.py").write_text("value = 2\n", encoding="utf-8")
    pr = _prior_thread_pr()
    github_client = _FakeGitHubClient(pr)
    workflow = ReviewWorkflow(
        _make_config(tmp_path, resolve_stale_threads=True),
        github_client=cast(Any, github_client),
        model_client=cast(Any, _FakeCodexClient(_prior_thread_review_response([]))),
    )

    workflow.process_review(7)

    assert github_client.resolved_thread_ids == []
