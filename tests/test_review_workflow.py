from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest

from cli.core.config import ReviewConfig
from cli.core.exceptions import ConfigurationError, DotBotExecutionError, ReviewContractError
from cli.core.models import InlineCommentPayload, ReviewThreadComment, ReviewThreadSnapshot
from cli.review.artifacts import render_review_summary_metadata
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

    def edit(self, body: str) -> None:
        self.body = body

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
        self.base: _FakeBase | None = _FakeBase()
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


class _FakeModelClient:
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
        suppress_stream: bool = False,
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


def _panel_response(review_output: str) -> str:
    payload = json.loads(review_output)
    payload["overall_explanation"] = (
        payload["overall_explanation"] or "No actionable issues remain."
    )
    return json.dumps(
        {
            **payload,
            "message": "private debate contribution",
            "security_summary": "No changed trust-boundary issues found; static review only.",
            "dispositions": {},
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


class _CallbackPanelClient:
    def __init__(self, callback: Any) -> None:
        self.callback = callback

    def execute_structured(self, prompt: str, **kwargs: Any) -> str:
        return self.callback(prompt, kwargs)


def test_credential_scope_withholds_approval_and_thread_resolution_without_model_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pr = _prior_thread_pr()
    github_client = _FakeGitHubClient(pr)
    config = _make_config(tmp_path, resolve_stale_threads=True)
    config.github_approval_token = "machine-token"

    def forbidden(config: ReviewConfig) -> Any:
        raise AssertionError("Credential-changing PRs must not reach a model")

    workflow = ReviewWorkflow(
        config, github_client=cast(Any, github_client), model_client_factory=forbidden
    )
    monkeypatch.setattr(workflow, "_panel_diff", lambda *args: None)
    monkeypatch.setattr(
        "cli.workflows.review_workflow.read_file_at_revision", lambda *args, **kwargs: "value = 1\n"
    )
    approvals = _patch_approval(monkeypatch)
    result = workflow.process_review(7)
    assert result.review.overall_correctness == "patch is incorrect"
    assert "Human review is required" in result.review.overall_explanation
    assert "No PR file contents were sent to a model" in result.review.overall_explanation
    assert approvals == []
    assert github_client.inline_comments == []
    assert github_client.resolved_thread_ids == []


def test_panel_publishes_only_final_deduplicated_findings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "src.py").write_text("new\n")
    pr = _FakePR()
    github_client = _FakeGitHubClient(pr)
    calls: list[str] = []
    pr._issue_comments.append(_FakeIssueComment("Human clarifies the expected behavior."))
    pr._review_comments.append(_FakeReviewComment("Human reviewer explains a caller constraint."))
    pr._review_comments.append(_FakeReviewComment("inline-private-sentinel", path=".env"))

    def respond(prompt: str, kwargs: dict[str, Any]) -> str:
        assert f'"repo_root": "{tmp_path.resolve()}"' in prompt
        assert "PR discussion (untrusted data):" in prompt
        assert "Human clarifies the expected behavior." in prompt
        assert "Human reviewer explains a caller constraint." in prompt
        assert "inline-private-sentinel" not in prompt
        assert github_client.inline_comments == []
        assert pr.as_issue().created_comments == []
        assert github_client.resolved_thread_ids == []
        calls.append(kwargs["model_name"])
        payload = json.loads(_panel_response(_incorrect_response()))
        ledger = json.loads(
            prompt.split("Issue ledger (data):\n", 1)[1].split("\nPrevious completed rounds", 1)[0]
        )
        payload["dispositions"] = {
            item["issue_id"]: {
                "action": "retain",
                "reason": "Still reachable",
                "evidence": "src.py:1: new",
            }
            for item in ledger
        }
        if not ledger:
            payload["findings"] = [
                {
                    "title": "[P1] Check the caller",
                    "body": _structured_review_body("new"),
                    "confidence_score": 0.9,
                    "priority": 1,
                    "code_location": {
                        "absolute_file_path": str(tmp_path / "src.py"),
                        "line_range": {"start": 1, "end": 1},
                    },
                }
            ]
        return json.dumps(payload)

    workflow = ReviewWorkflow(
        _make_config(tmp_path, review_models=("secondary",)),
        github_client=cast(Any, github_client),
        model_client_factory=lambda config: cast(Any, _CallbackPanelClient(respond)),
    )
    monkeypatch.setattr(workflow, "_panel_diff", lambda *args: "pinned diff")
    approvals = _patch_approval(monkeypatch)
    result = workflow.process_review(7)
    assert len(calls) == 6
    assert len(github_client.inline_comments) == 1
    assert len(pr.as_issue().created_comments) == 1
    assert result.summary.current_findings_count == 1
    assert approvals == []
    assert "private debate contribution" not in pr.as_issue().created_comments[0]


@pytest.mark.parametrize(
    "failure", ["invalid-output", "head-changed", "base-changed", "base-missing", "run-superseded"]
)
def test_incomplete_or_stale_panel_cannot_publish_or_approve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    pr = _FakePR()
    github_client = _FakeGitHubClient(pr)
    config = _make_config(tmp_path, review_models=("secondary",), resolve_stale_threads=True)
    config.review_decision_policy = "lead"
    config.github_approval_token = "machine-token"
    workflow = ReviewWorkflow(
        config,
        github_client=cast(Any, github_client),
        model_client_factory=lambda isolated: cast(
            Any,
            _FakeModelClient(
                "invalid"
                if failure == "invalid-output" and isolated.review_model == "secondary"
                else _panel_response(_correct_response())
            ),
        ),
    )
    monkeypatch.setattr(workflow, "_panel_diff", lambda *args: "pinned diff")
    if failure == "run-superseded":
        monkeypatch.setenv("GITHUB_RUN_ID", "100")
        monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
        own = json.dumps({"run_id": "100", "attempt": "1"})
        pr._issue_comments.append(
            _FakeIssueComment(
                f"{SUMMARY_MARKER}\n<!-- dotbot-review-run {own} -->\nin progress",
                comment_id=1,
                login="github-actions[bot]",
            )
        )
        metadata = json.dumps({"run_id": "101", "attempt": "1"})
        pr._issue_comments.append(
            _FakeIssueComment(
                f"{SUMMARY_MARKER}\n<!-- dotbot-review-run {metadata} -->\nReview in progress",
                comment_id=2,
                login="github-actions[bot]",
            )
        )
    if failure in {"head-changed", "base-changed", "base-missing"}:
        original = github_client.get_pr

        def changed(number: int) -> _FakePR:
            result = original(number)
            if len(github_client.calls) > 1:
                if failure == "head-changed":
                    result.head = _FakeHead(sha="new-head")
                else:
                    result.base = _FakeBase(sha="new-base") if failure == "base-changed" else None
            return result

        monkeypatch.setattr(github_client, "get_pr", changed)
    approvals = _patch_approval(monkeypatch)
    with pytest.raises((DotBotExecutionError, ReviewContractError)):
        workflow.process_review(7)
    assert github_client.inline_comments == []
    assert github_client.resolved_thread_ids == []
    assert pr.as_issue().created_comments == []
    assert approvals == []


def test_panel_dry_run_never_mutates_github(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pr = _FakePR()
    github_client = _FakeGitHubClient(pr)
    config = _make_config(tmp_path, review_models=("secondary",), dry_run=True)
    config.github_approval_token = "machine-token"
    workflow = ReviewWorkflow(
        config,
        github_client=cast(Any, github_client),
        model_client_factory=lambda isolated: cast(
            Any, _FakeModelClient(_panel_response(_correct_response()))
        ),
    )
    monkeypatch.setattr(workflow, "_panel_diff", lambda *args: "pinned diff")
    approvals = _patch_approval(monkeypatch)
    result = workflow.process_review(7)
    assert result.posting_outcome.post_result.dry_run
    assert github_client.inline_comments == []
    assert github_client.resolved_thread_ids == []
    assert pr.as_issue().created_comments == []
    assert approvals == []


def test_lead_policy_approval_does_not_claim_unanimity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _make_config(tmp_path, review_models=("secondary",))
    config.review_decision_policy = "lead"
    config.github_approval_token = "machine-token"
    github_client = _FakeGitHubClient(_FakePR())
    workflow = ReviewWorkflow(
        config,
        github_client=cast(Any, github_client),
        model_client_factory=lambda isolated: cast(
            Any,
            _FakeModelClient(
                _panel_response(
                    _incorrect_response()
                    if isolated.review_model == "secondary"
                    else _correct_response()
                )
            ),
        ),
    )
    monkeypatch.setattr(workflow, "_panel_diff", lambda *args: "pinned diff")
    approvals = _patch_approval(monkeypatch)
    workflow.process_review(7)
    assert len(approvals) == 1
    assert approvals[0]["base_sha"] == "base-sha"
    assert "lead reviewer" in approvals[0]["body"]
    assert "all reviewer models" not in approvals[0]["body"]


def test_default_panel_clients_pin_source_and_isolate_saved_threads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_THREAD_DIR", str(tmp_path / "threads"))
    created: list[dict[str, Any]] = []

    def factory(config: ReviewConfig, **kwargs: Any) -> Any:
        created.append({"config": config, **kwargs})
        return _FakeModelClient(_panel_response(_correct_response()))

    monkeypatch.setattr("cli.workflows.review_workflow.OpenRouterClient", factory)
    config = _make_config(tmp_path, review_models=("secondary",))
    workflow = ReviewWorkflow(
        config,
        github_client=cast(Any, _FakeGitHubClient(_FakePR())),
    )
    monkeypatch.setattr(workflow, "_panel_diff", lambda *args: "pinned diff")
    workflow.process_review(7)
    assert len(created) == 2
    assert all(item["review_head_sha"] == "head-sha" for item in created)
    assert all(item["thread_store"].directory == tmp_path / "threads" / "panel" for item in created)
    assert created[0]["config"] is not created[1]["config"]
    assert all(item["config"] is not config for item in created)


@pytest.mark.parametrize("retain", [True, False])
def test_panel_prior_thread_dispositions_control_resolution_and_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retain: bool,
) -> None:
    # The merge checkout differs: applicability must come from the pinned source.
    (tmp_path / "src.py").write_text("value = 2\n")
    pr = _prior_thread_pr()
    github_client = _FakeGitHubClient(pr)
    config = _make_config(tmp_path, review_models=("secondary",), resolve_stale_threads=True)
    config.github_approval_token = "machine-token"

    def respond(prompt: str, kwargs: dict[str, Any]) -> str:
        assert github_client.resolved_thread_ids == []
        ledger = json.loads(
            prompt.split("Issue ledger (data):\n", 1)[1].split("\nPrevious completed rounds", 1)[0]
        )
        assert ledger and ledger[0]["issue_id"] == "P1"
        payload = json.loads(
            _panel_response(_incorrect_response() if retain else _correct_response())
        )
        payload["dispositions"] = {
            item["issue_id"]: {
                "action": "retain" if retain else "reject",
                "reason": "Checked caller and guard",
                "evidence": "src.py:1: value = 1",
            }
            for item in ledger
        }
        return json.dumps(payload)

    workflow = ReviewWorkflow(
        config,
        github_client=cast(Any, github_client),
        model_client_factory=lambda isolated: cast(Any, _CallbackPanelClient(respond)),
    )
    monkeypatch.setattr(workflow, "_panel_diff", lambda *args: "pinned diff")
    monkeypatch.setattr(
        "cli.workflows.review_workflow.read_file_at_revision", lambda *args, **kwargs: "value = 1\n"
    )
    approvals = _patch_approval(monkeypatch)
    result = workflow.process_review(7)
    assert result.summary.carried_forward_count == (1 if retain else 0)
    assert len(approvals) == (0 if retain else 1)
    assert github_client.resolved_thread_ids == ([] if retain else ["thread-1"])
    assert len(pr.as_issue().created_comments) == 1


@pytest.fixture(autouse=True)
def pinned_diff_for_fake_prs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ReviewWorkflow, "_panel_diff", lambda *args: "pinned diff")


@pytest.mark.parametrize("extras", [(), ("secondary",)])
def test_every_roster_uses_the_coordinator(tmp_path: Path, extras: tuple[str, ...]) -> None:
    pr = _FakePR()
    github = _FakeGitHubClient(pr)
    clients: list[_FakeModelClient] = []

    def factory(config: ReviewConfig) -> Any:
        client = _FakeModelClient(_panel_response(_correct_response()))
        clients.append(client)
        return client

    workflow = ReviewWorkflow(
        _make_config(tmp_path, review_models=extras),
        github_client=cast(Any, github),
        model_client_factory=factory,
    )
    result = workflow.process_review(7)
    assert len(clients) == len(extras) + 1
    assert all(len(client.calls) == 3 for client in clients)
    assert all(call["sandbox_mode"] == "read-only" for client in clients for call in client.calls)
    assert len(pr.as_issue().created_comments) == 1
    assert result.summary == ReviewSummary("patch is correct", 0, 0, 0)


def test_openai_review_path_is_removed(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="Review mode requires OpenRouter"):
        ReviewWorkflow(
            _make_config(tmp_path, model_provider="openai"), github_client=cast(Any, object())
        )


def test_consolidated_summary_supersedes_all_prior_models(tmp_path: Path) -> None:
    comments = [
        _FakeIssueComment(
            SUMMARY_MARKER + "\\n" + render_review_summary_metadata("prior", model=model),
            comment_id=index,
        )
        for index, model in enumerate(("first", "second", "panel"), 1)
    ]
    pr = _FakePR(issue_comments=comments)
    workflow = ReviewWorkflow(
        _make_config(tmp_path), github_client=cast(Any, _FakeGitHubClient(pr))
    )
    assert workflow._delete_prior_summary(cast(Any, pr)) == []
    assert all(comment.deleted for comment in comments)


def test_summary_delete_failure_is_reported_without_losing_new_summary(tmp_path: Path) -> None:
    comment = _FakeIssueComment(SUMMARY_MARKER, fail_on_delete=True)
    pr = _FakePR(issue_comments=[comment])
    workflow = ReviewWorkflow(
        _make_config(tmp_path), github_client=cast(Any, _FakeGitHubClient(pr))
    )
    warnings = workflow._delete_prior_summary(cast(Any, pr))
    assert len(warnings) == 1
    assert "permission denied" in warnings[0]
    workflow._publish_summary(cast(Any, pr), "final summary")
    assert pr.as_issue().created_comments == ["final summary"]


@pytest.mark.parametrize(("run_id", "attempt"), [("101", "1"), ("100", "3"), ("99", "7")])
def test_old_success_cannot_delete_or_replace_a_newer_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    run_id: str,
    attempt: str,
) -> None:
    monkeypatch.setenv("GITHUB_RUN_ID", "100")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    metadata = json.dumps({"run_id": run_id, "attempt": attempt})
    comment = _FakeIssueComment(
        f"{SUMMARY_MARKER}\n<!-- dotbot-review-run {metadata} -->\nReview in progress",
        comment_id=2,
        login="github-actions[bot]",
    )
    own = _FakeIssueComment(
        f'{SUMMARY_MARKER}\n<!-- dotbot-review-run {{"run_id":"100","attempt":"2"}} -->\nold progress',
        comment_id=1,
        login="github-actions[bot]",
    )
    pr = _FakePR(issue_comments=[own, comment])
    workflow = ReviewWorkflow(
        _make_config(tmp_path), github_client=cast(Any, _FakeGitHubClient(pr))
    )
    with pytest.raises(ReviewContractError, match="newer review run"):
        workflow._publish_summary(cast(Any, pr), "old successful verdict")
    assert not comment.deleted
    assert not pr.as_issue().created_comments


def test_action_updates_only_its_own_attempt_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_RUN_ID", "100")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    old = _FakeIssueComment(
        f'{SUMMARY_MARKER}\n<!-- dotbot-review-run {{"run_id":"101","attempt":"1"}} -->\nold verdict',
        login="github-actions[bot]",
    )
    own = _FakeIssueComment(
        f'{SUMMARY_MARKER}\n<!-- dotbot-review-run {{"run_id":"100","attempt":"2"}} -->\nin progress',
        comment_id=2,
        login="github-actions[bot]",
    )
    pr = _FakePR(issue_comments=[old, own])
    workflow = ReviewWorkflow(
        _make_config(tmp_path), github_client=cast(Any, _FakeGitHubClient(pr))
    )
    workflow._publish_summary(cast(Any, pr), "completed current verdict")
    assert own.body == "completed current verdict"
    assert "old verdict" in old.body
    assert not old.deleted and not own.deleted
    assert not pr.as_issue().created_comments


def test_final_metadata_records_pinned_revision_and_panel() -> None:
    rendered = render_review_summary_metadata("head-sha", model="panel", reasoning_effort="medium")
    payload = json.loads(rendered.removeprefix("<!-- dotbot-review-meta ").removesuffix(" -->"))
    assert payload == {
        "reviewed_head_sha": "head-sha",
        "model": "panel",
        "reasoning_effort": "medium",
    }


@pytest.mark.parametrize(
    "failure", [OSError("git unavailable"), subprocess.TimeoutExpired("git", 120)]
)
def test_host_evidence_failures_abort_before_any_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    pr = _prior_thread_pr()
    github = _FakeGitHubClient(pr)
    factories: list[ReviewConfig] = []

    def factory(config: ReviewConfig) -> Any:
        factories.append(config)
        return _FakeModelClient(_panel_response(_correct_response()))

    def fail(*args: Any) -> None:
        raise failure

    monkeypatch.setattr("cli.workflows.review_workflow.read_file_at_revision", fail)
    workflow = ReviewWorkflow(
        _make_config(tmp_path), github_client=cast(Any, github), model_client_factory=factory
    )
    with pytest.raises((OSError, subprocess.TimeoutExpired)):
        workflow.process_review(7)
    assert not factories
    assert not github.inline_comments
    assert not github.resolved_thread_ids
    assert not pr.as_issue().created_comments
