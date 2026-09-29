from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from cli.core.exceptions import GitHubAPIError
from cli.review.approval import (
    APPROVAL_BODY_TEMPLATE,
    APPROVAL_MARKER,
    ApprovalOutcome,
    build_approval_body,
    submit_pr_approval,
)


def _debug_spy() -> tuple[list[tuple[int, str]], Any]:
    messages: list[tuple[int, str]] = []

    def _debug(min_level: int, message: str) -> None:
        messages.append((min_level, message))

    return messages, _debug


class _FakeRequester:
    def __init__(self) -> None:
        self.requests: list[tuple[str, str, dict[str, Any] | None]] = []

    def requestJsonAndCheck(
        self,
        verb: str,
        url: str,
        input: dict[str, Any] | None = None,
    ) -> object:
        self.requests.append((verb, url, input))
        return {}, {}


class _FakeReview:
    def __init__(
        self,
        *,
        login: str | None,
        state: str,
        commit_id: str,
        body: str | None = None,
    ) -> None:
        self.user = SimpleNamespace(login=login) if login is not None else None
        self.state = state
        self.commit_id = commit_id
        self.body = body


class _FakeApprovalPR:
    url = "https://api.example.test/repos/owner/repo/pulls/7"

    def __init__(self, reviews: list[_FakeReview] | None = None) -> None:
        self._reviews = reviews or []
        self._requester = _FakeRequester()

    def get_reviews(self) -> list[_FakeReview]:
        return list(self._reviews)


class _FakeApprovalRepo:
    def __init__(self, pr: _FakeApprovalPR) -> None:
        self._pr = pr

    def get_pull(self, pr_number: int) -> _FakeApprovalPR:
        assert pr_number == 7
        return self._pr


class _FakeApprovalUser:
    def __init__(self, login: str) -> None:
        self.login = login


class _FakeGithub:
    def __init__(
        self,
        *,
        user: _FakeApprovalUser | None = None,
        repo: _FakeApprovalRepo | None = None,
    ) -> None:
        self._user = user or _FakeApprovalUser("dotCMS-Machine-User")
        self._repo = repo

    def get_user(self) -> _FakeApprovalUser:
        return self._user

    def get_repo(self, repository: str) -> _FakeApprovalRepo:
        assert repository == "owner/repo"
        return self._repo or _FakeApprovalRepo(_FakeApprovalPR())


def _patch_github(monkeypatch: pytest.MonkeyPatch, gh: _FakeGithub) -> list[dict[str, Any]]:
    constructions: list[dict[str, Any]] = []

    def _factory(**kwargs: Any) -> _FakeGithub:
        constructions.append(kwargs)
        return gh

    monkeypatch.setattr("cli.review.approval.Github", _factory)
    return constructions


def test_submit_pr_approval_posts_approve_review(monkeypatch: pytest.MonkeyPatch) -> None:
    pr = _FakeApprovalPR(reviews=[])
    gh = _FakeGithub(repo=_FakeApprovalRepo(pr))
    constructions = _patch_github(monkeypatch, gh)
    debug_messages, debug = _debug_spy()

    outcome = submit_pr_approval(
        repository="owner/repo",
        pr_number=7,
        head_sha="abc123",
        token="machine-user-pat",
        body="LGTM",
        debug=debug,
    )

    assert outcome == ApprovalOutcome(submitted=True, login="dotCMS-Machine-User")
    assert constructions == [{"login_or_token": "machine-user-pat", "per_page": 100}]
    assert pr._requester.requests == [
        (
            "POST",
            "https://api.example.test/repos/owner/repo/pulls/7/reviews",
            {"commit_id": "abc123", "body": "LGTM", "event": "APPROVE"},
        )
    ]
    assert debug_messages


def test_submit_pr_approval_skips_when_already_approved(monkeypatch: pytest.MonkeyPatch) -> None:
    pr = _FakeApprovalPR(
        reviews=[
            _FakeReview(login="someone-else", state="APPROVED", commit_id="abc123"),
            _FakeReview(login="dotCMS-Machine-User", state="APPROVED", commit_id="older"),
            _FakeReview(login="dotCMS-Machine-User", state="APPROVED", commit_id="abc123"),
        ]
    )
    gh = _FakeGithub(repo=_FakeApprovalRepo(pr))
    constructions = _patch_github(monkeypatch, gh)
    _, debug = _debug_spy()

    outcome = submit_pr_approval(
        repository="owner/repo",
        pr_number=7,
        head_sha="abc123",
        token="machine-user-pat",
        body="LGTM",
        debug=debug,
    )

    assert outcome == ApprovalOutcome(
        submitted=False,
        login="dotCMS-Machine-User",
        reason="already_approved",
    )
    # The machine-user client was still constructed with the PAT.
    assert constructions == [{"login_or_token": "machine-user-pat", "per_page": 100}]
    # No POST request was issued.
    assert pr._requester.requests == []


def test_submit_pr_approval_submits_when_prior_approval_is_for_older_sha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pr = _FakeApprovalPR(
        reviews=[
            _FakeReview(login="dotCMS-Machine-User", state="APPROVED", commit_id="older-sha"),
            _FakeReview(login="dotCMS-Machine-User", state="CHANGES_REQUESTED", commit_id="abc123"),
        ]
    )
    gh = _FakeGithub(repo=_FakeApprovalRepo(pr))
    constructions = _patch_github(monkeypatch, gh)
    _, debug = _debug_spy()

    outcome = submit_pr_approval(
        repository="owner/repo",
        pr_number=7,
        head_sha="abc123",
        token="machine-user-pat",
        body="LGTM",
        debug=debug,
    )

    assert outcome.submitted is True
    assert constructions == [{"login_or_token": "machine-user-pat", "per_page": 100}]
    assert len(pr._requester.requests) == 1


def test_submit_pr_approval_wraps_pr_load_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class _NoPRGithub(_FakeGithub):
        def get_repo(self, repository: str) -> _FakeApprovalRepo:
            raise RuntimeError("boom")

    _patch_github(monkeypatch, _NoPRGithub())
    _, debug = _debug_spy()

    with pytest.raises(GitHubAPIError, match="failed to load owner/repo#7 for approval: boom"):
        submit_pr_approval(
            repository="owner/repo",
            pr_number=7,
            head_sha="abc123",
            token="machine-user-pat",
            body="LGTM",
            debug=debug,
        )


def test_submit_pr_approval_tolerates_token_that_cannot_read_its_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Installation tokens (${{ github.token }}) get 403 from GET /user.

    The approval must still be submitted: the identity is unknowable by design
    for the workflow token, not a configuration error.
    """

    class _InstallationTokenGithub(_FakeGithub):
        def get_user(self) -> _FakeApprovalUser:
            raise RuntimeError("Resource not accessible by integration: 403")

    pr = _FakeApprovalPR(reviews=[])
    gh = _InstallationTokenGithub(repo=_FakeApprovalRepo(pr))
    _patch_github(monkeypatch, gh)
    debug_messages, debug = _debug_spy()

    outcome = submit_pr_approval(
        repository="owner/repo",
        pr_number=7,
        head_sha="abc123",
        token="ghs_installation_token",
        body="LGTM",
        debug=debug,
    )

    assert outcome == ApprovalOutcome(submitted=True, login="")
    assert len(pr._requester.requests) == 1
    assert any("installation token" in message for _, message in debug_messages)


def test_submit_pr_approval_dedupes_by_body_marker_when_login_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a login, our own marker in an approval body keeps re-runs idempotent."""

    class _InstallationTokenGithub(_FakeGithub):
        def get_user(self) -> _FakeApprovalUser:
            raise RuntimeError("Resource not accessible by integration: 403")

    pr = _FakeApprovalPR(
        reviews=[
            _FakeReview(
                login="github-actions[bot]",
                state="APPROVED",
                commit_id="abc123",
                body=APPROVAL_BODY_TEMPLATE.format(models="openai/gpt-5"),
            )
        ]
    )
    gh = _InstallationTokenGithub(repo=_FakeApprovalRepo(pr))
    _patch_github(monkeypatch, gh)
    _, debug = _debug_spy()

    outcome = submit_pr_approval(
        repository="owner/repo",
        pr_number=7,
        head_sha="abc123",
        token="ghs_installation_token",
        body="LGTM",
        debug=debug,
    )

    assert outcome == ApprovalOutcome(submitted=False, login="", reason="already_approved")
    assert pr._requester.requests == []


def test_submit_pr_approval_ignores_unmarked_approval_when_login_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Another actor's approval must not suppress ours when we can't match logins."""

    class _InstallationTokenGithub(_FakeGithub):
        def get_user(self) -> _FakeApprovalUser:
            raise RuntimeError("Resource not accessible by integration: 403")

    pr = _FakeApprovalPR(
        reviews=[
            _FakeReview(
                login="some-human",
                state="APPROVED",
                commit_id="abc123",
                body="looks good to me",
            )
        ]
    )
    gh = _InstallationTokenGithub(repo=_FakeApprovalRepo(pr))
    _patch_github(monkeypatch, gh)
    _, debug = _debug_spy()

    outcome = submit_pr_approval(
        repository="owner/repo",
        pr_number=7,
        head_sha="abc123",
        token="ghs_installation_token",
        body="LGTM",
        debug=debug,
    )

    assert outcome.submitted is True
    assert len(pr._requester.requests) == 1


def test_build_approval_body_lists_models() -> None:
    body = build_approval_body(["openai/gpt-5", "google/gemini-2.5-pro"])
    assert "patch is correct" in body
    assert "openai/gpt-5" in body
    assert "google/gemini-2.5-pro" in body
    # The body marker is the idempotency key for tokens whose login is unknown.
    assert APPROVAL_MARKER in body
