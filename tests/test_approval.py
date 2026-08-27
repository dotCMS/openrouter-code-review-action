from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from cli.core.exceptions import GitHubAPIError
from cli.review.approval import ApprovalOutcome, build_approval_body, submit_pr_approval


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
    def __init__(self, *, login: str, state: str, commit_id: str) -> None:
        self.user = SimpleNamespace(login=login)
        self.state = state
        self.commit_id = commit_id


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


def test_submit_pr_approval_wraps_api_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BrokenGithub(_FakeGithub):
        def get_user(self) -> _FakeApprovalUser:
            raise RuntimeError("boom")

    _patch_github(monkeypatch, _BrokenGithub())
    _, debug = _debug_spy()

    with pytest.raises(GitHubAPIError, match="boom"):
        submit_pr_approval(
            repository="owner/repo",
            pr_number=7,
            head_sha="abc123",
            token="machine-user-pat",
            body="LGTM",
            debug=debug,
        )


def test_build_approval_body_lists_models() -> None:
    body = build_approval_body(["openai/gpt-5", "google/gemini-2.5-pro"])
    assert "patch is correct" in body
    assert "openai/gpt-5" in body
    assert "google/gemini-2.5-pro" in body
