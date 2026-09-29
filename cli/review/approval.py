"""Submit an approving PR review as the machine user (e.g. dotCMS-Machine-User).

The reviewer workflow runs a roster of LLM reviewers over a PR. When every
reviewer in the roster concludes ``Overall: patch is correct``, the workflow
approves the PR on their behalf. The token that submits the approval decides the
identity it appears under: a machine-user PAT shows a named user (e.g.
dotCMS-Machine-User), while ``${{ github.token }}`` shows ``github-actions[bot]``.

Installation tokens cannot call ``GET /user`` (GitHub answers 403 "Resource not
accessible by integration"), so the approver identity is resolved best-effort:
when it is unknown, idempotency is keyed off the marker in the review body
instead of the login.

The approval is idempotent per head SHA: if this action has already approved the
current head commit, we skip re-submitting so re-triggered runs do not spam
duplicate approvals.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from github import Github
from github.PullRequestReview import PullRequestReview

from ..core.exceptions import GitHubAPIError

APPROVAL_MARKER = "approved automatically by dotbot"

APPROVAL_BODY_TEMPLATE = (
    "✅ dotbot review: all reviewer models ({models}) agree — **patch is correct**.\n\n"
    f"<sub>{APPROVAL_MARKER}</sub>"
)


@dataclass(frozen=True)
class ApprovalOutcome:
    """Result of an approval attempt."""

    submitted: bool
    login: str = ""
    reason: str = ""


def build_approval_body(models: list[str]) -> str:
    """Render the approval review body for the given reviewer roster."""
    return APPROVAL_BODY_TEMPLATE.format(models=", ".join(models))


def _resolve_login(gh: Github, debug: Callable[[int, str], None]) -> str:
    """Return the approval token's user login, or ``""`` when it cannot be read.

    Installation tokens (``${{ github.token }}``) get a 403 from ``GET /user``.
    The login is only used for logging and for the login-based idempotency
    shortcut, so this is best-effort: an unresolvable token still submits the
    approval, deduplicated by :data:`APPROVAL_MARKER` in the review body.
    """
    try:
        return gh.get_user().login
    except Exception:  # noqa: BLE001 — an installation token cannot read /user
        debug(
            1,
            "Approval token cannot resolve its own user (installation token); "
            "falling back to the review body marker for idempotency",
        )
        return ""


def _is_our_approval(review: PullRequestReview, *, login: str, head_sha: str) -> bool:
    """True when ``review`` is an existing approval of ``head_sha`` that we made.

    Matches the resolved login, or — when the token cannot report a login (the
    installation-token case) — our own :data:`APPROVAL_MARKER` in the body.
    """
    if review.state != "APPROVED" or (review.commit_id or "") != head_sha:
        return False
    author = review.user.login if review.user is not None else None
    return bool(login and author == login) or APPROVAL_MARKER in (review.body or "")


def _approval_exists(reviews: Iterable[PullRequestReview], *, login: str, head_sha: str) -> bool:
    """True when an approval for ``head_sha`` came from us already."""
    return any(_is_our_approval(review, login=login, head_sha=head_sha) for review in reviews)


def submit_pr_approval(
    *,
    repository: str,
    pr_number: int,
    head_sha: str,
    token: str,
    body: str,
    debug: Callable[[int, str], None],
) -> ApprovalOutcome:
    """Submit an ``APPROVE`` review on the PR as the token's user.

    Uses a dedicated ``Github`` client authenticated with ``token`` so the review
    is attributed to that token rather than the action's default credentials —
    a machine-user PAT, or ``github.token`` for a ``github-actions[bot]``
    approval. Skips submission when this token has already approved this exact
    head commit.

    Raises:
        GitHubAPIError: if any GitHub API call fails.
    """
    gh = Github(login_or_token=token, per_page=100)

    login = _resolve_login(gh, debug)

    try:
        pr = gh.get_repo(repository).get_pull(pr_number)
    except Exception as exc:
        raise GitHubAPIError(
            f"failed to load {repository}#{pr_number} for approval: {exc}"
        ) from exc

    try:
        if _approval_exists(pr.get_reviews(), login=login, head_sha=head_sha):
            debug(1, f"Approval already submitted for {head_sha}; skipping")
            return ApprovalOutcome(submitted=False, login=login, reason="already_approved")
    except Exception as exc:
        raise GitHubAPIError(f"failed to list reviews on {repository}#{pr_number}: {exc}") from exc

    try:
        # POST /pulls/{n}/reviews directly (same shape the batched inline
        # review submission uses) instead of ``create_review`` — the
        # PyGithub typed wrapper wants a ``Commit`` object, while the
        # REST API accepts a plain ``commit_id`` SHA string.
        payload = {"commit_id": head_sha, "body": body, "event": "APPROVE"}
        pr._requester.requestJsonAndCheck("POST", f"{pr.url}/reviews", input=payload)
    except Exception as exc:
        raise GitHubAPIError(
            f"failed to submit approval review on {repository}#{pr_number} as "
            f"{login or 'installation token'}: {exc}"
        ) from exc

    debug(
        1,
        f"Submitted APPROVE review on {repository}#{pr_number} as {login or 'installation token'}",
    )
    return ApprovalOutcome(submitted=True, login=login)
