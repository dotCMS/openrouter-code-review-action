"""Submit an approving PR review as the machine user (e.g. dotCMS-Machine-User).

The reviewer workflow runs a roster of LLM reviewers over a PR. When every
reviewer in the roster concludes ``Overall: patch is correct``, the workflow
approves the PR on their behalf using a dedicated GitHub user token — so the
approval shows up as coming from the machine user, not from the action's
default ``GITHUB_TOKEN`` identity.

The approval is idempotent per head SHA: if the machine user has already
approved the current head commit, we skip re-submitting so re-triggered runs
do not spam duplicate approvals.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from github import Github

from ..core.exceptions import GitHubAPIError

APPROVAL_BODY_TEMPLATE = (
    "✅ dotbot review: all reviewer models ({models}) agree — **patch is correct**.\n\n"
    "<sub>approved automatically by dotbot</sub>"
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

    Uses a dedicated ``Github`` client authenticated with ``token`` so the
    review is attributed to the machine user (e.g. dotCMS-Machine-User)
    rather than the default action token. Skips submission when that user
    has already approved this exact head commit.

    Raises:
        GitHubAPIError: if any GitHub API call fails.
    """
    gh = Github(login_or_token=token, per_page=100)

    try:
        login = gh.get_user().login
    except Exception as exc:
        raise GitHubAPIError(f"failed to resolve approval token user: {exc}") from exc

    try:
        pr = gh.get_repo(repository).get_pull(pr_number)
    except Exception as exc:
        raise GitHubAPIError(
            f"failed to load {repository}#{pr_number} for approval: {exc}"
        ) from exc

    try:
        for review in pr.get_reviews():
            review_author = review.user.login if review.user is not None else None
            if (
                review_author == login
                and review.state == "APPROVED"
                and (review.commit_id or "") == head_sha
            ):
                debug(1, f"Approval already submitted by {login} for {head_sha}; skipping")
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
            f"failed to submit approval review on {repository}#{pr_number} as {login}: {exc}"
        ) from exc

    debug(1, f"Submitted APPROVE review on {repository}#{pr_number} as {login}")
    return ApprovalOutcome(submitted=True, login=login)
