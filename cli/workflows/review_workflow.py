from __future__ import annotations

import json
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..clients.github_client import GitHubClient, GitHubClientProtocol
from ..clients.model_client import ModelClientProtocol
from ..clients.openrouter_client import OpenRouterClient
from ..clients.openrouter_thread_store import OpenRouterThreadStore
from ..clients.openrouter_tools import read_file_at_revision
from ..core.config import ReviewConfig, make_debug
from ..core.exceptions import ConfigurationError, ReviewContractError
from ..core.github_types import (
    ChangedFileProtocol,
    IssueCommentLikeProtocol,
    PullRequestLikeProtocol,
    ReviewCommentLikeProtocol,
)
from ..core.models import PriorDotBotReviewComment, ReviewRunResult
from ..review.anchor_engine import build_anchor_maps
from ..review.approval import build_approval_body, submit_pr_approval
from ..review.artifacts import ReviewArtifacts, render_review_summary_metadata
from ..review.context_manager import ReviewContextWriter
from ..review.coordinator import ReviewCoordinator
from ..review.dedupe import (
    SUMMARY_MARKER,
    collect_dotbot_author_logins,
    collect_prior_dotbot_review_comments,
    render_prior_dotbot_comments_for_prompt,
)
from ..review.posting import (
    ReviewPostingOutcome,
    build_inline_comment_payloads,
    persist_anchor_maps,
    post_inline_comments,
    post_remapped_comments,
)
from ..review.review_prompt import load_guidelines, render_additional_review_instructions

SUMMARY_TIP = (
    'Tip: comment with "/dotbot address comments" to attempt automated fixes for unresolved '
    "review threads."
)


@dataclass(frozen=True)
class _ReviewSnapshots:
    review_comments: list[ReviewCommentLikeProtocol]
    issue_comments: list[IssueCommentLikeProtocol]
    prior_dotbot_comments: list[PriorDotBotReviewComment]


@dataclass(frozen=True)
class ReviewSummary:
    overall_correctness: str
    current_findings_count: int
    carried_forward_count: int
    active_findings_count: int


@dataclass(frozen=True)
class ReviewWorkflowResult:
    review: ReviewRunResult
    posting_outcome: ReviewPostingOutcome
    summary: ReviewSummary


def _build_review_summary(
    review: ReviewRunResult,
    summary: ReviewSummary,
    posting_outcome: ReviewPostingOutcome,
    *,
    reviewed_head_sha: str,
    model: str,
    reasoning_effort: str,
) -> str:
    summary_lines = [
        SUMMARY_MARKER,
        render_review_summary_metadata(
            reviewed_head_sha,
            model=model,
            reasoning_effort=reasoning_effort,
        ),
        f"- Reviewer: {model} ({reasoning_effort})",
        f"- Overall: {summary.overall_correctness.strip() or 'patch is correct'}",
        f"- New findings this run: {summary.current_findings_count}",
        f"- Prior unresolved dotbot findings still relevant: {summary.carried_forward_count}",
        f"- Active findings total: {summary.active_findings_count}",
    ]
    if posting_outcome.dropped_count > 0:
        summary_lines.append(
            f"- Findings not publishable: {posting_outcome.dropped_count} ({posting_outcome.describe_drops()})"
        )
    if posting_outcome.remapped_count > 0:
        summary_lines.append(
            f"- Findings remapped to general PR comments: {posting_outcome.remapped_count} "
            f"({posting_outcome.describe_remaps()})"
        )
    if posting_outcome.post_result.dry_run:
        summary_lines.append(f"- Inline comments ready: {posting_outcome.publishable_count}")
        if posting_outcome.remapped_count > 0:
            summary_lines.append(f"- General PR comments ready: {posting_outcome.remapped_count}")

    overall_explanation = _build_summary_explanation(review, summary)
    if overall_explanation:
        summary_lines.append("")
        summary_lines.append(overall_explanation)
    summary_lines.append("")
    summary_lines.append(SUMMARY_TIP)
    summary_lines.append("")
    summary_lines.append(f"<sub>reviewed by dotbot · {model} · {reasoning_effort}</sub>")
    return "\n".join(summary_lines)


def _build_summary_explanation(
    review: ReviewRunResult,
    summary: ReviewSummary,
) -> str:
    overall_explanation = review.overall_explanation.strip()
    current_findings_count = summary.current_findings_count
    carried_forward_count = summary.carried_forward_count
    active_findings_count = summary.active_findings_count
    if active_findings_count == 0:
        return overall_explanation or "No actionable bugs remain in the current review state."
    if current_findings_count == 0 and carried_forward_count > 0:
        finding_noun = "finding" if carried_forward_count == 1 else "findings"
        verb = "applies" if carried_forward_count == 1 else "apply"
        return (
            "No new actionable bugs were found in the current changes, but "
            f"{carried_forward_count} prior unresolved dotbot {finding_noun} still {verb}, "
            "so the patch remains incorrect."
        )
    if current_findings_count > 0 and carried_forward_count == 0:
        bug_noun = "bug" if current_findings_count == 1 else "bugs"
        verb = "was" if current_findings_count == 1 else "were"
        return overall_explanation or (
            f"{current_findings_count} actionable {bug_noun} {verb} found in the current "
            "changes, so the patch remains incorrect."
        )
    new_noun = "finding" if current_findings_count == 1 else "findings"
    prior_noun = "finding" if carried_forward_count == 1 else "findings"
    new_verb = "was" if current_findings_count == 1 else "were"
    prior_verb = "applies" if carried_forward_count == 1 else "apply"
    aggregate_summary = (
        f"{current_findings_count} new actionable {new_noun} {new_verb} identified in the current "
        f"changes, and {carried_forward_count} prior unresolved dotbot {prior_noun} still {prior_verb}, "
        "so the patch remains incorrect."
    )
    if not overall_explanation:
        return aggregate_summary
    return f"{aggregate_summary}\n\n{overall_explanation}"


class ReviewWorkflow:
    """Review every roster through the same pinned, read-only coordinator."""

    def __init__(
        self,
        config: ReviewConfig,
        *,
        github_client: GitHubClientProtocol | None = None,
        model_client_factory: Callable[[ReviewConfig], ModelClientProtocol] | None = None,
    ) -> None:
        if config.model_provider != "openrouter":
            raise ConfigurationError(
                "Review mode requires OpenRouter; use a v1 release for the old review provider"
            )
        self.config = config
        self._model_client_factory = model_client_factory
        self.context_manager = ReviewContextWriter()
        self.github_client = github_client or GitHubClient(config)
        self._debug = make_debug(config)

    def _build_rename_map(self, changed_files: list[ChangedFileProtocol]) -> dict[str, str]:
        rename_map: dict[str, str] = {}
        for changed_file in changed_files:
            if changed_file.status != "renamed":
                continue
            previous_filename = changed_file.previous_filename
            current_filename = changed_file.filename
            if previous_filename and current_filename:
                rename_map[previous_filename] = current_filename
        return rename_map

    def _require_head_sha(self, pr: PullRequestLikeProtocol) -> str:
        head_sha = pr.head.sha if pr.head else None
        if head_sha:
            return head_sha
        raise ReviewContractError(
            f"Missing PR head commit SHA for {self.config.repository}#{pr.number}"
        )

    def _debug_changed_files(self, changed_files: list[ChangedFileProtocol]) -> None:
        self._debug(1, f"Changed files: {len(changed_files)}")
        for changed_file in changed_files[:10]:
            patch_len = (
                len(changed_file.patch.splitlines()) if isinstance(changed_file.patch, str) else 0
            )
            self._debug(
                2,
                f" - {changed_file.filename} status={changed_file.status} patch_len={patch_len}",
            )

    def _capture_review_snapshots(
        self,
        pr: PullRequestLikeProtocol,
        *,
        repo_root: Path,
        source_reader: Callable[[str], str | None] | None = None,
    ) -> _ReviewSnapshots:
        try:
            review_comments_snapshot = list(pr.get_review_comments())
        except Exception as exc:
            raise ReviewContractError(
                f"Failed to retrieve review comments for {self.config.repository}#{pr.number}: {exc}"
            ) from exc
        try:
            issue_comments_snapshot = list(pr.get_issue_comments())
        except Exception as exc:
            raise ReviewContractError(
                f"Failed to retrieve issue comments for {self.config.repository}#{pr.number}: {exc}"
            ) from exc
        prior_dotbot_comments: list[PriorDotBotReviewComment] = []
        dotbot_author_logins = collect_dotbot_author_logins(issue_comments_snapshot)
        if dotbot_author_logins:
            try:
                review_threads_snapshot = self.github_client.get_review_threads(pr)
            except Exception as exc:
                raise ReviewContractError(
                    "Failed to retrieve review thread state for "
                    f"{self.config.repository}#{pr.number}: {exc}"
                ) from exc
            prior_dotbot_comments = collect_prior_dotbot_review_comments(
                review_threads_snapshot,
                dotbot_author_logins,
                repo_root,
                source_reader=source_reader,
            )
            self._debug(
                1,
                "Prior dotbot review thread matching: "
                f"{len(dotbot_author_logins)} normalized author login(s), "
                f"{len(prior_dotbot_comments)} unresolved thread(s) matched",
            )
        return _ReviewSnapshots(
            review_comments=review_comments_snapshot,
            issue_comments=issue_comments_snapshot,
            prior_dotbot_comments=prior_dotbot_comments,
        )

    def _build_summary(
        self,
        review: ReviewRunResult,
    ) -> ReviewSummary:
        current_findings_count = len(review.findings)
        carried_forward_count = len(review.carried_forward)
        active_findings_count = current_findings_count + carried_forward_count
        overall_correctness = review.overall_correctness.strip()
        if not overall_correctness:
            overall_correctness = (
                "patch is incorrect" if active_findings_count else "patch is correct"
            )
        elif active_findings_count > 0 and overall_correctness.casefold() == "patch is correct":
            overall_correctness = "patch is incorrect"
        return ReviewSummary(
            overall_correctness=overall_correctness,
            current_findings_count=current_findings_count,
            carried_forward_count=carried_forward_count,
            active_findings_count=active_findings_count,
        )

    def _publish_summary(self, pr: PullRequestLikeProtocol, summary: str) -> None:
        if self.config.dry_run:
            self._debug(1, "DRY_RUN: would refresh summary issue comment")
            return

        delete_warnings = self._delete_prior_summary(pr)
        for warning in delete_warnings:
            print(warning, file=sys.stderr)
        pr.as_issue().create_comment(summary)

    def process_review(self, pr_number: int) -> ReviewWorkflowResult:
        """Collect, deliberate, and publish one final review, even for a single model."""
        pr = self.github_client.get_pr(pr_number)
        changed_files = list(pr.get_files())
        rename_map = self._build_rename_map(changed_files)
        head_sha = self._require_head_sha(pr)
        self._debug_changed_files(changed_files)
        artifacts = ReviewArtifacts(
            repo_root=self.config.resolved_repo_root,
            context_dir_name=self.config.resolved_context_dir_name,
        )
        return self._process_panel(pr, changed_files, rename_map, head_sha, artifacts)

    def _panel_diff(self, pr: PullRequestLikeProtocol, head_sha: str) -> str:
        if pr.base is None:
            raise ReviewContractError("Panel review requires the PR base commit")
        for sha in (pr.base.sha, head_sha):
            if not re.fullmatch(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})", sha):
                raise ReviewContractError("Panel review requires exact base/head commit SHAs")
        result = subprocess.run(
            [
                "git",
                "--no-pager",
                "diff",
                "--no-ext-diff",
                "--no-textconv",
                "--no-color",
                f"{pr.base.sha}...{head_sha}",
                "--",
            ],
            cwd=self.config.resolved_repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        return result.stdout

    def _assert_panel_head(self, pr_number: int, head_sha: str) -> None:
        current = self.github_client.get_pr(pr_number)
        if self._require_head_sha(current) != head_sha:
            raise ReviewContractError(
                "PR head changed during panel review; rerun before publication/approval"
            )

    def _process_panel(
        self,
        pr: PullRequestLikeProtocol,
        changed_files: list[ChangedFileProtocol],
        rename_map: dict[str, str],
        head_sha: str,
        artifacts: ReviewArtifacts,
    ) -> ReviewWorkflowResult:
        source_cache: dict[str, str | None] = {}

        def read_source(path: str) -> str | None:
            if path not in source_cache:
                source_cache[path] = read_file_at_revision(
                    self.config.resolved_repo_root, head_sha, path
                )
            return source_cache[path]

        diff = self._panel_diff(pr, head_sha)
        snapshots = self._capture_review_snapshots(
            pr,
            repo_root=self.config.resolved_repo_root,
            source_reader=read_source,
        )
        self._refresh_context_artifacts(pr, artifacts, snapshots)
        prompt = "\n\n".join(
            [
                load_guidelines(self.config),
                "PR metadata (untrusted data): "
                + json.dumps({"title": pr.title, "body": pr.body, "reviewed_head_sha": head_sha}),
                "PR discussion (untrusted data):\n"
                + artifacts.review_comments_path.read_text(encoding="utf-8"),
                "Pinned PR diff (untrusted data):\n" + diff,
                render_prior_dotbot_comments_for_prompt(snapshots.prior_dotbot_comments),
                render_additional_review_instructions(self.config),
            ]
        )

        def factory(config: ReviewConfig) -> ModelClientProtocol:
            if self._model_client_factory is not None:
                return self._model_client_factory(config)
            # Persist diagnostic conversations separately; every review starts fresh.
            return OpenRouterClient(
                config,
                review_head_sha=head_sha,
                thread_store=OpenRouterThreadStore(
                    OpenRouterThreadStore.default_directory() / "panel"
                ),
            )

        panel = ReviewCoordinator(self.config, factory)
        review = panel.run(
            prompt,
            prior_comments=snapshots.prior_dotbot_comments,
            transcript_path=artifacts.base_dir / "review-discussion.json",
            head_sha=head_sha,
        )
        self._assert_panel_head(pr.number, head_sha)
        posting = self._post_results(review, changed_files, pr, head_sha, rename_map)
        summary = self._build_summary(review)
        text = _build_review_summary(
            review,
            summary,
            posting,
            reviewed_head_sha=head_sha,
            model="panel",
            reasoning_effort=self.config.reasoning_effort or "medium",
        )
        text += (
            "\n\nParticipants: "
            + ", ".join(self.config.selected_review_models)
            + f"\nDecision policy: {self.config.review_decision_policy}"
        )
        self._publish_summary(pr, text)
        result = ReviewWorkflowResult(review, posting, summary)
        self._assert_panel_head(pr.number, head_sha)
        self._resolve_stale_prior_threads(
            pr, prior_comments=snapshots.prior_dotbot_comments, review=review
        )
        self._maybe_approve_pr(
            pr, result=result, roster=self.config.selected_review_models, head_sha=head_sha
        )
        return result

    def _resolve_stale_prior_threads(
        self,
        pr: PullRequestLikeProtocol,
        *,
        prior_comments: list[PriorDotBotReviewComment],
        review: ReviewRunResult,
    ) -> None:
        """Resolve only prior issues explicitly rejected by the final panel decision.

        Unseen threads stay open for a human; API failures are reported without
        losing the completed review.
        """
        if not self.config.resolve_stale_threads:
            return
        if self.config.dry_run:
            self._debug(1, "DRY_RUN: would resolve stale prior dotbot threads")
            return
        if not prior_comments:
            return

        carried_ids = set(review.carried_forward_comment_ids)
        # Mirror render_prior_dotbot_comments_for_prompt's 200-comment cap so
        # we only ever resolve threads the reviewers actually saw.
        candidates = [comment for comment in prior_comments if comment.is_currently_applicable][
            :200
        ]
        stale_thread_ids: list[str] = []
        for comment in candidates:
            if comment.id in carried_ids:
                continue
            if comment.thread_id not in stale_thread_ids:
                stale_thread_ids.append(comment.thread_id)

        if not stale_thread_ids:
            self._debug(1, "No stale prior dotbot threads to resolve")
            return

        for thread_id in stale_thread_ids:
            try:
                self.github_client.resolve_review_thread(pr, thread_id)
                self._debug(1, f"Resolved stale prior dotbot thread {thread_id}")
            except Exception as exc:
                print(
                    f"Failed to resolve stale prior dotbot thread {thread_id}: {exc}",
                    file=sys.stderr,
                )

    _OVERALL_CORRECT = "patch is correct"

    def _maybe_approve_pr(
        self,
        pr: PullRequestLikeProtocol,
        *,
        result: ReviewWorkflowResult,
        roster: tuple[str, ...],
        head_sha: str,
    ) -> None:
        """Apply the final policy decision; never infer approval from missing results."""
        accepted = result.summary.overall_correctness.strip().casefold() == self._OVERALL_CORRECT
        if not accepted:
            self._debug(
                1,
                f"Skipping PR approval: final panel decision was not {self._OVERALL_CORRECT!r}",
            )
            return

        approval_token = self.config.github_approval_token.strip()
        if not approval_token:
            self._debug(
                1,
                "Final review decision accepts the patch; no approval token "
                "configured (set DOTBOT_GITHUB_USER_PAT to enable auto-approval)",
            )
            return

        policy = self.config.review_decision_policy
        body = build_approval_body(list(roster), policy=policy)
        if self.config.dry_run:
            self._debug(1, "DRY_RUN: would submit APPROVE review as machine user")
            print(f"DRY_RUN: would approve PR #{pr.number} as the machine user:")
            print(body)
            return

        try:
            outcome = submit_pr_approval(
                repository=self.config.repository,
                pr_number=pr.number,
                head_sha=head_sha,
                token=approval_token,
                body=body,
                debug=self._debug,
            )
        except Exception as exc:  # noqa: BLE001 — approval must never fail the review
            print(f"Warning: failed to submit PR approval: {exc}", file=sys.stderr)
            return
        if outcome.submitted:
            approver = f" as {outcome.login}" if outcome.login else ""
            print(f"Approved PR #{pr.number}{approver}: {policy} decision accepted the patch")
        elif outcome.reason == "already_approved":
            approver = f" by {outcome.login}" if outcome.login else ""
            print(f"PR #{pr.number} already approved{approver} for {head_sha}")
        elif outcome.reason == "head_changed":
            print(f"Skipped PR #{pr.number} approval: head changed; rerun the review")

    def _refresh_context_artifacts(
        self,
        pr: PullRequestLikeProtocol,
        artifacts: ReviewArtifacts,
        snapshots: _ReviewSnapshots,
    ) -> None:
        self.context_manager.write_context_artifacts(
            pr,
            artifacts,
            issue_comments=snapshots.issue_comments,
            review_comments=snapshots.review_comments,
        )

    def _post_results(
        self,
        result: ReviewRunResult,
        changed_files: list[ChangedFileProtocol],
        pr: PullRequestLikeProtocol,
        head_sha: str,
        rename_map: dict[str, str],
    ) -> ReviewPostingOutcome:
        """Post review results to GitHub.

        After Sub-AC 3.3.2 the posting flow has two distinct rungs:

        * Findings that anchor on the PR diff are posted as inline
          review comments (the original behaviour).
        * Findings that fail the anchor check but have usable content
          are remapped to general PR issue comments rather than
          dropped, so the reviewer's signal is preserved when the model
          aims at the wrong file or a deletion-only diff.

        Drops are still reported, separated from remaps in the
        operator-facing log line so the two are not conflated.
        """
        findings = list(result.findings)
        total_findings = len(findings)

        file_maps = build_anchor_maps(changed_files)
        repo_root = self.config.resolved_repo_root
        artifacts = ReviewArtifacts(
            repo_root=repo_root,
            context_dir_name=self.config.resolved_context_dir_name,
        )
        persist_anchor_maps(file_maps, artifacts)

        build_result = build_inline_comment_payloads(
            findings,
            file_maps,
            rename_map,
            repo_root,
            dry_run=self.config.dry_run,
            debug=self._debug,
        )
        if build_result.dropped_count > 0:
            print(
                "Posting dropped "
                f"{build_result.dropped_count}/{total_findings} findings before GitHub comment creation "
                f"({build_result.describe_drops()})"
            )
        if build_result.remapped_count > 0:
            print(
                "Posting remapped "
                f"{build_result.remapped_count}/{total_findings} findings as general PR comments "
                f"({build_result.describe_remaps()})"
            )
        post_result = post_inline_comments(
            self.github_client,
            pr,
            head_sha,
            build_result.payloads,
            dry_run=self.config.dry_run,
            debug=self._debug,
        )
        if not self.config.dry_run and build_result.payloads:
            if post_result.batch_submitted:
                print(
                    "Posted "
                    f"{post_result.posted_count}/{post_result.attempted_count} "
                    "inline comments via single batched PR review submission"
                )
            elif post_result.per_comment_fallback:
                print(
                    "Batched PR review rejected with 422; per-comment fallback posted "
                    f"{post_result.posted_count}/{post_result.attempted_count} inline comments "
                    f"({post_result.skipped_after_422} dropped after individual 422 retry)"
                )
        remapped_post_result = post_remapped_comments(
            self.github_client,
            pr,
            build_result.remapped_payloads,
            dry_run=self.config.dry_run,
            debug=self._debug,
        )
        return ReviewPostingOutcome(
            total_findings=total_findings,
            prefiltered_count=0,
            build_result=build_result,
            post_result=post_result,
            remapped_post_result=remapped_post_result,
        )

    def _delete_prior_summary(self, pr: PullRequestLikeProtocol) -> list[str]:
        """The consolidated summary supersedes all previous dotbot summaries."""
        warnings: list[str] = []
        for comment in pr.get_issue_comments():
            if not isinstance(comment.body, str) or SUMMARY_MARKER not in comment.body:
                continue
            try:
                comment.delete()
            except Exception as exc:
                warnings.append(
                    f"Failed to delete prior summary issue comment id={comment.id}: {exc}"
                )
        return warnings
