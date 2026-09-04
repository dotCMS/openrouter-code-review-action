from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from ..clients.git_ops import (
    git_commit_shas,
    git_diff_name_only,
    git_diff_text,
    git_is_ancestor,
)
from ..clients.github_client import GitHubClient, GitHubClientProtocol
from ..clients.model_client import ModelClientProtocol, create_model_client
from ..clients.openrouter_thread_store import OpenRouterThreadStore
from ..core.config import ReviewConfig, make_debug
from ..core.exceptions import DotBotExecutionError, ReviewContractError, ReviewResumeError
from ..core.github_types import (
    ChangedFileProtocol,
    IssueCommentLikeProtocol,
    PullRequestLikeProtocol,
    ReviewCommentLikeProtocol,
)
from ..core.models import (
    REVIEW_OUTPUT_SCHEMA,
    CarriedForwardReviewComment,
    PriorDotBotReviewComment,
    ReviewRunResult,
)
from ..core.review_state import PriorReviewKey
from ..core.sha_delta import ShaDeltaResolver, ShaDeltaResult
from ..review.anchor_engine import build_anchor_maps
from ..review.approval import build_approval_body, submit_pr_approval
from ..review.artifacts import ReviewArtifacts
from ..review.context_manager import ReviewContextWriter
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
from ..review.resume_state import (
    MAX_INLINE_INCREMENTAL_DIFF_LINES,
    load_latest_thread_id,
    parse_review_summary_meta,
    parse_reviewed_head_sha,
    render_review_summary_metadata,
)
from ..review.review_prompt import (
    compose_prompt,
    load_guidelines,
    render_additional_review_instructions,
)

# ``diff --git a/<path> b/<path>`` is the canonical header in unified-diff
# output. We use the *post-image* path (``b/...``) because that is the
# filename on the current HEAD, which is the side the review prompt and
# the inline comment posting layer both anchor against.
_DIFF_HEADER_RE = re.compile(r'^diff --git a/(?:[^\s]+) b/("(?:[^"\\]|\\.)*"|\S+)', re.MULTILINE)

SUMMARY_TIP = (
    'Tip: comment with "/dotbot address comments" to attempt automated fixes for unresolved '
    "review threads."
)


def _extract_paths_from_unified_diff(diff_text: str) -> frozenset[str]:
    """Return the unique post-image paths referenced by a unified diff.

    Sub-AC 4.3 — used to scope the model input to only the files that
    appear in the SHA-delta. We pull the post-image (``b/...``) path
    because that is the filename on the current HEAD; the workflow's
    ``ChangedFileProtocol.filename`` and the inline-comment posting
    layer both anchor against the same side. Quoted paths (binary
    files, paths with spaces) are unwrapped per ``git`` output rules.
    """
    paths: list[str] = []
    seen: set[str] = set()
    for match in _DIFF_HEADER_RE.finditer(diff_text):
        raw_path = match.group(1)
        if raw_path.startswith('"') and raw_path.endswith('"'):
            try:
                # ``git`` quotes paths with non-ASCII / whitespace using
                # C-style escapes; Python's unicode_escape decoder is
                # close enough for our scoping purpose (we only need to
                # recover the literal path string for set-membership).
                raw_path = raw_path[1:-1].encode("utf-8").decode("unicode_escape")
            except UnicodeDecodeError:
                raw_path = raw_path[1:-1]
        if raw_path and raw_path not in seen:
            seen.add(raw_path)
            paths.append(raw_path)
    return frozenset(paths)


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


@dataclass(frozen=True)
class _ReviewResumeState:
    previous_reviewed_sha: str
    resume_thread_id: str
    inline_diff: str | None
    commit_shas: tuple[str, ...]


@dataclass(frozen=True)
class _ShaDeltaScope:
    """Outcome of the SHA-delta scoping decision for a review run.

    Sub-AC 4.3 — wraps the :class:`ShaDeltaResult` produced by
    :class:`ShaDeltaResolver` together with the *scoped* changed-file
    list the rest of the workflow should feed to the model. When the
    decision is ``incremental`` the scoped list contains only the files
    touched between the previously reviewed SHA and the current HEAD;
    on every other branch (``same_sha``, ``fresh_review``, or no prior
    state) it is the full PR file list, preserving the legacy "review
    everything" behaviour.

    The wrapper exists so the workflow can pass *one* object — instead
    of a tuple of (decision, files, paths, prompt block) — through the
    review pipeline, and so the scoping decision can be inspected from
    tests as a single value.
    """

    result: ShaDeltaResult
    scoped_changed_files: list[ChangedFileProtocol]
    scoped_paths: tuple[str, ...]

    @property
    def is_incremental(self) -> bool:
        return self.result.is_incremental

    @property
    def is_fresh_review(self) -> bool:
        return self.result.is_fresh_review

    @property
    def is_same_sha(self) -> bool:
        return self.result.is_same_sha

    @property
    def previous_reviewed_sha(self) -> str | None:
        return self.result.previous_reviewed_sha


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
    """Main workflow for code review operations."""

    def __init__(
        self,
        config: ReviewConfig,
        *,
        github_client: GitHubClientProtocol | None = None,
        model_client: ModelClientProtocol | None = None,
    ) -> None:
        self.config = config
        self.model_client = model_client or create_model_client(config)
        self.context_manager = ReviewContextWriter()
        self.github_client: GitHubClientProtocol = github_client or GitHubClient(config)
        self._debug = make_debug(config)

    def _build_review_base_instructions(self, guidelines: str) -> str:
        """Construct base instructions for dotbot review runs."""
        parts: list[str] = [
            "You are an autonomous code review assistant.",
            "Follow the review guidelines below verbatim while producing prioritized, actionable findings.",
            "Treat 'REVIEW COMMENT FORMAT (REPO STANDARD)' as authoritative over generic formatting guidance.",
        ]

        guidelines_text = guidelines.strip()
        if guidelines_text:
            parts.append("\nReview guidelines:\n" + guidelines_text)

        parts.append(
            "Use git commands as needed to inspect the diff between the PR head and the base branch."
        )
        return "\n".join(parts).strip()

    def _latest_reviewed_head_sha(
        self,
        issue_comments: list[IssueCommentLikeProtocol],
    ) -> str | None:
        for comment in reversed(issue_comments):
            body = comment.body
            if not isinstance(body, str) or SUMMARY_MARKER not in body:
                continue
            reviewed_head_sha = parse_reviewed_head_sha(body)
            if reviewed_head_sha:
                return reviewed_head_sha
        return None

    def _previous_head_sha_hint(
        self,
        issue_comments: list[IssueCommentLikeProtocol],
    ) -> str | None:
        """Resolve the best-available previously-reviewed HEAD SHA hint.

        Sub-AC 4.3 — the SHA-delta resolver wants a previous-SHA hint
        to find the prior state record on disk. Two sources cover the
        common cases:

        * ``DOTBOT_REVIEW_PREVIOUS_HEAD_SHA`` — set by
          ``cli.review.prepare_resume_state`` from scanning prior PR
          summary comments. This is the action.yml-driven path.
        * Latest summary marker scraped from the in-memory issue
          comment list — covers local invocations and cases where the
          env var was not propagated.
        """
        env_value = os.environ.get("DOTBOT_REVIEW_PREVIOUS_HEAD_SHA")
        if isinstance(env_value, str):
            stripped = env_value.strip()
            if stripped:
                return stripped
        return self._latest_reviewed_head_sha(issue_comments)

    def _resume_cache_was_restored(self) -> bool:
        cache_hit = os.environ.get("DOTBOT_REVIEW_CACHE_HIT")
        if cache_hit is None:
            return True
        return cache_hit.strip().lower() == "true"

    def _resolve_review_resume_state(
        self,
        issue_comments: list[IssueCommentLikeProtocol],
        *,
        head_sha: str,
    ) -> _ReviewResumeState | None:
        previous_reviewed_sha = os.environ.get("DOTBOT_REVIEW_PREVIOUS_HEAD_SHA")
        if previous_reviewed_sha is not None:
            previous_reviewed_sha = previous_reviewed_sha.strip() or None
        if previous_reviewed_sha is None:
            previous_reviewed_sha = self._latest_reviewed_head_sha(issue_comments)
        if previous_reviewed_sha is None:
            self._debug(1, "No prior reviewed HEAD SHA found; starting fresh review")
            return None
        if not self._resume_cache_was_restored():
            self._debug(
                1,
                f"Resume cache miss for prior reviewed SHA {previous_reviewed_sha}; starting fresh",
            )
            return None

        try:
            is_ancestor = git_is_ancestor(previous_reviewed_sha, head_sha)
        except subprocess.CalledProcessError as exc:
            self._debug(
                1,
                "Failed to validate review resume ancestry "
                f"{previous_reviewed_sha} -> {head_sha}: {exc}; starting fresh review",
            )
            return None
        if not is_ancestor:
            self._debug(
                1,
                f"Prior reviewed SHA {previous_reviewed_sha} is not an ancestor of {head_sha}; starting fresh",
            )
            return None

        resume_thread_id: str | None
        if self.config.model_provider == "openai":
            codex_home_value = os.environ.get("CODEX_HOME")
            if not isinstance(codex_home_value, str) or not codex_home_value.strip():
                raise ReviewResumeError(
                    "Resume cache restored but CODEX_HOME is unset; cannot resume review thread"
                )
            codex_home = Path(codex_home_value)
            try:
                resume_thread_id = load_latest_thread_id(codex_home, Path.cwd())
            except ReviewResumeError as exc:
                self._debug(1, f"{exc}; starting fresh review")
                return None
        else:
            resume_thread_id = self._latest_openrouter_thread_id()
            if resume_thread_id is None:
                self._debug(
                    1,
                    "Resume cache restored but no OpenRouter thread was found for "
                    f"{self.config.repository}#{self.config.pr_number} "
                    f"model={self.config.review_model}; starting fresh review",
                )
                return None

        revision_range = f"{previous_reviewed_sha}..{head_sha}"
        try:
            incremental_diff = git_diff_text(revision_range)
            commit_shas = tuple(git_commit_shas(revision_range))
        except subprocess.CalledProcessError as exc:
            raise ReviewResumeError(
                f"Failed to compute incremental review context for {revision_range}: {exc}"
            ) from exc

        diff_line_count = len(incremental_diff.splitlines())
        inline_diff = None
        if diff_line_count <= MAX_INLINE_INCREMENTAL_DIFF_LINES:
            inline_diff = incremental_diff.strip() or None
        self._debug(
            1,
            "Resuming review from "
            f"{previous_reviewed_sha} with thread {resume_thread_id}; "
            f"incremental diff lines={diff_line_count}, "
            f"{'embedding diff' if inline_diff is not None else 'using commit range only'}",
        )
        return _ReviewResumeState(
            previous_reviewed_sha=previous_reviewed_sha,
            resume_thread_id=resume_thread_id,
            inline_diff=inline_diff,
            commit_shas=commit_shas,
        )

    def _latest_openrouter_thread_id(self) -> str | None:
        """Most recently persisted OpenRouter thread for this repo/PR/model."""
        store = OpenRouterThreadStore(OpenRouterThreadStore.default_directory())
        try:
            return store.latest_thread_id(
                repository=self.config.repository,
                pr_number=self.config.pr_number,
                model=self.config.review_model,
            )
        except OSError as exc:
            self._debug(1, f"Failed to list OpenRouter threads: {exc}; starting fresh review")
            return None

    def _resolve_sha_delta_scope(
        self,
        changed_files: list[ChangedFileProtocol],
        *,
        head_sha: str,
        previous_head_sha: str | None,
    ) -> _ShaDeltaScope | None:
        """Compute the SHA-delta-driven scope for the upcoming review run.

        Sub-AC 4.3 — this is the single integration point for the
        :class:`ShaDeltaResolver` (Sub-AC 4.2). It answers two questions
        the rest of the workflow needs:

        * *Did prior state exist for this PR/model?* — controls whether
          we emit a ``<sha_delta_scope>`` block to the model.
        * *If incremental, which files actually changed in the delta?*
          — controls which files we surface in the prompt's
          ``<changed_files>`` list and whose hunks we inline.

        Falls back to ``None`` when the run cannot be scoped (no PR
        number on the config, missing review model, or the resolver
        threw an unexpected non-``CalledProcessError``). On every other
        branch the wrapper carries the resolver's verdict and the
        scoped file list. ``fresh_review`` and ``same_sha`` reuse the
        full PR file list so downstream code keeps the "review
        everything" behaviour the seed says first runs and missing
        state must take.
        """
        if self.config.pr_number is None:
            return None
        review_model = (self.config.review_model or "").strip()
        if not review_model:
            return None

        runner_temp = os.environ.get("RUNNER_TEMP", "").strip() or None
        try:
            resolver = ShaDeltaResolver.from_runner_temp(
                runner_temp,
                debug=self._debug,
            )
        except Exception as exc:  # noqa: BLE001 — defensive: state-dir setup
            self._debug(
                1,
                f"SHA-delta resolver setup failed for runner_temp={runner_temp!r}: "
                f"{exc}; using full review",
            )
            return None

        try:
            current_key = PriorReviewKey(
                repository=self.config.repository,
                pr_number=self.config.pr_number,
                review_model=review_model,
                reviewed_head_sha=head_sha,
            )
        except ReviewContractError as exc:
            self._debug(1, f"SHA-delta key construction failed: {exc}; using full review")
            return None

        sha_delta_result = resolver.resolve(
            current_key,
            previous_head_sha=previous_head_sha,
        )

        scoped_files = changed_files
        scoped_paths: tuple[str, ...] = tuple(
            file.filename for file in changed_files if file.filename
        )
        if sha_delta_result.is_incremental:
            delta_paths = self._compute_sha_delta_paths(sha_delta_result)
            if delta_paths:
                filtered = [file for file in changed_files if file.filename in delta_paths]
                if filtered:
                    scoped_files = filtered
                    scoped_paths = tuple(file.filename for file in filtered if file.filename)
                    self._debug(
                        1,
                        "SHA-delta scoped review prompt: "
                        f"{len(filtered)}/{len(changed_files)} changed files "
                        "from PR limited to delta range "
                        f"{sha_delta_result.previous_reviewed_sha}.."
                        f"{sha_delta_result.current_head_sha}",
                    )
                else:
                    # Delta paths exist but none match the PR file list
                    # (e.g. base-branch merge that touched files outside
                    # the PR). Fall back to the full file set so we do
                    # not silently hide the rest of the PR from the
                    # model.
                    self._debug(
                        1,
                        "SHA-delta path filter produced empty intersection with PR file list; "
                        "keeping full PR file set",
                    )
            else:
                self._debug(
                    1,
                    "SHA-delta range produced no extractable file paths; keeping full PR file set",
                )

        return _ShaDeltaScope(
            result=sha_delta_result,
            scoped_changed_files=scoped_files,
            scoped_paths=scoped_paths,
        )

    def _compute_sha_delta_paths(self, result: ShaDeltaResult) -> frozenset[str]:
        """Best-effort path list for the SHA-delta range.

        Prefers parsing the inline diff text when the resolver embedded
        one (cheap, no extra subprocess) and falls back to
        ``git diff --name-only`` when the diff was capped above the
        inline-line limit. Returns an empty frozenset on failure so the
        caller can downgrade to "scope by hunk-only" without raising —
        the scoping is an optimisation, not a correctness gate.
        """
        if result.incremental_diff:
            paths = _extract_paths_from_unified_diff(result.incremental_diff)
            if paths:
                return paths

        previous_sha = result.previous_reviewed_sha
        if previous_sha is None:
            return frozenset()
        revision_range = f"{previous_sha}..{result.current_head_sha}"
        try:
            delta_paths = git_diff_name_only(revision_range)
        except subprocess.CalledProcessError as exc:
            self._debug(
                1,
                f"git diff --name-only probe failed for {revision_range}: {exc}; "
                "falling back to full PR file set",
            )
            return frozenset()
        return frozenset(delta_paths)

    def _build_sha_delta_block(self, scope: _ShaDeltaScope | None) -> str:
        """Render the SHA-delta scope block embedded in the review prompt.

        Empty when the scope is missing or non-incremental — the model
        sees the regular full-PR prompt and the workflow behaves the
        same as a first-run review.
        """
        if scope is None or not scope.is_incremental:
            return ""
        result = scope.result
        previous_reviewed_sha = result.previous_reviewed_sha or ""
        lines = [
            "<sha_delta_scope>",
            "Only review changes introduced since the previously reviewed commit.",
            f"<previous_reviewed_head_sha>{previous_reviewed_sha}</previous_reviewed_head_sha>",
            f"<current_head_sha>{result.current_head_sha}</current_head_sha>",
        ]
        if scope.scoped_paths:
            lines.append("<delta_files>")
            lines.extend(f"- {path}" for path in scope.scoped_paths)
            lines.append("</delta_files>")
        if result.incremental_diff is not None:
            lines.extend(
                [
                    "<incremental_diff>",
                    result.incremental_diff,
                    "</incremental_diff>",
                ]
            )
        elif result.commit_shas:
            lines.append("<incremental_commits>")
            lines.extend(result.commit_shas)
            lines.append("</incremental_commits>")
            lines.append(
                "Inspect the incremental delta locally with "
                f"`git diff {previous_reviewed_sha}..{result.current_head_sha}` as needed."
            )
        lines.append("</sha_delta_scope>")
        return "\n".join(lines)

    def _build_review_resume_block(
        self,
        resume_state: _ReviewResumeState | None,
        *,
        head_sha: str,
    ) -> str:
        if resume_state is None:
            return ""

        lines = [
            "<review_resume_context>",
            "This is a continuation of an existing review conversation.",
            "Only review changes introduced since the previously reviewed commit.",
            f"<previous_reviewed_head_sha>{resume_state.previous_reviewed_sha}</previous_reviewed_head_sha>",
            f"<current_head_sha>{head_sha}</current_head_sha>",
        ]
        if resume_state.inline_diff is not None:
            lines.extend(
                [
                    "<incremental_diff>",
                    resume_state.inline_diff,
                    "</incremental_diff>",
                ]
            )
        else:
            lines.extend(["<incremental_commits>"])
            lines.extend(resume_state.commit_shas)
            lines.extend(
                [
                    "</incremental_commits>",
                    "Inspect the incremental delta locally with "
                    f"`git diff {resume_state.previous_reviewed_sha}..{head_sha}` as needed.",
                ]
            )
        lines.append("</review_resume_context>")
        return "\n".join(lines)

    def _build_schema_prompt(self, existing_comments: list[PriorDotBotReviewComment]) -> str:
        """Build the turn-2 prompt for structured output, with optional dedup context."""
        prompt_context = render_prior_dotbot_comments_for_prompt(existing_comments)
        lines: list[str] = []
        if prompt_context:
            lines.append(prompt_context)
            lines.append(
                "Produce the JSON review output now. "
                'Use "findings" only for new, non-redundant findings from this review run. '
                'Use "carried_forward" only for entries from prior_dotbot_review_comments '
                "that still describe live issues in the current patch. "
                "For each carried_forward entry, copy the exact current_code snippet into "
                '"current_evidence" verbatim. '
                "Do not include stale or fixed comments in carried_forward. "
                "Do not include a carried-forward entry for an issue already captured in findings."
            )
        else:
            lines.append('Produce the JSON review output now. Return "carried_forward" as [].')
        return "\n".join(lines)

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

    def _execute_model_under_timeout(
        self,
        prompt: str,
        *,
        output_schema: dict[str, object],
        schema_prompt: str,
        resume_thread_id: str | None,
        model: str,
    ) -> str:
        """Run the model pass, bounded by the configured wall-clock budget.

        ``DOTBOT_MODEL_TIMEOUT_SECONDS`` (0 disables) caps a single
        reviewer model pass. A wedged stream or an over-enthusiastic tool
        loop otherwise has no natural ceiling — the per-socket HTTP timeout
        only fires on *idle* connections, so a trickling reasoning trace can
        spin indefinitely. On expiry the pass raises and the run fails fast
        instead of burning credits for an hour; the underlying request is
        abandoned when the process exits.
        """
        effective_model = (
            self.config.model_name if self.config.model_provider == "openai" else model
        )
        timeout_seconds = self.config.model_timeout_seconds
        started = time.monotonic()

        def _run() -> str:
            return self.model_client.execute_structured(
                prompt,
                model_name=effective_model,
                sandbox_mode="danger-full-access",
                output_schema=output_schema,
                schema_prompt=schema_prompt,
                resume_thread_id=resume_thread_id,
            )

        if timeout_seconds <= 0:
            output = _run()
            elapsed = time.monotonic() - started
            print(f"Reviewer {effective_model} finished in {elapsed:.1f}s")
            return output

        output_box: list[str] = []
        error_box: list[BaseException] = []

        def _worker() -> None:
            try:
                output_box.append(_run())
            except BaseException as exc:  # noqa: BLE001 — re-raised on the main thread
                error_box.append(exc)

        # Daemon thread: when the budget expires we abandon the request;
        # a non-daemon worker (or ThreadPoolExecutor's atexit join) would
        # otherwise block process exit until the stream dries up.
        worker = threading.Thread(target=_worker, daemon=True)
        worker.start()
        worker.join(timeout=timeout_seconds)
        if worker.is_alive():
            raise DotBotExecutionError(
                f"Reviewer {effective_model} exceeded the "
                f"{timeout_seconds}s wall-clock budget "
                "(DOTBOT_MODEL_TIMEOUT_SECONDS); aborting"
            )
        if error_box:
            raise error_box[0]
        elapsed = time.monotonic() - started
        print(f"Reviewer {effective_model} finished in {elapsed:.1f}s")
        return output_box[0]

    def _sanitize_review_result(
        self,
        result: ReviewRunResult,
        prior_dotbot_comments: list[PriorDotBotReviewComment],
    ) -> ReviewRunResult:
        carried_forward = self._normalize_carried_forward(
            result.carried_forward,
            prior_dotbot_comments,
        )
        if carried_forward == result.carried_forward:
            return result
        return ReviewRunResult(
            overall_correctness=result.overall_correctness,
            overall_explanation=result.overall_explanation,
            overall_confidence_score=result.overall_confidence_score,
            findings=list(result.findings),
            carried_forward=carried_forward,
        )

    def _normalize_carried_forward(
        self,
        raw_carried_forward: list[CarriedForwardReviewComment],
        prior_dotbot_comments: list[PriorDotBotReviewComment],
    ) -> list[CarriedForwardReviewComment]:
        valid_comments = {
            comment.id: comment
            for comment in prior_dotbot_comments
            if comment.is_currently_applicable
        }
        normalized_carried_forward: list[CarriedForwardReviewComment] = []
        seen_comment_ids: set[str] = set()
        dropped_count = 0
        for carried_forward in raw_carried_forward:
            comment_id = carried_forward.comment_id
            valid_comment = valid_comments.get(comment_id)
            if comment_id in seen_comment_ids or valid_comment is None:
                dropped_count += 1
                continue
            current_evidence = carried_forward.current_evidence.strip()
            if current_evidence != valid_comment.current_code.strip():
                dropped_count += 1
                continue
            normalized_carried_forward.append(
                CarriedForwardReviewComment(
                    comment_id=comment_id,
                    current_evidence=valid_comment.current_code,
                )
            )
            seen_comment_ids.add(comment_id)
        if dropped_count > 0:
            self._debug(
                1,
                f"Dropped {dropped_count} invalid carried_forward entries from structured output",
            )
        return normalized_carried_forward

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

    def _parse_structured_review_output(self, output: str) -> ReviewRunResult:
        try:
            payload = json.loads(output)
        except json.JSONDecodeError as parse_err:
            preview = output.strip()
            if not preview:
                preview = "(empty response)"
            if len(preview) > 1200:
                preview = preview[:1200] + "\n\n... (truncated)"
            self._debug(1, f"Structured output was not valid JSON: {parse_err}")
            print("Model did not return valid JSON (truncated preview):")
            print(preview)
            raise DotBotExecutionError(f"JSON parsing error: {parse_err}") from parse_err

        try:
            return ReviewRunResult.from_payload(payload)
        except ReviewContractError:
            raise
        except Exception as exc:
            raise ReviewContractError(f"Invalid structured review output: {exc}") from exc

    def _publish_summary(self, pr: PullRequestLikeProtocol, summary: str, *, model: str) -> None:
        if self.config.dry_run:
            self._debug(1, "DRY_RUN: would refresh summary issue comment")
            return

        delete_warnings = self._delete_prior_summary(pr, model=model)
        for warning in delete_warnings:
            print(warning, file=sys.stderr)
        pr.as_issue().create_comment(summary)

    def process_review(self, pr_number: int) -> ReviewWorkflowResult:
        """Process a code review for the given pull request.

        When a multi-model roster is configured (``review.models`` or
        ``DOTBOT_REVIEW_MODELS``), each model runs the full pipeline in
        sequence and posts its own findings and summary. Review state
        (resume threads, SHA-delta, cache keys) is isolated per model via
        ``config.review_model``, and snapshots are refreshed between models
        so a later model sees the earlier models' posted inline comments as
        prior dotbot context — that is the disagreement/refutation pass.
        """
        self._debug(1, f"Processing review for {self.config.repository} PR #{pr_number}")

        pr = self.github_client.get_pr(pr_number)
        changed_files = list(pr.get_files())
        rename_map = self._build_rename_map(changed_files)
        head_sha = self._require_head_sha(pr)
        self._debug_changed_files(changed_files)

        repo_root = self.config.resolved_repo_root
        context_dir_name = self.config.resolved_context_dir_name
        artifacts = ReviewArtifacts(repo_root=repo_root, context_dir_name=context_dir_name)

        roster = self.config.selected_review_models
        self._debug(1, f"Reviewer roster: {len(roster)} model(s) -> {', '.join(roster)}")

        results: list[ReviewWorkflowResult] = []
        initial_prior_comments: list[PriorDotBotReviewComment] = []
        for index, model in enumerate(roster):
            original_model = self.config.review_model
            self.config.review_model = model
            try:
                snapshots = self._capture_review_snapshots(pr, repo_root=repo_root)
                if index == 0:
                    initial_prior_comments = snapshots.prior_dotbot_comments
                self._refresh_context_artifacts(pr, artifacts, snapshots)
                result = self._run_review_model(
                    pr,
                    changed_files=changed_files,
                    rename_map=rename_map,
                    head_sha=head_sha,
                    artifacts=artifacts,
                    snapshots=snapshots,
                    model=model,
                )
                results.append(result)
            finally:
                self.config.review_model = original_model
        self._resolve_stale_prior_threads(
            pr, prior_comments=initial_prior_comments, results=results
        )
        self._maybe_approve_pr(pr, results=results, roster=roster, head_sha=head_sha)
        return results[-1]

    def _resolve_stale_prior_threads(
        self,
        pr: PullRequestLikeProtocol,
        *,
        prior_comments: list[PriorDotBotReviewComment],
        results: list[ReviewWorkflowResult],
    ) -> None:
        """Resolve prior dotbot threads every reviewer declined to carry forward.

        Each prior unresolved dotbot thread was rendered into the reviewer
        prompt, and the reviewers were asked to carry forward the ones that
        still describe live issues. A thread none of them carried forward is,
        by the model's own judgment, stale or already fixed — so close it.
        Threads the reviewers never saw (their current_code no longer matches
        the file) are left for a human. Failures are logged, never fatal.
        """
        if not self.config.resolve_stale_threads:
            return
        if self.config.dry_run:
            self._debug(1, "DRY_RUN: would resolve stale prior dotbot threads")
            return
        if not prior_comments or not results:
            return

        carried_ids = {
            comment_id
            for result in results
            for comment_id in result.review.carried_forward_comment_ids
        }
        stale_thread_ids: list[str] = []
        for comment in prior_comments:
            if not comment.is_currently_applicable:
                continue
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
        results: list[ReviewWorkflowResult],
        roster: tuple[str, ...],
        head_sha: str,
    ) -> None:
        """Approve the PR as the machine user when every reviewer agrees it is correct.

        Requires ``github_approval_token`` (a PAT owned by the machine user,
        e.g. dotCMS-Machine-User). Only fires when *every* model in the
        roster reported ``Overall: patch is correct`` — a single dissent
        (or any active finding) withholds the approval. Failures to submit
        are logged as warnings; they never fail the review run itself.
        """
        unanimous = bool(results) and all(
            result.summary.overall_correctness.strip().casefold()
            == self._OVERALL_CORRECT.casefold()
            for result in results
        )
        if not unanimous:
            self._debug(
                1,
                "Skipping PR approval: not every reviewer model reported "
                f"{self._OVERALL_CORRECT!r}",
            )
            return

        approval_token = self.config.github_approval_token.strip()
        if not approval_token:
            self._debug(
                1,
                "All reviewers agree the patch is correct; no approval token "
                "configured (set DOTBOT_GITHUB_USER_PAT to enable auto-approval)",
            )
            return

        body = build_approval_body(list(roster))
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
            print(
                f"Approved PR #{pr.number} as {outcome.login}: all reviewers agree the patch is correct"
            )
        elif outcome.reason == "already_approved":
            print(f"PR #{pr.number} already approved by {outcome.login} for {head_sha}")

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

    def _run_review_model(
        self,
        pr: PullRequestLikeProtocol,
        *,
        changed_files: list[ChangedFileProtocol],
        rename_map: dict[str, str],
        head_sha: str,
        artifacts: ReviewArtifacts,
        snapshots: _ReviewSnapshots,
        model: str,
    ) -> ReviewWorkflowResult:
        """Run a single-model review pass and post its findings."""
        guidelines = load_guidelines(self.config)
        base_instructions = self._build_review_base_instructions(guidelines)
        resume_state = self._resolve_review_resume_state(
            snapshots.issue_comments,
            head_sha=head_sha,
        )
        resume_block = self._build_review_resume_block(
            resume_state,
            head_sha=head_sha,
        )

        # Sub-AC 4.3 — scope the model input to only the SHA-delta
        # files/hunks when prior review state exists. Falls back to a
        # full review on first run, missing state, or when the scoping
        # probe cannot be completed; the legacy CODEX_HOME-driven
        # ``resume_block`` path still wins when active so existing
        # cached threads keep replacing the entire prompt with their
        # own incremental diff.
        sha_delta_scope = self._resolve_sha_delta_scope(
            changed_files,
            head_sha=head_sha,
            previous_head_sha=self._previous_head_sha_hint(snapshots.issue_comments),
        )
        scoped_changed_files = (
            sha_delta_scope.scoped_changed_files if sha_delta_scope is not None else changed_files
        )
        sha_delta_block = "" if resume_block else self._build_sha_delta_block(sha_delta_scope)

        raw_prompt = compose_prompt(self.config, scoped_changed_files, pr, artifacts)

        prompt_sections = [base_instructions]
        if resume_block:
            prompt_sections.append(resume_block)
            prompt_sections.append(render_additional_review_instructions(self.config))
        elif sha_delta_block:
            prompt_sections.append(sha_delta_block)
            prompt_sections.append(raw_prompt)
        else:
            prompt_sections.append(raw_prompt)
        prompt = "\n\n".join(section for section in prompt_sections if section)

        self._debug(2, f"Prompt length: {len(prompt)} chars")

        schema_prompt = self._build_schema_prompt(snapshots.prior_dotbot_comments)

        print(f"Running dotbot ({model}) to generate review findings...", flush=True)

        effective_model = (
            self.config.model_name if self.config.model_provider == "openai" else model
        )
        output = self._execute_model_under_timeout(
            prompt,
            output_schema=REVIEW_OUTPUT_SCHEMA,
            schema_prompt=schema_prompt,
            resume_thread_id=resume_state.resume_thread_id if resume_state is not None else None,
            model=model,
        )

        parsed_result = self._sanitize_review_result(
            self._parse_structured_review_output(output),
            snapshots.prior_dotbot_comments,
        )

        posting_outcome = self._post_results(
            parsed_result,
            changed_files,
            pr,
            head_sha,
            rename_map,
        )
        summary = self._build_summary(parsed_result)

        summary_text = _build_review_summary(
            parsed_result,
            summary,
            posting_outcome,
            reviewed_head_sha=head_sha,
            model=effective_model,
            reasoning_effort=self.config.reasoning_effort or "medium",
        )
        self._publish_summary(pr, summary_text, model=effective_model)

        return ReviewWorkflowResult(
            review=parsed_result,
            posting_outcome=posting_outcome,
            summary=summary,
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

    def _delete_prior_summary(self, pr: PullRequestLikeProtocol, *, model: str) -> list[str]:
        """Delete prior dotbot summary issue comments for the given model.

        Older summaries (written before the multi-model change) have no
        ``model`` in their metadata and are always superseded, so they are
        deleted regardless. Summaries carrying a model only supersede
        themselves — in a multi-model roster each reviewer keeps its own
        summary line.
        """
        warnings: list[str] = []
        comments = list(pr.get_issue_comments())
        for comment in comments:
            comment_body = comment.body
            body = comment_body.strip() if isinstance(comment_body, str) else ""
            if SUMMARY_MARKER not in body:
                continue
            meta = parse_review_summary_meta(body)
            prior_model = meta.get("model")
            if isinstance(prior_model, str) and prior_model and prior_model != model:
                continue
            try:
                comment.delete()
                self._debug(1, f"Deleted prior summary issue comment id={comment.id}")
            except Exception as exc:
                warning = f"Failed to delete prior summary issue comment id={comment.id}: {exc}"
                warnings.append(warning)
                self._debug(
                    1,
                    warning,
                )
        return warnings
