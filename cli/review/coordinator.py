"""Bounded, synchronous discussion rounds over concurrently investigating reviewers."""

from __future__ import annotations

import json
import os
import queue
import re
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..clients.model_client import ModelClientProtocol
from ..core.config import ReviewConfig
from ..core.exceptions import (
    CorrectableReviewError,
    DotBotExecutionError,
    DotBotReviewError,
    ReviewContractError,
)
from ..core.filesystem import write_text_atomic
from ..core.model_config import validate_panel_options
from ..core.models import (
    REVIEW_OUTPUT_SCHEMA,
    CarriedForwardReviewComment,
    PriorDotBotReviewComment,
    ReviewFinding,
    ReviewRunResult,
)
from .review_prompt import load_panel_instructions

_DISPOSITION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["retain", "reject"]},
        "reason": {"type": "string"},
        "evidence": {"type": "string"},
    },
    "required": ["action", "reason", "evidence"],
    "additionalProperties": False,
}
PANEL_OUTPUT_SCHEMA: dict[str, Any] = {
    **REVIEW_OUTPUT_SCHEMA,
    "properties": {
        **REVIEW_OUTPUT_SCHEMA["properties"],
        "overall_correctness": {
            "type": "string",
            "enum": ["patch is correct", "patch is incorrect"],
        },
        "message": {"type": "string"},
        "security_summary": {"type": "string"},
        "dispositions": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    "required": [*REVIEW_OUTPUT_SCHEMA["required"], "message", "security_summary", "dispositions"],
}


def _panel_output_schema(issue_ids: set[str]) -> dict[str, Any]:
    return {
        **PANEL_OUTPUT_SCHEMA,
        "properties": {
            **PANEL_OUTPUT_SCHEMA["properties"],
            "dispositions": {
                "type": "object",
                "properties": {issue_id: _DISPOSITION_SCHEMA for issue_id in sorted(issue_ids)},
                "required": sorted(issue_ids),
                "additionalProperties": False,
            },
        },
    }


# A whole reply wrapped in one Markdown code fence; JSON is never extracted from prose.
_FENCED_REPLY = re.compile(r"\A\s*```(?:json)?[ \t]*\n(.*)\n[ \t]*```\s*\Z", re.DOTALL)
_EXCERPT_CHARS = 300
# Built only from a validated status code and error type, so safe to log whole.
_LOG_SAFE_FAILURE = re.compile(
    r"OpenRouter returned an error instead of choices: code=(?:\d{3}|unknown) type=[a-z_]+"
)


def _carries_decision(text: str) -> bool:
    """Whether a rejected reply may hold a decision that a correction could erase.

    Allowlist, fail closed: only an empty reply, or a strictly parsed panel object that is
    visibly decision-free, may be corrected. Prose, partial, escaped, duplicated or unknown
    replies count as a possible decision (prose can state a verdict too).
    """
    if not text.strip():
        return False
    try:
        payload = json.loads(text, object_pairs_hook=_unique_json_object)
    except (ReviewContractError, ValueError):
        return True
    if not isinstance(payload, dict) or not set(payload) <= set(PANEL_OUTPUT_SCHEMA["properties"]):
        return True
    decision_fields = {"findings", "carried_forward", "dispositions"}
    if any(
        isinstance(value, (dict, list))
        for key, value in payload.items()
        if key not in decision_fields
    ):
        return True  # a finding could be hiding in a text field
    dispositions = payload.get("dispositions", {})
    return (
        payload.get("findings", []) != []
        or payload.get("carried_forward", []) != []
        or payload.get("overall_correctness", "patch is correct") != "patch is correct"
        or not isinstance(dispositions, dict)
        or any(
            not isinstance(item, dict)
            or not set(item) <= set(_DISPOSITION_SCHEMA["properties"])
            or item.get("action") != "reject"
            or any(isinstance(value, (dict, list)) for value in item.values())
            for item in dispositions.values()
        )
    )


def _log_safe_failure(error: BaseException) -> str:
    """The part of a reviewer failure that may be logged with model streaming off."""
    text = " ".join(str(error).split())
    if isinstance(error, ReviewContractError):
        return text  # fixed validation messages
    if not isinstance(error, DotBotReviewError):
        return ""
    if _LOG_SAFE_FAILURE.fullmatch(text):
        return text
    # Our own errors read "<category>: <detail>"; the detail may be provider or runtime text.
    return text.split(":", 1)[0][:120]


def _excerpt(text: str) -> str:
    if len(text) <= 2 * _EXCERPT_CHARS:
        return text
    return f"{text[:_EXCERPT_CHARS]} ... {text[-_EXCERPT_CHARS:]}"


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReviewContractError("Panel JSON contains a duplicate field")
        result[key] = value
    return result


@dataclass(frozen=True)
class PanelTurn:
    review: ReviewRunResult
    message: str
    security_summary: str
    dispositions: dict[str, bool]
    payload: dict[str, Any]

    @classmethod
    def parse(cls, output: str, issue_ids: set[str]) -> PanelTurn:
        fenced = _FENCED_REPLY.match(output)
        text = fenced.group(1) if fenced else output
        try:
            return cls._parse(text, issue_ids)
        except CorrectableReviewError:
            raise
        except ReviewContractError as exc:
            if _carries_decision(text):
                raise
            # Nothing decided can be lost, so one correction is equivalent to a fresh attempt.
            raise CorrectableReviewError(str(exc)) from exc

    @classmethod
    def _parse(cls, output: str, issue_ids: set[str]) -> PanelTurn:
        if not output.strip():
            raise ReviewContractError("Panel response is empty")
        try:
            payload = json.loads(output, object_pairs_hook=_unique_json_object)
        except json.JSONDecodeError as exc:
            raise ReviewContractError(
                f"Panel response is not valid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}"
            ) from exc
        if not isinstance(payload, dict) or set(payload) != set(PANEL_OUTPUT_SCHEMA["required"]):
            raise ReviewContractError("Panel response has missing or unknown fields")
        for field in ("message", "security_summary", "overall_explanation"):
            if not isinstance(payload[field], str) or not payload[field].strip():
                raise ReviewContractError(f"Panel {field} must be non-empty text")
        review = ReviewRunResult.from_payload(payload)
        if review.overall_correctness not in {"patch is correct", "patch is incorrect"}:
            raise ReviewContractError("Panel correctness must be an explicit final position")
        if review.carried_forward:
            raise ReviewContractError("Panel prior comments must be addressed by ledger ID")
        raw = payload["dispositions"]
        if not isinstance(raw, dict) or set(raw) != issue_ids:
            raise ReviewContractError("Panel dispositions must cover exactly the ledger IDs")
        dispositions: dict[str, bool] = {}
        for issue_id, item in raw.items():
            if not isinstance(item, dict) or set(item) != set(_DISPOSITION_SCHEMA["required"]):
                raise ReviewContractError("Invalid panel disposition fields")
            if any(not isinstance(value, str) or not value.strip() for value in item.values()):
                raise ReviewContractError("Each disposition needs a reason and code evidence")
            if item["action"] not in {"retain", "reject"}:
                raise ReviewContractError("Invalid panel disposition action")
            dispositions[issue_id] = item["action"] == "retain"
        turn = cls(review, payload["message"], payload["security_summary"], dispositions, payload)
        if (
            review.findings or any(dispositions.values())
        ) and review.overall_correctness == "patch is correct":
            raise _PanelVerdictConflict(turn)
        return turn


class _PanelVerdictConflict(CorrectableReviewError):
    def __init__(self, turn: PanelTurn) -> None:
        super().__init__("Panel approval contradicts retained findings")
        self.turn = turn


@dataclass(frozen=True)
class PanelIssue:
    issue_id: str
    finding: ReviewFinding | None = None
    prior: CarriedForwardReviewComment | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "finding": self.finding.as_dict() if self.finding else None,
            "prior_comment": {
                "comment_id": self.prior.comment_id,
                "current_evidence": self.prior.current_evidence,
            }
            if self.prior
            else None,
        }


class ReviewCoordinator:
    def __init__(
        self,
        config: ReviewConfig,
        client_factory: Callable[[ReviewConfig], ModelClientProtocol],
    ) -> None:
        validate_panel_options(config.review_discussion_rounds, config.review_decision_policy)
        self.config = config
        self.roster = config.selected_review_models
        # Each client owns its own usage counters and conversations, even in tests.
        self.clients = {
            model: client_factory(
                replace(config, review_model=model, stream_output=False, debug_level=0)
            )
            for model in self.roster
        }
        if len({id(client) for client in self.clients.values()}) != len(self.roster):
            raise ReviewContractError("Panel reviewers require isolated model clients")
        self.issues: dict[str, PanelIssue] = {}
        self._finding_ids: dict[str, str] = {}
        self.transcript: list[dict[str, Any]] = []
        # Models with a pass still running; their billed cost is not final.
        self._in_flight: set[str] = set()

    def run(
        self,
        prompt: str,
        *,
        prior_comments: list[PriorDotBotReviewComment],
        transcript_path: Path,
        head_sha: str,
    ) -> ReviewRunResult:
        instructions = load_panel_instructions(self.config)
        for comment in [c for c in prior_comments if c.is_currently_applicable]:
            issue_id = f"P{len(self.issues) + 1}"
            self.issues[issue_id] = PanelIssue(
                issue_id, prior=CarriedForwardReviewComment(comment.id, comment.current_code)
            )
        context = {
            "repository": self.config.repository,
            "pr": self.config.pr_number,
            "head_sha": head_sha,
            "policy": self.config.review_decision_policy,
            "rounds": self.config.review_discussion_rounds,
            "participants": self.roster,
        }
        write_text_atomic(
            transcript_path, self._redact(json.dumps({**context, "turns": []}) + "\n")
        )
        positions: dict[str, dict[str, bool]] = {}
        final: dict[str, PanelTurn] = {}
        phases = [
            "investigation",
            *[f"discussion-{i + 1}" for i in range(self.config.review_discussion_rounds)],
        ]
        if self.config.review_decision_policy == "lead":
            phases.append("adjudication")
        for phase in phases:
            participants = self.roster[:1] if phase == "adjudication" else self.roster
            issue_ids = set(self.issues)
            round_prompt = (
                prompt
                + "\n\n"
                + instructions
                + f"\nPhase: {phase}.\n"
                + (
                    "You are the primary reviewer adjudicating all disputes. Check the evidence for every retained or rejected issue.\n"
                    if phase == "adjudication"
                    else ""
                )
                + "\nIssue ledger (data):\n"
                + json.dumps([i.as_dict() for i in self.issues.values()])
                + "\nPrevious completed rounds (data):\n"
                + json.dumps(self.transcript)
            )
            turns = self._run_round(participants, round_prompt, phase, issue_ids)
            # Register in roster order only after the whole round finishes.
            for model in participants:
                turn = turns[model]
                vote = dict(turn.dispositions)
                added_ids: list[str] = []
                for finding in turn.review.findings:
                    issue_id = self._register_finding(finding)
                    if vote.get(issue_id) is False:
                        raise ReviewContractError("Panel rejected and reintroduced the same issue")
                    vote[issue_id] = True
                    added_ids.append(issue_id)
                positions[model] = vote
                final[model] = turn
                entry = {"model": model, "phase": phase, "new_issue_ids": added_ids, **turn.payload}
                self.transcript.append(entry)
                if self.config.stream_output:
                    print(
                        self._redact(f"[review-panel] {json.dumps(entry, ensure_ascii=True)}"),
                        flush=True,
                    )
            write_text_atomic(
                transcript_path,
                self._redact(json.dumps({**context, "turns": self.transcript}, indent=2) + "\n"),
            )
        voters = self.roster[:1] if self.config.review_decision_policy == "lead" else self.roster
        retained = [
            issue
            for issue_id, issue in self.issues.items()
            if any(positions[model].get(issue_id, False) for model in voters)
        ]
        findings = sorted(
            (issue.finding for issue in retained if issue.finding is not None),
            key=lambda finding: finding.priority if finding.priority is not None else 4,
        )
        carried = [issue.prior for issue in retained if issue.prior is not None]
        correct = not retained and all(
            final[m].review.overall_correctness == "patch is correct" for m in voters
        )
        explanation = "\n\n".join(
            f"{model}: {final[model].review.overall_explanation}\nSecurity: {final[model].security_summary}"
            for model in voters
        )
        return ReviewRunResult(
            overall_correctness="patch is correct" if correct else "patch is incorrect",
            overall_explanation=self._redact(explanation),
            overall_confidence_score=None,
            findings=findings,
            carried_forward=carried,
        )

    def cost_summary(self) -> str:
        """Billed cost per model; any pass without a reported cost makes the total unavailable."""
        total: float | None = 0.0
        parts = []
        for model, client in self.clients.items():
            # Read in-flight first so a pass count is never newer than the cost it is paired with.
            in_flight = model in self._in_flight
            passes = getattr(client, "billed_passes", 0)
            cost = None if in_flight else getattr(client, "billed_cost_usd", None)
            total = None if cost is None or total is None else total + cost
            shown = "cost unavailable" if cost is None else f"${cost:,.2f}"
            parts.append(f"{model} {shown} over {passes} pass{'' if passes == 1 else 'es'}")
        return f"{'cost unavailable' if total is None else f'${total:,.2f}'} ({'; '.join(parts)})"

    def _register_finding(self, finding: ReviewFinding) -> str:
        raw = finding.code_location.absolute_file_path
        root = self.config.resolved_repo_root
        path = Path(os.path.abspath(root / raw))
        if not path.is_relative_to(root.resolve()):
            # Models sometimes emit a foreign absolute path (e.g. /README.md); read it as repo-relative.
            path = Path(os.path.abspath(root / raw.lstrip("/")))
        if not path.is_relative_to(root.resolve()):
            raise ReviewContractError("Panel finding path escapes the repository")
        if (
            not finding.title.strip()
            or not finding.body.strip()
            or finding.code_location.end_line < finding.code_location.start_line
        ):
            raise ReviewContractError("Panel finding needs actionable content and a valid range")
        finding = replace(
            finding,
            title=self._redact(finding.title),
            body=self._redact(finding.body),
            code_location=replace(finding.code_location, absolute_file_path=str(path)),
        )
        # shortcut: exact text/location dedupe only; add semantic grouping if paraphrased duplicates become noisy.
        key = json.dumps(
            {
                "title": re.sub(
                    r"^(?:[🔴🟡⚪]\ufe0f?)?\s*\[P[0-3]\]\s*", "", finding.title.strip()
                ).casefold(),
                "body": finding.body.strip(),
                "location": finding.code_location.as_dict(),
                "side": finding.side or "RIGHT",
            },
            sort_keys=True,
        )
        if key not in self._finding_ids:
            issue_id = f"F{len(self._finding_ids) + 1}"
            self._finding_ids[key] = issue_id
            self.issues[issue_id] = PanelIssue(issue_id, finding=finding)
        else:
            issue_id = self._finding_ids[key]
            previous = self.issues[issue_id].finding
            if (
                previous is not None
                and finding.priority is not None
                and (previous.priority is None or finding.priority < previous.priority)
            ):
                self.issues[issue_id] = PanelIssue(issue_id, finding=finding)
        return self._finding_ids[key]

    def _run_round(
        self, participants: tuple[str, ...], prompt: str, phase: str, issue_ids: set[str]
    ) -> dict[str, PanelTurn]:
        completed: queue.Queue[tuple[str, PanelTurn | BaseException]] = queue.Queue()

        def worker(model: str) -> None:
            protected: PanelTurn | None = None
            validations = 0

            def validate(output: str) -> None:
                nonlocal protected, validations
                validations += 1
                try:
                    turn = PanelTurn.parse(output, issue_ids)
                except ReviewContractError as exc:
                    if isinstance(exc, _PanelVerdictConflict) and protected is None:
                        protected = exc.turn
                    correctable = validations == 1 and isinstance(exc, CorrectableReviewError)
                    self._log_rejected_reply(model, phase, validations, exc, output, correctable)
                    if correctable and self.config.stream_output:
                        print(
                            self._redact(
                                f"[review-panel-correction] {model} {phase}: one correction requested: {exc}"
                            ),
                            flush=True,
                        )
                    raise
                if protected is not None and (
                    turn.review.findings != protected.review.findings
                    or turn.dispositions != protected.dispositions
                ):
                    error = ReviewContractError(
                        "Correction changed findings or disposition actions"
                    )
                    self._log_rejected_reply(model, phase, validations, error, output, False)
                    raise error

            try:
                output = self.clients[model].execute_structured(
                    prompt + f"\nYour reviewer identity: {model}",
                    output_schema=_panel_output_schema(issue_ids),
                    output_validator=validate,
                    schema_prompt="Return the panel JSON now, covering every ledger ID with a reason and code evidence.",
                    model_name=model,
                    suppress_stream=True,
                    sandbox_mode="read-only",
                    resume_thread_id=None,
                )
                result: PanelTurn | BaseException = PanelTurn.parse(output, issue_ids)
            except BaseException as exc:
                result = exc
            self._in_flight.discard(model)
            completed.put((model, result))

        deadline = time.monotonic() + self.config.model_timeout_seconds
        self._in_flight.update(participants)
        for model in participants:
            threading.Thread(target=worker, args=(model,), daemon=True).start()
        turns: dict[str, PanelTurn] = {}
        while len(turns) < len(participants):
            remaining = (
                max(0, deadline - time.monotonic()) if self.config.model_timeout_seconds else None
            )
            try:
                model, result = completed.get(timeout=remaining)
            except queue.Empty as exc:
                pending = ", ".join(model for model in participants if model not in turns)
                raise DotBotExecutionError(
                    f"Panel {phase} timed out after {self.config.model_timeout_seconds}s "
                    f"waiting for {pending}"
                ) from exc
            if isinstance(result, BaseException):
                message = _log_safe_failure(result)
                detail = f": {self._redact(message)}" if message else ""
                raise DotBotExecutionError(
                    f"Panel {phase} failed for {model}: {type(result).__name__}{detail}"
                ) from result
            turns[model] = result
        return turns

    def _log_rejected_reply(
        self,
        model: str,
        phase: str,
        attempt: int,
        error: ReviewContractError,
        output: str,
        correctable: bool,
    ) -> None:
        """Log why a reply was rejected; the reply text itself only when model text may be logged."""
        if correctable:
            outcome = "correction allowed"
        elif attempt > 1:
            outcome = "correction rejected too"
        else:
            outcome = "not correctable: the reply may hold findings or votes"
        line = (
            f"[review-panel-rejected] {model} {phase} reply {attempt}: "
            f"{' '.join(str(error).split())} | {outcome} | {len(output)} chars"
        )
        if self.config.stream_output:
            # Redact before truncating; JSON escaping keeps model text on this one line.
            line += " | excerpt " + json.dumps(_excerpt(self._redact(output)), ensure_ascii=True)
        print(self._redact(line), file=sys.stderr, flush=True)

    def _redact(self, text: str) -> str:
        for secret in sorted(
            {
                self.config.github_token,
                self.config.github_approval_token,
                self.config.openai_api_key,
                self.config.openrouter_api_key,
            },
            key=len,
            reverse=True,
        ):
            if secret:
                text = text.replace(secret, "[REDACTED]")
        return text
