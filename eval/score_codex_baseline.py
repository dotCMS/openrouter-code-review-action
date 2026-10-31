#!/usr/bin/env python3
"""Score Codex baseline findings against the labeled ground-truth dataset.

Sub-AC 2.3.2 deliverable. Sub-AC 2.3.1 defined the labeled dataset
(``eval/labeled_dataset/dataset.yaml``); this module is its consumer.
For every PR in the bake-off sample registry, it walks the Codex
baseline findings (``eval/runs/<pr_id>/codex.json``), applies the
§4.5 matching rules against the ground-truth entries for that PR, and
writes a per-PR "scored" artifact:

    eval/runs/<pr_id>/codex_scored.json

Each Codex finding gets one **scored finding** record:

* ``matched_dataset_id`` — id of the best-matching ground-truth entry,
  or ``null`` when nothing in the dataset corresponds to this finding.
* ``assigned_label`` — ``TP`` if the matched entry is labeled ``TP``,
  ``FP`` if it is labeled ``FP``, ``UNMATCHED`` when no dataset entry
  could be paired (per methodology §3 these *count* as findings the
  labeling pass needs to triage; they do not silently drop).
* ``match_quality`` — one of ``"exact_line"`` (line ranges overlap),
  ``"line_proximity"`` (within ± tolerance lines), ``"keyword_only"``
  (matched on keyword overlap because the dataset entry's
  ``line_range_status`` is ``unverified``), or ``"none"``.
* ``match_distance`` — closest-line distance between the model finding's
  range and the dataset entry's range (``None`` for keyword-only and
  unmatched records). Mirrors the same field on
  ``evaluation/bakeoff/comparison/`` records so a labeler can compare
  the two views without re-deriving the proximity metric.

The output format is intentionally a *projection* — it does **not**
re-decide TP/FP. It records what the dataset says and how confidently
the matcher associated each Codex finding with that dataset entry.
The labeling pass (methodology §2) still adjudicates ambiguous matches;
this artifact is what makes that pass cheap.

Pending tolerance
-----------------

Codex baseline captures may be in ``capture.status = "pending"`` (the
default committed placeholder), ``"failed"`` (live capture errored), or
``"captured"`` (real findings present). The runner treats every
non-captured state as *not yet measurable* and writes a
``status = "skipped"`` scored artifact with a free-form ``notes`` field
that explains why. Real scoring only happens for ``status = "captured"``
artifacts. This mirrors the methodology §2.3 rule 5 discipline that we
never paper over missing captures with synthesized scores.

Sample-level rollup
-------------------

After per-PR records are written, the runner emits
``eval/runs/_codex_scoring_summary.json`` with status counts and
per-PR aggregate (matched_tp / matched_fp / unmatched / dataset
coverage). That single rollup is what downstream Sub-AC 1.5 (the
parity score) reads first to decide which PRs are score-ready.

Usage
-----

::

    python -m eval.score_codex_baseline                 # score every sample PR
    python -m eval.score_codex_baseline --pr-id dotcms-core-35567
    python -m eval.score_codex_baseline --check         # CI-style drift check

The ``--check`` mode rebuilds in memory and exits non-zero if the
on-disk scored artifacts differ from the rebuild (excluding the
volatile ``generated_at`` timestamp). Re-run without ``--check`` to
refresh.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from eval.labeled_dataset import (
    LabeledDataset,
    LabeledFinding,
    load_dataset,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PATH = REPO_ROOT / "eval" / "sample-prs.yaml"
RUNS_DIR = REPO_ROOT / "eval" / "runs"
SUMMARY_PATH = RUNS_DIR / "_codex_scoring_summary.json"

# Methodology §4.5 rule 2: "line ranges overlap by ≥ 1 line OR are within
# ± 3 lines of each other". We carry the same tolerance the comparison
# reference set uses so a single tweak propagates to both views.
LINE_PROXIMITY_TOLERANCE = 3

# Bumped when the scored-artifact schema changes in a way downstream
# consumers (Sub-AC 1.5 scoring) must care about.
SCHEMA_VERSION = "1"
GENERATOR_REL_PATH = "eval/score_codex_baseline.py"

# Matches the ``[P0]``/``[P1]``/``[P2]``/``[P3]`` severity tag the
# reviewer prompt instructs the model to put on every finding title.
_SEVERITY_TAG_RE = re.compile(r"\[P(?P<level>[0-3])\]")

# Capture statuses that mean the artifact carries real findings worth
# scoring. Anything else (pending / failed / unknown) gets a "skipped"
# scored artifact with a notes-field rationale.
_SCORABLE_CAPTURE_STATES: frozenset[str] = frozenset({"captured"})


# ---------------------------------------------------------------------------
# Sample loading (light-weight so we don't pull the runner module's logic)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SamplePR:
    """Per-PR sample registry entry — just the fields the scorer needs."""

    id: str
    head_sha: str


def _display_path(path: Path) -> str:
    """Render ``path`` as repo-relative when possible, absolute otherwise.

    The scorer is exercised against a tmp_path runs dir during tests
    and against the real repo runs dir in production. ``Path.relative_to``
    raises when the target is not under ``REPO_ROOT`` (e.g. tmp_path on
    macOS resolves under ``/private/var/...``), so we fall back to the
    full path string instead of crashing the user-facing notes message.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_sample(path: Path | None = None) -> list[_SamplePR]:
    if path is None:
        path = SAMPLE_PATH
    with path.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, Mapping):
        raise SystemExit(f"{path}: top-level document must be a mapping")
    raw = doc.get("prs") or []
    if not isinstance(raw, list):
        raise SystemExit(f"{path}: 'prs' must be a list")
    out: list[_SamplePR] = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            raise SystemExit(f"{path}: each prs[] entry must be a mapping")
        out.append(
            _SamplePR(
                id=str(entry["id"]),
                head_sha=str(entry["head_sha"]),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Codex artifact loading + finding normalization
# ---------------------------------------------------------------------------


def codex_artifact_path(pr_id: str, *, runs_dir: Path | None = None) -> Path:
    base = runs_dir if runs_dir is not None else RUNS_DIR
    return base / pr_id / "codex.json"


def scored_artifact_path(pr_id: str, *, runs_dir: Path | None = None) -> Path:
    base = runs_dir if runs_dir is not None else RUNS_DIR
    return base / pr_id / "codex_scored.json"


def _load_codex_artifact(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(doc, Mapping):
        return None
    # `dict` cast keeps mypy honest; mappings inside JSON are dicts.
    return dict(doc)


@dataclass(frozen=True)
class _NormalizedFinding:
    """Flat shape of a Codex finding the matcher consumes."""

    finding_id: str
    raw_index: int
    path: str | None
    line_start: int | None
    line_end: int | None
    severity: str | None
    title: str
    message: str
    confidence_score: float | None


def _extract_severity(title: str, priority: int | None) -> str | None:
    """Return ``"P0"``/``"P1"``/``"P2"``/``"P3"`` from a finding.

    Mirrors the comparison-set extractor: prefer the explicit numeric
    ``priority`` field (set by the prompt's "set priority to 0 for P0,
    1 for P1, ..." rule) and fall back to scanning the ``[P<n>]`` tag
    in the title.
    """
    if isinstance(priority, int) and 0 <= priority <= 3:
        return f"P{priority}"
    m = _SEVERITY_TAG_RE.search(title or "")
    if m is None:
        return None
    return f"P{m.group('level')}"


def _normalize_finding(
    raw: Mapping[str, Any],
    *,
    raw_index: int,
) -> _NormalizedFinding:
    """Reshape one raw Codex finding into the matcher-facing shape."""
    code_location = raw.get("code_location") if isinstance(raw, Mapping) else None
    path: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    if isinstance(code_location, Mapping):
        path_raw = code_location.get("absolute_file_path")
        if isinstance(path_raw, str) and path_raw.strip():
            path = path_raw.strip()
        line_range = code_location.get("line_range")
        if isinstance(line_range, Mapping):
            ls = line_range.get("start")
            le = line_range.get("end")
            if isinstance(ls, int) and ls > 0:
                line_start = ls
            if isinstance(le, int) and le > 0:
                line_end = le
    if line_end is None and line_start is not None:
        line_end = line_start

    title_raw = raw.get("title")
    title = title_raw.strip() if isinstance(title_raw, str) else ""
    body_raw = raw.get("body")
    message = body_raw if isinstance(body_raw, str) else ""

    priority_raw = raw.get("priority") if isinstance(raw, Mapping) else None
    priority = priority_raw if isinstance(priority_raw, int) else None

    confidence_raw = raw.get("confidence_score") if isinstance(raw, Mapping) else None
    if isinstance(confidence_raw, (int, float)):
        confidence_score: float | None = float(confidence_raw)
    else:
        confidence_score = None

    severity = _extract_severity(title, priority)
    return _NormalizedFinding(
        finding_id=f"codex.f{raw_index + 1:04d}",
        raw_index=raw_index,
        path=path,
        line_start=line_start,
        line_end=line_end,
        severity=severity,
        title=title,
        message=message,
        confidence_score=confidence_score,
    )


def _extract_findings(artifact: Mapping[str, Any]) -> list[_NormalizedFinding]:
    """Pull the normalized findings list out of one Codex artifact.

    Returns an empty list when ``review_run_result`` is absent or
    malformed — the scorer treats that the same as "no findings to
    score" and the artifact's capture status drives the surfaced state.
    """
    review = artifact.get("review_run_result")
    if not isinstance(review, Mapping):
        return []
    raw_findings = review.get("findings")
    if not isinstance(raw_findings, list):
        return []
    out: list[_NormalizedFinding] = []
    for idx, item in enumerate(raw_findings):
        if isinstance(item, Mapping):
            out.append(_normalize_finding(item, raw_index=idx))
    return out


# ---------------------------------------------------------------------------
# Matching: methodology §4.5
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchResult:
    """Outcome of pairing one Codex finding with the dataset.

    ``match_quality`` is the *strongest* signal the matcher used:

    * ``"exact_line"`` — line ranges overlap by ≥ 1 line.
    * ``"line_proximity"`` — line ranges are within ± tolerance lines.
    * ``"keyword_only"`` — the dataset entry has no line range
      (``line_range_status: unverified``); matched by path and
      keyword overlap alone.
    * ``"none"`` — no dataset entry matched.
    """

    matched: LabeledFinding | None
    match_quality: str
    match_distance: int | None
    matched_keywords: tuple[str, ...]


def _line_distance(
    a_start: int | None,
    a_end: int | None,
    b_start: int,
    b_end: int,
) -> int | None:
    """Closest-line distance between ranges ``a`` and ``b``.

    Returns ``0`` when the ranges overlap, a positive integer when
    they do not, and ``None`` when ``a`` is missing line numbers.
    """
    if a_start is None or a_end is None:
        return None
    if a_end < a_start:
        a_start, a_end = a_end, a_start
    if b_end < b_start:
        b_start, b_end = b_end, b_start
    if a_end < b_start:
        return b_start - a_end
    if b_end < a_start:
        return a_start - b_end
    return 0


def _keyword_hits(text: str, keywords: Iterable[str]) -> tuple[str, ...]:
    """Return the keywords that appear in ``text`` (case-insensitive).

    Used both as the unverified-entry anchor *and* as a tie-breaker
    when multiple confirmed entries are within tolerance of the same
    finding.
    """
    if not text:
        return ()
    lowered = text.lower()
    hits: list[str] = []
    for kw in keywords:
        if not isinstance(kw, str) or not kw:
            continue
        if kw.lower() in lowered:
            hits.append(kw)
    return tuple(hits)


def _path_matches(finding_path: str | None, gt_path: str | None) -> bool:
    """Return True when finding and ground-truth paths refer to the same file.

    Strict equality is required when both sides set a path; an
    unverified entry with ``path_hint = None`` accepts any path
    (it's a keyword-only anchor — the labeling pass adjudicates).
    """
    if gt_path is None:
        return True
    if finding_path is None:
        return False
    return finding_path == gt_path


def _score_quality(distance: int | None, *, has_line_range: bool) -> str:
    if not has_line_range:
        return "keyword_only"
    if distance is None:
        # Finding has no line numbers; treat as a missing anchor.
        return "keyword_only"
    if distance == 0:
        return "exact_line"
    return "line_proximity"


def _quality_rank(quality: str) -> int:
    """Lower number = stronger match; used to break ties deterministically."""
    return {
        "exact_line": 0,
        "line_proximity": 1,
        "keyword_only": 2,
        "none": 3,
    }.get(quality, 3)


def match_finding(
    finding: _NormalizedFinding,
    candidates: Sequence[LabeledFinding],
    *,
    tolerance: int = LINE_PROXIMITY_TOLERANCE,
) -> MatchResult:
    """Match ``finding`` against the labeled-dataset entries for one PR.

    Implements methodology §4.5:

    1. Same path (or dataset entry has no ``path_hint``).
    2. Line ranges overlap by ≥ 1 line OR within ± ``tolerance`` lines
       (skipped when the dataset entry's ``line_range_status`` is
       ``unverified`` and ``line_range`` is ``null``).
    3. The labeler's "same underlying bug" judgement is **not** done
       here — that's the human pass. We surface match candidates by
       quality and let the labeler ratify or reject.

    Selection rule when multiple candidates qualify:

    * Stronger quality wins (``exact_line`` > ``line_proximity`` >
      ``keyword_only``).
    * Within the same quality, the candidate with the smallest line
      distance wins; ties are broken by larger keyword-hit count, then
      by deterministic ``id`` order so the result is reproducible.
    """
    best: MatchResult = MatchResult(
        matched=None,
        match_quality="none",
        match_distance=None,
        matched_keywords=(),
    )

    for entry in candidates:
        if not _path_matches(finding.path, entry.path_hint):
            continue

        line_range = entry.line_range
        line_distance: int | None
        quality: str
        if line_range is None:
            # Unverified entry — keyword-only anchor. Still requires
            # at least one keyword hit so we don't pair every model
            # finding with every PR-anchored unverified entry.
            hits = _keyword_hits(
                _haystack_for_finding(finding),
                entry.match_keywords,
            )
            if not hits:
                continue
            line_distance = None
            quality = "keyword_only"
            keyword_hits: tuple[str, ...] = hits
        else:
            line_distance = _line_distance(
                finding.line_start,
                finding.line_end,
                line_range.start,
                line_range.end,
            )
            if line_distance is None or line_distance > tolerance:
                continue
            quality = _score_quality(line_distance, has_line_range=True)
            keyword_hits = _keyword_hits(
                _haystack_for_finding(finding),
                entry.match_keywords,
            )

        candidate = MatchResult(
            matched=entry,
            match_quality=quality,
            match_distance=line_distance,
            matched_keywords=keyword_hits,
        )
        if _is_stronger(candidate, best):
            best = candidate

    return best


def _haystack_for_finding(finding: _NormalizedFinding) -> str:
    """Concatenated text the keyword matcher scans.

    Pulls from title + body + path; lower-cased once at the call site
    in :func:`_keyword_hits`. We include the path because some FP-trap
    entries key on file-name fragments (e.g. integration-test paths).
    """
    parts: list[str] = []
    if finding.title:
        parts.append(finding.title)
    if finding.message:
        parts.append(finding.message)
    if finding.path:
        parts.append(finding.path)
    return "\n".join(parts)


def _is_stronger(candidate: MatchResult, current: MatchResult) -> bool:
    """Return True when ``candidate`` should replace ``current`` as best.

    Stronger quality wins. Ties are broken by smaller distance, then
    larger keyword-hit count, then dataset-entry id (for determinism).
    """
    cand_rank = _quality_rank(candidate.match_quality)
    cur_rank = _quality_rank(current.match_quality)
    if cand_rank != cur_rank:
        return cand_rank < cur_rank

    cand_dist = candidate.match_distance
    cur_dist = current.match_distance
    if cand_dist is not None and cur_dist is not None and cand_dist != cur_dist:
        return cand_dist < cur_dist

    cand_hits = len(candidate.matched_keywords)
    cur_hits = len(current.matched_keywords)
    if cand_hits != cur_hits:
        return cand_hits > cur_hits

    cand_id = candidate.matched.id if candidate.matched else ""
    cur_id = current.matched.id if current.matched else ""
    return cand_id < cur_id


# ---------------------------------------------------------------------------
# Per-PR scored artifact assembly
# ---------------------------------------------------------------------------


def _assigned_label(match: MatchResult) -> str:
    if match.matched is None:
        return "UNMATCHED"
    return match.matched.label


def _scored_finding_record(
    finding: _NormalizedFinding,
    match: MatchResult,
) -> dict[str, Any]:
    matched = match.matched
    return {
        "finding_id": finding.finding_id,
        "raw_index": finding.raw_index,
        "path": finding.path,
        "line_start": finding.line_start,
        "line_end": finding.line_end,
        "severity": finding.severity,
        "title": finding.title,
        "confidence_score": finding.confidence_score,
        "matched_dataset_id": matched.id if matched is not None else None,
        "assigned_label": _assigned_label(match),
        "match_quality": match.match_quality,
        "match_distance": match.match_distance,
        "matched_keywords": list(match.matched_keywords),
        "dataset_entry": (
            None
            if matched is None
            else {
                "id": matched.id,
                "label": matched.label,
                "severity": matched.severity,
                "class": matched.finding_class,
                "title": matched.title,
                "rationale": matched.rationale,
                "path_hint": matched.path_hint,
                "line_range": (
                    None
                    if matched.line_range is None
                    else {
                        "start": matched.line_range.start,
                        "end": matched.line_range.end,
                    }
                ),
                "line_range_status": matched.line_range_status,
                "confidence": matched.confidence,
            }
        ),
    }


def _aggregate(scored: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Per-PR roll-up: matched_tp / matched_fp / unmatched / by_quality."""
    by_label = {"TP": 0, "FP": 0, "UNMATCHED": 0}
    by_quality = {
        "exact_line": 0,
        "line_proximity": 0,
        "keyword_only": 0,
        "none": 0,
    }
    by_severity = {"P0": 0, "P1": 0, "P2": 0, "P3": 0, "unknown": 0}
    for row in scored:
        label = str(row.get("assigned_label") or "UNMATCHED")
        by_label[label] = by_label.get(label, 0) + 1
        quality = str(row.get("match_quality") or "none")
        by_quality[quality] = by_quality.get(quality, 0) + 1
        sev = row.get("severity")
        if isinstance(sev, str) and sev in by_severity:
            by_severity[sev] += 1
        else:
            by_severity["unknown"] += 1
    return {
        "total": len(scored),
        "matched_tp": by_label["TP"],
        "matched_fp": by_label["FP"],
        "unmatched": by_label["UNMATCHED"],
        "by_quality": by_quality,  # type: ignore[dict-item]
        "by_severity": by_severity,  # type: ignore[dict-item]
    }


def _dataset_coverage(
    pr_dataset_entries: Sequence[LabeledFinding],
    matched_dataset_ids: Iterable[str],
) -> dict[str, Any]:
    """How much of the ground-truth dataset for this PR did Codex hit?

    Reports the matched-vs-unmatched split at the *dataset* axis (not
    the model-finding axis). Methodology §4.4 uses TP coverage as the
    overlap denominator; surfacing it per-PR here makes the scoring
    pass downstream cheap.
    """
    matched_ids = set(matched_dataset_ids)
    by_label_total = {"TP": 0, "FP": 0}
    by_label_matched = {"TP": 0, "FP": 0}
    unmatched_ids: list[str] = []
    for entry in pr_dataset_entries:
        by_label_total[entry.label] = by_label_total.get(entry.label, 0) + 1
        if entry.id in matched_ids:
            by_label_matched[entry.label] = by_label_matched.get(entry.label, 0) + 1
        else:
            unmatched_ids.append(entry.id)
    return {
        "dataset_total": sum(by_label_total.values()),
        "dataset_matched": sum(by_label_matched.values()),
        "dataset_unmatched_ids": sorted(unmatched_ids),
        "by_label_total": by_label_total,
        "by_label_matched": by_label_matched,
    }


def _capture_state(artifact: Mapping[str, Any]) -> tuple[str, str | None]:
    """Return ``(capture_status, notes)`` from a Codex artifact."""
    capture = artifact.get("capture")
    if not isinstance(capture, Mapping):
        return ("unknown", None)
    status = str(capture.get("status") or "unknown")
    notes_raw = capture.get("notes")
    notes = notes_raw if isinstance(notes_raw, str) else None
    return (status, notes)


def _empty_aggregate() -> dict[str, Any]:
    return {
        "total": 0,
        "matched_tp": 0,
        "matched_fp": 0,
        "unmatched": 0,
        "by_quality": {
            "exact_line": 0,
            "line_proximity": 0,
            "keyword_only": 0,
            "none": 0,
        },
        "by_severity": {"P0": 0, "P1": 0, "P2": 0, "P3": 0, "unknown": 0},
    }


@dataclass(frozen=True)
class ScoredPRResult:
    """In-memory scoring output for one PR (before serialization)."""

    pr_id: str
    head_sha: str
    status: str  # "scored" | "skipped"
    capture_status: str
    findings_scored: list[dict[str, Any]]
    aggregate: dict[str, Any]
    dataset_coverage: dict[str, Any]
    notes: str | None


def score_pr(
    pr: _SamplePR,
    dataset: LabeledDataset,
    *,
    runs_dir: Path | None = None,
    tolerance: int = LINE_PROXIMITY_TOLERANCE,
) -> ScoredPRResult:
    """Score one PR's Codex baseline against the labeled dataset."""
    artifact_path = codex_artifact_path(pr.id, runs_dir=runs_dir)
    artifact = _load_codex_artifact(artifact_path)

    pr_dataset_entries = dataset.by_pr(pr.id)

    if artifact is None:
        return ScoredPRResult(
            pr_id=pr.id,
            head_sha=pr.head_sha,
            status="skipped",
            capture_status="missing",
            findings_scored=[],
            aggregate=_empty_aggregate(),
            dataset_coverage=_dataset_coverage(pr_dataset_entries, []),
            notes=(
                f"No Codex baseline artifact at {_display_path(artifact_path)}; "
                "run `python -m eval.run_codex_baseline` first."
            ),
        )

    capture_status, capture_notes = _capture_state(artifact)
    head_sha_at_capture = artifact.get("head_sha")

    if capture_status not in _SCORABLE_CAPTURE_STATES:
        return ScoredPRResult(
            pr_id=pr.id,
            head_sha=pr.head_sha,
            status="skipped",
            capture_status=capture_status,
            findings_scored=[],
            aggregate=_empty_aggregate(),
            dataset_coverage=_dataset_coverage(pr_dataset_entries, []),
            notes=(
                f"Codex baseline capture is {capture_status!r}; nothing to score yet. "
                + (capture_notes or "")
            ).strip(),
        )

    if head_sha_at_capture != pr.head_sha:
        # The methodology pins each PR's review to its registered
        # head_sha (§1.1, §4.1). A drifted SHA invalidates the score
        # because the diff being reviewed is not the diff the dataset
        # entries were anchored to.
        return ScoredPRResult(
            pr_id=pr.id,
            head_sha=pr.head_sha,
            status="skipped",
            capture_status=capture_status,
            findings_scored=[],
            aggregate=_empty_aggregate(),
            dataset_coverage=_dataset_coverage(pr_dataset_entries, []),
            notes=(
                f"Codex baseline captured {head_sha_at_capture!r} but registry pins "
                f"{pr.head_sha!r}; revalidate capture and label revisions before scoring. "
                "A PR head must not be equated with its historical merge commit."
            ),
        )

    findings = _extract_findings(artifact)
    scored_records: list[dict[str, Any]] = []
    matched_dataset_ids: list[str] = []
    for finding in findings:
        match = match_finding(finding, pr_dataset_entries, tolerance=tolerance)
        record = _scored_finding_record(finding, match)
        scored_records.append(record)
        if match.matched is not None:
            matched_dataset_ids.append(match.matched.id)

    aggregate = _aggregate(scored_records)
    coverage = _dataset_coverage(pr_dataset_entries, matched_dataset_ids)

    return ScoredPRResult(
        pr_id=pr.id,
        head_sha=pr.head_sha,
        status="scored",
        capture_status=capture_status,
        findings_scored=scored_records,
        aggregate=aggregate,
        dataset_coverage=coverage,
        notes=None,
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _build_scored_artifact(
    result: ScoredPRResult,
    *,
    generated_at: str,
    tolerance: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "pr_id": result.pr_id,
        "run": "codex",
        "head_sha": result.head_sha,
        "status": result.status,
        "capture_status": result.capture_status,
        "tolerance_lines": tolerance,
        "findings_scored": list(result.findings_scored),
        "aggregate": result.aggregate,
        "dataset_coverage": result.dataset_coverage,
        "notes": result.notes,
        "generated_at": generated_at,
        "generator": GENERATOR_REL_PATH,
    }


def _build_summary(
    results: Sequence[ScoredPRResult],
    *,
    generated_at: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    by_status: dict[str, int] = {}
    totals = {
        "scored_prs": 0,
        "skipped_prs": 0,
        "total_findings": 0,
        "total_matched_tp": 0,
        "total_matched_fp": 0,
        "total_unmatched": 0,
        "dataset_total": 0,
        "dataset_matched": 0,
    }
    for result in results:
        by_status[result.status] = by_status.get(result.status, 0) + 1
        if result.status == "scored":
            totals["scored_prs"] += 1
            totals["total_findings"] += result.aggregate["total"]
            totals["total_matched_tp"] += result.aggregate["matched_tp"]
            totals["total_matched_fp"] += result.aggregate["matched_fp"]
            totals["total_unmatched"] += result.aggregate["unmatched"]
        else:
            totals["skipped_prs"] += 1
        totals["dataset_total"] += result.dataset_coverage["dataset_total"]
        totals["dataset_matched"] += result.dataset_coverage["dataset_matched"]
        rows.append(
            {
                "pr_id": result.pr_id,
                "head_sha": result.head_sha,
                "status": result.status,
                "capture_status": result.capture_status,
                "findings_scored": result.aggregate["total"],
                "matched_tp": result.aggregate["matched_tp"],
                "matched_fp": result.aggregate["matched_fp"],
                "unmatched": result.aggregate["unmatched"],
                "dataset_total": result.dataset_coverage["dataset_total"],
                "dataset_matched": result.dataset_coverage["dataset_matched"],
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "generator": GENERATOR_REL_PATH,
        "sample_size": len(results),
        "status_counts": by_status,
        "totals": totals,
        "tolerance_lines": LINE_PROXIMITY_TOLERANCE,
        "prs": rows,
    }


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _serialize(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def write_scored_artifacts(
    results: Sequence[ScoredPRResult],
    *,
    runs_dir: Path | None = None,
    summary_path: Path | None = None,
    generated_at: str | None = None,
    tolerance: int = LINE_PROXIMITY_TOLERANCE,
) -> tuple[list[Path], Path]:
    """Persist per-PR scored artifacts and the sample-level summary."""
    if generated_at is None:
        generated_at = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    written: list[Path] = []
    for result in results:
        path = scored_artifact_path(result.pr_id, runs_dir=runs_dir)
        artifact = _build_scored_artifact(result, generated_at=generated_at, tolerance=tolerance)
        _atomic_write(path, _serialize(artifact))
        written.append(path)
    summary_target = summary_path if summary_path is not None else SUMMARY_PATH
    summary = _build_summary(results, generated_at=generated_at)
    _atomic_write(summary_target, _serialize(summary))
    return written, summary_target


# ---------------------------------------------------------------------------
# Drift check (CI-friendly)
# ---------------------------------------------------------------------------


def _strip_volatile(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy with the ``generated_at`` timestamp removed.

    Lets ``--check`` re-run on a fresh clone without flagging the
    walltime move as drift.
    """
    out = json.loads(json.dumps(payload))
    if isinstance(out, dict):
        out.pop("generated_at", None)
    return out  # type: ignore[no-any-return]


def check_scored_artifacts(
    results: Sequence[ScoredPRResult],
    *,
    runs_dir: Path | None = None,
    summary_path: Path | None = None,
    tolerance: int = LINE_PROXIMITY_TOLERANCE,
) -> list[str]:
    """Compare on-disk scored artifacts against an in-memory rebuild."""
    drift: list[str] = []
    for result in results:
        path = scored_artifact_path(result.pr_id, runs_dir=runs_dir)
        fresh = _build_scored_artifact(
            result, generated_at="<ignored-for-check>", tolerance=tolerance
        )
        on_disk = _load_codex_artifact(path)
        if on_disk is None:
            drift.append(f"{result.pr_id}: missing scored artifact at {path}")
            continue
        if _strip_volatile(on_disk) != _strip_volatile(fresh):
            drift.append(f"{result.pr_id}: on-disk scored artifact differs from rebuild")
    summary_target = summary_path if summary_path is not None else SUMMARY_PATH
    fresh_summary = _build_summary(results, generated_at="<ignored-for-check>")
    on_disk_summary = _load_codex_artifact(summary_target)
    if on_disk_summary is None:
        drift.append(f"{_display_path(summary_target)}: missing scoring summary")
    elif _strip_volatile(on_disk_summary) != _strip_volatile(fresh_summary):
        drift.append(f"{_display_path(summary_target)}: scoring summary differs from rebuild")
    return drift


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def score_all(
    sample: Sequence[_SamplePR] | None = None,
    *,
    dataset: LabeledDataset | None = None,
    runs_dir: Path | None = None,
    tolerance: int = LINE_PROXIMITY_TOLERANCE,
) -> list[ScoredPRResult]:
    """Score every PR in the sample registry against the labeled dataset."""
    if dataset is None:
        dataset = load_dataset()
    if sample is None:
        sample = _load_sample()
    return [score_pr(pr, dataset, runs_dir=runs_dir, tolerance=tolerance) for pr in sample]


def _select(sample: Sequence[_SamplePR], pr_ids: Sequence[str] | None) -> list[_SamplePR]:
    if not pr_ids:
        return list(sample)
    by_id = {pr.id: pr for pr in sample}
    missing = [pid for pid in pr_ids if pid not in by_id]
    if missing:
        raise SystemExit(
            f"unknown pr_id(s): {', '.join(missing)}. Known: {', '.join(sorted(by_id))}"
        )
    return [by_id[pid] for pid in pr_ids]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eval.score_codex_baseline",
        description=(
            "Score Codex baseline findings against the labeled ground-truth "
            "dataset. Writes eval/runs/<pr_id>/codex_scored.json per PR plus "
            "eval/runs/_codex_scoring_summary.json for the sample-level rollup."
        ),
    )
    parser.add_argument(
        "--pr-id",
        action="append",
        dest="pr_ids",
        metavar="ID",
        help="Score this PR id (may be repeated). Default: every sample PR.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Do not write; rebuild in memory and exit non-zero if the on-disk "
            "scored artifacts differ from the rebuild. Use in CI to keep the "
            "scored set in sync with eval/runs/<pr_id>/codex.json."
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=LINE_PROXIMITY_TOLERANCE,
        help=(
            f"± lines tolerance for §4.5 line-proximity matching "
            f"(default: {LINE_PROXIMITY_TOLERANCE})."
        ),
    )
    args = parser.parse_args(argv)

    sample = _load_sample()
    if not sample:
        raise SystemExit(f"{SAMPLE_PATH}: prs[] is empty. Populate the sample registry first.")
    targets = _select(sample, args.pr_ids)

    dataset = load_dataset()
    results = [score_pr(pr, dataset, tolerance=args.tolerance) for pr in targets]

    if args.check:
        drift = check_scored_artifacts(results, tolerance=args.tolerance)
        if drift:
            sys.stderr.write(
                "scored artifacts drifted from rebuild:\n  - "
                + "\n  - ".join(drift)
                + "\nRe-run `python -m eval.score_codex_baseline`\n"
            )
            return 1
        print(f"scored artifacts in sync ({len(results)} PRs)")
        return 0

    written, summary_target = write_scored_artifacts(results, tolerance=args.tolerance)
    by_status: dict[str, int] = {}
    for r in results:
        by_status[r.status] = by_status.get(r.status, 0) + 1
    by_status_str = ", ".join(f"{k}={v}" for k, v in sorted(by_status.items()))
    print(
        f"wrote {len(written)} scored artifacts and "
        f"{_display_path(summary_target)} ({by_status_str})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
