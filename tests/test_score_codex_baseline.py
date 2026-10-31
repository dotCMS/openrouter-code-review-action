"""Tests for the Codex baseline scorer (Sub-AC 2.3.2).

The scorer takes captured Codex baseline findings
(``eval/runs/<pr_id>/codex.json``) and matches them against the
ground-truth labeled dataset (``eval/labeled_dataset/dataset.yaml``)
per methodology §4.5. These tests pin:

* Methodology §4.5 line-overlap and ±-tolerance matching.
* Quality ranking when multiple ground-truth entries are within
  tolerance (``exact_line`` beats ``line_proximity`` beats
  ``keyword_only``).
* Keyword-only matching for entries with
  ``line_range_status: unverified`` (the dominant state in the
  shipped dataset until labelers pin lines).
* Pending-tolerant behavior — pending/failed/missing captures yield
  ``status = "skipped"`` records, never synthesized scores.
* Drift-check parity (``--check`` mode) against the on-disk artifacts
  so a regenerated set never silently diverges from the captures.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import score_codex_baseline as scorer  # noqa: E402
from eval.labeled_dataset import (  # noqa: E402
    LabeledDataset,
    LabeledFinding,
    LineRange,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_labeled(
    *,
    id: str = "gt-001-001",
    pr_id: str = "sample-pr-001",
    head_sha: str = "deadbeefcafe1234",
    label: str = "TP",
    severity: str = "P1",
    finding_class: str = "bug_correctness",
    title: str = "P1: example bug",
    rationale: str = "Real bug introduced by the diff.",
    match_keywords: tuple[str, ...] = ("example",),
    source: str = "retrospective_diff_review",
    confidence: str = "high",
    path_hint: str | None = "src/app.py",
    line_range: LineRange | None = LineRange(start=10, end=12),
    line_range_status: str = "confirmed",
) -> LabeledFinding:
    return LabeledFinding(
        id=id,
        pr_id=pr_id,
        head_sha=head_sha,
        label=label,
        severity=severity,
        finding_class=finding_class,
        title=title,
        rationale=rationale,
        match_keywords=match_keywords,
        source=source,
        confidence=confidence,
        path_hint=path_hint,
        line_range=line_range,
        line_range_status=line_range_status,
    )


def _normalized(
    *,
    finding_id: str = "codex.f0001",
    raw_index: int = 0,
    path: str | None = "src/app.py",
    line_start: int | None = 10,
    line_end: int | None = 12,
    severity: str | None = "P1",
    title: str = "[P1] off-by-one",
    message: str = "The loop iterates one too many times",
    confidence_score: float | None = 0.9,
) -> scorer._NormalizedFinding:
    return scorer._NormalizedFinding(
        finding_id=finding_id,
        raw_index=raw_index,
        path=path,
        line_start=line_start,
        line_end=line_end,
        severity=severity,
        title=title,
        message=message,
        confidence_score=confidence_score,
    )


def _codex_artifact(
    *,
    pr_id: str,
    head_sha: str,
    capture_status: str,
    findings: list[dict[str, Any]] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    review = (
        {
            "findings": findings or [],
            "carried_forward": [],
            "overall_correctness": "patch is correct",
            "overall_explanation": "fixture",
            "overall_confidence_score": 0.9,
        }
        if findings is not None
        else None
    )
    return {
        "pr_id": pr_id,
        "run": "codex",
        "head_sha": head_sha,
        "model": "gpt-5.4",
        "reasoning_effort": "medium",
        "web_search_mode": "disabled",
        "prior_review_state": None,
        "capture": {
            "status": capture_status,
            "runner_version": "1",
            "captured_at": "2026-05-06T12:00:00Z" if capture_status == "captured" else None,
            "command": f"python -m eval.run_codex_baseline --pr-id {pr_id}",
            "notes": notes,
        },
        "review_run_result": review,
        "posted": {
            "summary_id": None,
            "inline_ids": [],
            "posting_outcome": {
                "batch_submitted": 0,
                "per_comment_fallback": 0,
                "skipped_after_422": 0,
            },
        },
    }


def _raw_finding(
    *,
    path: str = "src/app.py",
    start: int = 10,
    end: int = 12,
    title: str = "[P1] off-by-one bug",
    body: str = "The loop iterates one too many times",
    priority: int | None = 1,
    confidence: float = 0.9,
) -> dict[str, Any]:
    return {
        "title": title,
        "body": body,
        "confidence_score": confidence,
        "priority": priority,
        "code_location": {
            "absolute_file_path": path,
            "line_range": {"start": start, "end": end},
        },
    }


def _dataset(findings: list[LabeledFinding]) -> LabeledDataset:
    return LabeledDataset(
        version="1.0",
        dataset_id="test-dataset",
        generated_at="2026-05-06",
        source_sample="eval/sample-prs.yaml",
        methodology="eval/baseline-methodology.md",
        findings=tuple(findings),
    )


# ---------------------------------------------------------------------------
# Severity extraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("title", "priority", "expected"),
    [
        ("[P0] critical", 0, "P0"),
        ("[P1] major", 1, "P1"),
        ("[P3] nit", 3, "P3"),
        ("no tag", None, None),
        # Title-tag fallback when priority is missing
        ("[P2] message", None, "P2"),
        # Priority wins over title tag
        ("[P3] looks like a nit", 0, "P0"),
    ],
)
def test_extract_severity(title, priority, expected):
    assert scorer._extract_severity(title, priority) == expected


# ---------------------------------------------------------------------------
# Line-distance arithmetic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        # Direct overlap
        ((10, 20), (15, 25), 0),
        # Touching exactly
        ((10, 20), (20, 30), 0),
        # Adjacent but separate
        ((10, 20), (22, 25), 2),
        ((10, 20), (5, 7), 3),
        # Far apart
        ((10, 20), (40, 50), 20),
        # Reversed input is normalized
        ((20, 10), (15, 25), 0),
    ],
)
def test_line_distance(a, b, expected):
    assert scorer._line_distance(a[0], a[1], b[0], b[1]) == expected


def test_line_distance_returns_none_when_finding_lacks_lines():
    assert scorer._line_distance(None, None, 10, 20) is None


# ---------------------------------------------------------------------------
# Path matching (allow None on dataset side, strict equality otherwise)
# ---------------------------------------------------------------------------


def test_path_matches_strict_on_both_sides():
    assert scorer._path_matches("src/a.py", "src/a.py")
    assert not scorer._path_matches("src/a.py", "src/b.py")


def test_path_matches_accepts_none_dataset_side():
    """Unverified FP-trap entries with no path_hint accept any model path."""
    assert scorer._path_matches("src/a.py", None)
    # Conversely, finding with no path can't match a confirmed dataset path.
    assert not scorer._path_matches(None, "src/a.py")


# ---------------------------------------------------------------------------
# Single-candidate matching: confirmed entries
# ---------------------------------------------------------------------------


def test_match_returns_exact_line_when_ranges_overlap():
    entry = _make_labeled(line_range=LineRange(10, 20))
    finding = _normalized(line_start=15, line_end=18)
    result = scorer.match_finding(finding, [entry])
    assert result.matched is entry
    assert result.match_quality == "exact_line"
    assert result.match_distance == 0


def test_match_returns_line_proximity_within_tolerance():
    entry = _make_labeled(line_range=LineRange(10, 20))
    finding = _normalized(line_start=22, line_end=23)
    result = scorer.match_finding(finding, [entry])
    assert result.matched is entry
    assert result.match_quality == "line_proximity"
    assert result.match_distance == 2


def test_match_returns_none_when_outside_tolerance():
    entry = _make_labeled(line_range=LineRange(10, 20))
    finding = _normalized(line_start=40, line_end=45)
    result = scorer.match_finding(finding, [entry])
    assert result.matched is None
    assert result.match_quality == "none"


def test_match_skips_entry_with_different_path():
    entry = _make_labeled(path_hint="src/a.py", line_range=LineRange(10, 20))
    finding = _normalized(path="src/b.py", line_start=15, line_end=18)
    result = scorer.match_finding(finding, [entry])
    assert result.matched is None


def test_match_uses_custom_tolerance():
    entry = _make_labeled(line_range=LineRange(10, 20))
    finding = _normalized(line_start=22, line_end=23)
    # Tolerance 0 forces no match
    assert scorer.match_finding(finding, [entry], tolerance=0).matched is None


# ---------------------------------------------------------------------------
# Keyword-only matching for unverified entries
# ---------------------------------------------------------------------------


def test_match_falls_back_to_keyword_when_unverified():
    entry = _make_labeled(
        line_range=None,
        line_range_status="unverified",
        path_hint=None,
        match_keywords=("CloseDBIfOpened", "atomicity"),
    )
    finding = _normalized(
        path="src/x.java",
        line_start=None,
        line_end=None,
        title="Removed @CloseDBIfOpened",
        message="restores atomicity with the contentlet save",
    )
    result = scorer.match_finding(finding, [entry])
    assert result.matched is entry
    assert result.match_quality == "keyword_only"
    assert "CloseDBIfOpened" in result.matched_keywords
    assert "atomicity" in result.matched_keywords
    assert result.match_distance is None


def test_keyword_match_requires_at_least_one_hit():
    """An unverified entry without any keyword hits must NOT silently match.

    Otherwise every model finding on a PR would pair with every
    unverified ground-truth entry that happens to lack a path, which
    would inflate matched-counts.
    """
    entry = _make_labeled(
        line_range=None,
        line_range_status="unverified",
        path_hint=None,
        match_keywords=("specific-keyword",),
    )
    finding = _normalized(
        path="src/x.java",
        title="generic finding",
        message="nothing distinctive here",
    )
    result = scorer.match_finding(finding, [entry])
    assert result.matched is None
    assert result.match_quality == "none"


def test_keyword_match_is_case_insensitive():
    entry = _make_labeled(
        line_range=None,
        line_range_status="unverified",
        path_hint=None,
        match_keywords=("AtomicITY",),
    )
    finding = _normalized(
        title="atomicity restored",
        message="...",
    )
    result = scorer.match_finding(finding, [entry])
    assert result.matched is entry


# ---------------------------------------------------------------------------
# Quality ranking when multiple candidates qualify
# ---------------------------------------------------------------------------


def test_match_prefers_exact_line_over_line_proximity():
    overlap_entry = _make_labeled(
        id="gt-001-001",
        line_range=LineRange(10, 20),
        match_keywords=("alpha",),
    )
    nearby_entry = _make_labeled(
        id="gt-001-002",
        line_range=LineRange(25, 30),
        match_keywords=("alpha",),
    )
    finding = _normalized(line_start=15, line_end=22)
    result = scorer.match_finding(finding, [nearby_entry, overlap_entry])
    assert result.matched is overlap_entry
    assert result.match_quality == "exact_line"


def test_match_prefers_line_proximity_over_keyword_only():
    proximity_entry = _make_labeled(
        id="gt-001-001",
        line_range=LineRange(10, 20),
    )
    unverified_entry = _make_labeled(
        id="gt-001-002",
        line_range=None,
        line_range_status="unverified",
        path_hint=None,
        match_keywords=("off-by-one",),
    )
    finding = _normalized(
        line_start=22,
        line_end=23,
        title="off-by-one near loop",
    )
    result = scorer.match_finding(finding, [unverified_entry, proximity_entry])
    assert result.matched is proximity_entry
    assert result.match_quality == "line_proximity"


def test_match_breaks_ties_deterministically_by_id():
    a = _make_labeled(id="gt-001-001", line_range=LineRange(10, 20))
    b = _make_labeled(id="gt-001-002", line_range=LineRange(10, 20))
    finding = _normalized(line_start=15, line_end=18)
    # Same path, same overlap, same keyword count → choose the smaller id.
    result = scorer.match_finding(finding, [b, a])
    assert result.matched is a


def test_match_breaks_ties_by_keyword_count():
    a = _make_labeled(
        id="gt-001-001",
        line_range=LineRange(10, 20),
        match_keywords=("one",),
    )
    b = _make_labeled(
        id="gt-001-002",
        line_range=LineRange(10, 20),
        match_keywords=("one", "two", "three"),
    )
    finding = _normalized(
        line_start=15,
        line_end=18,
        title="match one two three",
        message="mentions all keywords",
    )
    # Same overlap (distance 0) and same path, but b has 3 keyword hits.
    result = scorer.match_finding(finding, [a, b])
    assert result.matched is b


# ---------------------------------------------------------------------------
# Aggregate / dataset coverage
# ---------------------------------------------------------------------------


def test_aggregate_counts_labels_quality_severity():
    rows = [
        {"assigned_label": "TP", "match_quality": "exact_line", "severity": "P1"},
        {"assigned_label": "TP", "match_quality": "line_proximity", "severity": "P0"},
        {"assigned_label": "FP", "match_quality": "keyword_only", "severity": "P3"},
        {"assigned_label": "UNMATCHED", "match_quality": "none", "severity": None},
    ]
    agg = scorer._aggregate(rows)
    assert agg["total"] == 4
    assert agg["matched_tp"] == 2
    assert agg["matched_fp"] == 1
    assert agg["unmatched"] == 1
    assert agg["by_quality"]["exact_line"] == 1
    assert agg["by_quality"]["line_proximity"] == 1
    assert agg["by_quality"]["keyword_only"] == 1
    assert agg["by_quality"]["none"] == 1
    assert agg["by_severity"]["P0"] == 1
    assert agg["by_severity"]["P1"] == 1
    assert agg["by_severity"]["P3"] == 1
    assert agg["by_severity"]["unknown"] == 1


def test_dataset_coverage_splits_matched_from_unmatched():
    tp_a = _make_labeled(id="gt-001-tp-a", label="TP")
    tp_b = _make_labeled(id="gt-001-tp-b", label="TP")
    fp_a = _make_labeled(id="gt-001-fp-a", label="FP")
    coverage = scorer._dataset_coverage([tp_a, tp_b, fp_a], ["gt-001-tp-a"])
    assert coverage["dataset_total"] == 3
    assert coverage["dataset_matched"] == 1
    assert coverage["by_label_total"] == {"TP": 2, "FP": 1}
    assert coverage["by_label_matched"] == {"TP": 1, "FP": 0}
    assert coverage["dataset_unmatched_ids"] == sorted(["gt-001-tp-b", "gt-001-fp-a"])


# ---------------------------------------------------------------------------
# Per-PR scoring: pending tolerance + happy path + SHA drift detection
# ---------------------------------------------------------------------------


def _write_codex_artifact(runs_dir: Path, pr_id: str, payload: dict[str, Any]) -> None:
    target = runs_dir / pr_id
    target.mkdir(parents=True, exist_ok=True)
    (target / "codex.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_score_pr_skipped_when_artifact_missing(tmp_path: Path):
    pr = scorer._SamplePR(id="sample-pr-001", head_sha="deadbeefcafe1234")
    dataset = _dataset([_make_labeled()])
    result = scorer.score_pr(pr, dataset, runs_dir=tmp_path)
    assert result.status == "skipped"
    assert result.capture_status == "missing"
    assert "No Codex baseline artifact" in (result.notes or "")


def test_score_pr_skipped_when_capture_pending(tmp_path: Path):
    pr = scorer._SamplePR(id="sample-pr-001", head_sha="deadbeefcafe1234")
    _write_codex_artifact(
        tmp_path,
        "sample-pr-001",
        _codex_artifact(
            pr_id="sample-pr-001",
            head_sha="deadbeefcafe1234",
            capture_status="pending",
            notes="Live capture skipped: GITHUB_TOKEN not set",
        ),
    )
    dataset = _dataset([_make_labeled()])
    result = scorer.score_pr(pr, dataset, runs_dir=tmp_path)
    assert result.status == "skipped"
    assert result.capture_status == "pending"
    assert result.aggregate["total"] == 0


def test_score_pr_skipped_when_head_sha_drifts(tmp_path: Path):
    pr = scorer._SamplePR(id="sample-pr-001", head_sha="deadbeefcafe1234")
    _write_codex_artifact(
        tmp_path,
        "sample-pr-001",
        _codex_artifact(
            pr_id="sample-pr-001",
            head_sha="abcdabcdabcdabcd",
            capture_status="captured",
            findings=[_raw_finding()],
        ),
    )
    dataset = _dataset([_make_labeled()])
    result = scorer.score_pr(pr, dataset, runs_dir=tmp_path)
    assert result.status == "skipped"
    assert "registry pins" in (result.notes or "")


@pytest.mark.parametrize("captured_head", ["actual-pr-head", None, 0])
def test_merge_metadata_cannot_bypass_the_labeled_revision_guard(tmp_path: Path, captured_head):
    pr = scorer._SamplePR(id="sample-pr-001", head_sha="historical-merge")
    artifact = _codex_artifact(
        pr_id=pr.id, head_sha="actual-pr-head", capture_status="captured", findings=[_raw_finding()]
    )
    artifact["head_sha"] = captured_head
    artifact["merge_sha"] = pr.head_sha
    _write_codex_artifact(tmp_path, pr.id, artifact)
    result = scorer.score_pr(pr, _dataset([_make_labeled()]), runs_dir=tmp_path)
    assert result.status == "skipped"
    assert result.aggregate["total"] == 0


def test_score_pr_scores_captured_artifact_with_overlap_match(tmp_path: Path):
    pr = scorer._SamplePR(id="sample-pr-001", head_sha="deadbeefcafe1234")
    finding = _raw_finding(path="src/app.py", start=10, end=12)
    _write_codex_artifact(
        tmp_path,
        "sample-pr-001",
        _codex_artifact(
            pr_id="sample-pr-001",
            head_sha="deadbeefcafe1234",
            capture_status="captured",
            findings=[finding],
        ),
    )
    dataset = _dataset(
        [
            _make_labeled(
                id="gt-001-001",
                label="TP",
                path_hint="src/app.py",
                line_range=LineRange(10, 12),
            ),
        ]
    )
    result = scorer.score_pr(pr, dataset, runs_dir=tmp_path)
    assert result.status == "scored"
    assert result.aggregate["matched_tp"] == 1
    assert result.aggregate["matched_fp"] == 0
    assert result.aggregate["unmatched"] == 0
    row = result.findings_scored[0]
    assert row["assigned_label"] == "TP"
    assert row["match_quality"] == "exact_line"
    assert row["matched_dataset_id"] == "gt-001-001"
    assert row["dataset_entry"]["label"] == "TP"


def test_score_pr_marks_unmatched_findings_as_unmatched(tmp_path: Path):
    pr = scorer._SamplePR(id="sample-pr-001", head_sha="deadbeefcafe1234")
    finding = _raw_finding(path="src/elsewhere.py", start=100, end=110)
    _write_codex_artifact(
        tmp_path,
        "sample-pr-001",
        _codex_artifact(
            pr_id="sample-pr-001",
            head_sha="deadbeefcafe1234",
            capture_status="captured",
            findings=[finding],
        ),
    )
    dataset = _dataset(
        [
            _make_labeled(
                id="gt-001-001",
                label="TP",
                path_hint="src/app.py",
                line_range=LineRange(10, 12),
            ),
        ]
    )
    result = scorer.score_pr(pr, dataset, runs_dir=tmp_path)
    assert result.status == "scored"
    assert result.aggregate["unmatched"] == 1
    row = result.findings_scored[0]
    assert row["assigned_label"] == "UNMATCHED"
    assert row["matched_dataset_id"] is None
    assert row["dataset_entry"] is None


def test_score_pr_matches_fp_entry_when_finding_aligns(tmp_path: Path):
    pr = scorer._SamplePR(id="sample-pr-001", head_sha="deadbeefcafe1234")
    finding = _raw_finding(
        path="src/app.py",
        start=10,
        end=12,
        title="[P3] WADL hardening should include a test",
        body="suggest adding a test asserting 404",
    )
    _write_codex_artifact(
        tmp_path,
        "sample-pr-001",
        _codex_artifact(
            pr_id="sample-pr-001",
            head_sha="deadbeefcafe1234",
            capture_status="captured",
            findings=[finding],
        ),
    )
    dataset = _dataset(
        [
            _make_labeled(
                id="gt-001-fp-a",
                label="FP",
                path_hint="src/app.py",
                line_range=LineRange(10, 12),
                match_keywords=("WADL", "404"),
            ),
        ]
    )
    result = scorer.score_pr(pr, dataset, runs_dir=tmp_path)
    assert result.aggregate["matched_fp"] == 1
    row = result.findings_scored[0]
    assert row["assigned_label"] == "FP"


# ---------------------------------------------------------------------------
# Scored artifact serialization & summary
# ---------------------------------------------------------------------------


def test_write_scored_artifacts_round_trip(tmp_path: Path):
    pr = scorer._SamplePR(id="sample-pr-001", head_sha="deadbeefcafe1234")
    _write_codex_artifact(
        tmp_path,
        "sample-pr-001",
        _codex_artifact(
            pr_id="sample-pr-001",
            head_sha="deadbeefcafe1234",
            capture_status="captured",
            findings=[_raw_finding()],
        ),
    )
    dataset = _dataset(
        [
            _make_labeled(
                id="gt-001-001",
                label="TP",
                path_hint="src/app.py",
                line_range=LineRange(10, 12),
            ),
        ]
    )
    result = scorer.score_pr(pr, dataset, runs_dir=tmp_path)
    summary_target = tmp_path / "summary.json"
    written, summary_path = scorer.write_scored_artifacts(
        [result],
        runs_dir=tmp_path,
        summary_path=summary_target,
        generated_at="2026-05-06T12:00:00Z",
    )
    assert summary_path == summary_target
    assert len(written) == 1
    on_disk = json.loads(written[0].read_text(encoding="utf-8"))
    assert on_disk["pr_id"] == "sample-pr-001"
    assert on_disk["status"] == "scored"
    assert on_disk["aggregate"]["matched_tp"] == 1
    summary = json.loads(summary_target.read_text(encoding="utf-8"))
    assert summary["totals"]["scored_prs"] == 1
    assert summary["totals"]["total_matched_tp"] == 1
    assert summary["status_counts"] == {"scored": 1}


def test_check_scored_artifacts_detects_drift(tmp_path: Path):
    pr = scorer._SamplePR(id="sample-pr-001", head_sha="deadbeefcafe1234")
    _write_codex_artifact(
        tmp_path,
        "sample-pr-001",
        _codex_artifact(
            pr_id="sample-pr-001",
            head_sha="deadbeefcafe1234",
            capture_status="captured",
            findings=[_raw_finding()],
        ),
    )
    dataset = _dataset(
        [
            _make_labeled(
                id="gt-001-001",
                label="TP",
                path_hint="src/app.py",
                line_range=LineRange(10, 12),
            ),
        ]
    )
    result = scorer.score_pr(pr, dataset, runs_dir=tmp_path)
    summary_target = tmp_path / "summary.json"
    scorer.write_scored_artifacts(
        [result],
        runs_dir=tmp_path,
        summary_path=summary_target,
        generated_at="2026-05-06T12:00:00Z",
    )
    # Tamper: rewrite the on-disk artifact.
    target = scorer.scored_artifact_path("sample-pr-001", runs_dir=tmp_path)
    target.write_text(
        textwrap.dedent(
            """\
            {"schema_version": "1", "pr_id": "sample-pr-001", "tampered": true}
            """
        ),
        encoding="utf-8",
    )
    drift = scorer.check_scored_artifacts([result], runs_dir=tmp_path, summary_path=summary_target)
    assert drift, "drift list must not be empty after tampering"
    assert any("differs from rebuild" in d for d in drift)


def test_check_scored_artifacts_in_sync(tmp_path: Path):
    pr = scorer._SamplePR(id="sample-pr-001", head_sha="deadbeefcafe1234")
    _write_codex_artifact(
        tmp_path,
        "sample-pr-001",
        _codex_artifact(
            pr_id="sample-pr-001",
            head_sha="deadbeefcafe1234",
            capture_status="pending",
        ),
    )
    dataset = _dataset([_make_labeled()])
    result = scorer.score_pr(pr, dataset, runs_dir=tmp_path)
    summary_target = tmp_path / "summary.json"
    scorer.write_scored_artifacts(
        [result],
        runs_dir=tmp_path,
        summary_path=summary_target,
        generated_at="2026-05-06T12:00:00Z",
    )
    drift = scorer.check_scored_artifacts([result], runs_dir=tmp_path, summary_path=summary_target)
    assert drift == []


# ---------------------------------------------------------------------------
# CLI entrypoint smoke tests
# ---------------------------------------------------------------------------


def test_cli_check_mode_returns_zero_on_in_sync_state(monkeypatch, capsys, tmp_path):
    # Point the scorer at an isolated runs dir so we don't touch the
    # repo's checked-in artifacts.
    monkeypatch.setattr(scorer, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(scorer, "SUMMARY_PATH", tmp_path / "runs" / "_summary.json")
    monkeypatch.setattr(
        scorer,
        "_load_sample",
        lambda path=None: [scorer._SamplePR(id="sample-pr-001", head_sha="deadbeefcafe1234")],
    )
    monkeypatch.setattr(scorer, "load_dataset", lambda: _dataset([_make_labeled()]))

    rc = scorer.main([])
    assert rc == 0

    rc_check = scorer.main(["--check"])
    assert rc_check == 0
    captured = capsys.readouterr()
    assert "in sync" in captured.out


def test_cli_unknown_pr_id_exits_non_zero(monkeypatch, tmp_path):
    monkeypatch.setattr(scorer, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(scorer, "SUMMARY_PATH", tmp_path / "runs" / "_summary.json")
    monkeypatch.setattr(
        scorer,
        "_load_sample",
        lambda path=None: [scorer._SamplePR(id="sample-pr-001", head_sha="deadbeefcafe1234")],
    )
    monkeypatch.setattr(scorer, "load_dataset", lambda: _dataset([_make_labeled()]))

    with pytest.raises(SystemExit):
        scorer.main(["--pr-id", "no-such-pr"])


# ---------------------------------------------------------------------------
# End-to-end against the shipped repo state (smoke)
# ---------------------------------------------------------------------------


def test_real_repo_score_all_runs_against_pending_captures():
    """The shipped repo must score cleanly even with all captures pending.

    Methodology rule: pending captures are skipped, never synthesized.
    This test guards against accidental "score the placeholders as
    UNMATCHED" regressions that would inflate the parity report with
    zeros.
    """
    results = scorer.score_all()
    assert results, "sample registry must not be empty"
    assert all(r.status == "skipped" for r in results), (
        "every Codex baseline capture is currently pending — every score "
        "must be skipped, never synthesized. If a capture has landed live, "
        "this test should be relaxed to allow status=='scored' too."
    )
    # Every PR must still be cross-referenced to ground-truth coverage
    # so the summary stays meaningful before captures land.
    assert all(r.dataset_coverage["dataset_total"] >= 1 for r in results), (
        "dataset coverage must surface ≥ 1 ground-truth entry per PR"
    )
