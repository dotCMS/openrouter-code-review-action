"""Pin the Sub-AC 1.2 baseline PR fixture contract.

Every PR in ``evaluation/bakeoff/sample-prs.yml`` must ship with a
fixture directory under ``tests/fixtures/baseline/<pr-id>/`` containing:

* ``diff.patch`` — unified diff captured from ``gh pr diff``
* ``metadata.json`` — ``gh pr view --json`` payload

A machine-readable rollup at ``tests/fixtures/baseline/_index.json``
indexes the lot and records the SHA-256 + byte size of each diff so a
silently-corrupted fixture surfaces as a test failure rather than a
mysterious bake-off regression.

The fixture contract these tests pin:

1. One fixture dir exists for every PR in the sample registry.
2. Each fixture exposes the four metadata fields Sub-AC 1.2 requires —
   PR number, base SHA, head SHA, changed files.
3. The metadata's ``mergeCommit.oid`` matches the
   ``merge_commit`` pinned in ``sample-prs.yml`` (so a force-pushed
   merge commit is caught at fixture-regen time, not at review time).
4. ``_index.json`` is consistent with the on-disk diff bytes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = REPO_ROOT / "evaluation" / "bakeoff" / "sample-prs.yml"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "baseline"
INDEX_PATH = FIXTURES_DIR / "_index.json"


def _load_sample_prs() -> list[dict]:
    data = yaml.safe_load(SAMPLE_PATH.read_text(encoding="utf-8"))
    return list(data["prs"])


def _load_index() -> dict:
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def sample_prs() -> list[dict]:
    return _load_sample_prs()


@pytest.fixture(scope="module")
def index() -> dict:
    return _load_index()


def test_index_exists_and_covers_sample(sample_prs: list[dict], index: dict) -> None:
    sample_numbers = sorted(p["number"] for p in sample_prs)
    indexed_numbers = sorted(p["number"] for p in index["prs"])
    assert sample_numbers == indexed_numbers, (
        "sample-prs.yml and tests/fixtures/baseline/_index.json drifted; "
        "regenerate the fixtures (see baseline/README.md)"
    )
    assert index["count"] == len(sample_prs)
    assert index["repo"] == "dotCMS/core"


@pytest.mark.parametrize("pr_number", sorted(p["number"] for p in _load_sample_prs()))
def test_fixture_dir_present(pr_number: int) -> None:
    pr_id = f"pr-{pr_number}"
    pr_dir = FIXTURES_DIR / pr_id
    assert (pr_dir / "diff.patch").is_file(), f"missing diff for {pr_id}"
    assert (pr_dir / "metadata.json").is_file(), f"missing metadata for {pr_id}"


@pytest.mark.parametrize("pr_number", sorted(p["number"] for p in _load_sample_prs()))
def test_metadata_carries_required_fields(pr_number: int) -> None:
    """Sub-AC 1.2 requires PR number, base/head SHAs, changed files."""
    pr_id = f"pr-{pr_number}"
    meta = json.loads((FIXTURES_DIR / pr_id / "metadata.json").read_text(encoding="utf-8"))
    assert meta["number"] == pr_number
    # Base and head SHAs from the PR branch tips.
    assert isinstance(meta["baseRefOid"], str) and len(meta["baseRefOid"]) == 40
    assert isinstance(meta["headRefOid"], str) and len(meta["headRefOid"]) == 40
    # Merge commit SHA — the SHA reviews replay against.
    assert isinstance(meta["mergeCommit"]["oid"], str)
    assert len(meta["mergeCommit"]["oid"]) == 40
    # Changed files — list[{path, additions, deletions}].
    assert isinstance(meta["files"], list) and meta["files"], "no changed files"
    assert meta["changedFiles"] == len(meta["files"])
    for f in meta["files"]:
        assert isinstance(f["path"], str) and f["path"]
        assert isinstance(f["additions"], int)
        assert isinstance(f["deletions"], int)


def test_merge_commit_matches_sample(sample_prs: list[dict]) -> None:
    """A force-push that breaks ancestry should be caught at fixture
    regen, not at review time."""
    for pr in sample_prs:
        pr_id = f"pr-{pr['number']}"
        meta = json.loads((FIXTURES_DIR / pr_id / "metadata.json").read_text(encoding="utf-8"))
        assert meta["mergeCommit"]["oid"] == pr["merge_commit"], (
            f"{pr_id}: sample-prs.yml merge_commit drifted from on-disk metadata"
        )


def test_index_diff_hashes_match_disk(index: dict) -> None:
    for entry in index["prs"]:
        diff_path = FIXTURES_DIR / entry["diff_path"]
        data = diff_path.read_bytes()
        assert len(data) == entry["diff_bytes"], f"{entry['pr_id']}: byte mismatch"
        digest = hashlib.sha256(data).hexdigest()
        assert digest == entry["diff_sha256"], (
            f"{entry['pr_id']}: diff.patch hash drift; regenerate fixtures"
        )


def test_diff_starts_with_diff_header() -> None:
    """A diff captured from ``gh pr diff`` always starts with
    ``diff --git`` (or is empty for a no-op PR, which we don't sample).
    Catches accidental writes of HTML error pages or empty stdout."""
    for pr_dir in sorted(FIXTURES_DIR.glob("pr-*")):
        diff_text = (pr_dir / "diff.patch").read_text(encoding="utf-8", errors="replace")
        assert diff_text.startswith("diff --git "), (
            f"{pr_dir.name}: diff.patch is not a unified diff"
        )


# ---------------------------------------------------------------------------
# Sub-AC 1.3 — Codex baseline mirror contract
# ---------------------------------------------------------------------------
#
# Every PR fixture must ship with a ``codex-findings.json`` mirror of
# the corresponding ``eval/runs/dotcms-core-<NUMBER>/codex.json`` artifact.
# The mirror is produced by ``python -m eval.mirror_codex_baseline`` and
# preserves the upstream JSON shape so parity scoring can read either
# location.


_CODEX_MIRROR_REQUIRED_KEYS = frozenset(
    {
        "pr_id",
        "run",
        "head_sha",
        "model",
        "reasoning_effort",
        "web_search_mode",
        "capture",
        "review_run_result",
        "posted",
        "source",
    }
)


@pytest.mark.parametrize("pr_number", sorted(p["number"] for p in _load_sample_prs()))
def test_codex_findings_mirror_present(pr_number: int) -> None:
    """Sub-AC 1.3: codex-findings.json must exist for every fixture PR."""
    pr_id = f"pr-{pr_number}"
    findings = FIXTURES_DIR / pr_id / "codex-findings.json"
    assert findings.is_file(), (
        f"missing {findings.relative_to(REPO_ROOT)}; "
        "regenerate via `python -m eval.mirror_codex_baseline`"
    )
    payload = json.loads(findings.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    missing = _CODEX_MIRROR_REQUIRED_KEYS - payload.keys()
    assert not missing, f"{pr_id}: codex-findings.json missing keys {sorted(missing)}"
    assert payload["pr_id"] == pr_id
    assert payload["run"] == "codex"
    capture = payload["capture"]
    assert isinstance(capture, dict)
    assert capture["status"] in {"captured", "pending", "failed"}
    assert capture["runner_version"] in {"1", "2"}
    source = payload["source"]
    assert isinstance(source, dict)
    assert source["produced_by"] == "eval.run_codex_baseline"
    assert source["mirrored_by"] == "eval.mirror_codex_baseline"
    expected_source = f"eval/runs/dotcms-core-{pr_number}/codex.json"
    assert source["eval_artifact"] == expected_source


def test_codex_findings_mirror_in_sync() -> None:
    """The fixture mirror must match the upstream eval artifact byte-for-byte
    after id translation. Drift here means somebody edited the mirror by
    hand or forgot to re-run ``eval.mirror_codex_baseline``."""
    from eval.mirror_codex_baseline import main as mirror_main  # noqa: PLC0415

    rc = mirror_main(["--check"])
    assert rc == 0, (
        "codex-findings.json mirror drifted from eval/runs/<id>/codex.json; "
        "run `python -m eval.mirror_codex_baseline` to refresh"
    )
