"""Smoke tests for the Codex baseline runner (Sub-AC 1.2).

These tests pin the artifact contract: shape of the per-PR JSON
envelope, idempotency of placeholder writes, correct handling of
selection by ``--pr-id``, and the read-only SDK capture seam. They
stub the existing ``CodexClient`` and never make a live model call.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from cli.core.models import REVIEW_OUTPUT_SCHEMA

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import run_codex_baseline as runner  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


SAMPLE_DOC = {
    "prs": [
        {
            "id": "fixture-pr-1",
            "repo": "owner/repo",
            "pr": 1,
            "head_sha": "a" * 40,  # legacy merge commit
            "pr_head_sha": "deadbeef" * 5,  # actual PR head (40-char SHA)
            "pr_base_sha": "b" * 40,
            "web_search_mode_for_run": "disabled",
            "reasoning_effort_for_run": "medium",
        },
        {
            "id": "fixture-pr-2",
            "repo": "owner/repo",
            "pr": 2,
            "head_sha": "c" * 40,  # legacy merge commit
            "pr_head_sha": "feedface" * 5,
            "pr_base_sha": "d" * 40,
            "web_search_mode_for_run": "live",
            "reasoning_effort_for_run": "high",
        },
    ]
}


@pytest.fixture
def isolated_runs(tmp_path, monkeypatch):
    """Redirect the runner to a tmp sample registry + tmp runs/ dir.

    Keeps the real ``eval/runs/`` artifacts untouched while the smoke
    test exercises the placeholder-write path.
    """
    sample_path = tmp_path / "sample-prs.yaml"
    sample_path.write_text(yaml.safe_dump(SAMPLE_DOC, sort_keys=False))
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    monkeypatch.setattr(runner, "SAMPLE_PATH", sample_path)
    monkeypatch.setattr(runner, "RUNS_DIR", runs_dir)
    # Strip env so live capture path doesn't accidentally fire.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    return tmp_path


# ---------------------------------------------------------------------------
# Tests — sample loading
# ---------------------------------------------------------------------------


def test_load_sample_parses_pinned_metadata(isolated_runs):
    sample = runner.load_sample()
    assert [pr.id for pr in sample] == ["fixture-pr-1", "fixture-pr-2"]
    pr1 = sample[0]
    assert pr1.repo == "owner/repo"
    assert pr1.pr == 1
    # `head_sha` is the actual PR head from `pr_head_sha`; the legacy
    # `head_sha` merge commit is carried on `merge_sha`.
    assert pr1.head_sha == "deadbeef" * 5
    assert pr1.base_sha == "b" * 40
    assert pr1.merge_sha == "a" * 40
    assert pr1.web_search_mode == "disabled"
    assert pr1.reasoning_effort == "medium"
    # Default model pinned to action.yml's prior baseline so re-runs
    # against a fresh registry stay reproducible.
    assert pr1.model == runner.DEFAULT_BASELINE_MODEL


def test_load_sample_rejects_non_mapping_top_level(tmp_path, monkeypatch):
    bad = tmp_path / "sample.yaml"
    bad.write_text("- 1\n- 2\n")
    monkeypatch.setattr(runner, "SAMPLE_PATH", bad)
    with pytest.raises(SystemExit):
        runner.load_sample()


def test_load_sample_requires_pinned_pr_revision_fields(tmp_path, monkeypatch):
    # A row without pr_head_sha/pr_base_sha must fail loudly rather than
    # silently fall back to the legacy merge sha.
    doc = {
        "prs": [
            {
                "id": "x",
                "repo": "owner/repo",
                "pr": 1,
                "head_sha": "a" * 40,
                "web_search_mode_for_run": "disabled",
                "reasoning_effort_for_run": "medium",
            }
        ]
    }
    bad = tmp_path / "sample.yaml"
    bad.write_text(yaml.safe_dump(doc, sort_keys=False))
    monkeypatch.setattr(runner, "SAMPLE_PATH", bad)
    with pytest.raises(SystemExit, match="exact commit SHA"):
        runner.load_sample()


def test_canonical_registry_pins_real_pr_head_distinct_from_legacy_merge():
    # Regression: the shipped registry's legacy `head_sha` is the merge
    # commit and differs from the actual PR head, and every row carries a
    # base revision that exists as a commit during live capture.
    sample = runner.load_sample()
    assert sample, "canonical registry must not be empty"
    for pr in sample:
        assert pr.merge_sha is not None
        assert len(pr.merge_sha) == 40 and int(pr.merge_sha, 16) >= 0
        assert len(pr.head_sha) == 40 and int(pr.head_sha, 16) >= 0
        assert len(pr.base_sha) == 40 and int(pr.base_sha, 16) >= 0
        assert pr.head_sha != pr.merge_sha, (
            f"{pr.id}: pr_head_sha must differ from the legacy merge head_sha"
        )


# ---------------------------------------------------------------------------
# Tests — placeholder write contract
# ---------------------------------------------------------------------------


def _read_artifact(pr_id: str) -> dict:
    return json.loads((runner.RUNS_DIR / pr_id / "codex.json").read_text())


def test_dry_run_only_writes_pending_placeholder_for_every_pr(isolated_runs):
    rc = runner.main(["--all", "--dry-run-only"])
    assert rc == 0

    for pr_id in ("fixture-pr-1", "fixture-pr-2"):
        artifact = _read_artifact(pr_id)
        # Required top-level keys per methodology §4.2.
        assert artifact["pr_id"] == pr_id
        assert artifact["run"] == "codex"
        assert artifact["base_sha"]
        assert artifact["merge_sha"]
        assert artifact["model"] == runner.DEFAULT_BASELINE_MODEL
        assert artifact["prior_review_state"] is None
        # Placeholder has no review payload yet — downstream scoring
        # MUST treat this as not-yet-measurable.
        assert artifact["review_run_result"] is None
        assert artifact["capture"]["status"] == "pending"
        assert artifact["capture"]["runner_version"] == runner.RUNNER_VERSION
        # Posting telemetry shape matches AC 3's PostingOutcome dataclass —
        # downstream code that propagates batch_submitted /
        # per_comment_fallback / skipped_after_422 must find them present.
        outcome = artifact["posted"]["posting_outcome"]
        assert set(outcome) == {
            "batch_submitted",
            "per_comment_fallback",
            "skipped_after_422",
        }


def test_pending_placeholder_when_live_env_missing(isolated_runs):
    # Without OPENAI_API_KEY / codex module the runner MUST fall back
    # to writing a pending placeholder rather than silently skipping or
    # writing fabricated findings.
    rc = runner.main(["--all"])
    assert rc == 0
    artifact = _read_artifact("fixture-pr-1")
    assert artifact["capture"]["status"] == "pending"
    assert "OPENAI_API_KEY" in artifact["capture"]["notes"] or (
        "GITHUB_TOKEN" in artifact["capture"]["notes"]
    )


def test_idempotent_skips_already_captured(isolated_runs, monkeypatch):
    # Seed an already-captured artifact for fixture-pr-1.
    sample = runner.load_sample()
    pr1 = sample[0]
    runner.write_artifact(
        pr1,
        capture_status="captured",
        review_run_result={
            "findings": [],
            "carried_forward": [],
            "overall_correctness": "patch is correct",
            "overall_explanation": "no issues found",
            "overall_confidence_score": 0.9,
        },
        posted=None,
        notes="seeded by test",
    )

    # Spy on capture_one to confirm it short-circuits on captured PRs.
    captured_calls: list[str] = []
    real_capture = runner.capture_one

    def _spy(pr, *, dry_run_only, force):
        captured_calls.append(pr.id)
        return real_capture(pr, dry_run_only=dry_run_only, force=force)

    monkeypatch.setattr(runner, "capture_one", _spy)
    rc = runner.main(["--all", "--dry-run-only"])
    assert rc == 0
    assert "fixture-pr-1" in captured_calls  # called …
    # … but the captured artifact is unchanged.
    artifact = _read_artifact("fixture-pr-1")
    assert artifact["capture"]["status"] == "captured"
    assert artifact["capture"]["notes"] == "seeded by test"


def test_force_reruns_even_when_captured(isolated_runs):
    sample = runner.load_sample()
    pr1 = sample[0]
    runner.write_artifact(
        pr1,
        capture_status="captured",
        review_run_result={
            "findings": [],
            "carried_forward": [],
            "overall_correctness": "patch is correct",
            "overall_explanation": "no issues found",
            "overall_confidence_score": 0.9,
        },
        posted=None,
        notes="seeded by test",
    )
    rc = runner.main(["--pr-id", "fixture-pr-1", "--dry-run-only", "--force"])
    assert rc == 0
    artifact = _read_artifact("fixture-pr-1")
    assert artifact["capture"]["status"] == "pending"
    assert "Placeholder written by --dry-run-only" in artifact["capture"]["notes"]


def test_unknown_pr_id_errors(isolated_runs):
    with pytest.raises(SystemExit):
        runner.main(["--pr-id", "not-a-real-id", "--dry-run-only"])


# ---------------------------------------------------------------------------
# Tests — live SDK capture (stubbed CodexClient, no model calls)
# ---------------------------------------------------------------------------


def _sdk_result() -> dict[str, Any]:
    return {
        "findings": [],
        "carried_forward": [],
        "overall_correctness": "patch is correct",
        "overall_explanation": "no issues found",
        "overall_confidence_score": 0.9,
    }


class _FakeCodexClient:
    last_config: Any = None
    last_call: dict[str, Any] = {}

    def __init__(self, config: Any) -> None:
        _FakeCodexClient.last_config = config

    def execute_structured(self, prompt: str, **kwargs: Any) -> str:
        _FakeCodexClient.last_call = {"prompt": prompt, **kwargs}
        return json.dumps(_sdk_result())


def _fixture_pr() -> runner.SamplePR:
    return runner.SamplePR(
        id="fixture-pr-1",
        repo="owner/repo",
        pr=1,
        head_sha="deadbeef" * 5,
        base_sha="b" * 40,
        web_search_mode="disabled",
        reasoning_effort="medium",
        merge_sha="a" * 40,
    )


def test_invoke_codex_sdk_uses_read_only_structured_review(monkeypatch, tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setattr("cli.clients.codex_client.CodexClient", _FakeCodexClient)
    monkeypatch.setattr(
        "cli.clients.github_client.GitHubClient",
        lambda _: SimpleNamespace(
            get_pr=lambda n: SimpleNamespace(
                head=SimpleNamespace(sha="deadbeef" * 5), base=SimpleNamespace(sha="b" * 40)
            )
        ),
    )
    monkeypatch.setattr(
        runner.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0)
    )

    pr = _fixture_pr()
    out = runner._invoke_codex_sdk(pr, checkout)

    assert out == _sdk_result()
    assert _FakeCodexClient.last_call["sandbox_mode"] == "read-only"
    assert _FakeCodexClient.last_call["output_schema"] is REVIEW_OUTPUT_SCHEMA
    prompt = _FakeCodexClient.last_call["prompt"]
    assert "<repository>owner/repo</repository>" in prompt
    assert pr.head_sha in prompt
    assert "Review guidelines:" in prompt
    assert f"<base_sha>{'b' * 40}</base_sha>" in prompt
    assert f"git diff {'b' * 40}...{pr.head_sha}" in prompt
    assert _FakeCodexClient.last_config.mode == "review"
    assert _FakeCodexClient.last_config.openai_api_key == "test-key"


@pytest.mark.parametrize(
    ("head", "base"),
    [
        (None, "b" * 40),
        ("different-head", "b" * 40),
        ("deadbeef" * 5, None),
        # Negative case: the live PR base must equal the pinned base.
        ("deadbeef" * 5, "different-base"),
    ],
)
def test_sdk_capture_rejects_missing_or_changed_revision_before_execution(
    monkeypatch, tmp_path, head, base
):
    monkeypatch.setattr(
        "cli.clients.github_client.GitHubClient",
        lambda _: SimpleNamespace(
            get_pr=lambda n: SimpleNamespace(
                head=SimpleNamespace(sha=head) if head else None,
                base=SimpleNamespace(sha=base) if base else None,
            )
        ),
    )
    monkeypatch.setattr(
        "cli.clients.codex_client.CodexClient",
        lambda _: pytest.fail("must not execute an unpinned benchmark"),
    )
    with pytest.raises(RuntimeError, match="Benchmark PR"):
        runner._invoke_codex_sdk(_fixture_pr(), tmp_path)


def test_live_capture_writes_review_run_result(isolated_runs, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setattr(runner, "_have_codex_module", lambda: True)
    monkeypatch.setattr(
        runner, "_checkout_pr", lambda pr, dest, **kw: dest.mkdir(parents=True, exist_ok=True)
    )
    monkeypatch.setattr(runner, "_invoke_codex_sdk", lambda pr, repo_path: _sdk_result())

    rc = runner.main(["--pr-id", "fixture-pr-1"])
    assert rc == 0

    artifact = _read_artifact("fixture-pr-1")
    assert artifact["capture"]["status"] == "captured"
    assert artifact["review_run_result"] == _sdk_result()
    # Capture never posts or approves.
    assert artifact["posted"]["inline_ids"] == []
    assert artifact["posted"]["summary_id"] is None


# ---------------------------------------------------------------------------
# Tests — checkout history (base + merge-base evidence for the agent)
# ---------------------------------------------------------------------------


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def test_checkout_pr_fetches_full_history_for_base_evidence(tmp_path):
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", str(origin)], check=True, capture_output=True)

    seed = tmp_path / "seed"
    seed.mkdir()
    _git(["init"], seed)
    _git(["config", "user.email", "test@example.com"], seed)
    _git(["config", "user.name", "Test"], seed)
    (seed / "base.txt").write_text("base\n")
    _git(["add", "."], seed)
    _git(["commit", "-m", "base"], seed)
    base_sha = _git(["rev-parse", "HEAD"], seed).stdout.strip()
    _git(["branch", "-M", "main"], seed)
    _git(["remote", "add", "origin", str(origin)], seed)
    _git(["push", "-u", "origin", "main"], seed)

    _git(["checkout", "-b", "feature"], seed)
    (seed / "head.txt").write_text("head\n")
    _git(["add", "."], seed)
    _git(["commit", "-m", "head"], seed)
    head_sha = _git(["rev-parse", "HEAD"], seed).stdout.strip()
    _git(["push", "-u", "origin", "feature"], seed)

    pr = _fixture_pr()
    pr = runner.SamplePR(
        id=pr.id,
        repo=pr.repo,
        pr=pr.pr,
        head_sha=head_sha,
        base_sha=base_sha,
        web_search_mode=pr.web_search_mode,
        reasoning_effort=pr.reasoning_effort,
    )
    dest = tmp_path / "checkout"
    runner._checkout_pr(pr, dest, repo_url=str(origin))

    assert _git(["rev-parse", "HEAD"], dest).stdout.strip() == head_sha
    merge_base = subprocess.run(
        ["git", "-C", str(dest), "merge-base", base_sha, head_sha],
        capture_output=True,
        text=True,
    )
    assert merge_base.returncode == 0
    assert merge_base.stdout.strip() == base_sha
    assert _git(["cat-file", "-t", base_sha], dest).stdout.strip() == "commit"
