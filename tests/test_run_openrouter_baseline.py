"""Regression tests for the OpenRouter production-path baseline runner.

These pin the two integration seams that PR14 broke:

* ``_checkout_pr`` must fetch complete history so the panel's
  ``git diff <base>...<head>`` can resolve the PR base and merge base.
* ``_extract_review_run_result`` must take the consolidated
  ``[review-result]`` aggregate, never a streamed ``[review-panel]`` turn.

They also pin the trusted-source CLI invocation. No live OpenRouter or
GitHub access is required.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import run_openrouter_baseline as runner  # noqa: E402

SAMPLE_DOC = {
    "prs": [
        {
            "id": "fixture-pr-1",
            "repo": "owner/repo",
            "pr": 1,
            "head_sha": "a" * 40,
            "pr_head_sha": "deadbeef" * 5,
            "pr_base_sha": "b" * 40,
            "web_search_mode_for_run": "disabled",
            "reasoning_effort_for_run": "medium",
        }
    ]
}


@pytest.fixture
def isolated_runs(tmp_path, monkeypatch):
    sample_path = tmp_path / "sample-prs.yaml"
    sample_path.write_text(yaml.safe_dump(SAMPLE_DOC, sort_keys=False))
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    monkeypatch.setattr(runner, "SAMPLE_PATH", sample_path)
    monkeypatch.setattr(runner, "RUNS_DIR", runs_dir)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    return tmp_path


def _result(
    correctness: str = "patch is correct",
    findings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "overall_correctness": correctness,
        "overall_explanation": f"verdict: {correctness}",
        "overall_confidence_score": 0.8,
        "findings": findings or [],
        "carried_forward": [],
    }


def _panel_turn(correctness: str = "patch is incorrect") -> dict[str, Any]:
    # A streamed panel turn carries the same result keys as the final
    # aggregate plus panel-only fields, which is exactly why the old
    # scanner could mistake it for the result.
    return {
        **_result(correctness, findings=[]),
        "message": "intermediate position",
        "security_summary": "none",
        "dispositions": {},
    }


# ---------------------------------------------------------------------------
# Artifact / placeholder contract (kept in lockstep with the Codex runner)
# ---------------------------------------------------------------------------


def test_dry_run_only_writes_pending_placeholder(isolated_runs):
    rc = runner.main(["--all", "--dry-run-only"])
    assert rc == 0
    artifact = json.loads((runner.RUNS_DIR / "fixture-pr-1" / "openrouter.json").read_text())
    assert artifact["run"] == "openrouter"
    assert artifact["capture"]["status"] == "pending"
    assert artifact["review_run_result"] is None
    assert set(artifact["posted"]["posting_outcome"]) == {
        "batch_submitted",
        "per_comment_fallback",
        "skipped_after_422",
    }


def test_pending_placeholder_when_live_env_missing(isolated_runs):
    rc = runner.main(["--all"])
    assert rc == 0
    artifact = json.loads((runner.RUNS_DIR / "fixture-pr-1" / "openrouter.json").read_text())
    assert artifact["capture"]["status"] == "pending"
    assert (
        "OPENROUTER_API_KEY" in artifact["capture"]["notes"]
        or "GITHUB_TOKEN" in artifact["capture"]["notes"]
    )


# ---------------------------------------------------------------------------
# Result extraction — the [review-result] marker contract
# ---------------------------------------------------------------------------


def test_both_runners_load_the_same_verified_pr_revisions():
    from eval import run_codex_baseline as codex_runner

    path = ROOT / "eval/sample-prs.yaml"
    raw = yaml.safe_load(path.read_text())["prs"]
    for item, opened, sdk in zip(
        raw, runner.load_sample(path), codex_runner.load_sample(path), strict=True
    ):
        assert opened.head_sha == sdk.head_sha == item["pr_head_sha"]
        assert opened.base_sha == sdk.base_sha == item["pr_base_sha"]
        assert opened.merge_sha == sdk.merge_sha == item["head_sha"]
        assert opened.head_sha != opened.merge_sha


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pr_head_sha", "refs/heads/main"),
        ("pr_base_sha", "--upload-pack=evil"),
        ("pr_base_sha", None),
        ("pr_head_sha", "1" * 40 + "\n"),
    ],
)
@pytest.mark.parametrize("backend", ["openrouter", "codex"])
def test_registry_requires_exact_pinned_revisions(tmp_path, field, value, backend):
    from eval import run_codex_baseline as codex_runner

    doc = json.loads(json.dumps(SAMPLE_DOC))
    doc["prs"][0][field] = value
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(doc))
    loader = runner.load_sample if backend == "openrouter" else codex_runner.load_sample
    with pytest.raises(SystemExit, match="exact commit SHA"):
        loader(path)


def test_extract_uses_final_marker_over_intermediate_incorrect_turns():
    final = _result("patch is correct")
    stdout = (
        f"[review-panel] {json.dumps(_panel_turn('patch is incorrect'))}\n"
        f"[review-panel] {json.dumps(_panel_turn('patch is incorrect'))}\n"
        f"[review-result] {json.dumps(final)}\n"
        "Review completed: patch is correct, 0 findings\n"
    )
    assert runner._extract_review_run_result(stdout) == final


def test_extract_uses_final_marker_over_intermediate_correct_turns():
    final = _result("patch is incorrect", findings=[])
    stdout = (
        f"[review-panel] {json.dumps(_panel_turn('patch is correct'))}\n"
        f"[review-result] {json.dumps(final)}\n"
    )
    assert runner._extract_review_run_result(stdout) == final


def test_panel_output_without_final_marker_is_rejected():
    stdout = f"[review-panel] {json.dumps(_panel_turn())}\n"
    with pytest.raises(RuntimeError, match="missing the final"):
        runner._extract_review_run_result(stdout)


def test_marker_text_inside_panel_finding_body_is_not_the_result():
    turn = _panel_turn()
    turn["findings"] = [
        {
            "title": "[P2] mention",
            "body": "body mentions [review-result] but is not one",
            "confidence_score": 0.5,
            "priority": 2,
            "code_location": {
                "absolute_file_path": "foo.py",
                "line_range": {"start": 1, "end": 1},
            },
        }
    ]
    stdout = f"[review-panel] {json.dumps(turn)}\n"
    with pytest.raises(RuntimeError, match="missing the final"):
        runner._extract_review_run_result(stdout)


def test_untagged_json_is_not_accepted_as_a_final_result():
    stdout = json.dumps(_result("patch is correct"))
    with pytest.raises(RuntimeError, match="missing the final"):
        runner._extract_review_run_result(stdout)


def test_last_final_marker_wins_over_an_earlier_preview_marker():
    final = _result("patch is incorrect")
    stdout = (
        "[review-result] "
        + json.dumps(_result("patch is correct"))
        + "\n[review-result] "
        + json.dumps(final)
        + "\n"
    )
    assert runner._extract_review_run_result(stdout) == final


def test_malformed_json_after_marker_raises():
    with pytest.raises(RuntimeError, match="malformed JSON"):
        runner._extract_review_run_result("[review-result] {not json}\n")


# ---------------------------------------------------------------------------
# Checkout — real git history proves base + merge base are present
# ---------------------------------------------------------------------------


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _rev_parse(cwd: Path, rev: str) -> str:
    return _git(["rev-parse", rev], cwd).stdout.strip()


def test_checkout_pr_fetches_full_history_so_base_and_merge_base_exist(tmp_path):
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
    base_sha = _rev_parse(seed, "HEAD")
    _git(["branch", "-M", "main"], seed)
    _git(["remote", "add", "origin", str(origin)], seed)
    _git(["push", "-u", "origin", "main"], seed)

    _git(["checkout", "-b", "feature"], seed)
    (seed / "head.txt").write_text("head\n")
    _git(["add", "."], seed)
    _git(["commit", "-m", "head"], seed)
    head_sha = _rev_parse(seed, "HEAD")
    _git(["push", "-u", "origin", "feature"], seed)

    pr = runner.SamplePR(
        id="real-pr",
        repo="owner/repo",
        pr=1,
        head_sha=head_sha,
        base_sha=base_sha,
        web_search_mode="disabled",
        reasoning_effort="medium",
    )
    dest = tmp_path / "checkout"
    runner._checkout_pr(pr, dest, repo_url=str(origin))

    # The pinned head is checked out.
    assert _rev_parse(dest, "HEAD") == head_sha
    # The base commit is present and the merge base resolves.
    assert (
        subprocess.run(
            ["git", "-C", str(dest), "cat-file", "-e", base_sha],
            capture_output=True,
        ).returncode
        == 0
    )
    merge_base = subprocess.run(
        ["git", "-C", str(dest), "merge-base", base_sha, head_sha],
        capture_output=True,
        text=True,
    )
    assert merge_base.returncode == 0
    assert merge_base.stdout.strip() == base_sha
    # The panel's actual diff command succeeds against the checkout.
    diff = subprocess.run(
        ["git", "-C", str(dest), "diff", f"{base_sha}...{head_sha}", "--"],
        capture_output=True,
        text=True,
    )
    assert diff.returncode == 0
    assert "head.txt" in diff.stdout


# ---------------------------------------------------------------------------
# CLI invocation — trusted source, not the PR checkout
# ---------------------------------------------------------------------------


class _FakeProc:
    returncode = 0
    stdout = "[review-result] " + json.dumps(_result("patch is correct")) + "\n"
    stderr = ""


def test_invoke_review_cli_anchors_to_trusted_source(tmp_path, monkeypatch):
    captured: dict[str, Any] = {}
    monkeypatch.setenv("DOTBOT_REVIEW_STATUS_COMMENT_ID", "unrelated-action-comment")

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs["env"]
        captured["cwd"] = kwargs["cwd"]
        return _FakeProc()

    monkeypatch.setattr(runner.subprocess, "run", _fake_run)
    pr = runner.SamplePR(
        id="fixture-pr-1",
        repo="owner/repo",
        pr=1,
        head_sha="deadbeef" * 5,
        base_sha="b" * 40,
        web_search_mode="disabled",
        reasoning_effort="medium",
    )
    out = runner._invoke_review_cli(
        pr,
        tmp_path / "pr-checkout",
        review_model="anthropic/claude-opus-4.7",
        reasoning_effort="medium",
        web_search_mode="disabled",
    )
    assert out == _result("patch is correct")
    assert captured["env"]["PYTHONSAFEPATH"] == "1"
    assert captured["env"]["PYTHONPATH"].split(os.pathsep)[0] == str(runner.REPO_ROOT)
    assert captured["cwd"] == str(runner.REPO_ROOT)
    assert captured["env"]["DOTBOT_REVIEW_EXPECTED_HEAD_SHA"] == pr.head_sha
    assert captured["env"]["DOTBOT_REVIEW_EXPECTED_BASE_SHA"] == pr.base_sha
    assert "DOTBOT_REVIEW_STATUS_COMMENT_ID" not in captured["env"]
