from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from cli.review.artifacts import render_review_run_reference

REPORTER = Path("scripts/review-status.cjs").resolve()


def _report(
    tmp_path: Path,
    *,
    status: str = "failed",
    prior_run: str = "99",
    prior_attempt: str = "1",
    head: str = "head",
    author: str = "github-actions[bot]",
    legacy: bool = False,
) -> dict[str, Any]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to execute the action's status reporter")
    previous = json.dumps({"run_id": prior_run, "attempt": prior_attempt, "head_sha": "head"})
    comment = {
        "id": 7,
        "user": {"login": author},
        "body": (
            "dotbot code review:\nOverall: patch is correct"
            if legacy
            else f"dotbot code review:\n<!-- dotbot-review-run {previous} -->\nOverall: patch is correct"
        ),
    }
    script = tmp_path / "report.cjs"
    script.write_text(
        "const report = require(" + json.dumps(str(REPORTER)) + ");\n"
        "const comments = " + json.dumps([comment]) + ";\n"
        "const mutations = [];\n"
        "const github = {\n"
        " paginate: async () => comments,\n"
        " rest: {\n"
        "  pulls: {get: async () => ({data: {head: {sha: " + json.dumps(head) + "}}})},\n"
        "  issues: {\n"
        "   listComments: () => {},\n"
        "   updateComment: async (args) => mutations.push(['update', args]),\n"
        "   createComment: async (args) => mutations.push(['create', args]),\n"
        "  },\n"
        " },\n"
        "};\n"
        "const context = {runId: 100, repo: {owner: 'o', repo: 'r'},\n"
        " payload: {pull_request: {number: 14, head: {sha: 'head'}}}};\n"
        "report({github, context, core: {info: () => {}, warning: () => {}}, status: "
        + json.dumps(status)
        + "}).then(() => console.log(JSON.stringify({mutations})))\n"
        ".catch((error) => {console.error(error); process.exitCode = 1;});\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [node, str(script)],
        env={**os.environ, "GITHUB_RUN_ATTEMPT": "2", "GITHUB_SERVER_URL": "https://example.test"},
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize("status", ["in_progress", "failed", "cancelled"])
def test_status_replaces_stale_verdict_with_attempt_link(tmp_path: Path, status: str) -> None:
    result = _report(tmp_path, status=status)
    assert len(result["mutations"]) == 1
    verb, payload = result["mutations"][0]
    assert verb == "update"
    assert payload["comment_id"] == 7
    body = payload["body"]
    assert "patch is correct" not in body
    assert "no completed verdict" in body
    assert "No automatic approval" in body
    assert "https://example.test/o/r/actions/runs/100/attempts/2" in body
    assert f"Review {status.replace('_', ' ')}" in body


@pytest.mark.parametrize(("prior_run", "prior_attempt"), [("101", "1"), ("100", "3")])
def test_old_run_cannot_overwrite_newer_run_or_attempt(
    tmp_path: Path, prior_run: str, prior_attempt: str
) -> None:
    assert not _report(tmp_path, prior_run=prior_run, prior_attempt=prior_attempt)["mutations"]


def test_legacy_success_summary_is_replaced_on_failure(tmp_path: Path) -> None:
    body = _report(tmp_path, legacy=True)["mutations"][0][1]["body"]
    assert "Review failed" in body
    assert "patch is correct" not in body


def test_failure_replaces_its_own_attempts_completed_verdict(tmp_path: Path) -> None:
    body = _report(tmp_path, prior_run="100", prior_attempt="2")["mutations"][0][1]["body"]
    assert "Review failed" in body
    assert "no completed verdict" in body


def test_cancelled_old_head_cannot_overwrite_current_head_status(tmp_path: Path) -> None:
    assert not _report(tmp_path, status="cancelled", head="new-head")["mutations"]


def test_reporter_does_not_edit_human_comments(tmp_path: Path) -> None:
    result = _report(tmp_path, author="human")
    assert result["mutations"][0][0] == "create"


def test_successful_summary_records_same_run_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_RUN_ID", "100")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    monkeypatch.setenv("GITHUB_SERVER_URL", "https://example.test")
    rendered = render_review_run_reference("head")
    marker = rendered.splitlines()[0]
    metadata = json.loads(marker.removeprefix("<!-- dotbot-review-run ").removesuffix(" -->"))
    assert metadata == {"run_id": "100", "attempt": "2", "head_sha": "head"}
    assert "https://example.test/o/r/actions/runs/100/attempts/2" in rendered
    monkeypatch.delenv("GITHUB_RUN_ID")
    assert render_review_run_reference("head") == ""


def test_action_reports_failures_even_before_cli_and_skips_act_and_dry_run() -> None:
    steps = yaml.safe_load(Path("action.yml").read_text(encoding="utf-8"))["runs"]["steps"]
    start, end = steps[0], steps[-1]
    assert start["name"] == "Mark review in progress"
    for step in (start, end):
        assert "inputs.mode == 'review'" in step["if"]
        assert "inputs.dry_run != '1'" in step["if"]
        assert step["uses"] == "actions/github-script@v7"
        assert step["env"]["DOTBOT_ACTION_PATH"] == "${{ github.action_path }}"
    assert "always()" in end["if"]
    assert "failure() || cancelled()" in end["if"]
