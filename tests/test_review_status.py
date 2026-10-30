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
    newer_comment: bool = False,
) -> dict[str, Any]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to execute the action's status reporter")
    assert node is not None
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
    comments = [comment]
    if newer_comment:
        newer = json.dumps({"run_id": "101", "attempt": "1", "head_sha": "head"})
        comments.append(
            {
                **comment,
                "id": 8,
                "body": f"dotbot code review:\n<!-- dotbot-review-run {newer} -->\nReview in progress",
            }
        )
    script = tmp_path / "report.cjs"
    script.write_text(
        "const report = require(" + json.dumps(str(REPORTER)) + ");\n"
        "const comments = " + json.dumps(comments) + ";\n"
        "const mutations = [];\n"
        "const github = {\n"
        " paginate: async () => comments,\n"
        " rest: {\n"
        "  pulls: {get: async () => ({data: {head: {sha: " + json.dumps(head) + "}}})},\n"
        "  issues: {\n"
        "   listComments: () => {},\n"
        "   updateComment: async (args) => {mutations.push(['update', args]); return {data: {id: args.comment_id}};},\n"
        "   createComment: async (args) => {mutations.push(['create', args]); return {data: {id: 10}};},\n"
        "  },\n"
        " },\n"
        "};\n"
        "const context = {runId: 100, repo: {owner: 'o', repo: 'r'},\n"
        " payload: {pull_request: {number: 14, head: {sha: 'head'}}}};\n"
        "report({github, context, core: {info: () => {}, warning: () => {}}, status: "
        + json.dumps(status)
        + "}).then((result) => console.log(JSON.stringify({mutations, result})))\n"
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
    result = _report(
        tmp_path,
        status=status,
        prior_run="100" if status != "in_progress" else "99",
        prior_attempt="2" if status != "in_progress" else "1",
    )
    assert len(result["mutations"]) == 1
    verb, payload = result["mutations"][0]
    assert verb == ("create" if status == "in_progress" else "update")
    assert result["result"] == (10 if status == "in_progress" else 7)
    body = payload["body"]
    assert "patch is correct" not in body
    assert "no completed verdict" in body
    assert "No automatic approval" in body
    assert "https://example.test/o/r/actions/runs/100/attempts/2" in body
    assert f"Review {status.replace('_', ' ')}" in body


def test_older_workflow_rerun_can_start_a_new_attempt(tmp_path: Path) -> None:
    result = _report(tmp_path, status="in_progress", prior_run="101")
    assert result["mutations"][0][0] == "create"


def test_obsolete_attempt_of_same_workflow_does_not_create_new_status(tmp_path: Path) -> None:
    assert not _report(tmp_path, prior_run="100", prior_attempt="3")["mutations"]


@pytest.mark.parametrize("obsolete", ["attempt", "head"])
def test_obsolete_startup_stops_before_the_cli(tmp_path: Path, obsolete: str) -> None:
    with pytest.raises(subprocess.CalledProcessError):
        if obsolete == "attempt":
            _report(tmp_path, status="in_progress", prior_run="100", prior_attempt="3")
        else:
            _report(tmp_path, status="in_progress", head="new-head")


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_cleanup_never_creates_or_repurposes_an_attempt_record(tmp_path: Path, status: str) -> None:
    assert not _report(tmp_path, status=status, legacy=True)["mutations"]
    assert not _report(tmp_path, status=status, newer_comment=True)["mutations"]


def test_failure_replaces_its_own_attempts_completed_verdict(tmp_path: Path) -> None:
    body = _report(tmp_path, prior_run="100", prior_attempt="2")["mutations"][0][1]["body"]
    assert "Review failed" in body
    assert "no completed verdict" in body
    assert _report(tmp_path, prior_run="100", prior_attempt="2")["mutations"][0][0] == "update"


def test_cancelled_old_head_cannot_overwrite_current_head_status(tmp_path: Path) -> None:
    assert not _report(tmp_path, status="cancelled", head="new-head")["mutations"]


def test_superseded_head_finalizes_its_existing_attempt_comment(tmp_path: Path) -> None:
    result = _report(
        tmp_path, status="cancelled", head="new-head", prior_run="100", prior_attempt="2"
    )
    assert result["mutations"][0][0] == "update"
    assert result["mutations"][0][1]["comment_id"] == 7
    assert "Review cancelled" in result["mutations"][0][1]["body"]


def test_old_failed_run_updates_only_its_own_comment(tmp_path: Path) -> None:
    result = _report(tmp_path, prior_run="100", prior_attempt="2", newer_comment=True)
    assert len(result["mutations"]) == 1
    verb, payload = result["mutations"][0]
    assert verb == "update"
    assert payload["comment_id"] == 7
    assert "Review failed" in payload["body"]


def test_reporter_does_not_edit_human_comments(tmp_path: Path) -> None:
    result = _report(tmp_path, author="human", status="in_progress")
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
    assert start["id"] == "review_status"
    cli = next(step for step in steps if step.get("id") == "run_dotbot_cli")
    assert (
        cli["env"]["DOTBOT_REVIEW_STATUS_COMMENT_ID"] == "${{ steps.review_status.outputs.result }}"
    )
    assert "return await report" in start["with"]["script"]
    for step in (start, end):
        assert "inputs.mode == 'review'" in step["if"]
        assert "inputs.dry_run != '1'" in step["if"]
        assert step["uses"] == "actions/github-script@v7"
        assert step["env"]["DOTBOT_ACTION_PATH"] == "${{ github.action_path }}"
    assert "always()" in end["if"]
    assert "failure() || cancelled()" in end["if"]
