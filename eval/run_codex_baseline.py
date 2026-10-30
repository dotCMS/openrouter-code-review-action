#!/usr/bin/env python3
"""Capture the Codex baseline for the bake-off sample PRs.

Sub-AC 1.2 deliverable. The runner iterates over every PR pinned in
``eval/sample-prs.yaml`` and writes a per-PR raw findings artifact to
``eval/runs/<pr_id>/codex.json`` plus a ``meta.yaml`` with the
reproducibility metadata defined in ``eval/baseline-methodology.md``
§4.2.

Design notes
------------

* The runner is **idempotent**: re-running it without ``--force`` skips
  PRs whose ``codex.json`` already has ``capture.status == "captured"``.
  This is what makes the bake-off cheap to extend (one new PR added to
  the sample only re-runs that PR, not the other nine).

* It writes **schema-conforming placeholders** when the live tooling
  is unavailable. A placeholder is marked ``capture.status = "pending"``
  and downstream scoring Sub-ACs treat pending PRs as not-yet-measurable.
  This honours the methodology rule that the bake-off is scored on real
  captured data, never on synthesized findings.

* It **never posts comments to GitHub or approves any PR.** Live
  capture invokes the existing :class:`CodexClient` directly in
  read-only sandbox mode against a pinned checkout, reusing the
  production review guidelines and ``REVIEW_OUTPUT_SCHEMA``, and
  stores the parsed ``ReviewRunResult`` without touching GitHub. The
  "posted" block stays at its zero/default ``PostingOutcome`` shape
  because no comments are ever published.

* It **does not route through production review.** v2 moved
  production review to the OpenRouter coordinator; the eval keeps a
  standalone Codex SDK path so the baseline is measured on real Codex
  output rather than mislabelled OpenRouter output.

* It **isolates each PR's git checkout**: a fresh worktree per PR
  under ``${RUNNER_TEMP:-/tmp}/codex-baseline/<pr_id>/``. This avoids
  bleed-over of uncommitted state between runs and makes a single
  failed capture impossible to silently corrupt the next.

Usage
-----

::

    python -m eval.run_codex_baseline --pr-id dotcms-core-35567
    python -m eval.run_codex_baseline --all
    python -m eval.run_codex_baseline --all --force        # re-capture even completed
    python -m eval.run_codex_baseline --all --dry-run-only # write placeholders, do not invoke codex

Required env (live capture):

* ``OPENAI_API_KEY`` — Codex model auth.

* ``GITHUB_TOKEN`` — read the pinned PR head/base metadata; never post.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PATH = REPO_ROOT / "eval" / "sample-prs.yaml"
RUNS_DIR = REPO_ROOT / "eval" / "runs"
RUNNER_VERSION = "2"

# The legacy Codex baseline is pinned to the model + reasoning effort
# that was in production use at the time the sample was selected.
# See action.yml inputs (model: gpt-5.4, reasoning_effort: medium) and
# .github/workflows/codex-review.yml.
DEFAULT_BASELINE_MODEL = "gpt-5.4"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_WEB_SEARCH_MODE = "live"


# ---------------------------------------------------------------------------
# Sample loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SamplePR:
    id: str
    repo: str
    pr: int
    head_sha: str
    web_search_mode: str
    reasoning_effort: str
    model: str = DEFAULT_BASELINE_MODEL


def load_sample(path: Path | None = None) -> list[SamplePR]:
    """Load the sample PR registry into typed entries.

    ``path`` defaults to the module-level ``SAMPLE_PATH``. Resolving it
    lazily (rather than as a default-argument value) means tests can
    monkeypatch ``SAMPLE_PATH`` and load_sample() picks up the override.
    """
    if path is None:
        path = SAMPLE_PATH
    with path.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, Mapping):
        raise SystemExit(f"{path}: top-level document must be a mapping")
    raw = doc.get("prs") or []
    if not isinstance(raw, list):
        raise SystemExit(f"{path}: 'prs' must be a list")
    out: list[SamplePR] = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            raise SystemExit(f"{path}: each prs[] entry must be a mapping")
        out.append(
            SamplePR(
                id=str(entry["id"]),
                repo=str(entry["repo"]),
                pr=int(entry["pr"]),
                head_sha=str(entry["head_sha"]),
                web_search_mode=str(
                    entry.get("web_search_mode_for_run") or DEFAULT_WEB_SEARCH_MODE
                ),
                reasoning_effort=str(
                    entry.get("reasoning_effort_for_run") or DEFAULT_REASONING_EFFORT
                ),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Artifact I/O — atomic write, idempotency check, schema-conforming shape
# ---------------------------------------------------------------------------


def artifact_dir_for(pr: SamplePR) -> Path:
    return RUNS_DIR / pr.id


def codex_artifact_path(pr: SamplePR) -> Path:
    return artifact_dir_for(pr) / "codex.json"


def meta_artifact_path(pr: SamplePR) -> Path:
    return artifact_dir_for(pr) / "meta.yaml"


def is_already_captured(pr: SamplePR) -> bool:
    path = codex_artifact_path(pr)
    if not path.exists():
        return False
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    capture = data.get("capture") if isinstance(data, Mapping) else None
    if not isinstance(capture, Mapping):
        return False
    return capture.get("status") == "captured"


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


def write_artifact(
    pr: SamplePR,
    *,
    capture_status: str,
    review_run_result: dict[str, Any] | None,
    posted: dict[str, Any] | None,
    notes: str,
) -> None:
    artifact: dict[str, Any] = {
        "pr_id": pr.id,
        "run": "codex",
        "head_sha": pr.head_sha,
        "model": pr.model,
        "reasoning_effort": pr.reasoning_effort,
        "web_search_mode": pr.web_search_mode,
        "prior_review_state": None,
        "capture": {
            "status": capture_status,
            "runner_version": RUNNER_VERSION,
            "captured_at": (
                datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
                if capture_status == "captured"
                else None
            ),
            "command": (f"python -m eval.run_codex_baseline --pr-id {pr.id}"),
            "notes": notes,
        },
        "review_run_result": review_run_result,
        "posted": posted
        or {
            "summary_id": None,
            "inline_ids": [],
            "posting_outcome": {
                "batch_submitted": 0,
                "per_comment_fallback": 0,
                "skipped_after_422": 0,
            },
        },
    }
    _atomic_write(codex_artifact_path(pr), json.dumps(artifact, indent=2, sort_keys=True) + "\n")

    meta = {
        "pr_id": pr.id,
        "repo": pr.repo,
        "pr": pr.pr,
        "head_sha": pr.head_sha,
        "model": pr.model,
        "reasoning_effort": pr.reasoning_effort,
        "web_search_mode": pr.web_search_mode,
        "runner_version": RUNNER_VERSION,
        "capture_status": capture_status,
    }
    _atomic_write(
        meta_artifact_path(pr),
        yaml.safe_dump(meta, sort_keys=True, default_flow_style=False),
    )


# ---------------------------------------------------------------------------
# Live capture — invokes the existing Codex CLI in --dry-run mode
# ---------------------------------------------------------------------------


def _have_codex_module() -> bool:
    try:
        __import__("codex")
        return True
    except ImportError:
        return False


def _live_env_ok() -> tuple[bool, str]:
    if not os.environ.get("OPENAI_API_KEY"):
        return False, "OPENAI_API_KEY not set"
    if not os.environ.get("GITHUB_TOKEN"):
        return False, "GITHUB_TOKEN not set"
    if not _have_codex_module():
        return False, "codex python module not installed (pip install codex-python==1.122.0)"
    return True, ""


def _checkout_pr(pr: SamplePR, dest: Path, *, repo_url: str | None = None) -> None:
    """Clone ``pr.repo`` and check out the pinned ``head_sha`` at ``dest``.

    Fetches full history for every branch tip plus the pinned head so
    the PR base commit and its merge base are present. The read-only
    Codex agent runs ``git diff <base>...<head>`` in this checkout; a
    depth-1 head-only fetch omits both the base and the merge base.

    ``repo_url`` overrides the GitHub URL so tests can point at a local
    bare repository instead of the network.
    """
    dest.mkdir(parents=True, exist_ok=True)
    remote = repo_url or f"https://github.com/{pr.repo}.git"
    subprocess.run(["git", "init", str(dest)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(dest), "remote", "add", "origin", remote],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(dest), "fetch", "--no-tags", "origin"],
        check=True,
        capture_output=True,
    )
    # The pinned head may be unreachable from any branch (unmerged PR),
    # so fetch it explicitly. A server that refuses single-SHA fetches
    # still leaves the branch fetch above in place.
    subprocess.run(
        ["git", "-C", str(dest), "fetch", "--no-tags", "origin", pr.head_sha],
        check=False,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(dest), "checkout", "--detach", pr.head_sha],
        check=True,
        capture_output=True,
    )


def _build_sdk_prompt(pr: SamplePR, base_sha: str, guidelines: str) -> str:
    """Build the read-only Codex prompt with pinned benchmark context."""
    diff_rev = f"{base_sha}...{pr.head_sha}"
    parts = [
        "You are an autonomous code review assistant.",
        "Follow the review guidelines below verbatim while producing "
        "prioritized, actionable findings.",
        "",
        "<benchmark_context>",
        f"<repository>{pr.repo}</repository>",
        f"<pull_request>{pr.pr}</pull_request>",
        f"<head_sha>{pr.head_sha}</head_sha>",
        f"<base_sha>{base_sha or 'unknown'}</base_sha>",
        "</benchmark_context>",
    ]
    guidelines_text = guidelines.strip()
    if guidelines_text:
        parts.extend(["", "Review guidelines:", guidelines_text])
    parts.extend(
        [
            "",
            "The repository is checked out in the current working directory "
            "at the pinned head SHA.",
            f"Run `git diff {diff_rev}` to inspect the changes under review.",
            "Use HEAD (right side) line numbers when citing code locations.",
        ]
    )
    return "\n".join(parts).strip()


def _invoke_codex_sdk(pr: SamplePR, repo_path: Path) -> dict[str, Any]:
    """Run the existing Codex SDK read-only against the pinned checkout.

    v2 moved production review to the OpenRouter coordinator, so the
    eval talks to :class:`CodexClient` directly instead of shelling out
    to ``cli.main --mode review`` (which would now run OpenRouter). It
    reuses the production review guidelines and
    ``REVIEW_OUTPUT_SCHEMA`` and returns the parsed ``ReviewRunResult``
    as a dict, with no GitHub posting or approval.

    The SDK import is resolved before entering the PR worktree so a
    ``cli`` package inside the checked-out PR can never shadow the
    action's. It is intentionally lazy: the eval module must load
    without the optional ``codex-python`` package installed.
    """
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from cli.clients.codex_client import CodexClient
    from cli.clients.github_client import GitHubClient
    from cli.core.config import ReviewConfig
    from cli.core.models import REVIEW_OUTPUT_SCHEMA, ReviewRunResult
    from cli.review.review_prompt import load_guidelines

    # ReviewConfig.validate() blocks review mode with the openai provider
    # (v2 routes production review through OpenRouter); the eval calls
    # CodexClient directly and deliberately never validates as production.
    config = ReviewConfig(
        github_token=os.environ.get("GITHUB_TOKEN", ""),
        repository=pr.repo,
        pr_number=pr.pr,
        mode="review",
        model_provider="openai",
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        model_name=pr.model,
        reasoning_effort=pr.reasoning_effort,
        web_search_mode=pr.web_search_mode,
        debug_level=1,
        stream_output=False,
        repo_root=repo_path,
    )
    snapshot = GitHubClient(config).get_pr(pr.pr)
    if snapshot.head is None or snapshot.head.sha != pr.head_sha:
        raise RuntimeError("Benchmark PR head changed; refusing to capture a different revision")
    if snapshot.base is None or not snapshot.base.sha:
        raise RuntimeError("Benchmark PR has no base revision")
    base_sha = snapshot.base.sha
    subprocess.run(
        ["git", "-C", str(repo_path), "cat-file", "-e", f"{base_sha}^{{commit}}"],
        check=True,
        capture_output=True,
    )
    prompt = _build_sdk_prompt(pr, base_sha, load_guidelines(config))

    previous_cwd = Path.cwd()
    os.chdir(repo_path)
    try:
        output = CodexClient(config).execute_structured(
            prompt,
            sandbox_mode="read-only",
            output_schema=REVIEW_OUTPUT_SCHEMA,
            schema_prompt=('Produce the JSON review output now. Return "carried_forward" as [].'),
        )
    finally:
        os.chdir(previous_cwd)

    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Codex SDK returned non-JSON review output") from exc
    return ReviewRunResult.from_payload(payload).as_dict()


def capture_one(pr: SamplePR, *, dry_run_only: bool, force: bool) -> str:
    """Capture (or refresh) the Codex baseline for a single PR.

    Returns the resulting ``capture.status`` string.
    """
    if not force and is_already_captured(pr):
        print(f"[skip] {pr.id}: already captured", file=sys.stderr)
        return "captured"

    if dry_run_only:
        write_artifact(
            pr,
            capture_status="pending",
            review_run_result=None,
            posted=None,
            notes=(
                "Placeholder written by --dry-run-only. Re-run without "
                "--dry-run-only on a host with OPENAI_API_KEY and "
                "codex-python installed to capture the live baseline."
            ),
        )
        print(f"[pending] {pr.id}: placeholder written", file=sys.stderr)
        return "pending"

    ok, why = _live_env_ok()
    if not ok:
        write_artifact(
            pr,
            capture_status="pending",
            review_run_result=None,
            posted=None,
            notes=f"Live capture skipped: {why}",
        )
        print(f"[pending] {pr.id}: {why}; placeholder written", file=sys.stderr)
        return "pending"

    workdir = Path(os.environ.get("RUNNER_TEMP") or tempfile.gettempdir())
    repo_dir = workdir / "codex-baseline" / pr.id
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    try:
        _checkout_pr(pr, repo_dir)
        rrr = _invoke_codex_sdk(pr, repo_dir)
    except Exception as exc:
        write_artifact(
            pr,
            capture_status="failed",
            review_run_result=None,
            posted=None,
            notes=f"Live capture failed: {exc}",
        )
        print(f"[failed] {pr.id}: {exc}", file=sys.stderr)
        return "failed"

    write_artifact(
        pr,
        capture_status="captured",
        review_run_result=rrr,
        posted=None,
        notes="Captured via Codex SDK (read-only) on isolated worktree",
    )
    print(f"[captured] {pr.id}: {len(rrr.get('findings') or [])} findings", file=sys.stderr)
    return "captured"


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def _select(prs: Iterable[SamplePR], pr_ids: Sequence[str] | None) -> list[SamplePR]:
    if not pr_ids:
        return list(prs)
    by_id = {pr.id: pr for pr in prs}
    missing = [pid for pid in pr_ids if pid not in by_id]
    if missing:
        raise SystemExit(
            f"unknown pr_id(s): {', '.join(missing)}. Known: {', '.join(sorted(by_id))}"
        )
    return [by_id[pid] for pid in pr_ids]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eval.run_codex_baseline",
        description="Capture the Codex baseline for the bake-off sample PRs.",
    )
    sel = parser.add_mutually_exclusive_group(required=True)
    sel.add_argument("--all", action="store_true", help="Capture every PR in the sample registry.")
    sel.add_argument(
        "--pr-id",
        action="append",
        dest="pr_ids",
        metavar="ID",
        help="Capture this PR id (may be repeated).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-capture even if codex.json already has status == captured.",
    )
    parser.add_argument(
        "--dry-run-only",
        action="store_true",
        help=(
            "Do not invoke the Codex SDK; just write/refresh schema-"
            "conforming placeholders. Useful for bootstrapping the "
            "artifact directory before live-capture access is available."
        ),
    )
    args = parser.parse_args(argv)

    sample = load_sample()
    if not sample:
        raise SystemExit(
            f"{SAMPLE_PATH}: prs[] is empty. Populate the sample first "
            "(see Sub-AC 1.1.1 / evaluation/bakeoff/sample-prs.yml)."
        )
    targets = _select(sample, None if args.all else args.pr_ids)

    rc = 0
    statuses: dict[str, int] = {}
    for pr in targets:
        status = capture_one(pr, dry_run_only=args.dry_run_only, force=args.force)
        statuses[status] = statuses.get(status, 0) + 1
        if status == "failed":
            rc = 1

    print(
        "[summary] " + ", ".join(f"{k}={v}" for k, v in sorted(statuses.items())),
        file=sys.stderr,
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
