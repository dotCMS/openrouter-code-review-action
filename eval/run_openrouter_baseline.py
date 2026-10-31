#!/usr/bin/env python3
"""Capture the OpenRouter+aider action review baseline for the bake-off sample.

Sub-AC 2.2 deliverable. Mirror of ``eval/run_codex_baseline.py`` but
invoking the **OpenRouter+aider action in review mode** (i.e. the new
production review path in ``cli.main --mode review``) instead of the
legacy Codex CLI. Iterates every PR pinned in ``eval/sample-prs.yaml``,
checks each one out into an isolated worktree, shells out to the
in-tree CLI, and writes a per-PR raw findings artifact at
``eval/runs/<pr_id>/openrouter.json`` plus a ``meta.yaml`` reproducibility
sidecar (schema in ``eval/baseline-methodology.md`` §4.2).

Why a *separate* script from ``evaluation/bakeoff/run_openrouter.py``?
``evaluation/bakeoff/run_openrouter.py`` is a self-contained probe that
calls OpenRouter directly so the bake-off can be re-run on every model
bump without coupling to in-flight workflow code. *This* script does
the inverse: it exercises the **shipped action's review path end-to-end**
on each curated PR (checkout → cli.main → ReviewRunResult) so the
parity-evidence comparison in Sub-AC 2.4 is grounded in the production
code, not just in the raw chat-completions wire shape.

Design rules (held in lockstep with ``run_codex_baseline.py``):

* **Idempotent**: re-running without ``--force`` skips PRs whose
  ``openrouter.json`` already has ``capture.status == "captured"``.
  Re-runs cost only the PRs that changed.

* **Schema-conforming placeholders**: when ``OPENROUTER_API_KEY`` /
  ``GITHUB_TOKEN`` aren't available, write a placeholder marked
  ``capture.status = "pending"`` so downstream scoring sees the empty
  slot rather than silently missing the PR. The bake-off is scored on
  real captured data, never on synthesized findings.

* **Never posts to GitHub**: live capture invokes ``cli.main`` in
  ``--dry-run`` mode and parses the structured ``ReviewRunResult`` JSON
  envelope from stdout. The ``posted`` block in the artifact is the
  ``PostingOutcome`` shape (batch_submitted / per_comment_fallback /
  skipped_after_422 counts) zeroed out — the live PR is untouched.

* **Isolated git checkout per PR**: a fresh worktree under
  ``${RUNNER_TEMP:-/tmp}/openrouter-baseline/<pr_id>/`` per run. No
  state bleed-over.

Usage::

    python -m eval.run_openrouter_baseline --pr-id dotcms-core-35567
    python -m eval.run_openrouter_baseline --all
    python -m eval.run_openrouter_baseline --all --force          # re-capture
    python -m eval.run_openrouter_baseline --all --dry-run-only   # placeholders only

Required env (live capture):

* ``OPENROUTER_API_KEY`` — OpenRouter auth (every model call routes here).
* ``GITHUB_TOKEN``       — read access for diff fetch + prior-review thread.
"""

from __future__ import annotations

import argparse
import json
import os
import re
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

# Canonical stdout markers emitted by ``cli.main``. Panel turns stream under
# ``[review-panel]`` and are intermediate; only ``[review-result]`` carries
# the consolidated final ``ReviewRunResult``.
RESULT_MARKER = "[review-result]"

# Defaults read from the in-repo model config file when present, but
# the script must be runnable on any checkout (e.g. before the config
# is materialized in CI), so we keep defensive constants here too.
# Tracks `seed.yaml` constraint: "Initial review and act model:
# anthropic/claude-opus-4.7 via OpenRouter".
DEFAULT_REVIEW_MODEL = "anthropic/claude-opus-4.7"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_WEB_SEARCH_MODE = "live"


# ---------------------------------------------------------------------------
# Sample loading — accepts the canonical eval/sample-prs.yaml schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SamplePR:
    id: str
    repo: str
    pr: int
    head_sha: str
    base_sha: str
    web_search_mode: str
    reasoning_effort: str
    model: str = DEFAULT_REVIEW_MODEL
    merge_sha: str | None = None


def load_sample(path: Path | None = None) -> list[SamplePR]:
    """Load the sample PR registry into typed entries.

    ``path`` defaults to the module-level ``SAMPLE_PATH``. Resolved
    lazily so tests can monkeypatch ``SAMPLE_PATH`` and have
    ``load_sample()`` pick up the override.
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
        for field in ("pr_head_sha", "pr_base_sha"):
            value = entry.get(field)
            if not isinstance(value, str) or not re.fullmatch(
                r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})", value
            ):
                raise SystemExit(f"{path}: {field} must be an exact commit SHA")
        out.append(
            SamplePR(
                id=str(entry["id"]),
                repo=str(entry["repo"]),
                pr=int(entry["pr"]),
                head_sha=str(entry["pr_head_sha"]).lower(),
                base_sha=str(entry["pr_base_sha"]).lower(),
                merge_sha=str(entry["head_sha"]) if entry.get("head_sha") else None,
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
# Artifact I/O — atomic write, idempotency, schema-conforming shape
# ---------------------------------------------------------------------------


def artifact_dir_for(pr: SamplePR) -> Path:
    return RUNS_DIR / pr.id


def openrouter_artifact_path(pr: SamplePR) -> Path:
    return artifact_dir_for(pr) / "openrouter.json"


def meta_artifact_path(pr: SamplePR) -> Path:
    # Distinct meta filename so a side-by-side codex+openrouter capture in
    # the same eval/runs/<pr_id>/ folder does not clobber the codex meta.
    return artifact_dir_for(pr) / "openrouter-meta.yaml"


def is_already_captured(pr: SamplePR) -> bool:
    path = openrouter_artifact_path(pr)
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
    model: str | None = None,
) -> None:
    """Write the per-PR ``openrouter.json`` + ``openrouter-meta.yaml``.

    Mirrors the Codex baseline runner's envelope exactly so downstream
    scoring code can consume both ``codex.json`` and ``openrouter.json``
    via one shared parser.
    """
    resolved_model = model or pr.model
    artifact: dict[str, Any] = {
        "pr_id": pr.id,
        "run": "openrouter",
        "head_sha": pr.head_sha,
        "base_sha": pr.base_sha,
        "merge_sha": pr.merge_sha,
        "model": resolved_model,
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
            "command": (f"python -m eval.run_openrouter_baseline --pr-id {pr.id}"),
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
    _atomic_write(
        openrouter_artifact_path(pr),
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
    )

    meta = {
        "pr_id": pr.id,
        "repo": pr.repo,
        "pr": pr.pr,
        "head_sha": pr.head_sha,
        "base_sha": pr.base_sha,
        "merge_sha": pr.merge_sha,
        "model": resolved_model,
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
# In-repo model-config resolution (respects swap-speed: a single config
# edit must be enough to change the review model the runner exercises)
# ---------------------------------------------------------------------------


def resolve_review_model(repo_root: Path | None = None) -> tuple[str, str, str]:
    """Return ``(review_model, reasoning_effort, web_search_mode)`` for the run.

    Reads the in-repo ``.openrouter-review.yml`` via the production
    loader so the harness exercises the same swap-speed contract as
    the action itself. Falls back to the ``DEFAULT_*`` constants when
    the loader is unavailable (e.g. a checkout without the cli package
    on ``sys.path``) so the harness always has a usable defaults set.
    """
    root = repo_root or REPO_ROOT
    try:
        sys.path.insert(0, str(root))
        from cli.core.model_config import load_model_config  # local import
    except Exception:
        return (DEFAULT_REVIEW_MODEL, DEFAULT_REASONING_EFFORT, DEFAULT_WEB_SEARCH_MODE)

    try:
        cfg = load_model_config(repo_root=root)
    except Exception:
        return (DEFAULT_REVIEW_MODEL, DEFAULT_REASONING_EFFORT, DEFAULT_WEB_SEARCH_MODE)

    return (
        cfg.review_model or DEFAULT_REVIEW_MODEL,
        cfg.review_reasoning_effort or DEFAULT_REASONING_EFFORT,
        cfg.review_web_search_mode or DEFAULT_WEB_SEARCH_MODE,
    )


# ---------------------------------------------------------------------------
# Live capture — checkout + invoke cli.main --mode review --dry-run
# ---------------------------------------------------------------------------


def _live_env_ok() -> tuple[bool, str]:
    if not os.environ.get("OPENROUTER_API_KEY"):
        return False, "OPENROUTER_API_KEY not set"
    if not os.environ.get("GITHUB_TOKEN"):
        return False, "GITHUB_TOKEN not set"
    return True, ""


def _checkout_pr(pr: SamplePR, dest: Path, *, repo_url: str | None = None) -> None:
    """Clone ``pr.repo`` and check out the pinned ``head_sha`` at ``dest``.

    Fetches full history (no ``--depth``) for every branch tip plus the
    pinned head, so the PR base commit and its merge base are present.
    The production panel runs ``git diff <base>...<head>`` in this
    checkout and fails closed when either commit is missing; a depth-1
    head-only fetch omits both the base and the merge base.

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
    # Full fetch of every branch tip (including the PR base branch) with
    # complete history so ``git diff base...head`` can resolve the merge base.
    subprocess.run(
        ["git", "-C", str(dest), "fetch", "--no-tags", "origin"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(dest), "fetch", "--no-tags", "origin", pr.base_sha],
        check=True,
        capture_output=True,
    )
    # The pinned head may be unreachable from any branch (unmerged PR), so
    # fetch it explicitly, also with full history. A server that refuses
    # single-SHA fetches still leaves the branch fetch above in place.
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


def _invoke_review_cli(
    pr: SamplePR,
    repo_path: Path,
    *,
    review_model: str,
    reasoning_effort: str,
    web_search_mode: str,
) -> dict[str, Any]:
    """Run ``cli.main --mode review --dry-run`` against ``repo_path``.

    The CLI streams intermediate ``[review-panel]`` turns and then prints
    the consolidated ``[review-result] <ReviewRunResult JSON>`` line on
    stdout in ``--dry-run`` mode (it skips posting but still emits the
    final payload). We capture stdout and anchor on that final marker.

    OpenRouter routing: the action reads ``OPENROUTER_API_KEY`` from
    env, so we just inherit the parent process env (which the live-env
    pre-check ensures is set).
    """
    cmd = [
        sys.executable,
        "-m",
        "cli.main",
        "--repo",
        pr.repo,
        "--pr",
        str(pr.pr),
        "--repo-root",
        str(repo_path),
        "--mode",
        "review",
        "--model",
        review_model,
        "--reasoning-effort",
        reasoning_effort,
        "--web-search-mode",
        web_search_mode,
        "--dry-run",
        "--debug",
        "1",
    ]
    env = dict(os.environ)
    # Anchor the import to the trusted action source, never the PR checkout:
    # without PYTHONSAFEPATH, ``python -m`` would prefer a ``cli/`` package
    # present in the checked-out PR. Mirrors action.yml's hardening.
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + pythonpath if pythonpath else "")
    env["PYTHONSAFEPATH"] = "1"
    env["DOTBOT_REVIEW_EXPECTED_HEAD_SHA"] = pr.head_sha
    env["DOTBOT_REVIEW_EXPECTED_BASE_SHA"] = pr.base_sha
    env.pop("DOTBOT_REVIEW_STATUS_COMMENT_ID", None)
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "OpenRouter+aider review CLI dry-run failed "
            f"(rc={proc.returncode}); stderr={proc.stderr[-2000:]!r}"
        )
    return _extract_review_run_result(proc.stdout)


def _extract_review_run_result(stdout: str) -> dict[str, Any]:
    """Read only the final tagged aggregate, never a discussion turn or preview."""
    for line in reversed(stdout.splitlines()):
        line = line.lstrip()
        if not line.startswith(RESULT_MARKER):
            continue
        try:
            payload = json.loads(line[len(RESULT_MARKER) :])
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"malformed JSON after {RESULT_MARKER} marker") from exc
        if (
            not isinstance(payload, dict)
            or not {"findings", "carried_forward", "overall_correctness"} <= payload.keys()
        ):
            raise RuntimeError(f"{RESULT_MARKER} marker is not followed by a ReviewRunResult")
        return payload
    raise RuntimeError(f"review output is missing the final {RESULT_MARKER} marker")


def capture_one(
    pr: SamplePR,
    *,
    dry_run_only: bool,
    force: bool,
    review_model: str | None = None,
    reasoning_effort: str | None = None,
    web_search_mode: str | None = None,
) -> str:
    """Capture (or refresh) the OpenRouter+aider review baseline for one PR.

    Returns the resulting ``capture.status`` string.
    """
    if not force and is_already_captured(pr):
        print(f"[skip] {pr.id}: already captured", file=sys.stderr)
        return "captured"

    # Resolve model + reasoning + web-search from the in-repo config file
    # at call time so the swap-speed contract holds: editing
    # .openrouter-review.yml retargets the runner without code change.
    if review_model is None or reasoning_effort is None or web_search_mode is None:
        rm, re_, ws = resolve_review_model()
        review_model = review_model or rm
        reasoning_effort = reasoning_effort or re_
        web_search_mode = web_search_mode or ws

    if dry_run_only:
        write_artifact(
            pr,
            capture_status="pending",
            review_run_result=None,
            posted=None,
            notes=(
                "Placeholder written by --dry-run-only. Re-run without "
                "--dry-run-only on a host with OPENROUTER_API_KEY + "
                "GITHUB_TOKEN to capture the live baseline."
            ),
            model=review_model,
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
            model=review_model,
        )
        print(f"[pending] {pr.id}: {why}; placeholder written", file=sys.stderr)
        return "pending"

    workdir = Path(os.environ.get("RUNNER_TEMP") or tempfile.gettempdir())
    repo_dir = workdir / "openrouter-baseline" / pr.id
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    try:
        _checkout_pr(pr, repo_dir)
        rrr = _invoke_review_cli(
            pr,
            repo_dir,
            review_model=review_model,
            reasoning_effort=reasoning_effort,
            web_search_mode=web_search_mode,
        )
    except Exception as exc:
        write_artifact(
            pr,
            capture_status="failed",
            review_run_result=None,
            posted=None,
            notes=f"Live capture failed: {exc}",
            model=review_model,
        )
        print(f"[failed] {pr.id}: {exc}", file=sys.stderr)
        return "failed"

    write_artifact(
        pr,
        capture_status="captured",
        review_run_result=rrr,
        posted=None,
        notes="Captured via cli.main --mode review --dry-run on isolated worktree",
        model=review_model,
    )
    print(
        f"[captured] {pr.id}: {len(rrr.get('findings') or [])} findings",
        file=sys.stderr,
    )
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
        prog="eval.run_openrouter_baseline",
        description=(
            "Capture the OpenRouter+aider review baseline for the bake-off "
            "sample PRs by checking each one out and invoking the in-tree "
            "review CLI in --dry-run mode."
        ),
    )
    sel = parser.add_mutually_exclusive_group(required=True)
    sel.add_argument(
        "--all",
        action="store_true",
        help="Capture every PR in the sample registry.",
    )
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
        help="Re-capture even if openrouter.json already has status == captured.",
    )
    parser.add_argument(
        "--dry-run-only",
        action="store_true",
        help=(
            "Do not invoke the review CLI; just write/refresh schema-"
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

    review_model, reasoning_effort, web_search_mode = resolve_review_model()
    print(
        f"openrouter baseline: model={review_model} "
        f"reasoning_effort={reasoning_effort} "
        f"web_search_mode={web_search_mode} "
        f"sample_size={len(targets)}",
        file=sys.stderr,
    )

    rc = 0
    statuses: dict[str, int] = {}
    for pr in targets:
        status = capture_one(
            pr,
            dry_run_only=args.dry_run_only,
            force=args.force,
            review_model=review_model,
            reasoning_effort=reasoning_effort,
            web_search_mode=web_search_mode,
        )
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
