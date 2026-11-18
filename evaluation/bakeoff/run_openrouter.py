#!/usr/bin/env python3
"""OpenRouter bake-off runner.

Sub-AC 2.2 deliverable. Iterates the curated PR sample in
``evaluation/bakeoff/sample-prs.yml`` and writes one
``evaluation/bakeoff/runs/<pr_id>/openrouter.json`` artifact per PR
in the schema defined by ``eval/baseline-methodology.md`` §4.2.

The runner is self-contained on purpose: the in-tree review pipeline
still routes through ``CodexClient`` while AC 1's OpenRouter chat
path is being implemented. Rather than couple the parity-evidence
collection to the in-flight workflow rewrite, this script speaks
directly to OpenRouter's OpenAI-compatible chat-completions
endpoint, applies the strict ``response_format`` built by
``cli.core.models.openrouter_review_response_format``, and validates
the returned payload through ``validate_review_payload``. That gives
us a stable harness we can re-run on every model bump.

Modes
-----

* ``--mode live``    — POST to OpenRouter, validate the returned
  ``ReviewRunResult``, persist as ``status=completed``.
* ``--mode dry``     — Fetch the diff and build the OpenRouter
  request payload (model, messages, response_format, reasoning), but
  do *not* POST. Persists a ``status=skipped_dry_run`` artifact with
  the resolved payload-shape telemetry (diff_chars, message count,
  reasoning effort, response_format kind, online suffix). Useful for
  verifying the runner wiring end-to-end without spending tokens.
* ``--mode replay``  — Re-derive findings from a previously captured
  raw ``/api/v1/chat/completions`` response committed to
  ``evaluation/bakeoff/replay/<pr_id>.json``. Reuses the same
  extraction + schema-validation path as ``--mode live`` so a parser
  regression surfaces against real recorded data, not synthesized
  findings. Persists ``status=replayed`` with the full
  ``ReviewRunResult``.
* ``--mode scaffold`` (default) — Persist a
  ``status=unfilled`` placeholder that records the slot, the head
  SHA, the model that *will* be used, and what the operator must
  set to fill it. Designed to be checked in: makes downstream
  parity-scoring sub-ACs see the empty slot rather than silently
  miss the PR.

Required environment for ``--mode live``:

* ``OPENROUTER_API_KEY``
* ``GITHUB_TOKEN`` (or working ``gh`` CLI auth) — used to fetch the
  unified diff for the sample PR's pinned head SHA.

Per-PR artifact schema (mirrors §4.2 of the methodology):

```yaml
pr_id:                # e.g. pr-35567
run:                  # always "openrouter" here
status:               # completed | skipped_dry_run | unfilled | error
head_sha:             # the merge_commit reviewed
model:                # OpenRouter slug actually used
reasoning_effort:     # minimal | low | medium | high
web_search_mode:      # disabled | cached | live
prior_review_state:   # null (this is the initial fresh-review pass)
review_run_result:    # raw ReviewRunResult payload, or null if not run
posted: null          # bake-off runs do not post to GitHub
labels: []            # filled by the labeling sub-AC
solo_labeled: null    # filled by the labeling sub-AC
notes:                # free-form runner notes (errors, skip reason, etc.)
captured_at:          # ISO-8601 UTC timestamp
runner:               # this script's relative path, for provenance
runner_version:       # bumped if the runner contract changes
```
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

# Make the in-tree CLI importable so we can reuse the strict
# response_format builder + the schema validator.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cli.core.exceptions import ReviewContractError  # noqa: E402
from cli.core.model_config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    load_model_config,
)
from cli.core.models import (  # noqa: E402
    ReviewRunResult,
    openrouter_review_response_format,
    validate_review_payload,
)

RUNNER_VERSION = "1.0.0"
RUNNER_REL_PATH = "evaluation/bakeoff/run_openrouter.py"

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_REFERER = "https://github.com/dotcms/openrouter-code-review-action"
OPENROUTER_TITLE = "openrouter-code-review-action bake-off"

DEFAULT_SAMPLE = "evaluation/bakeoff/sample-prs.yml"
DEFAULT_OUT_DIR = "evaluation/bakeoff/runs"
DEFAULT_REPLAY_DIR = "evaluation/bakeoff/replay"

PROMPT_PATH = _REPO_ROOT / "prompts" / "review.md"
# The bakeoff sample is application code: pair the base with the default focus prompt.
FOCUS_PROMPT_PATH = _REPO_ROOT / "prompts" / "code_java.md"


# --------------------------------------------------------------------------- #
# Sample loader
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SamplePR:
    """One row from sample-prs.yml, normalized for the runner."""

    pr_id: str
    repo: str
    number: int
    title: str
    head_sha: str
    base_ref: str
    head_ref: str
    surface: str
    change_type: str
    security_signal: bool
    notes: str

    @classmethod
    def from_yaml_entry(cls, entry: dict[str, Any]) -> SamplePR:
        number = int(entry["number"])
        return cls(
            pr_id=f"pr-{number}",
            repo=str(entry["repo"]),
            number=number,
            title=str(entry.get("title", "")).strip(),
            head_sha=str(entry["merge_commit"]).strip(),
            base_ref=str(entry.get("base_ref", "main")),
            head_ref=str(entry.get("head_ref", "")),
            surface=str(entry.get("surface", "")),
            change_type=str(entry.get("change_type", "")),
            security_signal=bool(entry.get("security_signal", False)),
            notes=str(entry.get("why", "")).strip(),
        )


def load_sample(path: Path) -> list[SamplePR]:
    if not path.is_file():
        raise SystemExit(f"sample file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("prs"), list):
        raise SystemExit(f"unexpected sample-prs schema in {path}")
    return [SamplePR.from_yaml_entry(p) for p in data["prs"]]


# --------------------------------------------------------------------------- #
# Diff fetcher (used in --mode live only)
# --------------------------------------------------------------------------- #


def fetch_pr_diff(pr: SamplePR) -> str:
    """Fetch the unified diff for ``pr`` at its pinned head SHA.

    Prefers the ``gh`` CLI because it transparently handles auth and
    follows GitHub's diff redirect. Falls back to the REST API with
    ``GITHUB_TOKEN`` if ``gh`` is missing.
    """
    try:
        completed = subprocess.run(
            ["gh", "pr", "diff", str(pr.number), "--repo", pr.repo],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return completed.stdout
    except FileNotFoundError:
        pass
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"gh pr diff failed for {pr.pr_id}: {exc.stderr.strip()}") from exc

    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Cannot fetch diff: 'gh' CLI not on PATH and GITHUB_TOKEN unset")
    url = f"https://api.github.com/repos/{pr.repo}/pulls/{pr.number}"
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github.v3.diff",
            "Authorization": f"Bearer {token}",
            "User-Agent": "openrouter-code-review-action-bakeoff/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="replace")


# --------------------------------------------------------------------------- #
# OpenRouter call
# --------------------------------------------------------------------------- #


def _system_prompt() -> str:
    return "\n\n".join(
        path.read_text(encoding="utf-8") for path in (PROMPT_PATH, FOCUS_PROMPT_PATH)
    )


def _user_message(pr: SamplePR, diff: str) -> str:
    header = textwrap.dedent(
        f"""\
        <pull_request>
        <repo>{pr.repo}</repo>
        <number>{pr.number}</number>
        <title>{pr.title}</title>
        <head_sha>{pr.head_sha}</head_sha>
        <base_ref>{pr.base_ref}</base_ref>
        <head_ref>{pr.head_ref}</head_ref>
        </pull_request>

        Below is the unified diff to review. Use right-side (HEAD) line
        numbers in code_location.line_range. Emit your findings in the
        strict review_output JSON schema enforced by response_format.
        """
    )
    return f"{header}\n```diff\n{diff}\n```"


def _ensure_online_suffix(model: str, web_search_mode: str) -> str:
    """Apply OpenRouter's ``:online`` web-search variant when requested."""
    if web_search_mode != "live":
        return model
    return model if model.endswith(":online") else f"{model}:online"


def _build_payload(
    pr: SamplePR,
    diff: str,
    model: str,
    reasoning_effort: str,
    web_search_mode: str,
) -> dict[str, Any]:
    return {
        "model": _ensure_online_suffix(model, web_search_mode),
        "messages": [
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": _user_message(pr, diff)},
        ],
        "response_format": openrouter_review_response_format(),
        "reasoning": {"effort": reasoning_effort},
        # No streaming for the bake-off runner: we want the deterministic
        # final JSON payload, not partial frames. Streaming is exercised
        # separately by AC 6 (see tests/test_openrouter_stream.py).
        "stream": False,
    }


def _post_openrouter(api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OPENROUTER_CHAT_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": OPENROUTER_REFERER,
            "X-Title": OPENROUTER_TITLE,
            "User-Agent": "openrouter-code-review-action-bakeoff/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail[:500]}") from exc


def _extract_review_payload(api_response: dict[str, Any]) -> dict[str, Any]:
    """Pull the structured ReviewRunResult payload out of an OpenRouter response."""
    choices = api_response.get("choices") or []
    if not choices:
        raise ReviewContractError("OpenRouter response had no choices")
    message = (choices[0] or {}).get("message") or {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ReviewContractError("OpenRouter response message had no content")
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise ReviewContractError(
            f"OpenRouter returned non-JSON content despite strict schema: {exc}"
        ) from exc


def run_openrouter(
    pr: SamplePR,
    *,
    api_key: str,
    model: str,
    reasoning_effort: str,
    web_search_mode: str,
) -> tuple[ReviewRunResult, dict[str, Any]]:
    diff = fetch_pr_diff(pr)
    payload = _build_payload(pr, diff, model, reasoning_effort, web_search_mode)
    api_response = _post_openrouter(api_key, payload)
    review_payload = _extract_review_payload(api_response)
    review = validate_review_payload(review_payload)
    usage = api_response.get("usage") if isinstance(api_response, dict) else None
    telemetry = {
        "diff_chars": len(diff),
        "usage": usage,
        "model_resolved": payload["model"],
    }
    return review, telemetry


# --------------------------------------------------------------------------- #
# Replay support — load a previously-captured live response and re-derive
# findings without re-billing OpenRouter.
# --------------------------------------------------------------------------- #


def _replay_path_for(pr: SamplePR) -> Path:
    """Resolve the on-disk path for ``pr``'s recorded OpenRouter response."""
    return _REPO_ROOT / DEFAULT_REPLAY_DIR / f"{pr.pr_id}.json"


def replay_recorded_response(
    path: Path,
) -> tuple[ReviewRunResult, dict[str, Any]]:
    """Validate a committed ``/api/v1/chat/completions`` response and return the parsed review.

    The replay file is the *raw* OpenRouter response object captured by an
    operator running ``--mode live`` once. Replaying it reuses the same
    extraction + schema-validation path as the live runner, so a regression
    in the parser is caught against real captured data — not synthesized
    findings — and reviewers running the bake-off offline get deterministic
    output.
    """
    api_response = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(api_response, dict):
        raise ReviewContractError(f"Replay file {path} did not contain a JSON object")
    review_payload = _extract_review_payload(api_response)
    review = validate_review_payload(review_payload)
    usage = api_response.get("usage") if isinstance(api_response, dict) else None
    model_resolved = None
    if isinstance(api_response.get("model"), str):
        model_resolved = api_response["model"]
    telemetry = {
        "diff_chars": None,  # not recorded with the response
        "usage": usage,
        "model_resolved": model_resolved,
        "replay_source": str(path.relative_to(_REPO_ROOT)),
    }
    return review, telemetry


# --------------------------------------------------------------------------- #
# Artifact writer
# --------------------------------------------------------------------------- #


def _now_iso() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _artifact_skeleton(
    pr: SamplePR,
    *,
    status: str,
    model: str,
    reasoning_effort: str,
    web_search_mode: str,
    review_run_result: dict[str, Any] | None,
    notes: str,
    telemetry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "pr_id": pr.pr_id,
        "run": "openrouter",
        "status": status,
        "head_sha": pr.head_sha,
        "repo": pr.repo,
        "pr_number": pr.number,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "web_search_mode": web_search_mode,
        "prior_review_state": None,
        "review_run_result": review_run_result,
        "posted": None,
        "labels": [],
        "solo_labeled": None,
        "notes": notes,
        "telemetry": telemetry or {},
        "captured_at": _now_iso(),
        "runner": RUNNER_REL_PATH,
        "runner_version": RUNNER_VERSION,
    }


def _write_artifact(out_dir: Path, pr: SamplePR, doc: dict[str, Any]) -> Path:
    pr_dir = out_dir / pr.pr_id
    pr_dir.mkdir(parents=True, exist_ok=True)
    artifact = pr_dir / "openrouter.json"
    artifact.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return artifact


# --------------------------------------------------------------------------- #
# Mode handlers
# --------------------------------------------------------------------------- #


def _resolve_models(repo_root: Path, config_path: str | None) -> dict[str, str]:
    """Read the in-repo .openrouter-review.yml so the runner respects swap-speed."""
    cfg = load_model_config(repo_root=repo_root, config_path=config_path)
    return {
        "review_model": cfg.review_model or "anthropic/claude-opus-4.7",
        "reasoning_effort": cfg.review_reasoning_effort or "medium",
        "web_search_mode": cfg.review_web_search_mode or "live",
    }


def _handle_pr(
    pr: SamplePR,
    *,
    mode: str,
    out_dir: Path,
    model: str,
    reasoning_effort: str,
    web_search_mode: str,
    api_key: str | None,
) -> tuple[Path, str]:
    if mode == "scaffold":
        doc = _artifact_skeleton(
            pr,
            status="unfilled",
            model=model,
            reasoning_effort=reasoning_effort,
            web_search_mode=web_search_mode,
            review_run_result=None,
            notes=(
                "Placeholder. Re-run this file's parent script with --mode live "
                "in an environment that exports OPENROUTER_API_KEY (and either "
                "GITHUB_TOKEN or `gh` CLI auth) to populate review_run_result."
            ),
        )
        return _write_artifact(out_dir, pr, doc), "unfilled"

    if mode == "dry":
        # Dry mode exercises every step of the live path *up to* the POST:
        # fetch the diff, build the payload, validate the response_format
        # shape. We persist the payload-shape metadata (no findings) so
        # reviewers can see what the runner would send and so the bake-off
        # has reproducible per-PR evidence that the harness ran end-to-end.
        try:
            diff = fetch_pr_diff(pr)
            payload = _build_payload(
                pr,
                diff,
                model,
                reasoning_effort,
                web_search_mode,
            )
            telemetry: dict[str, Any] = {
                "diff_chars": len(diff),
                "model_resolved": payload["model"],
                "payload_messages": len(payload.get("messages") or []),
                "response_format_kind": (payload.get("response_format") or {}).get("type"),
                "stream": bool(payload.get("stream")),
                "reasoning_effort_resolved": (payload.get("reasoning") or {}).get("effort"),
                "usage": None,  # populated only on live runs
            }
            notes = (
                "Dry-run: diff fetched and OpenRouter request payload built; "
                "no POST issued. Re-run with --mode live to populate "
                "review_run_result."
            )
        except Exception as exc:  # one bad fetch shouldn't sink the rest of the sweep
            telemetry = {
                "diff_chars": None,
                "model_resolved": _ensure_online_suffix(model, web_search_mode),
                "payload_messages": None,
                "response_format_kind": None,
                "stream": False,
                "reasoning_effort_resolved": reasoning_effort,
                "usage": None,
                "dry_run_error": f"{type(exc).__name__}: {exc}",
            }
            notes = (
                "Dry-run: harness wiring exercised but diff fetch / payload build "
                f"failed for this PR ({type(exc).__name__}). The other PRs in the "
                "sweep are unaffected."
            )

        doc = _artifact_skeleton(
            pr,
            status="skipped_dry_run",
            model=model,
            reasoning_effort=reasoning_effort,
            web_search_mode=web_search_mode,
            review_run_result=None,
            notes=notes,
            telemetry=telemetry,
        )
        return _write_artifact(out_dir, pr, doc), "skipped_dry_run"

    if mode == "replay":
        replay_path = _replay_path_for(pr)
        if not replay_path.is_file():
            doc = _artifact_skeleton(
                pr,
                status="unfilled",
                model=model,
                reasoning_effort=reasoning_effort,
                web_search_mode=web_search_mode,
                review_run_result=None,
                notes=(
                    f"Replay: no recorded OpenRouter response at "
                    f"{replay_path.relative_to(_REPO_ROOT)} — leaving slot unfilled. "
                    "Capture once with --mode live (committing the raw "
                    "/api/v1/chat/completions response under "
                    f"{DEFAULT_REPLAY_DIR}/{pr.pr_id}.json) to enable replay."
                ),
            )
            return _write_artifact(out_dir, pr, doc), "unfilled"
        try:
            review, telemetry = replay_recorded_response(replay_path)
        except Exception as exc:
            doc = _artifact_skeleton(
                pr,
                status="error",
                model=model,
                reasoning_effort=reasoning_effort,
                web_search_mode=web_search_mode,
                review_run_result=None,
                notes=(
                    f"Replay failed: {type(exc).__name__}: {exc}. "
                    f"Recorded response at {replay_path.relative_to(_REPO_ROOT)} "
                    "did not validate against REVIEW_OUTPUT_SCHEMA."
                ),
            )
            return _write_artifact(out_dir, pr, doc), "error"

        doc = _artifact_skeleton(
            pr,
            status="replayed",
            model=model,
            reasoning_effort=reasoning_effort,
            web_search_mode=web_search_mode,
            review_run_result=review.as_dict(),
            notes=(
                "Replay: validated a previously-captured live OpenRouter "
                f"response from {replay_path.relative_to(_REPO_ROOT)}. "
                "Findings are real (no synthesis); replay defends the "
                "bake-off against re-billing on every run."
            ),
            telemetry=telemetry,
        )
        return _write_artifact(out_dir, pr, doc), "replayed"

    # mode == live
    if not api_key:
        raise SystemExit(
            "--mode live requires OPENROUTER_API_KEY to be exported in the environment"
        )
    try:
        review, telemetry = run_openrouter(
            pr,
            api_key=api_key,
            model=model,
            reasoning_effort=reasoning_effort,
            web_search_mode=web_search_mode,
        )
    except Exception as exc:  # capture-and-continue: one bad PR shouldn't sink the run
        doc = _artifact_skeleton(
            pr,
            status="error",
            model=model,
            reasoning_effort=reasoning_effort,
            web_search_mode=web_search_mode,
            review_run_result=None,
            notes=f"Live run failed: {type(exc).__name__}: {exc}",
        )
        return _write_artifact(out_dir, pr, doc), "error"

    doc = _artifact_skeleton(
        pr,
        status="completed",
        model=model,
        reasoning_effort=reasoning_effort,
        web_search_mode=web_search_mode,
        review_run_result=review.as_dict(),
        notes="Live OpenRouter call completed; payload validated against REVIEW_OUTPUT_SCHEMA.",
        telemetry=telemetry,
    )
    return _write_artifact(out_dir, pr, doc), "completed"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--sample",
        type=Path,
        default=_REPO_ROOT / DEFAULT_SAMPLE,
        help=f"Path to sample-prs.yml (default: {DEFAULT_SAMPLE})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / DEFAULT_OUT_DIR,
        help=f"Directory to write per-PR artifacts (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--mode",
        choices=["live", "dry", "scaffold", "replay"],
        default="scaffold",
        help=(
            "live: POST to OpenRouter; dry: fetch diff and build payload but skip "
            "the POST; scaffold: write unfilled placeholders; replay: re-derive "
            "findings from a committed evaluation/bakeoff/replay/<pr_id>.json "
            "live-response capture (no token spend, deterministic output)."
        ),
    )
    parser.add_argument(
        "--config-path",
        default=None,
        help=(
            f"Path to the in-repo model config file (default: {DEFAULT_CONFIG_PATH}). "
            "Overrides the env-default; lets the runner respect swap-speed."
        ),
    )
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        metavar="PR_ID",
        help="Restrict to specific pr-<number> ids (may be repeated)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sample = load_sample(args.sample)
    if args.only:
        wanted = set(args.only)
        sample = [pr for pr in sample if pr.pr_id in wanted]
        if not sample:
            raise SystemExit(f"No sample PRs matched --only {args.only!r}")

    resolved = _resolve_models(_REPO_ROOT, args.config_path)
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip() or None

    print(
        f"openrouter bake-off: mode={args.mode} "
        f"model={resolved['review_model']} "
        f"reasoning_effort={resolved['reasoning_effort']} "
        f"web_search_mode={resolved['web_search_mode']} "
        f"sample_size={len(sample)}",
        flush=True,
    )

    summary: dict[str, int] = {}
    for pr in sample:
        artifact_path, status = _handle_pr(
            pr,
            mode=args.mode,
            out_dir=args.out_dir,
            model=resolved["review_model"],
            reasoning_effort=resolved["reasoning_effort"],
            web_search_mode=resolved["web_search_mode"],
            api_key=api_key,
        )
        summary[status] = summary.get(status, 0) + 1
        rel = (
            artifact_path.relative_to(_REPO_ROOT) if artifact_path.is_absolute() else artifact_path
        )
        print(f"  {pr.pr_id} [{status}] -> {rel}", flush=True)

    print("\nsummary: " + ", ".join(f"{k}={v}" for k, v in sorted(summary.items())))
    if args.mode == "live" and summary.get("error"):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
