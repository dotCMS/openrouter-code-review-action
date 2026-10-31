from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def render_review_summary_metadata(
    reviewed_head_sha: str,
    *,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> str:
    payload: dict[str, Any] = {"reviewed_head_sha": reviewed_head_sha}
    if model is not None:
        payload["model"] = model
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    return (
        "<!-- dotbot-review-meta "
        + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        + " -->"
    )


def render_review_run_reference(head_sha: str) -> str:
    run_id = os.environ.get("GITHUB_RUN_ID")
    repository = os.environ.get("GITHUB_REPOSITORY")
    if not run_id or not repository:
        return ""
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "1")
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    metadata = json.dumps({"run_id": run_id, "attempt": attempt, "head_sha": head_sha})
    url = f"{server}/{repository}/actions/runs/{run_id}/attempts/{attempt}"
    return (
        f"<!-- dotbot-review-run {metadata} -->\n[Workflow run {run_id} — attempt {attempt}]({url})"
    )


def review_run_identity(body: str) -> tuple[int, int] | None:
    lines = body.split("\n", 2)
    prefix = "<!-- dotbot-review-run "
    if len(lines) < 2 or not lines[1].startswith(prefix) or not lines[1].endswith(" -->"):
        return None
    try:
        metadata = json.loads(lines[1][len(prefix) : -4])
    except json.JSONDecodeError:
        return None
    if not isinstance(metadata, dict):
        return None
    run_id, attempt = metadata.get("run_id"), metadata.get("attempt")
    if not isinstance(run_id, str) or not isinstance(attempt, str):
        return None
    if (
        not run_id.isascii()
        or not run_id.isdecimal()
        or not attempt.isascii()
        or not attempt.isdecimal()
    ):
        return None
    return int(run_id), int(attempt)


@dataclass(frozen=True)
class ReviewArtifacts:
    repo_root: Path
    context_dir_name: str

    @property
    def base_dir(self) -> Path:
        return (self.repo_root / self.context_dir_name).resolve()

    @property
    def pr_metadata_path(self) -> Path:
        return self.base_dir / "pr.md"

    @property
    def review_comments_path(self) -> Path:
        return self.base_dir / "review_comments.md"

    @property
    def anchor_maps_path(self) -> Path:
        return self.base_dir / "anchor_maps.json"

    def relative_to_repo_root(self, path: Path) -> Path:
        try:
            return path.relative_to(self.repo_root)
        except ValueError:
            return Path(self.context_dir_name) / path.name
