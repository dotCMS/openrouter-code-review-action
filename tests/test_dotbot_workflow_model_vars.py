"""Workflow-level tests for the DOTBOT_REVIEW_MODELS org/repo variable.

`DOTBOT_REVIEW_MODELS` is resolved by a shell step in dotbot-review.yml, not by
Python code, so these tests execute that step's real `run:` script with bash and
assert the rendered `.openrouter-review.yml`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from cli.core.model_config import load_model_config

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")


def test_review_roster_override_preserves_panel_and_act_settings(tmp_path: Path) -> None:
    workflow = yaml.safe_load(Path(".github/workflows/dotbot-review.yml").read_text())
    step = next(
        step
        for step in workflow["jobs"]["review"]["steps"]
        if str(step.get("name", "")).startswith("Override review roster")
    )
    config_path = tmp_path / ".openrouter-review.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "review": {
                    "model": "old/primary",
                    "models": ["old/secondary"],
                    "discussion_rounds": 3,
                    "decision": {"policy": "lead"},
                },
                "act": {"model": "kept/actor"},
            }
        )
    )
    result = subprocess.run(
        ["bash", "-c", step["run"]],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={**os.environ, "REVIEW_MODELS": "new/primary,new/secondary"},
        check=False,
    )
    assert result.returncode == 0, result.stderr
    config = load_model_config(config_path=str(config_path))
    assert config.review_model == "new/primary"
    assert config.review_models == ("new/secondary",)
    assert config.review_discussion_rounds == 3
    assert config.review_decision_policy == "lead"
    assert config.act_model == "kept/actor"


@pytest.mark.parametrize("module", ["yaml.py", "pathlib.py"])
def test_review_roster_override_does_not_import_pr_checkout_modules(
    tmp_path: Path, module: str
) -> None:
    workflow = yaml.safe_load(Path(".github/workflows/dotbot-review.yml").read_text())
    step = next(
        step
        for step in workflow["jobs"]["review"]["steps"]
        if str(step.get("name", "")).startswith("Override review roster")
    )
    (tmp_path / module).write_text(
        'open("shadow-imported", "w").write("executed")\nraise RuntimeError("PR module executed")\n'
    )
    result = subprocess.run(
        ["bash", "-c", step["run"]],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "REVIEW_MODELS": "new/primary,new/secondary"},
    )
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "shadow-imported").exists()
    assert load_model_config(repo_root=tmp_path).review_models == ("new/secondary",)
