"""Workflow-level tests for the DOTBOT_* org/repo model variables.

`DOTBOT_ACT_MODEL` is resolved by a shell step in dotbot-act.yml, not by Python
code, so these tests execute that step's real `run:` script with bash and assert
the generated config file the action is then pointed at via `config_path`.

Covered:

* a `~`-prefixed OpenRouter "latest" alias slug (the value the dotCMS org
  variable actually carries, e.g. ``~deepseek/deepseek-flash-latest``) is
  accepted and round-trips through ``load_model_config``;
* surrounding/embedded whitespace is stripped;
* a blank variable falls back to the in-repo file (no `config_path` output);
* a non-slug value fails the run instead of emitting an unparseable config.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from cli.core.model_config import load_model_config

ACT_WORKFLOW = Path(".github/workflows/dotbot-act.yml")
RENDER_STEP_PREFIX = "Override act model from DOTBOT_ACT_MODEL"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")


def _render_step_script() -> str:
    workflow = yaml.safe_load(ACT_WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["act"]["steps"]
    for step in steps:
        if str(step.get("name", "")).startswith(RENDER_STEP_PREFIX):
            assert step["id"] == "act_model"
            assert step["if"] == "${{ vars.DOTBOT_ACT_MODEL != '' }}"
            assert step["env"]["ACT_MODEL"] == "${{ vars.DOTBOT_ACT_MODEL }}"
            return str(step["run"])
    raise AssertionError(f"no step named '{RENDER_STEP_PREFIX} ...' in {ACT_WORKFLOW}")


def _run_step(
    tmp_path: Path,
    act_model: str | None,
) -> tuple[subprocess.CompletedProcess[str], Path, dict[str, str]]:
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()

    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(tmp_path),
        "RUNNER_TEMP": str(runner_temp),
        "GITHUB_OUTPUT": str(tmp_path / "github_output"),
    }
    if act_model is not None:
        env["ACT_MODEL"] = act_model

    result = subprocess.run(  # noqa: S603 - fixed, repo-owned script under test
        ["bash", "-c", _render_step_script()],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        check=False,
    )

    generated = runner_temp / "dotbot-act-model.yml"
    outputs: dict[str, str] = {}
    output_file = tmp_path / "github_output"
    if output_file.exists():
        for line in output_file.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            outputs[key] = value

    return result, generated, outputs


@pytest.mark.parametrize(
    "slug",
    [
        "~deepseek/deepseek-flash-latest",
        "~z-ai/glm-latest",
        "anthropic/claude-opus-4.7",
        "google/gemini-2.5-pro:free",
    ],
)
def test_render_step_accepts_openrouter_slugs(tmp_path: Path, slug: str) -> None:
    result, generated, outputs = _run_step(tmp_path, slug)

    assert result.returncode == 0, result.stderr
    cfg = load_model_config(config_path=str(generated))
    assert cfg.act_model == slug
    # The action reads this path from the config_path input.
    assert outputs["config_path"] == str(generated)


def test_render_step_strips_whitespace(tmp_path: Path) -> None:
    result, generated, _ = _run_step(tmp_path, "  ~deepseek/deepseek-flash-latest \n")

    assert result.returncode == 0, result.stderr
    assert load_model_config(config_path=str(generated)).act_model == (
        "~deepseek/deepseek-flash-latest"
    )


def test_render_step_falls_back_when_variable_blank(tmp_path: Path) -> None:
    """Variable set but whitespace-only: warn, emit no config_path, keep the file."""
    result, generated, outputs = _run_step(tmp_path, "   ")

    assert result.returncode == 0
    assert "::warning::" in result.stdout
    assert "config_path" not in outputs
    assert not generated.exists()


def test_render_step_fails_loudly_on_non_slug(tmp_path: Path) -> None:
    """A malformed variable must not silently produce a broken config file."""
    result, generated, outputs = _run_step(tmp_path, "evil: [1,2]")

    assert result.returncode == 2
    assert "::error::" in result.stdout
    assert "config_path" not in outputs
    assert not generated.exists()
