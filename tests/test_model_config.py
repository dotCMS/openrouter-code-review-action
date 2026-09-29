"""Tests for the in-repo model config file (.openrouter-review.yml).

These tests pin AC 8 from seed.yaml: "Changing review or act model is a single
edit to the in-repo config file." They assert that:

* Editing only the YAML file flips ``ReviewConfig.selected_model`` for both
  modes — no Python code changes, no action input changes.
* Defaults (``deepseek/deepseek-v4-pro-0813`` review / ``anthropic/claude-opus-4.7`` act) apply when the file is absent.
* Malformed files surface a ``ConfigurationError`` instead of silently using
  the defaults.
* Explicit ``from_args`` overrides still win over the file (so a workflow can
  pin a model for a one-off run) but in normal use the file is the source of
  truth.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cli.core.config import ReviewConfig
from cli.core.exceptions import ConfigurationError
from cli.core.model_config import (
    DEFAULT_ACT_MODEL,
    DEFAULT_CONFIG_PATH,
    DEFAULT_REVIEW_MODEL,
    ModelConfig,
    load_model_config,
    resolve_config_path,
)


def _write_config(repo_root: Path, body: str) -> Path:
    target = repo_root / DEFAULT_CONFIG_PATH
    target.write_text(body, encoding="utf-8")
    return target


# --- pure loader -----------------------------------------------------------


def test_load_model_config_returns_defaults_when_file_missing(tmp_path: Path) -> None:
    cfg = load_model_config(repo_root=tmp_path)

    assert cfg.review_model == DEFAULT_REVIEW_MODEL
    assert cfg.act_model == DEFAULT_ACT_MODEL
    assert cfg.source_path is None


def test_load_model_config_reads_per_mode_models(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        "review:\n  model: anthropic/claude-opus-4.7\nact:\n  model: openai/gpt-5.4\n",
    )

    cfg = load_model_config(repo_root=tmp_path)

    assert cfg.review_model == "anthropic/claude-opus-4.7"
    assert cfg.act_model == "openai/gpt-5.4"
    assert cfg.source_path == (tmp_path / DEFAULT_CONFIG_PATH).resolve()


def test_load_model_config_partial_review_only_keeps_act_default(tmp_path: Path) -> None:
    _write_config(tmp_path, "review:\n  model: anthropic/claude-opus-4.7\n")

    cfg = load_model_config(repo_root=tmp_path)

    assert cfg.review_model == "anthropic/claude-opus-4.7"
    assert cfg.act_model == DEFAULT_ACT_MODEL


def test_load_model_config_rejects_malformed_yaml(tmp_path: Path) -> None:
    _write_config(tmp_path, ":\nthis: is: : invalid")

    with pytest.raises(ConfigurationError, match="Invalid YAML"):
        load_model_config(repo_root=tmp_path)


def test_load_model_config_rejects_unknown_keys(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        "review:\n  model: anthropic/claude-opus-4.7\n  bogus: 1\n",
    )

    with pytest.raises(ConfigurationError, match="unknown 'review' keys: bogus"):
        load_model_config(repo_root=tmp_path)


def test_load_model_config_rejects_invalid_reasoning_effort(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        "review:\n  model: anthropic/claude-opus-4.7\n  reasoning_effort: extreme\n",
    )

    with pytest.raises(ConfigurationError, match="reasoning_effort"):
        load_model_config(repo_root=tmp_path)


def test_load_model_config_rejects_web_search_mode_under_act(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        "act:\n  model: anthropic/claude-opus-4.7\n  web_search_mode: live\n",
    )

    with pytest.raises(ConfigurationError, match="web_search_mode"):
        load_model_config(repo_root=tmp_path)


def test_load_model_config_rejects_top_level_non_mapping(tmp_path: Path) -> None:
    _write_config(tmp_path, "- a\n- b\n")

    with pytest.raises(ConfigurationError, match="must be a YAML mapping"):
        load_model_config(repo_root=tmp_path)


def test_resolve_config_path_uses_default_when_unset(tmp_path: Path) -> None:
    assert resolve_config_path(tmp_path, None) == (tmp_path / DEFAULT_CONFIG_PATH).resolve()
    assert resolve_config_path(tmp_path, "") == (tmp_path / DEFAULT_CONFIG_PATH).resolve()


def test_resolve_config_path_honors_absolute_path(tmp_path: Path) -> None:
    abs_path = tmp_path / "elsewhere" / "models.yml"
    assert resolve_config_path(tmp_path, str(abs_path)) == abs_path


def test_model_config_for_mode_picks_correct_slug() -> None:
    cfg = ModelConfig(review_model="r/x", act_model="a/y")
    assert cfg.model_for_mode("review") == "r/x"
    assert cfg.model_for_mode("act") == "a/y"
    # Unknown mode falls back to review (defensive — schema disallows other modes)
    assert cfg.model_for_mode("anything-else") == "r/x"


# --- ReviewConfig integration (the AC 8 contract) --------------------------


@pytest.fixture()
def base_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Minimal env that lets ReviewConfig.from_args succeed in either mode."""

    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("GITHUB_WORKSPACE", str(tmp_path))
    # Avoid leaking host env vars into the loader
    monkeypatch.delenv("OPENROUTER_REVIEW_CONFIG", raising=False)
    monkeypatch.delenv("OPENROUTER_REVIEW_MODEL", raising=False)
    monkeypatch.delenv("OPENROUTER_ACT_MODEL", raising=False)
    return tmp_path


def test_review_config_uses_defaults_when_no_in_repo_file(base_env: Path) -> None:
    config = ReviewConfig.from_args(pr_number=1)

    assert config.review_model == DEFAULT_REVIEW_MODEL
    assert config.act_model == DEFAULT_ACT_MODEL
    assert config.selected_model == DEFAULT_REVIEW_MODEL


def test_review_config_picks_up_review_model_from_in_repo_file(base_env: Path) -> None:
    _write_config(base_env, "review:\n  model: anthropic/claude-opus-4.7\n")

    config = ReviewConfig.from_args(pr_number=1, mode="review")

    assert config.selected_model == "anthropic/claude-opus-4.7"


def test_swap_review_model_is_single_file_edit(base_env: Path) -> None:
    """The full AC 8 contract — swapping models requires only a YAML edit."""

    target = base_env / DEFAULT_CONFIG_PATH

    target.write_text("review:\n  model: anthropic/claude-opus-4.7\n", encoding="utf-8")
    first = ReviewConfig.from_args(pr_number=1, mode="review")
    assert first.selected_model == "anthropic/claude-opus-4.7"

    # The "single edit" — change one line in the YAML, nothing else.
    target.write_text("review:\n  model: openai/gpt-5.4\n", encoding="utf-8")
    second = ReviewConfig.from_args(pr_number=1, mode="review")

    assert second.selected_model == "openai/gpt-5.4"
    # No other config drift from that single edit:
    assert second.review_model == "openai/gpt-5.4"
    assert second.act_model == DEFAULT_ACT_MODEL


def test_swap_act_model_is_single_file_edit(
    base_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same contract for act mode — the in-repo file is the source of truth."""

    monkeypatch.setenv("DOTBOT_MODE", "act")
    target = base_env / DEFAULT_CONFIG_PATH

    target.write_text("act:\n  model: anthropic/claude-opus-4.7\n", encoding="utf-8")
    first = ReviewConfig.from_args(pr_number=1, mode="act", act_instructions="x")
    assert first.selected_model == "anthropic/claude-opus-4.7"

    target.write_text("act:\n  model: google/gemini-2.5-pro\n", encoding="utf-8")
    second = ReviewConfig.from_args(pr_number=1, mode="act", act_instructions="x")
    assert second.selected_model == "google/gemini-2.5-pro"


def test_explicit_kwarg_still_wins_over_in_repo_file(base_env: Path) -> None:
    _write_config(base_env, "review:\n  model: anthropic/claude-opus-4.7\n")

    config = ReviewConfig.from_args(pr_number=1, mode="review", review_model="openai/gpt-5.4")

    assert config.selected_model == "openai/gpt-5.4"


def test_in_repo_file_can_override_reasoning_effort_and_web_search(
    base_env: Path,
) -> None:
    _write_config(
        base_env,
        "review:\n"
        "  model: anthropic/claude-opus-4.7\n"
        "  reasoning_effort: high\n"
        "  web_search_mode: cached\n",
    )

    config = ReviewConfig.from_args(pr_number=1, mode="review")

    assert config.reasoning_effort == "high"
    assert config.web_search_mode == "cached"


def test_custom_config_path_resolved_relative_to_repo_root(
    base_env: Path,
) -> None:
    custom = base_env / "ci" / "models.yml"
    custom.parent.mkdir(parents=True)
    custom.write_text("review:\n  model: anthropic/claude-sonnet-4.5\n", encoding="utf-8")

    config = ReviewConfig.from_args(pr_number=1, mode="review", config_path="ci/models.yml")

    assert config.selected_model == "anthropic/claude-sonnet-4.5"


def test_invalid_in_repo_file_raises_configuration_error(base_env: Path) -> None:
    _write_config(base_env, "review:\n  model: 12345\n")

    with pytest.raises(ConfigurationError, match="must be a non-empty string"):
        ReviewConfig.from_args(pr_number=1, mode="review")


def test_review_models_list_parsed_from_in_repo_file(base_env: Path) -> None:
    _write_config(
        base_env,
        "review:\n  model: deepseek/deepseek-v4-pro-0813\n  models:\n"
        "    - openai/gpt-5\n    - google/gemini-2.5-pro\n",
    )

    config = ReviewConfig.from_args(pr_number=1, mode="review")

    assert config.review_models == ("openai/gpt-5", "google/gemini-2.5-pro")
    assert config.selected_review_models == (
        DEFAULT_REVIEW_MODEL,
        "openai/gpt-5",
        "google/gemini-2.5-pro",
    )


def test_review_models_env_comma_separated(base_env: Path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("DOTBOT_REVIEW_MODELS", "openai/gpt-5, google/gemini-2.5-pro")

    config = ReviewConfig.from_args(pr_number=1, mode="review")

    assert config.review_models == ("openai/gpt-5", "google/gemini-2.5-pro")
    assert config.selected_review_models == (
        DEFAULT_REVIEW_MODEL,
        "openai/gpt-5",
        "google/gemini-2.5-pro",
    )


def test_review_models_deduplicated_and_empty_by_default(base_env: Path) -> None:
    config = ReviewConfig.from_args(pr_number=1, mode="review")
    assert config.review_models == ()
    assert config.selected_review_models == (DEFAULT_REVIEW_MODEL,)


def test_review_models_ignored_under_act(base_env: Path) -> None:
    _write_config(
        base_env,
        "review:\n  models:\n    - openai/gpt-5\n",
    )

    config = ReviewConfig.from_args(pr_number=1, mode="act")

    assert config.selected_model == DEFAULT_ACT_MODEL
    assert config.review_models == ("openai/gpt-5",)


def test_review_models_rejects_non_list(base_env: Path) -> None:
    _write_config(base_env, "review:\n  models: openai/gpt-5\n")

    with pytest.raises(ConfigurationError, match="non-empty list"):
        ReviewConfig.from_args(pr_number=1, mode="review")


def test_review_models_rejects_empty_list(base_env: Path) -> None:
    _write_config(base_env, "review:\n  models: []\n")

    with pytest.raises(ConfigurationError, match="non-empty list"):
        ReviewConfig.from_args(pr_number=1, mode="review")


# --- DOTBOT_ACT_MODEL: generated act config outside the checkout -------------


def test_act_model_from_generated_config_outside_repo(
    base_env: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirror the dotbot-act workflow's DOTBOT_ACT_MODEL override.

    The workflow renders ``act:\n  model: <slug>\n`` to a file in RUNNER_TEMP
    (outside GITHUB_WORKSPACE) and passes it as the ``config_path`` action
    input, which the action forwards as ``OPENROUTER_REVIEW_CONFIG``. The
    variable must win over the repo's committed ``act.model``, and the
    generated file must be usable from outside the checkout.
    """

    _write_config(base_env, "act:\n  model: anthropic/claude-opus-4.7\n")

    runner_temp = tmp_path_factory.mktemp("runner-temp")
    generated = runner_temp / "dotbot-act-model.yml"
    generated.write_text("act:\n  model: openai/gpt-5.4\n", encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_REVIEW_CONFIG", str(generated))

    config = ReviewConfig.from_args(pr_number=1, mode="act")

    assert config.act_model == "openai/gpt-5.4"
    assert config.selected_model == "openai/gpt-5.4"
    # Act-only file: review fields fall back to defaults. Act runs never read
    # them, and the absolute out-of-checkout path must not break loading.
    assert config.review_model == DEFAULT_REVIEW_MODEL


def test_act_model_generated_config_without_review_block_is_valid(
    tmp_path: Path,
) -> None:
    """The generated act-only file is a complete, valid config on its own."""

    generated = tmp_path / "dotbot-act-model.yml"
    generated.write_text("act:\n  model: google/gemini-2.5-pro\n", encoding="utf-8")

    cfg = load_model_config(config_path=str(generated))

    assert cfg.act_model == "google/gemini-2.5-pro"
    assert cfg.review_model == DEFAULT_REVIEW_MODEL
