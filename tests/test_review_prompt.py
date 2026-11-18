from pathlib import Path

import pytest

from cli.core.config import ReviewConfig
from cli.core.exceptions import PromptError
from cli.review.review_prompt import load_guidelines, load_panel_instructions


def _make_review_config() -> ReviewConfig:
    return ReviewConfig(github_token="token", repository="owner/repo", mode="review")


def test_load_guidelines_include_repo_standard_comment_format() -> None:
    guidelines = load_guidelines(_make_review_config())

    assert "REVIEW COMMENT FORMAT (REPO STANDARD):" in guidelines
    assert "**Current code:**" in guidelines
    assert "**Problem:** Brief description (max 20 words)." in guidelines
    assert "**Fix:**" in guidelines
    assert "severity emoji + priority tag: 🔴 [P0]/[P1], 🟡 [P2], ⚪ [P3]" in guidelines
    assert "Include file path and line number in the title when possible." in guidelines
    assert "Skip comments for formatting-only issues, personal style preferences" in guidelines

    # Preserve required JSON output fields.
    assert '"carried_forward": [' in guidelines
    assert '"comment_id": "<prior review comment id>"' in guidelines
    assert '"current_evidence": "<exact current-code snippet copied verbatim>"' in guidelines
    assert '"overall_correctness": "patch is correct" | "patch is incorrect"' in guidelines
    assert '"code_location": {' in guidelines


def test_panel_prompt_is_loaded_from_editable_markdown() -> None:
    expected = Path("prompts/review_panel.md").read_text(encoding="utf-8")
    assert load_panel_instructions(_make_review_config()) == expected
    assert "REVIEW COMMENT FORMAT (REPO STANDARD)" in expected
    assert "untrusted evidence" in expected
    assert "P2/P3" in expected


def test_missing_prompt_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> str:
        raise OSError("missing")

    monkeypatch.setattr(Path, "read_text", fail)
    with pytest.raises(PromptError, match="review_panel.md"):
        load_panel_instructions(_make_review_config())


def test_default_prompts_load_application_focus_after_base() -> None:
    guidelines = load_guidelines(_make_review_config())
    assert guidelines.index("## Output schema") < guidelines.index(
        "# Application code review focus"
    )
    assert "Infrastructure-as-code" not in guidelines


def test_iac_prompt_replaces_application_focus() -> None:
    guidelines = load_guidelines(
        ReviewConfig(github_token="t", repository="o/r", mode="review", prompts=("iac",))
    )
    assert "REVIEW COMMENT FORMAT (REPO STANDARD):" in guidelines
    assert "# Infrastructure-as-code review focus" in guidelines
    assert "try-with-resources" not in guidelines


def test_prompt_names_accept_md_suffix() -> None:
    def load(*prompts: str) -> str:
        return load_guidelines(
            ReviewConfig(github_token="t", repository="o/r", mode="review", prompts=prompts)
        )

    assert load("code_java.md", "iac.md") == load("code_java", "iac")


def test_prompts_env_is_comma_separated_and_empty_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cli.core.config import _config_values_from_environment

    monkeypatch.setenv("DOTBOT_PROMPTS", " code_java, iac ,")
    assert _config_values_from_environment()["prompts"] == ("code_java", "iac")
    monkeypatch.setenv("DOTBOT_PROMPTS", "")
    assert _config_values_from_environment()["prompts"] == ("code_java",)


@pytest.mark.parametrize("name", ["../review", "IaC", "nope"])
def test_unknown_or_unsafe_prompt_name_fails_fast(name: str) -> None:
    config = ReviewConfig(github_token="t", repository="o/r", mode="review", prompts=(name,))
    with pytest.raises(PromptError):
        load_guidelines(config)


def test_prompts_env_and_kwarg_reach_review_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOTBOT_PROMPTS", "iac")
    base = {"github_token": "t", "repository": "o/r", "pr_number": 1, "mode": "review"}
    assert ReviewConfig.from_args(**base).prompts == ("iac",)
    assert ReviewConfig.from_args(**base, prompts=["code_java", "iac"]).prompts == (
        "code_java",
        "iac",
    )
    assert ReviewConfig.from_args(**base, prompts="code_java, iac").prompts == ("code_java", "iac")
    assert ReviewConfig.from_args(**base, prompts="").prompts == ("code_java",)
