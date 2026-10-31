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
