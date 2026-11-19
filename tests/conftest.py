from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the repository root is on sys.path so `cli` is importable in tests
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _dummy_model_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide dummy provider API keys so config validation passes.

    ReviewConfig validates that the active provider's API key is present.
    Tests construct real configs from the environment, so without this
    fixture the suite only passes on machines that happen to export
    OPENROUTER_API_KEY / OPENAI_API_KEY. Tests that assert on missing-key
    validation errors call ``monkeypatch.delenv`` themselves and override
    this fixture's value.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


@pytest.fixture(autouse=True)
def _no_ci_step_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests from appending to the real GitHub Actions step summary in CI."""
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
