from __future__ import annotations

import email.message
import io
import json
import urllib.request
from pathlib import Path
from typing import Any, cast

import pytest

from cli.clients import openrouter_client
from cli.clients.openrouter_client import OpenRouterClient
from cli.clients.openrouter_thread_store import OpenRouterThreadStore
from cli.core.config import ReviewConfig
from cli.core.exceptions import ConfigurationError, DotBotExecutionError
from cli.review.coordinator import ReviewCoordinator


def _bedrock_config(tmp_path: Path, **overrides: Any) -> ReviewConfig:
    values: dict[str, Any] = dict(
        github_token="token",
        repository="owner/repo",
        pr_number=7,
        model_provider="bedrock",
        bedrock_api_key="bedrock-api-key-test",
        bedrock_region="us-east-1",
        review_model="deepseek.v3.2",
        stream_output=False,
        web_search_mode="live",
        repo_root=tmp_path,
    )
    values.update(overrides)
    return ReviewConfig(**values)


class _FakeResponse(io.BytesIO):
    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class _CapturingUrlopen:
    """Stands in for urllib.request.urlopen and records each request."""

    def __init__(self) -> None:
        self.requests: list[urllib.request.Request] = []

    def __call__(self, request: urllib.request.Request, timeout: float) -> _FakeResponse:
        self.requests.append(request)
        body = {"choices": [{"message": {"role": "assistant", "content": "done"}}]}
        return _FakeResponse(json.dumps(body).encode("utf-8"))


def _payload(request: urllib.request.Request) -> dict[str, Any]:
    data = request.data
    assert isinstance(data, bytes)
    decoded = json.loads(data)
    assert isinstance(decoded, dict)
    return decoded


def test_bedrock_review_config_is_accepted() -> None:
    config = ReviewConfig.from_args(
        github_token="token",
        repository="owner/repo",
        pr_number=1,
        model_provider="bedrock",
        bedrock_api_key="bedrock-api-key-test",
    )

    assert config.model_provider == "bedrock"
    assert config.bedrock_region == "us-east-1"


def test_bedrock_requires_an_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)

    with pytest.raises(ConfigurationError, match="Missing AWS_BEARER_TOKEN_BEDROCK"):
        ReviewConfig.from_args(
            github_token="token",
            repository="owner/repo",
            pr_number=1,
            model_provider="bedrock",
        )


@pytest.mark.parametrize("region", ["evil.example.com/x", "us-east-1.attacker", "", "US-EAST-1"])
def test_bedrock_rejects_anything_but_a_region_name(region: str) -> None:
    with pytest.raises(ConfigurationError, match="Invalid Bedrock region"):
        ReviewConfig.from_args(
            github_token="token",
            repository="owner/repo",
            pr_number=1,
            model_provider="bedrock",
            bedrock_api_key="bedrock-api-key-test",
            bedrock_region=region,
        )


def test_environment_supplies_bedrock_key_and_region(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("PR_NUMBER", "1")
    monkeypatch.setenv("DOTBOT_PROVIDER", "bedrock")
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-api-key-test")
    monkeypatch.setenv("AWS_REGION", "us-west-2")

    config = ReviewConfig.from_environment()

    assert config.bedrock_api_key == "bedrock-api-key-test"
    assert config.bedrock_region == "us-west-2"


def test_bedrock_request_goes_to_bedrock_with_its_own_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    urlopen = _CapturingUrlopen()
    monkeypatch.setattr(openrouter_client.urllib.request, "urlopen", urlopen)
    client = OpenRouterClient(
        _bedrock_config(tmp_path, bedrock_region="us-west-2"),
        thread_store=OpenRouterThreadStore(tmp_path / "threads"),
    )

    assert client.execute_text("inspect the repo", reasoning_effort="high") == "done"

    request = urlopen.requests[0]
    assert request.full_url == (
        "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1/chat/completions"
    )
    headers = {key.lower(): value for key, value in request.header_items()}
    assert headers["authorization"] == "Bearer bedrock-api-key-test"
    assert "http-referer" not in headers
    assert "x-title" not in headers
    payload = _payload(request)
    assert payload["reasoning_effort"] == "high"
    assert "reasoning" not in payload
    # Bedrock has no :online web search, even with web_search_mode=live.
    assert payload["model"] == "deepseek.v3.2"


def test_openrouter_request_is_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    urlopen = _CapturingUrlopen()
    monkeypatch.setattr(openrouter_client.urllib.request, "urlopen", urlopen)
    client = OpenRouterClient(
        _bedrock_config(tmp_path, model_provider="openrouter", openrouter_api_key="or-key"),
        thread_store=OpenRouterThreadStore(tmp_path / "threads"),
    )

    client.execute_text("inspect the repo")

    request = urlopen.requests[0]
    assert request.full_url == "https://openrouter.ai/api/v1/chat/completions"
    payload = _payload(request)
    assert payload["reasoning"] == {"effort": "medium"}
    assert payload["model"] == "deepseek.v3.2:online"


def test_bedrock_http_errors_name_bedrock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(request: urllib.request.Request, timeout: float) -> _FakeResponse:
        raise urllib.error.HTTPError(
            request.full_url,
            403,
            "Forbidden",
            hdrs=email.message.Message(),
            fp=io.BytesIO(b"denied"),
        )

    monkeypatch.setattr(openrouter_client.urllib.request, "urlopen", refuse)
    client = OpenRouterClient(
        _bedrock_config(tmp_path), thread_store=OpenRouterThreadStore(tmp_path / "threads")
    )

    with pytest.raises(DotBotExecutionError, match="^Bedrock HTTP 403"):
        client.execute_text("inspect the repo")


def test_bedrock_key_is_redacted_from_panel_logs(tmp_path: Path) -> None:
    coordinator = ReviewCoordinator(_bedrock_config(tmp_path), lambda config: cast(Any, None))

    assert coordinator._redact("key=bedrock-api-key-test") == "key=[REDACTED]"


def test_action_mints_the_bedrock_key_without_hiding_user_site_packages() -> None:
    # Runners install requirements with pip --user; python -I would hide them.
    action = Path(__file__).resolve().parent.parent.joinpath("action.yml").read_text()
    mint = next(line for line in action.splitlines() if "bedrock_api_key.py" in line)

    assert "python3 -P " in mint
    assert "python3 -I " not in mint
