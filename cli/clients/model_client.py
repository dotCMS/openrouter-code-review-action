"""Typed protocol and factory for the model-execution clients.

Workflows depend on this protocol rather than a concrete client so the
Codex SDK client and the OpenRouter client are interchangeable, and so
tests can inject fakes without touching provider imports.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from ..core.config import ReviewConfig


@runtime_checkable
class ModelClientProtocol(Protocol):
    """The execution surface both Codex and OpenRouter clients expose."""

    def execute_text(
        self,
        prompt: str,
        *,
        model_name: str | None = None,
        reasoning_effort: str | None = None,
        suppress_stream: bool = False,
        sandbox_mode: str = "read-only",
        resume_thread_id: str | None = None,
    ) -> str: ...

    def execute_structured(
        self,
        prompt: str,
        *,
        output_schema: dict[str, object],
        schema_prompt: str = "Produce the JSON output now.",
        output_validator: Callable[[str], object] | None = None,
        model_name: str | None = None,
        reasoning_effort: str | None = None,
        suppress_stream: bool = False,
        sandbox_mode: str = "read-only",
        resume_thread_id: str | None = None,
    ) -> str: ...


def create_model_client(config: ReviewConfig) -> ModelClientProtocol:
    """Build the model client selected by ``config.model_provider``.

    ``openai`` routes to the legacy Codex SDK client; everything else
    (notably the default ``openrouter``) routes to the OpenRouter
    chat-completions client with the built-in tool-calling agent loop.
    """
    if config.model_provider == "openai":
        from .codex_client import CodexClient

        return CodexClient(config)
    from .openrouter_client import OpenRouterClient

    return OpenRouterClient(config)
