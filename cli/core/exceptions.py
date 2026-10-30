from __future__ import annotations


class DotBotReviewError(Exception):
    """Base exception for dotbot review operations."""


class GitHubAPIError(DotBotReviewError):
    """GitHub API related errors."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ConfigurationError(DotBotReviewError):
    """Configuration validation errors."""


class PatchParsingError(DotBotReviewError):
    """Patch parsing related errors."""


class DotBotExecutionError(DotBotReviewError):
    """Model execution related errors."""


class PromptError(DotBotReviewError):
    """Prompt composition or loading errors."""


class ReviewContractError(DotBotReviewError):
    """Structured review payload or metadata contract violations."""


class CorrectableReviewError(ReviewContractError):
    """A response error whose validated decisions can be preserved during correction."""
