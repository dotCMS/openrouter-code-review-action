from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from .exceptions import ConfigurationError
from .model_config import (
    DEFAULT_ACT_MODEL,
    DEFAULT_CONFIG_PATH,
    DEFAULT_REVIEW_MODEL,
    ModelConfig,
    load_model_config,
)

_CONFIG_OVERRIDE_KEYS = frozenset(
    {
        "github_token",
        "github_approval_token",
        "repository",
        "pr_number",
        "mode",
        "model_provider",
        "openai_api_key",
        "openrouter_api_key",
        "model_name",
        "review_model",
        "review_models",
        "act_model",
        "config_path",
        "reasoning_effort",
        "web_search_mode",
        "act_instructions",
        "debug_level",
        "stream_output",
        "dry_run",
        "model_timeout_seconds",
        "additional_prompt",
        "repo_root",
        "context_dir_name",
        "allowed_commenter_associations",
        "resolve_stale_threads",
    }
)

_DEFAULT_ALLOWED_COMMENTER_ASSOCIATIONS = ("MEMBER", "OWNER", "COLLABORATOR")
_VALID_COMMENTER_ASSOCIATIONS = frozenset(
    {
        "COLLABORATOR",
        "CONTRIBUTOR",
        "FIRST_TIMER",
        "FIRST_TIME_CONTRIBUTOR",
        "MANNEQUIN",
        "MEMBER",
        "NONE",
        "OWNER",
    }
)


class _ReviewConfigValues(TypedDict):
    github_token: str
    github_approval_token: str
    repository: str
    pr_number: int | None
    mode: str
    model_provider: str
    openai_api_key: str
    openrouter_api_key: str
    model_name: str
    review_model: str
    review_models: tuple[str, ...]
    act_model: str
    config_path: str
    reasoning_effort: str
    web_search_mode: str
    act_instructions: str
    debug_level: int
    stream_output: bool
    dry_run: bool
    model_timeout_seconds: int
    additional_prompt: str
    repo_root: Path | None
    context_dir_name: str
    allowed_commenter_associations: tuple[str, ...]
    resolve_stale_threads: bool


@dataclass
class ReviewConfig:
    """Configuration for code review operations."""

    github_token: str
    repository: str
    pr_number: int | None = None
    mode: str = "review"  # "review" or "act"
    model_provider: str = "openrouter"
    openai_api_key: str = ""
    openrouter_api_key: str = ""
    model_name: str = "gpt-5.4"
    review_model: str = DEFAULT_REVIEW_MODEL
    review_models: tuple[str, ...] = ()
    act_model: str = DEFAULT_ACT_MODEL
    config_path: str = DEFAULT_CONFIG_PATH
    reasoning_effort: str = "medium"
    web_search_mode: str = "live"
    act_instructions: str = ""
    debug_level: int = 0
    stream_output: bool = True
    dry_run: bool = False
    # Wall-clock budget for a single reviewer model pass. 0 disables the cap.
    model_timeout_seconds: int = 900
    additional_prompt: str = ""
    repo_root: Path | None = None
    context_dir_name: str = ".dotbot-context"
    allowed_commenter_associations: tuple[str, ...] = _DEFAULT_ALLOWED_COMMENTER_ASSOCIATIONS
    # PAT owned by the machine user (e.g. dotCMS-Machine-User); when set and
    # every reviewer model agrees the patch is correct, the PR is approved as
    # that user. Empty disables auto-approval.
    github_approval_token: str = ""
    # When True, prior unresolved dotbot threads that the reviewer model saw
    # but declined to carry forward are resolved (closed) on GitHub.
    resolve_stale_threads: bool = False

    @classmethod
    def from_environment(cls) -> ReviewConfig:
        """Create configuration from environment variables."""
        values = _config_values_from_environment()
        _apply_in_repo_model_config(values, explicit_keys=set())
        return cls._from_values(values)

    @classmethod
    def from_args(cls, **kwargs: Any) -> ReviewConfig:
        """Create configuration from keyword arguments."""
        unknown = sorted(
            key
            for key, value in kwargs.items()
            if value is not None and key not in _CONFIG_OVERRIDE_KEYS
        )
        if unknown:
            joined = ", ".join(unknown)
            raise ConfigurationError(f"Unknown configuration arguments: {joined}")

        values = _config_values_from_environment()
        _apply_config_overrides(values, kwargs)

        repo_root = values.get("repo_root")
        if isinstance(repo_root, str):
            values["repo_root"] = Path(repo_root).resolve()

        explicit_keys = {key for key, value in kwargs.items() if value is not None}
        _apply_in_repo_model_config(values, explicit_keys=explicit_keys)

        return cls._from_values(values)

    @classmethod
    def from_github_event(cls, event: Mapping[str, Any]) -> ReviewConfig:
        """Create configuration for a GitHub Actions event payload."""
        pr_number = cls.extract_pr_number_from_event(event)
        if not pr_number:
            raise ConfigurationError("This workflow must be triggered by a PR-related event")
        return cls.from_args(pr_number=pr_number)

    def validate(self) -> None:
        """Validate the configuration."""
        if not self.github_token:
            raise ConfigurationError("GitHub token is required")

        if not self.repository or "/" not in self.repository:
            raise ConfigurationError("Repository must be in format 'owner/repo'")

        if self.pr_number is not None and self.pr_number <= 0:
            raise ConfigurationError("PR number must be positive")

        if self.mode not in ("review", "act"):
            raise ConfigurationError(f"Invalid mode: {self.mode}. Must be 'review' or 'act'")

        if self.mode == "review" and self.pr_number is None:
            raise ConfigurationError("PR number is required in review mode")

        if self.debug_level < 0:
            raise ConfigurationError("Debug level must be non-negative")

        if self.model_timeout_seconds < 0:
            raise ConfigurationError("model_timeout_seconds must be non-negative")

        if self.web_search_mode not in ("disabled", "cached", "live"):
            raise ConfigurationError(
                f"Invalid web_search_mode: {self.web_search_mode}. "
                "Must be 'disabled', 'cached', or 'live'"
            )

        if self.model_provider == "openrouter":
            if not self.openrouter_api_key.strip():
                raise ConfigurationError(
                    "Missing OPENROUTER_API_KEY for model provider 'openrouter'"
                )
        elif self.model_provider == "openai":
            if not self.openai_api_key.strip():
                raise ConfigurationError("Missing OPENAI_API_KEY for model provider 'openai'")
        else:
            raise ConfigurationError(
                f"Invalid model_provider: {self.model_provider}. Must be 'openrouter' or 'openai'"
            )

        associations = self.allowed_commenter_associations
        if not associations:
            raise ConfigurationError(
                "DOTBOT_ALLOWED_COMMENTER_ASSOCIATIONS must include at least one value"
            )

        invalid_associations = sorted(
            association
            for association in associations
            if association not in _VALID_COMMENTER_ASSOCIATIONS
        )
        if invalid_associations:
            joined = ", ".join(invalid_associations)
            raise ConfigurationError(
                "Invalid DOTBOT_ALLOWED_COMMENTER_ASSOCIATIONS values: "
                f"{joined}. Allowed values: {', '.join(sorted(_VALID_COMMENTER_ASSOCIATIONS))}"
            )

    def is_commenter_allowed(self, author_association: str | None) -> bool:
        """Return True when the comment author is allowed to trigger edit mode."""
        if not author_association:
            return False
        normalized = author_association.strip().upper()
        return normalized in self.allowed_commenter_associations

    @staticmethod
    def extract_pr_number_from_event(event: Mapping[str, Any]) -> int | None:
        """Extract PR number from a GitHub event payload."""
        pull_request = event.get("pull_request")
        if isinstance(pull_request, Mapping):
            number = pull_request.get("number")
            if isinstance(number, (int, float, str, bytes, bytearray)):
                try:
                    parsed = int(number)
                    return parsed if parsed > 0 else None
                except (TypeError, ValueError):
                    return None
            return None

        issue = event.get("issue")
        if isinstance(issue, Mapping) and issue.get("pull_request"):
            number = issue.get("number")
            if isinstance(number, (int, float, str, bytes, bytearray)):
                try:
                    parsed = int(number)
                    return parsed if parsed > 0 else None
                except (TypeError, ValueError):
                    return None
            return None
        return None

    @classmethod
    def _from_values(cls, values: _ReviewConfigValues) -> ReviewConfig:
        config = cls(**values)
        config.validate()
        return config

    @property
    def owner(self) -> str:
        """Extract owner from repository string."""
        return self.repository.split("/", 1)[0]

    @property
    def repo_name(self) -> str:
        """Extract repository name from repository string."""
        return self.repository.split("/", 1)[1]

    @property
    def resolved_repo_root(self) -> Path:
        """Return the configured repository root, defaulting to the current directory."""
        return self.repo_root or Path(".").resolve()

    @property
    def resolved_context_dir_name(self) -> str:
        """Return the configured context directory name with the default fallback applied."""
        return self.context_dir_name or ".dotbot-context"

    @property
    def selected_model(self) -> str:
        """Return the OpenRouter model slug to use for the active mode.

        ``act`` mode reads from :attr:`act_model`; everything else reads from
        :attr:`review_model`. Both fields are populated by the in-repo
        ``.openrouter-review.yml`` (or whatever ``config_path`` points at) so
        flipping the active model is a single edit to that file.
        """
        if self.mode == "act":
            return self.act_model or DEFAULT_ACT_MODEL
        return self.review_model or DEFAULT_REVIEW_MODEL

    @property
    def selected_review_models(self) -> tuple[str, ...]:
        """Return the complete reviewer model roster for review mode.

        Order: the primary :attr:`review_model` first, then any extra
        :attr:`review_models` (the multi-model "fight" roster from
        ``review.models`` in the in-repo config or ``DOTBOT_REVIEW_MODELS``).
        Deduplicated, preserving order.
        """
        roster: list[str] = [self.review_model or DEFAULT_REVIEW_MODEL]
        roster.extend(model for model in self.review_models if model not in roster)
        return tuple(roster)

    def load_model_config_file(self) -> ModelConfig:
        """Re-read the in-repo model config file off disk.

        Useful for diagnostics — the values are already applied to the
        ReviewConfig at construction time, so callers don't normally need this.
        """
        return load_model_config(repo_root=self.resolved_repo_root, config_path=self.config_path)


def make_debug(config: ReviewConfig) -> Callable[[int, str], None]:
    """Create a debug logging callback bound to a config's debug_level."""
    level = config.debug_level

    def _debug(min_level: int, message: str) -> None:
        if level >= min_level:
            print(f"[debug{min_level}] {message}", file=sys.stderr)

    return _debug


def _parse_debug_level(value: str) -> int:
    """Parse debug level from string with fallback."""
    try:
        return int(value.strip() or "0")
    except (ValueError, AttributeError):
        return 0


def _parse_non_negative_int(value: str | None, *, default: int) -> int:
    """Parse a non-negative int from an env string, falling back to default."""
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value.strip())
    except ValueError:
        return default
    return parsed if parsed >= 0 else default


def _parse_allowed_commenter_associations(value: str | None) -> tuple[str, ...]:
    """Parse a comma-separated GitHub author-association allowlist."""
    if value is None:
        return _DEFAULT_ALLOWED_COMMENTER_ASSOCIATIONS

    associations = tuple(item.strip().upper() for item in value.split(",") if item.strip())
    return associations


def _parse_model_list(value: str | None) -> tuple[str, ...]:
    """Parse a comma-separated list of OpenRouter model slugs for the reviewer roster."""
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _config_values_from_environment() -> _ReviewConfigValues:
    github_token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
    github_approval_token = os.environ.get("DOTBOT_GITHUB_USER_PAT", "").strip()
    repository = os.environ.get("GITHUB_REPOSITORY", "").strip()
    openai_api_key = os.environ.get("OPENAI_API_KEY", "").strip()

    pr_number = None
    if pr_num_str := os.environ.get("PR_NUMBER"):
        try:
            pr_number = int(pr_num_str)
        except ValueError:
            pr_number = None

    repo_root = None
    if workspace := os.environ.get("GITHUB_WORKSPACE"):
        repo_root = Path(workspace).resolve()

    return {
        "github_token": github_token,
        "github_approval_token": github_approval_token,
        "repository": repository,
        "pr_number": pr_number,
        "mode": os.environ.get("DOTBOT_MODE", "review").strip(),
        "model_provider": os.environ.get("DOTBOT_PROVIDER", "openrouter").strip(),
        "openai_api_key": openai_api_key,
        "openrouter_api_key": os.environ.get("OPENROUTER_API_KEY", "").strip(),
        "model_name": os.environ.get("DOTBOT_MODEL", "gpt-5.4").strip(),
        "review_model": os.environ.get("OPENROUTER_REVIEW_MODEL", DEFAULT_REVIEW_MODEL).strip()
        or DEFAULT_REVIEW_MODEL,
        "review_models": _parse_model_list(os.environ.get("DOTBOT_REVIEW_MODELS", "")),
        "act_model": os.environ.get("OPENROUTER_ACT_MODEL", DEFAULT_ACT_MODEL).strip()
        or DEFAULT_ACT_MODEL,
        "config_path": os.environ.get("OPENROUTER_REVIEW_CONFIG", DEFAULT_CONFIG_PATH).strip()
        or DEFAULT_CONFIG_PATH,
        "reasoning_effort": os.environ.get("DOTBOT_REASONING_EFFORT", "medium").strip(),
        "web_search_mode": os.environ.get("DOTBOT_WEB_SEARCH_MODE", "live").strip(),
        "act_instructions": os.environ.get("DOTBOT_ACT_INSTRUCTIONS", "").strip(),
        "debug_level": _parse_debug_level(os.environ.get("DEBUG_CODEREVIEW", "0")),
        "stream_output": os.environ.get("STREAM_AGENT_MESSAGES", "1") != "0",
        "dry_run": os.environ.get("DRY_RUN") == "1",
        "resolve_stale_threads": os.environ.get("DOTBOT_RESOLVE_STALE_THREADS", "") == "1",
        "model_timeout_seconds": _parse_non_negative_int(
            os.environ.get("DOTBOT_MODEL_TIMEOUT_SECONDS"), default=900
        ),
        "additional_prompt": os.environ.get("DOTBOT_ADDITIONAL_PROMPT", "").strip(),
        "repo_root": repo_root,
        "context_dir_name": ".dotbot-context",
        "allowed_commenter_associations": _parse_allowed_commenter_associations(
            os.environ.get("DOTBOT_ALLOWED_COMMENTER_ASSOCIATIONS")
        ),
    }


def _apply_in_repo_model_config(values: _ReviewConfigValues, *, explicit_keys: set[str]) -> None:
    """Overlay values from the in-repo model config file (e.g. ``.openrouter-review.yml``).

    Precedence is **explicit kwargs > in-repo config file > env > defaults**:
    a value supplied via :meth:`ReviewConfig.from_args` always wins over the
    file, but the file always wins over hardcoded defaults / env variables that
    were not explicitly set by the consumer.

    This is what makes "swap the model" a single edit to the in-repo file.
    """

    repo_root_value = values.get("repo_root")
    repo_root: Path | None
    if isinstance(repo_root_value, Path):
        repo_root = repo_root_value
    elif isinstance(repo_root_value, str) and repo_root_value:
        repo_root = Path(repo_root_value).resolve()
    else:
        repo_root = None

    file_config = load_model_config(repo_root=repo_root, config_path=values.get("config_path"))

    if "review_model" not in explicit_keys:
        values["review_model"] = file_config.review_model
    if "review_models" not in explicit_keys and file_config.review_models:
        values["review_models"] = file_config.review_models
    if "act_model" not in explicit_keys:
        values["act_model"] = file_config.act_model

    if "reasoning_effort" not in explicit_keys:
        mode = values.get("mode", "review")
        per_mode_effort = (
            file_config.act_reasoning_effort
            if mode == "act"
            else file_config.review_reasoning_effort
        )
        if per_mode_effort:
            values["reasoning_effort"] = per_mode_effort

    if "web_search_mode" not in explicit_keys and file_config.review_web_search_mode is not None:
        values["web_search_mode"] = file_config.review_web_search_mode


def _apply_config_overrides(values: _ReviewConfigValues, kwargs: Mapping[str, Any]) -> None:
    github_token = kwargs.get("github_token")
    if github_token is not None:
        values["github_token"] = github_token

    github_approval_token = kwargs.get("github_approval_token")
    if github_approval_token is not None:
        values["github_approval_token"] = str(github_approval_token).strip()

    repository = kwargs.get("repository")
    if repository is not None:
        values["repository"] = repository

    pr_number = kwargs.get("pr_number")
    if pr_number is not None:
        values["pr_number"] = pr_number

    mode = kwargs.get("mode")
    if mode is not None:
        values["mode"] = mode

    model_provider = kwargs.get("model_provider")
    if model_provider is not None:
        values["model_provider"] = model_provider

    openai_api_key = kwargs.get("openai_api_key")
    if openai_api_key is not None:
        values["openai_api_key"] = str(openai_api_key).strip()

    openrouter_api_key = kwargs.get("openrouter_api_key")
    if openrouter_api_key is not None:
        values["openrouter_api_key"] = str(openrouter_api_key).strip()

    model_name = kwargs.get("model_name")
    if model_name is not None:
        values["model_name"] = model_name

    review_model = kwargs.get("review_model")
    if review_model is not None:
        values["review_model"] = str(review_model).strip() or DEFAULT_REVIEW_MODEL

    review_models = kwargs.get("review_models")
    if review_models is not None:
        if isinstance(review_models, str):
            values["review_models"] = _parse_model_list(review_models)
        else:
            values["review_models"] = tuple(
                str(model).strip() for model in review_models if str(model).strip()
            )

    act_model = kwargs.get("act_model")
    if act_model is not None:
        values["act_model"] = str(act_model).strip() or DEFAULT_ACT_MODEL

    config_path = kwargs.get("config_path")
    if config_path is not None:
        values["config_path"] = str(config_path).strip() or DEFAULT_CONFIG_PATH

    reasoning_effort = kwargs.get("reasoning_effort")
    if reasoning_effort is not None:
        values["reasoning_effort"] = reasoning_effort

    web_search_mode = kwargs.get("web_search_mode")
    if web_search_mode is not None:
        values["web_search_mode"] = web_search_mode

    act_instructions = kwargs.get("act_instructions")
    if act_instructions is not None:
        values["act_instructions"] = act_instructions

    debug_level = kwargs.get("debug_level")
    if debug_level is not None:
        values["debug_level"] = debug_level

    stream_output = kwargs.get("stream_output")
    if stream_output is not None:
        values["stream_output"] = stream_output

    dry_run = kwargs.get("dry_run")
    if dry_run is not None:
        values["dry_run"] = dry_run

    resolve_stale_threads = kwargs.get("resolve_stale_threads")
    if resolve_stale_threads is not None:
        values["resolve_stale_threads"] = resolve_stale_threads

    additional_prompt = kwargs.get("additional_prompt")
    if additional_prompt is not None:
        values["additional_prompt"] = additional_prompt

    repo_root = kwargs.get("repo_root")
    if repo_root is not None:
        values["repo_root"] = repo_root

    context_dir_name = kwargs.get("context_dir_name")
    if context_dir_name is not None:
        values["context_dir_name"] = context_dir_name

    allowed_commenter_associations = kwargs.get("allowed_commenter_associations")
    if allowed_commenter_associations is not None:
        if isinstance(allowed_commenter_associations, str):
            values["allowed_commenter_associations"] = _parse_allowed_commenter_associations(
                allowed_commenter_associations
            )
        else:
            values["allowed_commenter_associations"] = tuple(
                str(item).strip().upper()
                for item in allowed_commenter_associations
                if str(item).strip()
            )
