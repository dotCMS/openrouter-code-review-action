"""In-repo model configuration file loader.

The OpenRouter code-review action lets consuming repositories pin which
OpenRouter model slugs are used for ``review`` and ``act`` mode by editing a
single YAML file checked in to their repo (default: ``.openrouter-review.yml``
at the repo root).

The intent is **swap speed**: changing review or act model — including switching
provider entirely (e.g. ``anthropic/claude-opus-4.7`` → ``openai/gpt-5``,
``google/gemini-2.5-pro``, etc.) — is a single edit to the in-repo config file
with no code change. The action picks up the change on the next run.

File shape
----------

.. code-block:: yaml

    # .openrouter-review.yml
    review:
      model: deepseek/deepseek-v4-pro-0813
      reasoning_effort: medium       # optional; default: medium
      web_search_mode: live          # optional; default: live
    act:
      model: anthropic/claude-opus-4.7
      reasoning_effort: medium       # optional; default: medium

Both blocks are optional. Missing keys fall back to action defaults so the file
can be as small as a single ``review.model`` line.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .exceptions import ConfigurationError

DEFAULT_CONFIG_PATH = ".openrouter-review.yml"
DEFAULT_REVIEW_MODEL = "deepseek/deepseek-v4-pro-0813"
DEFAULT_ACT_MODEL = "anthropic/claude-opus-4.7"

_VALID_REASONING_EFFORTS = frozenset({"minimal", "low", "medium", "high"})
_VALID_WEB_SEARCH_MODES = frozenset({"disabled", "cached", "live"})
_KNOWN_MODE_KEYS = frozenset(
    {"model", "models", "reasoning_effort", "web_search_mode", "discussion_rounds", "decision"}
)


def validate_panel_options(rounds: object, policy: object) -> None:
    if type(rounds) is not int or not 1 <= rounds <= 5:
        raise ConfigurationError("review.discussion_rounds must be an integer from 1 to 5")
    if not isinstance(policy, str) or policy not in {"unanimous", "lead"}:
        raise ConfigurationError("review.decision.policy must be 'unanimous' or 'lead'")


@dataclass(frozen=True)
class ModelConfig:
    """Resolved per-mode model configuration loaded from the in-repo file.

    Each field is the *effective* value after applying defaults — callers can
    use these values directly without re-checking for ``None``. ``source_path``
    is ``None`` when no config file was found and pure defaults were used.

    ``review_models`` is the optional multi-model reviewer roster: additional
    model slugs that run alongside ``review_model`` so two (or more) models
    review the same PR and comment on each other's findings. It is empty when
    only the single configured review model should run.
    """

    review_model: str = DEFAULT_REVIEW_MODEL
    review_models: tuple[str, ...] = ()
    review_discussion_rounds: int = 2
    review_decision_policy: str = "unanimous"
    act_model: str = DEFAULT_ACT_MODEL
    review_reasoning_effort: str | None = None
    act_reasoning_effort: str | None = None
    review_web_search_mode: str | None = None
    source_path: Path | None = None

    def model_for_mode(self, mode: str) -> str:
        """Return the model slug that should be used for the given mode."""
        if mode == "act":
            return self.act_model
        return self.review_model


def resolve_config_path(repo_root: Path | None, config_path: str | None) -> Path:
    """Resolve the absolute config path from a repo root and relative/absolute path.

    A relative ``config_path`` is anchored to ``repo_root``; an absolute path is
    returned unchanged. ``None``/empty values fall back to
    :data:`DEFAULT_CONFIG_PATH`.
    """

    rel = (config_path or DEFAULT_CONFIG_PATH).strip() or DEFAULT_CONFIG_PATH
    candidate = Path(rel)
    if candidate.is_absolute():
        return candidate
    base = repo_root or Path(".")
    return (base / candidate).resolve()


def load_model_config(
    repo_root: Path | None = None,
    config_path: str | None = None,
) -> ModelConfig:
    """Load and validate the in-repo model config file.

    Missing files yield a :class:`ModelConfig` populated with defaults so the
    action runs without any in-repo file present. A malformed file raises
    :class:`~cli.core.exceptions.ConfigurationError` with the file path so the
    consumer can fix and re-run.
    """

    resolved = resolve_config_path(repo_root, config_path)
    if not resolved.exists():
        return ModelConfig()

    try:
        raw = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(
            f"Failed to read OpenRouter model config at {resolved}: {exc}"
        ) from exc

    try:
        parsed = yaml.safe_load(raw) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"Invalid YAML in OpenRouter model config at {resolved}: {exc}"
        ) from exc

    if not isinstance(parsed, dict):
        raise ConfigurationError(f"OpenRouter model config at {resolved} must be a YAML mapping")

    review_block = _coerce_mode_block(parsed.get("review"), "review", resolved)
    act_block = _coerce_mode_block(parsed.get("act"), "act", resolved)

    return ModelConfig(
        review_model=review_block.get("model") or DEFAULT_REVIEW_MODEL,
        review_models=tuple(review_block.get("models") or ()),
        review_discussion_rounds=review_block.get("discussion_rounds", 2),
        review_decision_policy=review_block.get("decision", {}).get("policy", "unanimous"),
        act_model=act_block.get("model") or DEFAULT_ACT_MODEL,
        review_reasoning_effort=review_block.get("reasoning_effort"),
        act_reasoning_effort=act_block.get("reasoning_effort"),
        review_web_search_mode=review_block.get("web_search_mode"),
        source_path=resolved,
    )


def _coerce_mode_block(value: Any, mode: str, source: Path) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigurationError(f"OpenRouter model config at {source}: '{mode}' must be a mapping")

    unknown = sorted(set(value).difference(_KNOWN_MODE_KEYS))
    if unknown:
        joined = ", ".join(unknown)
        raise ConfigurationError(
            f"OpenRouter model config at {source}: unknown '{mode}' keys: {joined}. "
            f"Allowed: {', '.join(sorted(_KNOWN_MODE_KEYS))}"
        )

    block: dict[str, Any] = {}
    if mode != "review" and {"discussion_rounds", "decision"}.intersection(value):
        raise ConfigurationError("discussion_rounds and decision are only valid under review")
    decision = value.get("decision", {})
    if not isinstance(decision, dict) or set(decision) - {"policy"}:
        raise ConfigurationError("review.decision must be a mapping containing only policy")
    rounds = value.get("discussion_rounds", 2)
    validate_panel_options(rounds, decision.get("policy", "unanimous"))
    block["discussion_rounds"] = rounds
    block["decision"] = decision

    model = value.get("model")
    if model is not None:
        if not isinstance(model, str) or not model.strip():
            raise ConfigurationError(
                f"OpenRouter model config at {source}: '{mode}.model' must be a non-empty string"
            )
        block["model"] = model.strip()

    models = value.get("models")
    if models is not None:
        if mode != "review":
            raise ConfigurationError(
                f"OpenRouter model config at {source}: 'models' is only valid under "
                f"the 'review' block (found under '{mode}')"
            )
        if not isinstance(models, list) or not models:
            raise ConfigurationError(
                f"OpenRouter model config at {source}: 'review.models' must be a non-empty list"
            )
        normalized_models: list[str] = []
        for raw_model in models:
            if not isinstance(raw_model, str) or not raw_model.strip():
                raise ConfigurationError(
                    f"OpenRouter model config at {source}: 'review.models' entries "
                    "must be non-empty strings"
                )
            normalized_models.append(raw_model.strip())
        block["models"] = tuple(normalized_models)

    reasoning = value.get("reasoning_effort")
    if reasoning is not None:
        if not isinstance(reasoning, str) or reasoning not in _VALID_REASONING_EFFORTS:
            raise ConfigurationError(
                f"OpenRouter model config at {source}: '{mode}.reasoning_effort' "
                f"must be one of {sorted(_VALID_REASONING_EFFORTS)}"
            )
        block["reasoning_effort"] = reasoning

    web_search = value.get("web_search_mode")
    if web_search is not None:
        if mode != "review":
            raise ConfigurationError(
                f"OpenRouter model config at {source}: 'web_search_mode' is only "
                f"valid under the 'review' block (found under '{mode}')"
            )
        if not isinstance(web_search, str) or web_search not in _VALID_WEB_SEARCH_MODES:
            raise ConfigurationError(
                f"OpenRouter model config at {source}: '{mode}.web_search_mode' "
                f"must be one of {sorted(_VALID_WEB_SEARCH_MODES)}"
            )
        block["web_search_mode"] = web_search

    return block
