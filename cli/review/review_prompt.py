from __future__ import annotations

import re
from pathlib import Path

from ..core.config import ReviewConfig, make_debug
from ..core.exceptions import PromptError


def _load_builtin_prompt(config: ReviewConfig, filename: str) -> str:
    path = Path(__file__).resolve().parents[2] / "prompts" / filename
    make_debug(config)(1, f"Using built-in prompt: {path}")
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PromptError(f"Failed to read built-in prompt {path}: {exc}") from exc


def load_guidelines(config: ReviewConfig) -> str:
    """Base review guidelines plus each focus prompt named in ``config.prompts``."""
    if config.mode != "review":
        return ""
    # Accept the filename spelling too: "iac.md" means "iac".
    names = [name.removesuffix(".md") for name in config.prompts]
    for name in names:
        # Names become filenames: reject anything that could leave prompts/.
        if not re.fullmatch(r"[a-z0-9_]+", name):
            raise PromptError(f"Invalid prompt name {name!r}: use lowercase letters, digits, '_'")
    return "\n\n".join(_load_builtin_prompt(config, f"{name}.md") for name in ("review", *names))


def load_panel_instructions(config: ReviewConfig) -> str:
    return _load_builtin_prompt(config, "review_panel.md")


def render_additional_review_instructions(config: ReviewConfig) -> str:
    extra = config.additional_prompt
    return "<additional_instructions>\n" + extra + "\n</additional_instructions>\n" if extra else ""
