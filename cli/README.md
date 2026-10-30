# dotbot Code Review CLI

A modular, well-structured CLI for autonomous code review using dotbot.

## Overview

This CLI provides autonomous code review capabilities for GitHub pull requests. It has been refactored from the original monolithic script into a clean, modular architecture with proper separation of concerns.

## Architecture

```
cli/
├── __init__.py
├── main.py                    # CLI entry point with argparse
├── core/
│   ├── __init__.py
│   ├── config.py              # Configuration loading + validation
│   ├── models.py              # Typed dataclasses (comments, findings, payloads)
│   ├── exceptions.py          # Custom exception hierarchy
│   └── github_types.py        # Shared typed protocols for GitHub objects
├── clients/
│   ├── __init__.py
│   ├── codex_client.py        # Optional act backend and evaluation baseline
│   ├── codex_event_debugger.py # Protocol event debug formatting helpers
│   ├── github_client.py       # GitHub API helpers (PyGithub wrapper)
│   └── git_ops.py             # Git subprocess helpers
├── review/
│   ├── __init__.py
│   ├── artifacts.py           # Review artifact persistence + summary rendering
│   ├── coordinator.py         # Concurrent review panel, discussion, and decision policies
│   ├── dedupe.py              # Duplicate-detection helpers
│   ├── posting.py             # Inline comment payload/build/post helpers
│   ├── review_prompt.py       # Load editable Markdown review/panel prompts
│   ├── context_manager.py     # Context artifact writing
│   ├── patch_parser.py        # Patch parsing utilities
│   └── anchor_engine.py       # Diff anchor resolution utilities
├── workflows/
│   ├── __init__.py
│   ├── edit_prompt.py         # ACT prompt/context formatting helpers
│   ├── review_workflow.py     # Review orchestration
│   └── edit_workflow.py       # ACT mode orchestration
```

## Key Improvements

### 1. **Modular Design**
- **GitHub API**: PyGithub is wrapped in `clients/github_client.py`; review orchestration lives in `workflows/review_workflow.py`
- **Model API**: production review uses `clients/openrouter_client.py`; `clients/codex_client.py` remains for optional act-mode SDK execution and evaluation.
- **Patch Processing**: `review/patch_parser.py` contains utilities for parsing unified diffs
- **Configuration**: `ReviewConfig` centralizes all configuration management
- **Prompt Building**: `review/review_prompt.py` provides guideline loading and prompt composition helpers
- **Review Processing**: `ReviewWorkflow` orchestrates the entire review workflow

### 2. **Better Error Handling**
- Custom exception hierarchy instead of `sys.exit()` calls
- Structured error messages with context
- Graceful degradation for non-critical errors

### 3. **CLI Interface**
- Full command-line argument support using `argparse`
- Can be used as a standalone CLI or in GitHub Actions mode
- Comprehensive help and examples

### 4. **Configuration Management**
- Environment variable support with validation
- Command-line argument overrides
- Explicit mode selection (review vs act)

## Usage

### As a CLI Tool

```bash
# Review a specific PR (default mode)
python -m cli.main --repo owner/repo --pr 123

# Explicit review mode
python -m cli.main --repo owner/repo --pr 123 --mode review

# Act mode with custom instructions
python -m cli.main --repo owner/repo --pr 123 --mode act --act-instructions "Run tests after changes"

# Dry run mode
python -m cli.main --repo owner/repo --pr 123 --dry-run

# Debug mode
python -m cli.main --repo owner/repo --pr 123 --debug 2
```

### GitHub Actions

When used via the composite action, the CLI runs in GitHub Actions mode automatically and reads the event payload to determine whether to run a full review or a comment-triggered edit.

Review posting behavior:
- dotbot posts a PR-level issue comment with a review summary.
- Findings are posted as standalone inline PR review comments on the relevant lines.

### Collaborative review

Every `ReviewWorkflow` run uses the bounded OpenRouter coordinator, including
single-model rosters. The old sequential review path has been removed. Investigations
and each discussion round run concurrently over isolated clients. Round barriers
ensure every participant receives the same complete prior transcript. Models
reinspect pinned-commit code using read-only tools and explicitly retain/reject
stable issue IDs with reasons and evidence.

Configure in `.openrouter-review.yml`:

```yaml
review:
  model: deepseek/deepseek-v4-pro-0813
  models: [openai/gpt-5]
  discussion_rounds: 2  # 1–5
  decision:
    policy: unanimous  # or lead; the primary model adjudicates
```

Unanimous mode preserves all final retained issues and blocks approval on dissent.
Lead mode adds a primary-reviewer adjudication pass. Both fail closed on an
incomplete panel, timeout, unrepaired malformed output, or changed PR base/head.
A structurally valid but contradictory verdict gets one correction request with the
full original conversation/tool results, inside the original timeout and without
another investigation pass. Malformed responses whose decisions cannot be protected
abort immediately. A second invalid response aborts; verdict repairs must preserve
findings and disposition actions. Both prompt files use the same
all-priorities verdict rule. Only the final
findings and one summary are posted; debate remains in attributed run logs
(subject to stream settings) and `.dotbot-context/review-discussion.json`.
The composite Action creates a summary per attempt before startup and updates only
that comment with the final verdict or failed/cancelled status. Every attempt has
its own visible workflow link; historical verdicts remain tied to their own runs.
The local CLI does not run these Action lifecycle steps. Successful CLI reviews
emit the consolidated `ReviewRunResult` as one `[review-result] <JSON>` stdout line;
evaluation captures use that marker, never intermediate panel turns.

Configured credentials are redacted; other sensitive source information may still
appear in discussion, so protect log access. No private reasoning stream is logged.

Credential-path changes (`.env`, `.env.*`, `*.pem`, `*.key`, including deletions/renames)
require human review instead of model calls, automatic approval or thread resolution.
Public `.env.example`/`.env.sample` templates remain reviewable.

The checkout needs both base/head commits for the pinned diff and inspection.
Reviews start fresh; the session-resume, SHA-delta, cache restore/save, and direct
OpenAI review-provider paths are removed in v2. Saved conversations are diagnostic
artifacts only. Even a single reviewer uses the investigation plus discussion
passes. Each pass uses the existing model timeout; already-issued provider
requests are abandoned, not remotely cancelled, on timeout. Exact finding text
and locations are deduplicated; semantic duplicates should reuse ledger IDs.

Comment-triggered edits

- Add a comment on the PR that starts with:
  - `/dotbot <instructions>` or `/dotbot: <instructions>`
- Bare `/dotbot` comments are ignored; the command must include instructions.
- The remainder of the comment is passed to the coding agent. The workflow executes the agent with the configured dotbot runtime settings for this environment, then commits and pushes resulting branch changes (unless dry-run).

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GITHUB_TOKEN` | GitHub API token | *Required* |
| `OPENAI_API_KEY` | Optional act-mode SDK key; not supported for review | *Required for SDK act* |
| `DOTBOT_MODE` | Operation mode (review/act) | `review` |
| `DOTBOT_MODEL` | Model name | `gpt-5.4` |
| `DOTBOT_PROVIDER` | Review requires `openrouter`; act also supports `openai` | `openrouter` |
| `DOTBOT_REASONING_EFFORT` | Reasoning effort level | `medium` |
| `DOTBOT_ACT_INSTRUCTIONS` | Additional instructions for act mode | `` |
| `DOTBOT_ALLOWED_COMMENTER_ASSOCIATIONS` | Comma-separated GitHub comment roles allowed to trigger act mode | `MEMBER,OWNER,COLLABORATOR` |
| `DEBUG_CODEREVIEW` | Debug level (0-2) | `0` |
| `DRY_RUN` | Skip posting (1 for dry run) | `0` |

Invalid `DOTBOT_ALLOWED_COMMENTER_ASSOCIATIONS` values fail fast during configuration loading.

## Operation Modes

- **`review`** (default): Analyzes PR diffs using built-in guidelines from `prompts/review.md`
- **`act`**: Responds to `/dotbot` commands in PR comments to make autonomous code edits with optional custom instructions

## Testing

Each module can be tested independently:

```bash
pytest tests/ -v
```

## Deduplication on Repeated Runs

- The CLI detects if a prior dotbot review exists on the PR (looks for a summary containing "dotbot code review:" or earlier inline review comments).
- When detected, deduplication happens in three layers:
  - **dotbot-thread attribution**: only unresolved review threads whose root author matches a prior dotbot summary author are reused as rerun context.
  - **Inline semantic dedup**: the structured-output turn uses those prior dotbot comments to decide which issues are new vs already covered.
  - **Re-adjudicated summary carry-forward**: the model returns prior comment IDs that still seem relevant, and the summary reports those separately from new findings.
  - **Auto-resolution of fixed dotbot threads**: the model can also mark prior unresolved dotbot comments as fixed, and review mode resolves those GitHub review threads automatically.

## v2 Migration

Review always deliberates over the full pinned PR, without cached verdicts or
incremental review sessions. Applicable prior comments still enter the issue
ledger for explicit final dispositions. Use OpenRouter for review; consumers
requiring the removed execution paths can pin an exact v1 release or `@v1`.
The v2 release updates `v2` and `latest`, not `v1`.

### Customizing the Review Prompt

- General review guidelines live in `prompts/review.md`; panel roles, discussion, evidence, and verdict instructions live in `prompts/review_panel.md`.
- Provide extra reviewer guidance using env `DOTBOT_ADDITIONAL_PROMPT` (verbatim text). No Python changes are needed to edit the bundled Markdown steering prompts.

## Benefits

1. **Maintainable**: Clear separation of concerns makes code easier to maintain
2. **Testable**: Each component can be unit tested independently
3. **Extensible**: Easy to add new features or modify existing ones
4. **Flexible**: Works as CLI tool or GitHub Action
5. **Robust**: Better error handling and validation
6. **Debuggable**: Structured logging and debug levels
