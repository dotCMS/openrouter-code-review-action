# OpenRouter Code Review Action (Review + Act)

Run an OpenRouter-routed model to review pull requests and, on demand, make
autonomous edits driven by `/dotbot` comments. Review path uses direct
OpenRouter chat completions with schema-enforced JSON output; Act path runs the
built-in tool-calling agent loop so models and providers can be swapped quickly
when pricing or quality changes.

- **Review**: posts precise inline review comments and a PR-level summary. When
  there are no findings, only the summary is posted.
- **Act**: applies focused edits when trusted users comment `/dotbot`; commits
  and pushes to the PR branch.

## Quick Start (Review)

```yaml
name: PR Review
on:
  pull_request:
    types: [opened, synchronize, reopened, ready_for_review]
permissions:
  contents: read
  pull-requests: write
  issues: write
jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - name: OpenRouter autonomous review
        uses: wezell/openrouter-code-review-action@v1
        with:
          mode: review
          openrouter_api_key: ${{ secrets.OPENROUTER_API_KEY }}
          github_approval_token: ${{ secrets.DOTBOT_GITHUB_USER_PAT }}
```

### Automatic PR Approval

If the `DOTBOT_GITHUB_USER_PAT` secret is set (passed via the
`github_approval_token` input), the action approves the PR **as the PAT's
user** — e.g. `dotCMS-Machine-User` — whenever *every* reviewer model in the
roster concludes `Overall: patch is correct`. The approval is idempotent per
head commit: re-running the workflow on the same SHA will not spam duplicate
approvals.

If the secret is not set, or any reviewer model reports a finding, the action
simply posts its review comments and skips the approval step. Approval
submission failures are logged as warnings and never fail the review run.

> The PAT needs `pull-requests: write` scope, and its user must differ from
> the PR author (GitHub rejects approvals from the PR author).

## Act on `/dotbot` Comments

When a trusted user comments `/dotbot <instructions>` on a PR, the action checks
out the branch, runs the built-in agent loop (with `read_file`, `write_file`,
`run_command` tools) against the configured `act.model`, and pushes the result.
Give the agent a runnable environment so it can build/test before pushing.

```yaml
name: PR Act
on:
  issue_comment: { types: [created] }
  pull_request_review_comment: { types: [created] }
permissions:
  contents: write
  workflows: write       # needed if /dotbot may edit .github/workflows/*
  pull-requests: write
  issues: write
concurrency:
  group: dotbot-act-${{ github.event.issue.number || github.event.pull_request.number || github.ref }}
  cancel-in-progress: false
jobs:
  act:
    name: Act on /dotbot comments
    if: >-
      (
        (
          github.event_name == 'issue_comment' &&
          startsWith(github.event.comment.body, '/dotbot') &&
          github.event.issue.pull_request
        ) || (
          github.event_name == 'pull_request_review_comment' &&
          startsWith(github.event.comment.body, '/dotbot')
        )
      ) &&
      github.actor != 'dependabot[bot]'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
          ref: ${{ github.event.pull_request.head.sha || format('refs/pull/{0}/head', github.event.issue.number) }}
          token: ${{ secrets.REPO_ACCESS_TOKEN }}

      # Give the agent a working environment so it can build/test.
      # Replace with your own setup (install deps, run migrations, etc.).
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
      - run: npm ci

      - name: dotbot autonomous edits
        uses: wezell/openrouter-code-review-action@v1
        with:
          mode: act
          openrouter_api_key: ${{ secrets.OPENROUTER_API_KEY }}
          allowed_commenter_associations: MEMBER,OWNER,COLLABORATOR
```

### `/dotbot` Commands

- **`/dotbot <instructions>`** — apply minimal diffs matching the instructions.
- Bare **`/dotbot`** is ignored; include explicit instructions after the command.
- **`/dotbot address comments`** (or natural variants like "please fix the
  review comments") — address unresolved review threads. Only unresolved
  threads are considered; resolved threads are ignored.

## Swapping the Model (Single-Edit Config File)

Model selection lives in an in-repo file — `.openrouter-review.yml` at the
consuming repo's root by default — so flipping review or act to a different
OpenRouter slug is a single edit. No Python change, no workflow change, no
action-input change required.

```yaml
# .openrouter-review.yml
review:
  model: deepseek/deepseek-v4-pro-0813    # change this line to swap review model
  models:                                  # optional — extra reviewers to "fight"
    - openai/gpt-5
    - google/gemini-2.5-pro
  reasoning_effort: medium            # optional — minimal | low | medium | high
  web_search_mode: live               # optional — disabled | cached | live

act:
  model: anthropic/claude-opus-4.7    # change this line to swap act model
  reasoning_effort: medium
```

The defaults are `deepseek/deepseek-v4-pro-0813` for review mode and
`anthropic/claude-opus-4.7` for act mode. To pin a different
provider — e.g. `openai/gpt-5.4`, `google/gemini-2.5-pro`,
`anthropic/claude-sonnet-4.5` — change the slug on the matching `model:` line
and commit. The next run uses the new model.

### Multiple Reviewers ("the fight")

Set `review.models` (a list) — or `DOTBOT_REVIEW_MODELS` as a comma-separated
env var — to run additional review models alongside the primary `review.model`.
Each reviewer runs the full pipeline in sequence:

- posts its own inline findings and its own PR summary, tagged with its model
  and reasoning effort (the summary footer line and the metadata thumbprint).
- sees the earlier reviewers' posted comments as prior-context, so a later
  model can carry them forward, agree, or push back — a built-in disagreement
  pass. Configure two strong models and let them argue about the diff.

Review state (resume threads, SHA-delta scope, cache keys) is isolated per
model, so changing the roster won't reuse the wrong review.

Override the file path with the `config_path` action input or
`OPENROUTER_REVIEW_CONFIG` env var (e.g. `ci/openrouter-models.yml`). Per-call
action inputs (`model:`, `reasoning_effort:`) still win over the file when
present, so a single workflow can opt out for a one-off run.

`web_search_mode: live` appends OpenRouter's `:online` variant to the slug at
call time; `cached` and `disabled` skip the live web fetch.

## Inputs

| Input | Description | Default |
|-------|-------------|---------|
| `openrouter_api_key` | OpenRouter API key for routed model calls | *required* |
| `config_path` | Path to the in-repo model config file | `.openrouter-review.yml` |
| `mode` | `review` or `act` | `review` |
| **Model** | | |
| `model` | Per-call override; the in-repo config file is the normal swap point | `deepseek/deepseek-v4-pro-0813` (review) / `anthropic/claude-opus-4.7` (act) |
| `models` | Array of models (or comma separated arg) to code review | none  |
| `reasoning_effort` | `minimal` / `low` / `medium` / `high` | `medium` |
| `model_timeout_seconds` | Wall-clock budget per reviewer model pass; 0 disables | `900` |
| `web_search_mode` | `disabled` / `cached` / `live` | `live` |
| **Review-only** | | |
| `github_approval_token` | GitHub user PAT (e.g. `secrets.DOTBOT_GITHUB_USER_PAT`) owned by a machine user such as `dotCMS-Machine-User`. When every reviewer model reports `Overall: patch is correct`, dotbot approves the PR as that user; when unset (or any model dissents) reviews post as normal comments with no approval | *(unset)* |
| `additional_prompt` | Extra reviewer instructions (verbatim) | |
| `resolve_stale_threads` | `0` or `1` — resolve prior unresolved dotbot threads the reviewer model saw but declined to carry forward (judged stale/fixed); threads never shown to the model stay open | `0` |
| **Act-only** | | |
| `act_instructions` | Extra guidance appended to the edit prompt | |
| `allowed_commenter_associations` | Comma-separated GitHub `author_association` values allowed to trigger Act mode | `MEMBER,OWNER,COLLABORATOR` |
| `dry_run` | `0` or `1` — skip push | `0` |
| **Debug** | | |
| `debug_level` | `0` (off) / `1` (basic) / `2` (trace) | `1` |
| `stream_agent_messages` | `0` or `1` — stream model output to logs | `1` |
| **Advanced** | | |
| `extra_pip_args` | Additional pip flags (e.g., `--index-url`) | |

> `openai_api_key` is still accepted for legacy parity but the OpenRouter path
> is the supported route for all model calls. Migration drops direct vendor
> SDKs entirely — see `seed.yaml` for the constraint set.

## Agent Loop

OpenRouter chat completions have no built-in agent runtime, so the action ships
its own tool-calling loop. Every review/edit turn gives the model three tools:

- `read_file` — read a repo file (with optional line ranges)
- `write_file` — create/overwrite a repo file
- `run_command` — run a shell command in the repo root (e.g. `git diff`)

The loop runs until the model produces a final answer, with a 30-iteration cap.
Sandbox parity with the legacy Codex client is preserved: review and act turns
run with full access, and any future read-only turn would see the mutating
tools disabled. Tool results feed back as `role: "tool"` messages, so the
conversation (including tool traffic) is persisted for resume. Setting
`DOTBOT_PROVIDER=openai` routes to the legacy Codex SDK client instead.

## What It Posts

- **Inline comments** anchored to exact diff lines. If a line isn't in the
  current diff, the finding is filtered or remapped before submission so
  GitHub does not return 422.
- **PR-level summary** as an issue comment on each run (refreshed on re-runs;
  prior summaries are deleted). The summary footer records which model and
  reasoning effort produced the review, e.g. `<sub>reviewed by dotbot ·
  deepseek/deepseek-v4-pro-0813 · medium</sub>`.
- **Multi-line suggestions** only when contiguous and short; otherwise a
  single-line comment.

## Review Continuation

On repeated `pull_request` review runs, the action continues the prior review
instead of restarting from scratch.

1. The PR summary stores the previously reviewed head SHA in hidden metadata.
2. Review mode caches isolated review state keyed by repository, PR number,
   model, and reviewed SHA.
3. On the next push, the action restores that cache, resumes the latest stored
   review thread, and scopes the prompt to the SHA-delta since the previously
   reviewed SHA.
4. If the prior SHA is no longer an ancestor (force-push, rebase), the cache
   is missing, or no thread can be restored, the action falls back to a fresh
   full review.

## Deduplication on Repeated Runs

When a prior review exists on the PR, reruns reuse only **unresolved
action-authored review threads** as context.

1. **Inline semantic dedup** — prior unresolved comments are passed to the
   model's structured-output turn so it can avoid reposting the same issue as
   a new finding.
2. **Re-adjudicated carry-forward** — the model separately marks which of
   those prior unresolved comments are still relevant now. Only those count
   toward the PR summary.
3. **Separated counts** — the summary reports new findings and still-relevant
   prior findings separately.

## Security & Permissions

- Act mode enforces a built-in `author_association` allowlist. Keep the
  workflow-level `if:` guard as defense in depth if you want early job
  skipping.
- Invalid `allowed_commenter_associations` values fail fast at startup so auth
  policy drift is visible immediately.
- For forks, the default `GITHUB_TOKEN` generally cannot push — run Act only
  on branches in the main repo, or use a PAT with fork access.
- Grant only what's needed: `contents: write` (push), `pull-requests: write`
  (reviews), `issues: write` (summary comments and Act replies).
- **`workflows: write` is required for Act mode** when the `/dotbot` fix touches
  `.github/workflows/*`. The default `GITHUB_TOKEN` refuses to push edits to
  workflow files without it — you'll see `refusing to allow a GitHub App to
  create or update workflow ... without workflows permission`. Add
  `workflows: write` to the Act job's `permissions:` block if you want `/dotbot`
  to be able to modify workflows.

## Troubleshooting

- **422 Unprocessable Entity**: target line not in PR head diff. The action's
  filter/remap pipeline should catch this before submission; if you still see
  422s, set `debug_level: 2` to log anchors.
- **Model errors**: ensure your OpenRouter key has access to the selected slug
  and that the slug exists in the OpenRouter catalog.
- Review uses built-in prompts (see `prompts/review.md`). Customize with
  `additional_prompt`.

## Local Development

```bash
uv sync                # install deps
make lint              # format, lint, type-check
GITHUB_TOKEN=… OPENROUTER_API_KEY=… PYTHONPATH=. python -m cli.main \
  --repo owner/repo --pr 123 --mode review --dry-run
```

Environment variables of note:

| Variable | Purpose |
|----------|---------|
| `GITHUB_TOKEN` / `GH_TOKEN` | GitHub API token used for review comments (required) |
| `DOTBOT_GITHUB_USER_PAT` | Optional PAT owned by the machine user; enables auto-approval when all reviewers agree (see [Automatic PR Approval](#automatic-pr-approval)) |
| `DOTBOT_MODEL_TIMEOUT_SECONDS` | Per-reviewer-model wall-clock budget in seconds (0 disables) | `900` |
| `OPENROUTER_API_KEY` | OpenRouter API key (required for the default provider) |
| `OPENAI_API_KEY` | Legacy provider key (`DOTBOT_PROVIDER=openai`) |

## Release & Versioning

Releases are cut automatically on every merge to `main`: the `Auto Release on
Merge` workflow bumps the highest existing `v*` tag by a patch (e.g. `v1.1.0` →
`v1.1.1`), creates the tag and GitHub Release. After publish, the `Release`
workflow points the `v1` and `latest` tags at the new release commit, so
consumers pinning `@v1` always get the latest merged action code.

Because the `v1`/`latest` moving tags track releases, self-hosted workflow
files should pin the action to a specific release SHA.

## Project Status

This action started as a clone of the GitHub codex-review-action and was
migrated to route all model calls through [OpenRouter](https://openrouter.ai)
with a built-in tool-calling agent loop (see [Agent Loop](#agent-loop)). The
legacy Codex SDK client remains for parity behind `DOTBOT_PROVIDER=openai`, and
the `eval/` tooling compares new model candidates against a Codex baseline.

The evaluation harness scores candidate models for review-quality parity:

```bash
uv run python -m eval.run_openrouter_baseline   # run candidates over the PR sample
uv run python -m eval.overlap_score           # score overlap vs the Codex baseline
```
