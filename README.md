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

## v2 breaking migration

Review mode now has **one execution path**: the OpenRouter coordinator, including
single-model rosters. The sequential reviewer, direct OpenAI/Codex review provider,
session resume, SHA-delta scoping, and review-cache restore/save paths are removed.
Every run reviews the full pinned PR and reconsiders applicable unresolved comments.

- Use `dotCMS/openrouter-code-review-action@v2` and `OPENROUTER_API_KEY` for review.
  OpenAI models remain available through their OpenRouter slugs.
- No opt-in is required: existing model configuration uses the coordinator's
  defaults (two discussion rounds, unanimous policy). Even one reviewer now uses
  three passes; account for that model cost when migrating.
- Pin `@v1` or an exact v1 release to retain the old behavior. Publishing v2 updates
  only `v2` and `latest`, never `v1`. `@latest` intentionally opts into major changes.
- The separate act-mode SDK option remains. Codex evaluation calls the SDK directly in read-only mode; it never routes through production review.

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
        uses: dotCMS/openrouter-code-review-action@v2
        with:
          mode: review
          openrouter_api_key: ${{ secrets.OPENROUTER_API_KEY }}
          github_approval_token: ${{ secrets.DOTBOT_GITHUB_USER_PAT }}
```


### Automatic PR Approval

If a token is passed via the `github_approval_token` input — normally the
`DOTBOT_GITHUB_USER_PAT` secret owned by a machine user such as
`dotCMS-Machine-User` — the action approves the PR **as that user** whenever
the configured decision policy accepts the final review. The default,
`unanimous`, requires every model's **final** position to be `patch is correct`
with no retained findings. With `review.decision.policy: lead`, the primary
reviewer adjudicates disputed findings after discussion. Approval is idempotent
per head commit: re-running the workflow on the same SHA will not spam duplicates.

If the input is unset, or the final decision retains findings or rejects the
patch, the action posts its final review comments without approval. A missing,
failed, timed-out, or malformed reviewer response prevents panel publication and
approval under either policy. A changed PR base or head requires a new review. Approval
submission failures are logged as warnings and never fail the review run.

> The PAT needs `pull-requests: write` scope, and its user must differ from
> the PR author (GitHub rejects approvals from the PR author).
>
> The self-hosted review workflow wires both:
> `github_approval_token: ${{ secrets.DOTBOT_GITHUB_USER_PAT || github.token }}`
> — the machine-user PAT when the secret exists, the workflow token otherwise,
> so a consumer repo with no PAT still approves. Where no machine-user PAT is
> available, `github_approval_token: ${{ github.token }}`
> works too, provided the repo allows it (Settings → Actions → "Allow GitHub
> Actions to create and approve pull requests"). Installation tokens cannot read
> `GET /user`, so the action does not try to resolve an identity for them: the
> approval is submitted as `github-actions[bot]` and idempotency falls back to a
> marker in the review body. A bot cannot approve a PR the bot itself authored,
> so a PAT remains the better choice where one exists.

#### Token setup in the dotCMS org

`DOTBOT_GITHUB_USER_PAT` is an **organization-level** secret available to every
dotCMS repo, and the machine user already has write access to every repo
through an org team. Adopting dotbot in a dotCMS repo needs only the workflow
file — there is no per-repo secret to set and no per-repo access to grant.

- **Override**: a repository-level secret with the same name takes precedence
  over the org secret (GitHub precedence: environment > repository >
  organization).
- **Repos with outside collaborators**: if anyone outside the org has write
  access to a repo, set a repository-level `DOTBOT_GITHUB_USER_PAT` with a
  placeholder (dummy) value so the real org token is never exposed to them.
  Approval with the placeholder fails and is logged as a warning; reviews still
  post. To keep auto-approval there, pass `${{ github.token }}` instead.

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
  pull-requests: write
  issues: write
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
    # Job-level, not workflow-level: every comment starts this workflow, and a
    # workflow-level group is joined even when the job is skipped, so an
    # ordinary comment would cancel a waiting /dotbot run. queue: max keeps
    # back-to-back /dotbot requests queued.
    concurrency:
      group: dotbot-act-${{ github.event.issue.number || github.event.pull_request.number || github.ref }}
      cancel-in-progress: false
      queue: max
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
        uses: dotCMS/openrouter-code-review-action@v2
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
  models:                            # optional — extra panel reviewers
    - openai/gpt-5
    - google/gemini-2.5-pro
  discussion_rounds: 2               # optional — integer 1–5
  decision:
    policy: unanimous                # unanimous | lead (primary model adjudicates)
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

### Collaborative Review Panel

Set `review.models` (a list) — or the additive `review_models` Action input —
to select additional models alongside `review.model`. Every review uses the
coordinator, whether the roster contains one model or several:

1. Investigates the PR concurrently, with correctness and security review by every model.
2. Runs `review.discussion_rounds` shared rounds (default 2, allowed 1–5).
   Every model sees the complete preceding rounds, can inspect code again, and
   explicitly retains or retracts findings using stable issue IDs and evidence.
3. Applies `review.decision.policy`: `unanimous` preserves final minority findings
   and requires all final positions to approve; `lead` adds an adjudication turn
   by the primary model, which must explain each issue disposition with evidence.
4. Publishes only final actionable findings and **one consolidated PR summary per attempt**.
   No intermediate findings or debate are posted. Exact duplicate findings are
   collapsed; reviewers are instructed to address existing IDs rather than
   introduce paraphrased duplicates.

The Action creates an **in-progress** summary before validation/install. Each attempt
updates only its own comment to a final verdict, **failed**, or **cancelled**, with a
visible run/attempt link. Historical attempt summaries remain as history, rather
than being reused as the current result. Separate comment ownership prevents an old
cleanup step from overwriting a newer attempt's status.

Explicit discussion messages are logged when `stream_agent_messages` is enabled;
raw concurrent model streams are suppressed. A local transcript is saved to
`.dotbot-context/review-discussion.json`. Known configured credentials are redacted,
but repository-sensitive discussion still belongs only in appropriately protected
run logs. No external tracing service or additional framework is required.

Panel tools read the pinned PR commit, list its files, and search its code;
they cannot write files or execute arbitrary shell commands. PRs changing credential
paths (`.env`, `.env.*`, `*.pem`, `*.key`, including deletions/renames) require human
review: no PR file contents are sent to a review model, and approval/thread
resolution are withheld. Public `.env.example`/`.env.sample` templates are allowed.
The checkout must
contain the PR base and head commits (`fetch-depth: 0` is recommended). Existing
per-model timeouts apply to every pass; two discussion rounds mean three passes
per model, plus one primary-model pass with `lead`. Provider billing can continue
briefly for an already-issued request after a timeout.

A structurally validated response with a contradictory verdict gets **one correction
attempt**, reusing its full conversation/tool evidence without rerunning investigation.
It stays inside the original timeout and adds no discussion round. The correction
cannot discard findings or change disposition actions. Malformed JSON, missing fields,
or other errors whose original decisions cannot be protected fail immediately;
a second invalid response and provider/execution errors also fail closed.
The correction can incur one extra structured-output completion.

Reviews start fresh on every run. All applicable unresolved dotbot findings enter
the ledger and receive explicit dispositions; there is no 200-issue truncation.
An Action attempt is bound to its startup head SHA and status-comment ID; deletion or corruption
of that comment prevents publication and approval. Prior findings are reconsidered
through explicit dispositions; prior verdicts never count as current votes. There
is no sequential fallback or review resume/cache path. Panel options are configured
in the YAML file, not new CLI flags or Action inputs. The steering instructions
live in [`prompts/review_panel.md`](prompts/review_panel.md), alongside the general
review guidelines in [`prompts/review.md`](prompts/review.md).

### Org/Repo Variables (`vars.DOTBOT_*`)

The self-hosted workflows read two GitHub **Variables** (org- or repo-level),
so one place can pin the models for every consuming repo without editing its
config file:

| Variable | Mode | Effect |
|----------|------|--------|
| `DOTBOT_REVIEW_MODELS` | review | Org/repo Variable consumed by the self-hosted workflow: replaces the primary/additional model roster while preserving other YAML settings. The `review_models` Action input is additive to the primary model. |
| `DOTBOT_ACT_MODEL` | act | Model slug for `/dotbot` edits, including OpenRouter's `~` "latest" aliases (e.g. `~deepseek/deepseek-flash-latest`). Replaces the in-repo `act:` block for that run. |

Both are optional. With a variable unset, nothing is generated and the in-repo
`.openrouter-review.yml` (or the action default) applies. When set, the variable
is authoritative — a variable set to a malformed slug fails the run instead of
silently falling back.

The act override is rendered to `$RUNNER_TEMP/dotbot-act-model.yml` and passed
through the `config_path` input rather than written into the checkout: act mode
pushes commits to the PR branch, so a generated file inside the worktree could
otherwise be swept into the agent's commit. Review renders in place, since
review never commits.

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
| `github_approval_token` | Token used to approve the PR when the configured final decision policy accepts the patch. Normally a machine-user PAT (e.g. `secrets.DOTBOT_GITHUB_USER_PAT` for `dotCMS-Machine-User`); `${{ github.token }}` also works and approves as `github-actions[bot]`. When unset (or the final policy rejects the patch) reviews post as normal comments with no approval | *(unset)* |
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

> `openai_api_key` is only for the optional SDK-backed act mode. Review mode
> requires OpenRouter and rejects `DOTBOT_PROVIDER=openai`.

## Agent Loop

OpenRouter chat completions have no built-in agent runtime, so the action ships
its own tool-calling loop. Reviewers get read-only `read_file`, `list_files`, and
`search_code` tools against the pinned PR commit. They cannot write files or run
arbitrary shell commands. Act mode separately exposes `read_file`, `write_file`,
and `run_command` so it can implement requested edits.

The loop runs until the model produces a final answer, with a 30-iteration cap.
Tool results feed back as `role: "tool"` messages. Review conversations may be saved
for diagnostics, but are never resumed as a new review's verdict. The direct
`DOTBOT_PROVIDER=openai` SDK option is supported only for act mode.

## What It Posts

- **Inline comments** anchored to exact diff lines. If a line isn't in the
  current diff, the finding is filtered or remapped before submission so
  GitHub does not return 422.
- **PR-level summary** as one issue comment per Action attempt, updated as that
  attempt progresses. Earlier attempts remain identified history. The local CLI
  refreshes its summary by deleting prior summaries. The summary footer records which model and
  reasoning effort produced the review, e.g. `<sub>reviewed by dotbot ·
  deepseek/deepseek-v4-pro-0813 · medium</sub>`.
- **Multi-line suggestions** only when contiguous and short; otherwise a
  single-line comment.

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
- **Editing `.github/workflows/*` needs a PAT, not a permission.** There is no
  `workflows:` scope for `GITHUB_TOKEN` — declaring one makes GitHub reject the
  entire workflow file, so no run ever starts. The `GITHUB_TOKEN` used to push
  cannot create or update workflow files either; to let `/dotbot` edit them, set
  `secrets.REPO_ACCESS_TOKEN` to a PAT carrying the `workflow` scope (the
  workflow already prefers it over `github.token`). Otherwise you'll see
  `refusing to allow a GitHub App to create or update workflow … without
  workflows permission`.

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

The patched certifi, cryptography, PyJWT, and urllib3 minimums are enforced in
`requirements.txt` as well as the resolved `uv.lock`, so pip upgrades vulnerable
versions already installed on a runner.

CI also runs a pinned Trivy filesystem scan for vulnerabilities, misconfiguration,
and secrets, including locked development dependencies, failing on HIGH/CRITICAL
findings. It runs independently of lint/tests;
existing dependency advisories must be fixed rather than silently ignored.

Environment variables of note:

| Variable | Purpose |
|----------|---------|
| `GITHUB_TOKEN` / `GH_TOKEN` | GitHub API token used for review comments (required) |
| `DOTBOT_GITHUB_USER_PAT` | Optional PAT owned by the machine user; enables auto-approval when the final decision policy accepts the patch (see [Automatic PR Approval](#automatic-pr-approval)) |
| `DOTBOT_MODEL_TIMEOUT_SECONDS` | Per-reviewer-model wall-clock budget in seconds (0 disables) | `900` |
| `OPENROUTER_API_KEY` | OpenRouter API key (required for the default provider) |
| `OPENAI_API_KEY` | Optional act-mode SDK key (`DOTBOT_PROVIDER=openai`); not a review provider |

## Release & Versioning

Releases are cut on merges to `main`. Conventional Commit `!` markers or a
`BREAKING CHANGE:` footer bump the major version; ordinary `feat:` bumps the
minor, otherwise the patch increments. This breaking migration cuts v2.0.0.

A release updates only its matching moving major tag and `latest`: v2 releases
advance `v2`/`latest`, while `v1` consumers remain on v1. Editing an older release
cannot downgrade those moving tags. Pin an exact release SHA for immutable code,
a major tag for compatible releases, or `latest` to follow breaking changes.

## Project Status

This action started as a clone of the GitHub codex-review-action and was
migrated to route all model calls through [OpenRouter](https://openrouter.ai)
with a built-in tool-calling agent loop (see [Agent Loop](#agent-loop)). The
Codex SDK remains only for the separate optional act backend and the `eval/`
quality baseline. Production review always uses the OpenRouter coordinator.

The evaluation harness scores candidate models for review-quality parity:

```bash
uv run python -m eval.run_openrouter_baseline   # run candidates over the PR sample
uv run python -m eval.overlap_score           # score overlap vs the Codex baseline
```
