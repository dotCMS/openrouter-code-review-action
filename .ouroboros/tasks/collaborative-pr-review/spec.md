# Collaborative PR review coordinator

Approval status: APPROVED — user confirmed the complete spec with “yes”; implementation authorized.

## Goal

Make the selected OpenRouter reviewer models a collaborative review panel. They investigate the same PR concurrently, exchange evidence-backed challenges, reconsider findings, and publish only a consolidated final review and approval decision. Debate belongs in run logs, not PR comments.

## Agreed decisions

- Reuse the existing clients; add a bounded coordinator, not Agno or another framework.
- Agents must actually receive other agents' messages and have an opportunity to respond and inspect code again.
- Explicit discussion messages may be logged; the debate must not be posted to GitHub.
- Configure decision policy as `unanimous` or `lead`, defaulting to `unanimous`.
- Missing reviewers, timeouts, and invalid responses must prevent automatic approval under either policy.

## Proposed defaults and configuration

Use the existing consuming-repository `.openrouter-review.yml` (or its configured path):

```yaml
review:
  model: deepseek/deepseek-v4-pro-0813  # primary reviewer; lead when policy=lead
  models:
    - openai/gpt-5
    - google/gemini-2.5-pro
  discussion_rounds: 2               # integer 1–5; default 2
  decision:
    policy: unanimous                # unanimous | lead; default unanimous
```

No new CLI flags or Action inputs are necessary. Existing roster selection and config precedence remain supported. Invalid values, unknown decision keys, and review-only options under `act` fail validation. Existing workflow roster overrides must preserve other configuration rather than replace the entire file.

## Scope and behavior

1. Apply collaboration to OpenRouter review runs with at least two distinct selected models. Single-reviewer runs, the legacy OpenAI provider, and act mode retain their current execution paths.
2. Capture one consistent PR/context snapshot and reviewed head SHA. Each reviewer has isolated config, client usage accounting, and conversation state. Never mutate shared `config.review_model` during concurrent calls.
3. Run the initial investigations concurrently. Every reviewer checks correctness and security implications: trust boundaries, authorization, injection, data exposure, and changed dependency/configuration behavior where relevant. Findings require concrete code evidence, not speculative security claims.
4. Run the configured number of discussion rounds. Each reviewer receives the same immutable transcript through the preceding round, with model attribution and stable finding references. Reviewers can inspect repository code again, challenge or support findings, and explicitly retain, revise, or retract them. Collect a full round before broadcasting its messages; completion order must not change who sees what.
5. Require structured, validated outputs for evidence, issue dispositions, and final review positions. Malformed or incomplete positions are failures, not approval votes. Do not infer agreement from silence.
6. Under `unanimous`, retain the union of reviewers' final actionable findings, consolidating duplicates without losing distinct issues. Any remaining finding, applicable carried-forward issue, or dissent blocks approval. Initial findings explicitly retracted during discussion need not remain blocking.
7. Under `lead`, the primary reviewer adjudicates the discussion and explicitly explains each retained or rejected disputed finding using evidence. Its final result determines published findings and correctness. An unsupported/incomplete adjudication prevents approval; the lead cannot override an execution failure.
8. Publish the final findings through existing anchoring, remapping, and deduplication code, then publish one panel summary with the reviewed SHA, participants, policy, security conclusions, and concise unresolved issues where relevant. The full debate and intermediate findings are not posted. Keep existing token requirements and same-SHA approval idempotency. Do not introduce automatic merge or a new request-changes policy.
9. Validate that the PR head has not changed before publication/approval; never approve a different head based on stale deliberation. Preserve existing safeguards for applicable prior findings and optional stale-thread resolution. Partial panel execution must not resolve prior threads or publish a successful final verdict.
10. Log attributed, explicit discussion messages and decisions in round order and save a local transcript under existing context artifacts. Do not log private reasoning traces or credentials as debate. Suppress unlabelled concurrent model streams in this path. Existing stream-output settings govern discussion logging; dry-run still performs deliberation but never changes GitHub.
11. Keep existing per-model timeout and tool-iteration limits. Discussion rounds are bounded and cannot grow autonomously. Any timeout or model failure ends the panel without approval or partial publication; report the failing participant/phase. Do not silently skip a selected reviewer.
12. Use read-only exploration for collaborative passes: supply the pinned diff/context and allow repository inspection, but do not expose write tools, arbitrary shell execution, or GitHub mutation capabilities to reviewer models. Add only the read-only exploration support required for code investigation. Do not weaken act-mode validation or claim that prompt instructions alone sandbox code.
13. Preserve model-specific resume/cache separation. Reuse cached context only when its project/model/reviewed-SHA scope is valid; regenerate the current panel discussion and final decision rather than treating a prior verdict as a current vote. Changes to the roster or policy must not reuse a stale panel verdict.

## User-approved scope addition

The user additionally requested a Trivy workflow step. Add a pinned filesystem
scan as an independent CI job on the existing push/PR triggers, with read-only
permissions and no persisted checkout credentials. Scan vulnerabilities,
misconfiguration, and secrets; fail on HIGH/CRITICAL findings. Do not suppress
findings or upgrade unrelated dependencies silently.

AC10: workflow assertions and actionlint validate the pinned scan configuration;
a local Trivy run proves findings produce a nonzero exit. Existing vulnerability
findings are reported as outstanding dependency debt, not hidden to make CI green.

The user subsequently authorized fixing the dependency advisories and opening a
PR. Refresh only the four affected locked dependencies (certifi, cryptography,
pyjwt, urllib3), retain the scan's failure threshold without suppressions, rerun
QA/tests/Trivy, and publish the complete coordinator/security change as a PR.

AC11: the configured HIGH/CRITICAL Trivy scan exits zero, QA and the full test
suite pass after the targeted upgrades, and the PR is created with validation
results and monitored for CI/review feedback.

## Non-goals

- A general-purpose agent framework, dynamic specialist recruitment, or an unbounded chat service.
- MCP, new dependencies, external tracing services, hosted dashboards, or a transcript UI.
- Posting the debate, automatic PR edits, deployment, or merging.
- Broad sandbox redesign or unrelated act-mode hardening.
- Claiming multi-model agreement proves correctness or security.

## Acceptance criteria

| ID | Observable outcome | Verification |
| --- | --- | --- |
| AC1 | Existing configs still load; the new policy and bounded round count validate with the stated defaults; roster overrides preserve them. | Config and workflow-generation tests, including malformed types/values and act-mode rejection. |
| AC2 | At least two investigations overlap in execution without shared config, usage, or conversation corruption. | Deterministic fake-client synchronization test, not timing-only assertions; per-model state assertions. |
| AC3 | Every participant sees all prior-round messages and can revise/retract after challenge; no participant sees a partial current round. Security review is explicitly requested. | Fake transcripts and prompt tests covering challenges, follow-up inspection, retractions, and deterministic ordering. |
| AC4 | Both policies follow the final positions and handle disagreement as specified. Lead overrides have evidence-backed dispositions; incomplete execution never approves. | Policy tests covering unanimous approval, dissent, retraction, lead rejection/retention, malformed adjudication, missing votes, and timeout. |
| AC5 | No GitHub mutation happens during investigation/discussion; only final consolidated findings and one panel summary are published. | Posting spies assert ordering/counts; duplicate and distinct findings are checked; transcript content is absent from PR payloads. |
| AC6 | Stale SHA, invalid results, and failures prevent approval; prior findings, same-SHA idempotency, and dry-run safeguards remain intact. | Workflow/approval regressions and head-change tests. |
| AC7 | Collaborative agents can inspect needed code but cannot use write/arbitrary-command tools or post directly. | Tool-boundary tests plus an investigation fixture requiring context outside the changed hunk. |
| AC8 | Debate is identifiable in local logs/artifacts without leaking private reasoning or credentials, respects stream settings, and resumes do not reuse stale panel decisions. | Transcript, logging, and resume/isolation tests. |
| AC9 | Single-model review, legacy-provider behavior, act mode, and existing review output consumers remain compatible. | Existing targeted tests, full pytest suite, formatting/lint/type checks, and documentation examples. |

## Smallest implementation plan and expected touched areas

1. Add typed configuration and panel-message/decision contracts with focused validation tests (`cli/core/model_config.py`, `cli/core/config.py`, review models or a focused panel module).
2. Add one focused coordinator and discussion prompts in `cli/review/`. Use standard-library concurrency and isolated existing model clients. Keep decision policy code separate from model-generated claims so failure conditions are enforced deterministically.
3. Separate collecting review results from publishing them in `cli/workflows/review_workflow.py`, preserving the single-reviewer path and reusing posting, prior-comment, and approval helpers. Integrate read-only exploration, per-reviewer state, and transcript artifacts through existing client/review modules where needed.
4. Preserve new configuration when `.github/workflows/dotbot-review.yml` replaces the roster. Update relevant cache/resume handling and tests only where collaboration changes their assumptions.
5. Update `README.md`, `cli/README.md`, configuration examples, and Action descriptions that currently describe sequential reviewers/per-model publication or unconditional unanimity. Do not add unused public options.
6. Run focused regressions, the full test suite, and repository QA; inspect the final diff against this spec. Live paid-model/GitHub-write checks require separate authorization.

## Risks

- More calls and larger transcripts increase latency/cost; two discussion rounds add two passes per model, plus lead adjudication when selected.
- Discussion may cause correlated mistakes or persuade a correct minority to retract. Record evidence and preserve unresolved dissent rather than equating consensus with proof.
- Existing clients contain mutable per-call state and permissive review tools; concurrency must not merely wrap today's publishing loop in threads.
- Read-only source inspection is not execution isolation. Repository content and other agents' messages remain untrusted inputs.
- Actions logs can be accessible to repository readers; discussion must not include credentials or unnecessarily reproduce sensitive data.
