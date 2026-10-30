# Collaborative PR review progress

## Current phase

Implementation and local verification complete. The complete spec was approved
by the user. Trivy was separately requested and added to the spec as AC10.

The user subsequently authorized dependency remediation and PR publication.
The targeted upgrades are complete on `feat/collaborative-review-panel`.
Next action: commit/push this branch, create the PR, and check CI/review feedback.

## Baseline

The working tree was clean before task documents were created. The original
review loop was sequential and posted per model; the roster override replaced
all config settings. Baseline targeted command:

```text
.venv/bin/python -m pytest -q tests/test_model_config.py tests/test_config.py tests/test_review_workflow.py tests/test_review_posting.py tests/test_approval.py tests/test_openrouter_client.py tests/test_openrouter_tools.py tests/test_prepare_resume_state.py tests/test_main.py
```

Outcome: 155 passed in 1.98s. No baseline failures in that subset.

## Implementation

- Typed YAML configuration: discussion_rounds (1–5, default 2), decision.policy
  (unanimous or lead); no new CLI flags, Action inputs, or dependencies.
- Standard-library concurrent coordinator, immutable completed-round broadcasts,
  stable issue ledger, explicit evidence-bearing dispositions, deterministic
  minority-preserving or lead-adjudicated final results.
- Isolated per-model configs/clients, raw-stream suppression, redacted explicit
  discussion logs and local transcript; separate panel conversation namespace.
- Read-only pinned-commit inspection, listing and literal search. No shell/write
  tools exposed to the panel. Pinned diff disables external diff/textconv.
- Prior-comment applicability uses pinned content rather than a merge checkout.
  Git-object/diff paths do not follow unrelated live-checkout symlinks.
- Final-only consolidated publication, same-head checks, same-SHA approval
  idempotency, final prior-thread dispositions, and dry-run behavior.
- Workflow roster overrides preserve panel/act settings; docs/examples updated.
- Trivy runs independently in CI, pinned by action SHA and scanner version,
  with read-only permissions, no persisted checkout token, and HIGH/CRITICAL
  vulnerability/misconfiguration/secret failures.

## Final verification

- `make qa`: passed (Ruff formatting/lint and MyPy).
- `uv sync --dev --locked`: passed after updating only certifi 2026.1.4 →
  2026.10.23, cryptography 46.0.5 → 50.0.2, pyjwt 2.11.0 → 2.15.1, and
  urllib3 2.6.3 → 2.8.0 in uv.lock.
- `uv run pytest -q`: 870 passed in 9.81s after dependency remediation,
  including head-change, prior-thread, and live-checkout-symlink safeguards.
- `uvx --from actionlint-py==1.7.12.25 actionlint -ignore 'unexpected key "queue" for "concurrency" section'`: passed.
- `git diff --check`: passed.
- Local Trivy 0.74.0 scan (vuln,misconfig,secret; HIGH,CRITICAL; .venv skipped):
  exit 0, zero vulnerability/misconfiguration/secret findings after remediation
  (previously 14 vulnerability advisories). No ignore rules or unrelated upgrades
  added. Report: /tmp/dotbot-trivy-remediated.json.

## Acceptance criteria

AC1–AC10 and the local-validation part of AC11 are verified. The full suite,
QA, and the configured Trivy gate pass. PR publication/CI verification remains
in progress. See tests/test_review_coordinator.py,
tests/test_review_inspection.py, tests/test_review_workflow.py,
tests/test_model_config.py, tests/test_dotbot_workflow_model_vars.py,
tests/test_approval.py, and tests/test_ci_security.py.

## Changed paths

- cli/review/coordinator.py — new panel and decision contracts.
- cli/workflows/review_workflow.py — panel integration and final publication.
- cli/clients/openrouter_client.py, cli/clients/openrouter_tools.py — isolated,
  pinned read-only inspection.
- cli/core/config.py, cli/core/model_config.py — typed configuration.
- cli/review/approval.py, cli/review/dedupe.py, cli/review/patch_parser.py —
  final-policy approval, pinned prior evidence, and Git path handling.
- .github/workflows/ci.yml, .github/workflows/dotbot-review.yml — Trivy and
  configuration-preserving roster override.
- .openrouter-review.yml, action.yml, README.md, cli/README.md — examples/docs.
- uv.lock — targeted security updates to the four affected dependencies.
- Tests listed above and these task documents.

## Limits and remaining risks

- The configured Trivy severity gate is clean with the current database; this
  does not assert the absence of lower-severity or future advisories.
- Local provider/GitHub-write integration tests use fakes; live model review was
  not invoked directly. PR creation and its normal GitHub workflows are now
  authorized by the user.
- Deduplication uses matching issue text/location, ignoring severity labels and
  preserving the highest severity. Paraphrases should reuse existing ledger IDs;
  semantic deduplication is not guaranteed.
- Panels start fresh rather than resume old verdicts. Existing single-model and
  legacy paths remain supported.
- Already-issued provider requests cannot be remotely cancelled by the local
  timeout; isolated daemon workers are abandoned and may incur further billing
  until the process/request ends.
- Evidence citations and consensus are not proof of factual correctness. Known
  configured credentials are redacted, not all possible sensitive source data.
