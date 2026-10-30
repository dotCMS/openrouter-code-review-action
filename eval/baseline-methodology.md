# Codex → OpenRouter Bake-off Methodology

> **Scope.** This document defines how we measure parity between the
> existing Codex review path and the new OpenRouter review path. It
> binds the Seed's two parity ACs to falsifiable, reproducible
> procedures.
>
> | Seed AC | Pass condition |
> |---------|----------------|
> | AC 1 — `finding_overlap` | ≥ 80 % overlap on True-Positive findings between OpenRouter and Codex on the sample |
> | AC 2 — `false_positive_discipline` | Aggregate false-positive rate of OpenRouter ≤ Codex baseline on the sample |
>
> Anything not specified here defers to `prompts/review.md` (the
> reviewer prompt) and the schema in `cli/core/models.py`
> (`REVIEW_OUTPUT_SCHEMA`).

---

## 1. Sample PR set

### 1.1 Why a fixed sample

Picking PRs ad-hoc each time the model changes makes scores
incomparable. The sample is fixed once, snapshotted by **(repo,
PR-number, head-SHA)** triples, and re-used for every bake-off — the
initial Codex → OpenRouter swap and every subsequent model bump done
through `.openrouter-review.yml`.

### 1.2 Selection criteria (PR eligibility)

A PR is eligible for the sample if **all** of the following hold:

1. **Public, accessible source.** The PR diff and head SHA are
   reachable to anyone running the bake-off (no secrets, no private
   submodules required to reproduce the diff).
2. **Already merged or merge-ready.** Open PRs are allowed only if their
   head SHA is pinned and their diff will not change during evaluation.
3. **Real engineering signal.** The diff is non-trivial — at least one
   of: logic change, security-sensitive change, dependency/version
   change, or schema/config change. Pure-style or pure-rename PRs are
   excluded.
4. **Bounded size.** Diff is between **20 and 1,500** added+removed
   lines. Smaller PRs leave too little surface to overlap on; larger
   PRs blow context windows and confound the comparison.
5. **Reviewable in a single turn.** No binary/lockfile-only diffs; no
   PRs whose review depends on out-of-tree state Codex never had access
   to either.
6. **Belongs to a representative file mix.** Across the full sample,
   coverage must include each of: Python, TypeScript/JavaScript,
   Java/Kotlin (dotcms reality), YAML/Action config, and tests.

### 1.3 Required diversity (across the 5-10 sample)

The sample as a whole must cover the bake-off-critical scenarios.
A single PR can satisfy more than one slot.

| Slot | Why it matters | Minimum count |
|------|----------------|---------------|
| Real bug present in the diff | Anchors True-Positive measurement | ≥ 2 |
| Clean diff (no real bug) | Anchors False-Positive measurement | ≥ 2 |
| Continuation case (synchronize push after a prior review) | Exercises `prior_review_state` reuse on both paths | ≥ 1 |
| Force-push that breaks reviewed-SHA ancestry | Exercises the Seed's "fresh full review" fallback | ≥ 1 |
| Web-search-relevant change (CVE-tagged dep bump, deprecated API) | Exercises `web_search_mode` parity | ≥ 1 |
| Multi-line suggestion-worthy fix | Exercises inline-anchor validity | ≥ 1 |

### 1.4 Sample size

5 PRs minimum, 10 maximum (per `seed.yaml` AC 1). Bigger samples are
encouraged but cost reviewer time linearly; below 5 the overlap and
FP rate metrics become too noisy to defend the AC.

### 1.5 Registry format

The sample lives in `eval/sample-prs.yaml`. Each entry pins:

- `id` — short stable identifier (used in run-artifact filenames)
- `repo` — `owner/name`
- `pr` — PR number
- `head_sha` — the exact commit reviewed (snapshot)
- `slots` — one or more of the diversity-slot keys from §1.3
- `notes` — free-form rationale: why this PR is in the sample

A PR may be removed from the sample only by ADR-style note in
`sample-prs.yaml` recording why it was unsuitable. Silent swap-outs
defeat the purpose of a fixed sample.

---

## 2. Labeling rubric

For every finding emitted by either Codex or OpenRouter on a sample
PR, exactly one **label** is assigned.

### 2.1 Label set

| Label | Meaning |
|-------|---------|
| `TP` | **True positive.** Concrete, actionable bug introduced or surfaced by this diff. The author would fix it if shown. |
| `FP` | **False positive.** Reported finding is wrong, non-actionable, or excluded by the reviewer prompt. See §3 for the precise definition. |
| `DUP` | **Duplicate.** The same underlying issue is already counted under another finding from the **same** review run. The first instance keeps its real label (`TP` or `FP`); each subsequent instance is `DUP`. Cross-run duplicates (same issue in Codex and OpenRouter) are **not** `DUP` — they are the basis of finding-overlap (§4.1). |
| `CARRY` | **Carried-forward prior finding.** A finding that originates from `carried_forward[]` in the schema (a re-adjudicated prior comment), not a new issue. Reuses the prior run's label; does **not** count toward new-finding totals. |

`DUP` and `CARRY` exist purely to keep the TP/FP counts honest. They
are excluded from rate denominators.

### 2.2 Required fields per labeled finding

```yaml
finding_id:           # stable per-run id (e.g. f042)
run:                  # codex | openrouter
pr_id:                # matches sample-prs.yaml id
path:                 # finding code_location.absolute_file_path
line_start:           # finding code_location.line_range.start
line_end:             # finding code_location.line_range.end
severity:             # P0 | P1 | P2 | P3 (from finding title; see prompts/review.md)
label:                # TP | FP | DUP | CARRY
rationale:            # ≤ 1 sentence — why this label
matched_finding_id:   # for DUP, the id this duplicates
match_to_other_run:   # optional — id of the matching finding in the OTHER run, if any (overlap evidence)
```

Findings with `severity` of P3 (nits) are still labeled but treated
specially in scoring (§4.4).

### 2.3 Labeling procedure

1. **Two reviewers, independent.** Reviewer A and Reviewer B label
   each finding without seeing the other's labels.
2. **Tie-break on disagreement.** If A and B disagree, a third
   reviewer adjudicates. The adjudicated label is final and the
   disagreement is recorded in the run artifact.
3. **Inter-rater agreement** on `TP` vs `FP` should be tracked
   (Cohen's κ or simple percent-agreement). If agreement falls below
   0.6 κ on any single PR, the rubric itself is suspect — escalate
   before scoring.
4. **One reviewer is acceptable** for the initial bake-off if a second
   is unavailable, but the run must mark `solo_labeled: true` and the
   confidence interval on FP rate widens accordingly. Subsequent
   model swaps should restore two-reviewer labeling.
5. **No model self-grading.** A model under evaluation may not label
   its own findings.

### 2.4 Labeling order

Label **per PR**, not per run. Both runs' findings on the same PR
are labeled together, so duplicates and cross-run matches surface
naturally. Within a PR, label by file then by line to keep
neighboring findings adjacent in the reviewer's mental model.

---

## 3. False-positive definition

A finding is **FP** if **any** of the following hold. The list is
ordered by precedence — the first matching rule wins.

1. **Phantom behavior.** The reported behavior does not actually
   occur in the diff or the surrounding repository code as it stands
   at `head_sha`. Examples: nullable claim on a value the type system
   guarantees non-null at that path; off-by-one claim that the
   surrounding loop bound contradicts.
2. **Pre-existing.** The flagged code was not introduced or modified
   by this diff. Re-flagging unchanged code violates the prompt's
   rule 4 ("bug was introduced in the commit").
3. **Speculation without evidence.** Finding posits "this *may*
   break X" without naming the concrete repository code that would
   break. Violates prompt rule 7.
4. **Out-of-scope category.** Finding is purely:
   - formatting / style / personal preference,
   - documentation typo,
   - changes outside the PR diff,
   - intentional change clearly part of the PR's purpose.
   These are explicitly excluded by `prompts/review.md`.
5. **Schema-invalid.** Finding violates the structured-output
   contract: missing `code_location`, non-integer line range, line
   range that does not overlap the diff. *(These are also caught
   programmatically — see AC 3 — but they still count as FP if any
   slip through to a posted comment.)*
6. **Hallucinated reference.** Finding cites a file, function, or
   API that does not exist at `head_sha`.

**Not FP** (these stay TP unless another rule fires):

- Severity disagreement. A finding labeled P1 that a reviewer thinks
  is "really P3" is still TP if the underlying bug is real. Severity
  miscalibration is tracked separately (§4.4) but does not flip the
  label.
- Suggestion that is suboptimal but not wrong. A finding that
  identifies a real issue and proposes a workable-but-imperfect fix
  is TP.
- Verbose body text. Wordiness alone does not make a finding FP.

### 3.1 The non-actionable edge case

A finding can be **technically correct yet non-actionable** — e.g.
"this third-party library has a known CVE we cannot patch from this
repo." The prompt's rule 5 ("the author would fix it if made aware")
is the tiebreaker:

- If a reasonable PR author at this repo would open a follow-up to
  address it → **TP**.
- If no reasonable action is available from this repo → **FP**
  (rule 3, speculation without actionable repo evidence).

---

## 4. Codex baseline measurement procedure

### 4.1 Per-PR run protocol

The protocol below describes the historical single-agent parity benchmark. In v2,
OpenRouter captures exercise the collaborative production coordinator, while Codex
captures call the SDK directly, read-only, with pinned PR head/base and shared
review guidelines. Their orchestration and prompt context differ; new captures
are end-to-end workflow comparisons, not evidence of same-prompt parity. Both
runners retain dry-run/no-publication behavior. These captures record runner
version `2`; historical artifacts are not refreshed automatically (use `--force`
for an intentional paid recapture).

The registry's legacy `head_sha` is the merge revision used by the existing
human-labeled corpus; it is preserved, not silently relabeled. New captures use
explicit `pr_head_sha` and `pr_base_sha` verified against GitHub PR metadata and
record the original merge separately. OpenRouter additionally requires the live
PR snapshot to still match both pinned revisions. Historical merge-based labels
must be revalidated before scoring new PR-head captures; the scorer continues to
reject different (or missing) captured revisions.

For each PR in `sample-prs.yaml`:

1. **Pin the head SHA.** Check out `head_sha` exactly. Do not
   evaluate against a moving branch tip.
2. **Codex run.** Invoke the legacy Codex action against `head_sha`
   with the recorded model, reasoning effort, and web-search mode.
   Capture:
   - the structured `ReviewRunResult` JSON (all `findings[]` and
     `carried_forward[]`),
   - the posted PR-level summary,
   - the inline review comments actually posted (PostingOutcome).
3. **OpenRouter run.** Repeat with the new OpenRouter path on the
   same `head_sha`, same prompt, same web-search mode, same
   reasoning effort.
4. **Persist artifacts.** Write to `eval/runs/<pr_id>/codex.json`
   and `eval/runs/<pr_id>/openrouter.json` with the schema in §4.2.
5. **Label.** Apply the rubric in §2 across both runs together.
6. **Score.** Compute the metrics in §4.3 per PR; aggregate across
   the sample.

### 4.2 Per-run artifact schema

```yaml
pr_id:                # matches sample-prs.yaml
run:                  # codex | openrouter
head_sha:             # what was reviewed
model:                # OpenRouter slug or Codex model id
reasoning_effort:     # minimal | low | medium | high
web_search_mode:      # disabled | cached | live
prior_review_state:   # null | { reviewed_sha, summary_metadata, ... }
review_run_result:    # raw ReviewRunResult JSON
posted:
  summary_id:         # github comment id of summary
  inline_ids:         # list of github review-comment ids
  posting_outcome:    # batch_submitted / per_comment_fallback / skipped_after_422 counts
labels:               # list of labeled findings (§2.2)
solo_labeled:         # bool; true if only one reviewer
```

### 4.3 Per-PR metrics

Let, for a single PR:

- `TP_codex` = count of findings in the Codex run labeled `TP`.
- `FP_codex` = count labeled `FP`.
- `TP_openrouter`, `FP_openrouter` = same on the OpenRouter run.
- `TP_match` = count of OpenRouter `TP` findings whose
  `match_to_other_run` points at a Codex `TP` (and vice versa —
  the relation is symmetric and many-to-many is collapsed via
  §4.5).

Then:

- **Per-PR FP rate** for run `r` = `FP_r / max(1, TP_r + FP_r)`.
- **Per-PR finding overlap** = `TP_match / max(1, TP_codex)`.
  (Uses Codex TP set as the denominator — overlap is "what fraction
  of the Codex baseline did OpenRouter also catch." Symmetric
  Jaccard is reported alongside but not used as the gate.)

### 4.4 Sample-level aggregate (the AC gate)

- **Aggregate FP rate** for run `r` =
  `sum(FP_r) / max(1, sum(TP_r + FP_r))` across all PRs in the
  sample.
- **Aggregate overlap** =
  `sum(TP_match) / max(1, sum(TP_codex))`.

Pass conditions (per `seed.yaml`):

- `aggregate_overlap >= 0.80` → AC 1 passes.
- `aggregate_FP_rate_openrouter <= aggregate_FP_rate_codex` → AC 2
  passes. A small absolute tolerance of **+ 0.05** is acceptable to
  absorb rubric noise on a 5-10 PR sample, but only if it is called
  out explicitly in the bake-off report.

P3 findings are excluded from the FP-rate denominator and the
overlap numerator. They are reported separately so reviewer-prompt
nit-discipline regressions are visible without dominating scoring.

### 4.5 What counts as "the same finding" across runs

Two findings (one in each run) match for overlap purposes when
**all three** hold:

1. Same `path`.
2. Line ranges overlap by ≥ 1 line **or** are within ± 3 lines of
   each other (tolerance for slightly different anchor choices on
   the same bug).
3. The reviewer judges them to describe the same underlying bug —
   not just adjacent code. This is the rubric's `match_to_other_run`
   field; it is a labeling decision, not a regex.

A Codex finding can match at most one OpenRouter finding and vice
versa. Ties are broken by closest line-range center.

---

## 5. Reproducibility

- **Snapshot SHAs, not branches.** Always run against `head_sha`,
  never against a branch reference, so the diff is identical for
  every reviewer and every model bump.
- **Capture model + effort + web-search-mode in the artifact.** A
  bake-off score is meaningless without those.
- **Commit run artifacts.** `eval/runs/<pr_id>/*.json` is part of
  the bake-off evidence. Future model swaps re-run the same sample
  and append new artifacts; the historical artifacts stay
  in-repo for trend tracking.
- **Cost log.** Each run artifact records token usage from the
  streaming `usage` payload (already wired in
  `cli/clients/openrouter_stream.py::StreamingResult`). FP-rate
  parity at 10× cost is not a win.

---

## 6. Out of scope (explicitly)

This methodology covers AC 1 and AC 2 only. Other ACs have their
own evidence:

- **AC 3** (inline anchor validity) — covered by
  `tests/test_review_posting.py` and the `posting_outcome`
  telemetry.
- **AC 6** (token streaming) — covered by
  `tests/test_openrouter_stream.py`.
- **AC 8** (single-edit model swap) — covered by
  `cli/core/model_config.py` and `.openrouter-review.yml`.

A PR sampled here does **not** validate those ACs; do not conflate.
