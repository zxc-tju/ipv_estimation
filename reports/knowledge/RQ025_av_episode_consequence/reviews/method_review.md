# RQ025 WP7 Method Review v2

## Context

This review checks whether the already generated RQ025 matched-effect outputs are methodologically safe to summarize as a bounded or mixed descriptive result set. It does not recompute any effect. Upstream work already froze the pair design (`461/475` primary, `361/475` sensitivity), assembled the accepted ego/counterpart ledgers, computed point estimates in `effect_analysis_v4/point/`, and added branch-specific treated-case bootstrap intervals in `effect_analysis_v5/`.

The question for this review is narrower: do any method issues in the visible outputs overturn the bounded or no-stable-positive reading, or do they mainly require tighter wording?

## Decision

`REQUEST_FIX`

The visible issues are important reporting and interpretation risks, but none of them overturn the bounded or mixed bottom line. The needed fixes are all framing fixes:

- define the descriptive unit and inferential unit separately;
- describe bootstrap intervals as treated-case-cluster intervals under a frozen with-replacement ledger;
- prevent TTC mean-only narration;
- keep the seven-outcome family visible and non-selective.

## Study Design Readout

- Frozen matching branches:
  - `primary`: same-scenario, with replacement, `q50`, `461` matched treated episodes.
  - `sensitivity`: same-run, with replacement, `q50`, `361` matched treated episodes.
- Descriptive unit: matched treated-control episode-onset pair row.
- Nesting: pair rows are nested within treated case and treated run.
- Ego outcomes: all complete pairs for `future_min_ttc_s` and `future_min_ttc_lt_2s`.
- Counterpart main: complete pairs with `both_non_scripted = true`.
- Counterpart all-row: explicitly descriptive sensitivity only.
- Uncertainty: treated-case cluster bootstrap, `B=2000`, seed `20260824`, performed separately for primary and sensitivity branches.

## What Holds

- Denominators stay frozen and visible. Primary rows remain `461`; sensitivity rows remain `361`.
- Missingness is outcome-blind and explicitly tabulated. Ego shrinkage comes from non-closing rows; counterpart shrinkage comes from missing counterpart series and a small number of primary control alignment failures.
- The bootstrap unit is at least aligned with the strongest obvious dependence source on the treated side: repeated treated cases.
- The family is still carried as a full seven-outcome set rather than a post-hoc winner table.

## What Needs Fixing

### P1. Independent unit wording

The point report says the independent unit is the pair row, while the bootstrap reports use treated case as the resampling unit. These are not actually contradictory if separated into descriptive and inferential roles, but the current wording invites confusion.

Required fix:
- `pair row` for descriptive summaries.
- `treated case` for uncertainty.

### P1. With-replacement control reuse boundary

The control side is reused heavily:

- primary bootstrap: `150` unique control cases across `461` rows; `105` control cases reused more than once; maximum linked pair count `11`.
- sensitivity bootstrap: `120` unique control cases across `361` rows; `86` control cases reused more than once; maximum linked pair count `10`.

This does not invalidate the current descriptive bootstrap, but it does mean the intervals should not be sold as if an independent control pool were re-sampled.

### P1. TTC mean-versus-median conflict

The ego TTC result is not direction-stable:

- primary: mean difference `+14.159877 s`, median difference `-0.104292 s`, TTC<2 risk difference `-0.010667`.
- sensitivity: mean difference `-88.057032 s`, median difference `+0.076227 s`, TTC<2 risk difference `-0.016835`.

That is the strongest reason the write-up must not narrate TTC as a monotone benefit or monotone harm. The appropriate reading is bounded and mixed, with the mean strongly tail-sensitive.

### P1. Multiplicity and joint-family interpretation

The seven outcomes are carried through as required, but no branch-consistent single metric should be upgraded into a headline effect. The results should stay at the family level.

### P2. Missingness explanation

The tables already disclose counts, but the reviewer-facing narrative should add one sentence that the denominator shrinkage is pre-existing and outcome-blind:

- ego complete pairs: `375/461` primary, `297/361` sensitivity because some accepted ego windows have no closing rows;
- counterpart complete pairs: `445/461` primary, `351/361` sensitivity because some counterpart series are missing, plus `2` primary control alignment failures;
- counterpart main subset: `382/461` primary and `318/361` sensitivity because the predeclared non-scripted boundary excludes scripted rows.

### P2. Simplify direction labels

Labels such as `provisionally_branch_consistent_metric_mixed` are harder than needed. Zero medians should be described as concentration at no change, not as a second sign channel that inflates label complexity.

## Can Any Visible Issue Overturn The Bounded / No-Stable-Positive Conclusion?

No.

None of the visible issues creates hidden evidence for a stable positive effect. The main risks are the opposite:

- over-reading a tail-sensitive TTC mean,
- over-reading branch-consistent but small counterpart metrics,
- over-reading descriptive bootstrap intervals as stronger than they are.

If the wording is tightened, the bounded or mixed conclusion remains supportable.

## Allowed Sentence

在冻结匹配账本与 treated-case clustered bootstrap 下，七个预先列出的结局整体呈现以 bounded 或 mixed 为主的描述性对比，当前结果不支持稳定正向效应叙事。

## Forbidden Phrasings

- 证明自动驾驶更安全
- 显示明确收益
- 稳定改善
- 没有效应
- 等效
- 非劣
- CI 跨 0 因而证明无差异
- 显著改善
- 因而导致
- 优于人类
