# Codex Review: RQ006 Sigma Sensitivity

Status: review-complete, archived-review scope; not yet frozen in `decision.md`.

Review date: 2026-06-21.

## Scope

Reviewed study package:

- `reports/studies/RQ006_sigma_sensitivity/RQ006_1_sigma_compare_20260618/`

Primary evidence read:

- `01_results/sigma01_vs_sigma002_comparison_report.md`
- `00_entry/index.html`
- `TRACEABILITY.md`

## Overall Verdict

RQ006 should remain a robustness/parameter-sensitivity record, not a substantive
social-compliance result. It supports using the sigma=0.1 full rerun as the
healthier current analysis source, while warning that IPV magnitudes are
materially sensitive to sigma.

Paper-safe phrasing:

> The sigma=0.1 rerun greatly reduces the all-zero artifact seen at sigma=0.02,
> but sigma choice changes IPV magnitudes enough that quantitative conclusions
> should be reported with parameter-sensitivity boundaries.

## Claims That Can Be Carried Forward

1. **Sigma=0.1 is the healthier current data source.**
   The comparison aligned `38,228` sigma=0.1 rows with the sigma=0.02 table.
   Sigma=0.1 has only `17` all-zero cases, compared with `7,318` all-zero cases
   under sigma=0.02. There were no sigma=0.1-only keys, only `10` sigma=0.02-only
   keys, and no agent-type mismatches after track-ID alignment.

2. **The comparison preserves identity and matching integrity.**
   The alignment key includes dataset, folder, scenario index, sorted track ID
   pair, start and end. Swapped agent order did not create false differences.

3. **Sigma materially affects IPV magnitude.**
   Mean absolute differences are large enough to matter:
   `ipv_mean_track_low=0.205`, `ipv_mean_track_high=0.193`, with p95 absolute
   differences around `0.572` and `0.529`. This should be treated as a
   sensitivity boundary for numeric effect sizes.

## Claims To Reject Or Defer

- **Reject:** sigma sensitivity validates any social-compliance verifier claim.
  It only audits estimator-parameter behavior.
- **Reject:** sigma=0.02 should be used for headline conclusions. The high
  all-zero rate makes it unsuitable as the current primary source.
- **Defer:** parameter-invariant quantitative effect sizes. Directional patterns
  may be useful elsewhere, but this package does not establish that every
  manuscript effect is robust to sigma choice.

## Knowledge-Layer Action

Keep RQ006 as `archived-review` unless sigma sensitivity becomes
manuscript-relevant again. If cited, use it only as a robustness appendix note:
current analyses should prefer sigma=0.1, and numeric IPV magnitudes should not
be described as parameter-invariant without an updated sensitivity table.
