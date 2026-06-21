# ChatGPT Review — RQ001 Online IPV Interval Deployability

- Reviewer: ChatGPT (GPT-5.5 Pro)
- Review date: 2026-06-21
- Executions reviewed:
  - `RQ001_1_current_ipv_distribution_20260618`
  - `RQ001_2_interval_query_20260618` (superseded context only)
  - `RQ001_3_online_interval_lock_20260619` (primary evidence)
- Review status: **PASS WITH REQUIRED METHOD AND CLAIM BOUNDARIES**
- Recommended decision role: **accept the lane-referenced causal rolling-IPV + split-conformal result as the strongest currently evaluated online interval estimator; do not yet freeze stronger no-leakage, universal-transfer, hard-constraint, or group-norm claims**

## 1. Overall verdict

RQ001 provides a strong and practically useful answer to the interval-estimation question. The best evaluated online signal is not predicted risk or PET-bin lookup, but the driver's own early-window causal rolling-IPV reconstructed from a strict trajectory prefix and a static map-lane reference. Conditional quantile estimation followed by split-conformal calibration produces substantially sharper intervals while preserving near-nominal coverage, and it is the only compared method to reach approximately 0.90 coverage in the locked Leave-Waymo-Out evaluation.

The core interval-estimation finding should be retained. However, it must be separated from two stronger claims that RQ001 alone does not establish:

1. that the early-window target/label construction is completely free of overlap leakage; and
2. that a self-anchored interval by itself constitutes a valid population social norm.

The first requires a non-overlapping post-anchor target rebuild. The second belongs to RQ002 and currently requires a guarded hybrid design rather than pure self-anchor.

## 2. Findings supported by RQ001

### 2.1 Risk/PET is not the main lever for interval sharpness

Even oracle PET narrows the empirical IPV interval only modestly relative to the global reference. The primary report gives widths of approximately `0.833` versus `0.857`, or roughly a 3% reduction. This shows that the limitation of the original `predict PET -> risk bin -> empirical envelope` approach is not merely PET-prediction error; risk contains limited information about the large between-driver component of interval width.

Paper-safe interpretation:

> Risk helps locate the population norm, but it is not the principal online signal for sharply locating an individual within the human conditional distribution.

Primary evidence:

- `reports/studies/RQ001_online_ipv_interval/RQ001_3_online_interval_lock_20260619/README.md`
- `.../01_results/baseline_metrics.csv`
- `.../01_results/final_summary.md`

### 2.2 The causal rolling-IPV self-anchor is the strongest evaluated interval signal

On the locked balanced lane-referenced slice (5,000 cases / 10,000 agent rows), the primary report records:

| Evaluation | Method | Coverage | Mean width |
|---|---|---:|---:|
| TEST | oracle PET | 0.889 | 0.867 |
| TEST | no-roll causal kinematics | 0.896 | 0.738 |
| TEST | causal rolling-IPV | **0.899** | **0.591** |
| Leave-Waymo-Out | oracle PET | 0.860 | 0.840 |
| Leave-Waymo-Out | no-roll causal kinematics | 0.857 | 0.743 |
| Leave-Waymo-Out | causal rolling-IPV | **0.902** | **0.628** |

Within this protocol, causal rolling-IPV is materially sharper than both oracle PET and the self-anchor-free kinematic fallback. It is also the only compared method to attain the nominal 0.90 target in the Leave-Waymo-Out test.

This supports a comparative claim within the locked protocol, not a universal statement about arbitrary datasets, maps, routes, or source shifts.

Primary evidence:

- `.../01_results/metrics_balanced_lock.csv`
- `.../00_entry/index.html`
- `.../TRACEABILITY.md`

### 2.3 The map-lane reference is the key deployment assumption

The strict-prefix reconstruction result is compelling: an observed-prefix reference correlates weakly with offline IPV (`corr ~= 0.281`), whereas a static map-lane-centreline reference reaches approximately `corr = 0.993`, `MAE = 0.027`.

The appropriate conclusion is not that the estimator is universally causal in every sense. It is that the signal is non-anticipating and route-conditioned under the deployment assumption that the relevant lane or route reference is available at decision time.

Paper-safe wording should use:

- `strict-prefix`, `non-anticipating`, `online-admissible`, or `route-conditioned causal reconstruction`;
- not broad causal-effect language.

### 2.4 Split-conformal calibration is methodologically necessary

Raw conditional quantile intervals under-cover at roughly `0.86`; split-conformal/CQR calibration is needed to bring empirical coverage close to the nominal target. For a runtime verifier, calibrated uncertainty is more important than adding model complexity.

The deliberate use of standard quantile models is a strength:

- it isolates the contribution of the signal rather than model capacity;
- it keeps the online step auditable and low-cost;
- it supports finite-sample marginal coverage under the calibration assumptions;
- an independent QRF-leaf + conformal route reportedly reproduces the main sharpness/coverage pattern.

The paper should describe coverage guarantees only under the applicable exchangeability/calibration domain. It must not imply nominal coverage after arbitrary domain transfer.

### 2.5 The fallback path is useful but defines a coverage boundary

Approximately 74% of the locked cases have a usable lane/route reference. The remaining approximately 26% fall back to a self-anchor-free causal-kinematics CQR interval. The fallback is weaker but still sharper than the PET lookup in the reported comparison.

This supports a two-path runtime implementation:

1. lane-referenced self-anchor path when route support is valid;
2. causal-kinematic fallback or monitor-only mode otherwise.

It does not support saying that the preferred self-anchor path is available for every interaction.

## 3. Blocking issues for stronger claims

### 3.1 The current target construction has an early/full-window overlap issue

The existing task predicts a full-window IPV target from an early-window IPV anchor. Because the full-window target includes the anchor period, part of the apparent predictability can be mechanical. RQ002's E1 review identifies this as a blocking issue for a strict no-leakage claim.

Required fix:

- define the scoring target on a post-anchor, non-overlapping window (`t > anchor end`);
- rerun the locked comparison and conformal calibration;
- report both the original and corrected protocols rather than silently replacing the earlier result.

Until then, the map-lane reconstruction supports input-side non-anticipation, but not the strongest possible claim that the whole supervised evaluation is overlap-free.

### 3.2 Do not mix the locked interval table with the integrated verifier A/B numbers

Two valid but different protocols appear in the evidence base:

- locked balanced interval comparison: causal-roll TEST approximately `0.899 / 0.591`, Leave-Waymo-Out `0.902 / 0.628`;
- integrated verifier A/B: self-anchor TEST approximately `0.901 / 0.485` versus PET-bin approximately `0.900 / 0.833`, with a roughly 42% width reduction.

These should be labeled as separate experiments with separate denominators, feature/model paths, and evaluation roles. A headline such as `-42% width` must explicitly identify the A/B protocol and should not be presented as the locked production-table width.

### 3.3 Cross-source transfer is bounded

The strongest transfer result holds on a balanced, lane-referenced locked slice with Waymo held out. It is not an unconditional guarantee for arbitrary source shift. In the integrated verifier A/B, held-out-source coverage remains below nominal even though it improves over the PET-bin baseline.

Therefore:

- self-anchor can be called the strongest evaluated transfer signal in the locked comparison;
- the integrated system should remain a monitor/soft-cost/warning under uncalibrated source shift;
- hard constraints require target-domain recalibration and support checks.

### 3.4 Interval sharpness does not establish normative validity

RQ001 shows that self-anchor predicts a narrower human conditional interval. It does not establish that conditioning on the agent's own behavior is sufficient to define a group norm. RQ002 finds a norm-laundering risk and requires a situation floor plus abstention.

The paper and implementation must distinguish:

- **RQ001:** which signal gives a sharp calibrated interval;
- **RQ002:** when that interval is normatively valid and when the verifier must guard or abstain.

### 3.5 Paired-IPV and planner outputs should remain secondary interfaces

Pair-sum/pair-difference checks can identify jointly anomalous interactions, but current evidence positions them as a gate rather than a stronger primary estimator. Planner-facing warning, soft cost, fallback, and counterfactual injection results demonstrate interface actionability, not closed-loop benefit or safety improvement.

## 4. Claim classification

| Claim | Review decision |
|---|---|
| Oracle PET/risk-bin lookup provides little interval-sharpness gain | **Supported within the evaluated data/protocol** |
| Lane-referenced causal rolling-IPV is the strongest evaluated online interval signal | **Supported on the locked balanced lane-referenced slice** |
| Causal rolling-IPV is the only compared method reaching 0.90 Leave-Waymo-Out coverage | **Supported for the locked comparison** |
| Map-lane prefix reconstruction is online-admissible and route-conditioned | **Supported under lane/route availability** |
| Split-conformal calibration is required to correct raw under-coverage | **Supported** |
| The result transfers unconditionally across sources | **Rejected** |
| The integrated verifier is ready for hard-constraint deployment under source shift | **Rejected** |
| The current supervised task is fully free of target overlap | **Not yet supported; rebuild required** |
| A pure self-anchor interval is by itself a valid population social norm | **Outside RQ001 and unsafe under RQ002** |
| The 42% width result and the 0.591 locked width are the same protocol | **Rejected; keep protocols separate** |

## 5. Required improvements before freezing RQ001 paper claims

1. Rebuild the target on a non-overlapping post-anchor window and rerun calibration, TEST and Leave-Waymo-Out comparisons.
2. Publish a protocol crosswalk distinguishing locked interval metrics, integrated verifier A/B metrics, feature sets, splits and denominators.
3. State the lane/route support rate and fallback behavior wherever deployability is claimed.
4. Treat source guard, support detection and target-domain recalibration as part of the deployment contract.
5. Use two directional deviations (`D_comp`, `D_yield`) downstream rather than an ambiguously signed one-sided cost.
6. Import the RQ002 validity guard before describing the interval estimator as a complete social-compliance verifier.

## 6. Recommended decision handoff

```text
RQ001 may freeze the following bounded result: on the locked balanced lane-referenced InterHub slice, a strict-prefix map-lane causal rolling-IPV self-anchor followed by split-conformal calibration is the sharpest evaluated online IPV interval method and the only compared method to attain approximately 0.90 coverage in Leave-Waymo-Out evaluation. Risk/PET lookup is not the main interval-sharpness lever.

Do not freeze: unconditional cross-source robustness; hard-constraint deployment under shift; a universal lane-independent deployability claim; a strict whole-task no-leakage claim before the non-overlapping target rebuild; or the claim that self-anchor alone defines a valid population social norm.
```
