# ChatGPT Review — RQ002 Self-Anchor as Group Norm

- Reviewer: ChatGPT (GPT-5.5 Pro)
- Review date: 2026-06-21
- Executions reviewed:
  - `RQ002_1_self_anchor_validation_main_20260619`
  - `RQ002_2_self_anchor_validation_codex_20260619`
- Review status: **MUST REVISE — REJECT SELF-ANCHOR-ONLY NORMATIVE CLAIM**
- Recommended method direction: **retain self-anchor for interval sharpness, but require a situational population floor, calibrated abstention/support gates, and a non-overlapping target before treating the system as a social-compliance verifier**

## 1. Overall verdict

The two independent RQ002 validation packages converge on the same high-level conclusion: a self-anchored interval is informative, but self-anchor alone is not a defensible population group norm.

The central failure mode is norm laundering. If the expected interval is conditioned too strongly on the agent's own early behavior, a persistently aggressive but internally self-consistent agent can shift its own reference and appear normal. The validation does not justify discarding self-anchor: situation-only features are too weak and produce wider intervals. The supported design direction is therefore hybrid rather than substitutive:

```text
self-anchor signal for sharpness
+ split-conformal calibration
+ situational population floor
+ moderate-shift / out-of-support abstention
```

This is a required validity envelope around the same estimator, not evidence for a second competing verifier.

The hybrid architecture is an evidence-driven design recommendation. It should not yet be described as fully validated end-to-end until the floor and abstention thresholds are calibrated, integrated, and re-evaluated on a non-overlapping target.

## 2. What the validation establishes

### 2.1 Pure self-anchor cannot be equated with a population norm

The reports explicitly test whether self-anchored conformal intervals behave as a human conditional norm or merely as a self-consistency band. Both packages return `MUST_REVISE`, and the red-team review blocks a strong `self-anchor alone is a group norm` narrative.

The paper should define the normative object as:

> the human population conditional distribution given admissible early behavior and context,

not as an interval owned or redefined by the monitored agent.

Primary evidence:

- `reports/studies/RQ002_self_anchor_group_norm/RQ002_1_self_anchor_validation_main_20260619/00_entry/index.html`
- `.../02_process/board/FINAL_REPORT.md`
- `.../02_process/board/validation.md`
- `reports/studies/RQ002_self_anchor_group_norm/RQ002_2_self_anchor_validation_codex_20260619/00_entry/index.html`
- corresponding `findings.md` and `validation.md`

### 2.2 E1 exposes a target-overlap problem

The locked full-window IPV label overlaps the early approximately 2 s anchor. One independent run reports a very high early/full overlap statistic (`mean ~= 0.952`, `pct_one ~= 0.789`). This does not prove future-input leakage, but it means the supervised target partly contains the signal used to predict it.

Consequences:

- the existing reconstruction evidence can support input-side non-anticipation;
- it cannot by itself support the strongest whole-task `strict no-leakage` claim;
- the target should be rebuilt over a post-anchor, non-overlapping scoring window.

Required protocol:

```text
anchor = causal early window available at decision time
target = IPV over a later window beginning after anchor end
```

Original and corrected results should both be retained for traceability.

### 2.3 E2 provides useful transfer evidence but not true held-out-driver validation

The available data lack persistent cross-scenario driver identities. Consequently, a true held-out-individual test is not fully assessable. Leave-Waymo-Out and deterministic scenario/case/family substitutes show favorable coverage near 0.90, but they are not equivalent to unseen-driver validation.

Paper-safe interpretation:

- supported: source/case/family transfer substitutes are favorable within the evaluated slice;
- not supported: demonstrated generalization to previously unseen persistent individuals.

### 2.4 E3 shows that the laundering risk is not confined to extreme shifts

The main HGB/CQR stress route flags large synthetic shifts strongly, and an explicit out-of-support guard increases flag-or-abstain behavior. This is favorable. However, the independent empirical-bin replication identifies a residual washout region at moderate shifts, summarized in the evidence index as approximately `Delta = 0.4-0.6`.

This matters because a guard that catches only extreme out-of-support behavior is insufficient. Abstention should also trigger when the self-anchored conditional center materially diverges from the situation-only population center, even if the feature vector remains nominally inside marginal training support.

Recommended gate:

```text
abstain if |q50_self - s50_situation| > tau_abs
or if features are outside calibrated joint support
```

The threshold must be calibrated without outcome leakage and checked across the moderate-shift region.

### 2.5 E4 shows why self-anchor must be retained

Situation-only context is not an adequate replacement for self-anchor. The indexed results report:

- situation-only R² for early IPV: approximately `0.044`;
- incremental R² from disposition residual: approximately `0.45`;
- situation-only/self-anchor width ratio: approximately `1.34`.

These results indicate that stable individual disposition is a large and real component of the predictive signal. Replacing self-anchor with a situation-only norm would sacrifice substantial sharpness and misrepresent the uncertainty structure.

Therefore the valid conclusion is:

> self-anchor is necessary for individual localization, but insufficient for normative authority.

### 2.6 E5 provides the clearest evidence for a situational floor

In the high-risk segment (`PET <= 1` used offline for adjudication), the indexed results report:

- self-anchor flag lift: approximately `0.850`;
- situation-only flag lift: approximately `1.129`;
- cases passed by self-anchor but flagged by the situational method are enriched approximately `1.507x` for bad-outcome proxies.

This is the strongest direct warning that self-anchor can be too permissive where population context expects prosocial behavior. A consistently competitive agent should not be able to lower the permissive edge of its own acceptable interval indefinitely.

The recommended correction is a one-sided situational floor on the lenient/competitive edge:

```text
q05_guarded = max(q05_self, s05_situation - tau_flr)
```

The floor should use only online-admissible context features such as geometry, role, and a causal kinematic risk proxy. Observed PET must remain offline-only.

## 3. Required guarded architecture

### 3.1 Signal: causal early-window self-anchor

Use strict trajectory history plus a static lane/route reference to estimate the monitored agent's early rolling-IPV. This component locates the agent within the human distribution and provides interval sharpness.

### 3.2 Calibration: conditional quantiles plus split-conformal

Fit conditional quantiles using the self-anchor and admissible context, then calibrate the interval on a held-out calibration set. Keep the machinery standard and auditable; the methodological contribution is the signal and its validity envelope, not a bespoke high-capacity predictor.

### 3.3 Guard: situation floor

Fit a separate situation-only population reference:

```text
S(x_sit) = [s05, s50, s95]
```

where `x_sit` contains online-admissible context and excludes observed PET, full-window outcomes, and realized passing order.

Apply a one-sided floor to prevent an agent's self-anchor from making the competitive edge more permissive than the population context allows:

```text
q05_guarded = max(q05_self, s05 - tau_flr)
```

This floor is a policy/validity guard. It should not be misrepresented as proof that the situation-only model is the primary estimator.

### 3.4 Guard: moderate-discrepancy and support-aware abstention

Return `Abstain` or `Monitor` and revert to the situation-only interval when:

```text
|q50_self - s50| > tau_abs
or joint-support / provenance / source-health gates fail
```

Support checks should be multivariate or otherwise calibrated to the actual estimator domain; marginal min/max checks alone can miss internal low-density regions.

### 3.5 Separate empirical verifier from safety-policy guard

The implementation and manuscript should expose two logically distinct outputs:

1. empirical human-norm nonconformity;
2. safety/policy guard intervention under high-risk or low-support conditions.

External outcome analysis must not apply the guard first and then use guard-induced flags as evidence that the empirical human norm is valid. This separation is especially important for RQ003.

### 3.6 Use directional two-tail deviations

With `theta > 0` defined as prosocial, the two interval tails have different meanings:

```text
D_comp  = max(0, (Q_low  - theta) / w)   # competitive shortfall
D_yield = max(0, (theta - Q_high) / w)   # over-yielding excess
```

A single signed deviation combined with a one-sided `more competitive` trigger is vulnerable to sign inversion. Planner actions should preserve the distinction between competitive intrusion and excessive yielding/freezing.

## 4. Blocking issues before claiming a validated guarded verifier

### 4.1 The proposed hybrid has not yet been fully evaluated as an integrated system

The validation motivates the floor and abstention, but the reports reviewed here do not by themselves demonstrate that the final guarded implementation simultaneously achieves:

- nominal in-domain coverage;
- retained sharpness;
- improved high-risk bad-outcome flag lift;
- acceptable abstention rate;
- robustness across source shift;
- no outcome-tuned thresholds.

These must be measured after integration. Until then, describe the hybrid as the required revision/design supported by validation, not as a completed validation success.

### 4.2 Guard thresholds require frozen calibration

`tau_flr` and `tau_abs` must be selected on a calibration set with explicit objectives and without test/outcome tuning. The calibration should report trade-offs among coverage, width, bad-outcome enrichment, abstention, and false flags. Moderate-shift stress tests must be part of the acceptance criteria.

### 4.3 Persistent-driver generalization remains unresolved

Without stable driver identities, cross-individual normative coverage is not established. Future data with persistent identities should support a grouped held-out-driver evaluation that prevents the same driver's behavior from entering both fit/calibration and test partitions.

### 4.4 The situation reference must remain strictly online-admissible

The floor cannot use observed PET merely because PET is available offline for analysis. It should use a causal online risk proxy, geometry, role, and other information available at the decision time. Otherwise the method would reintroduce the risk-estimation dependency that RQ001 successfully removed from the primary interval path.

## 5. Claim classification

| Claim | Review decision |
|---|---|
| Self-anchor contains substantial real information about individual IPV disposition | **Supported** |
| Situation-only context can replace self-anchor without material loss | **Rejected** |
| Self-anchor alone is a valid population group norm | **Rejected / blocked** |
| Pure self-anchor is immune to norm laundering | **Rejected** |
| A situation floor is required on the permissive competitive edge | **Supported as an evidence-driven design revision** |
| Out-of-support abstention is required | **Supported** |
| Abstention should also cover moderate self-vs-situation disagreement | **Supported by the residual washout finding; threshold still to be calibrated** |
| The current full-window target is strictly non-overlapping with the anchor | **Rejected** |
| True held-out-driver coverage has been demonstrated | **Not assessable with current identifiers** |
| The final guarded verifier has already been validated end-to-end | **Not yet supported** |
| Observed PET may be used in the deployed situation floor | **Rejected** |
| `D_comp` and `D_yield` should replace an ambiguous one-sided signed trigger | **Supported** |

## 6. Required follow-up evaluation

1. Rebuild the target on a post-anchor non-overlapping window.
2. Implement the situational floor and discrepancy/support abstention in the same locked pipeline.
3. Freeze `tau_flr` and `tau_abs` using calibration-only objectives.
4. Re-evaluate coverage, width, Winkler score, false flags, high-risk flag lift, disagreement enrichment, and abstention rate.
5. Repeat the E3 moderate-shift sweep after integration.
6. Report subgroup behavior by source, geometry, role, lane availability, and support status.
7. Add grouped held-out-driver validation when persistent identities become available.
8. Preserve separate empirical-norm and policy-guard outputs in code, reports, and manuscript language.

## 7. Recommended decision handoff

```text
RQ002 should reject the strong claim that a self-anchored interval alone constitutes a population social norm. The two independent validation packages support a MUST_REVISE decision.

Retain self-anchor because it carries substantial individual-disposition information and produces materially sharper intervals. Require a guarded design consisting of self-anchor + split-conformal calibration + a one-sided situation floor + moderate-discrepancy/out-of-support abstention. Rebuild the evaluation target on a non-overlapping post-anchor window.

The guarded architecture is the evidence-supported revision direction, but should not be labeled fully validated until the integrated system is recalibrated and re-evaluated under frozen thresholds.
```
