# Codex Review: RQ001 Online IPV Interval

Status: review-complete, not yet frozen in `decision.md`.

Review date: 2026-06-21.

## Scope

Reviewed study packages:

- `reports/studies/RQ001_online_ipv_interval/RQ001_1_current_ipv_distribution_20260618/`
- `reports/studies/RQ001_online_ipv_interval/RQ001_2_interval_query_20260618/`
- `reports/studies/RQ001_online_ipv_interval/RQ001_3_online_interval_lock_20260619/`

RQ001_3 is the canonical result. RQ001_1 and RQ001_2 are useful provenance, but
RQ001_2's PET/TTC interval-query path is superseded by the lane-referenced
causal rolling-IPV interval.

## Overall Verdict

RQ001 supports a paper-safe verifier-method claim: a route/lane-conditioned
causal rolling-IPV self-anchor plus split-conformal calibration is a much
sharper online IPV interval than the PET-bin envelope. The claim must remain
route-conditioned and calibration-scoped; it is not a global formal guarantee.

Paper-safe phrasing:

> For cases with usable lane/route support, a prefix-only causal rolling-IPV
> self-anchor with conformal calibration provides calibrated, substantially
> narrower human-reasonable IPV intervals than PET-bin envelope lookup. Cases
> without route support should fall back to a wider no-roll kinematic conformal
> interval or abstain from hard verifier action.

## Claims That Can Be Carried Forward

1. **PET/risk lookup is not the main lever.**
   RQ001_2 and RQ001_3 agree that even oracle PET conditioning barely improves
   the interval. In RQ001_3's full-data primary test, oracle PET envelope
   coverage/width is `0.900 / 0.833`, close to the global floor
   `0.903 / 0.857`.

2. **The strongest online signal is self rolling-IPV.**
   RQ001_3's A/B verifier table reports the OnlineIPVIntervalEstimator at
   `0.901` coverage and `0.485` width on primary test, versus PET-bin
   `0.900` coverage and `0.833` width, with similar false-flag rate.

3. **The deployable locked version remains effective when causalized.**
   In the balanced lane-referenced causal rebuild, causal-roll reaches
   `0.899` coverage and `0.591` mean width on TEST. In Leave-Waymo-Out it
   reaches `0.902` coverage and `0.628` width, outperforming no-roll and oracle
   PET baselines on the same lane-supported slice.

4. **No-lane fallback is required.**
   The method is route-conditioned. RQ001_3 states that about 26% of cases lack
   the needed lane/route support and should use a wider no-roll kinematic CQR
   fallback rather than fabricated rolling-IPV features.

5. **The output contract is usable for the verifier.**
   The interval can feed `[q05, q50, q95]` and a signed deviation score into the
   existing social-compliance monitor. It should expose `calibration_mode`,
   `source_health`, and fallback reason.

## Claims To Reject Or Defer

- **Reject:** "Better PET prediction will fix the verifier." The report shows
  PET-bin lookup has limited ceiling value for IPV intervals.
- **Reject:** "The interval is globally calibrated for all datasets and all
  runtime conditions." Calibration is lane/route/support dependent and should
  expose fallbacks.
- **Reject:** "The verifier is ready for hard constraints without shift
  handling." The A/B integration Leave-Waymo-Out row still shows
  OnlineIPVIntervalEstimator coverage `0.823`, although it improves over PET-bin
  `0.786`; target-domain recalibration is still needed before hard action.
- **Defer:** full production latency/cost claims. RQ001_3 includes cost probes,
  but the knowledge layer should freeze only the methodological result unless a
  deployment benchmark is added.

## Knowledge-Layer Action

Update `synthesis.md` and `decision.md` around RQ001_3 as the canonical method
evidence. Freeze the accepted claim only with the route/lane support boundary,
no-lane fallback, cross-source recalibration need, and "not formal guarantee"
language intact.
