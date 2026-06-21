# Phase 7 Interpretation Correction

Worker: `RQ003_phase7_interp_fix_001`

Run: `RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0`

## Settled Conclusion

The corrected scenario-fixed results show a suggestive, nonsignificant (p=0.30, CI includes 0), non-generalizing (LOSO delta Spearman +0.017), non-IPV-specific favorable direction; robustness is not established. No robust incremental predictive utility relative to the prespecified kinematic+safety baseline was demonstrated.

This is not evidence that the effect is exactly zero. The corrected primary analysis is underpowered and cannot exclude smaller or context-specific effects.

## Evidence Basis

- Corrected primary LOTO: N=53, delta Spearman +0.136833, p=0.30, 95% CI [-0.038781, +0.305797].
- LOSO generalization: delta Spearman +0.016732, MAE reduction -0.060000, delta CV-R2 -0.004579.
- Safe subsets: S1 and S2 equal the primary 53 cells, so agreement is vacuous; S3 has n=6 and null/reverse behavior.
- Negative controls matching or exceeding primary: future_leaky_full_window_ipv +0.231817, ipv_time_shuffle +0.196823, counterpart_swap +0.168441, role_flip +0.136833, sign_flip +0.136833.
- Degradation controls failed: state_shuffle and wrong_state did not degrade as expected; state_shuffle improved baseline Spearman from -0.190937 to +0.105547.
- Independent replication2 closed the corrected-path reproduction blocker: `replication2_status.json` reports corrected N=53, exact reported-alpha refit direction +0.136833, and independently tuned favorable direction +0.054669.

## RT2 Blocker Resolution

- RT2-BLOCK-001: closed by explicitly stating that controls match or exceed the primary direction and that state_shuffle/wrong_state degradation failed. The negative-control and confirmatory reports now conclude no IPV-specific signal demonstrated.
- RT2-BLOCK-002: closed by citing `02_process/17_independent_replication/replication2/replication2_status.json`, which reproduced corrected N=53 and the favorable direction without importing Phase 4/Phase 7 modeling scripts.
- RT2-BLOCK-003: closed by stating S1=S2=primary 53 cells, S3 n=6 null/reverse, and the safe_subset_agreement_count=2 is vacuous. Safe-subset robustness support was removed.
- RT2-BLOCK-004: closed by downgrading wording to suggestive/nonsignificant/non-generalizing/non-IPV-specific and by using the plan-approved conclusion sentence.

## Scope Guardrails

No numeric result table was recomputed or edited. No freeze, feature, optimizer, tier decision, or plotting artifact was changed.
