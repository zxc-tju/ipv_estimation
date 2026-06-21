# Phase 4 Directional IPV Feature Report

Worker: `RQ003_phase4_features_001`

Run: `RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0`

Status: PARTIAL

## Scope

- Processed approved top-five cohort cells: 150 / 150.
- Real optimizer path: `sociality_estimation.core.ipv_estimation.estimate_ipv_current` with SciPy SLSQP via solver preset `balanced`.
- Candidate grid: `legacy7`.
- Conflict window: current-frame AV-NPC distance <= 41.3104272038 m.
- Human conditional norm: rebuilt from InterHub calibration split only, using current-frame `theta_npc_bin`, `state_condition`, and `tau_bin`.

## Key Results

- High-support primary cells: 58.
- Fallback-inclusive cells: 79.
- Fallback `D_comp` AUC range: 0 to 0.128626.
- Fallback `D_yield` AUC range: 0 to 0.132927.

## Firewall

No official coordination, efficiency, comprehensive score values, ranks, residuals, or predictor-outcome result tables were used to compute these features.

Spec deviation: during exploratory discovery before this script was written, a broad search touched score-field README/header text but exposed no outcome values. This is recorded in the worker manifests, so this run is marked `PARTIAL` rather than a clean `PASS`.

## Online Construction

- Counterpart selection uses the first current-frame conflict opportunity in the replay, preferring scripted `从车` vehicles when present; it does not use score outcomes.
- Per-frame IPV uses only rolling-prefix states up to the current frame.
- Rolling references are built from past/current positions plus current-velocity extrapolation, not from full-window future paths.
- Primary scalar columns use high-support frames only; fallback columns are clearly suffixed `_fallback`.
