# RQ009: Dynamic Counterpart-Conditioned Human Envelope (M3)

Status: APPROVED / greenlit by PI 2026-06-24 (was planning). Execution not yet run; no `decision.md` until results exist.

Knowledge layer for the primary dynamic verifier model (M3): the estimability-aware, conformally calibrated dynamic envelope conditioned on the current state and the counterpart's current IPV.

## Frozen upstream inputs (all satisfied)

- **RQ007 (estimability contract, frozen):** separate interaction opportunity / IPV estimability / human-reference support / deviation; never read high uncertainty as neutral IPV. Use the valid-window/estimability contract.
- **RQ008 (frozen, negative):** no directional temporal IPV law survived discovery → RQ009 must NOT rely on temporal motifs; use context + counterpart current IPV (and, at most, estimable-window structure marked pending).
- **RQ004 (frozen):** R1 state-conditioned response surface (risk × geometry × role × time) is the contextual norm.
- **RQ002 (frozen):** self-anchor → M4 ablation only; norm is the human population conditional distribution, not agent-owned.
- **RQ005 (frozen):** leakage contract — observed PET / post-hoc labels offline-only; causal provenance for runtime inputs.

## Scope (M0–M5)

M3 = context + counterpart current IPV + split-conformal (primary). M4 = context + ego self-anchor (sharpness ablation only). Plus M0 global scalar, M1 oracle PET-bin, M2 context-only, M5 source-aware/OOD-gated. Target = same-window current rolling IPV; support/OOD → abstain.

## Next action

Draft the RQ009 execution plan and run M0–M5 under the frozen contracts; it gates RQ010B (M3 preference test), RQ011B (matched-scenario), and RQ013. This README records the greenlight; it is not a results decision.
