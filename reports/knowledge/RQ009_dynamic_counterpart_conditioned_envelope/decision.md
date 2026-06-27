# RQ009 Decision: Dynamic Counterpart-Conditioned Human Envelope

Status: COMPLETE - knowledge-layer freeze, 2026-06-27.

Run ID: `RQ009_1_dynamic_envelope_20260625T121905Z_98c433de`
Git head: `4aef4d22bb639bf003c48094607c970e55445d5f`
Plan SHA-256: `b8c027a717af08cd70de6ef2b5221b387323f1131623f39fa5fd688f89baa254`

Basis for freeze: Phase 12a final independent review `PASS` with `ready_to_register=true`, after Phase 11b fixed the reader report. Process support gates include provenance PASS, calibration PASS, M3-vs-M4 no escalation, red-team PASS, replication reconciled, and final report gate PASS.

Reader-facing report:

- Entry page: `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/00_entry/index.html`
- English report: `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/90_report/index.html`
- Chinese report: `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/90_report/index.zh.html`
- Figure manifest: `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/01_results/figures/figure_manifest.csv`
- Evidence table: `reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/evidence.csv`

## Accepted Claims

| ID | Claim | Strength | Evidence pointer |
|---|---|---|---|
| RQ009-KC-C1 | Online IPV envelope is marginally valid under split conformal quantile calibration for gate-passing held-out test anchors. | MODERATE, with limitations | `evidence.csv` C1/P4, Fig. `c1_validity_envelope`: M3 coverage 80/90/95 = `0.816215 / 0.898666 / 0.949635`, within the +/-3 pp gate; M3 90% mean width `1.016152`, Winkler `1.422935`, `n=1209857`, cases `7550`. |
| RQ009-KC-C2 | Counterpart-conditioned IPV adds no practically meaningful, generalizing information about ego future IPV beyond context in this operationalization. | STRONG practical null | `evidence.csv` C2/C3/C4/C7/P9/P10, Figs. `c2_context_dominates`, `c3_counterpart_ipv_null`, `c4_horizon_reconciliation`, `c7_robustness_perturbation`: M3 is effectively M2/`ipv_removed`; 90% M3-minus-`ipv_removed` Winkler `-0.000211426` (`-0.014856%`), case-cluster CI `[-0.001886180, 0.001504979]`, case sign p `0.862943`; all observed effects are below the 5% meaningful-effect bar. |
| RQ009-KC-C3 | Context and kinematics dominate interval sharpness; ego self-anchor adds only a marginal increment. | MODERATE to STRONG descriptive decomposition | `evidence.csv` C2/C5, Figs. `c2_context_dominates`, `c5_self_anchor_marginal`: at 90%, M2 vs M0 mean width `-42.271248%` and Winkler `-35.611843%`; M4 vs M2 Winkler `-2.722934%`; M3 vs M2 Winkler `-0.014856%`. |
| RQ009-KC-C4 | M3-vs-M4 does not require escalation, but M3 is not better than M4. | GOVERNANCE PASS, scientific non-win | `evidence.csv` C5 and Phase 6 verdict: M3 vs M4 mean width/Winkler/coverage delta at 80% = `-4.429% / +0.505% / -0.036 pp`, at 90% = `+2.960% / +2.784% / +0.142 pp`, at 95% = `+6.599% / +4.501% / -0.396 pp`; no signed threshold arm crosses the PI escalation limits. |

## Counterpart-IPV Null Scope

The accepted null is a practical performance/adaptation null, not a literal independence claim and not a claim that social interaction has no effect.

Supporting details:

- M3, M2, and `ipv_removed` are effectively indistinguishable at the 90% interval endpoint: M3 vs `ipv_removed` width `+0.661%`, Winkler `-0.014856%`, and coverage `-0.022 pp`.
- The paired case-level result is null: case signs `3767` better / `3783` worse / `0` ties, sign p `0.862943`; case Wilcoxon p `0.522202` in the replication reconciliation.
- Across 15 registered guard-tune counterpart encodings, no point or interval screen was both BH-significant and practically meaningful; best point dR2 was `C04=-0.000912`, and best interval Winkler reduction was `C03=-1.1902%`, below the 5% bar.
- Longer-horizon checks over the registered 0.5-2.1 s class did not rescue the mechanism; the reported h=6..21 sweep spans about 0.6-2.1 s, with best point dR2 `0.001715` and best interval Winkler reduction `-0.2405%`.
- Dependency reconciliation shows the earlier approximately `-0.11` horizon signal was a residualization/estimand artifact. Canonical held-out row-level partial r is small, about `-0.0393..-0.0345`, and held-out case-level r is small positive/nonmonotone, `0.0226..0.0575`.
- Red team found no leakage or null-effect exploit. Replication divergence was reconciled: the frozen 90% effect remains `-0.014856%`, and the clean-room `-0.333852%` route remains far below the 5% meaningful bar.

## Not Supported

The primary hypothesis that counterpart conditioning sharpens or shifts the online IPV envelope in a practically meaningful way is NOT SUPPORTED. The data support a marginally valid dynamic envelope and a strong bounded null for the incremental counterpart-IPV channel beyond context/kinematics, not a positive counterpart-conditioning mechanism.

Also not supported:

- any causal claim that counterpart IPV changes ego future IPV;
- any literal-independence claim across all possible social encodings or outcomes;
- any subgroup-conditional or source-transfer guarantee from the marginal conformal result;
- any planner-performance, external-validity, or normative-behaviour claim.

## Limitations

These limitations must travel with any paper use:

- Conditional subgroup coverage is uneven: `126/264` supported subgroup rows fall outside the +/-3 pp band (`evidence.csv` C6, Fig. `c6_limitations`).
- Leave-one-dataset-out transfer is unstable: M3 90% coverage ranges `0.748790..0.991481` across held-out datasets (`evidence.csv` C6).
- The scored target has material atom/boundary mass: exact-zero atom `273819/1270566 = 0.215509`, which qualifies interval ties and creates 80% endpoint/nudge fragility (`evidence.csv` C6_ATOM; red-team RT-04).
- Results are associational and prediction-oriented, not causal.
- Results use one IPV operationalization, one InterHub-derived feature/target contract, and gate-passing rows after the support/OOD abstention rule.
- Future work may test other outcomes such as yield timing, gap acceptance, trajectory geometry, longer horizons, or lower-noise sociality labels; those would be new questions, not rescues of this frozen result.

## Evidence Index

| Evidence row | Figure/report | Metric/value | Status |
|---|---|---|---|
| C1 | `01_results/figures/c1_validity_envelope.png` | M3 coverage 80/90/95 = `0.816215;0.898666;0.949635`; M3 90% width `1.016152`, Winkler `1.422935` | supported |
| C2 | `01_results/figures/c2_context_dominates.png` | M2 vs M0 90% width/Winkler = `-42.271248% / -35.611843%`; M3 vs M2 Winkler `-0.014856%` | supported |
| C3 | `01_results/figures/c3_counterpart_ipv_null.png` | M3-minus-`ipv_removed` 90% Winkler `-0.000211426`; CI `[-0.001886180, 0.001504979]`; case sign p `0.862943` | supported |
| C4 | `01_results/figures/c4_horizon_reconciliation.png` | canonical held-out row partial r `-0.0393..-0.0345`; case-level r `0.0226..0.0575`; best dR2 below `0.02` | supported_with_scope |
| C5 | `01_results/figures/c5_self_anchor_marginal.png` | M4 vs M2 90% Winkler `-2.722934%`; M3 vs M4 90% Winkler `+2.783881%` | supported |
| C6/C6_ATOM | `01_results/figures/c6_limitations.png`; `90_report/index.html` | subgroup outside +/-3 pp `126/264`; LODO M3 90 range `0.748790-0.991481`; target zero atom `273819/1270566=0.215509` | limitation |
| C7 | `01_results/figures/c7_robustness_perturbation.png` | `validity_robust=true`; `null_robust=true`; perturbation gain range `-0.541%..1.193%` | supported |
| P2/P3/P4/P8/P9/P10 | process gates | provenance PASS; feature/matrix audit PASS; calibration PASS; independent review PASS; red team PASS; replication reconciled | supported |

## Paper Handoff

Paper-safe wording:

> In InterHub, the conformal dynamic IPV envelope is marginally calibrated on held-out gate-passing anchors, but the counterpart-IPV channel does not add a practically meaningful, generalizing increment beyond context/kinematics. Context dominates sharpness, ego self-history provides only a small ablation gain, and the primary counterpart-conditioning hypothesis is not supported.

Use only with the limitations above. Do not write this as causal evidence, source-conditional validity, or proof that social adaptation is absent.
