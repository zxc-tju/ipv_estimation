# Claude Code Review

Status: filed (2026-06-21)

Reviewer role: research reviewer / repository integrator.

Run reviewed: `RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0`

Reader entry:
`reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/00_entry/index.html`

## Verdict

Concur with the Tier B registrar conclusion. RQ003_6 is a power-limited
**boundary / null result**, not a verifier-validation success. No robust
incremental predictive utility for the directional IPV signal over the
prespecified kinematic+safety baseline was demonstrated in the approved NSFC
top-five cohort (N=53 cells, 9 teams, 14 scenarios). The accepted-claim row and
rejected/deferred table now in `decision.md` are well supported by the run
artifacts.

## Key Findings

| Analysis | Result | Reading |
|---|---|---|
| Primary LOTO (leave-one-team-out) | delta Spearman = +0.137, p = 0.30, CI [-0.039, 0.306] | Favorable direction only; nonsignificant, CI crosses 0. |
| Secondary LOSO (leave-one-scenario-out) | delta Spearman approximately +0.017; MAE reduction negative; CV-R^2 delta negative | Does not generalize across scenarios. |
| Sensitivity fallback (N=73) | delta Spearman = -0.037, p = 0.84 | Reverses; not confirmatory. |

**Decisive point - the apparent gain is not IPV-specific.** In
`negative_controls.csv`, controls that should show no gain reproduce or exceed
the primary delta: `role_flip` = +0.137 and `sign_flip` = +0.137 (identical to
primary), `counterpart_swap` = +0.168 (larger than primary). This indicates the
unconstrained ridge model absorbs added columns to manufacture a comparable
"improvement" regardless of whether the column carries real directional IPV
information. The primary +0.137 is therefore a model-capacity artifact, not
evidence that directional IPV encodes coordination-relevant signal.

## Boundaries And Blocked Items (confirmed)

- H3 blind two-human social-compliance annotation is **blocked**: no real
  two-human labels exist in this package.
- NPC material is boundary-only and non-identifiable under the available
  matching fields; it supports matched opportunity-structure wording only.
- The full 20-team NSFC universe is **not analysis-ready**; it must not back
  confirmatory claims from this run.
- The coordination outcome is the official/generated report score, not a direct
  human social-compliance label.
- Safe-subset agreement, LOSO, negative controls, and state-dependence must not
  be promoted as robustness support (S1/S2 duplicate the primary 53 cells; S3
  has only 6 cells and is null/reverse).

## Reproducibility / Process Assessment

- Identity gates verified by the Phase 12 registrar: run-manifest ID,
  `plan_sha256`, `tier_decision.json` = B, and `final_review_status.json` = PASS
  all matched. Final review status: PASS.
- Negative-control battery is appropriately broad (state_shuffle,
  ipv_time_shuffle, counterpart_swap, role_flip, sign_flip, wrong_envelope_cell,
  kinematics_only, ipv_removed, shuffled_ipv, future_leaky) and is what makes the
  non-specificity finding credible.
- `before_scenario_fix` companion tables are retained alongside corrected
  tables, so the scenario-crosswalk red-team fix is auditable.
- Minor caveat: the registrar noted a `GIT_HEAD` mismatch between the task brief
  (`c23074a...`) and the observed repo HEAD (`394bb61...`); it was judged
  non-blocking because it was not an identity gate and no commit/push occurred.
  Flagging for provenance completeness; it does not change the conclusion.

## Supporting Role For The NMI Manuscript

RQ003_6 supports the paper primarily as a **guardrail and a protocol**, not as a
positive validation result:

1. It justifies keeping the *External validation on an independent real-vehicle
   challenge* section (main.tex ~L199-209) inside `\planned{}` and keeping the
   rejected claim "NSFC formally validates the IPV verifier" rejected. This
   protects the submission from over-claiming.
2. It yields paper-safe boundary wording (see `decision.md` Paper Handoff): an
   honest, power-limited boundary result with negative controls is itself an
   acceptable manuscript element.
3. It stands up the full external-validation harness (provenance/coverage,
   missingness audit, LOTO/LOSO/LOFO, negative controls, state-dependence,
   OOD/abstention), giving a ready protocol for the planned analysis and a clear
   list of what is needed to convert `\planned{}` into a real result: higher
   power (full 20-team universe, once analysis-ready), real two-human
   social-compliance labels (to unblock H3), and identifiable NPC matching
   fields.

## Recommendation

Freeze RQ003_6 at the Tier B boundary as recorded in `decision.md`. Do not use
RQ003_6 to assert criterion validity, consequence-chain prediction, or
discriminant/incremental value as established findings. Re-open for a Tier A
attempt only when the 20-team universe is analysis-ready and/or real two-human
labels become available.

## Source Pointers

- `reports/studies/.../RQ003_6_..._fbd2d3f0/01_results/tables/confirmatory_results.csv`
- `reports/studies/.../RQ003_6_..._fbd2d3f0/01_results/tables/confirmatory_results_interpretation.csv`
- `reports/studies/.../RQ003_6_..._fbd2d3f0/01_results/tables/negative_controls.csv`
- `reports/studies/.../RQ003_6_..._fbd2d3f0/registrar_report.md`
- `reports/studies/.../RQ003_6_..._fbd2d3f0/proposed_knowledge_update.md`
