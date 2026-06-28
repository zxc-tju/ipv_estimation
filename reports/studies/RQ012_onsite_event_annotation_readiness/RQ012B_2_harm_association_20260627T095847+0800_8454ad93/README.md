# RQ012B Run 2 — deviation→harm (scientific endpoint) · COMPLETE

RUN_ID: `RQ012B_2_harm_association_20260627T095847+0800_8454ad93` · 2026-06-27/28
RQ: RQ012B (OnSite automatic-event harm; manuscript R5 "consequence" endpoint)
Orchestration: Claude = controller; Codex CLI fleet + Tongji HPC = execution.

## Result (definitive, full behavioural battery)
**BOUNDED/NULL.** An OnSite AV's deviation from the frozen human IPV envelope (RQ009 M3) does **not robustly, IPV-specifically predict realised interaction-failure harm**. Across the full battery (9 automatic events + behavioural groupings + 4 official subscores; kinematic baseline; cluster-aware permutation; label/placebo/M2/exposure controls; BH-FDR over 64 tests):
- **Direction is uniform but the link is weak**: more deviation → slightly more of every failure type and slightly worse every official score (all toward "worse"; partial r ≲ 0.17).
- **Clearest channels** (still weak): near-miss/contact, deadlock (E16), efficiency, total friction. **Weakest/≈null**: hard-braking (E02, 急刹), comfort, jerk.
- **Too-passive > too-aggressive**: over-passivity lines up with problems (deadlock, near-miss, safety, efficiency) more than aggression; aggression mainly with contact.
- **Nothing robustly SUPPORTED**: near-miss is context-explained (fails the context-only M2 control → not M3-specific); only too-passive→deadlock (E16) survives all controls but is UNDERPOWERED (52/243 units) + at the BH q≈0.05 edge → a bounded, UNCONFIRMED hypothesis.
- **Bounds manuscript R5** "behavioural-prior-mismatch → harm" on OnSite via the M3-deviation operationalization; flags **passivity→deadlock** for a future powered test.

## Start here
- Bilingual report: `00_entry/index.html` → `90_report/index.{html,zh.html}` (being refreshed to this full-battery headline + Fig 4)
- PI synthesis (authoritative): `02_process/09_report/CONCLUSIONS.md` (HA-1..HA-4)
- Pre-registration (+ amendment): `02_process/04_harm_association/PRE_REGISTRATION.md`
- Figures: `01_results/figures/` — fig1 OOD funnel, fig2 deviation signal, fig3 safety-pass null, **fig4 intuitive deviation↔failures matrix**
- Full-battery results: `02_process/04_harm_association/{harm_association_full_battery_report.md,results_full_battery.json}`; matrix `data/derived/onsite_competition/RQ012B_event_harm/stage4b/full_battery/deviation_consequence_matrix.csv`

## Pipeline (validated)
M3 scorer reconstructed from RQ009 frozen code (parity 0.0) → pinned IPV estimator fetched from Tongji HPC (git 5edd2810, sha byte-identical) → OnSite IPV + 67,861 anchors / 267 units on HPC (ProcessPool, 42 min) → OOD gate (28% in-support; 245/267 units; 840 out-of-band moments / 149 units) → pre-registered association + full behavioural battery + negative controls.

## Validation
design/pre-reg(+amendment) → association(safety pass + full battery) → red-team POWER-LIMITED-NULL → blind replication AGREE → independent full-battery recheck → final-review PASS. Under `02_process/`.

## Provenance
M3 parity 0.0 · estimator HPC git `5edd2810` (sha 30b9fd…/8c5c63…/0a0860…) · anchors 67,861/267 · analysis n=245/~19 teams · seed 20260628 · venv sklearn 1.6.1.

## Registration
knowledge `decision.md` (RQ012-KC-HARM-NULL + full-battery refinement) + `report_index.md` current; program index (registry/dashboard/STUDIES) re-synced 2026-06-28 to `accepted / RQ012B COMPLETE / bounded-null` (a prior origin/main merge had reverted them). Feeds RQ013 as a bounded/null prior.
