# Claude Code Review

Status: filed (2026-06-24)

Reviewer role: research reviewer / repository integrator.

Run reviewed: `RQ011_2_onsite_readiness_20260623T201415+0800_efdd75a5` (supersedes `RQ011_1_...20aaee57`; `overall_status: complete`).
Reader entry: `reports/studies/RQ011_onsite_full_universe_readiness/RQ011_2_onsite_readiness_20260623T201415+0800_efdd75a5/90_report/index.html`.

## Verdict

Concur with `READY_WITH_FROZEN_EXCLUSIONS` — as a **readiness/inventory** decision, not an analysis result.
The matched-scenario primary unit (`algorithm×scenario`) is well justified, and the separation of an
**outcome universe** (`full_300`: official scores/collisions/deductions on all 300 algorithm/team×scenario
cells) from a **replay/IPV universe** (`clean_285`, with `T19` frozen out) is the correct, disciplined
structure. The single frozen exclusion (T19, no unique vehicle-3190 replay/session) is reasonable and is
declared replay-only.

## Key Findings

| Item | Result | Reading |
|---|---|---|
| Primary unit | `algorithm×scenario` (matched scenario) | Identifiable; preserves algorithm identity over 15 scenario labels. |
| Outcome universe | full_300 | Official outcome fields complete; no outcome-side exclusion. |
| Replay/IPV universe | clean_285 (T19 excluded) | Replay/trajectory/IPV work limited to corrected-clean cells. |
| Run-level / repeated runs | NOT identifiable / false | No run-level, seed-level, or repeated-run claims possible. |
| Residual replay bias | T19 exclusion drops replay collision rate to ~8.4% vs 11.0% full | Replay/IPV findings are positively shifted; not full-300 coverage. |
| Identity collapses | algorithm_id == team_id; case_id == scenario_id | Limits separability; team kept as metadata/effect. |

## Boundaries And Watch-Items

- **Matched-scenario is the ceiling.** Run-level replication and repeated runs are not identifiable, so RQ011
  cannot support run-level or repeated-run "realised consequence" claims; the downstream framing must stay at
  algorithm×scenario.
- **Moderate replay selection bias** must be stated wherever replay/IPV results appear: the 285-cell replay set
  is outcome-positively-shifted relative to the 300-cell outcome set, and area composition shifts slightly away
  from Beijing.
- **Governance flag (review carefully):** the final `READY_WITH_FROZEN_EXCLUSIONS` leaf was produced by an RT10
  fix on 2026-06-24 (`pi_authorized_correction_applied: true`) that re-interpreted `run_level_claims_allowed=false`
  from a terminal final-status block into a scope boundary. The re-interpretation is defensible, but it is a
  human-authorized re-grade of a red-team finding — confirm the PI authorization is recorded and that no leaf was
  renamed. Also `git_head` (`32ebf75...`) differs from the 38063a2 baseline of the other RQ runs.
- **Documentation gap:** `evidence.csv` is header-only (empty). The readiness checks live in process files
  (`analysis_unit_decision.md`, `execution_status.json`) but are not in the evidence ledger; populate it.
- Exclusions/weights must not be tuned on outcomes, IPV predictors, or IPV–outcome associations (leakage guard,
  correctly stated).

## Reproducibility / Process Assessment

- Run 2 imports Phase-1 from run 1 and recomputes the corrected universe; phases 0–14 complete. Red team: RT10
  (decision-tree precedence defect) fixed; RT03/RT04/RT06 documented nonblocking. Plan SHA-256 pinned. The
  decision logic is auditable in `analysis_unit_decision.md`.

## Supporting Role For The Program

- Provides the frozen matched-scenario universe and analysis unit that RQ012 (annotation; explicitly depends on
  "RQ011 frozen universe"), RQ011B (matched-scenario validity), and RQ009 will consume. Usable as a readiness /
  scope decision; not as an outcome or IPV result.

## Recommendation

Accept `READY_WITH_FROZEN_EXCLUSIONS` scoped to `algorithm×scenario`, with the no-run-level and replay-bias
caveats attached. Before downstream use: populate `evidence.csv`, confirm the recorded PI authorization for the
RT10 re-grade, and never describe matched-scenario readiness as run-level, repeated-run, or causal realised harm.

## Source Pointers

- `02_process/08_analysis_unit/analysis_unit_decision.md`; `02_process/01_plan_review/readiness_decision_tree.json`
- `01_results/figures/source/readiness_decision_map.csv` (F5 readiness decision map)
- `execution_status.json` (RT10 fix block); prior run `RQ011_1_...20aaee57/`
