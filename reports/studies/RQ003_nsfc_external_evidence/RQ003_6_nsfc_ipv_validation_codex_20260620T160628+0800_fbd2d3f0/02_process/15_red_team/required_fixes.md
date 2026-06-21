# Required Fixes

1. **Fix scenario identity first.**
   Build and freeze a scenario crosswalk that reconciles `scenario_map_outcome_free.csv`, `replay_score_mapping.csv`, raw `tj_competition_case` structure, case names, and official scenario codes. The crosswalk must be outcome-clean or must explicitly document any intentional opening of score metadata without score values.

2. **Rerun the primary analysis from the corrected map.**
   Recompute primary inclusion, scenario/area residualization, LOTO, LOSO, LOFO, safe subsets, and `cv_predictions.csv`. Treat the current null/reverse result as provisional until this rerun is complete.

3. **Regenerate A1 and collision-free reporting.**
   Remove the false statement that the only non-100 safety rows are structural A1. Report catastrophic/zero-score rows by corrected official scenario and cell identity.

4. **Rerun controls and boundary analyses.**
   Recompute negative controls, future-leaky diagnostic, state-dependence/FDR tables, counterexamples, and any safe-subset summaries under the corrected labels.

5. **Repeat independent replication and red-team review.**
   Do not proceed to Tier review until a fresh replication confirms the corrected outputs and a fresh red team finds no blockers.

6. **Constrain final wording.**
   If the corrected rerun remains null, phrase the conclusion as "power-limited no detected incremental utility over the prespecified baseline in the scoped high-support top-five cohort." Do not write "proven no effect."
