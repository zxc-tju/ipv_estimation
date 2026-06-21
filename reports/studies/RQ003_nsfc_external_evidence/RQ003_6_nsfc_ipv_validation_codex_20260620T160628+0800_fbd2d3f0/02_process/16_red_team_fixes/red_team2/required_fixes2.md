# Required Fixes From Red Team v2

Status: `BLOCKERS_FOUND`

1. Run an independent corrected N=53 replication without importing Phase 4 or Phase 7 modeling scripts. Reproduce or reconcile `cv_predictions.csv`, `confirmatory_results.csv`, fold-local residualization, alpha selection, imputation, scaling, and per-cell predictions.

2. Downgrade all result wording. The allowed wording is: "corrected labels produce a weak favorable but nonsignificant direction." Forbidden wording includes robust, validated, established, final confirmatory support, or meaningful safe-subset requirement met.

3. Reinterpret or rerun negative controls. Controls that shuffle IPV, swap counterpart conditioning, or use future-leaky features match/exceed the primary delta. Degradation controls failed. This must be treated as a robustness failure unless a corrected control design explains it.

4. Remove safe-subset robustness support unless a genuinely distinct safe subset agrees. S1 and S2 are identical to the primary sample under corrected A1/safety membership, and S3 has only 6 cells with null/reverse direction.

5. Formalize the corrected scenario/fold contract. The in-memory official LOSO/LOFO regeneration is a reasonable error fix, but the frozen `fold_contract.csv` still contains the old erroneous A1-C5 grid.

6. Repair or clarify crosswalk provenance pointers. The corrected labels are supported, but stored SQL line pointers are exact-session matches for only 30/150 cells.

7. Update reader-facing metadata only after the blockers above are resolved. Current entry pages and run status still say no formal statistics were run.

