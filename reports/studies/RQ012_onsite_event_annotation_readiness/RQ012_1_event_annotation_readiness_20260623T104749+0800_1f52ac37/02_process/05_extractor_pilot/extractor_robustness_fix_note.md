# RQ012A Extractor Robustness Fix Note

Run ID: RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37  
Worker ID: RQ012-W17a-extractor-robustness  
Scope: V01/V03/V04/V05 extractor robustness only. No labels, agreement, IPV, outcomes, scores, ranks, team identity, event-IPV association, or paper repo content were read.

## Regression Evidence

Pre-fix command:

```text
python3 reports/studies/RQ012_onsite_event_annotation_readiness/RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37/02_process/05_extractor_pilot/tests/test_extractor_robustness.py
```

Pre-fix result: 5 tests total, 4 failed.

| finding | pre-fix failure |
|---|---|
| V03 | Duplicate ego timestamps in E09 pair extraction emitted one primary interval and did not record the pair-event impossible-value guard. |
| V04 | Same actor ID with changing originId/name emitted one primary E09 interval instead of splitting and flagging attribution failure. |
| V05 | High-band same-pair/time contact produced both E09 and E15 primary endpoint counts. |
| V01 | `EventResult` had no raw/primary de-overlap metadata and E02/E18 precedence could not be represented. |

Post-fix commands:

```text
python3 reports/studies/RQ012_onsite_event_annotation_readiness/RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37/02_process/05_extractor_pilot/tests/test_extractor_robustness.py
python3 reports/studies/RQ012_onsite_event_annotation_readiness/RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37/02_process/05_extractor_pilot/tests/test_extractor_pilot.py
python3 reports/studies/RQ012_onsite_event_annotation_readiness/RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37/02_process/05_extractor_pilot/extractor_pilot.py
```

Post-fix result: W17a robustness tests 5/5 pass; W12 extractor tests 30/30 pass; same-seed pilot regenerated with `phase_status.json` reporting `tests_pass: true`.

## Fix Map

| finding | code fix | post-fix evidence |
|---|---|---|
| V01 cross-event hierarchy | Added raw vs primary endpoint bookkeeping plus `cross_event_precedence()`. E01 is represented as an E02 subset and remains non-computable without frozen counterpart relation. E18 takes precedence over overlapping ego E02 hard-stop windows. | `cross_event_audit_summary.csv` includes E01/E02 and E02/E18 rows. Central pilot has E01/E02 overlap 0 and E02/E18 overlap 0, with raw and primary counts shown explicitly. Synthetic E02/E18 hard-stop regression passes with E02 raw=1, E02 primary=0, E18 primary=1. |
| V03 timestamp validation | Added pair-event prevalidation for ego and world time series before nearest-neighbor alignment. Missing timestamps are dropped; duplicate/non-monotonic time bases are rejected and counted as impossible. Nearest-neighbor ties now use a deterministic lower-timestamp tie-break. | Synthetic duplicate-ego E09 fixture now emits 0 primary intervals and records `impossible_values > 0`. Tied-neighbor fixture consistently selects the lower timestamp. Central pilot pair impossible diagnostics are now E09=4 and E15=4. |
| V04 actor stability | Added world actor identity signatures using actor_id plus originId/name where present. Pair rows split on signature changes; affected windows increment actor-attribution failures and any resulting intervals are diagnostic-only. | Synthetic same-id origin/name change fixture now emits 0 primary E09 intervals and records attribution failures. |
| V05 E09/E15 duplicate suppression | `extract_all_events()` computes E15 before E09 and passes the active E15 overlap tolerance into E09's audit path. Cross-event precedence suppresses same-pair/time E09 primary endpoints where E15 contact exists, while retaining raw intervals in `cross_event_audit.csv`. | High-band synthetic contact fixture now reports E15 primary=1 and E09 primary=0. Central pilot E09 raw=463, primary=350, suppressed_by_precedence=113; E09/E15 audit rows=114. |

## Same-Seed Pilot Deltas

| event_id | pre_count | post_raw_count | post_primary_count | note |
|---|---:|---:|---:|---|
| E01 | 0 | 0 | 0 | Still deferred until frozen counterpart relation exists. |
| E02 | 1770 | 1770 | 1770 | No central E02/E18 overlap in this pilot. |
| E09 | 950 | 463 | 350 | Pair timestamp guard plus E15 precedence removed ambiguous/contact-primary intervals. |
| E15 | 152 | 114 | 114 | Pair timestamp guard removed ambiguous contact candidates; E15 remains primary over E09 where overlapping. |
| E18 | 0 | 0 | 0 | Central E18 remains zero; high-band sensitivity still reports E18=202. |

## Output Artifacts

- `01_results/automatic_event_pilot.csv`
- `01_results/automatic_event_pilot_report.md`
- `01_results/automatic_event_extractor_spec.md`
- `data/derived/onsite_competition/RQ012_onsite_event_annotation_readiness/RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37/extractor_pilot/cross_event_audit.csv`
- `data/derived/onsite_competition/RQ012_onsite_event_annotation_readiness/RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37/extractor_pilot/cross_event_audit_summary.csv`
