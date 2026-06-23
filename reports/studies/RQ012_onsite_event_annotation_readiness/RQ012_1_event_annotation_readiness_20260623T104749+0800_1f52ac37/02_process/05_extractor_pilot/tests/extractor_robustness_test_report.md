# RQ012A Extractor Robustness Regression Report

Run ID: RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37
Worker ID: RQ012-W17a-extractor-robustness
Git HEAD: 13aafc86b29295187372173f408b6160b133c3c7

## Scope Firewall

Synthetic trajectory-only fixtures; no labels, IPV, scores, ranks, team identity, agreement files, outcomes, or paper repo content were read.

## Results

| finding | test | status | message |
|---|---|---|---|
| V03 | e09_e15_duplicate_ego_timestamp_rejected | pass |  |
| V03 | tied_nearest_neighbor_uses_lower_timestamp_deterministically | pass |  |
| V04 | actor_identity_change_splits_and_blocks_primary_pair_interval | pass |  |
| V05 | high_band_e15_suppresses_same_interval_e09_primary_count | pass |  |
| V01 | e02_e18_ego_hard_stop_precedence_deoverlaps_primary_counts | pass |  |

Overall: PASS
