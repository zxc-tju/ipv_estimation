# RQ014 PI decision D4: scoped G3R authorization record

Date: 2026-07-23

## Wave-A status

This document records the scoped decision surface for exactly one future managed
operation: `rq014_r3_full_rating_join_and_rank`. In this Wave-A build the
operation remains centrally absent from `allowed_operations`, and its execution
contract status remains
`DENY_PENDING_FROZEN_RATING_BLIND_FEATURE_BANK_AND_SEPARATE_RATING_AUTHORIZATION`.
This document does not itself submit a job, expose ratings, or override the
machine DENY. A separate reviewed Wave-B change must make the one-operation
allowlist/status transition.

## PI authority and scope

The PI/user full-autonomy grant recorded in the board runbook on 2026-07-20 is:

> 后续决策你可以自主推进，按照你觉得较好的方案执行即可，我授权你全部权限。HPC的资源很充沛，可以将效率作为主要目标，全力推进这个研究直到完成。

The Lead may execute the Wave-B authorization and the single managed R3 job on
that authority after all gates below pass. The authority is scoped to R3 only:
one immutable spec, one Slurm job, one controlled geometry-keyed rating-source
read, one join, and exactly 960 terminal rows (320 frozen predictor cells times
RWS, PSP, and PPR). R4, optional analyses, reranking, exploratory joins,
additional rating reads, and partial disclosure are not authorized.

## Frozen inputs and runtime boundary

- The predictor input is the immutable BANK_VERIFY=PASS G2R publication from
  `RQ014_2_blind_feature_build_20260722T210000Z_e41c8792`. Its umbrella manifest,
  PASS receipt (SHA-256 prefix `b74bb0e2`), DONE receipt, and every umbrella
  artifact are separately SHA-bound and revalidated before rating access.
- The scientific authority is `reports/plans/RQ014_recovery_lane_v3.json`.
  Its ordered B-to-I attrition, RWS/PSP/PPR kernels, stability statistics,
  terminal rollup, 960-row ledger order/hash chain, and total ranking are exact.
- Wave A does not open a rating-bearing path. The governed source location is
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/reframed_pref_analysis/phase3_preference_test/ratings_extracted.csv`,
  as recorded by the RQ010B rating-join verification. No prior reviewed receipt
  contains its content digest, so Wave B must freeze its byte size and SHA-256
  in the immutable spec; the managed R3 process verifies that binding before
  parsing the exact six-column join interface and emits only a counts-and-hashes
  rating-access receipt.
- Rating values remain in the private mode-0700 job namespace. A failure emits
  only stage/class/count/hash evidence, removes partial result bytes, writes no
  DONE, and publishes no row or rating value. PASS publication is one atomic
  rename after all 960 rows and their hash chain validate.

## Gates before the one submit

Wave B must bind a clean reviewed commit, fresh review manifest, distinct
statistics and execution/governance `NO_BLOCKER` reviews, Formal G1 PASS, final
bundle, exact env-v5 closure, the frozen G2R receipt chain, and the exact ratings
path/size/SHA. The immutable production spec must pass validate-only before the
Lead performs the explicit one-shot submit. Any missing gate, stale pin, source
drift, receipt mismatch, noncanonical row, chain error, or disclosure-policy
violation denies execution or fails closed.
