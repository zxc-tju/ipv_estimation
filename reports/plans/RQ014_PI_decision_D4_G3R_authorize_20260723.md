# RQ014 PI decision D4: scoped G3R authorization record

Date: 2026-07-23

## Wave-B authorization

This document authorizes exactly one managed operation:
`rq014_r3_full_rating_join_and_rank`. Wave B adds only that operation to central
`allowed_operations` and changes its execution-contract status to
`CONDITIONALLY_AUTHORIZED_AFTER_FORMAL_G1_AND_SCOPED_D4_DECISION`. The flip is
conditional, not a submission: no job may start until the immutable bank,
ratings path/size commitment, fresh dual reviews, Formal G1, final
bundle, env-v5 closure, immutable spec, and validate-only gate all pass.

The authorization boundary is one job, one controlled rating-source read, one
geometry-keyed join, and one atomic 960-row terminal publication. Partial
disclosure, additional joins, exploratory access, reranking, and any R4 work
remain forbidden.

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
- Neither Wave A nor this Wave-B build opens or hashes a rating-bearing path.
  The governed source location is
  `/share/home/u25310231/ZXC/RQ010B_wod_e2e/reframed_pref_analysis/phase3_preference_test/ratings_extracted.csv`,
  and its Lead-observed stat-only size is `337416` bytes. The blind pipeline's
  declassification receipt records `rating_access: NONE` and intentionally has
  no source digest. The immutable spec therefore binds the source as exact
  path plus size with `sha256: null`. The first controlled contact occurs only
  inside managed R3: it computes the source SHA-256 before parsing, records that
  digest in the counts-and-hashes-only rating-access receipt as the governed
  digest of record, and enforces any non-null digest supplied by a future
  governed source. This preserves rating blindness throughout authorization
  and review.
- Rating values remain in the private mode-0700 job namespace. A failure emits
  only stage/class/count/hash evidence, removes partial result bytes, writes no
  DONE, and publishes no row or rating value. PASS publication is one atomic
  rename after all 960 rows and their hash chain validate.

## Gates before the one submit

Wave B binds a clean reviewed commit, fresh review manifest, distinct
statistics and execution/governance `NO_BLOCKER` reviews, Formal G1 PASS, final
bundle, exact env-v5 closure, the frozen G2R receipt chain, and the exact ratings
path/size/null-digest first-contact commitment. The immutable production spec
must pass validate-only before the
Lead performs the explicit one-shot submit. Any missing gate, stale pin, source
drift, receipt mismatch, noncanonical row, chain error, or disclosure-policy
violation denies execution or fails closed.

## Wave-B runtime-binding status

<!-- WAVE_B_BINDING_STATUS:START -->
Bindings finalized: bank manifest `2b4da1df4a5328b80d88b815ac3cdb71546952bac4638b29f4fa263b527d4515`, bank receipt `b74bb0e2ab5966b9eaaab164130bd50791b5ceee5743030b0bb26719d79c37b9`, DONE `256750c71902e31e46335c331369c256b7b7d13a4fb08758f1b8234b6229efdb`, and ratings size `337416` with `sha256: null`; managed R3 will establish and record the governed source digest at first controlled contact. The ratings file was not opened or hashed by the build/finalizer.
<!-- WAVE_B_BINDING_STATUS:END -->
