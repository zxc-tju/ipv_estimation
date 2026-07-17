# RQ014 PI decision D3: authorize the rating-blind G2R build

Date: 2026-07-17

## Decision

The PI authorizes exactly one additional managed operation:
`rq014_r2_blind_feature_build`. This operation may leave
`DENY_PENDING_ACCEPTED_PREFLIGHT_PILOT_AND_PI_BUDGET` and become conditionally
authorized after Formal G1 because all three named prerequisites are now met.
No other RQ014 operation is authorized by this decision.

This authorization is rating-blind. `rating_access`, `rating_join`, and
`observed_rating_statistics` remain `NONE`/`FORBIDDEN`. G2R ends after the
canonical 320-cell blind feature bank, availability masks, anchor scores,
common-support manifest, predictor manifest, and output manifest. It must not
write a leaderboard or a recovery ledger. The rating join and rank operation
(`rq014_r3_full_rating_join_and_rank`, D4) remains a separate future PI gate.

## Preconditions and evidence

1. **D1 accepted preflight.** The scoped D1 decision is
   `reports/plans/RQ014_PI_decision_D1_preflight_v1p6_20260713.md` (2,371 bytes,
   SHA-256 `ca6237f6817a805156c2b2a243fdc42af8195be013a38b55557802924a2fdaf3`).
   Its accepted managed preflight subsequently completed PASS and supplied the
   reviewed lineage required by the pilot and G2R surfaces.
2. **D2 resource pilot passed.** D2 is recorded in
   `reports/plans/RQ014_PI_decision_D2_resource_pilot_20260714.md` (2,494 bytes,
   SHA-256 `4fa68ccdabd8e9f7fefc650d4c52a3464d1a5ed2033b18389323275bcde355a7`).
   Managed job `1930942` completed `0:0`; its immutable PASS receipt is copied at
   `.codex-fleet/rq014-execution-v1p6/board/w5e_evidence/pilot_receipt_PASS_1930942.json`
   (7,568 bytes, SHA-256
   `0f192b4e6b5db6b2ba4e889ac365f0a5e037c320673813d022a9d115c41cc184`).
   Every measured stage passed, every failure-taxonomy count is zero, and the
   receipt records `rating_access=NONE`, `rating_join=NONE`, and
   `observed_rating_statistics=NONE`.
3. **D3 compute budget approved.** The PI-approved measurement record is
   `.codex-fleet/rq014-execution-v1p6/board/runbook_next.md` (SHA-256
   `9774bae47d5a2064a5f2e526e90db3ca56c9a2a763eb0a160e2fca734baecc69`,
   2026-07-17 D3 entry). It binds pilot job `1930942` and projects the full
   320-cell G2R at `0.046839937489001185` parallel wall-clock hours (about
   2.8 minutes), `0.6670795025147223` CPU-hours, and about 11 GB peak RSS,
   comfortably inside the reviewed 16-CPU/32G/04:00:00 profile.
4. **The implementation is complete and reviewed.** W1 through W4 froze and
   tested the schemas/goldens, WOD-to-M3/anchor/NC kernel, scoring/readouts, and
   rating-blind 320-cell orchestration. W5a added only the managed operation
   surface and its hash-pin cascade. The W5a reviewed authority at base commit
   `7441f27fd7695aa1193a78716397fc12191553ac` binds Formal G1 SHA-256
   `20bd6afb9e6414259056a2793cae7d94a5594c36dabd937010c64176ca3c1c58`
   and the 144-row final bundle SHA-256
   `38c5f5357359de8cb055d9472e9e0bbec5750700c2aee36e56240686d9a60f73`.
   The frozen construction, NC, readout/status, portable M3, anchor-domain, and
   orchestration goldens remain the acceptance evidence for the authorized
   implementation.

## Operational boundary

This decision changes repository authorization bytes only. It does not publish
a run spec, mutate HPC state, or submit Slurm work. A production G2R run still
requires a fresh immutable spec and upstream lineage bound to the final reviewed
commit, successful validate-only, and the existing explicit operator submit
step. Any authority, environment, M3, receipt, rating-boundary, NC, or output
integrity mismatch fails closed.
