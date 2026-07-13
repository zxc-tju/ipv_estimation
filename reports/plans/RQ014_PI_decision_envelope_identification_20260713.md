# RQ014 PI scoped decision — envelope identification

Date: 2026-07-13. Source: interactive PI decision recorded in
`.codex-fleet/rq014-execution-v1p6/board/validation.md` under **PI ENVELOPE IDENTIFICATION** after review of
`.codex-fleet/rq014-execution-v1p6/board/reports/w4d_envelope_candidates.zh.html`.

## Exact identification

The PI identified the lost-recipe envelope as candidate ①, the frozen RQ009 **M3 conformal model** used by the
historical phase-2 lineage. The exact artifact set is:

| Artifact | Exact file SHA-256 | Size |
|---|---|---:|
| `models/rq009_m3/m3_scorer.joblib` | `b04999aba29a82fb71a97ac22c728479a7734e24a0b32189d08f95184d74f253` | 88,306,301 B |
| `models/rq009_m3/manifest.json` | `2efbdd0c39edabc419aad815a1eb7529af3623a06c4d3a0b0a99782bcb2f40f4` | 2,368 B |
| `models/rq009_m3/feature_spec_contract.json` | `3ad8ba8ab4c51422a7b2ef208683b7552b68f9e949f0087542ba208065677cce` | 6,829 B |

The scorer is intentionally ignored/private rather than a tracked Git blob, as documented by
`models/rq009_m3/README.md`; its bytes are bound by the tracked manifest. A read-only verification on 2026-07-13
found `/share/home/u25310231/ZXC/sociality_estimation/checkpoints/rq009_m3/m3_scorer.joblib` at exactly the stated
size and SHA-256, and found the checkpoint manifest and feature contract byte-identical to the managed-repository
files. No HPC file was written.

## Decision basis and transfer semantics

The decision followed the recorded three-candidate comparison: ① M3 conformal; ② phase2c M3 humannorm; and ③ the
path-type HV norm table. M3 alone is the identified model artifact and has the complete scorer/manifest/feature-spec
provenance chain. The historical phase-2 recipe maps M3 `q_0p5` to the center and `lo_90`/`hi_90` to the central-90
envelope.

The PI explicitly acknowledges that WOD deviations are computed under **out-of-support extrapolation semantics**.
The bounded historical pilot recorded 0/228 rows in support. The support-gate and OOD fields remain diagnostics;
they do not drop, mask, or abstain from otherwise eligible WOD rows. This acknowledgement permits computation but
does not turn extrapolated scores into an in-support or externally validated claim.

## Consequences and claim boundary

- The envelope is one frozen input, not a search axis: the recovery grid is 2 frequencies × 8 temporal recipes ×
  2 horizons × 10 readouts = 320 predictor cells and 320 × 3 associations = 960 terminal screen rows.
- A new InterHub envelope export is cancelled. G4R loads the exact frozen M3 artifact set while independently
  rewriting every other computation.
- X02 is inactive and unbound under option (b); its historical sites remain addressable only as legacy provenance.
- If the frozen grid ends `NOT_RECOVERED_WITHIN_FROZEN_RECOVERY_GRID`, wrong-envelope memory remains a candidate
  explanation, but D5B in `RQ014_plan_v1p6_execution_handoff_20260712.md` covers only bounded non-recovery. Widening
  the model, envelope, grid, or claim requires a new amendment and authority loop.

This scientific identification does not authorize HPC submission, rating access, G2R, G3R, or G4R. Machine use
still requires fresh dual review of the amended bytes, a new Formal G1, a final bundle, publication/sync, immutable
specification, validate-only PASS, and every applicable scoped authorization.
