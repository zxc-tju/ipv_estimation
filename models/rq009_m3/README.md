# Frozen RQ009 M3 verifier bundle

`m3_scorer.joblib` is a portable, LZMA-compressed re-export of the accepted
RQ009/RQ012 M3 scorer. It is not retrained. Its inference classes live under
`sociality_estimation.verifier.model`, and the absolute Mac calibration path
from the legacy serialization has been removed.

The fitted scorer is intentionally ignored by Git because its OOD tree contains
2,557,510 transformed reference rows. Git tracks this README, both contracts,
and `manifest.json`; the private artifact is synchronized through SSH and must
match the manifest SHA-256 before loading.

The scorer must never be replaced without updating `manifest.json` and passing
the compact local/HPC verifier parity fixture. Loaders verify the SHA-256 before
deserialization.

The original ignored scorer remains the provenance source with SHA-256
`bf9a0c7ae41ba9efcb2ad997aaac1b7881d7788cf8dadd01252c17ed7a6b0ba5`.
It is retained only at
`data/derived/_provenance_archive/rq009_m3_legacy_source_20260711/m3_scorer.joblib`
to rebuild and audit this portable bundle.  It is **not** a deployment model:
do not point `SOCIALITY_M3_SCORER` at it or copy it to HPC.  The only supported
runtime artifact is this directory's manifest-verified portable scorer.

`legacy_feature_spec_contract.json` preserves the original contract byte
semantics. `feature_spec_contract.json` corrects only the three padded quantile
column spellings (`q_0p10/q_0p50/q_0p90`) to the accepted runtime outputs
(`q_0p1/q_0p5/q_0p9`).
