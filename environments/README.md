# Reproducible environments

The pipeline intentionally uses two isolated Python environments.

- `requirements-ipv-exact-linux-64.txt` reproduces the Linux/HPC IPV-generation
  ABI. Formal sigma01-compatible IPV is computed on Linux because SLSQP is
  platform-sensitive.
- `requirements-ipv-verifier.txt` reproduces the frozen RQ009/RQ012 M3 scorer
  ABI. It performs quantile, conformal, OOD-gate, and deviation inference.

Do not install the verifier stack into the legacy
`/share/home/u25310231/ZXC/ipv_estimation` environment. Create a separate
environment below `/share/home/u25310231/ZXC/sociality_estimation/envs/`.

Every durable run records the Git commit, requirements-file SHA-256, profile
SHA-256, model-manifest SHA-256, input checksums, and output checksums.
