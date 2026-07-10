# Reproducible environments

The pipeline intentionally uses two isolated Python environments.

- `requirements-ipv-exact-linux-64.txt` records the direct Linux/HPC
  IPV-generation packages. Formal sigma01-compatible IPV is computed from a
  clone of the verified HPC binary environment because SLSQP also depends on
  the Python/BLAS build. The install job exports a reusable conda explicit spec.
- `requirements-ipv-verifier.txt` reproduces the frozen RQ009/RQ012 M3 scorer
  ABI. It performs quantile, conformal, OOD-gate, and deviation inference.

Do not install the verifier stack into the legacy
`/share/home/u25310231/ZXC/ipv_estimation` environment. Create a separate
environment below `/share/home/u25310231/ZXC/sociality_estimation/envs/`.

On Tongji HPC, submit `scripts/hpc/install_exact_env.sbatch` and
`scripts/hpc/install_verifier_env.sbatch`. Each job records the fully resolved
`pip freeze` under the durable deployment root's `manifests/` directory before
running its role-specific regression test.

Every durable run records the Git commit, requirements-file SHA-256, profile
SHA-256, model-manifest SHA-256, input checksums, and output checksums.
