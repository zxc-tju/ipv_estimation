# Reproducible IPV estimator and verifier deployment

## Canonical roles

The frozen legacy checkout at
`/share/home/u25310231/ZXC/ipv_estimation` remains an immutable historical
reference at Git commit `5edd2810`. New work uses the Git-managed checkout at
`/share/home/u25310231/ZXC/sociality_estimation/code/repo`.

Formal deployment is commit-addressed rather than branch-addressed. The HPC
checkout is detached at an exact commit and must stay clean. Data, logs,
environments, checkpoints, and outputs live beside `code/repo`, never inside it.
Git tracks the scorer contract and checksum manifest, but not the fitted
2.56-million-row OOD reference tree. The private model bundle is copied through
SSH with `scripts/hpc/sync_private_m3_bundle.sh` and verified before activation.

## Pipeline contract

1. Run IPV generation in the `ipv-exact` Linux environment with
   `--execution-profile configs/ipv_sigma01_exact.json`. The pipeline validates
   solver/window/reference/sigma/downsampling values and records the profile
   SHA-256 in its summary.
2. Map dataset-specific trajectories into the standardized causal-history
   columns and call `build_m3_anchor_features`; this produces the complete
   32-column M3 scorer input without relying on ignored report scripts.
3. Run `sociality_estimation.verifier.score_anchors` in the isolated verifier
   environment.
4. Compute raw signed/absolute envelope exceedance with
   `sociality_estimation.verifier.raw_envelope_deviation`.

The RQ011 half-width-normalized deviation is analysis-specific and is not the
canonical verifier output.

## Git synchronization

Publish a reviewed commit locally, then deploy exactly that commit:

```bash
git push -u origin codex/unify-ipv-pipeline
COMMIT=$(git rev-parse HEAD)
ssh tongji-hpc \
  "BRANCH=codex/unify-ipv-pipeline COMMIT=$COMMIT bash -s" \
  < scripts/hpc/sync_tongji_checkout.sh
```

Never use `git pull` with local HPC changes. If the HPC checkout is dirty, the
sync script fails closed.

If HPC cannot reach GitHub during the first clone, create and copy a Git bundle,
then point the same sync script at the remote bundle. The resulting checkout is
still a normal Git repository whose `origin` is GitHub:

```bash
git bundle create /tmp/sociality-estimation.bundle codex/unify-ipv-pipeline
scp /tmp/sociality-estimation.bundle \
  tongji-hpc:/share/home/u25310231/ZXC/sociality_estimation/manifests/
ssh tongji-hpc \
  "BRANCH=codex/unify-ipv-pipeline COMMIT=$COMMIT \
   BOOTSTRAP_BUNDLE=/share/home/u25310231/ZXC/sociality_estimation/manifests/sociality-estimation.bundle \
   bash -s" < scripts/hpc/sync_tongji_checkout.sh
```

Synchronize the checksum-bound private scorer after the code commit is deployed:

```bash
bash scripts/hpc/sync_private_m3_bundle.sh
```

Install and verify the two isolated runtimes through Slurm:

```bash
ssh tongji-hpc 'cd /share/home/u25310231/ZXC/sociality_estimation/code/repo && \
  sbatch scripts/hpc/install_exact_env.sbatch && \
  sbatch scripts/hpc/install_verifier_env.sbatch'
```

The exact-environment job clones the retained, verified sigma01-generation
binary environment without modifying it. It exports both a conda explicit spec
and `pip freeze`, then runs `scripts/verify_ipv_estimator_fixture.py`. Matching
only NumPy/SciPy version strings is insufficient because Python and BLAS builds
can change SLSQP output.

The release gate requires `exact` to match sigma01 within `1e-12`. The
non-canonical vectorized `fast` backend is reported separately and must remain
within `0.002` of `exact` on the frozen fixture; formal profiles always select
`exact`.
