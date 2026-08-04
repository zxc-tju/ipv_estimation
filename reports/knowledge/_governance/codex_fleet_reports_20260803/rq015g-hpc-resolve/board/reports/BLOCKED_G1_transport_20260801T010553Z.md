# BLOCKED_G1 - HPC Transport Unavailable

- time_utc: `2026-08-01T01:05:53Z`
- status: `BLOCKED_BEFORE_SSBATCH`
- scope: G1 did not create a remote work_dir, did not submit Slurm, and did not produce HPC resolve numbers.

## Evidence

The normal sandbox shell cannot open the HPC connection:

```text
ssh -o BatchMode=yes -o ConnectTimeout=12 tongji-hpc 'hostname; whoami; pwd'
ssh: connect to host 10.168.164.207 port 22: Operation not permitted
Connection closed by UNKNOWN port 65535
```

A direct socket probe failed the same way:

```text
nc -vz -G 8 10.168.164.207 22
nc: connectx to 10.168.164.207 port 22 (tcp) failed: Operation not permitted
```

The prepared staging script failed before remote directory creation:

```text
.codex-fleet/rq015g-hpc-resolve/work/stage_and_submit_g1_hpc.sh
ssh: connect to host 10.168.164.207 port 22: Operation not permitted
Connection closed by UNKNOWN port 65535
```

A node_repl fallback probe failed with the same transport error:

```text
ssh -o BatchMode=yes -o ConnectTimeout=12 tongji-hpc 'hostname; whoami; pwd'
ssh: connect to host 10.168.164.207 port 22: Operation not permitted
Connection closed by UNKNOWN port 65535
```

No password prompt appeared.

## Prepared Artifacts

- `.codex-fleet/rq015g-hpc-resolve/work/run_g1_hpc.py`
- `.codex-fleet/rq015g-hpc-resolve/work/submit_rq015g_anchor_resolve.sbatch`
- `.codex-fleet/rq015g-hpc-resolve/work/stage_and_submit_g1_hpc.sh`
- `.codex-fleet/rq015g-hpc-resolve/work/fetch_g1_hpc_outputs.sh`
- `.codex-fleet/rq015g-hpc-resolve/work/local_input_manifest.json`

Static checks completed:

- `run_g1_hpc.py`: Python syntax check passed with `PYTHONPYCACHEPREFIX` redirected into G1 work.
- shell scripts: `bash -n` passed for submit, stage, and fetch scripts.
- `local_input_manifest.json`: records the 9 sample-used PKL SHA-256 values and the frozen local input/file SHA values.

## Network-Restored Command

From the repository root, run:

```bash
.codex-fleet/rq015g-hpc-resolve/work/stage_and_submit_g1_hpc.sh
```

Then monitor:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=12 tongji-hpc 'squeue -u u25310231'
```

After completion, fetch:

```bash
.codex-fleet/rq015g-hpc-resolve/work/fetch_g1_hpc_outputs.sh
```

The fetched run is expected to contain `anchor_mse_hpc.csv`, `g1_compare.json`,
`g1_hpc_summary.json`, `hpc_environment_manifest.json`, and
`board/reports/G1_hpc_resolve_report.md`.
