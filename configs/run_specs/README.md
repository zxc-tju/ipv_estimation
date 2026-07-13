# Managed research run specifications

Production HPC runs are submitted only through
`scripts/hpc/submit_research_run.sh`.  The specification uses JSON syntax
(which is valid YAML 1.2) and must include the exact Git commit, input-manifest
path and SHA-256, RQ/operation authorization, and input paths.  Generated code,
logs, manifests, and outputs live under one immutable
`work_dirs/<RQ>/<run_id>/` tree.

For RQ014, the only supported operator invocation is:

```sh
/usr/bin/env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C /bin/sh -c 'wrapper=/share/home/u25310231/ZXC/sociality_estimation/code/repo/scripts/hpc/submit_research_run.sh; lock=/share/home/u25310231/ZXC/sociality_estimation/manifests/runtime_maintenance.lock; test ! -L "$wrapper" && test -f "$wrapper" && test ! -L "$lock" && exec 8>"$lock" && /usr/bin/flock -s 8 && exec 9<"$wrapper" && test "$(/usr/bin/readlink /proc/$$/fd/8)" = "$lock" && test "$(/usr/bin/readlink /proc/$$/fd/9)" = "$wrapper" && /usr/bin/printf "%s  %s\n" a504e4f1593575ca28251fe1b1d4ac7d959ea9257d0c2e4984a30ba42020106e /proc/$$/fd/9 | (cd / && /usr/bin/sha256sum --check --strict -) && exec /bin/sh /proc/$$/fd/9 "$@"' rq014-bootstrap --spec /share/home/u25310231/ZXC/sociality_estimation/manifests/RQ014/run_specs/REPLACE_RUN_ID.json --submit
```

This single clean-environment command is the authorization boundary rather
than a convenience example: it locks inherited fd8, opens the exact managed
wrapper as fd9, hashes that retained descriptor, and executes the same fd9
instead of reopening the wrapper path. The wrapper may only inherit these two
descriptors; it cannot create a missing capability itself. Before either RQ014
dependency is preloaded, the launcher checks both exact `/proc/self/fd` targets,
regular non-symlink path identities, and descriptor/path device, inode and mode.
This is a local machine provenance gate, not a cryptographic secret against
deliberate same-account descriptor emulation; the reviewed clean-bootstrap hash
remains the wrapper byte trust anchor.
A direct caller-supplied `--rq014-only`, a plain boolean, or an arbitrary object
cannot authorize RQ014 validation/submission; missing or mismatched descriptors
are rejected before `materialize_registry.py` or `preflight.py` is loaded.
A shell script cannot undo a dynamic-loader hook that ran before the script
started. The wrapper therefore also rejects `HPC_SOCIALITY_ROOT`,
`BASH_ENV`, `ENV`, `LD_PRELOAD`, `PYTHONHOME`, `PYTHONPATH`, and every inherited
`SBATCH_*` variable. It uses the fixed checkout and managed interpreter, starts
Python with `-I -S -B`, submits with an absolute `sbatch --export=NIL`, and the
generated job enters the Python entrypoint through a second empty environment.
`NIL` is intentional: Slurm's `NONE` mode can implicitly reconstruct the user
login environment, while `NIL` forbids user-variable export. Before either the
launcher or job starts managed Python, the runtime gate verifies the pinned
launcher, preflight and registry-materializer bytes, stdlib checksum manifest, exact regular-file
count and total size, zero stdlib symlinks, the absent `python39.zip`, and the
complete pinned native-library closure including exact loader links and final
regular-file digests.

Both isolated Python stages create `scripts` and `scripts.rq014` with empty
`__path__`. They explicitly preload the exact closed-snapshot
`materialize_registry.py` with `spec_from_file_location` as
`scripts.rq014.materialize_registry` before loading preflight and the fixed
entrypoint; ordinary path imports and local shadows remain unavailable.

RQ014 v1.5 initially allowlisted only `rq014_g2_declassification_export`. It
converts eight exact score-omitting Phase-1 bundles plus structural and
counterpart tables into the canonical score-stripped CSV/JSON bundle. The
launcher passes every reviewed role/size/SHA-256, and the exporter opens each
source once without following links, verifies and parses only the retained
descriptor bytes, then records that same digest; raw
TFRecords and rating tables are forbidden. After the 2026-07-13 D1 decision, the
candidate central authority also lists `rq014_g2_contract_preflight`, but the old
v1.5 Formal G1/final bundle cannot authorize the changed authority byte. Preflight
remains machine-denied until fresh dual review, a new checksum-bound
`FORMAL_G1_PASS`, a new final bundle, the exact published `origin/main` commit,
immutable spec, validate-only PASS, and the promised explicit user confirmation.
Both operations bind their operation-scoped PI decision and the exact reviewed source
inventory and `managed_python_environment_v3.json`; an arbitrary self-reported
source or environment hash is rejected. Contract preflight additionally
requires the prior export PASS receipt and `DONE.json`. Its required
`declassification_export_commit` field binds the exporter provenance to the
published prior-export commit independently of the current preflight `git_commit`.
Schema v2 also defines an exact `m3_artifact` block containing path, size
`88306301`, and SHA-256 `b04999aba29a82fb71a97ac22c728479a7734e24a0b32189d08f95184d74f253`.
The block is required for contract preflight and prohibited for declassification
export; an export spec containing it fails exact-key validation. For preflight,
the launcher opens the fixed managed checkpoint once with no-follow/nonblocking
flags, requires containment and a regular file, verifies size and SHA-256 before
any input-manifest or materialization-ledger processing, and never deserializes or
scores it. The generated job repeats the byte check at start; Python repeats the
retained-descriptor verification and binds it into the read-only preflight receipt.
`M3_ARTIFACT_MISMATCH` is a global abort with zero cell/ledger/rating processing.
InterHub is retained only as historical provenance: active G2 instead requires a
separately checksum-pinned WOD path-type mapping manifest. Scientific G2R remains
denied until a future managed-environment closure v4 separately freezes and reviews
the required joblib/numpy/pandas/scipy/scikit-learn runtime.
The WOD mapping freeze is defined by
`reports/plans/RQ014_plan_v1p7_addendum_pathtype_20260713.md`: its reviewed scene-level
table must equal the `valid.envelope.wod_path_type_mapping.mapping_table_sha256`
materialization binding. `UNMAPPED_EXCLUDED` scenes are absent from the four-value
table and fail closed as `MISSING_WOD_PATH_TYPE / INELIGIBLE_BLIND`; preflight never
infers a replacement class.

The preflight authorization chain is forward-bound to these exact v1.6 paths:

- review manifest: `reports/plans/RQ014_plan_v1p6_preflight_review_manifest_20260713.sha256`;
- statistics verdict: `reports/studies/RQ014_wod_e2e_rating_recovery/01_plan_review/RQ014_v1p6_preflight_statistics_review_20260713.json`;
- execution/governance verdict: `reports/studies/RQ014_wod_e2e_rating_recovery/01_plan_review/RQ014_v1p6_preflight_execution_governance_review_20260713.json`;
- Formal G1: `reports/studies/RQ014_wod_e2e_rating_recovery/01_plan_review/RQ014_formal_G1_v1p6_preflight_20260713.yaml`;
- final bundle: `reports/plans/RQ014_plan_v1p6_checksums_20260713.sha256`.

The verdicts, Formal G1, and final bundle do not exist before the candidate manifest passes fresh dual review.
Adding preflight to the central allowlist changes a reviewed authority byte and
therefore requires a rebuilt candidate manifest, fresh statistics and execution
reviews, a new formal G1 artifact and final bundle; the already reviewed
execution-contract status does not change. The two passing review payloads must
also declare different nonempty `reviewer_agent` identities.

RQ014 does not create a Git worktree. The launcher copies only regular files
registered by the final contract checksum bundle into a closed code snapshot.
Every snapshot byte is read directly from the declared published Git commit
tree and checked against the registered digest; the worktree is never a source.
The job checks the checksum bundle file's own pinned digest and then verifies
every registered byte. Repository hooks, fsmonitor commands, checkout filters,
dirty/untracked worktree bytes, and unregistered commit files therefore cannot
enter the execution tree.

Validate-only is side-effect free: it emits the exact commit-blob
`code_snapshot_files` plan, M3 path/size/hash plus retained-descriptor verification
evidence, job/resource/thread plan and pinned runtime metadata.
It does not create a snapshot receipt, run root or rendered sbatch script. Those
artifacts, including both concrete `--export=NIL` controls, are created and
revalidated only inside the authorized submit path.

The current review-candidate recovery route is defined in `reports/plans/RQ014_recovery_lane_v3.json`.
It fixes the checksum-bound RQ009 M3 conformal model as the sole envelope and defines a 320-cell
rating-blind feature grid followed by a 960-row full-data screen. Resource pilot, feature build,
rating recovery screen, clean replay, optional power/stability, and every other
rating-bearing operation remain centrally unauthorized.

Schema v2 is fixed by `research_run_spec_v2.schema.json`, while the managed
standard-library interpreter receipt is fixed by
`rq014_managed_python_environment_v3.schema.json`; the checked-in RQ014
template is intentionally non-executable until every placeholder is replaced
with an exact path/hash in an immutable managed manifest area. The final
production spec must be a direct child of
`/share/home/u25310231/ZXC/sociality_estimation/manifests/RQ014/run_specs/`, be a
regular non-symlink read-only file, and be written once as canonical JSON
(sorted keys, compact separators, one trailing newline). The launcher reads it
through one no-follow descriptor, retains those exact bytes through validation
and sealing, and never reopens the spec path.
