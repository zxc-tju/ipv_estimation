# Managed research run specifications

Production HPC runs are submitted only through
`scripts/hpc/submit_research_run.sh`.  The specification uses JSON syntax
(which is valid YAML 1.2) and must include the exact Git commit, input-manifest
path and SHA-256, RQ/operation authorization, and input paths.  Generated code,
logs, manifests, and outputs live under one immutable
`work_dirs/<RQ>/<run_id>/` tree.

RQ014 has no authorized operations.  Infrastructure-only preflight and parity
fixtures are registered under `INFRA`; they cannot perform rating joins.
