# RQ003 Phase 12 Registrar Report

Worker: `RQ003_phase12_registrar_001`

Run: `RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0`

Status: PASS

Generated local time: `2026-06-21T00:41:45+0800`

## Identity Verification

| Check | Expected | Observed | Result |
|---|---|---|---|
| Run root exists | yes | yes | PASS |
| `run_manifest.json` run ID | `RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0` | matched | PASS |
| `plan_sha256.txt` | `98900786a4819e58e91f16eb8da84724c928ca5a6d2c40b2252cf106b56625e1` | matched | PASS |
| `tier_decision.json` tier | `B` | `B` | PASS |
| `final_review_status.json` status | `PASS` | `PASS` | PASS |

## Shared Edits Applied

| File | Edit | Scope |
|---|---|---|
| `reports/studies/RQ003_nsfc_external_evidence/README.md` | Appended one RQ003_6 execution row. | Minimal append to existing Executions table. |
| `main_workflow.log` | Appended one registrar workflow summary block. | Append-only; existing bytes preserved. |
| `START_HERE.md` | Added one RQ003_6 Tier B reader-entry pointer. | Minimal operating-brief update because the latest stable RQ003 report pointer changed. |

## Proposal-Only Artifacts

| File | Purpose |
|---|---|
| `proposed_knowledge_update.md` | Human-review proposal for knowledge-layer interpretation and STUDIES.md handling. |
| `proposed_decision_patch.diff` | Suggested patch for `reports/knowledge/RQ003_nsfc_external_evidence/decision.md`; not applied. |

## Files Intentionally Not Modified

- `STUDIES.md`
- `reports/knowledge/RQ003_nsfc_external_evidence/decision.md`
- The standalone paper repository at `../9_overleaf/NMI---Online-Sociality-Verfication-for-Autonomous-Vehicle`

No commit or push was performed.

Git context note: the task brief listed `GIT_HEAD` as
`c23074a091f9ff57b1034144571f68f771db9d8d`, but final verification observed
the current repository HEAD as `394bb61a41cd224fc5c5366566039a5828b7ad70`.
This field was not one of the explicit blocked pre-write identity gates. Branch
status remained `main...origin/main`, so no commit or push was performed during
registration.

## Key Conclusion Registered

Tier B conclusion: no robust incremental predictive utility relative to the
prespecified kinematic+safety baseline was demonstrated. The run is
power-limited to the approved top-five cohort (N=53), and the apparent favorable
direction is not IPV-specific.

Boundary conditions:

- H3 is blocked because real two-human labels are absent.
- NPC is boundary-only and non-identifiable.
- The full 20-team universe is not analysis-ready.
- Reader-facing validation-success claims are not allowed by the Tier B decision.

## Conflict Check

The shared files were hashed once during initial read and again immediately
before editing. The second pre-edit hashes matched the first pre-edit hashes for
all shared files, so no mid-edit shared-file conflict was detected before
patching.

| File | First pre-edit SHA-256 | Second pre-edit status | Post-edit SHA-256 |
|---|---|---|---|
| `reports/studies/RQ003_nsfc_external_evidence/README.md` | `9992b70cb4da810c8ca5c65a9e0d9c9e4881af3f6f6912f71bfa5e6eb79283a8` | matched | `64e9be469f133d0aae483968e94b16344fbfd2bcd69d2b1385851043e502e69f` |
| `STUDIES.md` | `209295a4790739b4a2755db9ca853432af5df9820ff387e7e2fd8589e47d5e46` | matched | `209295a4790739b4a2755db9ca853432af5df9820ff387e7e2fd8589e47d5e46` |
| `START_HERE.md` | `95b05439f0be8933aecb559b38a94734a101aad369a03c90e5409c0809a21dd4` | matched | `5f3583a7325f137e351d838dfde6bed899c9bbedf9aabae8eadb9ce6b7794f5e` |
| `main_workflow.log` | `f7dd7a582085121ea5c018843bd12e9df8b4f6c18b04416fd296878a24d2e0d0` | matched | `8a23eb967e0c6153bcd128eba21480d9425de3fc9ad375dc2cd6e9946c6c21ad` |
| `reports/knowledge/RQ003_nsfc_external_evidence/decision.md` | `337b3f54211675fb8763d40c20bea7f89503a34dfc1d5ded2961c0005c242344` | matched | `337b3f54211675fb8763d40c20bea7f89503a34dfc1d5ded2961c0005c242344` |

## Diff Summary

- RQ003 study README: one execution row appended for RQ003_6.
- START_HERE: one RQ003_6 Tier B reader-entry pointer added.
- main_workflow.log: one registrar summary block appended.
- Knowledge and decision updates were written only as run-package proposals.

## Implementation Note

`main_workflow.log` contains invalid UTF-8 bytes, so the patch tool could not
open it safely. The workflow-log entry was appended with a binary-safe Python
append to preserve all existing bytes and avoid rewriting unrelated content.
