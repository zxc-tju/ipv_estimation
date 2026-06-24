# Claude Code Review

Status: filed (2026-06-24)

Reviewer role: research reviewer / repository integrator.

Run reviewed: `RQ008_1_temporal_ipv_discovery_20260622T234914+0800_3e3e776a` (`RQ008A_COMPLETE`; confirmation hold-out never opened).
Reader entry: `reports/studies/RQ008_interhub_temporal_ipv_discovery/RQ008_1_temporal_ipv_discovery_20260622T234914+0800_3e3e776a/90_report/index.html`.

## Verdict

Concur with the registered negative discovery boundary. RQ008A is a **credible exploratory negative**:
0/24 fixed candidate directional temporal structures (18 motifs, 2 alignments, 4 role/uncertainty effects)
survived the Phase-7 mechanical/compositional control gate. The accepted reading — apparent temporal
"dynamics" are explained by time-reversal invariance, source composition, duration, and estimation
uncertainty rather than a directional temporal IPV law — is the correct, disciplined conclusion. No
positive motif/feedback/reciprocity/lead-lag/risk-coupling/IPV-mean temporal-law claim is warranted.

## Key Findings

| Check | Result | Reading |
|---|---|---|
| Control survival | 0/24 | No candidate survives; headline negative. |
| Dominant killer | reversed_time (most motifs) | Patterns are reverse-invariant → not directional processes. |
| Split integrity | 22,937 discovery / 15,291 confirmation; confirmation never opened | Discovery-side negative is honest. |
| Independent checks | review PASS_RATIFIED; red team PASS_WITH_FIXES; replication PARTIAL | Negative holds; change-point divergence doesn't affect 0/24. |

## Critical Caveat (load-bearing)

The reversed-time + one-vs-rest SNR pairing can be **direction-insensitive** and therefore over-kill
reverse-invariant motif shape. The accepted boundary is correctly scoped to **directional** temporal IPV
structure, not "no temporal structure of any kind." The decision's mandatory pre-Wave-B Attack-10 amendment
(add a signed direction-sensitive companion statistic, or scope the null to directional structure exceeding
reverse-invariant shape) is the right gate and must precede any confirmation opening. I concur and flag this
as the single most important constraint on interpretation.

## Boundaries And Watch-Items (confirmed)

- Single dataset family / single sigma (InterHub sigma=0.1). Source/geometry imbalance: `waymo_train` 60.7%,
  `MP` geometry 90.3% — a composition confound the controls partly absorb but do not eliminate.
- Estimability here is a **proxy/error-based** quantity; no frozen RQ007 estimability/valid-window contract
  was available to this run. Cross-RQ note: future temporal work should consume RQ007's now-frozen contract
  rather than a local proxy.
- Change-point rows remain implementation-sensitive (replication divergence) but were already rejected by
  controls, so they do not affect the 0/24 result.
- Do not over-generalize: this is a discovery-layer negative on one setting, not a universal claim.

## Reproducibility / Process Assessment

- Split gate PASS before exploration; plan SHA-256 pinned (`3a94986...`). The conservative control battery
  (time-shuffle, reversed-time, pseudo-pair, duration-matched, random-alignment, source-balancing,
  uncertainty-only, estimability-matched) is what makes the negative credible — passing descriptive screens
  alone was not enough. Red team returned zero blocking findings (nonblocking: source composition, clustering
  instability, Attack-10 false-negative risk). Replication PARTIAL (4/5 checks agree; split reproduced exactly).

## Supporting Role For The Program / Manuscript

- Usable only as an exploratory negative boundary / limitation / program-planning result, scoped to directional
  temporal IPV structure on InterHub sigma=0.1. Prevents the manuscript and downstream RQ009 from inheriting
  motif labels as social/behavioural laws.
- Consistent with RQ007 (whose lifecycle is explicitly descriptive). Together they discipline online-IPV-dynamics
  claims.

## Recommendation

Accept the negative discovery boundary. Keep Wave B frozen (`FROZEN_RQ008A`); open only after explicit
authorization, fixed-code review, and the Attack-10 direction-sensitive amendment. Do not elevate any positive
temporal claim from RQ008A.

## Source Pointers

- `02_process/08_controls/survival_summary.csv` (Fig. 2 control survival matrix)
- `02_process/09_hypothesis_freeze/freeze_register.csv`
- `evidence.csv` (4 frozen rows), `execution_status.json`
- knowledge: `decision.md`, `synthesis.md`
