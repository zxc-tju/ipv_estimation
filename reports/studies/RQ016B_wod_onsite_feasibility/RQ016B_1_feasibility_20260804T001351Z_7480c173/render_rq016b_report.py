#!/usr/bin/env python3
"""Render the RQ016B-F1 feasibility report from audit_evidence.json."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / ".codex-fleet/rq016b-wod-onsite-feasibility/work/F1"
REPORT = ROOT / ".codex-fleet/rq016b-wod-onsite-feasibility/board/reports/RQ016B_1_feasibility.md"


def load() -> dict:
    return json.loads((WORK / "audit_evidence.json").read_text(encoding="utf-8"))


def evidence_cell(row: dict) -> str:
    if row["status"] == "AVAILABLE":
        ev = row["evidence"]
        return f"`OS_ALLVALID_M3::{ev['column']}` {ev['nonnull']}/{ev['total']} non-null"
    return "No local WOD trajectory/context input; see Q1 WOD source table."


def feature_table(e: dict) -> str:
    wod = {r["feature"]: r for r in e["feature_matrix"]["WOD"]}
    onsite = {r["feature"]: r for r in e["feature_matrix"]["OnSite"]}
    lines = [
        "| Feature | Kind | WOD | WOD evidence or reason | OnSite | OnSite evidence |",
        "|---|---|---|---|---|---|",
    ]
    for feature in [r["feature"] for r in e["feature_matrix"]["OnSite"]]:
        wr = wod[feature]
        orow = onsite[feature]
        onote = ""
        if orow.get("note"):
            onote = " Note: heuristic OnSite mapping, not InterHub audited source label."
        ev = orow["evidence"]
        onsite_ev = f"`OS_ALLVALID_M3::{ev['column']}` {ev['nonnull']}/{ev['total']} non-null.{onote}"
        lines.append(
            f"| `{feature}` | {orow['kind']} | {wr['status']} | "
            f"{wr['missing_reason']} | {orow['status']} | {onsite_ev} |"
        )
    return "\n".join(lines)


def lodo_table(e: dict) -> str:
    rows = e["lodo"]["m2_90_rows"]
    lines = [
        "| heldout_source | coverage column | n / total_n | abstained_n / total_n | fit_sources |",
        "|---|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['heldout_source']}` | {r['coverage']:.12f} | "
            f"{r['n']} / {r['total_n']} | {r['abstained_n']} / {r['total_n']} | "
            f"`{r['fit_sources']}` |"
        )
    return "\n".join(lines)


def q1_wod(e: dict) -> str:
    gf = e["gate_facts"]["wod_k2"]
    scope = e["scope"]["wod"]
    receipt = e["wod_receipt"]
    return f"""
### WOD

Result: the current local WOD package is not enough to rerun the seven-candidate MSE path. It is a sanitized audit projection, not a trajectory/materializer input bundle.

| Required input | Status | File, columns, count |
|---|---|---|
| Pair trajectories | MISSING | `WOD_PROJ` has only `segment_key`, `candidate_index`, `ego_ipv`, `ego_ipv_error`; all four columns are 906/906 non-null. No x/y/v/heading/history columns are present. |
| Sampling rate and timestep | MISSING | `WOD_PROJ` has no timestamp or frame column. `WOD_K2` has `frame_id` 0/906 non-null and `case_id` 0/906 non-null; filter all rows, source columns `frame_id`, `case_id`. |
| Usable history window | MISSING | No per-frame history rows. `WOD_K2` has 906 rows, `measurement_role=candidate` for 906/906 rows; source column `measurement_role`. |
| Map or reference line | MISSING | No map/lane/reference-line column in `WOD_PROJ`; local RQ010 placeholder README says no WOD dataset payload was stored. |
| Frozen grid/config fields | PARTIAL | `WOD_K2` has `candidate_grid_id=legacy7_pi_over_8` for 906/906 rows and `K=7` for 906/906 rows, but MSE/log-weight fields are empty by supervisor fact and no materializer input is present. |

`rq015a_full479_projected` contents: `WOD_PROJ` row count is {scope['projection_rows']}; `segment_key` 906/906 non-null and `candidate_index` 906/906 non-null. `WOD_K2` row count is {gf['row_count']}; `canonical_key` and `product_row_key` are each 906/906 non-null and 906 unique, while `case_id` and `frame_id` are 0/906 non-null. The source file for these counts is `data/derived/rq015k_logdomain_gate/l1_v1/artifact_id=wod_rq010b_full479_audited`, columns `canonical_key`, `product_row_key`, `case_id`, `frame_id`.

Boundary: the sanitization receipt says the projected local CSV has forbidden-column scan `match_count=0`, `matched_columns=[]`; it also records source-side forbidden columns `rating`, `selected_track_avg_score`, and `track_avg_score`, which were dropped before local transfer. I did not read the source WOD table. The WOD K2 schema contains `log_score_0` through `log_score_6`; the audit script treated these as mechanical `score` matches and did not read their content.
"""


def q1_onsite(e: dict) -> str:
    dense = e["scope"]["onsite"]["dense"]
    total_deltas = dense["physical_rows"] - dense["case_key_nunique"]
    top_100 = dense["timestamp_ms_delta_counts_top10"].get("100", 0)
    single = e["scope"]["onsite"]["single_coverage"]
    allv = e["scope"]["onsite"]["allvalid_coverage"]
    channel = e["onsite_channel_values"]["onsite_channel_exact_hw10"]
    return f"""
### OnSite

Result: OnSite has enough local trajectory/context evidence to define a rebuild path, but not enough to claim a true map/reference-line contract. A materializer still has to be written before seven-candidate MSE can be produced.

| Required input | Status | File, columns, count |
|---|---|---|
| Pair trajectories | AVAILABLE | `ONSITE_DENSE` has `ego_key_agent`, `counterpart_key_agent`, ego/counterpart x/y/vx/vy/heading, `distance_m`, `closing_rate_mps`, `relative_speed_mps`: each listed raw column is 70,317/70,317 non-null. |
| Sampling rate and timestep | AVAILABLE as observed timestamps | `ONSITE_DENSE` has `timestamp_ms` and `frame_index` 70,317/70,317 non-null. The most frequent adjacent within-case timestamp gap is 100 ms: {top_100}/{total_deltas} adjacent gaps, filter rows sorted by `case_key,timestamp_ms,frame_index`. |
| Usable history window | AVAILABLE for anchor scopes | `ONSITE_ALLVALID_M3` has `history_row_count` 67,861/67,861 non-null. All-valid provenance records `feature_history_window=10`, `target_history_window=4`, `min_observation=4`, `target_final_offset=6`. |
| Map or reference line | PARTIAL | `ONSITE_DENSE` has 0 true map/lane/route/reference-line columns by schema scan. Existing OnSite build used observed-trajectory fallback; all raw ego/counterpart trajectory columns above are populated. |
| Frozen grid/config fields | PARTIAL | `ONSITE_CHANNEL_EXACT_HW10` has `candidate_grid=legacy_seven_point_-3pi/8_to_3pi/8` for 278/285 rows, `current_history_window=10` for 278/285 rows, and `min_observation=4` for 278/285 rows. But its `target_history_window=10`, not the frozen config's future target history window 4, and it does not contain per-candidate MSE/log weights. |

Single-anchor coverage source `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/02_process/03_event_deviation/onsite_ipv/coverage.json`: `units_requested={single['units_requested']}`, `units_with_anchors={single['units_with_anchors']}`, `total_av_anchors={single['total_av_anchors']}`, `max_anchors_per_unit={single['max_anchors_per_unit']}`, `ipv_frame_rows={single['ipv_frame_rows']}`.

All-valid coverage source `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012B_2_harm_association_20260627T095847+0800_8454ad93/02_process/03_event_deviation/hpc_onsite_ipv/process_allvalid_processpool_amd/coverage.json`: `total_av_anchors={allv['total_av_anchors']}`, `valid_anchor_candidate_total={allv['valid_anchor_candidate_total']}`, `valid_anchor_candidates_after_cap={allv['valid_anchor_candidates_after_cap']}`, `anchors_excluded_by_cap={allv['anchors_excluded_by_cap']}`.
"""


def q2_section(e: dict) -> str:
    return f"""
## Q2 - M2 context features

Definitions: M2 is the RQ009 context feature set with 22 numeric context columns and 7 categorical context columns. The 12 support-gate distance columns in this audit are the RQ009 distance list after excluding the three `counterpart_ipv_*` columns. The machine-readable full table is `work/F1/feasibility_matrix.json`.

Aliases used in the table:

| Alias | Path |
|---|---|
| `WOD_PROJ` | `data/derived/wod_e2e/rq015a_full479_projected/rq010b_wod_full479_audited_candidate_ipv_projected.csv` |
| `WOD_K2` | `data/derived/rq015k_logdomain_gate/l1_v1/artifact_id=wod_rq010b_full479_audited` |
| `OS_ALLVALID_M3` | `data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi/onsite_m3_av_anchors_multi_allvalid.parquet` |

{feature_table(e)}

Interpretation:

WOD has 29/29 M2 features marked MISSING in the current local package. This is not a statement about what the original Waymo data could contain; it is a statement about the local sanitized artifacts inspected here.

OnSite has 29/29 M2 features marked AVAILABLE for the all-valid anchor scope. Two numeric features are sparse by definition or implementation: `closing_ttc_anchor` is 31,740/67,861 non-null and `apet_online_proxy` is 5,364/67,861 non-null in `OS_ALLVALID_M3`. The four InterHub audited-style labels `geometry_path_category`, `geometry_path_relation`, `turn_pair_label`, and `priority_role` exist in OnSite, but the mapping document states they are deterministic kinematic heuristics rather than InterHub audited map/path labels. This is a validity boundary, not a missing-column defect.

The 12 support-gate distance columns are AVAILABLE on OnSite under the same evidence rows above. They are all 67,861/67,861 non-null except `closing_ttc_anchor` at 31,740/67,861 and `apet_online_proxy` at 5,364/67,861, source `OS_ALLVALID_M3`, columns as named.
"""


def q3_section(e: dict) -> str:
    scope = e["scope"]
    oscope = scope["onsite"]
    gf = e["gate_facts"]["wod_k2"]
    return f"""
## Q3 - Scale and range

### WOD

The local WOD projection has 906 rows, source `WOD_PROJ`, filter all rows. `segment_key` and `candidate_index` are each 906/906 non-null. In `WOD_K2`, filter all rows: `canonical_key` is 906/906 non-null and 906 unique; `product_row_key` is 906/906 non-null and 906 unique; `case_id` is 0/906 non-null; `frame_id` is 0/906 non-null. Therefore the local package supports a 906 candidate-row ledger count, but it does not support a case/scene/frame denominator.

`rq015a_full479_projected` contains only `segment_key`, `candidate_index`, `ego_ipv`, and `ego_ipv_error`. This is enough to audit the old scalar channel and candidate identity, not enough to rebuild per-candidate MSE or M2 context.

### OnSite

| Range option | Unit | Count | Source and filter |
|---|---:|---:|---|
| A. All aligned frames | physical frame | {oscope['range_options']['A_all_aligned_frames']['rows']} | `ONSITE_DENSE`, all rows; columns `case_key`, `frame_index`, `timestamp_ms` |
| B. All RQ009 timing-valid anchor frames, current materialized scope | anchor frame | {oscope['range_options']['B_all_rq009_timing_valid_anchor_frames_current_materialized']['rows']} | `ONSITE_ALLVALID_M3`, all rows |
| B'. Candidate count before failed-unit loss | candidate anchor | {oscope['range_options']['B_all_rq009_timing_valid_candidates_before_failed_units']['rows']} | all-valid `coverage.json`, field `valid_anchor_candidate_total` |
| C. Continue one anchor per unit | anchor frame | {oscope['range_options']['C_one_anchor_per_unit']['rows']} | `ONSITE_SINGLE_M3`, all rows |

The three requested values reconcile as follows. `max_anchors_per_unit=1` and `total_av_anchors=267` are verified in single-anchor `coverage.json`, fields `max_anchors_per_unit` and `total_av_anchors`. The value 67,861 is verified as all-valid materialized anchor rows in `ONSITE_ALLVALID_M3` and as all-valid `coverage.json` field `total_av_anchors`. The reconstructed one-anchor cap exclusion is 67,594/67,861 anchor rows, computed as all-valid materialized rows 67,861 minus single-anchor rows 267; source files `ONSITE_ALLVALID_M3` and `ONSITE_SINGLE_M3`, row counts. However, the all-valid `coverage.json` field named `valid_anchor_candidate_total` is 68,420, not 67,861. I treat this as a field-name/denominator distinction: 68,420 is the candidate count before failed-unit loss; 67,861 is the currently materialized all-valid anchor table.
"""


def q4_section(e: dict) -> str:
    lodo = e["lodo"]
    heldout_sources = ", ".join(f"`{s}`" for s in lodo["heldout_sources"])
    return f"""
## Q4 - Transfer validity

### WOD

RQ009 LODO evidence source is `{lodo['path']}`, filter `tier == M2` and `alpha_label == 90`. Held-out sources are {heldout_sources}. The 90 nominal coverage column ranges from {lodo['m2_90_coverage_min']:.12f} to {lodo['m2_90_coverage_max']:.12f} under that filter.

{lodo_table(e)}

For this audit's WOD object, the exact artifact `wod_rq010b_full479_audited` is not in LODO. `waymo_train` is present, so there is source-family evidence for a Waymo training source, but there is no same-artifact evidence for the RQ010B/RQ014 WOD AV ledger inspected here.

### OnSite

OnSite is not in the LODO held-out source list. Therefore there is no same-source transfer evidence for OnSite in the accepted RQ009 LODO table. The OnSite context columns can be constructed for anchor scopes, but the envelope remains an InterHub-human envelope applied across datasets.
"""


def q5_section(e: dict) -> str:
    return """
## Q5 - Minimal feasible path and cost

### WOD

Minimal path that would produce a real WOD result:

1. Build a new WOD safe projection from the upstream RQ010B/RQ014-side source that includes only non-protected inputs: AV/counterpart paired trajectories, timestamps/frame indices, selected pairing identity, candidate grid ID, and map/reference-line or an approved reference fallback. Output: a sanitized trajectory/context source table with forbidden-column receipt.
2. Write a WOD materializer that maps that source table to frozen config `configs/ipv_sigma01_exact.json`: grid `legacy7_pi_over_8`, `K=7`, `sigma=0.1`, current history window 10, future target history window 4, target final offset 6. Output: per-candidate MSE/log-weight columns plus the mechanism-one status columns.
3. Build the 29 M2 context columns or a documented subset failure table. Output: WOD context matrix keyed to the same AV anchor rows.
4. Apply the rebuilt RQ016 envelope only to rows that pass mechanism one and have support-gate context. Output: WOD AV envelope decision table.

Cost: the local row count is only 906 candidate rows, but the local package lacks the actual solve input. Using the K1 InterHub pilot only as a timing reference, 906 solve units would be small after the materializer exists: K1 measured 1,120 units in 499.6056067943573 seconds on six workers, source `.codex-fleet/rq015k-fullcorpus-gate/board/reports/K1_preflight_and_plan.md`, fields reported from `k1_pilot_summary.json`; K1b measured 6.467854 units/s for P16, source `.codex-fleet/rq015k-fullcorpus-gate/board/reports/K1b_memory_pilot.md`. These timings do not price the WOD source projection or materializer work, which is the dominant cost.

Route that does not work: applying the rebuilt envelope directly to `WOD_PROJ` or `WOD_K2`. It fails at step 1 because the local files do not contain paired trajectories, timestamps, reference-line input, M2 context columns, or per-candidate MSE/log-weight values.

### OnSite

Minimal path that would produce a real OnSite result:

1. PI chooses the denominator: A all aligned frames, B all RQ009 timing-valid anchor frames, or C one anchor per unit. Output: a frozen row universe with source file and filter.
2. PI chooses the reference-line contract: continue observed-trajectory fallback or require true map/lane/reference-line. Output: a reference source rule.
3. Write the OnSite materializer for the chosen universe. Inputs: `ONSITE_DENSE` raw trajectories and `ONSITE_ALLVALID_M3` or `ONSITE_SINGLE_M3` anchors/context; frozen config `configs/ipv_sigma01_exact.json`. Output: seven-candidate MSE/log-weight columns and mechanism-one status columns.
4. Join materializer output to M2 context. Output: OnSite AV context-and-status table.
5. Apply the rebuilt RQ016 envelope to rows that pass mechanism one and the support gate. Output: OnSite AV envelope decision table.

Cost by denominator before role expansion:

| Option | Solve-unit denominator | Evidence |
|---|---:|---|
| A all aligned frames | 70,317 physical frames | `ONSITE_DENSE`, all rows |
| B current all-valid materialized anchors | 67,861 anchor frames | `ONSITE_ALLVALID_M3`, all rows |
| C one anchor per unit | 267 anchor frames | `ONSITE_SINGLE_M3`, all rows |

If the implementation follows the previous four-role expansion used by K2, option A maps to 281,268 role rows, source `data/derived/rq015k_logdomain_gate/l1_v1/artifact_id=onsite_dense_timeseries`, filter all rows, column `measurement_role`. Option B would need the commander to define whether one AV current role is enough or whether target/counterpart roles must also be generated; without that decision the denominator is not final.

Compute timing reference: the old OnSite all-valid anchor build completed 67,861 anchors in 2,522.0963592529297 seconds on the recorded HPC/processpool run, source all-valid `coverage.json` fields `total_av_anchors` and `elapsed_s`. That run did not produce seven-candidate MSE/log weights. K1/K1b exact-solve timing suggests option C is local-scale after a materializer exists; options A and B should be treated as HPC-scale because they are tens of thousands to hundreds of thousands of solve units and previous all-valid OnSite work was already run in the HPC lane.

Route that does not work: using current OnSite K2 rows as mechanism-one input. It fails because `gate_applicable=False` for 281,268/281,268 rows, `context_cell_key` is 0/281,268 non-null, and per-candidate MSE/log-weight inputs are absent for this materialization.

Which dataset should go first: OnSite. The criterion is local input readiness. OnSite has paired trajectories 70,317/70,317 non-null, all 29 M2 context columns for 67,861 all-valid anchors, and no RQ014 protected-source boundary. WOD has only a four-column sanitized projection locally and requires a new protected-source retrieval/projection step before any materializer can be defined.
"""


def self_check_section(e: dict) -> str:
    nc = e["negative_control"]
    return f"""
## Self-check

Column-level evidence: every OnSite Q2 AVAILABLE row has `file + column + non-null/total` in `feasibility_matrix.json`. WOD Q2 rows are all MISSING; the inspected WOD projection file has only four columns and the inspected K2 source lacks context values.

Negative control: rule `{nc['rule']}` on `{nc['source_file']}` column `{nc['column']}` normally gives `{nc['normal_result']}` with {nc['actual_nonnull']}/{nc['expected_total']}. I deliberately changed the expected total to {nc['disturbed_expected_total']}; output: `{nc['disturbed_output']}`.

Reproducibility: run `python3 .codex-fleet/rq016b-wod-onsite-feasibility/work/F1/rq016b_f1_audit.py` to rebuild `audit_evidence.json` and `feasibility_matrix.json`, then run `python3 .codex-fleet/rq016b-wod-onsite-feasibility/work/F1/render_rq016b_report.py` to rebuild this report. The audit script ran no estimator, no Slurm job, no model training, and wrote only this task's work/report outputs.

Boundary events: WOD source-side protected columns were encountered only inside the sanitization receipt metadata, not by opening source WOD content. WOD K2 schema field names `log_score_0..6` matched the mechanical guard; their content was not read. No RQ007 held-out content was parsed in this audit.
"""


def main() -> None:
    e = load()
    report = f"""# RQ016B-F1 WOD and OnSite feasibility audit

## Position

This work answers one narrow question: after RQ016 rebuilt the human envelope used by mechanism two, what would it cost to apply that envelope to WOD and OnSite AV data, and which parts cannot be done from current artifacts. The larger study is online validation of whether an AV's interaction tendency looks human-like. The overall pipeline has two serial gates: mechanism one asks whether the current IPV value carries candidate-discrimination information; mechanism two compares accepted values to a human reference envelope. RQ016 fixed the InterHub human envelope; this audit checks the WOD and OnSite application surface. It is read-only feasibility work: no estimator run, no Slurm submission, no materializer, no model training.

## Conclusions

WOD cannot produce a real result from the current local package. The local WOD package has 906 sanitized candidate rows and old scalar columns, but it does not have paired trajectories, timestamps, map/reference-line input, M2 context, or seven-candidate MSE/log-weight values.

OnSite is the better first target. It has local paired trajectories, all 29 M2 context columns for the 67,861 all-valid anchor table, and a documented mapping from OnSite fields to RQ009 context labels. The main gaps are materializer work, denominator choice, and the reference-line contract. The existing OnSite geometry labels are heuristic, not the InterHub audited labels.

## Decisions Needed

1. OnSite denominator: choose A all aligned frames, B all RQ009 timing-valid anchor frames, or C one anchor per unit. Without this, the result denominator is undefined.
2. OnSite reference-line contract: accept observed-trajectory fallback or require true map/lane/reference-line. Without this, the materializer input contract is not comparable to InterHub.
3. WOD source-access contract: decide whether a new protected-source projection may be prepared with trajectories and context but without rating/score/preference/human-score fields. Without this, WOD has no executable path.

{q1_wod(e)}

{q1_onsite(e)}

{q2_section(e)}

{q3_section(e)}

{q4_section(e)}

{q5_section(e)}

{self_check_section(e)}

state: WAITING_ON_COMMANDER
timestamp_utc: {e['generated_at_utc']}
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
