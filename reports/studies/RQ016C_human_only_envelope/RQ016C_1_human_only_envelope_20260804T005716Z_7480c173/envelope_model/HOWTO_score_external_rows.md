# HOWTO score external rows with the RQ016C-H2 envelope

This directory contains the fitted human-only envelope. Scoring does not refit any model.

## Required input columns

External rows must contain the same context columns used by this envelope:

- Numeric context: `elapsed_time_s, history_row_count, ego_vx_anchor, ego_vy_anchor, ego_heading_anchor, counterpart_vx_anchor, counterpart_vy_anchor, counterpart_heading_anchor, relative_dx_anchor, relative_dy_anchor, relative_distance_anchor, relative_dvx_anchor, relative_dvy_anchor, relative_speed_anchor, closing_rate_anchor, heading_difference_anchor, relative_distance_mean_wx, relative_distance_std_wx, relative_speed_mean_wx, closing_rate_mean_wx, closing_ttc_anchor, apet_online_proxy`
- Categorical context: `geometry_path_category, geometry_path_relation, turn_pair_label, priority_role`
- Support-gate columns: `geometry_path_category, priority_role`

Do not add `agent_type_pair`, `av_included`, or `source_dataset` as predictors. They are not part of the fitted context. If `target_ipv_future` is present, the scoring script also writes `support`, `not_support`, or `abstain` verdicts for each alpha layer. If `target_ipv_future` is absent, it writes intervals and the mechanism-two gate flag only.

## Command

```bash
~/.rq009_codex_fleet/venv/bin/python .codex-fleet/rq016c-human-only-envelope/work/H2/score_external_rows.py \
  --model .codex-fleet/rq016c-human-only-envelope/work/H2/envelope_model/rq016c_h2_envelope.pkl \
  --input path/to/external_rows.parquet \
  --output path/to/scored_rows.parquet
```

The input may be `.parquet` or `.csv`; the output suffix controls the output format.

## Output columns

- `mechanism2_gate_ok`: `True` when the row passes the support gate.
- `lo_80`, `hi_80`, `lo_90`, `hi_90`, `lo_95`, `hi_95`: interval bounds from the persisted global conformal radii.
- `verdict_80`, `verdict_90`, `verdict_95`: written only when `target_ipv_future` is present. Values are `support`, `not_support`, or `abstain`.

Per-cell calibration radii are stored in `conformal_radii_by_cell.json` for audit. The primary scoring path uses `conformal_radii_global.json`, matching the RQ016 split-conformal calculation.
