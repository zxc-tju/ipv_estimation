#!/usr/bin/env python3
"""RQ019 监督方独立复核。

不使用执行方脚本，从原始数据与原始竞赛日志重建，重点补执行方缺的那一项：
对「分布中部常规反应约 2 倍」的组间中位数差做 case 层 bootstrap 推断。

产出 rq019_supervisor_verification.json。
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent / "rq019_supervisor_verification.json"

RQ017 = REPO / "data/derived/rq017_onsite_gate/l1_v1"
M2 = REPO / ".codex-fleet/rq021-contemporaneous-envelope/work/E1/onsite_scoring_dryrun.parquet"
BASE = REPO / "data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi"
ANCHORS = BASE / "onsite_m3_av_anchors_multi_allvalid.parquet"
TSER = BASE / "onsite_ipv_timeseries_multi_allvalid.parquet"
SESSIONS_ROOT = REPO / "data/onsite_competition/all_teams_dataset/teams"

SCRIPTED = "online_first_conflict_nearest_timing_eligible_prefer_scripted_from_vehicle"
WINDOW_MS = 3000  # 固定 3 秒窗口，与执行方主组一致
ACCEL_ABS_CAP = 10.0  # 超过此值判为非物理，置缺失（执行方同口径）

result: dict = {}


def build_exposure() -> pd.DataFrame:
    files = sorted(glob.glob(str(RQ017 / "**/*.parquet"), recursive=True))
    g1 = pd.concat(
        [pd.read_parquet(f, columns=["product_row_key", "status", "ipv_log"]) for f in files]
    )
    g2 = pd.read_parquet(
        M2, columns=["product_row_key", "lo_90", "hi_90", "width_90", "mechanism2_gate_ok"]
    )
    d = g1[g1.status == "OK"].merge(
        g2[g2.mechanism2_gate_ok], on="product_row_key", how="inner"
    )
    kv = d.product_row_key.str.extract(
        r"case_key=([^|]+)\|anchor_frame_index=(\d+)\|perspective=([^|]+)"
    )
    d["case_key"], d["anchor_frame_index"], d["perspective"] = kv[0], kv[1].astype(int), kv[2]
    d["band"] = np.where(
        d.ipv_log < d.lo_90, "lower", np.where(d.ipv_log > d.hi_90, "upper", "inside")
    )
    result["two_gate_rows"] = int(len(d))
    result["band_counts"] = {k: int(v) for k, v in d.band.value_counts().items()}

    anc = pd.read_parquet(
        ANCHORS,
        columns=[
            "case_key",
            "anchor_frame_index",
            "perspective",
            "session_id",
            "counterpart_key_agent",
            "counterpart_selection",
        ],
    )
    d = d.merge(anc, on=["case_key", "anchor_frame_index", "perspective"], how="left")

    ts = pd.read_parquet(TSER, columns=["case_key", "frame_index", "timestamp_ms"])
    ts = ts.rename(columns={"frame_index": "anchor_frame_index", "timestamp_ms": "anchor_ts_ms"})
    d = d.merge(ts.drop_duplicates(["case_key", "anchor_frame_index"]),
                on=["case_key", "anchor_frame_index"], how="left")
    d["anchor_ts_ms"] = pd.to_numeric(d.anchor_ts_ms, errors="coerce")
    d["is_scripted"] = d.counterpart_selection == SCRIPTED
    result["scripted_rows"] = int(d.is_scripted.sum())
    result["anchor_ts_missing"] = int(d.anchor_ts_ms.isna().sum())
    return d


def load_session_series(session_id: str, wanted_ids: set) -> dict:
    """流式解析该 session 的原始日志，返回 {vehicle_id: DataFrame}。"""
    hits = list(SESSIONS_ROOT.glob(f"*/*/sessions/{session_id}/simulation_trajectory.log"))
    if not hits:
        return {}
    rows = defaultdict(list)
    with open(hits[0], "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            val = rec.get("value") or {}
            for r in (val.get("value") or []) + (val.get("obstacles") or []):
                vid = r.get("id")
                if vid not in wanted_ids:
                    continue
                ts, sp, ac, ca = (
                    r.get("globalTimeStamp"), r.get("speed"), r.get("acceleration"), r.get("courseAngle")
                )
                if ts is None or sp is None or ac is None or ca is None:
                    continue
                rows[vid].append((int(ts), float(sp), float(ac), float(ca)))
    out = {}
    for vid, rs in rows.items():
        df = pd.DataFrame(rs, columns=["ts", "speed_kmh", "accel", "course_deg"])
        df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
        df.loc[df.accel.abs() > ACCEL_ABS_CAP, "accel"] = np.nan
        out[vid] = df
    return out


def compute_outcomes(d: pd.DataFrame) -> pd.DataFrame:
    """对每个锚点，取锚点后 3 秒窗口内对手车的反应量。"""
    recs = []
    for sid, grp in d.groupby("session_id"):
        series = load_session_series(str(sid), set(grp.counterpart_key_agent.dropna().astype(int)))
        if not series:
            continue
        for row in grp.itertuples():
            vid = row.counterpart_key_agent
            if pd.isna(vid) or pd.isna(row.anchor_ts_ms):
                continue
            s = series.get(int(vid))
            if s is None or not len(s):
                continue
            t0 = float(row.anchor_ts_ms)
            w = s[(s.ts > t0) & (s.ts <= t0 + WINDOW_MS)]
            if len(w) < 2:
                continue
            acc = w.accel.dropna()
            # 航向：先 unwrap 再算变化，用真实时间差
            ang = np.unwrap(np.deg2rad(w.course_deg.values))
            dt = np.diff(w.ts.values) / 1000.0
            valid = dt > 0
            yaw = np.abs(np.rad2deg(np.diff(ang))[valid] / dt[valid]) if valid.any() else np.array([])
            # 锚点速度取窗口前最后一条
            prev = s[s.ts <= t0]
            anchor_speed = float(prev.speed_kmh.iloc[-1]) if len(prev) else np.nan
            recs.append(
                dict(
                    case_key=row.case_key,
                    band=row.band,
                    is_scripted=row.is_scripted,
                    speed_range_kmh=float(w.speed_kmh.max() - w.speed_kmh.min()),
                    anchor_speed_drop_kmh=(anchor_speed - float(w.speed_kmh.min()))
                    if not np.isnan(anchor_speed) else np.nan,
                    max_abs_yaw_rate_dps=float(yaw.max()) if len(yaw) else np.nan,
                    total_heading_change_deg=float(np.abs(np.rad2deg(ang[-1] - ang[0]))),
                    min_accel=float(acc.min()) if len(acc) else np.nan,
                    brake_share_2=float((acc < -2).mean()) if len(acc) else np.nan,
                    brake_share_3=float((acc < -3).mean()) if len(acc) else np.nan,
                )
            )
    return pd.DataFrame(recs)


def case_bootstrap_median_diff(df: pd.DataFrame, col: str, n_boot: int = 1000) -> dict:
    """对 lower vs inside 的中位数差做 case 层 bootstrap——执行方缺的正是这一项。"""
    sub = df[df.band.isin(["lower", "inside"])].dropna(subset=[col])
    lo, ins = sub[sub.band == "lower"][col], sub[sub.band == "inside"][col]
    if not len(lo) or not len(ins):
        return {"error": "empty group"}
    obs = float(lo.median() - ins.median())
    groups = {c: g for c, g in sub.groupby("case_key")}
    cases = np.array(list(groups))
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(cases, len(cases), replace=True)
        bs = pd.concat([groups[c] for c in pick])
        a, b = bs[bs.band == "lower"][col], bs[bs.band == "inside"][col]
        if len(a) and len(b):
            diffs.append(a.median() - b.median())
    lo95, hi95 = np.percentile(diffs, [2.5, 97.5])
    return {
        "median_lower": float(lo.median()),
        "median_inside": float(ins.median()),
        "observed_diff": obs,
        "ratio_lower_over_inside": float(lo.median() / ins.median()) if ins.median() else None,
        "case_bootstrap_ci95": [float(lo95), float(hi95)],
        "excludes_zero": bool(not (lo95 < 0 < hi95)),
        "n_lower": int(len(lo)),
        "n_inside": int(len(ins)),
        "n_cases": int(len(cases)),
        "n_boot": len(diffs),
    }


def main() -> None:
    d = build_exposure()
    out = compute_outcomes(d)
    result["outcome_rows"] = int(len(out))
    ns = out[~out.is_scripted]
    result["non_scripted_rows"] = int(len(ns))
    result["non_scripted_band_counts"] = {k: int(v) for k, v in ns.band.value_counts().items()}
    result["median_diff_case_bootstrap"] = {
        col: case_bootstrap_median_diff(ns, col)
        for col in (
            "speed_range_kmh",
            "anchor_speed_drop_kmh",
            "max_abs_yaw_rate_dps",
            "total_heading_change_deg",
            "min_accel",
            "brake_share_2",
            "brake_share_3",
        )
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
