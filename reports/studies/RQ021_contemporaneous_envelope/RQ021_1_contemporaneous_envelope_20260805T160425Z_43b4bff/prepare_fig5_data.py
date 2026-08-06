#!/usr/bin/env python3
"""为手稿 Figure 5 准备作图数据。

复用监督方复核脚本的重建逻辑，额外产出：
  1. 逐锚点的对手方反应表（缓存，避免重复解析原始日志）
  2. 比值 lower/inside 的 case 层 bootstrap CI（四个量单位不同，共用横轴必须用比值）
  3. 汇总成一个 fig5_data.json

输出与脚本同目录。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "rq019_rerun"))
import rq019_supervisor_verification as V  # noqa: E402

REPO = V.REPO
OUT_JSON = HERE / "fig5_data.json"
OUT_CP = HERE / "fig5_counterpart_outcomes.parquet"
OUT_EGO = HERE / "fig5_ego_ttc.parquet"

RQ018_DIR = HERE / "rq018_rerun"
RQ019_DIR = HERE / "rq019_rerun"


def case_bootstrap_ratio(df: pd.DataFrame, col: str, n_boot: int = 2000) -> dict:
    """lower/inside 中位数比值的 case 层 bootstrap CI。"""
    sub = df[df.band.isin(["lower", "inside"])].dropna(subset=[col])
    lo, ins = sub[sub.band == "lower"][col], sub[sub.band == "inside"][col]
    obs = float(lo.median() / ins.median())
    groups = {c: g for c, g in sub.groupby("case_key")}
    cases = np.array(list(groups))
    rng = np.random.default_rng(0)
    ratios = []
    for _ in range(n_boot):
        pick = rng.choice(cases, len(cases), replace=True)
        bs = pd.concat([groups[c] for c in pick])
        a, b = bs[bs.band == "lower"][col], bs[bs.band == "inside"][col]
        if len(a) and len(b) and b.median() != 0:
            ratios.append(a.median() / b.median())
    lo95, hi95 = np.percentile(ratios, [2.5, 97.5])
    return {
        "median_lower": float(lo.median()),
        "median_inside": float(ins.median()),
        "ratio": obs,
        "ci95": [float(lo95), float(hi95)],
        "excludes_one": bool(not (lo95 < 1.0 < hi95)),
        "n_lower": int(len(lo)),
        "n_inside": int(len(ins)),
        "n_cases": int(len(cases)),
        "n_boot": len(ratios),
    }


def main() -> None:
    out: dict = {}

    # ---- 自车侧：逐帧 TTC（面板 a）与危险阈值占比（面板 b）----
    rq018 = json.loads((RQ018_DIR / "rq018_supervisor_verification.json").read_text())
    out["ego_ttc_quantiles"] = rq018["ttc_quantiles_by_band"]
    out["ego_danger_shares"] = rq018["dangerous_frame_share"]
    out["ego_danger_bootstrap"] = rq018["dangerous_share_bootstrap"]
    # 四个阈值全部检验过：未标星的柱表示区间含 0，而不是「未检验」
    out["ego_danger_bootstrap_all"] = rq018["dangerous_share_bootstrap"]
    out["ego_band_counts"] = rq018["band_counts"]

    # 逐帧 TTC 原值（ECDF 需要）。用 RQ018 复核脚本重建，band 口径与其一致。
    sys.path.insert(0, str(HERE / "rq018_rerun"))
    import rq018_supervisor_verification as V18  # noqa: E402

    e = V18.build_analysis_set()
    e = V18.add_exceedance(e)
    e = V18.add_future_window(e)
    ego = e.loc[e.future_min_ttc_s.notna(), ["case_key", "band", "future_min_ttc_s"]].copy()
    ego.to_parquet(OUT_EGO, index=False)
    out["ego_ttc_rows"] = int(len(ego))

    # ---- 对手方侧：逐锚点反应（面板 c）----
    d = V.build_exposure()
    cp = V.compute_outcomes(d)
    cp = cp[~cp.is_scripted].copy()  # 主结论组
    cp.to_parquet(OUT_CP, index=False)
    out["counterpart_rows_non_scripted"] = int(len(cp))
    out["counterpart_band_counts"] = {k: int(v) for k, v in cp.band.value_counts().items()}

    out["counterpart_ratio_bootstrap"] = {
        col: case_bootstrap_ratio(cp, col)
        for col in (
            "anchor_speed_drop_kmh",
            "speed_range_kmh",
            "total_heading_change_deg",
            "max_abs_yaw_rate_dps",
        )
    }

    # ---- 对手方强制动阈值占比（面板 d）----
    dist = json.loads((RQ019_DIR / "distribution_results.json").read_text())
    rows = [
        r for r in dist["alpha90_case_bootstrap_threshold_contrasts"]
        if r.get("stratum") == "non_scripted"
        and r.get("window") == "fixed3"
        and "lower" in str(r.get("comparison", ""))
    ]
    out["counterpart_braking_thresholds"] = [
        {
            "threshold_mps2": r["threshold_mps2"],
            "lower_share": r["comparison_share"],
            "lower_num": r["comparison_numerator"],
            "lower_den": r["comparison_denominator"],
            "inside_share": r["inside_share"],
            "inside_num": r["inside_numerator"],
            "inside_den": r["inside_denominator"],
            "diff": r["share_difference"],
            "ci95": r["case_bootstrap_ci_95"],
            "p_case": r["case_equal_t_p"],
        }
        for r in rows
    ]

    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_EGO} ({len(ego)} rows)")
    print(f"wrote {OUT_CP} ({len(cp)} rows)")


if __name__ == "__main__":
    main()


def add_all_ego_thresholds() -> None:
    """补算自车全部四个危险阈值的 case 层 bootstrap（原先只算了 2.0/3.0）。

    面板 b 的星号必须能区分「不显著」与「未检验」；缺一个都会误导。
    """
    ego = pd.read_parquet(OUT_EGO)
    sub = ego[ego.band.isin(["lower", "inside"])].copy()
    groups = {c: g for c, g in sub.groupby("case_key")}
    cases = np.array(list(groups))
    out = {}
    for thr in (1.0, 1.5, 2.0, 3.0):
        sub["flag"] = (sub.future_min_ttc_s < thr).astype(float)
        groups = {c: g for c, g in sub.groupby("case_key")}
        obs = sub[sub.band == "lower"].flag.mean() - sub[sub.band == "inside"].flag.mean()
        rng = np.random.default_rng(0)
        diffs = []
        for _ in range(1000):
            pick = rng.choice(cases, len(cases), replace=True)
            bs = pd.concat([groups[c] for c in pick])
            a, b = bs[bs.band == "lower"].flag, bs[bs.band == "inside"].flag
            if len(a) and len(b):
                diffs.append(a.mean() - b.mean())
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        out[f"ttc_lt_{thr}"] = {
            "observed_diff_lower_minus_inside": float(obs),
            "case_bootstrap_ci95": [float(lo), float(hi)],
            "excludes_zero": bool(not (lo < 0 < hi)),
            "n_boot": len(diffs),
        }
        print(f"  TTC<{thr}: diff={obs:+.4f} CI=[{lo:+.4f}, {hi:+.4f}] excl0={out[f'ttc_lt_{thr}']['excludes_zero']}")
    d = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    d["ego_danger_bootstrap_all"] = out
    OUT_JSON.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"updated {OUT_JSON}")
