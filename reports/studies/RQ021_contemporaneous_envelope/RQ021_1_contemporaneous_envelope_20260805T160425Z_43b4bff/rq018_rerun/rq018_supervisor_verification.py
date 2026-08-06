#!/usr/bin/env python3
"""RQ018 监督方独立复核。

不使用执行方脚本，从原始数据重建分析集并复算，然后追加执行方未做的尾部风险分解。

产出 rq018_supervisor_verification.json。
"""
from __future__ import annotations

import glob
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent / "rq018_supervisor_verification.json"

RQ017 = REPO / "data/derived/rq017_onsite_gate/l1_v1"
M2 = REPO / ".codex-fleet/rq021-contemporaneous-envelope/work/E1/onsite_scoring_dryrun.parquet"
ANCHORS = (
    REPO
    / "data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi"
    / "onsite_m3_av_anchors_multi_allvalid.parquet"
)
TS = (
    REPO
    / "data/derived/onsite_competition/RQ012B_event_harm/stage3plus/onsite_anchors_multi"
    / "onsite_ipv_timeseries_multi_allvalid.parquet"
)

result: dict = {}


def build_analysis_set() -> pd.DataFrame:
    """两门交集，独立于执行方脚本重建。"""
    files = sorted(glob.glob(str(RQ017 / "**/*.parquet"), recursive=True))
    gate1 = pd.concat(
        [pd.read_parquet(f, columns=["product_row_key", "status", "ipv_log"]) for f in files]
    )
    gate2 = pd.read_parquet(
        M2,
        columns=[
            "product_row_key",
            "lo_90",
            "hi_90",
            "width_90",
            "mechanism2_gate_ok",
            "context_cell",
        ],
    )
    result["gate1_rows"] = int(len(gate1))
    result["gate1_ok"] = int((gate1.status == "OK").sum())
    result["gate2_rows"] = int(len(gate2))
    result["gate2_pass"] = int(gate2.mechanism2_gate_ok.sum())

    d = gate1[gate1.status == "OK"].merge(
        gate2[gate2.mechanism2_gate_ok], on="product_row_key", how="inner"
    )
    result["two_gate_rows"] = int(len(d))

    kv = d.product_row_key.str.extract(
        r"case_key=([^|]+)\|anchor_frame_index=(\d+)\|perspective=([^|]+)"
    )
    d["case_key"] = kv[0]
    d["anchor_frame_index"] = kv[1].astype(int)
    d["perspective"] = kv[2]
    result["n_cases"] = int(d.case_key.nunique())
    return d


def add_exceedance(d: pd.DataFrame) -> pd.DataFrame:
    """非负幅度形式的上侧/下侧越界，除以区间宽度归一。"""
    d["upper"] = np.where(d.ipv_log > d.hi_90, (d.ipv_log - d.hi_90) / d.width_90, 0.0)
    d["lower"] = np.where(d.ipv_log < d.lo_90, (d.lo_90 - d.ipv_log) / d.width_90, 0.0)
    d["band"] = np.where(d.lower > 0, "lower", np.where(d.upper > 0, "upper", "inside"))
    result["band_counts"] = {k: int(v) for k, v in d.band.value_counts().items()}
    # IPV 符号与人类下界的关系：回答“下侧越界是否等同于 IPV<0”
    result["ipv_sign"] = {
        "lower_rows_with_negative_ipv": int((d[d.band == "lower"].ipv_log < 0).sum()),
        "lower_rows_total": int((d.band == "lower").sum()),
        "inside_rows_with_negative_ipv": int((d[d.band == "inside"].ipv_log < 0).sum()),
        "inside_rows_total": int((d.band == "inside").sum()),
        "lo_90_all_negative": bool((d.lo_90 < 0).all()),
        "lo_90_median": float(d.lo_90.median()),
        "lo_90_p05_p95": [float(np.percentile(d.lo_90, 5)), float(np.percentile(d.lo_90, 95))],
    }
    return d


def add_future_window(d: pd.DataFrame) -> pd.DataFrame:
    """合同窗口内的未来最小 TTC 与最小距离。TTC 仅在接近帧上定义。"""
    anc = pd.read_parquet(
        ANCHORS,
        columns=[
            "case_key",
            "anchor_frame_index",
            "perspective",
            "target_window_end_frame_index",
            "unit_composite_key",
        ],
    )
    d = d.merge(anc, on=["case_key", "anchor_frame_index", "perspective"], how="left")
    result["anchor_join_misses"] = int(d.target_window_end_frame_index.isna().sum())

    ts = pd.read_parquet(TS, columns=["case_key", "frame_index", "distance_m", "closing_rate_mps"])
    by_case = {k: v.sort_values("frame_index") for k, v in ts.groupby("case_key", sort=False)}

    min_ttc, min_dist = [], []
    for ck, start, end in zip(
        d.case_key.values, d.anchor_frame_index.values, d.target_window_end_frame_index.values
    ):
        sub = by_case.get(ck)
        if sub is None or pd.isna(end):
            min_ttc.append(np.nan)
            min_dist.append(np.nan)
            continue
        win = sub[(sub.frame_index >= start) & (sub.frame_index <= int(end))]
        if not len(win):
            min_ttc.append(np.nan)
            min_dist.append(np.nan)
            continue
        min_dist.append(float(win.distance_m.min()))
        closing = win[win.closing_rate_mps > 0]
        min_ttc.append(
            float((closing.distance_m / closing.closing_rate_mps).min()) if len(closing) else np.nan
        )
    d["future_min_ttc_s"] = min_ttc
    d["future_min_distance_m"] = min_dist
    result["ttc_missing_total"] = int(d.future_min_ttc_s.isna().sum())
    # 缺失是否与曝露相关：执行方未报告此项
    result["ttc_missing_by_band"] = {
        b: {
            "missing": int(g.future_min_ttc_s.isna().sum()),
            "total": int(len(g)),
            "rate": float(g.future_min_ttc_s.isna().mean()),
        }
        for b, g in d.groupby("band")
    }
    return d


def cluster_ols(d: pd.DataFrame) -> None:
    """复算执行方的核心系数：log1p(TTC) ~ upper + lower + context_cell FE，case 聚类。"""
    t = d.dropna(subset=["future_min_ttc_s"])
    y = np.log1p(t.future_min_ttc_s.values)
    X = pd.get_dummies(t.context_cell, prefix="cc", drop_first=True).astype(float)
    X.insert(0, "lower", t.lower.values)
    X.insert(0, "upper", t.upper.values)
    X.insert(0, "const", 1.0)
    Xv, cols = X.values, list(X.columns)
    n, k = Xv.shape
    xtxi = np.linalg.pinv(Xv.T @ Xv)
    beta = xtxi @ (Xv.T @ y)
    resid = y - Xv @ beta

    cl = pd.factorize(t.case_key.values)[0]
    meat = np.zeros((k, k))
    for c in np.unique(cl):
        m = cl == c
        s = Xv[m].T @ resid[m]
        meat += np.outer(s, s)
    g = len(np.unique(cl))
    vcov = xtxi @ meat @ xtxi * (g / (g - 1)) * ((n - 1) / (n - k))
    se = np.sqrt(np.diag(vcov))

    out = {"n": int(n), "n_clusters": int(g)}
    for name in ("lower", "upper"):
        i = cols.index(name)
        tstat = beta[i] / se[i]
        out[name] = {
            "coef": float(beta[i]),
            "se_case_cluster": float(se[i]),
            "t": float(tstat),
            "p_case_cluster": float(2 * (1 - stats.t.cdf(abs(tstat), g - 1))),
            "ci95": [float(beta[i] - 1.96 * se[i]), float(beta[i] + 1.96 * se[i])],
        }
    result["log1p_ttc_model"] = out


def tail_risk(d: pd.DataFrame) -> None:
    """执行方未做的分解：负系数究竟来自危险端增加还是安全端压缩。"""
    t = d.dropna(subset=["future_min_ttc_s"])
    qs = [1, 5, 10, 25, 50, 75]
    result["ttc_quantiles_by_band"] = {
        b: {"n": int(len(g)), "q": {str(q): float(np.percentile(g.future_min_ttc_s, q)) for q in qs}}
        for b, g in t.groupby("band")
    }
    result["ttc_median_by_band"] = {
        b: float(g.future_min_ttc_s.median()) for b, g in t.groupby("band")
    }
    result["dangerous_frame_share"] = {}
    for thr in (1.0, 1.5, 2.0, 3.0, 5.0):
        result["dangerous_frame_share"][f"ttc_lt_{thr}"] = {
            b: {
                "n": int((g.future_min_ttc_s < thr).sum()),
                "total": int(len(g)),
                "share": float((g.future_min_ttc_s < thr).mean()),
            }
            for b, g in t.groupby("band")
        }

    # case 层 bootstrap：下侧 vs 区间内的危险帧占比差
    sub = t[t.band.isin(["lower", "inside"])].copy()
    cases = sub.case_key.unique()
    rng = np.random.default_rng(0)
    boot = {}
    for thr in (2.0, 3.0):
        sub["flag"] = (sub.future_min_ttc_s < thr).astype(float)
        obs = sub[sub.band == "lower"].flag.mean() - sub[sub.band == "inside"].flag.mean()
        groups = {c: g for c, g in sub.groupby("case_key")}
        diffs = []
        for _ in range(1000):
            pick = rng.choice(cases, len(cases), replace=True)
            bs = pd.concat([groups[c] for c in pick])
            a, b2 = bs[bs.band == "lower"].flag, bs[bs.band == "inside"].flag
            if len(a) and len(b2):
                diffs.append(a.mean() - b2.mean())
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        boot[f"ttc_lt_{thr}"] = {
            "observed_diff_lower_minus_inside": float(obs),
            "case_bootstrap_ci95": [float(lo), float(hi)],
            "excludes_zero": bool(not (lo < 0 < hi)),
            "n_boot": len(diffs),
        }
    # RQ021 comparison supplement: preserve the accepted 2.0/3.0 draws above,
    # then compute the two smaller pre-specified dangerous thresholds identically.
    for thr in (1.0, 1.5):
        sub["flag"] = (sub.future_min_ttc_s < thr).astype(float)
        obs = sub[sub.band == "lower"].flag.mean() - sub[sub.band == "inside"].flag.mean()
        groups = {c: g for c, g in sub.groupby("case_key")}
        diffs = []
        for _ in range(1000):
            pick = rng.choice(cases, len(cases), replace=True)
            bs = pd.concat([groups[c] for c in pick])
            a, b2 = bs[bs.band == "lower"].flag, bs[bs.band == "inside"].flag
            if len(a) and len(b2):
                diffs.append(a.mean() - b2.mean())
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        boot[f"ttc_lt_{thr}"] = {
            "observed_diff_lower_minus_inside": float(obs),
            "case_bootstrap_ci95": [float(lo), float(hi)],
            "excludes_zero": bool(not (lo < 0 < hi)),
            "n_boot": len(diffs),
            "supplementary_rq021_comparison": True,
        }
    result["dangerous_share_bootstrap"] = boot


def main() -> None:
    d = build_analysis_set()
    d = add_exceedance(d)
    d = add_future_window(d)
    cluster_ols(d)
    tail_risk(d)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
