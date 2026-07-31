#!/usr/bin/env python3
"""RQ014 后继诊断探针 v3 — 只读、rating-free。

v2 已确立：
  · 域内 M3 中心 SD=0.2267 vs 目标 SD=0.4301，corr=+0.504（中心并未塌缩；
    WOD 上的 sd≈0.06 属迁移失效）
  · 目标 IPV 有 42.1% 精确为 0，最近网格点 k=0 占 68.0%
  · 零值行区间宽度 0.248 vs 非零行 0.881（非零 = 更不确定）

v3 回答决定性问题：把"是否回落到 0"这道坎去掉之后，M3 还剩多少本事？
  H1 hurdle 分解：零/非零子集上分别看中心 SD、corr、R²
  H2 坎本身的可预测性：center / width 预测 zero 指示量的 AUC
  H3 方差分解：仅靠 zero/nonzero 二分能解释目标方差的多少（eta²）
  H4 零值率的结构性：按数据集/perspective/弃权状态分层

用法（仓库根）:
    python3 reports/knowledge/RQ014_wod_e2e_rating_recovery/reviews/m3_center_dispersion_probe.py
不写文件、不接触任何评分字段。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
PRED = (ROOT / "data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope"
        / "RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/04_calibration/predictions")
RATING_TOKENS = ("rating", "preference_score", "rater", "score_human")
WOD_FROZEN_CENTER_SD = 0.06
GRID = [k * math.pi / 8 for k in (-3, -2, -1, 0, 1, 2, 3)]
ZERO_EPS = 1e-6


def load(tier: str, fold: str) -> pd.DataFrame:
    p = PRED / f"tier={tier}" / f"fold={fold}" / "predictions.parquet"
    if not p.is_file():
        sys.exit(f"MISSING: {p}")
    df = pd.read_parquet(p)
    bad = [c for c in df.columns if any(t in c.lower() for t in RATING_TOKENS)]
    if bad:
        sys.exit(f"FAIL-CLOSED: rating-like columns: {bad}")
    return df


def desc(s: pd.Series, label: str) -> None:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        print(f"  {label}: EMPTY"); return
    q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    print(f"  {label}: n={len(s):,} mean={s.mean():+.4f} sd={s.std():.4f} "
          f"iqr={q[0.75]-q[0.25]:.4f} p05={q[0.05]:+.4f} p50={q[0.5]:+.4f} "
          f"p95={q[0.95]:+.4f}")


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """秩法 AUC：label 为 1 的样本 score 更高的概率。"""
    ok = np.isfinite(score) & np.isfinite(label)
    s, y = score[ok], label[ok].astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(s).rank(method="average").to_numpy()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main() -> None:
    df = load("M3", "test")
    lvl = sorted(df["nominal"].unique())[0]
    d = df[df["nominal"] == lvl].copy()
    d["center"] = (d["lo_cal"] + d["hi_cal"]) / 2.0
    d["center"] = d["center"].fillna((d["q_lo"] + d["q_hi"]) / 2.0)
    d["y"] = pd.to_numeric(d["y"], errors="coerce")
    d["is_zero"] = d["y"].abs() < ZERO_EPS
    d["abs_y"] = d["y"].abs()
    print(f"rows(nominal={lvl:.2f})={len(d):,}  zero_rate={d['is_zero'].mean():.4%}")

    # ---------- H1 hurdle 分解 ----------
    print("\n=== H1 零/非零子集上的 M3 表现 ===")
    for name, sub in (("ALL", d), ("ZERO (y==0)", d[d["is_zero"]]),
                      ("NONZERO (|y|>1e-6)", d[~d["is_zero"]])):
        sub = sub.dropna(subset=["center", "y"])
        if len(sub) < 100:
            print(f"\n[{name}] n={len(sub)} 太少"); continue
        sd_c, sd_y = sub["center"].std(), sub["y"].std()
        r = sub["center"].corr(sub["y"])
        print(f"\n[{name}] n={len(sub):,}")
        desc(sub["center"], "center")
        desc(sub["y"], "y     ")
        print(f"  SD(center)/SD(y) = {sd_c/sd_y:.4f}   corr = {r:+.4f}   R^2 = {r**2:.4f}")
    print("\n判读: 若 NONZERO 子集上 corr/R^2 相对 ALL 大幅下降，"
          "说明 M3 主要学到的是'会不会回落到 0'这道坎，\n"
          "      而不是社会性偏好的连续取值。")

    # ---------- H2 坎的可预测性 ----------
    print("\n=== H2 M3 预测'是否回落到 0'的能力（AUC，1=zero）===")
    lab = d["is_zero"].to_numpy()
    for col, sign, note in (("center", -1, "|center| 越小越像 zero"),
                            ("width", -1, "区间越窄越像 zero")):
        if col not in d.columns:
            continue
        score = (-d[col].abs().to_numpy() if col == "center"
                 else sign * d[col].to_numpy())
        a = auc(score, lab)
        print(f"  {col:8s}: AUC={a:.4f}   ({note})")
    print("  参考: 0.5=无预测力, >0.8=强预测力")

    # ---------- H3 方差分解 ----------
    print("\n=== H3 仅靠 zero/nonzero 二分解释的目标方差 ===")
    y = d["y"].dropna()
    z = d.loc[y.index, "is_zero"]
    grand = y.var(ddof=0)
    between = sum(len(g) * (g.mean() - y.mean()) ** 2 for _, g in y.groupby(z)) / len(y)
    print(f"  var(y)={grand:.6f}  between(zero/nonzero)={between:.6f}  "
          f"eta^2={between/grand:.4f}")
    print(f"  |y| 的 zero/nonzero eta^2 = ", end="")
    ay = d["abs_y"].dropna(); az = d.loc[ay.index, "is_zero"]
    b2 = sum(len(g) * (g.mean() - ay.mean()) ** 2 for _, g in ay.groupby(az)) / len(ay)
    print(f"{b2/ay.var(ddof=0):.4f}   ← |IPV| 有多少只是'零/非零'指示量")

    # ---------- H4 零值率的结构性 ----------
    print("\n=== H4 零值率分层 ===")
    for g in ("source_dataset", "perspective", "abstain"):
        if g not in d.columns:
            continue
        t = (d.groupby(g, observed=True)
               .agg(n=("y", "size"), zero_rate=("is_zero", "mean"),
                    width_mean=("width", "mean"), abs_y_mean=("abs_y", "mean")))
        print(f"\n--- by {g} ---")
        print(t.to_string(float_format=lambda v: f"{v:.4f}"))

    # 非零子集内 |y| 与 width 的关系（不确定性混淆的直接检查）
    nz = d[~d["is_zero"]].dropna(subset=["abs_y", "width"])
    if len(nz) > 100:
        print(f"\n=== H5 非零子集内 corr(|y|, width) = "
              f"{nz['abs_y'].corr(nz['width']):+.4f}  (n={len(nz):,}) ===")
        print("  正相关 ⇒ |IPV| 越大的行本身越不确定，"
              "'极端性↔评分'关联可能经由可辨识性通道产生。")


if __name__ == "__main__":
    main()
