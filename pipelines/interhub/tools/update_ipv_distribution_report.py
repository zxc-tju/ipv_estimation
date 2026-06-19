from __future__ import annotations

import base64
import html
import json
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from scipy import stats


ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = (
    ROOT
    / "reports"
    / "interhub"
    / "ipv_estimation_results"
    / "subsets_for_yiru"
    / "ipv_distribution_report"
)
FIG_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"
SUMMARY_PATH = REPORT_DIR / "report_summary.json"
HTML_PATH = REPORT_DIR / "ipv_distribution_report.html"
MATCHED_CSV = (
    ROOT
    / "reports"
    / "interhub"
    / "ipv_estimation_results"
    / "subsets_for_yiru"
    / "selected_interactive_segments_equalized_with_ipv_combined.csv"
)
PKL_ROOT = ROOT / "data" / "interhub" / "raw" / "subsets_for_yiru" / "pkl"

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}
NEUTRAL = {
    "light": "#E2E5EA",
    "base": "#C5CAD3",
    "mid": "#7A828F",
    "dark": "#464C55",
}
FAMILIES = {
    "blue": {
        "xlight": "#EAF1FE",
        "light": "#CEDFFE",
        "base": "#A3BEFA",
        "mid": "#5477C4",
        "dark": "#2E4780",
    },
    "orange": {
        "xlight": "#FFEDDE",
        "light": "#FFBDA1",
        "base": "#F0986E",
        "mid": "#CC6F47",
        "dark": "#804126",
    },
    "olive": {
        "xlight": "#D8ECBD",
        "light": "#BEEB96",
        "base": "#A3D576",
        "mid": "#71B436",
        "dark": "#386411",
    },
}
DATASET_LABELS = {
    "av2_motion_forecasting": "AV2",
    "waymo_train": "Waymo",
    "nuplan_train": "nuPlan",
}
DATASET_ORDER = ["AV2", "Waymo", "nuPlan"]
DATASET_PALETTE = {
    "AV2": FAMILIES["blue"]["base"],
    "Waymo": FAMILIES["orange"]["base"],
    "nuPlan": FAMILIES["olive"]["base"],
}
DATASET_EDGE = {
    "AV2": FAMILIES["blue"]["dark"],
    "Waymo": FAMILIES["orange"]["dark"],
    "nuPlan": FAMILIES["olive"]["dark"],
}


def use_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": "#FFFFFF",
            "savefig.edgecolor": "none",
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica Neue",
                "Microsoft YaHei",
                "Noto Sans CJK SC",
                "DejaVu Sans",
                "sans-serif",
            ],
            "font.monospace": ["Consolas", "DejaVu Sans Mono", "monospace"],
            "patch.linewidth": 1.0,
        },
    )


def savefig(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG_DIR / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def ci_mean(values: pd.Series) -> tuple[float, float, float]:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    mean = float(arr.mean())
    if len(arr) == 1:
        return mean, mean, mean
    sem = float(stats.sem(arr))
    lo, hi = stats.t.interval(0.95, len(arr) - 1, loc=mean, scale=sem)
    return mean, float(lo), float(hi)


def cliff_delta(first: pd.Series, second: pd.Series) -> float:
    """Rank-based effect size; positive means first tends to be larger."""
    a = pd.to_numeric(first, errors="coerce").dropna().to_numpy(dtype=float)
    b = pd.to_numeric(second, errors="coerce").dropna().to_numpy(dtype=float)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    u_stat, _ = stats.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic")
    return float(2 * u_stat / (len(a) * len(b)) - 1)


def pct(value: float) -> str:
    return f"{value:.1f}%"


def num(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def p_text(p_value: float) -> str:
    if pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "<0.001"
    return f"{p_value:.3f}"


def img_data(name: str) -> str:
    data = (FIG_DIR / name).read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def figure_card(
    number: str, title: str, image_name: str, caption: str, *, wide: bool = False
) -> str:
    cls = "figure-card wide" if wide else "figure-card"
    return f"""
    <article class="{cls}">
      <h3>{html.escape(number)}. {html.escape(title)}</h3>
      <img src="{img_data(image_name)}" alt="{html.escape(title)}">
      <p class="caption">{html.escape(caption)}</p>
    </article>
    """


def table_html(csv_name: str) -> str:
    frame = pd.read_csv(TABLE_DIR / csv_name)
    for col in frame.columns:
        if pd.api.types.is_float_dtype(frame[col]):
            frame[col] = frame[col].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
        elif pd.api.types.is_integer_dtype(frame[col]):
            frame[col] = frame[col].map(lambda v: "" if pd.isna(v) else f"{int(v):,}")
    return frame.to_html(index=False, classes="data-table", border=0, escape=True)


def metric_definitions() -> pd.DataFrame:
    rows = [
        {
            "category": "Data coverage",
            "english_metric": "case",
            "chinese_metric": "交互案例",
            "grain": "case",
            "meaning": "CSV 中的一行交互片段，也是报告的基本分析单元。",
            "calculation": "由 folder + scenario_idx + key_agents + track_id 匹配到一个 pkl 事件和一个输出目录。",
            "notes": "multi-agent 事件中仍只分析 CSV key_agents 指定的两个 agent。",
        },
        {
            "category": "Data coverage",
            "english_metric": "completed case",
            "chinese_metric": "已完成案例",
            "grain": "case",
            "meaning": "已有可用 IPV 结果的交互案例。",
            "calculation": "ipv_result_status == 'ok' 的行数。",
            "notes": "报告中 IPV 分布统计默认只使用 completed case。",
        },
        {
            "category": "Data coverage",
            "english_metric": "missing case",
            "chinese_metric": "缺失案例",
            "grain": "case",
            "meaning": "CSV 中存在但当前 cases.zip 中没有可用 IPV 结果的案例。",
            "calculation": "ipv_result_status != 'ok' 的行数。",
            "notes": "当前缺失均来自 nuPlan，因此跨数据集结论要等 nuPlan 补齐后复核。",
        },
        {
            "category": "Data coverage",
            "english_metric": "completion_pct",
            "chinese_metric": "完成率",
            "grain": "dataset",
            "meaning": "某数据集已完成案例占该数据集全部 CSV 案例的比例。",
            "calculation": "completed_cases / total_cases * 100。",
            "notes": "用于判断某个数据集的分布是否可能受缺失结果影响。",
        },
        {
            "category": "Data coverage",
            "english_metric": "agent_values",
            "chinese_metric": "key-agent IPV 值数量",
            "grain": "agent",
            "meaning": "进入 agent-level 分布统计的 IPV 数值个数。",
            "calculation": "2 * completed_cases；每个 case 有 key agent 1 和 key agent 2 两个值。",
            "notes": "如果某个 agent 的结果缺失，则该 agent 值会在统计中剔除。",
        },
        {
            "category": "Key-agent IPV",
            "english_metric": "key_agents",
            "chinese_metric": "关键交互车辆 ID",
            "grain": "case",
            "meaning": "CSV 中指定的两个关键 agent，是 IPV 填写和解释顺序的来源。",
            "calculation": "CSV key_agents 列按分号分隔；第 1 个 ID 对应 key agent 1，第 2 个 ID 对应 key agent 2。",
            "notes": "不要把 agent 1/2 直接解释成固定交通角色，除非再结合优先权、路径或车辆类型。",
        },
        {
            "category": "Key-agent IPV",
            "english_metric": "ipv_key_agent_1_mean / key agent 1 mean IPV",
            "chinese_metric": "关键 agent 1 的平均 IPV",
            "grain": "agent within case",
            "meaning": "CSV key_agents 中第 1 个 agent 在有效估计步上的有符号 IPV 均值。",
            "calculation": "mean(IPV_t for key agent 1 over valid estimated steps)。",
            "notes": "正负方向按估计器定义解释；报告不把该符号强行等同于某一交通角色。",
        },
        {
            "category": "Key-agent IPV",
            "english_metric": "ipv_key_agent_2_mean / key agent 2 mean IPV",
            "chinese_metric": "关键 agent 2 的平均 IPV",
            "grain": "agent within case",
            "meaning": "CSV key_agents 中第 2 个 agent 在有效估计步上的有符号 IPV 均值。",
            "calculation": "mean(IPV_t for key agent 2 over valid estimated steps)。",
            "notes": "与 key agent 1 按 CSV key_agents 顺序对应。",
        },
        {
            "category": "Key-agent IPV",
            "english_metric": "agent-level signed IPV",
            "chinese_metric": "agent 级有符号 IPV",
            "grain": "agent",
            "meaning": "把 key agent 1 和 key agent 2 的 mean IPV 合并成长表后得到的单个 agent IPV 值。",
            "calculation": "v in {ipv_key_agent_1_mean, ipv_key_agent_2_mean}。",
            "notes": "用于整体直方图、小提琴图、positive/negative/near-zero 比例等。",
        },
        {
            "category": "Key-agent type",
            "english_metric": "agent_type / key_agent_type",
            "chinese_metric": "关键对象车辆类型",
            "grain": "agent",
            "meaning": "每个 key agent 是自动驾驶车辆 AV 还是人类驾驶车辆 HV。",
            "calculation": "从 pkl 的 metadata.vehicle_type 与 metadata.track_ids 对齐后，按 CSV key_agents 顺序提取 key agent 1/2 的类型。",
            "notes": "不同于 CSV 的 AV_included；AV_included 是 case 级是否包含 AV，agent_type 是单个 key agent 级别的 AV/HV 身份。",
        },
        {
            "category": "Key-agent type",
            "english_metric": "AV",
            "chinese_metric": "自动驾驶车辆",
            "grain": "agent",
            "meaning": "pkl metadata.vehicle_type 标注为 AV 的 key agent。",
            "calculation": "agent_type == 'AV'。",
            "notes": "本报告新增的 AV/HV 分析只统计 key agents，不统计同一 multi-agent 事件中的其他非 key agents。",
        },
        {
            "category": "Key-agent type",
            "english_metric": "HV",
            "chinese_metric": "人类驾驶车辆",
            "grain": "agent",
            "meaning": "pkl metadata.vehicle_type 标注为 HV 的 key agent。",
            "calculation": "agent_type == 'HV'。",
            "notes": "HV 与 AV 的差异在每个数据集中分别比较，避免数据集结构差异混入总体 AV/HV 对比。",
        },
        {
            "category": "Key-agent type",
            "english_metric": "AV - HV delta",
            "chinese_metric": "AV 相对 HV 的差值",
            "grain": "dataset by agent type",
            "meaning": "同一数据集中 AV key agents 的指标相对 HV key agents 高或低多少。",
            "calculation": "metric_AV_within_dataset - metric_HV_within_dataset。",
            "notes": "比例类差值单位为 percentage points (pp)，mean/median 类差值单位为 IPV。",
        },
        {
            "category": "Key-agent type",
            "english_metric": "AV-HV paired case",
            "chinese_metric": "AV-HV 配对案例",
            "grain": "case",
            "meaning": "两个 key agents 中恰好一个是 AV、一个是 HV 的完成案例。",
            "calculation": "ipv_result_status == 'ok' 且 {key_agent_1_type, key_agent_2_type} == {'AV', 'HV'}。",
            "notes": "用于同一 case 内部的 AV 与 HV IPV 配对比较，避免跨 case 样本组成差异。",
        },
        {
            "category": "Key-agent type",
            "english_metric": "paired AV - HV IPV",
            "chinese_metric": "配对 AV-HV IPV 差值",
            "grain": "AV-HV paired case",
            "meaning": "同一 AV-HV 配对案例中，AV 的 mean IPV 相对 HV 的 mean IPV 高或低多少。",
            "calculation": "av_ipv - hv_ipv，其中 av_ipv/hv_ipv 分别来自该 case 中 AV/HV key agent 的 mean IPV。",
            "notes": "正值表示该 case 内 AV 比 HV 更亲社会或更少竞争；负值表示 HV 相对更亲社会或 AV 更竞争。",
        },
        {
            "category": "Distribution summary",
            "english_metric": "signed_mean",
            "chinese_metric": "有符号均值",
            "grain": "group",
            "meaning": "某组 agent-level signed IPV 的平均方向性偏移。",
            "calculation": "mean(v)，v 为 agent-level signed IPV。",
            "notes": "接近 0 表示整体方向性偏移弱，但不代表尾部不强。",
        },
        {
            "category": "Distribution summary",
            "english_metric": "signed_median",
            "chinese_metric": "有符号中位数",
            "grain": "group",
            "meaning": "某组 agent-level signed IPV 的中位数。",
            "calculation": "median(v)。",
            "notes": "本数据中大量值集中在 0，因此 signed_median 经常为 0。",
        },
        {
            "category": "Distribution summary",
            "english_metric": "abs_median / |IPV| median",
            "chinese_metric": "绝对 IPV 中位数",
            "grain": "group",
            "meaning": "忽略正负方向后，典型 IPV 强度的中位数。",
            "calculation": "median(|v|)。",
            "notes": "适合比较不同数据集的典型交互强度。",
        },
        {
            "category": "Distribution summary",
            "english_metric": "near-zero (%)",
            "chinese_metric": "近零 IPV 比例",
            "grain": "agent",
            "meaning": "IPV 很接近 0 的 agent-level 值占比。",
            "calculation": "count(|v| < 0.05) / count(v) * 100。",
            "notes": "阈值 0.05 是报告中用于描述“弱方向性”的统一阈值。",
        },
        {
            "category": "Distribution summary",
            "english_metric": "positive >0.05 (%)",
            "chinese_metric": "正向 IPV 比例",
            "grain": "agent",
            "meaning": "明显大于 0 的 agent-level signed IPV 占比。",
            "calculation": "count(v > 0.05) / count(v) * 100。",
            "notes": "用于看某组数据是否更偏正向。",
        },
        {
            "category": "Distribution summary",
            "english_metric": "negative <-0.05 (%)",
            "chinese_metric": "负向 IPV 比例",
            "grain": "agent",
            "meaning": "明显小于 0 的 agent-level signed IPV 占比。",
            "calculation": "count(v < -0.05) / count(v) * 100。",
            "notes": "用于看某组数据是否更偏负向。",
        },
        {
            "category": "Distribution summary",
            "english_metric": "strong tail / strong |IPV| >=0.30 (%)",
            "chinese_metric": "强 IPV 尾部比例",
            "grain": "agent",
            "meaning": "绝对 IPV 强度达到较高水平的 agent-level 值占比。",
            "calculation": "count(|v| >= 0.30) / count(v) * 100。",
            "notes": "用于凸显长尾强交互案例。",
        },
        {
            "category": "Case-level IPV",
            "english_metric": "pair mean IPV / signed pair mean",
            "chinese_metric": "双车平均有符号 IPV",
            "grain": "case",
            "meaning": "一个 case 中两个 key agents 的有符号 IPV 平均值。",
            "calculation": "(k1 + k2) / 2，其中 k1、k2 分别是 key agent 1/2 mean IPV。",
            "notes": "反映该 case 的整体方向性偏移，正负可能相互抵消。",
        },
        {
            "category": "Case-level IPV",
            "english_metric": "pair |IPV| / pair_abs_ipv",
            "chinese_metric": "双车平均绝对 IPV",
            "grain": "case",
            "meaning": "一个 case 中两个 key agents 的平均 IPV 强度。",
            "calculation": "(|k1| + |k2|) / 2。",
            "notes": "报告中 tail probability、pair |IPV| intensity 主要使用这个指标。",
        },
        {
            "category": "Case-level IPV",
            "english_metric": "max |IPV| / pair_max_abs_ipv",
            "chinese_metric": "双车最大绝对 IPV",
            "grain": "case",
            "meaning": "一个 case 中两个 key agents 里更强的 IPV 强度。",
            "calculation": "max(|k1|, |k2|)。",
            "notes": "用于识别至少有一方交互倾向很强的案例。",
        },
        {
            "category": "Case-level IPV",
            "english_metric": "agent1-agent2 asymmetry",
            "chinese_metric": "agent1 与 agent2 IPV 不对称性",
            "grain": "case",
            "meaning": "同一 case 中两个 key agents 的有符号 IPV 差异。",
            "calculation": "k1 - k2。",
            "notes": "正值表示 key agent 1 的 mean IPV 高于 key agent 2；解释时需回到 key_agents 顺序。",
        },
        {
            "category": "Uncertainty / error",
            "english_metric": "ipv_key_agent_1_error_mean / ipv_key_agent_2_error_mean",
            "chinese_metric": "关键 agent IPV 平均误差",
            "grain": "agent within case",
            "meaning": "估计器为对应 key agent 输出的平均误差。",
            "calculation": "mean(error_t over valid estimated steps)。",
            "notes": "用于辅助判断该 agent 的 IPV 估计可靠性。",
        },
        {
            "category": "Uncertainty / error",
            "english_metric": "mean IPV error / error_pair_mean",
            "chinese_metric": "双车平均 IPV 误差",
            "grain": "case",
            "meaning": "一个 case 中两个 key agents 的平均估计误差。",
            "calculation": "(e1 + e2) / 2，其中 e1、e2 为两个 key agents 的 error_mean。",
            "notes": "报告中 error vs |IPV| 和 effect-size heatmap 使用该指标。",
        },
        {
            "category": "Uncertainty / error",
            "english_metric": "95% CI / CI low / CI high",
            "chinese_metric": "均值 95% 置信区间",
            "grain": "group",
            "meaning": "某组均值的不确定性范围。",
            "calculation": "mean ± t_{0.975, n-1} * SEM。",
            "notes": "用于均值森林图和 dataset mean comparison。",
        },
        {
            "category": "Difference view",
            "english_metric": "difference from pooled",
            "chinese_metric": "相对总体差值",
            "grain": "dataset",
            "meaning": "某数据集指标相对于全部 completed cases 合并结果的偏移。",
            "calculation": "metric_dataset - metric_pooled。",
            "notes": "比例类指标以 percentage points (pp) 表示。",
        },
        {
            "category": "Difference view",
            "english_metric": "quantile difference vs AV2",
            "chinese_metric": "相对 AV2 的分位数差值",
            "grain": "dataset by quantile",
            "meaning": "在同一分位点上，Waymo 或 nuPlan 的 signed IPV 与 AV2 的差异。",
            "calculation": "Q_dataset(p) - Q_AV2(p)。",
            "notes": "用于发现看起来相近的分布在哪个尾部或分位段分开。",
        },
        {
            "category": "Difference view",
            "english_metric": "tail probability difference vs AV2",
            "chinese_metric": "相对 AV2 的尾部概率差值",
            "grain": "dataset by threshold",
            "meaning": "某数据集超过给定 pair |IPV| 阈值的概率相对 AV2 高或低多少。",
            "calculation": "100 * [P_dataset(pair |IPV| >= t) - P_AV2(pair |IPV| >= t)]。",
            "notes": "单位为 percentage points (pp)，用于凸显强交互尾部差异。",
        },
        {
            "category": "Difference view",
            "english_metric": "pooled",
            "chinese_metric": "合并总体",
            "grain": "all completed cases",
            "meaning": "把 AV2、Waymo、nuPlan 已完成结果合并后的总体。",
            "calculation": "所有 completed cases 或 agent-level values 的合并集合。",
            "notes": "nuPlan 未补齐，因此 pooled 本身也受当前完成情况影响。",
        },
        {
            "category": "Statistical test",
            "english_metric": "Kruskal-Wallis statistic",
            "chinese_metric": "Kruskal-Wallis 多组秩检验统计量",
            "grain": "metric across datasets",
            "meaning": "非参数检验，用于判断三个数据集的分布是否存在整体差异。",
            "calculation": "对各组秩次进行 Kruskal-Wallis H 检验。",
            "notes": "只提示分布差异是否显著，不说明差异机制。",
        },
        {
            "category": "Statistical test",
            "english_metric": "Mann-Whitney U",
            "chinese_metric": "Mann-Whitney 两组秩检验统计量",
            "grain": "pairwise datasets",
            "meaning": "非参数两组检验，用于比较两个数据集的指标分布。",
            "calculation": "基于两组样本秩次计算 U statistic。",
            "notes": "完整结果保存在 dataset_pairwise_tests.csv。",
        },
        {
            "category": "Statistical test",
            "english_metric": "Cliff's delta",
            "chinese_metric": "Cliff 效应量",
            "grain": "pairwise datasets",
            "meaning": "衡量两个数据集分布系统性偏移的效应量。",
            "calculation": "P(X_first > X_second) - P(X_first < X_second)，范围 [-1, 1]。",
            "notes": "正值表示列名中的第一个数据集在该指标上整体更大。",
        },
        {
            "category": "Statistical test",
            "english_metric": "Spearman rho",
            "chinese_metric": "Spearman 秩相关系数",
            "grain": "case-level metric pair",
            "meaning": "两个指标之间的单调相关强度。",
            "calculation": "对两个变量分别取秩后计算 Pearson correlation。",
            "notes": "rho 接近 0 表示单调关系弱；正负表示方向。",
        },
        {
            "category": "Statistical test",
            "english_metric": "p value",
            "chinese_metric": "p 值",
            "grain": "test",
            "meaning": "在零假设成立时观察到当前或更极端统计量的概率。",
            "calculation": "由对应统计检验计算。",
            "notes": "样本量很大时很小的差异也可能显著，建议结合 effect size 看。",
        },
        {
            "category": "Scenario / risk proxy",
            "english_metric": "PET",
            "chinese_metric": "后侵入时间 / Post-Encroachment Time",
            "grain": "case",
            "meaning": "CSV 提供的交互风险代理指标，通常表示两个道路使用者通过潜在冲突区域的时间间隔。",
            "calculation": "报告直接读取 CSV 的 PET 字段，不在报告脚本中重算。",
            "notes": "单位和精确定义以原始数据生产流程为准。",
        },
        {
            "category": "Scenario / risk proxy",
            "english_metric": "intensity",
            "chinese_metric": "交互强度指标",
            "grain": "case",
            "meaning": "CSV 提供的交互强度/风险代理指标。",
            "calculation": "报告直接读取 CSV 的 intensity 字段，不在报告脚本中重算。",
            "notes": "该字段长尾很重，因此图中经常使用 log10 intensity。",
        },
        {
            "category": "Scenario / risk proxy",
            "english_metric": "log10 intensity / log10_intensity",
            "chinese_metric": "交互强度的 10 底对数",
            "grain": "case",
            "meaning": "对 intensity 做对数变换后得到的尺度，用于压缩极端长尾。",
            "calculation": "log10(intensity)，当前 intensity 全为正值。",
            "notes": "用于 Spearman correlation 和 PET/intensity 分箱图。",
        },
        {
            "category": "Scenario / risk proxy",
            "english_metric": "duration_steps",
            "chinese_metric": "有效估计步数",
            "grain": "case",
            "meaning": "该 case 中参与 IPV 统计的有效时间步数量。",
            "calculation": "从结果明细中统计有效估计步数。",
            "notes": "不同数据集采样率处理会影响该指标，例如 nuPlan 已按 20Hz 到 10Hz 处理。",
        },
        {
            "category": "Scenario grouping",
            "english_metric": "two/multi",
            "chinese_metric": "双车 / 多 agent 事件标签",
            "grain": "case",
            "meaning": "标记该交互事件是否只有两个 agent，或还有其他非 key agents。",
            "calculation": "来自 CSV 字段；报告中 multi-agent 仍只计算两个 key agents。",
            "notes": "用于比较 two 和 multi 的 agent-level IPV 分布。",
        },
        {
            "category": "Scenario grouping",
            "english_metric": "AV_included",
            "chinese_metric": "是否包含自动驾驶车辆",
            "grain": "case",
            "meaning": "标记 key interaction 中是否包含 AV。",
            "calculation": "来自 CSV AV_included 字段。",
            "notes": "用于比较 AV 与 all_HV 相关分组。",
        },
        {
            "category": "Scenario grouping",
            "english_metric": "path_category",
            "chinese_metric": "路径类别",
            "grain": "case",
            "meaning": "对两个 key agents 的路径关系进行较粗粒度分类。",
            "calculation": "来自 CSV path_category 字段。",
            "notes": "报告中用于 path category forest plot。",
        },
        {
            "category": "Scenario grouping",
            "english_metric": "path_relation",
            "chinese_metric": "路径关系",
            "grain": "case",
            "meaning": "比 path_category 更细的路径关系标签。",
            "calculation": "来自 CSV path_relation 字段。",
            "notes": "样本量较小的 relation 需要谨慎解释。",
        },
        {
            "category": "Scenario grouping",
            "english_metric": "turn_label",
            "chinese_metric": "转向组合标签",
            "grain": "case",
            "meaning": "两个 key agents 的转向组合，如 S-S、L-S、R-R 等。",
            "calculation": "来自 CSV turn_label 字段。",
            "notes": "用于比较不同转向组合下的 mean IPV。",
        },
    ]
    return pd.DataFrame(rows)


def metric_definition_table_html() -> str:
    frame = metric_definitions()
    frame.to_csv(TABLE_DIR / "metric_definitions.csv", index=False, encoding="utf-8")
    display = frame.rename(
        columns={
            "category": "类别",
            "english_metric": "English metric",
            "chinese_metric": "中文名称",
            "grain": "统计粒度",
            "meaning": "含义",
            "calculation": "计算方式",
            "notes": "说明",
        }
    )
    return display.to_html(index=False, classes="data-table definition-table", border=0, escape=True)


def load_key_agent_types() -> pd.DataFrame:
    rows = []
    for pkl_path in sorted(PKL_ROOT.rglob("*.pkl")):
        with pkl_path.open("rb") as handle:
            data = pickle.load(handle)
        events = data.items() if isinstance(data, dict) else enumerate(data)
        for segment_id, event in events:
            if not isinstance(event, dict):
                continue
            metadata = event.get("metadata", {})
            if not isinstance(metadata, dict):
                continue
            key_agents = str(metadata.get("key_agents", ""))
            agent_ids = [part.strip() for part in key_agents.split(";") if part.strip()]
            track_ids = [str(item) for item in metadata.get("track_ids", [])]
            vehicle_types = [str(item) for item in metadata.get("vehicle_type", [])]
            type_by_track = {
                track_id: vehicle_types[i]
                for i, track_id in enumerate(track_ids)
                if i < len(vehicle_types)
            }
            if len(agent_ids) != 2:
                continue
            rows.append(
                {
                    "folder": str(metadata.get("folder") or metadata.get("subdata")),
                    "scenario_idx": str(metadata.get("scenario_idx")),
                    "key_agents": key_agents,
                    "track_id": ";".join(track_ids),
                    "key_agent_1_id": agent_ids[0],
                    "key_agent_2_id": agent_ids[1],
                    "key_agent_1_type": type_by_track.get(agent_ids[0]),
                    "key_agent_2_type": type_by_track.get(agent_ids[1]),
                    "agent_type_pkl_file": str(pkl_path.relative_to(PKL_ROOT)),
                    "agent_type_segment_id": str(segment_id),
                }
            )
    frame = pd.DataFrame(rows)
    key_cols = ["folder", "scenario_idx", "key_agents", "track_id"]
    if frame.empty:
        raise ValueError(f"No key-agent type metadata found under {PKL_ROOT}")
    if frame.duplicated(key_cols).any():
        duplicates = frame.loc[frame.duplicated(key_cols, keep=False), key_cols].head().to_dict("records")
        raise ValueError(f"Duplicate key-agent type keys detected: {duplicates}")
    return frame


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(MATCHED_CSV)
    df["dataset_label"] = df["dataset"].map(DATASET_LABELS).fillna(df["dataset"])
    numeric_cols = [
        "ipv_key_agent_1_mean",
        "ipv_key_agent_2_mean",
        "ipv_key_agent_1_error_mean",
        "ipv_key_agent_2_error_mean",
        "PET",
        "intensity",
    ]
    for col in numeric_cols:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    key_cols = ["folder", "scenario_idx", "key_agents", "track_id"]
    agent_types = load_key_agent_types()
    for col in key_cols:
        df[col] = df[col].astype(str)
        agent_types[col] = agent_types[col].astype(str)
    df = df.merge(agent_types, on=key_cols, how="left", validate="one_to_one")
    missing_type_count = int(df[["key_agent_1_type", "key_agent_2_type"]].isna().any(axis=1).sum())
    if missing_type_count:
        raise ValueError(f"Missing key-agent AV/HV type for {missing_type_count} CSV rows")

    ok = df[df["ipv_result_status"].eq("ok")].copy()
    ok["pair_mean_ipv"] = ok[["ipv_key_agent_1_mean", "ipv_key_agent_2_mean"]].mean(axis=1)
    ok["pair_abs_ipv"] = ok[["ipv_key_agent_1_mean", "ipv_key_agent_2_mean"]].abs().mean(axis=1)
    ok["pair_max_abs_ipv"] = ok[
        ["ipv_key_agent_1_mean", "ipv_key_agent_2_mean"]
    ].abs().max(axis=1)
    ok["agent_asymmetry"] = ok["ipv_key_agent_1_mean"] - ok["ipv_key_agent_2_mean"]
    ok["mean_error"] = ok[["ipv_key_agent_1_error_mean", "ipv_key_agent_2_error_mean"]].mean(axis=1)

    agent_long = pd.concat(
        [
            ok[["dataset", "dataset_label", "key_agent_1_id", "key_agent_1_type", "ipv_key_agent_1_mean"]]
            .rename(columns={"ipv_key_agent_1_mean": "ipv"})
            .rename(columns={"key_agent_1_id": "agent_id", "key_agent_1_type": "agent_type"})
            .assign(agent_order="agent 1"),
            ok[["dataset", "dataset_label", "key_agent_2_id", "key_agent_2_type", "ipv_key_agent_2_mean"]]
            .rename(columns={"ipv_key_agent_2_mean": "ipv"})
            .rename(columns={"key_agent_2_id": "agent_id", "key_agent_2_type": "agent_type"})
            .assign(agent_order="agent 2"),
        ],
        ignore_index=True,
    ).dropna(subset=["ipv"])

    ok["dataset_label"] = pd.Categorical(ok["dataset_label"], DATASET_ORDER, ordered=True)
    agent_long["dataset_label"] = pd.Categorical(
        agent_long["dataset_label"], DATASET_ORDER, ordered=True
    )
    return df, ok, agent_long


def build_dataset_tables(
    df: pd.DataFrame, ok: pd.DataFrame, agent_long: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for label in DATASET_ORDER:
        all_part = df[df["dataset_label"].eq(label)]
        ok_part = ok[ok["dataset_label"].eq(label)]
        agent_part = agent_long[agent_long["dataset_label"].eq(label)]
        mean, lo, hi = ci_mean(agent_part["ipv"])
        pair_abs_mean, pair_abs_lo, pair_abs_hi = ci_mean(ok_part["pair_abs_ipv"])
        rows.append(
            {
                "dataset": label,
                "total_cases": len(all_part),
                "completed_cases": len(ok_part),
                "missing_cases": int((~all_part["ipv_result_status"].eq("ok")).sum()),
                "completion_pct": 100 * len(ok_part) / max(len(all_part), 1),
                "agent_values": len(agent_part),
                "signed_mean": mean,
                "signed_ci_low": lo,
                "signed_ci_high": hi,
                "signed_median": float(agent_part["ipv"].median()) if len(agent_part) else np.nan,
                "abs_median": float(agent_part["ipv"].abs().median()) if len(agent_part) else np.nan,
                "pair_abs_mean": pair_abs_mean,
                "pair_abs_ci_low": pair_abs_lo,
                "pair_abs_ci_high": pair_abs_hi,
                "pair_abs_median": float(ok_part["pair_abs_ipv"].median()) if len(ok_part) else np.nan,
                "positive_gt_0_05_pct": 100 * float((agent_part["ipv"] > 0.05).mean())
                if len(agent_part)
                else np.nan,
                "negative_lt_minus_0_05_pct": 100
                * float((agent_part["ipv"] < -0.05).mean())
                if len(agent_part)
                else np.nan,
                "near_zero_abs_lt_0_05_pct": 100
                * float((agent_part["ipv"].abs() < 0.05).mean())
                if len(agent_part)
                else np.nan,
                "strong_abs_ge_0_30_pct": 100
                * float((agent_part["ipv"].abs() >= 0.30).mean())
                if len(agent_part)
                else np.nan,
                "mean_error": float(ok_part["mean_error"].mean()) if len(ok_part) else np.nan,
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(TABLE_DIR / "dataset_comparison_summary.csv", index=False, encoding="utf-8")

    tests = []
    metrics = [
        ("agent_signed_ipv", agent_long, "ipv"),
        ("case_pair_abs_ipv", ok, "pair_abs_ipv"),
        ("case_pair_mean_ipv", ok, "pair_mean_ipv"),
        ("case_agent_asymmetry", ok, "agent_asymmetry"),
        ("case_mean_error", ok, "mean_error"),
    ]
    for metric_name, source, value_col in metrics:
        groups = [
            pd.to_numeric(
                source.loc[source["dataset_label"].eq(label), value_col], errors="coerce"
            )
            .dropna()
            .to_numpy()
            for label in DATASET_ORDER
        ]
        kw_stat, kw_p = stats.kruskal(*groups) if all(len(g) for g in groups) else (np.nan, np.nan)
        tests.append(
            {
                "metric": metric_name,
                "comparison": "Kruskal-Wallis all datasets",
                "statistic": kw_stat,
                "p_value": kw_p,
                "median_delta_first_minus_second": np.nan,
                "n": sum(len(g) for g in groups),
            }
        )
        for i, first in enumerate(DATASET_ORDER):
            for second in DATASET_ORDER[i + 1 :]:
                first_values = pd.to_numeric(
                    source.loc[source["dataset_label"].eq(first), value_col], errors="coerce"
                ).dropna()
                second_values = pd.to_numeric(
                    source.loc[source["dataset_label"].eq(second), value_col], errors="coerce"
                ).dropna()
                if len(first_values) and len(second_values):
                    stat, p_value = stats.mannwhitneyu(
                        first_values.to_numpy(), second_values.to_numpy(), alternative="two-sided"
                    )
                    delta = float(first_values.median() - second_values.median())
                else:
                    stat, p_value, delta = np.nan, np.nan, np.nan
                tests.append(
                    {
                        "metric": metric_name,
                        "comparison": f"{first} vs {second}",
                        "statistic": stat,
                        "p_value": p_value,
                        "median_delta_first_minus_second": delta,
                        "n": len(first_values) + len(second_values),
                    }
                )
    tests_frame = pd.DataFrame(tests)
    tests_frame.to_csv(TABLE_DIR / "dataset_pairwise_tests.csv", index=False, encoding="utf-8")
    return summary, tests_frame


def plot_pair_density(ok: pd.DataFrame) -> float:
    lim = 0.30
    pair = ok[["ipv_key_agent_1_mean", "ipv_key_agent_2_mean"]].dropna()
    inside = pair["ipv_key_agent_1_mean"].between(-lim, lim) & pair[
        "ipv_key_agent_2_mean"
    ].between(-lim, lim)
    pair_zoom = pair[inside]
    inside_pct = 100 * len(pair_zoom) / max(len(pair), 1)

    fig = plt.figure(figsize=(8.6, 8.0), facecolor="#FFFFFF")
    gs = GridSpec(
        2,
        2,
        width_ratios=[4.6, 1.05],
        height_ratios=[1.05, 4.6],
        wspace=0.05,
        hspace=0.05,
        figure=fig,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "nature_blue",
        ["#F4F5F7", FAMILIES["blue"]["light"], FAMILIES["blue"]["mid"], FAMILIES["blue"]["dark"]],
    )
    hb = ax.hexbin(
        pair_zoom["ipv_key_agent_1_mean"],
        pair_zoom["ipv_key_agent_2_mean"],
        gridsize=56,
        extent=(-lim, lim, -lim, lim),
        mincnt=1,
        cmap=cmap,
        norm=LogNorm(vmin=1),
        linewidths=0,
    )
    ax.axhline(0, color=TOKENS["ink"], linewidth=0.9, alpha=0.75)
    ax.axvline(0, color=TOKENS["ink"], linewidth=0.9, alpha=0.75)
    ax.plot(
        [-lim, lim],
        [-lim, lim],
        color=NEUTRAL["mid"],
        linestyle="--",
        linewidth=0.9,
        label="agent 1 = agent 2",
    )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Key agent 1 mean IPV")
    ax.set_ylabel("Key agent 2 mean IPV")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.10))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.10))
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.025)
    cb.set_label("Cases per hexbin, log scale", fontsize=8, color=TOKENS["muted"])
    cb.ax.tick_params(labelsize=7, colors=TOKENS["muted"])

    bins = np.linspace(-lim, lim, 52)
    ax_top.hist(
        pair_zoom["ipv_key_agent_1_mean"],
        bins=bins,
        color=FAMILIES["blue"]["base"],
        edgecolor=FAMILIES["blue"]["dark"],
        linewidth=0.35,
    )
    ax_top.axvline(0, color=TOKENS["ink"], linewidth=0.8)
    ax_top.set_xlim(-lim, lim)
    ax_top.set_ylabel("Cases", fontsize=8)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.tick_params(axis="y", labelsize=7)

    ax_right.hist(
        pair_zoom["ipv_key_agent_2_mean"],
        bins=bins,
        orientation="horizontal",
        color=FAMILIES["blue"]["light"],
        edgecolor=FAMILIES["blue"]["dark"],
        linewidth=0.35,
    )
    ax_right.axhline(0, color=TOKENS["ink"], linewidth=0.8)
    ax_right.set_ylim(-lim, lim)
    ax_right.set_xlabel("Cases", fontsize=8)
    ax_right.tick_params(axis="y", labelleft=False)
    ax_right.tick_params(axis="x", labelsize=7)

    fig.subplots_adjust(top=0.82)
    left = ax.get_position().x0
    fig.text(
        left,
        0.965,
        "Pair-level IPV density, zoomed to the core range",
        ha="left",
        va="top",
        fontsize=13,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        left,
        0.925,
        f"Display range is [-0.30, 0.30] for both key agents and covers {inside_pct:.1f}% of completed cases; color uses log density so the central cluster remains readable.",
        ha="left",
        va="top",
        fontsize=9,
        color=TOKENS["muted"],
    )
    sns.despine(fig=fig)
    savefig(fig, "04_pair_ipv_hexbin.png")
    return inside_pct


def plot_dataset_overview(
    ok: pd.DataFrame, agent_long: pd.DataFrame, summary: pd.DataFrame
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.8), facecolor="#FFFFFF")
    ax_cov, ax_dist, ax_ci, ax_pair = axes.ravel()
    fig.subplots_adjust(top=0.84, hspace=0.45, wspace=0.30)
    cov = summary.set_index("dataset").loc[DATASET_ORDER].reset_index()

    y = np.arange(len(cov))
    ax_cov.barh(
        y,
        cov["completed_cases"],
        color=FAMILIES["blue"]["base"],
        edgecolor=FAMILIES["blue"]["dark"],
        linewidth=0.9,
        label="completed",
    )
    ax_cov.barh(
        y,
        cov["missing_cases"],
        left=cov["completed_cases"],
        color=NEUTRAL["light"],
        edgecolor=NEUTRAL["mid"],
        linewidth=0.9,
        label="missing",
    )
    for yi, row in cov.iterrows():
        ax_cov.text(
            row["completed_cases"] + 35,
            yi,
            f"{int(row['completed_cases']):,}/{int(row['total_cases']):,}",
            va="center",
            fontsize=8,
            color=TOKENS["ink"],
        )
    ax_cov.set_yticks(y, cov["dataset"])
    ax_cov.set_xlabel("Cases")
    ax_cov.set_title(
        "Coverage: blue = completed, gray = missing",
        loc="left",
        fontsize=10,
        fontweight="semibold",
    )
    ax_cov.invert_yaxis()

    plot_agent = agent_long.copy()
    clip_lim = 0.35
    plot_agent["ipv_display"] = plot_agent["ipv"].clip(-clip_lim, clip_lim)
    sns.violinplot(
        data=plot_agent,
        x="dataset_label",
        y="ipv_display",
        hue="dataset_label",
        order=DATASET_ORDER,
        hue_order=DATASET_ORDER,
        palette=DATASET_PALETTE,
        inner="quartile",
        cut=0,
        linewidth=0.9,
        legend=False,
        ax=ax_dist,
    )
    ax_dist.axhline(0, color=TOKENS["ink"], linewidth=0.85)
    ax_dist.set_xlabel("")
    ax_dist.set_ylabel("Agent-level signed IPV, clipped")
    ax_dist.set_ylim(-clip_lim, clip_lim)
    ax_dist.set_title("Signed distribution", loc="left", fontsize=10, fontweight="semibold")
    ax_dist.text(
        0.02,
        0.04,
        "display clipped to +/-0.35",
        transform=ax_dist.transAxes,
        fontsize=8,
        color=TOKENS["muted"],
    )

    ci_y = np.arange(len(cov))
    for i, row in cov.iterrows():
        label = row["dataset"]
        ax_ci.hlines(
            i,
            row["signed_ci_low"],
            row["signed_ci_high"],
            color=DATASET_EDGE[label],
            linewidth=1.0,
        )
        ax_ci.scatter(
            row["signed_mean"],
            i,
            s=58,
            facecolor=DATASET_PALETTE[label],
            edgecolor=DATASET_EDGE[label],
            linewidth=1.0,
            zorder=3,
        )
        ax_ci.text(
            0.072,
            i,
            f"near-zero {row['near_zero_abs_lt_0_05_pct']:.1f}% | strong {row['strong_abs_ge_0_30_pct']:.1f}%",
            va="center",
            fontsize=8,
            color=TOKENS["muted"],
        )
    ax_ci.axvline(0, color=TOKENS["ink"], linewidth=0.85)
    ax_ci.set_yticks(ci_y, cov["dataset"])
    ax_ci.set_xlabel("Mean signed IPV with 95% CI")
    ax_ci.set_xlim(-0.04, 0.11)
    ax_ci.set_title("Mean and composition", loc="left", fontsize=10, fontweight="semibold")
    ax_ci.invert_yaxis()

    plot_pair = ok[["dataset_label", "pair_abs_ipv"]].dropna().copy()
    plot_pair["pair_abs_display"] = plot_pair["pair_abs_ipv"].clip(0, 0.42)
    sns.boxplot(
        data=plot_pair,
        x="dataset_label",
        y="pair_abs_display",
        hue="dataset_label",
        order=DATASET_ORDER,
        hue_order=DATASET_ORDER,
        palette=DATASET_PALETTE,
        width=0.52,
        showfliers=False,
        linewidth=0.9,
        legend=False,
        ax=ax_pair,
    )
    for i, label in enumerate(DATASET_ORDER):
        vals = plot_pair.loc[plot_pair["dataset_label"].eq(label), "pair_abs_ipv"]
        mean, lo, hi = ci_mean(vals)
        ax_pair.scatter(
            i, mean, s=54, facecolor="#FFFFFF", edgecolor=DATASET_EDGE[label], linewidth=1.2, zorder=4
        )
        ax_pair.vlines(i, lo, hi, color=DATASET_EDGE[label], linewidth=1.0, zorder=3)
    ax_pair.set_xlabel("")
    ax_pair.set_ylabel("Case-level pair |IPV|, clipped")
    ax_pair.set_ylim(0, 0.42)
    ax_pair.set_title("Pair |IPV| intensity", loc="left", fontsize=10, fontweight="semibold")
    ax_pair.text(
        0.02,
        0.92,
        "open dot = mean, whisker = 95% CI",
        transform=ax_pair.transAxes,
        fontsize=8,
        color=TOKENS["muted"],
    )

    left = axes[0, 0].get_position().x0
    fig.text(
        left,
        0.98,
        "Dataset comparison overview",
        ha="left",
        va="top",
        fontsize=13,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        left,
        0.945,
        "Coverage, signed IPV distribution, mean CI, and case-level pair |IPV| are shown together; nuPlan remains incomplete, so its differences should be treated as provisional.",
        ha="left",
        va="top",
        fontsize=9,
        color=TOKENS["muted"],
    )
    sns.despine(fig=fig)
    savefig(fig, "13_dataset_comparison_overview.png")


def plot_sign_strength(summary: pd.DataFrame) -> None:
    metric_order = [
        "near-zero (%)",
        "positive >0.05 (%)",
        "negative <-0.05 (%)",
        "strong |IPV| >=0.30 (%)",
    ]
    rows = []
    for _, row in summary.iterrows():
        rows.extend(
            [
                {"dataset": row["dataset"], "metric": metric_order[0], "value": row["near_zero_abs_lt_0_05_pct"]},
                {"dataset": row["dataset"], "metric": metric_order[1], "value": row["positive_gt_0_05_pct"]},
                {"dataset": row["dataset"], "metric": metric_order[2], "value": row["negative_lt_minus_0_05_pct"]},
                {"dataset": row["dataset"], "metric": metric_order[3], "value": row["strong_abs_ge_0_30_pct"]},
            ]
        )
    metrics_long = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9.8, 5.6), facecolor="#FFFFFF")
    x = np.arange(len(metric_order))
    width = 0.23
    for i, label in enumerate(DATASET_ORDER):
        values = metrics_long[metrics_long["dataset"].eq(label)].set_index("metric").loc[
            metric_order, "value"
        ]
        bars = ax.bar(
            x + (i - 1) * width,
            values,
            width=width,
            label=label,
            color=DATASET_PALETTE[label],
            edgecolor=DATASET_EDGE[label],
            linewidth=0.9,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=TOKENS["ink"],
            )
    ax.set_xticks(x, metric_order)
    ax.set_ylabel("Agent-level share")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_ylim(0, max(80, float(metrics_long["value"].max()) + 9))
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, ncol=3, fontsize=8)
    fig.subplots_adjust(top=0.80)
    left = ax.get_position().x0
    fig.text(
        left,
        0.98,
        "Dataset comparison of IPV sign and strength",
        ha="left",
        va="top",
        fontsize=13,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        left,
        0.935,
        "Shares are computed on agent-level IPV values from completed cases only; strong cases use |IPV| >= 0.30.",
        ha="left",
        va="top",
        fontsize=9,
        color=TOKENS["muted"],
    )
    sns.despine(fig=fig)
    savefig(fig, "14_dataset_comparison_sign_strength.png")


def plot_dataset_difference_lens(
    ok: pd.DataFrame, agent_long: pd.DataFrame, summary: pd.DataFrame
) -> None:
    """Show where similar-looking dataset distributions actually differ."""
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.9), facecolor="#FFFFFF")
    ax_share, ax_mean, ax_quantile, ax_tail = axes.ravel()
    fig.subplots_adjust(top=0.76, hspace=0.58, wspace=0.34)

    summary = summary.set_index("dataset").loc[DATASET_ORDER].reset_index()
    overall_shares = {
        "near-zero": 100 * float((agent_long["ipv"].abs() < 0.05).mean()),
        "positive": 100 * float((agent_long["ipv"] > 0.05).mean()),
        "negative": 100 * float((agent_long["ipv"] < -0.05).mean()),
        "strong tail": 100 * float((agent_long["ipv"].abs() >= 0.30).mean()),
    }
    share_cols = {
        "near-zero": "near_zero_abs_lt_0_05_pct",
        "positive": "positive_gt_0_05_pct",
        "negative": "negative_lt_minus_0_05_pct",
        "strong tail": "strong_abs_ge_0_30_pct",
    }
    share_metrics = list(share_cols)
    x = np.arange(len(share_metrics))
    width = 0.23
    for i, label in enumerate(DATASET_ORDER):
        row = summary[summary["dataset"].eq(label)].iloc[0]
        values = [row[share_cols[m]] - overall_shares[m] for m in share_metrics]
        bars = ax_share.bar(
            x + (i - 1) * width,
            values,
            width=width,
            color=DATASET_PALETTE[label],
            edgecolor=DATASET_EDGE[label],
            linewidth=0.9,
            label=label,
        )
        for bar, value in zip(bars, values):
            ax_share.text(
                bar.get_x() + bar.get_width() / 2,
                value + (0.55 if value >= 0 else -0.75),
                f"{value:+.1f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=7.5,
                color=TOKENS["ink"],
            )
    ax_share.axhline(0, color=TOKENS["ink"], linewidth=0.9)
    ax_share.set_xticks(x, share_metrics)
    ax_share.set_ylabel("Difference from pooled share, pp")
    ax_share.set_title("Sign and tail shares: difference from pooled", loc="left", fontsize=10, fontweight="semibold")

    overall_means = {
        "signed mean": float(agent_long["ipv"].mean()),
        "abs median": float(agent_long["ipv"].abs().median()),
        "pair |IPV| mean": float(ok["pair_abs_ipv"].mean()),
    }
    mean_cols = {
        "signed mean": "signed_mean",
        "abs median": "abs_median",
        "pair |IPV| mean": "pair_abs_mean",
    }
    mean_rows = []
    for label in DATASET_ORDER:
        row = summary[summary["dataset"].eq(label)].iloc[0]
        for metric, col in mean_cols.items():
            mean_rows.append(
                {
                    "dataset": label,
                    "metric": metric,
                    "delta": row[col] - overall_means[metric],
                }
            )
    mean_frame = pd.DataFrame(mean_rows)
    y_positions = np.arange(len(mean_cols))
    for i, label in enumerate(DATASET_ORDER):
        values = (
            mean_frame[mean_frame["dataset"].eq(label)]
            .set_index("metric")
            .loc[list(mean_cols), "delta"]
            .to_numpy()
        )
        ax_mean.plot(
            values,
            y_positions + (i - 1) * 0.12,
            marker="o",
            color=DATASET_EDGE[label],
            markerfacecolor=DATASET_PALETTE[label],
            linewidth=1.0,
            label=label,
        )
        for value, y_pos in zip(values, y_positions + (i - 1) * 0.12):
            ax_mean.text(
                value + (0.002 if value >= 0 else -0.002),
                y_pos,
                f"{value:+.3f}",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=7.5,
                color=TOKENS["muted"],
            )
    ax_mean.axvline(0, color=TOKENS["ink"], linewidth=0.9)
    ax_mean.set_yticks(y_positions, list(mean_cols))
    ax_mean.set_xlabel("Difference from pooled value")
    ax_mean.set_title("Central tendency and intensity deltas", loc="left", fontsize=10, fontweight="semibold")
    ax_mean.set_xlim(-0.035, 0.045)
    ax_mean.invert_yaxis()

    quantiles = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    av2_quantiles = (
        agent_long.loc[agent_long["dataset_label"].eq("AV2"), "ipv"].quantile(quantiles)
    )
    for label in ["Waymo", "nuPlan"]:
        q_values = agent_long.loc[agent_long["dataset_label"].eq(label), "ipv"].quantile(quantiles)
        delta = q_values.to_numpy() - av2_quantiles.to_numpy()
        ax_quantile.plot(
            quantiles * 100,
            delta,
            marker="o",
            color=DATASET_EDGE[label],
            markerfacecolor=DATASET_PALETTE[label],
            linewidth=1.0,
            label=f"{label} - AV2",
        )
        ax_quantile.text(
            quantiles[-1] * 100 + 1.2,
            delta[-1],
            f"{label} - AV2",
            va="center",
            fontsize=8,
            color=DATASET_EDGE[label],
        )
    ax_quantile.axhline(0, color=TOKENS["ink"], linewidth=0.9)
    ax_quantile.set_xlabel("Signed IPV quantile")
    ax_quantile.set_ylabel("Quantile difference vs AV2")
    ax_quantile.set_title("Where the signed distributions separate", loc="left", fontsize=10, fontweight="semibold")
    ax_quantile.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax_quantile.set_xlim(4, 104)

    thresholds = np.array([0.02, 0.05, 0.10, 0.20, 0.30, 0.40])
    av2_tail = np.array(
        [
            100 * float((ok.loc[ok["dataset_label"].eq("AV2"), "pair_abs_ipv"] >= threshold).mean())
            for threshold in thresholds
        ]
    )
    for label in ["Waymo", "nuPlan"]:
        tail = np.array(
            [
                100
                * float((ok.loc[ok["dataset_label"].eq(label), "pair_abs_ipv"] >= threshold).mean())
                for threshold in thresholds
            ]
        )
        tail_delta = tail - av2_tail
        ax_tail.plot(
            thresholds,
            tail_delta,
            marker="o",
            color=DATASET_EDGE[label],
            markerfacecolor=DATASET_PALETTE[label],
            linewidth=1.0,
            label=f"{label} - AV2",
        )
        label_idx = int(np.nanargmax(tail_delta))
        ax_tail.text(
            thresholds[label_idx] + 0.012,
            tail_delta[label_idx],
            f"{label} - AV2",
            va="center",
            fontsize=8,
            color=DATASET_EDGE[label],
        )
    ax_tail.axhline(0, color=TOKENS["ink"], linewidth=0.9)
    ax_tail.set_xlabel("pair |IPV| threshold")
    ax_tail.set_ylabel("Tail probability difference vs AV2, pp")
    ax_tail.set_title("How much heavier the interaction tail is", loc="left", fontsize=10, fontweight="semibold")
    ax_tail.set_xlim(0.015, 0.445)

    left = axes[0, 0].get_position().x0
    fig.text(
        left,
        0.98,
        "Dataset difference lens",
        ha="left",
        va="top",
        fontsize=13,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        left,
        0.945,
        "Instead of repeating raw distributions, this view shows deviations from the pooled result or from AV2 so close-looking distributions reveal where they separate.",
        ha="left",
        va="top",
        fontsize=9,
        color=TOKENS["muted"],
    )
    handles = [
        mpl.patches.Patch(facecolor=DATASET_PALETTE[label], edgecolor=DATASET_EDGE[label], label=label)
        for label in DATASET_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(left, 0.895),
        frameon=False,
        ncol=3,
        fontsize=8.5,
        borderaxespad=0,
    )
    sns.despine(fig=fig)
    savefig(fig, "13_dataset_difference_lens.png")


def plot_dataset_effect_size_heatmap(ok: pd.DataFrame, agent_long: pd.DataFrame) -> pd.DataFrame:
    agent_abs = agent_long.assign(abs_ipv=agent_long["ipv"].abs())
    metric_sources = {
        "agent signed IPV": (agent_long, "ipv"),
        "agent |IPV|": (agent_abs, "abs_ipv"),
        "case pair |IPV|": (ok, "pair_abs_ipv"),
        "case pair mean IPV": (ok, "pair_mean_ipv"),
        "agent1-agent2 asymmetry": (ok, "agent_asymmetry"),
        "mean IPV error": (ok, "mean_error"),
    }
    comparisons = [
        ("Waymo - AV2", "Waymo", "AV2"),
        ("nuPlan - AV2", "nuPlan", "AV2"),
        ("nuPlan - Waymo", "nuPlan", "Waymo"),
    ]
    rows = []
    for metric, (source, value_col) in metric_sources.items():
        for comparison, first, second in comparisons:
            first_values = source.loc[source["dataset_label"].eq(first), value_col]
            second_values = source.loc[source["dataset_label"].eq(second), value_col]
            rows.append(
                {
                    "metric": metric,
                    "comparison": comparison,
                    "cliffs_delta": cliff_delta(first_values, second_values),
                    "median_delta": float(first_values.median() - second_values.median()),
                    "n_first": int(first_values.notna().sum()),
                    "n_second": int(second_values.notna().sum()),
                }
            )
    effects = pd.DataFrame(rows)
    effects.to_csv(TABLE_DIR / "dataset_effect_size_summary.csv", index=False, encoding="utf-8")

    matrix = effects.pivot(index="metric", columns="comparison", values="cliffs_delta")
    matrix = matrix.loc[list(metric_sources), [c[0] for c in comparisons]]
    fig, ax = plt.subplots(figsize=(8.8, 5.7), facecolor="#FFFFFF")
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "delta_diverging",
        [FAMILIES["orange"]["mid"], "#FFFFFF", FAMILIES["blue"]["mid"]],
    )
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        center=0,
        vmin=-0.35,
        vmax=0.35,
        linewidths=0.8,
        linecolor=TOKENS["grid"],
        annot=True,
        fmt="+.2f",
        cbar_kws={"label": "Cliff's delta"},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    fig.subplots_adjust(top=0.78, left=0.24)
    left = ax.get_position().x0
    fig.text(
        left,
        0.98,
        "Pairwise dataset effect sizes",
        ha="left",
        va="top",
        fontsize=13,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        left,
        0.925,
        "Positive values mean the first dataset in the column tends to be larger; effect size makes small but systematic distribution shifts visible.",
        ha="left",
        va="top",
        fontsize=9,
        color=TOKENS["muted"],
    )
    sns.despine(fig=fig, left=True, bottom=True)
    savefig(fig, "14_dataset_effect_size_heatmap.png")
    return effects


def build_agent_type_tables(agent_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for dataset in DATASET_ORDER:
        for agent_type in ["AV", "HV"]:
            part = agent_long[
                agent_long["dataset_label"].eq(dataset) & agent_long["agent_type"].eq(agent_type)
            ].copy()
            mean, lo, hi = ci_mean(part["ipv"])
            rows.append(
                {
                    "dataset": dataset,
                    "agent_type": agent_type,
                    "agent_values": int(part["ipv"].notna().sum()),
                    "signed_mean": mean,
                    "signed_ci_low": lo,
                    "signed_ci_high": hi,
                    "signed_median": float(part["ipv"].median()) if len(part) else np.nan,
                    "abs_median": float(part["ipv"].abs().median()) if len(part) else np.nan,
                    "positive_gt_0_05_pct": 100 * float((part["ipv"] > 0.05).mean()) if len(part) else np.nan,
                    "negative_lt_minus_0_05_pct": 100 * float((part["ipv"] < -0.05).mean()) if len(part) else np.nan,
                    "near_zero_abs_lt_0_05_pct": 100 * float((part["ipv"].abs() < 0.05).mean()) if len(part) else np.nan,
                    "strong_abs_ge_0_30_pct": 100 * float((part["ipv"].abs() >= 0.30).mean()) if len(part) else np.nan,
                }
            )
    stats_frame = pd.DataFrame(rows)
    stats_frame.to_csv(TABLE_DIR / "agent_type_dataset_stats.csv", index=False, encoding="utf-8")

    delta_rows = []
    for dataset in DATASET_ORDER:
        part = agent_long[agent_long["dataset_label"].eq(dataset)]
        av = part.loc[part["agent_type"].eq("AV"), "ipv"].dropna()
        hv = part.loc[part["agent_type"].eq("HV"), "ipv"].dropna()
        if len(av) and len(hv):
            u_stat, p_value = stats.mannwhitneyu(av, hv, alternative="two-sided", method="asymptotic")
            abs_u, abs_p = stats.mannwhitneyu(
                av.abs(), hv.abs(), alternative="two-sided", method="asymptotic"
            )
            signed_delta = 2 * u_stat / (len(av) * len(hv)) - 1
            abs_delta = 2 * abs_u / (len(av) * len(hv)) - 1
        else:
            p_value = abs_p = signed_delta = abs_delta = np.nan
        delta_rows.append(
            {
                "dataset": dataset,
                "n_av": int(len(av)),
                "n_hv": int(len(hv)),
                "mean_delta_av_minus_hv": float(av.mean() - hv.mean()) if len(av) and len(hv) else np.nan,
                "positive_delta_pp_av_minus_hv": 100
                * (float((av > 0.05).mean()) - float((hv > 0.05).mean()))
                if len(av) and len(hv)
                else np.nan,
                "negative_delta_pp_av_minus_hv": 100
                * (float((av < -0.05).mean()) - float((hv < -0.05).mean()))
                if len(av) and len(hv)
                else np.nan,
                "near_zero_delta_pp_av_minus_hv": 100
                * (float((av.abs() < 0.05).mean()) - float((hv.abs() < 0.05).mean()))
                if len(av) and len(hv)
                else np.nan,
                "strong_delta_pp_av_minus_hv": 100
                * (float((av.abs() >= 0.30).mean()) - float((hv.abs() >= 0.30).mean()))
                if len(av) and len(hv)
                else np.nan,
                "cliffs_delta_signed_av_vs_hv": signed_delta,
                "p_value_signed_mannwhitney": p_value,
                "cliffs_delta_abs_av_vs_hv": abs_delta,
                "p_value_abs_mannwhitney": abs_p,
            }
        )
    deltas = pd.DataFrame(delta_rows)
    deltas.to_csv(TABLE_DIR / "agent_type_dataset_deltas.csv", index=False, encoding="utf-8")
    return stats_frame, deltas


def plot_agent_type_dataset_lens(agent_long: pd.DataFrame, deltas: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.9), facecolor="#FFFFFF")
    ax_mix, ax_mean, ax_quantile, ax_count = axes.ravel()
    fig.subplots_adjust(top=0.80, hspace=0.54, wspace=0.34)

    delta_metrics = [
        ("prosocial", "positive_delta_pp_av_minus_hv"),
        ("competitive", "negative_delta_pp_av_minus_hv"),
        ("selfish", "near_zero_delta_pp_av_minus_hv"),
        ("strong tail", "strong_delta_pp_av_minus_hv"),
    ]
    x = np.arange(len(delta_metrics))
    width = 0.23
    for i, dataset in enumerate(DATASET_ORDER):
        row = deltas[deltas["dataset"].eq(dataset)].iloc[0]
        values = [row[col] for _, col in delta_metrics]
        bars = ax_mix.bar(
            x + (i - 1) * width,
            values,
            width=width,
            color=DATASET_PALETTE[dataset],
            edgecolor=DATASET_EDGE[dataset],
            linewidth=0.9,
            label=dataset,
        )
        for bar, value in zip(bars, values):
            ax_mix.text(
                bar.get_x() + bar.get_width() / 2,
                value + (0.6 if value >= 0 else -0.8),
                f"{value:+.1f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=7.5,
                color=TOKENS["ink"],
            )
    ax_mix.axhline(0, color=TOKENS["ink"], linewidth=0.9)
    ax_mix.set_xticks(x, [label for label, _ in delta_metrics])
    ax_mix.set_ylabel("AV - HV share difference, pp")
    ax_mix.set_title("Behavior mix difference within each dataset", loc="left", fontsize=10, fontweight="semibold")

    mean_rows = []
    for dataset in DATASET_ORDER:
        for agent_type in ["AV", "HV"]:
            part = agent_long[
                agent_long["dataset_label"].eq(dataset) & agent_long["agent_type"].eq(agent_type)
            ]
            mean, lo, hi = ci_mean(part["ipv"])
            mean_rows.append(
                {"dataset": dataset, "agent_type": agent_type, "mean": mean, "lo": lo, "hi": hi}
            )
    mean_frame = pd.DataFrame(mean_rows)
    y = np.arange(len(DATASET_ORDER))
    for i, dataset in enumerate(DATASET_ORDER):
        part = mean_frame[mean_frame["dataset"].eq(dataset)].set_index("agent_type")
        ax_mean.hlines(i, part.loc["HV", "mean"], part.loc["AV", "mean"], color=NEUTRAL["mid"], linewidth=1.0)
        for agent_type, marker, xoffset in [("HV", "s", -0.03), ("AV", "o", 0.03)]:
            row = part.loc[agent_type]
            face = "#FFFFFF" if agent_type == "HV" else DATASET_PALETTE[dataset]
            ax_mean.errorbar(
                row["mean"],
                i + xoffset,
                xerr=[[row["mean"] - row["lo"]], [row["hi"] - row["mean"]]],
                fmt=marker,
                color=DATASET_EDGE[dataset],
                markerfacecolor=face,
                markeredgecolor=DATASET_EDGE[dataset],
                markersize=6,
                linewidth=1.0,
                capsize=2,
            )
            ax_mean.text(
                row["mean"] + 0.003,
                i + xoffset,
                agent_type,
                va="center",
                fontsize=8,
                color=DATASET_EDGE[dataset],
            )
    ax_mean.axvline(0, color=TOKENS["ink"], linewidth=0.9)
    ax_mean.set_yticks(y, DATASET_ORDER)
    ax_mean.set_xlabel("Mean signed IPV with 95% CI")
    ax_mean.set_title("AV tends to higher signed IPV in AV2 and Waymo", loc="left", fontsize=10, fontweight="semibold")
    ax_mean.set_xlim(-0.035, 0.055)
    ax_mean.invert_yaxis()

    quantiles = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    for dataset in DATASET_ORDER:
        part = agent_long[agent_long["dataset_label"].eq(dataset)]
        av = part.loc[part["agent_type"].eq("AV"), "ipv"]
        hv = part.loc[part["agent_type"].eq("HV"), "ipv"]
        q_delta = av.quantile(quantiles).to_numpy() - hv.quantile(quantiles).to_numpy()
        ax_quantile.plot(
            quantiles * 100,
            q_delta,
            marker="o",
            color=DATASET_EDGE[dataset],
            markerfacecolor=DATASET_PALETTE[dataset],
            linewidth=1.0,
        )
        ax_quantile.text(
            quantiles[-1] * 100 + 1.2,
            q_delta[-1],
            dataset,
            va="center",
            fontsize=8,
            color=DATASET_EDGE[dataset],
        )
    ax_quantile.axhline(0, color=TOKENS["ink"], linewidth=0.9)
    ax_quantile.set_xlabel("Signed IPV quantile")
    ax_quantile.set_ylabel("AV - HV quantile difference")
    ax_quantile.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax_quantile.set_xlim(4, 104)
    ax_quantile.set_title("Where AV/HV signed distributions separate", loc="left", fontsize=10, fontweight="semibold")

    count_frame = agent_long.groupby(["dataset_label", "agent_type"], observed=True).size().reset_index(name="n")
    y = np.arange(len(DATASET_ORDER))
    hv_counts = count_frame[count_frame["agent_type"].eq("HV")].set_index("dataset_label").reindex(DATASET_ORDER)["n"].fillna(0)
    av_counts = count_frame[count_frame["agent_type"].eq("AV")].set_index("dataset_label").reindex(DATASET_ORDER)["n"].fillna(0)
    ax_count.barh(y, hv_counts, color=NEUTRAL["light"], edgecolor=NEUTRAL["mid"], linewidth=0.9, label="HV")
    ax_count.barh(y, av_counts, left=hv_counts, color=FAMILIES["blue"]["base"], edgecolor=FAMILIES["blue"]["dark"], linewidth=0.9, label="AV")
    for i, dataset in enumerate(DATASET_ORDER):
        ax_count.text(hv_counts.iloc[i] + av_counts.iloc[i] + 40, i, f"AV {int(av_counts.iloc[i])} / HV {int(hv_counts.iloc[i])}", va="center", fontsize=8)
    ax_count.set_yticks(y, DATASET_ORDER)
    ax_count.set_xlabel("Completed key-agent values")
    ax_count.set_title("AV sample sizes are smaller, especially nuPlan", loc="left", fontsize=10, fontweight="semibold")
    ax_count.legend(loc="lower right", frameon=False, ncol=2, fontsize=8)
    ax_count.invert_yaxis()

    left = axes[0, 0].get_position().x0
    fig.text(
        left,
        0.98,
        "Key-agent AV vs HV IPV differences by dataset",
        ha="left",
        va="top",
        fontsize=13,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        left,
        0.94,
        "Vehicle type is read from pkl metadata.vehicle_type and aligned to CSV key_agents; all comparisons are within dataset to avoid mixing dataset composition effects.",
        ha="left",
        va="top",
        fontsize=9,
        color=TOKENS["muted"],
    )
    handles = [
        mpl.patches.Patch(facecolor=DATASET_PALETTE[label], edgecolor=DATASET_EDGE[label], label=label)
        for label in DATASET_ORDER
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(left, 0.895), frameon=False, ncol=3, fontsize=8.5)
    sns.despine(fig=fig)
    savefig(fig, "15_agent_type_dataset_lens.png")


def build_av_hv_paired_case_tables(ok: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_mask = (
        ok["key_agent_1_type"].isin(["AV", "HV"])
        & ok["key_agent_2_type"].isin(["AV", "HV"])
        & ok["key_agent_1_type"].ne(ok["key_agent_2_type"])
    )
    paired = ok.loc[pair_mask].copy()
    paired["av_ipv"] = np.where(
        paired["key_agent_1_type"].eq("AV"),
        paired["ipv_key_agent_1_mean"],
        paired["ipv_key_agent_2_mean"],
    )
    paired["hv_ipv"] = np.where(
        paired["key_agent_1_type"].eq("HV"),
        paired["ipv_key_agent_1_mean"],
        paired["ipv_key_agent_2_mean"],
    )
    paired["av_agent_id"] = np.where(
        paired["key_agent_1_type"].eq("AV"),
        paired["key_agent_1_id"],
        paired["key_agent_2_id"],
    )
    paired["hv_agent_id"] = np.where(
        paired["key_agent_1_type"].eq("HV"),
        paired["key_agent_1_id"],
        paired["key_agent_2_id"],
    )
    paired["av_order"] = np.where(paired["key_agent_1_type"].eq("AV"), "agent 1", "agent 2")
    paired["av_minus_hv_ipv"] = paired["av_ipv"] - paired["hv_ipv"]
    paired["abs_pair_gap"] = paired["av_minus_hv_ipv"].abs()

    record_cols = [
        "dataset",
        "dataset_label",
        "folder",
        "scenario_idx",
        "ipv_segment_id",
        "key_agents",
        "track_id",
        "av_agent_id",
        "hv_agent_id",
        "av_order",
        "av_ipv",
        "hv_ipv",
        "av_minus_hv_ipv",
        "abs_pair_gap",
        "path_category",
        "path_relation",
        "turn_label",
        "PET",
        "intensity",
    ]
    record_cols = [col for col in record_cols if col in paired.columns]
    paired[record_cols].to_csv(
        TABLE_DIR / "av_hv_paired_case_records.csv", index=False, encoding="utf-8"
    )

    rows = []
    for label, part in [("Overall", paired)] + [
        (dataset, paired[paired["dataset_label"].astype(str).eq(dataset)]) for dataset in DATASET_ORDER
    ]:
        diffs = part["av_minus_hv_ipv"].dropna()
        av = part["av_ipv"].dropna()
        hv = part["hv_ipv"].dropna()
        mean, lo, hi = ci_mean(diffs)
        if len(diffs) > 0:
            try:
                wilcoxon = stats.wilcoxon(av, hv, zero_method="wilcox", alternative="two-sided", method="auto")
                wilcoxon_p = float(wilcoxon.pvalue)
            except ValueError:
                wilcoxon_p = np.nan
            try:
                spearman = stats.spearmanr(av, hv)
                spearman_rho = float(spearman.statistic)
                spearman_p = float(spearman.pvalue)
            except ValueError:
                spearman_rho = spearman_p = np.nan
        else:
            wilcoxon_p = spearman_rho = spearman_p = np.nan
        rows.append(
            {
                "dataset": label,
                "paired_cases": int(len(part)),
                "av_mean_ipv": float(av.mean()) if len(av) else np.nan,
                "hv_mean_ipv": float(hv.mean()) if len(hv) else np.nan,
                "av_minus_hv_mean": mean,
                "av_minus_hv_ci_low": lo,
                "av_minus_hv_ci_high": hi,
                "av_minus_hv_median": float(diffs.median()) if len(diffs) else np.nan,
                "av_higher_gt_0_05_pct": 100 * float((diffs > 0.05).mean()) if len(diffs) else np.nan,
                "hv_higher_gt_0_05_pct": 100 * float((diffs < -0.05).mean()) if len(diffs) else np.nan,
                "similar_abs_diff_lt_0_05_pct": 100 * float((diffs.abs() < 0.05).mean()) if len(diffs) else np.nan,
                "av_positive_gt_0_05_pct": 100 * float((av > 0.05).mean()) if len(av) else np.nan,
                "hv_positive_gt_0_05_pct": 100 * float((hv > 0.05).mean()) if len(hv) else np.nan,
                "av_negative_lt_minus_0_05_pct": 100 * float((av < -0.05).mean()) if len(av) else np.nan,
                "hv_negative_lt_minus_0_05_pct": 100 * float((hv < -0.05).mean()) if len(hv) else np.nan,
                "wilcoxon_p_value": wilcoxon_p,
                "spearman_rho_av_vs_hv": spearman_rho,
                "spearman_p_value": spearman_p,
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(TABLE_DIR / "av_hv_paired_case_summary.csv", index=False, encoding="utf-8")
    return paired, summary


def plot_av_hv_paired_case_lens(paired: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.0), facecolor="#FFFFFF")
    ax_mean, ax_scatter, ax_dist, ax_mix = axes.ravel()
    fig.subplots_adjust(top=0.80, hspace=0.56, wspace=0.34)

    dataset_rows = summary[summary["dataset"].isin(DATASET_ORDER)].set_index("dataset").reindex(DATASET_ORDER)
    y = np.arange(len(DATASET_ORDER))
    for i, dataset in enumerate(DATASET_ORDER):
        row = dataset_rows.loc[dataset]
        ax_mean.hlines(i, row["hv_mean_ipv"], row["av_mean_ipv"], color=NEUTRAL["mid"], linewidth=1.1)
        ax_mean.scatter(
            row["hv_mean_ipv"],
            i - 0.035,
            marker="s",
            s=48,
            facecolor="#FFFFFF",
            edgecolor=DATASET_EDGE[dataset],
            linewidth=1.0,
            label="HV" if i == 0 else None,
        )
        ax_mean.scatter(
            row["av_mean_ipv"],
            i + 0.035,
            marker="o",
            s=54,
            facecolor=DATASET_PALETTE[dataset],
            edgecolor=DATASET_EDGE[dataset],
            linewidth=1.0,
            label="AV" if i == 0 else None,
        )
        ax_mean.text(
            max(row["av_mean_ipv"], row["hv_mean_ipv"]) + 0.004,
            i,
            f"Δ {row['av_minus_hv_mean']:+.3f}",
            va="center",
            fontsize=8,
            color=DATASET_EDGE[dataset],
        )
    ax_mean.axvline(0, color=TOKENS["ink"], linewidth=0.9)
    ax_mean.set_yticks(y, DATASET_ORDER)
    ax_mean.set_xlabel("Mean signed IPV")
    ax_mean.set_title("Within AV-HV pairs, AV mean IPV is higher", loc="left", fontsize=10, fontweight="semibold")
    ax_mean.set_xlim(-0.04, 0.05)
    ax_mean.invert_yaxis()
    ax_mean.legend(loc="lower right", frameon=False, ncol=2, fontsize=8)

    scatter_limit = 0.35
    for dataset in DATASET_ORDER:
        part = paired[paired["dataset_label"].astype(str).eq(dataset)]
        ax_scatter.scatter(
            part["hv_ipv"].clip(-scatter_limit, scatter_limit),
            part["av_ipv"].clip(-scatter_limit, scatter_limit),
            s=18,
            alpha=0.45,
            facecolor=DATASET_PALETTE[dataset],
            edgecolor=DATASET_EDGE[dataset],
            linewidth=0.35,
            label=dataset,
        )
    ax_scatter.plot([-scatter_limit, scatter_limit], [-scatter_limit, scatter_limit], color=TOKENS["ink"], linewidth=0.9)
    ax_scatter.axhline(0, color=NEUTRAL["mid"], linewidth=0.7, linestyle="--")
    ax_scatter.axvline(0, color=NEUTRAL["mid"], linewidth=0.7, linestyle="--")
    ax_scatter.set_xlim(-scatter_limit, scatter_limit)
    ax_scatter.set_ylim(-scatter_limit, scatter_limit)
    ax_scatter.set_xlabel("HV mean IPV in same case")
    ax_scatter.set_ylabel("AV mean IPV in same case")
    ax_scatter.set_title("Paired AV vs HV IPV, clipped to +/-0.35", loc="left", fontsize=10, fontweight="semibold")
    ax_scatter.legend(loc="lower right", frameon=False, fontsize=8)

    dist = paired.copy()
    dist["dataset_label"] = pd.Categorical(dist["dataset_label"].astype(str), DATASET_ORDER, ordered=True)
    sns.violinplot(
        data=dist,
        x="av_minus_hv_ipv",
        y="dataset_label",
        hue="dataset_label",
        order=DATASET_ORDER,
        hue_order=DATASET_ORDER,
        ax=ax_dist,
        cut=0,
        inner="quartile",
        linewidth=0.8,
        palette={dataset: DATASET_PALETTE[dataset] for dataset in DATASET_ORDER},
        legend=False,
    )
    ax_dist.axvline(0, color=TOKENS["ink"], linewidth=0.9)
    ax_dist.axvline(0.05, color=NEUTRAL["mid"], linewidth=0.7, linestyle="--")
    ax_dist.axvline(-0.05, color=NEUTRAL["mid"], linewidth=0.7, linestyle="--")
    ax_dist.set_xlim(-0.30, 0.30)
    ax_dist.set_xlabel("AV - HV IPV in the same case")
    ax_dist.set_ylabel("")
    ax_dist.set_title("Most paired differences remain near zero", loc="left", fontsize=10, fontweight="semibold")

    mix_cols = [
        ("HV higher", "hv_higher_gt_0_05_pct", FAMILIES["orange"]["base"], FAMILIES["orange"]["dark"]),
        ("similar", "similar_abs_diff_lt_0_05_pct", NEUTRAL["light"], NEUTRAL["mid"]),
        ("AV higher", "av_higher_gt_0_05_pct", FAMILIES["blue"]["base"], FAMILIES["blue"]["dark"]),
    ]
    lefts = np.zeros(len(DATASET_ORDER))
    for label, col, color, edge in mix_cols:
        values = dataset_rows[col].to_numpy(dtype=float)
        bars = ax_mix.barh(y, values, left=lefts, color=color, edgecolor=edge, linewidth=0.8, label=label)
        for bar, value, left in zip(bars, values, lefts):
            if value >= 8:
                ax_mix.text(left + value / 2, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", ha="center", va="center", fontsize=8)
        lefts += values
    ax_mix.set_yticks(y, DATASET_ORDER)
    ax_mix.set_xlim(0, 100)
    ax_mix.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax_mix.set_xlabel("Share of AV-HV paired cases")
    ax_mix.set_title("AV higher is more common, but near-zero is largest", loc="left", fontsize=10, fontweight="semibold")
    ax_mix.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), frameon=False, ncol=3, fontsize=8)
    ax_mix.invert_yaxis()

    left = axes[0, 0].get_position().x0
    fig.text(
        left,
        0.98,
        "Within-case AV-HV paired IPV comparison",
        ha="left",
        va="top",
        fontsize=13,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        left,
        0.94,
        "Only completed cases with exactly one AV and one HV key agent are included; positive AV-HV means the AV is more prosocial or less competitive than the HV in the same interaction.",
        ha="left",
        va="top",
        fontsize=9,
        color=TOKENS["muted"],
    )
    sns.despine(fig=fig)
    savefig(fig, "16_av_hv_paired_case_lens.png")


def dataset_tests_html() -> str:
    tests = pd.read_csv(TABLE_DIR / "dataset_pairwise_tests.csv")
    table = tests[tests["comparison"].eq("Kruskal-Wallis all datasets")].copy()
    table["p"] = table["p_value"].map(p_text)
    table = table[["metric", "n", "statistic", "p"]].rename(
        columns={
            "metric": "Metric",
            "n": "N",
            "statistic": "Kruskal-Wallis statistic",
            "p": "P value",
        }
    )
    table["Metric"] = table["Metric"].map(
        {
            "agent_signed_ipv": "Agent-level signed IPV",
            "case_pair_abs_ipv": "Case-level pair |IPV|",
            "case_pair_mean_ipv": "Case-level pair mean IPV",
            "case_agent_asymmetry": "Agent1 - agent2 asymmetry",
            "case_mean_error": "Mean IPV error",
        }
    )
    for col in table.columns:
        if pd.api.types.is_float_dtype(table[col]):
            table[col] = table[col].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
        elif pd.api.types.is_integer_dtype(table[col]):
            table[col] = table[col].map(lambda v: "" if pd.isna(v) else f"{int(v):,}")
    return table.to_html(index=False, classes="data-table", border=0, escape=True)


def build_html(summary_json: dict, inside_pct: float, ok: pd.DataFrame, agent_long: pd.DataFrame) -> str:
    completed = int(summary_json.get("completed_cases", len(ok)))
    missing = int(summary_json.get("missing_cases", 0))
    agent_values = int(summary_json.get("agent_values", len(agent_long)))
    near_zero = pct(float(summary_json.get("near_zero_pct", (agent_long["ipv"].abs() < 0.05).mean() * 100)))
    positive = pct(float(summary_json.get("positive_pct", (agent_long["ipv"] > 0.05).mean() * 100)))
    negative = pct(float(summary_json.get("negative_pct", (agent_long["ipv"] < -0.05).mean() * 100)))
    strong = pct(
        float(summary_json.get("strong_abs_ge_0_30_pct", (agent_long["ipv"].abs() >= 0.30).mean() * 100))
    )
    overall_mean = float(summary_json.get("overall_signed_mean", agent_long["ipv"].mean()))
    overall_median = float(summary_json.get("overall_signed_median", agent_long["ipv"].median()))

    figures = [
        figure_card("01", "数据覆盖率", "01_coverage_by_dataset.png", "Waymo 与 AV2 已完整覆盖；nuPlan 仍有缺失，因此跨数据集比较需要把 nuPlan 的缺失偏差单独看待。"),
        figure_card("02", "整体 key-agent IPV 分布", "02_overall_ipv_distribution.png", "IPV 明显集中在零附近，near-zero 是主体；长尾案例可作为后续人工复核和机制解释的重点。"),
        figure_card("03", "按数据集分组的 IPV 分布", "03_dataset_violin_ipv.png", "小提琴图展示各数据集的分布形态差异；nuPlan 当前样本未完全覆盖，解释时应避免过度外推。"),
        figure_card("04", "两个 key agents 的 IPV 配对关系", "04_pair_ipv_hexbin.png", f"图 4 已缩放到 [-0.30, 0.30] 的核心范围，覆盖 {inside_pct:.1f}% 的完成案例，并使用 log-density 色标显示零附近密集点。"),
        figure_card("05", "case-level IPV 指标分布", "05_pair_metric_distributions.png", "从 pair mean、pair |IPV|、agent1-agent2 asymmetry 和 max |IPV| 四个角度观察单案例强度。"),
        figure_card("06", "IPV 符号组成", "06_pair_sign_composition.png", "按分组比较 positive、negative 和 near-zero 的比例，用于判断不同场景类型是否存在方向性偏移。"),
        figure_card("07", "Path category 与 IPV", "07_path_category_forest.png", "点为均值，横线为置信区间；用于快速定位哪些 path category 的平均 IPV 明显偏离零。"),
        figure_card("08", "Path relation 与 IPV", "08_path_relation_forest.png", "更细粒度地展示 path relation 对 IPV 的影响，样本量较小的 relation 需要谨慎解释。"),
        figure_card("09", "Turn label 与 IPV", "09_turn_label_forest.png", "按转向组合观察 IPV，帮助判断直行、左转、右转组合是否带来不同交互倾向。"),
        figure_card("10", "交互类型与 AV 参与情况", "10_interaction_type_av_forest.png", "比较 two/multi-agent 案例以及是否包含 AV 的均值和置信区间。"),
        figure_card("11", "PET 与 intensity 的分箱关系", "11_pet_intensity_binned_abs_ipv.png", "展示风险代理指标与 pair |IPV| 的关系；intensity 使用 log10 以降低极端值影响。"),
        figure_card("12", "估计误差分布及其与 |IPV| 的关系", "12_error_distribution_and_abs_ipv.png", "估计误差用于辅助判断结果可靠性；误差和 |IPV| 的关系可以提示哪些强交互估计更稳定。"),
    ]

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>subsets_for_yiru IPV Distribution Report</title>
<style>
:root {{ --ink: #111111; --muted: #595959; --line: #d8d8d8; --soft: #f7f7f7; --blue: #0072B2; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; color: var(--ink); background: #ffffff; font-family: Arial, "Helvetica Neue", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif; line-height: 1.58; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 34px 28px 64px; }}
header {{ border-bottom: 1.2px solid var(--ink); padding-bottom: 18px; margin-bottom: 26px; }}
h1 {{ font-size: 28px; line-height: 1.18; margin: 0 0 10px; font-weight: 700; letter-spacing: 0; }}
h2 {{ font-size: 19px; margin: 36px 0 14px; padding-top: 12px; border-top: 1px solid var(--line); }}
h3 {{ font-size: 14px; margin: 0 0 10px; font-weight: 700; }}
p {{ margin: 8px 0 12px; }}
.note {{ color: var(--muted); font-size: 13px; }}
.grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 18px 0 20px; }}
.metric {{ border-top: 2px solid var(--ink); padding: 10px 8px 8px; background: #fff; }}
.metric .value {{ font-size: 23px; font-weight: 700; margin-bottom: 2px; }}
.metric .label {{ color: var(--muted); font-size: 12px; }}
ul {{ padding-left: 20px; }}
li {{ margin: 6px 0; }}
.figure-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 24px; }}
.figure-card {{ margin: 0; break-inside: avoid; }}
.figure-card.wide {{ grid-column: 1 / -1; }}
.figure-card img {{ width: 100%; display: block; border: 1px solid #e2e2e2; background: #fff; }}
.caption {{ color: var(--muted); font-size: 12px; margin-top: 7px; }}
table.data-table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin: 8px 0 22px; }}
.data-table th {{ text-align: left; border-top: 1.4px solid #111; border-bottom: 1px solid var(--line); padding: 8px 7px; background: #fff; }}
.data-table td {{ border-bottom: 1px solid #e6e6e6; padding: 7px; vertical-align: top; }}
.data-table td:not(:first-child), .data-table th:not(:first-child) {{ text-align: right; }}
.table-wrap {{ width: 100%; overflow-x: auto; margin: 8px 0 22px; }}
.definition-table {{ min-width: 1180px; font-size: 12px; line-height: 1.42; }}
.definition-table td:not(:first-child), .definition-table th:not(:first-child) {{ text-align: left; }}
.definition-table td:nth-child(1) {{ white-space: nowrap; color: var(--muted); }}
.definition-table td:nth-child(2), .definition-table td:nth-child(3), .definition-table td:nth-child(4) {{ white-space: nowrap; }}
.callout {{ border-left: 4px solid var(--ink); background: var(--soft); padding: 12px 14px; margin: 22px 0; }}
.code {{ font-family: Consolas, Monaco, monospace; }}
a {{ color: var(--blue); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
@media (max-width: 900px) {{
  main {{ padding: 24px 18px 52px; }}
  .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .figure-grid {{ grid-template-columns: 1fr; }}
  .figure-card.wide {{ grid-column: auto; }}
}}
</style>
</head>
<body>
<main>
<header>
  <h1>subsets_for_yiru IPV 分布分析报告</h1>
  <p class="note">数据来源：已从 <span class="code">cases.zip</span> 匹配到 CSV 的现有结果。报告采用接近 Nature 期刊的白底、细线、低饱和配色和紧凑排版；所有分布统计只使用 <span class="code">ipv_result_status = ok</span> 的 case。</p>
  <p class="note">生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}；匹配 CSV：<span class="code">{html.escape(rel(MATCHED_CSV))}</span></p>
</header>

<section class="grid">
  <div class="metric"><div class="value">{completed:,}</div><div class="label">已完成 case</div></div>
  <div class="metric"><div class="value">{missing:,}</div><div class="label">缺失 case</div></div>
  <div class="metric"><div class="value">{agent_values:,}</div><div class="label">key-agent IPV 值</div></div>
  <div class="metric"><div class="value">{near_zero}</div><div class="label">near-zero 比例，|IPV| &lt; 0.05</div></div>
</section>

<h2>核心发现</h2>
<ul>
  <li>当前 zip 中共有 <strong>{completed:,}</strong> 个完成 case，可形成 <strong>{agent_values:,}</strong> 个 key-agent 级 IPV 值；另有 <strong>{missing:,}</strong> 个 case 尚缺结果，全部来自 nuPlan。</li>
  <li>整体 IPV 高度集中在 0 附近：near-zero 比例为 <strong>{near_zero}</strong>，signed mean 为 <strong>{num(overall_mean)}</strong>，signed median 为 <strong>{num(overall_median)}</strong>。</li>
  <li>正向 IPV 比例为 <strong>{positive}</strong>，负向 IPV 比例为 <strong>{negative}</strong>；强 IPV，即 |IPV| ≥ 0.30，仅占 <strong>{strong}</strong>。</li>
  <li>图 4 已改为核心范围密度图：坐标范围固定为 <strong>[-0.30, 0.30]</strong>，覆盖 <strong>{inside_pct:.1f}%</strong> 的完成案例，并用 log-density 色标解决零附近点云过密的问题。</li>
  <li>数据集对比已改为差异导向视图：重点看各数据集相对整体/相对 AV2 的偏移、分位数差异、尾部概率差异和 pairwise effect size，而不是只重复展示三个相近分布。</li>
</ul>

<h2>指标释义与计算方式</h2>
<p class="note">下表给出报告中主要英文指标的中文对应、统计粒度、含义和计算方式。公式中 <span class="code">k1</span> 表示 <span class="code">ipv_key_agent_1_mean</span>，<span class="code">k2</span> 表示 <span class="code">ipv_key_agent_2_mean</span>，<span class="code">v</span> 表示 agent-level signed IPV。除覆盖率外，IPV 分布统计默认只使用 <span class="code">ipv_result_status = ok</span> 的完成案例。</p>
<div class="table-wrap">
{metric_definition_table_html()}
</div>

<h2>数据集差异总览</h2>
<p class="note">本节把数据集之间的差异集中展示，重点回答“这些分布虽然相近，但到底哪里不一样”。统计均基于已完成 case；nuPlan 的缺失比例较高，跨数据集结论需要等 nuPlan 补齐后再最终确认。</p>
<div class="figure-grid">
{figure_card('13', '数据集差异镜头', '13_dataset_difference_lens.png', '把 raw distribution 转成差值视角：相对整体的比例/均值偏移、相对 AV2 的分位数差异，以及 pair |IPV| 尾部概率差异。', wide=True)}
{figure_card('14', '数据集两两效应量热图', '14_dataset_effect_size_heatmap.png', '用 Cliff’s delta 显示系统性分布偏移；数值为正表示列名中的第一个数据集在该指标上整体更大。', wide=True)}
</div>

<h3>数据集对比汇总表</h3>
{table_html('dataset_comparison_summary.csv')}

<h3>数据集差异显著性检验</h3>
{dataset_tests_html()}
<p class="note">注：显著性检验用于提示分布差异是否明显，不直接说明机制原因；完整 pairwise Mann-Whitney 检验结果已保存到 <span class="code">{html.escape(rel(TABLE_DIR / 'dataset_pairwise_tests.csv'))}</span>，effect-size 汇总保存到 <span class="code">{html.escape(rel(TABLE_DIR / 'dataset_effect_size_summary.csv'))}</span>。</p>

<h2>可视化结果</h2>
<div class="figure-grid">
{''.join(figures)}
</div>

<h2>主要统计表</h2>
<h3>按数据集汇总的 agent-level IPV</h3>
{table_html('dataset_agent_stats.csv')}

<h3>按 two/multi 汇总的 agent-level IPV</h3>
{table_html('two_multi_agent_stats.csv')}

<h3>按 path category 汇总的 agent-level IPV</h3>
{table_html('path_category_agent_stats.csv')}

<h3>PET、intensity、error 与 IPV 的 Spearman 相关</h3>
{table_html('spearman_correlations.csv')}

<p class="note">注：以上表格只展示报告正文所需的核心结果，完整表格已保存到 <span class="code">{html.escape(rel(TABLE_DIR))}</span>。</p>

<h2>解释和后续建议</h2>
<div class="callout">
  <p><strong>IPV 分布解释：</strong>当前结果显示大多数交互案例的 IPV 接近 0，说明在多数片段中两车的策略倾向并不强烈偏向一方。长尾的高 |IPV| 案例更适合作为交互行为机制分析、异常案例复核和论文展示案例。</p>
  <p><strong>数据覆盖解释：</strong>Waymo 与 AV2 已完整覆盖，nuPlan 仍有 738 个缺失 case。由于 nuPlan 此前涉及 20Hz 到 10Hz 的重算逻辑，当前报告适合用于已有结果的初步分布分析；最终跨数据集结论建议等 nuPlan 补齐后再确认。</p>
  <p><strong>方法注意：</strong>CSV 中 IPV 均值按原始 <span class="code">key_agents</span> 顺序对应，agent 1 和 agent 2 的均值差异不应直接解释为某一物理角色差异，除非再结合角色、优先权、路径关系或是否为 AV。</p>
</div>

<h2>输出文件</h2>
<ul>
  <li>HTML 报告：<span class="code">{html.escape(rel(HTML_PATH))}</span></li>
  <li>图像目录：<span class="code">{html.escape(rel(FIG_DIR))}</span></li>
  <li>统计表目录：<span class="code">{html.escape(rel(TABLE_DIR))}</span></li>
  <li>摘要 JSON：<span class="code">{html.escape(rel(SUMMARY_PATH))}</span></li>
  <li>指标定义表：<span class="code">{html.escape(rel(TABLE_DIR / 'metric_definitions.csv'))}</span></li>
</ul>
</main>
</body>
</html>
"""


def format_report_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
        elif pd.api.types.is_integer_dtype(display[col]):
            display[col] = display[col].map(lambda v: "" if pd.isna(v) else f"{int(v):,}")
    return display.to_html(index=False, classes="data-table", border=0, escape=True)


def figure_block(number: str, title: str, image_name: str, caption: str, wide: bool = False) -> str:
    class_name = "report-figure wide" if wide else "report-figure"
    return f"""
    <figure class="{class_name}">
      <h4>{html.escape(number)}. {html.escape(title)}</h4>
      <img src="{img_data(image_name)}" alt="{html.escape(title)}">
      <figcaption>{html.escape(caption)}</figcaption>
    </figure>
    """


def conclusion_link(anchor: str, title: str, summary: str) -> str:
    return f"""
    <a class="index-card" href="#{html.escape(anchor)}">
      <strong>{html.escape(title)}</strong>
      <span>{html.escape(summary)}</span>
    </a>
    """


def build_conclusion_html(
    summary_json: dict, inside_pct: float, ok: pd.DataFrame, agent_long: pd.DataFrame
) -> str:
    completed = int(summary_json.get("completed_cases", len(ok)))
    missing = int(summary_json.get("missing_cases", 0))
    agent_values = int(summary_json.get("agent_values", len(agent_long)))
    overall_mean = float(summary_json.get("overall_signed_mean", agent_long["ipv"].mean()))
    near_zero_pct = float(summary_json.get("near_zero_pct", (agent_long["ipv"].abs() < 0.05).mean() * 100))
    positive_pct = float(summary_json.get("positive_pct", (agent_long["ipv"] > 0.05).mean() * 100))
    negative_pct = float(summary_json.get("negative_pct", (agent_long["ipv"] < -0.05).mean() * 100))
    strong_pct = float(
        summary_json.get("strong_abs_ge_0_30_pct", (agent_long["ipv"].abs() >= 0.30).mean() * 100)
    )

    dataset_table = pd.read_csv(TABLE_DIR / "dataset_comparison_summary.csv")
    context_table = pd.concat(
        [
            pd.read_csv(TABLE_DIR / "two_multi_agent_stats.csv").assign(对比维度="two/multi").rename(
                columns={"two/multi": "分组"}
            ),
            pd.read_csv(TABLE_DIR / "av_agent_stats.csv").assign(对比维度="AV included").rename(
                columns={"AV_included": "分组"}
            ),
        ],
        ignore_index=True,
    )
    context_table = context_table[
        ["对比维度", "分组", "n", "mean", "positive_%", "negative_%", "near_zero_%"]
    ].rename(
        columns={
            "n": "agent values",
            "mean": "mean IPV",
            "positive_%": "亲社会比例 >0.05 (%)",
            "negative_%": "竞争比例 <-0.05 (%)",
            "near_zero_%": "利己比例 |IPV|<0.05 (%)",
        }
    )
    path_table = pd.read_csv(TABLE_DIR / "path_category_agent_stats.csv")
    path_table = path_table[
        ["path_category", "n", "mean", "positive_%", "negative_%", "near_zero_%"]
    ].rename(
        columns={
            "path_category": "path category",
            "n": "agent values",
            "mean": "mean IPV",
            "positive_%": "亲社会比例 >0.05 (%)",
            "negative_%": "竞争比例 <-0.05 (%)",
            "near_zero_%": "利己比例 |IPV|<0.05 (%)",
        }
    )
    spearman = pd.read_csv(TABLE_DIR / "spearman_correlations.csv").rename(
        columns={
            "relationship": "关系",
            "n": "case 数",
            "spearman_rho": "Spearman rho",
            "p_value": "p value",
        }
    )
    agent_type_stats = pd.read_csv(TABLE_DIR / "agent_type_dataset_stats.csv")
    agent_type_deltas = pd.read_csv(TABLE_DIR / "agent_type_dataset_deltas.csv")
    av_hv_paired_summary = pd.read_csv(TABLE_DIR / "av_hv_paired_case_summary.csv")
    paired_overall = av_hv_paired_summary[av_hv_paired_summary["dataset"].eq("Overall")].iloc[0]

    dataset_display = dataset_table[
        [
            "dataset",
            "completed_cases",
            "missing_cases",
            "completion_pct",
            "signed_mean",
            "positive_gt_0_05_pct",
            "negative_lt_minus_0_05_pct",
            "near_zero_abs_lt_0_05_pct",
            "strong_abs_ge_0_30_pct",
            "pair_abs_mean",
        ]
    ].rename(
        columns={
            "dataset": "数据集",
            "completed_cases": "已完成 case",
            "missing_cases": "缺失 case",
            "completion_pct": "完成率 (%)",
            "signed_mean": "mean IPV",
            "positive_gt_0_05_pct": "亲社会比例 (%)",
            "negative_lt_minus_0_05_pct": "竞争比例 (%)",
            "near_zero_abs_lt_0_05_pct": "利己比例 (%)",
            "strong_abs_ge_0_30_pct": "强尾部比例 (%)",
            "pair_abs_mean": "pair |IPV| mean",
        }
    )
    agent_type_display = agent_type_stats[
        [
            "dataset",
            "agent_type",
            "agent_values",
            "signed_mean",
            "abs_median",
            "positive_gt_0_05_pct",
            "negative_lt_minus_0_05_pct",
            "near_zero_abs_lt_0_05_pct",
            "strong_abs_ge_0_30_pct",
        ]
    ].rename(
        columns={
            "dataset": "数据集",
            "agent_type": "key-agent 类型",
            "agent_values": "agent values",
            "signed_mean": "mean IPV",
            "abs_median": "median |IPV|",
            "positive_gt_0_05_pct": "亲社会比例 (%)",
            "negative_lt_minus_0_05_pct": "竞争比例 (%)",
            "near_zero_abs_lt_0_05_pct": "利己比例 (%)",
            "strong_abs_ge_0_30_pct": "强尾部比例 (%)",
        }
    )
    agent_type_delta_display = agent_type_deltas[
        [
            "dataset",
            "n_av",
            "n_hv",
            "mean_delta_av_minus_hv",
            "positive_delta_pp_av_minus_hv",
            "negative_delta_pp_av_minus_hv",
            "near_zero_delta_pp_av_minus_hv",
            "strong_delta_pp_av_minus_hv",
            "cliffs_delta_signed_av_vs_hv",
            "p_value_signed_mannwhitney",
        ]
    ].copy()
    agent_type_delta_display["p_value_signed_mannwhitney"] = agent_type_delta_display[
        "p_value_signed_mannwhitney"
    ].map(p_text)
    agent_type_delta_display = agent_type_delta_display.rename(
        columns={
            "dataset": "数据集",
            "n_av": "AV agent values",
            "n_hv": "HV agent values",
            "mean_delta_av_minus_hv": "AV-HV mean IPV",
            "positive_delta_pp_av_minus_hv": "亲社会比例差 pp",
            "negative_delta_pp_av_minus_hv": "竞争比例差 pp",
            "near_zero_delta_pp_av_minus_hv": "利己比例差 pp",
            "strong_delta_pp_av_minus_hv": "强尾部比例差 pp",
            "cliffs_delta_signed_av_vs_hv": "Cliff's delta",
            "p_value_signed_mannwhitney": "Mann-Whitney p",
        }
    )
    av_hv_paired_display = av_hv_paired_summary[
        [
            "dataset",
            "paired_cases",
            "av_mean_ipv",
            "hv_mean_ipv",
            "av_minus_hv_mean",
            "av_minus_hv_ci_low",
            "av_minus_hv_ci_high",
            "av_minus_hv_median",
            "av_higher_gt_0_05_pct",
            "hv_higher_gt_0_05_pct",
            "similar_abs_diff_lt_0_05_pct",
            "wilcoxon_p_value",
            "spearman_rho_av_vs_hv",
        ]
    ].copy()
    av_hv_paired_display["wilcoxon_p_value"] = av_hv_paired_display["wilcoxon_p_value"].map(p_text)
    av_hv_paired_display = av_hv_paired_display.rename(
        columns={
            "dataset": "数据集",
            "paired_cases": "AV-HV paired cases",
            "av_mean_ipv": "AV mean IPV",
            "hv_mean_ipv": "HV mean IPV",
            "av_minus_hv_mean": "AV-HV mean IPV",
            "av_minus_hv_ci_low": "CI low",
            "av_minus_hv_ci_high": "CI high",
            "av_minus_hv_median": "AV-HV median IPV",
            "av_higher_gt_0_05_pct": "AV 高于 HV >0.05 (%)",
            "hv_higher_gt_0_05_pct": "HV 高于 AV >0.05 (%)",
            "similar_abs_diff_lt_0_05_pct": "近零差异 |Δ|<0.05 (%)",
            "wilcoxon_p_value": "paired Wilcoxon p",
            "spearman_rho_av_vs_hv": "AV-HV Spearman rho",
        }
    )

    index_cards = [
        conclusion_link("c1", "C1. 利己性是主体", "65.7% 的 key-agent IPV 接近 0，亲社会和竞争都是少数。"),
        conclusion_link("c2", "C2. 数据集有不同社会性风格", "AV2 更利己，Waymo 更亲社会，nuPlan 当前结果更竞争且尾部更重。"),
        conclusion_link("c3", "C3. AV/HV 差异因数据集而异", "agent 级和 AV-HV paired case 都显示 AV 更偏亲社会，但大量配对差异仍近零。"),
        conclusion_link("c4", "C4. F 类路径关系最竞争", "F 类 negative 比例和 strong tail 明显高于其他 path category。"),
        conclusion_link("c5", "C5. 强社会性案例少但最值得复核", "强尾部只占 2.7%，但对机制解释和案例展示最有价值。"),
    ]

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IPV 行为倾向结论报告</title>
<style>
:root {{ --ink:#111; --muted:#5f6675; --line:#d8dce6; --soft:#f7f8fb; --blue:#2E4780; --orange:#804126; --green:#386411; }}
* {{ box-sizing: border-box; }}
html {{ scroll-behavior: smooth; }}
body {{ margin:0; color:var(--ink); background:#fff; font-family: Arial, "Helvetica Neue", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif; line-height:1.62; }}
main {{ max-width:1180px; margin:0 auto; padding:34px 28px 70px; }}
header {{ border-bottom:1.4px solid var(--ink); padding-bottom:18px; margin-bottom:28px; }}
h1 {{ font-size:30px; line-height:1.18; margin:0 0 10px; letter-spacing:0; }}
h2 {{ font-size:21px; line-height:1.25; margin:42px 0 14px; padding-top:14px; border-top:1px solid var(--line); }}
h3 {{ font-size:16px; margin:22px 0 10px; }}
h4 {{ font-size:14px; margin:0 0 8px; }}
p {{ margin:8px 0 12px; }}
.note {{ color:var(--muted); font-size:13px; }}
.summary {{ display:grid; gap:10px; margin:16px 0 22px; }}
.summary p {{ margin:0; padding:12px 14px; background:var(--soft); border-left:4px solid var(--ink); }}
.index-grid {{ display:grid; grid-template-columns:repeat(5, minmax(0,1fr)); gap:10px; margin:14px 0 26px; }}
.index-card {{ display:block; color:var(--ink); text-decoration:none; border-top:2px solid var(--ink); background:#fff; padding:10px 9px 11px; min-height:112px; }}
.index-card strong {{ display:block; font-size:13px; line-height:1.25; margin-bottom:8px; }}
.index-card span {{ color:var(--muted); font-size:12px; line-height:1.35; }}
.kpi-row {{ display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); gap:10px; margin:14px 0 18px; }}
.kpi {{ border-top:2px solid var(--ink); padding:9px 8px; background:#fff; }}
.kpi .value {{ font-size:22px; font-weight:700; }}
.kpi .label {{ color:var(--muted); font-size:12px; }}
.finding {{ margin-top:34px; }}
.takeaway {{ font-size:16px; font-weight:700; }}
.evidence-box {{ background:var(--soft); border-left:4px solid var(--ink); padding:12px 14px; margin:14px 0 18px; }}
.evidence-box ul {{ margin:6px 0 0; padding-left:20px; }}
.evidence-box li {{ margin:4px 0; }}
.fig-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:22px; margin:16px 0 22px; }}
.report-figure {{ margin:0; }}
.report-figure img {{ width:100%; display:block; border:1px solid #e2e4ea; background:#fff; }}
.report-figure figcaption {{ color:var(--muted); font-size:12px; margin-top:7px; }}
.wide {{ grid-column:1 / -1; }}
table.data-table {{ width:100%; border-collapse:collapse; font-size:13px; margin:8px 0 22px; }}
.data-table th {{ text-align:left; border-top:1.4px solid #111; border-bottom:1px solid var(--line); padding:8px 7px; background:#fff; }}
.data-table td {{ border-bottom:1px solid #e6e8ee; padding:7px; vertical-align:top; }}
.data-table td:not(:first-child), .data-table th:not(:first-child) {{ text-align:right; }}
.table-wrap {{ width:100%; overflow-x:auto; margin:8px 0 22px; }}
.definition-table {{ min-width:1180px; font-size:12px; line-height:1.42; }}
.definition-table td:not(:first-child), .definition-table th:not(:first-child) {{ text-align:left; }}
.definition-table td:nth-child(1), .definition-table td:nth-child(2), .definition-table td:nth-child(3), .definition-table td:nth-child(4) {{ white-space:nowrap; }}
.code {{ font-family:Consolas, Monaco, monospace; }}
.callout {{ border-left:4px solid var(--ink); background:var(--soft); padding:12px 14px; margin:18px 0; }}
@media (max-width: 900px) {{
  main {{ padding:24px 18px 56px; }}
  .index-grid, .kpi-row, .fig-grid {{ grid-template-columns:1fr; }}
  .wide {{ grid-column:auto; }}
}}
</style>
</head>
<body>
<main>
<header>
  <h1>IPV 行为倾向结论报告</h1>
  <p class="note">范围：基于 <span class="code">cases.zip</span> 已匹配到 CSV 的现有结果；统计只使用 <span class="code">ipv_result_status = ok</span> 的完成案例。IPV 正值表示亲社会性，越大越亲社会；负值表示竞争性，越小越竞争；接近 0 表示利己性。</p>
  <p class="note">生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}；完成案例 {completed:,} 个，key-agent IPV 值 {agent_values:,} 个；nuPlan 当前仍缺失 {missing:,} 个 case。</p>
</header>

<section>
  <h2>Executive Summary</h2>
  <div class="summary">
    <p><strong>总体以利己性为主。</strong> 已完成样本中，<strong>{near_zero_pct:.1f}%</strong> 的 key-agent IPV 落在 <span class="code">|IPV| &lt; 0.05</span>，亲社会比例为 <strong>{positive_pct:.1f}%</strong>，竞争比例为 <strong>{negative_pct:.1f}%</strong>。整体 mean IPV 为 <strong>{overall_mean:+.3f}</strong>，说明总体仅略偏亲社会。</p>
    <p><strong>数据集之间的行为风格不同。</strong> AV2 更集中在利己区间，Waymo 更偏亲社会，nuPlan 当前结果更偏竞争且强交互尾部更重；但 nuPlan 只完成 1,762/2,500 个 case，所以该差异需要补齐后复核。</p>
    <p><strong>AV/HV 差异需要在 key-agent 层面、分数据集看。</strong> pkl 的 <span class="code">metadata.vehicle_type</span> 显示，AV2 和 Waymo 中 AV key agents 更偏亲社会；nuPlan 中 AV/HV 的 signed IPV 差异较弱，且 AV 样本只有 188 个，应谨慎解释。</p>
    <p><strong>强亲社会或强竞争案例很少，但最值得分析。</strong> <span class="code">|IPV| ≥ 0.30</span> 仅占 <strong>{strong_pct:.1f}%</strong>，这些尾部 case 更适合做机制解释、案例筛选和论文图示。</p>
  </div>
</section>

<section>
  <h2>结论索引</h2>
  <div class="index-grid">
    {''.join(index_cards)}
  </div>
</section>

<section>
  <h2>读图约定：IPV 如何对应行为倾向</h2>
  <p><strong>本报告把 IPV 解释为社会性取向。</strong> <span class="code">IPV &gt; 0.05</span> 记为明显亲社会，<span class="code">IPV &lt; -0.05</span> 记为明显竞争，<span class="code">|IPV| &lt; 0.05</span> 记为近零/利己。阈值用于报告层面的分组描述，不等于模型估计过程中的硬边界。</p>
  <div class="kpi-row">
    <div class="kpi"><div class="value">{near_zero_pct:.1f}%</div><div class="label">利己性，|IPV| &lt; 0.05</div></div>
    <div class="kpi"><div class="value">{positive_pct:.1f}%</div><div class="label">亲社会性，IPV &gt; 0.05</div></div>
    <div class="kpi"><div class="value">{negative_pct:.1f}%</div><div class="label">竞争性，IPV &lt; -0.05</div></div>
    <div class="kpi"><div class="value">{strong_pct:.1f}%</div><div class="label">强尾部，|IPV| ≥ 0.30</div></div>
  </div>
</section>

<section class="finding" id="c1">
  <h2>C1. 大多数交互是利己性的，亲社会与竞争都是少数</h2>
  <p class="takeaway">最核心的整体结论是：车辆交互并不是普遍强亲社会或强竞争，而是大多落在近零 IPV 区间。</p>
  <div class="evidence-box">
    <strong>支撑信息</strong>
    <ul>
      <li>利己性比例 <strong>{near_zero_pct:.1f}%</strong>，远高于亲社会 <strong>{positive_pct:.1f}%</strong> 和竞争 <strong>{negative_pct:.1f}%</strong>。</li>
      <li>整体 mean IPV = <strong>{overall_mean:+.3f}</strong>，只有很弱的亲社会偏移。</li>
      <li>强尾部 <span class="code">|IPV| ≥ 0.30</span> 仅 <strong>{strong_pct:.1f}%</strong>，说明强社会性案例是少数。</li>
    </ul>
  </div>
  <div class="fig-grid">
    {figure_block('02', '整体 key-agent IPV 分布', '02_overall_ipv_distribution.png', '用于看总体分布形态：密度集中在 0 附近，长尾较少。')}
    {figure_block('06', 'IPV 符号组成', '06_pair_sign_composition.png', '用于看亲社会、竞争、利己三类在不同分组中的比例。')}
    {figure_block('04', '两个 key agents 的 IPV 配对关系', '04_pair_ipv_hexbin.png', f'核心范围 [-0.30, 0.30] 覆盖 {inside_pct:.1f}% 完成案例，可看到两车组合主要集中在零附近。')}
    {figure_block('05', 'case-level IPV 指标分布', '05_pair_metric_distributions.png', '用于确认 case-level mean、pair |IPV| 和 asymmetry 的整体强度与尾部。')}
  </div>
</section>

<section class="finding" id="c2">
  <h2>C2. 数据集之间不是同一种行为风格：AV2 更利己，Waymo 更亲社会，nuPlan 当前更竞争</h2>
  <p class="takeaway">如果只看三个小提琴图，它们会显得相近；但用差异视角看，三个数据集的社会性倾向分化很清楚。</p>
  <div class="evidence-box">
    <strong>支撑信息</strong>
    <ul>
      <li>AV2 的利己性最高：near-zero <strong>71.5%</strong>，强尾部仅 <strong>1.2%</strong>。</li>
      <li>Waymo 的亲社会比例最高：<strong>21.2%</strong>，mean IPV = <strong>+0.014</strong>。</li>
      <li>nuPlan 的竞争比例最高：<strong>24.3%</strong>，强尾部 <strong>5.8%</strong>，pair |IPV| mean 也最高。</li>
    </ul>
  </div>
  <div class="fig-grid">
    {figure_block('13', '数据集差异镜头', '13_dataset_difference_lens.png', '把分布差异转成相对总体/相对 AV2 的偏移，突出差异所在。')}
    {figure_block('14', '数据集两两效应量热图', '14_dataset_effect_size_heatmap.png', '用 Cliff’s delta 显示系统性分布偏移，补充 p value 的不足。')}
    {figure_block('01', '数据覆盖率', '01_coverage_by_dataset.png', '用于提醒 nuPlan 当前缺失较多，跨数据集结论需要补齐后复核。')}
    {figure_block('03', '按数据集分组的 IPV 分布', '03_dataset_violin_ipv.png', '保留 raw distribution 视角，用于和差异镜头相互校验。')}
  </div>
  <h3>数据集证据表</h3>
  {format_report_table(dataset_display)}
  <p class="note">解释重点：nuPlan 的结果方向性最强，但它当前完成率只有 70.5%，因此更适合表述为“当前已完成样本显示 nuPlan 更竞争且尾部更重”。</p>
</section>

<section class="finding" id="c3">
  <h2>C3. 在 key-agent 层面，AV/HV 的 IPV 分布差异因数据集而异</h2>
  <p class="takeaway">pkl 中的 <span class="code">metadata.vehicle_type</span> 可以把每个关键交互对象标为 AV 或 HV。按 agent 级身份看，AV2 和 Waymo 的 AV 更偏亲社会；进一步限制到同一 case 内一个 AV、一个 HV 的配对案例后，AV 的 IPV 仍平均高于 HV，但差异主要来自一部分右尾案例，而不是每一对都显著分开。</p>
  <div class="evidence-box">
    <strong>支撑信息</strong>
    <ul>
      <li>AV2：AV mean IPV = <strong>+0.018</strong>，HV = <strong>+0.002</strong>；AV 亲社会比例 <strong>28.8%</strong>，明显高于 HV 的 <strong>13.9%</strong>，并且利己比例更低。</li>
      <li>Waymo：AV mean IPV = <strong>+0.028</strong>，HV = <strong>+0.011</strong>；AV 亲社会比例 <strong>26.7%</strong>，竞争比例 <strong>10.0%</strong>，表现为更亲社会且更少竞争。</li>
      <li>nuPlan：AV mean IPV = <strong>-0.004</strong>，HV = <strong>-0.010</strong>；二者都存在较高竞争比例，且 AV 仅 <strong>188</strong> 个 agent values，Mann-Whitney signed p ≈ <strong>0.493</strong>，因此不应把 AV/HV 差异解释得过强。</li>
      <li>AV-HV paired case 内部对比：现有完成结果中共有 <strong>{int(paired_overall['paired_cases']):,}</strong> 个一 AV 一 HV 配对案例；同一 case 内 <span class="code">AV-HV mean IPV</span> 平均为 <strong>{paired_overall['av_minus_hv_mean']:+.3f}</strong>，paired Wilcoxon p = <strong>{html.escape(p_text(paired_overall['wilcoxon_p_value']))}</strong>。</li>
      <li>但 paired case 的差异并非普遍强：<strong>{paired_overall['similar_abs_diff_lt_0_05_pct']:.1f}%</strong> 的 AV-HV 差异仍在 <span class="code">|Δ| &lt; 0.05</span>；AV 明显高于 HV 的比例为 <strong>{paired_overall['av_higher_gt_0_05_pct']:.1f}%</strong>，HV 明显高于 AV 的比例为 <strong>{paired_overall['hv_higher_gt_0_05_pct']:.1f}%</strong>。</li>
      <li>旧的 <span class="code">AV_included</span> 是 case-level 变量，只能说明案例是否含 AV；本节的 <span class="code">agent_type</span> 是 key-agent-level 身份，更适合回答 AV 和 HV 各自的 IPV 分布差异。</li>
    </ul>
  </div>
  <div class="fig-grid">
    {figure_block('16', 'AV-HV paired case 内部 IPV 对比', '16_av_hv_paired_case_lens.png', '只保留同一 case 内恰好一个 AV 和一个 HV 的 key-agent 配对；Δ=AV-HV，正值表示 AV 相对 HV 更亲社会或更少竞争。', wide=True)}
    {figure_block('15', 'Key-agent AV/HV 按数据集的 IPV 差异', '15_agent_type_dataset_lens.png', 'agent_type 来自 pkl metadata.vehicle_type，并按 key_agents 顺序对齐；所有 AV-HV 比较都在同一数据集内进行。')}
    {figure_block('10', '交互类型与 AV 参与情况', '10_interaction_type_av_forest.png', 'case-level AV_included 与 two/multi 是辅助视角，不等同于单个 key agent 的 AV/HV 身份。')}
  </div>
  <h3>AV-HV paired case 内部对比证据表</h3>
  {format_report_table(av_hv_paired_display)}
  <h3>key-agent AV/HV 证据表</h3>
  {format_report_table(agent_type_display)}
  <h3>数据集内 AV-HV 差值表</h3>
  {format_report_table(agent_type_delta_display)}
  <h3>case-level AV_included 与 two/multi 辅助表</h3>
  {format_report_table(context_table)}
</section>

<section class="finding" id="c4">
  <h2>C4. F 类路径关系最竞争，HO 类最接近利己/弱交互</h2>
  <p class="takeaway">路径关系是解释竞争性的关键切口。F 类 path category 的竞争比例和强尾部明显更高，是筛选竞争案例的优先入口。</p>
  <div class="evidence-box">
    <strong>支撑信息</strong>
    <ul>
      <li>F 类 mean IPV = <strong>-0.025</strong>，竞争比例 <strong>27.8%</strong>，strong tail <strong>6.8%</strong>。</li>
      <li>HO 类 near-zero <strong>75.9%</strong>，更接近利己/弱交互。</li>
      <li>path_relation 和 turn_label 图可进一步定位更细的路径/转向组合。</li>
    </ul>
  </div>
  <div class="fig-grid">
    {figure_block('07', 'Path category 与 IPV', '07_path_category_forest.png', '比较粗粒度路径类别的 mean IPV 与置信区间。')}
    {figure_block('08', 'Path relation 与 IPV', '08_path_relation_forest.png', '在更细路径关系上查找竞争或亲社会更强的子类。')}
    {figure_block('09', 'Turn label 与 IPV', '09_turn_label_forest.png', '补充转向组合维度，帮助判断路径差异是否来自转向结构。')}
  </div>
  <h3>Path category 证据表</h3>
  {format_report_table(path_table)}
</section>

<section class="finding" id="c5">
  <h2>C5. 强亲社会/强竞争是尾部现象，应该作为案例复核重点</h2>
  <p class="takeaway">强社会性案例比例不高，但它们最可能承载可解释的互动机制，例如显著让行、抢行、冲突规避或竞争性通过。</p>
  <div class="evidence-box">
    <strong>支撑信息</strong>
    <ul>
      <li>整体 strong tail 只有 <strong>{strong_pct:.1f}%</strong>，不应把尾部当成普通案例。</li>
      <li>nuPlan 当前 pair |IPV| 尾部明显重于 AV2；F 类 path category 的强尾部也最高。</li>
      <li>error 与 pair |IPV| 的相关性较强，筛选尾部案例时应同时查看 error mean，避免把不稳定估计误读为行为机制。</li>
    </ul>
  </div>
  <div class="fig-grid">
    {figure_block('11', 'PET 与 intensity 的分箱关系', '11_pet_intensity_binned_abs_ipv.png', '检查风险代理指标和 pair |IPV| 的关系，辅助解释强交互是否与冲突强度相关。')}
    {figure_block('12', '估计误差分布及其与 |IPV| 的关系', '12_error_distribution_and_abs_ipv.png', '用于判断强 IPV 案例的估计可靠性。')}
  </div>
  <h3>相关性证据表</h3>
  {format_report_table(spearman)}
</section>

<section>
  <h2>建议的下一步</h2>
  <ol>
    <li><strong>先补齐 nuPlan。</strong> 当前最强的数据集差异来自 nuPlan，但它仍缺失 738 个 case；补齐后应重新确认 C2 的方向和 effect size。</li>
    <li><strong>优先复核尾部案例。</strong> 建议按 <span class="code">|IPV| ≥ 0.30</span>、F 类 path category、nuPlan 高 pair |IPV| 三个条件抽样复核轨迹。</li>
    <li><strong>把角色语义加进分析。</strong> 当前 key agent 1/2 只是 CSV 顺序；若能加入优先权、让行方、冲突点先后顺序，亲社会/竞争解释会更强。</li>
  </ol>
</section>

<section>
  <h2>关键假设与限制</h2>
  <div class="callout">
    <p><strong>阈值限制：</strong>亲社会、竞争、利己的阈值分别用 <span class="code">&gt;0.05</span>、<span class="code">&lt;-0.05</span>、<span class="code">|IPV|&lt;0.05</span> 表述，是报告解释层面的阈值。</p>
    <p><strong>覆盖限制：</strong>Waymo 和 AV2 已完整覆盖，nuPlan 未完整覆盖；所有跨数据集结论都应视为当前结果。</p>
    <p><strong>角色限制：</strong>IPV 与 key_agents 顺序匹配，但 key agent 1/2 不等同于固定交通角色；涉及行为机制时必须回到轨迹和场景语义。</p>
  </div>
</section>

<section>
  <h2>指标字典</h2>
  <p class="note">下表给出报告中主要英文指标的中文对应、统计粒度、含义和计算方式。公式中 <span class="code">k1</span> 表示 <span class="code">ipv_key_agent_1_mean</span>，<span class="code">k2</span> 表示 <span class="code">ipv_key_agent_2_mean</span>，<span class="code">v</span> 表示 agent-level signed IPV。</p>
  <div class="table-wrap">
    {metric_definition_table_html()}
  </div>
</section>

</main>
</body>
</html>
"""


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    use_theme()
    df, ok, agent_long = load_data()
    dataset_summary, dataset_tests = build_dataset_tables(df, ok, agent_long)
    agent_type_stats, agent_type_deltas = build_agent_type_tables(agent_long)
    av_hv_paired, av_hv_paired_summary = build_av_hv_paired_case_tables(ok)
    inside_pct = plot_pair_density(ok)
    plot_dataset_difference_lens(ok, agent_long, dataset_summary)
    dataset_effects = plot_dataset_effect_size_heatmap(ok, agent_long)
    plot_agent_type_dataset_lens(agent_long, agent_type_deltas)
    plot_av_hv_paired_case_lens(av_hv_paired, av_hv_paired_summary)

    summary_json = json.loads(SUMMARY_PATH.read_text(encoding="utf-8")) if SUMMARY_PATH.exists() else {}
    summary_json.update(
        {
            "csv": rel(MATCHED_CSV),
            "completed_cases": int(len(ok)),
            "missing_cases": int((~df["ipv_result_status"].eq("ok")).sum()),
            "agent_values": int(len(agent_long)),
            "near_zero_threshold": 0.05,
            "overall_signed_mean": float(agent_long["ipv"].mean()),
            "overall_signed_median": float(agent_long["ipv"].median()),
            "overall_abs_median": float(agent_long["ipv"].abs().median()),
            "positive_pct": 100 * float((agent_long["ipv"] > 0.05).mean()),
            "negative_pct": 100 * float((agent_long["ipv"] < -0.05).mean()),
            "near_zero_pct": 100 * float((agent_long["ipv"].abs() < 0.05).mean()),
            "strong_abs_ge_0_30_pct": 100 * float((agent_long["ipv"].abs() >= 0.30).mean()),
            "agent1_mean": float(ok["ipv_key_agent_1_mean"].mean()),
            "agent2_mean": float(ok["ipv_key_agent_2_mean"].mean()),
            "asymmetry_median_agent1_minus_agent2": float(ok["agent_asymmetry"].median()),
            "tables_dir": rel(TABLE_DIR),
            "figures_dir": rel(FIG_DIR),
            "figure_04_display_range": [-0.30, 0.30],
            "figure_04_display_coverage_pct": inside_pct,
            "dataset_comparison_summary_csv": rel(TABLE_DIR / "dataset_comparison_summary.csv"),
            "dataset_pairwise_tests_csv": rel(TABLE_DIR / "dataset_pairwise_tests.csv"),
            "dataset_effect_size_summary_csv": rel(TABLE_DIR / "dataset_effect_size_summary.csv"),
            "metric_definitions_csv": rel(TABLE_DIR / "metric_definitions.csv"),
            "agent_type_dataset_stats_csv": rel(TABLE_DIR / "agent_type_dataset_stats.csv"),
            "agent_type_dataset_deltas_csv": rel(TABLE_DIR / "agent_type_dataset_deltas.csv"),
            "av_hv_paired_case_summary_csv": rel(TABLE_DIR / "av_hv_paired_case_summary.csv"),
            "av_hv_paired_case_records_csv": rel(TABLE_DIR / "av_hv_paired_case_records.csv"),
            "av_hv_paired_cases": int(len(av_hv_paired)),
        }
    )
    SUMMARY_PATH.write_text(json.dumps(summary_json, indent=2, ensure_ascii=False), encoding="utf-8")
    html_text = build_conclusion_html(summary_json, inside_pct, ok, agent_long)
    HTML_PATH.write_text(html_text, encoding="utf-8")
    print(
        json.dumps(
            {
                "html": str(HTML_PATH),
                "figure_04_inside_pct": inside_pct,
                "question_runs": html_text.count("????"),
                "new_figures": [
                    "13_dataset_difference_lens.png",
                    "14_dataset_effect_size_heatmap.png",
                    "15_agent_type_dataset_lens.png",
                    "16_av_hv_paired_case_lens.png",
                ],
                "dataset_summary_rows": len(dataset_summary),
                "dataset_tests_rows": len(dataset_tests),
                "dataset_effect_rows": len(dataset_effects),
                "agent_type_stats_rows": len(agent_type_stats),
                "agent_type_delta_rows": len(agent_type_deltas),
                "av_hv_paired_cases": len(av_hv_paired),
                "av_hv_paired_summary_rows": len(av_hv_paired_summary),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
