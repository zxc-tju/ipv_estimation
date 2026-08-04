"""RQ015 Phase B — log 域轨迹似然权重（**B1 原型 + B2 scaffold**，与 legacy 并存）。

状态：`BUILD_WHILE_DENY / B1_PROTOTYPE / B2_SCAFFOLD_NOT_WIRED`。
本模块**未被任何生产路径导入**，不改变现有行为。生产接线（`estimate_ipv_pair`、
InterHub 导出/绘图、verifier anchor 的数值接口兼容层）**尚未交付**，是 Phase B 的
明列交付物，不得据本模块声称"可部署弃权链"。
启用需经独立双路复审 + Formal G1 + scoped authorization。

计划：`reports/plans/RQ015_plan_v1_ipv_estimability_and_estimator_repair_20260726.md` §4
复审：`.../codex_dual_review_synthesis_v1_20260726.md` 与 `.../codex_dual_review_synthesis_v1p1_20260726.md`
（后者 4 blocker 的修复见本文 `STATUS_SET`、`__post_init__` 不变量、`_check_*` 校验器、
D2 语义降级为 `D2_FLAT_UNDER_CURRENT_GRID_AND_MODEL`）

核心数学事实（计划 §4.0）：legacy 的
    var_i = [∏_k φ(d_ik)]^(1/n),  φ(d) = (1/(σ√2π))·exp(−d²/2σ²)
取对数后 log var_i = −log(σ√2π) − MSE_i/(2σ²)；首项对所有候选相同、归一化时约掉，故
    w = softmax(−MSE_i / (2σ²))
连乘是绕路，且正是 legacy 下溢之处。

充分性边界（复审 C-BLOCK-4.1）：`mse_per_candidate` **仅**对当前 Gaussian
squared-error 似然及其 σ 变更是充分统计量。它**不足以**换成重尾核——反例：
逐步残差 (0, √2) 与 (1, 1) 的 MSE 同为 1，但在 ν=3 的 Student-t 下对数似然
分别约为 −1.02165 与 −1.15073。若要支持换核，必须另存 `step_sq_residuals`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

SCHEMA_VERSION = "rq015-reliability-result-v2"
ESTIMATOR_VERSION = "rq015-logdomain-b1-prototype"

# ---- 主测量状态（互斥，恰好一个）----
STATUS_OK = "OK"
STATUS_NOT_ATTEMPTED = "NOT_ATTEMPTED"          # 上游按 frame_index < MIN_OBSERVATION 分流
STATUS_NON_FINITE_INPUT = "NON_FINITE_INPUT"
STATUS_SOLVER_FAILURE = "SOLVER_FAILURE"
STATUS_MODEL_MISFIT = "MODEL_MISFIT"
STATUS_FLAT_LIKELIHOOD = "FLAT_LIKELIHOOD"
STATUS_PRIORITY = (                              # 多因并存时的冻结优先级
    STATUS_NON_FINITE_INPUT, STATUS_SOLVER_FAILURE,
    STATUS_MODEL_MISFIT, STATUS_FLAT_LIKELIHOOD, STATUS_OK,
)
# 互斥主状态的**唯一**合法取值集合（NOT_ATTEMPTED 由上游发射）。
# `AT_GRID_BOUNDARY` **不是** status，只能是 flag（复审 v1.1 blocker 1）。
STATUS_SET = frozenset(STATUS_PRIORITY) | {STATUS_NOT_ATTEMPTED}

# ---- 诊断 flag（可并存，与 status 正交）----
FLAG_AT_GRID_BOUNDARY = "AT_GRID_BOUNDARY"
FLAG_LEGACY_PARTIAL_UNDERFLOW = "LEGACY_PARTIAL_UNDERFLOW"
FLAG_LEGACY_TOTAL_UNDERFLOW = "LEGACY_TOTAL_UNDERFLOW"

# ---- 零值机制（计划 §4.4；D0 由上游 frame_index 分流）----
MECH_D1_NUMERICAL = "D1_NUMERICAL_UNDERFLOW"
MECH_D2_FLAT = "D2_FLAT_UNDER_CURRENT_GRID_AND_MODEL"   # 不得称"固有不可辨识"
MECH_D3_MISFIT = "D3_MODEL_MISFIT"
MECH_D4_FAILURE = "D4_SOLVER_OR_INPUT_FAILURE"
MECH_OK = "OK"

K_EFF_FLAT_RATIO = 0.93          # K_eff >= ratio*K -> 近乎均匀（与计划 §3 一致）
LEGACY_DIVERGENCE_TOL = 1e-6     # 新旧权重最大绝对差，超过即判 legacy 被下溢污染


class EstimatorInputError(ValueError):
    """fail-closed 输入校验（复审 5.1；v1.1 blocker 4 扩展）。"""


class ResultInvariantError(AssertionError):
    """结果自洽性被破坏（如 status=OK 却带 NaN）。"""


@dataclass
class ReliabilityResult:
    """正交结果合同（复审 C-BLOCK-1 第 2 条）：主状态 + 并存 flags + 版本三元组。"""

    weights: np.ndarray
    ipv: float
    ipv_error: float
    status: str                       # 互斥主状态
    flags: Tuple[str, ...]            # 可并存诊断
    reason_code: Optional[str]
    K: int
    grid_id: str
    min_mse: float
    loglike_gap: float
    mse_per_candidate: np.ndarray
    k_eff: float
    step_sq_residuals: Optional[np.ndarray] = None   # 换核所需；默认不存（体积）
    schema_version: str = SCHEMA_VERSION
    estimator_version: str = ESTIMATOR_VERSION
    sufficiency_scope: str = field(default="gaussian_sigma_only")

    def __post_init__(self) -> None:
        """结果不变量（复审 v1.1 blocker 4）：status=OK 必须携带有限数值。"""
        if self.status not in STATUS_SET:
            raise ResultInvariantError(f"status {self.status!r} not in STATUS_SET")
        if FLAG_AT_GRID_BOUNDARY in (self.status,):
            raise ResultInvariantError("AT_GRID_BOUNDARY is a flag, never a status")
        if len(set(self.flags)) != len(self.flags):
            raise ResultInvariantError(f"duplicate flags: {self.flags}")
        if self.status == STATUS_OK:
            if not np.isfinite(self.ipv):
                raise ResultInvariantError("status=OK requires finite ipv (no OK+NaN)")
            if not (np.isfinite(self.ipv_error) and np.isfinite(self.k_eff)):
                raise ResultInvariantError("status=OK requires finite ipv_error and k_eff")
            if not np.all(np.isfinite(self.weights)):
                raise ResultInvariantError("status=OK requires finite weights")
        elif np.isfinite(self.ipv):
            raise ResultInvariantError(f"status={self.status} must carry ipv=NaN, got {self.ipv}")


def _check_sigma(sigma: object) -> float:
    if isinstance(sigma, bool) or not isinstance(sigma, (int, float, np.floating)):
        raise EstimatorInputError(f"sigma must be a real number, got {sigma!r}")
    sigma = float(sigma)
    if not (np.isfinite(sigma) and sigma > 0):
        raise EstimatorInputError(f"sigma must be finite positive, got {sigma!r}")
    denom = 2.0 * sigma ** 2
    if not (np.isfinite(denom) and denom > 0):     # 极小正 σ 使 2σ² 下溢为 0
        raise EstimatorInputError(
            f"sigma={sigma!r} underflows 2*sigma**2 to {denom!r}; likelihood undefined")
    return sigma


def _check_ratio(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise EstimatorInputError(f"{name} must be a real number, got {value!r}")
    value = float(value)
    if not (np.isfinite(value) and 0.0 < value <= 1.0):
        raise EstimatorInputError(f"{name} must lie in (0, 1], got {value!r}")
    return value


def _check_positive(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise EstimatorInputError(f"{name} must be a real number, got {value!r}")
    value = float(value)
    if not (np.isfinite(value) and value > 0):
        raise EstimatorInputError(f"{name} must be finite positive, got {value!r}")
    return value


def _validate(act: np.ndarray, vir: Sequence[np.ndarray], sigma: float) -> None:
    _check_sigma(sigma)
    if len(vir) == 0:
        raise EstimatorInputError("empty candidate collection")
    if act.ndim != 2 or act.shape[0] == 0:
        raise EstimatorInputError(f"observed track must be (n>=1, d), got {act.shape}")
    for i, v in enumerate(vir):
        if v.shape != act.shape:
            raise EstimatorInputError(
                f"candidate {i} shape {v.shape} != observed {act.shape} (broadcast forbidden)")


def step_sq_residuals(act_track: np.ndarray, vir_track_coll: Sequence[np.ndarray]) -> np.ndarray:
    """逐步平方残差 d_ik²，shape (K, n)。换核（如 Student-t）所需的统计量。"""
    act = np.asarray(act_track, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):   # 溢出由调用方按 SOLVER_FAILURE 处理
        return np.vstack([
            np.sum((np.asarray(v, dtype=float) - act) ** 2, axis=1) for v in vir_track_coll
        ])


def candidate_mse(act_track: np.ndarray, vir_track_coll: Sequence[np.ndarray]) -> np.ndarray:
    """每候选均方距离（Gaussian 似然 + σ 变更的充分统计量）。"""
    return step_sq_residuals(act_track, vir_track_coll).mean(axis=1)


def weights_from_mse(mse: np.ndarray, sigma: float) -> np.ndarray:
    """稳定 softmax：减最大值后最大项恒为 exp(0)=1，分母 >= 1，永不为零。"""
    mse = np.asarray(mse, dtype=float)
    sigma = _check_sigma(sigma)
    if mse.ndim != 1 or mse.size == 0:
        raise EstimatorInputError(f"mse must be a non-empty 1-D array, got shape {mse.shape}")
    if not np.all(np.isfinite(mse)):
        raise EstimatorInputError("non-finite mse (squared-distance overflow?)")
    logw = -mse / (2.0 * sigma ** 2)
    if not np.all(np.isfinite(logw)):
        raise EstimatorInputError("non-finite log-weights (sigma too small for this mse range?)")
    logw = logw - np.max(logw)
    w = np.exp(logw)
    total = float(w.sum())
    if not (np.isfinite(total) and total >= 1.0 - 1e-12):   # legacy 兜底分支不可达
        raise ResultInvariantError(f"stable softmax denominator collapsed to {total!r}")
    return w / total


def legacy_var(act_track: np.ndarray, vir_track_coll: Sequence[np.ndarray],
               sigma: float) -> np.ndarray:
    """legacy 的逐候选 `var`（未归一化），用于识别**部分**下溢（复审 C-BLOCK-2）。"""
    act = np.asarray(act_track, dtype=float)
    n = act.shape[0]
    out = np.zeros(len(vir_track_coll), dtype=float)
    for i, v in enumerate(vir_track_coll):
        d = np.linalg.norm(np.asarray(v, dtype=float) - act, axis=1)
        val = np.power(
            np.prod((1.0 / sigma / np.sqrt(2 * np.pi)) * np.exp(-d ** 2 / (2 * sigma ** 2))),
            1.0 / n)
        out[i] = 0.0 if val < 0 else val
    return out


def legacy_weights_reference(act_track: np.ndarray, vir_track_coll: Sequence[np.ndarray],
                             sigma: float) -> np.ndarray:
    """legacy 概率域算法的忠实复刻，**仅供平价测试与机制判别**，不得用于生产。"""
    var = legacy_var(act_track, vir_track_coll, sigma)
    if var.sum():
        return var / var.sum()
    return np.ones(len(vir_track_coll)) / len(vir_track_coll)   # 均匀兜底（缺陷所在）


def estimate_reliability(
    act_track: np.ndarray,
    vir_track_coll: Sequence[np.ndarray],
    ipv_range: Sequence[float],
    sigma: float,
    *,
    min_mse_misfit: float,
    grid_id: str,
    k_eff_flat_ratio: float = K_EFF_FLAT_RATIO,
    keep_step_residuals: bool = False,
) -> ReliabilityResult:
    """log 域权重 + 充分统计量 + 正交状态/flags。

    `min_mse_misfit` 与 `grid_id` 为**必填**：D3 阈值须由 Phase A 在
    dev+guard（剔除 RQ007 sealed）上冻结后显式传入，禁止默认关闭 D3
    （复审 C-BLOCK-2）。
    """
    act = np.asarray(act_track, dtype=float)
    stacked = [np.asarray(v, dtype=float) for v in vir_track_coll]
    _validate(act, stacked, sigma)
    K = len(stacked)
    ipv_range = np.asarray(ipv_range, dtype=float)
    if ipv_range.shape != (K,):
        raise EstimatorInputError("candidate count / ipv_range mismatch")
    min_mse_misfit = _check_positive("min_mse_misfit", min_mse_misfit)
    k_eff_flat_ratio = _check_ratio("k_eff_flat_ratio", k_eff_flat_ratio)
    if not isinstance(grid_id, str) or not grid_id:
        raise EstimatorInputError("grid_id must be a non-empty string")

    nan_k = np.full(K, np.nan)
    if not np.all(np.isfinite(act)) or not all(np.all(np.isfinite(v)) for v in stacked):
        return ReliabilityResult(
            weights=nan_k, ipv=float("nan"), ipv_error=float("nan"),
            status=STATUS_NON_FINITE_INPUT, flags=(), reason_code="non_finite_track",
            K=K, grid_id=grid_id, min_mse=float("nan"), loglike_gap=float("nan"),
            mse_per_candidate=nan_k, k_eff=float("nan"))

    sq = step_sq_residuals(act, stacked)
    if not np.all(np.isfinite(sq)):
        return ReliabilityResult(
            weights=nan_k, ipv=float("nan"), ipv_error=float("nan"),
            status=STATUS_SOLVER_FAILURE, flags=(), reason_code="squared_distance_overflow",
            K=K, grid_id=grid_id, min_mse=float("inf"), loglike_gap=float("nan"),
            mse_per_candidate=nan_k, k_eff=float("nan"))

    mse = sq.mean(axis=1)
    w = weights_from_mse(mse, sigma)
    ipv_error = float(1.0 - np.sqrt(np.sum(w ** 2)))
    k_eff = float(1.0 / np.sum(w ** 2))

    order = np.argsort(mse)
    min_mse = float(mse[order[0]])
    loglike_gap = (float((mse[order[1]] - mse[order[0]]) / (2.0 * sigma ** 2))
                   if K > 1 else float("inf"))

    flags = []
    if order[0] in (0, K - 1):
        flags.append(FLAG_AT_GRID_BOUNDARY)
    lv = legacy_var(act, stacked, sigma)
    if (lv == 0).all():
        flags.append(FLAG_LEGACY_TOTAL_UNDERFLOW)
    elif (lv == 0).any():
        flags.append(FLAG_LEGACY_PARTIAL_UNDERFLOW)

    if min_mse > min_mse_misfit:
        status, reason = STATUS_MODEL_MISFIT, "all_candidates_fit_poorly"
    elif k_eff >= k_eff_flat_ratio * K:
        status, reason = STATUS_FLAT_LIKELIHOOD, "weights_near_uniform"
    else:
        status, reason = STATUS_OK, None

    return ReliabilityResult(
        weights=w, ipv=float(np.sum(ipv_range * w)) if status == STATUS_OK else float("nan"),
        ipv_error=ipv_error, status=status, flags=tuple(flags), reason_code=reason,
        K=K, grid_id=grid_id, min_mse=min_mse, loglike_gap=loglike_gap,
        mse_per_candidate=mse, k_eff=k_eff,
        step_sq_residuals=sq if keep_step_residuals else None)


def classify_zero_mechanism(
    act_track: np.ndarray,
    vir_track_coll: Sequence[np.ndarray],
    ipv_range: Sequence[float],
    sigma: float,
    *,
    min_mse_misfit: float,
    grid_id: str,
    k_eff_flat_ratio: float = K_EFF_FLAT_RATIO,
    legacy_divergence_tol: float = LEGACY_DIVERGENCE_TOL,
) -> Tuple[str, ReliabilityResult]:
    """冻结的 D1/D2/D3/D4 可执行分类器（复审 C-BLOCK-2）。

    优先级：D4 > D1 > D3 > D2 > OK。
    **D1 不以"是否命中均匀兜底"定义**——legacy 可只压掉部分候选而不走兜底分支；
    判据是新旧权重的最大绝对差超过 `legacy_divergence_tol`，即 legacy 结果已被下溢污染。
    """
    legacy_divergence_tol = _check_positive("legacy_divergence_tol", legacy_divergence_tol)
    res = estimate_reliability(
        act_track, vir_track_coll, ipv_range, sigma,
        min_mse_misfit=min_mse_misfit, grid_id=grid_id,
        k_eff_flat_ratio=k_eff_flat_ratio)
    if res.status in (STATUS_NON_FINITE_INPUT, STATUS_SOLVER_FAILURE):
        return MECH_D4_FAILURE, res

    legacy_w = legacy_weights_reference(act_track, vir_track_coll, sigma)
    if float(np.max(np.abs(legacy_w - res.weights))) > legacy_divergence_tol:
        return MECH_D1_NUMERICAL, res
    if res.status == STATUS_MODEL_MISFIT:
        return MECH_D3_MISFIT, res
    if res.status == STATUS_FLAT_LIKELIHOOD:
        return MECH_D2_FLAT, res
    return MECH_OK, res


def underflow_rms_threshold(sigma: float, n_steps: int, boundary: str = "subnormal") -> float:
    """legacy 连乘（开 1/n 次方**之前**）下溢的临界 RMS 距离（复审 5.2）。

    `boundary="subnormal"`：乘积跌入 subnormal 区（< 最小正规数）的临界值；
    `boundary="zero"`     ：乘积舍入为精确 0、从而触发均匀兜底的临界值。
    二者相差约 0.03–0.04 m，必须分别命名与测试。
    """
    limits = {"subnormal": np.finfo(float).tiny, "zero": 5e-324}
    if boundary not in limits:
        raise EstimatorInputError(f"boundary must be one of {sorted(limits)}")
    c = np.log(1.0 / (sigma * np.sqrt(2 * np.pi)))
    mse_crit = 2.0 * sigma ** 2 * (c - np.log(limits[boundary]) / n_steps)
    return float(np.sqrt(max(mse_crit, 0.0)))
