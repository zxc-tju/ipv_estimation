"""RQ015 Phase B1/B2 平价、机制判别与 fail-closed 测试（无真实数据、无 HPC）。

覆盖 v1 双路复审关闭清单：正交结果合同、D1/D2/D3 可执行分类、部分下溢、
充分性边界（Gaussian/σ vs 换核）、σ/shape/overflow 入口校验、两个下溢阈值定义。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sociality_estimation.core.reliability_logdomain import (
    FLAG_AT_GRID_BOUNDARY,
    FLAG_LEGACY_PARTIAL_UNDERFLOW,
    FLAG_LEGACY_TOTAL_UNDERFLOW,
    MECH_D1_NUMERICAL,
    MECH_D2_FLAT,
    MECH_D3_MISFIT,
    MECH_D4_FAILURE,
    MECH_OK,
    SCHEMA_VERSION,
    STATUS_FLAT_LIKELIHOOD,
    STATUS_MODEL_MISFIT,
    STATUS_NON_FINITE_INPUT,
    STATUS_OK,
    EstimatorInputError,
    candidate_mse,
    classify_zero_mechanism,
    estimate_reliability,
    legacy_var,
    legacy_weights_reference,
    step_sq_residuals,
    underflow_rms_threshold,
    weights_from_mse,
)

SIGMA = 0.1
IPV_GRID = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=float) * math.pi / 8
GRID_ID = "legacy7_pi_over_8"
MISFIT = 4.0          # 测试用冻结阈值（真实值待 Phase A 在 dev+guard 上冻结）


def _tracks(rng, n_steps, offsets):
    act = np.cumsum(rng.normal(0, 1.0, size=(n_steps, 2)), axis=0)
    return act, [act + np.array([o, 0.0]) for o in offsets]


def _kw(**over):
    kw = dict(min_mse_misfit=MISFIT, grid_id=GRID_ID)
    kw.update(over)
    return kw


# ---------------- B1 平价 ----------------

def test_parity_no_underflow():
    """未下溢时新旧权重逐位一致（平价门 <= 1e-12）。"""
    rng = np.random.default_rng(20260726)
    for trial in range(50):
        n = int(rng.integers(5, 12))
        act, vir = _tracks(rng, n, rng.uniform(0.0, 0.25, size=7))
        legacy = legacy_weights_reference(act, vir, SIGMA)
        assert legacy_var(act, vir, SIGMA).sum() > 0
        assert np.max(np.abs(legacy - weights_from_mse(candidate_mse(act, vir), SIGMA))) <= 1e-12, trial


def test_uniform_fallback_branch_unreachable():
    rng = np.random.default_rng(11)
    for scale in (0.01, 1.0, 100.0, 1e4):
        act, vir = _tracks(rng, 8, rng.uniform(0, scale, size=7))
        w = weights_from_mse(candidate_mse(act, vir), SIGMA)
        assert abs(w.sum() - 1.0) < 1e-12 and np.isfinite(w).all()


# ---------------- 机制判别 ----------------

def test_d1_total_underflow():
    """legacy 全候选下溢 -> 均匀兜底 -> IPV 精确 0；分类为 D1，新实现给出真实权重。"""
    rng = np.random.default_rng(7)
    n = 5
    thr = underflow_rms_threshold(SIGMA, n, "zero")
    act, vir = _tracks(rng, n, np.linspace(thr * 1.5, thr * 2.5, 7))

    legacy = legacy_weights_reference(act, vir, SIGMA)
    assert np.allclose(legacy, 1 / 7)
    assert abs(float(np.sum(IPV_GRID * legacy))) < 1e-12
    assert abs((1 - math.sqrt(float((legacy ** 2).sum()))) - (1 - 1 / math.sqrt(7))) < 1e-12

    mech, res = classify_zero_mechanism(act, vir, IPV_GRID, SIGMA, **_kw(min_mse_misfit=1e9))
    assert mech == MECH_D1_NUMERICAL
    assert FLAG_LEGACY_TOTAL_UNDERFLOW in res.flags
    assert res.weights.max() > 0.99 and res.k_eff < 1.5


def test_partial_underflow_far_candidates_is_harmless():
    """部分下溢**本身**不等于污染：被清零的候选若本就权重可忽略，legacy 结果不变。

    这一条把 D1 的定义钉死为"legacy 结果是否被改变"，而不是"是否发生过下溢"。
    """
    rng = np.random.default_rng(21)
    n = 5
    thr = underflow_rms_threshold(SIGMA, n, "zero")
    act, vir = _tracks(rng, n, [0.05, 0.10, 0.15, thr * 2, thr * 3, thr * 4, thr * 5])

    lv = legacy_var(act, vir, SIGMA)
    assert (lv == 0).any() and (lv > 0).any(), "构造失败：需部分下溢"
    mech, res = classify_zero_mechanism(act, vir, IPV_GRID, SIGMA, **_kw(min_mse_misfit=1e9))
    assert FLAG_LEGACY_PARTIAL_UNDERFLOW in res.flags     # 诊断 flag 仍如实记录
    assert mech == MECH_OK                                # 但机制判定为无污染


def test_d1_partial_underflow_at_boundary_is_detected():
    """真正有害的部分下溢：候选恰好跨越下溢边界，被清零者本应携带可观权重。

    构造：以常量偏移 o 使 MSE 精确等于 o²；`zero` 边界处 MSE_crit≈3.005（σ=0.1, n=5）。
    取一个略低于、六个略高于边界的候选，被清零者的真实相对权重约 e^{-0.87}≈0.42。
    """
    rng = np.random.default_rng(22)
    n = 5
    crit = underflow_rms_threshold(SIGMA, n, "zero")           # ≈1.7336
    offsets = [crit - 0.004] + [crit + d for d in (0.002, 0.003, 0.004, 0.005, 0.006, 0.007)]
    act, vir = _tracks(rng, n, offsets)

    lv = legacy_var(act, vir, SIGMA)
    assert (lv == 0).sum() >= 1 and (lv > 0).sum() >= 1, "构造失败：需跨边界"
    legacy_w = legacy_weights_reference(act, vir, SIGMA)
    new_w = weights_from_mse(candidate_mse(act, vir), SIGMA)
    assert np.max(np.abs(legacy_w - new_w)) > 0.1, "构造失败：需实质差异"

    mech, res = classify_zero_mechanism(act, vir, IPV_GRID, SIGMA, **_kw(min_mse_misfit=1e9))
    assert FLAG_LEGACY_PARTIAL_UNDERFLOW in res.flags
    assert mech == MECH_D1_NUMERICAL, "跨边界的部分下溢必须判为 D1"


def test_d2_flat_versus_d3_misfit():
    """同为近均匀权重，`min_mse` 小=D2（固有不可辨识）、大=D3（模型失配）。"""
    rng = np.random.default_rng(31)
    act = np.cumsum(rng.normal(0, 1, size=(8, 2)), axis=0)

    same = [act.copy() for _ in range(7)]                       # 候选完全一致 -> min_mse=0
    mech, res = classify_zero_mechanism(act, same, IPV_GRID, SIGMA, **_kw())
    assert mech == MECH_D2_FLAT and res.status == STATUS_FLAT_LIKELIHOOD
    assert math.isnan(res.ipv), "不可估计必须弃权，不得落回中性 0"
    assert res.k_eff == pytest.approx(7.0)

    far = [act + np.array([100.0, 0.0]) for _ in range(7)]      # 一致但全体拟合极差
    mech2, res2 = classify_zero_mechanism(act, far, IPV_GRID, SIGMA, **_kw())
    assert mech2 == MECH_D3_MISFIT and res2.status == STATUS_MODEL_MISFIT
    assert math.isnan(res2.ipv)


def test_d3_reachable_by_default_contract():
    """D3 阈值为必填参数：不得像 v1 草案那样默认关闭 D3。"""
    rng = np.random.default_rng(41)
    act, vir = _tracks(rng, 6, [0.1] * 7)
    with pytest.raises(TypeError):
        estimate_reliability(act, vir, IPV_GRID, SIGMA)          # 缺必填 kwargs
    with pytest.raises(EstimatorInputError):
        estimate_reliability(act, vir, IPV_GRID, SIGMA, min_mse_misfit=float("nan"), grid_id=GRID_ID)


def test_d4_failure_paths():
    rng = np.random.default_rng(51)
    act, vir = _tracks(rng, 6, [0.1] * 7)
    bad = act.copy(); bad[0, 0] = np.nan
    mech, res = classify_zero_mechanism(bad, vir, IPV_GRID, SIGMA, **_kw())
    assert mech == MECH_D4_FAILURE and res.status == STATUS_NON_FINITE_INPUT

    huge = act * 1e200
    mech2, res2 = classify_zero_mechanism(act, [huge] * 7, IPV_GRID, SIGMA, **_kw())
    assert mech2 == MECH_D4_FAILURE      # 平方距离溢出 -> SOLVER_FAILURE，而非静默断言失败


# ---------------- 充分性边界 ----------------

def test_mse_sufficient_for_gaussian_sigma_only():
    """MSE 对 Gaussian/σ 充分；但对重尾核不充分（复审给出的反例）。"""
    rng = np.random.default_rng(3)
    act, vir = _tracks(rng, 9, rng.uniform(0, 0.2, size=7))
    mse = candidate_mse(act, vir)
    for sigma in (0.05, 0.1, 0.5, 2.0):
        assert np.array_equal(weights_from_mse(candidate_mse(act, vir), sigma),
                              weights_from_mse(mse, sigma))

    # 反例：两组逐步残差 MSE 相同，Student-t(ν=3) 对数似然不同
    a = np.array([0.0, 2.0]); b = np.array([1.0, 1.0])          # 平方残差，均值均为 1
    assert a.mean() == b.mean() == 1.0
    nu = 3.0
    ll = lambda sq: float(np.sum(-(nu + 1) / 2 * np.log1p(sq / nu)))
    assert abs(ll(a) - (-1.02165)) < 1e-4
    assert abs(ll(b) - (-1.15073)) < 1e-4
    assert ll(a) != ll(b), "换核必须另存 step_sq_residuals"


def test_step_residuals_optional_and_shaped():
    rng = np.random.default_rng(61)
    act, vir = _tracks(rng, 7, rng.uniform(0, 0.2, size=7))
    res = estimate_reliability(act, vir, IPV_GRID, SIGMA, **_kw(), keep_step_residuals=True)
    assert res.step_sq_residuals.shape == (7, 7)
    assert np.allclose(res.step_sq_residuals.mean(axis=1), res.mse_per_candidate)
    assert res.sufficiency_scope == "gaussian_sigma_only"
    assert estimate_reliability(act, vir, IPV_GRID, SIGMA, **_kw()).step_sq_residuals is None
    assert np.allclose(step_sq_residuals(act, vir).mean(axis=1), candidate_mse(act, vir))


# ---------------- fail-closed 入口 ----------------

@pytest.mark.parametrize("sigma", [0.0, -0.1, float("nan"), float("inf")])
def test_sigma_validation(sigma):
    rng = np.random.default_rng(71)
    act, vir = _tracks(rng, 6, [0.1] * 7)
    with pytest.raises(EstimatorInputError):
        estimate_reliability(act, vir, IPV_GRID, sigma, **_kw())


def test_shape_and_empty_validation():
    rng = np.random.default_rng(81)
    act, vir = _tracks(rng, 6, [0.1] * 7)
    with pytest.raises(EstimatorInputError):
        estimate_reliability(act, [v[:-1] for v in vir], IPV_GRID, SIGMA, **_kw())  # 禁止广播
    with pytest.raises(EstimatorInputError):
        estimate_reliability(act, [], IPV_GRID, SIGMA, **_kw())
    with pytest.raises(EstimatorInputError):
        estimate_reliability(act, vir, IPV_GRID[:-1], SIGMA, **_kw())


# ---------------- 契约与阈值 ----------------

def test_orthogonal_result_contract():
    """状态互斥、flags 可并存、版本三元组齐全。"""
    rng = np.random.default_rng(91)
    act = np.cumsum(rng.normal(0, 1, size=(6, 2)), axis=0)
    far = [act + np.array([100.0, 0.0]) for _ in range(7)]
    res = estimate_reliability(act, far, IPV_GRID, SIGMA, **_kw())
    assert res.status == STATUS_MODEL_MISFIT
    assert FLAG_AT_GRID_BOUNDARY in res.flags          # 主状态与 flag 同时成立
    assert res.schema_version == SCHEMA_VERSION and res.grid_id == GRID_ID
    assert res.estimator_version.startswith("rq015-logdomain")

    act2, vir2 = _tracks(rng, 8, np.linspace(0.0, 0.3, 7))
    ok = estimate_reliability(act2, vir2, IPV_GRID, SIGMA, **_kw())
    assert ok.status == STATUS_OK and np.isfinite(ok.ipv)
    assert ok.min_mse == pytest.approx(float(ok.mse_per_candidate.min()))


def test_two_distinct_underflow_thresholds():
    """subnormal 进入点与舍入为精确 0 是两个不同边界，必须分别命名（复审 5.2）。"""
    sub5, zero5 = underflow_rms_threshold(0.1, 5, "subnormal"), underflow_rms_threshold(0.1, 5, "zero")
    sub11, zero11 = underflow_rms_threshold(0.1, 11, "subnormal"), underflow_rms_threshold(0.1, 11, "zero")
    assert sub5 == pytest.approx(1.6915, abs=5e-4) and zero5 == pytest.approx(1.7336, abs=5e-4)
    assert sub11 == pytest.approx(1.1470, abs=5e-4) and zero11 == pytest.approx(1.1752, abs=5e-4)
    assert sub5 < zero5 and sub11 < zero11
    with pytest.raises(EstimatorInputError):
        underflow_rms_threshold(0.1, 5, "bogus")


def test_sigma_sharpness_is_conditional_on_candidate_spread():
    """σ=0.1 是否坍缩成硬 argmax 取决于候选 MSE 间距（v1 §4.3 的更正依据）。"""
    rng = np.random.default_rng(5)

    def k_eff(offsets, sigma):
        act, vir = _tracks(rng, 10, offsets)
        w = weights_from_mse(candidate_mse(act, vir), sigma)
        return 1.0 / float(np.sum(w ** 2))

    narrow, wide = np.linspace(0.0, 0.6, 7), np.linspace(0.0, 6.0, 7)
    assert 1.5 < k_eff(narrow, 0.1) < 3.5
    assert k_eff(wide, 0.1) < 1.05
    seq = [k_eff(narrow, s) for s in (0.05, 0.1, 0.5, 1.0)]
    assert all(a < b for a, b in zip(seq, seq[1:])), seq


# ---------------- v1.1 复审 4 blocker 的关闭测试 ----------------

def test_status_set_excludes_grid_boundary():
    """blocker 1：`AT_GRID_BOUNDARY` 只能是 flag，不得混入 status 取值集合。"""
    from sociality_estimation.core.reliability_logdomain import (
        FLAG_AT_GRID_BOUNDARY as FB, STATUS_SET, STATUS_PRIORITY,
    )
    assert FB not in STATUS_SET
    assert STATUS_SET == {
        "OK", "NOT_ATTEMPTED", "NON_FINITE_INPUT",
        "SOLVER_FAILURE", "MODEL_MISFIT", "FLAT_LIKELIHOOD",
    }
    assert set(STATUS_PRIORITY) < STATUS_SET and len(STATUS_PRIORITY) == 5


def test_d2_label_is_not_intrinsic():
    """blocker 2：D2 只能声称"在当前网格与模型下平坦"，不得称"固有不可辨识"。"""
    from sociality_estimation.core.reliability_logdomain import MECH_D2_FLAT
    assert MECH_D2_FLAT == "D2_FLAT_UNDER_CURRENT_GRID_AND_MODEL"
    assert "INTRINSIC" not in MECH_D2_FLAT


def test_invariant_forbids_ok_with_nan():
    """blocker 4：status=OK 携带 NaN 必须直接失败，不得静默通过。"""
    from sociality_estimation.core.reliability_logdomain import (
        ReliabilityResult, ResultInvariantError, STATUS_OK, STATUS_FLAT_LIKELIHOOD,
    )
    ok_kw = dict(weights=np.full(7, 1 / 7), ipv_error=0.6, status=STATUS_OK, flags=(),
                 reason_code=None, K=7, grid_id=GRID_ID, min_mse=0.1,
                 loglike_gap=1.0, mse_per_candidate=np.zeros(7), k_eff=2.0)
    with pytest.raises(ResultInvariantError):
        ReliabilityResult(ipv=float("nan"), **ok_kw)                     # OK + NaN
    with pytest.raises(ResultInvariantError):
        ReliabilityResult(ipv=0.5, **{**ok_kw, "status": STATUS_FLAT_LIKELIHOOD})  # 弃权却带数值
    with pytest.raises(ResultInvariantError):
        ReliabilityResult(ipv=0.5, **{**ok_kw, "status": "BOGUS"})
    with pytest.raises(ResultInvariantError):
        ReliabilityResult(ipv=0.5, **{**ok_kw, "flags": ("A", "A")})     # 重复 flag
    assert ReliabilityResult(ipv=0.5, **ok_kw).status == STATUS_OK       # 合法构造仍可用


@pytest.mark.parametrize("ratio", [0.0, -0.1, 1.5, float("nan"), float("inf"), True])
def test_invalid_k_eff_ratio_rejected(ratio):
    rng = np.random.default_rng(101)
    act, vir = _tracks(rng, 6, [0.1] * 7)
    with pytest.raises(EstimatorInputError):
        estimate_reliability(act, vir, IPV_GRID, SIGMA, **_kw(), k_eff_flat_ratio=ratio)


@pytest.mark.parametrize("tol", [0.0, -1e-6, float("nan"), float("inf")])
def test_invalid_divergence_tol_rejected(tol):
    rng = np.random.default_rng(102)
    act, vir = _tracks(rng, 6, [0.1] * 7)
    with pytest.raises(EstimatorInputError):
        classify_zero_mechanism(act, vir, IPV_GRID, SIGMA, **_kw(), legacy_divergence_tol=tol)


@pytest.mark.parametrize("sigma", [1e-200, 5e-324, True])
def test_tiny_positive_sigma_is_typed_error_not_assertion(sigma):
    """blocker 4：极小正 σ 曾以 AssertionError 泄漏，现须为 EstimatorInputError。"""
    rng = np.random.default_rng(103)
    act, vir = _tracks(rng, 6, [0.1] * 7)
    with pytest.raises(EstimatorInputError):
        estimate_reliability(act, vir, IPV_GRID, sigma, **_kw())
    with pytest.raises(EstimatorInputError):
        weights_from_mse(np.array([0.0, 1.0]), sigma)


def test_grid_id_and_mse_shape_validated():
    rng = np.random.default_rng(104)
    act, vir = _tracks(rng, 6, [0.1] * 7)
    with pytest.raises(EstimatorInputError):
        estimate_reliability(act, vir, IPV_GRID, SIGMA, min_mse_misfit=MISFIT, grid_id="")
    with pytest.raises(EstimatorInputError):
        weights_from_mse(np.zeros((2, 2)), SIGMA)
    with pytest.raises(EstimatorInputError):
        weights_from_mse(np.array([]), SIGMA)


def test_assert_stripping_does_not_disable_the_softmax_guard():
    """blocker 4：兜底守卫改为显式 raise，`python -O` 剥离 assert 也不会失效。"""
    import subprocess, sys, textwrap
    code = textwrap.dedent("""
        import numpy as np
        from sociality_estimation.core.reliability_logdomain import (
            weights_from_mse, EstimatorInputError)
        try:
            weights_from_mse(np.array([np.nan, 0.0]), 0.1)
        except EstimatorInputError:
            print("TYPED_OK")
    """)
    out = subprocess.run([sys.executable, "-O", "-c", code], capture_output=True,
                         text=True, env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"})
    assert "TYPED_OK" in out.stdout, out
