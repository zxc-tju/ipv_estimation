"""RQ015A 合同 fixtures（不读真实数据）。关闭 v2 复审 blocker 3/4 的可执行证据。"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "rq015a"))

from rq015a_contracts import (  # noqa: E402
    ATTEMPTED, NOT_ATTEMPTED, UNKNOWN, ContractViolation, REPORT_BINS_PRIMARY,
    L2Unit, aggregate_l2, aggregate_l3, band_shares, bins_stability, c0_route,
    c0_route_with_sensitivity, check_conservation, episode_summaries,
    k_eff_from_error, load_schema, local_positions, q_eff,
)

SCHEMA = ROOT / "reports" / "plans" / "RQ015A_ledger_schema_v2.json"


# ---------------- 派生量 ----------------

def test_q_eff_and_warmup_degeneracy():
    assert q_eff(0.0, 7) == pytest.approx(1 / 7)              # one-hot -> K_eff=1
    assert q_eff(1 - 1 / math.sqrt(7), 7) == pytest.approx(1.0, abs=1e-9)  # 均匀 -> q=1
    assert q_eff(1.0, 7) is None                              # warm-up 占位 error=1
    assert k_eff_from_error(1.0) is None                      # 不得除零
    assert q_eff(0.61, 7) == pytest.approx(1 / (1 - 0.61) ** 2 / 7)
    with pytest.raises(ContractViolation):
        q_eff(0.5, None)
    with pytest.raises(ContractViolation):
        q_eff(None, 7)


# ---------------- 行守恒（blocker 3）----------------

def test_conservation_expansion_1_to_2_uses_pre_d0_base():
    """sigma01：1 物理行 -> 2 agent-值。基数必须是**未排除 D0** 的物理行数。

    preflight C1 实测：dev+guard 物理行 2,598,536，其中 D0（frame_index<4）107,544，
    差 2,490,992。v1 拿 2,490,992 当 identity_1 基数，会让 identity_2 的
    NOT_ATTEMPTED 项恒为 0——断言退化成空话。
    """
    rep = check_conservation("interhub_sigma01_hw4_timeseries", 2_598_536, 2, 1,
                             {ATTEMPTED: 4_500_000, NOT_ATTEMPTED: 215_088,
                              UNKNOWN: 481_984},
                             {"L1_DIRECT": 5_197_072})
    assert rep.measurement_rows_expected == 5_197_072
    assert rep.status_counts[NOT_ATTEMPTED] == 107_544 * 2   # D0 行必须可见


def test_conservation_feature_matrix_expansion_2_no_collapse():
    """C3/C4：三个角色实际在 feature matrix，不在 predictions。

    predictions.parquet 的 15 列里没有任何 ipv_error，3x alpha 折叠只属于它；
    feature matrix 每 (anchor, perspective) 恰 1 行，故 E=2 / C=1。
    实测 dev+guard 行 4,497,368。
    """
    rep = check_conservation("rq009_feature_matrix", 4_497_368, 2, 1,
                             {ATTEMPTED: 8_994_736},
                             {"L1_DIRECT": 8_994_736})
    assert rep.measurement_rows_expected == 8_994_736


def test_conservation_expansion_1_to_4():
    """OnSite dense：1 物理行 -> 4 通道。"""
    assert 70_317 * 4 == 281_268
    rep = check_conservation("onsite_dense_timeseries", 70_317, 4, 1,
                             {ATTEMPTED: 275_000, NOT_ATTEMPTED: 4_272,
                              UNKNOWN: 1_996},
                             {"L1_DIRECT": 281_268})
    assert rep.measurement_rows_expected == 281_268


def test_conservation_collapse_3_to_1_generic_capability():
    """3->1 折叠能力（通用）。

    注意：这**不再**描述 rq009_feature_matrix。predictions.parquet 确实有 3x alpha
    （实测 fold=test 3,811,698 == 1,270,566 * 3），但它没有 ipv_error，已移出台账。
    本用例只保留折叠算术能力的覆盖。
    """
    rep = check_conservation("generic_collapsing_artifact", 5_335_782, 1, 3,
                             {ATTEMPTED: 1_778_594}, {"L1_DIRECT": 1_778_594})
    assert rep.measurement_rows_expected == 1_778_594


def test_conservation_failures_are_fail_closed():
    with pytest.raises(ContractViolation):        # identity_1
        check_conservation("x", 10, 2, 1, {ATTEMPTED: 19}, {"L1_DIRECT": 19})
    with pytest.raises(ContractViolation):        # identity_3
        check_conservation("x", 10, 2, 1, {ATTEMPTED: 20}, {"L1_DIRECT": 19})
    with pytest.raises(ContractViolation):        # 非法状态标签
        check_conservation("x", 10, 2, 1, {"BOGUS": 20}, {"L1_DIRECT": 20})
    with pytest.raises(ContractViolation):        # 不可整除
        check_conservation("x", 10, 1, 3, {ATTEMPTED: 3}, {"L1_DIRECT": 3})


def test_check_conservation_rejects_malformed_counts_and_labels():
    with pytest.raises(ContractViolation):
        check_conservation("x", 10.5, 2, 1, {ATTEMPTED: 21}, {"L1_DIRECT": 21})
    with pytest.raises(ContractViolation):
        check_conservation("x", -10, 2, 1, {ATTEMPTED: 20}, {"L1_DIRECT": 20})
    with pytest.raises(ContractViolation):
        check_conservation("x", True, 2, 1, {ATTEMPTED: 2}, {"L1_DIRECT": 2})
    with pytest.raises(ContractViolation):
        check_conservation("x", 10, 2.0, 1, {ATTEMPTED: 20}, {"L1_DIRECT": 20})
    with pytest.raises(ContractViolation):
        check_conservation("x", 10, 2, False, {ATTEMPTED: 20}, {"L1_DIRECT": 20})
    with pytest.raises(ContractViolation):
        check_conservation("x", 10, 2, 1, {ATTEMPTED: -1, UNKNOWN: 21}, {"L1_DIRECT": 20})
    with pytest.raises(ContractViolation):
        check_conservation("x", 10, 2, 1, {ATTEMPTED: True, UNKNOWN: 19}, {"L1_DIRECT": 20})
    with pytest.raises(ContractViolation):
        check_conservation("x", 10, 2, 1, {ATTEMPTED: 20}, {"L1_DIRECT": False})
    with pytest.raises(ContractViolation):
        check_conservation("x", 10, 2, 1, {ATTEMPTED: 20}, {"BOGUS": 20})


# ---------------- OnSite 局部序号 ----------------

def test_local_positions_not_frame_index_minus_min():
    """首帧非 0 且不连续时，局部序号必须靠排序而非减最小值。"""
    rows = [(1000, 7), (1040, 9), (1020, 8), (1100, 20)]     # frame_index 有跳跃
    pos = local_positions(rows)
    assert pos == [0, 2, 1, 3]
    naive = [fi - min(f for _, f in rows) for _, fi in rows]
    assert naive == [0, 2, 1, 13] and naive != pos           # 朴素规则给出 13，错


def test_local_position_d0_counts_differ_from_global_rule():
    rows = [(100 * i, 50 + i) for i in range(10)]            # frame_index 全部 >= 4
    pos = local_positions(rows)
    assert sum(1 for p in pos if p < 4) == 4                 # 局部规则命中 4 行
    assert sum(1 for _, fi in rows if fi < 4) == 0           # 全局规则命中 0 行


# ---------------- L1→L2→L3 ----------------

def _row(case, persp, cfg, status, q, artifact_id="interhub_sigma01_hw4_timeseries"):
    return {"artifact_id": artifact_id, "case_id": case, "perspective": persp,
            "configuration": cfg, "attempt_status": status, "q_eff": q}


def test_l2_minimum_support_and_l3_zero_support():
    rows = ([_row("c1", "p1", "hw4", ATTEMPTED, 0.4) for _ in range(5)]
            + [_row("c1", "p2", "hw4", ATTEMPTED, 0.9) for _ in range(3)]   # <5 -> 不足
            + [_row("c2", "p1", "hw4", UNKNOWN, None) for _ in range(6)])
    l2 = aggregate_l2(rows)
    by = {(u.case_id, u.perspective): u for u in l2}
    assert by[("c1", "p1")].status == "OK" and by[("c1", "p1")].mean_q_eff == pytest.approx(0.4)
    assert by[("c1", "p2")].status == "INSUFFICIENT_SUPPORT"
    assert by[("c2", "p1")].status == "OK" and by[("c2", "p1")].mean_q_eff is None

    l3 = {u.case_id: u for u in aggregate_l3(l2)}
    assert l3["c1"].mean_q_eff == pytest.approx(0.4)     # 不足支持的 L2 被排除
    assert l3["c1"].n_l2_ok == 1 and l3["c1"].n_l2_total == 2
    assert l3["c2"].status == "ZERO_SUPPORT" and l3["c2"].mean_q_eff is None
    assert {u.artifact_id for u in l2} == {"interhub_sigma01_hw4_timeseries"}


def test_aggregate_l2_nullable_keys_sort_none_before_strings():
    rows = ([_row(None, None, "hw4", ATTEMPTED, 0.2, artifact_id="onsite") for _ in range(5)]
            + [_row("s", "p1", None, ATTEMPTED, 0.4, artifact_id="onsite") for _ in range(5)])
    l2 = aggregate_l2(rows)
    assert [(u.case_id, u.perspective, u.configuration, u.mean_q_eff) for u in l2] == [
        (None, None, "hw4", 0.2),
        ("s", "p1", None, 0.4),
    ]
    l3 = aggregate_l3(l2)
    assert [u.case_id for u in l3] == [None, "s"]


def test_aggregate_l3_rejects_cross_artifact_l2_units():
    units = [
        L2Unit("c1", "p1", "hw4", 5, 5, 0, 0.4, "OK", artifact_id="A"),
        L2Unit("c1", "p2", "hw4", 5, 5, 0, 0.6, "OK", artifact_id="B"),
    ]
    with pytest.raises(ContractViolation):
        aggregate_l3(units)


def test_aggregation_is_permutation_invariant_and_deterministic():
    rows = [_row("c1", "p1", "hw4", ATTEMPTED, 0.1 * i) for i in range(1, 6)]
    a = aggregate_l2(rows)
    b = aggregate_l2(list(reversed(rows)))
    assert a == b, "聚合必须对输入顺序逐位确定（sorted + fsum）"
    assert a[0].mean_q_eff == 0.3          # 逐位等于 0.3，而非 0.30000000000000004
    assert [u.case_id for u in aggregate_l3(a)] == ["c1"]
    # episode 摘要同样要求逐位确定
    import random
    ipvs = [0.11, -0.27, 0.53, 0.02, -0.41]
    qs = [0.2, 0.5, 0.3, 0.9, 0.44]
    base = episode_summaries(ipvs, qs)
    for seed in range(5):
        idx = list(range(5)); random.Random(seed).shuffle(idx)
        shuffled = episode_summaries([ipvs[i] for i in idx], [qs[i] for i in idx])
        assert shuffled == base, "episode 摘要对输入顺序必须逐位确定"


def test_aggregate_l2_rejects_invalid_attempt_status_minimal_counterexample():
    rows = [_row("c1", "p1", "hw4", "BOGUS", 0.2, artifact_id="A")
            for _ in range(5)]
    with pytest.raises(ContractViolation, match="invalid attempt_status"):
        aggregate_l2(rows)


def test_aggregate_l2_rejects_invalid_q_eff_values():
    bad_cases = [
        (ATTEMPTED, 2.0),
        (ATTEMPTED, 0.0),
        (ATTEMPTED, float("inf")),
        (ATTEMPTED, float("nan")),
        (ATTEMPTED, True),
        (ATTEMPTED, "0.5"),
        (UNKNOWN, 2.0),
    ]
    for status, bad_q in bad_cases:
        rows = [_row("c1", "p1", "hw4", status, bad_q, artifact_id="A")
                for _ in range(5)]
        with pytest.raises(ContractViolation):
            aggregate_l2(rows)


# ---------------- episode 摘要（不使用 bins）----------------

def test_episode_definition_sensitivity():
    ipvs = [0.5, -0.5, 0.0]
    qs = [0.2, 0.9, None]                      # 第三帧 q 缺失 -> 同步剔除
    out = episode_summaries(ipvs, qs)
    assert out["n_used"] == 2
    assert out["unweighted"] == pytest.approx(0.0)
    # w = 1-q -> (0.8, 0.1)；加权均值偏向更集中的那帧
    assert out["concentration_wtd"] == pytest.approx((0.5 * 0.8 + -0.5 * 0.1) / 0.9)
    assert out["concentration_wtd"] > out["unweighted"]
    assert episode_summaries([], [])["unweighted"] is None


# ---------------- bins 仅描述 + 稳定性 ----------------

def test_band_shares_and_instability_verdict():
    qs = [0.2, 0.3, 0.6, 0.95, 0.99, None]
    s = band_shares(qs, *REPORT_BINS_PRIMARY)
    assert s["n"] == 5 and s["CONCENTRATED"] == pytest.approx(40.0)
    assert s["NEAR_UNIFORM"] == pytest.approx(40.0)
    with pytest.raises(ContractViolation):
        band_shares(qs, 0.9, 0.5)                     # lo>=hi
    st = bins_stability([0.5, 0.55, 0.6, 0.62, 0.66])  # 全部挤在 lo 附近 -> 不稳定
    assert st["verdict"] == "BINS_WITHHELD_UNSTABLE"
    assert bins_stability([0.05] * 50)["verdict"] == "BINS_REPORTABLE"


# ---------------- C0 路由（连续量，不依赖 bins）----------------

def test_c0_four_terminals_and_priority():
    assert c0_route(False, 0, 0, 0, [], True)["terminal"] == "NOT_APPLICABLE"
    # unknown 占比高 -> INDETERMINATE（优先级最高）
    r = c0_route(True, 100, 50, 30, [0.99] * 20, True)
    assert r["terminal"] == "INDETERMINATE_UNKNOWN_PROVENANCE"
    # 映射非 1:1 也走 INDETERMINATE
    assert c0_route(True, 100, 0, 0, [0.1] * 100, False)["terminal"] \
        == "INDETERMINATE_UNKNOWN_PROVENANCE"
    # 不可用占比超阈 -> 需重估
    r = c0_route(True, 100, 10, 0, [0.1] * 90, True)
    assert r["terminal"] == "OWNER_REANALYSIS_REQUIRED"
    assert r["reason_code"] == "unavailable_share_ge_cut"
    # 平均集中度差 -> 也需重估
    r = c0_route(True, 100, 0, 0, [0.85] * 100, True)
    assert r["terminal"] == "OWNER_REANALYSIS_REQUIRED" and r["reason_code"] == "mean_q_eff_ge_cut"
    # 全部低于阈值 -> 未触发
    assert c0_route(True, 100, 1, 0, [0.2] * 99, True)["terminal"] == "NO_AUDIT_TRIGGER_DETECTED"


def test_c0_routing_never_consumes_report_bins():
    """路由只吃连续量：同一批数据换 report bins 不影响路由结果。"""
    kw = dict(uses_ipv=True, n_rows=100, n_not_attempted=1, n_unknown=0,
              q_effs_attempted=[0.2] * 99, mapping_is_1to1=True)
    base = c0_route(**kw)["terminal"]
    for lo, hi in ((0.45, 0.90), (0.65, 0.96)):
        _ = band_shares(kw["q_effs_attempted"], lo, hi)   # 改 bins 不参与路由
        assert c0_route(**kw)["terminal"] == base


def test_c0_sensitivity_reports_stability():
    out = c0_route_with_sensitivity(uses_ipv=True, n_rows=100, n_not_attempted=0,
                                   n_unknown=0, q_effs_attempted=[0.1] * 100,
                                   mapping_is_1to1=True)
    assert out["primary"]["terminal"] == "NO_AUDIT_TRIGGER_DETECTED" and out["stable"]
    out2 = c0_route_with_sensitivity(uses_ipv=True, n_rows=100, n_not_attempted=7,
                                     n_unknown=0, q_effs_attempted=[0.1] * 93,
                                     mapping_is_1to1=True)
    assert not out2["stable"]          # 7% 落在 2%/5%/10% 之间 -> 不稳定，必须披露


def test_c0_route_rejects_negative_count_minimal_counterexample():
    with pytest.raises(ContractViolation):
        c0_route(True, 100, -1, 0, [0.2] * 101, True)


def test_c0_route_rejects_non_integral_and_bool_counts():
    bad_calls = [
        dict(uses_ipv=True, n_rows=100.0, n_not_attempted=0, n_unknown=0,
             q_effs_attempted=[0.2] * 100, mapping_is_1to1=True),
        dict(uses_ipv=True, n_rows=True, n_not_attempted=0, n_unknown=0,
             q_effs_attempted=[0.2] * 100, mapping_is_1to1=True),
        dict(uses_ipv=True, n_rows=100, n_not_attempted=1.0, n_unknown=0,
             q_effs_attempted=[0.2] * 99, mapping_is_1to1=True),
        dict(uses_ipv=True, n_rows=100, n_not_attempted=False, n_unknown=0,
             q_effs_attempted=[0.2] * 100, mapping_is_1to1=True),
        dict(uses_ipv=True, n_rows=100, n_not_attempted=0, n_unknown=None,
             q_effs_attempted=[0.2] * 100, mapping_is_1to1=True),
    ]
    for kw in bad_calls:
        with pytest.raises(ContractViolation):
            c0_route(**kw)


def test_c0_route_rejects_count_parts_exceeding_total():
    with pytest.raises(ContractViolation, match="exceeds n_rows"):
        c0_route(True, 5, 4, 2, [0.2], True)


def test_c0_route_with_sensitivity_rejects_shared_bad_counts():
    with pytest.raises(ContractViolation):
        c0_route_with_sensitivity(uses_ipv=True, n_rows=100,
                                  n_not_attempted=-1, n_unknown=0,
                                  q_effs_attempted=[0.2] * 101,
                                  mapping_is_1to1=True)


# ---------------- schema 自检 ----------------

def test_schema_loads_and_is_consistent():
    d = load_schema(SCHEMA)
    ids = [a["artifact_id"] for a in d["artifacts"]]
    assert len(ids) == len(set(ids)) == 7
    assert d["ledger_bearing_artifact_ids"] == [
        "interhub_sigma01_hw4_timeseries", "rq009_feature_matrix", "onsite_dense_timeseries"]
    by = {a["artifact_id"]: a for a in d["artifacts"]}
    assert by["interhub_sigma01_hw4_timeseries"]["expansion_factor"] == 2
    assert by["onsite_dense_timeseries"]["expansion_factor"] == 4
    assert by["rq009_m3_predictions"]["status"] == "PROVENANCE_ONLY_NOT_IN_LEDGER"
    # OnSite 必须用局部序号规则，且明文禁止 frame_index-min
    rule = by["onsite_dense_timeseries"]["not_attempted_rule"]
    assert rule["kind"] == "local_position"
    assert any("frame_index - min" in s for s in rule["local_position_algorithm"])
    # M3 join 键必须四元且含 source_dataset
    keys = by["rq009_feature_matrix"]["row_key_fields"]
    assert keys == ["case_key", "anchor_frame_index", "perspective", "source_dataset"]
    # split 过滤必须白名单且禁止 != 'sealed'
    sf = d["split_filter"]
    assert sf["allowlist"] == ["development", "guard"]
    assert "split != 'sealed'" in sf["forbidden_expressions"]
    assert sf["assertions"]["held_out_parsed_rows"] == 0
    # 跨产物 pooling 必须禁止
    assert d["cross_artifact_pooling"]["policy"] == "FORBIDDEN"
    # 守恒三恒等式齐备
    rc = d["row_conservation"]
    assert all(k in rc for k in ("identity_1_expansion", "identity_2_partition",
                                 "identity_3_recoverability"))


def test_schema_rejects_tampering(tmp_path):
    d = json.loads(SCHEMA.read_text())
    d["schema_id"] = "wrong"
    p = tmp_path / "s.json"; p.write_text(json.dumps(d))
    with pytest.raises(ContractViolation):
        load_schema(p)
    d = json.loads(SCHEMA.read_text())
    d["artifacts"].append(dict(d["artifacts"][0]))
    p2 = tmp_path / "s2.json"; p2.write_text(json.dumps(d))
    with pytest.raises(ContractViolation):
        load_schema(p2)


def test_load_schema_accepts_real_v2_schema():
    d = load_schema(SCHEMA)
    assert d["schema_id"] == "rq015a-concentration-ledger-v2"
    assert any(a["artifact_id"] == "rq014_g2r_anchor_scores"
               for a in d["non_ledger_artifacts"])


def test_load_schema_rejects_ledger_missing_expansion_factor(tmp_path):
    d = json.loads(SCHEMA.read_text())
    d["artifacts"][0].pop("expansion_factor")
    p = tmp_path / "missing_expansion.json"; p.write_text(json.dumps(d))
    with pytest.raises(ContractViolation, match="missing expansion_factor"):
        load_schema(p)


def test_load_schema_rejects_ledger_zero_expansion_factor(tmp_path):
    d = json.loads(SCHEMA.read_text())
    d["artifacts"][0]["expansion_factor"] = 0
    p = tmp_path / "zero_expansion.json"; p.write_text(json.dumps(d))
    with pytest.raises(ContractViolation, match="bad expansion_factor"):
        load_schema(p)


def test_load_schema_rejects_non_integral_constant_k_source(tmp_path):
    d = json.loads(SCHEMA.read_text())
    d["artifacts"][0]["K_source"]["value"] = 7.9
    p = tmp_path / "float_k_source.json"; p.write_text(json.dumps(d))
    with pytest.raises(ContractViolation, match="K_source.value"):
        load_schema(p)

    d = json.loads(SCHEMA.read_text())
    d["artifacts"][0]["K_source"]["value"] = True
    p = tmp_path / "bool_k_source.json"; p.write_text(json.dumps(d))
    with pytest.raises(ContractViolation, match="K_source.value"):
        load_schema(p)


def test_load_schema_rejects_v1_schema_id(tmp_path):
    d = json.loads(SCHEMA.read_text())
    d["schema_id"] = "rq015a-concentration-ledger-v1"
    p = tmp_path / "v1.json"; p.write_text(json.dumps(d))
    with pytest.raises(ContractViolation, match="schema_id mismatch"):
        load_schema(p)


def test_load_schema_records_non_ledger_entries_by_status(tmp_path):
    d = {
        "schema_id": "rq015a-concentration-ledger-v2",
        "artifacts": [
            {"artifact_id": "ledger", "expansion_factor": 1, "collapse_factor": 1},
            {"artifact_id": "provenance", "status": "PROVENANCE_ONLY_NOT_IN_LEDGER"},
            {"artifact_id": "absent", "status": "ARTIFACT_NOT_PRESENT_LOCALLY",
             "expansion_factor": 0},
        ],
    }
    p = tmp_path / "non_ledger.json"; p.write_text(json.dumps(d))
    out = load_schema(p)
    assert out["ledger_bearing_artifact_ids"] == ["ledger"]
    assert out["non_ledger_artifacts"] == [
        {"artifact_id": "provenance", "status": "PROVENANCE_ONLY_NOT_IN_LEDGER"},
        {"artifact_id": "absent", "status": "ARTIFACT_NOT_PRESENT_LOCALLY"},
    ]


def test_load_schema_rejects_duplicate_artifact_id(tmp_path):
    d = json.loads(SCHEMA.read_text())
    d["artifacts"].append(dict(d["artifacts"][0]))
    p = tmp_path / "duplicate.json"; p.write_text(json.dumps(d))
    with pytest.raises(ContractViolation, match="duplicate artifact_id"):
        load_schema(p)


def test_aggregate_l2_missing_perspective_raises_contract_violation():
    row = _row("c1", "p1", "hw4", ATTEMPTED, 0.3)
    row.pop("perspective")
    with pytest.raises(ContractViolation, match="interhub_sigma01_hw4_timeseries: missing perspective"):
        aggregate_l2([row])


def test_aggregate_l2_missing_configuration_raises_contract_violation():
    row = _row("c1", "p1", "hw4", ATTEMPTED, 0.3)
    row.pop("configuration")
    with pytest.raises(ContractViolation, match="interhub_sigma01_hw4_timeseries: missing configuration"):
        aggregate_l2([row])


def test_aggregate_l3_missing_key_raises_contract_violation():
    with pytest.raises(ContractViolation, match="L2 unit missing case_id"):
        aggregate_l3([{"status": "OK", "mean_q_eff": 0.3}])


# ---------------- v3 复审：三处 fail-open 的关闭测试 ----------------

def test_q_eff_is_fail_closed_not_truncating():
    """v3 的 min(val,1.0) 会把非法 ipv_error=1.1 静默变成合法的 1.0。"""
    from rq015a_contracts import q_eff as qe
    with pytest.raises(ContractViolation):
        qe(1.1, 7)                       # 越界 -> 必须报错，不得返回 1.0
    with pytest.raises(ContractViolation):
        qe(-0.01, 7)
    with pytest.raises(ContractViolation):
        qe(float("inf"), 7)
    with pytest.raises(ContractViolation):
        qe(0.5, 0)                       # K < 1
    with pytest.raises(ContractViolation):
        qe(0.9, 2)                       # k_eff=100 > K=2，数据自相矛盾
    with pytest.raises(ContractViolation):
        qe(True, 7)                      # bool 不是合法输入
    assert qe(1.0, 7) is None            # 退化仍走 UNKNOWN 而非报错
    with pytest.raises(ContractViolation):
        qe(None, 7)
    with pytest.raises(ContractViolation):
        qe(0.5, None)


def test_k_eff_and_q_eff_domains_are_fail_closed():
    with pytest.raises(ContractViolation):
        k_eff_from_error(1.1)
    with pytest.raises(ContractViolation):
        k_eff_from_error(-0.1)
    with pytest.raises(ContractViolation):
        k_eff_from_error(float("inf"))
    with pytest.raises(ContractViolation):
        k_eff_from_error(float("nan"))
    with pytest.raises(ContractViolation):
        k_eff_from_error(True)
    with pytest.raises(ContractViolation):
        q_eff(0.5, 7.0)
    with pytest.raises(ContractViolation):
        q_eff(0.5, True)


def test_legal_numeric_outputs_match_pre_fix_baseline():
    rows = ([_row("c2", "p1", "hw4", ATTEMPTED, 0.1 * i, artifact_id="A")
             for i in range(1, 6)]
            + [_row("c2", "p2", "hw4", ATTEMPTED, 0.2, artifact_id="A")
               for _ in range(5)]
            + [_row("c3", "p1", "hw4", UNKNOWN, None, artifact_id="A")
               for _ in range(5)])
    l2 = aggregate_l2(rows)
    l3 = aggregate_l3(l2)
    episode = episode_summaries(
        [0.11, -0.27, 0.53, 0.02, -0.41],
        [0.2, 0.5, 0.3, 0.9, 0.44])
    route = c0_route(True, 100, 1, 0, [0.2] * 99, True)

    assert k_eff_from_error(0.61).hex() == "0x1.a4c69b2e9f38bp+2"
    assert q_eff(0.61, 7).hex() == "0x1.e0e2fa7e6cd31p-1"
    assert [None if u.mean_q_eff is None else u.mean_q_eff.hex() for u in l2] == [
        "0x1.3333333333333p-2",
        "0x1.999999999999ap-3",
        None,
    ]
    assert [None if u.mean_q_eff is None else u.mean_q_eff.hex() for u in l3] == [
        "0x1.0000000000000p-2",
        None,
    ]
    assert episode["unweighted"].hex() == "-0x1.0624dd2f1a9f4p-8"
    assert episode["concentration_wtd"].hex() == "0x1.28e20cc7df5afp-5"
    assert band_shares([0.2, 0.3, 0.6, 0.95, 0.99, None], *REPORT_BINS_PRIMARY) == {
        "CONCENTRATED": 40.0,
        "INTERMEDIATE": 20.0,
        "NEAR_UNIFORM": 40.0,
        "n": 5,
    }
    assert route["terminal"] == "NO_AUDIT_TRIGGER_DETECTED"
    assert route["metrics"]["mean_q_eff_attempted"].hex() == "0x1.999999999999ap-3"


def test_cross_artifact_pooling_is_code_enforced():
    from rq015a_contracts import assert_single_artifact
    rows_a = [_row("c1", "p1", "hw4", ATTEMPTED, 0.3, artifact_id="A") for _ in range(5)]
    rows_b = [_row("c1", "p1", "hw4", ATTEMPTED, 0.3, artifact_id="B")]
    assert assert_single_artifact(rows_a) == "A"
    with pytest.raises(ContractViolation):
        assert_single_artifact(rows_a + rows_b)
    with pytest.raises(ContractViolation):
        aggregate_l2(rows_a + rows_b)            # 聚合入口也必须拦
    no_id = _row("c1", "p1", "hw4", ATTEMPTED, 0.3); no_id.pop("artifact_id")
    with pytest.raises(ContractViolation):
        assert_single_artifact([no_id])                    # 缺 artifact_id
    assert aggregate_l2(rows_a)[0].n_l1 == 5     # 单一产物照常工作


def test_c0_without_q_evidence_is_indeterminate():
    """有 ATTEMPTED 行却拿不到任何 q 值时，不得判 NO_AUDIT_TRIGGER_DETECTED。"""
    r = c0_route(True, 100, 0, 0, [], True)
    assert r["terminal"] == "INDETERMINATE_UNKNOWN_PROVENANCE"
    assert r["reason_code"] == "attempted_rows_without_q_evidence"
    r2 = c0_route(True, 100, 0, 0, [None] * 100, True)   # 全是 None 也算无证据
    assert r2["terminal"] == "INDETERMINATE_UNKNOWN_PROVENANCE"
    # 全部 NOT_ATTEMPTED（n_attempted=0）不受此规则影响，仍按占比路由
    r3 = c0_route(True, 100, 100, 0, [], True)
    assert r3["terminal"] == "OWNER_REANALYSIS_REQUIRED"
