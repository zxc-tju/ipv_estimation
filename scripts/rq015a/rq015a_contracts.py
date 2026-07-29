#!/usr/bin/env python3
"""RQ015A 唯一算法实现（可执行合同）。

关闭 v2 三路复审的 blocker 3/4：raw-row 守恒必须处理 1→2 / 1→4 展开与 M3 3→1 折叠；
L1–L3 聚合、episode 摘要、因素分析与 C0 路由必须是**唯一算法**而非 prose。

状态：`BUILD_WHILE_DENY`。本模块不读取任何真实数据、不被生产路径导入。
计划：`reports/plans/RQ015A_plan_v3_concentration_audit_20260726.md`
schema：`reports/plans/RQ015A_ledger_schema_v2.json`

**关键设计（关闭 blocker 1/4 的冲突）**：下游（episode 摘要、C0 路由）**只使用连续量**，
不使用报告用的 policy bins。bins 仅出现在描述性摘要里，且与任何判定无关。
"""

from __future__ import annotations

import json
import math
import numbers
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SCHEMA_VERSION = "rq015a-concentration-ledger-v2"
NON_LEDGER_STATUSES = {"PROVENANCE_ONLY_NOT_IN_LEDGER", "ARTIFACT_NOT_PRESENT_LOCALLY"}

ATTEMPTED = "ATTEMPTED"
NOT_ATTEMPTED = "NOT_ATTEMPTED"
UNKNOWN = "UNKNOWN"
LEDGER_STATUSES = {ATTEMPTED, NOT_ATTEMPTED, UNKNOWN}
RECOVERABILITY_LABELS = {
    "L1_DIRECT",
    "L2_PROVENANCE",
    "L3_PENDING_RQ015B",
    "L4_UNRECOVERABLE",
    "RECOVERABLE_BY_REPLAY_OUT_OF_SCOPE",
    "ARTIFACT_NOT_PRESENT_LOCALLY",
}

# C0 路由的 triage 阈值：**operational triage**，不是科学边界，不由数据导出。
ROUTING_PRIMARY = {"unavailable_share": 0.05, "mean_q_eff": 0.80, "unknown_share": 0.20}
ROUTING_SENSITIVITY = (
    {"unavailable_share": 0.02, "mean_q_eff": 0.70, "unknown_share": 0.10},
    {"unavailable_share": 0.05, "mean_q_eff": 0.80, "unknown_share": 0.20},
    {"unavailable_share": 0.10, "mean_q_eff": 0.90, "unknown_share": 0.30},
)
# 报告用 policy bins（仅描述，禁止进入任何判定）
REPORT_BINS_PRIMARY = (4.0 / 7.0, 0.93)
REPORT_BINS_SENSITIVITY = tuple(
    (lo, hi) for lo in (0.45, 4.0 / 7.0, 0.65) for hi in (0.90, 0.93, 0.96)
)
BINS_INSTABILITY_PP = 10.0   # 三档占比极差 > 10pp -> BINS_WITHHELD_UNSTABLE

MIN_SUPPORT_L1_PER_L2 = 5


def _det_mean(values) -> Optional[float]:
    """逐位确定的均值：先升序排序再 fsum，使结果不依赖输入顺序。

    朴素 `sum(...)/n` 在不同输入顺序下会给出 0.3 与 0.30000000000000004，
    违反复审要求的"唯一算法"。
    """
    vals = sorted(float(v) for v in values)
    if not vals:
        return None
    return math.fsum(vals) / len(vals)


class ContractViolation(RuntimeError):
    """守恒或合同断言失败（fail closed）。"""


def _require_integral(value: object, name: str, artifact_id: str,
                      minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ContractViolation(f"{artifact_id}: {name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ContractViolation(f"{artifact_id}: {name} must be >= {minimum}")
    return value


def _validate_count_mapping(artifact_id: str, name: str, counts: Dict[str, int],
                            allowed_keys: set) -> Dict[str, int]:
    if not isinstance(counts, dict):
        raise ContractViolation(f"{artifact_id}: {name} must be a dict")
    extra = set(counts) - allowed_keys
    if extra:
        raise ContractViolation(
            f"{artifact_id}: unknown {name} labels {sorted(extra)}")
    return {
        key: _require_integral(value, f"{name}[{key!r}]", artifact_id, 0)
        for key, value in counts.items()
    }


def _validate_ipv_error(ipv_error: object) -> float:
    if isinstance(ipv_error, bool) or not isinstance(ipv_error, numbers.Real):
        raise ContractViolation(f"invalid ipv_error {ipv_error!r}")
    ipv_error = float(ipv_error)
    if not math.isfinite(ipv_error):
        raise ContractViolation(f"non-finite ipv_error {ipv_error!r}")
    if ipv_error < 0.0 or ipv_error > 1.0:
        raise ContractViolation(f"ipv_error {ipv_error!r} outside [0, 1]")
    return ipv_error


def _validate_attempt_status(status: object, artifact_id: str) -> str:
    if not isinstance(status, str) or status not in LEDGER_STATUSES:
        raise ContractViolation(f"{artifact_id}: invalid attempt_status {status!r}")
    return status


def _validate_optional_q_eff(value: object, artifact_id: str) -> Optional[float]:
    """Validate a consumed q_eff value.

    The accepted numeric path is intentionally float-based.  `Decimal` remains
    rejected even for numerically valid values so the deterministic aggregation
    contract stays on one `float` + `math.fsum` path.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ContractViolation(f"{artifact_id}: invalid q_eff {value!r}")
    q = float(value)
    if not math.isfinite(q):
        raise ContractViolation(f"{artifact_id}: non-finite q_eff {value!r}")
    if q <= 0.0 or q > 1.0:
        raise ContractViolation(f"{artifact_id}: q_eff {value!r} outside (0, 1]")
    return q


def _nullable_string_sort_key(value: object) -> Tuple[bool, str]:
    if value is None:
        return (False, "")
    if isinstance(value, str):
        return (True, value)
    raise ContractViolation(f"expected string or None sort key, got {value!r}")


def _group_key_sort_key(key: Tuple[object, object, object]) -> Tuple[Tuple[bool, str],
                                                                    Tuple[bool, str],
                                                                    Tuple[bool, str]]:
    return tuple(_nullable_string_sort_key(part) for part in key)


# ---------------------------------------------------------------- 基础派生量

def k_eff_from_error(ipv_error: float) -> Optional[float]:
    ipv_error = _validate_ipv_error(ipv_error)
    denom = (1.0 - ipv_error) ** 2
    if denom <= 0.0:
        return None                      # ipv_error == 1（warm-up 占位）
    return 1.0 / denom


def q_eff(ipv_error: float, K: int) -> Optional[float]:
    """主量。**fail-closed**：非法 ipv_error 或 K 一律抛错，绝不静默截断。

    v3 的 `min(val, 1.0)` 是 fail-open：`ipv_error=1.1` 会返回看似合法的 1.0。
    合法域：`ipv_error ∈ [0, 1]` 且 `K ≥ 1`；`k_eff` 必须 `≤ K`（否则数据自相矛盾）。
    退化（error=1，含 warm-up 占位）返回 None，调用方须置 `attempt_status=UNKNOWN`。
    `Decimal` 有意拒绝：R7 确定性求和只保留 `float` + `math.fsum` 一条数值路径。
    """
    ipv_error = _validate_ipv_error(ipv_error)
    K = _require_integral(K, "K", "q_eff", 1)
    ke = k_eff_from_error(ipv_error)
    if ke is None:                       # error == 1 -> 退化
        return None
    if ke > K * (1.0 + 1e-9):            # 有效候选数不可能超过候选总数
        raise ContractViolation(
            f"k_eff {ke!r} exceeds K {K}: ipv_error/K inconsistent")
    val = min(ke / K, 1.0)               # 仅吸收 1e-9 级浮点噪声
    if not math.isfinite(val) or val <= 0.0:
        raise ContractViolation(f"degenerate q_eff {val!r}")
    return val


# ---------------------------------------------------------------- 行守恒

@dataclass(frozen=True)
class ConservationReport:
    artifact_id: str
    physical_rows: int
    expansion_factor: int
    collapse_factor: int
    measurement_rows_expected: int
    measurement_rows_observed: int
    status_counts: Dict[str, int]
    recoverability_counts: Dict[str, int]

    def assert_ok(self) -> None:
        status_counts = _validate_count_mapping(
            self.artifact_id, "status_counts", self.status_counts, LEDGER_STATUSES)
        recoverability_counts = _validate_count_mapping(
            self.artifact_id, "recoverability_counts", self.recoverability_counts,
            RECOVERABILITY_LABELS)
        if self.measurement_rows_observed != self.measurement_rows_expected:
            raise ContractViolation(
                f"{self.artifact_id}: identity_1 failed "
                f"{self.physical_rows}*{self.expansion_factor}/{self.collapse_factor}"
                f"={self.measurement_rows_expected} != {self.measurement_rows_observed}")
        s = sum(status_counts.values())
        if s != self.measurement_rows_observed:
            raise ContractViolation(
                f"{self.artifact_id}: identity_2 failed {s} != {self.measurement_rows_observed}")
        r = sum(recoverability_counts.values())
        if r != self.measurement_rows_observed:
            raise ContractViolation(
                f"{self.artifact_id}: identity_3 failed {r} != {self.measurement_rows_observed}")


def check_conservation(artifact_id: str, physical_rows: int, expansion_factor: int,
                       collapse_factor: int, status_counts: Dict[str, int],
                       recoverability_counts: Dict[str, int]) -> ConservationReport:
    physical_rows = _require_integral(physical_rows, "physical_rows", artifact_id, 0)
    expansion_factor = _require_integral(expansion_factor, "expansion_factor", artifact_id, 1)
    collapse_factor = _require_integral(collapse_factor, "collapse_factor", artifact_id, 1)
    status_counts = _validate_count_mapping(
        artifact_id, "status_counts", status_counts, LEDGER_STATUSES)
    recoverability_counts = _validate_count_mapping(
        artifact_id, "recoverability_counts", recoverability_counts, RECOVERABILITY_LABELS)
    num = physical_rows * expansion_factor
    if num % collapse_factor != 0:
        raise ContractViolation(
            f"{artifact_id}: {num} not divisible by collapse_factor {collapse_factor}")
    rep = ConservationReport(
        artifact_id=artifact_id, physical_rows=physical_rows,
        expansion_factor=expansion_factor, collapse_factor=collapse_factor,
        measurement_rows_expected=num // collapse_factor,
        measurement_rows_observed=sum(status_counts.values()),
        status_counts=status_counts,
        recoverability_counts=recoverability_counts)
    rep.assert_ok()
    return rep


# ---------------------------------------------------------------- OnSite 局部序号

def local_positions(rows: Sequence[Tuple[int, int]]) -> List[int]:
    """OnSite D0 判据：按 (timestamp_ms, frame_index) 稳定升序取 0-based 序号。

    禁止 `frame_index - min(frame_index)`（254/267 cases 首帧非 0，36 cases 不连续）。
    `rows` 为单个 case 的 (timestamp_ms, frame_index) 序列。
    """
    order = sorted(range(len(rows)), key=lambda i: (rows[i][0], rows[i][1]))
    pos = [0] * len(rows)
    for rank, idx in enumerate(order):
        pos[idx] = rank
    return pos


def validate_artifact_id(value: object) -> str:
    if not isinstance(value, str):
        raise ContractViolation("every row must carry string artifact_id")
    if not value.strip():
        raise ContractViolation("artifact_id must be non-empty")
    if value != value.strip():
        raise ContractViolation(
            "artifact_id must not contain leading or trailing whitespace")
    return value


def assert_single_artifact(rows: Iterable[dict]) -> str:
    """跨产物 pooling 守卫（schema 声明 FORBIDDEN，此处**代码强制**）。

    M3 与 RQ009 current/target 均为 sigma01 派生；合并会重复加权同一原始 observation。
    """
    ids = {validate_artifact_id(r.get("artifact_id")) for r in rows}
    if len(ids) > 1:
        raise ContractViolation(
            f"cross-artifact pooling forbidden; got {sorted(map(str, ids))}")
    if not ids:
        raise ContractViolation("every row must carry artifact_id")
    return ids.pop()


# ---------------------------------------------------------------- L1→L2→L3

def _require_l1_key(row: dict, key: str, artifact_id: str) -> object:
    if key not in row:
        raise ContractViolation(f"{artifact_id}: missing {key}")
    return row[key]


def _require_l2_value(unit, key: str) -> object:
    if isinstance(unit, dict):
        if key not in unit:
            raise ContractViolation(f"L2 unit missing {key}")
        return unit[key]
    try:
        return getattr(unit, key)
    except AttributeError:
        raise ContractViolation(f"L2 unit missing {key}")

@dataclass(frozen=True)
class L2Unit:
    case_id: Optional[str]
    perspective: Optional[str]
    configuration: Optional[str]
    n_l1: int
    n_attempted: int
    n_unknown: int
    mean_q_eff: Optional[float]
    status: str          # OK | INSUFFICIENT_SUPPORT
    artifact_id: Optional[str] = None


def aggregate_l2(l1_rows: Iterable[dict]) -> List[L2Unit]:
    """L1 → L2。分组键 (case_id, perspective, configuration)；<5 个 L1 记 INSUFFICIENT_SUPPORT。

    `mean_q_eff` 只对 ATTEMPTED 且 q_eff 非 None 的行取平均；无此类行则为 None。
    分组内排序不影响结果（均值与计数皆置换不变）。
    """
    l1_rows = list(l1_rows)
    artifact_id = assert_single_artifact(l1_rows)  # fail-closed：禁止跨产物
    groups: Dict[Tuple[Optional[str], Optional[str], Optional[str]], List[dict]] = {}
    for r in l1_rows:
        row_artifact_id = str(r.get("artifact_id", "<missing artifact_id>"))
        groups.setdefault((
            _require_l1_key(r, "case_id", row_artifact_id),
            _require_l1_key(r, "perspective", row_artifact_id),
            _require_l1_key(r, "configuration", row_artifact_id),
        ), []).append(r)
    out: List[L2Unit] = []
    for key in sorted(groups, key=_group_key_sort_key):  # 确定性输出顺序，None 在字符串前
        rows = groups[key]
        qs = []
        statuses = []
        for r in rows:
            artifact_id = str(r.get("artifact_id", "<missing artifact_id>"))
            status = _validate_attempt_status(
                _require_l1_key(r, "attempt_status", artifact_id), artifact_id)
            statuses.append(status)
            if status == ATTEMPTED:
                q = _require_l1_key(r, "q_eff", artifact_id)
                q_value = _validate_optional_q_eff(q, artifact_id)
                if q_value is not None:
                    qs.append(q_value)
            elif "q_eff" in r:
                _validate_optional_q_eff(r["q_eff"], artifact_id)
        out.append(L2Unit(
            case_id=key[0], perspective=key[1], configuration=key[2],
            n_l1=len(rows),
            n_attempted=sum(1 for status in statuses if status == ATTEMPTED),
            n_unknown=sum(1 for status in statuses if status == UNKNOWN),
            mean_q_eff=_det_mean(qs),
            status="OK" if len(rows) >= MIN_SUPPORT_L1_PER_L2 else "INSUFFICIENT_SUPPORT",
            artifact_id=artifact_id))
    return out


@dataclass(frozen=True)
class L3Unit:
    case_id: str
    n_l2_total: int
    n_l2_ok: int
    mean_q_eff: Optional[float]
    status: str          # OK | ZERO_SUPPORT


def aggregate_l3(l2_units: Sequence[L2Unit]) -> List[L3Unit]:
    """L2 → L3。只有 status=OK 且 mean_q_eff 非 None 的 L2 参与平均（等权）。

    某 case 若无任何合格 L2 -> ZERO_SUPPORT，`mean_q_eff=None`，**不以 0 参与任何平均**。
    """
    l2_units = list(l2_units)
    if not l2_units:
        raise ContractViolation("aggregate_l3 requires at least one L2 unit")
    groups: Dict[Optional[str], List[L2Unit]] = {}
    artifact_ids = set()
    for u in l2_units:
        groups.setdefault(_require_l2_value(u, "case_id"), []).append(u)
        artifact_id = _require_l2_value(u, "artifact_id")
        artifact_ids.add(validate_artifact_id(artifact_id))
        mean_q_eff = _require_l2_value(u, "mean_q_eff")
        if mean_q_eff is not None:
            _validate_optional_q_eff(mean_q_eff, "aggregate_l3")
    if len(artifact_ids) > 1:
        raise ContractViolation(
            f"cross-artifact pooling forbidden; got {sorted(map(str, artifact_ids))}")
    out: List[L3Unit] = []
    for case_id in sorted(groups, key=_nullable_string_sort_key):
        units = groups[case_id]
        ok = [u for u in units
              if _require_l2_value(u, "status") == "OK"
              and _require_l2_value(u, "mean_q_eff") is not None]
        out.append(L3Unit(
            case_id=case_id, n_l2_total=len(units), n_l2_ok=len(ok),
            mean_q_eff=_det_mean([_require_l2_value(u, "mean_q_eff") for u in ok]),
            status="OK" if ok else "ZERO_SUPPORT"))
    return out


# ---------------------------------------------------------------- episode 摘要

def episode_summaries(ipvs: Sequence[float], q_effs: Sequence[Optional[float]]) -> Dict[str, Optional[float]]:
    """两种 episode IPV 摘要的 definition sensitivity（**不使用 bins**）。

    * `unweighted`         : 全部 ATTEMPTED 帧等权平均
    * `concentration_wtd`  : 以 `w = 1 − q_eff` 加权（越集中权重越大）
    绝不声称哪一种更准确。q_eff 为 None 的帧从两个摘要中同步剔除。
    本函数只接收已抽取的数值序列；调用方必须先对源行执行 `assert_single_artifact()`。
    """
    if len(ipvs) != len(q_effs):
        raise ContractViolation("ipvs/q_effs length mismatch")
    pairs = []
    for v, q in zip(ipvs, q_effs):
        q_value = _validate_optional_q_eff(q, "episode_summaries")
        if q_value is not None and math.isfinite(v):
            pairs.append((v, q_value))
    if not pairs:
        return {"unweighted": None, "concentration_wtd": None, "n_used": 0}
    pairs = sorted(pairs)                      # 逐位确定：先排序再 fsum
    vals = [v for v, _ in pairs]
    ws = [1.0 - q for _, q in pairs]
    wsum = math.fsum(sorted(ws))
    return {
        "unweighted": _det_mean(vals),
        "concentration_wtd": (math.fsum(sorted(v * w for (v, _), w in zip(pairs, ws))) / wsum)
                             if wsum > 0 else None,
        "n_used": len(pairs),
    }


# ---------------------------------------------------------------- 报告用 bins（仅描述）

def band_shares(q_values: Sequence[Optional[float]], lo: float, hi: float) -> Dict[str, float]:
    """三档占比（**仅描述性报告**，禁止进入任何判定或路由）。"""
    if not (0.0 < lo < hi <= 1.0):
        raise ContractViolation(f"invalid policy bins lo={lo} hi={hi}")
    vals = []
    for q in q_values:
        q_value = _validate_optional_q_eff(q, "band_shares")
        if q_value is not None:
            vals.append(q_value)
    n = len(vals)
    if n == 0:
        return {"CONCENTRATED": 0.0, "INTERMEDIATE": 0.0, "NEAR_UNIFORM": 0.0, "n": 0}
    c = sum(1 for q in vals if q <= lo)
    u = sum(1 for q in vals if q >= hi)
    return {"CONCENTRATED": 100.0 * c / n,
            "INTERMEDIATE": 100.0 * (n - c - u) / n,
            "NEAR_UNIFORM": 100.0 * u / n, "n": n}


def bins_stability(q_values: Sequence[Optional[float]]) -> Dict[str, object]:
    """九组敏感性；任一档占比极差 > 10pp -> BINS_WITHHELD_UNSTABLE。"""
    grid = {f"{lo:.6f}|{hi:.6f}": band_shares(q_values, lo, hi)
            for lo, hi in REPORT_BINS_SENSITIVITY}
    spans = {}
    for band in ("CONCENTRATED", "INTERMEDIATE", "NEAR_UNIFORM"):
        vs = [g[band] for g in grid.values()]
        spans[band] = max(vs) - min(vs)
    verdict = ("BINS_WITHHELD_UNSTABLE"
               if any(v > BINS_INSTABILITY_PP for v in spans.values()) else "BINS_REPORTABLE")
    return {"grid": grid, "spans_pp": spans, "verdict": verdict,
            "primary": band_shares(q_values, *REPORT_BINS_PRIMARY)}


# ---------------------------------------------------------------- C0 路由（连续量）

def c0_route(uses_ipv: bool, n_rows: int, n_not_attempted: int, n_unknown: int,
             q_effs_attempted: Sequence[float], mapping_is_1to1: bool,
             cuts: Dict[str, float] = ROUTING_PRIMARY) -> Dict[str, object]:
    """确定性四态路由。**只用连续量与 triage 阈值，不使用 report bins。**

    优先级：INDETERMINATE > OWNER_REANALYSIS_REQUIRED > NO_AUDIT_TRIGGER_DETECTED > NOT_APPLICABLE
    """
    n_rows = _require_integral(n_rows, "n_rows", "c0_route", 0)
    n_not_attempted = _require_integral(
        n_not_attempted, "n_not_attempted", "c0_route", 0)
    n_unknown = _require_integral(n_unknown, "n_unknown", "c0_route", 0)
    if n_not_attempted + n_unknown > n_rows:
        raise ContractViolation(
            "c0_route: n_not_attempted + n_unknown exceeds n_rows")
    qs = []
    for q in q_effs_attempted:
        q_value = _validate_optional_q_eff(q, "c0_route")
        if q_value is not None:
            qs.append(q_value)
    if not uses_ipv:
        return {"terminal": "NOT_APPLICABLE", "reason_code": "no_ipv_derived_quantity"}
    if n_rows <= 0:
        return {"terminal": "INDETERMINATE_UNKNOWN_PROVENANCE", "reason_code": "no_rows"}
    unknown_share = n_unknown / n_rows
    unavailable_share = (n_not_attempted + n_unknown) / n_rows
    mean_q = _det_mean(qs)
    n_attempted = n_rows - n_not_attempted - n_unknown
    metrics = {"unknown_share": unknown_share, "unavailable_share": unavailable_share,
               "mean_q_eff_attempted": mean_q, "n_rows": n_rows,
               "n_attempted": n_attempted, "n_q_evidence": len(qs)}
    if n_attempted < 0:
        raise ContractViolation("counts inconsistent: attempted < 0")
    # fail-closed：有 ATTEMPTED 行却无任何 q 证据 -> 不得判"未触发"
    if n_attempted > 0 and not qs:
        return {"terminal": "INDETERMINATE_UNKNOWN_PROVENANCE",
                "reason_code": "attempted_rows_without_q_evidence", "metrics": metrics}

    if unknown_share >= cuts["unknown_share"] or not mapping_is_1to1:
        return {"terminal": "INDETERMINATE_UNKNOWN_PROVENANCE",
                "reason_code": ("unknown_share_ge_cut" if mapping_is_1to1
                                else "ledger_mapping_not_1to1"), "metrics": metrics}
    if unavailable_share >= cuts["unavailable_share"] or (
            mean_q is not None and mean_q >= cuts["mean_q_eff"]):
        return {"terminal": "OWNER_REANALYSIS_REQUIRED",
                "reason_code": ("unavailable_share_ge_cut"
                                if unavailable_share >= cuts["unavailable_share"]
                                else "mean_q_eff_ge_cut"), "metrics": metrics}
    return {"terminal": "NO_AUDIT_TRIGGER_DETECTED", "reason_code": "below_all_cuts",
            "metrics": metrics}


def c0_route_with_sensitivity(**kw) -> Dict[str, object]:
    primary = c0_route(**kw, cuts=ROUTING_PRIMARY)
    alts = [c0_route(**kw, cuts=c)["terminal"] for c in ROUTING_SENSITIVITY]
    return {"primary": primary, "sensitivity_terminals": alts,
            "stable": len(set(alts)) == 1}


# ---------------------------------------------------------------- schema 自检

def load_schema(path: str | Path) -> dict:
    d = json.loads(Path(path).read_text())
    if d.get("schema_id") != SCHEMA_VERSION:
        raise ContractViolation(f"schema_id mismatch: {d.get('schema_id')}")
    seen = set()
    ledger_bearing_artifact_ids = []
    non_ledger_artifacts = []
    for a in d["artifacts"]:
        if "artifact_id" not in a:
            raise ContractViolation("artifact missing artifact_id")
        artifact_id = a["artifact_id"]
        if artifact_id in seen:
            raise ContractViolation(f"duplicate artifact_id {artifact_id}")
        seen.add(artifact_id)
        status = a.get("status")
        if status in NON_LEDGER_STATUSES:
            non_ledger_artifacts.append({"artifact_id": artifact_id, "status": status})
            continue
        ledger_bearing_artifact_ids.append(artifact_id)
        for factor in ("expansion_factor", "collapse_factor"):
            if factor not in a:
                raise ContractViolation(f"{artifact_id}: missing {factor}")
            value = a[factor]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ContractViolation(f"{artifact_id}: bad {factor}")
        k_source = a.get("K_source")
        if isinstance(k_source, dict) and k_source.get("kind") == "constant":
            if "value" not in k_source:
                raise ContractViolation(f"{artifact_id}: missing K_source.value")
            _require_integral(k_source["value"], "K_source.value", artifact_id, 1)
    d["ledger_bearing_artifact_ids"] = ledger_bearing_artifact_ids
    d["non_ledger_artifacts"] = non_ledger_artifacts
    return d
