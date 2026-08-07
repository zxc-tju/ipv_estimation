#!/usr/bin/env python3
"""RQ021-E2：同期 envelope 的留一源迁移检验（leave-one-dataset-out）。

执行说明：本轮原派 codex agent 执行，两次均死于其后端网络中断（HTTP 503，
见 board/progress.log 2026-08-07 两条 exit 记录），由监督方按任务书
`board/RQ021_E2_kickoff.md` 直接本地执行。规格与任务书一致：

- 参照池、特征、fold、conformal 流程全部复用 E1 脚本（作为模块导入，零改动）；
- 对每个源 S：条件分位数模型、支持门、conformal 半径都只用非 S 行拟合，
  在 test fold 的 S 行上评估；
- 事前判读标准（先于结果固定）：alpha=90，四个留出源覆盖率全部落在
  [0.87, 0.93] 判「迁移获得支持」，任一带外判「边界维持」；
- 次要对照表：E1 持久化模型（全源拟合）对全部 test 行打分一次，按源分组
  报告覆盖率，用于区分「S 本来就难」与「S 只在未见时难」。
  其全源合计覆盖率必须逐位复现 E1 的 0.902798，否则本表作废。
"""
from __future__ import annotations

import gc
import importlib.util
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
E1_DIR = REPO_ROOT / ".codex-fleet/rq021-contemporaneous-envelope/work/E1"
E1_PKL = E1_DIR / "envelope_model/rq016c_h2_envelope.pkl"
OUT_JSON = HERE / "key_numbers_e2.json"
LOG = HERE / "lodo_run.log"

SOURCES = ["waymo_train", "nuplan_train", "lyft_train_full", "av2_motion_forecasting"]
BAND_90 = (0.87, 0.93)  # 事前判读带，与旧 RQ009 检验同一把 ±3pp 尺
RANDOM_STATE = 20260626  # 与 E1 默认一致
ALPHAS = ["80", "90", "95"]

# E1 全池不变量（对不上立即停）
EXPECTED_TOTAL = 2_442_625
EXPECTED_FOLDS = {"train": 974_984, "calibration": 481_088, "guard_tune": 499_893, "test": 486_660}
E1_TEST_COVERAGE_90 = 0.9027984335526273  # 次要对照表的复现锚


def log(msg: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def import_e1_module():
    path = HERE / "run_rq016c_h2_human_only_envelope.py"
    spec = importlib.util.spec_from_file_location("rq021_e1_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def coverage_by_source(e1, frames_test, artifact) -> dict:
    """次要对照表：全源拟合的 E1 模型对 test 行打分，按源分组算覆盖率。"""
    scored = e1.score_with_artifact(artifact, frames_test)
    y = frames_test[e1.ref.TARGET_COLUMN].to_numpy(dtype=np.float64)
    src = frames_test["source_dataset"].to_numpy()
    gate = scored["mechanism2_gate_ok"].to_numpy(dtype=bool)
    out = {}
    for label in ALPHAS:
        lo = scored[f"lo_{label}"].to_numpy(dtype=np.float64)
        hi = scored[f"hi_{label}"].to_numpy(dtype=np.float64)
        covered = (y >= lo) & (y <= hi) & gate
        per = {}
        for s in SOURCES + ["ALL"]:
            mask = np.ones(len(y), dtype=bool) if s == "ALL" else (src == s)
            g = int((gate & mask).sum())
            c = int((covered & mask).sum())
            per[s] = {
                "test_rows": int(mask.sum()),
                "gate_pass_rows": g,
                "covered_rows": c,
                "coverage": (c / g) if g else None,
                "abstention": 1.0 - g / int(mask.sum()) if mask.sum() else None,
            }
        out[label] = per
    return out


def main() -> int:
    started = time.time()
    log("import E1 pipeline module")
    e1 = import_e1_module()

    log("load K2 ledger + join RQ009 matrix (full pool, identical to E1)")
    ledger, ledger_diag = e1.load_k2_current_ledger()
    joined_folds, join_diag = e1.load_joined_folds(ledger)
    frames = e1.human_frames(joined_folds)
    counts = e1.validate_counts(frames, join_diag)
    del joined_folds, ledger
    gc.collect()

    total = sum(len(f) for f in frames.values())
    assert total == EXPECTED_TOTAL, f"pool total {total} != {EXPECTED_TOTAL}"
    for fold, expected in EXPECTED_FOLDS.items():
        got = len(frames[fold])
        assert got == expected, f"fold {fold} rows {got} != {expected}"
    assert counts["invalid_split_rows"] == 0, "rq007_split outside {development, guard}"
    log(f"pool invariants OK: total {total:,}; invalid_split_rows=0")

    per_source_pool = {
        fold: frames[fold]["source_dataset"].value_counts().to_dict() for fold in frames
    }

    selected_params = e1.ref.load_selected_hgb_params()

    log("secondary table: score all test rows with the persisted all-source E1 model")
    with E1_PKL.open("rb") as fh:
        artifact = pickle.load(fh)
    insample = coverage_by_source(e1, frames["test"], artifact)
    del artifact
    gc.collect()
    got_cov = insample["90"]["ALL"]["coverage"]
    assert abs(got_cov - E1_TEST_COVERAGE_90) < 1e-9, (
        f"all-source 90 coverage {got_cov!r} does not reproduce E1 {E1_TEST_COVERAGE_90!r}"
    )
    log(f"secondary table anchored: ALL@90 coverage {got_cov:.6f} == E1")

    lodo: dict = {}
    for s in SOURCES:
        t0 = time.time()
        log(f"=== holdout {s}: filter fit folds to non-{s}, test fold to {s} ===")
        fit = {
            fold: frames[fold].loc[frames[fold]["source_dataset"] != s].reset_index(drop=True)
            for fold in ("train", "guard_tune", "calibration")
        }
        held_test = frames["test"].loc[frames["test"]["source_dataset"] == s].reset_index(drop=True)
        log(
            f"rows: train {len(fit['train']):,} guard_tune {len(fit['guard_tune']):,} "
            f"calibration {len(fit['calibration']):,} | held-out test {len(held_test):,}"
        )

        gate, gate_payload, _, guard_diag = e1.ref.fit_gate(fit["train"], fit["guard_tune"])
        cal_ok, cal_diag = e1.ref.apply_gate(fit["calibration"], gate)
        test_ok, test_diag = e1.ref.apply_gate(held_test, gate)
        log(f"gate refit without {s}; held-out gate pass {int(test_ok.sum()):,}/{len(held_test):,}")

        model = e1.ref.fit_tier_model(fit["train"], selected_params, RANDOM_STATE)
        q_cal, _ = e1.ref.predict_quantiles(model, fit["calibration"])
        y_cal = fit["calibration"][e1.ref.TARGET_COLUMN].to_numpy(dtype=np.float32, copy=False)
        radii = e1.ref.compute_radii(q_cal, y_cal, cal_ok)
        del q_cal, y_cal
        gc.collect()

        q_test, _ = e1.ref.predict_quantiles(model, held_test)
        metrics = e1.ref.score_test_frame(held_test, q_test, radii, test_ok)
        del model, q_test, gate
        gc.collect()

        lodo[s] = {
            "fit_rows": {k: int(len(v)) for k, v in fit.items()},
            "held_out_test_rows": int(len(held_test)),
            "gate": {"fit": gate_payload, "guard_tune": guard_diag,
                     "calibration": cal_diag, "held_out_test": test_diag},
            "held_out_gate_pass": int(test_ok.sum()),
            "held_out_abstention": float(1.0 - test_ok.mean()),
            "conformal_radii": radii,
            "metrics": metrics,
            "elapsed_s": round(time.time() - t0, 1),
        }
        del fit, held_test, cal_ok, test_ok
        gc.collect()
        cov90 = lodo[s]["metrics"]["90"]["coverage"]
        log(f"holdout {s} done in {lodo[s]['elapsed_s']}s; held-out coverage@90 = {cov90:.6f}")

    verdict_rows = {s: lodo[s]["metrics"]["90"]["coverage"] for s in SOURCES}
    all_in_band = all(BAND_90[0] <= v <= BAND_90[1] for v in verdict_rows.values())
    verdict = "TRANSFER_SUPPORTED" if all_in_band else "BOUNDARY_STANDS"
    log(f"pre-registered verdict at alpha=90 band {BAND_90}: {verdict}  {verdict_rows}")

    payload = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "execution_note": (
            "codex agent 两次死于后端网络中断（HTTP 503），由监督方按任务书本地执行；"
            "见 board/progress.log 与本文件 docstring"
        ),
        "pre_registered_band_90": list(BAND_90),
        "random_state": RANDOM_STATE,
        "pool_counts": counts,
        "per_source_pool_rows": per_source_pool,
        "k2_ledger": ledger_diag,
        "join_health": join_diag,
        "insample_by_source": insample,
        "lodo": lodo,
        "verdict_alpha90": {"per_source_coverage": verdict_rows, "verdict": verdict},
        "elapsed_s": round(time.time() - started, 1),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"wrote {OUT_JSON.relative_to(REPO_ROOT)} in {payload['elapsed_s']}s total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
