#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE = ROOT / "reports/knowledge/RQ027_known_truth_ipv_recovery"
DECISION = KNOWLEDGE / "decision.md"

if DECISION.exists():
    raise SystemExit("decision.md already exists; refuse to overwrite")

DECISION.write_text(
    """# RQ027 Decision

Date: 2026-08-28  
Status: `ACCEPTED / CLOSED`  
Final state: `CLOSED_BY_PI_SCOPE_DECISION`  
Execution verdict retained: `PILOT_NO_GO`  
Further recovery research in this RQ: `STOPPED`

## Problem and stage

RQ027 tested whether the frozen online IPV estimator could recover a simulation-controlled IPV value when the trajectory generator did not share its planner, search, cost implementation or likelihood, and whether candidate-weight concentration could select lower-error readings. The bounded `240 interactive + 48 negative-control` pilot executed completely and was independently recomputed.

## PI ruling

For the current Nature Machine Intelligence manuscript, IPV recovery will not be developed as a new research line. The estimator is treated as an established method component introduced and validated in the published T-ITS study:

> Zhao, X., Sun, J. & Wang, M. Measuring sociality in driving interaction. IEEE Transactions on Intelligent Transportation Systems 25, 9224–9237 (2024).

The validation scope used by the manuscript is the controlled, model-consistent VGIM evidence already published there. The present paper focuses on the downstream human-reference monitor.

## Accepted interpretation

1. The RQ027 pilot is retained as a bounded diagnostic of cross-model numerical transportability for one frozen estimator, one independent generator family and one concentration policy.
2. The pilot does not support a general claim that IPV cannot be estimated, and it does not overturn the published VGIM-consistent recovery evidence.
3. The pilot does not reopen or revise the accepted RQ017, RQ018, RQ019, RQ021, RQ024 or RQ025 decisions.
4. Candidate-weight concentration may be used as the frozen operational readability rule of the present monitor; it is not presented as a calibrated guarantee of numerical recovery accuracy.
5. S2 perturbation expansion, the 3,120/14,040-run scale-up, sealed confirmatory testing and within-RQ retuning remain stopped.

## Manuscript directive

- Main text: cite the T-ITS paper in one short sentence and state that the estimator is held fixed as the behavioural reading supplied to the monitor.
- Supplementary Information: summarise the published controlled validation, the present online implementation, interaction-specific diagnostics and episode-summary sensitivity.
- Keep RQ027 figures and negative findings in the research record; they are not part of the paper's frontstage evidence chain.
- Use `readable/readability` and `candidate-discrimination` for the operational gate. Do not frame the current paper as a new universal parameter-recovery study.
- Follow the publication-conference principle: centre the paper on the context-conditioned human-reference monitor, without a defensive excursus on an abandoned comparison dimension.

## Closure

RQ027 is complete and closed. No additional recovery experiment is authorised under this RQ. Reopening requires an explicit new PI instruction and a new research contract.

## Evidence

- Plan: `reports/plans/RQ027_plan_v0_known_truth_ipv_recovery_20260828.md`
- Execution: `reports/studies/RQ027_known_truth_ipv_recovery/RQ027_1_known_truth_recovery_20260828T094332Z_889b49be/00_entry/index.html`
- Report: `reports/studies/RQ027_known_truth_ipv_recovery/RQ027_1_known_truth_recovery_20260828T094332Z_889b49be/01_results/REPORT.md`
- Independent validation: `reports/studies/RQ027_known_truth_ipv_recovery/RQ027_1_known_truth_recovery_20260828T094332Z_889b49be/01_results/independent_validation.md`
- Synthesis: `reports/knowledge/RQ027_known_truth_ipv_recovery/synthesis.md`
""",
    encoding="utf-8",
)

(KNOWLEDGE / "README.md").write_text(
    """# RQ027 Knowledge Layer

Status: `PI-ACCEPTED / CLOSED_BY_PI_SCOPE_DECISION`

Decision: `reports/knowledge/RQ027_known_truth_ipv_recovery/decision.md`

Execution layer: `reports/studies/RQ027_known_truth_ipv_recovery/`

Plan: `reports/plans/RQ027_plan_v0_known_truth_ipv_recovery_20260828.md`

## Research question

Can the frozen online IPV estimator recover a simulation-controlled IPV parameter when the trajectory generator does not share its planner, search, cost implementation or likelihood, and can candidate concentration distinguish lower-error interactive windows from uninformative or mismatched controls?

## Final interpretation

The bounded independent-generator pilot remains a `PILOT_NO_GO` for cross-model numerical recovery and concentration-based accuracy selection. The PI has closed this RQ and chosen not to pursue further recovery experiments for the current manuscript. The paper will cite the previously published T-ITS validation of IPV identification in controlled VGIM interactions and will use the estimator as a fixed input to the downstream human-reference monitor.

RQ027 is preserved as a research boundary. It does not establish universal IPV failure and does not revise accepted downstream monitor decisions.

## Reviews

- `reviews/plan_rationality_review.md`
- `synthesis.md`
""",
    encoding="utf-8",
)

(KNOWLEDGE / "synthesis.md").write_text(
    """# RQ027 Synthesis

Status: `CLOSED / PI ACCEPTED`

## Final ruling

The development-only `240 interactive + 48 negative-control` feasibility pilot completed and independently reproduced `PILOT_NO_GO` for the frozen estimator under the independent-generator contract. The execution record remains unchanged.

The PI has ended this RQ. The current manuscript will not present cross-model known-truth recovery as a new contribution. It will cite the published T-ITS validation of IPV identification in controlled VGIM interactions and treat the estimator as a fixed measurement component of the human-reference monitor.

## Evidence retained in the research record

- `288/288` scheduled runs completed with no engineering failure.
- Persistent opportunity-aware readings existed for `215/240` interactive runs.
- Accepted-run MAE was `0.553907 rad`, compared with `0.553432 rad` for the same-run zero predictor.
- Candidate concentration did not select lower-error frames in this pilot.
- Negative-control persistent concentration acceptance was `35/48`.

## Scope carried forward

- RQ027 is a bounded cross-model transportability diagnostic.
- It does not negate the previously published controlled VGIM validation.
- It does not reopen accepted RQ017+ decisions.
- No S2, sealed expansion or retuned rerun is authorised.
- Paper-facing treatment is concise: one main-text citation and a supplementary method note, with no defensive detour.
""",
    encoding="utf-8",
)

studies = (ROOT / "STUDIES.md").read_text(encoding="utf-8")
pattern = re.compile(r"^\| RQ027 \|.*$", re.MULTILINE)
replacement = (
    "| RQ027 | Known-truth IPV recovery and abstention validation under independent simulation | "
    "`CLOSED / PI_SCOPE_DECISION`; bounded pilot retained as cross-model diagnostic; no further recovery work for the current manuscript | "
    "`reports/studies/RQ027_known_truth_ipv_recovery/RQ027_1_known_truth_recovery_20260828T094332Z_889b49be/00_entry/index.html` | "
    "`reports/knowledge/RQ027_known_truth_ipv_recovery/decision.md` | "
    "plan `reports/plans/RQ027_plan_v0_known_truth_ipv_recovery_20260828.md`; published T-ITS controlled VGIM validation is the manuscript basis; no S2/sealed/retuning |"
)
studies, count = pattern.subn(replacement, studies)
if count != 1:
    raise SystemExit(f"Expected one RQ027 STUDIES row, replaced {count}")
(ROOT / "STUDIES.md").write_text(studies, encoding="utf-8")

start_here_path = ROOT / "START_HERE.md"
start_here = start_here_path.read_text(encoding="utf-8")
pattern = re.compile(r"^- \*\*【2026-08-28 RQ027 bounded feasibility pilot.*?】\*\*$", re.MULTILINE)
replacement = (
    "- **【2026-08-28 RQ027 已由 PI 最终裁定为 `CLOSED_BY_PI_SCOPE_DECISION`：独立生成器 pilot 的 `PILOT_NO_GO` 作为跨模型数值迁移边界保留；当前 NMI 手稿不再推进新的 recovery 研究，改为引用已发表 T-ITS 工作中受控 VGIM 条件下的 IPV identification，并将 estimator 作为冻结方法组件。禁止在 RQ027 内继续 S2、sealed 扩展或调门重跑。正式裁决：`reports/knowledge/RQ027_known_truth_ipv_recovery/decision.md`。】**"
)
start_here, count = pattern.subn(replacement, start_here)
if count != 1:
    raise SystemExit(f"Expected one RQ027 START_HERE bullet, replaced {count}")
start_here_path.write_text(start_here, encoding="utf-8")

log_path = ROOT / "main_workflow.log"
with log_path.open("a", encoding="utf-8") as handle:
    handle.write(
        "\n[2026-08-28 RQ027 PI closure]\n"
        "背景：PI 决定不再为当前 NMI 手稿推进新的 IPV recovery 研究，改用已发表 T-ITS 受控 VGIM 验证作为 estimator 的方法依据。\n"
        "结果：新增 RQ027 decision.md；知识层、STUDIES 与 START_HERE 已同步为 CLOSED_BY_PI_SCOPE_DECISION。RQ027 pilot 继续作为跨模型数值迁移的内部边界保存；S2、sealed 扩展与调门重跑终止。\n"
        "论文指令：正文一句引用，细节进入 Supplementary Information；不展开防御性论述。\n"
    )

# Remove the one-shot updater and its workflow from the final branch state.
for relative in (
    ".github/workflows/apply_rq027_pi_closure.yml",
    "scripts/apply_rq027_pi_closure.py",
):
    path = ROOT / relative
    if path.exists():
        path.unlink()

print("RQ027 closure files updated successfully")
