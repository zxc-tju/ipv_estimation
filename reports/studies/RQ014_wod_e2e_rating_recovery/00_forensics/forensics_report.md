# RQ014 Phase F — G1 取证最终报告

日期：2026-07-10。执行：Claude（PI）+ 用户手动运行只读扫描脚本（沙盒无 codex 认证/HPC 密钥）。
原始输出：本目录 `mac_forensics_raw / mac_forensics_pass2 / hpc_forensics_raw / hpc_forensics_pass2`；
pass3（Mac 因 OneDrive 云占位 find 挂起中止；HPC 因 ssh 超时未跑）脚本保留于
`reports/plans/prompts/RQ014_forensics_*_pass3*.sh`，可随时补跑。

## VERDICT: NOT_FOUND（已扫面全部阴性）

在所有已扫描面上，**不存在 RQ010B（2026-06-22）之前的任何 WOD-E2E 评分×IPV 偏离实验残留**。
"旧阳性实验"的配置无法从现有数字痕迹中恢复。

## 已扫面（全部阴性）

| 面 | 结果 |
|---|---|
| 本仓库：跟踪文件、git 已删除文件史、archived/、data/derived、workflow log 全期 | WOD 痕迹最早 2026-06-22（RQ010 立项）；更早无 |
| PAPER001 旧稿（≤2026-06-19 的 main.tex/structure/evidence_index） | 零 WOD-E2E；所有 "Waymo" 均为 InterHub-WOMD 语料 |
| HPC sacct（保留窗仅约 8 天） | wod/e2e/rating/pref 命中全为 RQ010B 作业 |
| HPC bash_history（502 行） | waymo/ipv/wod 零命中 |
| HPC 家目录全景 + 非 ZXC 目录 | 其余为他人项目（XN/CJE/JXY/HBY/KRX/ZZ）；/ZXC 2026-06 才建 |
| HPC 全盘 rating/preference 命名文件 | 全为 RQ010B 2026-06-29 之后 |
| Mac codex 会话（2026-03 至今，严格正则+片段） | 5–7 月命中 = Obsidian 论文笔记（WOD-E2E 数据集论文及引用它的 VLA 论文）+ RQ010B 自身的 worker prompts + 今日 RQ014 会话 |
| Mac codex history.jsonl | 零命中 |
| Mac Claude Code 转录（主仓库项目 + 旧 interhub_traj_lane 项目） | 主项目全为 6 月后；旧项目命中为 InterHub 包络工作，非 WOD |

## 未扫/未完成面（按残余价值排序）

1. **HPC 6-29 pilot 修复链 4 个结果目录的中间相关数**（`rq010b_wod_e2e_ipv_rating_pilot_{原始,dtfix,fixed,routefix}`）——脚本已备（hpc_pass3），ssh 超时未跑。直接检验假说 H-A。
2. `1_Codes/archived/` 兄弟目录 + 论文仓库（9_overleaf）git 全历史 pickaxe——脚本已备（mac_pass3b）。
3. Obsidian `WODE2E.md` 笔记正文。
4. Cowork 会话库全文（pass2 中断，仅有宽正则候选清单）。
5. 旧 Windows 机（`C:\Users\xiaocongzhao\OneDrive\...`）、OneDrive 网页版历史/回收站——仅能人工。

## 发现的相关片段（非目标本体，但构成记忆来源假说）

- 2026-06-18 旧手稿规划文档（`NMI_论文撰写思路` 快照）：R3 计划线 "★外部验证：在 NSFC
  获奖算法上，社会偏离 ↔ 独立排名/评分/安全事件 → Fig 5，新核心结果"。6-19 审计将其降级：
  `trajectory_friction_z` 仅为探索性运动学代理，NSFC verifier-validity 判 null。
- 2026-06-29 pilot 链存在 4 个相隔数分钟的结果目录；仓库文档只记录了最终 routefix 的
  rho=0.123（null），**中间（含带 dt 失配、candidate 自参考 bug 的）版本的相关数从未入档**。

## 记忆来源假说（排序）

- **H-A（可检验，优先）**：6-29 pilot 链某中间/含 bug 输出曾短暂显示"评分高↔偏离低"的强相关，
  修复后归零。若成立 → 旧结果 = ARTIFACT，RQ010B null 站得住。
- **H-B（可检验）**：NSFC/OnSite 的"偏离↔评分/排名"探索性信号在记忆中移植到了 WOD-E2E。
- **H-C（不可排除）**：真实的更早独立研究存在于未扫面（旧 Windows 机 / OneDrive 已删目录 /
  archived 兄弟目录）。与"数据已确认缺失"的表述最相容，但无任何数字痕迹支持。

## G1 决定建议

1. 取证判定 **NOT_FOUND**：Phase R 注册表按全宽配置空间冻结，不做基于"已恢复配置"的收窄。
2. 缺陷格（dt 失配、candidate 自参考、CV 外推 counterpart）保留在注册表内——它们同时充当
   H-A 的正式检验单元。
3. 记忆证据仅"方向+显著"，且全部已扫面阴性 ⇒ **T1 确认关卡（封存半集 + 全网格 max-stat）不得放宽**。
4. 剩余两个脚本（hpc_pass3 / mac_pass3b）不阻塞 Phase R，随时可补跑；结果只会
   改进 artifact 判别，不改变搜索空间。
