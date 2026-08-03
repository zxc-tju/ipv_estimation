# E1 RQ009 Dependency Sensitivity

Generated: 2026-07-31T16:51:34Z

## 1. 判定

**量级变了，性质没变。**

RQ009-KC-R3 的**组成性质在受限子集 C 上仍然成立**
（`coverage_within_3pp = True`、`directional_tails_ok = True`、M2 仍显著窄于 M0）；
但**两个幅度数字站不住**：M2 相对 M0 的收窄从 **−42.41% 降到 −29.65%**，
Winkler 从 **−35.47% 降到 −29.51%**。
该差值 **+12.76 pp 是下界**，因为可用性口径是 **case 级**，受限集 C 内部仍含
**33.27%** 的近均匀权重行（`q_eff ≥ 0.93`）—— 可用子集只是**稀释**了低判别力的行，
并没有把它们清干净；行级过滤只会让 C 与 B 的差距更大，不会更小。

同时移动的还有：M2 coverage 0.898949 → **0.882762**（Δ −1.62 pp，仍在名义 ±3pp 内），
abstention 4.94% → **6.29%**（Δ +1.34 pp）。

**是否重估 RQ009 属 PI 决策，本报告只给证据与边界，不给建议。**

## 2. Numbers Table

A is quoted only and annotated as not held-out-free.

| quantity | A quoted | B | C | D | Δ(C-B) | rel. Δ(C-B)/|B| |
|---|---:|---:|---:|---:|---:|---:|
| M2 vs M0 mean_width | -42.27% | -42.41% | -29.65% | -59.05% | +12.76 pp | 30.08% |
| M2 vs M0 winkler | -35.61% | -35.47% | -29.51% | -44.71% | +5.96 pp | 16.80% |
| M2 coverage | 0.898889 | 0.898949 | 0.882762 | 0.920065 | -1.62 pp | -1.80% |
| abstention | 4.78% | 4.94% | 6.29% | 3.14% | +1.34 pp | 27.11% |

## 3. What The Restriction Selected

- Case counts: B=5306 (expected 5306), C=2825 (expected 2825), D=2481 (expected 2481).
- 0.90 prediction rows: B=888892, C=510368, D=378524.
- q_eff >= 0.93 marks near-uniform candidate weights; that IPV value carries no discriminative information between candidates.

| set | cases | ledger rows | q_eff valid | q_eff median | q_eff mean | share q_eff >= 0.93 | k_eff median |
|---|---:|---:|---:|---:|---:|---:|---:|
| C | 2825 | 1020736 | 1020736 | 0.6968 | 0.6458 | 33.27% | 4.88 |
| D | 2481 | 757048 | 757048 | 0.9943 | 0.8166 | 62.77% | 6.96 |

## 4. Secondary Variant

Prediction perspective vocab: ['key_agent_1', 'key_agent_2']; usable vocab: ['key_agent_1', 'key_agent_2']; mapping used: key_agent_1->key_agent_1, key_agent_2->key_agent_2 (identity).

| quantity | B | C_case_perspective | D_case_perspective | Δ(C_case_perspective-B) |
|---|---:|---:|---:|---:|
| M2 vs M0 mean_width | -42.41% | -27.31% | -54.58% | +15.10 pp |
| M2 vs M0 winkler | -35.47% | -30.26% | -40.56% | +5.21 pp |
| M2 coverage | 0.898949 | 0.877875 | 0.915936 | -2.11 pp |
| abstention | 4.94% | 6.71% | 3.47% | +1.77 pp |

## 5. Internal Ablation

M3-M2 paired 90% Winkler on C: mean difference -0.00141265, pairs n=478290. Case-cluster bootstrap p-value skipped. This is an internal ablation, not a manuscript claim.

## 6. Verification

- Ledger split counts: {'development': 6459684, 'guard': 2535052}; development + guard equals total ledger rows: True; held_out rows in ledger: 0.
- Join identities by tier:
- M0: 2666676 + 1145022 = 3811698 (True)
- M2: 2666676 + 1145022 = 3811698 (True)
- M3: 2666676 + 1145022 = 3811698 (True)
- ipv_removed: 2666676 + 1145022 = 3811698 (True)
- held_out_rows_entering_any_statistic = 0. Mechanism: the dev+guard case set was built first from the RQ007 split ledger, then prediction rows were materialized through `pyarrow.dataset` with `field('case_key').isin(...)`; no `y`, `lo_cal`, `hi_cal`, or `width` value from an unmatched row entered a statistic. This does not claim unmatched parquet bytes were untouched by the reader.
- C subset B verified by set operation: True.
- `base_metric_row` imported from the frozen `evaluate.py` via `importlib.util.spec_from_file_location`; coverage / width / Winkler / pinball / tails / abstention were not reimplemented here.
- Numeric parity against published set A is not performable under H1; code-level reuse is the parity argument.
- Abstention identical across tiers within each set: True.
- Health: nonfinite metric entries=0, empty groups=0, tier row-count differences=0, overall pass=True.
- Expected case-count checks: {'B': {'observed': 5306, 'expected': 5306, 'matches': True}, 'C': {'observed': 2825, 'expected': 2825, 'matches': True}, 'D': {'observed': 2481, 'expected': 2481, 'matches': True}}.

## 7. Limits

- This is a re-aggregation of frozen predictions, not a refit, so it cannot speak to how the model would have been calibrated on the restricted set.
- Set A is quoted for context and is not reproduced here.
- The restriction is defined by RQ015A's single primary usable policy: `primary: q_n>=30; attempted_share>=0.80; median_q_eff<=0.75; share(q_eff<=0.75)>=0.60`.
- Results are descriptive sensitivity evidence only; no recommendation about re-analysis is made here.

---

# 附录 A — leader 验收附注（2026-07-31T16:54:47Z，track E leader 撰写）

E1 的数字已由 leader **独立复算**通过：不复用 E1 的脚本、不 import `base_metric_row`，
自行写 coverage / mean_width / Winkler 公式，从同样的冻结 parquet 重算 B/C/D。
结果逐位一致（B: dW −0.42409, dWink −0.35470, cov 0.898949, abst 0.049447；
C: −0.29652 / −0.29511 / 0.882762 / 0.062853；D: −0.59050 / −0.44710 / 0.920065 / 0.031372）。

以下三点是验收中补充的事实，E1 报告正文未涵盖，但对判读很关键：

## A.1 判定应当写成"量级变了、性质没变"，而不是笼统的"结论变了"

按 `restricted_metrics.csv` 的门限列，在受限集 C 上：

| RQ009-KC-R3 的组成性质 | 在 C 上是否保持 |
|---|---|
| M2 覆盖率接近名义（±3pp） | **保持**。`coverage_within_3pp = True`（0.882762 vs 0.90，差 −1.72pp） |
| directional tails 通过 | **保持**。`directional_tails_ok = True`（M2 在 B/C/D 与 perspective 变体上**全部**通过） |
| M2 显著窄于 M0 | **保持**，但幅度收窄：−29.65%（B 上为 −42.41%） |
| 幅度本身（−42.3% / −35.6%） | **变了**：宽度 +12.76 pp，Winkler +5.96 pp |
| abstention 4.78% | **变了**：4.94%（B）→ 6.29%（C），+1.34 pp |

即：受限后**站不住的是那两个幅度数字**，"sharp 且 near-nominal 且尾部合规"这一性质本身在
C 上仍然成立。这个区分直接决定 PI 面对的是"改数字"还是"改主张"。

## A.2 差异全部来自 M2 一侧，M0 是常数——这是定位"变在哪"的关键锚点

leader 复算发现 **`mean_width(M0)` 在 B / C / D / perspective 变体上完全相同 = 1.748666**
（M0 是全局 floor，宽度不随子集变化）。因此宽度比值的全部变化都来自 M2：

```
mean_width(M2):   B 1.007080  →  C 1.230155  →  D 0.716080
```

在补集 D（`share(q_eff≥0.93)` = 62.77%，`k_eff` 中位数 6.96/7）上，M2 给出的区间**更窄**
（−59.05%），在受限集 C 上**更宽**（−29.65%）。换言之：被 RQ015A 可用性口径**排除**的那部分行，
恰恰是 M2 收窄幅度看起来最大的那部分。方向与"结论建立在无判别信息的行上"这一担忧一致，
但本轨只报告这一共现关系，**不作因果归因**。

## A.3 三条应当随证据一起交给 PI 的边界

1. **C 并非不含近均匀权重的行 ⇒ +12.76 pp 是下界，不是点估计。**
   可用性口径是 case 级（`median_q_eff≤0.75` 且 `share(q_eff≤0.75)≥0.60`），
   C 内部仍有 **33.27%** 的台账行 `q_eff ≥ 0.93`。
   C 是"近均匀权重占比更低的子集"，不是"无近均匀权重的子集"——低判别力的行被**稀释**了，
   没有被清除。因此若改用**行级**过滤，C 与 B 的差距只会更大，不会更小：
   报告中的 +12.76 pp / +5.96 pp 应当读作**下界**。（已同步写入 §1 判定段。）
2. **成分变化未被拆解。** C 与 D 的差异不只是 `q_eff` 一个维度；可用性口径同时改变了
   场景几何、来源数据集与角色的构成。本轨没有做成分匹配，因此不能把 C↔B 的差值
   全部归到判别信息这一个因素上。
3. **参照系 A 不可复现，这是硬约束的直接后果。** 冻结的 fold=test 含 2,270/7,576（29.96%）
   个 RQ007 held_out case。本轨用 B 作参照系。附带事实：B 与已发表的 A 高度接近
   （A: cov 0.898889 / dW −42.27% / dWink −35.61%；B: 0.898949 / −42.41% / −35.47%），
   说明用 B 替代 A 并未引入可见偏移——但这只是事后观察，不构成对 A 的复现。

**本轨不就"是否重估 RQ009"给出任何建议。以上均为描述性敏感性证据。**

## A.4 具名候选下一轮（本轮**不做**，PI 想要时一句话即可触发）

**候选 E2 —— 按 `q_eff` 连续分层的 width / coverage 剖面。**

- **做什么**：不再用 case 级二分（可用/不可用），而是把 fold=test ∩ dev+guard 的
  预测行按台账 `q_eff` 连续分箱（例如十分位），在每一箱上用同一套
  `base_metric_row` 报 `mean_width(M2)`、`coverage(M2)`、`winkler(M2)`、`abstention`，
  以及 M2/M0 的收窄幅度。
- **能压掉什么**：
  （a）把 A.3-1 的"下界"从定性论证变成可读的单调剖面——直接看出收窄幅度随 `q_eff` 怎么走；
  （b）部分压掉 A.3-2 的成分混淆——若在**每一个** `q_eff` 分箱内部按几何/来源/角色做分层或匹配，
  就能把"判别信息"与"场景构成"两个维度分开看。
- **成本**：与本轮同量级（同样是冻结预测的过滤+重聚合，不重跑管线），一个 agent。
- **为何本轮不做**：本轨的问题是"RQ009 的结论依不依赖低判别力的行"，A/B/C/D 已经回答，
  且结论按保守口径表述（只报共现、不作因果）。按项目的速度原则，此处停手。

