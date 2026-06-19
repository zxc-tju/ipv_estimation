# 在线 IPV verifier — 简洁叙事 + 三部件 Methods 修订备忘

> **本地过程文档（LOCAL ONLY）。按 CLAUDE.md Overleaf 规则，不要推送到 Overleaf**
> （只有 `main.tex` / `bibliography/biblio.bib` / `structure.md` / 正文 `\includegraphics`
> 的图可推）。本备忘服务于 Methods 改写与 response letter，属研究过程材料。
>
> 日期：2026-06-19 · 依据：`reports/.../02_reports/3-online_ipv_interval-2`、
> 两份 `4-self_anchor_validation*`（均判 MUST REVISE）、当前 `paper/main.tex`。

---

## 0. 一页纸结论（TL;DR）

- **核心模型不换**：车道参照因果 rolling-IPV 自锚 + CQR + split-conformal。数据全力支持，
  且故意用"无聊的标准工具"——这是 NMI 的加分项，不是短板。
- **要补的只有"有效性护栏"**：情境下限（situation floor）+ 中等偏移弃权
  （moderate-Δ abstention）。它们不是第二个模型，是**同一个想法的有效边界**。
- **叙事保持一句话**："在线判断社会合规，不该估风险，而该锚定驾驶人自己早窗的偏好，
  再用 conformal 校准。"护栏只作为这句话的支撑域声明出现。
- **三处 `main.tex` 实质改动**：Algorithm 2 加 floor + abstain 两行；norm-laundering
  从 "planned validation" 改写为 "已测、需护栏"；E1 标签重叠在做无泄漏强声明前修复。

---

## 1. 简洁叙事框架（the spine）

### 一句话论点（paste-ready, EN）
> *The right online handle on social compliance is not the situation's risk but the
> driver's own early-window preference; a self-anchored, conformally calibrated interval
> is sharp, nominally covered and transfers across sources, and a minimal situational
> guard keeps it from laundering a persistently non-compliant agent.*

### 叙事弧（五拍，与 structure.md 对齐）
1. **重构（reframe）**：社会合规 = 在线运行时验证一个数据驱动的经验规范（IPV）。
2. **反直觉核心**：预测风险几乎不收窄合理区间（oracle PET 仅 −3%）；驾驶人自己的
   早窗因果偏好才收窄（−42%）。
3. **一个机制**：自锚 + conformal → 锐、达名义覆盖、唯一跨源迁移（Waymo 留出 0.902）。
4. **有效性包络**（护栏在此登场，而非新模型）：自锚只在其支撑域内是有效的人群条件规范；
   超出支撑或落入高风险段，退回人群情境规范。
5. **可执行**：deviation → planner 的 soft cost / warning / fallback；paired 联合检查作 gate。

### 为什么这是"NMI 形状"的
NMI 奖励的是**思路清晰 + 洞见 + 严谨**，不是模型复杂度，也不是模型极简本身。本文恰好是
"简单想法 + 出人意料结果 + 标准工具 + 诚实边界"。复杂度只应花在**诚实地划定边界**上
（floor / abstain / E1 修复），而不是花在 estimator 上。

---

## 2. 三部件 Methods（one signal · one calibration · one guard）

把 Methods 重组为三个部件，每个都由一条结果逼出来，没有一个为复杂而复杂。

### 部件 1 — 信号（the signal）：因果自锚
- **是什么**：`RealtimeIPVEstimator`(history_window≈10, `parallel_accurate`) 在前缀轨迹 +
  **静态地图车道中心线参照**上，算前 1–2s 的因果 rolling-IPV；聚合 mean/last/slope@1s,2s。
- **为何在这**：它是唯一跨源迁移的信号；且严格前缀重建对离线 IPV **corr 0.993 / MAE 0.027**
  （观测前缀参照只有 0.281）→ 信号来自前缀+静态地图，非未来泄漏，route-conditioned 可部署。
- **关键数**：相对 oracle-PET 包络宽度 **−42%**（0.485 vs 0.833），覆盖近名义。

### 部件 2 — 校准（the calibration）：split-conformal
- **是什么**：条件分位数模型（HistGBM/QRF-quantile）出 [q05,q50,q95]，再 split/CQR
  conformal 校准到 90%。特征 = 自锚聚合 + 几何 + 路权角色 + AV/HV + 早窗运动学，**不含 PET**。
- **为何在这**：原始分位数欠覆盖（~0.86），conformal 提供 distribution-free、有限样本覆盖保证
  ——对一个"verifier"是刚需（输出要可审计、可信）。
- **关键数**：A/B 同分布 0.901 / 0.485 / false-flag 0.051（vs PET-bin 0.900 / 0.833 / 0.051）。

### 部件 3 — 护栏（the guard）：情境下限 + 弃权 ＝ 有效性包络
> 这是相对当前 `main.tex` 唯一的实质新增；表述上务必归位成"有效性边界"，不是"第二个模型"。

- **情境下限（situation floor）**：自锚区间的**宽松侧下界 q05** 不得比"仅情境"人群规范
  `s05`（几何 + 角色 + **在线运动学 risk proxy**，不用观测 PET）更宽松。即
  `q05 ← max(q05, s05 − τ_flr)`。这堵住 **E5**：高风险 (PET≤1) 段自锚 flag lift 0.850 <
  情境法 1.129，"自锚放行但情境报警"子集对坏结果富集 **1.507×**——一个自洽的激进司机不能
  自定义宽松规范。
- **中等偏移弃权（moderate-Δ abstention）**：当 `|q50_self − s50_sit| > τ_abs` 或 out-of-support
  时，退回情境区间 / Monitor。按 **E3** 的残留洗白窗口 Δ≈0.4–0.6 标定（不是只抓极端）。
- **为何不能丢自锚（E4）**：情境法单独对早窗 IPV 的 R² 仅 0.044，自身倾向残差增量 R² 0.45，
  M2/M1 宽度比 1.34 → 自锚有真实贡献，只能**加护栏**，不能替换。
- **车道 fallback**（已有）：~26% 无车道 → 情境法 / no-roll 运动学 CQR（仍优于 oracle PET）。

**复杂度**：floor 是一次 max，abstain 是一次比较，situational 规范是一次查表/估计 →
在线仍 **O(1)**，每个规划周期内可跑、可审计。

---

## 3. Algorithm 1 / 2 伪代码补丁（drop-in 到 main.tex）

> 与现有 `algorithmic` 风格一致；**新增行已用注释标注**。把它们替换 main.tex 中
> Algorithm 1（offline 校准）与 Algorithm 2（online step）即可。

### Algorithm 1 — 追加 3 行（校准 situational 规范 + 两个护栏半径 + E1 修复）
```latex
% --- add inside Algorithm 1, after fitting \hat{Q} and the conformal radius \kappa ---
\STATE define the scoring target on a post-early, non-overlapping window ($t>$ anchor end)
       \COMMENT{removes the E1 early/full label overlap before no-leakage claims}
\STATE fit situational norm $\hat{S}_{[s_{05},s_{50},s_{95}]}$ on context-only features
       (geometry, role, \emph{online} kinematic risk proxy; no observed PET)
\STATE calibrate floor slack $\tau_{\mathrm{flr}}$ and abstain radius $\tau_{\mathrm{abs}}$ on
       $\mathcal{D}_k$ to (i) hold coverage $\ge 1-\alpha$, (ii) match the situation-only
       high-risk bad-outcome flag-lift, (iii) fire across moderate-$\Delta$ shifts
\RETURN $\hat{Q},\,\kappa,\,\hat{S},\,\tau_{\mathrm{flr}},\,\tau_{\mathrm{abs}},\,\text{source-guard}$
```

### Algorithm 2 — guarded self-anchor（整段替换）
```latex
\begin{algorithm}[t]
\caption{Online social-compliance verification step at time $t$ (guarded self-anchor)}
\label{alg:online}
\begin{algorithmic}[1]
\REQUIRE prefix $W_t$; map-lane ref; calibrated $\hat{Q},\kappa$; situational norm $\hat{S}$;
         guard radii $\tau_{\mathrm{flr}},\tau_{\mathrm{abs}}$; source health
\ENSURE guarded interval, deviation score, joint flag, planner signal
\IF{lane/route reference available}
    \STATE $\theta^{\mathrm{self}}_i\leftarrow$ causal rolling-IPV from $W_t$ + map-lane \COMMENT{strict-online}
\ELSE
    \STATE fall back to self-anchor-free causal-kinematics features \COMMENT{$\approx$26\% of cases}
\ENDIF
\STATE feats $\leftarrow(\theta^{\mathrm{self}}_i,\text{geometry},\text{role},\text{AV/HV},\text{kinematics},\theta_j^{\mathrm{lag}})$ \COMMENT{PET excluded}
\STATE mode $\leftarrow$ source-guard \textbf{if} source unknown/drift \textbf{else} CQR
\STATE $[q_{05},q_{50},q_{95}]\leftarrow\hat{Q}(\text{feats})$ widened by $\kappa(\text{mode})$ \COMMENT{self-anchor CQR interval}
\STATE $[s_{05},s_{50},s_{95}]\leftarrow\hat{S}(\text{geometry},\text{role},\text{online risk proxy})$ \COMMENT{situational norm; no observed PET}
\STATE $q_{05}\leftarrow\max\!\big(q_{05},\,s_{05}-\tau_{\mathrm{flr}}\big)$ \COMMENT{\textbf{situation floor}: lenient edge cannot dip below the population norm where prosociality is expected}
\IF{$|q_{50}-s_{50}|>\tau_{\mathrm{abs}}$ \textbf{ or not } \textsc{InSupport}(feats)}
    \RETURN \textsc{Abstain}/\textsc{Monitor} reverting to $[s_{05},s_{50},s_{95}]$ \COMMENT{\textbf{moderate-$\Delta$ abstention}}
\ENDIF
\STATE $\delta\leftarrow$ signed nonconformity of current $\theta_i$ vs the guarded interval
\STATE joint $\leftarrow$ (pair-sum and pair-diff of $\theta_i,\theta_j$ within human bands)
\IF{\textbf{not} \textsc{QualityGates}(support, provenance, source health)}
    \RETURN \textsc{Monitor}($\delta$, joint)
\ENDIF
\STATE signal $\leftarrow$ \textsc{Warning}+soft cost \textbf{if} $\delta$ exceeds threshold \textbf{else} \textsc{Monitor}
\RETURN (guarded interval, $\delta$, joint, signal)
\end{algorithmic}
\end{algorithm}
```

**注**：`\textsc{InSupport}` 与 `\textsc{QualityGates}` 可合并；这里分开写是为了让 E3 的
"中等偏移弃权"和原有的支撑/源健康门各自可读。floor 写成单侧（只钳宽松侧下界），与正文
"one-sided soft cost active only where the norm expects prosociality" 一致——只是从 soft cost
升级成区间上的硬钳。

---

## 4. 「为什么模型刻意保持标准/简单」辩护话术

### 4a. Methods 版（paste-ready, EN — 放在 Reasonable-interval estimation 段末）
> We deliberately keep the verifier's machinery standard and parsimonious. The reasonable
> interval is produced by conditional quantile regression calibrated with split-conformal
> prediction—both well-understood, distribution-free tools—rather than a bespoke deep
> architecture. This is a design choice, not a limitation. Our contribution is the
> identification of the informative online signal (the driver's own early-window causal
> preference) and its calibration into a coverage-guaranteed interval, not a new estimator;
> off-the-shelf components isolate the contribution to the signal and ensure that the
> observed sharpness, nominal coverage and cross-source transfer are attributable to *what
> the verifier reads* rather than to model capacity. Standard tools also keep each online
> step $O(1)$, auditable and certifiable—properties a runtime verifier requires. Where
> additional structure is genuinely needed—the situational floor and out-of-support
> abstention—it is the minimal one-sided guard that the validation experiments show to be
> necessary to prevent norm-laundering, and it adds no online cost.

### 4b. Response-letter 版（paste-ready, EN — 应对 "why not a more sophisticated/deep model?"）
> We thank the reviewer. We chose deliberately standard components (quantile regression
> with split-conformal calibration) for three reasons. **First**, our claim concerns *what
> an online verifier should read*, not a new estimator; using off-the-shelf tools isolates
> the contribution to the signal and guards against the result being an artifact of model
> capacity. **Second**, conformal calibration provides distribution-free, finite-sample
> coverage guarantees that a black-box predictor would not—essential for a runtime
> *verifier* whose outputs must be auditable and trustworthy, and used as a soft cost,
> warning or fallback trigger. **Third**, the verifier must run within a planning cycle
> ($O(1)$) and be inspectable for certification. We did evaluate a richer route: an
> independent QRF-leaf + conformal pipeline reproduces the headline (coverage 0.904, width
> 0.478 versus 0.485), confirming the result is method-agnostic and that added model
> complexity buys nothing here. The only structure we add—the situational floor and
> out-of-support abstention—is the minimal guard our validation (E1–E5) showed to be
> necessary, not gratuitous sophistication.

> 事实锚（均已对回源文件）：QRF 复现 0.904 / 0.478 见 `3-online_ipv_interval-2/01_results/
> final_summary.md` §4；自锚 A/B 0.901 / 0.485 见 `01_results/ab_metrics.csv`。

---

## 5. 落地改动清单（edits to main.tex）

1. **Algorithm 1 / 2**：按 §3 替换/追加（floor + abstain + situational 规范 + E1 后窗目标）。
2. **Fig.1 架构图**：在 self-anchor 路径后加一个 "situational floor / abstain" 分支盒，
   标注 "validity envelope, not a second model"。
3. **Results R3 / Discussion**：把 norm-laundering 从 "planned validation" 改为
   "已由内部 E1–E5 测试；自锚单独不足，混合护栏支撑有效性"；保留 NSFC 作**外部**验证。
4. **Methods Reasonable-interval 段**：贴入 §4a 辩护段；补 situational floor 的定义与
   τ_flr/τ_abs 校准目标（对齐 E5 high-risk flag-lift、E3 Δ0.4–0.6）。
5. **Methods Leakage contract**：声明 situational 规范只用**在线运动学 risk proxy**
   （非观测 PET），守住"PET 不入在线路径"。
6. **E1 修复**：在任何"strict no-leakage"措辞前，确保评估目标在 t>2s 非重叠后窗上重算；
   现有 corr 0.993 只解决*重建*因果性，未解决*标签重叠*。

### 一句话给审稿人的"简"的定位
> 模型让人觉得理所当然；复杂度只花在诚实地划定有效边界上；思路一句话讲完。
