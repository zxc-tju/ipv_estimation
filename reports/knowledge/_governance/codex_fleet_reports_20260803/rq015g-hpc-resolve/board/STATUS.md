# STATUS — track G（rq015g-hpc-resolve）

state: DONE
released_by: commander 于 2026-08-03T02:36:44Z 放行（结论早已裁定，此前仅未翻牌）
updated_at: 2026-08-03T02:36:44Z
phase: 三项交付 + commander 03:10Z 三处更正全部完成，等监督方放行结项
leader_pid: 70604
agents: G1 首段 pid 72935（沙箱禁网，仅出准备件）→ G1b pid 76889（执行+分析，已退出）
        → G2 pid 97527（跨节点闸门，已退出）

summary: |
  2,300 冻结锚点已在受管 HPC 冻结环境重解完毕（Slurm 2023332 COMPLETED, fata02, 6 worker）。
  8 条结论 4 存活 / 4 数值微调（幅度均 <2pp），**无一条结论改变**。

  【commander 02:18Z + 02:43Z 要求的三项，全部补齐】
  1. provenance —— 已写明 G1首段 / G1b / G2 / leader 各自产出，七次投递逐行列出原因与修法
     （七次修补全部由 G1b 完成，leader 未改任何计算代码）
  2. 节点画像 —— fata02 = AMD EPYC 9654 96-Core（avx2/avx512f/fma 全有）；
     参照作业 2022476/2022477 均在 intel 分区 cpui158（CoresPerSocket=48）
     ⇒ 微架构确实不同，01:56Z 的闸门被触发
  3. 跨节点探针 —— **未等 intel 队列**。改用 2022477 在 cpui158 的现成产物当参照，
     在 fata02 用完全相同输入/环境/参数重算同一 case（Slurm 2024766 COMPLETED）：
     max|Δ| = 0.0，mean|Δ| = 0.0，**float64 逐位相同 348/348**，CSV 数值 4/4。
     判据是 ≤1e-15 即算同一等价类，**实测是 0**
     ⇒ fata02 与 cpui158 跨节点逐位确定，"HPC 的答案"良定义，2,300 锚点结果直接采信

  【核心科学结果】
  · D1 未被高估而是被低估 0.16pp（fallback 翻转 0 vs 10，单向）
  · 分源最大偏移仅 1.73pp（waymo OK），D1 全线不动 ⇒ Mac 端流行病学未被环境污染
  · σ 地板（总失效并集下确界）Mac 59.48% → HPC 61.52% @ σ=0.02，**+2.04pp，结论变强**
  · 400 退化锚点 400/400 逐位相同，且**全部来自 nuplan、waymo 零行**（限定不得省略）
  · 结构性发现：曲面越平，argmin 越不可复现，而由曲面形状定义的量越可复现（三组独立证据）
  · 硬警告已写入：单锚点 IPV 点估计跨环境不可复现（ipv_log 最大差 1.8375 rad / 全幅 2.356 rad），
    且最不可复现处恰是最想用它的 N 类

  leader 独立复算全部头条数字（含 σ 地板与 6 组单调性），与 agent 报告及 commander 复算三方一致。

residual: |
  2023904（intel 全量冗余复算，预计 08-01T21:13Z）与 2024625（intel 节点画像，预计 08-02T16:43Z）
  仍在 intel 队列（该分区 184/232 节点 offline）。二者为**冗余确认**，
  闸门已由 2024766 以更强形式（逐位相同）关闭。未换第三个分区。

boundaries: HEAD 511b936c 无 commit；agent.py 未动；rq015b/rq015d 零改动；held_out=0；无禁用术语
corrections_applied: |
  按 commander 03:10Z：
  · 更正 1 —— D-3 支撑数字改为并集下确界（Mac 59.48% / HPC 61.52% @ σ=0.02），
    并补上 ALL 全网格 Mac/HPC 并列表（含 σ* 两侧 0.2347412257 / 0.2290908968）
  · 更正 2 —— 微架构一点的归属改为「commander 01:56Z 提出，leader 02:58Z 核查、03:00Z 派 G2 闭合」
  · 更正 3 —— provenance 段已按「谁写的/谁改的」逐项列出（七次修补全部出自 G1b）
  并写入结项表述：fata02(AMD EPYC 9654) 与 cpui158(Intel) 对本计算属同一数值等价类，
  2,300 锚点结果不受分区切换影响；2023904 因 intel 队列不可得未执行。

next: 等监督方核数字放行（leader 不自行转 DONE）
