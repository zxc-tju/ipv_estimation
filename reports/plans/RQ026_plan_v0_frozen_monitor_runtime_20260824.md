# RQ026 Plan v0（Approved / Executing）
状态：`APPROVED / EXECUTING` ｜ user_approval_date=`2026-08-24` ｜ `frozen_monitor_runtime=authorized` ｜ `managed_hpc_submit=authorized` ｜ `protected_data=NONE`

本文件定义 RQ026 的 frozen-monitor runtime evidence 合同：先做 WP4 本地 no-refit reproduction 与 full-sign local baseline，再做受管 Tongji HPC exact/fast preflight、pilot 与 full runtime evidence。只允许围绕已冻结 monitor/runtime 口径取证；不做 refit、不做 paper/decision/data 改写，不跨越 RQ024 的 accuracy boundary。

## 1. Context

RQ026 要解决的问题不是“模型是否更准”，而是：在 frozen monitor 配置下，端到端 runtime 路径能否在本地与受管 Tongji HPC 上给出可复现、可持久化、接口分离良好的运行证据。

当前已知前置事实：

- 用户已于 `2026-08-24` 明确授权 formal frozen-monitor runtime evidence。
- 用户已于 `2026-08-24` 明确授权 managed Tongji HPC submit。
- RQ024 仍是 `ACCEPTED / MIXED_DIAGNOSTIC / Tier2 blocked`，因此 RQ026 不得把 runtime evidence 写成 accuracy 修复或性能优越证明。

## 2. Research Question

在不 refit、不断言 accuracy improvement、且保持接口分离的前提下，冻结 monitor/runtime 配置在本地与 Tongji HPC 上能否给出：

- full `67,861` sign local baseline；
- exact/fast preflight；
- managed HPC pilot；
- managed HPC full；
- 以及 P50/P95/P99、deadline、failure、persistence、key conservation 的正式证据包。

## 3. Deterministic Unit

- 本地基线单位：frozen monitor runtime 的单行/单 key 执行记录。
- 总 baseline universe：`67,861` sign rows。
- HPC 运行单位：受管一次 runtime job（preflight / pilot / full）。
- 不允许重训、重拟合、阈值重调、accuracy retune、数据补写或结果挑选。

## 4. Approved Scope

允许：

1. WP4 本地 no-refit reproduction。
2. full `67,861` sign local baseline。
3. managed Tongji HPC exact preflight。
4. managed Tongji HPC fast preflight。
5. managed Tongji HPC pilot。
6. managed Tongji HPC full。
7. interface separation 证据。
8. runtime distribution / deadline / failure / persistence / key conservation 证据汇总。

不允许：

- 任何 refit / retrain / recalibration。
- 任何 accuracy claim、RQ024 式诊断复做、Tier2 修复叙事。
- paper repo、`decision.md`、原始数据、衍生数据主产物改写。
- 非 Tongji HPC 的远端执行面。

## 5. Runtime Surface

- Tongji HPC durable root：`/share/home/u25310231/ZXC/RQ026_frozen_monitor_runtime`
- 新提交的 Slurm job name 必须以 `zxc-` 开头。
- 运行环境必须记录 exact Linux env：OS / Python / interpreter path / core package versions / submit host / runtime host / command / working root。
- 本地与 HPC 必须保持 interface separation：输入契约、输出契约、监控契约分开记录，不得把临时 shell 状态当结果。

## 6. Required Evidence

必须产出并保留：

- local no-refit reproduction 证据；
- full `67,861` sign local baseline；
- exact preflight 与 fast preflight 对照；
- pilot 与 full 的 runtime 证据；
- `P50 / P95 / P99` runtime；
- deadline hit / miss；
- failure taxonomy；
- persistence checks；
- key conservation checks。

## 7. Output Location Contract

RQ026 的执行输出应进入其专属 work / report 路径；本计划只冻结合同，不提前规定未授权的额外报告路径。

## 8. One-pass Checks

1. 确认本地 reproduction 为 no-refit。
2. 确认 full-sign local baseline 分母固定为 `67,861`。
3. 确认 HPC root 位于 `/share/home/u25310231/ZXC/RQ026_frozen_monitor_runtime`。
4. 确认所有 Slurm job names 以 `zxc-` 开头。
5. 确认 exact Linux env 被逐项记录。
6. 确认 interface separation 文档存在。
7. 确认 P50/P95/P99、deadline、failure、persistence、key conservation 均有数值或结构化状态。

## 9. Stop Gates

出现任一情况立即停止并上报：

- 需要 refit/retrain 才能继续；
- baseline key 数不守恒，无法对齐 `67,861`；
- HPC 输出不能落到 `/share/home/u25310231/ZXC/RQ026_frozen_monitor_runtime`；
- Slurm job name 不符合 `zxc-` 约束；
- exact Linux env 无法完整记录；
- runtime 叙事试图跨入 RQ024 accuracy boundary；
- 需要改写 paper、decision、data 主产物；
- 触及任何 protected data。

## 10. Claim Boundaries

本计划只允许声明：

- frozen monitor/runtime 的 reproducibility 与 runtime evidence；
- local / HPC runtime distribution；
- deadline / failure / persistence / key conservation 的描述性证据；
- interface separation 是否成立。

本计划不允许声明：

- accuracy improved；
- monitor is better than baseline；
- causal effect；
- production readiness by default；
- 对 RQ024 诊断的修复已经完成。
