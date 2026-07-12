# RQ014 Plan v1.3 — Claude (PI 角色) 独立复核意见

日期：2026-07-11。对象：v1.3 amendment + 三套 resolved registries + FL05/pass4 实现 + fixtures，
及 codex targeted review（`PASS_AS_NONEXECUTABLE_DESIGN_CONTRACT`）。

## 结论：CONCUR（同意 v1.3 取代 v1.2 作为当前非执行设计合同）

## 独立核验（本次会话沙盒内完成）

1. checksum manifest 14/14 `OK`（`sha256sum -c`，仓库根）；
2. 三份 contract tests 独立复跑 **42 passed**（与 codex 评审宣称一致）；
3. v1p3 config registry：12 configs 原样、8 个授权布尔全 `false`、`delta_NI=0.08`、
   单侧 alpha 统一 0.025、`execution_authorized: false`；
4. START_HERE / STUDIES / workflow log 已登记 v1.3 verdict（R13 条件 4 满足）。

## 对 v1.3 改动的 PI 立场

- **科学内核完整保留**：12-config valid family、三层主张（构造/简约/增量）、扩展臂、
  非循环 required-cell、fail-closed FL05 全部延续；hardening 未改变任何科学命题。
- **同意的实质改进**：(a) 扩展臂推断措辞降级为 complete-null exploratory（varying supports
  下不宣称 strong FWER，诚实）；(b) 独占性改为 V04 配对对比 + 同步上界（修正了我 v1.1/1.2
  的非正式定义）；(c) `delta_NI=0.08` 依据改为评分前 PI 规范性选择而非经验估计——
  epistemically 更干净；(d) R12 指纹升级四条件防止无关负相关误判"找到旧结果"；
  (e) 八布尔授权 AND 拆分消除"一次批准全解锁"。
- **接受但记录的取舍**：R8 将 sign/role flip 降为纯描述、废除自动升级 forensic。
  同意理由：数据驱动的自动门会重新引入结果依赖。PI 承诺：若出现
  `FLIP_STRONGER_DESCRIPTIVE`，以新 amendment 人工处置，不静默忽略。
- **预期管理（watch items，非阻断）**：(1) X02 资格门极严，大概率终态为
  `INACCESSIBLE_DEFINITION` 或 `INELIGIBLE_SCALE_INCOMPATIBLE`——可接受：宁可"未测"
  也不"伪测"；(2) F07/F08 因无 cutoff 前 receipt 只能永久 `INACCESSIBLE`——诚实边界；
  (3) specificity 校准的算力天花板（≤20k CPU-h）仅在走到 confirmation 后才会触发，
  比例性受 gate 保护；(4) 全案的期望情形仍是 specification-recovery 榜单 +
  可能的 `INCONCLUSIVE_LOW_POWER`——该机器保证的是诚实，不是阳性结果。

## 下一步（按 base gate 顺序）

唯一待 PI 决定的事项：是否翻转 `g0_readonly_forensics_authorized=true`（只读取证授权，
零评分接触、零科学计算）。翻转后运行 checksum-bound pass4/FL05 关闭 F05–F10，再进 formal G1。
