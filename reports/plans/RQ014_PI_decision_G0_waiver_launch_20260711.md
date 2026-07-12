# RQ014 PI Decision — G0 豁免关闭与盲搜开工授权

日期：2026-07-11。决定人：PI（Desmond Zhao，经 Cowork 会话明示："不用再去翻了……直接开始盲搜"）。
性质：v1.3 合同框架下的 scoped decision（v1.3 R13：授权只能由后续 scoped decision 逐项开启）。
本文件为 checksum-bound amendment 的一部分，登记于 `RQ014_plan_v1p4_checksums_20260711.sha256`。

## 决定 1 — G0 剩余取证面豁免关闭

F05、F06、F07、F08、F10 由 PI 明示放弃扫描，关闭状态一律为：

```text
INACCESSIBLE_PI_WAIVED   (reason: PI explicitly waived remaining forensics, 2026-07-11)
```

G0 终态：`CLOSED_WITH_INACCESSIBLE_SURFACES`（F01–F04 = NOT_FOUND_ON_SCANNED_SURFACES；
F05–F10 = INACCESSIBLE 系）。

### 残余风险声明（必须随最终报告）

1. FL05 历史统计索引未执行 ⇒ `HISTORICAL_RATING_IPV_FINGERPRINT_CANDIDATE` 与
   `HISTORICAL_SIGNATURE_RECONSTRUCTED` 两类结论在本 run 内**不可达**；
2. 若盲搜终态为 NOT_RECOVERED / INCONCLUSIVE，将无法区分"记忆有误 / 旧结果为 bug
   产物 / 旧记录躺在未扫描档案中"三种解释；
3. 6-29 pilot 链中间统计未核查 ⇒ artifact 假设只能靠 forensic FT twins 的重算证据
   （FT01–FT04 为计算型，不依赖档案，仍可执行）；
4. 上述代价由 PI 知情接受，为换取立即开工。

## 决定 2 — Formal G1 判定：PASS

依据：codex v1.3 targeted review（无 blocker，`PASS_AS_NONEXECUTABLE_DESIGN_CONTRACT`）+
Claude PI 独立复核（CONCUR，含 42/42 测试与 14/14 校验和独立复跑）+ 本决定使 G0 达到
合法关闭。base plan 的 gate 顺序（G0→G1）由此满足。

## 决定 3 — 授权翻转（八布尔中开启两项）

```text
g2_ratings_blind_build_authorized = true    # 特征/包络/资格门/anchor parity 盲构建
scientific_compute_authorized     = true    # G2 资源试点 + G2P 功效模拟所需
```

其余六项维持 false。特别地：`discovery_rating_join_authorized` 保持 false ——
**评分 join 需在 G2/G2P 完成、PI 审阅功效结论后另行 scoped decision**。
extension/forensic 计算授权同样待 G2P 后另批。

## 决定 4 — 执行路径

执行按 `RQ014_G2_kickoff_prompt_20260711.md`（Mac 端 codex fleet 编排；HPC durable root
`/share/home/u25310231/ZXC/RQ014_recovery/<RUN_ID>/`，job 前缀 `zxc-rq014-`）。三个 v1p3
registry 为唯一机器权威；prose 冲突即停。

## 登记

- forensic registry v1p3 的 F05–F08/F10 状态字段已按本决定改写（见 v1p4 checksums）；
- 三个 registry 的授权布尔已同步翻转两项；
- `tests/test_rq014_v1p3_registry_contract.py` 两处断言按本决定同步修订
  （授权向量、豁免面字段），修订后合同测试 42/42 通过；
- START_HERE / STUDIES / workflow log 同步更新。
