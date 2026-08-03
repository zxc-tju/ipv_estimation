# STATUS — track K2（rq015k-fullcorpus-gate）

state: DONE
updated_at: 2026-08-03T00:50:27Z
released_by: commander（监督方），见 commander_notes.md 2026-08-03T00:50:27Z 一节
phase: 结项验收通过，已放行
summary: InterHub 全语料 log 域门判据台账已交付。4,981,984 个求解单元：OK 3,502,340（70.3001%）、NEAR_UNIFORM 1,457,746（29.2604%）、NO_IPV_EFFECT 19,964（0.4007%）、SOLVER_FAILURE 1,934（0.0388%）。RQ009 回填 8,994,736 行 exact-one join，misses 0、duplicates 0（实测）。G 锚点在正确 HPC 基线下 2,300/2,300 逐位相同、max_abs_diff=0.0。总行数 14,473,982。产物在 data/derived/rq015k_logdomain_gate/l1_v1/。
next: 遗留项转后续——(1) J 的 HT 分母 2,646,058 与 RQ009 8,994,736 的确切关系【唯一科学遗留】(2) launch_leader.sh nohup 早退（第 4 次，基础设施债）(3) validate_outputs 三次全量物化改流式（优化项）(4) OnSite/WOD materializer（PI 已裁定本轮不做）
