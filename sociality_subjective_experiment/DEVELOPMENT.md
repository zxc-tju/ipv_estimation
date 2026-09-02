# 开发与实验配置说明

## 1. 系统边界

网页只负责呈现、随机化和采集。网页不计算 IPV、人类参考区间或越界方向。研究者需要先在研究管线中生成 verdict，再把刺激及隐藏元数据录入系统。

## 2. 正式刺激应冻结的信息

```text
stimulus_id
scenario_id
trajectory_id
actor_source                 # human / av
verdict_class                # inside / outside
deviation_side               # none / assertive / accommodating
deviation_magnitude
role
priority_state
readability / human-support audit fields（写入 metadata_json）
video render version
```

正式采集开始后，不应替换视频或修改条件标签。建议在启动正式采集前：

1. 备份 `data/experiment.sqlite3`；
2. 备份 `uploads/`；
3. 记录配置、数据库和视频文件 SHA-256；
4. 将 `ALLOW_PLACEHOLDER_TRIALS` 设置为 0；
5. 用独立测试账号完成一整次 session；
6. 清空测试数据后再开放正式入口。

## 3. Pairwise trial

采集：

```text
preference_raw = A | B | NO_PREFERENCE
preferred_stimulus_id
choice_confidence = 1..5
playback_a_complete
playback_b_complete
replay_count
response_time_ms
```

两段视频按 A → B 顺序播放，A/B 映射由后端随机，并保存在 `session_trials`。

## 4. Single-clip trial

主要绝对终点：`acceptability`。

独立采集：

```text
predictability
comfort
perceived_unsafe
interaction_burden
assertiveness
hesitation
rating_confidence（可选）
free_text_reason（可选）
```

可接受性、感知不安全和互动负担分开保存，便于检验“不受欢迎”“增加调整负担”和“直接危险感”之间的差异。

## 5. API

Participant:

```text
GET  /api/public/config
POST /api/session/start
GET  /api/session/{token}/state
POST /api/session/{token}/pairwise
POST /api/session/{token}/single
POST /api/session/{token}/post
POST /api/session/{token}/complete
POST /api/session/{token}/event
```

Work mode:

```text
GET  /api/admin/summary
GET  /api/admin/stimuli
PUT  /api/admin/stimuli/{id}
POST /api/admin/stimuli/{id}/video
GET  /api/admin/pairs
POST /api/admin/pairs
GET  /api/admin/export/{table}.csv
```

工作模式接口通过 `X-Admin-Key` 鉴权。

## 6. 下一轮建议

- 刺激配置 JSON/CSV 批量导入；
- 正式 practice video 和 trial-level 理解检查；
- participant code 与重复参与控制；
- 中英文切换；
- PostgreSQL；
- 研究版本冻结和哈希页面；
- 自动数据质量报告；
- 招募平台回调；
- 机构 SSO；
- 服务器端加密备份。
