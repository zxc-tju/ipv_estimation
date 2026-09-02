# Sociality Subjective Experiment System

一个可本地运行的主观实验系统，面向“区间内 / 区间外驾驶互动行为”的人类评价。

## 已实现

- **受试者模式 `/`**
  - 知情同意与持照资格筛查；
  - 匿名驾驶背景；
  - 任务说明与理解检查；
  - 成对选择：依次播放 A、B，选择更愿意互动的车辆；
  - 单片段评分：可接受性、可预测性、舒适度、互动负担、感知不安全、过度激进、过度谨慎；
  - 实验后盲法检查与开放反馈；
  - 断点续做、进度显示、重播次数、响应时长和页面事件记录。

- **工作模式 `/work`**
  - Admin key 登录；
  - 查看会话和数据量；
  - 查看隐藏条件；
  - 上传或替换视频；
  - 修改刺激名称、场景和启用状态；
  - 查看和新增成对任务；
  - 逐表导出 CSV。

- **后端**
  - FastAPI；
  - SQLite；
  - session 创建时完成平衡抽样与 A/B 随机化；
  - Human/AV、inside/outside、偏离方向和幅度不会返回给受试者前端。

## 快速启动

需要 Python 3.9 或更高版本。

```bash
cd sociality_subjective_experiment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env，必须把 ADMIN_KEY 改为随机长密钥；示例占位值会被拒绝
./start.sh
```

浏览器打开：

- 受试者界面：`http://127.0.0.1:8000/`
- 工作模式：`http://127.0.0.1:8000/work`

Windows PowerShell：

```powershell
$env:ADMIN_KEY="your-long-random-key"
uvicorn app:app --host 0.0.0.0 --port 8000
```

## 正式采集前

`.env` 中设置：

```text
ADMIN_KEY=<长随机密钥>
ALLOW_PLACEHOLDER_TRIALS=0
```

工作模式仪表盘中的“缺少视频”必须为 0。开发期保留 `ALLOW_PLACEHOLDER_TRIALS=1`，网页会显示“模拟完成播放”按钮，方便测试空视频流程。

## 默认实验流程

1. 知情同意和基本信息；
2. 说明与理解检查；
3. 12 个成对选择 trial；
4. 18 个单片段评分 trial；
5. 实验后问卷；
6. 完成并锁定 session。

默认刺激为 `2 × 3` 设计：

| 来源 | 区间内 | 强势侧区间外 | 礼让侧区间外 |
|---|---|---|---|
| Human | H-in | H-out-A | H-out-C |
| AV | AV-in | AV-out-A | AV-out-C |

## 视频要求

支持 `.mp4`、`.webm`、`.mov`、`.m4v`，单文件上限 500 MB。建议保持：

- 相同画幅、分辨率、帧率和时长；
- 相同目标车辆标记和视角；
- pair 内使用同一场景；
- 文件名、车辆外观和水印不能泄露 Human/AV 或 monitor verdict。

上传后，服务器会用随机 UUID 文件名保存到 `uploads/`。

## 数据表

默认数据库：`data/experiment.sqlite3`。

- `participants`：匿名背景；
- `sessions`：会话状态与实验后问卷；
- `stimuli`：隐藏条件、视频与研究者元数据；
- `pair_definitions`：成对任务；
- `session_trials`：每名受试者冻结后的 trial 顺序与 A/B 映射；
- `pairwise_responses`：偏好、置信度、播放完成和响应时长；
- `single_responses`：逐片段评分；
- `events`：播放、重播、页面离开等日志。

## 随机化

创建 session 时：

- pairwise trial 按 `contrast_code` 平衡抽样；
- single trial 按 `actor_source × verdict_class × deviation_side` 平衡抽样；
- 每个 pair 的 A/B 顺序随机；
- 随机种子写入 `sessions`；
- trial 队列写入 `session_trials`，断点续做不会重新随机。

## 隐私与部署

当前版本适合本地、实验室局域网或机构服务器原型。系统默认不记录 IP 地址，也不收集直接身份信息。

正式网络部署还应完成：

- HTTPS；
- 机构反向代理和访问控制；
- 数据库加密备份；
- 伦理审批文本替换；
- 数据保留与删除规则；
- 管理员密钥轮换；
- 防重复参与和招募平台 participant code；
- 服务器日志隐私审计。

不要把带 `--reload` 的开发服务器直接暴露到公网。

## 测试

```bash
pytest -q
```

测试使用临时数据库，不接触正式数据。
