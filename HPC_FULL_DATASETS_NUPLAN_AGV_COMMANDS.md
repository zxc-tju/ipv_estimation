# HPC full_datasets/nuplan_agv_all IPV 运行说明

这批数据位于本地：

```text
C:/Users/46936/OneDrive/Desktop/Projects/1_Codes/2_sociality_estimation/interhub_traj_lane/0_raw_data/full_datasets/nuplan_agv_all
```

CSV 共有 10335 行，当前本地预检结果是 `pkl_events=10335`、`matched_rows=10335`、`unmatched_rows=0`。其中 `nuplan_train=7825`、`av2_motion_forecasting=2510`。现有 `process_subsets_for_yiru_ipv.py` 已经对 `nuplan_train` 使用下采样因子 2，也就是 20Hz -> 10Hz；Argoverse/AV2 保持原采样。

本次 full dataset 计算默认排除已经在 `interhub_traj_lane/0_raw_data/subsets_for_yiru/selected_interactive_segments_equalized.csv` 中出现过的 case。按 `folder + scenario_idx + key_agents + track_id` 匹配，当前重叠 5000 行，排除后实际需要计算 5335 行：`nuplan_train=5325`、`av2_motion_forecasting=10`。子集结果后续再统一汇总进 full 结果。

## 1. 本地上传

在本地 Git Bash 或 WSL 终端执行。HPC 说明里的登录端口是 `10022`，不要把密码写进命令或脚本。

Git Bash 路径：

```bash
LOCAL_REPO="/c/Users/46936/OneDrive/Desktop/Projects/1_Codes/2_sociality_estimation"
REMOTE_HOST="u25310231@logini.tongji.edu.cn"
REMOTE_REPO="/share/home/u25310231/ZXC/ipv_estimation"

rsync -avz --progress -e "ssh -p 10022" \
  "$LOCAL_REPO/process_subsets_for_yiru_ipv.py" \
  "$LOCAL_REPO/submit_full_datasets_nuplan_agv_ipv_array.sh" \
  "$LOCAL_REPO/submit_full_datasets_nuplan_agv_ipv_merge.sh" \
  "$REMOTE_HOST:$REMOTE_REPO/"

rsync -avz --progress -e "ssh -p 10022" \
  "$LOCAL_REPO/interhub_traj_lane/0_raw_data/full_datasets/nuplan_agv_all/" \
  "$REMOTE_HOST:$REMOTE_REPO/interhub_traj_lane/0_raw_data/full_datasets/nuplan_agv_all/"
```

如果用 WSL，把 `LOCAL_REPO` 改成：

```bash
LOCAL_REPO="/mnt/c/Users/46936/OneDrive/Desktop/Projects/1_Codes/2_sociality_estimation"
```

## 2. HPC 上预检

```bash
ssh -p 10022 u25310231@logini.tongji.edu.cn
cd /share/home/u25310231/ZXC/ipv_estimation
source /share/apps/miniconda3/etc/profile.d/conda.sh
conda activate ipv

python process_subsets_for_yiru_ipv.py \
  --csv interhub_traj_lane/0_raw_data/full_datasets/nuplan_agv_all/pkl/selected_interactive_segments_nuplan_agv_full.csv \
  --pkl-root interhub_traj_lane/0_raw_data/full_datasets/nuplan_agv_all/pkl \
  --output-root interhub_traj_lane/1_ipv_estimation_results/full_datasets/nuplan_agv_all \
  --exclude-csv interhub_traj_lane/0_raw_data/subsets_for_yiru/selected_interactive_segments_equalized.csv \
  --preflight-only
```

预期核心结果：

```text
csv_rows=10335
selected_rows=5335
excluded_rows=5000
pkl_events=10335
matched_rows=5335
unmatched_rows=0
```

## 3. 检查已完成和未完成 case

如果之前任务被终止，先扫描 case artifacts。这个命令不会计算，只统计已经完成和还没完成的 case：

```bash
python process_subsets_for_yiru_ipv.py \
  --csv interhub_traj_lane/0_raw_data/full_datasets/nuplan_agv_all/pkl/selected_interactive_segments_nuplan_agv_full.csv \
  --pkl-root interhub_traj_lane/0_raw_data/full_datasets/nuplan_agv_all/pkl \
  --output-root interhub_traj_lane/1_ipv_estimation_results/full_datasets/nuplan_agv_all \
  --exclude-csv interhub_traj_lane/0_raw_data/subsets_for_yiru/selected_interactive_segments_equalized.csv \
  --scan-incomplete-only
```

重点看输出里的 `incomplete_rows`。只要还有未完成 case，就可以重复提交下面的 4 节点 continuation 作业。

## 4. 提交 4 节点计算作业

默认使用 4 个数组 shard，也就是最多 4 个节点；每个 shard 1 个节点、96 个 workers，不保存 plot。脚本已内置：

- `--exclude-csv interhub_traj_lane/0_raw_data/subsets_for_yiru/selected_interactive_segments_equalized.csv`
- `--only-incomplete`

所以它会跳过子集内已经完成的 case，也会跳过 full-dataset 输出目录里已经有完整 artifacts 的 case。

```bash
SHARD_COUNT=4 WORKERS=96 sbatch submit_full_datasets_nuplan_agv_ipv_array.sh
squeue -u u25310231
```

如果作业再次因为时间或资源限制中断，重新执行同一条 `sbatch` 命令即可续跑。它会重新扫描并只处理还没完成的 case。

## 5. 后续汇总

本轮先只计算子集以外的 case。等所有 `incomplete_rows` 变成 0 后，再做 full dataset 结果汇总，并把 `subsets_for_yiru` 已完成结果合并进来。

临时检查 case artifacts：

```bash
find interhub_traj_lane/1_ipv_estimation_results/full_datasets/nuplan_agv_all/cases -name metadata.json | wc -l
find interhub_traj_lane/1_ipv_estimation_results/full_datasets/nuplan_agv_all/cases -name ipv_results.xlsx | wc -l
```
