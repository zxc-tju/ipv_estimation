# HPC full_datasets/nuplan_agv_all IPV 运行说明

这批数据位于本地：

```text
C:/Users/xiaocongzhao/OneDrive/Desktop/Projects/1_Codes/2_sociality_estimation/interhub_traj_lane/0_raw_data/full_datasets/nuplan_agv_all
```

CSV 共有 10335 行，当前本地预检结果是 `pkl_events=10335`、`matched_rows=10335`、`unmatched_rows=0`。其中 `nuplan_train=7825`、`av2_motion_forecasting=2510`。现有 `process_subsets_for_yiru_ipv.py` 已经对 `nuplan_train` 使用下采样因子 2，也就是 20Hz -> 10Hz；Argoverse/AV2 保持原采样。

## 1. 本地上传

在本地 Git Bash 或 WSL 终端执行。HPC 说明里的登录端口是 `10022`，不要把密码写进命令或脚本。

Git Bash 路径：

```bash
LOCAL_REPO="/c/Users/xiaocongzhao/OneDrive/Desktop/Projects/1_Codes/2_sociality_estimation"
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
LOCAL_REPO="/mnt/c/Users/xiaocongzhao/OneDrive/Desktop/Projects/1_Codes/2_sociality_estimation"
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
  --preflight-only
```

预期核心结果：

```text
csv_rows=10335
pkl_events=10335
matched_rows=10335
unmatched_rows=0
```

## 3. 提交数组作业和合并作业

默认使用 12 个数组 shard，每个 shard 1 个节点、96 个 workers，不保存 plot。

```bash
ARRAY_JOB=$(sbatch --parsable submit_full_datasets_nuplan_agv_ipv_array.sh | cut -d';' -f1)
SHARD_COUNT=12 sbatch --dependency=afterok:${ARRAY_JOB} submit_full_datasets_nuplan_agv_ipv_merge.sh
squeue -u u25310231
```

如果想手动合并：

```bash
SHARD_COUNT=12 sbatch submit_full_datasets_nuplan_agv_ipv_merge.sh
```

合并后的主要结果：

```text
interhub_traj_lane/1_ipv_estimation_results/full_datasets/nuplan_agv_all/selected_interactive_segments_equalized_with_ipv.csv
interhub_traj_lane/1_ipv_estimation_results/full_datasets/nuplan_agv_all/selected_interactive_segments_nuplan_agv_full_with_ipv.csv
```

第二个是 merge 脚本额外复制出的友好命名版本。

## 4. 超时后的断点续跑

如果部分 case 因超时没有完成，可以只处理 incomplete 行：

```bash
ONLY_INCOMPLETE=1 SHARD_COUNT=12 sbatch submit_full_datasets_nuplan_agv_ipv_array.sh
```

续跑完成后，用现有合并 CSV 作为 base 进行 patch merge：

```bash
ONLY_INCOMPLETE=1 SHARD_COUNT=12 sbatch submit_full_datasets_nuplan_agv_ipv_merge.sh
```
