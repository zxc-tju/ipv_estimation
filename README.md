# IPV Estimation Toolkit

This repository provides reusable tools for estimating Interaction Preference Values (IPVs) from paired vehicle trajectories. The codebase packages the optimisation model that previously lived in a notebook into importable modules and command-line scripts so it can be plugged into new datasets with minimal effort.

## Highlights
- **Reusable estimator** – `ipv_estimation.py` exposes `MotionSequence`, `estimate_ipv_pair`, and helpers for building `[x, y, vx, vy, heading]` time-series from raw data.
- **Argoverse pipeline** – `argoverse/argoverse_process.py` scans the expected Argoverse directory layout, batches IPV inference for every case, and saves both Excel summaries and diagnostic plots.
- **Underlying game-theoretic model** – `agent.py`, `simulator.py`, and utilities under `tools/` contain the optimisation logic used by the estimator; they can still be used for simulation or for extending the estimator to new behaviours.

## Repository Layout
```
.
├── agent.py                     # Agent class with IPV-aware optimisation routines
├── simulator.py                 # Legacy simulator (shares optimisation core)
├── ipv_estimation.py            # High-level IPV estimation helpers
├── argoverse/
│   ├── argoverse_process.py     # Batch processor for Argoverse rush/yield sets
│   └── 0_souce_data/            # (Ignored) expected location for raw CSV files
├── tools/
│   ├── utility.py               # Path and trajectory utilities
│   ├── Lattice.py               # Supporting data structures
│   └── lattice_planner.py       # Planner utilities
└── requirements.txt             # Reference dependency versions used originally
```

> **Note**  
> All large datasets and notebooks are ignored by `.gitignore`. Place Argoverse CSV files under `argoverse/0_souce_data/` following the existing subfolder naming convention (`argo1/HV_HV_rush`, `argo2/interaction_hv/left_turn_rush`, …).

## Environment Setup
The estimator depends on `numpy`, `scipy`, `pandas`, `matplotlib`, `shapely`, `openpyxl`, `seaborn`, and `tqdm`. The pinned versions in `requirements.txt` match the original development environment.

Example using conda:

```bash
conda create -n ipv_estimate -y python=3.9 \
      numpy=1.21 scipy=1.7 pandas=1.4 matplotlib=3.4 shapely=1.8 \
      openpyxl seaborn tqdm pip
conda activate ipv_estimate
pip install -r requirements.txt   # optional if you need the exact pip wheel builds
```

## Running IPV Estimation on Argoverse
1. Download the processed interaction CSVs and place them in the appropriate directory under `argoverse/0_souce_data/` (the script expects the same naming scheme used in the original notebook).
2. From the repository root, run:
   ```bash
   python argoverse/argoverse_process.py
   ```
   The script will:
   - enumerate rush and yield cases in both Argoverse 1 and 2,
   - estimate IPV time-series for left-turn and go-straight agents,
   - write Excel summaries to `argoverse/1_experiment_result/ipv_estimation/.../data/`,
   - generate trajectory/IPV plots in the parallel `fig/` directories.

To process only a subset of cases, modify `ARGO_CONFIG` or add guard clauses inside `process_dataset` (for example, slicing `case_ids` or filtering `path_name`).

## Using the Estimator with Other Datasets
The central function accepts generic motion sequences:

```python
from ipv_estimation import MotionSequence, estimate_ipv_pair, concat_motion

# Example: build motion arrays from position-only samples with 0.1 s spacing
left_motion = concat_motion(left_positions, sample_time=0.1)
gs_motion = concat_motion(gs_positions, sample_time=0.1)

lt_seq = MotionSequence(data=left_motion, target="lt_argo", reference=left_refline)
gs_seq = MotionSequence(data=gs_motion, target="gs_argo", reference=gs_refline)

ipv_values, ipv_errors = estimate_ipv_pair(lt_seq, gs_seq)
```

Adapt the `target` strings and reference polylines to match how the `Agent` class retrieves lane geometry in your scenario.

## Contributing
- File issues or pull requests for bug fixes and feature additions.
- Keep large datasets and notebooks out of version control; update `.gitignore` if new generated folders appear.
- When extending the estimator, favour adding self-contained helpers alongside tests to ensure the optimisation behaviour remains stable.

