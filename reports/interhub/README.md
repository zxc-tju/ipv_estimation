# InterHub Reports

Canonical InterHub generated result and report root:

`reports/interhub/ipv_estimation_results/`

The raw InterHub CSV/pkl inputs live under `data/interhub/raw/`.
New InterHub processing runs should use `pipelines/interhub/process_interhub.py`
and write outputs under this reports root unless a task explicitly specifies a
temporary output directory.
