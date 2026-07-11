# InterHub Data

Raw InterHub data now lives under `data/interhub/raw/`:

- Subset CSV: `data/interhub/raw/subsets_for_yiru/selected_interactive_segments_equalized.csv`
- Subset pkl root: `data/interhub/raw/subsets_for_yiru/pkl/`
- Full-dataset raw data: `data/interhub/raw/full_datasets/`

Generated InterHub results and reports live under
`reports/interhub/ipv_estimation_results/`.

## Tongji HPC topology

The managed HPC deployment root is
`/share/home/u25310231/ZXC/sociality_estimation`.  Its
`data/interhub/raw` path is a compatibility symlink to the retained historical
raw-data root:

```text
/share/home/u25310231/ZXC/sociality_estimation/data/interhub/raw
  -> /share/home/u25310231/ZXC/ipv_estimation/interhub_traj_lane/0_raw_data
```

This preserves the default subset CLI paths and full-dataset paths without a
second large copy.  Create or verify this mapping only with:

```bash
bash scripts/hpc/ensure_interhub_data_topology.sh
```

The script is fail-closed: it refuses to replace a real directory or a link to
a different target.  A symlink does not enforce read-only permissions: until
the immutable snapshot migration is complete, production launchers must treat
the target as input-only and verify its registered manifest.  The legacy
payload is historical input data, not an alternate code checkout; new derived
outputs belong under the managed project root.
