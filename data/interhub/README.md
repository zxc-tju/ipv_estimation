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
immutable raw-data snapshot:

```text
/share/home/u25310231/ZXC/sociality_estimation/data/interhub/raw
  -> /share/home/u25310231/ZXC/sociality_estimation/data/interhub/snapshots/interhub_legacy_20260711_v1
```

The retired legacy path contains a compatibility link to the same snapshot;
it is not an alternate data owner. Historical results are frozen under
`archives/historical-results/interhub_legacy_20260711_v1/`. The migration was
verified against 51 raw and 173,034 result SHA-256 entries before switching,
then reverified against the final snapshot paths.

This preserves the default subset CLI paths and full-dataset paths without a
second active copy. Before the one-time migration, the mapping was created or
verified with:

```bash
bash scripts/hpc/ensure_interhub_data_topology.sh
```

The script is fail-closed and is retained only for provenance; do not use it to
point the managed path back at the retired legacy root. Production launchers
verify the registered raw manifest and write only under per-run managed work
directories. New derived outputs belong under the managed project root.
