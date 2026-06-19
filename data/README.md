# Data

This folder is the canonical local data entrypoint.

Track README/manifest/index files when useful. Large raw payloads live in
ignored subdirectories and should not be staged:

- `data/interhub/raw/`
- `data/onsite_competition/raw/`
- `data/onsite_competition/top5_research_subset/teams/`
- `archived/argoverse/0_souce_data/` for legacy Argoverse data

Use manifest files or README notes here when a workflow needs a stable data
entrypoint without duplicating raw data.
