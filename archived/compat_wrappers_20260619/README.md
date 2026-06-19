# Compatibility Wrapper Archive

Archived on 2026-06-19 after the light migration to the canonical `src/` and
`pipelines/` layout.

These files were temporary compatibility wrappers:

- `agent.py`
- `ipv_estimation.py`
- `process_interhub.py`
- `simulator.py`
- `tools/*.py`

They were removed from the repository root because keeping both the old root
layout and the new canonical layout made the active project ambiguous.

Use the canonical paths instead:

- Core IPV code: `src/sociality_estimation/core/`
- Planning helpers: `src/sociality_estimation/planning/`
- InterHub pipeline: `pipelines/interhub/process_interhub.py`
- Simulation entrypoint: `pipelines/simulation/simulator.py`
