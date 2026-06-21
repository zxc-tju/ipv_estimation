# Anonymization Audit

Run ID: `RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0`

Package description: a blind behavioral criterion reference independent of the official score and IPV outputs.

## Result

Leakage audit result: PASS

The annotator-facing files under `reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/01_results/annotations` were checked for concrete team codes, source replay paths, session-path markers, official score fields, IPV numeric field names, and rank fields. The controlled mapping is stored separately at `reports/studies/RQ003_nsfc_external_evidence/RQ003_6_nsfc_ipv_validation_codex_20260620T160628+0800_fbd2d3f0/02_process/12_blind_annotation/controlled_identity_map.csv` and is not annotator-facing. Long run-root paths were removed from annotator-facing CSVs; case cards are referenced as local `blinded_items/...` files.

## Scope Distinction

- Mechanism sample: purposive case-card set for mechanism discovery only; not for natural-rate estimation and not for H3 testing.
- Validation sample: scenario-stratified random sample with fixed seed `20260620`; no IPV or official-outcome extremes were used for selection.
- Two annotator templates contain blank coding fields only; no labels were generated.

## Findings

No concrete identity, source path, official score, IPV value, or rank leakage found in annotator-facing files.

## Controlled-Access Boundary

`controlled_identity_map.csv` contains true cell IDs, team/session identifiers, and original media/trajectory paths so a later coordinator can materialize blinded files. It intentionally omits official score values and IPV numeric outputs. Do not distribute it to annotators.
