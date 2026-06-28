# RQ012B Run — W0 extractor-health (M3-independent slice)

RUN_ID: `RQ012B_1_event_harm_20260625T202307+0800_38f47437` · 2026-06-25
RQ: RQ012B (OnSite automatic-event harm analysis; no human labels)
Orchestration: Claude = controller only; all execution by Codex CLI fleet (gpt-5.5, xhigh).

## Status
- **Scientific endpoint (M3 deviation → automatic-event / official-harm association): NOT RUN — `BLOCKED_PENDING_M3`** (RQ009 M3 unfrozen; plan stages 3/4/5 deferred — see `02_process/0{3,4,5}_*/BLOCKED_PENDING_M3.md`).
- **Executed: M3-independent slice + full wrapper.** Stage-1 plan review + stage-2/W0 frozen 9-event extractor health on clean_285, with independent review, red team, blind replication, bilingual report, final review.
- **OVERALL_STATUS: COMPLETE (slice)** — 4-role validated; no new manuscript claim.

## Start here
- Bilingual report entry: `00_entry/index.html` → `90_report/index.html` (EN) / `90_report/index.zh.html` (ZH)
- PI synthesis: `02_process/09_report/CONCLUSIONS.md`
- Figures (Nature-figure skill): `01_results/figures/` (fig1 extractor health, fig2 sampling sensitivity; PNG+PDF+SVG + source_data + manifest)

## Extractor-health conclusions (evidence only — NOT scientific claims)
- EH-1 computable 280/285 (98.25%); precedence 2.66% (E09<E15); identity 100%; E01=0 definitionally inert.
- EH-2 sampling-rate sensitive → **pin native ~10 Hz for all downstream event extraction**.
- EH-3 absolute counts are extraction-convention dependent; counts are evidence, never outcomes.

## Provenance
- GIT_HEAD `4aef4d22…` · PLAN_SHA256 `3634181f…c050899`
- Frozen extractor SHA256 `f0da56b4…a56cef`; config `035c2cdf…2ca5`; universe mask `fdb055fc…6ab3`
- Env: Python 3.9.6, pandas 2.3.3, numpy 2.0.2 · NATURE_SKILL_STATUS=USED
- Stage records under `02_process/`; prompts under `02_process/00_meta/prompts/`; meta `02_process/00_meta/RUN_META.md`.
- Derived data: `data/derived/onsite_competition/RQ012B_event_harm/`.
