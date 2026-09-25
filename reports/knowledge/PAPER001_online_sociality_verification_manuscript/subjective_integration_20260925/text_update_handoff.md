# Text integration handoff — 2026-09-25

## Author instruction

Update manuscript text using the final subjective-rating evidence and the agreed interpretation; then provide a systematic main-figure plan for a local agent. Figure drawing is explicitly deferred.

## Version record

- Manuscript repository: `zxc-tju/NMI---Online-Sociality-Verfication-for-Autonomous-Vehicle`.
- Base manuscript: `2dab39d60216633bb6b1a9713d4ff2585eae2024`.
- Integration commit: `02d961cc650521e34b81333efa48fc04b9555658`.
- Final text commit: `6ce95d397c4c89259b5c2f499f05371e5c2a0678`.
- The manuscript `main` was fast-forwarded to the final text commit after its CI passed. No force update was used.
- Evidence release: `ipv_estimation@0e05a65b93e89f58f75accd7a6674f943b8302ad`, final RQ028 extraction package only.
- Figure plan: [figure_revision_plan.md](figure_revision_plan.md), added in research commit `fd02d0f096a86d79d28dab87e971b14f54f4569f`.

## Changes

Four manuscript text files changed: `main.tex`, `CLAUDE.md`, `structure.md`, `claims_register.md`. No image, bibliography, original research dataset, threshold or frozen analysis code changed.

The 150-word abstract, introduction and discussion now connect conditional atypicality with complementary subjective experience and vehicle-motion endpoints. New Results 2.6 follows the matched human-driving audit. New Methods 4.7 keeps the 40-person/20-pair subjective protocol separate from the 20-driver arm. Supplementary Note 2 and Table S2 retain the principal contrasts and interpretive exceptions. Figure 6 has an explicit artwork placeholder; it does not contain simulated results.

The main claim is direction-specific subjective experience: both outside-range segment categories receive higher counterpart-position atypicality ratings; assertive segments have larger comfort/cost differences; accommodating segments also differ from within-range segments. Subjective results remain exploratory. Direct dispreference, appropriateness, psychological IPV recovery, warning latency and realised harm are not newly validated.

## Actual figure audit

Inspected the baseline CI PDF, run `33226183680`, artifact `9706959011`, pages 5, 8, 9, 13 and 16. Figure 2 contained only three panels despite its six-panel caption. Figure 5 lacked the side-split panel described in its caption. Figure 4 showed assertive-only interval summaries despite broader caption language. Interim text/captions now match those existing assets, while the new figure plan specifies the required complete replacements. The plan is a proposed content organization for local execution, not a completed redraw.

## Checks performed

- Integration commit CI: run `36086700954`, Compile LaTeX and Source lint succeeded.
- Final text CI: run `36087818584`, Compile LaTeX and Source lint succeeded; undefined-reference/citation check passed.
- Final artifact: `10844652946`, `main-pdf`, from exact final text SHA; downloaded PDF has 49 pages.
- Final PDF SHA-256: `216a7c27f6043ad77cc521e675d209d85d49db77dc411035442a13d739a5f0ad`.
- Inspected rendered abstract and subjective table; checked extracted final text for the explicit Figure 6 placeholder, two protocol/sharing completion markers, correct Supplementary Note 2 reference and clarified Scenario mean column.
- Final polishing diff contains only the abstract, the unnumbered-note reference and the table heading; result values were unchanged.
- No independent statistical reanalysis, independent reviewer certification or completed figure validation is claimed for this writing pass.

## Remaining work

1. Execute the six-figure plan, prioritising Figure 2 semantic panels, Figure 5 direction split and new Figure 6; then Figure 4 bilateral completeness and Figure 1/3 de-duplication.
2. Supply and verify the subjective study's own recruitment, apparatus, order/allocation, full questionnaire anchors, ethics/consent, stimulus-selection and frozen-monitor version records. Complete its separate data-sharing statement. The two visible completion markers must remain until verified.
3. Add the full subjective Source Data and common-scenario figure with non-conflicting Supplementary numbering. Do not publish participant-level derivatives on assumed consent.
4. Update images, captions and panel references atomically, compile and inspect all figure pages, and append a concise cross-reference to the historical agent_handoff log during the local figure handoff. This standalone dated entry preserves the historical log without rewriting its long prior record.

Text integration and figure planning are complete within this scope. Figure rendering and protocol completion remain open by design; this is not a submission-ready declaration.
