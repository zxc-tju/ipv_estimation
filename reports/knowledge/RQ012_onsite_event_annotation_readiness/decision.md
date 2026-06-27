# RQ012 Decision: OnSite Event Annotation Readiness

Status: ACCEPTED — scope revised by PI 2026-06-24. Automatic-event readiness accepted; **two-human blind annotation DEPRECATED (not pursued).** Consequence/behaviour reference will use automatic events + OnSite official outcomes; human alignment is relocated to WOD-E2E preference + InterHub.

Run ID: `RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37`
Plan SHA-256: `921f6bb3b850126189895dcca52f054a1c6f5e95a16b05159bec0a13c26ad77e`
Basis: final review PASS; `reviews/claude_review.md` + `reviews/codex_review.md`; PI decision 2026-06-24 to drop human blind annotation.

## PI Decision (2026-06-24): drop human blind annotation

The two-human blinded annotation (convergent human-judgment leg) is **deprecated**. Rationale: it was the optional convergent-validity leg; the program retains two stronger, already-available signals — WOD-E2E released human preference scores (human alignment) and OnSite official rankings/scores/collisions/deductions (objective criterion + consequence). A 2-annotator study is weak/slow evidence; automatic event extraction covers event-aligned analysis without humans.

Coupling condition (noted): with annotation dropped, the human-alignment leg rests on **WOD-E2E (RQ010) + InterHub**; this raises the importance of RQ010B (now authorized).

## Accepted Claims (readiness)

| ID | Claim |
|---|---|
| RQ012-KC-READINESS | Gates 012-0/012-1 pass; 012-2 text/surface-cleared; 012-3 ready. The blinded, outcome-free event design and extractor-readiness checks exist. |
| RQ012-KC-AUTOEVENTS | The automatic event extractor (9 events; precedence/identity guards) is computable without humans — usable for event-aligned analysis (extractor health only, not outcomes). |
| RQ012-KC-CODEBOOK | The codebook separates automatic, human-only, and removed events; construct-proximal labels are secondary. |

## Deprecated / Not Pursued

| Item | Disposition |
|---|---|
| Two real human blind labels + kappa/AC1 agreement | DEPRECATED (PI 2026-06-24); Gate 012B closed as not-pursued. |
| Human-only events as primary endpoints | Dropped; use automatic events + official outcomes instead. |

## Downstream Effect

RQ012B (event-aligned harm) is reframed to **automatic events + OnSite official outcomes** (no human labels). This removes the human-label dependency from RQ012B and RQ013 (beyond-safety). `evidence.csv` back-filled 2026-06-24.

## Paper Handoff

OnSite consequence/realised-deficit evidence = automatic events + official collisions/scores/deductions (objective). Do NOT claim a human-judgment convergent leg from OnSite; human alignment is carried by WOD-E2E preference + InterHub.

## RQ012B Execution Result — W0 extractor-health (2026-06-25)

Run `RQ012B_1_event_harm_20260625T202307+0800_38f47437` (GIT_HEAD 4aef4d22; PLAN_SHA256 3634181f…c050899; via Codex fleet orchestration, Claude = controller). **Scientific endpoint NOT run — `BLOCKED_PENDING_M3`** (RQ009 M3 unfrozen → plan stages 3/4/5 deferred). Executed the M3-independent slice only (PI+user decision 2026-06-25): stage-1 plan review + stage-2/W0 frozen 9-event extractor health on clean_285, with independent review, red team, blind replication, and a bilingual offline report.

Extractor-health findings (evidence only — **NOT** scientific/manuscript claims; counts are never outcomes):
- **EH-1** — computable **280/285 (98.25%)** (5 data-availability misses: MissingCaseWindow×3, MissingEgoRows×2); precedence guard suppresses **2.66%** (all E09<E15); identity **100%**; **E01=0 definitionally inert** (spec defines no frozen counterpart relation — by design, not a bug).
- **EH-2** — the frozen extractor is **sampling-rate sensitive** (frame-difference jerk/accel + frame-spacing-duration triggers); independently reproduced under principled physical-time resampling (5 Hz +88%, 20 Hz −30%). **Carried condition: pin native ~10 Hz for all event extraction feeding the future stages 3–5.**
- **EH-3** — absolute event counts are extraction-convention dependent (blind replication AGREES on E01=0 / E18 / computability, diverges ~1–5% on totals); reinforces "counts = extractor evidence only".

Data-health/negative checks PASS (leakage clean — neutral `usecols`, no outcome/IPV/score columns in any derived CSV; universe = `corrected_clean_mask.csv` = 285 replay rows, T19 replay-excluded; no silent caps). Validation: plan-review PROCEED-WITH-CONDITIONS; red-team CLEAR (no blocker; 1 major = sampling-framing split, applied); blind replication AGREE on health/direction; final-review PASS. Nature-figure skill USED (no fallback). **No new accepted manuscript claim this round** (endpoint blocked; feeds the post-M3 RQ012B run and RQ013). Detail: run `02_process/09_report/CONCLUSIONS.md`; report `…/RQ012B_1_event_harm_20260625T202307+0800_38f47437/00_entry/index.html`.
