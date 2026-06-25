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
