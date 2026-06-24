# Codex Review: RQ012 OnSite Event Annotation Readiness

Status: review-complete; ready for registration as Wave-A annotation-readiness package; overall status remains `BLOCKED_FOR_HUMAN_LABELS`.
Review date: 2026-06-24.

## Scope

Reviewed study package:

- `reports/studies/RQ012_onsite_event_annotation_readiness/RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37/`

Primary study evidence read:

- `90_report/index.html`
- `README.md`
- `TRACEABILITY.md`
- `execution_status.json`
- `final_review.md`
- `annotation_readiness_status.json`
- `annotation_codebook_v2.md`
- `automatic_event_pilot_report.md`
- `independent_review.md`
- `red_team_findings.md`

## Overall Verdict

RQ012 supports Wave-A annotation readiness, not completed annotation evidence.
It is ready to register as a blinded, outcome-free event-labeling package, while
remaining blocked for real human labels, agreement statistics, and any
event-IPV or event-outcome association. The final study review passes, but the
study status correctly remains `BLOCKED_FOR_HUMAN_LABELS`.

Paper-safe phrasing:

> The OnSite event package establishes a blinded annotation design and
> extractor-readiness checks for selected interaction events. It does not yet
> provide human labels, inter-annotator agreement, or validated associations
> between events, IPV, and outcomes.

## Claims That Can Be Carried Forward

1. Gates 012-0 and 012-1 pass. Gate 012-2 is text/surface-cleared, Gate 012-3
   is ready-pending-humans, and Gate 012B remains blocked.
2. No simulated real labels were introduced. The label templates are blank and
   `human_labels_present=false`.
3. The package does not compute or claim event-IPV, event-score, event-rank, or
   team-identity associations.
4. The codebook separates automatic events, human-only events, and removed
   events. Construct-proximal labels are secondary only, not primary endpoints.
5. The automatic pilot is useful as extractor/data-health evidence only. It
   reports computability, primary counts, precedence suppression, and
   sampling-rate sensitivity without behavioral interpretation.
6. Red-team extractor issues were materially addressed through pair-event
   timestamp prevalidation, actor-identity stability guards, and cross-event
   precedence for E01/E02, E02/E18, and E09/E15.
7. The 90_report and final review close the historical B11 report blocker, so
   the current package can be registered as readiness evidence despite older
   review files preserving the earlier blocker state.

## Claims To Reject Or Defer

- Do not claim behavioral validation, event-IPV association, event-outcome
  association, or event-rate generalization.
- Do not claim human annotation is complete. Two accepted independent labels,
  kappa, and AC1 agreement are still required.
- Do not treat automatic event counts as scientific outcome results. The pilot
  shows extractor stability and weaknesses only.
- Do not use construct-proximal labels such as aggressive intrusion,
  appropriate assertiveness, or over-yielding freeze as primary event-IPV
  endpoints.
- Do not start Gate 012B without final media/card issuance, auditor sign-off,
  human labels, agreement statistics, required upstream freezes, and explicit
  authorization.

## Quality And Compliance Notes

The final review passes all readiness checks: offline report availability,
resolved links and artifacts, figure provenance, no simulated labels, no
event-IPV association, and explicit blocked-for-human-labels boundary. The
independent review and red-team findings are valuable but earlier in the
lifecycle; their old blocker language should be interpreted through the final
review and the rendered 90_report.

Several housekeeping issues remain. `README.md` and `TRACEABILITY.md` still read
like scaffold/bootstrap files. Some status surfaces still say figure rendering
or media issuance is pending even though figures are rendered; the final neutral
media/card package and auditor sign-off remain separate dependencies.

## Knowledge-Layer Action

Recommended decision state: register RQ012 as Wave-A annotation readiness with
overall status `BLOCKED_FOR_HUMAN_LABELS`. Keep Gate 012B blocked until real
human labels, agreement metrics, final neutral media/card issuance, auditor
sign-off, required RQ freezes, and explicit authorization are all present.
