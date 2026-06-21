# Tried / Null / Negative / Failed Results Ledger

WIP. No analyses were run in Phase 0A.

## 2026-06-20 Phase 3 Analysis Freeze v2

Worker: `RQ003_phase3_freeze_002`

Status: frozen specification written; no predictor-outcome analysis run.

Exploratory prior context from the allowed plan and user brief:

- Prior signed directional association, near-null absolute association, and safe-but-low-coordination patterns are treated as exploratory context only.
- Those prior patterns did not set thresholds, model classes, fold assignments, safe-subset definitions, or confirmatory pass/fail criteria in this freeze.
- Any later use must be labelled exploratory unless reproduced by the frozen confirmatory comparison after Gate 0 condition `G0R-COND-001` passes.

Rejected confirmatory spec choices:

- Rejected broad model ladders as confirmatory; the freeze allows only one confirmatory comparison.
- Rejected comprehensive score, area rank, and overall rank as confirmatory endpoints; all are exploratory.
- Rejected p90, max, onset, latency, gain, alternative windows, phase, and expanded safe-subset combinations as confirmatory; all are sensitivity or exploratory with FDR where applicable.
- Rejected full-window, observed-PET, realized-order, post-hoc-phase, future-frame, and outcome-tuned features for the confirmatory model.
- Rejected deriving scenario/family cell membership from score-joined tables; Phase 4 must stop if no outcome-free scenario/session source is available.
