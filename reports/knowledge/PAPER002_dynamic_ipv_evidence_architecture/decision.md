# PAPER002 Architecture Decision Ledger

Status: `writing` — pointer ledger, not a claim ledger. This tracks **which accepted RQ claims enter the v4.1 manuscript spine**. Each row's authority is the cited RQ `decision.md`; this file freezes no new claims.

## Claims admitted to the v4.1 spine

| Spine element | Source claim (canonical) | Manuscript framing |
|---|---|---|
| Estimability gate | RQ007 `decision.md` | descriptive, proximity-bounded caveat |
| Context-conditioned dynamic envelope (R3) | `RQ009-KC-R3` | empirical runtime monitor; context-conditioned only |
| OnSite matched-scenario universe | `RQ011-KC-UNIT/-OUTCOME-300/-REPLAY-285` | readiness/scope; no run-level claims; T19 replay-excluded |
| OnSite consequence reference | `RQ012-KC-AUTOEVENTS` | automatic events + official outcomes; no human-judgment leg |
| WOD-E2E preference path | RQ010 `decision.md` (feasibility) | `\externalpending{R4}` until RQ010B delivers |

## Explicitly excluded from manuscript claims

- IPV-conditioning channels (RQ009 M3/M4) — null internal ablations.
- Any RQ008 temporal-IPV motif as a behavioural law.
- OnSite human blind-annotation convergent leg (RQ012 deprecated).
- WOD-E2E as a completed human-preference validation (pending RQ010B).

## Open dependencies

RQ010B execution (human-preference leg) and RQ013 (beyond-safety increment) are not yet accepted; the spine carries them as pending. Update this ledger only when a constituent RQ `decision.md` changes.

Sources: constituent RQ `decision.md` files; `reviews/chatgpt_review_wave_b.md`.
