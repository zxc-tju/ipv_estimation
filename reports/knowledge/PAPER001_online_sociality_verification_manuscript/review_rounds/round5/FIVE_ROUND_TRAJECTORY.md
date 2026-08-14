# Five-round verdict trajectory — simulated NMI review of the social-monitoring manuscript

Sources: `../round1/AGGREGATION.md` … `AGGREGATION.md` (this directory); reviewer reports
`../roundN/review_codex_{A,B,C}.md`, `../roundN/review_claude_D.md`; revision logs
`../roundN/REVISION_LOG*.md`. Percentages are each reviewer's own acceptance probabilities:
as-submitted → assuming a competent major revision. Reviewers were blind to all prior rounds.

## Verdict table

| Round | Commit | A (social/planning) | B (formal methods) | C (statistics) | D (NMI editor) | Mean post-rev | Rejects |
|---|---|---|---|---|---|---|---|
| 1 | 486382e | 5 → 35, **Reject** | 5 → 40, **Reject** | 5 → 40, Major rev | 5 → 35, Major rev | 37.5 | 2 |
| 2 | 41b819a | 8 → 55, Major rev | 8 → 55, Major rev | 5 → 45, Major rev | 7 → 45, Major rev | 50.0 | 0 |
| 3 | a4f99f8 | 5 → 35, **Reject** | 5 → 45, Major rev | 5 → 45, Major rev | 5 → 60, Major rev | 46.25 | 1 |
| 4 | ad18609 | 10 → 45, Major rev | 5 → 45, Major rev | 5 → 45, Major rev | 8 → 55, Major rev | 47.5 | 0 |
| 5 | 985a757 | 8 → 38, **Reject** | 8 → 45, Major rev | 10 → 55, Major rev | 8 → 45, Major rev | 45.75 | 1 |

Round-over-round means (as-submitted / post-revision): 5.0/37.5 → 7.0/50.0 → 5.0/46.25 → 7.0/47.5 → 8.5/45.75.

## Reading (grounded in the round records)

1. The revision loop repaired every FACTUAL defect it was shown, and the repairs stuck: the round-1
   ">40% vs 21/20/8%" contradiction never recurred; the round-3 "certifies" verb was fixed in round
   4 (U03) and drew nothing in round 5; the round-4 LODO-caption error was fixed (U01) and round-5's
   editor cites the corrected numbers as evidence. Round 5 found zero arithmetic or factual breaks
   in the text and zero factual reviewer errors — both firsts.
2. What the loop could NOT move is exactly what the records predicted from round 1: the as-submitted
   probability is pinned at 5–10 by the PI-gated compliance block (ethics E5 — five unanimousish
   rounds; declarations/TOPS E8/E14) and by the C7-gated uncertainty-free headline (E2/E10). These
   need facts and the offline-server swap, not prose.
3. The post-revision mean plateaued at ~46–50 from round 2 onward. The residual mass is stable and
   identical across rounds 2–5: construct validity (B01), consequence identification (B03),
   cluster-aware conformal (B05), sequential semantics (B04/B13), crossed inference + human-arm
   funnel (E2/E10/B11/B12/B22), baselines (B02). These are new-evidence demands; text editing has
   reached its ceiling on them, which is why round-4/5 plans shrank to wording precision and
   frozen-number surfacing.
4. Reviewer A oscillates (Reject 35 → MR 55 → Reject 35 → MR 45 → Reject 38) on one axis — whether
   IPV atypicality may be called social compliance — while citing no fact the other rounds lacked;
   the round-3 record already diagnosed this as panel variance, not manuscript signal. B and C
   converged upward as disclosure improved (C ends at its five-round high, 55, explicitly crediting
   the freeze/audit discipline); D stays at Major revision throughout with the standards-desk block
   as the stated gate.
5. Reviewer-error incidence fell to zero: the ED1 "with with" phantom (asserted rounds 2–4, three
   reviewers at once in round 4, grep-refuted each time) was not asserted in round 5; the round-4
   fixes U02/U05/U06 each extinguished their round-4 objection lines.
6. Net position after five rounds: a Major-revision paper whose remaining levers, in order of
   probability impact per the reviews themselves, are (i) the E5/E8/E14 declarations package,
   (ii) C7 completion with the E2 presentation rule and B22 band, (iii) B01/B03-class new evidence.
   The text itself is now clean enough that no reviewer in the final round found a factual defect
   bigger than a mis-scoped adverb.
