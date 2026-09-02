# Chair findings — round 8

Findings reached by the chair while the referees were reviewing, from checking round-7 carry-overs.
These are NOT referee findings and must be attributed as chair findings in the aggregation. The
manuscript was deliberately left unmodified while the panel was reviewing.

Reviewed article: commit `1054581`, clean working tree, `main.tex` sha256 prefix `e54fa94181e816c9`,
40 pages, dated 2026-08-20.

## C1 — the long-standing provenance gap on the median lower band edge is CLOSED

Rounds 6 and 7 both recorded that the printed median lower edge of the reference range
(`-1.03` rad, Methods) had no traceable source and could not be recomputed from the frozen
evidence. It can. The frozen envelope scoring table for the benchmark carries per-row band edges at
all three levels (`lo_80/90/95`, `hi_*`, `width_*`) for all 67,861 candidate rows, and it joins to
the 14,099 judgeable moments on the row key with **zero unmatched rows**.

Recomputed on exactly the population the sentence names — the 14,099 judgeable moments at the 90%
level:

| quantity | recomputed | printed |
|---|---|---|
| median lower edge | **−1.0264 rad** | −1.03 rad |
| share of those moments whose lower edge is negative | **100.00%** | "negative in essentially every situation" |
| median of the per-situation-cell medians (8 cells) | −1.0177 rad | — |
| share of situation cells with a negative median lower edge | 100.00% | — |

The printed value rounds correctly and is computed on the population the sentence is about. Two
neighbouring populations give different values and are therefore excluded as the source: all scored
rows give −0.7635, gate-passing rows give −0.8877. Both the moment-level and the cell-level reading
of "in essentially every situation" hold at 100%, so the hedge is if anything weaker than the
evidence.

**Disposition: no manuscript change. Record the provenance so the item is not re-raised.** The
recomputation and its source are logged in the credentials section below.

**Process note.** Two rounds recorded this as unrecoverable. What was missing was not the evidence
but the join: the moment-level table carries verdicts and outcome windows but no band edges, and the
envelope table carries band edges but no verdicts. Neither file alone answers the question. Before
declaring a number untraceable, check whether it is split across two frozen tables that share a key.

## C2 — round-7 repairs verified as landed, at every occurrence

The round-7 rule was that a repair means editing every occurrence, not the one a referee quoted.
Checked by whole-text search on the submitted source:

- **Global-range self-contradiction (round 7 FC5).** The body clause asserting that the global range
  at the strictest level can never flag a moment is gone; zero occurrences of the phrase anywhere.
  The caption's correct version stands.
- **Attrition sentence (FC3).** Replaced. The manuscript now gives all three groups with their own
  numerators and denominators (3.9% assertive, 20/519; 6.0% accommodating, 52/869; 2.9%
  within-range, 367/12,711), states that the accommodating group loses most and why, states the
  direction of the resulting bias against the paper's own claim, and reports the retention
  sensitivity analysis (assertive upper-quartile contrast 0.626, [0.410, 0.829]).
- **Displayed ego window (FC4).** The fixed three-second window is now stated as the pre-specified
  primary and as the one displayed, with the shorter alternative window named as the anchor's own
  prediction-target horizon (median 0.60 s) and its crossing interval given.
- **"Whole body" caption sentence (FC2).** Gone; zero occurrences.

## C3 — new material entering review for the first time

The human reference arm is measured in this round. Rounds 6 and 7 reviewed it as a watermarked
synthetic target under a charter clause instructing referees to treat those values as final; that
clause is retired in the round-8 charter. Referees see: total flag rate 5.0% of judgeable moments
(786/15,598) against the 9.72% natural baseline, automated systems 2.0x the human rate, 15 of 15
scenarios above parity, and the consequence signature reproduced within humans on all six endpoints.
Any referee finding on these numbers is a first look, not a carry-over.

## Standing exposures not expected to move this round

Round 7's converging findings 3-5 are long-running demands for new evidence, not wording defects,
and nothing in this week's work addresses them: the "twice as often" headline is carried mostly by
the accommodating side and the narrative does not say so; there is no construct validation of the
measured quantity; and the audited human population differs from the reference population in
country, apparatus and task. Expect them again.

## Credentials for re-checking

- Band edges: `.codex-fleet/rq021-contemporaneous-envelope/work/E1/onsite_scoring_dryrun.parquet`
  (67,861 rows; columns `product_row_key`, `context_cell`, `lo_80/90/95`, `hi_*`, `width_*`,
  `mechanism2_gate_ok`).
- Verdicts and moment set: the round-7 recomputed moment table (14,099 rows; `product_row_key`,
  `band`), joined on `product_row_key`, zero unmatched.
- Manuscript sentence: Methods, the paragraph beginning "Of the 14,099 judgeable moments at the 90%
  level".
