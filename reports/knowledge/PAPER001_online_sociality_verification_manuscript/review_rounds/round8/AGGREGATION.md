# Round 8 aggregation — fact-checked findings

Panel: three Claude referees (no codex referee this round, on PI instruction), same remit structure
as rounds 6-7 so the rounds stay comparable. All three blind to rounds 1-7. Manuscript reviewed:
commit `1054581`, clean working tree, `main.tex` sha256 prefix `e54fa94181e816c9`, 40 pages, dated
2026-08-20. First round in which every human-arm value is measured; the watermark and the
placeholder wrappers are gone.

## Verdicts

| Referee | Remit | As submitted | After major revision | Recommendation |
|---|---|---|---|---|
| A | social interaction / interactive planning | 3% | 25% | **Major revision** |
| B | methodology + statistics + runtime verification | 4% | 32% | **Major revision** |
| C | journal generalist editor | 3% | 38% | **Major revision** |

Mean post-revision **31.7**, against 41.7 in round 7, 24.0 in round 6 and 37.5 / 50.0 / 46.25 /
47.5 / 45.75 in rounds 1-5.

**Three major revisions, no rejects — but the score fell 10 points from round 7.** The reason is not
a new defect introduced this week. All three referees, independently and with different remits,
converged on one structural question that no previous round had put this sharply: **does an
assertive flag measure behaviour, or does it measure the estimator hitting the edge of its own
candidate grid?** Every referee reached it by a different route (A by pixel-measuring the figure, B
by inverting the weighted mean, C by comparing per-frame movement with the width of the flag zone),
and each says explicitly that if the answer is "the grid", the results sections do not survive
rewriting. That is why "as submitted" collapsed to 3-4% while the framing itself continues to be
praised.

## Priority fact-checks (done before any disposition)

### FC1 — "an assertive flag is arithmetically a grid-saturation event" (A M1, B M1, C M1) — **PARTLY TRUE, AND OVERSTATED AS PUT**

This is the finding that decides the round, so it was checked against the frozen readings rather
than against the manuscript. Joining the 14,099 judgeable moments to the frozen band edges (zero
unmatched) and to the frame-level readings:

| population | n | median reading | at the floor (within 0.001 rad) | within 0.01 of either bound |
|---|---|---|---|---|
| within-range | 12,711 | +0.1092 | 0.53% | 1.78% |
| assertive-flagged | 519 | **−0.5941** | **11.75%** | 19.85% |
| accommodating-flagged | 869 | +0.3206 | 0.92% | 15.30% |

**What survives.** Floor readings are enriched **22-fold** among assertive flags relative to
within-range moments. That is real, it is undisclosed, and a referee is entitled to ask about it.

**What does not.** 88.25% of assertive flags are not at the floor, and the median assertive reading
sits 0.584 rad clear of it. The flag is not, as a population statement, a saturation event.

**Where the referees' arithmetic goes wrong.** C computes the assertive flag zone as
1.1781 − 1.03 = 0.148 rad, taking the median lower edge over *all* judgeable moments. But flags do
not occur at the median edge — they occur where the local edge is unusually high. At the moments
actually flagged assertive the local zone has median width **0.222 rad**, and 60.9% of them exceed
C's 0.148. More decisively: the median flagged reading sits **0.293 rad below its own local edge**,
so flags are not marginal crossings of a hairline. B's inversion (a reading of −1.03 requires
≥62.3% of the weight on the single most extreme candidate) is arithmetically correct but applies to
a moment at the median edge, which is not where flags happen.

**Honest position.** The saturation concern is legitimate and the paper is silent on it; the
strong form ("a flag *is* a saturation event") is refuted by the readings. This is answerable with a
disclosure, not with a retreat — but the disclosure has to include the 11.75%.

### FC2 — "the flags do not persist" (A M2, B M4, C M1) — **CONFIRMED, AND STRONGER THAN THE REFEREES SAY**

Computed on contiguous frame runs at 10 Hz:

| | assertive | accommodating |
|---|---|---|
| distinct flag runs | 376 | 646 |
| isolated single-frame runs | 320 (85.1% of runs) | 534 (82.7%) |
| share of flagged **moments** that are isolated single frames | **61.7%** | 61.4% |
| flagged moments in a run of ≥5 frames (0.5 s) | 17.7% | 12.4% |
| longest run | 12 frames | 11 frames |

Only **13 of the 120** scenario runs containing an assertive flag (10.8%) contain a contiguous
five-frame stretch.

**Correction to an earlier chair note, and a defect found in passing.** C attributes "20 of 120"
to the manuscript's own funnel, and C is right: the number is printed in the case-example figure's
funnel graphic (an earlier draft of this section said it was not in the manuscript; that was wrong,
the graphic had been checked only against the caption text). Traced to the screening code: alerts
are grouped into clusters with a **one-second gap tolerance** (a gap longer than 1.0 s starts a new
cluster), and the funnel counts runs whose largest cluster holds at least five alerts. The caption
describes a different criterion — "at least five flagged frames in one contiguous stretch" — and
strict frame-by-frame contiguity gives **13**, not 20. Reconstructing the one-second-tolerance rule
on the recomputed moment table returns 21, one off the printed 20, consistent with the screening
file's alert definition differing slightly from the recomputed table's.

So the number is traceable and correct under its real criterion, and the **caption mis-describes
that criterion**. Repair: restate the caption as the two-second clustering window. Note also that
under either reading the substantive point stands and is understated by C — flags do not persist.

### FC3 — "the stated operating point implies a ~99.5% run-level alarm rate" (B M4) — **FALSE**

B derives this by treating per-moment flags as independent across the moments in a run. Measured
directly: of the 231 scenario runs carrying judgeable moments, **157 (68.0%)** contain at least one
flag of either side and **120 (51.9%)** contain at least one assertive flag. Flags cluster within
runs (376 assertive runs of flags fall inside only 120 scenario runs), so the independence
calculation overstates it by a wide margin. The underlying point — that a per-moment operating point
translates into a much larger run-level alarm rate, and the paper never states the run-level number
— stands, and 68% is itself worth printing.

### FC4 — "the audit is reported one-sidedly: the AVs sit at the native rate and the human arm under-flags twofold" (A M5, B M6, C M2) — **CONFIRMED, EXACTLY**

Recomputed from the printed counts:

| level | AV rate | ÷ native | matched humans | ÷ native | AV ÷ humans |
|---|---|---|---|---|---|
| 80% | 19.49% | **0.98** | 9.97% | **0.50** | 1.96 |
| 90% | 9.84% | **1.01** | 5.04% | **0.52** | 1.95 |
| 95% | 4.97% | **1.12** | 2.28% | **0.52** | 2.18 |

The automated systems sit at the reference population's own alarm rate at all three levels. The
matched human drivers sit at almost exactly half of it at all three levels, and the paper's own
interval on the human rate (3.4-5.5%) excludes the 9.72% native rate. The "twice as often" is
produced by the human arm's halving, not by the machines being flagged more than the reference
predicts. Section 2.5 states the three-rate rule correctly and the Abstract then tells the story
with exactly one pair; the figure prints "AV = 2.0× humans" as the panel headline and leaves the
AV-versus-reference parity unlabelled. The ratio carries no interval anywhere.

This is round 7's finding 3 returning in a much sharper form, and it is now the panel's second
unanimous item.

### FC5 — "the human arm's headline ego endpoint carries no interval, and the figure's own convention then reads as a failure to replicate" (B M8) — **CONFIRMED**

In the shared-signature panel an asterisk marks a ratio whose interval excludes parity. The two
ego-margin rows (median and upper quartile) are drawn with no interval and no asterisk for either
arm, because no interval was computed for them in either arm's battery. The text nonetheless asserts
that "the same consequence signature appears", and the upper quartile is the endpoint that carries
the assertive-side result. A reader applying the caption's own rule sees the load-bearing endpoint
unmarked and cannot tell "interval admits parity" from "no interval computed".

### FC7 — "the benchmark is a harder corpus, so its flag rate is not comparable to natural driving" (PI question, 2026-08-20) — **CONFIRMED AND QUANTIFIED**

Raised by the PI rather than by a referee, and it bears directly on FC4. Measured:

- **Difficulty appears first as abstention, not as false alarms.** The human-support gate passes
  32.32% of benchmark candidate moments against a 5.08% abstention rate on the natural test fold —
  the benchmark abstains **13.3x more often**. The monitor's designed response to an unfamiliar
  situation is to decline to judge, and that is what it does.
- **The two corpora have very different situation mixes.** Total-variation distance between the
  benchmark's judgeable moments and the human reference pool is **82.1 percentage points**. The
  benchmark is 63.3% one cell (following geometry, priority role) which is 1.9% of the human pool;
  the human pool is 90.3% two merging-geometry cells which are 12.0% of the benchmark.
- **The mix inflates the benchmark's flag rate by about two points.** The cells over-represented in
  the benchmark are also the higher-flagging ones (11.16% in the dominant cell against 6.98-9.42%
  in the human pool's dominant cells). Reweighting the benchmark's own judgeable moments to the
  human pool's situation mix moves the automated-systems flag rate from **9.84% to 7.79%**.

**Consequence for FC4.** Composition-matched, the automated systems sit *below* the instrument's
native alarm rate (7.79% against 9.72%), not at it. This strengthens the apparatus-validity claim
rather than weakening it. It does **not** license printing 7.79% beside the human arm's 5.04%: the
human arm has no situation-cell breakdown in the entered data, so the same correction cannot yet be
made for it, and a matched number must not be set against an unmatched one. The cell composition of
the human arm's judgeable moments is a bounded ask on the archival moment-level table.

**Also note which comparison is difficulty-matched.** The within-benchmark automated-versus-human
ratio is matched by construction — same 15 scenarios, same counterparts, same instrument. The
cross-corpus comparison is the unmatched one. The referees' framing (that the ratio is the weak
claim and the native comparison the strong one) is exactly inverted on this axis, and the paper
should say which comparison controls what.

### FC6 — the chair's own carry-over check — **the −1.03 rad provenance gap is CLOSED**

Recorded separately in `CHAIR_FINDINGS.md` (C1). The number recomputes to −1.0264 on exactly the
population the sentence names, with 100% of moments and 100% of situation cells negative. Two
rounds had recorded it as unrecoverable; the evidence was split across two frozen tables sharing a
key. No manuscript change.

## Converging substantive findings

1. **Estimator behaviour at the flag boundary is undisclosed.** (A M1, B M1, C M1 — unanimous.) The
   strong form is refuted (FC1) but the paper says nothing about floor enrichment, per-frame
   movement relative to the flag zone, or grid-width sensitivity. This is the item that sets the
   ceiling this round.
2. **The audit is a two-sided calibration result reported one-sidedly.** (A M5, B M6, C M2 —
   unanimous, FC4 confirmed.) New this round because the measured human arm made the comparison
   real; in rounds 6-7 the same structure was masked by synthetic digits.
3. **Everything is per-moment and the persistence layer is unevaluated.** (A M2, B M4, C M1 —
   unanimous, FC2 confirmed.)
4. **Transfer: the leave-one-source-out result sits against the transfer claim.** (B M5, C M3.)
5. **The counterpart is one traffic-simulation model, disclosed only in Methods.** (A M7, C M4.)
6. **The partner-preference correlation is asserted and never quantified.** (A M4, C M6.) Carried
   over from rounds 2-7, unchanged.
7. **No construct validation of the measured quantity.** (A M3/M8, C M7.) Carried over unchanged;
   still the single largest limit on the ceiling.

## Items requiring a decision from the PI

1. **Whether to run and report the saturation analysis** (floor-proximity shares by arm and band,
   contiguous run-length distribution, and the headline rates recomputed under one pre-specified
   persistence rule), or to disclose the floor enrichment in Methods without recomputing. The
   panel is unanimous that some answer is required, and two referees say the paper does not survive
   if the answer comes back badly. The measurements in FC1-FC3 above are already done and can be
   reported as they stand; the persistence recomputation is not, and it is the one that could move
   the headline numbers.
2. **How to rebalance the headline comparison.** Options: state all three rates in the Abstract;
   make AV-versus-reference parity the primary calibration statement and the AV-versus-human ratio
   secondary; or keep the present framing and add the parity observation explicitly. Doing nothing
   is not viable — three referees independently called it the sentence the panel supports least.
   Note this is not a retreat: parity with the reference population at all three levels is a
   *stronger* calibration result than the ratio, and the ratio survives as a secondary statement.
3. **Whether to compute intervals for the two ego-margin rows in both arms** (which would let the
   figure's asterisk convention apply to the load-bearing endpoint) or to change the convention and
   say plainly which entries have no interval.
4. **Whether the human arm's apparatus and protocol asymmetries get a stated limitation**
   (head-mounted display versus perception interface; fixed non-counterbalanced scenario order;
   supervised, insured, instructed-to-drive-naturally participants), which two referees raise as
   competing explanations for the halving in FC4.

## Repairs that need no decision

- Print the run-level alarm rate (68.0% either side, 51.9% assertive, of 231 runs) wherever the
  per-moment rate is given as the operating point.
- Give the AV-versus-human ratio a confidence interval, or stop printing it as a bare bold headline.
- Two undefined denominators flagged independently by A and B: the readability pass rate (70.3%
  printed; A computes 71.21% from the printed components) and the support pass rate (32.32%).
  Either state the denominators or reconcile the numbers.
- The main text reports the passing permutation test and not the failing one; report both or neither
  (B M7).
- Record the −1.03 rad provenance (FC6) in the claims register so it is not re-raised.

## What did not recur

No referee re-raised the global-range self-contradiction, the attrition sentence, the displayed-window
choice or the "whole body" caption line. All four were repaired after round 7 and verified as
repaired at every occurrence before this round opened (`CHAIR_FINDINGS.md` C2). The round-7 process
rule — a repair means every occurrence, not the one the referee quoted — held.

---

# Dispositions — PI rulings 2026-08-20

The PI ruled on three of the four decision items above. Recorded here so the next round does not
re-open them.

## Ruling 1 — the saturation analysis is not run; the discreteness is a display problem

The PI's position: boundary saturation is not itself a meaningful question for this instrument, and
a false alarm carries little cost, because the monitor reports on social behaviour and is not an
intervention trigger. What is worth fixing is that a displayed reading flickers frame to frame.
Detection stays per frame; **filtering applies to the displayed value only**.

Implemented in the case-example figure: the reading panel now carries two traces from the same data
— the per-frame readings, which the verdicts are computed on, and those readings under a centred
2.1-second median, which is what a display would show. The same filter length the speed panel above
already uses. The smoothed trace is masked to readable frames so it never bridges an abstention, and
the flag markers stay on the raw readings. The caption states that filtering never touches a verdict.

Not done, deliberately: the persistence recomputation of the headline rates. Under this ruling a
persistence rule is a display choice, not a detection rule, so recomputing the headline under one
would misstate what the instrument does.

## Ruling 2 — the cross-corpus comparison stays, with the difficulty difference stated

The PI's position: keep it, but say plainly that the staged scenarios differ from natural driving and
are relatively harder, even though the drivers were told to drive as they normally would.

Implemented in Results: the passage now says the two settings are not interchangeable, that the
scenario set is staged conflicts selected for a benchmark and harder than what a naturalistic corpus
mostly contains, that the cross-corpus reading is therefore an alarm-inflation check rather than a
comparison of levels, and that the comparison of levels is the within-course one, where both kinds of
driver meet the same fifteen scenarios and the same counterparts under the same control.

No new number was printed. The composition-matched flag rate measured during the fact-check
(7.79% after reweighting the automated arm to the natural-driving situation mix) stays out of the
manuscript: it is a new number and has not been through the decision process, and it must never be
printed beside the unmatched 5.04% human rate.

## Ruling 3 — the ego-margin intervals are added

Implemented in the human-arm figure, panel b. The automated-arm ego-margin rows now carry the
case-bootstrap intervals from the frozen three-second-window battery (median ratio 0.9014,
[0.7626, 1.1072], which admits parity and takes no asterisk; upper quartile 0.6009, [0.4362, 0.8308],
which excludes parity and takes one). The two human ego-margin entries have no interval in the frozen
battery and are now labelled as such on the panel, so a bare marker is no longer read as an interval
that admits parity. The caption states the convention, including that the two emergency-tail rows
carry an interval on the difference in shares rather than on the ratio.

## Still open — no ruling yet

- **Rebalancing the headline comparison** (decision item 2 above). Three referees independently
  called the "twice as often" sentence the one the panel supports least. Doing nothing was judged
  not viable by the panel; no decision has been taken.
- **A stated limitation for the human arm's apparatus and protocol asymmetries** (decision item 4).
- The "repairs that need no decision" list is unimplemented. Two of its items add or remove a printed
  number in the manuscript and therefore need a ruling despite the heading: printing the run-level
  alarm rate, and either giving the automated-to-human ratio an interval or not printing it bare.

## Repairs made without a ruling

Two accuracy repairs and one legibility repair, none of which change a claim:

- **The case figure's selection criterion was mis-stated.** The caption said the displayed run needed
  five flagged frames "in one contiguous stretch". The screening code clusters flags with a
  one-second gap tolerance, so a stretch with short gaps qualifies. The caption now states the real
  rule.
- **The human-arm figure hid part of its own evidence.** Panel b's bottom row label was composed
  wide enough to overlap the side counts printed under panel a, obscuring digits the caption sends
  readers to. The row label is now plain and the braking threshold moved into the caption.
- The caption for the case figure now describes both traces in the reading panel.

---

# Dispositions, part two — PI rulings 2026-08-20 (second sitting)

The PI first challenged the baseline the analysis rested on: both arms must be judged against the
human corpus, not against a standard that could have been shaped by automated-vehicle behaviour.
Verified before anything else was changed — the reference pool holds 2,442,625 rows and every one is
a human-vehicle-to-human-vehicle pair, the columns recording whether an automated vehicle is present
and which dataset a row came from are excluded from the model's inputs, and both arms were scored
with the same frozen envelope. The Results now state this invariant, so that the on-course human arm
cannot be read as a second yardstick.

## Ruling on decision item 2 — the headline keeps "twice", and gains an interval

Measured for this ruling: resampling whole scenarios (20,000 draws), the automated-to-human ratio is
**1.95, 95% CI [1.62, 2.39]**, above parity in every draw. The same resampling shows the human arm
below the corpus rate in every draw and the automated arm below it in 45.9% of draws. The panel's
preferred reframing — promote "the automated arm is at parity with the reference population" — would
therefore have made the *unstable* quantity the primary claim. It was declined on that evidence.

What the manuscript now says instead: the flag rate is a property of the scenario set (1.7% to 12.4%
across the fifteen for humans, 2.7% to 17.4% for the automated systems), an absolute alarm rate is
not portable, and that is why the audit carries its own human control arm. The corpus rate keeps one
job — checking the instrument does not fire more once carried somewhere new.

Two explanations for the human arm's sub-nominal rate were tested and do not hold: the range is not
wider in these situations (it is slightly narrower, 1.811 rad against 1.897), and the near-disjoint
situation mix cannot arithmetically produce a halving because per-cell coverage is close to nominal
everywhere that carries data.

## Ruling on the "repairs that need no decision" list

The per-run alarm rate is printed once, in Methods: 157 of 231 runs (68.0%) contain a flagged moment
on either side, 120 (51.9%) on the assertive side, with a note that the per-moment rate is the
operating point. This also forecloses the referee estimate of roughly 99.5%, which FC3 measured as
false.

## Ruling on decision item 4 — apparatus asymmetry goes to limitations only

Not into any claim. Written as a design statement rather than a concession: the fixed scenario order
is what makes the arm a matched control, and the apparatus produces the vehicle motion the monitor
reads directly rather than in simulation. Drafted, not yet inserted.

## Figure decluttering (PI raised this himself)

The PI judged the figure set cluttered with annotation that belongs in captions or main text. All
eight generators were re-run unmodified and pixel-compared against the shipped figures first — all
eight identical. Multi-word text blocks across the set fell from 170 to 145; the largest single
reduction removed sixteen interval strings printed beside error bars that already drew them.

One item was deliberately not removed. Panel a of the consequence figure carries three call-outs, and
two of them mark the unresolved median and the uncompressed lower quartile. The claims register
records those as deliberate — the median is to be printed and never cited as a compression result —
so removing them would have quietly retired an honesty marker. Left in place; flagged to the PI.
