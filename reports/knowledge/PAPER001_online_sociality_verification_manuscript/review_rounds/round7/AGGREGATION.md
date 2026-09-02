# Round 7 aggregation — fact-checked findings

Panel: three referees, same remit structure as round 6 so the rounds are comparable. All three
blind to rounds 1-6. Manuscript reviewed: base commit `5288590` plus uncommitted working-tree
changes (`main.tex` sha256 prefix `8536f31a630966ac`), 39 pages.

## Verdicts

| Referee | Remit | As submitted | After major revision | Recommendation |
|---|---|---|---|---|
| A (codex) | social interaction / interactive planning | 12% | 55% | **Major revision** |
| B (claude) | methodology + statistics + runtime verification | 5% | 35% | **Major revision** |
| C (claude) | journal generalist editor | 4% | 35% | **Major revision** |

Mean post-revision **41.7**, against 24.0 in round 6 and 37.5 / 50.0 / 46.25 / 47.5 / 45.75 in
rounds 1-5.

**Three major revisions, no rejects.** Round 6 was the first unanimous reject; this is a reversal of
it, and the post-revision mean recovers to roughly the rounds 1-5 band. Every referee now treats the
paper as fixable rather than foundationally unsound. Note that "as submitted" stays very low (4-12%)
— the panel is saying the framing is sound and the claims are overreaching, not that the work is
close to acceptable as written.

## Priority fact-checks (done before any disposition)

### FC1 — "the estimator is, in operation, a 7-point argmax" (B, M4) — **FALSE**

This is round 6's withdrawn F1 returning in a much stronger form, and it had to be checked against
data rather than against the manuscript.

B's argument is not the round-6 one. B accepts that the reading is a weighted mean and argues that
the likelihood is so sharp that the mean collapses onto a single candidate in practice. The
arithmetic is correct: with sigma = 0.1 m, two candidates hold weights within a factor of two only
if their one-second rollouts differ by less than about 0.12 m RMS.

The premise is false. Measured on the concentration ledger of the onsite corpus (2,974 attempted
frames, candidate count 7), the effective number of candidates carrying weight is:

| statistic | value |
|---|---|
| median | **4.10 of 7** |
| mean | 4.40 |
| 25th percentile | 2.52 |
| 75th percentile | 6.99 |
| frames effectively on one candidate (< 1.05) | **5.9%** |
| frames with two or more candidates carrying weight (>= 2.0) | **82.5%** |

In the median frame roughly four of the seven candidates carry real weight. The estimator is not
argmax-like in practice; it is the opposite. B's headline for M4 falls.

**Scope limit, stated because it matters:** this ledger is the onsite corpus, not the benchmark
corpus that produces the headline numbers. The same measurement has not been made on the benchmark
frames. The refutation is strong but not corpus-matched, and the honest position is that the
concentration measured where we can measure it contradicts the claim by a wide margin.

**What survives from M4:** two subsidiary points, both legitimate and both unanswered. (i) At the
median situation the assertive flag region is 0.148 rad wide against a floor of -1.1781 rad, so an
assertive flag there requires weight >= 0.623 on the single most assertive candidate; that
arithmetic is right, and it means assertive flags are structurally hard to produce in typical
situations. (ii) The reading moves about 0.30 rad per frame where it is most readable, which is
twice that flag-region width, and no result uses the persistence layer. Neither is refuted by FC1,
and both deserve an answer.

**Method note, third round running:** a referee's arithmetic being correct says nothing about
whether the premise connecting it to the mechanism is correct. FC1 was settled by measuring, not by
re-deriving.

### FC2 — "compression appears on the assertive side only" is not supported (B M2, C M2 and M8) — **CONFIRMED**

Both referees reached this independently, and they are right. There is no between-side test
anywhere in the paper. On the endpoint where the two sides can be compared directly:

- accommodating ego-margin median ratio 0.81x, interval [0.70, 1.15]
- assertive ego-margin median ratio 0.751x

The accommodating interval **contains the assertive point estimate**, so these data do not
distinguish the sides on that endpoint. The panel title asserts that they do.

The caption sentence "the accommodating distribution tracks the within-range one across its whole
body" is also overstated: the accommodating quartile ratios are 1.07x, 0.81x and 0.96x, so the
median sits 19% below within-range, which is not "tracking".

Worse for the title, on the short-margin share endpoints all four accommodating differences exclude
zero (-1.07, -2.64, -3.49, -4.74 percentage points) while the assertive side excludes zero at three
of four. On that endpoint family the accommodating side is the **better**-supported arm. The
differences are negative — fewer short margins — so this is not compression; but it is a detected
side-specific effect pointing the other way, and the title claims the accommodating side shows
nothing.

**This is a defect introduced by this week's own revision.** Before the accommodating side was
computed, the paper made no side-specificity claim. Adding the data was right; the claim attached to
it went further than the data. The evidence is unchanged and fine — the wording is what fails.

### FC3 — the attrition sentence is now wrong (B, M2) — **CONFIRMED**

Methods states "the undefined fraction is similar in the two groups (9.1% of flagged vs 8.2% of
within-range moments), so this exclusion does not preferentially remove either group." Recomputed
from the frozen counts:

| group | flagged total | with a defined margin | dropped |
|---|---|---|---|
| assertive | 519 | 472 | 47 = **9.06%** |
| accommodating | 869 | 747 | 122 = **14.04%** |
| both sides ("flagged" as the main text defines it) | 1,388 | 1,219 | 169 = **12.18%** |
| within-range | 12,711 | 11,669 | 1,042 = **8.20%** |

The sentence is a survival from the assertive-only era. Under the main text's own definition of
"flagged" the figure is 12.2%, not 9.1%, and the accommodating arm loses 1.7x the within-range rate.
Since the margin is undefined where the pair is not closing, the exclusion removes preferentially
from exactly the group whose defining behaviour is yielding — the direction that biases the
accommodating arm toward the within-range distribution, which is the direction that flatters the
"assertive side only" claim.

### FC4 — the displayed ego-side window is the non-primary one (B, M1) — **CONFIRMED**

Methods define the ego-side outcome over "the post-verdict window, which runs from the verdict to
the end of that run's evaluated window", i.e. the open-ended contract window. The same section then
states that "the fixed three-second window is the pre-specified primary, and its case-clustered
intervals exclude zero at all three levels, while the open-ended contract-window interval crosses
zero at the 90% level ([-2.6100, +0.1372])" — and 90% is the level at which verdicts are issued.

So the figure displays the ego-side endpoint on the window the paper's own sensitivity analysis
identifies as the weaker one, and the abstract leads with it. The two headline ego numbers
(median -24.9%, upper quartile 47.4%) additionally carry no interval in the figure; a median-ratio
interval exists in Methods and does exclude the null, but the upper-quartile figure has no interval
anywhere.

### FC5 — the round-6 self-contradiction was only half repaired (chair, independently C M6) — **CONFIRMED**

Round 6 recorded F3 as repaired. The repair reached the figure caption and not the Results body.
Both are still in the paper: the body says the global range at the strictest level "can never flag a
moment", the caption of the same figure says "it still flags, but only where the reading sits at the
very edge of what the estimator can express". The caption's version is the correct one and is
consistent with the 2.8-point over-coverage printed beside it. Referee C found this without seeing
round 6.

Second time a round-6 item has been logged as repaired when only one occurrence was edited. Repair
means grepping every occurrence — body, caption, Methods.

### FC6 — the "120 runs" in the case-example caption — **RESOLVED: correct, and so is the 108**

Recomputed from the study's own analysis table, rebuilt with its own functions:

| population | moments | distinct scenario runs |
|---|---|---|
| all assertive-flagged moments | 519 | **120** |
| assertive-flagged moments with a defined post-verdict margin | 472 | **108** |
| any flagged moment, either side, all moments | 1,388 | 157 |

Both printed numbers are right. The case-example caption's 120 counts runs containing any of the
519 assertive moments; the consequence caption's 108 counts runs containing one of the 472 that
survive the defined-margin filter, and that caption states the 472 explicitly, so the two are
distinguishable as printed. No manuscript change needed. The provenance is now recorded here so the
pair is not re-raised as an inconsistency.

Note this also retires the earlier working assumption that no definition yielded 120: the run count
for the pre-filter population was never computed, because the frozen figure table contains only
post-filter rows. Rebuilding the analysis set upstream settles it.

## Converging substantive findings

1. **The consequence claims outrun their intervals.** (B M1, C M1, both confirmed above.) The
   headline ego-side numbers have no interval in the display, sit on the non-primary window, and the
   abstract states them as fact.
2. **Side-specificity is asserted without a between-side test.** (B M2, C M2/M8, confirmed above.)
   This is the new exposure created by this week's revision.
3. **The "twice as often" headline is carried by the accommodating side.** (B M3, C M3.) Round 6
   raised this as F4; the decomposition is unchanged (roughly four-fifths of the gap sits on the
   accommodating side) and neither the abstract nor the Discussion says so. Computing the
   accommodating battery answered the evidence gap but did not change the narrative exposure,
   because the answer came back negative.
4. **No construct validation of the measured quantity.** (A M1, C M7.) Unchanged from rounds 2-6;
   still the item that most limits the ceiling.
5. **The audited human population is not the reference population.** (A M4, C M4.) Different
   country, apparatus and task.

Items 3-5 are long-running demands for new evidence. Items 1-2 are wording defects fixable now, and
item 2 was introduced this week.

## Items requiring a decision from the PI

1. **The side-specificity claim has to come down** to what the data support. Options: drop "only"
   and state that the interval-supported counterpart-side compression appears on the assertive side
   while the ego-margin endpoints do not distinguish the sides; or run and report an explicit
   between-side test. My recommendation is the first, because the second will very likely return "no
   difference" and would then have to be reported.
2. **Whether to move the ego panel to the pre-specified three-second window**, with the open-ended
   window as a sensitivity panel, or to keep the display and state the window explicitly with its
   interval. The first is what the referees ask for and is more defensible; it changes a figure.
3. **Whether the accommodating-side attrition asymmetry gets a sensitivity analysis** (retaining
   non-closing moments, censored at window length) or only an accurate disclosure sentence.

## Repairs that need no decision

- The half-repaired global-range contradiction (FC5): delete the body clause. The argument does not
  rest on it.
- The attrition sentence (FC3): restate with all three groups and the correct denominators.
- The "tracks across its whole body" caption sentence (FC2): replace with what the quartiles show.
- The unverified run count (FC6): either trace it or restate it against the population that can be
  recomputed.
