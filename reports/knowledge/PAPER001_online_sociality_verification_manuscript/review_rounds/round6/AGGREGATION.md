# Round 6 aggregation — fact-checked findings

Panel: three referees (previous rounds used four; the formal-methods and statistics remits were
merged into one technical referee). All three blind to rounds 1-5. Manuscript reviewed: paper
commit `4e3b2de`, 38 pages.

## Verdicts

| Referee | Remit | As submitted | After major revision | Recommendation |
|---|---|---|---|---|
| A (codex) | social interaction / interactive planning | 3% | 25% | **Reject** |
| B (claude) | methodology + statistics + runtime verification | 3% | 25% | **Reject** |
| C (claude) | journal generalist editor | 4% | 22% | **Reject** |

Mean post-revision **24.0**, against 37.5 / 50.0 / 46.25 / 47.5 / 45.75 in rounds 1-5. Three
rejects out of three; the previous maximum was two out of four, in round 1.

This is the first round in which all referees converge on reject, and the first in which the
post-revision estimate falls below every earlier round. Rounds 1-5 were reviewed by four referees
and this one by three, so the means are not strictly comparable; the unanimity and the size of the
drop are not explained by that difference.

## Priority fact-checks (done before any disposition)

### F1 — the monitored quantity is estimated on a seven-point grid, and in the median situation only one of those seven points can produce an assertive-side flag — **TRUE as arithmetic, and it follows from numbers the manuscript itself prints**

Methods state the candidate set is seven values, the multiples of pi/8 from -3pi/8 to +3pi/8, i.e.
-1.1781, -0.7854, -0.3927, 0, +0.3927, +0.7854, +1.1781 rad. The Methods also state that at the 90%
level the lower edge of the reference range is negative in essentially every situation, with median
-1.03 rad. Only one candidate, -1.1781, lies below -1.03. So in the median situation an
assertive-side flag is possible only when the estimate lands on the single most extreme candidate.

### F2 — "every flagged reading sits pinned at the grid extreme" — **FALSE, refuted by the manuscript's own Extended Data figure**

Referee B's stronger formulation does not survive. The reference-band Extended Data figure shows a
flagged reading at -0.39 rad -- three grid steps away from the extreme -- and that flag is the
annotated example the panel is built around: the same reading is inside a 2.19-rad band earlier in
the interaction and outside a 1.49-rad band later. Flags therefore do occur away from the boundary
when the band narrows, which is the panel's entire point.

**Disposition: F1 stands, F2 falls.** The residual concern is real but unquantified: nobody has
measured what fraction of assertive-side flags sit on the extreme candidate. Acting on F2 as stated
would have meant "fixing" a defect the paper already disproves in its own display items.

### F3 — the figure that establishes the reference range contradicts itself — **TRUE**

The Results text states that at the 95% level the global range spans 100.00% of the admissible scale
"and so can never flag anything". The same figure's caption states that the global range over-covers
by 2.8 points at 95%, i.e. achieves 97.8% coverage rather than 100%. A range that contains every
attainable reading has coverage 100.00% by construction. Both statements cannot be true.

The likely reconciliation is that "spans 100.00% of the admissible range" describes the range's
WIDTH as a fraction of the candidate span, not its containing every candidate: a range of width
2.356 rad positioned off-centre inside the wider declared domain can have 100% relative width and
still exclude the topmost candidate. If that is the intended meaning, the clause "and so can never
flag anything" is simply wrong as written and must go.

### F4 — the headline ratio is driven by the over-yielding side, not the assertive side — **TRUE**

Recomputed from the side counts and judgeable totals printed in the human-arm figure, 90% level:

- humans, 15,102 judgeable: 391 assertive (2.59%), 322 accommodating (2.13%)
- machines, 14,099 judgeable: 519 assertive (3.68%), 869 accommodating (6.16%)
- overall 2.09x (the manuscript prints 2.1x)
- **assertive side 1.42x; accommodating side 2.89x**

The manuscript's consequence evidence concerns the assertive side. The side on which machines most
exceed humans is the side for which no consequence signature is reported. A referee can read this
decomposition straight off the figure the authors added this week, and two of three did.

Note for interpretation, not a defence: the human-side counts are synthetic placeholders and the
decomposition may change when measured values arrive. The referees were instructed to treat them as
final, correctly. But the structural exposure -- a headline ratio whose mass sits on the side with no
consequence story -- is a property of the narrative, not of the placeholder values, and it will need
an answer whatever the measurements say.

### F5 — provenance gap found while checking F1

The knowledge layer holds no record of the reference range's edge distribution, and the printed
median lower edge of -1.03 rad has no traceable source anywhere under `reports/knowledge`. It is a
number in the manuscript that cannot currently be recomputed from the frozen evidence. This is a
claims-discipline defect independent of anything the referees raised, and it blocks quantifying F1.

## Converging substantive findings across the three referees

1. **Consequence evidence sits in the wrong part of the outcome distribution.** (C, with B
   concurring.) The separation between flagged and within-range moments appears at the median and
   upper quartile of post-verdict time-to-collision, while the two groups are indistinguishable in
   the lower tail and genuine emergencies are *rarer* at flagged moments. The paper says this
   plainly; the referees judge that saying it plainly does not make it support the claim being made.
2. **The counterpart is simulation software, and that is disclosed only in Methods.** (C.) Half the
   consequence battery is measured on the other vehicle's response; that vehicle is traffic-
   microsimulation software.
3. **Measurement-equivalence of the human-versus-machine comparison.** (A and B.) Whether a
   difference between the two arms can mean anything other than the two populations having been
   observed through different apparatus.
4. **Entitlement to the word "social".** (A.) Whether a quantity fitted from a two-agent interaction
   model measures social preference at all, and whether deviation from a human range may be called
   social atypicality.

Items 1-4 are all restatements of demands recorded in rounds 2-5 as needing new evidence rather than
rewriting. Nothing in this round is a text defect of the kind rounds 1-4 fixed.

## Items requiring a decision from the PI

Recorded in ESCALATIONS.md for this round. In summary: whether the headline ratio survives the
side decomposition in F4; whether F3 is repaired by deleting the over-claim or by restating the
width statistic; and whether the F1/F5 pair is answered with a measurement or with a stated
limitation.

---

# CORRECTION, 2026-08-14 — F1 is FALSE. The estimator is not confined to the candidate values.

The PI directed me to read the estimation code rather than reason from the manuscript's description.
Two lines settle it. The likelihood weights over the seven candidates are normalised
(`weight = var / sum(var)`), and the reported estimate is their weighted average
(`subject.ipv = sum(ipv_range * ipv_weight)`), where the weights come from a Gaussian likelihood on
the distance between the observed trajectory and each candidate's simulated trajectory.

**The estimate is a posterior mean over the candidate set, and is therefore continuous on
[-3pi/8, +3pi/8].** Every value between -1.178 and -1.03 rad is attainable. The premise that a
reading must equal one of seven values is wrong, and with it the entire "only one candidate can
produce an assertive-side flag" argument.

F1 is withdrawn. It was recorded as TRUE above; that verdict was reached by checking the referee's
ARITHMETIC (only one candidate lies below -1.03, which is true) without checking the referee's
PREMISE (that the estimate can only equal a candidate, which is false). The candidate grid is the
support of the likelihood, not the range of the estimator.

This voids referee B's stated single most serious weakness, and with F2 already refuted by the
manuscript's own display item, the whole grid-saturation line of attack in round 6 is empty.

**Method note for future rounds, and the reason this nearly cost a revision cycle:** when a referee
derives a conclusion from two numbers printed in the manuscript, verifying that the derivation is
arithmetically sound is not verification. The premise connecting those numbers to the mechanism must
be checked against the implementation. The manuscript describes the candidate set without stating
that the estimate is their weighted mean, which is what let a careful referee — and then me — read it
as a discrete estimator. **That is a manuscript defect worth fixing on its own: the Methods should
state plainly that the reported value is a likelihood-weighted mean over the candidate set and is
continuous.** Two independent readers made the same wrong inference from the current wording.

F5 (no traceable source for the printed median lower edge of -1.03 rad) still stands as a
claims-discipline item, but it is no longer load-bearing for anything.

# F3 — repaired

The self-contradiction is fixed in the manuscript. The clause asserting that a global range "can
never flag anything" is gone; the passage now states that at the strictest level the global range's
width equals the whole admissible span, so it retains almost no power to discriminate and pays for
that width by over-covering. The argument for conditioning survives intact and the figure caption no
longer contradicts the body. Manuscript recompiles: 38 pages, no errors.

# F4 — PI ruling received, and an evidence gap that must be closed before it can be written

PI ruling: margin compression is NOT a property of the assertive side specifically. Mutual yielding
that ends in deadlock compresses the interaction just as an aggressive merge does. The claim the data
supports is therefore "**deviation** compresses the interaction", not "**assertiveness** compresses
the interaction". No retreat from the headline ratio is needed; the wording is what changes.

This is the right frame and it removes the exposure. **But the frozen evidence cannot currently
support it.** Of the seven outcome measures in the consequence battery, six were computed for the
assertive side and the within-range group only; only the emergency-rate measure has an
accommodating-side value. To state that deviation on either side compresses the interaction, the
battery has to be recomputed on the accommodating side.

The one measure that does have both sides points the PI's way: emergency rate is 8.84% within range,
4.66% on the assertive side and 5.35% on the accommodating side — both deviation sides sit below the
within-range group, and the two are close to each other. That is consistent with a side-symmetric
effect, and is not evidence for one.
