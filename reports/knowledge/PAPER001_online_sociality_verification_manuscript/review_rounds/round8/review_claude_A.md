# Referee report — "Online monitoring of socially compliant autonomous driving"

**Referee A — social interaction, human behaviour and interactive motion planning**

*Note on method.* Where I quote a number that is printed in the manuscript I give the section or
figure. Where I read a value off a published figure I say so and give the axis calibration I used,
so that the authors can check my reading. All arithmetic I assert is shown.

---

## 1. Summary assessment

The paper reframes social compliance as an online membership test: is the ego's current
interaction preference value (IPV) inside a human reference range conditioned on the observable
situation, and if the reading is not identifiable, abstain. It delivers a frozen estimator, a
conformally calibrated conditional range (coverage within 0.6 pp of nominal), an abstention
taxonomy, a matched-scenario real-vehicle benchmark, and — genuinely unusual and creditable — an
audit arm in which twenty licensed humans drive the same scenarios under the same instrument.

The framing is good and the bookkeeping is unusually honest. My difficulty is with the measurement
underneath it. On the authors' own frozen configuration, a flag is arithmetically almost equivalent
to the estimator saturating at an endpoint of its seven-point candidate grid; the paper's own
worked example (Fig. 1c) has a reference band whose lower edge coincides with the estimator's
floor, and whose two flagged readings sit within ~0.01 rad of its ceiling. The reading also moves
0.30 rad between consecutive frames, three-quarters of one grid step. The consequence signature is
a shift of the *75th percentile* of minimum time-to-collision from ≈12.5 s to ≈7.7 s, plus a
0.4 m s⁻¹ speed change in a counterpart that is a traffic-simulation model responding to the ego.
The headline "twice as often" compares an automated arm sitting exactly at the instrument's native
alarm rate against a human arm sitting significantly below it.

*(198 words)*

---

## 2. Major weaknesses

### M1. A flag is, arithmetically, an estimator-saturation event rather than a graded social judgement

**[Methods 4.1 and 4.5; Fig. 4a; Fig. 1c; Fig. ED1; Fig. ED2c]**

**What is wrong.** The estimator is a weighted mean over seven candidate preferences
θ_k ∈ {−3,−2,−1,0,1,2,3} × π/8 (Methods 4.1), so the reading is confined to
[−3π/8, +3π/8] = [−1.178, +1.178] rad. Methods 4.5 states that on the benchmark, at the 90% level,
"the lower edge is negative in essentially every situation (median −1.03 rad)". Combining the two:

- To read θ̂ ≤ −1.03 rad, the most favourable allocation puts weight *a* on the extreme candidate
  −1.178 and the remainder on its neighbour −0.785:
  −1.178a − 0.785(1−a) ≤ −1.03 ⇒ 0.3927a ≥ 0.2446 ⇒ **a ≥ 0.623**.
  If the residual mass sits at θ = 0 instead, −1.178a ≤ −1.03 ⇒ **a ≥ 0.874**.

So an "assertive-side atypicality" verdict at the median benchmark situation *requires between 62%
and 87% of the entire posterior mass to sit on the single most extreme grid point*. At the 95%
level the position is starker: Fig. 4a gives the conditioned width as 92% of the admissible span,
so the excluded region totals 0.08 × 2.356 = 0.188 rad, about 0.094 rad per side — roughly a
quarter of one grid step (π/8 = 0.393 rad).

The figures confirm this is not a theoretical worry. In **Fig. 1c**, the paper's headline worked
example, I measured the panel against its own y-axis ticks (0.5 rad = 245 px; zero at y = 2503.5):
the 90% band's **lower edge sits at −1.179 rad**, i.e. exactly the smallest value the estimator can
produce, for the whole displayed "reading issued" stretch — an assertive flag was arithmetically
impossible anywhere in the example. The two salmon triangles labelled "outside the human reference
range (accommodating side)" sit at **≈ +1.17 rad**, within about 0.01 rad of the ceiling +1.178.
**Fig. ED1** and **Fig. ED2c** show the same thing: the flagged red triangles lie in a flat row at
the estimator floor, and the flags begin precisely where the band's lower edge lifts off that
floor.

A second, related point that the paper never confronts. Methods 4.1 defines each candidate as
optimising cos θ_k · (own-progress cost) + sin θ_k · (interaction cost). At the extreme assertive
candidate θ = −3π/8, sin θ = −0.924: the candidate does not merely ignore the counterpart, it is
*rewarded* for increasing the counterpart's cost. Indifference is at θ = 0. So the operational
content of "atypically assertive" is "best explained by a near-maximally adversarial objective" —
which is exactly the region a candidate family will fall back on when it cannot generate the
observed trajectory at all.

**Why it matters at this journal's level.** The paper's most quotable claim — "under this common
instrument the automated systems are flagged 2.0 times as often as the matched human drivers"
(Section 2.5) — then has an entirely non-social competing explanation: automated planners produce
trajectories that a human-tuned candidate generator fits worse, the likelihood collapses onto a
grid endpoint more often, and the monitor flags the misfit. That is a statement about the
estimator, not about how the vehicle treated another road user. NMI readers will ask whether the
monitored quantity measures sociality or model misspecification, and the manuscript as written
cannot answer.

**Remedy.** (i) Report the distribution of the maximum normalised candidate weight, separately for
flagged and within-range moments, for both arms; (ii) report what fraction of flagged readings lie
within one grid step of an endpoint; (iii) re-run with a denser and wider grid (e.g. 11 or 13
candidates spanning the declared domain [−π/2, π/2]) and show that flag rates and the AV/human
ratio are not driven by boundary censoring; (iv) report a goodness-of-fit statistic for the winning
candidate (e.g. the absolute MSE of the best candidate) and show that flagged moments are not
simply moments at which *no* candidate fits.

---

### M2. The reading is effectively discrete and highly unstable, and every result in the paper is a single-frame verdict

**[Methods 4.1; Fig. 2b; Fig. ED1; Fig. ED2 selection funnel; Methods 4.4]**

**What is wrong.** Methods 4.1 asserts that "the reading is their weighted mean θ̂_a = Σ_k w_k θ_k,
which varies continuously over [−3π/8, 3π/8]" and that "the reported reading is not one of the
seven candidates". Four independent pieces of the paper's own evidence contradict the practical
force of that claim.

1. The likelihood is ℓ ∝ exp{−MSE/(2σ²)} with **σ = 0.1 m**, i.e. exp{−50·MSE} with MSE in m².
   A 0.02 m² difference in fit between two candidates already produces a likelihood ratio of
   e ≈ 2.7. Wherever the candidates are distinguishable at all, this is a hard arg-min in
   everything but name.
2. **Fig. 2b**: the mean frame-to-frame movement of the reading where it is *most* readable is
   0.30 rad (ego) and 0.31 rad (counterpart) at 10 Hz. The grid spacing is π/8 = 0.3927 rad.
   0.30 / 0.3927 = **0.76** — the fingerprint of a reading that hops between grid nodes about
   three frames in four, not of a continuously varying quantity.
3. **Fig. ED1** chooses as its illustration two frames that "carry the same reading, −0.393 rad".
   −0.393 rad is **exactly −π/8**, a grid node. For a genuinely continuous estimator, finding two
   frames with identical readings to three decimals would be a coincidence.
4. **Fig. ED2** funnel: of 120 runs containing a flagged assertive moment, only **20 (16.7%)**
   contain five flagged frames in one contiguous stretch. In the great majority of flagged runs,
   half a second of sustained flagging does not exist.

Against this, Methods 4.4 states that the persistence layer "belongs to the deployment interface:
no result in this paper uses it, every reported verdict and flag rate is per-moment, and its
operating characteristics ... are not evaluated in this paper."

**Why it matters.** The paper is proposing a *runtime monitor*. The quantities that determine
whether such a monitor is usable — episode-level false-alarm rate, detection delay, alert duration
— are the ones explicitly not evaluated, and the paper's own numbers suggest the per-frame flag
stream is chatter. Fig. 2b is presented as a virtue ("readable does not mean settled"), but its
consequence for the membership test is never drawn: if the reading moves 0.30 rad per 100 ms and
the distance from the middle of the band to its edge is of the same order, a flag is a sample of
estimator jitter at the boundary.

**Remedy.** Report the run-level alert statistics under a persistence rule of k = 3, 5, 10
consecutive judgeable frames, for both arms; show whether the AV/human ratio and the consequence
signature survive; and either drop the continuity claim in Methods 4.1 or support it with the
empirical distribution of θ̂ (a histogram will settle it in one panel).

---

### M3. The consequence signature has a mundane kinematic reading, and the magnitudes do not support the behavioural gloss

**[Section 2.4; Fig. 5a,c; Methods 4.5]**

**What is wrong.** Three separate problems compound.

*(a) The magnitude.* Section 2.4 says "the upper quartile falls by two-fifths, while the lower
quartile ... is not compressed at all", and the paper prints the absolute values only for the
quartile where nothing happens (3.46 vs 2.99 s). I measured Fig. 5a against its own log axis
(10⁰ at x = 759 px, 10¹ at 1378, 10² at 1997; 0.75 at y = 435):

| quantile | within-range | assertive | ratio |
|---|---|---|---|
| lower quartile | 2.87–3.08 s | 3.41–3.57 s | ≈1.16 |
| median | 5.49–5.62 s | 4.68–5.08 s | ≈0.90 |
| **upper quartile** | **12.3–12.8 s** | **7.4–8.2 s** | **≈0.62** |

The lower-quartile and median readings reproduce the paper's printed 2.99 / 3.46 s and −9.9%, so
the calibration is sound. The headline result is therefore that the 75th percentile of the minimum
time-to-collision over the following three seconds falls from about **12.5 s to about 7.7 s**. The
caption itself explains what that region is: "frames in which the pair is not closing carry no
time-to-collision and do not enter the minimum, so very large values mark a window whose only
closing frames close very slowly." A minimum TTC of 7.7 s sustained across a three-second window is
not an interaction under pressure. Calling this "a compression of ordinary interaction quality" and
"a measurably tighter interaction for both vehicles" attaches a behavioural meaning to a change
that occurs entirely in the regime of no interaction pressure — and the paper's own Fig. 5b shows
that at every threshold where anything is resolved, flagged moments are *less* likely to be tight
(0.46–0.61×).

*(b) The counterpart side is not an independent convergence.* Section 2.4 argues: "The two sides
are measured from different sources and different quantities, so their agreement is a convergence
rather than a restatement". But Methods 4.5 states that "the counterpart vehicles are driven by
traffic-simulation software (TESS NG) and respond to what the ego vehicle does". The counterpart's
speed reduction is therefore a deterministic response of a car-following/gap-acceptance model to
the ego's closing kinematics — the same kinematics from which the IPV is read. A different log file
is not a different construct. And the effect sizes are +1.41 km h⁻¹ [+0.08, +3.37] of speed
reduction (0.39 m s⁻¹ over three seconds; the lower bound is 0.02 m s⁻¹) and +2.60 km h⁻¹ of speed
range. These are reported in the main text as "nearly twice" and "roughly twice" a near-zero
baseline.

*(c) The confounder control is undefined.* The only adjustment in the consequence analysis is
"Comparisons are made within the frozen situation cells" (Methods 4.5). The phrase "situation cell"
occurs exactly twice in the manuscript and is defined at neither occurrence, and the situation
description has four categorical descriptors and 22 numeric channels (Methods 4.3) — one cannot
form cells over 22 continuous channels. Whether closing rate and distance-to-conflict are actually
balanced between the flagged and within-range groups is therefore unknown to the reader.

*(d) The effect vanishes on the estimator's own horizon.* Methods 4.5: on "the anchor's own
prediction-target horizon, a median 0.60 s", the contrast "crosses zero at the 90% level
([−2.6100, +0.1372])". The signature appears only on a window five times longer than the horizon
the estimate is about.

**Why it matters.** This is the section that converts a descriptive monitor into something worth
publishing at NMI. If the flagged group simply contains more moments in which the pair is genuinely
closing, then every endpoint follows from vehicle dynamics: smaller huge-TTC tail, more counterpart
speed adjustment, more speed range. The paper offers no test that separates the social reading from
the kinematic one.

**Remedy.** Define the situation cells explicitly and report the balance of closing rate,
range-rate and distance-to-conflict across the three groups at the verdict moment; then report the
*incremental* value of the flag over a baseline that uses only the raw kinematics available at
that moment. Print the absolute upper-quartile values. Replace or supplement the counterpart
endpoints with a quantity that is not a mechanical function of the ego's closing rate.

---

### M4. The "no need to read hidden intention" claim rests on an unquantified correlation and a single unpowered null

**[Sections 2.3 and 3; Methods 4.3]**

**What is wrong.** The Discussion calls this "the most informative finding": "At the level of a
whole interaction the partners' preferences are correlated; yet online, conditioning the reference
range on the counterpart's inferred preference — or on the ego agent's own history — adds no
measurable value beyond the current situation."

- **The correlation is never quantified.** It is asserted five times (Sections 2.2, 2.3, 3, and
  Methods 4.3) — "The two interacting agents' preferences, moreover, are correlated at this event
  level" — and no coefficient, confidence interval, sign or n is given anywhere in the manuscript.
  If the event-level partner correlation is small, the "apparent paradox" that organises the paper
  does not exist.
- **The null is one metric on one frozen model.** Methods 4.3 gives "paired 90% interval-score
  difference −0.0002, case-clustered p = 0.86", with hyperparameters frozen (learning rate 0.06,
  72 boosting iterations, 31 leaves) — presumably chosen for the situation-only model. No power
  analysis is offered: what magnitude of sharpening would have been detectable?
- **An untested alternative explanation.** The counterpart-IPV channel is produced by the same
  estimator whose frame-to-frame movement is 0.31 rad (Fig. 2b). Classical attenuation predicts
  that a channel that noisy will add nothing regardless of whether the underlying quantity is
  informative. The paper's resolution — "the partner correlation is already carried by the shared,
  observable situation" — is one of at least two explanations, and the noisier one is not excluded.
- **The stated bound is not honoured.** Methods 4.3 says "Neither is interpreted as a mechanism".
  Section 2.3 then writes "resolves because the partner correlation is already carried by the
  shared, observable situation", and the Discussion concludes "a question about behaviour in a
  situation, not about inferred minds". That is a mechanistic interpretation of a null, in the main
  text, on the paper's flagship finding.

**Why it matters.** For a readership that works on interactive planning and theory-of-mind-style
inference, "you do not need to infer the other agent's intention" is a strong and consequential
claim. It cannot rest on p = 0.86 from one ablation on one interval score.

**Remedy.** Report the event-level partner correlation with a confidence interval and n. Report the
detectable-effect size for the ablation. Add an upper-bound variant: condition on an *offline,
smoothed* counterpart summary (or an oracle counterpart IPV), which is not available online but
bounds how much partner information there is to be had. Report the ablation on flag agreement, not
only on interval score. Then soften Section 2.3 and the Discussion to what the ablation supports.

---

### M5. Figure 6 does not carry Section 2.5, and the headline ratio is anchored on the arm that moved

**[Section 2.5; Fig. 6a,b; Methods 4.5, 4.6]**

**(a) Which arm is the anomalous one?** From Fig. 6a the three flag rates at each level are:

| level | natural driving (held-out) | matched humans | automated |
|---|---|---|---|
| 80% | 20.0% | 10.0% | 19.5% |
| 90% | 9.7% | 5.0% | 9.8% |
| 95% | 4.4% | 2.3% | 5.0% |

(These reconcile with the printed side counts: at 90%, humans 435+351 = 786, 786/15,598 = 5.04%;
automated 519+869 = 1,388, 1,388/14,099 = 9.84%; ratio 1.95.)

The automated arm sits at the instrument's native alarm rate at **all three** levels. The matched
human arm sits at **half** of it, and the panel's own interval — "90% human: 95% CI 3.4–5.5%" —
excludes the native 9.72%. So under this instrument the automated systems are statistically
indistinguishable from the natural-driving human population that defined the reference, and the
only arm that departs from the instrument's calibration is the twenty humans on the test track.
Section 2.5 does say "all three rates belong together, and no pair of them tells the story alone",
which I credit. But the Abstract says "automated systems flagged about twice as often" with no
anchor, and the Discussion mentions the ratio not at all. The Abstract's version is the one that
will be quoted, and it points the reader in the opposite direction from the panel.

**(b) The two arms are not apparatus-matched.** Methods 4.5: "the simulated counterparts reach an
automated system through its perception interface and a human driver through a head-mounted
display." A head-mounted display, a safety supervisor on board who can intervene at any time,
staged conflicts and a fixed scenario sequence all plausibly compress human behaviour toward the
conforming middle — which is precisely the direction of the observed human deficit. The shared
source shift is controlled by the design; this arm-specific difference is not.

**(c) Fig. 6b does not mark the endpoint the text leads with.** I analysed the marker positions
programmatically. Asterisks (the caption: "an asterisk marks a ratio whose interval excludes
parity") appear for **both** arms on rows 3–6 (ego emergency <2 s; counterpart speed reduction;
counterpart speed variation; counterpart braking), and for **neither** arm on rows 1–2 (ego margin
median; **ego margin upper quartile**). Horizontal intervals are drawn on only two of six rows
(4 and 5), yet rows 3 and 6 carry asterisks without drawn intervals — so intervals exist and are
simply not shown, and the absence of an asterisk on row 2 must be read as "admits parity".

Two consequences. First, Section 2.5 opens the shared-signature claim with "the ego margin's upper
quartile contracts by about three-tenths" — an endpoint that, by the figure's own convention, is
not resolved for the human arm. Second, the same AV quantity **is** resolved in Fig. 5a
(0.601 [0.436, 0.831], i.e. −39.9% [−16.9, −56.4]) and is **not** marked in Fig. 6b. These two
figures disagree about the paper's single most load-bearing endpoint, the one Methods 4.5 says
carries the side-specific statement ("the side-specific statement is made at the quartile that
supports it and withheld at the one that does not").

**Remedy.** Give a confidence interval on the AV/human ratio itself and on the AV arm's rate; state
in the Abstract that the automated rate coincides with the reference population's native rate;
reconcile the Fig. 5 / Fig. 6b intervals or explain the different resampling; draw all intervals;
add per-driver flag rates for the twenty drivers; and either run a subset of human drivers through
the perception-interface staging or bound the HMD's effect on driving style.

---

### M6. Section 2.2's context-dependence effects are smaller than an analysis choice the paper itself shows to be large

**[Section 2.2; Figs. 2c, 3a,b,c; Methods 4.4]**

**What is wrong.** Section 2.2 rests on the risk-gated reversal of the priority effect: +0.058 →
+0.001 → −0.034 rad, a total swing of 0.092 rad, computed from **episode means**. Fig. 2c, two
pages earlier, establishes that the three admissible episode-summary rules "disagree by 0.26 rad on
average and flip the sign of the summary in 7–22% of episodes", and that the median episode summary
moves from +0.00 to +0.19 rad depending on the rule. The paper reports robustness of the risk-gated
effect to "dropping the largest source and ... alternative risk and geometry binnings" — but not to
the one factor it has itself demonstrated to be an order of magnitude larger than the effect.

Two further generality problems sit on the same result:

- **Source heterogeneity.** In Fig. 3b the merge/pass shift bars (all measured from the same zero
  line, so their ratio is scale-free) have lengths 1508, 538, 301, 255 px for Waymo, Lyft,
  Argoverse-2 and nuPlan — a **5.9-fold** spread between largest and smallest. The caption concedes
  "the size of the shift differs by source". The pooled reference is 61% Waymo
  (23,218 / 38,228 = 60.7%).
- **The mapping does not transfer between sources.** Fig. 3c: held-out R² is +0.026, +0.017,
  −0.195, −0.276. Methods 4.4: leave-one-source-out 90% coverage is 0.743 (Waymo) and 0.750 (Lyft).
  Fig. 4's caption states plainly: "Transfer to a data source not seen during fitting is not
  established and is reported as a boundary of the present monitor."

**Why it matters.** Sections 2.4 and 2.5 then deploy the frozen reference on a fifth, unseen source
(a different country, a closed course, staged conflicts, a simulator counterpart) and report the
results as the paper's principal evidence. The paper declares the boundary and then works outside
it. The human audit arm is the right instrument for this and it partly answers the concern — but
the paper must say so explicitly rather than leaving the reader to reconcile Fig. 4's caption with
Section 2.4.

**Remedy.** Repeat Fig. 3a under all three summary rules of Fig. 2c. Report per-source conditional
ranges and per-source coverage on the benchmark situations. Add one sentence in Section 2.4 stating
that the benchmark is an out-of-source deployment, that leave-one-source-out coverage of 0.74–0.75
would predict inflated flag rates, and that the human arm is what bounds this.

---

### M7. Every consequence measurement in the paper — in both arms — is the response of one traffic-simulation model

**[Methods 4.5, 4.6; Figs. 5c,d, 6b]**

**What is wrong.** The counterpart in the automated arm and in the human arm is the same
TESS-NG-driven vehicle (Methods 4.5: "the same counterpart control is used in the human reference
arm below"). Every counterpart-side endpoint in Fig. 5c,d and Fig. 6b is therefore a property of a
single rule-based traffic model's response function. The "shared consequence signature" across
humans and machines — the Discussion's claim that "what it flags carries the same consequence
signature whether the driver is a human or a machine" — is equally consistent with the reading that
the signature is a property of the simulator, which responds to closing kinematics the same way
whoever produced them.

The paper does check that the counterpart is reactive ("the same counterpart's trajectory differs
between systems by 4.35–9.03 m"), which is necessary but not sufficient: a deterministic
car-following model also differs between systems.

There is a second-order consequence. The reference range is learned from human–human pairs
(Methods 4.3) but applied where the counterpart is a simulator. Only 20.8% of candidate moments
are judgeable, and in the 36 always-silent runs the human-support pass rate is 0.07% against 55.3%
readability — i.e. the situations these interactions generate largely fall outside anything the
human record contains. The paper reads this as an honest abstention (correctly), but does not
consider that the same non-human counterpart kinematics may bias the readings that *do* pass.

**Why it matters.** For a readership in interactive behaviour, "what a flag means for the
interaction" is the central question, and the paper answers it only for interactions with one piece
of software. This bounds the generality claim considerably more than the manuscript admits.

**Remedy.** State the dependence explicitly in Section 2.4 and the Discussion. If any subset of
runs involved a human-driven or differently-controlled counterpart, report the signature there. At
minimum, report the counterpart model's own deceleration policy and show what its speed response
would have been to a matched non-flagged closing profile.

---

### M8. The separation between "atypical" and "inappropriate" is declared but not carried by the vocabulary or by Section 2.6

**[Sections 2.4, 2.6, 3; Methods 4.4; Fig. 5]**

**What is wrong.** The bound itself is stated well and repeatedly, and I am not attacking the paper
for failing to prove harm — it explicitly declines to. What I am flagging is that the bound is not
honoured consistently.

- Every conventional measure points the *other* way. Flagged assertive moments have 0.46–0.61× the
  rate of short margins at three thresholds and 3.1–4.4 pp less counterpart hard braking. The
  symmetric reading — that these are moments of decisive, promptly resolved conflict, which is
  competent driving — is never allowed onto the page.
- The vocabulary is uniformly a burden vocabulary: the counterpart "absorbs" the speed reduction;
  the interaction is "tighter"; "compression of ordinary interaction quality"; the Introduction's
  "burden the interaction". The verdict label itself, "Over-Yielding" (Methods 4.4), asserts
  excess, and "Competitive-Deviation" asserts competition.
- Section 2.6 argues the monitor's value is that "a monitor built on those thresholds would report
  nothing, while the interaction has nonetheless become measurably tighter for both vehicles". With
  the magnitudes established in M3 (12.5 s → 7.7 s at the 75th percentile; 0.39 m s⁻¹ of
  counterpart speed change), the neutral statement is that neither the safety monitor nor the
  social monitor has registered anything of operational consequence.

**Why it matters.** The paper's own validation route ("human-dispreferred behaviour → interaction-
harmful behaviour") specifies exactly the missing piece: a human-preference judgement. Collecting
human ratings of, say, 200 flagged and 200 matched within-range clips is cheap, fast, and within
this group's demonstrated capability. Without it, Section 2.6 is asking the reader to accept a
value judgement that the paper elsewhere says it is not making.

**Remedy.** Either add a released-preference arm (human ratings of flagged vs matched clips), or
rewrite Section 2.4 and 2.6 in neutral language and rename the two verdicts to
directionally-descriptive labels.

---

## 3. Minor issues

1. **Readable-fraction arithmetic.** Methods 4.4 gives the natural-corpus accounting over 4,497,368
   anchor rows as 3,202,646 readable + 1,275,480 near-uniform + 17,416 ties + 1,826 solver failures
   (these sum exactly to 4,497,368). But 3,202,646 / 4,497,368 = **71.21%**, whereas Methods 4.5
   quotes **70.3%** for the same quantity ("the fraction of moments whose reading carries
   discriminative information ... 70.3% vs 55.3%"). The gap is 0.9 pp ≈ 41,000 rows. State which
   denominator each uses.

2. **Undeclared gate denominators.** Methods 4.5 gives a benchmark human-support pass rate of
   "32.32% overall". If that is a share of *readable* moments (0.553 × 67,861 = 37,527), then
   judgeable would be 0.3232 × 37,527 = 12,129 — fewer than the stated 14,099, which is impossible
   since judgeable ⊆ readable. It must be a share of candidate moments (0.3232 × 67,861 = 21,933,
   which is consistent). Say so.

3. **Fig. 6 funnel, "count unavailable".** The automated arm's readable count is given as
   unavailable while 55.3% is printed; 0.553 × 67,861 = 37,527. Either report the count or explain
   why the percentage exists without it.

4. **"uncertainty not shown"** is printed on two panels (Fig. 2b and Fig. 6c). At this journal that
   is not acceptable, particularly for Fig. 6c, which supports a 15-of-15 claim.

5. **Fig. 6c** gives no interval, no test statistic and no accounting for the fact that the fifteen
   scenarios are not independent draws from a scenario population.

6. **Two permutation schemes disagree.** Methods 4.5: the exposure placebo gives p = 0.0199 while
   "A case-level label permutation on the same battery does not reach significance (p = 0.1493)".
   The justification for preferring the first ("the exposure placebo above is the test specific to
   flag timing") is asserted, not demonstrated. Also, 200 draws give a resolution of 1/201 ≈ 0.005;
   p = 0.0199 is four exceedances. Use ≥ 2,000 draws.

7. **Fig. ED2b uses a centred smoother to make a before/after claim.** The lines are "centred
   21-frame medians", so the "pre-alert baseline" window necessarily contains post-alert data and
   vice versa. As drawn, the counterpart has already reached its floor speed of 1.40 m s⁻¹ at the
   first flagged frame, and is *rising* at the last two flags. The panel title "Counterpart speed
   falls across the flagged stretch" is not what the panel shows. Use a causal (trailing) smoother
   or raw traces.

8. **Fig. 5c** reports "not supported" for six of eight entries, including all four
   accommodating-side entries and both steering endpoints, on a log axis running to 20×. The
   unsupported grey entries are visually as prominent as the two supported ones.

9. **"No alarm inflation" (Abstract) is not the measured direction.** The measured direction is
   deflation to roughly half the native rate, which is a materially different and more interesting
   fact.

10. **"Ego margin to counterpart"** is a minimum time-to-collision restricted to closing frames,
    with an upper tail beyond 100 s (Fig. 5a). Calling a 12-second value a "margin to counterpart"
    invites over-reading. Rename and truncate or censor the axis at a physically meaningful value.

11. **Section 2.2 is episode-level; the monitor is per-moment.** Fig. 2c is used to argue that
    episode summaries are not interpretable, and Section 2.2 is then used to justify the monitor's
    conditioning set. Say explicitly why the episode-level evidence licenses a per-moment design
    choice.

12. **Reference-frame agreement.** Methods 4.1 reports correlation ≈ 0.993 against the static
    map-lane reference but "bounded to the lane-referenced slice (about three quarters of cases)".
    What is the agreement on the remaining quarter, and are those cases over-represented on the
    benchmark?

13. **Data availability.** The human-arm records are not released and the code is "available to
    referees on request". The human arm is the paper's principal validity argument; consider
    releasing a de-identified moment-level verdict and outcome series (not raw driving records),
    which the consent constraint would plausibly permit.

14. **Competing interests.** The statement covers the benchmark but not the human reference arm,
    which is also run by the authors' group and whose data are not released.

15. The Discussion offers the framework as "a template for monitoring whether autonomous agents
    behave within human normative ranges online" while the evidence is pairwise vehicle–vehicle at
    mapped conflict points. That is a large step; either support it or scope it.

---

## 4. Questions to the authors

1. What fraction of flagged moments (each side, each arm) has its maximum normalised candidate
   weight on an endpoint of the grid, and what is that maximum weight's distribution for flagged vs
   within-range moments?
2. In Fig. 1c the 90% band's lower edge coincides with the estimator's floor (−1.178 rad) for the
   whole displayed stretch, and the two flagged readings sit within ~0.01 rad of the ceiling.
   Across the whole benchmark, in what fraction of judgeable moments is an assertive flag
   arithmetically attainable at all, i.e. in what fraction does the band's lower edge lie strictly
   above −1.178 rad by more than one grid step?
3. What is the event-level correlation between the two agents' IPVs — coefficient, confidence
   interval, n, and sign — and how is it computed?
4. What sharpening would the counterpart-IPV ablation have been powered to detect? What happens if
   hyperparameters are re-tuned for the augmented model, or if an offline/oracle counterpart
   summary is used as an upper bound?
5. What exactly is a "frozen situation cell"? How many cells are there, over which of the 26
   channels, and what are the flagged / within-range balance statistics for closing rate,
   range-rate and distance-to-conflict within them?
6. What are the absolute upper-quartile margins for the three groups (my reading of Fig. 5a gives
   ≈12.5 s and ≈7.7 s)? Does the assertive-side contrast survive restriction to moments whose
   post-verdict minimum TTC is under, say, 5 s?
7. Why does the ego-margin upper quartile carry an interval excluding parity in Fig. 5a
   (0.601 [0.436, 0.831]) but no asterisk in Fig. 6b, for the same automated arm?
8. Under a persistence requirement of 3, 5 or 10 consecutive judgeable flagged frames, what are the
   flag rates in each arm, the AV/human ratio, and the consequence signature?
9. How were the twenty human drivers distributed over the flagged moments? Please give per-driver
   flag rates and the number of runs contributing the 435 assertive human moments.
10. Can you bound the effect of the head-mounted display on the human arm's driving style, or run
    any human trials through the same staging the automated systems saw?
11. Does the risk-gated priority reversal of Fig. 3a hold under all three episode-summary rules of
    Fig. 2c?
12. Is the naturalistic readable fraction 71.21% (from the Methods 4.4 accounting) or 70.3% (as
    printed in Methods 4.5)?

---

## 5. Prioritised revision requests

1. **Establish that the flag is not a grid-saturation artefact** (M1). Weight distribution for
   flagged vs within-range moments; fraction of flagged readings within one grid step of an
   endpoint; re-run with a denser/wider grid and show the AV/human ratio is stable; report
   best-candidate fit quality at flagged moments.
2. **Separate the consequence signature from kinematics** (M3). Define the situation cells; report
   group balance on closing kinematics; report the flag's incremental value over a raw-kinematics
   baseline; print the absolute upper-quartile margins.
3. **Re-anchor the Abstract and Discussion on all three flag rates** (M5a). Say that the automated
   arm sits at the instrument's native rate and that the matched human arm sits significantly
   below it; give an interval on the ratio.
4. **Quantify the partner correlation and the ablation's power, and soften the "no hidden
   intention" claim to what they support** (M4).
5. **Report run-level operating characteristics under a persistence rule** (M2), and correct or
   support the continuity claim for θ̂.
6. **Reconcile Fig. 5a and Fig. 6b**, draw all intervals in Fig. 6b, and restate Section 2.5's
   shared-signature sentence around the endpoints that are actually marked (M5c).
7. **Add a human-preference arm, or neutralise the outcome vocabulary and Section 2.6** (M8).
8. **Scope the counterpart dependence and the source-transfer boundary explicitly in Sections 2.4
   and 2.5** (M6, M7).
9. Fix the arithmetic and denominator issues (minors 1–3) and remove "uncertainty not shown" from
   both panels (minor 4).

---

## 6. Acceptance probability

- **(a) As submitted:** **3%**. The central measurement-validity question (M1–M3) is unanswered in
  the manuscript and is visible in the authors' own figures; no NMI editor will accept an online
  social monitor whose flag cannot be distinguished from estimator saturation.
- **(b) Assuming a competent major revision:** **25%**. The framing (online conditional
  atypicality, explicit abstention, conformal calibration, a human audit arm run under the same
  instrument) is genuinely novel and well matched to this journal, and most of my requests are
  analyses on data the authors already hold. The residual risk is that M1 and M3 do not survive:
  if flags turn out to be grid-boundary events and the consequence signature disappears once
  closing kinematics are balanced, there is no publishable claim left at this level.

---

## 7. Recommendation

**Major revision.**
