# Referee B — methodology, statistics, runtime verification / formal semantics

## 1. Summary assessment

The paper reframes social compliance as an online membership test: is the ego's current
interaction-preference reading (IPV) inside a conformally calibrated human reference range
conditioned on the observable situation, and if the reading is not identifiable, abstain. It
delivers a large four-corpus reference (2,442,625 anchor rows), a conditioned range ~21%/20%
narrower than a global one at coverage within 0.6 pp of nominal, a matched-scenario real-vehicle
benchmark (19 systems, 267 runs) and — its best feature — a matched human arm that audits the
reference with the population that defined it.

The framing is genuinely new and the claims are bounded with unusual care for this
literature. But the measurement layer does not currently support the inferences drawn from it.
The estimator saturates on a seven-point grid, so an assertive-side flag is arithmetically an
"all likelihood on the extreme candidate" event that the readability gate co-selects; the verdict
propagates no uncertainty from a reading the authors themselves show moves 0.30 rad between
consecutive frames; and the cross-corpus audit conclusion is contradicted by the paper's own
leave-one-source-out numbers, reconciled only by a 12-fold rise in support abstention that is
never examined.

**Major revision.**

---

## 2. Major weaknesses

### MW1 — [Methods 4.1, 4.5; Figs ED1, ED2, 4a/b] An assertive-side flag is arithmetically a saturation event, and the readability gate co-selects exactly those moments

The reading is a weighted mean over seven candidates, θ̂ = Σ w_k θ_k with
θ_k ∈ {−3,…,3}×π/8, so θ̂ is confined to [−1.178, +1.178] rad. Methods 4.5 states that "at the
90% level the lower edge is negative in essentially every situation (median −1.03 rad)". Two
consequences follow by arithmetic I have done:

*(a) The assertive-flaggable region is 6.3% of the admissible span.* At the median lower edge only
1.178 − 1.03 = 0.148 rad of the 2.356 rad grid can produce an assertive flag.

*(b) An assertive flag forces near-total weight concentration.* Writing θ̂ = (π/8)·Σ w_k k,
a flag needs Σ w_k k ≤ −1.03/0.392699 = −2.6229. With a = w_{−3}, the most negative attainable
value is −3a − 2(1−a) = −(2+a), so −(2+a) ≤ −2.6229 ⟹ **a ≥ 0.623**; any other placement of the
residual mass forces a higher a. The readability gate (Methods 4.1) admits a reading whenever
"the largest normalised weight falls below 0.20" is not triggered. So the within-range group
contains everything from w_max = 0.20 upward while the assertive-flagged group is a strict subset
of {w_max ≥ 0.623}. The same computation on the accommodating side (upper edge ≈ +0.84 rad, from
the stated 90% mean width 1.87 rad and the 0.148 rad lower tail) gives w_max ≥ 0.380.

The three groups compared throughout Section 2.4 and Fig. 5 therefore differ systematically in
estimator sharpness, and they differ *unequally on the two sides* — which is precisely the
asymmetry the paper's headline result is built on ("Compression follows atypicality on the
assertive side and not on the accommodating side"). If weight concentration correlates with
closing kinematics — and it must, since concentration is driven by how strongly the candidate
trajectories separate — then the consequence signature can be manufactured by the gate ordering
rather than by atypicality. This is the failure mode the construction is most exposed to and the
paper does not test it.

*(c) The figures show the saturation directly.* In Fig. ED2c **every** flagged assertive moment
sits at the floor of the reading axis (≈ −1.18 rad), including two isolated single-frame plunges
at 16.85 s and 17.3 s that are flagged. Fig. ED1a shows a dense band of readings pinned at the
floor and repeated frame-to-frame excursions between the two extreme candidates.

*(d) Fig. 4 independently implies a large atom.* Fig. 4a prints the global width as 100% of the
admissible span at both the 90% and 95% levels (the caption states the 95% value "equals 100.00%
of the admissible span"), while Fig. 4b gives global achieved-minus-nominal coverage of −0.01 pp
and +2.77 pp — i.e. 89.99% and 97.77%. Adding at most 1 pp of width (0.024 rad, given the stated
whole-percent rounding) cannot move coverage by 7.78 pp unless the density in that sliver is
≥ 0.0778/0.024 = 3.30 per rad, against an average density over the support of 1/2.356 = 0.424 per
rad and 0.189 per rad in the 78%→100% shell — a factor ≥ 7.8 above average and ≥ 17 above the
adjacent shell. Either the reading distribution has a large atom at ±3π/8, or the two printed
widths differ by far more than the rounding admits.

**Why it matters.** If the assertive flag is an estimator-saturation indicator, then "the current
preference weights the counterpart's cost less than humans do in the same situation" is not what
is being measured, and every downstream claim in Sections 2.4–2.6 inherits the ambiguity.

**Remedy.** Report the empirical distribution of θ̂ with the point masses at ±3π/8 quantified;
report the fraction of situations in which the 90% lower edge lies at or below the grid floor (so
that an assertive flag is structurally impossible); refit with a substantially finer grid
(e.g. 21–41 candidates) and show that flag rates, the side asymmetry and the Fig. 5 contrasts are
stable; report the distribution of w_max separately in the assertive, accommodating and
within-range groups.

---

### MW2 — [Eq. (1) vs Methods 4.1] The implemented weights are a tempered likelihood whose sharpness depends on window occupancy, and none of the estimator's free constants is validated

Eq. (1) defines ℓ_{a,k}(t) = p(X_a(W_t) | X̃_a(W_t;θ_k), X_j(W_t), M) — a likelihood — and π_{a,k}
as its normalisation. Methods 4.1 implements
"ℓ_{a,k} ∝ exp{−MSE_k/(2σ²)} with σ = 0.1, where MSE_k is the **mean** squared Euclidean distance
… over the window". Under the i.i.d. Gaussian observation model the exponent would be
−n_f·MSE_k/(2σ²) with n_f the number of observed frames. Dividing by n_f makes π_{a,k} the n_f-th
root of the stated likelihood, i.e. a *tempered* posterior, not the object Eq. (1) defines.

This is not pedantry: the window admits 4 to 10 observed frames ("A rolling window of the ten most
recent frames at 10 Hz … with at least four observed frames required"). A 4-frame window is
tempered by 1/4 and a 10-frame window by 1/10, so identical per-frame fit quality yields sharper
weights — and therefore a higher chance of passing the readability gate — on shorter windows.
Readability, the paper's central conceptual construct, thus depends mechanically on tracking
continuity. This is a candidate explanation for the unexplained readability gap between corpora
("70.3% vs 55.3%"; "we report this as an observed difference without attributing a cause") and for
Fig. 2b's finding that the reading *moves more* where it is more readable.

Compounding this: with σ = 0.1 m the exponent is −50·MSE_k, so an MSE difference of 0.1 m²
between candidates gives a likelihood ratio of e⁵ ≈ 148. The likelihood surface is extremely
peaked, which is consistent with both the saturation in MW1 and with the 17,416 "exact ties"
reported in Methods 4.4 (all seven likelihoods underflowing). The word "sensitiv-" does not appear
anywhere in the manuscript: there is no sensitivity analysis to σ, to the 0.20 near-uniform
threshold, to the seven-point grid, or to the 1 s window. Two of the four gate thresholds in
Eq. (2), q₀ and c₀, are never given a numerical value anywhere in the paper, and u_a itself is
never given a formula ("a dispersion statistic of the normalised candidate weights, not a standard
deviation") — the gate is instead operationalised through a different statistic (max weight).

**Why it matters.** A paper whose contribution is an "auditable" frozen monitor must publish the
monitor's parameters and show that its verdicts are not artefacts of them.

**Remedy.** Correct Eq. (1) or the Methods so they define the same object; state whether the
n_f-tempering is intended and show readability rates stratified by n_f; give q₀, c₀ and the u_a
formula in the paper; add a sensitivity table over σ, grid resolution, window length and the 0.20
threshold, reporting flag rate, coverage and the Fig. 5 upper-quartile contrast for each.

---

### MW3 — [Section 2.3; Fig. 2b; Methods 4.4] The verdict propagates no uncertainty from the reading, so the conformal coverage is not the monitor's error rate

Split-conformal calibration here controls where a *human's* reading would fall given z_t. The
verdict is a membership test on θ̂_i(t), which is an estimate, treated as exact. u_a(t) is used
only as a gate, never propagated. The consequence is that Fig. 4b's "+0.03, +0.28, +0.57 pp" is
a property of the reference, not a false-alarm rate for the monitor — yet the paper repeatedly
presents near-nominal coverage as evidence that the *monitor* is calibrated.

The paper supplies the number that makes this binding. Fig. 2b: the mean movement of the reading
between consecutive frames is "0.30 rad for the ego and 0.31 for the counterpart" **at the most
readable moments**, and the authors correctly conclude that "reliability must not be read as
convergence". Adjacent frames share 9 of 10 window samples, so this forces a dilemma:

- if the movement is real, the monitored preference changes at ~3.0 rad s⁻¹ against a total
  admissible span of 2.356 rad, and is not a stable object at the timescale at which verdicts are
  issued; or
- if it is estimation error, then assuming independent Gaussian errors (E|Δ| = 1.1284·σ_ε) gives
  σ_ε ≥ 0.30/1.1284 = **0.266 rad**, which is 28% of the 90% band half-width (1.87/2 = 0.935 rad).
  Positive error correlation across overlapping windows would make this a *lower* bound.

At σ_ε = 0.266 rad, a vehicle whose true preference sits exactly at the band edge is flagged on
about half of frames, and one sitting a quarter of a band-half-width inside the edge is still
flagged on roughly a sixth. Fig. ED2c shows exactly this: isolated single-frame flags amid ~1 rad
frame-to-frame swings, all entering the 519-moment assertive group with equal weight.

**Remedy.** Define the verdict on the reading's own uncertainty (e.g. flag only when the reading's
interval is disjoint from the reference band), or report the verdict's operating characteristics
under an explicit noise model; report a test–retest or split-half reliability for θ̂; and
distinguish reference coverage from monitor false-alarm rate throughout.

---

### MW4 — [Section 2.4; Methods 4.4; planner interface] The monitor is only ever evaluated per moment, and its own operating point implies a near-certain run-level alarm

Methods 4.4: "A sequential warning layer (persistence over consecutive verdicts) belongs to the
deployment interface: no result in this paper uses it, every reported verdict and flag rate is
per-moment, and its operating characteristics (episode-level false-alarm rate, detection delay,
warning duration) are not evaluated". Methods 4.4 also states that trajectory-wise simultaneous
coverage is not established. Yet the planner interface specifies "a fallback candidate under
**sustained** competitive deviation", i.e. an action defined on exactly the sequential layer that
is not evaluated.

The run-level consequence is computable from the paper's numbers. There are 14,099 judgeable
moments in 267 runs = 52.8 per run, and the reference's native outside rate is 9.72%. Under
independence, P(at least one flag in a nominal run) = 1 − 0.9028^52.8 = **99.5%**. Flags are in
fact clustered (Fig. ED2 shows contiguous stretches), and the paper's own funnel gives 120 of 267
runs (45%) containing at least one flagged *assertive* moment — so even with clustering, a
correctly calibrated monitor alarms on close to half of all runs on the assertive side alone. For
a contribution positioned explicitly against runtime verification of formal specifications
(refs 36–39), the absence of any episode-level guarantee or measured episode-level alarm rate is
the gap that decides whether this is a monitor or a per-frame descriptive statistic.

**Remedy.** Report run-level and episode-level alarm rates for all three arms; report how many
independent flag stretches produce the 519 assertive moments (and the stretch-length
distribution); either evaluate the persistence layer with detection delay and episode false-alarm
rate, or remove the planner-interface claims that depend on it.

---

### MW5 — [Section 2.5; Fig. 3c; Methods 4.4] The transfer conclusion is contradicted by the paper's own transfer measurements and reconciled only by a 12-fold rise in support abstention

Section 2.5 concludes that "carrying the reference across country, apparatus and task does not
inflate its alarms". The paper's two direct measurements of transfer say the opposite:

- Fig. 3c: fitting the **same** full state description on three sources and predicting the fourth
  gives R² = +0.026, +0.017, −0.195, −0.276. Two of four folds are worse than predicting the
  held-out mean.
- Methods 4.4: leave-one-source-out 90% coverage is 0.743 (Waymo), 0.990 (nuPlan), 0.750 (Lyft),
  0.900 (Argoverse-2) — outside rates of 25.7% and 25.0% in two folds, 2.5× nominal.

The real-vehicle benchmark is a new source in every respect (country, apparatus, task, vehicle,
mixed-reality staging). By the paper's own leave-one-source-out evidence the expected alarm rate
there is ~25%, not the observed 5.0% (humans) / 9.8% (systems). The available reconciliation is
the support gate — and the paper supplies the number without ever stating it. On natural driving,
support abstention among readable moments is 5.08%. On the benchmark it is
1 − 14,099/(0.553 × 67,861) = 1 − 14,099/37,527 = **62.4%**, a **12.3-fold** increase (the human
arm is 1 − 15,598/40,993 = 61.9%). Section 2.3 describes this as "abstention on the real-vehicle
benchmark is larger". Methods 4.4 explicitly adopts the correct reading for the LOSO folds —
"coverage and abstention must be read together" — but Section 2.5 does not apply it to the audit.

Related, and quantitatively decisive for the headline comparison: Methods 4.4 reports by-source
conditional coverage on the deployed reference of 88.2%, 95.9%, 85.9%, 88.2% at the 90% level and
concludes "the pooled figure is not masking a source in which the range fails". Weighting these
by the source sizes implied by the LOSO counts (193,096 / 150,587 / 91,069 / 12,576) reproduces
the pooled figure exactly — 0.4317(88.2) + 0.3366(95.9) + 0.2036(85.9) + 0.0281(88.2) = **90.32%**
— which shows the opposite of the claim: three of four sources under-cover (outside rates 11.8%,
14.1%, 11.8%) and the pool is rescued by nuPlan's over-coverage. The resulting spread of
source-level outside rates, 4.1% to 14.1% (10.0 pp), is more than twice the AV-versus-human
difference the paper headlines (9.8% − 5.0% = 4.8 pp). The instrument's own conditional
miscalibration across sources exceeds the effect it is being used to detect.

**Remedy.** Report LOSO abstention for all four folds, not only Argoverse-2; report the situational
composition of the benchmark's judgeable set relative to the reference support region; test
whether the automated and human judgeable sets are matched on z_t (they are the two arms of the
headline comparison and are each selected by the same endogenous filter); and rewrite Section 2.5
so the audit conclusion is stated jointly over coverage and abstention.

---

### MW6 — [Section 2.5; Fig. 6a] The audit is a two-sided calibration statement reported one-sidedly, and the headline ratio has no interval

I verified the flag counts against the panel. Human arm: 879+676 = 1,555 → 1,555/15,598 = 9.97%;
435+351 = 786 → 5.04%; 189+167 = 356 → 2.28%. Automated arm: 1,174+1,574 = 2,748 → 19.49%;
519+869 = 1,388 → 9.84%; 314+387 = 701 → 4.97%. Nominal outside rates are 20 / 10 / 5, and the
native (held-out natural-driving) rates are 20.0 / 9.7 / 4.4, consistent with Fig. 4b.

The automated systems therefore sit **at nominal at all three levels**. The anomaly in the audit
is not the machines; it is that the matched human drivers are flagged at almost exactly half the
nominal rate at every level (0.50×, 0.50×, 0.46×). Under-flagging by a factor of two is as much a
calibration failure as over-flagging: it says the frozen reference does not describe the
closed-course human population, and therefore that the monitor's operating point on this
apparatus is unknown. The Results sentence acknowledging this ("The automated systems' flag rate
of 9.8% sits at the native level while exceeding that of the matched human drivers") does not
survive into the abstract ("automated systems flagged about twice as often"), the Introduction, or
the Discussion ("it flags its own defining population no more often than at home" — it flags them
half as often).

Second, the paper's most quotable number has no uncertainty attached anywhere. The only interval
in Fig. 6a is on the human arm (3.4–5.5%); there is none for the automated arm and none for the
2.0× ratio, and no test is reported. The one inferential statement — 15 of 15 scenarios above
parity, a paired sign test at p ≈ 6×10⁻⁵ — is never labelled as a test, and Fig. 6c prints
"uncertainty not shown".

In fairness, I checked whether the ratio is an artefact of the judgeable denominator: on candidate
moments it is 1,388/67,861 = 2.045% versus 786/78,903 = 0.996%, a ratio of 2.05 against 1.95 on
judgeable moments. The ratio is robust to that choice, and the authors should say so.

**Remedy.** Give a run-clustered interval for the ratio and state the sign test formally; present
the human arm's 2× under-flagging as a calibration finding in its own right and discuss what it
implies for the operating point; align abstract, Introduction and Discussion with the three-rate
statement already in Section 2.5.

---

### MW7 — [Section 2.4; Methods 4.5] The main text reports the more favourable of two permutation tests, the placebo has too few draws, and there is no multiplicity control anywhere

Section 2.4 states that "the association survives a placebo test that reassigns whole flag
sequences across scenario runs (Methods 4.5)". Methods 4.5 adds, and the main text does not:
"A case-level label permutation on the same battery does not reach significance (p = 0.1493)".
The label permutation is the standard nonparametric test of whether flagged and within-range
moments differ; it fails at conventional levels while the analytic case-clustered intervals on the
same battery exclude zero. When a permutation test disagrees with a clustered analytic interval on
the same statistic, the permutation result is normally the more trustworthy, and a reader of the
main text alone would not know it exists. Two further Methods disclosures point the same way and
also do not reach the main text: the alternative (median 0.60 s) window "crosses zero at the 90%
level ([−2.6100, +0.1372])", and "Weighted equally per run, the assertive-side counterpart braking
contrast does not reach significance (p = 0.4704)".

The placebo itself is under-powered as a headline test: p = 0.0199 = 4/201 from 200
whole-trajectory draws, whose Monte-Carlo standard error is √(0.0199·0.9801/200) = 0.0099. The
resampling error alone spans roughly [0.001, 0.039].

Finally, no multiplicity handling appears anywhere — the words "multiplicity", "Bonferroni" and
"false discovery" do not occur. The battery is ~13 endpoints run on two sides plus six
level-window combinations. Pre-specification (which the authors did well, and describe carefully)
is not multiplicity control. The between-side result that carries Fig. 5's title has p = 0.014
uncorrected; the accommodating-side "<1 s" interval, [−5.56, −0.02] pp, would not survive any
correction. And the side-specific claim is made at the endpoint that worked: "the side-specific
statement is made at the quartile that supports it and withheld at the one that does not". Two
pre-specified quartiles, claim made at one of them post hoc, is a selection the paper is candid
about but does not price into its inference.

**Remedy.** Report both permutation tests in the main text with equal prominence; raise the
placebo to ≥10,000 draws; nominate one primary endpoint and control family-wise error (or report
FDR-adjusted values) across the declared battery and across both sides.

---

### MW8 — [Section 2.5; Fig. 6b] The human-arm replication fails, by the figure's own convention, on the endpoint that carries the automated-arm result — and is asserted anyway

Fig. 6b's caption defines the convention: "an asterisk marks a ratio whose interval excludes
parity (the no-difference value, ratio 1); entries whose intervals admit parity carry no
asterisk." Section 2.5 states that at flagged human moments "the same consequence signature
appears that Fig. 5 documents for automated vehicles: the ego margin's upper quartile contracts by
about three-tenths, the counterpart absorbs 2.5 times the routine speed reduction, and every
emergency tail is rarer".

Inspecting the panel at high magnification, the asterisks are placed with a constant horizontal
offset (~+80 px at native resolution) from their marker, in every row where the pairing is
unambiguous. On the "Ego margin (upper quartile)" row that offset places the single (blue)
asterisk on the automated-vehicle marker; the human marker carries none. So the human-arm
upper-quartile contraction — the direct counterpart of the −39.9% that is the paper's headline
ego-side result — has an interval that admits parity, and is nevertheless quoted as a number in
the main text with no hedge. The paper applies its own support convention scrupulously in Fig. 5c
("A point ratio is printed only where the interval supports it; grey entries are reported but not
supported") and abandons it here.

Two further defects in the same panel:

- The three ego rows carry **no whiskers at all**, although the caption says whiskers are drawn
  "where defined" and Fig. 5a supplies intervals for exactly those quantities on the automated
  arm. For the automated upper quartile, 0.601 [0.436, 0.831], a whisker would be about five times
  the marker width, so it is genuinely absent. Three of six endpoints therefore carry significance
  marks with no displayed interval.
- The automated "Ego emergency (< 2 s)" entry is *supported* on the percentage-point scale in
  Fig. 5b (−6.73 [−11.67, −1.76]) but carries no asterisk on the ratio scale in Fig. 6b. Ratio and
  difference intervals can legitimately disagree when the denominator is resampled, but the paper
  should say which is primary and why the same contrast changes status between two figures.

**Remedy.** Draw the intervals for all six endpoints in both arms; restate the human-arm sentence
inside the support convention the paper defines; reconcile the two scales for the < 2 s endpoint.

---

### MW9 — [Section 2.4; Fig. 5a; Fig. ED2] The one resolved ego-side effect is reported without its absolute scale and sits in a regime where the endpoint is not a margin

Fig. 5a plots the ego margin on a log axis running past 10² s. Reading the quartile markers
against the decade grid (10⁰ and 10¹ ticks 649 px apart at the crop scale I used), the
within-range upper quartile is ≈ 12.6 s and the assertive upper quartile ≈ 7.6 s — a ratio of
0.60, which reproduces the reported 0.601 and confirms the calibration; the medians read ≈ 5.2 s
and ≈ 4.7 s, ratio 0.91 against the reported 0.901. So the headline "the upper quartile falls by
two-fifths" is a change from about 13 s to about 8 s of minimum time-to-collision over a 3 s
window.

The paper prints the absolute value of exactly one quartile — the lower one, 3.46 vs 2.99 s —
and it is the one with no effect; the median and upper quartile are given only as percentages.
A minimum TTC of 13 s over a 3 s window is, by the caption's own description, "a window whose only
closing frames close very slowly". Describing that as the "comfortable end" of an interaction
margin, and its contraction as "a measurably tighter interaction", overstates what the endpoint
can carry. The upper quartile is also the least stable of the three, because it sits at the
boundary of the population of barely-closing windows created by the rule that non-closing frames
do not enter the minimum (a rule that already drops 3.9% / 6.0% / 2.9% of moments differentially
by group).

Fig. ED2 compounds this rather than resolving it. The illustrative run shows the flagged automated
vehicle holding constant speed (3.33 → 3.32 m s⁻¹) while the counterpart decelerates — and the
plotted counterpart trace collapses from ~3.8 to ~1.4 m s⁻¹ between 15.2 s and 15.4 s, i.e. the
deceleration is essentially complete *before* the first flag in the shaded stretch. The panel is
moreover a pre-flag-baseline-versus-flagged-stretch contrast (4.31 → 1.40 m s⁻¹), not the
post-verdict three-second contrast that Fig. 5c actually measures, and the run was selected under
a criterion that requires a ≥20% counterpart speed drop. The figure offered to make the mechanism
visible shows the counterpart's response preceding the verdict.

This matters for the paper's central convergence argument ("The two sides are measured from
different sources and different quantities, so their agreement is a convergence rather than a
restatement"). The two sides are also measured on different run sets (227 ego runs, 175
counterpart runs), and both are driven by the same closing kinematics that produced the verdict
one tenth of a second earlier.

**Remedy.** Print the absolute quartiles for all three groups; repeat the analysis restricted to
windows with an operationally meaningful margin (e.g. min TTC below 10 s) and to genuinely closing
pairs; show the contrast survives matching on the counterpart's pre-verdict deceleration; replace
or re-caption Fig. ED2 so it displays the quantity Fig. 5c analyses.

---

## 3. Minor issues

1. **Readability denominator.** Methods 4.5 compares "70.3% vs 55.3%". The Methods 4.4 accounting
   gives 3,202,646 readable of 4,497,368 anchor rows (the four categories sum exactly to the
   total) = 71.21%, not 70.3%. Most likely a different denominator; state it.
2. **Support pass denominator.** "the human-support pass rate is 0.07% (against 32.32% overall)".
   Over candidate moments this implies positive dependence with readability
   (0.553 × 0.3232 = 17.9% < 20.8% judgeable); over readable moments it should be
   14,099/37,527 = 37.57%. State which.
3. **Fig. 6, funnel row.** "count unavailable" is printed for the automated arm's readable stage
   although 0.553 × 67,861 = 37,527 is recoverable from the same figure. Print it or explain why
   the count cannot be given in a paper claiming full readability accounting.
4. **Fig. 2a caption.** "The gain is present only for the real interaction" is contradicted two
   clauses later by "weakens when the same moment is paired with a different partner (−0.043)":
   the different-partner control retains a third of the real-interaction effect (−0.043 vs
   −0.132) with an interval that appears to exclude zero. Rewrite as partial specificity, and
   discuss what the residual non-specific sharpening implies for construct validity.
5. **"Uncertainty not shown"** is printed on Fig. 2b and Fig. 6c, both of which support main-text
   claims. In a paper about calibrated uncertainty this needs fixing rather than labelling.
6. **Fig. 2c annotation.** "first two rules 0.26 rad apart" sits beside medians of +0.08 and
   +0.00. The 0.26 rad is the mean per-episode absolute difference (as the caption says); the
   in-panel label should say so.
7. **Gate parameters.** Eq. (2) uses (u₀, q₀, c₀); only u₀ is operationalised, and through a
   different statistic. Publish all thresholds and the u_a definition in the paper.
8. **Sealed confirmatory split.** Fig. 2 states "Panels use the development split; the
   confirmatory split is held sealed." It is never used anywhere in the paper, so the paper's
   construct-validity evidence is exploratory throughout. Say what the split is reserved for, or
   open it.
9. **Harm layer has no numbers.** "atypicality does not robustly or specifically predict harm
   outcomes" is stated with no count of realised interaction failures, no incidence, no effect
   estimate and no precision. The only harm-adjacent number is that 3 of the 18 lost
   system–scenario cells were collision failures; the total number of collisions is never given. A
   bound with no numbers is not auditable.
10. **Counterpart-side moment counts.** Fig. 5c/d give records (13,800 / 21,025 / 310,246); only
    the accommodating side's anchor count (719) appears in Methods. Give assertive and
    within-range anchor counts so the two rows of Fig. 5 are comparable.
11. **Reason codes are order-dependent.** "reason codes are assigned by the first failing gate in
    the evaluation order of Algorithm 2". The reported reason distribution is therefore not a
    decomposition of causes; note this wherever it is used.
12. **Coverage precision.** Fig. 4b prints achieved-minus-nominal to 0.01 pp with no interval,
    over moments that the paper itself says are dependent. Give a scene- or case-clustered
    interval; the claim "within 0.6 pp" is a precision claim.
13. **Counterpart-log retention.** The retained set is enriched ~3× in flagged runs (132/175 =
    75.4% vs 7/29 = 24.1%). The paper reports this and declines to argue it; a bound or sensitivity
    analysis is needed, since the counterpart side carries the convergence claim.
14. **Section 2.6's value argument.** The section's case for the monitor requires that compression
    of comfortable margins is undesirable — the normative step the paper elsewhere declines to
    take. As written, a monitor that flags moments with roughly half the emergency rate has no
    demonstrated operational value; the section should be framed as orthogonality of measurement
    rather than as added safety-relevant information.
15. **Abstract.** "the same consequence signature" and "no alarm inflation" both overstate; see
    MW6 and MW8.

---

## 4. Questions to the authors

1. What fraction of moments produce a reading exactly at (or within 10⁻³ rad of) ±3π/8? Please
   give the histogram of θ̂ and the distribution of w_max, separately for within-range,
   assertive-flagged and accommodating-flagged moments.
2. In what fraction of judgeable situations does the 90% lower edge lie at or below −1.178 rad, so
   that an assertive-side flag is structurally impossible? Fig. ED1 appears to show roughly the
   first ten seconds of that interaction in exactly this state.
3. Is the n_f-tempering of the likelihood in Methods 4.1 intentional? What are readability rates
   as a function of the number of observed frames in the window?
4. How do the flag rate, the side asymmetry and the Fig. 5 upper-quartile contrast change under a
   finer candidate grid and under σ ∈ {0.05, 0.1, 0.2, 0.5} m?
5. What is the test–retest or split-half reliability of θ̂? What share of the 79.1% residual
   variance in Fig. 4c is estimator noise rather than signal?
6. What are the absolute quartiles of the ego margin in seconds for all three groups? Does the
   upper-quartile contrast survive restriction to windows with min TTC below, say, 10 s?
7. Why does the case-level label permutation (p = 0.1493) fail while the analytic clustered
   intervals on the same battery exclude zero? Which do you take as primary?
8. How many independent flag stretches (not moments) produce the 519 assertive moments, and in how
   many distinct scenario runs?
9. What is the run-level and episode-level alarm rate for each of the three arms at the 90% level?
10. What is the clustered 95% interval for the 2.0× automated-versus-human ratio?
11. Are the automated and human judgeable sets matched on z_t? Please give the standardised
    differences on the 22 numeric channels and four categorical descriptors between the two arms'
    judgeable moments.
12. What is the LOSO abstention rate for the Waymo, nuPlan and Lyft folds, alongside the 44.3%
    given for Argoverse-2?
13. For the counterpart-ablation null (interval-score difference −0.0002, p = 0.86), what
    equivalence margin does the design exclude? The paper is careful to call the accommodating-side
    nulls "non-detections at the achieved precision"; the same standard should apply to the null
    that carries the Discussion's "most informative finding".
14. How many realised interaction failures (by the frozen extractor) occur in the benchmark, how
    many involve flagged moments, and what precision does that support?
15. What are q₀ and c₀?

---

## 5. Prioritised revision requests

1. **Resolve the saturation question (MW1).** Distribution of θ̂ and w_max by verdict class;
   fraction of situations where an assertive flag is impossible; refit on a finer grid and
   demonstrate that Sections 2.4–2.6 are unchanged. *If the assertive flag is a saturation
   indicator, the paper's central empirical claims do not stand in their present form.*
2. **Propagate the reading's uncertainty into the verdict (MW3)**, and separate reference coverage
   from monitor false-alarm rate throughout the text and figures.
3. **Fix the measurement model and validate its constants (MW2):** reconcile Eq. (1) with Methods
   4.1, publish all gate parameters, and add a sensitivity table over σ, grid, window and
   threshold.
4. **Rewrite the transfer/audit argument (MW5, MW6)** so coverage and abstention are read together
   for the benchmark as they are for the LOSO folds; state the 62% benchmark support abstention and
   the by-source coverage spread in the main text; present the human arm's 2× under-flagging as a
   calibration finding; give an interval for the 2.0× ratio.
5. **Repair the inference on the consequence battery (MW7):** both permutation tests in the main
   text, ≥10,000 placebo draws, one declared primary endpoint, family-wise or FDR control across
   the battery and both sides.
6. **Add run-level/episode-level operating characteristics (MW4)**, or remove the planner-interface
   claims that presuppose the unevaluated persistence layer.
7. **Fix Fig. 6b (MW8):** draw all intervals, restate the human-arm sentence within the paper's own
   support convention, reconcile the < 2 s discordance with Fig. 5b.
8. **Report the ego-margin endpoint in absolute units and re-run it in an operationally meaningful
   TTC regime (MW9);** re-caption or replace Fig. ED2.
9. Quantify the harm layer (Minor 9) so that "bounded" is an auditable statement rather than a
   qualitative one.
10. Address the minor issues, in particular the two undefined denominators (Minor 1, 2) and the
    two "uncertainty not shown" panels (Minor 5).

---

## 6. Acceptance probability

- **(a) As submitted: 4%.** The construction is interesting and the reporting is unusually
  disciplined, but the estimator's behaviour at the flag boundary, the absence of uncertainty
  propagation into the verdict, and the tension between the audit conclusion and the paper's own
  transfer measurements are each independently sufficient to block acceptance at this journal.
- **(b) After a competent major revision: 32%.** Conditional on the saturation analysis showing the
  flags are not grid-boundary artefacts, and on the consequence battery surviving multiplicity
  control and the label permutation, this becomes a strong and distinctive paper. Both conditions
  are genuinely uncertain: if the assertive flag is a saturation event, no amount of rewriting
  recovers Sections 2.4–2.6.

---

## 7. Recommendation

**Major revision.**
