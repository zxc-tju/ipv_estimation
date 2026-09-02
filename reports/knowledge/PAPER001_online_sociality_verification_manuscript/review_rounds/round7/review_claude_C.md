# Referee report — "Online monitoring of socially compliant autonomous driving"

*Remit: generalist editor-referee (breadth, venue fit, claim support, declarations, display items).*

Quotations are taken from the submitted PDF; ligatures lost in text extraction (fl, fi, ff) are silently
restored. Every arithmetic assertion below is shown as a computation.

---

## 1. Summary assessment

The paper reframes social compliance for automated driving as an online membership test: is the
vehicle's current interaction preference inside a human reference range conditioned on the observable
situation, and where the reading is not identifiable, abstain rather than pass. It delivers a large
descriptive analysis (38,228 naturalistic interaction cases), a situation-conditioned quantile reference
with near-nominal marginal coverage, a consequence analysis on a 19-system matched-scenario
real-vehicle benchmark, and an audit in which 20 human drivers face the same scenarios under the
frozen monitor. The framing, the readability/abstention discipline and the audit design are new and
carefully executed, and much of the internal bookkeeping is exemplary.

The claims nonetheless run ahead of the reported uncertainty in precisely the places the abstract
advertises. The headline ego-margin "compression" carries no interval anywhere, and the two places
where its interval is indicated both show it admitting no effect. Side-specificity is asserted with no
between-side test. The "flagged about twice as often" result is driven four-fifths by the
accommodating side, for which the paper reports no consequence signature. The audited population is
not the reference's defining population.

**Major revision.**

---

## 2. Major weaknesses

### M1. The headline consequence result is stated as fact and reported without an interval; the paper's own two other reports of it indicate that interval admits no effect

**[Section 2.4; Fig. 5a; Fig. 6b; Methods 4.5]**

Section 2.4 states without hedging: *"The ego's own margin to its counterpart contracts across the
body of its distribution, the median falling by about a quarter and the upper quartile by nearly half."*
Fig. 5a's caption gives the numbers — *"the median falling 24.9% and the upper quartile 47.4%"* —
and no interval. Searching the manuscript, no confidence interval for either quantity appears anywhere.
This is asymmetric: the *accommodating*-side counterpart of the same quantity **is** given with an
interval (Methods 4.5: *"the ego's own margin median is 0.81× ([0.70, 1.15])"*).

Two independent parts of the manuscript indicate that the assertive-side interval also admits no effect.

1. **Fig. 6b.** The caption states the rule: *"an asterisk marks a ratio whose interval excludes parity
   (the no-difference value, ratio 1); entries whose intervals admit parity carry no asterisk."* Of the
   six rows in that panel, exactly two carry no asterisk on either arm: **Ego margin (median)** and
   **Ego margin (upper quartile)**. I verified this by marker geometry rather than by eye: in the
   published PNG every AV asterisk sits 53–54 px above and 46–48 px right of its AV marker (rows
   3–6: markers at y = 952, 1067, 1181, 1295; asterisks at y = 899, 1013, 1127, 1241), and every
   human asterisk sits 13 px below and ~45 px right of its human marker (markers y = 983, 1097,
   1211, 1325; asterisks y = 995, 1110, 1224, 1338). At the predicted positions for rows 1 and 2
   there is no asterisk on either series. The AV-side values in this figure are stated to be measured,
   so this is not an artefact of the human arm.
2. **Methods 4.5.** *"the fixed three-second window is the pre-specified primary, and its
   case-clustered intervals exclude zero at all three levels, while the open-ended contract-window
   interval crosses zero at the 90% level ([−2.6100, +0.1372])."* The ego-side outcome plotted in
   Fig. 5a,b is defined on exactly that open-ended window: *"The ego-side outcome is the minimum
   time-to-collision to the counterpart over the post-verdict window, which runs from the verdict to
   the end of that run's evaluated window (the counterpart-side battery instead uses the fixed
   three-second window below)."* And the 90% level is the operating level of the whole paper
   (Fig. 4 caption: *"Verdicts elsewhere in this paper are issued at the 90% level."*).

So the paper displays the ego margin under the **non-primary** window, at the level at which that
window's interval crosses zero, without an interval, and asserts the contraction in the abstract
(*"moments flagged on the assertive side are followed by tighter interactions on both sides"*) and the
Discussion (*"the real-vehicle route establishes a compression of ordinary interaction quality on both
sides"*). The pre-specified primary window's ego-margin result is never displayed.

**Why it matters.** This is the single empirical claim that converts the monitor from a descriptive
instrument into something worth deploying. At this journal a headline effect must be shown with its
uncertainty, under the pre-specified analysis, in the main display.

**Remedy.** Print case-clustered intervals on the median and upper-quartile ratios in Fig. 5a; make the
fixed three-second window the displayed primary for the ego side and relegate the open-ended window
to sensitivity; rewrite Section 2.4, the Discussion and the abstract to the level the interval supports.

---

### M2. Side-specificity ("assertive side only") is asserted with no between-side test, and the one endpoint where the comparison can be reconstructed contradicts it

**[Fig. 5a panel title; Section 2.4; Methods 4.5]**

Fig. 5a is titled *"Compression appears on the assertive side only."* Section 2.4 states *"atypicality on
the two sides is followed by different things"* and *"The complementary direction — a preference more
accommodating than the human range — carries no such signature on either side."*

No test of the assertive-versus-accommodating difference appears anywhere in the manuscript. The
argument is entirely "significant here, not significant there", which does not establish a difference.
Where the comparison can be reconstructed, it fails:

- Assertive ego-margin median ratio = 1 − 0.249 = **0.751**.
- Accommodating ego-margin median ratio = **0.81, 95% CI [0.70, 1.15]** (Methods 4.5).
- **0.70 ≤ 0.751 ≤ 1.15** — the accommodating interval contains the assertive point estimate.

The accommodating group is also the *larger* one (747 versus 472 ego moments), so this is not a power
asymmetry running in the direction the text implies. Methods 4.5 concedes the point — *"These are
non-detections at the achieved precision … and not demonstrations of equivalence"* — and the figure
title and Results prose then override the concession.

A differential exclusion compounds this. Of the flagged moments at the 90% level (Methods 4.5:
*"519 lie below the range, 869 above"*), the ego-margin panels retain 472 and 747:

- accommodating undefined margin: (869 − 747)/869 = **14.0%**
- assertive: (519 − 472)/519 = **9.1%**
- within-range: (12,711 − 11,669)/12,711 = **8.2%**

Methods 4.5 says *"the undefined fraction is similar in the two groups (9.1% of flagged vs 8.2% of
within-range moments)"* — a comparison that silently omits the accommodating group, whose exclusion
rate is 1.7× the within-range rate and is applied to the very group used to argue side-specificity.

**Remedy.** Report the assertive-minus-accommodating contrast with an interval for every endpoint in
Fig. 5 and Fig. 6b; retitle the panel; disclose the accommodating-side exclusion rate and show the
result is not driven by it.

---

### M3. The "twice as often" headline is driven ~79% by the accommodating side, for which the paper reports no consequence signature — and neither the abstract nor the Discussion says so

**[Abstract; Section 2.5; Fig. 6a side counts]**

The abstract's third empirical claim is *"automated systems flagged about twice as often."* Section 2.5:
*"the automated systems are flagged 2.1 times as often as the matched human drivers (9.8% vs 4.7%
of judgeable moments)."* The side counts printed beneath Fig. 6a allow that number to be decomposed
(90% level; H 391 | 322 of 15,102; AV 519 | 869 of 14,099):

| | assertive | accommodating | total |
|---|---|---|---|
| humans | 391/15,102 = 2.589% | 322/15,102 = 2.132% | 4.721% |
| automated | 519/14,099 = 3.681% | 869/14,099 = 6.164% | 9.845% |
| ratio | **1.42×** | **2.89×** | 2.09× |

Gap = 9.845 − 4.721 = 5.124 pp, of which the assertive side contributes 3.681 − 2.589 = 1.092 pp
(**21.3%**) and the accommodating side 6.164 − 2.132 = 4.032 pp (**78.7%**). The same pattern holds
at the 80% and 95% levels (accommodating ratios 2.16× and 2.82×; assertive 1.38× and 1.89×).

So the automated systems' excess atypicality is predominantly *over*-yielding relative to matched
humans — the side on which the paper reports no consequence signature at all. A reader of the
abstract, and of the Discussion, will take "flagged about twice as often" together with the
consequence section and conclude that automated systems are twice as often assertively deviant.
They are not, by the paper's own counts.

**Why it matters.** This is the sentence a broad readership and the press will carry away, and it is
the one place where the paper's two empirical strands (flag rate, consequence) are implicitly joined.
The decomposition is computable from the figure and should be in the text.

**Remedy.** Report the two sides separately wherever the 2.1× appears; state explicitly which side
drives it; and either explain what an accommodating-side excess means or stop letting the aggregate
ratio carry the assertive-side interpretation.

---

### M4. "Its defining population" is a different human population, in a different country, under a different apparatus

**[Abstract; Section 2.5 title and text; Discussion; Methods 4.3, 4.6]**

The abstract says *"Auditing the reference with its defining population — human drivers in the same
scenarios."* Section 2.5 is titled *"The reference is audited by the population that defined it."* The
Discussion says the reference *"flags its own defining population no more often than at home."*

Against this, in the same paper: Section 2.5's own opening sentence, *"The reference range was
learned from natural human driving recorded in other countries on open roads"*; Methods 4.3, *"The
corpora are fleet-collected in specific cities in the United States and Singapore"*; Methods 4.6, the
human arm approved by Tongji University and compensated at CNY 150 per hour. The genuine
defining population appears in Fig. 6a only as the held-out natural-driving bars.

The apparatus differs too, in every respect that could matter to an interaction-preference reading.
Methods 4.5: *"The counterpart vehicles are driven by traffic-simulation software (TESS NG)"* and
*"the simulated counterparts reach … a human driver through a head-mounted display."* The reference
was estimated from human–human pairs on open roads (Methods 4.3: *"The reference pool contains
human–human vehicle pairs only"*). Calling the closed-course VR arm "the population that defined the
reference" makes the audit sound like a self-consistency check when it is in fact a transfer test
across population, country, counterpart type and display modality simultaneously.

**Remedy.** Restrict "defining population" to the held-out natural-driving arm. Retitle Section 2.5 and
rewrite the abstract sentence to state what was actually varied.

---

### M5. A two-fold *under*-flagging of the audit arm is presented as validation, and the cross-source coverage evidence points the other way

**[Section 2.5; Fig. 6a; Methods 4.4]**

Section 2.5: *"Human drivers on the course are flagged on 4.7% of judgeable moments (713/15,102),
no more often than that native rate: carrying the reference across country, apparatus and task does
not inflate its alarms."* Check: 713/15,102 = 4.72%. The nominal outside rate at the 90% level is
10%; the native rate is 9.72%; Fig. 6a prints *"90% human: 95% CI 3.8–5.8%"*. That interval **excludes
both** 9.72% and 10%. The frozen reference therefore over-covers this population by roughly a factor
of two — it has lost about half its sensitivity in the new setting. "Does not inflate its alarms" is true
and beside the point; the finding is a calibration failure in the deflationary direction, reported as a
success.

This is not an isolated reading. The manuscript's own transfer diagnostics show the reference's
coverage is strongly source-dependent:

- Methods 4.4, leave-one-source-out at the 90% level: *"0.743 (Waymo …), 0.990 (nuPlan …),
  0.750 (Lyft …) and 0.900 (Argoverse-2 …)"* — i.e. achieved coverage ranges from 74.3% to 99.0%
  against a 90% nominal.
- Methods 4.4, within-fold conditional coverage by source: *"88.2%, 95.9%, 85.9% and 88.2% …
  against 90.3% pooled, so the pooled figure is not masking a source in which the range fails."*
  Lyft at 85.9% means an outside rate of 14.1% against a 10% nominal — a 41% relative excess in
  alarm rate. Whether that is "a source in which the range fails" is a judgement the reader should be
  allowed to make from stated numbers rather than have settled for them.
- Fig. 3c: held-out-source R² of the full state description is *"+0.026 Waymo, +0.017 Lyft, −0.195
  Argoverse-2, −0.276 nuPlan"*, i.e. negative in two of four folds, against the within-source 20.9%
  of Fig. 4c. The conditioning that produces the sharpening in Fig. 4 does not survive a source change.

The abstract nonetheless says conditioning *"gives a sharper, calibrated, auditable runtime monitor"*
with no qualifier. A monitor whose achieved coverage moves between 74% and 99% depending on the
source, and which under-flags a new population by half, is not calibrated in the sense a deployment
reader will assume.

**Remedy.** Report the human-arm flag rate against *nominal* with its interval and name the direction
of the miscalibration; reconcile it with the leave-one-source-out coverage in the same place; qualify
"calibrated" in the abstract as marginal, within-source, and empirical.

---

### M6. Internal contradiction on the global reference, and the only baseline is one the paper itself calls unusable

**[Section 2.3 vs Fig. 4 caption and Fig. 4b; Methods 4.3]**

Section 2.3 says the conditioned range *"stays informative at the strictest level, where the global range
spans essentially the whole admissible scale and can never flag a moment."* Fig. 4's caption says the
opposite of the same object: *"it still flags, but only where the reading sits at the very edge of what
the estimator can express."* Fig. 4b settles it arithmetically: the global range at the 95% level is
printed as +2.77 pp of over-coverage, so achieved coverage = 95 + 2.77 = 97.77%, so it flags
100 − 97.77 = **2.23%** of moments — on the n = 461,937 printed in the same panel, ≈10,300 flagged
moments. "Can never flag a moment" is false by the paper's own panel.

More consequentially, the global range is the *only* quantitative comparator. The paper concedes it is
a straw man (*"A global range is not a usable reference"*). Methods 4.3 mentions a second comparator
— *"We compare a global reference, an offline oracle-risk ceiling, and the context-conditioned
reference"* — and asserts the conclusion (*"the context-conditioned reference supplies essentially all of
the achievable sharpening"*) without ever printing a number for the ceiling. And no comparison is made
to any published detector, including the three the Introduction itself names as closest in aim
(refs 45, 46, 47: unexplained deviation from learned traffic behaviour; surprise-based models).
Section 2.6 compares only against fixed-threshold safety checks, and does so by showing flagged
moments have *fewer* emergencies — a demonstration of difference, not of added value.

**Remedy.** Fix the contradiction. Print the oracle-risk ceiling. Add at least two non-trivial baselines
evaluated on the same benchmark: (i) a range conditioned on kinematics alone, without the IPV, and
(ii) one published trajectory-anomaly or surprise detector. Without (i) in particular, a reader cannot
tell whether the IPV contributes anything beyond the kinematics already in z_t.

---

### M7. The central measurement has no construct validation, no hyperparameter sensitivity analysis, and a per-moment instability the paper measures but does not confront

**[Methods 4.1, 4.4, 4.5; Fig. 2b; Figs ED1, ED2]**

*Construct.* The IPV is supported by two references (24, 25), both by the authors. Nothing in this
paper shows that the quantity corresponds to anything a human observer would call a social preference.
The paper is candid that the human-preference route is future work, but that leaves the entire
construction resting on an unvalidated scalar.

*Frozen configuration, no sensitivity.* Methods 4.1 fixes: seven candidates on
θ ∈ {−3,…,3}×π/8; a ten-frame (1 s) window with at least four observed frames; a Gaussian
likelihood with σ = 0.1 m; and the reliability rule *"the reading is discarded as near-uniform when the
largest normalised weight falls below 0.20 (against 1/7 ≈ 0.14 for exactly uniform weights)."* No
sensitivity analysis for σ, window length, grid resolution or the 0.20 threshold appears anywhere.
Every downstream quantity — readability rate, flag rate, consequence contrast — is a function of these
four numbers, and the 0.20 rule in particular sits very close to the uniform value 0.143.

*Per-moment instability.* Fig. 2b reports that where the reading is most readable it moves
0.30–0.31 rad between consecutive frames. Figs ED1 and ED2 make this visible: the reading swings
across most of the 2.36 rad admissible span between adjacent 0.1 s frames. Methods 4.4 states
*"no result in this paper uses it [the persistence layer], every reported verdict and flag rate is
per-moment."* So every headline number is computed on a signal with that much frame-to-frame
movement, and the paper presents the movement statistic only as a caution about *episode summaries*,
never as a caution about the per-moment verdicts on which everything rests.

*Boundary pinning.* The reading is a weighted mean over the grid and therefore lives on
[−3π/8, 3π/8] = [−1.178, +1.178] rad. Methods 4.5 states that at the 90% level *"the lower edge is
negative in essentially every situation (median −1.03 rad)."* In the median situation the entire
assertive-side detection region is therefore 1.1781 − 1.03 = **0.148 rad wide = 6.3% of the 2.356 rad
admissible span**, and in Figs ED1 and ED2 the assertive flags visibly sit at the grid floor (≈ −1.17 rad).
This has three consequences the paper does not discuss: assertive flags are largely readings for which
the likelihood collapsed onto the most extreme candidate (the least informative configuration of the
estimator); the assertive side is structurally under-flagged relative to the accommodating side
(3.68% versus 6.16% at a nominally symmetric 90% range), which is a plausible mechanical
explanation for the asymmetry in M3; and the "flag" is not a graded departure but a saturation event.

**Remedy.** Sensitivity analyses over σ, window, grid and the 0.20 rule. Report the fraction of
assertive-side flags within (say) 0.05 rad of the grid floor and the distribution of the 90% lower edge
relative to the floor. Recompute every headline rate under a minimal persistence rule and show the
conclusions survive. Add at least one external check on the IPV construct.

---

### M8. Fig. 5b's caption misdescribes the panel, and the "no signature on the accommodating side" sentence is contradicted by the panel beside it

**[Fig. 5b; Section 2.4]**

Fig. 5b's caption: *"All four thresholds were tested and all four are displayed with their intervals; the
< 3 s interval admits no difference."* The panel prints **two** < 3 s intervals: −3.14 [−8.33, +3.15]
(assertive) and −4.74 [−8.63, −0.59] (accommodating). The second **excludes zero**. The caption's
singular claim is false for one of the two displayed series.

This is not cosmetic. Section 2.4 closes the emergency-tail paragraph with *"The complementary
direction — a preference more accommodating than the human range — carries no such signature on
either side."* But on the ego-margin emergency thresholds the accommodating side shows all four
intervals excluding zero (−1.07, −2.64, −3.49, −4.74 pp), i.e. **one more supported threshold than the
assertive side**. Whatever the intended scope of "no such signature", as placed it tells the reader
something the adjacent panel refutes.

**Remedy.** Make the caption side-specific. Separate the "body-of-distribution compression" signature
from the "emergency tail" signature explicitly in the text, and state which side shows which.

---

### M9. Declarations, pre-specification and human-subjects reporting are incomplete for a paper that leans this heavily on pre-specification

**[Methods 4.5, 4.6; Data Availability; Code Availability; Fig. 5 caption]**

*Pre-specification.* The manuscript rests on pre-specification at least six times: *"a split frozen before
this study"*; *"the fixed three-second window is the pre-specified primary"*; *"Its endpoints, windows,
thresholds, clustering unit and resampling settings were fixed for the assertive-side comparison and
were not altered afterwards"*; *"the analysis plan, which was specified and frozen before any outcome
was examined"*; *"display criteria fixed in advance"*; *"the selection score fixed together with them"*.
No registration, no deposited plan, no date, no third-party time stamp is offered. These claims are
load-bearing for the entire consequence section and are currently unverifiable.

*Human participants.* No demographics (age, sex, driving experience), no sample-size justification for
n = 20, and no debriefing statement despite a deliberately withheld purpose (*"they were not told that
their runs would serve as a human reference against automated systems"*). Order is fully confounded
with scenario: *"Each driver completed the scenario set as a single fixed sequence, identical across
drivers"* — which directly affects the per-scenario comparison in Fig. 6c, where learning and fatigue
cannot be separated from scenario identity.

*Third parties.* Nineteen competitor systems are analysed and compared unfavourably with humans.
The Competing Interests statement covers the authors' role in operating the benchmark, but nothing
states whether the entrants consented to this secondary analysis or how the systems are de-identified.

*Availability.* Data Availability names no repository for the derivatives (*"will be made available upon
publication"*) and does not mention the Source Data files that seven of the eight captions promise.
Code and the "frozen configuration … released with the code" — which is required to reproduce any
number in the paper — are not available now.

*Production defect.* Fig. 5's caption is **truncated mid-sentence at the page break**: page 14 ends
*"Error bars and bracketed values are 95% confidence intervals from 1,000"* and page 15 resumes the
Results text. The resample count is therefore never stated, and Fig. 5 is the only display item lacking
its "Source data are provided as a Source Data file" sentence.

**Remedy.** Deposit and cite the frozen analysis plan with a verifiable date. Complete the
human-participant reporting (demographics, power, debriefing, order). Add a statement on the use of
third-party competitor runs. Name repositories. Fix the truncated caption.

---

### M10. Abstract, Discussion and stated limitations do not cover the paper's real exposure; the generality claim is unsupported

**[Abstract; Discussion; Methods scope note; Section 2.4]**

The Discussion's limitations are: harm is bounded not confirmed, and the evidence covers *"pairwise
vehicle–vehicle interactions at mapped conflict points"*. Both are real and honestly stated. The
following are not mentioned in the Discussion at all:

- **Duty cycle.** *"Across its 67,861 candidate moments the monitor returns a verdict on 14,099
  (20.8%)"*, and in 36 of 267 runs (13.5%) it never speaks. A runtime monitor that is silent on four
  moments in five, and on one run in seven, is a materially different proposition from what the abstract
  describes, and this belongs in the abstract, not only in Section 2.4.
- **No cross-source transfer.** Fig. 4's caption concedes *"Transfer to a data source not seen during
  fitting is not established"*, but the Discussion's summary of the same result reads *"A reference range
  conditioned on the current observable situation is sharp and near-nominally calibrated"* with no
  qualifier (see M5).
- **No real-time evaluation.** Methods' scope note: *"end-to-end real-time operation on vehicle
  hardware is not evaluated in this paper."* For a paper titled "Online monitoring", the absence of any
  latency or compute measurement should be in the Discussion, not buried in a Methods preamble.
- **Construct validity of the IPV** (M7).

*Venue fit.* The generality claim is a single sentence: *"Beyond driving, the framework offers a
template for monitoring whether autonomous agents behave within human normative ranges online,
under uncertainty and distribution shift, while being explicit about when a social judgement should be
withheld."* Nothing in the paper tests, instantiates or even sketches that template outside driving.
The statistical machinery is entirely off-the-shelf and the paper says so (split conformal, CQR, reject
option; *"Conformal calibration of trajectory atypicality has precedent in unconditional surveillance
settings"*). The conformal layer moreover does nothing here: *"The fitted radii are near zero
(c_α = 1.4 × 10⁻³, 1.2 × 10⁻⁶ and 0.0 rad at the 80%, 90% and 95% levels), so the conformal step
finds essentially nothing to repair"* — at the 95% level the radius is exactly zero, so the "conformal"
interval *is* the raw quantile-model interval, and the word is carrying rhetorical weight
("calibrated, auditable") that the fitted object does not.

What is genuinely transferable is not the estimator or the conformal wrapper but (i) the four-way
separation of interaction opportunity / readability / human support / deviation with an explicit
abstain-with-reason-code, and (ii) the audit design in which a frozen normative instrument is turned
back on a matched population of the kind that defined it. Those are the contributions a broad
machine-intelligence readership can take away, and the paper currently sells them under a
"conditional conformal monitor" heading that a reviewer from the conformal-prediction community
would find thin.

**Remedy.** Either add one non-driving instantiation, or demote the template sentence to explicitly
marked speculation and rebuild the significance claim around the abstention taxonomy and the audit
protocol. Bring the duty cycle, the transfer boundary and the absence of real-time evaluation into the
abstract and the Discussion.

---

## 3. Minor issues

1. **Readability denominator.** Methods 4.5 reports the naturalistic readability rate as 70.3%
   (*"70.3% vs 55.3%"*), but the anchor-row accounting in Methods 4.4 gives 3,202,646 readable of
   4,497,368 rows = **71.21%**. The components sum exactly (3,202,646 + 1,275,480 + 17,416 +
   1,826 = 4,497,368), so the discrepancy is presumably a different denominator. State it.
2. **Fig. 3a**: the middle band's estimate (+0.001, n = 22,833) is given with no confidence interval in
   the text, unlike the two flanking bands, and no visible interval in the panel.
3. **Fig. 3b**: the x-axis is *"Difference in reading (rad)"* but the reference category of the contrast
   ("merge/pass" and "same-direction" relative to *what*?) is never stated in the caption or Methods.
4. **Fig. 2b** prints *"uncertainty not shown"* in-panel and gives no n. The claim it supports
   ("readable does not mean settled") is used in the Results as an established fact.
5. **Fig. 2c**: the in-panel annotation attributes *"0.26 rad apart"* to "the first two rules", whose
   printed medians are +0.08 and +0.00; the caption instead attributes 0.26 rad to the rules on
   average. Reconcile.
6. **Fig. 6 funnel**: the automated arm's Gate-1 box reads *"count unavailable / 55.3% pass"*, yet
   0.553 × 67,861 = 37,527 is computable from the same figure. This reads as a pipeline artefact in
   a headline display.
7. **Fig. 6b caption** says the panel expresses *"each quantity of Fig. 5"*; it shows six of roughly a
   dozen Fig. 5 endpoints, omitting the two counterpart endpoints that were unsupported there
   (net heading change, peak yaw rate). Selective, and should be stated as a selection.
8. **Fig. 6c** labels three points (A3, A5, B1) that the caption never explains; markers overlap so the
   reader cannot verify 15 distinct scenarios; and *"uncertainty not shown"* is printed while the text
   draws a 15-of-15 conclusion from the panel. A sign test (p = 2⁻¹⁵ ≈ 3 × 10⁻⁵ one-sided) would cost
   nothing and would make the claim checkable.
9. **"Side" is overloaded**, sometimes within one sentence. The abstract's *"moments flagged on the
   assertive side are followed by tighter interactions on both sides"* uses "side" first for a side of the
   reference range and then for a vehicle. Use "assertive/accommodating side" for the range and
   "ego/counterpart" for the vehicles throughout.
10. **Fig. ED1 legend** labels only the upper marker *"above range · atypical"* while the lower is
    *"below range · more assertive"*; the caption calls both sides atypical. Align them.
11. **Selection-bias sentence contradicts its own arithmetic.** Methods 4.5: *"The runs dropped at the
    last step are not the eventful ones — 7 of those 29 contain any flagged moment, against 132 of the
    175 retained — so the analysed set is not the flag-rich remainder of a larger pool."* But
    7/29 = 24.1% and 132/175 = 75.4%: the retained set is **3.1× as likely** to contain a flagged
    moment. The benign reading (nothing eventful was discarded) is defensible; the stated conclusion is
    not.
12. **Braking endpoint fragility.** Methods 4.5: *"Weighted equally per run, the assertive-side
    counterpart braking contrast does not reach significance (p = 0.4704)."* That endpoint nonetheless
    appears in Fig. 6b as part of the "shared consequence signature" with no such caveat.
13. **Selective reporting of the two permutation tests.** Section 2.4 cites only the passing one (*"the
    association survives a placebo test that reassigns whole flag sequences"*, p = 0.0199 in Methods);
    the null case-level label permutation on the same battery (p = 0.1493) appears only in Methods.
14. **"Frozen situation cells"** (Methods 4.5) is the confounding control for the entire consequence
    analysis and is never defined — number of cells, construction, or how the comparison is stratified
    or weighted within them.
15. **Effect size and practical significance.** The counterpart contrast is a difference in medians of
    **+1.41 km h⁻¹ [+0.08, +3.37]** (2.74 versus 1.33 km h⁻¹). Reported as "roughly twice the routine
    speed reduction" this sounds substantial; in absolute terms it is 0.4 m s⁻¹, and the interval nearly
    touches zero. Both framings are given in the caption, which is good practice, but the main text and
    abstract use only the ratio framing.
16. **Absolute margins are given only for the quartile that did not move.** Fig. 5a prints the lower
    quartile in seconds (4.09 versus 4.18 s) but the median and upper quartile only as percentages.
    Reading the plotted curves against the printed decade ticks, the median moves from ≈8.9 s to
    ≈6.7 s and the upper quartile from ≈22 s to ≈12 s. Both the before and after values are far outside
    any operationally meaningful margin, which is exactly why the absolute numbers should be printed.
17. **Fig. 5a caption**: *"The accommodating distribution tracks the within-range one across its whole
    body."* Methods 4.5 reports its median at 0.81× the within-range median — a 19% shortfall, four
    fifths of the assertive-side contraction. That is not "tracking".
18. **Fig. ED2 funnel.** Of the 20 runs with ≥5 contiguous flagged frames, 10 show a counterpart speed
    drop ≥20% and only 2 also show a near-constant ego speed. The mechanism the case illustrates is
    therefore present in 2 of 20 clustered-flag runs. The caption discloses the funnel honestly but the
    Results reference to it (*"a single run … is shown for illustration"*) does not.
19. **Fig. ED2a** is compressed to a near-horizontal strip (x spans 35 m, y spans 4 m) and is a small
    fraction of the area given to panels b and c.

---

## 4. Questions to the authors

1. For the assertive-side ego-margin median and upper-quartile ratios in Fig. 5a: what are the
   case-clustered 95% intervals, under **both** the open-ended and the pre-specified fixed three-second
   window? If the open-ended-window interval is [−2.6100, +0.1372] at the 90% level as Methods 4.5
   states, on what basis does the main text assert contraction as fact?
2. Why are the "Ego margin (median)" and "Ego margin (upper quartile)" rows the only entries in
   Fig. 6b without an asterisk on either arm? By the caption's own rule this means their intervals admit
   parity. If that is not the intended reading, what is?
3. What is the assertive-minus-accommodating contrast, with an interval, for every endpoint in Fig. 5?
   Without it, on what evidence does the panel title "Compression appears on the assertive side only"
   rest?
4. Why does 14.0% of the accommodating group lack a defined post-verdict margin against 8.2% of the
   within-range group, and does the accommodating-side null survive a sensitivity analysis on that
   exclusion?
5. Four fifths of the AV-versus-human flag-rate gap is on the accommodating side (4.03 of 5.12 pp at
   the 90% level, from the counts printed under Fig. 6a). What is the intended interpretation of an
   automated fleet that over-yields relative to matched humans, and how should "flagged about twice
   as often" be read in light of that decomposition?
6. The human arm's flag rate (4.7%, 95% CI 3.8–5.8%) is significantly below both nominal (10%) and
   the native rate (9.72%). Why is a two-fold loss of sensitivity in the new setting reported as evidence
   that the reference transfers?
7. How do you reconcile the claim that the reference transfers to a new country with your own
   leave-one-source-out coverage of 0.743 (Waymo) and 0.750 (Lyft) at a 90% nominal, and with the
   negative held-out-source R² in two of four folds (Fig. 3c)?
8. What fraction of assertive-side flags have a reading within 0.05 rad of the grid floor (−3π/8)?
   Given that the median 90% lower edge is −1.03 rad, the median assertive detection region is
   0.148 rad — 6.3% of the admissible span. How should a flag at the estimator's saturation point be
   interpreted?
9. What are the sensitivity results for σ = 0.1 m, the ten-frame window, the seven-point grid, and the
   0.20 near-uniform threshold? How much do the flag rate and the consequence contrasts move?
10. Do any of the headline results survive a minimal persistence rule (e.g. three consecutive verdicts),
    given that the reading moves 0.30 rad between consecutive frames where it is most readable?
11. Where is the analysis plan that "was specified and frozen before any outcome was examined", and
    can its date be independently verified?
12. Section 2.3 says the global range "can never flag a moment"; Fig. 4's caption says "it still flags",
    and Fig. 4b implies it flags 2.23% of moments at the 95% level. Which is correct?
13. What is the offline oracle-risk ceiling numerically, at each of the three levels?
14. What are the "frozen situation cells" — how many, how constructed, and how is the flagged-versus
    within-range comparison stratified within them?
15. Were participants debriefed about the withheld purpose of the study, and what were the
    participant demographics and the basis for n = 20?
16. Did the entrants whose systems are analysed consent to this secondary analysis?
17. Methods 4.5 gives the naturalistic readability rate as 70.3% while Methods 4.4's accounting gives
    3,202,646/4,497,368 = 71.21%. What are the two denominators?

---

## 5. Prioritised revision requests

**Tier 1 — the paper cannot be assessed without these**

1. Report intervals on the ego-margin median and upper-quartile ratios, under the pre-specified primary
   window, in Fig. 5a; align Section 2.4, the Discussion and the abstract with what those intervals
   support (M1).
2. Add and report the assertive-versus-accommodating contrast with an interval for every endpoint;
   retitle Fig. 5a; disclose the differential exclusion (M2).
3. Decompose the 2.1× flag-rate ratio by side wherever it appears, in the abstract included, and state
   that ~79% of the gap is accommodating-side (M3).
4. Correct "its defining population" throughout; describe the human arm as a matched human
   population under a different country, counterpart and display apparatus (M4).
5. Report the human-arm flag rate against nominal with its interval, name the direction of
   miscalibration, and reconcile it with the leave-one-source-out coverage in the same paragraph (M5).
6. Fix the Section 2.3 / Fig. 4 contradiction on the global range (M6).
7. Correct the Fig. 5b caption and the "no such signature on either side" sentence (M8).
8. Fix the truncated Fig. 5 caption and state the resample count (M9).

**Tier 2 — required for the contribution to stand at this venue**

9. Add two baselines on the same benchmark: a kinematics-only conditional range without the IPV, and
   one published trajectory-anomaly or surprise detector (M6).
10. Sensitivity analyses over σ, window, grid and the 0.20 rule; report the boundary-pinning fraction
    and the lower-edge distribution relative to the grid floor; recompute headline rates under a minimal
    persistence rule (M7).
11. Deposit and cite the frozen analysis plan with a verifiable date; complete the human-participant
    reporting; add a third-party-data statement; name repositories (M9).
12. Move the duty cycle (20.8%; 36 silent runs), the transfer boundary and the absence of real-time
    evaluation into the abstract and the Discussion limitations (M10).

**Tier 3 — presentation**

13. Print the oracle-risk ceiling; define "frozen situation cells"; state Fig. 3b's reference category;
    give the Fig. 3a middle-band interval; add uncertainty or n to Fig. 2b; explain the Fig. 6c point
    labels and add the sign test; resolve the "side" terminology collision; align the Fig. ED1 legend
    with its caption; fix the Fig. 6 "count unavailable" box; correct the retained-versus-dropped-run
    sentence in Methods 4.5.

---

## 6. Acceptance probability

- **(a) As submitted: 4%.** The headline consequence claim is not supported by the manuscript's own
  uncertainty reporting, the abstract misdescribes the audited population, and the "twice as often"
  result is composed opposite to the way the paper's narrative implies. Any of the three would
  ordinarily be decisive at this journal.
- **(b) After a competent major revision: 35%.** Most defects are reporting defects and are fixable
  without new data collection. The risk is that the honest versions of M1 and M2 leave the consequence
  section with a counterpart-side association of ~1.4 km h⁻¹ and no supported ego-side effect, which
  would materially weaken the paper's principal advance. Whether the remaining contribution — the
  abstention taxonomy, the situation-conditioned reference, and the audit protocol — clears the bar
  will depend on whether Tier-2 baselines show the IPV adds something a kinematic monitor does not.

---

## 7. Recommendation

**Major revision.**
