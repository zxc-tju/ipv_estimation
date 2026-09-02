# Referee report — "Online monitoring of socially compliant autonomous driving"

**Remit: methodology and statistics, extended to runtime verification and formal semantics.**

All quotations are from the submitted PDF; figure numbers follow the manuscript's own numbering
(Fig. 1 concept, Fig. 2 measurability, Fig. 3 context, Fig. 4 reference range, Fig. 5 consequence,
Fig. 6 human arm, ED1 reference band, ED2 case example). Arithmetic I report is recomputed from
numbers printed in the manuscript or its figures.

---

## 1. Summary assessment

The paper reframes social compliance as an online membership test: is the ego's current
interaction preference value (IPV) inside a situation-conditioned human reference range, or should
the monitor abstain? The reframing is genuinely interesting, the abstention discipline is
principled, and the disclosure of limitations in Methods is unusually honest for this literature.

What is delivered is weaker than what is claimed. The monitored quantity is a weighted average
over seven candidate preferences spanning ±3π/8; the manuscript's own Extended Data show flagged
readings pinned at those extremes, and Methods 4.5 places the median lower edge of the 90% range
at −1.03 rad, above which only one grid point lies. An assertive-side flag therefore looks like
estimator saturation rather than a measured preference. The calibration is marginal over dependent
moments, its conformal radius is ~0 (so the step is inert), and the paper's own leave-one-source-out
coverage (0.743, 0.750 at nominal 0.90) shows it does not transport — while Section 2.5 asserts it
does. The consequence analysis conditions on counterpart behaviour that is an input to the verdict.
There is no baseline monitor and no detection performance. I cannot recommend publication.

---

## 2. Major weaknesses

### M1. The estimator saturates at its own candidate grid, and a "flag" appears to be that saturation rather than a measurement
**[Methods 4.1; Methods 4.5; Fig. ED1; Fig. ED2c; Fig. 1c]**

*What is wrong.* Methods 4.1 states: "The frozen configuration uses seven candidate preferences
θ_k ∈ {−3, −2, −1, 0, 1, 2, 3} × π/8". The extreme candidates are therefore ±1.178 rad and the grid
spacing is 0.393 rad. Methods 4.5 states: "at the 90% level the lower edge is negative in
essentially every situation (median −1.03 rad)". Only one grid point (−1.178) lies below −1.03. So
in the median situation an assertive-side flag requires the candidate weights to be dominated by
the single most assertive candidate in the grid — i.e. the estimator hitting the boundary of its
own parameterisation.

The figures confirm this directly. In Fig. ED1 every red marker — both the below-range triangles
around 11–13 s and the above-range triangles at 3.6, 5.0, 6.5 and 12.1 s — sits at approximately
∓1.18 rad, flat on the grid extremes, while unflagged readings vary continuously in between. In
Fig. ED2c the six flagged frames of the showcase stretch form a perfectly flat line at ≈ −1.17 rad.
The ED1 caption's own worked value, "The two outlined frames carry the same reading, −0.393 rad",
is −π/8 to three decimals.

The likelihood makes this expected. Methods 4.1 specifies "a Gaussian trajectory likelihood
ℓ_{a,k} ∝ exp{−MSE_k/(2σ²)} with σ = 0.1", with MSE in m² over a ten-frame window. Two candidates
differing by 0.3 m RMS over the window then differ in likelihood by a factor of ≈ e^4.5 ≈ 90. The
posterior over candidates is near-degenerate almost everywhere, so θ̂ = Σ π_k θ_k is effectively a
seven-level argmax, and the reliability gate — "the reading is discarded as near-uniform when the
largest normalised weight falls below 0.20 (against 1/7 ≈ 0.14 for exactly uniform weights)" —
excludes only the fully non-identified case, not the saturated one.

*Why it matters.* If flags are grid-boundary events, then every headline quantity in the paper —
the 9.8% and 4.7% flag rates, the 2.1× ratio, the consequence contrasts — is a statistic about
where a seven-point grid was censored, not about social preference. The width comparison in Fig. 4a
is affected too: a conditioned 90% range of 1.87 rad on a 2.36 rad candidate span covers roughly
five of the seven grid points, so the "reference range" is a device that admits five levels and
flags two. Nothing in the manuscript rules this out, and nothing in it reports the distribution of
θ̂.

*Remedy.* Report the empirical distribution of θ̂ over accepted moments (histogram, and the mass
within 0.02 rad of each grid point); report the fraction of assertive-side and over-yielding flags
whose θ̂ lies at or adjacent to a boundary candidate; report the distribution of max_k π_k. Then
re-run every headline rate on (i) a finer grid over the same span and (ii) a grid extended to the
declared domain [−π/2, π/2], and show the rates are stable. If they are not, the paper's central
object needs redefinition before any of its claims can stand.

---

### M2. The calibration claim is marginal over dependent moments, the conformal step is inert, and the paper's own numbers show the guarantee does not transport — while Section 2.5 says it does
**[Methods 4.4; Fig. 4b; Fig. 3c; Fig. 4 caption vs Section 2.5]**

*What is wrong.* Four separate problems compound.

(a) *Dependence and the unit of validity.* Methods 4.4 is candid — "Moments within an interaction
are dependent, so the coverage statement is marginal over accepted moments — not a per-interaction,
sequential or conditional guarantee" — but the construction is then not adjusted for it. Splits are
by whole scenes, so a test moment's cluster-mates are all in test and none in calibration:
exchangeability holds at the scene level, not the moment level, which is the level at which the
nonconformity scores are pooled and the radius chosen. The effective sample size for the coverage
statement is the number of scenes/cases (26,828 interaction cases across four folds), not the
461,937 moments the figure quotes.

(b) *No uncertainty on the coverage numbers.* Fig. 4b reports "+0.03, +0.28, +0.57" percentage
points to two decimals with **no interval of any kind** on any marker. A clustered-data coverage
estimate reported without a cluster-bootstrap interval cannot support the claim in the panel title
("Coverage within 0.6 pp of nominal") or in the abstract ("calibrated").

(c) *The conformal step does nothing.* "The fitted radii are near zero (c_α = 1.4 × 10⁻³,
1.2 × 10⁻⁶ and 0.0 rad at the 80%, 90% and 95% levels), so the conformal step finds essentially
nothing to repair in the situational quantile model." With a radius of zero the deployed interval
*is* the raw gradient-boosted quantile interval. The distribution-free finite-sample language that
"calibrated by split-conformal prediction" carries for a reader is therefore doing no work here;
the paper is relying on a boosted quantile model being well calibrated, which is exactly what
conformal exists to avoid relying on.

(d) *It demonstrably does not transport, and the paper says both things.* Methods 4.4 reports
leave-one-source-out coverage "at the 90% level ... 0.743 (Waymo ...), 0.990 (nuPlan ...), 0.750
(Lyft ...) and 0.900 (Argoverse-2 ..., with 44.3% held-out abstention...)". Two of four sources
lose 15 coverage points — a false-alarm rate of 25% where 10% is nominal — and the one that holds
does so by abstaining on nearly half its moments. Even *within* the fitted sources, conditional
coverage runs 85.9%–95.9% ("88.2%, 95.9%, 85.9% and 88.2% ... against 90.3% pooled"), a 10-point
spread that the text describes as "not masking a source in which the range fails"; Lyft at 85.9%
is a 41% relative inflation of the alarm rate. Fig. 3c shows the same failure for the mean
function: held-out R² of "+0.026 Waymo, +0.017 Lyft, −0.195 Argoverse-2, −0.276 nuPlan". And the
Fig. 4 caption itself concedes: "Transfer to a data source not seen during fitting is not
established and is reported as a boundary of the present monitor." Section 2.5 nonetheless
concludes that "carrying the reference across country, apparatus and task does not inflate its
alarms." These two statements cannot both be the paper's position.

*Why it matters.* The monitor's entire claim to be more than an ad-hoc detector is the calibration.
As constructed, the guarantee is (i) marginal, (ii) over a population of human natural-driving
moments from US/Singapore fleet corpora, (iii) reported without uncertainty, and (iv) shown by the
authors' own leave-one-source-out analysis not to hold under exactly the kind of shift the
real-vehicle deployment represents. The 9.8%/4.7% rates on the benchmark are empirical frequencies
in a shifted population, not controlled error rates.

*Remedy.* Re-do the calibration cluster-aware: (i) report per-interaction coverage and a
cluster-bootstrap CI over cases/scenes for every number in Fig. 4b; (ii) implement and compare a
cluster-conformal or subsampled-one-moment-per-interaction variant; (iii) state explicitly, in the
main text, that no coverage claim is made on the benchmark population and give the empirical
alarm rate there with its interval; (iv) reconcile the Fig. 4 caption with Section 2.5.

---

### M3. The consequence analysis is confounded by construction: counterpart behaviour is an input to the verdict and the outcome window overlaps the same manoeuvre
**[Section 2.4; Methods 4.5; Eq. 1; Fig. 5; Fig. ED2b]**

*What is wrong.* Eq. (1) conditions the candidate likelihood on the counterpart trajectory:
ℓ_{a,k}(t) = p(X_a(W_t) | X̃_a(W_t; θ_k), X_j(W_t), M). The counterpart's behaviour over the
one-second window *ending* at t is therefore an input to the estimate that produces the verdict at
t. The counterpart-side outcome is then read "over a fixed three-second window from the same
moment". If the counterpart begins yielding at t − 0.5 s, the ego's unchanged trajectory becomes
best explained by a more assertive candidate (flag), and the same continuous deceleration is then
counted as the outcome.

This is visible in the paper's own showcase. In Fig. ED2b the counterpart's speed falls from ≈ 4.9
to ≈ 1.8 m s⁻¹ **before** the first red alert band begins, reaching its floor essentially at alert
onset; the caption's headline contrast, "The counterpart median moves from 4.31 to 1.40 m s⁻¹",
compares a 2 s pre-alert baseline to a stretch across which most of the decline had already
occurred.

The ego side has a parallel problem of a different kind: θ̂(t) is computed from the kinematics of
the window ending at t, and the ego outcome is "the minimum time-to-collision to the counterpart
over the post-verdict window" beginning at t. Adjacent windows of the same continuous motion are
strongly autocorrelated, so a smaller subsequent TTC at moments where the recent motion was
assertive is close to mechanical.

The manuscript's defence does not reach either issue. Section 2.4 argues: "The two sides are
measured from different sources and different quantities, so their agreement is a convergence
rather than a restatement." That addresses data provenance, not temporal and mechanical dependence.
Methods 4.5's non-anticipation statement — "the trajectory samples that produce a verdict (the
one-second window ending at that moment) precede every outcome sample" — establishes that the
*ego's* verdict inputs precede the outcome; it does not address that the counterpart's pre-verdict
behaviour is a verdict input while its post-verdict behaviour is the outcome.

The placebo does not resolve it either. The exposure placebo reassigns "whole exposure trajectories
across scenario runs" (p = 0.0199 from 200 draws), which controls for run-level composition but not
for within-run kinematic autocorrelation. The test that would control for it — a within-run
time-shift placebo — is not run. Notably, "A case-level label permutation on the same battery does
not reach significance (p = 0.1493)"; of the two permutation schemes reported, the paper designates
the one that reaches significance as "the test specific to flag timing".

*Why it matters.* The consequence analysis is the only evidence in the paper that a flag
corresponds to anything at all. If it is partly a restatement of the estimation window, the paper
has an interesting construct and no demonstration that it means anything.

*Remedy.* (i) Re-run every consequence contrast with the counterpart-side outcome window starting
at t + 1 s or later, so that no outcome sample overlaps any verdict input. (ii) Add a within-run
time-shift placebo (flags displaced ±1, ±2, ±3 s inside the same run). (iii) Add the matched
kinematic-baseline comparison of M5. (iv) Report the exposure-placebo p with the granularity of 200
draws made explicit (resolution 1/201) and, preferably, with ≥ 2,000 draws.

---

### M4. The human-versus-machine comparison is confounded by apparatus, and the headline 2.1× is dominated by the deviation direction for which the paper reports no signature
**[Section 2.5; Fig. 6a; Methods 4.5, 4.6]**

*What is wrong.* Two separate problems.

(a) *Apparatus.* Methods 4.5 states: "the ego is a real vehicle driven at a test site, and the
simulated counterparts reach an automated system through its perception interface and a human
driver through a head-mounted display." The human arm additionally drove a closed course with "a
safety supervisor on board who could intervene or terminate a run at any time". A head-mounted
display, a supervisor and a staged conflict are all expected to make a driver more conservative.
The comparison therefore contrasts machine behaviour with human behaviour *under a different
sensory and social apparatus*, and the paper offers no manipulation check, no calibration of the
HMD condition against on-road behaviour, and no sensitivity analysis.

Fig. 6a makes the alternative reading unavoidable. The natural-driving reference population is
flagged at 20.0 / 9.7 / 4.4 per cent at the 80/90/95% levels; the automated systems at 19.5 / 9.8 /
5.0; the on-course humans at 11.2 / 4.7 / 2.2. The automated systems sit *at* the reference's own
rate at every level; the on-course humans sit at roughly half it at every level, and the printed CI
("90% human: 95% CI 3.8–5.8%") excludes 9.72%. On the paper's own instrument, it is the human arm
that is the outlier — a reading the text partly concedes ("all three rates belong together, and no
pair of them tells the story alone") and then sets aside in favour of the pair in the panel
annotation, "AV = 2.1× humans", and in the abstract, "automated systems flagged about twice as
often".

(b) *Direction.* Fig. 6a prints the side counts. At the 90% level: humans 391 assertive | 322
accommodating of 15,102 judgeable; AVs 519 | 869 of 14,099. Recomputing: the assertive-side rate
ratio is 3.68% / 2.59% = **1.42×**; the over-yielding rate ratio is 6.16% / 2.13% = **2.89×**. The
same pattern holds at 80% (1.38× / 2.16×) and 95% (1.89× / 2.82×). The headline 2.1× is therefore
driven predominantly by *over-yielding* flags — the direction of which Section 2.4 states: "The
complementary direction—a preference more accommodating than the human range—carries no such
signature on either side." The paper's most quotable number is largely a count of the deviation
type it has shown to be behaviourally inert.

(c) *No uncertainty on the comparison.* Only one bar in Fig. 6a carries a whisker (the human 90%
rate). There is no interval on any AV rate, none on the natural-driving rate, and none on the 2.1×
ratio itself. Fig. 6c is annotated "uncertainty not shown", and the caption confirms "per-scenario
points are shown without intervals". The supporting "15 of 15 scenarios" is a sign test on
scenario-level points that are themselves pooled over 19 systems and 20 drivers; a uniform
apparatus effect would produce 15/15 exactly as behaviour would.

*Why it matters.* Section 2.5 is the paper's external-validity chapter and the source of its most
newsworthy sentence. As it stands the sentence is confounded by protocol, decomposes into the wrong
direction, and carries no interval.

*Remedy.* Report the 2.1× ratio with a cluster-bootstrap interval; decompose it by side in the main
text; report per-driver flag rates for the 20 participants (to show the human rate is not carried
by a subset); and either provide evidence that the HMD/supervisor condition does not depress the
flag rate, or restate the comparison as bounded by apparatus and remove it from the abstract.

---

### M5. There is no baseline monitor, no detection performance, and the one available human-preference endpoint was pre-excluded
**[Sections 2.3–2.6; Methods 4.3, 4.6]**

*What is wrong.* The only comparator anywhere in the paper is a global (unconditioned) IPV range,
and Fig. 4a shows that comparator to be degenerate at both levels at which the paper issues
verdicts: its width is 100% of the admissible range at the 90% and the 95% level. Beating a
reference that spans the whole scale is not evidence that the construction is good.

Absent are the comparisons that would decide whether the IPV pipeline earns its complexity:

- A **kinematic conditional-quantile monitor**: fit the same quantile machinery on the same z_t but
  with a directly observable behavioural scalar as target (time gap at the conflict point, closing
  rate, longitudinal acceleration) and flag the same tail fraction. Does it flag the same moments,
  at the same rate, with the same consequence signature? This is the decisive control, because
  Fig. 4c reports that the situation explains only 20.9% of the reading's variance — so a flag is
  by construction a large residual of a model whose inputs are 22 kinematic channels plus four
  categorical descriptors.
- A **published monitor**: refs [44]–[47] are positioned as the nearest neighbours and none is run
  on this benchmark. The Introduction's claim that "None of these reads a social preference, refers
  it to a calibrated situation-conditioned human range..." is a novelty claim, not a performance
  claim, and is currently the paper's only comparative statement.
- **Any detection performance.** No ROC, no sensitivity/specificity, no precision at any operating
  point, against any label.

On the last point: Methods 4.6 records that "Official competition scores, harm labels and
preference ratings were excluded as endpoints in the analysis plan". The Introduction's validation
route is "human reference range → conditional social atypicality → human-dispreferred behaviour →
interaction-harmful behaviour". The paper had access to preference ratings — the third stage of its
own route — and excluded them by design. The conflict-of-interest rationale is understandable, but
the consequence is that the promised route has no evidence at its one testable link, and the reader
is not told this trade-off was made.

*Why it matters.* At this journal a new runtime monitor must be shown to beat the obvious cheaper
thing. Without a baseline, no reader can tell whether the inverse-planning layer contributes
anything over an anomaly detector on the kinematics that are already in z_t.

*Remedy.* Add the kinematic-baseline monitor and report flag agreement (Cohen's κ), rate, and the
full consequence battery for it side by side with the IPV monitor. Add at least one published
monitor on the same benchmark. Report the preference-rating analysis as a pre-specified secondary
endpoint, with the conflict-of-interest handling stated (e.g. analysis by a blinded party), or
explain in the main text why it cannot be done.

---

### M6. There are no episode-level operating characteristics, and the per-moment framing conceals that the monitor alarms in about three of every four runs
**[Methods 4.4; Methods 4.5; Fig. 2b; Fig. ED2 funnel]**

*What is wrong.* Every rate in the paper is per moment. Methods 4.4 states: "A sequential warning
layer (persistence over consecutive verdicts) belongs to the deployment interface: no result in
this paper uses it, every reported verdict and flag rate is per-moment, and its operating
characteristics (episode-level false-alarm rate, detection delay, warning duration) are not
evaluated in this paper."

The manuscript nonetheless contains the numbers that make the per-moment framing untenable:

- Methods 4.5: "7 of those 29 contain any flagged moment, against **132 of the 175 retained**". A
  monitor advertised as calibrated at the 90% level raises at least one alarm in 75% of runs.
- Fig. ED2's selection funnel: "**120** runs with a flagged moment → **20** with ≥5 clustered
  frames". Only 17% of runs containing an assertive flag contain a contiguous half-second of them;
  flags are overwhelmingly isolated or near-isolated frames.
- Fig. 2b: the mean frame-to-frame movement of the reading among readable moments is "0.30 rad for
  the ego and 0.31 for the counterpart". At 10 Hz that is 76% of the 0.393 rad grid spacing every
  100 ms. The monitored signal jumps by nearly a full grid step between consecutive verdicts.
- Fig. 1c — the paper's flagship illustration — shows a flag consisting of a two-frame spike above
  a 90% edge at ≈ +1.02 rad, on the over-yielding side.

*Why it matters.* A signal that moves 0.3 rad per 100 ms, judged instantaneously against a fixed
boundary, will cross that boundary from jitter alone. Without a run-length distribution, an
episode-level false-alarm rate, or a stability analysis, the reported 9.8% cannot be distinguished
from estimator noise crossing a threshold. Fig. 2c's demonstration that episode summaries are
unstable is used to justify per-moment verdicts, but a per-moment verdict is *more* exposed to the
same instability, not less; the natural conclusion is that a persistence rule is mandatory, and
that rule is precisely what the paper declines to evaluate.

*Remedy.* Report the run-length distribution of flags; the fraction of flags that are singletons;
the episode-level (per-run) alarm rate for all three arms with intervals; and flag stability under
perturbation (window length 0.5/1.0/1.5 s, ±1-frame time shift, resampled solver seeds). Then state
in the main text what fraction of runs the monitor alarms in — that, not the moment rate, is the
number a deployment reader needs.

---

### M7. Directional language outruns the battery, and Fig. 6b puts protective and adverse effects on the same side of one axis
**[Section 2.4; Fig. 5a–d; Fig. 6b; abstract]**

*What is wrong.* Read as a whole, the AV consequence battery has six endpoints and four of them
move in the *protective* direction. From Fig. 5b, the ego's sub-threshold-margin rate falls from
≈2.3% to ≈0.9% (< 1 s), ≈5.8% to ≈2.1% (< 1.5 s) and ≈8.8% to ≈4.7% (< 2 s); from Fig. 5d,
counterpart hard braking falls from ≈8.5% to ≈4.1%, ≈6.3% to ≈2.9% and ≈5.7% to ≈2.6%. Only the
median/upper-quartile ego margin and the counterpart's routine speed reduction/variation move
adversely.

Three consequences follow.

(a) The abstract renders halved emergency rates as "with no rise in emergency rates at any
supported threshold". That is true and materially understated; Fig. 5b's own title says "rarer".

(b) "Supported" is defined as "A threshold is called supported when its per-endpoint interval
excludes a zero difference"; the panel title then reads "Emergency margins are rarer at every
supported threshold". As written the title is true by construction. The honest statement is that
three of four margin thresholds and three of three braking thresholds excluded zero, per endpoint,
with no adjustment across a nested and correlated family of ~11 endpoints in Fig. 5 alone.

(c) Fig. 6b places "Ego margin (median)" (ratio ≈ 0.8; smaller margin = worse) and "Ego emergency
(< 2 s)" (ratio ≈ 0.48; fewer emergencies = better) on the same side of parity under the single
label "compressed at flagged moments". Five of six markers cluster as if telling one story when
they point in opposite normative directions. The axis label and annotations conflate direction of
change with direction of harm.

(d) Most seriously, the paper's lead consequence number is not supported on its own convention. In
Fig. 6b the "Ego margin (median)" row carries **no asterisk for either arm**, and the caption states
"an asterisk marks a ratio whose interval excludes parity ...; entries whose intervals admit parity
carry no asterisk". Yet Section 2.4 reports "the median falling by about a quarter" and Section 2.5
"the ego margin's median contracts by about a fifth" as findings, and Fig. 5a reports "median
−24.9%" with no interval at all (Fig. 5's caption attaches intervals only to panels b, c and d).

*Why it matters.* The claim that a social flag marks something a reader should care about rests on
two endpoints, one of which does not clear the paper's own significance convention, against four
that move the other way. A referee reading only Fig. 5b/5d would conclude the monitor's alarms
anti-predict danger.

*Remedy.* Report an interval for the median and upper-quartile margin ratios in Fig. 5a and mark
them consistently in Fig. 6b; retitle Fig. 5b/5d non-circularly; split Fig. 6b's axis by direction
of harm rather than direction of change; add a multiplicity statement (or a pre-registered primary
endpoint) for the battery; and state plainly in the main text that emergency-tail events are one-third
to one-half as frequent at flagged moments.

---

### M8. Internal contradiction between Fig. 4a and Fig. 4b about the global reference
**[Fig. 4a, Fig. 4b and caption]**

*What is wrong.* Fig. 4a labels the global range's width as **100** (per cent of the admissible
range) at both the 90% and the 95% level, and the caption concludes: "A global range is not a
usable reference: at the 95% level it spans 100.00% of the admissible range and so can never flag
anything." But Fig. 4b reports the *same* global range, on the *same* 461,937 accepted moments, at
−0.01 pp (90%) and +2.77 pp (95%) relative to nominal — i.e. achieved coverage of 89.99% and
97.77%, meaning it excludes 10.01% and 2.23% of those moments. A range that spans the entire
admissible support of the quantity must cover every reading. The two panels cannot both be right.

*Why it matters.* The degeneracy of the global reference is the paper's stated justification for
conditioning, and this is the figure that carries it. Whichever panel is wrong, the comparison that
motivates the whole construction is currently unreliable. The caption also singles out the 95%
level while the printed bar shows the same value at 90% — the level at which every verdict in the
paper is issued.

*Remedy.* Reconcile the width normalisation and the coverage evaluation (state the admissible set
used in each), correct whichever panel is in error, and state the global range's status at the 90%
level explicitly.

---

### M9. The paper's headline "tension" rests on an unquantified correlation and an accepted null
**[Section 2.2; Section 2.3; Discussion; Methods 4.3]**

*What is wrong.* The Discussion calls it "The most informative finding ... a tension between two
levels", and both halves are under-reported.

The first half — "The two interacting agents' preferences, moreover, are correlated at this event
level" (Section 2.2), repeated in Section 2.3 and the Discussion — appears **four times with no
coefficient, no confidence interval, no sample size and no test** anywhere in the manuscript,
Methods included.

The second half is an accepted null: "the counterpart channel in particular is statistically
indistinguishable from removing the IPV input given the situation (paired 90% interval-score
difference −0.0002, case-clustered p = 0.86)". There is no confidence interval on the difference,
no pre-specified equivalence margin, and no statement of the smallest sharpening the design could
have detected. Absence of evidence is presented as evidence of absence, and it is presented as the
paper's most informative finding.

*Remedy.* Give the partner correlation with its coefficient, CI, unit of analysis and n. Recast the
counterpart ablation as an equivalence test against a pre-specified margin (e.g. "the counterpart
channel narrows the 90% interval score by less than X"), with a CI and a power statement. Do the
same for the self-history ablation.

---

### M10. No stated monitor semantics, no threshold sensitivity, and no runtime characterisation, in a paper about a runtime monitor
**[Title; Methods scope note; Methods 4.1–4.4; Algorithm 2]**

*What is wrong.* Three gaps that matter specifically for the runtime-verification framing.

(a) *Semantics.* The paper adopts monitoring vocabulary (verdicts, abstention, reason codes,
Algorithm 2) and cites runtime verification [36–39], but never states what is guaranteed, over what
population, at what time index. Methods 4.4's "The estimand is pointwise marginal coverage over
accepted moments" is the closest thing and it appears in Methods only. What the construction
provides is one-sided: an approximate false-alarm rate on human natural-driving moments drawn from
the fitting corpora. It provides **no soundness statement** for the alarm ("if
Competitive-Deviation is emitted then ..."), and **no detection guarantee or power statement** at
all. The main text and abstract use "calibrated" and "auditable" without conveying either
limitation.

(b) *No sensitivity to any frozen threshold.* The results depend on: the near-uniform rule
(max weight < 0.20); the support gate (95th percentile of guard distances, 1.081; k = 25; ≥ 50
training anchors; ≥ 10 cases); the interaction-opportunity rule; the 1 s window with ≥ 4 observed
frames; and σ = 0.1 m. Freezing thresholds in advance is good practice and is not a substitute for
showing that the conclusions survive perturbing them. No sensitivity analysis of any of these
appears.

(c) *No runtime characterisation.* The Methods scope note concedes "end-to-end real-time operation
on vehicle hardware is not evaluated in this paper". The estimator solves seven trajectory
optimisations per agent per frame at 10 Hz. No latency, throughput, memory or hardware figure is
given anywhere. For a paper titled "Online monitoring", that is a conspicuous absence.

*Remedy.* Add a short, explicit semantics statement to the main text (the null being tested, the
population, the time index, the verdict domain, and what abstention does and does not assert). Add
a threshold-sensitivity table (flag rate and consequence contrasts across a grid of gate settings).
Report per-frame computation time on a stated platform.

---

### M11. Availability and selection: the monitor speaks on a fifth of moments and there is no evidence that fifth is representative
**[Section 2.4; Methods 4.5; Fig. 6 top]**

*What is wrong.* On the benchmark "the monitor returns a verdict on 14,099 (20.8%)" of candidate
moments and is entirely silent in 36 of 267 runs. Methods 4.5 explains why: in the silent runs the
readability pass rate is normal (53.78%) but "the human-support pass rate is 0.07% (against 32.32%
overall)", and support varies enormously between cells ("one high-volume situation cell with
1,148,133 human anchor rows passes support on 14.58% of its benchmark moments while a 45,283-row
cell passes on 47.03%").

This means the monitor speaks precisely where a staged conflict resembles ordinary US/Singapore
natural driving, and falls silent where it does not. Every reported rate — 9.8%, 4.7%, the 2.1×,
the whole consequence battery — is conditional on that selection, and no analysis is offered
showing the judgeable subset is representative of the safety-relevant moments. Because AV and human
runs are matched at the scenario level but judgeability is determined per moment, the two arms'
judgeable sets may also differ in composition *within* a scenario, which would confound the 2.1×
independently of M4.

Two arithmetic issues sit here as well. (i) 14,099 judgeable of 37,527 readable (55.3% of 67,861)
is 37.6%, which cannot be reconciled with a "32.32% overall" support pass rate if support is the
second gate. (ii) Fig. 6's funnel prints "count unavailable / 55.3% pass" for the automated arm's
readable stage while printing an exact count (39,580) for the human arm — yet 55.3% of 67,861 is a
computable number. If the 55.3% is not a pooled moment-level rate on the same denominator, then the
funnel's central like-for-like comparison (55.6% human vs 55.3% AV) is not a comparison of like
quantities.

*Remedy.* Report the composition of judgeable versus non-judgeable moments (situation-cell
distribution, distance to conflict, TTC distribution) for both arms; restrict the AV-vs-human
comparison to situation cells judgeable in both arms and report it there; resolve the 32.32%
arithmetic; and supply the missing count.

---

## 3. Minor issues

1. **Fig. 2a has no stated baseline.** All four rows, including "Real interaction" (−0.132), carry a
   value on an axis labelled "Change in how sharply the reading is identified", but the reference
   against which the change is measured is never named in the caption or the text.
2. **Fig. 2a caption is internally inconsistent.** "The gain is present only for the real
   interaction: it disappears when the same interaction is misaligned in time (+0.006), **weakens**
   when the same moment is paired with a different partner (−0.043)". A third of the effect survives
   substituting the partner; "present only for the real interaction" is not what the panel shows.
   The four rows also have different n (4,743 / 4,605 / 4,701 / 8,130), so these are not paired
   contrasts on a common set.
3. **"uncertainty not shown"** is printed inside Fig. 2b and Fig. 6c. For a quantitative claim in a
   Nature-family paper this should not appear in a figure; either show the intervals or move the
   panel to Extended Data.
4. **Resampling unit may not cover the data in Fig. 5a,b.** The caption says intervals come from
   "bootstrap resamples over the 175 scenario runs, the unit of resampling throughout", but the
   472/11,669 moments are defined in Methods 4.5 as all judgeable moments with a defined margin,
   which come from the 231-run set; the caption separately notes they "cluster within 120 scenario
   runs". Please confirm that the 120 runs are a subset of the 175, or correct the resampling unit.
5. **Mixed estimands within one figure.** Fig. 5c reports ratios of medians while Fig. 5b/5d report
   differences in percentage points; Fig. 6b converts everything to ratios. Ratios of medians have
   awkward interval behaviour (the AV speed-reduction interval's lower edge sits just above parity)
   and are not comparable across panels.
6. **Readability accounting does not reproduce.** Methods 4.4 gives "3,202,646 readable" of
   "4,497,368 anchor rows" = 71.2%, while Methods 4.5 quotes 70.3% for the same quantity.
7. **Asterisk assignment in Fig. 6b is ambiguous** at printed size: in the counterpart speed-reduction
   and speed-variation rows the markers sit close together and the asterisks cannot be reliably
   attributed to the AV or human entry.
8. **Fig. ED1 legend is asymmetric**: "below range · more assertive" versus "above range · atypical".
   Both sides are atypical; the labelling implies otherwise.
9. **Placebo p-value granularity.** 200 draws give a resolution of 1/201 ≈ 0.005; p = 0.0199 is the
   fourth-smallest possible value. Please increase the number of draws.
10. **"a small fraction of readable moments (5.08%...)"** (Section 2.3) is computed as 24,723 of
    486,660 test-fold *anchor rows* (Methods 4.4). The identity of these two denominators has to be
    reconstructed by the reader from the fold sums; state it.
11. **Data availability.** The human reference arm — the paper's external-validity backbone — "is
    reported only in aggregate", yet Fig. 6's caption states "Source data are provided as a Source
    Data file". Clarify what aggregate source data will be released (at minimum per-driver and
    per-scenario judgeable counts and flag counts).
12. **Ethics.** Participants "were not told that their runs would serve as a human reference against
    automated systems". No debriefing procedure is described; please state whether participants were
    debriefed and re-consented.
13. **No limitations section.** Limitations are distributed across Methods; a broad readership needs
    them collected.
14. **Unsupported closing claim.** "Beyond driving, the framework offers a template for monitoring
    whether autonomous agents behave within human normative ranges online" is not evidenced by
    anything in the paper.
15. **Effect sizes in Fig. 3a** (+0.058 / +0.001 / −0.034 rad) are ~3% of the 1.87 rad conditioned
    90% range, as the text concedes; the section heading "Human social behaviour is context-dependent"
    is carried operationally by the 20% width reduction in Fig. 4a, not by Fig. 3a.

---

## 4. Questions to the authors

1. What is the empirical distribution of θ̂ over accepted moments? What fraction of its mass lies
   within 0.02 rad of each of the seven candidates, and what is the distribution of max_k π_k?
2. What fraction of assertive-side flags have θ̂ at the boundary candidate −3π/8, and what fraction
   of over-yielding flags at +3π/8? Do the 9.8% / 4.7% / 2.1× figures survive a finer grid over the
   same span, and a grid extended to the declared domain [−π/2, π/2]?
3. Fig. 4a labels the global range width as 100% of the admissible range at the 90% and 95% levels
   and the caption says such a range "can never flag anything", while Fig. 4b reports 89.99% and
   97.77% achieved coverage for that same range on the same moments. Which is correct?
4. What are the cluster-bootstrap confidence intervals (over scenes or interaction cases) on each
   achieved-coverage value in Fig. 4b? What is the per-interaction coverage at the 90% level?
5. Given leave-one-source-out coverage of 0.743 and 0.750, on what basis does Section 2.5 conclude
   that carrying the reference "across country, apparatus and task does not inflate its alarms",
   when the Fig. 4 caption states that transfer "is not established"?
6. The counterpart trajectory X_j(W_t) enters Eq. (1). What happens to the counterpart-side results
   when the outcome window is displaced to begin at t + 1 s or later, so that no outcome sample
   overlaps a verdict input? What does a within-run time-shift placebo give?
7. In Fig. ED2b the counterpart's deceleration begins before the flagged stretch. How much of the
   "4.31 → 1.40 m s⁻¹" change occurs at or before the first flag?
8. What is the run-length distribution of flags? What fraction are single frames? What is the
   per-run alarm rate, with an interval, for each of the three arms?
9. Please give the event-level partner correlation: coefficient, confidence interval, unit of
   analysis, n.
10. For the counterpart-IPV ablation, what is the confidence interval on the −0.0002 interval-score
    difference, and what sharpening would the design have had 80% power to detect?
11. Decomposed by side, the 90%-level AV/human rate ratio is 1.42× (assertive) and 2.89×
    (over-yielding). Given that Section 2.4 reports no consequence signature on the over-yielding
    side, what should a reader conclude from the pooled 2.1×?
12. Fig. 6a shows the automated systems at the reference's own rate at all three levels and the
    on-course humans at roughly half it. Why is the human arm not the anomaly here? What evidence
    excludes the head-mounted display, the safety supervisor and the closed course as the cause?
13. Are the AV and human judgeable moments matched in situation-cell composition within each
    scenario? What is the flag-rate ratio restricted to cells judgeable in both arms?
14. In Fig. 6b, the "Ego margin (median)" row carries no asterisk for either arm. Does its interval
    admit parity? If so, on what basis is "the median falling by about a quarter" reported as a
    finding in Section 2.4?
15. How is 14,099 judgeable of 37,527 readable (37.6%) consistent with a "32.32% overall"
    human-support pass rate? And why is the automated arm's readable count "unavailable" when the
    percentage is given?
16. How do the flag rate and the consequence battery move under perturbation of the near-uniform
    threshold (0.20), the support percentile (95th), k (25), the window length (1 s) and σ (0.1 m)?
17. What does a kinematic conditional-quantile baseline — same z_t, same quantile machinery, an
    observable behavioural target — flag, and does it reproduce the consequence signature?
18. What is the per-frame computation time of the estimator plus gates on a stated platform?
19. Preference ratings from the same challenge exist and were excluded as endpoints. Can they be
    analysed under a blinded protocol, and if not, why not?
20. What are the per-driver flag rates across the 20 participants?

---

## 5. Prioritised revision requests

1. **Establish that a flag is not an artefact of the candidate grid.** Distribution of θ̂ and of
   max_k π_k; fraction of flags at the boundary candidates; full re-run of all headline rates on a
   finer and on an extended grid. Nothing else in the paper can be assessed until this is settled.
   *(M1)*
2. **Re-do the consequence analysis without overlap between verdict inputs and outcomes.** Displaced
   counterpart windows, within-run time-shift placebo, and both permutation schemes reported on
   equal footing. *(M3)*
3. **Make the calibration claim cluster-aware and state its scope.** Per-interaction coverage,
   cluster-bootstrap intervals on every number in Fig. 4b, a cluster-conformal comparison, and an
   explicit statement in the main text that no coverage guarantee is claimed on the benchmark
   population. Reconcile Section 2.5 with the Fig. 4 caption. *(M2)*
4. **Add a baseline monitor.** A kinematic conditional-quantile monitor on the same z_t, plus at
   least one published monitor, evaluated on the same benchmark with the same battery. *(M5)*
5. **Report episode-level operating characteristics.** Flag run-lengths, singleton fraction, per-run
   alarm rate with intervals for all three arms, and stability under window/threshold perturbation.
   *(M6)*
6. **Repair the human-versus-machine comparison.** Interval on the ratio, side decomposition in the
   main text, per-driver rates, restriction to jointly judgeable cells, and an explicit apparatus
   caveat — or removal of the 2.1× from the abstract. *(M4)*
7. **Fix the Fig. 4a/4b contradiction.** *(M8)*
8. **Correct the directional reporting.** Intervals for the median/upper-quartile margin ratios;
   non-circular panel titles; Fig. 6b redrawn so that protective and adverse effects are not on the
   same side of one axis; multiplicity statement for the battery; the halved emergency rates stated
   plainly in the abstract. *(M7)*
9. **Quantify both halves of the paper's central tension.** Partner correlation with CI; the
   counterpart ablation as an equivalence test with a margin and a power statement. *(M9)*
10. **Add a monitor-semantics paragraph, a threshold-sensitivity table, and a latency figure.**
    *(M10)*
11. **Characterise the judgeable subset.** Composition of judgeable versus non-judgeable moments in
    both arms; resolve the 32.32% arithmetic; supply the missing count in Fig. 6. *(M11)*
12. Minor issues 1–15.

---

## 6. Acceptance probability

- **(a) As submitted: 3%.** The central measurement may be a grid-boundary artefact, the calibration
  guarantee is shown by the authors' own analysis not to transport, the only evidence that a flag
  means anything is confounded by construction, and the headline comparison is confounded by
  apparatus and decomposes into the direction with no reported signature. There is no baseline and
  no detection performance.
- **(b) After a competent major revision: 25%.** The reframing is genuinely novel and well suited to
  this journal, the abstention discipline is a real contribution, and the disclosure standard in
  Methods is above average for the field. But requests 1–4 require new analysis whose outcome could
  change every headline number — and could invert the paper's own conclusion, since Fig. 6a as
  printed reads more naturally as "the automated systems sit at the reference's nominal rate". I
  cannot put the probability higher without knowing how those analyses come out.

---

## 7. Recommendation

**Reject.**

I would encourage the authors to resubmit a substantially re-analysed version. The framing —
conditional social atypicality with explicit abstention, monitored online against a
situation-conditioned human reference — is worth publishing if the measurement can be shown to be a
measurement, the calibration claim is stated at the level at which it holds, and the consequence
evidence survives a construction that does not feed the outcome back into the verdict. As
submitted, those three conditions are not met, and the paper's most quotable sentence is the one
its own Fig. 6a most directly undercuts.
