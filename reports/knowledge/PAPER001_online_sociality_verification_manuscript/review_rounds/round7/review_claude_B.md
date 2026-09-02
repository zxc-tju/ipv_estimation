# Referee report — "Online monitoring of socially compliant autonomous driving"

**Remit: methodology, statistics, runtime verification / formal semantics.**

---

## 1. Summary assessment

The paper reframes social compliance as an online membership test: is the ego's current
interaction preference (IPV) inside a human reference range conditioned on the observable
situation, and if the reading is not identifiable, abstain rather than pass. It delivers a
frozen estimator, a split-conformal situation-conditioned range with near-nominal *marginal*
coverage, an explicit abstention vocabulary, a matched-scenario real-vehicle benchmark, and a
same-scenario human arm audited by the unchanged monitor. The construction is genuinely novel
and the abstention semantics are handled with more care than is usual.

The statistical case, however, does not carry the claims made for it. The abstract's central
empirical result — flagged assertive moments are "followed by tighter interactions on both
sides" — rests on an ego-side endpoint that the paper's own Fig. 6b marks as *not* excluding
parity, computed on the window whose interval Methods 4.5 reports as crossing zero. The
"assertive side only" contrast is contradicted by Fig. 5a,b. The "twice as often" headline is
mostly carried by the side the paper says has no signature. Underneath, the estimator is
effectively a 7-point argmax and the assertive flag region is a sliver at its floor.

---

## 2. Major weaknesses

### M1. The headline ego-side effect is presented without an interval, is reported on the non-primary window, and the paper's own figure marks it as not distinguishable from parity

*Location: Abstract; §2.4; Fig. 5a and caption; Fig. 6b; Methods 4.5.*

The abstract states that flagged moments "are followed by tighter interactions on both sides".
§2.4 makes this concrete: "The ego's own margin to its counterpart contracts across the body of
its distribution, the median falling by about a quarter and the upper quartile by nearly half."
Fig. 5a prints "median −24.9%" and "upper quartile 47.4%" — with no confidence interval on
either.

Three checks contradict the strength of that claim.

(i) **The window.** Methods 4.5 defines the ego-side outcome as "the minimum time-to-collision
to the counterpart over the post-verdict window, which runs from the verdict to the end of that
run's evaluated window", explicitly contrasting it with the counterpart battery, which "instead
uses the fixed three-second window". The same section then reports: "the fixed three-second
window is the pre-specified primary, and its case-clustered intervals exclude zero at all three
levels, while the open-ended contract-window interval crosses zero at the 90% level
([−2.6100, +0.1372])." The ego panel therefore displays the *non-primary* window, and it is
exactly that window whose interval crosses zero at the 90% level — the level at which, per the
Fig. 4 caption, "Verdicts elsewhere in this paper are issued".

(ii) **The figure's own significance convention.** Fig. 6b's caption states that "an asterisk
marks a ratio whose interval excludes parity ... entries whose intervals admit parity carry no
asterisk." I located every marker and asterisk glyph in that panel by colour and centroid.
Marker rows sit at image y = 725/839/952/1067/1181/1295 (AV, filled) and y =
754/869/983/1097/1211/1325 (human, open); the six row labels centre at y =
745/859/975/1088/1202/1305, so the row order is as printed. Blue asterisks occur at y =
899/1013/1127/1241, each 53–54 px above an AV marker and 46–52 px to its right; the x-offsets
identify rows 4, 5 and 6 unambiguously (asterisk x = 2570/2538/2031 against marker x =
2518/2480/1984), which fixes the convention and assigns the remaining asterisk (y = 899) to row
3. Grey asterisks occur at y = 995/1110/1224/1338, each ~13 px below a human marker, i.e. rows
3–6. **Rows 1 and 2 — "Ego margin (median)" and "Ego margin (upper quartile)" — carry no
asterisk for either arm.** This is internally coherent with (i) and with Fig. 5b, where the
AV assertive <2 s difference (−4.18 [−7.28, −1.03], row 3) does exclude zero.

(iii) **Consistency.** So the two numbers the abstract and §2.4 lead with are, by the paper's
own machinery, non-detections; and the two ego-side quantities that *are* interval-supported
(emergency margin rates, row 3) point the *other* way — flagged moments have **fewer** short
margins.

**Why it matters.** At this journal the abstract claim would be read as the empirical payload of
the whole monitor. As it stands, the ego half of "tighter interactions on both vehicles" is an
uncertainty-free point estimate on a secondary window, silently marked non-significant three
figures later. §2.6 then compounds it: "the interaction has nonetheless become measurably
tighter for both vehicles."

**Remedy.** Report the assertive-side ego-margin median and upper-quartile ratios with their
case/run-clustered intervals directly in Fig. 5a; state which window each panel uses; and
rewrite the abstract, §2.4 and §2.6 so that "measurably tighter" is claimed only for the
endpoints whose intervals exclude the null. If the pre-specified primary is the three-second
window, Fig. 5a,b should show it, with the open-ended window as a sensitivity panel.

---

### M2. "Compression appears on the assertive side only" is contradicted by the panel that carries the title

*Location: Fig. 5a (panel title and caption), Fig. 5b, §2.4, Methods 4.5.*

Fig. 5a is titled "Compression appears on the assertive side only" and its caption asserts:
"The accommodating distribution tracks the within-range one across its whole body." §2.4:
"neither the counterpart's speed variation nor its speed reduction separates detectably from
within-range moments, and the ego's own margin does not either."

But Methods 4.5 reports, for the accommodating side, "the ego's own margin median is 0.81×
([0.70, 1.15])", while the assertive median ratio is 1 − 0.249 = **0.751**. The accommodating
point estimate is a 19% median contraction; its interval **contains the assertive point
estimate**, so these data do not distinguish the two sides on this endpoint. In the panel
itself the two median markers sit adjacent on the log axis, well to the left of the within-range
marker. "Tracks the within-range one across its whole body" is not what the figure shows.

Fig. 5b is stronger against the title. Reading the printed differences: at <1 s, <1.5 s, <2 s
and <3 s the accommodating intervals are −1.07 [−2.03, −0.05], −2.64 [−4.62, −0.38],
−3.49 [−6.36, −0.68] and −4.74 [−8.63, −0.59] — **all four exclude zero**, whereas the
assertive side excludes zero at only three of four (<3 s: −3.14 [−8.33, 3.15]). So on the
ego-margin endpoint family, the accommodating side is if anything the *better*-supported arm.
Fewer short margins together with a lower median is precisely the "compression of ordinary
interaction quality, not a shift of mass into the tail" that §2.4 reserves for the assertive
side.

Both the main text ("at the widest margin tested (under 3 s) the interval admits no
difference") and the Fig. 5b caption ("the < 3 s interval admits no difference") use the
singular where two intervals are displayed, and the one that excludes zero at <3 s is the
accommodating one. Similarly, the parenthetical list "(margins under 1, 1.5 and 2 s; braking
beyond −2, −3 and −4 m s⁻²)" describes the assertive arm only, while its grammatical subject is
"flagged moments", which §2.4 defines as "counting both sides".

A related asymmetry is buried in Methods 4.5: "The ego-margin panels use the moments with a
defined post-verdict margin (472 flagged, 11,669 within-range) ... the undefined fraction is
similar in the two groups (9.1% of flagged vs 8.2% of within-range moments), so this exclusion
does not preferentially remove either group." Those numbers check out for the assertive arm
(519 − 472 = 47; 47/519 = 9.06%) and the within-range arm (12,711 − 11,669 = 1,042;
1,042/12,711 = 8.20%). But the accommodating arm loses 869 − 747 = 122 moments, i.e.
122/869 = **14.04%**, 1.7× the within-range rate; and "flagged" as the main text defines it
loses (47+122)/1,388 = **12.2%**, not 9.1%. Since the margin is undefined "where the pair is not
closing", the exclusion removes disproportionately from exactly the group whose defining
behaviour is over-yielding — the direction that would bias the accommodating arm toward the
within-range distribution.

**Remedy.** Retitle the panel; delete or correct the "tracks ... across its whole body"
sentence; report both sides' intervals for every ego-margin statistic; report the attrition for
all three groups and a sensitivity analysis in which non-closing moments are retained (e.g.
censored at the window length) rather than dropped.

---

### M3. The "automated systems flagged about twice as often" headline is carried by the side the paper says has no consequence signature, has no interval, and is level-dependent

*Location: Abstract; §2.5; Fig. 6a.*

§2.5: "under this common instrument the automated systems are flagged 2.1 times as often as the
matched human drivers (9.8% vs 4.7% of judgeable moments)". Fig. 6a prints the side counts:
humans 912|780, 391|322, 178|147 and AVs 1,174|1,574, 519|869, 314|387 at the 80/90/95% levels
(assertive|accommodating), against 15,102 and 14,099 judgeable moments.

Decomposing the 90% level:

| side | AV | human | ratio |
|---|---|---|---|
| assertive | 519/14,099 = 3.681% | 391/15,102 = 2.589% | **1.42** |
| accommodating | 869/14,099 = 6.163% | 322/15,102 = 2.132% | **2.89** |
| both | 9.844% | 4.721% | 2.09 |

The excess decomposes as 1.092 pp (assertive) + 4.031 pp (accommodating) = 5.123 pp, so
**78.7% of the AV–human gap sits on the accommodating side** — the side for which §2.4 reports
"neither the counterpart's speed variation nor its speed reduction separates detectably" and
Methods 4.5 reports that "Every interval admits no difference." The paper's single most
quotable population result is therefore driven mostly by flags it has itself characterised as
behaviourally inert.

The ratio is also level-dependent: 19.5/11.2 = 1.74 at 80%, 2.09 at 90%, 4.972/2.152 = 2.31 at
95%; and assertive-side-only it is 1.38 / 1.42 / 1.89 across the three levels. Only the 90%
pooled value is reported.

Finally, no interval is given for the ratio or for the AV rate. The only interval in Fig. 6a is
on the human 90% bar ("95% CI 3.8–5.8%"). That interval is itself informative about the
inferential problem: a naive binomial interval on 713/15,102 has half-width 0.34 pp, against the
printed 1.0 pp, implying a design effect of ≈ (1.0/0.34)² ≈ 8.8 — i.e. an effective sample size
around 1,700, not 15,000.

**Remedy.** Report the ratio with a clustered interval; report it separately by side and by
level; and either drop "about twice as often" from the abstract or state that the excess is
predominantly on the accommodating side.

---

### M4. The estimator is, in operation, a 7-point argmax; the assertive flag region is a narrow sliver at its floor; and per-frame verdicts are unstable

*Location: Methods 4.1, 4.2, 4.5; Fig. 2b; Fig. 1c; Fig. ED1; Fig. ED2 funnel.*

Methods 4.1 specifies "a Gaussian trajectory likelihood ℓₐ,ₖ ∝ exp{−MSEₖ/(2σ²)} with σ = 0.1,
where MSEₖ is the mean squared Euclidean distance (m²)". Then 2σ² = 0.02, and the weight ratio
between two candidates differing by ΔMSE is exp(−50 ΔMSE). For two candidates to hold weights
within a factor of two requires ΔMSE ≤ ln2/50 = 0.0139 m², i.e. an RMS positional difference of
≈ 0.12 m over the one-second window. Any candidate whose one-second rollout differs by more
than about a tenth of a metre RMS is effectively excluded. The "weighted mean ... which varies
continuously over [−3π/8, 3π/8]" is therefore, in practice, close to a hard argmax over seven
grid values. Fig. ED1 shows exactly this: readings repeatedly pinned at ±1.18 rad (= ±3π/8),
with excursions across the whole span between consecutive frames. It is also why the two frames
chosen in Fig. ED1 to "carry the same reading, −0.393 rad" sit at −π/8 = −0.3927 to three
decimals — a grid point.

This has a direct consequence for what an assertive flag means. Methods 4.5: "at the 90% level
the lower edge is negative in essentially every situation (median −1.03 rad)". The estimator
floor is −3π/8 = −1.1781 rad. The assertive flag region at the median situation is therefore
**0.148 rad wide — 38% of one grid step (π/8 = 0.3927 rad)**. Writing the reading as
Σ wₖθₖ, and putting all non-extreme mass on the next candidate (the most favourable case),
θ̂ ≤ −1.03 requires −1.1781w − 0.7854(1−w) ≤ −1.03, i.e. **w ≥ 0.623** on the single most
assertive candidate. Any other allocation requires more. So "Competitive-Deviation" is, in
practice, an indicator that the likelihood has saturated on the extreme grid point — not a
graded departure from a human range. Fig. 1c and Fig. ED1 show long stretches where the lower
edge coincides with the floor, in which an assertive flag is *structurally impossible*.

Two corroborating arithmetic points. First, Fig. 4a reports the global range at 100% of the
admissible span at both the 90% and 95% levels ("at the 95% level its width equals 100.00% of
the admissible span"), while Fig. 4b reports coverages of 89.99% and 97.77%. If those widths
are literal, the two nested intervals differ by at most 0.5% of 2.356 rad = 0.0118 rad, so at
least 7.78% of the 461,937 accepted test moments (≥ 35,900 moments) lie inside a set of total
measure ≤ 0.012 rad — a mass concentration consistent only with atoms at grid values. If
instead the widths are clipped at the admissible span, then "100.00%" is not the true width and
the conditioned-vs-global comparison in Fig. 4a is not like-for-like. Either reading needs
addressing. Second, the near-zero conformal radii ("cα = 1.4 × 10⁻³, 1.2 × 10⁻⁶ and 0.0 rad")
are what one expects when nonconformity scores are heavily tied at zero, not evidence that "the
conformal step finds essentially nothing to repair".

Stability is the other half. Fig. 2b: the reading moves 0.30 rad (ego) / 0.31 (counterpart) per
frame where it is most readable. That is **twice the width of the median assertive flag
region** and ~0.76 of a grid step, at 10 Hz. Methods 4.4 states that "no result in this paper
uses" the persistence layer and "every reported verdict and flag rate is per-moment". The
Fig. ED2 selection funnel quantifies the resulting fragmentation: of "120 runs with a flagged
moment", only "20 with ≥5 clustered frames" — i.e. in 100 of 120 runs the assertive flags never
form a contiguous half-second.

**Why it matters.** A runtime monitor whose verdict flips faster than the phenomenon it claims
to measure, and whose positive verdict is close to a saturation indicator of a coarse candidate
grid, is a different object from the graded, calibrated instrument the paper describes.

**Remedy.** Report the empirical distribution of θ̂ (is it atomic at the seven grid values?);
report the fraction of accepted moments whose lower edge lies at or below −3π/8 and whose upper
edge lies at or above +3π/8, i.e. the fraction structurally unflaggable on each side; report
verdict run-length and frame-to-frame flip-rate distributions; and show sensitivity to σ and to
grid resolution.

---

### M5. Calibration is marginal only, and the conditional-coverage evidence the paper does report shows heterogeneity larger than the effect it is used to detect

*Location: §2.3; Fig. 4b; Methods 4.3, 4.4; §2.5.*

Credit where due: the calibration/test separation is clean ("Splits are made by whole scenes
under a split frozen before this study, so no scene contributes to more than one fold"), the
folds are disjoint (481,088 calibration; coverage on 486,660 test / 461,937 accepted), and the
paper says plainly that the statement is "marginal over accepted moments — not a
per-interaction, sequential or conditional guarantee". Three problems remain.

**(a) By-source coverage.** Methods 4.4: "on the test fold the deployed reference contains
88.2%, 95.9%, 85.9% and 88.2% of moments at the 90% level within Waymo, nuPlan, Lyft and
Argoverse-2 respectively, against 90.3% pooled, so the pooled figure is not masking a source in
which the range fails." As *alarm* rates — which is what the monitor emits — those are 11.8%,
4.1%, 14.1% and 11.8% against a nominal 10%: a **3.4× spread** (14.1/4.1), and a 41% relative
inflation on Lyft. The paper's own headline effect (AV = 2.1× humans) is smaller than the
instrument's between-source variation in its own alarm rate. "Not masking a source in which the
range fails" is a generous reading of those numbers.

**(b) Leave-one-source-out.** Refitting with a source held out covers it at 0.743 (Waymo),
0.750 (Lyft), 0.990 (nuPlan) and 0.900 (Argoverse-2, with 44.3% abstention) against nominal
0.90 — i.e. alarm rates of 25.7% and 25.0% on two of four sources. Fig. 3c independently shows
out-of-source R² of +0.026, +0.017, −0.195, −0.276. The conditioning demonstrably does not
transfer between the very sources it was fitted on, yet it is deployed unchanged on a fifth
domain (different country, closed course, mixed reality).

**(c) The transfer test is one-sided and cannot separate "transferred" from "went silent".**
§2.5 argues that "the instrument survives the move" because human drivers on the course are
"flagged on 4.7% of judgeable moments (713/15,102), no more often than [the 9.72%] native
rate". But 4.7% with a printed CI of 3.8–5.8% *excludes* 9.72%: on the deployment domain the
reference does not merely fail to inflate, it fires at roughly **half** nominal, i.e. achieved
coverage ≈ 95.3% against a nominal 90%. A monitor that has become conservative on the new
domain is not thereby validated; low alarm rate is exactly what an over-wide, mis-transported
reference produces. The same finding also undercuts "The automated systems' flag rate of 9.8%
sits at the native level", which compares a deployment-domain rate to a home-domain nominal
that the paper has just shown does not carry across.

**(d) The conformal step.** The nonconformity is "sₖ = max{q̂l(xₖ) − yₖ, yₖ − q̂u(xₖ), 0}",
citing conformalized quantile regression [52]. Clipping the CQR score at zero makes cα ≥ 0 by
construction, so the procedure can only *widen* the base quantile interval and can never repair
over-coverage — which is the direction of every deviation reported in Fig. 4b (+0.03, +0.28,
+0.57). It also creates mass ties at zero, which voids the usual upper coverage bound
1 − α + 1/(n+1). The near-nominal numbers are therefore empirical observations about this
quantile model, with no conformal guarantee on the conservative side.

**Remedy.** Report achieved coverage conditional on situation cells (at minimum the categorical
strata) and on the deployment domain; state explicitly that the deployment-domain human
coverage is ≈95.3% at a nominal 90% and what that implies for reading the AV rate; use the
unclipped CQR score or justify the clipping; and soften "calibrated" in the abstract to
"marginally calibrated on held-out natural driving".

---

### M6. Abstention is endogenous to the monitored vehicle's behaviour, and on the deployment domain it removes most of the evidence

*Location: §2.4; Methods 4.3, 4.4, 4.5.*

§2.4: "Abstention here is a statement about the reference rather than about the vehicle, and it
is auditable as such." This is asserted, not demonstrated. The support gate "scores each moment
by the mean Euclidean distance to its k = 25 nearest training anchors" in the space of zₜ, and
zₜ comprises (Methods 4.3) "ego and counterpart velocity components and headings at the anchor;
relative position components, distance, velocity components, speed, closing rate and heading
difference at the anchor; short-window means and a dispersion of relative distance, relative
speed and closing rate", plus two online risk proxies. Every one of those is a function of the
ego's own behaviour. "The support gate compares kinematic neighbourhoods, not IPV values"
(Methods 4.5) answers a narrower question: it rules out dependence on the *reading*, not on the
*behaviour*. A vehicle that behaves unusually manufactures an unusual zₜ and is abstained. The
censoring is therefore informative with respect to the very quantity being estimated, and every
flag rate in the paper is conditional on surviving it.

The magnitude on the deployment domain makes this decisive rather than pedantic. Of 67,861
candidate moments, 55.3% are readable (0.553 × 67,861 = 37,527) and 14,099 are judgeable, so
**62.4% of readable benchmark moments are abstained for want of human support** (1 −
14,099/37,527), against 5.08% on the held-out natural-driving fold — a >12-fold increase. §2.3
describes the home-domain figure as "only a small fraction of readable moments (5.08% ...)" and
defers the benchmark figure; the benchmark figure is never stated as a percentage of readable
moments anywhere I could find. In the 36 always-silent runs the support pass rate is 0.07%.

**Why it matters.** For a runtime monitor, the pair (alarm rate, abstention rate) has to be read
jointly — the paper says as much for leave-one-source-out ("coverage and abstention must be read
together") but not for its own deployment result. If the gate is preferentially removing the
situations in which the reference would be wrong, the clean human alarm rate in §2.5 is partly
an artefact of the gate.

**Remedy.** Test whether abstention depends on the monitored agent's behaviour (e.g. abstention
rate as a function of the ego's reading in the immediately preceding readable frames, or of
kinematic extremity); report the benchmark abstention as a fraction of readable moments in the
main text; and report the AV/human comparison restricted to situation cells supported in both
arms.

---

### M7. The inference machinery is inconsistently specified, unadjusted for multiplicity, and demonstrably sensitive to choices the paper makes without justification

*Location: Methods 4.5; Fig. 2a, 3a, 5, 6 captions.*

**(a) The clustering unit is stated three different ways for the same analyses.** Methods 4.4
defines "an interaction case is one interacting vehicle pair within a scene" and "a scenario run
is one system's or one driver's traversal of it" — different objects. Methods 4.5 then says
"inference resamples scenario runs (the unit of clustering throughout)", but in the same
paragraph reports "the absolute case-clustered t" and "its case-clustered intervals exclude
zero at all three levels"; Fig. 5's caption says "resampled over the 175 scenario runs"; Fig. 6b
says "case-bootstrap 95% intervals"; Fig. 6a says "bootstrapped over driver-by-scenario runs";
Fig. 3a says only "95% bootstrap confidence intervals". A referee cannot tell which intervals in
Fig. 5 and Fig. 6b resample runs and which resample cases. Since cases nest within runs,
case-level resampling would be anti-conservative here.

**(b) No multiplicity control anywhere.** The words "multiplicity", "Bonferroni", "false
discovery" and "familywise" do not occur. Figure 5 alone displays 4 margin thresholds × 2 sides
+ 4 counterpart endpoints × 2 sides + 3 braking thresholds × 2 sides = 22 interval-based
contrasts, plus the ego median and upper-quartile ratios; Fig. 6b adds twelve. Several of the
"supported" results sit at the boundary: the headline counterpart speed reduction is
"+1.41 km h⁻¹ [+0.08, +3.37]" (Fig. 5c), a lower bound of 0.08 km h⁻¹ = 0.022 m s⁻¹, which is
not a behaviourally meaningful response.

**(c) The ratio-of-medians statistics are unstable and their intervals are not printed.**
Fig. 5c reports "2.06×" and "1.89×" against within-range medians of 1.33 and 2.93 km h⁻¹.
Methods 4.5 concedes the general point ("a ratio becomes unstable where the within-range median
is near zero, as it is in one of the two sites") but the panel prints point ratios with no
interval; from the plotted whiskers the AV speed-reduction ratio interval spans roughly
[1.09, 4.16] — a factor of four, with a lower end at parity. The difference and the ratio do
*not* tell the same story here: 2.06× reads as a doubling, +1.41 [+0.08, +3.37] km h⁻¹ reads as
a marginal detection.

**(d) The estimand is moment-weighted, and the weighting decides one of the results.** Methods
4.5: "Estimates are weighted by moment rather than by scenario run ... Weighted equally per run,
the assertive-side counterpart braking contrast does not reach significance (p = 0.4704)." The
rationale given (the monitor issues a verdict per moment) is reasonable but it means the
estimand is dominated by whichever runs contribute the most flagged moments. With 519 assertive
moments in 120 runs and only 20 runs containing a contiguous half-second of them, the
concentration could be extreme. No distribution of flagged moments across runs is reported.

**(e) The two permutation tests disagree, and the paper adopts the one that is significant.**
Methods 4.5: the exposure placebo gives "empirical p = 0.0199" (= 4/201, so 3 of 200 draws
exceeded the statistic; Monte-Carlo SE = √(0.02·0.98/200) = 0.010, so p = 0.020 ± 0.010 and is
not distinguishable from 0.04), while "A case-level label permutation on the same battery does
not reach significance (p = 0.1493)" (= 30/201, also 200 draws). The paper resolves this by
declaring that "the exposure placebo above is the test specific to flag timing". That may be
correct, but the two nulls are never written down, so the reader cannot judge whether the
significant test is the one that answers the question posed in §2.4 ("the association survives a
placebo test that reassigns whole flag sequences across scenario runs"). Nor is it stated which
endpoint the placebo statistic is computed on, or whether it was run for each endpoint.

**Remedy.** Fix one clustering unit and use it everywhere, or state per-panel which is used;
report the number of contrasts and either a multiplicity adjustment or an explicit
confirmatory/exploratory split; print ratio intervals wherever a ratio is printed; report the
per-run distribution of flagged moments and effective sample sizes; write out both permutation
nulls formally, increase the draw count to ≥2,000, and report both p-values for every endpoint.

---

### M8. The paper's self-declared "most informative finding" is a null with no equivalence margin, on a noise-dominated input

*Location: §2.3; §3 (Discussion); Methods 4.3.*

The Discussion states: "The most informative finding is a tension between two levels ... online,
conditioning the reference range on the counterpart's inferred preference — or on the ego
agent's own history — adds no measurable value beyond the current situation ... so a runtime
social monitor does not need to read the other agent's hidden intention." The supporting
evidence is one non-detection: "paired 90% interval-score difference −0.0002, case-clustered
p = 0.86" (Methods 4.3).

Two objections. First, the paper elsewhere applies exactly the right discipline to nulls —
"These are non-detections at the achieved precision ... and not demonstrations of equivalence"
(Methods 4.5, accommodating side) — but not to this one, which is escalated to a conceptual
conclusion. No equivalence margin, minimum detectable effect, or power statement is given.
Second, the added regressor is the counterpart's IPV, produced by the same estimator whose
frame-to-frame movement is 0.31 rad (Fig. 2b) and which is near-discrete (M4). A null from
adding a heavily attenuated regressor to a boosted quantile model does not license "does not
need to read the other agent's hidden intention"; it is at least as consistent with the
counterpart channel being too noisy to help.

**Remedy.** State an equivalence margin in interval-score units and report a two-one-sided-test
or an interval on the difference; test the mediation claim directly (does the counterpart IPV
predict the ego IPV after conditioning on zₜ?); and repeat the ablation with a smoothed or
window-aggregated counterpart reading to separate "no information" from "too noisy".

---

## 3. Minor issues

1. **Readability fraction does not reconcile.** Methods 4.4: "3,202,646 readable" of "4,497,368
   anchor rows" gives 3,202,646/4,497,368 = **71.21%**. Methods 4.5 states the same quantity as
   "70.3% vs 55.3%". Excluding solver failures from the denominator gives 71.24%. Please state
   the denominator that yields 70.3%, or correct one of the two figures.

2. **"The gain is present only for the real interaction" (Fig. 2a caption)** is inconsistent with
   the same panel: the different-partner control is −0.043 with a visibly tight interval — about
   a third of the −0.132 real-interaction value, and clearly not zero. "Weakens" (used in the
   next clause) is right; "only" is not. Also, the four rows have different cluster counts
   (4,743 / 4,605 / 4,701 / 8,130), which is odd for matched controls, and the baseline against
   which "change" is measured is never defined.

3. **Fig. 3a stratifies on a realised joint outcome.** The bands are formed on realised
   post-encroachment time, "an offline quantity used here for description only". PET is a
   function of both trajectories, and both agents' IPVs are estimated from those same
   trajectories; conditioning on it can induce a within-pair difference reversal with no change
   in underlying preferences. Worth an explicit caution, since this result motivates the whole
   conditioning argument.

4. **Gate pass-rate denominators are unstated.** "the human-support pass rate is 0.07% (against
   32.32% overall)" cannot be reconciled with 14,099/37,527 = 37.6% unless 32.32% is taken over
   candidate rather than readable moments. Please state the denominator for each rate.

5. **One situation cell dominates the reference.** "one high-volume situation cell with
   1,148,133 human anchor rows" is 47.0% of the 2,442,625-row reference pool (25.5% of the
   4,497,368-row ledger). For a "context-conditioned" reference this deserves comment.

6. **Base rates missing in Fig. 5b,d.** "flagged moments are one-third to one-half as frequent
   as typical ones" cannot be checked because only differences are printed. Reading the plotted
   points, the <2 s assertive contrast is ≈4.7% vs ≈8.9%, a ratio of ≈0.53, marginally outside
   the stated band. Please tabulate the underlying rates.

7. **Inconsistent "n" conventions.** Fig. 2a reports n as case clusters; Figs. 4, 5, 6 report n
   as moments. Given design effects of order 8–9 (M3), moment counts overstate precision to a
   reader skimming the panels.

8. **Fig. 6 funnel prints "count unavailable"** for the automated arm's Gate-1 count. For a
   paper whose selling point is auditability, this should be recoverable (it is 0.553 × 67,861 ≈
   37,527).

9. **Fig. 4c** defines R² as "1 − SSE/SST of the conditional median against the test-fold mean".
   A median predictor does not minimise squared error, so this understates the explained
   variance and is not comparable with the R² in Fig. 3c unless the latter is defined the same
   way.

10. **Fig. ED1's illustrative pair** is drawn from near the tails of the width distribution
    (1.49 and 2.19 rad against a mean of 1.87, 5th–95th 1.35–2.28). Say so.

11. **Preference ratings were excluded as endpoints** (Methods 4.6). Given that the
    "human-dispreferred" step is the next link in the paper's own validation chain, an
    explicitly exploratory report of those ratings would be far more valuable than their
    omission.

12. **Fig. 6a shows no interval on the AV bars** or on the natural-driving bars, only on the
    human same-scenario 90% bar.

---

## 4. Questions to the authors

1. What are the case/run-clustered 95% intervals for the assertive-side ego-margin median ratio
   and upper-quartile ratio, on both the fixed three-second window and the open-ended window,
   at all three levels? Why is the open-ended window used for Fig. 5a,b when Methods 4.5 names
   the three-second window as the pre-specified primary?
2. Fig. 6b carries no asterisk on "Ego margin (median)" or "Ego margin (upper quartile)" for
   either arm. Do you agree that both intervals admit parity, and if so how do you reconcile
   this with the abstract's "tighter interactions on both sides"?
3. Given that the accommodating-side ego-margin median ratio is 0.81× [0.70, 1.15] and the
   assertive-side value is 0.751×, on what statistic do you distinguish the two sides on this
   endpoint?
4. What is the empirical distribution of θ̂ over accepted moments? Is it atomic at the seven
   grid values? What fraction of accepted moments have a lower reference edge at or below
   −3π/8, and an upper edge at or above +3π/8?
5. What is the distribution of verdict run-lengths and frame-to-frame verdict flips? Of the 519
   assertive moments, how many belong to contiguous stretches of ≥3 frames, and how are they
   distributed over the 120 runs?
6. Which resampling unit — interaction case or scenario run — produced each interval in Figs. 5
   and 6b? What are the effective sample sizes?
7. Write out the two permutation nulls explicitly. On which endpoint is the placebo statistic
   computed, and what are the placebo and label-permutation p-values for *every* endpoint in
   Fig. 5?
8. What fraction of *readable* benchmark moments is abstained for lack of human support, and
   does abstention depend on the monitored agent's own kinematic extremity?
9. What is the equivalence margin for the counterpart-IPV ablation, and what interval-score
   improvement would the design have detected at 80% power?
10. Why does Fig. 4a report the global range as 100.00% of the admissible span at both the 90%
    and 95% levels, when the achieved coverages differ by 7.8 pp? Are the reported widths
    clipped at the admissible span?
11. Which denominator gives the 70.3% readability figure in Methods 4.5?
12. 19 systems × 15 scenarios = 285 cells, of which 267 yield anchors and 13 of the 18 losses
    "fall in a single system". Is any result sensitive to dropping that system entirely?

---

## 5. Prioritised revision requests

1. **Restate the consequence result to match the intervals.** Remove "tighter interactions on
   both sides" from the abstract unless the ego-side intervals support it; put clustered
   intervals on every quantity in Fig. 5a and Fig. 6b; state per panel which outcome window is
   used and make the pre-specified primary the displayed one.
2. **Fix the two-sided story.** Retitle Fig. 5a; correct "tracks the within-range one across its
   whole body"; correct the singular "<3 s interval" statements; report all three groups'
   attrition and a sensitivity analysis retaining non-closing moments.
3. **Decompose the 2.1× headline** by side and by level, with intervals, and say in the main
   text that ~79% of the excess is on the accommodating side.
4. **Characterise the estimator's discreteness and the flag geometry**: distribution of θ̂,
   fraction of moments structurally unflaggable per side, verdict run-lengths and flip rates,
   sensitivity to σ and grid resolution.
5. **Strengthen the calibration reporting**: conditional coverage by situation cell; explicit
   statement that deployment-domain human coverage is ≈95.3% at nominal 90%; unclipped CQR score
   or a justification for the clipping; remove or qualify "sits at the native level".
6. **Fix the inference specification**: one clustering unit, stated everywhere; multiplicity
   handling or a declared confirmatory subset; ratio intervals wherever ratios are printed; both
   permutation tests for all endpoints with ≥2,000 draws.
7. **Treat the counterpart-ablation null as a null**: equivalence margin, power, and a direct
   test of the mediation claim.
8. **Add the abstention-endogeneity analysis** and report benchmark abstention as a fraction of
   readable moments in the main text.
9. Minor items 1–12.

---

## 6. Acceptance probability

- **(a) As submitted: 5%.** The abstract's principal empirical claim is not supported by the
  paper's own intervals, and the side-asymmetry and population-ratio headlines do not survive
  decomposition. These are not presentation defects; they change what the paper reports.
- **(b) After a competent major revision: 35%.** The framing, the abstention semantics and the
  matched human-arm audit are genuinely strong and well suited to this journal. But once the
  ego-side compression claim is withdrawn or properly bounded, the surviving empirical payload
  is thinner — a counterpart-side speed signature, an emergency-tail null, and a flag-rate
  contrast largely on the inert side of the range — and whether that clears the bar here is a
  real editorial question rather than a formality.

---

## 7. Recommendation

**Major revision.**
