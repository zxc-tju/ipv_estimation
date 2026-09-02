# Referee report — "Online monitoring of socially compliant autonomous driving"

**Remit: generalist editor-referee (breadth, venue fit, claim support, declarations, display items).**

---

## 1. Summary assessment

The paper reframes social compliance for automated vehicles as an online membership test: is the
ego's current interaction preference value (IPV) inside a human reference range conditioned on the
observable situation, and if the reading is not identifiable, abstain. It delivers a
situation-conditioned conditional-quantile reference calibrated by split conformal, an explicit
abstention taxonomy, a large naturalistic characterisation (38,228 interaction cases), and two
applications on a staged real-vehicle benchmark: an automated-systems arm and a matched human arm.

The construction is careful, the abstention discipline is genuine, and the "audit the reference with
human drivers in the same scenarios" move is the most original idea in the paper. But the delivered
evidence does not reach the level the first two pages claim. The one empirical link between a flag
and anything consequential sits where the vehicles are not meaningfully interacting; one of the two
"sides" of that interaction is traffic-microsimulation software, disclosed only in Methods; the human
comparison arm is not apparatus-matched to the machine arm and is significantly *under*-flagged by
the reference, which the text reports one-sidedly; and the abstract misdescribes which population
defined the reference. Transferable content beyond driving amounts to one Discussion sentence.

*(197 words)*

---

## 2. Major weaknesses

### MW1 — [Whole paper; Discussion, final paragraph] The advance is a framing, and the framing is demonstrated in exactly one domain with one construct. The transferable content is a single sentence.

The paper's claim on a broad machine-intelligence readership rests on the idea that one can monitor
machine behaviour against a *conditional* human reference, abstain where the measurement is not
identifiable, and audit the reference by running its defining population through the same instrument.
That is a real idea. But the entire delivery for a non-driving reader is one sentence: "Beyond
driving, the framework offers a template for monitoring whether autonomous agents behave within human
normative ranges online, under uncertainty and distribution shift, while being explicit about when a
social judgement should be withheld" (Discussion, final paragraph).

Nothing else in the paper is domain-general in substance. The statistical machinery is entirely
off-the-shelf and the authors say so: histogram gradient-boosted quantile regression, split-conformal
calibration (Methods 4.4: "we use the split-conformal form for auditability"), a k-nearest-neighbour
support gate, and a reject option. There is no new estimator, no new calibration guarantee, no new
theory of when a behavioural construct is identifiable. The monitored quantity is a single scalar
recovered by inverse planning against a two-term hand-specified utility on a seven-point grid — a
driving-specific instrument.

*Why it matters here.* At this journal the editorial question is whether a reader in language-model
oversight, clinical decision support, or robot manipulation learns a method they can carry away. As
written they learn a slogan. The specialist reader, meanwhile, gets a well-made contribution that
sits naturally in a transportation or robotics venue.

*Remedy.* Either (i) instantiate the construction on one non-driving behavioural stream, however
small, so that "conditional reference range + readability gate + population audit" is shown to be a
transferable recipe rather than a description of this pipeline; or (ii) abstract the recipe
explicitly — state the conditions a construct must satisfy for this monitor to be well-posed
(identifiability signature, conditional support, an auditing population), and show they are checkable
outside driving. Option (ii) is cheap and would materially change the venue argument.

### MW2 — [Abstract] Two of the four empirical sentences in the abstract are not supported as written; one is contradicted by the paper's own figure.

(a) **"Auditing the reference with its defining population—human drivers in the same scenarios—shows
no alarm inflation…"** The population that defined the reference is *not* the human drivers in the
same scenarios. Methods 4.3: "The corpora are fleet-collected in specific cities in the United States
and Singapore; the reference is a sample of natural driving there, not of all human driving." Section
2.5 opens by saying the same: "The reference range was learned from natural human driving recorded in
other countries on open roads." Figure 6a's own legend separates the two as distinct arms — "Humans,
natural driving (held-out), n = 461,937" versus "Humans, same scenarios, n = 15,102". The section
heading ("The reference is audited by the population that defined it") repeats the error. The audit
is a *transfer* test to a different human population in a different country on a different apparatus,
which is a weaker and more interesting claim than the one the abstract makes.

(b) **"…with no rise in emergency rates at any supported threshold."** "Supported" here carries a
private definition that exists only in the Figure 5 caption: "A threshold is called supported when
its per-endpoint interval excludes a zero difference — a statement about the interval, distinct from
the monitor's human-support gate; the panel titles use the word in exactly this sense." A reader of
the abstract cannot decode this, and the sentence as parsed by an ordinary reader ("safety was
unaffected") is both weaker and stronger than the truth: at three of four thresholds emergencies are
*less* frequent at flagged moments, and at the fourth (< 3 s) the interval is uninformative. Using a
term in the abstract that requires a caption to disambiguate against the paper's own second use of
the same word is not acceptable at triage.

(c) **"…automated systems flagged about twice as often."** See MW5: the ratio is level-dependent
(1.74× at 80%, 2.09× at 90%, 2.27× at 95%, from Figure 6a), and its composition is not what the
sentence implies.

*Remedy.* Rewrite the abstract so each empirical sentence survives a literal reading. Say "human
drivers driving the same staged scenarios in a different country" rather than "its defining
population". Drop "supported" or replace it with "at every threshold we could resolve". State the
ratio with its level and its decomposition.

### MW3 — [Section 2.4; Fig. 5a,b] The only empirical link between a flag and a consequence lies entirely in the region of the outcome distribution where the two vehicles are not meaningfully interacting.

I read the quantiles directly off Figure 5a (pixel measurement calibrated on the decade ticks;
my reconstruction reproduces the paper's own reported lower quartile to within 1%, so the
calibration is sound):

| Quantile of post-verdict minimum TTC | Within human range | Flagged assertive | Paper's figure |
|---|---|---|---|
| 25th | 4.24 s | 4.11 s | "4.09 vs 4.18 s" ✓ |
| 50th | 8.89 s | 6.68 s | "median falling 24.9%" ✓ (6.68/8.89 = 0.751) |
| 75th | 21.9 s | 11.6 s | "the upper quartile 47.4%" ✓ (11.6/21.9 = 0.53) |
| 90th | ~83 s | ~28 s | — |

So the entire "compression of ordinary interaction quality" is a shift in minimum time-to-collision
from about 8.9 s to about 6.7 s at the median, and from about 22 s to about 12 s at the upper
quartile. Both regimes are an order of magnitude above any margin at which a driver or a planner
responds. Below the 25th percentile — around 4 s, the only part of this distribution where TTC is
behaviourally live — the two groups are indistinguishable, and in the genuine emergency tail (< 2 s,
< 1.5 s, < 1 s) flagged moments are *strictly rarer* (Fig. 5b). The counterpart-side companion result
is a median speed reduction of 2.74 vs 1.33 km h⁻¹ (Fig. 5c caption) — an absolute difference of
1.4 km h⁻¹, presented in the figure as "2.06×" on a log axis running to 20×, and in the text as
"absorbs roughly twice the routine speed reduction".

Three further problems compound this.

- **No uncertainty on the two headline numbers.** "−24.9%" and "47.4%" appear only as annotations in
  Fig. 5a and in its caption. The caption states that intervals come from bootstrap resamples for
  panels b, c and d; panel a carries none. The two numbers the Results lead with therefore have no
  interval anywhere in the manuscript. Figure 6b, which repeats the ego-margin median as a ratio for
  both arms, shows that row with neither a whisker nor an asterisk — i.e. either undefined or
  admitting parity.
- **The plotted window may be the non-significant one.** Methods 4.5 defines the ego-side outcome as
  "the minimum time-to-collision to the counterpart over the post-verdict window, which runs from the
  verdict to the end of that run's evaluated window (the counterpart-side battery instead uses the
  fixed three-second window below)". The same section then reports: "the fixed three-second window is
  the pre-specified primary, and its case-clustered intervals exclude zero at all three levels, while
  the open-ended contract-window interval crosses zero at the 90% level ([−2.6100, +0.1372])". Every
  verdict in the paper is issued at the 90% level (Fig. 4 caption). On the natural reading, Fig. 5a/b
  — the flagship panels — display the open-ended window, whose interval at the operative level
  includes zero, while the interval that excludes zero belongs to a window that is never plotted. The
  term "contract-window" appears exactly once and is never defined, so I cannot close this from the
  text; it must be resolved.
- **The open-ended window has variable length.** Its length depends on when in the run the verdict
  occurs. A longer remaining window mechanically lowers a *minimum*. The paper never reports whether
  flagged and within-range moments have comparable remaining-window durations. If flagged-assertive
  moments occur earlier in runs, the whole panel-a result is an artefact.

*Why it matters here.* This is the paper's answer to "so what". If the answer is "the minimum TTC
falls from 22 s to 12 s at the upper quartile while genuine emergencies become rarer", a broad
readership will not accept that a socially meaningful event has been detected.

*Remedy.* Report absolute medians and quartiles with bootstrap intervals; plot the pre-specified
window; report remaining-window length by group; and either identify an outcome that is behaviourally
interpretable at 4 s and below, or state plainly that the flagged moments differ only in the
non-interacting tail.

### MW4 — [Section 2.4 vs Methods 4.5] The main text and abstract describe a two-vehicle real interaction; one vehicle is traffic-microsimulation software, and this is disclosed only in Methods.

The words "simulation", "simulated", "mixed-reality", "TESS NG" and "head-mounted display" appear
nowhere in the abstract, Introduction, Results or Discussion. They appear only in Methods 4.5–4.6:
"The counterpart vehicles are driven by traffic-simulation software (TESS NG) and respond to what the
ego vehicle does"; "The staging is mixed-reality: the ego is a real vehicle driven at a test site,
and the simulated counterparts reach an automated system through its perception interface and a human
driver through a head-mounted display."

What the main text says instead: "a matched-scenario real-vehicle benchmark … in which identical
scenarios are replayed against many independent driving systems" (2.4); "The counterpart, read
independently from the other vehicle's own control record, absorbs roughly twice the routine speed
reduction … The two sides are measured from different sources and different quantities, so their
agreement is a convergence rather than a restatement" (2.4); and the abstract's "tighter interactions
on both sides". Figure 5's caption reinforces it: "Speed quantities are read from the other vehicle's
own logged control record."

A reader spending ninety seconds on the abstract and Figure 5 will conclude that a second real
vehicle independently corroborated the ego-side finding. In fact the second "side" is a car-following
model whose response to a fast-approaching ego is a designed property of the software, and the claim
of *independent convergence* between the two sides is correspondingly weak: both quantities are
downstream of the same ego trajectory, one through the ego's own kinematics and one through a
deterministic reactive model of it.

*Why it matters here.* This is not a subtle over-claim; it changes what the central result is. It is
also exactly the kind of thing that decides a triage outcome.

*Remedy.* State in the abstract and at the first mention in Results that the counterpart is
simulator-driven and reactive; delete or heavily qualify "read independently" and "convergence rather
than a restatement"; and rename the counterpart quantities so no reader takes them for a second real
vehicle's log.

### MW5 — [Section 2.5; Fig. 6] The human comparison arm is not apparatus-matched, the reference significantly *under*-flags it, and the "twice as often" headline is composed mainly of the flag class the paper says has no consequence.

Three separate problems in the paper's most important comparison.

(a) **Not apparatus-matched.** The automated systems receive the simulated counterparts "through
[their] perception interface"; the human drivers receive them "through a head-mounted display"
(Methods 4.5). Section 2.5 nonetheless states that human drivers drove "against the same counterparts
under the same control, and the unchanged monitor — same estimator, same gates, same frozen reference
— audits them exactly as it audits the automated systems." The counterpart controller is the same;
the sensing channel, field of view, latency and task demand are not. Driving a real vehicle at a test
site while wearing an HMD, with a safety supervisor aboard, is a different task from driving with
one's own eyes, and it is a task that would plausibly push behaviour toward the cautious middle. That
confound is unmeasured and is aligned in sign with the reported result.

(b) **The deflation is significant and is reported one-sidedly.** Figure 6a gives the human
same-scenario flag rate as 4.7% with a 95% CI of 3.8–5.8%, against a native outside rate of 9.72%.
The interval excludes the native rate comfortably. So the reference does not transfer to the human
arm — it *over-covers* it by a factor of about two, i.e. it is systematically insensitive on that
apparatus. The text records this as "no more often than that native rate: carrying the reference
across country, apparatus and task does not inflate its alarms." Calibration is two-sided.
Under-flagging by 2× is a transfer failure, not a clean bill of health, and it is precisely what
manufactures the AV-versus-human gap: the automated systems are flagged at essentially the nominal
rate (19.5 / 9.8 / 5.0 against nominal 20 / 10 / 5), so the anomalous arm in Figure 6a is the human
one, not the machine one.

(c) **The 2.1× is dominated by the inert flag class.** From the side counts printed under Figure 6a
at the 90% level — humans 391 assertive | 322 accommodating (713/15,102); AV 519 | 869 (1,388/14,099):

- assertive side: AV 3.68% vs humans 2.59% → **1.42×**
- over-yielding side: AV 6.16% vs humans 2.13% → **2.89×**

The paper states of the over-yielding side: "The complementary direction—a preference more
accommodating than the human range—carries no such signature on either side" (Section 2.4). So the
abstract's "automated systems flagged about twice as often" is carried mainly by the flag class the
paper's own consequence analysis found inert. The ratio is also unstable across the three nominal
levels shown in the same panel (1.74×, 2.09×, 2.27×); the abstract quotes the middle one without the
level.

(d) **Clustering.** The human arm has 20 participants. Figure 6a's interval is "bootstrapped over
driver-by-scenario runs" (300 units), not over drivers (20). If driver effects are non-trivial this
understates uncertainty on the very quantity that anchors the headline ratio.

*Remedy.* Report the human-arm flag rate against nominal as a two-sided calibration result and say
plainly that the reference over-covers this population. Decompose the ratio by side and by nominal
level. Cluster on driver. Either run a subset of drivers without the HMD, or retire the claim that
the two arms are audited "exactly" alike and treat the ratio as apparatus-confounded.

### MW6 — [Section 2.4 vs Section 2.5] Direct internal contradiction on whether the monitor separates machines from humans as populations.

Section 2.4: "The verdict does not separate automated driving from human driving as a population; it
locates the particular moments at which the interaction that follows is measurably tighter."

Section 2.5, two pages later: "under this common instrument the automated systems are flagged 2.1
times as often as the matched human drivers (9.8% vs 4.7% of judgeable moments), and the per-scenario
flag rate is higher for the automated systems in 15 of 15 scenarios."

The second is a population-level separation and is the third empirical sentence of the abstract. The
mitigating sentence that follows ("The population-level rates answer a calibration question … neither
is a ranking of vehicles or drivers") does not repair it: a 2.1× population rate difference either is
or is not a separation of the two populations. As it stands the paper asserts both.

*Remedy.* Decide. If the ratio is a calibration diagnostic, remove it from the abstract. If it is a
finding about machine behaviour, delete the disclaimer in 2.4 and defend the comparison against MW5.

### MW7 — [Section 2.4; Fig. 6 funnel; Methods 4.4–4.5] In the setting the monitor is built for, it cannot judge four moments in five, and the reason is structural rather than incidental.

On the benchmark the monitor "returns a verdict on 14,099 [of 67,861] (20.8%)". Of the moments it can
*read*, only 20.8/55.3 = 37.6% are judgeable; **62% of readable benchmark moments abstain for lack of
human support**, against 5.08% on the natural-driving test fold — a twelvefold increase. Methods 4.5
identifies the mechanism precisely: in the 36 always-silent runs, readability is normal (53.78% vs
55.30% overall) while "the human-support pass rate is 0.07% (against 32.32% overall)".

That is not a data-volume problem, as the authors themselves demonstrate ("one high-volume situation
cell with 1,148,133 human anchor rows passes support on 14.58% of its benchmark moments while a
45,283-row cell passes on 47.03%"). It is the expected consequence of building a reference from
naturalistic driving and applying it to *deliberately staged conflicts*: the situations of greatest
interest are, by construction, the ones with the least human precedent. The monitor is therefore
systematically silent where social atypicality is most likely to occur — a structural ceiling on the
method, not a tuning issue.

The paper frames this as a virtue throughout ("Abstention here is a statement about the reference
rather than about the vehicle, and it is auditable as such"). Abstention *is* the right behaviour; but
a runtime monitor that declines to speak on four fifths of the candidate moments in the only
deployment-like evidence presented has not yet been shown to be operationally useful, and the paper
never states the 62%/79% figures in one place.

*Remedy.* Report the silence rate as a headline operating characteristic, not a decomposition.
Characterise *which* situations lose support (conflict severity, closing rate, geometry). State
explicitly whether the abstentions are concentrated in the highest-conflict moments, and if so, say
what that implies for deployment.

### MW8 — [Whole paper] The monitored construct is never validated against anything external, and the one external criterion the authors possess was excluded by design.

Figure 2a shows that the candidate weights concentrate more sharply inside real interaction windows
than in three scrambled controls. That is an identifiability check, not construct validation: it shows
the estimator responds to real interaction structure, not that θ̂ measures anything a human would
recognise as assertiveness or accommodation. There is no comparison against human ratings of the
behaviour, no convergent measure, no manipulation check. The candidate model — "cos θₖ ·
(own-progress cost) + sin θₖ · (interaction cost)" on a seven-point grid with a fixed Gaussian
likelihood (σ = 0.1 m) — is certainly misspecified relative to real drivers, and any misspecification
is absorbed into θ̂ and hence into every flag.

The paper sets out its own validation ladder in the Introduction: "human reference range → conditional
social atypicality → human-dispreferred behaviour → interaction-harmful behaviour." Neither of the
last two rungs is attempted. The harm rung is honestly bounded (Section 2.4). But the
human-preference rung is not attempted *despite the data being in hand*: Methods 4.6 states that
"Official competition scores, harm labels and preference ratings were excluded as endpoints in the
analysis plan". Preference ratings exist in the benchmark the authors' own group operates.

*Why it matters here.* A paper whose entire claim is "we can monitor social behaviour" cannot leave
the validity of the social measurement untested, particularly when the decisive test is available and
was set aside. Pre-specification protects against endpoint shopping; it does not require that an
available criterion go unanalysed. A pre-registered secondary or an explicitly labelled exploratory
analysis would have cost nothing.

*Remedy.* Analyse the released preference ratings against flag status, labelled as
secondary/exploratory with the pre-specification history stated. If that analysis cannot be run,
explain concretely why, and add some external anchor for the construct (e.g. expert or naive-observer
ratings of a sample of flagged and within-range moments).

### MW9 — [Figs. 1, 2, 3, 5, 6, ED1, ED2] The display items do not carry the argument to a non-specialist, and two of them work against it.

Judged as a reader who will give them ninety seconds:

- **Fig. 6b is direction-ambiguous.** The axis is annotated "compressed at flagged moments" (left of
  parity) and "elevated" (right). But the left side contains both worse outcomes (ego margin median
  and upper quartile) and *better* ones (ego emergency < 2 s; counterpart braking < −3 m s⁻²), while
  the right side contains worse ones (counterpart speed reduction and variation). A reader cannot tell
  which points are good news. This is the paper's summary display item for the human-versus-machine
  audit.
- **Fig. 5a's caption gives absolute values only where nothing happens.** "4.09 vs 4.18 s" is given for
  the lower quartile — the quantile the paper describes as "nearly unchanged" — while the median and
  upper quartile, where the effect lives, appear only as percentages. That inverts the usual reporting
  priority and prevents a reader from seeing that the effect is at 7–22 s (MW3).
- **Fig. 3a magnifies a very small effect.** The y-axis spans roughly ±0.07 rad, about 3% of the
  2.36 rad admissible span, so a "sign reversal" of ±0.05 rad reads as dramatic. The text is honest
  ("The role shifts are small against the width of the human range — a few per cent of it"), but the
  figure is not, and the figure is what triage sees.
- **Fig. 3c is a negative result titled as a positive.** "Why the preference must be read online"
  displays held-out-source R² of +0.026, +0.017, −0.195, −0.276 — i.e. in two of four folds the fitted
  state description is worse than predicting the held-out source's mean. That is a transfer failure;
  the title reframes it as a design rationale. It also sits uneasily with deploying the frozen
  reference in a different country (see MW5, and Methods 4.4's leave-one-source-out coverage of 0.743
  for Waymo and 0.750 for Lyft against nominal 0.90).
- **Fig. 2b prints "uncertainty not shown".** Four point estimates support the claim that "a readable
  moment is not a settled one", with no intervals. So does Fig. 6c ("uncertainty not shown").
- **Fig. 1's worked example flags the wrong side.** The illustrative flags in panel c are on the
  over-yielding side — the side the paper later reports "carries no such signature on either side".
  The paper's opening figure therefore illustrates the verdict class with no demonstrated consequence.
- **ED1's insets occlude the data.** The two inset boxes are drawn over the upper-right region of the
  main time series, hiding part of the band and trace they are meant to explain. The "80%" band label
  also floats mid-band rather than sitting at an edge, unlike Fig. 1c.
- **ED2 shows the counterpart decelerating before the flag.** In panel b the counterpart's speed falls
  from ~4.9 to ~1.5 m s⁻¹ *within the shaded pre-alert baseline*, before the first red alert band. The
  quoted "4.31 → 1.40 m s⁻¹" therefore compares a baseline median that already contains most of the
  fall against the alert stretch. In the single run selected to illustrate the mechanism, the visual
  evidence does not show the flag preceding the counterpart's response.

*Remedy.* Rebuild Fig. 6b with an unambiguous good/bad encoding; put absolute values on Fig. 5a's
median and upper quartile; give Fig. 3a a y-axis referenced to the admissible span or an inset showing
it; retitle Fig. 3c; add intervals to Fig. 2b and 6c; choose an assertive-side example for Fig. 1; move
ED1's insets outside the axes; re-baseline ED2 to a window that ends before the counterpart's response
begins.

### MW10 — [Methods 4.6; Data/Code Availability; Competing Interests] Human-participant reporting and availability statements fall short of what the standards desk requires.

- **No participant characteristics whatsoever.** The manuscript reports only "Licensed drivers",
  "20 drivers × 15 scenarios", "Participants held a valid driving licence, were compensated at CNY 150
  per hour, and were instructed to drive naturally". There is no age, no sex or gender, no driving
  experience, no recruitment route, no inclusion/exclusion criteria, no sample-size justification, and
  no statement on how sex and gender were considered in the design. Nature Portfolio requires this for
  research with human participants; a Reporting Summary is not mentioned anywhere in the manuscript.
- **Incomplete disclosure without a stated debriefing.** "they were not told that their runs would
  serve as a human reference against automated systems." This is deception by omission. Approval
  (tjdxsr2025011) is stated, but the manuscript does not state that the committee approved the
  incomplete disclosure, nor that participants were debriefed afterwards, nor that they could withdraw
  their data after learning the true purpose.
- **The data behind an abstract-level claim will never be released.** "Records from the human
  reference arm are not released… the human arm is reported only in aggregate." The 2× comparison is
  the third empirical sentence of the abstract. Aggregate-only reporting of a headline claim, from a
  benchmark operated by the authors' own group, with the automated arm's raw records also "available
  upon publication", leaves nothing independently checkable at review time.
- **Code is offered "on request".** "…are available to referees on request during peer review." For a
  paper whose contribution is an instrument defined by frozen thresholds, gates and a frozen split,
  the code and the frozen configuration should be deposited and accessible at submission.
- **Competing interests.** The declaration is candid and appropriate — the authors' group operates the
  benchmark, no author competed, labels are public, funding is separate. I do not regard this as
  disqualifying, but combined with the non-release of the human arm it raises the bar for
  independently verifiable reporting rather than lowering it.

*Remedy.* Add full participant characteristics and a Reporting Summary; state the ethics committee's
position on incomplete disclosure and the debriefing procedure; release de-identified moment-level
derivatives for the human arm (verdict series and kinematic summaries carry no re-identification risk
that pseudonymisation cannot handle) or obtain consent for it; deposit code now.

---

## 3. Minor issues

1. **Title and abstract are purely automotive.** "Online monitoring of socially compliant autonomous
   driving", opening with "Autonomous vehicles are engineered and assessed for collision safety…".
   Nothing in the first 60 words signals a general contribution. At triage this reads as a
   transportation paper.
2. **Fig. 2a caption contradicts itself.** "The gain is present only for the real interaction: it
   disappears when the same interaction is misaligned in time (+0.006), *weakens* when the same moment
   is paired with a different partner (−0.043)…" A weakened gain is still a gain, so "only" is wrong.
   Separately, the panel's units and baseline are never given: −0.132 in what, relative to what
   absolute level of concentration? A reader cannot judge whether it is large.
3. **"With no engineering failures in this replay" (2.4) sits oddly beside Methods 4.5**, which records
   18 of 285 system–scenario cells lost in the authors' own replay processing, plus one of 20 systems
   set aside entirely, with "13 of the 18 fall[ing] in a single system". The Results sentence is
   scoped to candidate moments, but the plain reading overstates. Since the AV arm feeds the 2.1×,
   please report flag rates with and without the near-absent system.
4. **The placebo test is thin.** p = 0.0199 is 4/201, i.e. 3 of 200 permutation draws exceeded the
   observed statistic. Use ≥10,000 draws. Also, the Results report only that "the association survives
   a placebo test", while Methods discloses that "A case-level label permutation on the same battery
   does not reach significance (p = 0.1493)". The pre-specification argument for preferring the
   exposure placebo is reasonable, but both results belong in the main text.
5. **ED1 shows a signal that swings across most of the admissible span within tenths of a second**, and
   every reported verdict and flag rate is per-moment on that signal, with the persistence layer
   explicitly unused ("no result in this paper uses it"). Please state what a single-moment verdict
   means on a signal with this much frame-to-frame movement, and report flag rates under a minimal
   persistence rule as a robustness check.
6. **A 90% range averaging 1.87 rad of a 2.36 rad span** (Section 2.3) is 79% of the admissible
   parameter range; at 95% the conditioned range is 92% of it (Fig. 4a). "Stays informative at the
   strictest level" is a stretch for a range covering 92% of the scale.
7. **"Absorbs" is causal language** ("the counterpart absorbs roughly twice the routine speed
   reduction", 2.4 and 2.5) in a paper whose figure captions correctly say "Associations are
   descriptive".
8. **Fig. 6's funnel prints "count unavailable"** for the automated arm's readable-moment count in a
   main display item. The number is recoverable from the printed 55.3%; print it.
9. **No Supplementary Information** accompanies a paper with this many frozen thresholds, gates,
   ablations and analysis batteries.
10. **"Contract window" (Methods 4.5) is used once and never defined.**
11. Section 2.2's finding that "the two interacting agents' preferences [are] correlated at this event
    level" is asserted without a reported correlation coefficient, sample or interval, yet it is one
    half of the "most informative finding" of the Discussion.

---

## 4. Questions to the authors

1. Which window does Figure 5a/b plot — the fixed three-second window or the open-ended one? If the
   latter, why is the flagship panel the specification whose interval crosses zero at the 90% level,
   the level at which every verdict in the paper is issued?
2. What are the absolute median and upper-quartile post-verdict minimum TTCs for the two groups, with
   bootstrap intervals? Do you accept that a shift from ~22 s to ~12 s at the upper quartile describes
   vehicles that are not interacting?
3. Is the mean remaining-window length equal between flagged and within-range moments? If not, how much
   of the panel-a difference survives conditioning on it?
4. Why is the counterpart's simulated, reactive nature absent from the abstract and Results? In what
   sense are the ego-side and counterpart-side measures "independent" when both are downstream of the
   same ego trajectory through a deterministic car-following model?
5. On what basis is a human driver wearing a head-mounted display "audited exactly as" an automated
   system receiving the same counterparts through its perception interface? What evidence bounds the
   HMD's effect on IPV?
6. The matched human arm is flagged at 4.7% [3.8, 5.8] against a native 9.72%. Do you accept that the
   reference over-covers this population by a factor of about two, and that this is a transfer failure
   rather than "no alarm inflation"?
7. Decomposed by side, the AV:human ratio is 1.42× (assertive) and 2.89× (over-yielding). Given that
   the over-yielding side "carries no such signature on either side", what does the abstract's "twice
   as often" tell a reader?
8. How do you reconcile "The verdict does not separate automated driving from human driving as a
   population" (2.4) with the abstract's population-level 2.1×?
9. What fraction of *readable* benchmark moments abstain for lack of human support, stated as one
   number? Are abstentions concentrated in the higher-conflict moments, and if so what does the monitor
   offer at deployment?
10. Preference ratings exist in the benchmark your group operates and were excluded as endpoints. Can
    they be analysed now as a declared exploratory endpoint? If not, why not?
11. What evidence is there that θ̂ measures social assertiveness rather than residual misspecification
    of the two-term candidate utility?
12. Excluding the system that lost 13 of its 15 cells and the system set aside entirely, what is the AV
    flag rate? Is the excess broad across the 19 systems or concentrated in a few?
13. Participant age range, sex/gender distribution, driving experience, recruitment route, and
    sample-size rationale for n = 20; and did the ethics committee approve the incomplete disclosure,
    with what debriefing?

---

## 5. Prioritised revision requests

1. **Resolve the window question and re-plot Figure 5 on the pre-specified window**, with absolute
   quantiles and bootstrap intervals for the median and upper quartile, and a report of
   remaining-window balance. If the effect is confined to TTC above ~5 s, say so in the Results.
2. **Disclose the simulated, reactive counterpart in the abstract and at first mention in Results**,
   and withdraw the "independent … convergence rather than a restatement" characterisation.
3. **Rewrite the abstract** so that (i) the audit population is described accurately, (ii) "supported
   threshold" is removed or replaced, (iii) the AV:human ratio is given with its level and its
   assertive/over-yielding decomposition, and (iv) the counterpart's nature is stated.
4. **Report the human-arm calibration two-sidedly** and analyse the HMD as a confound; cluster
   inference on driver (n = 20); reconcile with the Section 2.4 disclaimer.
5. **Add an external validity anchor for the IPV**, ideally the existing preference ratings as a
   declared exploratory endpoint.
6. **State the silence rate as a single headline number** and characterise which situations lose human
   support.
7. **Bring the human-participant reporting up to standard**: full characteristics, Reporting Summary,
   ethics position on incomplete disclosure and debriefing; deposit code now; release de-identified
   human-arm derivatives.
8. **Rebuild the display items** per MW9, in particular Figure 6b's direction encoding and Figure 3c's
   title.
9. **Make the generality concrete** — either a non-driving instantiation or an explicit, checkable
   statement of the conditions under which this recipe transfers — and revisit the title.
10. Strengthen the placebo test to ≥10,000 draws and report both permutation schemes in the main text.

---

## 6. Acceptance probability

- **(a) As submitted: 4%.** The abstract contains a claim contradicted by the paper's own figure, the
  central consequence result sits where nothing is at stake, one side of the headline interaction is
  undisclosed simulation software, and the human-participant reporting does not meet the standards
  desk's requirements.
- **(b) Assuming a competent major revision: 22%.** Several of the requests above are writing and
  re-analysis and would materially improve the paper. But two of the most serious problems are not
  repairable by revision: the location of the consequence effect in the non-interacting tail is a
  property of the data, and the apparatus mismatch between the human and machine arms would need new
  data collection. Absent those, the paper would remain a careful, honest, single-domain contribution
  whose central validation is explicitly incomplete.

---

## 7. Recommendation

**Reject.**

I want to be clear about what I am not saying. The paper is unusually disciplined: the abstention
taxonomy is principled and consistently applied; the monitoring-versus-certification framing is held
throughout (I checked — there is no slippage into "verify" or "guarantee" anywhere in the authors'
own claims); the limitations sections are candid; the accounting of dropped runs and cells is more
thorough than most submissions. The idea of auditing a normative reference by running its defining
population through the same instrument is a good one and I would like to see it published somewhere.

But my remit is whether this belongs in a broad machine-intelligence venue, and on the evidence
presented the answer is no. The transferable content is one sentence; the statistical apparatus is
standard; the construct is never validated against anything external, including a criterion the
authors hold; the one demonstrated consequence of a flag is a change in time-to-collision between
about 22 and 12 seconds while genuine emergencies become rarer; and the comparison that gives the
paper its most quotable sentence rests on a human arm that the reference itself fails to calibrate
and that drove the scenarios through a head-mounted display. A specialist transportation or
human-factors venue would evaluate this work on its considerable merits without needing the
generality claim to carry weight it cannot yet bear.

I would change this recommendation if the authors returned with (i) the preference-rating analysis
linking atypicality to human dispreference, (ii) an apparatus-matched human arm, and (iii) an outcome
measure that is behaviourally interpretable inside four seconds. With those three, this would be a
strong candidate here.
