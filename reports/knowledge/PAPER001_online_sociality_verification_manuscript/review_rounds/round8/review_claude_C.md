# Referee C — generalist editor-referee

**Manuscript:** "Online monitoring of socially compliant autonomous driving"
**Remit:** breadth and venue fit, claim support, declarations and reporting standards, display items.

---

## 1. Summary assessment

The paper recasts social compliance as an online membership test: is the vehicle's current
interaction preference inside a human reference range conditioned on the observable situation, and
if the moment is not readable or not supported, abstain. It delivers a per-moment estimator with an
explicit abstention taxonomy, a conditional-quantile reference calibrated on ~2.4 M human anchor
rows from four naturalistic corpora, and two validation arms — a matched-scenario benchmark of 19
automated systems and a 20-driver human comparator.

The framing is genuinely novel and well suited to this journal, and the authors' evidentiary
discipline is unusually good: both sides of the range are reported, non-detections are labelled as
such, exclusions are accounted for. The problems are visible precisely because the authors printed
the numbers that reveal them. Three are decisive. The instrument's frame-to-frame noise (0.30 rad)
is about twice the median width of the region in which an assertive flag is possible (0.148 rad),
and no persistence requirement is used anywhere. The headline "automated systems flagged about
twice as often" is generated entirely by the human arm being flagged at half the reference
population's rate, under a different apparatus. And the paper's own leave-one-source-out test shows
the reference does not transfer.

---

## 2. Major weaknesses

### M1. The per-moment verdict is noisier than the decision it is asked to make, and the paper never tests persistence

**[Location: Fig. 2b; Methods 4.1; Methods 4.5; Fig. ED1; Fig. ED2; all flag rates in Sections 2.4–2.5]**

Four numbers the paper prints, taken together, undermine every per-moment result:

- Methods 4.1: the reading is a weighted mean over seven candidates spanning
  [−3π/8, 3π/8], so it is hard-bounded at ±1.1781 rad.
- Methods 4.5: "at the 90% level the lower edge is negative in essentially every situation
  (median −1.03 rad)". The median assertive-flag zone is therefore
  1.1781 − 1.03 = **0.148 rad wide**, i.e. 0.148/2.3562 = **6.3% of the admissible span**.
- Fig. 2b: the mean frame-to-frame movement of a readable reading is **0.30 rad** (ego) and 0.31
  (counterpart), at 10 Hz.
- 0.30/0.148 = **2.0**. The mean single-frame movement of the instrument is about twice the median
  width of the entire region in which an assertive flag can occur.

Figures ED1 and ED2 show what this looks like. In ED1 the trace swings the full ±1.18 rad within
two or three frames, repeatedly, and sits pinned at the −1.18 floor for long runs. In ED2 — the
paper's hand-picked showcase — **every** flagged assertive frame plotted in panel c sits at
approximately −1.18, i.e. at the estimator's floor, and two of them are isolated single-frame
excursions from readings near −0.3. ED2's own selection funnel says only **20 of 120** runs
containing an assertive flag (16.7%) contain "at least five flagged frames in one contiguous
stretch": in 100 of 120 runs, assertive flags do not persist for half a second.

Methods 4.4 concedes that the persistence layer exists and is unused: "A sequential warning layer
(persistence over consecutive verdicts) belongs to the deployment interface: no result in this
paper uses it, every reported verdict and flag rate is per-moment". That is exactly the problem.
The 9.8% and 5.0% flag rates, the 2.0× ratio, the 15-of-15 scenario sweep and the entire
consequence analysis are computed on a classifier whose measurement noise exceeds its decision
region, whose flags are concentrated at a hard estimator boundary, and whose flags mostly do not
persist for 0.5 s.

**Why it matters here:** the paper's central promise is an *auditable runtime monitor*. A monitor
whose alarms are dominated by grid saturation and single-frame jitter is not auditable in any
operational sense, and the reader cannot tell from the manuscript how much of the reported signal
is behaviour and how much is estimator boundary behaviour.

**Remedy:** report (i) the fraction of readings within, say, 0.01 rad of ±3π/8, overall and among
flagged moments, in both arms; (ii) the distribution of contiguous flag-run lengths; (iii) all
headline rates (9.8%, 5.0%, 2.0×, 15/15, and the Fig. 5 and Fig. 6b batteries) recomputed under at
least one pre-specified persistence rule; (iv) a grid-width sensitivity analysis (e.g. 9 or 11
candidates spanning the full declared domain [−π/2, π/2]) showing the flag rates are not an
artefact of the candidate span. If the results do not survive (iii), the paper's claims must be
rewritten around what does.

---

### M2. The headline cross-population comparison is the one reading the paper's own panel does not support

**[Location: Abstract; Section 2.5; Fig. 6a]**

Fig. 6a contains three rates per level, and I read them as: natural driving (held-out) 20.0 / 9.7 /
4.4; matched human drivers 10.0 / 5.0 / 2.3; automated systems 19.5 / 9.8 / 5.0. These cross-check
exactly against Fig. 4b's coverage deviations (+0.03, +0.28, +0.57 pp → outside rates 19.97, 9.72,
4.43) and against the side counts printed under the axis (2,748/14,099 = 19.49%;
1,388/14,099 = 9.84%; 701/14,099 = 4.97%; 1,555/15,598 = 9.97%; 786/15,598 = 5.04%;
356/15,598 = 2.28%). So:

| level | AV ÷ reference population | matched humans ÷ reference population |
|---|---|---|
| 80% | 19.49/19.97 = **0.98** | 9.97/19.97 = **0.50** |
| 90% | 9.84/9.72 = **1.01** | 5.04/9.72 = **0.52** |
| 95% | 4.97/4.43 = **1.12** | 2.28/4.43 = **0.52** |

The automated systems are flagged at essentially the rate the reference predicts for natural human
driving, at all three levels. The matched human drivers are the anomaly: flagged at almost exactly
half the reference population's rate at every level, and the paper's own interval (Fig. 6a, "90%
human: 95% CI 3.4–5.5%") excludes the 9.72% native rate. The entire "2.0×" is produced by the
human arm's halving.

Section 2.5 states this correctly — "all three rates belong together, and no pair of them tells the
story alone" — and then the Abstract tells the story with exactly one pair: "automated systems
flagged about twice as often". Fig. 6a reinforces it by printing **"AV = 2.0× humans"** in bold as
the panel's headline while leaving the AV-vs-reference comparison (ratio ≈ 1.0) unlabelled. Two
further problems: the 2.0× carries **no interval anywhere** (only the human bar has a whisker), and
Section 2.5's framing of the halving as validation — "no more often than that native rate: carrying
the reference across country, apparatus and task does not inflate its alarms" — treats a
statistically clear two-fold *deflation* as a success. It is equally evidence that the human arm's
behaviour, or the monitor's reading of it, differs systematically from natural driving; and if the
transfer deflates flags for humans, there is no reason to assume it does not also deflate them for
the machines.

The arms are also not matched in apparatus. Methods 4.5: "the simulated counterparts reach an
automated system through its perception interface and a human driver through a head-mounted
display." Methods 4.6 adds that drivers were supervised, insured, instructed to drive naturally,
and ran "the scenario set as a single fixed sequence, identical across drivers" — i.e. no
counterbalancing, so practice and habituation are confounded with scenario identity in the human
arm and absent in the machine arm. Every one of these is a plausible route to more stereotypical,
less-flagged human behaviour.

**Why it matters here:** "AVs are twice as socially atypical as humans" is the sentence a broad
readership will take from this paper, and it is the claim the paper's own panel supports least.

**Remedy:** demote the 2.0× from the Abstract, or state all three rates there. Give the ratio a
confidence interval. Make the AV-vs-reference-population parity the primary calibration statement
in Fig. 6a's headline. Report participant characteristics and address the apparatus, instruction,
supervision and order-effect asymmetries explicitly as limits on the comparison.

---

### M3. The paper's own leave-one-source-out test shows the reference does not transfer, and the deployment depends on transfer

**[Location: Methods 4.4; Fig. 3c; Fig. 4 caption; Sections 2.4–2.5; Abstract; Discussion]**

Methods 4.4 reports leave-one-source-out coverage at the 90% level: "0.743 (Waymo,
143,380/193,096), 0.990 (nuPlan, 149,068/150,587), 0.750 (Lyft, 68,270/91,069) and 0.900
(Argoverse-2, 11,315/12,576, with 44.3% held-out abstention)". I verified each ratio. Converting to
outside rates against a **nominal 10.0%**:

- Waymo held out: 25.7% (2.57× nominal)
- Lyft held out: 25.0% (2.50× nominal)
- Argoverse-2 held out: 10.0% (1.00×, but only after abstaining on 44.3% of its moments)
- nuPlan held out: 1.0% (0.10× nominal)

The alarm rate on an unseen source ranges over a **26-fold** span, in both directions. Fig. 3c says
the same thing in a different currency: out-of-source R² of +0.026, +0.017, −0.195, −0.276 — worse
than predicting the held-out mean in two of four folds.

The benchmark is a fifth source, and by any measure a more distant one than any held-out member of
the four (different country, closed course, staged conflicts, mixed reality, a different vehicle
fleet). Fig. 4's caption is honest about this — "Transfer to a data source not seen during fitting
is not established and is reported as a boundary of the present monitor" — and Sections 2.4–2.6
then proceed as if it were established, on the strength of a single successful transfer to a
20-driver arm. Against a 26-fold source-driven spread, a 1.95-fold human/machine contrast carries
very little information.

The Abstract calls the monitor "calibrated" and reports "no alarm inflation". The Discussion offers
the framework as "a template for monitoring whether autonomous agents behave within human normative
ranges online, under uncertainty and distribution shift". Distribution shift is precisely the
condition under which the paper's own experiment shows the calibration failing.

**Remedy:** state the LOSO outside-rate range in the Results, not only as a coverage triple buried
in Methods 4.4. Qualify "calibrated" as in-distribution and marginal wherever it appears, including
the Abstract. Remove or heavily qualify the "under distribution shift" clause in the Discussion.
Present the human-arm result as one transfer observation consistent with a wide prior, not as
establishing transfer.

---

### M4. The counterpart is traffic-simulation software, and neither the main text nor Fig. 5 says so

**[Location: Abstract; Section 2.4; Fig. 5 and its caption; disclosed only in Methods 4.5]**

The Abstract says "a matched-scenario real-vehicle benchmark". Section 2.4 says "The counterpart,
read independently from the other vehicle's own control record, carries nearly twice the routine
speed variation", and draws the key inference: "The two sides are measured from different sources
and different quantities, so their agreement is a convergence rather than a restatement (Fig. 5)".
Fig. 5's caption says only "Speed quantities are read from the other vehicle's own logged control
record". A reader of the Abstract, Section 2.4 and Fig. 5 will conclude that two vehicles were
observed.

Only Methods 4.5 discloses the truth: "The counterpart vehicles are driven by traffic-simulation
software (TESS NG) and respond to what the ego vehicle does" and "The staging is mixed-reality: the
ego is a real vehicle driven at a test site". The counterpart-side endpoints are therefore
properties of a car-following/gap-acceptance model's reaction function. If the ego progresses more
assertively, a reactive simulator yields more — largely by construction. The word "convergence" is
not earned: both sides are downstream of the same ego behaviour, one through the ego's own
trajectory and one through a deterministic reaction to it. The authors' rebuttal ("in the four
scenarios whose counterpart identity recurs across systems, the same counterpart's trajectory
differs between systems by 4.35–9.03 m") establishes that the simulator reacts; it does not
establish independence.

**Why it matters here:** the "tighter interaction for **both** vehicles" claim is the paper's
strongest-sounding empirical result and appears in the Abstract. A broad readership must be told,
in the Abstract and in the figure, that one of those two vehicles is software.

**Remedy:** state in the Abstract, in Section 2.4 and in Fig. 5's caption that the ego is a real
vehicle and the counterpart is simulator-driven and reaches each arm through a different interface.
Replace "convergence" with a statement of what the counterpart channel actually shows.

---

### M5. The ego-side consequence is close to a restatement of the exposure, and the effect lives where nothing is at stake

**[Location: Section 2.4; Section 2.6; Fig. 5a,b]**

The exposure is "the ego weights the counterpart's cost less than humans do in the same situation".
The ego-side outcome is the ego's own minimum time-to-collision over the next three seconds. These
are the same behaviour one second apart. Methods 4.5 establishes temporal separation ("the
trajectory samples that produce a verdict … precede every outcome sample"), which is necessary but
not sufficient: driving is continuous, so an assertive preference at *t* mechanically implies
assertive motion over *t*..*t*+3 s. The defence the paper offers — "measured from different sources
and different quantities" — applies to the counterpart channel, not this one.

Where the effect sits is also a problem. Fig. 5a's x-axis runs past 10² s. The caption explains
that non-closing frames are excluded, "so very large values mark a window whose only closing frames
close very slowly". The compression is confined to the upper quartile of that distribution — a
regime in which nothing is happening — while the lower quartile, the only safety-relevant end,
moves the *other* way (3.46 s for flagged assertive vs 2.99 s within range). The paper prints the
lower quartile in seconds but the upper quartile only as a percentage (−39.9%), so the reader
cannot see what absolute value is being compressed by two-fifths.

And Fig. 5b shows the flagged moments are *safer*: at every threshold whose interval excludes zero,
assertive-side moments are "0.46 to 0.61 times" as likely to fall below the emergency margin. On
the paper's own evidence, an assertive-side flag predicts fewer sub-1-second margins and a wider
worst-case margin. Section 2.6 turns this into an argument for the monitor's distinctiveness — "a
monitor built on those thresholds would report nothing" — but the reader is entitled to the
converse question, which the paper never asks: what should a developer *do* with a flag whose
immediate correlate is a safer interaction? Methods' planner interface nonetheless proposes "a
fallback candidate under sustained competitive deviation at high physical risk". Nothing in the
paper motivates that action.

**Remedy:** print the absolute quartile values in Fig. 5a. Add an analysis that separates the flag's
information from the ego's contemporaneous kinematics (e.g. contrast against a matched non-flagged
group with the same instantaneous speed/closing rate, or condition on the ego's own realised
progress). State plainly in the Discussion that assertive-side flags are associated with fewer
emergency-threshold events, and say what the monitor is therefore *for*.

---

### M6. A mechanism is asserted in Results and Discussion that Methods explicitly disclaims, and its supporting statistic is never reported

**[Location: Section 2.2; Section 2.3; Discussion; Methods 4.3]**

Methods 4.3: "Two further inputs are evaluated as ablations and add no measurable value … **Neither
is interpreted as a mechanism**; the counterpart channel in particular is statistically
indistinguishable from removing the IPV input given the situation (paired 90% interval-score
difference −0.0002, case-clustered p = 0.86)."

Section 2.3: "The apparent paradox … **resolves because the partner correlation is already carried
by the shared, observable situation** (role, geometry, relative kinematics, risk)."

Discussion: "**The relational structure that is visible across interactions is already carried by
the observable situation**, so a runtime social monitor does not need to read the other agent's
hidden intention."

This is a direct contradiction. The Results and Discussion state as fact the mechanism the Methods
say is not being claimed, and the paper calls it "the most informative finding".

Worse, the evidence for the mechanism is a different claim from the null that was tested. A null
improvement in one interval score, from one gradient-boosted quantile model with frozen
hyperparameters (learning rate 0.06, 72 boosting iterations, 31 leaves), does not show that the
partner correlation is *carried by* the situation. The direct test — the residual ego–counterpart
IPV correlation after conditioning on *z_t* — is never reported.

Nor is the correlation itself. Section 2.2 states, in a single clause with no number, no interval,
no *n* and no panel: "The two interacting agents' preferences, moreover, are correlated at this
event level". This is one of the two legs of the paper's headline tension, and Fig. 3 has no panel
for it.

**Remedy:** report the event-level partner correlation with its coefficient, interval, *n* and
clustering, and give it a panel. Report the partial correlation conditional on *z_t*. Either
demonstrate the mechanism or delete the causal wording from Section 2.3 and the Discussion so they
match Methods 4.3.

---

### M7. Display items: several panels do not support, or contradict, what is said about them

**[Location: Figs. 1, 2, 5, 6, ED2]**

Individually small; collectively they mean the reader cannot rely on the figures.

1. **Fig. 1b omits the abstention branch its own caption describes.** The diagram runs
   trajectory window → mechanism 1 (discriminative?) → no reading / mechanism 2 → inside range /
   outside range. Mechanism 2 has exactly two outputs. The caption says it returns "inside the
   range, outside it (atypical), **or an abstention**." The missing branch is the human-support
   gate, which on the benchmark discards 0.553 × 67,861 − 14,099 = 23,428 of 37,527 readable
   moments (**62.4%**) and silences 36 of 267 runs entirely. The concept figure hides the mechanism
   that does most of the work.
2. **Fig. 1c's grey intervals carry no reason codes.** Methods 4.4 defines six. The paper's claim
   that "Abstention here is a statement about the reference rather than about the vehicle, and it is
   auditable as such" (Section 2.4) is not demonstrated in the one worked example. In panel a the
   monitor's silence covers the vehicle's actual traversal of the conflict zone (22.0 s) — the
   moment a reader would most expect a verdict — with no explanation.
3. **Fig. 2a's caption contradicts itself in one sentence:** "The gain is present only for the real
   interaction: it disappears when the same interaction is misaligned in time (+0.006), **weakens**
   when the same moment is paired with a different partner (−0.043)". The panel shows the
   different-partner control at −0.043 with an interval clearly excluding zero — a third of the
   real-interaction gain, not an absence. Separately, the x-axis quantity ("Change in how sharply the
   reading is identified") is a dimensionless dispersion statistic whose scale is never given, so no
   reader can judge whether −0.132 is large; and the four rows have different *n* (4,743 / 4,605 /
   4,701 / 8,130), suggesting different case sets, which is not explained.
4. **Fig. 2b prints "uncertainty not shown" inside the panel** while the caption asserts a
   quantitative contrast (0.30/0.31 vs 0.17). **Fig. 6c does the same** for the 15-of-15 paired
   result, which is one of three findings in Section 2.5.
5. **Fig. 2c's in-panel annotation is misleading.** "first two rules 0.26 rad apart" is bracketed
   directly against two medians that are 0.08 rad apart (+0.08 and +0.00). The 0.26 rad is a mean
   per-episode absolute disagreement, per the caption; the annotation reads as a distance between
   the plotted medians.
6. **Fig. 5's caption is truncated mid-sentence** — it ends "…; panels" and stops. The *n* values
   for panels c and d are never given in the caption (they appear only as in-panel text), and Fig. 5
   is the only display item in the paper without a Source Data statement.
7. **Fig. 6b prints significance asterisks on rows that have no intervals drawn.** Whiskers appear
   only on the two counterpart-speed rows; the other four rows (ego margin median, ego margin upper
   quartile, ego emergency <2 s, counterpart braking) carry asterisks with no interval, while the
   caption says "whiskers are case-bootstrap 95% intervals **where defined**". An asterisk defined as
   "a ratio whose interval excludes parity" cannot be printed for a ratio whose interval is not
   defined. Asterisk placement also makes ownership ambiguous: on the upper-quartile row the single
   asterisk is blue but sits nearer the grey marker. On my reading of the colours, the human
   upper-quartile ratio is **not** starred — yet Section 2.5 asserts "the ego margin's upper quartile
   contracts by about three-tenths" for the human arm as an established part of the shared signature.
8. **Fig. 6b uses a different resampling unit from the rest of the benchmark.** Its caption says
   "case-bootstrap 95% intervals"; Methods 4.5 says "inference resamples scenario runs (**the unit of
   clustering throughout**)". This may explain why the AV ego-emergency (<2 s) contrast excludes zero
   in Fig. 5b (−6.73 pp [−11.67, −1.76]) but appears unstarred in Fig. 6b — the same data, the same
   endpoint, two verdicts.
9. **Fig. 6's funnel prints "count unavailable"** for the automated arm's gate-1 count, which is
   0.553 × 67,861 = 37,527. Printing an unavailable count in an audit figure, next to the fraction
   and the denominator that determine it, is a poor advertisement for the paper's auditability
   claim.
10. **Fig. ED2 panel b's title asserts what the panel contradicts.** "Counterpart speed falls across
    the flagged stretch" — but the plotted 2.1 s centred median has already fallen from ~4.9 to
    ~1.4 m/s by the time the first flag fires at ~15.4 s, is flat across the flagged stretch, and
    rises afterwards. The caption's disclaimer ("The display is descriptive and assigns no
    causation") does not undo a panel title that states the causal reading.
11. **Fig. 6c labels three points (A5, B1, A3) with no key** for what those scenarios are.

**Remedy:** all of the above are fixable, but every one needs fixing. In particular: add the support
branch and reason codes to Fig. 1; correct Fig. 2a's caption; draw the intervals in Fig. 6b or
remove the asterisks; reconcile the bootstrap unit; complete Fig. 5's caption; retitle ED2b.

---

### M8. Reporting standards fall short of what this journal requires

**[Location: Methods 4.6; Data Availability; Code Availability; whole manuscript]**

- **No Supplementary Information and no Reporting Summary** anywhere in the submission.
- **No participant characterisation.** The human arm is described only as "Licensed drivers", "20
  drivers", "held a valid driving licence". No age, no sex or gender (SAGER reporting is expected),
  no driving experience, no recruitment route, no exclusions. A paper whose central audit rests on
  this population must characterise it; without this, "matched human drivers" is unverifiable and M2
  cannot be adjudicated.
- **Deliberate non-disclosure with no debriefing statement.** Methods 4.6: "they were not told that
  their runs would serve as a human reference against automated systems." Whether the ethics
  approval covered this withholding, and whether participants were debriefed and re-consented
  afterwards, is not stated.
- **The human arm's data are not released at all** ("Records from the human reference arm are not
  released … the human arm is reported only in aggregate"). The consent limitation is legitimate,
  but the consequence is that the paper's flagship audit — the comparison that produces the 2.0× —
  cannot be reproduced or re-analysed by anyone. A de-identified moment-level verdict and endpoint
  series, with no trajectories, would likely fall within a reasonable reading of the consent and
  should be sought.
- **Code is "available to referees on request during peer review"** rather than provided. For a
  submission whose entire contribution is an instrument, referees should have it.
- **The instrument is not fully specified in the paper.** The estimator's objective is given only as
  "cos θk · (own-progress cost) + sin θk · (interaction cost)", with "the full cost terms and solver
  configuration … released with the code". The readability thresholds *q0* and *c0* of Eq. (2) are
  never given a numerical value anywhere. The interaction-opportunity rule is qualitative ("a mapped
  conflict point reachable by both agents within a bounded horizon with closing geometry") with no
  horizon. Methods must be self-contained.
- **Two announced analyses are never reported.** Methods 4.3 says "We compare a global reference, an
  offline oracle-risk ceiling, and the context-conditioned reference" and then concludes the
  conditioned reference "supplies essentially all of the achievable sharpening" — but the oracle
  ceiling's value never appears. Methods 4.5 says 27 runs "are held out of the pooled
  counterpart-response analyses and **reported separately**" — they are not reported anywhere.
- **No multiplicity control.** Fig. 5 alone reports roughly thirty intervals (8 in panel b, 10 in
  panel c, 6 in panel d, plus the ego quartile battery and the side-vs-side comparisons). The
  counterpart-side "convergence" rests on two supported contrasts out of ten, one of which has a
  lower bound of **+0.08 km h⁻¹** — 0.02 m s⁻¹ accumulated over three seconds, a mean additional
  deceleration of 0.007 m s⁻². No adjustment is discussed.
- **No public analysis plan.** The paper leans heavily on pre-specification ("frozen", "fixed in
  advance", "pre-specified primary", "specified and frozen before any outcome was examined"). None
  of it is verifiable without a timestamped public deposit.
- **Competing interests, entrant data.** The disclosure that the benchmark is operated by the
  authors' group is welcome. But the organisers also decide which entrants enter the analysis: "one
  is set aside in full because its replay records are not clean", and "13 of the 18 [lost
  system-scenario cells] fall in a single system". Whether those exclusions were made blind to the
  verdict series is not stated, and no statement covers the terms under which third-party entrants'
  runs were used and characterised.

---

### M9. The motivating result and the deployed monitor condition on different things; and the episode summary that carries it is one the paper shows to be unstable

**[Location: Section 2.2; Fig. 3a; Eq. (3); Fig. 2c; Methods 4.2]**

Section 2.2 is the paper's argument that "the reference cannot be a scalar". Its evidence, Fig. 3a,
bins on realised post-encroachment time — and the caption concedes this is "an offline quantity used
here for description only; the online monitor never reads it (Eq. 3)". Eq. (3) instead uses
"online-computable risk proxies". The paper never shows that the sign reversal (+0.058 → −0.034)
survives when the offline PET bands are replaced by the online proxy the monitor actually uses. The
motivation and the mechanism are therefore not connected by any reported result.

Second, Fig. 3a is computed on an episode summary — "the mean IPV over its valid interaction
segment" (Section 2.2). Fig. 2c and Methods 4.2 show that the three candidate summary rules "differ
by 0.26 rad on average and flip the episode sign in 7–22% of episodes", with medians of +0.08, +0.00
and +0.19. The reported role effect (0.058 rad; total swing across bands 0.092 rad) is smaller than
the disagreement between the rules the paper itself documents. Group means do attenuate per-episode
noise, so this is not automatically fatal — but the paper reports robustness of the sign reversal to
*dropping the largest source* and to *alternative risk and geometry binnings*, and not to the one
analytical choice it has independently shown to be consequential.

**Remedy:** reproduce Fig. 3a using the online risk proxy of Eq. (3), and under all three summary
rules.

---

### M10. Venue fit: the generalisation beyond driving is one decorative sentence, and it claims a property the paper disproves

**[Location: Discussion]**

The entire beyond-driving argument is: "Beyond driving, the framework offers a template for
monitoring whether autonomous agents behave within human normative ranges online, under uncertainty
and distribution shift, while being explicit about when a social judgement should be withheld [7]."
One sentence, one citation, that citation being the machine-behaviour agenda paper the Introduction
already leans on.

Nothing in the monitored construct obviously travels: the IPV is defined by a two-agent trajectory
optimisation over a mapped conflict point with lane-centreline geometry. The Discussion does not say
what the analogue of the readability gate, the situation vector or the support gate would be in any
second domain, nor what evidence would be needed. And, per M3, "under distribution shift" is the one
condition the paper's own leave-one-source-out experiment shows the monitor does not satisfy.

For a broad machine-intelligence readership this matters: the transferable idea here — *a calibrated
conditional normative range plus a principled reject option, with the reject option treated as a
first-class verdict rather than a failure* — is genuinely interesting and is the strongest reason to
publish this at this journal. It deserves a developed paragraph naming at least one concrete second
setting and what would have to be established there. As written it reads as an editorial gesture.

---

## 3. Minor issues

1. **Arithmetic query (readability fraction).** Methods 4.4's ledger gives 3,202,646 readable of
   4,497,368 anchor rows = **71.21%**. Methods 4.5 states "The fraction of moments whose reading
   carries discriminative information differs between the naturalistic corpora and the benchmark
   (**70.3%** vs 55.3%)". Methods 4.4's vocabulary defines a readable moment as one that "passes the
   discriminative-information rule", so these should be the same quantity. Which denominator gives
   70.3%?
2. **Arithmetic query (support pass rate).** Methods 4.5 gives the benchmark's overall human-support
   pass rate as **32.32%**. From Section 2.4, 0.553 × 67,861 = 37,527 readable and 14,099 judgeable,
   giving 14,099/37,527 = **37.57%**. Is 32.32% an unweighted mean across the 267 runs? Please state
   the denominator.
3. **The conformal layer does no work but is presented as a pillar.** Methods 4.4: "The fitted radii
   are near zero (cα = 1.4 × 10⁻³, 1.2 × 10⁻⁶ and 0.0 rad …), so the conformal step finds essentially
   nothing to repair". The Abstract's "calibrated", Fig. 4's calibration panel and eight references
   [51,52,54–58] therefore rest on a null operation; the good coverage is a property of the quantile
   regression. Say so plainly, and report the *global* model's conformal radius and the pre-conformal
   widths so readers can see how much of the 21%/20%/8% sharpening is conditioning and how much is
   the conformal correction applied to the baseline.
4. **The comparison baseline in Fig. 4a is degenerate.** The panel shows the global range at
   **100%** of the admissible span at both the 90% and the 95% level, though the caption singles out
   only the 95% level. Any conditioning improves on a baseline that spans the whole scale. A more
   informative comparator (e.g. the situation vector permuted within strata, or the announced
   oracle-risk ceiling) would show what the conditioning is actually worth.
5. **The "sharpened" range is still four-fifths of the scale.** The conditioned 90% range averages
   1.87 rad of 2.36 rad. The paper says this honestly; it should also say what it implies — the
   monitor is an extreme-value detector, not a graded measure of social behaviour.
6. **"at any supported threshold" is unparseable in the Abstract**, where the word is undefined. It
   is defined only in Fig. 5b's caption. Also, the double negative ("no rise") understates the actual
   result (a substantial fall) and thereby hides the interpretive problem raised in M5.
7. **"the population that defined it"** (Section 2.5 heading; Abstract "its defining population") is
   inaccurate. The reference was defined by fleet-recorded natural driving in the United States and
   Singapore; the audit arm is 20 drivers in another country on a closed course. They are the same
   *kind* of population, not the same one — which is exactly why the transfer question in M3 arises.
8. **Fig. 3c's title, "Why the preference must be read online", asserts a conclusion the panel does
   not reach.** Failure of a fitted model to transfer across corpora is a statement about
   generalisation, not about online-vs-offline reading.
9. **Figs. 3c and 4c are not reconciled.** One reports R² never above +0.03 and negative in two
   folds; the other reports the situation explaining 20.9%. The units and splits differ, but a reader
   will juxtapose them. One sentence would fix it.
10. **Section 2.6 is a ten-line section that restates Fig. 5b** and could fold into Section 2.4.
11. **Methods 4.5 reads as an audit log.** Roughly two and a half pages of run-level bookkeeping
    (267/231/204/175/227 accounting, naming conventions, sentinel checks) belongs in Supplementary,
    with the load-bearing conclusions retained.
12. **Four runs unaccounted.** "231 contain at least one judgeable moment"; "the ego-side panels use
    all 227 runs with an evaluated ego window". Where do the other four go?
13. **Fig. 5c and 5d appear to contradict each other to a casual reader** (counterpart speed
    reduction 2.06× but counterpart hard braking 3.40 pp *less* frequent). One caption sentence
    reconciling them — gentler, sustained deceleration rather than harder braking — would help.
14. **Absolute magnitudes are missing throughout the human arm.** Fig. 6b is entirely in ratio units,
    although Methods 4.5 itself warns that "a ratio becomes unstable where the within-range median is
    near zero, as it is in one of the two sites". Give the human arm's absolute medians and per-endpoint *n*.
15. **Reference [48] and [49] are dated 2026** and [53] is a website with a 2024 date; please confirm
    accessibility and give an access date for the challenge platform.

---

## 4. Questions to the authors

1. What fraction of readings lie at (within 0.01 rad of) the candidate-grid boundary ±3π/8, overall
   and among flagged moments, in each arm? In Fig. ED2c every flagged frame appears to sit at the
   floor.
2. What is the distribution of contiguous flagged-frame run lengths? Fig. ED2's funnel implies only
   20 of 120 runs with an assertive flag contain five consecutive flagged frames.
3. What are the flag rates (9.8%, 5.0%), the ratio (2.0×) and the Fig. 5/6b batteries under a
   pre-specified persistence rule of, say, 0.5 s?
4. What is the confidence interval on the 2.0× ratio, and on the automated arm's 9.8%?
5. Why is the matched human arm flagged at almost exactly half the reference population's rate at all
   three levels? Which of apparatus (head-mounted display vs perception interface), instruction,
   supervision, fixed scenario order, or driver population do you consider responsible, and what
   evidence bears on it?
6. What are the participants' age range, sex/gender distribution, and driving experience? Were they
   debriefed after the deliberate non-disclosure of purpose, and did the ethics approval cover it?
7. Does the sign reversal in Fig. 3a survive when the offline PET bands are replaced by the online
   risk proxy of Eq. (3)? And under all three episode summary rules?
8. What is the event-level ego–counterpart IPV correlation (coefficient, interval, *n*, clustering)?
   What is it after conditioning on *z_t*?
9. What is the offline oracle-risk ceiling's width at each level, announced in Methods 4.3 but never
   reported?
10. Where are the 27 naming-convention runs "reported separately"?
11. Which denominators give the 70.3% readability fraction (Methods 4.5) and the 32.32% support pass
    rate (Methods 4.5)? My computations from the printed counts give 71.21% and 37.57%.
12. Fig. 6b: what are the intervals for the four rows drawn without whiskers, and which marker owns
    each asterisk? Is the human-arm upper-quartile ratio supported? Why does the AV ego-emergency
    (<2 s) contrast exclude zero in Fig. 5b but appear unstarred in Fig. 6b — is the bootstrap unit
    different, and if so, how does that square with "the unit of clustering throughout"?
13. Was the exclusion of the one system "set aside in full because its replay records are not clean"
    made blind to its verdict series? Under what terms were third-party entrants' runs used?
14. What are *q0* and *c0*, and what is the interaction-opportunity horizon?
15. Can a de-identified moment-level verdict and endpoint series for the human arm be released?

---

## 5. Prioritised revision requests

1. **Re-run every per-moment result under a persistence rule, and report grid-saturation
   statistics.** (M1) This is the gate. If the headline rates do not survive, the paper's claims must
   be rebuilt around what does.
2. **Rewrite the cross-population claim.** (M2) Give all three rates in the Abstract or drop the 2×
   from it; attach an interval; make AV-vs-reference-population parity the primary statement in
   Fig. 6a; characterise the participants; address the apparatus and order-effect asymmetries as
   limits.
3. **Bring the transfer limitation into the Results and the Abstract.** (M3) Report the LOSO outside
   rates (25.7%, 1.0%, 25.0%, 10.0% against 10.0% nominal); qualify "calibrated"; remove "under
   distribution shift" from the Discussion.
4. **Disclose the mixed-reality staging and the simulator counterpart in the Abstract, Section 2.4
   and Fig. 5's caption, and drop "convergence".** (M4)
5. **Reconcile Results and Discussion with Methods 4.3 on the counterpart-IPV null, and report the
   event-level and conditional partner correlations with a panel.** (M6)
6. **Fix the display items,** item by item as listed in M7 — in particular Fig. 1b's missing
   abstention branch, Fig. 2a's self-contradicting caption, Fig. 6b's asterisks-without-intervals,
   Fig. 5's truncated caption, and ED2b's title.
7. **Meet reporting standards:** Reporting Summary; participant characterisation and debriefing;
   Supplementary Information; code to referees now; full estimator specification and all Eq. (2)
   thresholds in Methods; the oracle ceiling and the 27 held-out runs actually reported; a
   multiplicity statement; a public analysis-plan deposit; a de-identified human-arm release. (M8)
8. **Reproduce Fig. 3a with the online risk proxy and under all three summary rules.** (M9)
9. **Address what a flag is for,** given that flagged assertive moments are followed by fewer
   emergency-threshold events, and print the absolute quartile values in Fig. 5a. (M5)
10. **Develop the beyond-driving paragraph** into a real argument with a named second setting, or
    remove it. (M10)

---

## 6. Acceptance probability

- **(a) As submitted: 3%.** The truncated Fig. 5 caption, the missing Reporting Summary and
  participant details, the undisclosed simulator counterpart in the Abstract, and the Methods/Results
  contradiction over the counterpart-IPV mechanism would each on its own require a revision cycle.
- **(b) After a competent major revision: 38%.** Most of the report is addressable by better
  reporting and more careful claims, and the underlying idea is worth publishing here. The residual
  risk is concentrated in M1: if the flag rates and consequence signature do not survive a
  persistence requirement and a grid-saturation audit, the empirical spine of Sections 2.4–2.6 goes
  with them, and the paper becomes a methods contribution without a validated instrument.

---

## 7. Recommendation

**Major revision.**

The reframing is novel, the abstention machinery is a real contribution, and the authors' reporting
discipline is better than most submissions in this area — the criticisms above were possible only
because they printed the numbers. But three load-bearing claims currently outrun their evidence: the
per-moment verdict is noisier than the region it partitions, the two-fold human/machine contrast is
generated by an apparatus-confounded comparator and contradicted by the paper's own third rate, and
the reference's transfer is asserted where the paper's own experiment shows it failing in both
directions. I would want to see M1 answered before any decision is made on the rest.
