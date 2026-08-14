# Reviewer D — Round 4 review (journal-side editor-generalist)

Manuscript: "Online monitoring of socially compliant autonomous driving" (36 pp., 58 refs, 6 figures incl. 2 Extended Data)

---

## 1. Summary assessment

The manuscript recasts social compliance for autonomous vehicles as online monitoring of conditional social atypicality: an interaction preference value (IPV) estimated per frame by likelihood-based inverse planning, gated by an explicit readability boundary, judged against a situation-conditioned human reference range with split-conformal calibration, and audited on a matched-scenario real-vehicle benchmark that includes a human reference arm driving the same scenarios. The reframing is genuinely novel for a broad machine-intelligence audience; the abstention discipline, frozen gates, reason codes and two-population audit are exemplary practice; and the in-domain calibration evidence is solid. The deployment-facing claims are weaker than the prose: the headline 2.1× machine-vs-human flag comparison and "no alarm inflation" carry no uncertainty and rest on asymmetric denominators; the consequence signature's inferential support is thin (200-draw placebo p ≈ 0.02, a case-level permutation null, one window variant crossing zero) with an unaddressed shared-source circularity on the ego side; the central "situation suffices" claim is a null result without equivalence bounds; no baseline monitor is compared; and the human-participant experiment has no ethics or consent statement, with empty Acknowledgements and Author Contributions — a standards-desk stop. All remediable. Major revision.

---

## 2. Major weaknesses

**M1. No ethics oversight, consent, or safety reporting for the human-participant experiment. [Methods 4.6; Results 2.5; back matter p. 36]**
Twenty licensed drivers drove a real vehicle through 15 deliberately staged conflict scenarios against scripted counterpart injections. The only nod to participant protection is "Driver identities are pseudonymised before analysis." There is no ethics-committee/IRB approval statement, no informed-consent statement, no compensation or safety-protocol description, and nothing on the authorisation to analyse the 19 competition teams' runs as research data. Nature Portfolio policy makes an ethics statement for research involving human participants a hard requirement; as submitted this manuscript would not pass the standards desk regardless of scientific merit. Remedy: add a full ethics statement (approving body, protocol number, consent procedure, safety measures for staged conflicts) and state the terms under which benchmark runs and team data may be analysed and published.

**M2. Reporting completeness and undisclosed proximity to the benchmark operator. [p. 36; ref. [53]; Data Availability]**
Acknowledgements and Author Contributions are section headers with no content. The paper is single-authored, yet describes a 20-driver real-vehicle campaign and a national benchmark deployment — an editor needs to know who designed, ran and funded those operations. More pointedly: the benchmark is operated by the "Tongji University TOPS Group" (ref. [53]) and the author is at Tongji's College of Transportation Engineering; the Data Availability text refers only to "its organisers" without disclosing this relationship, while the Competing Interests section declares none. That proximity is not disqualifying — it may even explain the enviable access — but it must be disclosed and its role in scenario selection, data access and outcome labelling described. Remedy: complete both sections, disclose the institutional relationship, and state funding.

**M3. The headline arm comparison (9.8% vs 4.7%, "2.1×", "no alarm inflation") has no uncertainty quantification and rests on non-comparable denominators. [Results 2.5; Fig. 6a,c; Methods 4.4, 4.6]**
Every figure-level uncertainty is conscientious elsewhere, yet Fig. 6a and 6c are annotated "uncertainty not shown" and the text supplies no interval for 4.7% vs 9.72%, for the 2.1× ratio, or for "15 of 15 scenarios" (scenario-paired, but no formal inference; runs share systems and scenarios, so independence cannot be assumed). Second, the denominators differ in selection intensity: on the natural held-out fold the support gate removes ~5% of readable moments, whereas on the benchmark only 20.8% of candidate moments are judgeable out of 55.3% readable — i.e. roughly 62% of readable benchmark moments are discarded for lack of human support. The judgeable-moment fraction for the human arm is never reported at all (only 15,102 judgeable moments appear). Comparing flag rates across arms whose accepted subsets are selected this differently is not self-evidently meaningful. Third, the audit is structurally one-sided: a transported reference that is too wide (under-flagging) produces exactly the observed "no inflation" signature — the humans' 4.7% being half the native 9.72% is as consistent with a conservatively miscalibrated transported reference as with well-behaved drivers, and the design cannot distinguish these. Fourth, the 9.8% pools 19 heterogeneous systems; if a few immature competition entries drive most flags, the population claim "automated systems are flagged about twice as often" mischaracterises the population. Remedy: case/scenario-clustered CIs for all arm-level rates and ratios; report candidate→readable→judgeable funnels for all three populations side by side; an anonymised per-system flag-rate distribution; and an explicit statement of the one-sidedness of the audit (it can detect alarm inflation, not reference dilation).

**M4. The consequence-signature evidence is thinner than the main-text claims, and a mechanical-circularity explanation is not excluded. [Results 2.4; Fig. 5; Methods 4.5]**
The main text asserts flagged-assertive moments are followed by "measurably tighter interaction on both sides" and that "the association survives a placebo test." Methods reveals a more fragile picture: the exposure placebo yields empirical p = 0.0199 from only 200 draws (resolution ~0.005); a case-level label permutation on the same battery "does not reach significance (p = 0.1493)"; and the open-ended contract-window interval crosses zero at 90% ([−2.6100, +0.1372]), leaving one pre-specified window carrying the result. None of this nuance reaches the main text. Separately, the ego-side outcome (minimum TTC after the verdict) is computed from the same kinematic stream that produced the assertive reading: an assertive IPV is inferred from trajectories that close on the counterpart, and closing trajectories mechanically presage tighter margins over the following seconds. The counterpart-side reading from the other vehicle's own control record mitigates this for one side only, and the paper's "convergence rather than a restatement" is asserted, not tested. Remedy: raise placebo draws to ≥1,000; reconcile the main text with the case-permutation null; and add a matched-kinematics control (e.g. within-range moments matched on pre-flag closing rate/relative speed) showing the compression survives conditioning on the kinematics that triggered the flag.

**M5. The counterpart's control policy on the benchmark is never described, yet the headline consequence quantities are properties of it. [Results 2.4; Methods 4.5, 4.6]**
"Counterpart speed reduction 2.06×" and "absorbs roughly twice the routine speed reduction" are read from the counterpart's logged control record in the 175 non-scripted runs, where the counterpart "can respond to the ego." Responding under what controller — a rule-based agent, a learned policy, a human teleoperator? The reactivity, gains and braking logic of that policy determine every counterpart-side number and the cross-arm claim that "the flag means the same thing for either kind of driver." Without this description the ecological validity of the consequence signature cannot be assessed, nor whether the same policy served both arms. Remedy: specify the counterpart policy (and its identity across arms) in Methods, with enough detail to judge how human-like its responses are.

**M6. The paper's most distinctive scientific claim — partners' preferences co-vary, yet the situation suffices online — is a null result reported without the statistics needed to accept a null. [Results 2.2, 2.3; Fig. 2/3; Methods 4.3]**
The event-level correlation itself is never quantified: no coefficient, no CI, no method, anywhere in the manuscript — the tension that headlines the Discussion rests on an unreported number. Its resolution rests on "adds no measurable value": a paired 90% interval-score difference of −0.0002 with case-clustered p = 0.86. Absence of significance is not evidence of absence at NMI level: no equivalence test, no minimum detectable effect, no power statement is provided, so the reader cannot tell whether the ablation could have detected a practically relevant sharpening had one existed. The same applies to the self-history ablation. Remedy: report the event-level correlation with CI; recast both ablations as equivalence analyses (e.g. TOST or CI-within-margin on the interval-score and width differences) with a pre-stated practical-equivalence margin, and state the MDE at 80% power.

**M7. No empirical baseline: the monitor is positioned against unusual-driving and surprise-based detectors but never compared with any of them. [Introduction p. 3; refs 45–47; Results 2.4–2.6]**
The introduction claims none of the adjacent methods "reads a social preference, refers it to a calibrated situation-conditioned human range, or separates the moments…". Conceptually fair — but the paper never shows the IPV-based flag finds moments that a simpler or existing detector (learned-abnormality [45], surprise/violated expectation [46, 47], or even a kinematics-only conditional range on, say, closing speed) would miss or mislabel. Section 2.6 argues only against emergency-threshold checks — the easiest foil. Since the consequence battery is already built, running one credible alternative flagger through the identical battery would establish that the *social* content of the IPV, not just conditioning on kinematic context, is doing the work. This is the difference between a compelling framework paper and a validated instrument. Remedy: add at least one alternative-monitor comparison under the same frozen protocol, or an ablation replacing the IPV with a kinematic deviation score.

**M8. "Narrow" and "transfers" overstate what the numbers show. [Abstract; Results 2.3; Fig. 4 caption; Methods 4.4]**
The abstract promises a "narrow, calibrated, auditable runtime monitor." The relative sharpening over a global range (21%/20%/8%) is real, but the absolute 90% conditioned width averages 1.87 rad — 79% of the 2.36-rad admissible span (5th–95th percentiles 1.35–2.28) — so the instrument flags only near-extreme preferences; "narrow" is not an accurate description and the honest relative-vs-absolute contrast appears only in a caption parenthesis. On transfer, the contributions list says the reference "transfers across country, apparatus and task without inflating its alarms," yet within-corpus leave-one-source-out coverage at the 90% level is 0.743 (Waymo held out) and 0.750 (Lyft), and the Argoverse-2 fold achieves 0.900 only with 44.3% abstention. If the reference misses nominal coverage by 15 points across data sources inside the same corpus, cross-country "transfer" supported solely by a one-sided flag-rate audit (M3) deserves far more guarded language. Remedy: state absolute widths in the main text next to the relative reductions; rewrite transfer claims to "did not inflate alarms in one cross-domain audit," and surface the leave-one-source-out numbers in Results rather than Methods.

---

## 3. Minor issues

1. **Text–figure contradiction** [2.2 vs Fig. 3c]: main text says held-out variance explained is "at or barely above zero in all four folds"; the figure shows −0.195 (AV2) and −0.276 (nuPlan). "At or below zero" is the correct direction; fix the sentence (the scientific point survives).
2. **Fig. 6b asterisk convention**: an asterisk marking intervals that *exclude no difference* inverts the universal convention (asterisk = significant); relabel before a reader mis-reads the panel.
3. **"Supported threshold" is circular as used** [Abstract; 2.4; 2.6]: a threshold is "supported" iff its interval excludes zero, so "no rise at any supported threshold" is close to true by construction; at < 3 s the interval admits up to +3.15 pp. State all four thresholds with intervals (as Fig. 5b already does) and drop the term.
4. **Permutation resolution**: 200 placebo draws is inconsistent with the 1,000–2,000 bootstrap resamples used elsewhere; report ≥1,000 draws.
5. **No sensitivity analysis for frozen constants**: σ = 0.1 m, near-uniform gate 0.20, k = 25 support neighbours, 95th-percentile support radius, ≥50 anchors/≥10 cases. The readability accounting (70.3% vs 55.3%) and the abstention rates all depend on these; one supplementary sweep would establish robustness.
6. **Human-arm reporting gaps** [2.5; Methods 4.6]: number of completed human runs (20 × 15 = 300?), assertive-vs-over-yielding split of the 713 human flags, and which Fig. 6b cells lack defined intervals ("where defined") are all unreported.
7. **Latency**: "online" is the selling word, yet no per-frame compute cost is reported even though real-time operation is explicitly out of scope; a rough figure would cost one sentence.
8. **Data/Code availability is thin for Nature standards**: "will be made available upon publication" with no repository, DOI, licence, or named access process for the restricted benchmark data; reviewers should be able to access code and derived data during review.
9. **Terminology load**: anchor/readable/accepted/candidate/judgeable moments are defined mid-4.4 after several have been used; a one-line glossary (or Extended Data table) would help the broad readership.
10. **Fig. 1c legibility**: the grey "no reading" gaps and the top "reading issued" strip are hard to reconcile at displayed size; consider annotating the abstention reasons on the gaps.

---

## 4. Questions to the authors

1. Which ethics body approved the 20-driver staged-conflict study, and what consent, compensation and safety protocols applied? Under what agreement are the 19 competition systems' runs analysed and published?
2. What controls the counterpart vehicle in the 175 non-scripted runs, and is it identical in the human arm? How human-like is its response law?
3. What is the anonymised per-system distribution of flag rates behind the pooled 9.8%? Do a minority of systems account for the majority of flags?
4. Provide clustered CIs for 4.7% vs 9.72%, for 9.8% vs 4.7% (the 2.1×), and a formal statement for "15 of 15 scenarios." What is the human arm's candidate→judgeable funnel?
5. What sharpening (rad of width, or interval-score units) could the counterpart-IPV ablation have detected with 80% power? Same for self-history.
6. What is the event-level partner correlation (estimate, CI, estimator)?
7. Can the ego-margin compression survive matching flagged and within-range moments on pre-flag closing kinematics (relative speed, range rate)? If not, what remains of the "both sides" claim beyond the counterpart channel?
8. Why does the readable fraction differ between corpus and benchmark (70.3% vs 55.3%)? "Reported without attributing a cause" is honest but a deployment instrument needs at least a diagnostic hypothesis.
9. How sensitive are the readability and abstention accountings to σ = 0.1 m and the 0.20 near-uniform gate?
10. At the 90% level the natural outside rate is 9.72% rather than 10.0%; confirm that all cross-arm comparisons use the achieved (not nominal) rate as the anchor, and that the +0.28 pp over-coverage does not differentially affect the benchmark's support-gated subset.

---

## 5. Prioritised revision requests

1. **(Gatekeeping)** Add the ethics/consent/safety statement for the human-driver study and the authorisation basis for benchmark-team data; complete Acknowledgements, Author Contributions and funding; disclose the author–benchmark-operator institutional relationship (M1, M2).
2. **(Headline claims)** Add clustered uncertainty to every arm-level comparison in 2.5/Fig. 6; report the three populations' selection funnels side by side; add the anonymised per-system flag distribution; state explicitly that the audit detects inflation, not dilation, of the transported reference (M3).
3. **(Core claim rigor)** Quantify the event-level correlation; convert both null ablations into equivalence tests with pre-stated margins and MDEs (M6).
4. **(Consequence evidence)** Describe the counterpart policy; add the matched-kinematics control for ego-margin compression; ≥1,000 placebo draws; reconcile main text with the case-permutation null and the crossing window variant (M4, M5).
5. **(Positioning)** Run one alternative flagger (surprise-based or kinematic-deviation) through the identical frozen consequence battery (M7).
6. **(Language–evidence alignment)** Replace "narrow" with the absolute widths; move leave-one-source-out coverage into Results; retune the "transfers across country, apparatus and task" sentence (M8). Fix the Fig. 3c text mismatch and Fig. 6b asterisk convention.
7. **(Robustness)** Sensitivity sweep over frozen constants; report per-frame compute cost.
8. **(Availability)** Concrete repositories, DOIs, licences and an access route for restricted data, available to referees.

---

## 6. Acceptance probability

- **(a) As submitted:** 8% — the missing ethics statement and empty declarations alone would stop it at the standards desk; beyond that, the headline arm comparison and consequence claims need uncertainty and controls no careful referee would waive.
- **(b) After a competent major revision** (all of requests 1–6, with requests 4–5 delivering usable results): 55%. The conceptual contribution, claim discipline and audit design are genuinely strong and well matched to NMI; the residual risk is that the matched-kinematics control or the baseline comparison dissolves the consequence signature, which would force a further narrowing of the paper to its (still publishable, but less compelling) calibration core.

---

## 7. Recommendation

**Major revision.**
