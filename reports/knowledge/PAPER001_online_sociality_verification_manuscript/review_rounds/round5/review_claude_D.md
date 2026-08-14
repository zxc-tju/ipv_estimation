# Reviewer D — Round 5 review (journal-side editor-generalist, NMI)

Manuscript: "Online monitoring of socially compliant autonomous driving" (single author; 36 pp incl. Methods, Extended Data, 58 refs)

---

## 1. Summary assessment

The manuscript recasts AV social compliance as online monitoring of conditional social atypicality: a frame-level interaction preference value (IPV) is estimated by likelihood-based inverse planning, gated by readability/abstention logic, and judged against a situation-conditioned human reference range calibrated by split conformal prediction. Evidence spans 38,228 naturalistic interactions from four corpora, a 267-run matched-scenario real-vehicle benchmark of 19 driving systems, and a 20-driver human arm audited under the identical frozen instrument. The conceptual contribution — clinical-style reference-range monitoring with explicit abstention, imported into machine behaviour — is genuinely attractive for NMI, and the freeze/audit discipline (frozen gates, outcome-blind crosswalk, honest bounding of the harm claim) is unusually good. But as submitted the paper would not pass the standards desk: human-participant real-vehicle experiments carry no ethics approval or consent statement, and Author Contributions and Acknowledgements are empty. Scientifically, the transfer claim is one-sided (course humans are flagged at half the native rate; 74–75% leave-one-source-out coverage is Methods-only), the headline 2.1× comparison lacks uncertainty and per-arm selection funnels, mechanism claims rest on a development split, the consequence inference is mixed across its own placebo tests, and abstract/title wording overruns the paper's own definitions. All addressable. Major revision.

---

## 2. Major weaknesses

**M1. No ethics approval, consent, or safety reporting for human-participant experiments.**
[Location: Methods 4.6; Section 2.5; back matter]
Licensed human drivers drove a real vehicle through deliberately staged conflict scenarios against scripted counterpart injections (20 drivers × 15 scenarios). This is human-subjects research with non-trivial physical risk, yet the manuscript contains no institutional ethics approval, no informed-consent statement, no compensation or recruitment description, and no safety protocol for the staged conflicts. Pseudonymisation ("Driver identities are pseudonymised before analysis") is noted but is not a substitute. The Acknowledgements and Author Contributions sections are section headers with no content, and there is no funding statement despite the benchmark being an NSFC-affiliated challenge. At any Nature-family journal this fails the reporting checklist before scientific review begins.
*Remedy:* Add ethics committee name and approval number, consent procedure, safety protocol summary, recruitment/compensation details; complete the Contributions, Acknowledgements and funding declarations.

**M2. The transfer/audit argument is one-sided, and the 2.1× headline is not statistically secured.**
[Location: Section 2.5; Fig. 6a,c; Methods 4.4 (leave-one-source-out); Abstract]
Three interlocking problems:
(a) *Direction of the calibration argument.* The claim "carrying the reference across country, apparatus and task does not inflate its alarms" reads 4.7% (course humans) vs 9.72% (native) as instrument survival. But a halving of the alarm rate is itself a large behavioural change of the instrument-plus-population system; it is equally consistent with the monitor becoming less sensitive on the course (wider effective ranges, differential support-gate selection) or with course humans driving conservatively under observation (demand characteristics — instructed drivers on a closed track). Only inflation is treated as disconfirming; deflation is absorbed as success. The AV-vs-human contrast (9.8% vs 4.7%) inherits this asymmetry.
(b) *The paper's own evidence says cross-source transfer is fragile.* Leave-one-source-out coverage at the 90% level is 0.743 (Waymo held out) and 0.750 (Lyft) — reported only in Methods 4.4. Fig. 3c likewise shows the fitted state structure does not transfer predictively (R² ≤ +0.03, negative in two sources). The benchmark is a far larger shift (country, closed course, staged conflicts) than the within-corpus shifts under which coverage demonstrably breaks. The main text never confronts this tension; the transfer boundary lives in Methods while the transfer success story leads Section 2.5 and the abstract.
(c) *Missing selection funnels and uncertainty.* The 2.1× ratio is computed over "judgeable moments" only. For the automated arm the funnel is given (67,861 candidates → 20.8% judgeable); for the human arm only the endpoint (713/15,102) appears — no candidate count, readability rate, or support-abstention rate. If the two arms pass the gates at different rates, the comparison is conditioned on different selections. Fig. 6a and 6c say "uncertainty not shown"; the paper's single most quotable number (2.1×) carries no interval anywhere, and the 15/15-scenarios-above-parity statement has no dependence-aware inference (scenarios share the same 19 systems).
*Why it matters:* This is the abstract's culminating claim and the result general readers will carry away; at NMI it must be evidence-proof in both directions.
*Remedy:* Report full per-arm funnels (candidate → readable → judgeable, with reason codes); give a scenario-run-clustered CI for the flag-rate ratio and per-scenario uncertainty in Fig. 6c; surface the LOSO coverage boundary in the main text; discuss the conservative-human confound and the maturity class of the 19 competition systems; reword "does not inflate its alarms" into a two-sided statement about what did and did not change across the move.

**M3. Abstract and title overrun the paper's own definitions (internal consistency).**
[Location: Title; Abstract; Section 2.5 heading; Fig. 6 title; Abstract vs Fig. 5b]
(a) *"Defining population" conflation.* The abstract says the reference is audited "with its defining population—human drivers in the same scenarios", and Section 2.5 is titled "The reference is audited by the population that defined it". The section's own first sentence contradicts this: the reference was learned from natural driving in the United States and Singapore; the auditors are different licensed drivers, in a different country, on a closed course. The 9.72% figure is the defining population; the 4.7% figure is not.
(b) *"No rise in emergency rates at any supported threshold" is circular.* A threshold is defined as "supported" precisely when its interval excludes zero difference (Fig. 5b caption); at the <3 s threshold the interval [−8.33, +3.15] admits a rise. The abstract phrasing therefore excludes, by construction, exactly the threshold at which the data are ambiguous, while sounding like an unconditional negative finding.
(c) *Title vs claims.* The paper's central and repeated discipline is that atypicality is not compliance/appropriateness, and that the harm link is bounded, not confirmed. A title promising monitoring of "socially compliant autonomous driving" asserts the very reading the text disclaims.
*Remedy:* Retitle (e.g., around "conditional social atypicality" or "social-compliance monitoring" with explicit scoping) or justify; rewrite the two abstract sentences precisely ("human drivers driving the same scenarios", "emergency rates are lower at thresholds under 2 s and inconclusive at 3 s").

**M4. Abstention dominates the benchmark, and the promised accounting is not delivered.**
[Location: Section 2.4; Methods 4.4–4.5]
On the benchmark the monitor issues verdicts on 20.8% of candidate moments; 55.3% are readable, so roughly 62% of readable moments fail human support or later gates, and in 36/267 runs the monitor never speaks. Methods 4.4 promises "the total abstention rate with its reason distribution, so that filtering is never hidden", and Section 2.3 concedes benchmark abstention "is larger and is reported with its reasons in Methods" — but no numeric reason distribution for the benchmark (or the human arm) appears anywhere in the manuscript. Substantively, if the human record covers only ~38% of readable moments in staged conflicts — precisely the regime the monitor targets — the operational utility of the instrument in its intended domain is limited, and the flag-rate comparisons ride on the surviving minority.
*Remedy:* Tabulate the full abstention funnel with reason codes for both benchmark arms (main text or ED); discuss what coverage of conflict-rich regimes is needed for deployment relevance and how the reference would be extended.

**M5. The consequence signature's inferential support is mixed, but the abstract states it as settled.**
[Location: Methods 4.5; Section 2.4; Fig. 5; Abstract]
The exposure placebo (whole-trajectory reassignment) gives p = 0.0199, but a case-level label permutation on the same battery does not reach significance (p = 0.1493), and the open-ended contract-window interval crosses zero at the 90% level ([−2.6100, +0.1372]); only the pre-specified fixed-window primary excludes zero. The authors disclose all of this in Methods — commendably — and then the abstract asserts "moments flagged on the assertive side are followed by tighter interactions on both sides" without qualification. With 15 scenario templates, 120 flag-containing runs and scenario-run clustering, the effective sample is modest, and two of the four inferential probes are equivocal.
*Remedy:* Bring both permutation tests into the main text; phrase the finding as a within-cell association supported by the primary window and timing placebo, with the case-level permutation reported alongside; or add analyses (e.g., more templates, hierarchical model) that resolve the discrepancy.

**M6. Construct validity of the IPV as a *social* measure is asserted, not evidenced, and estimator sensitivity is unexamined.**
[Location: Section 2.1; Methods 4.1; Eq. 1]
The paper is admirably explicit that online/offline agreement (r ≈ 0.993 on the lane-referenced slice) is "agreement between estimators, not recovery of a latent preference". But the entire semantic layer — "assertive side", "weights the counterpart's cost less than humans do" — is internal to a 7-candidate planner family with a Gaussian likelihood of σ = 0.1 m over a 1-s window. No evidence in the paper links IPV readings to human judgements of assertiveness or cooperativeness (the human-dispreference layer is explicitly deferred). Sensitivity of headline results to σ, the candidate grid (which stops at ±3π/8 inside the declared ±π/2 domain), the 0.20 near-uniform cutoff, and the k = 25 / 95th-percentile support gate is not reported; with σ = 0.1 m the likelihood is extremely peaked, so the "reliability" statistic's behaviour deserves scrutiny.
*Why it matters:* NMI readers will ask whether the monitor measures sociality or a planner-family projection of kinematics; the flag semantics inherit whatever the construct is.
*Remedy:* Add sensitivity analyses over σ, grid density/extent, and gate thresholds; either include one human-judgement linkage (even a small rater study on flagged vs within-range clips) or systematically confine the language to "atypical under the IPV instrument".

**M7. The Section 2.1 mechanism claims rest on the development split only.**
[Location: Fig. 2 caption]
Fig. 2's caption states "Panels use the development split; the confirmatory split is held sealed." The control contrasts that ground the readability construct (sharpening specific to real interactions; readable-does-not-mean-settled; summary-rule instability) are therefore exploratory as published. The paper's own freeze-then-confirm discipline is incomplete at exactly the subsection that licenses the monitor's front gate.
*Remedy:* Consume and report the confirmatory split for these analyses before publication, or mark the claims as exploratory in the main text.

**M8. Reproducibility and disclosure gaps around the real-vehicle benchmark.**
[Location: Methods 4.5–4.6; Data/Code Availability; ref. 53; Competing Interests]
The physical apparatus is essentially undescribed: what vehicle, what course, what a "scenario-scripted counterpart injection" physically is, how the counterpart is controlled in the 175 non-scripted runs, what the 19 "independent" systems are (maturity class: competition entrants, not production stacks — this materially scopes the 2.1× claim). The run accounting is incomplete (267 total; 27 scripted excluded; 175 used "with recorded outcomes" — the remaining 65 runs are unexplained, and outcome-recording failures could be selective). The central benchmark is cited as a URL (ref. 53) with no archived documentation. Data and code availability are both "upon publication" with no repository, licence or accession pathway named. Finally, the benchmark is "operated by its organisers" — a Tongji group — and the author is at Tongji; the relationship between the author and the benchmark organisation (scenario design, team selection, data access privileges) must be disclosed even if it is not a competing interest, because the paper leans on the benchmark's independence.
*Remedy:* Add an apparatus subsection; complete the run accounting; archive benchmark documentation; name repositories and access terms; state the author–organiser relationship and the safeguards (the outcome-blind crosswalk helps here).

---

## 3. Minor issues

1. Section 2.3: "Where a reading is issued, it abstains for lack of human support on only a small fraction of moments" — self-contradictory as written (issued readings cannot abstain); rephrase around eligible moments.
2. Fig. 2a: the quantity "change in how sharply the reading is identified" has no stated units and the baseline of each control pairing is under-explained for a general reader.
3. Run arithmetic: 267 = 175 (used) + 27 (scripted) + 65 (unaccounted); make the third category explicit (also raised in M8).
4. Human arm: 20 drivers × 15 scenarios implies 300 runs; the realised run count, dropouts, and per-driver exposure are not stated.
5. "Admissible span" is used both for the candidate span (3π/4) and next to the declared domain ([−π/2, π/2]); disclosed, but the dual usage will confuse readers of Fig. 4a.
6. Online–offline agreement (r ≈ 0.993) is quoted for the lane-referenced slice ("about three quarters of cases"); agreement on the remaining quarter is unreported.
7. Empirical p = 0.0199 from 200 permutation draws overstates precision (granularity ≈ 1/201); report as p ≈ 0.02.
8. The sixfold contributions list is not Nature-format style and reads as a checklist; fold into prose.
9. The controlled vocabulary (anchor row / readable / accepted / candidate / judgeable moment) is defined only in Methods 4.4; a small main-text box or table would spare the general reader.
10. Fig. 3a effect sizes (±0.05 rad on a ~1.9 rad range) are conceded to be "a few per cent" of the range; keep all summary statements of Section 2.2 calibrated to that concession (the current text mostly is).
11. Fig. 6c's "15/15 scenarios above parity" is presented without any inferential statement; scenarios are not independent across the shared systems.

---

## 4. Questions to the authors

1. What ethics approvals, consent procedures and on-track safety protocols governed the human-driver arm and the staged-conflict benchmark generally?
2. Provide the complete moment funnels for both benchmark arms (candidate → readable → within-support/judgeable, with reason-code distributions). Are the judgeable fractions comparable between the automated and human arms? If not, how does differential selection affect the 2.1× comparison?
3. What is the per-source achieved coverage of the deployed (all-source-fitted) reference on the natural test fold — not only the leave-one-source-out refits? Pooled coverage within 0.6 pp of nominal could mask per-source miscalibration dominated by Waymo's share.
4. Why is the confirmatory split for the Section 2.1 analyses still sealed at submission, and will it be consumed before publication?
5. What physically is a "counterpart injection", and what controls the counterpart in the 175 non-scripted runs? Describe vehicle, course and injection mechanism.
6. What is the author's relationship to the OnSite organisation (scenario design, team admission, data governance)? What safeguards ensured the monitor's development was independent of benchmark outcomes?
7. How sensitive are the headline numbers (flag rates, 21%/20%/8% narrowing, consequence ratios) to σ = 0.1 m, the seven-candidate grid and its ±3π/8 extent, the 0.20 near-uniform cutoff, and the support-gate parameters (k = 25, 95th percentile, ≥50 anchors/≥10 cases)?
8. Why do 65 of the 240 non-scripted runs lack recorded outcomes, and can outcome-recording failure correlate with interaction intensity?
9. What maturity class are the 19 driving systems (research prototypes, competition stacks, production-intent)? In what sense are they "independent"?
10. Can you provide a scenario-run-clustered confidence interval for the 9.8%/4.7% ratio and dependence-aware inference for the per-scenario comparison?

---

## 5. Prioritised revision requests

1. **Complete mandatory reporting** (M1): ethics approval, consent, safety protocol, funding, author contributions, acknowledgements. Non-negotiable before re-review.
2. **Secure or temper the Section 2.5 headline** (M2): per-arm funnels; clustered CI on the flag-rate ratio; uncertainty in Fig. 6a/c; two-sided discussion of the 4.7% vs 9.72% halving; scope "automated systems" to the benchmark's system class.
3. **Fix wording overclaims** (M3): defining-population conflation in abstract and Section 2.5 heading; "supported threshold" circularity; reconcile the title with the atypicality-not-compliance stance.
4. **Publish the abstention accounting** (M4): numeric reason distributions for benchmark and human arms; main-text statement of the 20.8% verdict rate and its implications; surface LOSO coverage in the main text.
5. **Report the confirmatory split for Fig. 2** (M7), or label those claims exploratory.
6. **Balance the consequence-signature inference** (M5): both permutation tests in the main text; qualified phrasing in the abstract.
7. **Add apparatus and availability specifics** (M8): benchmark hardware/protocol description; run accounting; archived benchmark documentation; named repositories for data and code; author–organiser disclosure.
8. **Add estimator sensitivity analyses and construct scoping** (M6); if feasible, a small human-judgement linkage study for flagged vs within-range moments.

---

## 6. Acceptance probability

- **(a) As submitted:** 8% — the missing ethics/consent reporting alone returns the paper from the standards desk; the headline-claim gaps would independently draw major objections.
- **(b) Assuming a competent major revision** (all of requests 1–7, credible progress on 8): 45% — the conceptual frame, scale, and audit discipline are genuinely NMI-grade; residual risk sits in construct validity and the operational-utility question (a monitor that speaks on one moment in five, with a 1.87-rad-wide 90% band).

---

## 7. Recommendation

**Major revision.**
