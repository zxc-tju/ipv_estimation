# Referee Report — Nature Machine Intelligence
## "Online monitoring of socially compliant autonomous driving" (31 pp., 58 refs.)

---

## 1. Summary assessment

The manuscript proposes runtime monitoring of "conditional social atypicality" for autonomous vehicles: an interaction preference value (IPV) estimated online by likelihood-based inverse planning, an explicit readability/abstention gate, a situation-conditioned human reference range calibrated by split-conformal quantile regression, and a consequence audit on a matched-scenario real-vehicle benchmark including a human-driver arm judged by the same frozen instrument. The framing is genuinely novel and well suited to this journal: it moves social compliance from offline aggregate scores to a per-moment, auditable instrument, and the abstention discipline ("not readable is not neutral") is exemplary. The context-dependence analysis (38,228 interactions, four corpora) and the closed-loop audit by the defining population are the strongest elements. However: the normative meaning of the monitored scalar is deferred entirely to future work; the consequence signature is vulnerable to mechanical circularity and is never compared against a situation-conditioned kinematic baseline; a headline sharpening number (">40%") contradicts the figure it cites (21/20/8%); the "2.1× humans" headline depends on which of three baselines is adopted; the benchmark population is essentially uncharacterised; and ethics statements plus several promised materials are absent. Substantial, salvageable work — major revision.

---

## 2. Major weaknesses

**M1. The construct being monitored is never validated as "social" within the paper.**
[Abstract; §1 validation path; §2.4; §3; Methods 4.1]
The IPV originates in the author's own prior work [24, 25], and the only in-paper validation of the online estimator is agreement with the offline version of itself (r ≈ 0.993 on the lane-referenced ~75% slice), which the authors candidly label "agreement between estimators, not recovery of a latent preference." No evidence in this manuscript connects an IPV flag to anything a human would recognise as socially assertive, discourteous, or dispreferred — no human-rating study, no convergent validity against established constructs (e.g., SVO), nothing. The validation chain drawn in §1 (atypicality → dispreference → harm) is honest, but the paper occupies only its first link. At NMI level this is decisive: the title, abstract and framing promise *social* compliance monitoring; what is delivered is a well-calibrated atypicality monitor for an unvalidated model-derived scalar. The consequence signature (Fig. 5) is the paper's only bridge to meaning, and it is compromised (see M2). *Remedy:* a human-judgement validation (raters assess flagged vs. situation-matched unflagged moments), or convergent-validity analysis against an independent behavioural construct; failing that, claims and title should be narrowed to behavioural atypicality.

**M2. The consequence signature may be partially mechanical, and no incremental-value baseline is provided.**
[§2.4, §2.6, Fig. 5]
An assertive-side flag means the trajectory window is best explained by a low-θ candidate — i.e., the ego is pressing the gap. The "consequences" measured over the following window — ego margin contraction, counterpart deceleration — are near-mechanical correlates of exactly that kinematic pattern. Reading counterpart speed from the other vehicle's control record removes sensor sharing, not physical coupling; calling the two sides "a convergence rather than a restatement" overstates their independence. Section 2.6 argues social monitoring sees what safety checks cannot, but the only comparator is binary emergency thresholds (TTC bins, hard braking). The natural competitor — a *graded, situation-conditioned reference range built identically on plain kinematic quantities* (closing speed, gap-closure rate) — is never tested and might reproduce most of Fig. 5. Without it, "social atypicality and conventional safety register different things" is unsupported at the level that matters. *Remedy:* construct the kinematic-baseline monitor under the identical conformal pipeline and show IPV flags carry incremental information about the downstream signature.

**M3. The headline sharpening claim contradicts the paper's own figure.**
[Abstract; §2.3; Fig. 4; Methods 4.3]
Abstract and §2.3 claim "more than a 40% reduction in interval width at near-nominal coverage," and Methods 4.3 repeats ">40% width reduction with a Winkler-score improvement above 35%." Figure 4a — the cited evidence — shows reductions of **21%, 20% and 8%** at the 80/90/95% levels. Nowhere is the 40% derivable from displayed numbers, and the Winkler figure appears nowhere else. Compounding this, the conditioned range still spans 62–92% of the *entire admissible scale* [−π/2, π/2], and the situation explains only 20.9% of reading variance (Fig. 4c) — so "narrow, calibrated, auditable" is doing heavy rhetorical work for an instrument that, in absolute terms, is coarse and flags only the extreme tail. A quantitative contradiction on a headline claim is disqualifying at Nature-family standard until resolved. *Remedy:* reconcile or correct the 40%/35% figures with an explicit definition (quantity, baseline, subset); temper "narrow" with the absolute widths.

**M4. The three flag rates support incompatible stories, and the paper headlines the favourable one.**
[Abstract; contribution 6; §2.5; Fig. 6]
The audit produces three rates at the 90% level: natural-driving held-out humans 9.72%, matched closed-course humans 4.7%, automated systems 9.845%. The text correctly notes "no pair of them tells the story alone" — and then the abstract tells it with one pair ("automated systems flagged about twice as often"). Two problems. (i) *Baseline instability:* against natural-driving humans, the automated systems sit essentially at the native outside rate — parity, not deficit; the 2.1× exists only against closed-course humans. (ii) *The favourable reading of 4.7%:* a frozen 90% instrument flagging its new-domain human population at 4.7% is not evidence that "the instrument survives the move"; it is over-coverage — conservative calibration drift — plausibly caused by instructed, observed, closed-course drivers behaving cautiously. That same drift mechanically inflates the 2.1× ratio. The transfer celebration in §2.5 also sits uneasily beside Fig. 3c (held-out-source R² ≤ 0) and Fig. 4's own caption ("transfer to a data source not seen during fitting is not established"). The paper cannot simultaneously disclaim ranking ("no panel ranks vehicles") and headline a ratio in the abstract and contribution list. *Remedy:* present all three rates symmetrically with uncertainty; analyse the 4.7% explicitly as possible conservative shift (per-driver dispersion, instruction effects); state the parity-with-native reading; align the abstract.

**M5. The benchmark and reference populations are under-characterised to the point that key numbers cannot be reconciled.**
[§2.4–2.5; Methods 4.5–4.6; refs. 50, 53]
Nowhere is it stated how many automated systems were tested, what they are (production stacks or research-grade competition planners — which determines what the 2.1× can possibly generalise to), how many runs per scenario, or which edition of the benchmark was used — indeed reference [53] still contains the author note "Verify the specific challenge edition/year for the data used." The universes do not reconcile from the text: Fig. 5 uses 175 scenarios with 472 + 11,669 = 12,141 judgeable moments; Fig. 6 uses 15 matched scenarios with n = 14,099 automated-vehicle moments — more moments from one-twelfth the scenarios, with no explanation of the run structure connecting them. Separately, the "human" reference is built from AV-perception corpora (Waymo 23,218 of 38,228 cases): were the data-collection vehicles' own (machine-driven) trajectories excluded from the reference population? This is never stated, and machine contamination of the human reference would undercut the paper's central object. *Remedy:* a full benchmark reporting table (systems, runs, scenario counts, moment universes) and an explicit reference-population provenance statement.

**M6. Statistical prose outruns the intervals in several load-bearing places.**
[§2.4; §2.5; Figs. 5–6]
"Without added emergencies" (abstract) rests on interval estimates of which at least one (< 3 s margin bin: −3.14 [−8.33, +3.15] pp) admits no difference and cannot exclude a ~20% relative increase against its base rate; absence claims require equivalence bounds, not non-significance. "Every emergency tail is rarer, not more common" (human arm) is asserted while Fig. 6b concedes intervals exist only "where defined." The 472 flagged moments cluster into ~120 runs (ED2), i.e. ~4 frames (~0.4 s) per flagged run — the scenario-level bootstrap is appropriate, but the effective number of independent flagged episodes should be reported alongside moment counts. The paper's headline comparison figure carries no uncertainty at all in panels a and c ("uncertainty not shown"). *Remedy:* equivalence testing for no-increase claims; episode-level counts; confidence intervals on every comparison panel; prose tempered to what intervals support.

**M7. Compliance and completeness gaps that independently block publication.**
[Methods 4.6; Extended Data; back matter]
(i) Twenty licensed drivers drove staged conflicts against scripted injections in a real vehicle — human-subjects research with physical risk — yet there is no ethics-approval or informed-consent statement (only pseudonymisation). (ii) The counterfactual-injection planner-interface demonstration promised in Methods ("reported in Extended Data") does not exist in the manuscript (Extended Data contains only ED1–ED2). (iii) Methods defers required reporting (abstention reason distribution, leave-one-source-out coverage, active-interaction and readability rates) to supplementary material whose section headers are empty. (iv) Methods cites `claims_register.md`, an internal project file unavailable to readers. (v) Reproducibility: the quantile-model class, gate thresholds (u₀, q₀, c₀), split sizes and feature dimensionalities are all unstated; data and code are "available upon publication" with no repository or accession. *Remedy:* ethics statement, complete Extended Data/Supplementary, remove internal artefacts, full reproducibility reporting per Nature checklist.

---

## 3. Minor issues

1. Eq. (2): u_a is introduced as a "reliability measure" but the gate requires u_a < u₀, implying u_a is a spread/uncertainty quantity; define it precisely — the central equation should not need reverse-engineering.
2. The primary consequence variable, the ego "margin" (seconds, log-scaled to 10² in Fig. 5a), is never defined, and the length of "the window following the verdict" is unstated in the main text; margins of ~100 s suggest post-resolution frames enter the window.
3. §2.4 "fewer than one judgeable moment in ten is flagged" is ambiguous: assertive-only is 472/12,141 ≈ 3.9%; both sides ≈ 9.8%. State which.
4. Fig. 3b sample sizes (23,211/5,098/2,404/7,498) differ slightly from §2.2 totals (23,218/5,105/2,406/7,499) with no explanation of the exclusions.
5. The validation-path display on p. 3 overflows the right margin (text clipped at "interact…").
6. Methods register slips: "the the onset" (§4.5); "we do not call u_a a standard deviation until its definition in the released time series is audited" reads as an internal audit note, not journal prose.
7. Reference [35] is printed with several hundred author names across nearly two pages; truncate. Reference [53]'s leftover instruction must be removed.
8. Fig. 5c prints point ratios (2.35×, 2.04×) for entries explicitly marked "not supported" — this invites quotation of unsupported numbers; suppress the values.
9. Fig. 4a's percentage annotations mix signs (−21%, +20%, −8%) for the same directional quantity; Fig. 4c is a single stacked bar and could be a sentence.
10. The end-to-end fraction of time the monitor speaks on natural data (461,937/3,695,981 ≈ 12.5% of frames) is never stated plainly; readability and active-interaction rates belong in the main text.
11. §2.5 describes the reference as "natural human driving recorded in other countries"; the corpora are fleet-collected in specific US/Singapore cities, and human behaviour around data-collection AVs may itself be adapted — one sentence on this population's limits is needed.
12. Acknowledgements and Author Contributions are empty headers.

---

## 4. Questions to the authors

1. Derive the ">40% width reduction" and ">35% Winkler improvement" exactly: which quantity, which baseline, which subset? Why does Fig. 4a show 21/20/8%?
2. How many automated systems are in the benchmark, of what kind (production vs. research prototypes), with how many runs per scenario? Reconcile the 175-scenario universe (12,141 moments, Fig. 5) with the 15-scenario universe (14,099 moments, Fig. 6), and name the benchmark edition.
3. Were the data-collection vehicles' own trajectories (e.g., the Waymo ego) excluded from the human reference population? Give the per-source composition of reference egos.
4. Does a situation-conditioned conformal reference range built on plain kinematic quantities (closing speed, gap-closure rate) reproduce the Fig. 5 consequence signature? What incremental predictive content do IPV flags carry beyond it?
5. Given frame-to-frame reading movement of 0.30 rad where most readable (Fig. 2b) against bands of ~2–2.5 rad, what is the distribution of flagged-stretch durations, how much verdict flicker occurs, and what are the persistence-gate parameters?
6. Is unreadability correlated with assertive kinematics — i.e., could a planner evade flags by remaining unreadable? Report the benchmark abstention fraction by reason code.
7. Human arm: what were the drivers' instructions and incentives, were they aware conflicts were scripted, and what is the per-driver flag-rate dispersion? Does any driver exceed the automated-system mean?
8. Do you accept that 4.7% outside a 90% reference indicates conservative domain shift (over-coverage) on the closed course? Provide the leave-one-source-out coverage numbers Methods says are "reported as boundaries."
9. What ethics approval and safety protocol governed the staged-conflict driving study?
10. Where is the counterfactual-injection demonstration referenced in Methods 4.6?

---

## 5. Prioritised revision requests

1. **Repair claims-evidence consistency on the sharpening result** (M3): reconcile abstract, §2.3, Methods 4.3 and Fig. 4; report absolute band widths alongside relative reductions.
2. **Add the situation-conditioned kinematic-baseline control** (M2): identical pipeline, kinematic target; report incremental value of IPV flags. This is the experiment on which the paper's "beyond safety" thesis stands or falls.
3. **Fully characterise benchmark and reference populations** (M5): systems, runs, universes, edition, reference-ego provenance; reconcile all n's.
4. **Rebalance the three-rate audit** (M4): symmetric presentation with uncertainty, per-driver dispersion, explicit treatment of the conservative-shift reading; align the abstract's "twice as often" with the "no ranking" discipline.
5. **Complete compliance and reporting** (M7): ethics/consent statement, missing Extended Data demonstration, populated Supplementary (abstention reasons, LOSO coverage), reproducibility details, repository commitments, internal artefacts removed.
6. **Temper statistical prose** (M6): equivalence bounds for "no added emergencies," episode-level effective n, uncertainty on all comparison panels.
7. **Close or narrow the construct-validity gap** (M1): add a human-judgement or convergent-validity study, or recast title and claims as behavioural atypicality monitoring with social meaning as forward validation.

---

## 6. Acceptance probability

- **(a) As submitted:** 5%. The claims-figure contradiction (M3), the uncharacterised benchmark (M5) and the missing ethics statement (M7) would each independently prevent acceptance in the current state.
- **(b) Assuming a competent major revision:** 35%. The framework, abstention discipline and closed-loop audit are NMI-calibre ideas; the residual risk is that the kinematic-baseline control (request 2) shows the IPV flags to be largely redundant, which would cap the contribution at machinery rather than measurement.

---

## 7. Recommendation

**Major revision.**
