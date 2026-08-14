# Reviewer D — Round 3 (journal-side editor-generalist)

Manuscript: "Online monitoring of socially compliant autonomous driving" (35 pp., 58 refs)

---

## 1. Summary assessment

The manuscript recasts social compliance for autonomous vehicles as online monitoring of conditional social atypicality: a per-moment interaction preference value (IPV), gated by an explicit abstention ("readability") layer, is tested against a situation-conditioned, conformally calibrated human reference range. Evidence spans 38,228 naturalistic interactions from four corpora, a matched-scenario real-vehicle benchmark (19 automated systems, 267 runs), and a symmetric human-driver arm (20 drivers, same scenarios). The framing is genuinely fresh for this journal; the epistemic discipline (abstention semantics, atypicality distinguished from inappropriateness, harm explicitly bounded rather than claimed) is exemplary, and the numerical bookkeeping is unusually consistent. However: the human-participant arm carries no ethics statement; the headline comparative claims (automated systems flagged 2.1 times as often; 15/15 scenarios above parity) carry no uncertainty; the transfer narrative conflicts with the paper's own leave-one-source-out under-coverage; a relative-concentration-only reliability gate leaves model misfit unexcluded as an explanation for machine flag rates; and the centrepiece "the situation suffices" rests on an unquantified null. All are fixable largely without new data collection. A promising, careful paper that is not yet at Nature-family submission standard. Major revision.

---

## 2. Major weaknesses

**M1. Reporting and ethics completeness fails the standards desk.**
[Location: Methods 4.6 (p.23–24); Declarations (p.35); Data/Code Availability (p.25)]
Twenty licensed human drivers drove staged conflict scenarios in a real vehicle against scripted counterpart injections. This is human-participant research, yet the manuscript contains no ethics-committee approval, no informed-consent statement, no safety protocol, and no recruitment/instruction/compensation detail ("Driver identities are pseudonymised before analysis" is the only related sentence). Author Contributions and Acknowledgements are empty; funding is undeclared. Data and code availability are "will be made available upon publication" with no repository, licence, accession or reviewer-access plan — for a paper whose entire contribution is an auditable instrument, this is self-undermining. Finally, the benchmark [53] is operated by a group at the author's own institution; whatever the author's actual role, the relationship and the terms of access to official benchmark data (which the analysis plan says were examined and excluded as endpoints) must be disclosed. Why it matters: any of these alone triggers a return from the Nature Portfolio standards desk before scientific assessment. Remedy: full ethics statement (approval body, protocol number, consent), completed declarations, competing-interest/role clarification regarding the benchmark organisers, and a concrete deposit plan with peer-review access.

**M2. The headline comparative claims carry no uncertainty and no population scoping.**
[Location: Abstract; §2.5 (p.15); Fig. 6a,c (p.16)]
The abstract's most quotable result — "automated systems flagged about twice as often" (9.8% vs 4.7%) — is reported without any interval, as is "per-scenario flag rate higher in 15 of 15 scenarios"; Fig. 6a and 6c are annotated "uncertainty not shown". These rates aggregate moments that cluster within drivers (20), systems (19), scenarios (15) and runs; a cluster-robust interval is both necessary and easy to compute, and the 15/15 sign claim needs an analysis that respects scenario-level dependence across shared systems. Separately, the 19 automated systems are entries in a national algorithm challenge, nowhere characterised by maturity or stack type; a broad NMI readership will read the abstract as a statement about autonomous vehicles in general. Why it matters: the journal cannot headline an uncertainty-free, population-unscoped ratio; this is the number that press coverage would carry. Remedy: clustered CIs for every comparative rate (including Fig. 6a/c), a description of the systems, and abstract wording scoped to "19 automated driving systems on this benchmark".

**M3. The transfer story is internally inconsistent, and "no alarm inflation" is conditional on heavy, asymmetrically reported abstention.**
[Location: contribution 6 (p.4); §2.5 title and text (pp.13–15); Abstract; Fig. 4 caption (p.11); Methods 4.4 (p.21–22)]
Contribution 6 states the reference "transfers across country, apparatus and task without inflating its alarms", and §2.5 is titled "The reference is audited by the population that defined it". But (i) the Fig. 4 caption states "Transfer to a data source not seen during fitting is not established", and Methods reports leave-one-source-out coverage of 0.743 (Waymo) and 0.750 (Lyft) at 90% nominal — i.e., roughly 2.5x alarm inflation across naturalistic sources, with Argoverse-2 reaching 0.900 only alongside 44.3% held-out abstention; (ii) the human arm is not "the population that defined" the reference — the reference comes from natural driving in the United States and Singapore, the audit arm from different licensed drivers in another country on a closed course; (iii) on the benchmark only 20.8% of candidate moments receive verdicts, and the human arm's candidate-to-judgeable funnel is never reported, so the 4.7% vs 9.8% comparison could be shaped by differential abstention between arms; (iv) the human-on-course rate (4.7%) is half the pre-specified anchor (9.72%) — a 2x conservative calibration shift that the text presents as success, while the AV rate "sits at the native level" is a coincidence across different context mixes dressed as meaning. Why it matters: the paper's own strongest honesty (LOSO, abstention accounting) contradicts its own strongest marketing (contribution 6, section title, abstract phrase); at NMI this reads as a framing failure that a rebuttal cannot paper over. Remedy: temper contribution 6 and retitle §2.5 accurately ("audited by human drivers under the same instrument"); report both arms' full funnels (candidate → readable → supported → judgeable) next to the flag rates; foreground the paired per-scenario contrast (Fig. 6c), which is the analysis that actually controls context mix; state explicitly that cross-source deployment would require support-gate abstention at the observed LOSO levels.

**M4. Construct validity for machine policies: no absolute-fit gate, so model misfit is an unexcluded explanation for the higher machine flag rate.**
[Location: Eq. 1–2 (p.5); Methods 4.1 (pp.18–19); §2.5 (p.15)]
The reliability gate is purely relative: a reading is kept when the largest normalised candidate weight exceeds 0.20 (uniform is 0.14). Nothing gates absolute goodness of fit, so a trajectory poorly explained by *all* seven candidates — e.g., a machine policy outside the human planner family — can still be confidently mapped to the extreme assertive candidate and flagged. The consequence signature (Fig. 5) and its replication in the human arm are a genuine defence that flags mark real interaction tightening, but they do not exclude that part of the 2.1x machine-vs-human flag ratio reflects worse absolute fit of machine trajectories to the human-derived candidate manifold rather than stronger assertive preference. Why it matters: the central comparative claim of §2.5 is exactly the kind of measurement-artifact question a statistically minded NMI referee will ask first. Remedy: report absolute-fit distributions (best-candidate MSE) separately for human-arm and automated-arm judgeable moments; test an absolute-fit gate and show the flag-rate ratio's sensitivity to it; state the limitation explicitly if a residual gap remains.

**M5. The centrepiece "the situation suffices; the counterpart adds nothing" is an unquantified null with an obvious attenuation confound.**
[Location: §2.3 (p.10); Methods 4.3 (p.20); Discussion (p.17)]
The evidence is a paired 90% interval-score difference of -0.0002 (case-clustered p = 0.86). No equivalence margin is pre-specified, no power analysis is given, and the counterpart channel enters as the *online-estimated* counterpart IPV — a noisy measurement whose error alone could erase any incremental value (classical attenuation), especially since Fig. 2b shows readable readings moving 0.30 rad between consecutive frames. Yet the Discussion converts this null into a deployment-level conclusion: "a runtime social monitor does not need to read the other agent's hidden intention." Why it matters: the paper's self-declared most informative finding (the event-vs-case tension) currently rests on absence of evidence, not evidence of absence; this is the difference between an NMI headline and an overreach. Remedy: equivalence testing with a justified margin; an upper-bound analysis conditioning on the *offline* (less noisy) counterpart IPV to bound what a noiseless counterpart channel could add; soften the Discussion claim to match whichever bound survives.

**M6. Per-moment verdicts versus reading volatility: the flag population may be dominated by single-frame flicker, and no persistence sensitivity is reported.**
[Location: Fig. 2b (p.7); §2.4 (pp.12–13); Fig. ED2 caption (p.27); Methods 4.4 (p.22)]
Readable readings move 0.30 rad frame-to-frame — comparable to the distance between a typical reading and the band edge — and the ED2 selection funnel reveals that of 120 runs containing a flagged assertive moment, only 20 contain at least five clustered flagged frames: in five of six flagged runs, the flags last under half a second. Every reported flag rate and the entire consequence battery anchor on per-moment flags, while the persistence layer that a deployment would actually use is explicitly not evaluated. Why it matters: if most flags are estimator flicker, the operational meaning of "auditable runtime monitor" weakens, and the consequence signature could be driven by a small persistent subset. Remedy: report the flagged-stretch length distribution for both arms; recompute the Fig. 5/Fig. 6b signatures under minimum-persistence thresholds (e.g., >=3, >=5 frames); either evaluate the persistence layer's operating characteristics or temper the runtime-monitor language.

**M7. Title and construct scope overreach the paper's own boundaries.**
[Location: Title; Abstract; §1 (p.3); Discussion (p.17)]
The title promises monitoring of "socially compliant autonomous driving" while the text repeatedly and correctly insists that what is monitored is atypicality, that atypicality is not inappropriateness, and that the harm link is bounded, not confirmed. By the paper's own three distinctions, the title claims the thing the paper explicitly does not deliver. Moreover, "social compliance" is operationalised as a single model-derived scalar (the IPV trade-off angle) for pairwise vehicle-vehicle conflicts at mapped points; signalling, courtesy conventions, gap acceptance, VRUs and multi-agent scenes are all outside the construct (acknowledged only late in the Discussion). Why it matters: NMI titles and abstracts are read stand-alone; the mismatch invites exactly the misreading the authors have taken such care to prevent in the body. Remedy: retitle toward what is delivered (e.g., "Online monitoring of social atypicality in autonomous driving") or add explicit scoping in title/abstract; state early that the IPV is one dimension of social interaction.

---

## 3. Minor issues

1. "Judgeable" is used in Results (§2.4, p.12) before its definition (Methods 4.4, p.21); same for "anchor row". Add a one-line vocabulary note at first use.
2. Run accounting is incomplete: 19 systems x 15 scenarios = 285, but 267 runs are reported (18 missing, unexplained); 175 non-scripted + 27 scripted = 202, leaving 65 of 267 runs unaccounted for in the consequence analyses; the human arm's run count (20 x 15 = 300?) and any exclusions are never stated.
3. The counterpart vehicle's control is underdescribed for a general reader: who or what drives it, how "scenario-scripted injections" differ between the 27 fixed-script runs and the responsive runs, and whether counterpart behaviour was identical across the automated and human arms.
4. Online/offline estimator agreement (r ~ 0.993) is established only on the lane-referenced slice ("about three quarters of cases"); the status and treatment of the remaining quarter is unclear.
5. p.12: duplicated parenthetical "(no engineering failures (no solver or pipeline failure on any candidate moment))".
6. Fig. 4c (20.9% explained / 79.1% residual bar) is a weak visualisation of a single number; consider a sentence instead, or an R^2-by-source display.
7. "Sits at the native level" (§2.5, p.15) for the AV rate (9.8% vs 9.72%) compares rates across different context mixes; the paper's own human-arm shift (9.72% -> 4.7%) shows such coincidences are not meaningful. Drop or reword.
8. Ref. [53] is a URL-style citation to the benchmark platform; give a formal citable reference with access date, and align it with the disclosure requested in M1.
9. Fig. 3a effect sizes (+0.058 to -0.034 rad) are a few per cent of the ~1.87 rad range width; the text handles this honestly, but the §2.2 topic sentence ("does not reduce to a single global sociality score") would be better anchored to the operational sharpening argument than to the small sign reversal.
10. Abstract phrase "with its defining population" repeats the §2.5 imprecision (see M3(ii)); fix in both places.
11. The Fig. 1c caption says marked readings lie "above the 90% range (the over-yielding side)" while the in-panel annotation reads simply "outside the human reference range"; make the side explicit in the panel.
12. Methods 4.4 reports conditional coverage and trajectory-wise coverage as "not established"; this important boundary deserves one sentence in the main text near the calibration claim, not only in Methods.

---

## 4. Questions to the authors

1. What ethics approval and consent governed the 20-driver arm, and what safety protocol governed staged conflicts with a real vehicle? How were drivers recruited, instructed and compensated, and could instruction-induced conservatism explain part of the 4.7% rate?
2. Provide cluster-robust intervals for 9.8% vs 4.7% and for the 15/15 per-scenario claim. How variable are flag rates across the 19 systems and 20 drivers (ranges, not rankings)?
3. Report the human-arm funnel (candidate moments → readable → supported → judgeable) alongside the automated arm's 67,861 → 55.3% → 20.8%. Are judgeable fractions comparable, and if not, how does differential abstention affect the comparison?
4. For the counterpart ablation: what incremental sharpening could your design have detected (power/equivalence bound), and what does conditioning on the offline counterpart IPV — a less noisy proxy for the true preference — add? 
5. What are the absolute-fit (best-candidate MSE) distributions for human vs automated judgeable moments, and how does the 2.1x ratio change under an absolute-fit gate?
6. What is the distribution of contiguous flagged-stretch lengths in both arms, and does the Fig. 5 consequence signature survive a minimum-persistence requirement?
7. How do you reconcile LOSO under-coverage (0.743/0.750) across naturalistic sources with successful cross-country transfer to the benchmark? Is the implied deployment claim that the support gate will abstain its way to validity on any new source?
8. Account for the 285-vs-267 run discrepancy, the 65 runs absent from consequence analyses, and the human-arm run count.
9. Who controls the counterpart vehicle in each run type, and is counterpart behaviour matched across arms?
10. What is the author's relationship to the OnSite benchmark organisers, and under what terms were official outcome labels accessed and then excluded as endpoints?
11. What will actually be deposited (datasets, verdict series, code), where, under what licence, and can reviewers access it during revision?

---

## 5. Prioritised revision requests

1. **Declarations and deposit (M1):** ethics statement for the human arm; completed Author Contributions, Acknowledgements, funding; benchmark-role disclosure; concrete data/code repository plan with reviewer access; Nature reporting summary.
2. **Uncertainty and scoping for headline rates (M2):** clustered CIs for all comparative flag rates including Fig. 6a/c; characterise the 19 systems; scope the abstract claim to the benchmark population.
3. **Reconcile the transfer narrative (M3):** temper contribution 6; retitle §2.5 and fix the abstract's "defining population"; report both arms' abstention funnels next to the flag rates; foreground the paired per-scenario contrast; state LOSO-implied deployment behaviour.
4. **Absolute-fit analysis (M4):** fit distributions by arm; flag-rate sensitivity to an absolute-fit gate; explicit limitation if a gap remains.
5. **Equivalence treatment of the counterpart null (M5):** pre-specified margin, attenuation/oracle analysis, and correspondingly softened Discussion claim.
6. **Persistence sensitivity (M6):** flagged-stretch distributions; consequence signature under minimum-persistence thresholds; temper or evaluate the runtime layer.
7. **Title/scope alignment (M7):** retitle toward atypicality or scope explicitly; early statement that IPV is one dimension of social interaction.
8. **Minor fixes:** accounting (runs), counterpart control description, terminology-before-definition, figure and wording items listed above.

---

## 6. Acceptance probability

- (a) As submitted: **5%** (the missing ethics statement alone would halt the manuscript at the standards desk; uncertainty-free headline comparisons and the internal transfer-claim conflict independently preclude acceptance in current form).
- (b) Assuming a competent major revision addressing requests 1–7: **60%** (the underlying evidence structure is careful and largely already in place; residual risk sits in the editorial novelty judgement, the null-centred centrepiece after equivalence analysis, and how the absolute-fit analysis lands on the 2.1x claim).

---

## 7. Recommendation

**Major revision.**
