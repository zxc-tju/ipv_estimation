# Round-2 revision plan — executable items only (FIX-TEXT / FIX-FIGURE / CLARIFY)

Base: `main.tex` @ 41b819a (= HEAD, clean tree). All line anchors refer to this state.
Evidence root: `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/` (research repo). RQ021 run dir = `reports/studies/RQ021_contemporaneous_envelope/RQ021_1_contemporaneous_envelope_20260805T160425Z_43b4bff/`.

Standing constraint screen applied to EVERY item: evidence numbers frozen (text may only move TO frozen values); the 14 `\targetnum` call sites byte-identical (L88, L184, L477, L484 x3, L491 x2, L494, L495, L496 x2, L704, L705); figH SYNTHETIC watermark + red header untouched; two-layer discipline (no codenames/audit vocabulary in main text); vocabulary rules (no estimability family; lower side = "more assertive"; abstention never neutral); 发布会原则 (accuracy corrections only, no self-weakening); frozen external-validation order; no invented facts (→ ESCALATE instead). Tags: [NOFAB] = wording deliberately claims nothing beyond frozen evidence; [VOCAB]; [PRESS]; [TARGETSAFE] = edit verified not to touch a `\targetnum` argument.

Compile after each tier; `\targetnum` count must remain exactly 21 tokens (14 call sites + macro def + 6 comments) with byte-identical arguments.

---

## Tier 0 — process

### Q00 — Rebuild the review PDF from HEAD (stale-figure regression)
- Target: build artifact only; NO repo change.
- Problem: `manuscript_41b819a.pdf` embeds the pre-P14 `fig5_consequence.pdf` (its p.13 text layer contains "2.35x"/"2.04x", absent from the repo figure) and by the same token likely the pre-P15 `fig0_concept.pdf`. Round-2 reviewers partly reviewed retired figure content.
- Action: `pdflatex; bibtex; pdflatex x2` from the current tree; verify p.13 no longer contains "2.35"; archive the fresh PDF as the round-3 review object.
- Resolves: R2-50; defuses the figure-facing half of A-m10.

## Tier 1 — correctness (claims-evidence and text-figure breaks)

### Q01 — Remove the duplicated "References" heading
- Target: main.tex L814. Current: `\section*{References}` immediately above `\bibliographystyle{unsrt}` + `\bibliography{bibliography/biblio}`.
- Edit: delete the `\section*{References}` line (unsrt's thebibliography emits its own heading).
- Constraint check: layout only. Resolves R2-37(i) (D-m11).

### Q02 — End-matter order and Extended Data float control
- Target: main.tex L759–813 block. Current order: Extended Data heading -> ED1 figure -> ED2 figure -> Data Availability -> Code Availability -> References -> declarations; in the PDF the [t] floats jump above the heading and Data Availability interleaves between ED figures.
- Edit: reorder to Data Availability -> Code Availability -> `\clearpage` -> `\section*{Extended Data}` -> `\suppressfloats[t]` -> ED1 figure (`[!htb]`) -> ED2 figure (`[!htb]`) -> References -> declarations. (Any equivalent arrangement in which the heading demonstrably precedes ED1 on the rendered page is acceptable; verify visually.)
- Constraint check: layout only; captions untouched. Resolves R2-37(ii) (D-m11).

### Q03 — Benchmark abstention accounting: replace the "limiting factor" sentence with the frozen two-level truth
- Target: main.tex L391–398 (§2.4 "The monitor is, first, able to speak" paragraph).
- Current (L393–397): "…returns a verdict on 14,099 (about one in five; no engineering failures), and in 231 of the 267 scenario runs (nearly nine in ten) it speaks at least once. No scenario run is silenced because the ego's own preference carries too little information to read; where the monitor abstains, the limiting factor is the availability of comparable human behaviour in that situation."
- Problem: true at run level (RQ017-KC-C2) but false as a moment-level reading — 44.7% of candidate moments fail the discriminative-information rule (A-m8's inconsistency).
- Proposed: "…returns a verdict on 14,099 ($20.8\%$; no engineering failures), and in 231 of the 267 scenario runs ($86.5\%$) it speaks at least once. At the moment level, $55.3\%$ of candidate moments carry discriminative information between the candidates, and $20.8\%$ additionally lie within human support; at the run level, none of the 36 runs in which the monitor never speaks is silenced by readability alone --- in every such run there are readable moments, and what is missing there is comparable human behaviour for the situation. Abstention here is a statement about the reference rather than about the vehicle, and it is auditable as such."
- Evidence: RQ017 decision.md KC-C1 (37,520/67,861 = 55.2971%; 14,099 = 20.7763%; engineering failures 0) and KC-C2 (231/267 = 86.5169%; never-judgeable 36; of which mechanism-one-all-run 0 = 0.0000%).
- Constraint check: [NOFAB] every figure frozen; [PRESS] the 0-of-36 fact is a strength; two-layer (no reason-code names in main text — "discriminative information"/"human support" are already main-text vocabulary). Resolves R2-05, R2-24 (A-m8, B-m14, B-q3, D-m5-part, C-M3-part).

### Q04 — Scope the §2.3 abstention sentence
- Target: main.tex L337. Current: "It abstains on only a small fraction of supported cases, giving an auditable runtime monitor…"
- Problem: "supported cases" is self-contradictory shorthand for support-gate abstention and invites the overall-abstention misreading (D-m5).
- Proposed: "Where a reading is issued, it abstains for lack of human support on only a small fraction of moments ($5.08\%$ on the held-out natural-driving test fold; abstention on the real-vehicle benchmark is larger and is reported with its reasons in Methods), giving an auditable runtime monitor…"
- Evidence: RQ021 key_numbers gate.test (24,723/486,660 = 5.08%); RQ017 for the benchmark cross-ref.
- Constraint check: [NOFAB]; consistent with Fig. 4 caption's existing 5.08%. Resolves R2-05 facet (D-m5).

### Q05 — One-decimal flag rates in prose (keep exact counts)
- Target: three sites, none inside `\targetnum` [TARGETSAFE]:
  (i) L445: "$9.845\%$ of judgeable moments are flagged ($1{,}388$ of $14{,}099$; …)" -> "$9.8\%$ of judgeable moments are flagged ($1{,}388$ of $14{,}099$; …)".
  (ii) L486: "The automated systems' flag rate of $9.845\%$ sits at the native level" -> "$9.8\%$".
  (iii) L494: "($9.845\%$ vs \targetnum{$4.7\%$} of judgeable moments)" -> "($9.8\%$ vs \targetnum{$4.7\%$} of judgeable moments)" — edit ONLY the plain string; the two `\targetnum` tokens in L494–496 stay byte-identical.
  Also L482 "9.72\%" -> "9.7\%" ONLY IF the PI confirms; default: leave 9.72% (it is the apparatus-validity anchor quoted at 2 dp in Methods 4.6 — change both together or neither).
- Evidence: 1,388/14,099 = 9.8447% (RQ021-KC-C4); Fig. 6a already labels 1-dp values.
- Constraint check: [TARGETSAFE] verified against the call-site list; counts preserve exactness; matches frozen values at stated precision. Resolves R2-41 (D-m4).

### Q06 — Methods 4.4: promise only what is delivered; drop the word "guarantee" from the caveat
- Target A: main.tex L659–661. Current: "Conditional coverage across state groups, leave-one-source-out coverage, and a trajectory-wise simultaneous variant are reported as boundaries rather than assumed nominal: at the 90% level, refitting with one source held out covers…"
- Proposed A: "Conditional coverage across state groups and trajectory-wise simultaneous coverage are not established here and remain stated boundaries of the construction; leave-one-source-out coverage quantifies the source-shift boundary: at the 90% level, refitting with one source held out covers…" (LODO numbers unchanged).
- Target B: the earlier caveat sentence (Methods 4.4, "Moments within an interaction are dependent, so the guarantee is marginal over accepted moments, not a per-interaction or conditional guarantee.")
- Proposed B: "Moments within an interaction are dependent, so the coverage statement is marginal over accepted moments --- not a per-interaction, sequential or conditional guarantee --- and the achieved coverage is reported empirically (Fig. 4b)."
- Evidence: state-stratified coverage is not frozen (round-1 R1-19); LODO digits verified in key_numbers_e2.json; empirical coverage 0.8003/0.9028/0.9557 in key_numbers.
- Constraint check: accuracy-mandated (C-M2's promise-mismatch is TRUE); [PRESS] boundary statement, not concession. Resolves R2-04 core, softens R2-03 (C-M2, C-M1-part, A-M3-part, B-M2-part).

### Q07 — Re-scope the sequential-warning layer as unevaluated interface
- Target A: main.tex L675–676. Current: "Sequential warnings are tuned on an independent guard set, not read off the pointwise coverage."
- Proposed A: "A sequential warning layer (persistence over consecutive verdicts) belongs to the deployment interface: no result in this paper uses it, every reported verdict and flag rate is per-moment, and its operating characteristics (episode-level false-alarm rate, detection delay, warning duration) are not evaluated in this paper."
- Target B: Algorithm 2 line 15 (L748): "\STATE apply separately-calibrated persistence for warnings; map verdict to a planner-facing signal" -> "\STATE optionally map the verdict to a planner-facing signal through a persistence rule (deployment interface; not evaluated in this paper)".
- Evidence: no persistence parameters or operating characteristics exist in the knowledge layer (re-verified this round; round-1 R1-32). The claim "tuned on an independent guard set" is not auditable from the workspace and is removed rather than repeated [NOFAB].
- Constraint check: [PRESS] scope statement in the E9-decided spirit (the Methods scope note already declares real-time execution unevaluated); removes B-M1's strongest quote ("the planner-facing object is unspecified"). Resolves R2-10 text facet (B-M1, D-M8-part, C-m8, A-M8-part). B04/B13 carry the evaluation demands.

### Q08 — Restore the register's "measurable" qualifiers on the ablation claim
- Target A: L375: "yet reading the counterpart adds nothing online---resolves because…" -> "yet reading the counterpart adds no measurable sharpening online---resolves because…"
- Target B: L544: "---adds nothing beyond the current situation." -> "---adds no measurable value beyond the current situation."
- Evidence: register C3a/C3b word the accepted claim as "adds no measurable value" (paper-repo claims_register.md L20–21); frozen statistic −0.0002, p=0.863 (RQ009 decision L17–18) has no CI, so the unqualified "nothing" exceeds the evidence.
- Constraint check: register compliance = accuracy, not self-weakening; the "hidden intention" sentence is NOT touched (sanctioned narrative). Resolves R2-33 text facet (D-M7-part).

### Q09 — "Pre-registration" -> pre-specified frozen plan
- Target: L712. Current: "Official competition scores, harm labels and preference ratings were excluded as endpoints by pre-registration."
- Proposed: "Official competition scores, harm labels and preference ratings were excluded as endpoints in the analysis plan, which was specified and frozen before any outcome was examined."
- Evidence: internal freeze records exist; no citable public registration exists in the workspace (cannot be invented).
- Constraint check: [NOFAB]; D-m10 explicitly accepts this wording. Resolves R2-36 (D-m10, A-m16, C-q10-part).

### Q10 — Finish the "causal" rename
- Target: L612. Current: "…given the current observable situation $z_t$ (geometry, role, causal interaction progress, relative kinematics, online-computable risk proxies, and support/readability state)…"
- Edit: "causal interaction progress" -> "interaction progress" (the tuple's online-computability rider from round-1 P12 already governs; Eq. 3's main-text tuple says plain "progress").
- Resolves R2-51 facet (A-m5). Constraint check: naming only.

### Q11 — Align u_a's direction at first mention
- Target: L207–208. Current: "The concentration of the weights $\pi_{a,k}(t)$ gives a reliability measure $u_a(t)$."
- Proposed: "The spread of the weights $\pi_{a,k}(t)$ gives a reliability statistic $u_a(t)$: the more sharply the weights concentrate, the smaller $u_a(t)$, and a reading is used only where $u_a(t)$ is below a frozen threshold (Methods)."
- Evidence: Methods 4.1 (dispersion statistic; gate $u_a<u_0$; near-uniform rule) — internal consistency fix.
- Constraint check: [VOCAB] "readability" untouched. Resolves R2-15 (B-m4, D-m1).

### Q12 — Precision rider on "legal safety"
- Target: L114. Current: "…and can guarantee legal safety as an online layer around a planner [18–21]…"
- Proposed: "…and can guarantee a formally specified notion of legal safety --- within the assumptions of its model --- as an online layer around a planner [18–21]…"
- Evidence: cited works' own claim structure (Pek et al. 2020; RSS). Constraint check: precision, not concession; abstract untouched (that is E13). Resolves R2-12 intro facet (B-m2).

### Q13 — Fig. 5 caption: retire the asterisk convention, name the <3 s row, label intervals per-endpoint
- Target: Fig. 5 caption, three sentences.
  (i) L423–425: "All four thresholds were tested; the absence of a mark at $<3$\,s means its interval admits no difference, not that it went untested." -> "All four thresholds were tested and all four are displayed with their intervals; the $<3$\,s interval admits no difference."
  (ii) L435–438: "Error bars and asterisks denote $95\%$ confidence intervals from $1{,}000$ ($\mathbf{c}$: $2{,}000$) bootstrap resamples over the $175$ scenario runs, the unit of resampling throughout; the $472$ flagged moments cluster within $120$ scenario runs; an asterisk marks an interval excluding no difference." -> "Error bars and bracketed values are $95\%$ confidence intervals from $1{,}000$ ($\mathbf{c}$: $2{,}000$) bootstrap resamples over the $175$ scenario runs, the unit of resampling throughout; the $472$ flagged moments cluster within $120$ scenario runs. Intervals are per endpoint; the displayed battery is the frozen analysis battery and no threshold was selected post hoc." (fold the existing "All displayed thresholds…" sentence in, avoiding duplication)
- Rationale: NO asterisk glyph exists anywhere in the regenerated Fig. 5 (verified in the repo PDF/PNG); the stale convention caused A-m10's misreading. Fig. 6's caption keeps its asterisk sentence (that figure does use asterisks).
- Evidence: figure file itself; fig5_data.json for the interval values; B-m12's endpoint-wise labelling fallback.
- Constraint check: [PRESS] "admits no difference" is the sanctioned phrasing (RQ018-KC-C2 forbids citing <3 s as evidence); no number changes. Resolves R2-54, R2-22 label facet (A-m10, B-m12).

### Q14 — Fig. 4 caption: define the admissible range; add the rounding note
- Target: Fig. 4 caption L349–352 area.
  (i) After "as a percentage of the admissible range of the quantity." add: "The admissible range is the span of the candidate grid, $3\pi/4\approx2.36$\,rad; the parameterisation domain of Eq.~(1) is $[-\pi/2,\pi/2]$ and the seven candidates span $[-3\pi/8,3\pi/8]$."
  (ii) After "Conditioning narrows it by $21\%$, $20\%$ and $8\%$ at the three levels" add: "(reductions computed from unrounded widths --- $20.7\%$, $20.5\%$, $8.4\%$; bar labels are rounded to whole percent, so recomputing from them can differ by a point)".
- Evidence: RQ021 key_numbers — conditioned widths 1.4516/1.8651/2.1588, global 1.8314/2.3453/2.3562, span 2.3562; reductions 20.74/20.47/8.38%.
- Constraint check: numbers TO frozen values only; retires D-m3's recurring arithmetic; answers A-m4/B-m3/D-m2 definitional ambiguity. Resolves R2-40, R2-14 (grid-endpoint rationale stays out — not frozen; sensitivity → B07).

### Q15 — Disambiguate "scripted": scenario-scripted injections vs script-following counterpart control
- Target A: L692–693 (Methods 4.5). Current: "(a scripted-counterpart subset, $27$ runs, is excluded)".
- Proposed A: "(a subset of $27$ runs in which the counterpart follows a fixed script and cannot respond to the ego is excluded from counterpart-response analyses and reported only in isolation)".
- Target B: L703–704 (Methods 4.6). Current: "Licensed drivers drove the same scenario set in a real vehicle against the same scripted counterpart injections, giving a drivers-by-scenarios matrix (\targetnum{20} drivers $\times$ \targetnum{15} scenarios)…"
- Proposed B: "…against the same scenario-scripted counterpart injections (the same injection events; counterpart control as in the automated-systems benchmark), giving…" — the two `\targetnum` tokens stay byte-identical [TARGETSAFE].
- Target C: L477–478 (§2.5). Current: "…drove the same \targetnum{15} scenarios in a real vehicle, against the same scripted counterparts, and the unchanged monitor…" -> "…against the same scenario-scripted counterpart injections, and the unchanged monitor…" [TARGETSAFE].
- Evidence: RQ019 decision — excluded scripted group "1,075 rows / 27 cases, isolated reporting only; scripted counterparts may not react"; main analyses on the 175 non-scripted runs.
- Constraint check: [NOFAB] states only the frozen distinction; C7 prose wording (not digits) may be edited; digits untouched. Resolves R2-42 (A-M6, A-q11).

## Tier 2 — completeness (CLARIFY: surfacing frozen facts)

### Q16 — Print the frozen candidate-objective family (Methods 4.1)
- Target: Methods 4.1, after the sentence introducing candidate-trajectory generation (near L590–600, "for each agent we generate candidate trajectories over a grid of preferences").
- Proposed addition: "Each candidate trajectory for preference $\theta_k$ optimises the frozen implementation's utility $\cos\theta_k\cdot(\text{own-progress cost}) + \sin\theta_k\cdot(\text{interaction cost})$, so a more negative $\theta$ weights the joint term negatively --- the assertive direction; the full cost terms and solver configuration are released with the code."
- Evidence: `util = cos(ipv) x own cost + sin(ipv) x interaction cost` (agent.py:1193), quoted verbatim in RQ021 decision.md L12 and RQ018 decision.md L31 with "more negative IPV = more competitive".
- Constraint check: [NOFAB] exactly the frozen quote, no invented cost details ("released with the code" already appears for the config). Resolves R2-02 definitional facet (A-m3, A-q1, B-m5-part, C-M7-part) — the single cheapest answer to the construct-definition family.

### Q17 — Print RQ018-KC-C3: the flag is not a sign test
- Target: Methods 4.5, after the verdict-count sentence (L692–694 area), or §2.4 near L445.
- Proposed addition: "The criterion is the situation-conditioned range, not the sign of the reading: at the $90\%$ level the lower edge is negative in essentially every situation (median $-1.03$\,rad), $40.0\%$ of within-range moments also carry a negative reading, and of all negative-reading moments only $9.25\%$ are flagged."
- Evidence: RQ018-KC-C3 (lo_90 median −1.0264, 5–95 pct [−1.1781, −0.5984]; 5,090/12,711 = 40.04%; 519/5,609 = 9.25%). The KC's required qualification mandates the 40.04% counterexample whenever cited.
- Constraint check: [PRESS] pure strength (pre-empts the "assertive = negative IPV" misreading behind A-M1/B-M8 normative critique); numbers frozen. Resolves R2-06 facet; supports R2-01 rebuttal.

### Q18 — Print the placebo p-value and the margin-missingness balance
- Target: Methods 4.5 placebo sentence (L697 area: "…a placebo test that reassigns whole exposure trajectories across scenario runs distinguishes the real flag timing from permuted timings").
- Proposed: append "(empirical $p=0.0199$)". Then after the 472/11,669 sentence append: "The margin is undefined where the pair is not closing; the undefined fraction is similar in the two groups ($9.1\%$ of flagged vs $8.2\%$ of within-range moments), so this exclusion does not preferentially remove either group."
- Evidence: RQ018 decision (placebo exposure control p=0.0199; B4: 47/519 = 9.06% vs 1,042/12,711 = 8.20%).
- Constraint check: [NOFAB]; the case-label permutation (p=0.1493) is E11's disclosure decision — NOT printed here. Resolves R2-21, A-q9, part of D-M4.

### Q19 — Fig. 3 caption: offline-margin note + panel-c definition
- Target: Fig. 3 caption (L297–310 area).
  (i) After the panel-a band sentence add: "The bands are formed on the realised post-encroachment margin, an offline quantity used here for description only; the online monitor never reads it (Eq.~3)."
  (ii) For panel c add: "$R^2$ is the case-level variance explained on the held-out source by the full state description fitted on the other three; $0$ marks predicting the held-out source's mean."
- Evidence: RQ004 analyses bin by realised PET; leave-one-dataset-out full-case R² AV2 −0.195 / nuPlan −0.276 / Lyft +0.017 / Waymo +0.026 (RQ004_1 round2_3_synthesis.md L6); Eq. 3 rider already excludes observed post-encroachment measures.
- Constraint check: [NOFAB]; CIs for panel c are NOT added (not frozen → B09). Resolves R2-18, R2-47 (B-m8, A-m15).

### Q20 — Fig. 4 caption: define the 20.9%
- Target: Fig. 4 caption panel-c sentence (L356–359).
- Proposed: after "The situation explains $20.9\%$" insert "(defined as $1-\mathrm{SSE}/\mathrm{SST}$ of the conditional median against the test-fold mean, over all $486{,}660$ test moments; $21.1\%$ over the $461{,}937$ accepted moments)".
- Evidence: RQ021 key_numbers D2_contemporaneous_test_r2 (definition string, r2 0.20936, n 486,660) and D2_…gate_passing (0.21112, 461,937).
- Constraint check: [NOFAB] frozen definition verbatim; uncertainty not added (→ B09). Resolves R2-19 (B-m9, C-m5).

### Q21 — Denominator glossary (Methods 4.4)
- Target: Methods 4.4, before the split sentence (L645 area).
- Proposed addition: "Vocabulary, used consistently: an anchor row is one agent-moment eligible for estimation; a readable moment passes the discriminative-information rule; an accepted moment additionally lies within human support (these receive verdicts); on the benchmark we write candidate moment for an anchor row entering the two gates, and judgeable moment for one that passes both."
- Evidence: definitional restatement of the frozen pipeline stages (RQ021 gate accounting; RQ017 two-gate counts); no new numbers.
- Constraint check: two-layer safe (no codenames). Resolves R2-52 (C-m9, A-m12-part, D-m6-part).

### Q22 — Fig. 2 caption: n = case clusters
- Target: Fig. 2 caption panel-a CI sentence (L261). Current: "Points are means with $95\%$ confidence intervals over case clusters."
- Proposed: "Points are means with $95\%$ confidence intervals over case clusters; the $n$ column counts those case clusters."
- Evidence: RQ007_F1_source_data.csv n_cases = 4,743/4,605/4,701/8,130 exactly as displayed.
- Constraint check: [NOFAB]. Resolves R2-52 facet (A-m12).

### Q23 — Fig. 5 caption: the 469 flow
- Target: Fig. 5 caption panel-c sentence (L433–435 area).
- Proposed: after "panel \textbf{c} uses $n=469$ atypical and $n=10{,}483$ within-range moments" add "(the flagged moments falling inside the $175$ non-scripted runs with recorded counterpart logs)".
- Evidence: RQ019-KC-C1 (non-scripted main group, fixed 3 s, alpha=90: 469 / 10,483, 175 cases).
- Constraint check: [NOFAB]. Resolves R2-21 facet (B-m11).

### Q25 — Fig. 3a magnitude-context clause (optional but recommended)
- Target: §2.2 after the risk-reversal sentence (L285–290 area) or Fig. 3 caption.
- Proposed: "The role shifts are small against the width of the human range --- a few per cent of it; what conditioning contributes operationally is the range sharpening of Fig.~4 --- but their direction reverses with risk, which no scalar score can represent."
- Evidence: +0.058/−0.034 (RQ004-KC-PRIORITY) vs width 1.87 rad (RQ021); reductions Fig. 4.
- Constraint check: [PRESS] concedes magnitude while converting it into the conditioning argument. Resolves R2-55 (D-m7).

### Q26 — Counterpart-Unreadable parenthetical
- Target: Methods 4.4 reason-code list (L670–672) or Algorithm 2 abstain line.
- Proposed: after "\textsc{Counterpart-Unreadable}" in the Methods list add "(the counterpart's identity is not stable enough to anchor a reading, $c_a$ of Eq.~2)".
- Constraint check: consistent with Eq. 2's definition; naming only. Resolves R2-16 (B-m6).

### Q27 — Define "no engineering failures"
- Target: L393 (inside the Q03 rewrite). Add after "no engineering failures": "(no solver or pipeline failure on any candidate moment)".
- Evidence: RQ017-KC-C1 required qualification ("engineering failures 0" with denominator 67,861); L1 status/reason codes.
- Constraint check: [NOFAB]. Resolves R2-28 (B-m18). Execute together with Q03.

### Q28 — One-sentence case/row flow (Methods 4.3 or 4.4)
- Target: Methods 4.4, after the fold sentence (L649–651 area).
- Proposed addition: "The universes nest as follows: of the $38{,}228$ corpus interaction cases (Section 2.2), $26{,}828$ enter the frozen anchor ledger, contributing $4{,}497{,}368$ anchor rows (the readability accounting above); the human--human reference pool retains $2{,}442{,}625$ of those rows (Methods 4.3), which are exactly the train/guard/calibration/test folds ($974{,}984+499{,}893+481{,}088+486{,}660$)."
- Evidence: key_numbers.json `k2_case_key_filter_unique_cases` = 26,828 with joined rows 4,497,368; reference-pool rows 2,442,625; fold rows sum = 2,442,625 (verified exactly).
- Constraint check: [NOFAB] chain verified end-to-end this round; the K2 filter's internal criteria stay in the release docs (B06). Resolves R2-20 (B-m10, A-m6, D-m6).

## Tier 3 — figure text batch (FIX-FIGURE via the restyle pipeline; frozen data only, label/title changes and one marker restyle)

### Q24 — Four-figure text/label pass
- (i) Fig. 2 (`fig1_measurable`): panel-a title "The reading becomes readable only during a real interaction" -> "The reading is identified most sharply during the real interaction". Reason: the different-partner control is −0.043 (CI [−0.047,−0.039], same direction), so "only" overstates; the caption already says "weakens". Evidence: RQ007_F1_source_data.csv. [C-m3]
- (ii) Fig. 4 (`fig3_monitor`): panel-b title "Coverage stays at nominal" -> "Coverage within 0.6 pp of nominal". Evidence: +0.03/+0.28/+0.57 pp frozen. [C-m2]
- (iii) Fig. 5 (`fig5_consequence`): panel-b title "Emergency margins stay at or below the within-range rate" -> "Emergency margins are rarer at every supported threshold"; panel-d title "Counterpart braking stays at or below the within-range rate" -> "Counterpart braking is rarer at every threshold". Reason: the <3 s interval admits an increase (CI [−8.33, +3.15]), so panel b's "stay at or below" is an equivalence claim the evidence does not carry; panel d's three thresholds all exclude zero (−4.43/−3.40/−3.13). [B-m13, C-M6 figure facet]
- (iv) Fig. 1 (`fig0_concept`): enlarge/darken the conflict-zone marker in panel a and thin/offset the time labels so they clear the trajectories. No data change. [D-m12]
- Constraint check: restyle pipeline only, data files untouched; SYNTHETIC watermark not involved (figH untouched); titles screened against [PRESS] (each new title is a supported positive claim) and RQ018-KC-C2 (no <3 s evidence claim).
- After regeneration: recompile and re-verify the PDF embeds the new binaries (guard against a Q00 recurrence).

---

## Execution order and verification

1. Q00 (rebuild + verify embeds) — do first so every later visual check is against real state.
2. Tier 1 Q01–Q15 in order; compile after Q02, Q07, Q15.
3. Tier 2 Q16–Q28; compile once.
4. Tier 3 Q24 figure batch; final compile.
5. Final checks: `\targetnum` = 21 tokens, arguments byte-identical (diff must show no line containing `targetnum`); zero occurrences of the estimability family and of "passive"; no model codenames/audit vocabulary introduced in main text; figH files untouched; abstract byte-identical EXCEPT nothing (abstract untouched this round — E13 pending); pages render without the duplicated References heading; ED heading precedes ED1.

## Deliberately NOT planned (and why)

- Abstract sentences ("certified for collision safety", "makes testable") — E13, PI decision.
- "The instrument survives the move" and the three-rate presentation — E2 (open, C7-coupled).
- Fig. 6 uncertainty/precision/scenario-key changes — E10 (C7 regeneration spec).
- Ethics/consent/safety statement, Acknowledgements, Author Contributions — E5/E8 (facts absent from workspace).
- Benchmark edition/archival citation, scenario-template descriptions, 285→267/240→175 reasons — E6/B16 (organiser facts absent).
- Episode-level/cluster-conformal/baseline/oracle reanalyses, CI/MDE for ablations, persistence evaluation, over-yielding outcomes, missingness audits, alternative denominators — backlog B01–B17 and new B18–B21 (frozen-evidence rule).
- Removing the "hidden intention" Discussion sentence — sanctioned narrative (project objective); Q08's qualifier restoration is the accuracy fix.
