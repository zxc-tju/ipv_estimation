# Round-1 revision plan — executable items only (FIX-TEXT / FIX-FIGURE / CLARIFY)

Target manuscript: `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/2_PaperWriting/NMI---Online-Sociality-Verfication-for-Autonomous-Vehicle/main.tex` (line numbers @ commit 486382e).
Research-repo root for evidence paths: `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation/`.
Figure restyle pipeline: `.codex-fleet/paper-figure-upgrade/work/T5_style_harmonisation/restyle_*.py` (+ `T2_concept/make_fig_concept.py`); regenerated PDFs/PNGs are copied into the paper repo `figures/`.

Constraint legend (every item was screened against ALL hard constraints; the tags name the ones that bind):
[FROZEN] numbers only from frozen evidence, path cited · [TARGETNUM] the 14 `\targetnum` call sites stay byte-identical · [FIGH] figH SYNTHETIC watermark/red header untouched · [2LAYER] no model codenames / audit vocabulary in main text · [VOCAB] readability family; lower side = "more assertive"; abstention never neutral/compliant · [PRESS] precise repositioning, no self-weakening · [ORDER] no later-stage result retro-justifies an earlier gate · [NOFAB] nothing invented; gaps go to escalation.

Rule for all items: keep `main.tex` compilable after each edit; do not touch any `\targetnum{...}` argument; do not touch `figures/figH_human_arm_TARGET_SYNTHETIC.*`.

---

## Tier 1 — correctness blockers

### P01 — Replace the stale ">40 %" sharpening claim with the frozen 21/20/8 % numbers (two sites) — ⚠ execute after PI ratifies wording (ESCALATIONS E1)
- Resolves: R1-01 (A-M8, B-M8, C-M10iv/q14, D-M3/q1/P1); touches R1-23's 95 %-span point.
- **Site 1** §2.3, L320-323. Current: "the human range is much sharper than a single global range---more than a 40\% reduction in interval width at near-nominal coverage---while abstaining on only a small fraction of supported cases".
  Proposed: "the human range is sharper than a single global range---a fifth narrower at the 80\% and 90\% levels (21\% and 20\%; 8\% at the 95\% level, where the global range spans the entire admissible scale and can never flag anything)---at coverage within 0.6 percentage points of nominal, while abstaining on only a small fraction of supported cases".
- **Site 2** Methods 4.3, L576-578. Current: "the context-conditioned reference supplies essentially all of the achievable sharpening over the global reference (a width reduction above 40\% with a Winkler-score improvement above 35\% at near-nominal coverage)."
  Proposed: "the context-conditioned reference narrows the global reference by 21\%, 20\% and 8\% at the 80\%, 90\% and 95\% levels at coverage within 0.6 percentage points of nominal (mean widths 1.452, 1.865 and 2.159 rad against 1.831, 2.345 and 2.356 rad for the global range)."
  Note: the ">35\% Winkler" clause and "essentially all of the achievable sharpening" (an oracle-ceiling comparison from the superseded configuration) have **no frozen counterpart under the current envelope** and are dropped, not restated.
- Evidence: `reports/studies/RQ021_contemporaneous_envelope/RQ021_1_contemporaneous_envelope_20260805T160425Z_43b4bff/key_numbers.json` → `human_only_envelope.metrics.{80,90,95}.mean_width` = 1.4516/1.8651/2.1588 and `…circularity_diagnostics.marginal_envelopes.ipv_log.metrics.{80,90,95}.mean_width` = 1.8314/2.3453/2.3562; coverage +0.03/+0.28/+0.57 pp; paper repo `claims_register.md` C3 mandates exactly these numbers and bans mixing in the old RQ009 −42.3 %/−35.6 %.
- Constraints: [FROZEN] [PRESS] (accuracy repair; the strongest true facts — global unusable at 90/95 %, calibration within 0.6 pp — lead) [2LAYER] (no M0/M2 codenames in the replacement).

### P02 — Make Methods 4.4 + Algorithm 1 describe the frozen calibration actually run
- Resolves: R1-02 (C-M1/q1-q3, B-M2/q4, A-M4); prerequisite for honest answers to C-q3.
- Target: Methods 4.4 L598-601. Current: "The primary estimand is case-balanced pointwise marginal coverage at prespecified progress anchors $\tau\in\{0.2,0.4,0.6,0.8\}$, with at most one calibration score per case per anchor so that long interactions do not dominate; the readability gate is frozen…".
  Proposed: "The estimand is pointwise marginal coverage over accepted moments: the conformal radius $c_\alpha$ is the $\lceil(n+1)(1-\alpha)\rceil$-th smallest nonconformity score over all accepted calibration-fold moments ($n=455{,}723$), and coverage is evaluated over all accepted test-fold moments ($461{,}937$). Splits are made by whole scenes under a split frozen before this study, so no scene contributes to more than one fold; the readability gate is frozen on training/guard data and applied identically to calibration and test. Moments within an interaction are dependent, so the guarantee is marginal over accepted moments, not a per-interaction or conditional guarantee."
- Also: Algorithm 1 line 7 (L650) "sample $\le 1$ score per case per anchor $\tau\in\mathcal{A}$" → "score every accepted calibration window"; drop `progress anchors $\mathcal{A}$` from the Require line (L642) if no other use remains.
- Evidence: same key_numbers.json → `human_only_envelope.conformal_radii` (calibration_n 455,723; ranks 364,580/410,152/432,938), `gate.test` (461,937 accepted of 486,660), `join_health` + `k2_ledger.invalid_rq007_split_rows: 0` (whole-scene frozen split inherited from the RQ007 split).
- Constraints: [FROZEN] [PRESS] (the honest marginal-claim statement is a scope statement, not a concession) [2LAYER].

### P03 — Rewrite the Methods 4.5 protocol sentence to the frozen design (also kills "the the onset")
- Resolves: R1-03 (A-M8/q6, B-m13, C-M5/q8, D-m6), R1-32 (persistence wording).
- Target: Methods 4.5 L617-620. Current: "…we order the interaction-opportunity onset, the
the onset of a readable segment and the first persistent deviation, and analyse only consequences after the deviation onset, controlling for scenario and initial risk."
  Proposed: "…verdicts are issued per moment; outcome windows begin at the flagged moment, so the trajectory samples that produce a verdict (the one-second window ending at that moment) precede every outcome sample. The ego-side outcome is the minimum time-to-collision to the counterpart over the post-verdict window; the counterpart-side outcomes are read over a fixed three-second window from the same moment, from the other vehicle's own logged control record. Comparisons are made within the frozen situation cells, inference resamples scenario runs (the unit of clustering throughout), and a placebo test that reassigns whole exposure trajectories across scenario runs distinguishes the real flag timing from permuted timings."
- Evidence: `reports/knowledge/RQ018_abnormal_ipv_degradation/decision.md` (post-verdict future-min-TTC battery; unit = moment; case clustering), `reports/knowledge/RQ019_counterpart_burden/decision.md` (fixed 3 s window; raw control logs; B6), `rq018_rerun/frame_level_results.json` (`context_fixed_effects: true`), `rq018_rerun/negative_controls.json` (case-label permutation; placebo-exposure pass p=0.0199 recorded in RQ018-KC-C1).
- Constraints: [FROZEN] [NOFAB] (drops "initial risk" and "first persistent deviation", which have no frozen counterpart) [2LAYER] (no RQ numbers in text).

### P04 — Repair the emergency-threshold sentences in §2.4 and §2.6 (the <3 s bin cannot carry evidence)
- Resolves: R1-04, R1-35 (main-text part) (C-M6, D-M6, B-M5 part); honours RQ018-KC-C2's required qualification.
- Target 1: §2.4 L422-426. Current: "At every threshold that would mark an emergency…atypical moments are roughly half as frequent as typical ones or fewer, not more frequent, and the lower quartile of the ego's margin is nearly unchanged between them."
  Proposed: "At the emergency thresholds---a closing margin short enough to demand immediate action, or braking hard enough to count as evasive---flagged moments are one-third to one-half as frequent as typical ones at every threshold whose interval excludes no difference (margins under 1, 1.5 and 2\,s; braking beyond $-2$, $-3$ and $-4$\,m\,s$^{-2}$); at the widest margin tested (under 3\,s) the interval admits no difference. The lower quartile of the ego's margin is nearly unchanged between the groups."
- Target 2: §2.6 L490-492. Current: "at every emergency threshold the atypical moments are no more frequent than the typical ones, so a monitor built on those thresholds would report nothing". Proposed: "at no emergency threshold are the flagged moments more frequent than the typical ones---at the supported thresholds they are one-third to one-half as frequent---so a monitor built on those thresholds would report nothing".
- Evidence: `fig5_data.json` in the RQ021 run dir → `ego_danger_shares` (<1 s 0.85 vs 2.28 %; <1.5 s 2.12 vs 5.85 %; <2 s 4.66 vs 8.84 %; <3 s 12.71 vs 15.85 %, CI [−8.33, +3.15] pp crosses 0) and `counterpart_braking_thresholds` (all three CIs exclude 0); RQ018-KC-C2 ("TTC<3 s … 不得作为证据引用").
- Constraints: [FROZEN] [PRESS] (precision, not concession: the supported claim — fewer emergencies at every supported threshold — stays and is stated crisply) [VOCAB].

### P05 — Remove the `claims_register.md` citation from Methods
- Resolves: R1-05 (A-M8, B-m14, D-M7iv).
- Target: L533-536. Current: "…and the two validation protocols. Per-study evidence status is tracked in \texttt{claims\_register.md}."
  Proposed: delete the last sentence; end the scope note at "…the two validation protocols." (Hard constraint: remove, do not expand into reader-facing audit vocabulary.)
- Evidence: n/a (internal artefact removal). Constraints: [2LAYER] (this IS the two-layer rule).

### P06 — Stop promising a counterfactual-injection demonstration that is not in the manuscript
- Resolves: R1-06 (D-M7ii, D-q10).
- Target: L676-681 "Planner interface (demonstration)." Current: "A counterfactual-injection demonstration is reported in Extended Data as an interface demonstration, not an external-validity result; a planner fallback is a control action and is never labelled \textsc{Within-Norm}."
  Proposed: "\noindent\textbf{Planner interface.} The verdicts map to planner actions---audit, soft cost or warning, a fallback candidate under sustained competitive deviation at high physical risk, a hesitation/efficiency warning, and a degraded unverified mode on abstention. This mapping is an interface specification, not an external-validity result; a planner fallback is a control action and is never labelled \textsc{Within-Norm}."
  (If the PI holds an actual demonstration artefact, restore the pointer instead — see ESCALATIONS E7 note.)
- Evidence: absence verified in main.tex (Extended Data = ED1, ED2 only) and in the frozen evidence tree. Constraints: [NOFAB] [PRESS] (a false promise is worse than a specification statement).

### P07 — Delete the leftover instruction in reference [53] (onsite)
- Resolves: R1-07 (A-M8, B-m15, C-m9, D-M5/m7) — the certain half; the edition/year fact is ESCALATIONS E6.
- Target: `bibliography/biblio.bib` L43. Current: `note         = {Verify the specific challenge edition/year for the data used}`.
  Proposed: remove the `note` line (and the trailing comma on the preceding line as needed to keep BibTeX valid).
- Constraints: [NOFAB] (do not substitute a guessed edition; E6 carries that decision).

## Tier 2 — consistency

### P08 — Truncate the reference [35] author list
- Resolves: R1-08 (D-m7).
- Target: `bibliography/biblio.bib` `@article{bethlehem2022brain}` author field. Proposed: keep the first author + "and others" (`author = {Bethlehem, R. A. I. and Seidlitz, J. and White, S. R. and others}`), which `unsrt` renders as "et al."
- Constraints: none binding (bibliographic form only).

### P09 — Fix register slips: "an readability indicator"; the u_a audit-note sentence
- Resolves: R1-09 (A-m1, C-m10, D-m1/m6), part of R1-47.
- Target 1: L229 "an \emph{readability} indicator" → "a \emph{readability} indicator".
- Target 2: L543-544. Current: "This is inverse planning, not a single joint maximum-likelihood optimum, and we do not call $u_a(t)$ a standard deviation until its definition in the released time series is audited."
  Proposed: "This is inverse planning, not a single joint maximum-likelihood optimum; $u_a(t)$ is a dispersion statistic of the normalised candidate weights, not a standard deviation, and the reliability gate $u_a(t)<u_0$ of Eq.~(2) is operationalised by the frozen rule below: the reading is discarded as near-uniform when the largest normalised weight falls below $0.20$ (against $1/7\approx0.14$ for exactly uniform weights) or the candidate likelihoods are exactly tied."
  Then shorten the now-duplicated near-uniform sentence at L556-558 to avoid saying the 0.20 rule twice.
- Evidence: mechanism-1 frozen rules (NEAR_UNIFORM = max weight <0.20; NO_IPV_EFFECT = exact tie): `reports/knowledge/RQ017_onsite_mechanism_one/decision.md` terminology + `.codex-fleet/rq022-matched-scenario/work/T1_target_figure/DATA_INTERFACE.md` §2.1; K2 reason codes in RQ021 key_numbers `k2_ledger.reason_code_counts`.
- Constraints: [FROZEN] [VOCAB] (readability wording kept; no estimability family).

### P10 — Fix the overflowing validation-path display
- Resolves: R1-10 (D-m5).
- Target: L125-128, the `\[ ... \]` chain "human reference range → conditional social atypicality → human-dispreferred behaviour → interaction-harmful behaviour."
  Proposed: break into two centred lines (e.g. an `aligned` or `array` environment with the arrow continuing the second line: line 1 "human reference range → conditional social atypicality →"; line 2 "human-dispreferred behaviour → interaction-harmful behaviour."), or set the whole chain in `\small` text inside a single line that provably fits the measure. Verify in the compiled PDF that nothing is clipped.
- Constraints: layout only; keep compilable.

### P11 — Reframe §2.6's opening away from the "beyond safety" contest (claims register C6)
- Resolves: R1-11 (A-M3, B-M5, D-M2 — the wording half; the experiment half is BACKLOG B02).
- Target: L488-494. Current opening: "Finally, social monitoring is worthwhile only if it carries information beyond conventional safety and kinematic checks. The previous section sharpens what that question is about."
  Proposed: "Finally, what does a social flag register that a conventional safety check does not? The previous section sharpens the answer." (Keep the rest; its dissociation logic — thresholds silent while the interaction tightens — is register-sanctioned. The closing sentence "Social atypicality and conventional safety therefore register different things" stays.)
- Evidence: paper repo `claims_register.md` C6 (NOT_CLAIMED; dissociation-only wording mandated; "registers something conventional checks do not", never "adds value beyond safety").
- Constraints: [PRESS] (removes a contest the paper cannot win and was ruled not to enter) [2LAYER].

### P12 — Rename "causal risk proxies"
- Resolves: R1-12 (C-m4).
- Target: Eq. 3 (L314) "causal risk proxies" and Methods 4.3 (L573-574) "online causal risk proxies".
  Proposed: "online-computable risk proxies" in both places, with one clause at first use: "risk proxies computable from information available at time $t$ (no observed post-encroachment measure enters online)". The Fig. 4 caption already states the exclusion.
- Evidence: RQ021 key_numbers `feature_contract.numeric_context` (closing_ttc_anchor, apet_online_proxy) + exclusion list in Methods 4.3.
- Constraints: [FROZEN] [VOCAB].

### P13 — Fig. 5 caption: colour word, definitions, absolute values, true denominators, episode count
- Resolves: R1-13 (B-m9, C-m3/m5, A-m7/m9, D-m2, D-M6-part), R1-44 (display note), R1-48 (keeps per-moment phrasing).
- Target: caption L392-413. Edits:
  (a) "Throughout, blue marks moments" → "Throughout, teal marks moments".
  (b) After the panel-a sentence add: "The margin is the minimum time-to-collision between the ego and its counterpart over the window following the verdict; very large values arise where the paths are already diverging."
  (c) Panel c: add absolute medians: "(medians 2.74 vs 1.33\,km\,h$^{-1}$ for the speed reduction and 5.53 vs 2.93\,km\,h$^{-1}$ for the speed range, read over a fixed three-second window)".
  (d) Panel d: replace "panels \textbf{c},\textbf{d} use $n=469$ and $n=10{,}483$" with "panel \textbf{c} uses $n=469$ atypical and $n=10{,}483$ within-range moments; panel \textbf{d} aggregates the acceleration records within those windows ($13{,}800$ vs $310{,}246$ records)".
  (e) After "the unit of resampling throughout" add: "; the $472$ flagged moments cluster within $120$ scenario runs".
  (f) Add: "All displayed thresholds are the frozen analysis battery; none was selected post hoc."
- Evidence: figure PNG (teal); `fig5_data.json` (`counterpart_ratio_bootstrap` medians; `counterpart_braking_thresholds` record denominators 13,800/310,246; `ego_band_counts`); RQ018 restatement (lower side 519 → 120 cases; defined-TTC subset 472/11,669); RQ019-E1/E2 (3 s window, km/h logs); RQ022 DATA_INTERFACE ("no threshold scans", frozen battery).
- Constraints: [FROZEN] [VOCAB] [PRESS].

### P14 — FIX-FIGURE: suppress the numeric ratio labels on Fig. 5c's unsupported rows
- Resolves: R1-14 (D-m8; aligned with RQ019's rejection of any steering claim).
- Target: `restyle_fig5_consequence.py` (T5 pipeline) → regenerate `figures/fig5_consequence.pdf/.png`: for `total_heading_change_deg` and `max_abs_yaw_rate_dps` print only "not supported" (grey), no "2.35×"/"2.04×" numerals; keep the grey interval whiskers so the display remains complete. No data change.
- Evidence: `fig5_data.json` `excludes_one: false` for both rows; RQ019 decision "任何转向类主张 … 不成立".
- Constraints: [FROZEN] (uses existing data only) [PRESS] (removes quotable unsupported numbers).

### P15 — FIX-FIGURE: route Fig. 1b's "no" branch only into "no reading"
- Resolves: R1-15 (B-m3); tightens the abstention semantics the paper's own vocabulary rules demand.
- Target: `make_fig_concept.py` / `restyle_fig0_concept.py` → regenerate `figures/fig0_concept.pdf/.png`: the mechanism-1 "no" arrow terminates at the "no reading" box alone; mechanism-2's two outcomes (inside / outside·atypical) get their own arrows; optionally annotate the outside box "either side" to reflect the two-sided rule. No data involved.
- Constraints: [VOCAB] (abstention must never look like a verdict) [FIGH] untouched (different figure).

### P16 — Algorithm 2: emit Counterpart-Unreadable where its component fails
- Resolves: R1-16 (B-m12).
- Target: L666 `\IF{$g_i(t)=0$} \RETURN \textsc{Abstain}(\textsc{Ego-Unreadable}) \ENDIF`.
  Proposed: "\IF{$g_i(t)=0$} \RETURN \textsc{Abstain}(\textsc{Ego-Unreadable} if $u_i$ or $q_i$ failed; \textsc{Counterpart-Unreadable} if $c_i$ failed) \ENDIF" — i.e. the reason code names the failing component of Eq.~(2) ($c_a$ is the counterpart-identity stability term).
- Evidence: Eq. 2 component list (L230-235); Methods 4.4 reason-code list (L606-610). Constraints: [2LAYER] (Methods-layer only).

## Tier 3 — completeness (CLARIFY: surfacing frozen numbers reviewers could not find)

### P17 — Benchmark accounting sentence in §2.4 + Methods 4.5
- Resolves: R1-17, R1-21 (D-M5/q2, A-m12, C-M11/q10, B-q13; RQ017-KC-C1's own required qualification).
- Target: §2.4 L373-378 ("The monitor is, first, able to speak…") and Methods 4.5 L613-616. Add (main text, plain words): "The benchmark replays 15 scenario templates against 19 independent driving systems (267 scenario runs). Across its 67,861 candidate moments the monitor returns a verdict on 14,099 (about one in five; no engineering failures), and in 231 of the 267 scenario runs (nearly nine in ten) it speaks at least once." In Methods 4.5 add the universe ladder: "…of the 14,099 judgeable moments at the 90\% level, 519 lie below the range, 869 above and 12,711 inside; consequence analyses use the 175 non-scripted scenario runs with recorded outcomes (a scripted-counterpart subset, 27 runs, is excluded), and the ego-margin panels use the moments with a defined post-verdict margin (472 flagged, 11,669 within-range)."
  Also replace "scenarios" by "scenario runs" where the 175/267 unit is meant (Fig. 5 caption "over the 175 scenarios" → "over the 175 scenario runs"; §2.4 L773 "nine of every ten scenarios" → covered by the new sentence).
- Evidence: RQ017-KC-C1/C2 (67,861; 55.30 %; 14,099 = 20.78 %; 267/231; failures 0); RQ018-E1 (231 cases, 19 teams; 519/120); RQ021-KC-C4 (519/869/12,711); RQ019-E1+B4 (175 non-scripted; scripted 27); `av_reference_values.json` (15 templates). 
- Constraints: [FROZEN] [TARGETNUM] (the sentence must NOT touch the `\targetnum{15}` sites in §2.5/4.6 — the 15 templates here are stated for the AV benchmark from the frozen AV-side file) [2LAYER] [PRESS].

### P18 — State the flag-rate denominators and the both-sides reading in §2.4
- Resolves: R1-21 (D-m3, A-m12).
- Target: L417 "Such moments are also rare: fewer than one judgeable moment in ten is flagged."
  Proposed: "Such moments are also rare: counting both sides, 9.845\% of judgeable moments are flagged (1,388 of 14,099; 519 on the assertive side, 869 on the over-yielding side)."
- Evidence: RQ021-KC-C4 (`reports/knowledge/RQ021_contemporaneous_envelope/decision.md`).
- Constraints: [FROZEN] [VOCAB] (lower side described as assertive).

### P19 — Reference-population provenance sentence (human–human pairs only)
- Resolves: R1-18 (D-M5/q3, D-m11).
- Target: Methods 4.3 (after L574 "…support/readability state"). Add: "The reference pool contains human--human vehicle pairs only (2,442,625 anchor rows): any interaction in which either agent is an automated data-collection vehicle is excluded from the reference, so machine behaviour cannot contaminate the human range. The corpora are fleet-collected in specific cities in the United States and Singapore; the reference is a sample of natural driving there, not of all human driving." (Second sentence satisfies D-m11's population-limits request; phrase as scope, not weakness.)
- Evidence: RQ021 key_numbers `source_reference_pool` (filter `agent_type_pair == HV;HV`, 2,442,625 rows) + `negative_controls.agent_type_pair_back_in_support_gate: EXPECTED_FAIL`; LODO key_numbers_e2 `pool_counts.agent_type_pair_counts/av_included_counts`.
- Constraints: [FROZEN] [PRESS] (scope statement).

### P20 — Methods 4.4: frozen split sizes, gate parameters, abstention accounting, MSE units
- Resolves: R1-19, R1-22, R1-47 (A-M4/m13/q8, B-M1/m1/q3, C-M1/M4/m7, D-M7v).
- Target: Methods 4.2/4.4. Add one compact paragraph (Methods layer): fold sizes train/guard/calibration/test = 974,984/499,893/481,088/486,660 human anchor rows (26,828 interaction cases; whole-scene split frozen in advance); support gate = mean Euclidean distance to the k=25 nearest training anchors after standardisation and one-hot situation categories, abstain above the 95th percentile of guard distances (1.081), with categorical support requiring ≥50 training anchors and ≥10 cases; support abstention on the test fold 5.08\% (24,723/486,660) with zero categorical failures; mechanism-level reasons on the natural corpus: near-uniform weights 1,275,480, exact ties 17,416, solver failures 1,826, readable 3,202,646 (of 4,497,368 anchor rows). In 4.1, after the likelihood: "MSE$_k$ is the mean squared Euclidean distance (m$^2$) between observed and candidate positions over the window, so $\sigma$ is in metres."
  Also (same item) repair the reporting-promise sentence L604-606: replace "…case-balanced unconditional performance, so that filtering is never hidden" with "…and the abstention rate with its reason distribution, so that filtering is never hidden" (drop "case-balanced unconditional performance", which has no frozen number).
- Evidence: RQ021 key_numbers `human_only_envelope.row_counts`, `gate.fit.params/definition`, `gate.test`, `k2_ledger.reason_code_counts`, `join_health.k2_case_key_filter_unique_cases`; `src/sociality_estimation/core/reliability_logdomain.py` (`candidate_mse`, `weights_from_mse`).
- Constraints: [FROZEN] [2LAYER] (parameters live in Methods) [NOFAB] (quantile-model family stays unstated → BACKLOG B06).

### P21 — Surface the leave-one-source-out coverage numbers (Methods or Extended Data table)
- Resolves: R1-20, R1-36-part (B-q10, C-M2/q4, D-q8, A-M4).
- Target: Methods 4.4 after "leave-one-source-out coverage … reported as boundaries" (L602-604): attach the numbers instead of only promising them: "at the 90\% level, refitting with one source held out covers the held-out source at 0.743 (Waymo, 143,380/193,096), 0.990 (nuPlan, 149,068/150,587), 0.750 (Lyft, 68,270/91,069) and 0.900 (Argoverse-2, 11,315/12,576, with 44.3\% held-out abstention — the support gate intercepting unsupported situations); transfer across sources is therefore a bounded property of source heterogeneity, and coverage and abstention must be read together."
- Evidence: RQ021 decision B2 addendum + `RQ021_2_lodo_transfer_20260807T114305Z_0c4d280/key_numbers_e2.json`; register C3-bnd sanctions the surfacing.
- Constraints: [FROZEN] [ORDER] (the boundary statement pre-exists the benchmark stage; no retro-tuning) [PRESS] (frame as the abstention mechanism working, per the register's own reading).

### P22 — Fig. 4 caption: attribution softening + width spread + accepted counts
- Resolves: R1-23 (C-M10ii/m8, A-m6, B-m8-part).
- Target: caption L340-348. Edits: (a) "the remaining $79.1\%$ is carried by the interaction itself and is not recoverable from the situation description" → "the remaining $79.1\%$ is not recoverable from the situation description"; (b) after the width sentence add "(at the 90\% level the conditioned width has mean $1.87$ rad, 5th--95th percentiles $1.35$--$2.28$ rad)"; (c) after "Panels a,b use $n=461{,}937$ evaluated moments" add "(the accepted moments; identical at all three levels)"; (d) add "Verdicts elsewhere in this paper are issued at the $90\%$ level."
- Evidence: RQ021 key_numbers `metrics.90.{mean_width,width_p05,width_p95}` (1.8651/1.3521/2.2792), `abstained_n` identical across levels; the 90 % operating level is the level of every flag rate quoted (RQ021-KC-C4, Fig. 6).
- Constraints: [FROZEN] [PRESS] (drops an over-attribution; adds precision).

### P23 — Fig. 2 caption + Methods: name the plotted statistic; state what is and is not shown
- Resolves: R1-24 (A-m2/m4, B-m5/m2).
- Target: Fig. 2 caption L247-265 + Methods 4.2. Edits: (a) caption axis sentence: "How sharply the reading is identified" → add "(the change in the concentration of the candidate weights — the reliability quantity of Eq.~2 — relative to control pairings; negative = more sharply identified)". (b) Methods 4.2 add: "The event-level summary is the mean current IPV over the valid interaction segment; alternative rules (all valid frames; readability-weighted) differ by 0.26 rad on average and flip the episode sign in 7--22\% of episodes (Fig. 2c), which is why the monitor never uses an episode summary online and every reading is per-moment." (c) In the caption, keep "development split; confirmatory split held sealed" unchanged (deliberate, register C0E).
- Evidence: RQ007-KC-C1/C2/C3 (`reports/knowledge/RQ007_interaction_conditioned_ipv_estimability/decision.md`); register C0E.
- Constraints: [FROZEN] [VOCAB] (concentration/reliability wording, no banned family) [NOFAB] (panel-b/c CIs not invented → BACKLOG B09).

### P24 — Fig. 3 caption: reconcile the two count universes
- Resolves: R1-25 (D-m4, B-m7, C-M11-part).
- Target: caption L289-303. Add after panel-b sentence: "Panel \textbf{b} uses the cases entering the episode-level analysis for each source (Waymo 23,211; Lyft 5,098; Argoverse-2 2,404; nuPlan 7,498 of the 23,218/5,105/2,406/7,499 interaction cases)." Add after panel-a sentence: "The three bands cover the 37,495 cases with a defined post-encroachment margin."
- Evidence: RQ004 cohort table (`…/agent_G_avhv/round3_avhv_scalar_failure_results.json` `cohort_distribution.full_valid_cases`); RQ004-KC-PRIORITY (bin deltas + n's as printed); arithmetic on printed bin n's (9,799+22,833+4,863).
- Constraints: [FROZEN] (the 37,495 is a sum of already-printed frozen n's; the exclusion reason is definitional — bins partition defined PET values).

### P25 — Scope sentence in the Discussion
- Resolves: R1-26, R1-38-part (B-m18, A-M9).
- Target: Discussion, end of the boundaries paragraph (after L525 "…created by, not hidden by, the present monitor."). Add: "The evidence in this paper covers pairwise vehicle--vehicle interactions at mapped conflict points; interactions with cyclists and pedestrians, and multi-agent scenes, are outside the present reference and are the natural next extension of the same construction."
- Evidence: reference-pool filter (HV;HV; P19) + benchmark composition (vehicle counterparts; RQ019 terminology).
- Constraints: [PRESS] (scope statement with a forward direction, not a deficiency list).

### P26 — Methods nomenclature mapping + upper-side counts
- Resolves: R1-27 (A-m11, B-m4).
- Target: Methods 4.4 after the verdict list (L609-610). Add: "In the main text, \textsc{Competitive-Deviation} is written `outside the range on the assertive side' and \textsc{Over-Yielding} `outside on the over-yielding side'; both sides are tested at every judgeable moment." Also harmonise captions (Fig. 1c and ED1) so each outside-range mention names its side the same way ("above the range (over-yielding side)" / "below the range (more assertive)").
- Evidence: verdict taxonomy L606-610; side conventions θ>0 cooperative (L202-203); RQ021-KC-C4 side counts (P18 carries the numbers).
- Constraints: [VOCAB] (lower side is "more assertive"; never "passive") [2LAYER] (SC names stay in Methods; captions use plain words).

### P27 — Remove the empty back-matter scaffolding
- Resolves: R1-28-part (D-M7iii).
- Target: L751-754. Delete the empty "Additional Information / Supplementary Information / Supplementary Tables / Supplementary Figures" headers (nothing exists behind them; the surfaced numbers now live in Methods/ED per P17–P24). Keep "Competing Interests" (has content).
- Constraints: none binding (removing empty promises; Acknowledgements/Author Contributions stay as-is pending E8).

### P28 — Ground the ablation null with its frozen statistic
- Resolves: R1-29 (C-M10i/q13).
- Target: Methods 4.3 L578-583, after "…add no measurable value". Add: "(the counterpart-IPV channel is statistically indistinguishable from removing the IPV input given the situation: paired 90\% interval-score difference $-0.0002$, case-clustered $p=0.86$)".
- Evidence: RQ009 decision, Internal Ablations section (paired 90 % Winkler diff −0.0002, case-cluster p=0.863). Note: use "interval score (Winkler)" only here in Methods; fine under two-layer.
- Constraints: [FROZEN] [2LAYER] (Methods layer; no M3/M4 codenames).

### P29 — State the estimator-fit boundary across populations (measurement-invariance hedge)
- Resolves: R1-30-part (C-M9/q12).
- Target: Methods 4.5 or 4.4 boundary sentence. Add: "The fraction of moments whose reading carries discriminative information differs between the naturalistic corpora and the benchmark (70.3\% vs 55.3\%); we report this as an observed difference without attributing a cause, and the readability gate treats both populations identically." 
- Evidence: RQ017 rejected-claims table (OnSite OK 55.2971 % vs InterHub 70.3001 %; explanation explicitly not licensed).
- Constraints: [FROZEN] [NOFAB] (no explanation offered) [PRESS] (transparency framed as identical treatment).

### P30 — Surface the placebo-exposure control in §2.4 (one clause)
- Resolves: R1-31-part (A-M2, B-M5, C-M5 — the part answerable now).
- Target: §2.4 L380-388, end of the consequence paragraph. Add: "; the association survives a placebo test that reassigns whole flag sequences across scenario runs" (Methods 4.5 carries the mechanics per P03).
- Evidence: RQ018-KC-C1 (placebo exposure control passes, empirical p=0.0199; `rq018_rerun/negative_controls.json`).
- Constraints: [FROZEN] [2LAYER] (plain words in main text) — descriptive-association framing retained ("Associations are descriptive" stays in the caption).

---

## Execution notes

1. Order within a tier is the execution order; P01 waits for E1 ratification, everything else is unblocked.
2. After the text edits: recompile; verify the 14 `\targetnum` call sites byte-identical — baseline @486382e: `grep -o 'targetnum' main.tex | wc -l` = 21 (14 call sites + 1 macro definition + 6 comment mentions); the count and every call-site argument must be unchanged. Verify no `estimability` family token anywhere; verify figH files untouched.
3. After P14/P15: regenerate only `fig5_consequence` and `fig0_concept` outputs via the T5 pipeline; do not touch `figH_human_arm_TARGET_SYNTHETIC.*`.
4. Handoff: append an entry to `reports/knowledge/PAPER001_online_sociality_verification_manuscript/imported_from_paper_repo_20260620/agent_handoff.md` after the edits land (per repo contract).
5. structure.md L90 still carries ">40% width reduction" — flag to PI with E1 so the planning layer is synced in the same pass (do not edit as part of this plan without PI instruction).
