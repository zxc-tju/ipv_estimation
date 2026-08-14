# Escalations — round-2 status of open items + new decisions

Context for a reader who has not followed the rounds: this manuscript runs simulated NMI review rounds; items the revision layer may not decide (headline wording, claim boundaries, C7 human-arm material, facts absent from the workspace) go to the PI as escalations. Round 1 opened E1–E12; the PI decided E1, E3, E4, E9, E12 (all executed in commit 41b819a) and left E2, E5, E6, E7, E8, E10, E11 open. Round 2 re-raises are mapped onto those ids below — no re-litigation, only new facets. One genuinely new escalation this round: E13.

---

## Still OPEN from round 1 — status and round-2 re-raises

### E2 — Three-flag-rate story, "flagged about twice as often", transfer language (C7-coupled) — OPEN
- Round-2 re-raises: A-M4/q10/q12/q14; B-M6/q8; C-M3/M4/q6/q7, C-m6; D-M6/q7, plus D-M3's demand to hedge the 2.1x to the tested system population.
- New facets to fold in when deciding: (i) the reviewers now explicitly demand a PRE-STATED two-sided acceptance band for the on-course human outside rate (insensitivity is currently narrated as success — the asymmetry objection appears independently in C, D, A); (ii) a formal cross-arm interaction/equivalence analysis for the "shared consequence signature" (C-m6); (iii) an explicit reconciliation sentence between the LOSO bounds (0.743/0.750) and carrying the reference to the course; (iv) estimand prespecification for the contrast (per-frame vs per-run vs per-scenario — B20); (v) D-M3's population hedge ("the automated systems entered in this benchmark", not "automated driving"). §2.5's "the instrument survives the move" (main.tex L481) is the sentence carrying most of the load.
- Recommendation unchanged from round 1 (option a): decide nothing textual now; pre-commit the presentation rule (all three rates with uncertainty, conservative-shift reading stated, abstract sentence re-derived at C7 swap), and additionally ratify the two-sided band BEFORE the real human rate is seen (see B22 — deciding it after voids it).

### E5 — Ethics / consent / safety statement for the human driving study — OPEN, now urgent
- Round-2 re-raises: all four again, harder — D-M5 grades it "hard, non-negotiable… independent of scientific merit" and now also names the empty Author Contributions/Acknowledgements (→ E8); A-M7 calls the arm "not reportable or auditable"; C-m7; B-q7.
- Status: nothing in the workspace has changed; the facts must come from the PI (IRB body + number, consent process, safety protocol, recruitment/compensation, whether approval sits under the benchmark organiser's framework).
- Recommendation: supply in the NEXT edit pass regardless of C7 timing (round-1 option a). This is the cheapest large probability move available: every reviewer marks it blocking and it needs no analysis.

### E6 — Benchmark edition/year, archival citation, scenario/system documentation — OPEN
- Round-2 re-raises: D-M3/m13/q1 (bare-URL ref for the load-bearing benchmark; what ARE the 19 systems; maturity class changes the meaning of the 2.1x), B-m15 (scenario-template sampling frame), A-m7 (285 vs 267), D-q3 (18 missing runs, 65 missing-outcome runs).
- New facet: the run-count reconciliation reasons (285→267, 240→175) are organiser facts — fold their retrieval into the same request as the edition/year (B16/B19 execute once facts exist).
- Recommendation: one consolidated request to the benchmark organisers: edition/year/round for both arms, archival citation or report, scenario-template one-liners (A1–C4), system provenance/maturity classes, and the run-ledger explaining missing runs and missing outcome recordings.

### E7 — Data/code availability and review-time access — OPEN
- Round-2 re-raises: A-m17 (editor should obtain confidential artefacts sufficient to reproduce central tables/figures), C-P7 (unedited pre-specified analysis plan), B-P8, D implicit.
- New facet: C's demand for the frozen analysis-plan document itself as a review artefact pairs naturally with a reviewer-only bundle (frozen verdict series + figure data + monitor code). Plan Q09 meanwhile rewords "pre-registration" to the honest "pre-specified and frozen".
- Recommendation: decide the release vehicle before resubmission; a reviewer-only bundle neutralises the reproducibility objection cluster (R2-48/R2-53) at low cost.

### E8 — Acknowledgements and Author Contributions — OPEN
- Round-2 re-raises: D-m11/D-M5 ("empty section stubs on p.33"; "completed declarations" required). Competing Interests is filled; the other two remain empty pending PI content. Unchanged ask.

### E10 — Fig. 6 (C7 target figure) regeneration spec — OPEN
- Round-2 re-raises: D-M9 ("uncertainty not shown" on the paper's most consequential comparisons — panels a and c — "honest annotation is not a remedy at this level"); C-m1 (cluster-valid CIs for every rate and the 2.1x); B-M6 (scenario-paired uncertainty for 15/15); D-q9 (which Fig. 6b intervals exclude parity; human-arm candidate-moment denominator).
- New facets for the spec: print the human-arm candidate-moment denominator (the analogue of 67,861); state which panel-b intervals exclude parity; per-cell diagnostics link to B22. The data interface already carries the flag-rate CI field and per-unit tables, so most demands are satisfiable at swap time without endpoint changes — the exceptions (acceptance band, formal cross-arm test, alternative denominators) are the E2 decision.
- Recommendation: approve the round-1 spec plus these facets in one sitting, before the offline-server swap starts.

### E11 — Disclose the event-level null companion tests? — OPEN
- Round-2 re-raises: B-M5/B-q6 and C-M5/C-q8 again demand episode-level inference; C-M6 additionally attacks selective summarisation. The frozen record still contains the RQ019 B8 event-level nulls and now also the RQ018 case-label permutation p=0.1493 (vs the disclosed exposure placebo p=0.0199, which plan Q18 prints).
- Sharpened consequence: if E7 grants reviewers data access, these nulls become discoverable; round-1's option (a) — one Methods sentence naming the estimand boundary without printing p-values — remains the recommendation, and its value rises with E7.

---

## Decided in round 1 — re-raised, no reopening recommended

- E1 (>40% headline → 21/20/8% story): executed; no round-2 reviewer disputes the numbers. New minor facet (D-m3 rounding tension) is handled by plan Q14 without PI input.
- E3 (abstract "without added emergencies" → "no rise in emergency rates at any supported threshold"): executed. C-M6/B-m13 still call any no-rise phrasing an equivalence claim; the full remedy is B08 (margins + adjusted bounds). Standing by the decided wording is defensible — it scopes to supported thresholds and the <3 s row is disclosed; reopen only if the PI wants the descriptive-only fallback ("flagged moments are one-third to one-half as frequent at the supported thresholds").
- E4 (title/"social compliance" packaging): executed as option (b) (question-vs-delivered-object recast). B-M8/A-m2 re-raise retitling; no new fact accompanies the demand. The response letter should carry the recast argument; a title change remains available as a concession card if the editor sides with B. The "hidden intention" Discussion sentence stays (register-backed); plan Q08 restores the "measurable" qualifiers, which is the accuracy component of D-M7.
- E9 (online = non-anticipation scope note): executed and now quoted by reviewers. B-M7 escalates to demanding either a safety-architecture evaluation or renaming to "offline-validated advisory monitor". Plan Q07 (persistence layer marked unevaluated; Algorithm 2 line re-scoped) plus B04/B13 is the round-2 response; renaming the contribution remains NOT recommended (three of four reviewers accept the scope note as the boundary).
- E12 (structure.md sync): executed — no ">40%" remains in structure.md (verified this round).

---

## NEW this round

### E13 — Abstract precision: opening and closing sentences
- **Issue.** Two abstract sentences draw fire that the body no longer deserves. (1) Opening (L77): "Autonomous vehicles are certified for collision safety, yet a crash-free manoeuvre can still be socially atypical." A-m1 and B-m1 independently object: no universal collision-safety certification exists (ISO 21448 / UL 4600 are assessment frameworks, not guarantees). (2) Closing (L89–90): "whether a deviation is harmful is the next validation layer, which the monitor makes testable." D-m14 reads "makes testable" as promissory. Both sentences are abstract-level wording → PI territory (the intro's parallel "guarantee legal safety" is already handled by plan Q12 without you).
- **Options — opening.**
  - (a) ★ "Autonomous vehicles are engineered and assessed for collision safety, yet a crash-free manoeuvre can still be socially atypical." Keeps the rhetorical pivot, states only what standards do.
  - (b) "Autonomous vehicles can be verified collision-safe…" — slightly stronger than (a), still defensible via the online-verification literature the intro cites.
  - (c) Keep "certified". Guarantees the same first-line objection from two reviewers in every future round.
- **Options — closing.**
  - (a) ★ Keep "which the monitor makes testable" and defend in the response letter: the monitor defines the deviation set, locates the moments, and fixes the endpoints — that IS what makes the harm question testable; the sentence claims an instrument property, not a result.
  - (b) Soften to "…the next validation layer, for which the monitor supplies the instrument." Marginally humbler, same content.
  - (c) Drop the clause. Weakens the abstract's close more than accuracy requires [PRESS].
- **Consequences.** Opening (a) removes a two-reviewer objection for zero narrative cost; closing (a) costs nothing now and (b) is available as a concession. Neither touches any `\targetnum` or the recast thesis sentence (E4 territory).
- **Reviews.** A-m1; B-m1; D-m14.
- **Also note (no decision needed).** The abstract is otherwise untouched by the round-2 plan; every other abstract-adjacent demand routes through E2 (rates/twice), E3 (emergencies, decided), or E4 (framing, decided).
