# Escalations — round-3 status of open items + one new decision

Context for a reader who has not followed the rounds: this manuscript runs simulated NMI review
rounds; items the revision layer may not decide (headline wording, claim boundaries, C7 human-arm
material, facts absent from the workspace) go to the PI. Round 1 opened E1–E12 (E1/E3/E4/E9/E12
decided); round 2 added E13 (decided and executed). Open going into round 3: E2, E5, E6, E7, E8,
E10, E11. Round-3 re-raises are mapped below without re-litigation. One genuinely new escalation:
E14. One open item (E11) is sharpened by an aggregator fact-check finding.

Reading order if time is short: E5 (cheapest large probability move, third unanimous round),
E2 (the named-sites list is now concrete), E11 (register-compliance gap found), E14 (new).

---

## Still OPEN — status and round-3 re-raises

### E2 — Three-flag-rate story, "twice as often", transfer language (C7-coupled) — OPEN, now with a named-sites list
- Round-3 re-raises: D-M3 (the most systematic statement yet), D-m7, D-m10, D-P3; C-M4, C-q4, C-P2; A-M5, A-M6 (wording), A-q11; B-M3, B-P5.
- The five load-bearing sites, all verified verbatim at a4f99f8 (none touched by the S-plan):
  1. Contribution 6 (L181-184): "the reference transfers across country, apparatus and task without inflating its alarms" — D-M3 sets this against the Fig-4 caption's own "Transfer to a data source not seen during fitting is not established" and the printed LOSO 0.743/0.750.
  2. §2.5 title (L485): "The reference is audited by the population that defined it" — the course drivers are different people, different country, closed course; D proposes "audited by human drivers under the same instrument"; A demands removal of the "same population" reading.
  3. Abstract (L87-88): "Auditing the reference with its defining population" — same defect (D-m10).
  4. §2.5 (L498): "the instrument survives the move" — round-2's known carrier sentence.
  5. §2.5 (L503): "The automated systems' flag rate of 9.8% sits at the native level" — D-m7: a cross-context-mix coincidence "dressed as meaning"; the paper's own 9.72→4.7 shift shows such coincidences are not meaningful.
- New facets to fold into the decision (unchanged recommendation core): D's population hedge for the abstract ("19 automated driving systems on this benchmark", D-M2); the funnel-beside-rates presentation (D-M3 remedy, pairs with E10); the LOSO-implied deployment sentence ("cross-source deployment would require support-gate abstention at the observed LOSO levels"); foregrounding the paired per-scenario contrast as the context-controlling analysis.
- Recommendation (updated): still decide nothing textual before C7, BUT pre-commit now to (a) the presentation rule (all three rates with uncertainty + funnels; conservative-shift reading stated); (b) softening the five sites in one pass at swap time (draft wordings exist in D-M3's remedy list); (c) ratifying the two-sided acceptance band BEFORE the real human rate is seen (B22 — deciding it after voids it). The five-site pass is mechanical once ruled; holding it until swap avoids editing \targetnum-adjacent prose twice.

### E5 — Ethics / consent / safety statement for the human driving study — OPEN, third unanimous round, urgent
- Round-3 re-raises: D-M1 (again grades it standards-desk blocking, now bundled with declarations and deposit as "any of these alone triggers a return"), D-q1; A-M6, A-q10, A-P6; C-m8.
- Status unchanged: no ethics facts exist in the workspace; nothing may be invented. Required from the PI: approval body + protocol number, consent process, safety protocol for staged conflicts, recruitment/instruction/compensation, order/counterbalancing, missing-trial rules, whether approval sits under the benchmark organiser's framework.
- Recommendation unchanged and stronger: supply in the next edit pass regardless of C7 timing. Every round has priced this as the single cheapest probability move; D's 60% post-revision estimate explicitly conditions on it.

### E6 — Benchmark edition/year, archival citation, scenario/system documentation, run ledger — OPEN
- Round-3 re-raises: D-m8 (formal citable reference + access date), D-m2/D-q8 (285 vs 267; the 65 consequence-absent runs; human-arm run count), C-q7, A-q9, A-q14 (are the 19 systems independent — frozen basis is "19 team clusters", RQ018-B6; full provenance is organiser material), D-m3/D-q9 (who drives the counterpart vehicle; matched across arms?).
- Note: the bib entry is now formal in shape (organiser, title, URL, year 2024) but still lacks edition/round and access date, and the edition facts remain outside the workspace.
- Recommendation unchanged: one consolidated organiser request (edition/year/round for both arms, archival citation, scenario one-liners, system maturity classes, run ledger incl. the 18 missing and 65 outcome-absent runs, counterpart-vehicle control description). Feeds B16/B19 the day it lands.

### E7 — Data/code availability and review-time access — OPEN
- Round-3 re-raises: A-m13 (archive available during evaluation), D-M1/D-q11 (what will be deposited, where, licence, reviewer access), B-m12/C-q10 (dated protocol artifacts).
- Sharpened consequence: E7 access makes the per-endpoint placebo ledger discoverable (see E11) — deciding E7 and E11 together is now clearly preferable.
- Recommendation unchanged: decide the release vehicle before resubmission; a reviewer-only bundle (frozen verdict series + figure data + monitor code + dated freeze records) neutralises the reproducibility cluster at low cost.

### E8 — Acknowledgements and Author Contributions — OPEN
- Round-3 re-raises: D-M1 (empty declarations; funding undeclared). Unchanged ask: PI-supplied content. Pairs with E14 (the Competing Interests section will need the benchmark-relationship sentence).

### E10 — Fig. 6 (C7 target figure) regeneration spec — OPEN
- Round-3 re-raises: A-m8, B-m9, C-m3 (CIs for panels a/c and the ratio), C-M5/D-q3 (human-arm candidate→judgeable funnel printed beside the rates), B-M5 (scenario-paired uncertainty for 15/15), A-q8 (per-scenario counts — the AV side already exists frozen in `av_reference_values.json` per_scenario_alpha90).
- Recommendation unchanged: approve the accumulated spec in one sitting before the offline-server swap (CIs on all three rates; per-scenario paired uncertainty; the human funnel as the analogue of 67,861 → 20.8%; which panel-b intervals exclude parity; per-unit dispersion note). The data interface already carries the CI and per-unit fields; the additions are display-spec, not endpoint changes — except the acceptance band, which is the E2(c) ruling.

### E11 — Event-level nulls and placebo-companion disclosure — OPEN, SHARPENED by a compliance finding
- What is new: fact-checking the round-2 insertion found that the manuscript now cites RQ018-KC-C1
  (the quantile compressions AND the placebo p=0.0199) without the decision file's Required
  Qualification. `RQ018 decision.md` (L40 and boundary B5) makes two companions mandatory when C1
  is cited: (i) the case-level label permutation p=0.1493 did not reach 0.05; (ii) the contract-window
  case-clustered CI crosses zero (90%: [−2.6100, +0.1372]) — only the fixed 3-s window excludes zero.
  Also frozen and discoverable under E7: the per-endpoint placebo ledger
  (`rq018_rerun/negative_controls.json`) with p from 0.005 to 0.985 across endpoint × window × side
  (the sanctioned 0.0199 is the fixed-3s log1p lower-side entry). Round-3 reviewers demanded exactly
  this resolution level (B-M4: placebos for every primary endpoint; C-M8: selective-summary concern).
- Options:
  - (a) ★ Comply with the frozen ruling: one Methods sentence adding the two companions, phrased as
    scope ("the placebo distinguishes timing for the fixed-window magnitude endpoint; a case-label
    permutation on the same battery does not reach significance (p=0.1493), and the open-ended-window
    regression interval admits zero — the primary evidence remains the quantile comparison").
    Cost: two null numbers enter Methods. Gain: register compliance, immunity to E7 discovery,
    directly answers B-M4/C-M8.
  - (b) Remove the placebo p from the text entirely (cite the quantile comparison only). Cost: loses
    a passing control the reviewers value; the p returns via E7 access anyway.
  - (c) Keep as is. Leaves the manuscript out of compliance with its own frozen decision — not
    recommended under any reading of the claims discipline.
- Consequence of deciding: (a) closes R3-27 and part of B-M4 now; (b) is defensible but weaker; (c)
  risks a future integrity flag from a reviewer with data access.
- Reviews: B-M4, B-q6; C-M8, C-q10; (round-2: B-M5/C-M5/C-M6 lineage).

---

## Decided in earlier rounds — round-3 status, no reopening recommended

- E1 (>40% → 21/20/8 story): no round-3 reviewer disputes any width number; the Q14 rounding note
  worked (zero rounding complaints). CLOSED in effect.
- E3 (abstract "no rise in emergency rates at any supported threshold"): C-M8 re-attacks "supported"
  as selection-by-significance; S04 defines the term in the Fig-5 caption; the abstract wording
  stands. Standing by the decision remains defensible.
- E4 (title/"social compliance"): re-raised a third time (A-M1/P1, B-M1/P1, D-M7/P7) with no new
  fact. The concession card (retitle toward "social atypicality") remains available if the editor
  sides with the panel; D-M7's actionable sliver (early "one dimension" scope sentence) is executed
  as S15 without touching the title. Response letter carries the recast argument.
- E9 (online = non-anticipation scope): quoted and accepted by C ("the paper uses 'online' correctly");
  B-M8/C-M10 still want the runtime evaluation → B13/B04. S16 adds the safety-subordination
  interface sentence, which was B-M8's sharpest governance point. No renaming recommended.
- E12 (structure.md sync): stable; nothing new.
- E13 (abstract opening/closing): RETIRED — zero round-3 objections to "engineered and assessed"
  and zero to the kept "makes testable" closing. The round-2 prediction that the closing needed a
  response-letter defence was not even tested; no NONE disposition was needed this round.

---

## NEW this round

### E14 — Competing-interest / role disclosure for the benchmark ([53])
- **Issue.** D-M1/D-q10 demand disclosure of the author's relationship to the benchmark organisers
  and the terms under which official outcome labels were accessed (the Methods states they were
  examined and excluded as endpoints). The workspace CORROBORATES the premise: `bibliography/biblio.bib`
  L38-43 names the operator as "Tongji University TOPS Group" — the author's own institution. What
  the workspace does NOT contain: the author's actual role (none / participant / organiser-adjacent),
  the data-access terms, and whether any authorship or funding overlaps with the challenge series.
  None of this may be invented; the disclosure text must come from the PI.
- **Options.**
  - (a) ★ Add a Competing Interests sentence (and a Methods access-terms clause) stating the
    relationship precisely: institution shared; author's role in the challenge (as it actually is);
    official labels obtained under [terms], examined once under the outcome-blind crosswalk and
    excluded as endpoints. Pairs with E8's declarations pass.
  - (b) Silence. Guarantees the question returns from any Nature-family standards desk (D grades the
    bundle as return-triggering), and the institutional link is already visible in the bib entry —
    silence reads as concealment of a checkable fact.
  - (c) Independent-operation statement only ("the benchmark is operated independently of the
    authors") — ONLY if factually true; the PI must confirm before any such sentence exists.
- **Consequence.** (a) costs one sentence and closes D-q10 cleanly; (b) risks an integrity-flavoured
  desk query, the worst class of reviewer interaction; (c) is (a)'s special case if the facts allow.
- **Reviews.** D-M1, D-q10, D-m8 (formal citation + access date, executed separately under E6 once
  edition facts arrive); A-q14 (system-independence, adjacent).
- **Recommendation.** (a), decided together with E5/E8 in one declarations pass — the three form a
  single "standards-desk completeness" package and are jointly the largest as-submitted probability
  lever available without any new analysis.
