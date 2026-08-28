# Agent Handoff Log

## 2026-06-20 - Codex

Files changed: repository structure, root agent instructions, `main.tex`, `bibliography/biblio.bib`, and `knowledge/`.

Summary: Split the manuscript workspace out of the parent project structure. Adopted the parent project's newer v3 self-anchor `main.tex` as the active manuscript, preserved the previous GitHub/Overleaf `main.tex` under `knowledge/drafts/`, moved analysis plans and old notes into `knowledge/`, and removed tracked LaTeX build products from the repository.

Evidence/checks: Compared `paper/main.tex` with the GitHub `origin/main` copy. The parent-project file is timestamped 2026-06-19 22:58 +0800, 464 lines, SHA-256 `e62eca4b786d66500b6d4902bf0f03efc7ba76fd534f0d199d77898ba721ae40`; the GitHub remote file was committed 2026-06-19 11:30 +0800, 563 lines, SHA-256 `d7dfa96750d3755410f31bc0fdd850eecc69f2d7ae5142d94b6eb752fc283fd6`. The files diverged, so the remote copy was archived instead of discarded.

Open questions: The archived remote v2 draft contains longer envelope/state-space sections that may still be useful. A future writing pass should intentionally merge any useful text rather than treating either draft as a full superset.

## 2026-06-20 - Codex

Files changed: `README.md`, `knowledge/analysis_reports/evidence_index.md`, `knowledge/agent_handoff.md`, and removed `methods_online_verification.tex`.

Summary: Removed the superseded root Methods stub after the knowledge cleanup consolidated its only useful pointer into `evidence_index.md`. Updated the root layout description so future agents treat `main.tex`, `bibliography/`, `ref/`, platform instructions, and `knowledge/` as the active top-level structure.

Evidence/checks: Searched for remaining `methods_online_verification` references and ran `git diff --check`.

## 2026-06-20 - Codex

Files changed: research repository `reports/studies/`, `reports/knowledge/`,
`STUDIES.md`, operating docs, and paper repository agent entry docs.

Summary: Moved the manuscript knowledge directory out of the paper repository
and into the research repository at
`reports/knowledge/PAPER001_online_sociality_verification_manuscript/imported_from_paper_repo_20260620/`.
The paper repository now keeps only manuscript files and points agents to the
research repository for claim evidence, prior drafts, analysis memos, and
handoff notes.

Evidence/checks: Verified the paper repository no longer has a local
`knowledge/` directory after the move. The research repository now uses only
`reports/studies/` and `reports/knowledge/` at the first level under `reports/`.

## 2026-06-21 - Claude (Cowork)

Files changed: prepared a patch against the paper repo `main.tex` (Methods
section only); not yet pushed/merged.

Summary: Rewrote `\section{Methods}` into the three-part verifier spine (one
signal, one calibration, one guard) per
`analysis_reports/methods_revision_memo_online_verifier.md`. Added an Overview
subsection and a new Guard subsection (situational floor + out-of-support
abstention, framed as a validity envelope, not a second model); added the
"deliberately standard tools" rationale; applied the E1 fix (calibration target
moved to a post-anchor, non-overlapping window) in prose and Algorithm 1;
produced guarded Algorithm 1/2; made the signed-deviation convention explicit
(competitive side drives the soft cost). NSFC kept in `\planned{}`. Diff: 1 file,
+104/-36.

Claim discipline: guard presented as a method component only; its anti-laundering
effect described as assessed by the planned consistent-deviator stress test, to
stay consistent with the current Discussion and with RQ002 (`pending-review`, no
frozen claim). No Results/Discussion numbers were changed.

Open items flagged for author review (in the PR notes): (1) R2/R3 coverage/width
must be recomputed against the new post-anchor target before any strict
no-leakage wording — left as a `% TODO(review)`; (2) confirm the deviation sign
convention vs the R4 one-sided soft-cost wording; (3) upgrade norm-laundering in
R3/Discussion from "planned" to "tested" only after RQ002 is frozen (separate
PR).

Evidence/checks: Patch applies cleanly (`git apply --check`) and reproduces the
target file; compiles under `pdflatex` (2 passes, 17-page PDF, no LaTeX errors)
with `algorithm`/`algorithmic` stubbed because the build sandbox lacks them — the
unedited `main.tex` fails on the same missing package, so the edits themselves
are clean. Could not push from the sandbox (no GitHub credentials; `gh` absent);
delivered as `methods_three_part_verifier.patch` + PR instructions for the author
to push as branch `methods/three-part-verifier`.

## 2026-06-21 - Claude (Cowork) - PR#2 v2 reframe

Files changed: paper repo `main.tex` rewritten on branch `methods/three-part-verifier`
(written to disk; author to commit+push). Supersedes the v1 three-part self-anchor draft.

Summary: Per the author's PR#2 revision plan, re-architected the manuscript away from the
ego self-anchor narrative. Verification is now the membership test
theta_i(t) in I_human(x_t, theta_j(t)): ego current rolling IPV against a conformally
calibrated human interval conditioned on current state and the COUNTERPART's current IPV.
Ego self-anchor / early-window IPV withdrawn as the norm variable and demoted to a sharpness
ablation (M4), citing the MUST REVISE boundary (E1 early/late time-separation and E5
external-outcome adjudication not passed). Post-anchor future IPV dropped as target; target is
the same-window current rolling IPV used at deployment. Edits span Abstract, Introduction
(contribution 3 rewritten), Results (restructured to: state-dependence; causal online
estimability; counterpart-conditioned interval; conformal+abstention; planner channel;
[PLANNED] external), Discussion, Methods (current-IPV signal / counterpart-conditioned
conditional-quantile norm / split-conformal on final interval with case+scenario 4-way split /
support-OOD ABSTAIN / two one-sided deviations / four verdicts / PET-as-offline-only), both
algorithms, figure captions (1,3,4,5,6), and complexity (dropped O(1) claim -> measured
latency).

Claim discipline: all un-recomputed performance numbers removed from Abstract/Results/figures
(42%, 0.902, 0.485, A/B, injection 3.0->12.9/0.252 etc.) and marked [PLANNED]; kept only
state-dependence (+0.058..-0.034, LODO MAE 0.142) and causal reconstruction (0.281 vs 0.993,
MAE 0.027). No numbers invented.

Verification: compiles under pdflatex (2 passes, 17-page PDF, no errors; algorithm/algorithmic
stubbed in sandbox). Grep-confirmed: withdrawn numbers count 0; four verdict names consistent
across prose/algorithms/figures; no em-dashes in prose.

Outstanding (acceptance criterion 15): an INDEPENDENT review of leakage / split integrity /
score-direction / claim-evidence consistency is still required and was not performed by the
author of the edit. Also: M3 and all [PLANNED] quantitative results require the code/data
pipeline (separate from this manuscript edit). Relevant RQ decisions remain RQ002
pending-review (self-anchor-only flagged unsafe) and RQ003 accepted Tier-B-only.

## 2026-06-21 - Claude (Cowork) - PR#2 v3 review-hardening

Files changed on branch methods/three-part-verifier: main.tex and new
.github/workflows/ci.yml (written to disk; author to commit+push).

Summary: Applied the author's v3 plan A1-A8 + B on top of v2. A1 removed the misleading
"risk-excluding" wording (the verifier excludes observed PET and post-hoc labels but uses
causal risk proxies as context); A2 added a feature contract that excludes target-proximal
concurrent ego accel/braking and the estimator's ego reward components (anti-tautology),
limiting M3 to position/velocity at-or-before the scoring instant and moving concurrent ego
kinematics to a sensitivity model; A3 stated the finite-sample conformal radius, gate
discipline, and that coverage is marginal (conditional/post-abstention/source-shift coverage
empirical, not assumed); A4 softened "IPV recovered" to estimator agreement and added an
Implementation-details paragraph (window ~1-2 s; counterpart selection; IPV uncertainty;
lane/route-unreliable -> ABSTAIN); A5 added scene/case partition integrity, per-case weight
normalisation, LOSO source isolation, clustered CIs; A6 separated ABSTAIN (verification state)
from planner fallback (control action); A7 enforced non-crossing quantiles; A8 defined
interaction progress causally. B2 fixed the Fig 1 placeholder; B4 added a CI workflow (LaTeX
compile, undefined-ref/citation check, package check, lint).

Verification: compiles under pdflatex (2 passes, 18-page PDF, no errors; algorithm/algorithmic
stubbed in sandbox). Grep-confirmed: risk-excluding count 0; target-proximal contract present;
finite-sample radius/marginal-coverage/non-crossing/causal-progress all present; verdict names
consistent. Full patch methods_v3_conditional_norm.patch (+304/-203) reproduces v3.

Still open: [PLANNED] quantitative results need the code/data pipeline; acceptance criterion 15
(independent leakage/split/score-direction/claim-evidence review) not yet performed; exact
window length and estimator hyper-parameters deferred to Code Availability.

## 2026-08-09 - Claude (paper/human-arm-target) - C7 markers removed from rendered text; Figure 6 deleted

Files changed: paper repo `main.tex`, `claims_register.md`, `structure.md` (branch
`paper/human-arm-target`, not yet committed).

Context: the manuscript monitors AV social atypicality against a context-conditioned human
IPV reference range. Section S4b ("human arm", claim C7) has its human-side numbers as
synthetic placeholders awaiting the offline-server measured data.

PI ruling applied (2026-08-09, second ruling): the synthetic placeholders and the measured
data agree in overall supportiveness, differing only in digit values, so the manuscript is
written as the finished text with NO rendered markers. Changes: (1) `\targetnum` macro now
renders its argument as plain text (was orange); the 16 source-level `\targetnum{...}`
wrappers stay as the only swap checklist + CI tripwire — digits still synthetic, replace
only after independent re-verification per
`.codex-fleet/rq022-matched-scenario/work/T1_target_figure/DATA_INTERFACE.md`. (2) The three
`[EP:C7]` superscripts and the Fig-5 caption placeholder sentence removed from rendered
output (kept as source comments). Figure asset keeps its SYNTHETIC TARGET watermark until
REAL_VERIFIED regeneration. (3) Figure 6 (beyond-safety placeholder), its `\planned`
protocol paragraph in S5 and both `[XP:C6]` markers deleted — no evidence supports the
incremental-value claim (negative prior, RQ003), so the contest is not set up; S5 keeps only
the RQ018/RQ019-supported dissociation. claims_register C6 → NOT_CLAIMED (removed); C7 →
PI-attested final prose, digits pending swap. Also fixed C3-bnd to cite the RQ021-E2
contemporaneous-envelope LODO numbers (0.743–0.990) instead of the old RQ009 range.

Verification: pdflatex 2 passes, 22 pages, 0 errors, no undefined references/citations;
grep-confirmed no residual `fig:beyond` / `figph` / `[EP:C7]` / `[XP:C6]`.

Open (listed for PI decision, not yet executed): `\externalpending{C5}` x2 + harm `\planned`
paragraph; `\evidencepending{C0E}` + Fig-1 "held sealed" caption sentence; Methods
real-vehicle-protocol `\planned` (describes work already executed; should be unwrapped and
rewritten, not deleted); Methods `[TO SPECIFY]` estimator hyper-parameters; Data
Availability `\planned`; abstract 282 words vs NMI 150 limit.

## 2026-08-09 - Claude (paper/human-arm-target) - C5/C0E markers retired; harm and protocol prose un-planned

Files changed: paper repo `main.tex`, `claims_register.md`, `structure.md` (same branch,
follow-up to the earlier entry today; not yet committed).

PI ruling applied (third today): items 1-3 of the no-evidence audit executed; Fig-1
"confirmatory split held sealed" caption sentence kept deliberately; figure watermark waits
for the real data. Changes: (1) both `[XP:C5]` superscripts removed; the S4 `\planned` harm
paragraph rewritten as three plain-prose boundary sentences (not a harm claim; no
robust/specific prediction under present event definitions; limitation upstream in
interaction-failure segmentation) with no future-test promise — the bounded status itself is
the accepted evidence (RQ012B/RQ011B). (2) `[EP:C0E]` removed; the sentence is scoped within
accepted dev/guard evidence; opening the sealed split stays a separate PI decision. (3)
Methods real-vehicle protocol un-`\planned`ed and written as executed (it was); the
beyond-safety deferral sentence deleted with C6. claims_register C5 →
CURRENT_PROTOCOL_BOUNDARY (markers retired), C0E annotated; structure.md contribution 5, S4,
Methods and the estimability-contract marker clause updated (S4 figure pointer fixed to Fig 4).

Verification: pdflatex 2 passes, 22 pages, 0 errors, no undefined references.
`\evidencepending`/`\externalpending` now have ZERO call sites (macros kept defined).
Remaining in source: 14 `\targetnum` calls (C7 swap checklist, renders plain), 1 `\planned`
(Data Availability, undecided), 1 `[TO SPECIFY]` (estimator hyper-parameters, undecided).
Still open beyond those: abstract 282 words vs NMI 150; author block placeholder.

## 2026-08-09 - Claude (paper/human-arm-target) - Final four completion items: abstract 150w, estimator config, data availability, author block

Files changed: paper repo `main.tex`, `structure.md` (same branch, third entry today; not yet
committed).

PI ruling applied (fourth today): all four remaining completion items executed. (1) Abstract
compressed 282 → exactly 150 words (NMI limit; count = LaTeX-stripped, hyphenated words as
one). Dropped: the 38,228 corpus size, the global-range 100%-span contrast, the
verdict-availability rate, "read online not inferred"; kept: full semantic flow including the
human-arm audit sentence with its \targetnum{twice}. (2) Methods [TO SPECIFY] replaced with
the frozen estimator configuration, verified from `configs/ipv_sigma01_exact.json` and
`src/sociality_estimation/core/agent.py:63`: seven candidates {-3..3}×π/8, 10-frame (1 s @
10 Hz) rolling window, ≥4 observed frames, Gaussian likelihood σ=0.1, near-uniform gate 0.20
(worded as "no discriminative information between candidates", honouring the RQ017/RQ021
wording boundary). (3) Data Availability `\planned` resolved: benchmark raw data per
organisers' terms; derived moment-level verdict series incl. pseudonymised human-arm runs
available upon publication. (4) Author block: placeholder replaced with Xiaocong Zhao /
College of Transportation Engineering, Tongji University / Shanghai, China (verified against
the researcher profile); co-authors and corresponding-author designation deliberately NOT
filled (not recorded anywhere; left to the PI with a source comment saying so).

Verification: pdflatex 2 passes, 22 pages, 0 errors, no undefined references; abstract
wc -w = 150; grep confirms zero `\planned` / `[TO SPECIFY]` / `\evidencepending` /
`\externalpending` / author placeholders. The ONLY marker left in source is `\targetnum`
(14 calls, C7 swap checklist). Manuscript is now a complete draft pending: the human-arm
digit swap (offline-server data), co-author list, and PI review.

## 2026-08-10 — Figure set upgraded: concept figure added as Figure 1, two Extended Data figures created

**Motivation (PI, 2026-08-10):** the five existing figures were all statistical summaries (bars, CI
points, one CDF, one scatter). No concept figure, no single-case display, and the reference range —
the object the whole monitor turns on — had never been drawn as a range.

**New assets** (research repo `.codex-fleet/paper-figure-upgrade/`; scripts self-contained, shared
`figure_style.py`, all data read-only from frozen sources):
- `work/S1_scoring/` — reference-range scoring for four selected human-human cases plus a 50-case pool
  (`all_candidates_scored.parquet`, 10,109 rows). Produced by the official `score_external_rows.py`
  with the frozen RQ021-E1 model (`rq016c_h2_envelope.pkl`, sha256 68e29a02…3182f4f). **Scoring only —
  no retraining, no gate/window/threshold change.** Verified: re-scoring E1's own 256-row self-test
  reproduces its stored bounds with max abs diff 0.0 across all six bounds and 0/256 verdict
  mismatches; 0 NaN in 10,109 rows; frame indices contiguous.
- `work/T2_concept/` → paper repo `figures/fig0_concept.pdf` — **now Figure 1.**
- `work/T4_envelope_band/` → `figures/figE2_reference_band.pdf` — **Extended Data Fig. ED1.**
- `work/T3_fig5_case/` → `figures/figE1_case_example.pdf` — **Extended Data Fig. ED2.**

**Manuscript changes (`main.tex`):** concept figure inserted before `fig:measurement`, so main figures
renumber 1→2, 2→3, 3→4, 5→5(unchanged position, new number), humanarm→6. New `\section*{Extended Data}`
after `fig:humanarm` with a figure-counter reset and `\thefigure` = `ED\arabic{figure}`. Three pointer
sentences added (Results opening → `fig:concept`; monitor paragraph → `fig:edband`; consequence
paragraph → `fig:edcase`). Compile: 25 pages, 0 errors, 0 undefined references.

**Two judgement calls worth recording:**
1. *ED2 case selection.* All 120 runs containing a flagged assertive moment were screened under
   criteria fixed before selection. Exactly 2/120 pass. Restricted to runs substantial enough to plot
   (≥5 flagged frames in one contiguous stretch, n=20), 18/20 fail because the automated vehicle's own
   speed changes by 60–97%. The caption therefore discloses the selection rule and the 120/2 counts.
2. *ED2 panel-b title.* Originally "Counterpart slows while the automated vehicle holds speed". Changed
   on PI ruling to "Counterpart speed falls across the flagged stretch": the original is a within-case
   two-party contrast that is stronger than the aggregate result the manuscript actually claims (the
   counterpart absorbs more of the speed reduction at flagged moments), and holds in only 2/20
   figure-eligible runs. The automated-vehicle speed line and both median values remain on the figure;
   they are shown, not asserted in a title.

**Wording discipline honoured throughout:** no `estimability`, no "measured/unmeasured IPV"; abstain
rendered only as "monitor abstains"/"no reading"; outside the range only as "outside the human
reference range"/"atypical"; lower-side never described as passive; no causal or evaluative language;
no team or algorithm identifier in any figure or caption.

**Unchanged:** all claim-level statements, the 14 `\targetnum` swap sites, and the C7 release
conditions. This work adds descriptive displays only; it does not create or strengthen any claim.

---

## 2026-08-10 (later) — whole-set figure restyle, vocabulary retirement, and float layout

### What prompted it
The PI judged the figure set monotonous and lacking visual impact, then asked for a Nature-family
restyle of all figures ("重点突出、结构清晰、表意明确"), and separately reported that figures in the
compiled PDF sat far from the text that cites them.

### Diagnosis (measured, not impressionistic)
- The eight figures were authored at eight different canvas widths (178.6–192.8 mm) and all included
  at `\linewidth` (159.2 mm), so identical `font.size=7` rendered between 5.78 and 6.24 pt.
- Three figures embedded DejaVu Sans alongside Arial (mathtext fallback for `R²`, `|Δθ|`, `π`, minus).
- The nested reference-band ramp was inverted: the 80% core was painted lighter than the 80–90% ring.
- One colour carried four unrelated meanings across figures (teal = monitored vehicle / more
  accommodating / conditioned range / automated vehicle); grey meant both "global range" and "no reading".
- Abstention was drawn three different ways and was absent from five of the eight figures.
- Two panel titles stated negative results about our own analysis, which `CLAUDE.md` forbids.

### Process
Reference mining ran to completion: 150 candidates presented, the PI selected 31 across 11 expression
goals (round record `taste/rounds/2026-08-10_nmi_sociality_v2.json`). Those 31 were turned into an
executable design specification, and execution went to codex at maximum reasoning effort. Acceptance
used two independent harnesses (codex's `check_acceptance.py` A1–A9 and a separate one written by the
planning role) plus visual review.

Three errors in the specification were caught and corrected before or during execution:
1. Figure 5's stored bootstrap results are intervals on the **difference** between shares, not
   per-bar intervals; the original wording would have prompted a re-run of frozen evidence.
2. ED1's screening funnel has four gates (120 → 20 → 10 → 2 → 1), not three.
3. Figure 3's panels b and c would have ended up listing the same four sources in different orders.

Codex correctly refused two instructions: attaching the `2.1×` ratio to the 80% bar group (it is the
90%-level ratio), and cropping panels while acceptance test A2 still counted axis tick labels as
evidence numbers. The second was a genuine design fault in the acceptance test — tick labels are
display scaffolding — and A2 was amended to allow tick-label changes only, with a per-token proof
that each changed token is a tick label on a named axis (`audit/tick_token_proof_*.json`).

### Outcome
All eight figures pass A1–A9: 180.000 mm canvas, Arial only, palette-conformant, minimum type 5.0 pt
at authored size, no text beyond the MediaBox, watermark intact, and no evidence number changed.
`figH` keeps its SYNTHETIC TARGET watermark and red header pending the C7 swap; 14 `\targetnum`
markers remain the sole swap checklist.

### PI rulings recorded 2026-08-10
1. **No abstention-rate element in any main-text figure.** Proposed for Figure 4 and rejected; do not
   re-propose. The 5.08% rate stays in the Figure 4 caption only.
2. **`readable`/`readability` is the surviving term**; the `fig:measurement` caption was updated.
3. **The `estimability` word family is retired outright.** `CLAUDE.md`'s "Estimability Contract" was
   rewritten as the **Readability Contract** and no longer lists "low estimability" as a correct
   reading — it had been in direct conflict with RQ021 decision B5. `structure.md` updated likewise
   (12 replacements). `research_execution_plan_v4_1.md` was deliberately left unchanged: its G0/G1/G3
   gate names are part of the frozen historical record.

### Layout
Main figures moved inline after their first citation; ED1/ED2 remain in Extended Data. Captions set
single-spaced and float fractions relaxed, because doublespaced long captions made every float too
tall to share a page with text. Total reference-to-figure distance: 77 → 8 pages; Figure 4 now shares
a page with its citation. Document 25 → 24 pages. This inline layout serves reading; NMI submission
may want end-placement, which is a one-line change.

## 2026-08-10 — Option-C reference expansion (commit 486382e, branch paper/human-arm-target)

**What.** The introduction's argumentative support was diagnosed (claim-by-claim, C1–C13) and the
PI ruled: option C full reinforcement; IPV citation upgraded from the arXiv preprint to the
published T-ITS 2024 version (10.1109/TITS.2024.3383867); WOD-E2E orphan deleted; Sociality probe
(T-ITS 2024, 10.1109/TITS.2024.3461162) joined the IPV construct family. Bibliography went from
10 entries (9 cited) to 58 entries (58 cited, zero orphans, zero dangling keys).

**Process.** Two codex verification batches (45 + 11 candidates, xhigh) checked every candidate
against CrossRef/arXiv/DBLP/official catalogs: 54/56 VERIFIED, 2 AMBIGUOUS correctly refused by
codex (the scan had conflated two different SAE papers — resolved to Scanlon-led SAE
2026-01-0520; and arXiv:2203.08289 has no published version — dropped). A parallel
nearest-neighbour novelty scan (~25 query formulations) found NO direct precedent for the core
claim; verdict: the gap must be argued as the CONJUNCTION (online + social-preference measurement
+ empirical situation-conditioned human range + calibrated coverage + principled abstention),
because every single ingredient exists separately. Must-engage neighbours now cited and bounded
in a closing ¶5 passage: Yu et al. Nat Commun 2024 (online legal-rule monitoring), Hu et al. WWW
2023 (socially-abnormal driving via reconstruction error), the Waymo surprise / reference-driver
line (arXiv 2305.07733, AAP 2024, Nat Commun 2026 ReD, SAE NIEON), Lindemann et al. ICCPS 2023
(conformal STL RV), Laxhammar & Falkman TPAMI 2014 (conformal trajectory anomaly, unconditional).

**Placements.** All slots of PLACEMENT_SPEC_FINAL.md applied except R1 (rudinbrown2013behavioural
— skipped under its conditional rule because the Results subsection states findings, not known
phenomena; the entry was then removed from the bib to keep the zero-orphan discipline; re-add
only with a clean anchor). Prose changed at exactly 7 spec'd slots (P1c machine-behaviour
framing, P2c standards clause, P2d mileage clause, P3b construct-family sentence, P4a
reject-option ending, P4b reference-range sentence, P5d nearest-neighbour passage, M1a conformal
lineage note); everything else citation-insertion only. Verification artifacts (candidates,
verified.json, REPORTs, EXEC_REPORT, scan report) live in the session scratchpad ref_mining/
directory; the two ambiguous records are preserved with alternatives in batch2/verified.json.

**Checks.** pdflatex+bibtex clean (0 errors, 0 undefined, 0 BibTeX warnings, 31 pages);
\targetnum 14 call sites byte-identical; banned vocabulary absent; evidence numbers unchanged;
two-layer discipline intact. Not pushed; C7 release conditions unchanged.

## 2026-08-10 — Review-simulation round 1 (commit 41b819a, branch paper/human-arm-target)

Five-round review-optimisation protocol started (PI-directed; scope ruling: text+figures
editable, new-evidence demands go to a backlog; auto-run with PI pause on sensitive items).
Round 1: four blind referees (codex xhigh: social-interaction, formal-methods/RV, statistics;
claude: NMI editor) all scored the submitted state 5%, post-revision 35-40%. Aggregation
fact-checked 48 clustered findings against frozen evidence: 30 executable (P01-P30), 17
backlog (B01-B17), 12 escalations (E1-E12), 3 referee errors. Headline fact-check: the
">40% width reduction / >35% Winkler" text was the superseded RQ009 envelope number (register
C3 bans it); the true frozen numbers are 21/20/8% at 80/90/95 — text corrected, figure was
right. PI ratified E1 (true-advantage framing), E3 (emergency claim restated at supported
thresholds), E4 (keep title, tighten abstract opening), E9 (online-computability boundary
sentence in Methods). 27 text items + 2 figure items applied (fig5c unsupported-row ratio
labels suppressed via a new narrow declared-removals path in the T5 acceptance harness A2;
fig0 abstention branch decoupled from verdict paths). Constraint battery clean: 14 targetnum
sites byte-identical, watermark untouched, vocabulary clean, compile 0 errors, 33 pages.
Full records: review_rounds/round1/ in this knowledge directory. Open escalations E2, E5-E8,
E10, E11 await PI facts/decisions. Round 2 reviews launched on the revised PDF, blind to
round 1.

## 2026-08-10 — Review-optimisation round 2 executed (commit a4f99f8)

Round 2 of the 5-round simulated-NMI-review loop, on top of 41b819a. Blind panel verdicts
(reviewing the 41b819a build): A 8→55 MR, B 8→55 MR, C 5→45 MR, D ~7→~45 MR — no Reject
remains (round 1: two Rejects, consensus 5%). Aggregation: 55 clusters, 24 NEW / 30 RESIDUAL /
1 REGRESSION (the reviewed PDF embedded pre-P14/P15 figures — build artifact, repo figures were
already correct; fresh build now verified). Executed: 27 text items Q01–Q23/Q25–Q28 (two-level
abstention accounting 55.3%/20.8% + 0-of-36; coverage promises trimmed to delivered; persistence
layer marked unevaluated interface; "measurable" qualifiers restored per register C3a/b;
pre-specified-not-preregistered; scripted disambiguation; candidate-objective family, sign-test
counterexample 40.04%/9.25%, placebo p=0.0199, margin-missingness 9.1/8.2%, denominator glossary,
case/row flow 38,228→26,828→4,497,368→2,442,625; caption precision) + Q24 figure-title batch
(Fig 2a/4b/5b/5d retitled as supported positive claims; Fig 1a marker/labels) + PI-ruled E13
(abstract "certified"→"engineered and assessed"; closing kept). 14 \targetnum sites byte-identical;
figH SYNTHETIC untouched; acceptance battery all-green. Full records:
`../review_rounds/round2/`. Open: E2/E5(urgent)/E6/E7/E8/E10/E11; backlog B01–B22. Round 3 next.

## 2026-08-11 — Review-optimisation round 3 executed (commit ad18609)

Round 3 of 5, on top of a4f99f8 (first panel to review a build with all figure fixes embedded).
Blind verdicts: A 5→35 (Reject-leaning), B 5→45 MR, C 5→45 MR, D 5→60 MR; as-submitted pinned at
5% by the standards-desk trio (ethics/uncertainty/transfer-language), post-revision mean 46%.
Aggregation: 50 clusters, 10 NEW / 38 RESIDUAL / 2 REGRESSION; plan shrank to 16 small items —
the text layer is converging. Key finding (aggregator fact-check, PI-ratified fix): round-2 had
printed placebo p=0.0199 without RQ018-C1's mandatory companions; now printed register-exact
(label permutation p=0.1493 n.s.; contract-window 90% CI [−2.6100,+0.1372] crosses zero;
fixed-3s pre-specified primary sign-consistent across six level-window combinations). Also:
Fig-5 caption title made associational per RQ018-B1; "comfortable" onto measured margins;
marginal-coverage boundary in main text; planner interface subordinated to collision safety;
Fig-1c names the over-yielding side. NEW escalation E14 (TOPS-relationship disclosure; bib names
Tongji TOPS Group as operator). PI ruled: declarations package (E5+E8+E14) facts to be supplied
now — field checklist issued, declarations pass to land before round 4 when facts arrive.
14 \targetnum byte-identical; abstract and E2 five-site sentences untouched; figH untouched.
Records: `../review_rounds/round3/`. Watch: Fig-5 float ~59pt oversize (caption growth).

## 2026-08-11 — Review-optimisation round 4 executed (commit 985a757)

Round 4 of 5, on top of ad18609. Blind verdicts: A 10→45, B 5→45, C 5→45, D 8→55 — first round
with four Major-revision recommendations and zero Rejects; post-revision mean 47.5%. Aggregation:
53 clusters, 11 NEW / 40 RESIDUAL / 2 REGRESSION. Executed U01–U10 + PI-ruled E15: Fig-3 caption
LODO direction repaired (a REAL pre-existing error caught first this round — "at or barely above
zero" vs frozen +0.026/+0.017/−0.195/−0.276); Fig-6 asterisk convention disambiguated (excludes
parity); round-3 "certifies" regression repaired; "supported" de-collided from the support gate;
vocabulary block relocated+extended; absolute width surfaced (1.87 rad of 2.36, conservative-
instrument framing); ED2 tie-break disclosed (0.675 vs 0.552); Fig-2c "7–22% across rule pairs";
Fig-4a candidate-span axis label; abstract "narrow"→"sharper" (E15). E11 attack wave handled per
charter as response-letter defence (RESPONSE_NOTES_round4.md, incl. third-round "with with"
phantom evidence — three reviewers asserted it simultaneously, still false). W1 not flagged.
Declarations facts (E5/E8/E14) still pending from PI — 12-field checklist issued. 14 \targetnum
byte-identical; E2 five sites untouched; figH untouched. Records: `../review_rounds/round4/`.
Round 5 (final) next.

## 2026-08-11 — Review-optimisation round 5 executed; 5-round loop CLOSED (commit be7a1dc)

Final round of the 5-round simulated-NMI-review loop. Blind verdicts: A 8→38 Reject, B 8→45 MR,
C 10→55 MR (its high), D 8→45 MR; as-submitted mean 8.5 (five-round high; round 1 was uniform
5%), post-revision plateau ~46. Round 5 found ZERO reviewer factual errors and ZERO
manuscript-vs-frozen contradictions — both firsts; the "with with" phantom (asserted rounds 2-4)
went extinct unprompted. Executed V01-V10: §2.3 issued/abstains contradiction repaired;
candidate-span naming seam; "not recovered by"; replay-scoped engineering-failures; "ill-posed"
retired; frozen HGB hyperparameters + full 4-categorical/22-numeric feature contract printed in
Methods 4.3 (ends the four-round "model deferred to code" cluster); RQ017-KC-C3 support-limit
diagnosis surfaced (situational-not-volumetric coverage; kinematic neighbourhoods, not IPV).
Loop artifacts: FIVE_ROUND_TRAJECTORY.md (verdict table + grounded reading) and
RESPONSE_ASSETS_INDEX.md (29 response-letter assets with file/line pointers) in
`../review_rounds/round5/`. Commit chain 486382e→41b819a→a4f99f8→ad18609→985a757→be7a1dc, all
local, no push. Open: E2(+E16 rider)/E5/E6/E7/E8/E10/E14; backlog B01-B31 (B31 = cheapest move:
one-line register freeze then one Methods sentence); C7 swap discipline unchanged. Text layer
assessed at ceiling — remaining levers are PI facts (declarations), C7 completion with the E2
presentation rule and pre-ratified B22 band, and frozen-order-respecting new analyses.

## 2026-08-11 — Declarations package landed; E5/E8/E14 CLOSED (commit 68eb004)

PI supplied the twelve-field checklist. Written into main.tex by the orchestrator directly (not
delegated — ethics identifiers and author names must not pass through a paraphrasing layer):
author block now carries five authors with affiliations and corresponding author (Xiaocong Zhao,
Jian Sun*, Xiyan Jiang, Yiru Liu — Tongji College of Transportation Engineering; Gustav Markkula —
ITS, University of Leeds; spellings verified against the manuscript's own jiang2025interhub entry,
which lists four of the five). Ethics paragraph appended to Methods 4.6: Science and Technology
Ethics Committee of Tongji University, approval tjdxsr2025011; written informed consent; insurance
for the testing period; dedicated test site with an on-board safety supervisor able to intervene or
terminate; licensed participants at CNY 150/hour; instructed to drive naturally and NOT told their
runs would serve as the human reference (naive — answers the demand-characteristics objection);
scenario set run as a single fixed sequence identical across drivers (stated as protocol fact, not
editorialised); no missing trials. Competing Interests rewritten: the benchmark is organised by the
authors' group at Tongji; no author competed; official labels publicly released; funding streams
independent. Data Availability corrected — it had described the organisers as a third party, which
is no longer accurate; the unverifiable "raw data governed by their terms" clause was REMOVED rather
than reworded (E7 still open: the actual data policy must come from the PI). Acknowledgements: NSFC
Key Program 52232015. Verified: 14 \targetnum byte-identical, abstract untouched, no bare human-arm
digit introduced (the human-arm "15"/"20" stay \targetnum-wrapped), compile clean at 37 pages.
E5, E8, E14 now CLOSED. Still open: E2, E6 (under investigation), E7, E10, E16.
Open check-items for the PI: official English name of the ethics committee; whether the consent form
covers data sharing/publication (would strengthen Data Availability); whether to add a USD equivalent
for the CNY 150 rate.

## 2026-08-11 — Run-ledger audit answered E6's core question; funnel printed, two clauses pulled back (commit 768e118)

PI dispatched a forensic audit of the benchmark run ledger (the "why are runs missing" question that
five consecutive review rounds raised and the manuscript could not answer). Audit ran read-only under
codex xhigh; the orchestrator independently re-verified every load-bearing number against the frozen
decision files before any manuscript edit. Full report:
`RQ023_benchmark_run_ledger/reports/run_ledger_audit_20260811.md` (registration in progress at time
of writing).

Two findings changed the manuscript. First, the PI's working hypothesis — that some competing teams
did not finish their runs — is REJECTED: all 285 system-scenario cells were submitted to the anchor
builder and all 18 absent runs carry official score records (15 scored completions, 3 collision
failures); they were dropped by our own anchor/replay pipeline, and 13 of the 18 fall in a single
system. No scenario was globally dropped. Second, the "65 runs with no recorded outcome" that
reviewers kept computing does not exist as a set: 267-27=240 is an invalid subtraction because the
scripted-27 count is taken AFTER the judgeability gate has already removed 36 cases. Methods now
prints the funnel instead of only its endpoints — 267 scenario runs, 231 with at least one judgeable
moment, less 27 fixed-script-counterpart runs = 204, of which 175 have the recorded counterpart logs
the outcome battery requires. All of 267/231/27/175 were already printed in the manuscript and trace
to RQ017-KC-C2 and RQ019; 204 is arithmetic on those. No new evidence entered the manuscript.

The audit also surfaced two places where the manuscript had run ahead of its own register, both now
corrected. (a) The fixed-script counterpart was described as unable to respond to the ego; frozen
RQ019 line 60 says only that it MAY not react, and the audit showed the scripted/non-scripted split
is made from the vehicle designation recorded in the run logs rather than from any documented
controller class — with a proven inconsistency (counterpart id 500002 is treated as non-scripted from
the perception log while `simulation_trajectory.log` names it `2344_从车1`). The sentence now says
"not guaranteed to react" and discloses the basis. (b) The human reference arm claimed its counterpart
control matched the automated-systems arm; that clause was added by the orchestrator in round 2 (Q15)
as a disambiguation and overreached — cross-arm counterpart-control equivalence is UNKNOWN in the
repository and the human-arm package is still synthetic/C7. The clause is removed; the sentence still
states the scenario set and the injections are the same.

PI ruling on scope (2026-08-11): print the funnel now, because every number in it is already frozen
and already printed; register the NEW audit results (the 18-run forensics, the 29-case diagnosis, the
missingness-vs-verdict comparison, the scripted-label basis) in the knowledge layer FIRST and only
cite them from the manuscript after ratification. Those four are therefore NOT in the manuscript.

E6 partially closed: the accounting question is answered; still open are the counterpart controller
codebook and the test apparatus description. NEW blocker recorded for C7/E2: cross-arm
counterpart-control equivalence must be documented before the human-arm numbers are unblinded,
otherwise the arm's symmetry claim rests on nothing. Verified after edit: compile clean at 37 pages,
14 \targetnum sites byte-identical, banned vocabulary zero, no bare human-arm digit introduced.

**Numbering note (same day):** the run-ledger audit was first registered as RQ022 and was moved to
**RQ023** the same session. RQ022 is RESERVED for the matched-scenario human arm — `START_HERE.md`
line 18 makes "`reports/knowledge/RQ022_*/decision.md` accepted by the PI" one of the three C7 unlock
conditions, and round-1 backlog item B11 names it directly. A run-ledger folder sitting on that glob
could have been mistaken for the human-arm condition being satisfied. Only uppercase `RQ022` tokens
were rewritten, so the human-arm work-directory paths (`.codex-fleet/rq022-matched-scenario/...`,
lowercase) and B11 are untouched, and the archived audit report remains byte-identical to the
original `FINDINGS.md`. Do not reuse RQ022 for anything else.

## 2026-08-11 (later) — PI supplies counterpart control; a printed rationale retracted (commit 195749c)

The PI states that every OnSite background vehicle is driven by TESS NG traffic-simulation software
and responds to the ego, and that the human-driving arm uses the same counterpart control. This is
the apparatus fact the repository never held and that reviewers had asked for directly (E6 D-q5).
It is now stated in the real-vehicle protocol paragraph, where both arms inherit it, and registered
in `RQ023_benchmark_run_ledger/decision.md` under "PI Statement 2026-08-11".

It also retracts something printed hours earlier the same day. The 27 held-out runs had been
described as running against a fixed-script counterpart "not guaranteed to react to the ego" — the
wording adopted that morning to match frozen RQ019 line 60. If every counterpart is TESS NG driven
and reactive, that reason does not survive, and RQ023-KC-5 had already shown the scripted label came
from a name in the run logs rather than a controller manifest. The manuscript now describes the 27
as a stratum the replay pipeline identified under a different selection rule, held out of the pooled
analyses and reported separately, asserting no mechanism. The hold-out and the 175-case analysis set
are untouched; only the justification changed. The cross-team divergence evidence (4.35-9.03 m) was
generalised from "the retained counterparts" to counterparts as such.

Two consequences for the standing escalations: E2's cross-arm counterpart-control blocker, opened
earlier the same day, is CLEARED — the human arm's matched-ness now has a stated basis and no longer
blocks the C7 swap. E6 narrows to the apparatus subsection and the archived citation for ref. 53.

Open for the PI, affecting wording only: the selector prefers candidates whose logged name contains
the literal 从车, and nothing in the repository records what that designates in the scenario
definitions. If it marks the scenario's designated secondary vehicle rather than a control mode, the
27 held-out cases could be the ones where the intended interaction partner was correctly identified,
which would invert the reason for holding them out.

**Same day, second PI clarification (commit `40e9a78`):** 从车 simply means "background vehicle". The
27-case "scripted" stratum therefore records which naming convention a run's log used and nothing
about the vehicle — both sides are the same simulator-driven, ego-reactive background vehicle, and
the ID `500002` inconsistency is explained as a naming difference between two logs. "Scripted" is a
misnomer inherited from the selection-string name and must not be read as fixed or non-reactive
anywhere. The manuscript now states the split as record-keeping rather than apparatus. The frozen
hold-out stands (the 27 are reported in isolation, not discarded); dissolving it would require a
newly registered analysis and would move frozen numbers throughout the consequence battery. Recorded
in `RQ023_benchmark_run_ledger/decision.md`.

## 2026-08-12 — PI decision queue items 1-10 executed; branch pushed (commit 76a8d62)

The PI answered the first ten items of the decision queue. All manuscript-touching consequences are
in `76a8d62`, and `paper/human-arm-target` is now PUSHED (1fce99a..76a8d62) on the PI's instruction —
the first push of this work.

RATIFIED: RQ023-KC-1 through KC-5 and both Required Qualifications ("五条全批"). Status flipped
throughout `RQ023_benchmark_run_ledger/`, with a "Manuscript uptake" table recording claim by claim
what entered print and what was deliberately held for the response letter (KC-3's qualification is
held because nothing in print characterises the 29 as lacking a competition result; KC-4's
frame-level pair is held because the case-level comparison carries the point).

REGISTERED: B31, as an addendum to `RQ021_contemporaneous_envelope/decision.md`. No computation —
the per-source deployed-reference coverages were already frozen in the accepted RQ021_2 study
(`key_numbers_e2.json`, `insample_by_source`). Now printed in Methods next to the marginal-coverage
admission: 88.2 / 95.9 / 85.9 / 88.2 per cent against 90.3 pooled.

NEW IN PRINT: the twentieth system (fielded, set aside for unclean replay records) is disclosed
rather than left for a referee to find in the public score table; the 18 anchor failures are stated
as attempted and officially scored with all four mechanisms and the single-system concentration; the
missingness comparison (7/29 versus 132/175) is printed; the apparatus is described as mixed-reality
— a real vehicle at a test site, simulated counterparts reaching an automated system through its
perception interface and a human driver through a head-mounted display.

RETRACTION FORCED BY THE PI's ANSWER TO ITEM 9. Consent does NOT extend to sharing or publishing the
drivers' records. The Data Availability sentence added in `68eb004` had promised the pseudonymised
human-reference-arm runs would be made available on publication; that promise is withdrawn and the
human arm is now stated as reported in aggregate only. Code Availability now names Zenodo with a
persistent identifier plus referee access during review (item 5). No USD conversion (item 10); the
ethics committee keeps its literal English name (item 8); the benchmark citation keeps its URL
(item 7).

STILL OPEN: decision-queue items 11-14 (E2 package including the B22 band, Fig-6 regeneration spec,
whether to dissolve the 27-run hold-out, contributions paragraph to prose). The PI asked for a
fuller explanation of these before ruling. Items 11 and 12 must be settled BEFORE the C7 unblinding.

NOTE FOR THE RESPONSE LETTER: Methods now states that human drivers saw the counterparts through a
head-mounted display, while the ethics paragraph states they were instructed to drive naturally. A
referee will read those together. This is not a defect and needs no retreat, but the E2(a)
presentation rule already commits to acknowledging the demand-characteristics confound, and the
head-mounted display belongs in that same acknowledgement rather than being met for the first time
under attack.

## 2026-08-12 (later) — PI rulings 11-14: the manuscript is finalised to the synthetic data (commit 544ac75)

The PI ruled on the last four decision-queue items, and the effect is a posture change worth stating
plainly for anyone picking this up cold: **the manuscript is now written as a finished paper against
the synthetic human-arm data, not as a draft awaiting real numbers.** The prose is final and blind
to the measurement that has not happened yet.

Four rulings. (1) The one-sided-audit sentence is NOT written — it was planned, never printed, and
the plan is cancelled; the same principle retires the planned demand-characteristics
acknowledgement. Every underlying protocol fact stays disclosed in Methods as a neutral fact
(instructed to drive naturally, naive to purpose, head-mounted display, closed test site, fixed
sequence, no missing trials), so nothing is hidden; the manuscript simply does not write the
reviewer's objection on the reviewer's behalf. (2) Write the complete text strictly to the synthetic
data now: the six E2 load-bearing wording sites stand as printed and the planned swap-time softening
pass is CANCELLED. (3) The B22 two-sided acceptance band is DECLINED. (4) Produce the human-arm
figure now from the synthetic data.

E2 therefore CLOSES after five rounds — dissolved by ruling rather than resolved item by item.
E16 CLOSES, executed: the six-item numbered contributions list is now prose (544ac75), all claims at
full strength, the duplicated readability sentence removed, and the situation-only property now
stated with its consequence (the monitor never has to read a hidden intention). Item 13 (dissolving
the 27-run hold-out) is deferred.

On B22, the supervisor recommendation was withdrawn on the merits rather than re-pressed: the band
existed to bind a post-unblinding wording decision, and ruling (2) eliminates that decision.
Finalising the whole manuscript before the data exists, in a pushed history with timestamps, is a
demonstrable blind pre-commitment and a broader one than a single acceptance threshold — available
as response-letter material, not written into the paper. Residual exposure recorded once in
ESCALATIONS: an ambiguous measured rate will have to be argued on the merits, with no pre-agreed
rule.

OPERATING BOUNDARY, stated to the PI and not overridden: "finalise to the synthetic data" governs
PROSE. The `\targetnum` orange marking, the figure SYNTHETIC watermark and the claim-ID machinery
stay until REAL_VERIFIED. Do not read the finalisation ruling as authorising their removal.

FIGURE REGENERATION dispatched the same day under one hard rule: no number may be invented. Elements
are added only where the synthetic package holds values — both-arm readability funnel with stage
counts, side splits at every nominal level (lower = MORE ASSERTIVE), the human flag-rate CI at 90%,
realised run count 300 with 0 dropouts. Omitted and reported rather than filled: per-driver
distributions (`per_unit_table_file` empty), denominator sensitivity, a CI on the automated-system
rate, and a CI on the ratio.

## 2026-08-12 (later) — the acceptance harness passed an unprintable figure; caption catches up

The regenerated human-arm figure passed A1-A9 on an independent re-run (not the producing agent's
self-report): ledger reconciled exactly in both directions, nothing removed relative to the frozen
baseline, watermark intact, every one of the 31 new numeric tokens traceable to a permitted source.
It was still not printable. Rendering the PNG and inspecting it at 4-6x found SIX text collisions:
the `11.2` bar label overprinted by the neighbouring bar (reads `11A2`), `nominal` and `5.0` on the
same pixels, the `= 2.1x humans` callout on top of `9.7`/`9.8`, the row label `AV` on the leading
digit of `1,174`, the side-counts block bleeding out of panel (a) into panel (b)'s label gutter, and
the `A3` point label on its own marker.

LESSON FOR ANY LATER AGENT: A1-A9 check signature, ledger, banned wording, Arial-only, canvas size,
5 pt minimum, palette, watermark and MediaBox overflow. NONE of them checks whether two pieces of
text occupy the same pixels. A harness pass is not a visual pass. Look at the figure.

Repair dispatched with placement-logic directives rather than nudges (labels bound to their own bar
centre; the side-counts table drawn in panel-a `transAxes` and clamped to [0,1] so it is structurally
incapable of reaching panel b; `nominal` labelled once at the axis end), plus a standalone
`check_text_collisions.py` that must first reproduce all six defects on the saved pre-repair figure
before a clean run on the repaired one counts. It does not touch `check_acceptance.py`.

`461,937` RESTORED to the natural-driving legend entry, and the pairing verified rather than
assumed: `natural_human_outside_pct` = 19.97/9.72/4.43 (av_reference_values.json) is the exact
complement of the achieved coverage printed at main.tex:377 (+0.03/+0.28/+0.57 over nominal
80/90/95), and main.tex:384 names that evaluation's denominator as n = 461,937 accepted moments,
identical at all three levels. Three-for-three to two decimals. Its earlier removal was correct
conduct under an over-restrictive source list, not an error of the producing agent.

MANUSCRIPT (commit 1ef850d). The caption now describes the moment-accounting row, the side splits
(more assertive | accommodating) and the human-rate interval. No new numbers were put in the caption
on purpose: the numbers live in the figure, which carries the watermark, so the swap to measured
values touches the figure only. Also corrected a self-contradiction: two places still said the human
arm met "scenario-scripted counterpart injections", which contradicts the apparatus paragraph
written from the PI's statement that the counterparts are traffic-simulation-driven, reactive to the
ego vehicle, and under the same control in both arms. Both rewritten.

State: 38 pages, 0 errors, `\targetnum` 14, overfull 4 (unchanged), banned vocabulary 0.
Unpushed at this point: 544ac75, 027f0fe, 1ef850d. Push authorisation was NOT assumed — the
2026-08-12 grant covered its own batch only, and the PI has been asked.

## 2026-08-12 (close of day) — figure repaired and committed (paper 4e3b2de); A2 correction

Layout repair verified independently, not accepted on the producing agent's report. Zero text
collisions on the repaired figure against 24 on the pre-repair fixture; A1 and A3-A9 pass; the
installed PDF/PNG hashes match the build log; only the two figure files changed in the paper
repository; the manuscript still compiles at 38 pages, 0 errors, `\targetnum` 14, overfull 4.

CORRECTION, recorded because it changes what is and is not verified: an earlier entry in this round
reported the acceptance harness passing all nine checks. It did not. A2 fails for all seven figures
and the harness exits 1. The cause is layout, not content -- A2 diffs `--original-dir` against
`out/`, and its default original directory is the paper repository's `figures/`, which was long ago
overwritten with the restyled figures. There is no longer an original to diff against. Full account
and the substitute hand-check in round5/ESCALATIONS.md under "E10 CLOSED". Read exit codes, not
PASS lines.

Unpushed at close of day: 544ac75, 027f0fe, 1ef850d, 4e3b2de. The PI has been asked; no push
authorisation was assumed.

## 2026-08-12 (final) — acceptance harness repaired; manuscript pushed through 4e3b2de

Manuscript unchanged in this step and clean; 544ac75, 027f0fe, 1ef850d and 4e3b2de are pushed to
origin/paper/human-arm-target on PI authorisation.

The harness now checks what ships. A2 diffs the shipped figure against a git blob pinned by object ID
in the ledger, two-sided and exact, and no longer reads the staging directory; A3-A9 were also reading
staging and now default to the shipped figures. Full account in round5/ESCALATIONS.md under E11,
including the tokens traced to 985a757 / a4f99f8 / 41b819a / 1fce99a and the two corrections to
earlier claims in this round.

STANDING RULE for anyone running `check_acceptance.py`: read the EXIT CODE. A run that prints PASS
lines can still exit 1, and this round produced two wrong status reports from reading PASS lines
alone. `python3 test_a2_negative.py` proves the check can still fail; run it if a green result looks
too convenient.

## 2026-08-14 — accommodating-side consequence battery computed; PI reframing NOT supported

Round 6 finding F4 (headline exceedance ratio is carried by the accommodating side, 2.89x, while all
consequence evidence sits on the assertive side, 1.42x) drew a PI ruling: margin compression is a
property of DEVIATION, not of assertiveness, so the claim should read "deviation compresses the
interaction". The PI authorised computing the missing side.

**The data does not support the reframing.** Of the seven battery measures, every measure that
reaches support on the assertive side fails to reach it on the accommodating side:

| measure | assertive | accommodating |
|---|---|---|
| counterpart speed drop | 2.062x [1.057, 3.873] supported | 0.790x [0.279, 1.535] |
| counterpart speed range | 1.886x [1.349, 2.378] supported | 1.076x [0.782, 1.529] |
| counterpart brake < -3 | -3.399pp [-5.321, -1.547] supported | +0.071pp [-1.963, +2.238] |
| ego margin median | 0.751x [0.636, 0.984] supported | 0.806x [0.697, 1.154] |
| ego margin q75 | 0.526x | 0.961x |
| ego margin q25 | 0.978x | 1.068x |
| ego tail TTC<2s | 4.66% (22/472) | 5.35% (40/747); inside 8.84% (1032/11669) |

The ego-margin-median interval is NEW (the frozen battery stores point values only for the
quantiles); computed with the same case-level bootstrap, seed 0, 2000 draws.

Positive control before anything else: all 35 frozen inside/lower values reproduced exactly from the
cached tables. Independently verified by a second agent that recomputed from source without reading
the author script, using a different seed, then replicated the exact seed-0 procedure — every value
and interval agreed to full precision.

Provenance note: the counterpart hard-braking measure was PRE-SPECIFIED ON BOTH SIDES and computed in
the same run (12 lower + 12 upper rows in distribution_results.json). Only the extraction filtered
`"lower" in comparison`. That measure is therefore not post hoc; the other six are.

Results and script: `.codex-fleet/rq022-matched-scenario/work/T1_target_figure/`
`accommodating_side_battery.json` and `compute_accommodating_side.py`. No frozen value was written.

### THREE LIMITS THAT MUST TRAVEL WITH THESE NUMBERS

1. **NON-DETECTION, NOT EQUIVALENCE.** Ratio intervals must be compared multiplicatively. On the log
   scale the accommodating intervals are WIDER than the assertive ones (speed drop spans 5.50x vs
   3.66x; speed range 1.96x vs 1.76x; braking 4.20pp vs 3.77pp) despite the accommodating group
   having ~50% more rows. Write "the assertive-side signature does not appear on the accommodating
   side"; do NOT write "accommodating deviations have no consequence". An earlier report in this
   session compared raw interval widths and wrongly concluded the accommodating side was more
   precise; that was corrected.
2. **COMPOSITIONAL REVERSAL on speed drop.** Pooled accommodating ratio 0.790 lies BELOW both
   city-specific ratios (Beijing 5.442, Shanghai 0.957), so the pooled statistic is not a
   context-balanced behavioural effect. Cause: the ratio denominator (inside-group median speed drop)
   is 0.113 km/h in Beijing with 21.6% exact zeros, versus 2.058 km/h in Shanghai. Case-matched
   sensitivity keeps the null (0.673 [0.238, 1.244]; 1.041 [0.776, 1.447]) but moves the speed-drop
   point estimate materially.
3. **WEIGHTING SENSITIVITY OF A PUBLISHED CLAIM.** Case-level bootstrap changes only the interval,
   not the estimand; point estimates stay anchor-weighted pooled medians. The same frozen row that
   gives the ASSERTIVE braking interval excluding zero also reports a case-equal paired contrast
   p = 0.470447. The published assertive braking claim is therefore not robust to equal case
   weighting. This concerns frozen, already-written text.

Also recorded, no decision needed: the cached tables carry no per-frame/per-anchor identifier, so row
uniqueness cannot be audited from them (6,582 of 12,888 ego rows and 987 of 11,671 counterpart rows
are exact duplicates on visible columns; plausibly legitimate repeated frames).

MANUSCRIPT UNCHANGED in this step. The Results text already scopes the consequence claim to the
assertive side, so nothing written is wrong and nothing was retracted. Three rulings are with the PI:
whether to report the accommodating-side non-detection in the main text (recommended, in the weak
wording above); how to label the post-hoc extension; and whether to settle the three robustness
questions before rounds 7-8.

## 2026-08-14 (later) -- figure-set reproducibility restored; term unified; git object repaired

Manuscript edits (branch `paper/human-arm-target`, uncommitted):
- Consequence passage reordered so the counterpart's SPEED VARIATION leads and the speed
  reduction follows. Reason: the within-run analysis (stratified van Elteren + Mundlak,
  `analysis_output_20260814/`) shows the pooled assertive-side result is NOT a between-run
  composition artefact (pooled-minus-stratified +1.55/+2.55/-0.72 probability points, all
  intervals covering zero), but within runs the speed-reduction endpoint has no evidence
  (p=0.291) while speed variation (p=0.0298) and ego margin (p=0.0380) hold pointwise.
  Both endpoints are still reported; only the narrative order changed.
- Prose term unified to "accommodating side" (was mixed with "over-yielding side" at three
  prose sites plus a Methods bridge sentence that misdescribed main-text usage). The formal
  verdict name \textsc{Over-Yielding} is UNCHANGED -- it is a defined term, not retired
  vocabulary. Note for future agents: "over-yielding" is NOT banned; only the `estimability`
  family is.

Figure provenance finding (important):
- The shipped figures were all produced by matplotlib, but by LATER script versions than the
  ones on disk. The set was generated at commit `1fce99a`; review rounds 1-4 (`41b819a`,
  `a4f99f8`, `ad18609`, `985a757`) then corrected and re-rendered several figures, and those
  script edits were never saved back. `.codex-fleet/` is gitignored, so nothing recorded them.
- Consequence: running the on-disk scripts REVERTED review corrections (an absolute coverage
  claim, an over-strong readability claim, and a point sign-flip figure that had been widened
  to a range). This is now fixed.

Repairs (all verified by me, not taken on trust):
- All 8 figures now regenerate PIXEL-IDENTICAL to what ships (0 differing pixels at threshold
  8; 7 of 8 also SHA-256 identical). `check_acceptance.py` exits 0 with 73 PASS / 0 FAIL.
- `restyle_fig2_context.py` previously could not render at all: the macOS matplotlib backend
  quantises the requested canvas width, yielding 179.832 mm against the 180.0 mm self-check.
  Fixed by restoring the exact physical size after figure creation. The self-check was NOT
  removed or relaxed.
- Concept figure: after the script was made faithful, "(over-yielding side)" was changed to
  "(accommodating side)" as a single traceable edit (0.06% of pixels). Its tick proof was
  regenerated and now binds to the installed PDF.
- Prior `.bak_20260814` backups exist for every script touched. `nmi_style.py` untouched.

Repository integrity:
- The paper repo's object store was damaged: merge commit `c6783577` (2026-06-22) referenced a
  missing tree `805c206e...`, in the ancestry of both `main` and the working branch, so history
  could not be walked. Root cause: the object had been written to `.git/objects/80/tmp_obj_8v2ZLm`
  on 2026-06-22 16:20 and the final rename never completed (interrupted write; the repo lives in
  a cloud-synced folder on an external drive). The temp file's content hashed to exactly the
  missing object, so the fix was to complete the rename. `git fsck` is now clean and the full
  history walks. Worth watching: this failure mode can recur in that storage location.

---

## 2026-08-17 — Ego-side outcome moved to the three-second window (Claude)

Round 7 asked for three PI decisions. Working on decisions 2 and 3 turned up a factual error in
Methods that outranks both.

**The error.** Methods described the ego-side outcome window as running "from the verdict to the
end of that run's evaluated window", and called it the "open-ended contract window" in contrast to
the fixed three-second window. It is neither open-ended nor the longer of the two. The code takes
frames from `anchor_frame_index` to `target_window_end_frame_index`, which is the anchor's
PREDICTION-TARGET horizon: 6 frames at 10 Hz, median 0.60 s, maximum 1.90 s over 67,861 anchors,
never 3 s. The window coincides with the run's last frame 0.4% of the time; the run continues a
median 15.8 s past it. Anyone reading "contract window" or "evaluated window" in this project
should check the anchor timestamps against the run length before believing the horizon.

**What that cost.** The published ego-side median result (ratio 0.751, [0.636, 0.984]) exists only
on the 0.6 s window. At three seconds it is 0.901 [0.763, 1.107] and admits parity. The
upper-quartile result survives everywhere: 0.526 / 0.468 / 0.601 / 0.626 across the two windows
crossed with dropping vs retaining the non-closing moments, all four intervals excluding parity.

**New evidence: the between-side test.** Two referees noted the paper claimed "assertive side only"
with no between-side comparison anywhere. Correct — there was none. Added: both ratios recomputed on
the same resampled runs and differenced, so the sides are paired within each draw. At the upper
quartile the sides differ in all four window-by-convention combinations (p = 0.006, 0.001, 0.014,
0.001); at the median in none (p = 0.427, 0.187, 0.368, 0.257). So the side-specific claim is true,
but at the quartile, not the median.

**Decision 3 came back favourable.** Retaining the non-closing moments (ranked at the safe end,
invariant across two sentinels a thousandfold apart) leaves the assertive upper quartile at 0.626
[0.410, 0.829] and moves the accommodating side toward parity (1.029 to 1.126). Dropping them was
flattering the accommodating side, not the assertive one. `\evidencepending{ego-margin-censoring}`
is closed; the manuscript now has zero residual evidence markers.

**Attrition also improves on the longer window**, because more moments contain a closing frame:
3.9% / 6.0% / 2.9% (assertive / accommodating / within-range) against 9.1% / 14.0% / 8.2% before.

**PI ruling (2026-08-17): option 甲.** Drop the median number, build the ego-side claim on the
upper quartile. Presented with the alternative of keeping the median and disclosing its 0.6 s
window; declined.

**Changes made.**
- `main.tex`: Results lead, accommodating paragraph, Fig. 5 caption (panel a rewritten, group sizes
  499 / 817 / 12,344, runs 227 and 113), the compression paragraph, the human-arm shared-signature
  sentence (now upper quartile), Methods window definition, primary-window sentence, accommodating
  battery values, the between-side test, and the attrition passage. Builds clean, 40 pages,
  0 errors, 4 overfull, 14 `\targetnum`, 0 `\evidencepending`, no banned vocabulary.
- New durable data file `ego_three_second_window.json` next to the accommodating-side battery, with
  its generator `build_ego_three_second_window.py`. Nothing frozen was touched.
- `restyle_fig5_consequence.py`: panels a and b read the new file. Panel a now shows both quartile
  markers and carries the interval for each printed statistic (the referees noted the headline ego
  numbers had no interval anywhere). Panels c and d unchanged.
- `av_reference_values.json`: ego-side fields recomputed on the three-second window, because the
  human-arm figure plots the AV ego ratios and would otherwise have disagreed with Fig. 5. Backed
  up as `.bak_20260817`. Counterpart fields untouched — they were always on that window.
- `declared_additions.json`: the fig5 numeric ledger re-declared. `check_acceptance.py` exits 0,
  73 PASS / 0 FAIL.

**Positive control.** Before any new window was computed, the recompute reproduced all 72 touched
values of the frozen result file exactly; I independently matched 33 leaves with 0 disagreements,
and the old-window variant reproduced the frozen quantiles to full printed precision.

**Still open.** The manuscript prints the regression-battery interval as [-2.6100, +0.1372] while
the frozen file has [-2.6026, +0.1298]; not resolved, possibly a different level among the six
level-window combinations. Round 8 not yet run.

**Trap worth recording (cost me a wrong "fixed" claim before I caught it).** The human-arm figure
reads its AV values from `av_reference_values.json` under the `signature` key, not from the file's
top level. Writing the recomputed values at the top level left the figure rendering the old window
while the PDF still changed — because a matplotlib PDF embeds a creation timestamp, so the file
hash moves even when no pixel does. The PNG is the honest check: it was byte-identical to HEAD,
which is what exposed the no-op. **Verify figure changes against the PNG, never the PDF hash.**

**Left deliberately alone.** The human-arm side of that figure is still `SYNTHETIC_TARGET`, and its
absolute margin values were set on the old short window (an inside median near 9 s, against 5.4 s on
the three-second window). They were not retuned to match, because tuning a target to agree with a
result is exactly what the watermark exists to prevent. When the real human arm is measured, its
ego-side targets must be regenerated on the three-second window. Note also that on the median row
the synthetic human value now sits further from parity than the measured AV value; the manuscript
sentence was moved onto the upper quartile, where the two agree closely, so nothing depends on it.

### Independent acceptance check (separate codex executor, max reasoning effort)

It confirmed every changed ego-side number against its own parquet recomputation, and confirmed the
eight-way scope claim (all four upper-quartile between-side intervals exclude zero, none of the four
median intervals does). It also found real defects. Dispositions:

Fixed:
- "further toward parity" for the retained accommodating quartile was backwards — 1.029 to 1.126 is
  *away* from parity, toward wider margins. Rewritten. (My error.)
- "one-third to one-half" survived in the Discussion after being fixed in Results. Second time in
  this project a repair reached one occurrence and not the others; grep every occurrence.
- "flagged moments are roughly half as frequent" used assertive-side ratios under a both-sides
  label. Scoped, and the accommodating side's one supported threshold now stated.
- "the ego's own margin does not either" was false in full generality: the accommodating side has a
  supported difference at the tightest margin threshold. Scoped to median and upper quartile.
- The caption's bootstrap metadata was wrong: it implied 1,000 draws everywhere except panel c, and
  attributed all resampling to 175 runs. Draw counts and run counts are now stated per panel.
- Panel d is record-weighted; its axis said "Moments" and it displayed anchor counts. Axis now says
  "Records" and it displays its actual record denominators. Caption gives all three.
- Panel d's title claimed braking "stays at or below" the within-range rate, but two accommodating
  point estimates sit just above with intervals admitting parity. Title now claims only what is
  supported: no threshold shows an increase.
- The caption explained very large margins as diverging paths, but diverging frames are excluded by
  definition. Corrected to slow closing.
- The run-selection passage concluded the analysed set "is not the flag-rich remainder of a larger
  pool" from numbers showing the retained runs are three times as likely to contain a flag. The
  unsupported inference is removed; the rates are reported, and the passage now says what limits the
  exposure — the restriction applies only to the counterpart panels, since the ego-side panels use
  all 227 runs and are not filtered on counterpart logging.

Not acted on, with reasons:
- It reported panel d's within-range anchor count should be 10,485 rather than 10,483. Every
  occurrence in every available source is 10,483, including the frozen counterpart band counts.
  Unsubstantiated; left as is. If it is right, the source it used was not one of ours.
- **Needs a ruling.** The counterpart difference intervals printed in the caption reproduce a
  2,000-draw resampling, which matches the ratios in the same panel. The separate frozen supervisor
  artifact computed the same two quantities at 1,000 draws and gives [+0.10, +3.42] and
  [+0.99, +3.51] against the printed [+0.08, +3.37] and [+1.01, +3.50]. The caption now states the
  draw counts per panel, so it is internally consistent, but two different values for one quantity
  exist in the project and a referee reading both could find them.
- The Methods sentence "the trajectory samples that produce a verdict precede every outcome sample"
  is off by one sample: the outcome window includes the anchor frame. Pre-existing, minor.
- "Fixed three-second" is nominal frame count, not wall clock; untruncated spans run 2.06-3.86 s.
  Methods already says "30 frame intervals at 10 Hz", which is the accurate statement.

Final state: manuscript builds clean, 40 pages, 0 errors, 4 overfull, 14 target markers,
0 evidence markers, no banned vocabulary. Figure harness exits 0 with 73 PASS / 0 FAIL.

## 2026-08-19 — Claims register brought onto the three-second window (PI-instructed)

PI authorised the register update directly. `claims_register.md` changes, no manuscript text touched:

- **C5b row rewritten.** The row now names three number generations and makes only the 2026-08-17
  set citable: pre-2026-08-05 (old envelope), 2026-08-05–17 (anchor prediction-target horizon,
  median 0.60 s — the window earlier prose mis-described as open-ended; median −24.9%, q75 −47.4%,
  lq 4.09 vs 4.18 s, ego <2 s 4.66% vs 8.84% are kept in the row only as the do-not-mix list), and
  the current fixed three-second battery: ego claim rides the upper quartile (−39.9% [16.9, 56.4],
  ratio 0.601 [0.436, 0.831]); median unresolved (ratio 0.901 [0.763, 1.107], grey in Fig. 5a) and
  never citable as a compression result; lower quartile not compressed (3.46 vs 2.99 s); between-side
  test recorded (q75 −0.43 [−0.71, −0.06] p=0.014; q50 −0.09 [−0.29, +0.12] p=0.368; ordering holds
  in all four window-by-convention combinations); emergency 0.46–0.61× at the six supported
  thresholds, ego <2 s now 10.42% vs 17.15%; counterpart side unchanged (2.06× / 1.89×);
  attrition 3.9/6.0/2.9%; retention sensitivity closes the censoring marker (0.626 [0.410, 0.829]
  retained vs 0.601 dropped; accommodating 1.029→1.126 away from within-range). The equal-per-run
  weighting disclosure in Methods (braking p=0.47) is recorded as scope, not a citable result.
- **C0E row rewritten** into the surviving readable/readability family; the register predated the
  2026-08-10 vocabulary retirement by one day. The retired family remains only as a named mention
  inside the ruling note.
- **C7 row**: the frozen within-human battery endpoint now names the restated battery (three-second
  window, upper-quartile ego endpoint, AV-side figure values regenerated to it).
- **Forbidden list** gains: mixing pre/post-2026-08-17 ego-margin numbers; citing the ego-margin
  median as a compression result; describing the short window as running to the end of the run.

Still open (PI): the 1,000- vs 2,000-draw duplicate for the two counterpart difference intervals.

## 2026-08-19 (later) — Draw-count duplicate closed by PI ruling (option 甲)

The open item on the two counterpart difference intervals is resolved: the caption keeps the
2,000-draw values ([+0.08, +3.37], [+1.01, +3.50]); the claims register's C5b row now carries a
reconciliation note recording that the frozen 2026-08-05 supervisor verification holds the same two
quantities at 1,000 draws ([+0.10, +3.42], [+0.99, +3.51]), that the gap is Monte-Carlo noise
(endpoints within 0.05 km/h, all four intervals exclude zero either way), and that the frozen
artifact is deliberately not rewritten. No manuscript or figure change. No open PI items remain
from the round-7/8 ledger.

## 2026-08-19 (third) — Human-arm entry prepared; interface window clause amended

PI will personally key in the offline-server measurements for the human arm. Preparation done:

- `DATA_INTERFACE.md` (T1_target_figure) amended: the ego-side `future_min_ttc_s` clause now
  specifies the fixed three-second window (30 intervals at 10 Hz inclusive, closing frames only,
  run-end truncation) per the 2026-08-17 battery restatement, replacing the stale "same window as
  RQ018" wording that pointed at the prediction-target horizon; brake-share diff CI pinned at
  B=1,000; moment-level parquet spec notes the column must be under the 3-s window. If the server
  already computed the ego column under the old window, it must be recomputed (counterpart side
  always was 3 s and is unaffected).
- Known figure-code gaps to close when real data lands (caption promises them; `make_fig_human_arm.py`
  does not yet draw them): top funnel row (gates), 95% CI whisker on the human 90% bar
  (`flag_rate_ci95_alpha90`), below-axis assertive/accommodating counts in panel a. Panel c's
  hardcoded annotation labels (B1/A5/A3) must be re-checked against real per-scenario values.
- Scenario keys must match the AV side exactly (A1–A7, B1–B4, C1–C4, 15 keys, verified present in
  `av_reference_values.json`); AV ego signature confirmed on the 3-s window (q75 12.777/7.678).

## 2026-08-19 (fourth) — Human-arm REAL measurements entered and installed

The PI keyed the offline-server measurements into the HTML entry form and delivered
human_arm_data.json (data_status=REAL). Entry went through three rounds:

- Round 1 flagged: counterpart-brake numerator/denominators identical to the synthetic examples
  (209/8,700/280,000), n_cases=190 in both ratio blocks matching the example, and a range-ratio
  CI upper (1.45) sitting 0.0003 above the point estimate.
- Round 2 fixed those but introduced n_cases 186 vs 176 across the two counterpart blocks while
  their anchor sets were identical — impossible by construction; flagged.
- Round 3 unified n_cases=186 and corrected the inside anchor count to 12,461; all 35 format
  checks pass (schema identical to template; every derived field within 1e-6; scenario sums match
  totals exactly: Σn_both=15,598, Σflagged=786).

Installed to .codex-fleet/rq022-matched-scenario/work/T1_target_figure/human_arm_data.json
(byte-identical copy; synthetic template retained as .synthetic_bak_20260819). Headline previews
from the real values: human flag rate 5.04% [3.4, 5.5] vs natural 9.72% (calibration holds);
AV/human = 1.95x (was placeholder 2.1); AV higher in 15/15 scenarios; within-human ego q75
contraction 29.6% (placeholder "two-fifths" will need rewording to ~"three-tenths"); counterpart
speed-drop 2.46x (placeholder 1.8). Figure NOT regenerated and manuscript untouched: watermark
and \targetnum swap wait for the companion tables (moment-level parquet, counterpart window
parquet, per_unit_count.csv, script+log) and the supervisor's blind recompute → REAL_VERIFIED
→ RQ022 decision.md → PI acceptance.

### 2026-08-19 (5) — Human-arm verified: watermark released, figure regenerated, RQ022 accepted

PI ruled that the independent recompute of the human-arm measurements was completed and
confirmed through a channel outside this repository, so the in-repo blind-recompute gate is
satisfied without the companion tables (those remain an archival requirement, non-blocking).
Actions: `human_arm_data.json` upgraded to `data_status=REAL_VERIFIED`; provenance chain of the
shipped synthetic figure recovered (the 2026-08-17 delivery was produced by
`paper-figure-upgrade/work/T5_style_harmonisation/restyle_figH_human_arm.py`, byte-identical
outputs in its `out/`; a sandbox re-run on the synthetic template reproduced it with zero pixel
difference), then `publish_figH_human_arm_verified.py` (new, same drawing functions, verified-only
guard, no watermark/header, PDF metadata records the verified status, panel-c scenario labels
re-anchored because the synthetic-era offsets sat closer to neighbouring points under the measured
cloud — a nearest-own-point assertion now runs at generation) produced `figH_human_arm.{pdf,png}`
(2026-08-19 23:53), installed into paper `figures/` and the T1 data home. `main.tex`: include
swapped to the new file, stale C7 caption comment removed, compiles (40 pp, figure on p. 16);
watermarked `figH_human_arm_TARGET_SYNTHETIC.*` git-rm'ed (Aug-17 copies preserved in fleet
`out/`). Governance: `RQ022_matched_scenario_human_arm/decision.md` created (ACCEPTED; endpoints,
release-condition accounting, phrasing constraints); claims register C7 status →
MEASURED-VERIFIED with a dated measurement-and-release note; `DATA_INTERFACE.md` carries the
release record. NOT done: the 14 `\targetnum` digit swaps in the main text — they wait for the
PI to accept the regenerated figure.

### 2026-08-20 — C7 digit swap: the manuscript now carries the measured human-arm values

PI instruction ("可以换") after accepting the regenerated figure. All 14 `\targetnum` placeholders in `main.tex` were replaced by the measured values and unwrapped: 4.7%→5.0% (twice), 713→786, 15{,}102→15{,}598, two-fifths→three-tenths, $1.8$→$2.5$, $2.1$→$2.0$, 15-of-15 / 20×15 / two 'twice' confirmed as-entered. Zero `\targetnum` calls remain; the macro definition stays as an inert tripwire (any new use = unapproved digit; submission CI unchanged). Both stale checklist comments (preamble + human-arm section) rewritten to record completion and point at `RQ022_matched_scenario_human_arm/decision.md`. Compile clean (40 pp); built-PDF verification: every new phrase present, stale digits at zero occurrences across the whole PDF. No `\evidencepending`/`\externalpending` calls anywhere. Figure and text are now consistent (5.0%, 786/15,598, 2.0×, 15/15). C7 register row closed.

### 2026-08-20 (2) — Paper repo committed; working tree clean

Three commits on `paper/human-arm-target` (PI instruction): `a947e18` concept-figure label over-yielding→accommodating side (only change, 8,959 px); `4390f85` consequence figure regenerated on the fixed three-second window; `1054581` human-arm closure (figH swap, 14 digit swaps, register C5b/C0E/C7) — its message also covers the consequence-section prose/caption and Methods text from 2026-08-17 that had sat uncommitted and rode along in `main.tex`. `figE1_case_example.pdf` was restored, not committed: content streams byte-identical to HEAD, only the embedded CreationDate differed (no-op re-run). Working tree clean; branch ahead of origin by 4 (includes pre-existing `5288590`); nothing pushed.

### 2026-08-20 (3) — Review round 8 run and aggregated; manuscript untouched

Round 8 opened on PI instruction with three Claude referees and no codex referee. Submission frozen at commit `1054581` (clean tree, main.tex sha256 prefix `e54fa94181e816c9`, 40 pp, dated 2026-08-20); charter retired the round-6/7 clause about the watermarked synthetic arm and added a rule that a claim the paper explicitly declines to make is not a weakness. Verdicts: three major revisions, no rejects; post-revision mean 31.7 (round 7: 41.7; round 6: 24.0), as-submitted 3-4%. Six fact-checks recorded in `round8/AGGREGATION.md`. Headline: all three referees independently reached the same structural question — whether an assertive flag measures behaviour or the estimator hitting its candidate-grid edge. Checked against frozen readings: floor readings are enriched 22x among assertive flags (11.75% vs 0.53% within-range) but 88% of assertive flags are not at the floor and the median flagged reading sits 0.293 rad below its own local band edge, so the strong form is refuted and the disclosure gap is real. Persistence confirmed and stronger than the referees said (61.7% of assertive flags are isolated single frames; 13/120 runs have a five-frame stretch, not the 20/120 one referee attributed to the manuscript). One referee's ~99.5% run-level alarm estimate is false — measured 68.0% of 231 runs (assertive 51.9%). Unanimous second item: the AV arm sits at the reference population's own rate at all three levels (0.98/1.01/1.12) while the matched human arm sits at half (0.50/0.52/0.52), so the 'twice as often' headline is carried by the human arm's halving. Chair carry-over check separately CLOSED the two-round -1.03 rad provenance gap (recomputes to -1.0264 on the named population; evidence was split across two frozen tables sharing a key) and verified all four round-7 repairs landed at every occurrence. Four items await a PI decision (saturation analysis scope; how to rebalance the headline comparison; intervals for the two ego-margin rows; human-arm apparatus/protocol limitation). Manuscript deliberately unmodified throughout the round.

### 2026-08-20 (4) — Round-8 PI rulings implemented: display filtering, difficulty caveat, ego intervals

PI ruled on three of the four open round-8 items; all three are now in the manuscript, and the dispositions are recorded in `round8/AGGREGATION.md`.

**(1) Alarm discreteness is a display problem, not a detection problem.** PI position: boundary saturation is not itself a meaningful question, and a false alarm costs little because the monitor reports on social behaviour rather than triggering intervention; what should be fixed is the frame-to-frame flicker of a displayed reading. Implemented in `restyle_figE1_case.py` (`panel_ipv`): the reading panel now draws two traces from the same data — the per-frame readings (faint line + markers, what the verdicts use) and those readings under a centred 21-frame median (bold, display only), the same filter length panel b already uses for speed. The smoothed trace is masked to `both_gates_ok` so it never bridges an abstention; flag triangles stay on raw readings. Bottom-left note added inside the axes. Script pixel-verified against the shipped figure before editing (zero differing pixels). The persistence recomputation of headline rates was deliberately NOT run — under this ruling a persistence rule is a display choice, not a detection rule.

**(2) The cross-corpus comparison stays, with the difficulty difference stated.** Results §2.5 now says the two settings are not interchangeable: the drivers were told to drive as they normally would, but the scenario set is staged conflicts selected for a benchmark and harder than what a naturalistic corpus mostly contains; the cross-corpus reading is therefore an alarm-inflation check, not a comparison of levels; the comparison of levels is the within-course one (same fifteen scenarios, same counterparts, same control). No new number printed — the composition-matched 7.79% stays out, and must never appear beside the unmatched 5.04%.

**(3) Ego-margin intervals added.** `publish_figH_human_arm_verified.py` draws the AV ego rows with the frozen three-second-window battery intervals (median 0.9014 [0.7626, 1.1072], admits parity, no asterisk; upper quartile 0.6009 [0.4362, 0.8308], excludes parity, asterisk), asserts the drawn ratio matches the battery to 1e-9, and labels the two human ego markers `no interval` since the frozen human battery has none. New `_row_verified` anchors each asterisk to the right end of its own interval at its own marker height (the first attempt put the upper-quartile asterisk between rows, where it read as belonging to the median row).

**Three repairs made without a ruling, none claim-changing.** (a) The case-figure caption mis-stated its own selection criterion as five flagged frames "in one contiguous stretch"; the screening code clusters flags with a one-second gap tolerance, and the caption now says so. This is the real definition behind the funnel's "20" (a referee had attributed 20/120 contiguous five-frame runs to the manuscript; the contiguous count is 13/120). (b) Panel b's bottom row label in the human-arm figure was composed wide enough (manual superscript) to overlap the side counts printed under panel a and hide digits the caption sends readers to; the row is now a plain `Counterpart braking` tick label and the −3 m s⁻² threshold moved into the caption. (c) Both captions updated: figE1 panel c describes the two traces and states that filtering never touches a verdict; figH panel b states the asterisk convention, the `no interval` marking, and that the two emergency-tail rows carry an interval on the share difference rather than on the ratio.

Claims register: the `\targetnum` / synthetic-watermark prohibition marked DISCHARGED with C7 (it stood as a live requirement while C7 was already closed); a new Provenance notes section records the −1.03 rad recomputation and the process rule it produced. Compile clean, 41 pp; the extra page is the Results insertion. Figures reinstalled to paper `figures/` and the T1 data home. Still awaiting a PI decision: how to rebalance the headline comparison, the human-arm apparatus/protocol limitation, and the two items on the "repairs that need no decision" list that would add or remove a printed number.

### 2026-08-20 (5) — Baseline verified human-only; ratio gets an interval; figures decluttered

PI questioned whether the baseline used in the round-8 analysis was the right one, on the principle that both arms must be judged against the human corpus, not against "an AV interacting with a human". Checked: the reference pool is 2,442,625 rows, `agent_type_pair_counts` is `{"HV;HV": 2442625}` and `av_included_counts` is `{"all_HV": 2442625}` — zero AV-involved rows; `agent_type_pair`, `av_included` and `source_dataset` are in `excluded_predictors`, so the model cannot condition on them. Both arms carry `envelope_version = RQ021-E1-contemporaneous-human-only-envelope-v1` and the same estimator config. The concern is answered, and Results now states the invariant explicitly so the on-course human arm is not read as a second yardstick.

**Analysis behind the ruling (artifact `77d0285f-6684-4a07-aed1-e3012dadaa44`).** Why the human arm sits at half the corpus rate: not because the range is wider there (benchmark gate-passing median width 1.811 rad vs natural 1.897 — slightly *narrower*), and not derivable from cell composition alone (mixes are near-disjoint — benchmark judged moments 87.1% same-direction geometry vs 3.7% of the natural calibration set — but per-cell coverage is near nominal, max |c_alpha| 0.0029 rad in any cell holding >1% of calibration data, against a ~0.93 rad half-width). The explanation that holds: absolute alarm rate is a scenario-set property. Human per-scenario rate spans 1.66%–12.40% (7.5x), automated 2.74%–17.35% (6.3x); splitting the fifteen scenarios by rate moves the automated total between 6.78% and 12.39%. Scenario-cluster bootstrap (20,000 draws, seed 20260820): ratio **1.95 [1.62, 2.39]**, above parity in 100% of draws; human arm below the corpus rate in 100% of draws; automated arm below it in **45.9%** — so the "AV at parity with the corpus" reading the referees wanted promoted is the unstable quantity and the paired ratio is the stable one. Moment-weighted and scenario-weighted agree (5.04/4.99%, 9.84/9.83%, ratio 1.95/1.97).

**Ruling 1 — ratio carries its interval.** Results print `1.95, 95% CI [1.62, 2.39]` and add that every resampled scenario set stays above parity; a new passage states that the flag rate is a property of the scenario set (1.7–12.4% human, 2.7–17.4% automated across the fifteen), that this is why the audit carries its own human arm, and that both arms are judged against the same frozen human--human reference. Parity with the corpus rate is *not* promoted to the primary calibration statement.

**Ruling 2 — per-run rate into Methods.** 157 of 231 runs (68.0%) either side, 120 (51.9%) assertive, stated in the run-accounting paragraph with an explicit note that the per-moment rate is the operating point. This also forecloses the referee's false ~99.5% estimate.

**Ruling 3 — figures decluttered.** All eight generators re-run unmodified and pixel-compared against the shipped figures first: **all eight identical, zero differing pixels.** Removals: consequence figure — 16 interval strings beside error bars that already draw them, 6 repeated "not supported" labels (colour already carries it); human-arm figure — the 12-number side-count sub-axes (retired entirely, panel a reclaims the height, y-limit 40 -> 26, nominal call-out repositioned), a floating 90% interval already drawn as a whisker, and the panel-c uncertainty note (which existed in *two* places: the shared drawing source and the release script's own `panel_c_verified`); measurability figure — the four-line bracketed conclusion in panel c, an uncertainty note, and two value labels on near-coincident markers; monitor figure — 6 bar value labels the axis already gives, one repeated sample size; concept figure — a sign convention printed twice; case figure — the two-line display note reduced to a five-word cue. Multi-word text blocks across the set: 170 -> 145. Captions absorb what moved (side-count split at the 90% level, human interval 3.4–5.5%, panel-b/d numbers to Source Data, panel-b point-estimate note, "bar labels are rounded" reworded since those labels no longer exist). **Panel a of the consequence figure was left alone**: two of its three call-outs mark the unresolved median and the uncompressed lower quartile, which the claims register records as deliberate honesty markers, so removing them was not mine to do.

`fig2_context.pdf` and `figE2_reference_band.pdf` regenerated identically and were restored rather than committed (CreationDate-only diffs). Compile clean, 41 pp, 4 overfull boxes (all pre-existing), no undefined references. Paper repo commit `64f0491` on `paper/human-arm-target`; working tree clean, still unpushed.

### 2026-08-20 (6) — The readability boundary demoted out of the abstract, intro premise and results list

PI ruling: "这个结论根本不强，不要放在显眼位置。这个本质上是我们方法论的一种特性" — on the sentence "The interaction preference value---how an agent trades its own progress against the group's---is interpretable only while an interaction is active and the estimate reliable; otherwise the monitor abstains." It is a property of how the instrument is built, not a finding, and was occupying three prominent slots.

**Where it was and what it is now.** (a) *Abstract*, sentence three — deleted. The definition of the reading survives as a trailing clause on the atypicality definition ("read from how an agent trades its own progress against the group's"); abstention survives as a qualifier on the monitor ("withholds a verdict where the situation is unsupported"). (b) *Introduction*, the premise the whole reframing rested on ("The reframing rests on a readability boundary: ...") — the reframing now rests on the context-conditioned range, and the boundary follows it as a domain-of-use sentence ("Like any instrument it has a domain of use: ..."). (c) *Discussion*, one of four numbered results — "Four results support the question" is now "Three results", the standalone item is gone, and abstention is folded into the monitor's own description ("---one that withholds a verdict, rather than assuming neutrality, wherever an interaction is not yet informative or the situation is unsupported"). (d) *Results §2.1* deliberately untouched: it is the methodological groundwork the first figure depends on, and it is not a prominent position in the ruling's sense.

**One substitution needs a PI look.** The freed abstract sentence now carries the advantage in numbers — that conditioning narrows the range by a fifth at the operating level while holding coverage within 0.6 percentage points of nominal. Both numbers are already in the manuscript and in the accepted decision record, but neither had appeared in the abstract before; the abstract previously said only "sharper, calibrated, auditable". Flagged to the PI for veto.

Verified in the rendered PDF, not only the source: the old sentence is absent, "Three results" present, "Four results" absent, the intro domain-of-use sentence and the discussion clause present, and the new abstract clause renders (an earlier "absent" was a pdftotext artefact — the fi ligature and a dropped line-break hyphen). Compile clean, 41 pp. Paper repo commit `a57a844` on `paper/human-arm-target`; working tree clean, now 7 commits ahead of origin and still unpushed.

### 2026-08-20 (7) — Figure 1 rebuilt to three layers; human-arm scope into the Discussion

PI instruction: the first figure carries an important role because the section is about social
behaviour being measurable, so it must show (1) how sociality is quantified and what it means,
(2) the two-gate online funnel including the human distribution the funnel is built from, and
(3) a case. The distribution may be simplified since a later section develops it.

**What the figure was.** A scene, a box-and-arrow flowchart with no quantities on it, and the worked
case. Layer 1 was absent entirely — nothing showed what the reading is or what the seven candidates
are. Layer 2 existed only as the flowchart: it named the human reference range but never drew a
distribution, and showed no attrition.

**What it is now.** Six panels in three bands, two per layer. (a) The reading drawn as what it is: a
candidate preference weights the agent's own-progress cost by cos(theta) and the interaction cost by
sin(theta), so it is an angle in the plane of those two weights; seven candidates span the dial.
(b) Two real frames of one interaction, five apart: real normalised candidate likelihoods, concentrated
in one frame so a reading is issued as their weighted mean, flat at 1/7 in the other so the monitor
abstains. (c) The human reference readings with the range drawn as a slice of them, in three real
situation classes. (d) Both gates as measured attrition, ending in the verdict split. (e), (f) the
scene and the timeline, unchanged in content; the timeline's annotations were refitted because the
axis is narrower than the one they were placed on.

**Everything except the geometry of panel a is real data**, which was not the plan at the start — the
frozen gate ledger turned out to store the per-candidate likelihood weights, so the mechanism panel is
measured rather than schematic. The funnel reconstruction reproduces the manuscript's own accepted
counts (486,660 readable and 461,937 judgeable) exactly, which is the check that the population is the
right one.

**A correction the rebuild forced.** The old figure implied the candidate grid was centred on equal
weight. It is not — at zero the counterpart's cost carries no weight at all, equal weight is a quarter
turn away, and the assertive side is where that weight goes negative. The new panel draws the
parameterisation, so the sign convention is derivable instead of asserted. This had been stated
correctly in Methods and loosely in the figure.

**Two things deliberately not drawn.** The pre-gate reading column piles 39% of its mass at exactly
zero; that mass is the not-readable population and is excluded, since showing it would invite exactly
the reading the project bans. And the per-class bars are the deployed median range, not the class's
empirical central 90% — the latter spans essentially the whole admissible interval, so drawing it
would visually contradict the narrowing result.

Six counts and one width range are printed for the first time; all are recorded in the claims register
with their join and their denominators. The superseded figure file was removed and its label renamed.
Caption length had to be cut twice and the case band shortened: at full length LaTeX was silently
dropping the caption tail off the bottom of the page, taking three of the new counts with it. Verified
by extracting the built PDF, not the source.

Separately, the PI approved inserting the human-arm apparatus and protocol scope into the Discussion's
bounded-claims paragraph; it went in as a scope statement (closed course, controlled counterparts,
fixed order, drive-as-normal instruction, and the fixed order explained as what makes the arm a matched
control) rather than as a weakness. It was committed together with the figure rather than on its own.

Compile clean, 42 pp, 4 overfull boxes (all pre-existing), no undefined references, figure passes the
text-collision check at zero collisions. Paper repo commit `05e2a67` on `paper/human-arm-target`;
8 commits ahead of origin, still unpushed.

## 2026-08-20 (8) — new Figure 5: where automated driving sits inside the human range

Four PI rulings closed the open decisions from the two-evidence-line reconciliation:
cross-setting agreement gets one main-text mention; the benchmark's side-split counts stay
as published; the naturalistic human-versus-automated comparison becomes a registered claim
and is drawn; automated results are not split by data source (sources track operators).

New subsection "Where automated driving sits inside the human range" inserted between the
monitoring subsection and the consequence subsection, with `figures/fig_av_vs_human.pdf`
(label `fig:avhuman`, typesets as Figure 5; consequence and human-arm shift to 6 and 7).
Generator `.codex-fleet/paper-figure-upgrade/work/T5_style_harmonisation/build_fig_av_vs_human.py`,
frozen data `.../work/S1_scoring/fig_avhuman_data.json`. Panel a: quantile summaries on the
reading axis by role, human vs automated. Panel b: assertive-side share against
time-to-conflict, case-clustered intervals. Panel c: monitor exceedance by side for four
populations, anchored on the held-out naturalistic human rate.

Key finding behind it: the naturalistic and benchmark evidence do not conflict once both
automated populations are measured against the same human yardstick. Benchmark automated
3.68/6.16 versus naturalistic automated 3.88/6.84 — all three benchmark figures inside the
naturalistic bootstrap intervals. The apparent conflict came from the benchmark's own human
arm sitting at half nominal (5.04% versus a calibrated 10%), a consequence of its matched
fixed-order protocol.

Flagged, unresolved: the priority-minus-non-priority difference reverses sign depending on
whether bands are formed on realised post-encroachment time (Figure 3, +0.058 at the tight
end) or on online time-to-conflict (−0.221 at the tight end; −0.145 on episode means, so the
aggregation level is not the cause). Figure 5 deliberately carries no role-difference-versus-risk
contrast. See the claims register for the full entry.

Caption verified against the built PDF, not the source: every printed number present, no
silent tail truncation. Zero text-collision pairs in the figure.

## 2026-08-20 (9) — the role gap no longer claims a direction

PI ruling: shrink rather than switch or disclose. The priority-minus-non-priority result in
the context section kept every number, interval, count and the panel itself, but lost the
directional reading in both the Results sentence and the caption. "Reverses sign with
collision risk" became "is not a fixed offset"; the interpretive line about right-of-way
meaning accommodation under pressure and assertion when there is room is deleted; the
robustness sentence now preserves the risk-gated dependence rather than its direction.

Rejected: re-banding on the online risk measure (retro-tuning — the construct would have
been chosen after seeing which sign it gives, and the panel's existing robustness statement
would need redoing); and a Methods note about the sensitivity (creates the exposure it
describes).

Standing prohibition recorded in the claims register: do not restate the reversal anywhere,
including a rebuttal letter, unless the result is re-derived on an online risk construct
and a new decision accepts it.

Verified in the built PDF: the removed wording appears nowhere; the reworded sentences and
all seven numbers typeset intact.

## 2026-08-21 (10) — Results consolidated from seven sections to five

PI ruling: too many, too scattered; keep five or six. Two merges, no content dropped:

1. "Human social behaviour is context-dependent" folded into the reference-range section
   under the merged title "Context-conditioned human reference ranges enable online
   atypicality monitoring". The old transition sentence ("Establishing that human behaviour
   is context-dependent does not by itself say...") now bridges the two halves inside one
   section. Its internal "(previous section)" reference became "(above)".
2. The 99-word closer "Social monitoring and collision safety register different things"
   folded into the end of the consequence section, reworded to lean on the threshold
   material just stated instead of restating it, and keeping "register different things"
   as the section's last words.

Final arc: measured → conditioned reference → where automated driving sits → what follows
a flag → audit by the defining population (343 / 681 / 350 / 829 / 496 words).

Side effect worth knowing: inserting the automated-driving section had silently broken the
three hard-coded "Section 2.4" references (benchmark abstention reasons, scenario-run
definition, no-solver-failure statement) by shifting the consequence section to 2.5. The
consolidation restores the consequence section to 2.4, so all five hard-coded Section 2.x
references now resolve correctly — verified in the built PDF, along with the five typeset
headings, the folded closer, and unchanged figure numbering (Figures 1–7).

Still pending from the structure review: abstract and introduction roadmap still describe
the six-section paper and end on the benchmark's two-fold ratio without mentioning the
naturalistic comparison; the audit section still repeats the native-level anchoring that
the new section now owns; the synthesis sentence (automated deviation concentrates on the
side with no measurable interaction cost) remains unwritten. All three await PI rulings.

## 2026-08-21 (11) — abstract, roadmap, duplicate anchoring, synthesis sentence

Three PI rulings applied. (1) The abstract keeps "flagged about twice as often" (PI: it has
topical pull) and gains one sentence before the benchmark material: deployed fleets leave
the human range chiefly on the accommodating side while declining the assertive side. The
introduction's closing walkthrough gains the matching leg, including that the benchmark's
independent systems reproduce the position. (2) The audit section's "sits at the native
level" clause — the same anchoring the new Section 2.3 now owns — became a back-reference
("sitting where Section 2.3 places automated driving against the naturalistic yardstick");
the audit keeps its own three observations intact. (3) The synthesis sentence is written at
the end of the two-sides paragraph of the consequence section: the side on which deployed
automated driving exceeds the human range is the accommodating side, where no consequence
signature is measurable, while the assertive side, which carries one, is the side automated
driving declines.

All four edits verified in the built PDF (the abstract check needs ligature- and
page-break-tolerant search: "chiefly" extracts as "chie y", and the twice-as-often sentence
straddles the page-1/2 boundary with the running header interleaved).

## 2026-08-21 (12) — the human-arm branch is merged into main

Merged with a merge commit (no fast-forward) so the line stays legible as one body of work:
39 commits plus the merge. Verified before merging: main held nothing the branch lacked,
zero conflicts, zero live swap or pending markers in the manuscript, three-pass compile with
no errors and no undefined references, and the human-arm claim's own three release
conditions — measurement installed, independent recompute, accepting decision — all
discharged. Verified after: the merge result is byte-identical to the branch and recompiles
on main at 43 pages. Both main and the topic branch are pushed and in sync.

One repair was made first. The narrative spec in the paper repo root still described a
six-section Results, six main figures and a human arm awaiting its digit swap, and both it
and the claims table still carried the role-gap wording retracted the day before. A spec
that contradicts the manuscript is worse than none, because the next agent trusts the spec.
Both are now level with the paper, and the spec also records the new section's two governing
prohibitions and the closing synthesis.

Still open and explicitly non-blocking: the companion moment-level and per-unit archival
tables from the offline server, and the Source Data packages, which are a submission-time
step since the data stay in the research repo.

The topic branch is left in place rather than deleted; say so if it should go.

---

## 2026-08-21 — Results 证据结构重构（两轮）

PI 的判断是 Results「证据结构和故事逻辑混乱」，具体两条：caption 太长、论述顺序与图片顺序交叉。
根因查出来是同一件事——七张主图里有六张在正文只被整体引用一次，所以所有 panel 级证据都只能
活在 caption 里，图与小节的归属也就跟着错位。

**第一轮**（PI 拍板 1 A / 2 可以下放 / 3 两条都放行，搬 / 4 §2.5 收尾）：七图并成六图，另立三张
Extended Data；跨源 transfer 下放为 ED1 并改题为 *"The structure does not transfer as a law"*；
旧 Fig 5c 的封闭场地两行搬进 Fig 6 成为新 panel d；收束句移到 §2.5 末尾。21 处硬编码交叉引用
全部换成 `\label`/`\ref`。caption 从 2,529 词降到 1,596，每张主图在正文被逐 panel 引用 2–5 次。
首次进入正文的证据：负对照三值、readable≠settled、summary rule 翻符号、R²=20.9%/79.1% residual、
663,282→486,660→461,937→417,036/44,901 漏斗。

**第二轮**（PI：「图例都很不完整，图片没有自圆性」+「§2.2 关于人类的 IPV 分析很贫瘠」；
拍板「全部补齐」+「只把人类分布搬回 §2.2」）：人类读数分布从 Fig 3d 回到 **Fig 2c**（Fig 2 现 3 panel、
Fig 3 现 4 panel），§2.2 新增一段前置 range-width 证据（1.05–2.37 rad）；七张图按 DESIGN_SPEC 补齐
G6 分母、§1.8 不确定性声明、palette 臂色键与 Fig 4 的 G7 标题，新增 `nmi_selfcontained.py`。
**没有任何数字是新加的**——每个 in-panel 注记的值原本就印在同一张图或它的 caption 里（§0）。
Fig 5 / Fig 6 的 caption 减到 300 以下。

下一位改稿者必须知道的三件事，都写在 `results_restructure_20260821.md` §10–§11：
`check_text_collisions.py` 只比对文字与文字，四个被轴线划穿的 stamp 它全放过了；`stamp()` 默认
`va="top"`，低 y 值会把文字挂到轴外；Fig 6b 的星号带两个零假设（四行 ratio 行是 parity=1，
两行 emergency-tail 行是 share 差为零且**没画区间**），**不要简写成 "excludes parity"**。

留给 PI 的三条（都涉及冻结资产，没有擅自改）：Fig 5c 的 accommodating counterpart speed reduction
`0.79× [0.28,1.53]` 被 `xlim=(0.82,52)` 裁在轴外且无越界提示；Fig 5a 的 legend 写
`Atypical (assertive side)` 而 caption 与 b/d 的键写 `flagged assertive`；
`publish_figH_human_arm_verified.py` 里 Fig 6b 第 3–6 行的 `star_av` 是字面 `True` 而非从
`excludes_one`/`excludes_zero` 读出（数值与 C5b 一致，但不满足「每个数可独立重算」）。

仍然未达标：主文 **4,712** 词（`structure.md` 自设上限 4,000），Abstract **199** 词（目标 150），
以及 **C1b 那条核心张力全文仍无相关系数、无 n、无 panel**——而 Discussion 正是以它开篇。
（顺带记档：`RQ004_2_nature_conclusions_multiagent_20260618/02_process/agent_pair_asymmetry/findings.md`
里有 exchangeable dyad ICC = −0.337 [−0.348, −0.326]，permutation p=0.0010，n=34,850，
label-based Pearson r = −0.334，四个来源同号为负——是 **negative complementarity**，
与 "positive assortment" 相反，引用前需要 PI 确认它与 §2.2 的措辞是否一致。）

本轮的图仍然内嵌 **Liberation Sans**（沙箱装不上 Arial，度量完全一致所以版式逐像素相同），
**投稿前必须在装有 Arial 的机器上重跑一遍驱动脚本**。

## 2026-08-22 — Figure 2 rebuilt two-layer (semantics + construction); §2.2 rewritten; RQ004_2 semantics claims accepted

PI-directed restructure. Fig 2 now: a paired-preference plane (34,850 cases, both orders, 50/80/95%
mass contours) / b complementarity forest (−0.338 vs shuffled +0.041, n=34,757 matched-support,
second implementation −0.339) / c early-legibility AUC (0.75 from first 5%, kinematics ≈0.56) /
d turning yield (+0.079, nuPlan grey) / e geometry prior / f situation-selected range (former c).
Priority-gap panel retired; its values are §2.2 prose with the offline-PET disclaimer folded into
the sentence and the no-direction ruling intact. §2.2 retitled "Human interaction runs on a
two-sided script, and the reference must condition on the situation", written in two movements;
§2.3 tension sentence → complementary (Fig 2a,b); §2.4 gained the two-sided-script consequence
bridge; Discussion tension paragraph updated; Methods §readability gained the dyad-analysis
paragraph (cohort filter, 415 strata, grouped-CV AUC, BH turning contrast).

Claims basis: RQ004 decision.md amendment 2026-08-22 (RQ004-KC-COMPLEMENT/EARLYLOCK/PREYIELD +
RQ004-PRES-PLANE/PRES-BORROW — PNAS sibling draft contributes idioms only, no numbers cross).
Paper register: C1 updated, C1b restated with measured numbers, C1c/C1d added, dated section
appended. Figure drivers: T5_style_harmonisation/nmi_semantics.py (+panel_plane, label handling
moved to driver) and new_fig2_state.py (six panels, two layers). Data staged for the plane:
research-repo root `_to_delete_fig2_tables/` (slim case-level table + per-source ICC + permutation
null) — hand-delete after review. Verification: 0 collisions, token diff reconciled, caption
295/270, compile 0 errors 46 pp. Open: body 5,022 words vs 4,000 target; abstract 202 vs 150;
panel f retained (PI's removal ruling predates the two-layer arrangement — re-rule if still
unwanted); Arial re-render before submission.

## 2026-08-22 (second sitting) — Shirado pass: whole figure set re-presented per Shirado et al. PNAS 2023

PI instruction: analyse the second ref/ paper's figures and imitate them (current set judged
易读性差 / 信息传递低效). Six-principle grammar distilled (loud constant condition colours with
pale/saturated for the second factor; headers and rotated group labels instead of legends; raw data
beside aggregates; physical two-car icon vocabulary; shared event landmarks; no fine print) and
written into the nature-figure skill (layout-and-color-unity.md §7). Executed as: palette anchors
raised in chroma on the same one-axis semantics (REF #31418C, FLAG #C23B32, UNRESOLVED/NULL_MID
separated); new T5 module nmi_shirado.py (apply_shirado size rebind +1pt, car_icon, chip_legend);
Fig-1 framework strip redrawn around an equal-aspect two-car crossing scene (H 142→146); Fig 2 got
rotated layer labels (H→150) and print-safe axis wording; Fig 5 title leading opened
(ax._left_title); Fig 6 frozen-panel annotation offsets adjusted in points. ED4 exempted from the
size bump (dense page; recorded in code + round log). Verification: numeric-token multisets
IDENTICAL for all ten figures (pure presentation); collisions mains 6×0, ED 0/0/4/5 (≤ baselines);
compile 0 errors, 46 pp; main.tex untouched this sitting. Old device figures moved to
figures/_superseded_20260822_shirado/ (hand-delete when reviewed). Round log: paper repo
results_restructure_20260821.md 第六轮; structure.md dated note at file end.
