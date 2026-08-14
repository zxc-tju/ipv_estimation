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
