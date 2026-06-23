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

## 2026-06-21 - Codex - Merge Overleaf 2026-06-21-1457 into main

Files changed in paper repo: `main.tex` resolved to the Overleaf branch version, and
`structure.md` restored from the Overleaf branch. The existing `main` repository files
including CI, collaboration docs, `.gitignore`, and bibliography were retained.

Summary: Per author instruction, merged `origin/overleaf-2026-06-21-1457` into `main`.
The merge conflict represented a narrative fork: `main` contained the recently merged
counterpart-conditioned current-IPV verifier rewrite, while the Overleaf branch contained
the v3 self-anchor manuscript outline and full text. Conflict resolution intentionally keeps
the Overleaf manuscript text as the newer drafting source: self-anchor is again the main
online handle, with risk/PET framed as weak for interval narrowing, source-transfer numbers
restored, planner-facing interface-demo numbers restored, and external validation kept
inside `\planned{...}`. This reverts the manuscript narrative away from the
counterpart-conditioned-current-IPV version in PR#2 while preserving the repository hygiene
work from `main`.

Verification: no conflict markers remained; `git diff --check` passed; CI-equivalent source
lint passed for algorithm packages and dev markers; LaTeX environment begin/end counts
matched for document, abstract, algorithm, algorithmic and figure; bibliography keys cited
from `main.tex` exist in `bibliography/biblio.bib`. Full PDF compilation was not run locally
because no TeX engine (`latexmk`, `pdflatex`, `tectonic`) was available in the Codex
environment; GitHub CI should perform the full compile.

Open risk: `structure.md` is restored as an outline and may become stale if the manuscript
narrative changes again. The revived self-anchor claims should be checked against the latest
RQ decisions before submission.
