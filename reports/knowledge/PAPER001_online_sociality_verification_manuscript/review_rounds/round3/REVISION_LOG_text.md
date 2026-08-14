# Round-3 text revision execution log

## Execution summary

- Manuscript: `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/2_PaperWriting/NMI---Online-Sociality-Verfication-for-Autonomous-Vehicle/main.tex`
- Verified starting point: branch `paper/human-arm-target`, HEAD `a4f99f8`, clean worktree at the text run's initial status check.
- Executed text items: 16/16 requested (`S01`--`S13`, `E11-COMPANION`, `S15`, `S16`).
- Exact executions without deviation: 15.
- Executions with a logged deviation: 1 (`S09`, capitalization-only adaptation).
- Explicitly skipped: `S14` (assigned to the separate figure agent).
- Anchor mismatches: none.
- Git actions: no add, commit, or push.

## Item log

### S01 — PASS

- Anchor matched once: `returns a verdict on 14{,}099 ($20.8\%$; no engineering failures (no solver or pipeline failure on any`.
- Companion Methods anchor matched once: the natural-corpus accounting ending `$17{,}416$ exact ties and $1{,}826$ solver failures.`
- Applied both approved replacements/additions with all numbers unchanged.
- Deviation: none.

### S02 — PASS

- Anchor matched once: `the monitor registers is that comfortable interactions stop being comfortable, not that dangerous ones`.
- Replaced with the approved margins/events sentence.
- Deviation: none.

### S03 — PASS

- Anchor matched once: `\caption{\textbf{Atypicality compresses ordinary interaction on both sides without adding emergencies.}`.
- Replaced the caption title with the approved associational wording.
- Deviation: none.

### S04 — PASS

- Anchor matched once across the existing caption wrap: `All four thresholds were tested and all four are displayed with their intervals; the $<3$\,s interval admits no difference.`
- Appended the approved definition of `supported`.
- Deviation: none.

### S05 — PASS

- Anchor matched once: `at coverage within 0.6 percentage points of nominal---and it stays informative at the`.
- Inserted the approved marginal-coverage boundary immediately after the sentence ending `can never flag a moment.`
- Deviation: none.

### S06 — PASS

- Anchor matched once: `counting both sides, $9.8\%$ of judgeable moments are flagged`.
- Added the approved Results gloss for judgeable moments.
- Deviation: none.

### S07 — PASS

- Anchor matched once across the line wrap: `is this behaviour within the range that competent humans exhibit?`
- Replaced it with the approved same-situation human-driver wording.
- Deviation: none.

### S08 — PASS

- Algorithm anchor matched once: `\RETURN (range, verdict, signal)`.
- Methods anchor matched once: `\textsc{Low-Human-Support} or \textsc{Out-of-Distribution}; otherwise it emits`.
- Applied the approved optional-output comment and first-failing-gate reason-code precedence.
- Deviation: none.

### S09 — PASS WITH DEVIATION

- Anchor matched once after: `The frozen configuration uses seven candidate preferences $\theta_k\in\{-3,-2,-1,0,1,2,3\}\times\pi/8$;`.
- Inserted the approved grid/domain reconciliation sentence verbatim.
- DEVIATION: the existing following wording `a rolling window` was changed to `A rolling window` because the inserted text ends with a period. This is capitalization-only; it changes no fact, number, or claim strength.

### S10 — PASS

- Anchor matched once: `The ego-side outcome is the minimum time-to-collision to the counterpart over the post-verdict window`.
- Applied the approved contract-window and TTC-computation definition.
- Deviation: none.

### S11 — PASS

- Anchor matched once: `distinguishes the real flag timing from permuted timings (empirical $p=0.0199$)`.
- Applied the approved 200-draw and absolute case-clustered-$t$ specification.
- Deviation: none.

### E11-COMPANION — PASS

- Anchor matched once immediately after the S11 placebo sentence.
- Inserted the PI-ratified two-sentence block verbatim, including `[-2.6100, +0.1372]` and `$p=0.1493$`.
- Deviation: none.

### S12 — PASS

- Anchor matched once: the sentence ending `over all accepted test-fold moments ($461{,}937$).`
- Inserted the approved conformal-radii sentence before `Vocabulary, used consistently:`.
- Deviation: none.

### S13 — PASS

- Anchor matched once: `panel \textbf{d} aggregates the acceleration records within those windows ($13{,}800$ vs $310{,}246$ records).`
- Appended the approved threshold-nesting and record-weighting sentences.
- Deviation: none.

### S14 — SKIPPED

- Reason: the execution scope explicitly assigns S14 to a separate figure agent.
- This text run did not edit, generate, copy, or revert figure binaries.
- Concurrent-worktree note: during this run the separate S14 agent modified `figures/fig0_concept.pdf` and `figures/fig0_concept.png`; its log confirms those files were copied at `2026-08-11 06:39:21 +0800`. Those concurrent changes were preserved untouched. `figH` remained byte-identical.

### S15 — PASS

- Anchor matched once: the sentence ending `($\theta>0$ cooperative).`
- Inserted the approved early IPV scope sentence.
- Deviation: none.

### S16 — PASS

- Anchor matched once: `a hesitation/efficiency warning, and a degraded unverified mode on abstention. This mapping is`.
- Applied the approved safety-subordination and degraded-mode wording.
- Deviation: none.

## Compile checkpoints

| Checkpoint | Command sequence | Exit | Pages | Result |
|---|---|---:|---:|---|
| After S05 | `pdflatex -interaction=nonstopmode -halt-on-error main.tex` | 0 | 35 | PASS |
| After S13 | `pdflatex -interaction=nonstopmode -halt-on-error main.tex` | 0 | 36 | PASS |
| After S16 | `pdflatex`; `bibtex`; `pdflatex`; `pdflatex` (all halt on error) | 0 | 36 | PASS |

Final `main.log` contains `0` fatal-error lines and `0` undefined-reference/citation warnings. A non-fatal `Float too large for page by 59.26468pt` warning remains for Figure 5 after the approved caption additions; rendered-page inspection confirmed that the caption text remains present and legible, although it occupies the lower page margin. No unplanned layout edit was made.

## Final verification battery

| Battery item | Actual output | Result |
|---|---|---|
| 1. `\targetnum` ordered-token identity | `targetnum_count=14`; before/after token diff output empty (`targetnum_diff_lines=0`) | PASS |
| 2. Abstract byte identity | diff output empty; SHA-256 before = after = `1a4c0d30ca1c8ecd5861cde095591b8fd70d1526dff1bb1775a6b11f7264f0bf` | PASS |
| 3. Vocabulary / audit tokens | `estimab_count=0`; `standalone_passive_count=0`; `M0--M5/CQR/source-guard before=0, after=0` | PASS |
| 4. Superseded wording gone | `no engineering failures (no=0`; `comfortable interactions stop=0`; `Atypicality compresses=0`; `(range, verdict, signal)=0` | PASS |
| 5. E11 uniqueness | `0.1493=1`; `-2.6100=1`; `0.0199=1` | PASS |
| 6. Compile and document structure | full sequence exit `0`; fatal errors `0`; References headings `1`; Extended Data line `2039` precedes ED1 caption line `2081`; pages `36` | PASS |
| E2-gated text protection | contribution-6 paragraph diff `0`; human-arm gated paragraph diff `0`; section-title diff `0`; abstract diff `0` | PASS |
| Figure protection / concurrency | `figH` SHA-256 `a70513f58d4c6a3962511f6797030d7537d13405d5f4010f5e904d6e1309e7d3` unchanged; text run made no figure edit; current `fig0_concept.pdf/.png` changes belong to concurrent S14 agent | PASS (scope-isolated) |
| Git hygiene | `git diff --check` empty; no add/commit/push. Current worktree lists `main.tex` plus the separate S14 agent's two `fig0_concept` binaries | PASS |

## Snapshots

- Before: `ROUND3_text_before_main.tex`
- After: `ROUND3_text_after_main.tex`
- Compiled after: `ROUND3_text_after_main.pdf`

