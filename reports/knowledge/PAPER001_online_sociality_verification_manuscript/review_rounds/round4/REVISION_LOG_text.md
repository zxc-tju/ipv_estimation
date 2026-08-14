# Round-4 text revision execution log

## Context and scope

This run executed the approved Round-4 manuscript text revisions against `main.tex` at baseline commit `ad18609bfeb0`. The overall revision loop contains ten plan items; this text lane executed U01, U02, U03, U05, U07, U04, U06 and U10 in that order, while U08 and U09 remained assigned to the separate figure lane. The run edited `main.tex`, created `RESPONSE_NOTES_round4.md`, compiled the manuscript, and did not stage, commit or push anything.

Baseline evidence: the repository was clean; the ordered `grep -o` extraction contained 14 `\targetnum{...}` call sites; its diff against `HEAD:main.tex` was empty; and the extracted abstract diff was empty.

## Item log

### U01

- Status: APPLIED / PASS.
- Anchor matched, allowing the documented source wrap: `predicting the held-out source leaves the variance explained at or barely` followed by `above zero in all four folds.`
- Applied wording: `Fitting the full state description on every source but one and predicting the held-out source leaves the variance explained never above $+0.03$ in any fold and below zero in two ($+0.026$ Waymo, $+0.017$ Lyft, $-0.195$ Argoverse-2, $-0.276$ nuPlan).`
- Deviation: none.

### U02

- Status: APPLIED / PASS.
- Anchor matched: `whiskers are case-bootstrap $95\%$ intervals where defined, and`.
- Applied wording: `an asterisk marks a ratio whose interval excludes parity (the no-difference value, ratio $1$); entries whose intervals admit parity carry no asterisk.`
- Deviation: none.

### U03

- Status: APPLIED / PASS.
- Anchor matched: `so the conformal step certifies, rather than repairs, the situational quantile model.`
- Applied wording: `so the conformal step finds essentially nothing to repair in the situational quantile model.`
- Deviation: none.

### U05

- Status: APPLIED / PASS (three of three micro-edits).
- Anchor 1 matched: `A threshold is called supported when its per-endpoint interval excludes no difference; the`.
- Applied target 1 wording: `A threshold is called supported when its per-endpoint interval excludes a zero difference --- a statement about the interval, distinct from the monitor's human-support gate; the panel titles use the word in exactly this sense.`
- Anchor 2 matched: `at every threshold whose interval excludes no difference`.
- Applied target 2 wording: `at every threshold whose interval excludes a zero difference`; the remainder of the sentence was unchanged.
- Anchor 3 matched: `at the supported thresholds they are one-third`.
- Applied target 3 wording: `at the thresholds whose intervals exclude a zero difference (Section 2.4) they are one-third to one-half as frequent`.
- Deviation: none. The abstract occurrence `at any supported threshold` was not touched.

### U07

- Status: APPLIED WITH DEVIATION / PASS.
- Anchor matched: `Vocabulary, used consistently: an anchor row is one agent-moment eligible for`.
- The existing vocabulary sentence was moved unchanged, and the approved definitions of scene, interaction case, scenario and scenario run were appended exactly as specified.
- DEVIATION: the plan body says to place the block immediately after the first Methods 4.4 sentence, whose final words are `over accepted calibration moments.` The execution prompt's final battery instead requires `Vocabulary, used consistently` to appear before the first use of `accepted calibration moments`. Those two placements cannot both hold. To satisfy the explicit final battery, the block was placed immediately after the Methods 4.4 subsection heading, before the formula sentence. Actual final lines: vocabulary at line 665; first `accepted calibration moments` at line 674.

### U04

- Status: APPLIED / PASS.
- Anchor matched: the sentence ending `spans essentially the whole admissible scale and can never flag a moment.`
- Inserted the approved absolute-width sentence with frozen values `1.87`, `1.35`, `2.28` and `2.36` before the empirical marginal-coverage sentence.
- Deviation: none.

### U06

- Status: APPLIED / PASS.
- Anchor matched across the documented wrap: `(at least five flagged frames in one contiguous stretch; counterpart speed decrease of at least $20\%$; automated-vehicle speed change within $10\%).`
- Appended the approved deterministic tie-break disclosure with frozen scores `$0.675$ vs $0.552$`.
- Deviation: none.

### U10

- Status: APPLIED / PASS (both deliverables).
- Manuscript anchor matched across the documented wrap: `the association survives a placebo test that reassigns whole flag sequences across scenario runs`.
- Applied pointer: appended `(Methods 4.5)` before the semicolon. The E11 companion block was not edited; its extracted diff against `HEAD:main.tex` is empty.
- Response-letter material: created `RESPONSE_NOTES_round4.md` with four numbered, self-contained points covering (1) the distinct case-label and exposure-placebo questions, (2) the fixed-three-second-window magnitude basis and acknowledged contract-window boundary, (3) the Results/Methods two-layer placement, and (4) the ED1 reviewer-error note.
- ED1 grep evidence included in the response note: `grep -n -F 'with with' manuscript_ad18609.txt` returned no output and exit status 1; the correct single-`with` embedded title is at line 1771, and the whitespace-tolerant caption match begins at line 1829.
- Deviation: none.

### U08

- Status: SKIPPED as instructed; assigned to the separate figure agent.
- No figure action was taken by this text lane.

### U09

- Status: SKIPPED as instructed; assigned to the separate figure agent.
- No figure action was taken by this text lane.

## Compile log

- After U05: `pdflatex -interaction=nonstopmode -halt-on-error main.tex` exited 0 and wrote a 36-page PDF.
- After U10: `pdflatex` exited 0; `bibtex main` exited 0; both subsequent `pdflatex` passes exited 0; output was 36 pages.
- The separate figure lane then wrote the four U08/U09 binaries into the shared worktree. The complete four-command build was repeated against that final shared state: exits were `0, 0, 0, 0`, and the final PDF remained 36 pages (`754797` bytes).
- Final `main.log`: fatal-error lines `0`; undefined-reference/citation matches `0`. Existing non-fatal layout/header warnings remain outside this text plan; no layout remedy was attempted.

## Final verification battery

| Check | Status | Actual output |
|---|---|---|
| 1a. Ordered `\targetnum` extraction | PASS | current `14`; HEAD `14`; `diff -u` exit `0`; diff bytes `0`; output empty |
| 1b. Abstract byte identity | PASS | extracted `\begin{abstract}...\end{abstract}` diff exit `0`; diff bytes `0`; output empty |
| 2a. Forbidden vocabulary | PASS | `estimab*=0`; standalone `passive=0`; `M0-M5=0`; `CQR=0`; `source-guard=0`; `claims-register=0` |
| 2b. Protected E2 sites | PASS | zero changed diff lines matching `defining population`, contribution-6 transfer wording, section 2.5 title, `instrument survives the move`, or `sits at the native level` |
| 3a. Removed plan phrases | PASS | wrapped `at or barely above zero=0`; `asterisk marks an interval excluding no difference=0`; `certifies, rather than repairs=0` |
| 3b. No-difference ambiguity | PASS | combined `excludes no difference` / `excluding no difference` count `0` |
| 3c. Vocabulary order | PASS | `Vocabulary, used consistently` line `665`; first `accepted calibration moments` line `674`; before=`1` |
| 3d. Frozen values surfaced | PASS | section 2.3 body `1.87` count `1`; ED2 caption `0.675` count `1` |
| 4a. E11 uniqueness | PASS | `0.1493=1`; `-2.6100=1`; `+0.1372=1`; `0.0199=1` |
| 4b. E11 block identity | PASS | extracted companion-block diff exit `0`; diff bytes `0`; output empty |
| 5a. Full compile | PASS | final command exits `pdflatex:0, bibtex:0, pdflatex:0, pdflatex:0`; fatal errors `0`; undefined references/citations `0` |
| 5b. PDF structure | PASS | `References` heading count `1`; `Extended Data` line `1780`; `Figure ED1` line `1839`; ED-before-ED1=`1`; pages `36` |
| 6. U10 response note | PASS | file exists; numbered points `4` |
| Diff hygiene | PASS | `git diff --check` exit `0`, output empty; staged paths `0` |
| figH protection | PASS | PDF SHA-256 `a70513f58d4c6a3962511f6797030d7537d13405d5f4010f5e904d6e1309e7d3`; PNG SHA-256 `bb1c5817ac7c1adf3230d870a055cf01269226ab4a9465c4e7e1b585f3835f15`; both match the clean baseline |

## Concurrent shared-worktree observation

The baseline repository was clean. During this text run, the separate figure agent wrote exactly the four files assigned to U08/U09: `figures/fig1_measurable.pdf`, `figures/fig1_measurable.png`, `figures/fig3_monitor.pdf` and `figures/fig3_monitor.png`. This text lane did not edit, generate or copy figure binaries. The changes were preserved, not reverted, and the final full compile was repeated after they appeared. No `figH` file changed.

One read-only verification retry occurred: the first final `git status` command was issued from `OUTPUT_DIR`, which is not a Git worktree, and returned `fatal: not a git repository (or any of the parent directories): .git`. It made no changes. The command was immediately rerun from the manuscript repository and returned the five expected unstaged paths (`main.tex` plus the separate figure lane's four U08/U09 binaries), staged-path count `0`, and `git diff --check` exit `0`.

## Final state

- Text items applied: 8 of 8 requested (U01, U02, U03, U05, U07, U04, U06, U10).
- Figure items skipped by this lane: 2 of 2 instructed (U08, U09).
- Plan deviations: 1 (U07 placement conflict, documented above).
- Read-only verification retries: 1 (wrong working directory for the first final `git status`; corrected immediately).
- Anchor mismatches: 0.
- Battery failures: 0.
- Git staging/commit/push: none.
