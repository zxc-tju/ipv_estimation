# Round-2 text revision execution log

Base manuscript: `main.tex` at `41b819a`.

## Q01

- Status: DONE
- Anchor matched: `\section*{References}` immediately above `\bibliographystyle{unsrt}`.
- Action: removed the duplicate manual References heading.

## Q02

- Status: DONE
- Anchor matched: end-matter block beginning `% Extended Data figures` and ending at the bibliography.
- Action: reordered the end matter to Data Availability, Code Availability, `\clearpage`, Extended Data, bibliography and declarations; added `\suppressfloats[t]`; changed both Extended Data figures from `[t]` to `[!htb]`; captions were left byte-identical. The first render still began References between ED1 and ED2, so the plan's permitted equivalent layout was used: a second `\clearpage` after ED2 flushes both Extended Data floats before References.

## Q03 (+Q27)

- Status: DONE
- Anchor matched: `The monitor is, first, able to speak.` paragraph beginning with the 15 scenario templates and 19 systems.
- Action: replaced the approximate run/moment accounting with the frozen $20.8\%$, $86.5\%$, $55.3\%$ and 36-run two-level account; defined no engineering failures as no solver or pipeline failure on any candidate moment.

## Q04

- Status: DONE
- Anchor matched: `It abstains on only a small fraction of supported cases` in Results section 2.3.
- Action: scoped the $5.08\%$ figure to the held-out natural-driving test fold and cross-referenced the larger benchmark abstention account in Methods.

## Q05

- Status: DONE
- Anchors matched: all three plain-text `$9.845\%$` flag-rate sites in Results; the third is adjacent to protected `\targetnum` tokens.
- Action: changed the three plain-text rates to `$9.8\%$`; exact counts and all `\targetnum` calls were preserved. Per the specified default, `$9.72\%$` was left unchanged.

## Q06

- Status: DONE
- Anchors matched: the marginal-coverage sentence beginning `Moments within an interaction are dependent` and the sentence beginning `Conditional coverage across state groups` in Methods 4.4.
- Action: replaced the guarantee wording with the empirical pointwise-coverage statement; separated unestablished conditional/simultaneous coverage from the reported leave-one-source-out boundary without changing any LODO values.

## Q07

- Status: DONE
- Anchors matched: `Sequential warnings are tuned on an independent guard set` in Methods 4.4 and `apply separately-calibrated persistence for warnings` in Algorithm 2.
- Action: respecified persistence as an optional, unevaluated deployment interface and stated that all reported verdicts and flag rates are per-moment.

## Q08

- Status: DONE
- Anchors matched: `yet reading the counterpart adds nothing online` in Results and `adds nothing beyond the current situation` in Discussion.
- Action: restored the evidence-bounded qualifiers `adds no measurable sharpening online` and `adds no measurable value beyond the current situation`.

## Q09

- Status: DONE
- Anchor matched: `Official competition scores, harm labels and preference ratings were excluded as endpoints by pre-registration.`
- Action: replaced the registration claim with the frozen-before-outcomes analysis-plan wording.

## Q10

- Status: DONE
- Anchor matched: `causal interaction progress` in the Methods 4.3 situation tuple.
- Action: renamed it `interaction progress`.

## Q11

- Status: DONE
- Anchor matched: `The concentration of the weights $\pi_{a,k}(t)$ gives a reliability measure $u_a(t)$.`
- Action: aligned the first mention with the Methods definition: spread defines $u_a(t)$, greater concentration means smaller $u_a(t)$, and readings require the frozen threshold.

## Q12

- Status: DONE
- Anchor matched: `can guarantee legal safety as an online layer around a planner` in the Introduction.
- Action: scoped the guarantee to a formally specified notion of legal safety within model assumptions.

## Q13

- Status: DONE
- Anchors matched: `All four thresholds were tested; the absence of a mark` and the Fig. 5 caption sentence beginning `Error bars and asterisks denote`.
- Action: stated that all four thresholds and intervals are displayed, removed the nonexistent asterisk convention, described bracketed values and per-endpoint intervals, and folded the frozen-battery/no-post-hoc-selection sentence into the replacement.

## Q14

- Status: DONE
- Anchors matched: `as a percentage of the admissible range of the quantity.` and `Conditioning narrows it by $21\%$, $20\%$ and $8\%$ at the three levels` in the Fig. 4 caption.
- Action: defined the candidate-grid span and parameterisation domain; added the frozen unrounded reductions and whole-percent bar-label rounding note.

## Q15

- Status: DONE
- Anchors matched: `(a scripted-counterpart subset, $27$ runs, is excluded)` in Methods 4.5; `against the same scripted counterpart injections` in Methods 4.6; and `against the same scripted counterparts` in Results 2.5.
- Action: distinguished fixed-script counterpart control from scenario-scripted injection events at all three sites; all adjacent `\targetnum` calls were preserved byte-identically.

## Q16

- Status: DONE
- Anchor matched: the Methods 4.1 sentence beginning `for each agent we generate candidate trajectories over a grid of preferences`.
- Action: added the frozen cosine/sine candidate-objective family, the more-negative/assertive direction, and the code-release boundary.

## Q17

- Status: DONE
- Anchor matched: the Methods 4.5 verdict-count sentence ending `$519$ lie below the range, $869$ above and $12{,}711$ inside`.
- Action: added the frozen sign-test counterexample using the $90\%$ lower edge, within-range negative readings and negative-reading flag rate. The existing semicolon after the verdict count was changed to a full stop and the following `Consequence` capitalised solely to form the required grammar join.

## Q18

- Status: DONE
- Anchors matched: the Methods 4.5 placebo sentence ending `distinguishes the real flag timing from permuted timings` and the defined-margin count ending `$472$ flagged, $11{,}669$ within-range`.
- Action: added empirical `$p=0.0199$` and the frozen `$9.1\%$` versus `$8.2\%$` undefined-margin balance.

## Q19

- Status: DONE
- Anchors matched: the Fig. 3a sentence ending `The three bands cover the $37{,}495$ cases` and the panel-c sentence ending `variance explained at or barely above zero in all four folds`.
- Action: identified the realised post-encroachment margin as offline/descriptive only and defined held-out-source case-level $R^2$ and its zero baseline.

## Q20

- Status: DONE
- Anchor matched: `The situation explains $20.9\%$` in the Fig. 4c caption.
- Action: added the frozen SSE/SST definition and the all-test-moment versus accepted-moment values and denominators.

## Q21

- Status: DONE
- Anchor matched: the Methods 4.4 test-fold coverage sentence immediately before `Splits are made by whole scenes`.
- Action: added the denominator glossary for anchor, readable, accepted, candidate and judgeable moments.

## Q22

- Status: DONE
- Anchor matched: `Points are means with $95\%$ confidence intervals over case clusters.` in the Fig. 2a caption.
- Action: specified that the displayed `$n$` column counts case clusters.

## Q23

- Status: DONE
- Anchor matched: `panel \textbf{c} uses $n=469$ atypical and $n=10{,}483$ within-range moments` in the Fig. 5 caption.
- Action: identified these as the flagged moments inside the 175 non-scripted runs with recorded counterpart logs.

## Q25

- Status: DONE
- Anchor matched: the Results 2.2 risk-reversal sentence ending `under low risk ($>2.0$\,s)`.
- Action: added the ratified magnitude context, contrasting the small role shifts with range sharpening while retaining their risk-dependent directional reversal.

## Q26

- Status: DONE
- Anchor matched: `\textsc{Counterpart-Unreadable}` in the Methods 4.4 reason-code list.
- Action: defined it as counterpart identity being insufficiently stable to anchor a reading, tied to `$c_a$` of Eq. 2.

## Q27

- Status: DONE
- Anchor matched: `no engineering failures` inside the Q03 benchmark-accounting paragraph.
- Action: folded into Q03 as specified; defined the phrase as no solver or pipeline failure on any candidate moment.

## Q28

- Status: DONE
- Anchor matched: the Methods 4.4 frozen train/guard/calibration/test fold sentence.
- Action: added the frozen nested flow from 38,228 corpus cases to 26,828 anchor-ledger cases, 4,497,368 anchor rows, 2,442,625 reference rows and the four fold counts.

## Execution summary

- In-scope items: 27.
- DONE: 27.
- DONE-WITH-DEVIATION: 0.
- SKIPPED: 0.
- Q00 and Q24 were excluded from this text-agent scope exactly as instructed and are not counted above.
- Permitted layout/grammar adaptations: Q02 uses a second `\clearpage` after ED2 because the first render otherwise began References between ED1 and ED2; Q17 changes the pre-existing semicolon to a full stop and capitalises `Consequence` to join the inserted sentence grammatically. Neither changes claim strength.
- Current-anchor mismatches: none; every quoted `Current`/target anchor used for an edit was found in `41b819a` (allowing only source line wrapping).
- Concurrent workspace observation: during this run, the separate Q24 lane modified the PDF/PNG pairs for `fig0_concept`, `fig1_measurable`, `fig3_monitor` and `fig5_consequence`. This text agent did not edit or revert those files. No `figH` file is in the tracked diff.

## Compile checkpoints

- After Q02: PASS, full `pdflatex` + `bibtex` + 2x `pdflatex`, exit 0. Visual correction confirmed Data/Code Availability before Extended Data, ED heading before ED1, ED2 before References.
- After Q07: PASS, `pdflatex`, exit 0; protected baselines unchanged.
- After Q15: PASS, 2x `pdflatex`, exit 0; protected baselines and robust full-token comparison unchanged.
- After Q28/final: PASS, full `pdflatex` + `bibtex` + 2x `pdflatex`, exit 0.

## Final verification battery

| # | Result | Actual output |
|---|---|---|
| 1 | PASS | `target_diff_exit=0`, `diff_output_bytes=0`, `targetnum_total=21`, `call_sites=14`, `after_matches_actual=0`, robust full-token comparison against `41b819a`: `0` |
| 2 | PASS | `estimab_count=0`, standalone case-insensitive `passive_word_count=0` |
| 3 | PASS | `M0-M5 before=0 after=0 delta=0`; `CQR before=0 after=0 delta=0`; `source-guard before=0 after=0 delta=0`; `claims register before=2 after=2 delta=0` |
| 4 | PASS | `manual_references_count=0` |
| 5 | PASS | `references_heading_count=1`; `extended_data_line=1960`; `ED1_caption_line=2002`; `order=PASS`; `stale_2.35_count=0`. Final page check: Data/Code Availability p.25, Extended Data heading and ED1 p.26, ED2 p.27, References p.28. |
| 6 | PASS | `abstract_diff_exit=0`, `diff_output_bytes=0`, `after_matches_actual=0` |
| 7 | PASS | `final_compile_exit=0`, `latex_error_lines=0`, `git diff --check` exit 0. Baseline/current warning counts: `fancyhdr=32/34`, `overfull=3/3`, `underfull=1/1`, `LaTeX=0/0`; no new warning class or box warning, with two additional instances of the existing `fancyhdr` head-height warning because the revised PDF is 35 pages versus 33 at `41b819a`. |
