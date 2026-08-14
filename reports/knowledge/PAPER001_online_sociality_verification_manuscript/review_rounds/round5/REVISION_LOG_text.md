# Round-5 text revision execution log

## Execution basis

- Plan read in full before editing: `ROUND5_REVISION_PLAN.md`.
- Manuscript base verified: `985a757b0939e2fff6f50d230e5d0dd9822514aa` on `paper/human-arm-target`.
- Initial worktree: clean.
- Pre-edit `\targetnum` ordered `grep -o` extraction count: 14; diff against HEAD: empty.
- Pre-edit abstract diff against HEAD: empty.
- Pre-edit E11 counts: `0.1493=1`, `0.0199=1`, `-2.6100=1`, `+0.1372=1`.
- Pre-edit figH SHA-256: `a70513f58d4c6a3962511f6797030d7537d13405d5f4010f5e904d6e1309e7d3`.

## Item log

### V01 (includes V07)

- Status: EXECUTED.
- Anchor matched: `Where a reading is issued, it abstains for lack of human support on only a small`.
- Result: replaced the sentence exactly with the approved readable-moment wording and the `Section 2.4 and Methods` pointer.
- Deviations: none.

### V07

- Status: EXECUTED INSIDE V01; no separate edit.
- Anchor matched: the same V01 sentence containing the benchmark-reasons pointer.
- Result: pointer changed from `Methods` to `Section 2.4 and Methods` in the single V01 pass.
- Deviations: none.

### V02

- Status: EXECUTED.
- Anchor matched: `(5th--95th percentiles $1.35$--$2.28$\,rad) of the $2.36$\,rad admissible span: the reference is wide,`.
- Result: replaced `admissible span` with `candidate span`; all numbers and the rest of the sentence stayed unchanged.
- Deviations: none.

### V03

- Status: EXECUTED.
- Anchor matched: `; the remaining $79.1\%$ is not recoverable`.
- Result: replaced `is not recoverable from` with `is not recovered by`.
- Deviations: none.

### V04

- Status: EXECUTED.
- Anchor matched: `returns a verdict on 14{,}099 ($20.8\%$), with no engineering failures --- no solver or pipeline failure on any`.
- Result: inserted `in this replay` after `failures`.
- Deviations: none.

### V05

- Status: EXECUTED.
- Anchor matched: `state; a scalar sociality score is ill-posed. We report this as a state-dependent regularity of the`.
- Result: replaced the approved clause exactly with `a single scalar sociality score cannot serve as that reference`.
- Deviations: none.

### Compile checkpoint after V05

- Command: `pdflatex -interaction=nonstopmode -halt-on-error main.tex`.
- Status: PASS, exit 0.
- Actual output: `Output written on main.pdf (36 pages, 754849 bytes).`

### V06 — target 1

- Status: EXECUTED.
- Anchor matched: `online-computable risk proxies, and support/readability state), estimated by conditional quantile regression`.
- Result: appended the approved histogram-based gradient-boosted quantile-model description and frozen hyperparameters.
- Deviations: none.

### V06 — target 2

- Status: EXECUTED.
- Anchor matched: `Excluded from the reference are observed post-encroachment time, ...`.
- Placement matched: immediately before the exclusion sentence.
- Result: inserted the approved four-categorical/22-numeric situation-vector feature contract.
- Deviations: none.

### V10

- Status: EXECUTED.
- Anchor matched across the existing source-line wrap: `we report this as an observed difference without attributing a cause, and the readability gate treats both populations identically.`
- Result: appended the approved support-limit diagnosis and the statement that the support gate compares kinematic neighbourhoods, not IPV values.
- Deviations: none.

### V08

- Status: EXECUTED in `RESPONSE_NOTES_round5.md`.
- Anchor matched in plan: `Absolute-width attack response (B-m8, D §6b)`.
- Result: response-ready prose covers all three required points: own disclosure; diversity/relative-sharpening/calibration/conservative-flag framing; printed width distribution plus backlog B09.
- Deviations: none.

### V09

- Status: EXECUTED in `RESPONSE_NOTES_round5.md`.
- Source matched: `../round4/RESPONSE_NOTES_round4.md`.
- Result: paragraphs 1--3 reused verbatim (`diff exit 0`), followed by the three approved additions: timing-test rationale; `4/201` and `30/201` frozen counts; B27 draw-count pre-commitment note.
- Deviations: none.

## Full compile

| Step | Exit | Actual output |
|---|---:|---|
| `pdflatex` after V10 | 0 | `main.pdf (37 pages, 756872 bytes)` |
| `bibtex main` | 0 | top-level auxiliary `main.aux`; style `unsrt.bst`; database `bibliography/biblio.bib` |
| second `pdflatex` | 0 | `main.pdf (37 pages, 756872 bytes)` |
| third `pdflatex` | 0 | `main.pdf (37 pages, 756872 bytes)` |

Non-failing pre-existing layout warnings remain in the LaTeX transcript (including `fancyhdr` head-height and overfull/float warnings); the requested error count is zero and undefined-reference/citation warning count is zero. No warning-driven out-of-scope edit was made.

## Final verification battery

| # | Check | Status | Actual output |
|---:|---|---|---|
| 1 | Ordered `\targetnum` extraction and abstract | PASS | before=14; after=14; targetnum diff exit=0 (empty); abstract diff exit=0 (empty) |
| 2 | E11 quadruple | PASS | `0.1493=1`; `0.0199=1`; `-2.6100=1`; `+0.1372=1` |
| 3 | Vocabulary/codenames | PASS | `estimab*=0`; standalone `passive=0`; `CQR=0`; `M0--M5=0`; `source-guard=0`; `claims-register=0` |
| 4a | `Where a reading is issued` | PASS | 0 |
| 4b | `is not recoverable` | PASS | 0 |
| 4c | `rad admissible span` | PASS | 0 |
| 4d | `no engineering failures in this replay` | PASS | 1 |
| 4e | `ill-posed` | PASS | 0 |
| 4f | `gradient-boosted` | PASS | 1 |
| 4g | `1{,}148{,}133` | PASS | 1 |
| 4h | `Section 2.4 and Methods` | PASS | 1 |
| 4i | Fig.-4 retained definition | PASS | `admissible range is the span of the candidate grid=1` |
| 5a | Compile errors and unresolved references/citations | PASS | `compile_error_lines=0`; `undefined_ref_or_citation_warnings=0` |
| 5b | References heading | PASS | extracted-PDF exact heading count=1 |
| 5c | Extended Data order | PASS | extracted-PDF `Extended Data` line 1807; `Figure ED1` line 1866 |
| 5d | Page count | PASS | `Pages: 37` |
| 5e | figH identity | PASS | final SHA-256 `a70513f58d4c6a3962511f6797030d7537d13405d5f4010f5e904d6e1309e7d3`; figure diff files=0 |
| 6 | Response notes | PASS | file exists; V09 paragraphs 1--3 verbatim diff exit=0; V08 has `1.87`, `1.35`, `2.28`, `21\%`, `20\%`, `8\%`, and B09; V09 has `4/201`, `30/201`, B27 and the pre-commitment note |

## Repository-scope safeguards

- `git diff --check`: exit 0.
- Staged files: 0.
- Figure files changed: 0.
- No commit or push performed.

## Deviations and anchor mismatches

- Deviations: none.
- Skipped items: none.
- Anchor mismatches: none.
