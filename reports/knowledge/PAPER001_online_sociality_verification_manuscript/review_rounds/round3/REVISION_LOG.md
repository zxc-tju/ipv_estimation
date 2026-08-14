# Round-3 revision log (index)

Commit: `ad18609` on `paper/human-arm-target` (base `a4f99f8`). 3 files: main.tex + fig0_concept pdf/png.

## What ran

- Reviews (blind, on `manuscript_a4f99f8.pdf`, 35 pp — first round reviewing a build with all
  round-1/2 figure fixes embedded): A 5→35 Reject-leaning, B 5→45 MR, C 5→45 MR, D 5→60 MR.
- Aggregation: `AGGREGATION.md` (50 clusters R3-01..R3-50; 10 NEW / 38 RESIDUAL / 2 REGRESSION),
  `ROUND3_REVISION_PLAN.md` (S01–S16), `BACKLOG_delta.md` (B23–B26), `ESCALATIONS.md`
  (E2 five-site list; E5 third unanimous round; E11 sharpened into a register-compliance gap;
  NEW E14 TOPS-relationship disclosure; E13 retired with zero objections).
- PI decisions this round: E11 = print the companion qualification (executed, exact PI-ratified
  wording); declarations package (E5+E8+E14) = facts to be supplied now by PI (checklist issued;
  pending at commit time).
- Execution:
  - Text (codex, xhigh): S01–S13, S15, S16 + E11 companion — 16/16 DONE, one capitalization-only
    deviation (S09). Battery 10/10 PASS incl. E11 uniqueness (0.1493/−2.6100/0.0199 once each)
    and E2-protected-region empty diffs. Details: `REVISION_LOG_text.md`.
  - Figure (codex, xhigh): S14 fig0_concept panel-c label "(over-yielding side)" — acceptance
    exit 0; label wrapped to two lines after A9 caught a MediaBox overflow on the one-line
    render; pixel diff confined to declared region; figH SHAs unchanged. Details:
    `REVISION_LOG_fig.md`.
  - Orchestrator: independent verification (targetnum 14/14 byte-identical; abstract unchanged;
    removed/kept phrase greps; E2 sentinel strings present; vocab zero).

## Fact-check headlines (aggregator, spot-verified by orchestrator)

- E11 compliance gap REAL: RQ018 decision.md C1 Required Qualification mandates the two
  companions whenever the placebo p is cited; main.tex had p=0.0199 without them (now fixed).
- E14 premise REAL: biblio.bib names "Tongji University TOPS Group" as benchmark operator.
- Reviewer errors: A's ED1 "with with" phantom (second consecutive round a reviewer reports it;
  figure and caption both single "with"); C-M5 inverted the printed readability/support split.
- Round-2 regressions found and fixed this round: nested engineering-failures parenthetical
  (S01); "supported threshold" panel title needing its caption definition (S04).

## Watch items

- W1: Figure 5 float oversized by ~59 pt; caption renders fully but extends into the lower
  margin (caption grew via S03/S04/S13). Candidate round-4 layout fix (caption trim or float
  page) — no content change required.

## Still open

E2 (five named sites, C7-coupled, two-sided band must be pre-ratified), E5/E8/E14 (facts
checklist with PI — declarations pass to land before round 4 if facts arrive), E6, E7 (couple
with E11 ledger discoverability), E10. C7 unchanged: 14 \targetnum swaps + watermark await
REAL_VERIFIED.
