# Round-2 revision log (index)

Commit: `a4f99f8` on `paper/human-arm-target` (base `41b819a`). 9 files: main.tex + 4 figure pdf/png pairs.

## What ran

- Reviews (blind, on `manuscript_41b819a.pdf` — NOTE: that build embedded the pre-P14/P15
  figures; see Q00): `review_codex_A.md` (8%→55%, Major revision), `review_codex_B.md`
  (8%→55%, MR), `review_codex_C.md` (5%→45%, MR), `review_claude_D.md` (~7%→~45%, MR).
- Aggregation: `AGGREGATION.md` (55 clusters R2-01..R2-55; 24 NEW / 30 RESIDUAL / 1 REGRESSION),
  `ROUND2_REVISION_PLAN.md` (Q00–Q28), `BACKLOG_delta.md` (B18–B22), `ESCALATIONS.md`
  (E2/E5/E6/E7/E8/E10/E11 still open; new E13, PI-decided this round).
- Execution:
  - Text (codex, xhigh): Q01–Q23, Q25–Q28 — 27/27 DONE, 0 deviations, 2 permitted
    adaptations (Q02 second `\clearpage`; Q17 grammar join). Details: `REVISION_LOG_text.md`.
  - Figures (codex, xhigh): Q24 — 4 changes, acceptance harness exit 0, pixel diffs confined
    to declared regions, figH SHA-256 unchanged. Details: `REVISION_LOG_fig.md`.
  - Orchestrator: E13 abstract edit per PI ruling — opening "certified" → "engineered and
    assessed" (L77); closing "which the monitor makes testable" KEPT (defend in response letter).
  - Q00 folded into the final orchestrator rebuild: fresh PDF verified to embed the round-1-fixed
    and round-2-retitled figures (stale "2.35" absent; new titles present).

## Final acceptance battery (orchestrator, independent)

- Compile exit 0, 0 LaTeX errors; PDF 35 pages.
- 14 `\targetnum` call sites byte-identical to 41b819a (token-list diff empty).
- Abstract diff vs 41b819a = exactly the one PI-ruled E13 line.
- `estimab*` = 0; standalone `passive` = 0; no manual `\section*{References}`.
- References heading appears once; Extended Data heading (p.26) precedes ED1 content.
- Fig 2a/4b/5b/5d new titles verified embedded (5b/5d split across two lines in the text
  layer; presence confirmed by prefix match + the fig agent's pixel comparison).

## PI decisions this round

- E13 opening: option (a) "engineered and assessed" — executed.
- E13 closing: option (a) keep and defend — no edit.
- Q05 conditional: default taken, 9.72% left unchanged (2-dp apparatus anchor).

## Still open (unchanged)

E2 (three-rate story, C7-coupled; two-sided acceptance band must be ratified BEFORE the real
human rate is seen), E5 (ethics facts — all four reviewers, urgent), E6 (benchmark
edition/run-ledger facts), E7 (data/code release vehicle), E8 (acknowledgements/contributions),
E10 (Fig 6 C7 regeneration spec + new facets), E11 (event-level null disclosure).
C7: 14 `\targetnum` swaps + watermark removal await REAL_VERIFIED.
