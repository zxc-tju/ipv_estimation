# Round-1 revision log — text items P02–P30 (P01/P14/P15 excluded)

Executor: round-1 revision agent, 2026-08-10.
Base state: repo HEAD = 486382e (exactly the plan's line-number reference), working tree clean.
Files edited: `main.tex`, `bibliography/biblio.bib` only. No commit, no push; build products left untracked.

Status summary: **27/27 in-scope items APPLIED** (0 skipped). 7 minor logged deviations, all in service of the item's stated intent; no constraint was traded away.

Legend: line numbers cited as "old L…" refer to the plan's baseline (486382e).

---

## Tier 1

### P02 — APPLIED (with 2 consistency extensions)
Methods 4.4 estimand (old L598–601) replaced:
- Before: "The primary estimand is case-balanced pointwise marginal coverage at prespecified progress anchors $\tau\in\{0.2,0.4,0.6,0.8\}$, with at most one calibration score per case per anchor … applied identically to calibration and test."
- After (plan wording): "The estimand is pointwise marginal coverage over accepted moments: the conformal radius $c_\alpha$ is the $\lceil(n+1)(1-\alpha)\rceil$-th smallest nonconformity score over all accepted calibration-fold moments ($n=455{,}723$), and coverage is evaluated over all accepted test-fold moments ($461{,}937$). Splits are made by whole scenes under a split frozen before this study … Moments within an interaction are dependent, so the guarantee is marginal over accepted moments, not a per-interaction or conditional guarantee."

Algorithm 1: line 7 "sample $\le 1$ score per case per anchor $\tau\in\mathcal{A}$" → "score every accepted calibration window"; `progress anchors $\mathcal{A}$` dropped from the Require line (no remaining use).

DEVIATION (in-scope consistency): the same "accepted calibration cases" phrase survived at two sites the plan did not quote and would have contradicted the new estimand text — Methods radius definition (old L593) → "over accepted calibration moments", Algorithm 1 line 8 → "over accepted calibration windows".

### P03 — APPLIED
Methods 4.5 (old L617–620). The "we order the interaction-opportunity onset, the / the onset of a readable segment and the first persistent deviation … controlling for scenario and initial risk" sentence (including the "the the" typo, R1-32) replaced by the plan's frozen-design text: per-moment verdicts; one-second verdict window precedes every outcome sample; ego outcome = min TTC over the post-verdict window; counterpart outcomes over a fixed three-second window from the same moment, from the other vehicle's own logged control record; comparisons within frozen situation cells; scenario runs as the clustering unit; placebo test reassigning whole exposure trajectories. "Realised-consequence events are defined by a frozen automatic extractor" kept as its own sentence (it still governs the harm-bounding sentence that follows). "Initial risk"/"first persistent deviation" dropped per [NOFAB].

### P04 — APPLIED (both sites)
- §2.4 (old L422–426) → "At the emergency thresholds---…---flagged moments are one-third to one-half as frequent as typical ones at every threshold whose interval excludes no difference (margins under $1$, $1.5$ and $2$\,s; braking beyond $-2$, $-3$ and $-4$\,m\,s$^{-2}$); at the widest margin tested (under $3$\,s) the interval admits no difference. The lower quartile of the ego's margin is nearly unchanged between the groups."
- §2.6 (old L490–492) → "at no emergency threshold are the flagged moments more frequent than the typical ones---at the supported thresholds they are one-third to one-half as frequent---so a monitor built on those thresholds would report nothing".

### P05 — APPLIED
Methods scope note: sentence "Per-study evidence status is tracked in \texttt{claims\_register.md}." deleted; note now ends at "…the two validation protocols." (Remaining "claims register" strings in the file are LaTeX comments only — never rendered.)

### P06 — APPLIED
"Planner interface (demonstration)." → "Planner interface." and "A counterfactual-injection demonstration is reported in Extended Data as an interface demonstration, not an external-validity result" → "This mapping is an interface specification, not an external-validity result". Fallback-never-Within-Norm sentence kept.

### P07 — APPLIED
`biblio.bib` @misc{onsite}: `note = {Verify the specific challenge edition/year for the data used}` removed; trailing comma on the `year` line removed; entry remains valid BibTeX. (Edition/year fact itself untouched — E6.)

## Tier 2

### P08 — APPLIED
`biblio.bib` @article{bethlehem2022brain}: 4,013-character author field → `Bethlehem, R. A. I. and Seidlitz, J. and White, S. R. and others`. Renders in main.bbl as "R. A. I. Bethlehem, J. Seidlitz, S. R. White, et al." (verified).

### P09 — APPLIED (1 wording deviation)
- "an \emph{readability} indicator" → "a \emph{readability} indicator" (old L229).
- u_a sentence (old L543–544) → "…; $u_a(t)$ is a dispersion statistic of the normalised candidate weights, not a standard deviation, and the reliability gate $u_a(t)<u_0$ of Eq.~\eqref{eq:readability} is operationalised by the following frozen rule: the reading is discarded as near-uniform when the largest normalised weight falls below $0.20$ (against $1/7\approx0.14$ for exactly uniform weights) or the candidate likelihoods are exactly tied."
  DEVIATION: plan draft said "the frozen rule below"; the rule is stated in the same sentence, so "the following frozen rule" — pure grammar, same content.
- Duplicate rule at old L556–558 shortened to "A reading failing the near-uniform rule above carries no discriminative information between the candidates, and the monitor abstains." (0.20 now stated once, as instructed.)

### P10 — APPLIED
Old L125–128 validation chain re-set as a two-line `aligned` block (line 1 ends after "conditional social atypicality →"; line 2 indented). Baseline log showed this display 192.8pt overfull; final log shows no overfull there. Page 3 of the PDF visually verified: both lines centred, nothing clipped.

### P11 — APPLIED (note for PI)
§2.6 opening → "Finally, what does a social flag register that a conventional safety check does not? The previous section sharpens the answer." Rest of the paragraph and the closing dissociation sentence kept, per plan.
NOTE (not acted on — plan quoted only the sentences): the \subsection heading still reads "What social monitoring adds beyond collision safety", which is the contest framing register C6 bans. Flag for PI whether the heading should follow ("What a social flag registers that safety checks do not" or similar).

### P12 — APPLIED (with 1 consequential layout repair)
"causal risk proxies" → "online-computable risk proxies" in Eq. (3) and Methods 4.3 ("online causal risk proxies" → "online-computable risk proxies"). First-use clause added after the equation: "; the risk proxies are those computable from information available at time $t$ (no observed post-encroachment measure enters online)."
DEVIATION (layout, forced by the longer term): eq. (3) was ALREADY 78.9pt overfull at baseline; the rename pushed it to 141.5pt (off the physical page). Re-set as a three-row `gathered` block (membership test on row 1; the $z_t$ tuple split over rows 2–3), same technique as P10. Final log: no overfull at eq. (3); page 8 visually verified, equation number (3) intact.

### P13 — APPLIED (Fig. 5 caption; 1 wording deviation)
(a) "blue marks moments" → "teal marks moments". (b) Margin definition sentence added after the panel-a opening ("The margin is the minimum time-to-collision … very large values arise where the paths are already diverging."). (c) Absolute medians added: "(medians $2.74$ vs $1.33$\,km\,h$^{-1}$ for the speed reduction and $5.53$ vs $2.93$\,km\,h$^{-1}$ for the speed range, read over a fixed three-second window)". (d) "panels c,d use $n=469$ and $n=10{,}483$" → "panel c uses $n=469$ atypical and $n=10{,}483$ within-range moments; panel d aggregates the acceleration records within those windows ($13{,}800$ vs $310{,}246$ records)". (e) "; the $472$ flagged moments cluster within $120$ scenario runs" added after "the unit of resampling throughout". (f) "All displayed thresholds are the frozen analysis battery; none was selected post hoc." added. Plus "over the $175$ scenarios" → "over the $175$ scenario runs" (P17 unit rule).
DEVIATION: panel-c opener "Counterpart response over the same window" → "Counterpart response from the same moment" — the plan's own added parenthetical fixes the window at three seconds, while panel a's window is the post-verdict window; "the same window" would have contradicted the new text. Matches P03's Methods phrasing "from the same moment".

### P16 — APPLIED
Algorithm 2 abstain line → `\textsc{Abstain}(\textsc{Ego-Unreadable} if $u_i$ or $q_i$ failed; \textsc{Counterpart-Unreadable} if $c_i$ failed)`.

## Tier 3

### P17 — APPLIED (1 unit fix folded in)
- §2.4: "The monitor is, first, able to speak." now followed by "The benchmark replays 15 scenario templates against 19 independent driving systems (267 scenario runs). Across its 67{,}861 candidate moments the monitor returns a verdict on 14{,}099 (about one in five; no engineering failures), and in 231 of the 267 scenario runs (nearly nine in ten) it speaks at least once." — replacing the vague "about one interaction moment in five … nine of every ten scenarios" clause (the plan's stray "L773" anchor = this L374 sentence; resolved by meaning). "No scenario is silenced" → "No scenario run is silenced" (plan's scenario-runs unit rule). The plain "15" here is the AV-benchmark count from the frozen AV-side file — deliberately NOT \targetnum-wrapped, and no \targetnum site was touched.
- Methods 4.5 universe ladder appended: "Of the $14{,}099$ judgeable moments at the $90\%$ level, $519$ lie below the range, $869$ above and $12{,}711$ inside; consequence analyses use the $175$ non-scripted scenario runs with recorded outcomes (a scripted-counterpart subset, $27$ runs, is excluded), and the ego-margin panels use the moments with a defined post-verdict margin ($472$ flagged, $11{,}669$ within-range)."
- Fig. 5 caption "175 scenario runs" done under P13.

### P18 — APPLIED
Old L417 → "Such moments are also rare: counting both sides, $9.845\%$ of judgeable moments are flagged ($1{,}388$ of $14{,}099$; $519$ on the assertive side, $869$ on the over-yielding side)." Lower side named assertive [VOCAB].

### P19 — APPLIED
Methods 4.3, after the $z_t$ definition sentence: reference pool HV–HV only ($2{,}442{,}625$ anchor rows), AV-agent interactions excluded ("machine behaviour cannot contaminate the human range"); corpora fleet-collected in specific cities in the United States and Singapore; "a sample of natural driving there, not of all human driving" (scope phrasing, per [PRESS]).

### P20 — APPLIED (three parts)
- 4.1: "MSE$_k$ is the mean squared error between the observed track and the candidate track" → "the mean squared Euclidean distance (m$^2$) between observed and candidate positions over the window, so $\sigma$ is in metres".
- 4.4 compact paragraph added after the P02 estimand block: fold sizes 974,984/499,893/481,088/486,660 (26,828 cases; whole-scene split frozen in advance); support gate = mean distance to k=25 nearest training anchors, abstain above 95th pct of guard distances (1.081), categorical support ≥50 anchors and ≥10 cases; test-fold support abstention 5.08% (24,723 of 486,660), zero categorical failures; natural-corpus readability accounting over 4,497,368 anchor rows: 3,202,646 readable, 1,275,480 near-uniform, 17,416 exact ties, 1,826 solver failures.
- Reporting sentence: "…the total abstention rate and its reason distribution, and case-balanced unconditional performance, so that filtering is never hidden" → "…and the total abstention rate with its reason distribution, so that filtering is never hidden" (unfrozen "case-balanced unconditional performance" dropped; no "case-balanced" token remains anywhere).

### P21 — APPLIED
LODO numbers attached to the boundaries sentence with a colon: 0.743 (Waymo, 143,380/193,096), 0.990 (nuPlan, 149,068/150,587), 0.750 (Lyft, 68,270/91,069), 0.900 (Argoverse-2, 11,315/12,576, with 44.3% held-out abstention---the support gate intercepting unsupported situations); "transfer across sources is therefore a bounded property of source heterogeneity, and coverage and abstention must be read together."

### P22 — APPLIED (Fig. 4 caption, all four edits)
(a) "is carried by the interaction itself and is not recoverable" → "is not recoverable". (b) width parenthetical "(at the $90\%$ level the conditioned width has mean $1.87$ rad, 5th--95th percentiles $1.35$--$2.28$ rad)". (c) "(the accepted moments; identical at all three levels)" after $n=461{,}937$. (d) "Verdicts elsewhere in this paper are issued at the $90\%$ level." inserted before the transfer-boundary sentence.

### P23 — APPLIED (1 dedup deviation)
- Fig. 2 caption panel a: parenthetical added "(the change in the concentration of the candidate weights---the reliability quantity of Eq.~\ref{eq:readability}---relative to control pairings; negative = more sharply identified)".
  DEVIATION: the caption's later clause "; more negative means more sharply identified" became an exact duplicate of the new parenthetical and was removed ("Points are means with $95\%$ confidence intervals over case clusters." kept).
- Methods 4.2 episode-summary sentence extended with the 0.26 rad / 7–22% sign-flip facts and "which is why the monitor never uses an episode summary online and every reading is per-moment" (cites Fig.~\ref{fig:measurement}c).
- "development split; confirmatory split held sealed" left untouched (deliberate, register C0E).

### P24 — APPLIED
Fig. 3 caption: after panel a — "The three bands cover the $37{,}495$ cases with a defined post-encroachment margin."; after panel b — "Panel \textbf{b} uses the cases entering the episode-level analysis for each source (Waymo $23{,}211$; Lyft $5{,}098$; Argoverse-2 $2{,}404$; nuPlan $7{,}498$ of the $23{,}218$/$5{,}105$/$2{,}406$/$7{,}499$ interaction cases)."

### P25 — APPLIED
Discussion, after "…created by, not hidden by, the present monitor.": "The evidence in this paper covers pairwise vehicle--vehicle interactions at mapped conflict points; interactions with cyclists and pedestrians, and multi-agent scenes, are outside the present reference and are the natural next extension of the same construction."

### P26 — APPLIED
- Methods 4.4 after the verdict list: "In the main text, \textsc{Competitive-Deviation} is written `outside the range on the assertive side' and \textsc{Over-Yielding} `outside on the over-yielding side'; both sides are tested at every judgeable moment."
- Caption harmonisation: Fig. 1c "above the $90\%$ range" → "above the $90\%$ range (the over-yielding side)"; ED1 "above it (atypical)" → "above it (the over-yielding side, atypical)" (its "below it (more assertive, atypical)" already conformed). Fig. 1b's generic two-sided "outside it (atypical)" mention names no side by design and was left.

### P27 — APPLIED
Empty "Additional Information / Supplementary Information / Supplementary Tables / Supplementary Figures" headers deleted. "Competing Interests" kept; Acknowledgements / Author Contributions untouched (pending E8).

### P28 — APPLIED (1 placement deviation)
Statistic "(paired $90\%$ interval-score difference $-0.0002$, case-clustered $p=0.86$)" attached to the existing "the counterpart channel in particular is statistically indistinguishable from removing the IPV input given the situation" clause.
DEVIATION: the plan's insertion point (directly after "add no measurable value") would have restated the indistinguishability claim the sentence two lines later already makes; the statistic was attached to that existing clause instead. Same content, no duplication. "Winkler" naming not repeated (the word already appears once in the adjacent P01-frozen sentence).

### P29 — APPLIED
End of Methods 4.5: "The fraction of moments whose reading carries discriminative information differs between the naturalistic corpora and the benchmark ($70.3\%$ vs $55.3\%$); we report this as an observed difference without attributing a cause, and the readability gate treats both populations identically."

### P30 — APPLIED
§2.4 consequence paragraph: "…a convergence rather than a restatement (Fig.~\ref{fig:consequence}); the association survives a placebo test that reassigns whole flag sequences across scenario runs; a single run containing flagged assertive moments is shown for illustration in Fig.~\ref{fig:edcase}." (Mechanics in Methods 4.5 per P03; "Associations are descriptive" retained in the caption.)

---

## Constraint verification (final state)

| Check | Result |
|---|---|
| Abstract | byte-identical to HEAD (diff of the abstract environment: empty; no hunk touches old L75–89) |
| `\targetnum` | 21 tokens exactly as baseline (14 call sites + 1 macro def + 6 comments); `git diff` contains **no** line with `targetnum` — every call-site argument byte-identical |
| `estimability` family | 0 occurrences in main.tex and biblio.bib |
| "passive" | 0 occurrences; lower side written "assertive side" throughout |
| Model codenames / audit vocabulary in main text | none introduced; "case-balanced" fully removed; rendered `claims_register.md` reference removed (P05); remaining matches are LaTeX comments only |
| P01 targets (escalation-blocked) | both ">40%" sentences byte-identical (§2.3 old L320–323; Methods 4.3 old L576–578) |
| figH files | `figures/figH_human_arm_TARGET_SYNTHETIC.pdf/.png` untouched; `git status figures/` clean |
| Doubled words ("the the" class) | cross-line scan clean (only false positive: "Argoverse-2 2{,}406") |
| Files modified | `main.tex`, `bibliography/biblio.bib` — nothing else |

## Compile evidence

Sequence: `pdflatex -interaction=nonstopmode main.tex; bibtex main; pdflatex ×2` (plus one extra pdflatex pass after the eq. (3) layout fix). All exit 0.

- Errors (`^!` in main.log): **0**
- Undefined citations/references: **none** (grep of main.log)
- BibTeX: `main.blg` — 0 errors, 0 warnings; bethlehem2022brain renders "…White, et al."; onsite entry renders without the instruction note
- Pages: **32** (`Output written on main.pdf (32 pages)`); baseline 31 — growth from the Methods additions (P17/P20/P21)
- Overfull hboxes: baseline had 5 → final has 3, all three **pre-existing at baseline and out of plan scope** (22.9pt in the eq. (1) align; 14.9pt and 6.5pt paragraph hyphenation around "counterpart-identity"). The two display overflows in scope are fixed: validation chain 192.8pt → none (P10); eq. (3) 78.9pt (baseline) → none (P12 repair). Pages 3 and 8 rendered to PNG and visually confirmed un-clipped.

## Open notes for the coordinator / PI

1. P11: §2.6 heading "What social monitoring adds beyond collision safety" still carries the C6 contest framing; plan quoted only the opening sentences. Decide whether the heading follows.
2. Repo handoff contract (append to `…/PAPER001…/agent_handoff.md` in the research repo) NOT executed by this pass — my edit permission was explicitly limited to main.tex + biblio.bib. The coordinator should append the round-1 entry once text + figure (P14/P15) + escalation items land together.
3. structure.md still carries ">40% width reduction" (plan note §5, tied to E1) — untouched per plan.
4. Build products (main.pdf/aux/log/blg/bbl) left untracked as required.
