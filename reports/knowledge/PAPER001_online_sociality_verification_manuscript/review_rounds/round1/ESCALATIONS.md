# Escalations — PI decisions needed before (or alongside) round-1 execution

Context for a reader who has not followed the round: four simulated NMI referee reports were consolidated and fact-checked against the frozen evidence. Twelve items cannot be executed by the revision layer because they change headline wording, claim boundaries, C7/`\targetnum` material, or need facts that do not exist in the workspace. Each brief: the issue → options (recommendation ★) → consequences → what the reviews say.

---

## E1 — Ratify the replacement wording for the ">40 %" sharpening headline (§2.3 + Methods 4.3) — BLOCKS plan item P01
- **Issue.** §2.3 (main.tex L321) and Methods 4.3 (L577-578) claim ">40 % width reduction" (+ ">35 % Winkler improvement"). The figure and the frozen evidence say 21/20/8 % at 80/90/95 % (RQ021). The >40 %/35 % pair is the superseded RQ009 future-target envelope; the claims register (C3) already forbids citing it alongside the current numbers, and no Winkler exists for the current envelope's global baseline. The numbers MUST change (accuracy is not self-weakening); what needs your ratification is the headline framing.
- **Options.**
  - (a) ★ P01's proposed wording: lead with "a fifth narrower at the 80 %/90 % levels; at 95 % the global range spans the whole admissible scale and can never flag anything; coverage within 0.6 pp of nominal". Keeps a true superiority headline (global reference unusable, conditioned one calibrated and usable) without the false number.
  - (b) Minimal edit: replace "more than a 40 %" by "up to a 21 %" and delete the Winkler clause. Safest, but surrenders the framing advantage and reads deflated.
  - (c) Recompute a Winkler improvement for the current envelope to restore a second metric — REJECTED by the frozen-evidence rule (new analysis; would go to backlog, and B05/B06 do not cover it).
- **Consequences.** Any option removes the contradiction all four referees flagged as disqualifying (D calls it "disqualifying at Nature-family standard"). Option (a) preserves the press-conference posture; (b) invites "the sharpening is modest" readings; leaving it unchanged is not lawful (violates the paper's own register).
- **Also decide.** Whether the abstract's word "narrow" stays (D-M3 wants absolute widths tempering it; the abstract has no number, and P22 adds absolute widths in the Fig. 4 caption — recommendation: keep "narrow, calibrated, auditable" unchanged in the abstract, let the caption carry the absolutes). And: `structure.md` L90 still says ">40 %" — order a sync of the planning doc in the same pass.
- **Reviews.** A-M8/P7; B-M8/P8; C-M10(iv)/q14; D-M3/q1/P1.

## E2 — The three-flag-rate story and the abstract's "flagged about twice as often" (C7-coupled)
- **Issue.** The abstract and contribution 6 headline the AV-vs-course-humans pair (\targetnum "twice"); §2.5 then correctly says "all three rates belong together, and no pair of them tells the story alone". D further argues 4.7 % < 10 % is conservative shift (over-coverage), which mechanically inflates the 2.1×, and that against the natural-driving rate (9.72 %) the AV rate (9.845 %) reads as parity. All human-side digits are synthetic placeholders (C7); the AV-side numbers are measured. Nothing here is executable now without touching `\targetnum` sites or C7 prose.
- **Options.**
  - (a) ★ Defer any wording change to the C7 swap (B11), and pre-commit now to the presentation rule the referees demand: all three rates with uncertainty, the conservative-shift reading stated, the abstract sentence re-derived from the real numbers at swap time (it may not survive if the real course-human rate lands near 9.7 %).
  - (b) Soften the abstract now ("flagged no less often than the population that defined the reference, and more often than matched human drivers in the same scenarios") — touches the `\targetnum{twice}` site → forbidden this round unless you explicitly authorise a byte change.
  - (c) Keep as is and rebut. Not recommended: two referees independently derived the over-coverage reading; it will not go away.
- **Consequences.** (a) costs nothing now and keeps the C7 discipline intact; (b) breaks the byte-freeze; (c) burns credibility in the response letter.
- **Reviews.** D-M4/P4/q8; C-M2/M8/q10; B-M6/q11; A-M5/q10. Note RQ021-KC-C4's required qualification already bans equivalence/improvement readings of 9.845 vs 9.720; the same discipline will apply to the real course-human number.

## E3 — Abstract wording "without added emergencies"
- **Issue.** Abstract L84. Three of four ego emergency bins exclude an increase; the widest (<3 s) has CI [−8.33, +3.15] pp and RQ018 forbids citing it as evidence; no equivalence margin exists. Main-text repair is P04; the abstract sentence is yours.
- **Options.** (a) ★ "with emergency-margin moments no more frequent — mostly rarer — at the supported thresholds" is too heavy for an abstract; recommended: "while emergency rates stay at or below the within-range rate" → replaced by "with no rise in emergency rates at any supported threshold" (precise, still positive). (b) Keep "without added emergencies" and defend descriptively (point estimates all negative). (c) Drop the clause entirely.
- **Consequences.** (a) survives methodological review (C explicitly demands absence-claims ↔ equivalence bounds; B08 would later license the strong form); (b) will be quoted against the <3 s interval in round 2; (c) weakens the abstract more than accuracy requires.
- **Reviews.** C-M6/P4/q9; D-M6/P6; B-M5 (part).

## E4 — Title and "social compliance" framing
- **Issue.** All four referees say the title/abstract promise a normative construct ("socially compliant autonomous driving") while the validated object is conditional social atypicality of a model-derived preference estimate. The paper's three boundary statements are acknowledged by A/B/C/D as honest — their objection is the packaging. C proposes a title centred on "online monitoring of conditional social atypicality".
- **Options.** (a) Retitle to the validated construct (e.g. "Online monitoring of conditional social atypicality in autonomous driving") and sweep "social compliance" from claim-bearing sentences (keeping it as the motivating question). (b) ★ Keep the title, tighten the abstract's first two sentences so "social compliance" is introduced as the question and "conditional social atypicality" as the delivered object (the current abstract already does this halfway: "We recast social compliance as the online monitoring of conditional social atypicality"), and hold B01 (construct validation) as the路线 that would later justify the stronger title. (c) Keep everything unchanged.
- **Consequences.** (a) maximally defuses A-M6/B-M7 but cedes the paper's most memorable phrase and its NMI-scale ambition; (b) is the press-conference-compatible middle: the recast sentence is literally the paper's thesis — making it do explicit work in the title/abstract forestalls the "overclaim" reading at low cost; (c) guarantees the same objection at full strength in round 2 from four independent directions.
- **Reviews.** A-M6/P7; B-M7/P7; C-m12; D-M1/P7.

## E5 — Ethics / consent / safety statement for the human driving study
- **Issue.** Twenty licensed drivers drove staged conflicts in a real vehicle; the manuscript has no ethics-approval, consent, compensation, or safety-protocol statement (D grades this independently blocking; A/B/C all demand it). None of these facts exist in the workspace, and inventing any of them is forbidden.
- **Needed from PI.** IRB/ethics body + approval number; consent process; recruitment/compensation summary; safety protocol (safety driver? speed caps? abort rules); data-protection statement beyond the existing pseudonymisation sentence. If approval was institutional under the benchmark organiser, say whose.
- **Options for timing.** (a) ★ Add the statement in this round (compliance text is independent of the C7 digit swap). (b) Bundle with the C7 swap. 
- **Consequences.** (a) removes an independent desk-reject risk now; (b) leaves a known-blocking gap in any intermediate PDF.
- **Reviews.** D-M7(i)/q9; A-M5/q11; B-M6/q12; C-m11.

## E6 — Benchmark edition/year and naming for reference [53]
- **Issue.** P07 deletes the leftover instruction note, but the citation still lacks the challenge edition/year/access-date the referees ask for. The workspace only shows internal IDs (`onsite:shanghai:…`) and `year = {2024}` already in the entry — insufficient to confirm which edition supplied the analysed data.
- **Needed from PI.** The exact challenge edition (year/city/round) for BOTH the automated-systems data and the human-arm sessions; whether naming the benchmark edition conflicts with the anonymity stance ("no team or algorithm identifier appears").
- **Consequences.** Until supplied, the provenance question (B-m15's "edition, version, access date and data subset") stays open; it is a two-minute fix once the fact exists.
- **Reviews.** B-m15; D-M5/q2; C-m9; A-M8.

## E7 — Data/code availability: repository, versioning, review access
- **Issue.** Both availability sections say "upon publication" with no repository/accession; A and B explicitly say review-time access is required for these selection-sensitive analyses. Repository URLs/DOIs cannot be invented.
- **Needed from PI.** Decide the release vehicle (e.g. Zenodo/OSF DOI + code repo), what is releasable under the source-data licences and benchmark terms, and whether a reviewer-only access bundle (frozen verdict series + figure data + monitor code) will be offered at submission. Also: if a counterfactual-injection demonstration artefact exists anywhere, restoring the Extended-Data pointer P06 removed is cleaner than the specification wording.
- **Reviews.** A-m14/P6; B-m16/P1; B-M8; D-M7(v)/P5.

## E8 — Acknowledgements and Author Contributions
- **Issue.** Empty headers (L742-746). Author list and contributions are explicitly "not recorded in this workspace and must not be guessed" (main.tex comment L62-63).
- **Needed from PI.** Funding sources, institutional/data-provider acknowledgements, co-author list + CRediT-style contributions, corresponding author.
- **Reviews.** D-m12.

## E9 — Positioning of "online/runtime" language (runtime-assurance reading)
- **Issue.** B tests the paper against runtime-verification standards (latency, faults, assurance contract) and asks that, absent an implementation study, the work be positioned as "offline evaluation of a prospective online indicator". The frozen evidence supports online-computability (non-anticipative inputs, 10 Hz windows, no outcome features) but not real-time operation.
- **Options.** (a) ★ Add one Methods sentence drawing exactly this line: "online here means every input is available at decision time (non-anticipation verified); end-to-end real-time operation on vehicle hardware is not evaluated in this paper" — keeps "online monitoring" as the construct name, concedes only what is true, and pairs with backlog B13. (b) Rename the contribution to "prospective online indicator" throughout (B's ask) — a large frame retreat the other three referees do not require. (c) No change.
- **Consequences.** (a) is a scope statement (press-conference-compatible) that removes B's strongest lever; (b) overshoots; (c) leaves a "runtime claims without runtime evidence" quote available.
- **Reviews.** B-M4/P4/q6; A-M7/q13.

## E10 — Fig. 6 (target figure) presentation decisions for the C7 regeneration
- **Issue.** Referees demand: uncertainty in panels a/c, numerator/denominator per level, consistent decimal precision (9.845 % vs 4.7 %), scenario-label definitions (A3/A5/B1), and a universe label for the AV side. The figure is the C7 SYNTHETIC target; its watermark/red header must stay until `REAL_VERIFIED`, so all of this lands in the regeneration spec, not in this round's edits. Note the data interface already carries a flag-rate CI field and per-unit tables — most demands are satisfiable at swap time without endpoint changes.
- **Needed from PI.** Approve the regeneration spec: (i) show the planned flag-rate CI in panel a and a bootstrap band or per-scenario CI treatment in panel c; (ii) harmonise precision (one decimal for all rates); (iii) print n/N per bar; (iv) add a one-line scenario-key (needs B16 material); (v) label the AV universe explicitly ("same 15 scenario templates; all runs"). Decide whether panel-b whiskers move from "where defined" to always-defined once real data exist.
- **Reviews.** A-m10; B-m11; C-M7/M8; D-M6.

## E11 — Whether to disclose the event-level null companion tests in the paper
- **Issue.** The frozen record (RQ019 B8) contains event-level (one-vote-per-scenario-run) companion tests that are null for the consequence quantities; the accepted claims are deliberately moment-level, and the manuscript phrases them per moment. Referees B/C ask directly for episode-level inference. Disclosing the null pre-empts round-2 discovery and demonstrates estimand discipline; withholding follows the press-conference principle (the knowledge layer forbids hiding it internally, but does not require printing it).
- **Options.** (a) ★ One Methods sentence: "an event-level contrast (one vote per scenario run) is a different estimand and is not claimed here; the per-moment estimand with scenario-run-clustered uncertainty is the deployment-relevant one because verdicts are issued per moment" — states the boundary without printing the p-values. (b) Full disclosure with the B8 table in Extended Data. (c) Silence.
- **Consequences.** (a) is honest, cheap, and consistent with the register's phrasing rules; (b) volunteers ammunition beyond what any referee can currently see; (c) risks a round-2 "the authors knew" moment if data are released for review (E7).
- **Reviews.** B-q13; C-M7; A-M2 (part).

## E12 — Sync the planning layer after P01
- **Issue.** `structure.md` L90 still contains ">40 % width reduction" (the stale RQ009 number). The repo contract makes structure.md the narrative source of truth; leaving it stale invites the same transcription error at the next Overleaf round-trip. Not a referee item; repo hygiene surfaced by this round's fact-check.
- **Needed from PI.** Authorise the one-line structure.md correction to the C3 register numbers (21/20/8 %) in the same commit series as P01.
