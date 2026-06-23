# RQ012A Plan Review

Worker: RQ012-W01-plan-review  
Role: independent plan review, phase 1  
Run ID: RQ012_1_event_annotation_readiness_20260623T104749+0800_1f52ac37  
Reviewed SPEC: reports/plans/RQ012_plan_v0_onsite_event_annotation_readiness_20260622.md

## Snapshot And Scope Checks

- SPEC SHA256: 921f6bb3b850126189895dcca52f054a1c6f5e95a16b05159bec0a13c26ad77e.
- Snapshot SHA256: 921f6bb3b850126189895dcca52f054a1c6f5e95a16b05159bec0a13c26ad77e.
- Byte identity: PASS. `cmp -s` reported identical.
- Git HEAD: 38063a2ff9cdc717098cf3f821c2bb162a0ac1d9.
- Scope: read START_HERE.md, AGENTS.md, CLAUDE.md, SPEC, spec snapshot hash/identity, RQ012 run directory names, RQ003 directory/file names, and README/codebook headings only. No PAPER_REPO files, raw media, raw trajectories, IPV values, score/rank/team values, labels, agreement computation, or event-IPV analysis were read or generated.

RQ003 foundation sanity check: PASS. Directory/file names plausibly show a codebook, two annotator templates, mechanism and validation sample manifests, blinded item materials, an anonymization audit, an artifact manifest, a controlled identity map in RQ003_6, and merge scripts in RQ003_6/RQ003_8. README/codebook headings confirm blind behavioral annotation material exists without opening raw media or protected values.

## Consolidated Axis Ratings

| Axis | Rating | Justification |
| --- | --- | --- |
| 1. Candidate event list | FAIL | W0 lists physical and interaction events, including hard braking, deceleration, jerk, forced yielding, stop-go, near miss, intervention, fallback, replanning, rejection, and mission failure (lines 54-70), and W1 requires class separation (lines 94-95). However W5 retains construct-proximal labels such as aggressive intrusion and over-yielding/freezing (lines 142-153) while the research question asks for a behavioural consequence reference for competitive and over-yielding IPV deviations (lines 10-12). The SPEC does not explicitly prevent those construct-proximal motifs from becoming primary consequence endpoints, which risks circular validation. |
| 2. Denylist | WEAK | The denylist covers IPV/deviation labels, official coordination scores/ranks/team identities, future associations, outcome-dependent thresholds, simulated/model labels, and another annotator's completed labels (lines 40-48). It is mostly sound, but it does not explicitly deny indirect leakage proxies such as area/scenario/run-path/order encodings during all pre-annotation work, nor borrowed prior human labels outside the current annotator-pair wording. W4 later covers filenames, metadata, thumbnails, paths, ordering, team, area, score, rank, and IPV for blind-package audit (lines 137-138), so this is not a standalone blocker. |
| 3. Threshold sources | FAIL | Permitted threshold sources are generally outcome-blind (engineering/safety standards, literature, measurement resolution, platform thresholds; lines 99-105), and the SPEC forbids choosing thresholds by later association strength/sign (lines 107-108). The development-subset option at line 105 is not specified tightly enough: it lacks pre-frozen selection rules, seed/hash/provenance, no-overlap rules with formal annotation/validation samples, scenario stratification rules, and a rule that subset construction itself cannot use official outcomes or outcome-correlated proxies. Sensitivity bands are required (line 107) but not enough to prevent leakage if the development subset is chosen after outcome knowledge. |
| 4. Blinding methodology | WEAK | The leakage audit explicitly includes filenames, metadata, thumbnails, paths, ordering, team, area, score, rank, and IPV (lines 125-138). Training and formal samples are separated in W5/W6 (lines 155-156, 160-166). The plan should still require a blind issuance manifest with randomized order, neutral IDs, stripped embedded media/trajectory metadata, and checksums so that the audit is reproducible. |
| 5. Real two-human requirement | PASS | W6 requires at least two independent human annotators, blinded materials only, no mutual label access, separated training/formal samples, raw label preservation, and no adjudication before independent agreement statistics (lines 160-169). Gate 012-3 keeps status at BLOCKED_FOR_HUMAN_LABELS until two real annotators complete the task (lines 211-218). Acceptance criteria and non-goals prohibit simulated labels/model substitutes (lines 261-263, 271). |
| 6. Merge/agreement protocol | WEAK | W7 freezes the primary agreement statistic, prevalence-aware secondary statistic, clip/event level agreement, minimum usable agreement, missing/uncertain handling, adjudication, and later-analysis criteria before labels are opened (lines 171-185). W8 rejects empty templates, copied duplicates, simulated labels, wrong item IDs, identity-revealing files, and incomplete fields (lines 186-195). The remaining weakness is operational: "simulated labels" cannot be reliably rejected by content tests alone, so the plan needs provenance/attestation or audit metadata. |
| 7. Stop conditions and gates | FAIL | Gates 012-0 through 012-3 are ordered before Gate 012B (lines 197-228), and Wave A repeatedly forbids event-IPV analysis (lines 14-15, 123, 268). However Gate 012B's explicit prerequisite list only names two real annotation files, agreement, and RQ007/RQ009/RQ011 freezes (lines 220-228). It does not explicitly require Gates 012-0, 012-1, 012-2, and 012-3 to have PASS status, nor frozen ontology/threshold/agreement-protocol deliverables, before later event-IPV analysis. That makes the authorization gate bypassable on the face of the SPEC. |
| 8. Scope boundary | WEAK | The SPEC correctly forbids Wave-A event-IPV association and causal claims (lines 14-15, 123, 268-270), and it correctly states the behavioural reference is not fully independent ground truth because it uses the same observed trajectory behaviour (lines 271-273). The weakness is wording: line 11 calls the reference "independent", while the non-goal later qualifies that independence. The plan should consistently call it an outcome-blind behavioural reference unless a stricter independence tier is defined. |

## Blocking Findings

### B01 - Construct-proximal motif labels may become circular consequence endpoints

- Axis: 1 and 8.
- Finding: W5 retains aggressive intrusion and over-yielding/freezing labels (lines 142-153) while the RQ seeks an independent behavioural consequence reference for competitive and over-yielding IPV deviations (lines 10-12). The SPEC does not state that construct-proximal motifs are ineligible as primary consequence endpoints.
- Why it blocks: Later validation could test an over-yielding or competitive IPV deviation against a human label that semantically restates the same construct, creating circular evidence rather than a consequence reference.
- Required fix: Add an ontology eligibility field that separates independent consequence endpoints, construct-proximal descriptors, planner/system events, and annotation-quality labels. Explicitly forbid construct-proximal descriptors such as aggressive intrusion and over-yielding/freezing from serving as primary event-IPV validation endpoints unless treated only as secondary descriptive context.

### B02 - Development-subset threshold source is under-specified

- Axis: 3.
- Finding: W2 permits distributions in a development subset that excludes IPV and official outcomes (line 105), but does not define how that subset is selected, frozen, separated from formal validation/annotation materials, or protected from outcome-correlated proxies.
- Why it blocks: Thresholds could be tuned on a subset chosen with prior outcome knowledge or proxy information while still appearing to exclude explicit IPV/official outcome columns.
- Required fix: Predefine development-subset construction before threshold work: sample frame, random/scenario-stratified method, seed/hash, no overlap with formal annotation/validation items unless explicitly justified, prohibited metadata/proxies, authorized builder role, and a frozen manifest produced before viewing event frequencies beyond outcome-blind data-health summaries.

### B03 - Gate 012B is not explicitly dependent on all Wave-A gates

- Axis: 7.
- Finding: Gate 012B allows later event-IPV analysis only after two real annotation files, agreement, and RQ007/RQ009/RQ011 freezes (lines 220-228), but does not explicitly require Gates 012-0, 012-1, 012-2, and 012-3 to have PASS status, nor require frozen Wave-A ontology, threshold, blind-package, and agreement-protocol deliverables.
- Why it blocks: A future executor could satisfy the literal 012B list while bypassing signal feasibility, threshold-blinding, blind-package integrity, or human-readiness gates.
- Required fix: Amend Gate 012B to require documented PASS for Gates 012-0, 012-1, 012-2, and 012-3; frozen ontology and threshold rationale; frozen blind-package audit; frozen agreement-analysis protocol before opening agreement results; and explicit authorization record before event-IPV analysis.

## Non-Blocking Findings

### A01 - Candidate list should include or explicitly exclude additional consequence classes

- Axis: 1.
- Finding: W0 covers several important event families (lines 54-70), but it does not explicitly list collision/contact, off-route or no-progress timeout, human/operator takeover, emergency stop, abrupt lateral comfort events, or yielding to non-counterpart actors.
- Required fix: Add these as audit candidates or document why they are unavailable/non-goals under W0/W1.

### A02 - Denylist should name indirect leakage proxies globally

- Axis: 2 and 4.
- Finding: The denylist is strong for explicit outcomes (lines 40-48), and W4 covers blind-package leakage fields (lines 137-138). The same proxy language should be promoted to the general denylist for all pre-analysis work.
- Required fix: Add area/scenario/run IDs, filenames, paths, order, thumbnails, manifest-derived strata, and prior annotation files as prohibited leakage channels unless explicitly approved as outcome-blind design variables.

### A03 - Blind issuance should be reproducibility-auditable

- Axis: 4.
- Finding: W4 requires leakage audit over annotator-facing surfaces (lines 125-138), but does not specify checksums, randomized issuance order, neutral item IDs, or embedded metadata stripping.
- Required fix: Require an issuance manifest with checksums, neutral IDs, randomized order seed, metadata-stripping evidence, and auditor sign-off before materials are released.

### A04 - Simulated-label rejection needs provenance evidence

- Axis: 6.
- Finding: W8 says the merge pipeline should reject simulated labels (line 192), but a merge script cannot reliably infer simulation from CSV content alone.
- Required fix: Add annotator attestation, file provenance checks, controlled submission channels, and metadata/audit logs as part of the merge-validation protocol.

## Verdict

VERDICT: BLOCKED
