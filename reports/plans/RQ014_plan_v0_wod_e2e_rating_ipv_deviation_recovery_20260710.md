# RQ014 Plan v0 — WOD-E2E rating↔IPV-deviation lost-result recovery (configuration-space search)

Status: PROPOSED (PI review pending). Author: Claude (PI role), 2026-07-10.
Execution model: Claude = principal investigator (judgment only); codex CLI fleet = all
hands-on implementation/experiments; HPC (Tongji, `/share/home/u25310231/ZXC`, job prefix
`zxc-`) for compute.

## 0. Background and premise

A prior independent study (data and configuration now confirmed lost) reportedly showed a
strong, significant association on WOD-E2E: higher human rating → lower deviation of the
candidate trajectory from the human IPV envelope (IPV verifier score), and vice versa.
The PI remembers only the direction and significance — not the effect size, unit of
analysis, counterpart source, envelope definition, trajectory-frequency handling, or the
IPV estimation window (history / future / combined).

Tension: the registered RQ010B result (2026-07-03, decision.md) is a bounded NULL for the
executed operationalizations (Scheme 1 future-only n=75 ρ=0.148 wrong-sign; Scheme 2
history+future n=98 ρ=0.031; 10 Hz sensitivity null; full479 audited pooled ρ=-0.038,
n=906). RQ014 therefore is NOT free confirmation of a believed result; it is a registered
recovery search whose three possible outcomes are all informative:

- **RECOVERED_ROBUST** — a defensible configuration reproduces the remembered effect and
  survives sealed confirmation + red team + replication → formally revise the RQ010B
  boundary via a new accepted claim.
- **ARTIFACT_RECOVERED** — the remembered effect reappears only under a configuration
  containing a known defect (e.g., dt mismatch, candidate-as-own-reference, degenerate
  IPV fallback) → the lost result is explained as an artifact; RQ010B stands.
- **NOT_RECOVERED** — no cell in the registered space reproduces it → RQ010B null is
  strengthened; the lost result remains unexplained.

No manuscript claim may change except through `reports/knowledge/RQ014_*/decision.md`
after Gate G5.

## 1. Research question

RQ014: Does there exist, within a pre-registered configuration space of WOD-E2E
IPV-deviation operationalizations, a defect-free configuration under which human rating
is negatively and significantly associated with deviation from a human IPV envelope,
replicating on a sealed confirmation split under family-wide multiplicity control?

## 2. Phase structure

### Phase F — Forensics first (cheapest path to recovery; run before any compute)
One codex agent sweeps for residue of the lost study before we burn a search budget:
- HPC: `sacct` full accounting history (job names/scripts/workdirs), `~/.bash_history`,
  all result/log/code dirs under `/share/home/u25310231/ZXC/` (including deleted-job
  sbatch files under `scripts/`, stray result dirs, python files matching
  rating|score|envelope|deviation).
- Local: `archived/` (compat wrappers, legacy_scripts, report_process, report_local_state),
  `main_workflow.log` full read, git reflog + dangling objects in this repo and the paper
  repo, `~/.rq009_codex_fleet`, sibling project folders under `../`, OneDrive version
  history for candidate files.
- Session transcripts of prior agent runs, if retained.
Deliverable: forensic report; any recovered config fragment becomes a **prioritized cell**
in the registry. Gate G1: PI reads report before Phase S launch.

### Phase R — Registry freeze (pre-registration)
Freeze `config_space.yaml` enumerating every tested cell (axes §3), the split (§4), the
acceptance tiers (§4), and the defect catalog (§5). SHA-256 recorded. After freeze, no
axis may be added without a v1 amendment logged before viewing any new results.

### Phase H — Harness + calibration anchors
Implementer agents build the sweep harness on HPC reusing: rated479 extracted segments
(`data/rated479_segments/`, 479/479 with ≥91 frames 10 Hz history), cached full479
detections/tracks (result dirs of 20260630), the gated counterpart selector, and the
**pinned legacy IPV estimator HEAD `5edd2810`** (local estimator has known drift
`a0fee535`; sigma01-compatible IPV requires the pinned HPC build).
**Gate G2 (harness validity):** before any search, the harness must exactly reproduce the
published RQ010B anchor numbers (Scheme 1 ρ=0.148/n=75; Scheme 2 ρ=0.031/n=98; 10 Hz
ρ=0.165/0.128; full479 pooled ρ=-0.0384/n=906) by setting the corresponding cells. A
harness that cannot reproduce known outputs cannot be trusted to find lost ones.

### Phase S — Screening (discovery split only)
Run the full registered grid on the discovery half. Rank cells by signed Spearman ρ
(rating vs deviation; remembered sign is negative). Leaderboard + per-cell health metrics
(IPV non-degeneracy, n, scene coverage). PI judgment at Gate G3 selects ≤5 candidate
cells (prioritizing defect-free cells; defect-bearing hits are tracked separately for the
ARTIFACT branch).

### Phase C — Confirmation (sealed split)
The confirmation half is untouched until Phase C. Each candidate cell runs once, frozen.
Family-wide max-statistic permutation over the ENTIRE registered grid (not just the
candidates) controls the search multiplicity. Acceptance per §4.

### Phase V — Validation
Red-team agent attacks surviving cells with the defect catalog (§5); a blind replicator
agent reimplements the top cell from the written spec only (no code access) on the sealed
split. Divergence is a finding.

### Phase D — Decision
Synthesis to `reports/knowledge/RQ014_wod_e2e_rating_recovery/decision.md`; STUDIES.md and
RQ010B decision cross-reference updated per outcome; workflow log entry.

## 3. Configuration space (registry axes)

Two-layer cost design: **expensive layer** = anything requiring IPV re-estimation;
**cheap layer** = post-processing over cached per-cell IPV series. Cheap-layer axes
multiply freely; expensive-layer combinations are capped (≤ ~96 configs) with a pilot
cost measurement (12-segment pilot per expensive config) before full submission.

Expensive layer:
- **E1 Trajectory frequency/preprocessing:** native 4 Hz; 10 Hz interpolated; resample
  dt ∈ {0.1, 0.25, 0.5}; smoothing ∈ {none, Savitzky-Golay}; estimator-dt matched vs
  MISMATCHED (mismatch is a *defect cell*, kept deliberately to test the ARTIFACT branch).
- **E2 IPV window:** history-only trailing hw ∈ {1 s, 2.5 s, 4 s}; future-only (candidate
  horizon); history+future combined; computed on candidate vs driven trajectory.
- **E3 Counterpart source:** gated multi-frame tracks (RQ010B final gates); relaxed gates
  (pre-gate track set); single-frame + constant-velocity extrapolation (early-pilot
  variant; defect-adjacent, flagged); nearest-track without conflict gate.
- **E4 Ego reference line:** §6 route-based constant-curvature reference; candidate-as-own
  reference (*known defect cell*, kept for the ARTIFACT branch).

Cheap layer:
- **C1 Envelope/reference:** path-type HV norm (RQ010B); InterHub sigma01 static
  state-conditioned quantile envelope (support gate ON and OFF); RQ009 M3 conformal
  (gate ON and OFF); per-scene candidate-set-internal envelope.
- **C2 Deviation metric:** signed/absolute band exceedance at {80, 90, 95}%; |IPV − norm
  median|; fraction-outside over the window; max exceedance.
- **C3 Statistical unit:** pooled candidates; within-scene rank (Spearman/Kendall mean);
  scene-mean.
- **C4 Scene subset:** full retained set; Scheme-1-like; Scheme-2-like.

Every executed cell — including failures and degenerate cells — is catalogued; the
max-stat family is the full catalog.

## 4. Split and acceptance (frozen at Phase R)

Split: 479 rated scenes split ~50/50 discovery/confirmation, stratified by path-type
geometry and abstention status, seed fixed in the registry. Ratings are already unblinded
at the project level (RQ010B); the scene-level sealed split + registry freeze is the
guard against selection leakage into confirmation.

- **T0 candidate (discovery):** ρ ≤ −0.30, cell p < 0.01, IPV non-degenerate
  (≥50 % |IPV|>1e-6, ≥20 distinct values), ≥30 scenes contributing.
- **T1 RECOVERED (confirmation):** frozen cell on sealed split: ρ < 0 with family-wide
  max-stat permutation p < 0.05; leave-one-scene-out sign-stable; not attributable to any
  defect-catalog item.
- **T2 manuscript-grade:** T1 + red team CLEAR + blind replication AGREE + registered
  statement of whether the effect survives partialing the RQ010B physics features
  (`driven_ade/fde` etc.) — survival not required for recovery, but the claim wording
  depends on it.
- **ARTIFACT_RECOVERED:** a cell meets T0/T1 magnitudes but only with ≥1 defect-catalog
  item active, and its defect-free twin cell fails → registered artifact explanation.

Because only direction+significance are remembered, no effect-size match to memory is
required; T1's family-wide control is the sole significance bar.

## 5. Defect catalog (red-team checklist, frozen)

1. Estimator dt ≠ trajectory dt (reproduces the invalid 20260629 pilot degeneracy).
2. Candidate-as-own-reference-line ego reference.
3. Uniform-fallback zero-IPV rows dominating (ego_ipv=0 fallback).
4. IPV value rate-sensitivity (4↔10 Hz Spearman ~0.3) exploited by post-hoc frequency pick.
5. Rating join before estimation (ratings must join only at the final test step).
6. Single-scene / few-scene dominance; n<30 cells.
7. Envelope built on data overlapping the evaluation scenes (leakage).
8. Support/OOD gate silently off where the norm is OOD (M3 ≤15 % in-support finding).
9. Local drifted estimator used instead of pinned `5edd2810`.
10. Sign-convention flips across modules (θ>0 prosocial).

## 6. Fleet decomposition (PI = Claude; executors = codex exec, xhigh)

| Agent | Role | Phase | Deliverable |
|---|---|---|---|
| W1 forensics | experimenter | F | forensic report + prioritized cells |
| W2 registry | designer | R | `config_space.yaml` + cost model |
| W3 harness | implementer (worktree) | H | sweep harness + G2 anchor parity report |
| W4 sweep | experimenter | S | discovery leaderboard + health tables |
| W5 confirm | experimenter | C | sealed-split confirmation report |
| W6 red team | reviewer | V | defect audit of survivors |
| W7 replicator | replicator (blind) | V | independent re-derivation verdict |

PI checkpoints (user-visible): G1 forensics, G2 anchor parity, G3 candidate-cell
selection, G4 confirmation verdict, G5 final decision. Claude writes prompts/judgments
only; all code and runs are codex/HPC.

## 7. Compute plan

Reuse cached detections/tracks (no new GPU except if E3 relaxed-gate variants need
re-selection from cached post-NMS records — CPU). Expensive layer: ≤96 configs × ~240
discovery scenes, IPV estimation CPU-bound on AMD 192-core nodes as Slurm arrays
(`zxc-rq014-*`); a 12-segment pilot measures per-config cost before the PI approves the
full submission. Cheap layer runs as a single aggregation job over cached IPV series.
All HPC artifacts under `/share/home/u25310231/ZXC/RQ014_recovery/`; durable outputs
synced to `data/derived/wod_e2e/RQ014_recovery/` and
`reports/studies/RQ014_wod_e2e_rating_recovery/`.

## 8. Governance

- New RQ (RQ014) row added to STUDIES.md (status: planning → approved on PI sign-off).
- Independent plan review is the first gate after PI approval (repo convention).
- RQ010B decision.md is NOT edited unless T1+T2 pass; then a cross-referencing amendment
  is proposed, never a silent change.
- Execution starts only on explicit PI assignment (repo rule).
