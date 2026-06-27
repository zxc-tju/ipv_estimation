# RQ010B Plan v1 — WOD-E2E Tracking Build + Human-Preference Validity

Status: `proposed` (revises PI-approved v0 after Stage-1 independent review returned **BLOCK**) · Wave: B · Work group: Group 4B · Date: 2026-06-25
Supersedes (does not delete): `RQ010B_plan_v0_wod_e2e_tracking_and_preference_validity_20260624.md` (PI-approved). v1 must pass an independent re-review and PI approval before it governs execution.

> This v1 closes the 12 blocking fixes from `reports/studies/RQ010_wod_e2e_tracking_feasibility/RQ010B_1_tracking_preference_20260625T201647+0800_695fa83f/02_process/01_plan_review/codex_plan_review.md`. A fix-mapping table is in §13. Points needing PI sign-off are gathered in §14 and marked **[PI]** inline.

## 0. Dataset ground truth (verified 2026-06-25, web + bucket probe)

- Source: `gs://waymo_open_dataset_end_to_end_camera_v_1_0_0` (non-commercial research licence; not requester-pays). 8 surround cameras (360°, 10 Hz JPEG) + intrinsics/extrinsics + ego pose/history + high-level routing command. **No LiDAR. No surrounding-actor tracks. No HD map.** (→ confirms `T2_FULL_TRACKING_REQUIRED`.)
- **Human preference = "Rater Feedback":** at a decision point in each segment, expert raters score **3 distinct 5 s future ego trajectories** on **0–10** (≥1 candidate scores >6). Stored in `E2EDFrame.preference_trajectories` (pos_x, pos_y, preference_score). **Labels exist ONLY on the validation split (≈479 segments / ≈93 shards / 225 GiB).** Rater Feedback Score (RFS) uses a trust region at t∈{3,5} s with exponential decay outside.
- Consequence: the B2 universe is bounded at ≈479 scenes × 3 candidates. This caps power and forces an explicit min-effective-N rule (§7, §8).

## 1. Research question

After building a multi-camera 3D/BEV tracker to recover counterpart tracks from WOD-E2E, does an ego candidate trajectory with **lower frozen-M3 IPV deviation** (deviation from the counterpart-conditioned human envelope) receive **higher human preference** among the 3 same-scene candidates — IPV-specifically, beating a frozen kinematic+safety baseline and the full RQ003 negative-control battery, with honest abstention accounting (manuscript R4, human-alignment leg)?

## 2. Phases + gate state machine (fix #1)

```text
STATE B1_INFRA (start when: signed-in WOD-E2E access OK):
  build Route 4 tracker; resolve critical-frame index; build map/route fallback;
  define+run rating-blind tracking QA. Ratings remain SEALED.
  -> on QA PASS (§5) AND alignment PASS (§6): set B1_DONE.
  -> on QA FAIL or unresolved critical-frame/map at population scale: STOP = feasibility/bounded-negative (§7); do NOT enter B2.

GATE G_RQ009 (external): RQ009 publishes a frozen M3 decision.md.
  -> if M3 materially worse than M4 (Wave B pivot): PAUSE, escalate to PI; do NOT auto-run B2.

STATE B2_TEST (enter only when: B1_DONE AND G_RQ009 frozen AND PI clears pivot gate AND
  unlock checklist §11 signed):
  freeze baseline+controls+inclusion (still ratings-sealed) -> unlock ratings ONCE ->
  run the single pre-registered test (§8) -> report result or bounded-negative.
```

No path reaches ratings without B1_DONE + frozen M3 + signed unlock. B1 produces only ratings-free artifacts (§11 manifest).

## 3. Causal inputs, counterpart/candidate inclusion, critical-frame (fixes #5, #7)

**Critical frame (latest allowed observation):** the rating decision timestamp `t*` = the frame at which the 3 candidate 5 s futures are anchored (their common origin in `preference_trajectories`). Resolve `t*` from the proto per segment; validate camera/ego/BEV transforms at `t*`. **No observation after `t*` may enter any runtime input** (enforces the post-critical denylist). If `t*` cannot be resolved or transforms fail validation at `t*` → **abstain** (case excluded with reason code; never guessed).

**Counterpart selection (rating-independent, pre-rating, deterministic):** from tracker output at/just before `t*`, select the interacting counterpart by a frozen rule — the actor with the minimum predicted spatiotemporal conflict (lane-/route-reference gap × time-to-conflict) against the ego routing corridor within a fixed radius/horizon window; deterministic tie-break by (smaller TTC, then smaller lateral gap, then smaller track-id). Inputs use only data ≤ `t*`. If no counterpart meets the support threshold → abstain. **[PI]** radius/horizon/conflict-threshold defaults in §14.

**Candidate set:** the 3 rated ego futures, taken as given by the dataset (we do NOT generate candidates → removes a candidate-generation leakage path). Unit of analysis = candidate-within-scene.

**Allowed runtime inputs:** ego state/history ≤ `t*`, routing command, camera-derived counterpart track ≤ `t*`, constructed route/reference geometry (§6), counterpart current-IPV estimate under the RQ007 estimability contract. Everything else forbidden (§12).

## 4. Frozen inputs / contracts

RQ010 feasibility decision (T2_FULL_TRACKING_REQUIRED; Route 4 preferred); RQ009 frozen M3 (B2 only, context + counterpart current IPV, no temporal motifs per RQ008); RQ007 estimability contract; RQ005 leakage contract (full list §12); RQ003 negative-control discipline (§10). Ratings-blind until the single final test.

## 5. Tracking quality gate — numeric, rating-blind (fixes #3, #4)

**Rating-blind 3D/BEV reference protocol (fix #4):** WOD-E2E ships no actor tracks, so the reference is built independently of ratings:
- Reference set = a frozen random sample of **N_ref = [PI, default 60]** validation segments drawn under seed `2026062306`, stratified by routing command (left/straight/right). Ratings columns are not loaded during reference construction (enforced by a loader that drops `preference_*` fields; audited in §11).
- Reference labels = 3D/BEV counterpart boxes at and around `t*` produced by **(a)** a high-compute offline multi-camera bundle-adjustment/triangulation pass (no real-time constraint), cross-checked by **(b)** manual 3D-box annotation on key frames by an annotator blind to ratings. Disagreement > tolerance → adjudicated by a second blind annotator; unresolved → dropped from reference (logged).
- The tracker under test is run with only causal ≤ `t*` inputs and compared to this reference.

**Numeric pass/fail (all under seed `2026062306`; CIs by case-level bootstrap, B=[PI, default 2000]):**

| Metric | Pass bound **[PI defaults]** | Rationale |
|---|---|---|
| HOTA (counterpart, ≤`t*` window) | ≥ 0.50 | overall track quality |
| AMOTA | ≥ 0.40 | detection+association under camera-only depth |
| ID switches / track | ≤ 0.20 | identity stability through interaction |
| Track fragmentation | ≤ 0.30 | continuity to `t*` |
| Critical-frame availability | ≥ 0.80 of sampled scenes have a valid counterpart track at `t*` | else abstain-heavy |
| Depth/uncertainty calibration | 90% predicted interval empirical coverage ∈ [0.85, 0.95] | honest uncertainty (§6) |

PASS = all bounds met with lower-CI not crossing the bound on the primary three (HOTA, AMOTA, critical-frame availability). Otherwise FAIL → feasibility/bounded-negative (§7); B2 not entered.

## 6. Map/route fallback + camera-only depth budget (fix #6)

No HD map → build a **route/reference frame** from ego trajectory history + routing command (lane-free ego-centric reference line); conflict geometry = counterpart motion relative to that reference. Allowed sources: ego pose, routing command, camera-derived geometry; **no external map import**. Depth: propagate per-detection camera depth uncertainty into the track; a case enters M3 only if, at `t*`, counterpart relative-position uncertainty ≤ **[PI, default 1.5 m lateral / 3.0 m longitudinal]** AND conflict geometry is resolvable. Else **abstain** (reason code). The fallback is frozen before ratings; it cannot become a rating-sensitive exclusion path (§11 audit).

## 7. Estimability + abstention accounting (fix #11)

Per candidate-scene, record separable fields (RQ007 contract): `opportunity` (is there an interaction?), `estimability` (is counterpart current IPV estimable?), `support` (in-distribution / OOD), `deviation` (M3), `abstain` (bool + reason ∈ {no_counterpart, t*_unresolved, transform_fail, depth_over_budget, geometry_unresolved, OOD, not_estimable}). **Hard M2-downgrade guard:** if counterpart track/IPV is missing or not estimable, the case **abstains** — it may NEVER fall back to M2 context-only and be reported as M3 (guard is an inspectable assertion in code + an audit field, not a comment). Denominators reported at every stage (opportunity → estimable → supported → tested). Min effective N after abstention: **[PI, default 80 candidate-scenes]**; below it the result is labeled "underpowered", not "negative".

## 8. Pre-registered final test (fix #8) — single, frozen before unlock

- **Unit:** candidate-within-scene (3 per scene); **outcome:** released rater preference score (0–10), used as **within-scene ranking** of the 3 candidates (per-scene normalization removes cross-scene scale differences).
- **Predictor:** frozen-M3 IPV deviation of each candidate (lower = better expected).
- **Model:** scene-stratified rank association — primary = conditional (within-scene) rank model / mixed model with scene random intercept; report standardized effect + 95% CI (case/scene-clustered bootstrap). **[PI]** exact estimator (conditional logit on best-candidate vs. Kendall-τ within scene vs. mixed ordinal) frozen in §14.
- **Direction:** one-sided (lower deviation → higher preference); **α = [PI, default 0.05]**; practical-effect floor **[PI, default standardized β or τ ≥ 0.1]**.
- **Multiplicity:** one primary endpoint; negative-control + sensitivity tests are secondary and reported with the full set (no cherry-picking).
- **PASS** = IPV-specific, baseline-beating (§9), control-surviving (§10), LOSO-generalizing association at α with effect ≥ floor and effective N ≥ min. **Else bounded-negative** (and we state which: tracker-QA-fail / abstain-heavy / null-association / underpowered).

## 9. Frozen kinematic+safety baseline (fix #9)

Baseline predicts within-scene candidate preference from **only** causal kinematic+safety features (curvature/jerk/comfort, headway/TTC/min-gap to the tracked counterpart, route adherence) — no IPV. Model form, features, hyperparameters, and case-isolated split discipline are **frozen and hashed before ratings unlock**; tuned only on ratings-free signals. The IPV claim requires M3 to add information **beyond** this baseline (nested comparison).

## 10. RQ003 negative-control battery + LOSO (fix #10)

Any positive IPV increment must survive ALL of: `role_flip`, `sign_flip`, `counterpart_swap`, `kinematics_only`, `IPV_removed`; and generalize under **LOSO** (leave-one-scenario/cluster-out, using `val_sequence_name_to_scenario_cluster.json`). Each control re-runs the §8 test under the perturbation; a control that does NOT collapse the effect falsifies IPV-specificity. All control outcomes reported in full (including nulls).

## 11. Ratings-unlock checklist + artifact manifest (fixes #1/#8; recommended 1,2)

Before the one-time ratings unlock, a signed checklist must hold: B1 QA PASS; RQ009 M3 frozen; pivot gate cleared by PI; baseline + 5 controls + inclusion/counterpart files **hashed**; abstention table built with rating columns **absent**; loader-level audit shows `preference_*` never read in B1. Pre-unlock artifact manifest (ratings-free): tracker code+weights, QA labels/metrics, calibration plots, critical-frame/map resolver logs, abstention table, input hashes. Unlock is logged once; any later change voids the pre-registration.

## 12. Full leakage denylist (fix #12)

Forbidden as runtime inputs until the final test: rating values / rating-derived filters / rating-tuned hyperparameters; observed PET; post-`t*` (post-critical) actor observations; **realized order; post-hoc phase; full-window IPV; closest-frame selection**; silent M2 context-only substitution for missing tracks; temporal motif features (RQ008). Ratings-unlock + audit procedure per §11.

## 13. Fix-mapping (review item → section)

1→§2; 2→§2/§5(route trigger); 3→§5; 4→§5; 5→§3; 6→§6; 7→§3; 8→§8; 9→§9; 10→§10; 11→§7; 12→§12. Recommended 1,2→§11; 3→§8(forecasting sensitivity-only, cannot rescue a failed primary); 4→§7(min-N/underpowered); 5→§15 deliverables; 6→§0/§15 licence.

## 14. PI decisions required (defaults proposed; change any)

- Numeric QA bounds (§5 table) and N_ref/bootstrap B.
- Counterpart-selection radius/horizon/conflict threshold (§3).
- Depth uncertainty budget (§6).
- Final estimator family + α + practical-effect floor + min effective N (§7, §8).

## 15. Deliverables, stop conditions, claims

Deliverables (reader-facing vs process-archive separated): tracker + QA report; preference-validity result OR bounded-negative; negative-control table; abstention manifest; `decision.md`. Stop: tracker QA fail or unresolved critical-frame/map at scale → freeze as feasibility/bounded-negative; never downgrade to context-only and call it M3. Claims: open-loop preference alignment only; not closed-loop, not realised harm; non-commercial research use (Waymo licence + required attribution string).

## 16. Dependencies

Upstream: RQ010 (frozen); RQ009 (B2 only, not yet frozen). Hosting: full run on 同济 HPC (Slurm/GPU; access pending VPN). B1 is the long pole — start when access + this v1 are approved.
