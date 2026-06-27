# RQ010B Plan v2 — WOD-E2E Tracking Build + Human-Preference Validity

Status: `proposed` (revises v1 after the second independent review returned **BLOCK/converging**) · Wave: B · Date: 2026-06-25
Supersedes (does not delete): v1 (`RQ010B_plan_v1_...20260625.md`) and v0 (PI-approved). v2 must pass an independent re-review and PI approval before it governs execution.
Closes the v1 re-review (`.../01_plan_review/codex_plan_review_v1.md`): #1,#12 already CLOSED; this revision resolves #2–#11 with firm values + the PI-chosen reference-truth strategy. Change-map in §13.

## 0. Dataset ground truth (verified 2026-06-25)

`gs://waymo_open_dataset_end_to_end_camera_v_1_0_0` (non-commercial; not requester-pays; byte-exact pilot verified). 8 surround cameras (360°,10 Hz JPEG)+intrinsics/extrinsics+ego pose/history+routing command. **No LiDAR, no actor tracks, no HD map.** Human "Rater Feedback": at a decision frame, 3 distinct 5 s ego futures scored 0–10 (≥1 >6), in `E2EDFrame.preference_trajectories` (pos_x,pos_y,preference_score); **validation-only**. Universe = **479 scenes** across **10 scenario clusters** (`val_sequence_name_to_scenario_cluster.json`, sha256 `49d9633a…`): Intersections 116, Foreign-Object-Debris 78, Cyclist 71, Pedestrian 52, Multi-Lane 42, Single-Lane 38, Special-Vehicles 25, Others 22, Cut-ins 20, Construction 15. IPV/counterpart-relevant clusters ≈ 339 scenes (Intersections, Cyclist, Pedestrian, Cut-ins, Multi/Single-Lane); FOD+Construction likely abstain (no social counterpart).

## 1. Research question

Among the 3 same-scene candidate ego futures, does **lower frozen-M3 IPV deviation** (deviation from the counterpart-conditioned human envelope) predict **higher human preference** — IPV-specifically, beating a frozen kinematic+safety baseline and the full RQ003 control battery, generalizing across scenario clusters (LOSO), with honest abstention accounting (manuscript R4)?

## 2. Phases + gate state machine (fix #1, CLOSED — retained)

`B1_INFRA` (start when signed-in access OK): build Route 4 tracker; **calibrate tracker error on Waymo Perception (§5)**; resolve t* (§3); build route/geometry fallback (§6); run rating-blind QA (§5). Ratings SEALED. → PASS ⇒ `B1_DONE`; QA FAIL or population-scale t*/geometry failure ⇒ STOP=feasibility/bounded-negative, no B2.
`G_RQ009` (external): frozen M3 decision.md; if M3 materially worse than M4 (Wave B pivot) ⇒ PAUSE→PI.
`B2_TEST` (enter only when B1_DONE ∧ G_RQ009 frozen ∧ pivot cleared ∧ §11 unlock signed): freeze baseline+controls+inclusion (still sealed) → unlock ratings ONCE → run §8 → result or bounded-negative.

## 3. Critical frame t*, counterpart & candidate inclusion (fixes #5, #7)

**t* (latest allowed observation) — field-level:** t* = the `E2EDFrame` current-frame timestamp; `preference_trajectories` waypoints are 5 s **futures anchored at t*** (waypoint times strictly > t*). Camera/ego history frames with time ≤ t* are the only admissible observations. **Mandatory B1 pre-flight** (first HPC parse, blocking): assert on a sample that (a) every val frame has exactly 3 `preference_trajectories` with `preference_score` present, (b) all candidate waypoints have t>t*, (c) ≥1 admissible camera-history frame ≤ t* exists, (d) intrinsics/extrinsics resolve at t*. If the pre-flight fails on >10% of sampled frames ⇒ the **study abstains** (feasibility result); never silently proceed.

**Counterpart selection (rating-independent, deterministic, ≤t* inputs):** within radius **R=50 m** of ego at t*, over horizon **H=5 s**, select the actor with minimum predicted time-to-conflict (TTC) against the ego routing corridor; require **≥1.0 s** of pre-t* track history (else not estimable). Conflict eligibility: predicted TTC ≤ **4 s** OR lateral route-gap ≤ **2 m** within H. Tie-break: smaller TTC → smaller lateral gap → smaller track-id. No eligible counterpart ⇒ abstain (`no_counterpart`).

**Candidate set:** the 3 dataset-provided futures (we do not generate candidates). Unit = candidate-within-scene.

## 4. Frozen inputs / contracts

RQ010 feasibility (T2_FULL_TRACKING_REQUIRED; Route 4); RQ009 frozen M3 (B2 only; context+counterpart current IPV; no temporal motifs, RQ008); RQ007 estimability; RQ005 leakage (full list §12); RQ003 controls (§10).

## 5. Reference truth, calibration & numeric QA gate (fixes #3, #4 — PI choice: Perception-LiDAR transfer)

**Independent reference via Waymo Perception (has LiDAR + 3D track GT):**
- Pull a Waymo Perception subset (`gs://waymo_open_dataset_v_*`), camera+LiDAR+3D labels. Run the **same** Route 4 camera-only tracker on the **camera subset common to both rigs** (front cameras, to control the 5-vs-8-camera rig difference — a documented caveat), compare to LiDAR-derived 3D GT.
- Output a **calibrated error model**: counterpart position/depth error as a function of range, occlusion, and class (vehicle/cyclist/pedestrian), with predicted-interval coverage. This model is the independent yardstick E2E lacks.
- **Transfer to E2E:** each E2E counterpart detection gets a position-uncertainty from the calibrated model; it propagates into M3 deviation and the abstention budget (§6).

**Numeric QA (seed `2026062306`; case-bootstrap B=2000; lower-CI must clear the bound on starred metrics):**

| Metric | Pass bound | Note |
|---|---|---|
| HOTA (counterpart, ≤t*) on Perception | ≥ 0.55 ★ | realistic camera-only floor |
| AMOTA on Perception | ≥ 0.45 ★ | detection+assoc |
| ID switches / track | ≤ 0.15 | identity stability |
| Track fragmentation | ≤ 0.25 | continuity to t* |
| Calibration coverage (90% PI) | empirical ∈ [0.86,0.94] ★ | honest uncertainty |
| E2E IPV-relevant scenes within M3 position budget (§6) | ≥ 70% ★ | else abstain-heavy ⇒ bounded-negative |

PASS = all starred bounds met with lower-CI clearing; non-starred are reported with CIs and must meet point bound. FAIL ⇒ feasibility/bounded-negative; no B2.

## 6. Route/geometry fallback + state-dependent depth budget (fix #6)

No HD map ⇒ build an ego-centric route/reference line from ego history + routing command; conflict geometry = counterpart motion vs that reference. Allowed sources: ego pose, routing command, camera-derived geometry — **no external map import**. **State-dependent budget:** with gap g = ego–counterpart distance at t*, a case enters M3 only if calibrated (§5) lateral uncertainty ≤ **min(0.5 m, 0.1·g)** and longitudinal ≤ **min(2.0 m, 0.2·g)**, AND conflict geometry resolvable; else abstain (`depth_over_budget`/`geometry_unresolved`). Fallback frozen pre-unlock (§11 audit) so it cannot become a rating-sensitive exclusion path.

## 7. Estimability & abstention accounting (fix #11)

Per candidate-scene record separable fields (RQ007): `opportunity`,`estimability`,`support`,`deviation`,`abstain`(+reason∈{no_counterpart,t*_unresolved,transform_fail,depth_over_budget,geometry_unresolved,OOD,not_estimable}). **Hard M2-downgrade guard:** missing/unestimable counterpart ⇒ abstain; NEVER fall back to M2 context-only reported as M3 (inspectable code assertion + audit field). Denominators reported at every stage (opportunity→estimable→supported→complete-scene). **Effective unit = COMPLETE scene** (all 3 candidates carry valid deviation + counterpart-bearing estimable interaction + RFS). **Min N = 60 complete scenes; LOSO requires ≥6 of 10 clusters each with ≥8 complete scenes.** Below ⇒ "underpowered" (not "negative").

## 8. Pre-registered final test (fix #8) — single primary, frozen pre-unlock

- **Primary:** within-scene **conditional logit** for P(candidate is the rater-top choice = argmax RFS) as a function of M3 deviation, scene as the conditioning stratum (McFadden conditional logit over the 3 alternatives). Effect = deviation coefficient β; hypothesis β<0 (lower deviation→preferred). **Tie rule:** if top-2 RFS within ε=1.0, the scene is a tie ⇒ excluded from the primary, included in the secondary.
- **Secondary (robustness):** within-scene Kendall τ_b between (−deviation) and RFS, aggregated, scene-clustered bootstrap.
- **Inference:** one-sided **α=0.025**; practical floor: β implying ≥1.5× odds across the interquartile deviation gap, OR τ_b ≥ 0.15. Scene/cluster-clustered bootstrap CIs.
- **One primary endpoint**; controls/sensitivity are secondary, reported in full. **Candidate-conditioned forecasting is sensitivity-only and cannot rescue a failed primary** (recommended #3).
- **PASS** = IPV-specific (§9 nested gain) + control-surviving (§10) + LOSO-generalizing + α-significant + effect ≥ floor + N ≥ min. Else bounded-negative, labeled: tracker-QA-fail / abstain-heavy / null-association / underpowered.

## 9. Frozen kinematic+safety baseline (fix #9)

Within-scene conditional logit (same form as §8 primary) on **only** causal kinematic+safety features: {min predicted TTC, min gap, headway, candidate longitudinal accel & jerk (comfort), curvature, route-adherence error} — no IPV. Logistic + L2 (λ=1.0), fixed; case/scene-isolated splits; **frozen + hashed pre-unlock**. IPV claim requires M3 deviation to add value over this baseline (nested likelihood-ratio test + ΔAUC, scene-clustered).

## 10. RQ003 control battery + LOSO (fix #10)

Implementations (each re-runs §8 primary under the perturbation): **role_flip** (swap ego/counterpart in IPV+envelope); **sign_flip** (negate IPV before envelope/deviation); **counterpart_swap** (replace true counterpart with a random support-matched actor from a *different* scene); **kinematics_only** (deviation with counterpart IPV fixed at population mean); **IPV_removed** (M3 with counterpart-IPV input ablated to neutral). **Collapse criterion:** a control "passes IPV-specificity" only if its effect falls below the §8 practical floor or loses significance; the true config must beat **all 5**. **LOSO:** leave-one-cluster-out over the 10 clusters (sha `49d9633a…`); generalization = β<0 direction in ≥80% of populated folds with pooled significance. All controls/folds reported including nulls.

## 11. Unlock checklist + pre-unlock manifest (fixes #1/#8; rec. 1,2)

Pre-unlock signed checklist: B1 QA PASS; RQ009 M3 frozen; pivot cleared; baseline+5 controls+inclusion/counterpart files + cluster file **hashed**; abstention table built with rating columns **absent**; loader audit shows `preference_*` unread in B1. Ratings-free manifest: tracker code+weights, Perception-calibration error model+plots, QA metrics, t* pre-flight log, route/geometry logs, abstention table, input hashes. Unlock logged once; later change voids pre-registration.

## 12. Full leakage denylist (fix #12, CLOSED — retained)

Forbidden runtime inputs until the final test: rating values/derived filters/tuned hyperparameters; observed PET; post-t* actor observations; realized order; post-hoc phase; full-window IPV; closest-frame selection; silent M2 substitution; temporal motifs (RQ008). Ratings-unlock + audit per §11.

## 13. Change-map (v1 re-review item → v2 resolution)

#2 route trigger→§5/§2 (Route 4 fails ⇒ Route 5 if Perception-QA HOTA<0.55 after frozen effort budget; rating-blind); #3→§5 (stricter starred bounds+CI rule); #4→§5 (Perception-LiDAR calibration transfer, PI-chosen); #5→§3 (field-level t* + blocking pre-flight); #6→§6 (state-dependent budget); #7→§3 (R/H/TTC/gap numbers); #8→§8 (single conditional-logit primary + tie rule + α=0.025); #9→§9 (frozen feature/model/λ/splits); #10→§10 (perturbations+collapse criterion+LOSO over real 10 clusters); #11→§7 (complete-scene min-N + LOSO fold minima). New-gap #6 (denominator) fixed in §7; #8/LOSO provenance fixed via the decoded cluster file; #10 forecasting rule added in §8.

## 14. Deliverables, stop, claims

Reader-facing vs process-archive separated: tracker + Perception-calibration report + QA report; preference-validity result OR bounded-negative; control table; abstention manifest; `decision.md`. Stop: QA fail or population-scale t*/geometry failure ⇒ feasibility/bounded-negative; never context-only-as-M3. Claims: open-loop preference alignment only; not closed-loop, not realised harm; non-commercial research use + required Waymo attribution.

## 15. Dependencies / hosting

Upstream: RQ010 (frozen); RQ009 (B2 only, not frozen). Hosting: full run on 同济 HPC (Slurm/GPU; access pending VPN). Adds a Waymo Perception calibration subset (§5). B1 starts when access + v2 approved.
