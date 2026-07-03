# RQ010B Reframed Preference-Validity — PI Pre-Registration Contract (2026-06-30)

Status: GOVERNING. Frozen by PI 2026-06-30. The earlier full-479 "null" is RETRACTED
(it used a constant-velocity straight-line extrapolated counterpart future; IPV against a
non-reactive counterpart is ill-determined — see memory `rq010b-reframed-ipv-deviation-contract`).

## IPV computation contract (key)
- **Time window:** primary = candidate FUTURE window t* → t*+5s (full 21-pt horizon). Past window BANNED
  (the 3 candidates share the past → no discriminative power).
- **Pairing:** each candidate IPV = ego_candidate_future × a SHARED FROZEN counterpart future
  (freeze one forecast from the pre-t* counterpart history, OR use the observed post-t* counterpart track;
  same for all 3 candidates, only the ego varies) → open-loop opportunity structure.
  MUST declare: this is an open-loop approximation, NOT closed-loop causal, NOT real harm.
- **Candidates lack vel/accel:** finite-difference the 21 positions at 4Hz to recover velocity
  (accel is noisy → smooth, or use the velocity term only).
- **Estimability gate (RQ007):** if the candidate future window has no active interaction
  (no closing / no conflict) → ABSTAIN (do not force). Usable sample = estimable interaction segments.
- **Deviation:** use the FROZEN M3 (context-conditioned envelope) to compute the candidate IPV's
  signed/abs deviation from the human norm for that state.

## Analysis design (within-segment ranking)
- Unit = candidate; group = segment (3 candidates/segment).
- **Primary endpoint:** within-segment "smaller deviation ↔ higher score" via conditional logit /
  Bradley–Terry / within-segment rank correlation; wild cluster bootstrap by segment.
  Plus a hit-rate: "is the top-rated candidate the minimum-deviation one?".
- **IPV-specificity controls (MANDATORY — decide the conclusion):** kinematics_only
  (progress / smoothness / proximity etc.), IPV_removed, shuffled_ipv. IPV must explain preference
  ABOVE these baselines; with max-statistic permutation.
- **Window robustness:** main = full 5s + estimability gate; sensitivity = early window 0–2s,
  estimable sub-window only.
- **Pre-registered PASS/FAIL:** positive requires (a) non-trivial within-segment effect,
  (b) beats ALL controls (IPV-specific), (c) permutation significant; else report bounded/null honestly.

## Leakage denylist
- Do NOT read preference_score VALUES before the final pre-registered test (ratings-blind).
- Counterpart selection & IPV estimation must NOT use rating.
- No rating-tuned predictor. Observed PET / post-hoc labels must not enter the online path.
- Do NOT pass M2 off as M3 (context-conditioned envelope is intended, but label it honestly).

## Delegation contract / return format
Each codex task includes: ROLE / WORKER_ID / OBJECTIVE / INPUTS / READ_SCOPE / WRITE_SCOPE /
DENYLIST / TASKS / DELIVERABLES / ACCEPTANCE_CRITERIA / NON_GOALS / STOP_CONDITIONS / RETURN_FORMAT.
Fixed return: STATUS(PASS|FAIL|BLOCKED|PARTIAL) / WORKER_ID / ROLE / SCOPE_COMPLETED / FILES_* /
KEY_EVIDENCE / ACCEPTANCE_CRITERIA_RESULTS / UNRESOLVED_BLOCKERS / RECOMMENDED_NEXT / GIT_DIFF_SUMMARY.
Implementer may NOT self-review; independent reviewer / red-team / replication MUST be new codex sessions.

## Phases
0 identity + data location/provenance (verify A/B/C + counterpart trajectories, sampling rate,
  t* aligned with the scored frame, M3 parity) → 1 IPV build (finite-diff velocity, principal
  counterpart, frozen counterpart, estimability gate) → 2 candidate deviation (frozen M3) →
  3 within-segment ranking main analysis + clustered uncertainty → 4 IPV-specificity controls +
  permutation → 5 window sensitivity → 6 independent review → 7 red team (open-loop assumption,
  finite-diff noise, rating leakage, estimability misuse, within-segment leakage) → 8 independent
  replication → 9 nature figures + EN/zh HTML → 10 final review + registration (update RQ010B result
  section; if M3 unavailable or estimable sample too small → bounded / underpowered null).

## Stop conditions
RUN_ROOT identity mismatch; A/B/C or counterpart trajectories not locatable/verifiable; t* or scored
frame cannot be aligned; M3 unavailable; rating read before final test; estimable sample too small to
support within-segment inference (→ report bounded/underpowered null); nature skill unavailable; any
HTML version won't open; a worker tries to modify PAPER_REPO.

## Completion
Done only when provenance, IPV build, within-segment main, IPV-specificity controls, window sensitivity,
independent review, red team, replication, and EN/zh HTML all PASS (a null also counts as done if honestly
presented). Final Chinese report lists: RUN_ID / preference-validity result (within-segment effect + CI) /
IPV_SPECIFIC (yes/no) / usable segment N / window sensitivity / RED_TEAM_STATUS / REPLICATION_STATUS /
NATURE_SKILL_STATUS / HTML_ENTRY_EN / HTML_ENTRY_ZH / OVERALL_STATUS / UNRESOLVED_BLOCKERS.

## Recorded caveat (PI)
IPV computed at 4Hz may differ from the original 10Hz operationalization. For final validation metrics,
consider interpolating trajectories to ~10Hz and recomputing.

## ADDENDUM 2026-07-03 — FINAL FROZEN DESIGN (frozen BEFORE any rating join)

M3 transfer FAILED (Phases 2b–2f): the RQ009 InterHub 32-feature M3 does NOT transfer to WOD-E2E.
Categorical support alignable via HV-HV encoding (684/684), but the numeric kinematic distribution is OOD;
support only 35/228 even after fixing construction (ipv_removed baseline, W_x recompute, elapsed neutralize);
`m3_center` near-constant (sd ~0.06). Human IPV norm ≈ NEUTRAL (median 0) across path types
(Kruskal–Wallis ε²=0.004, tiny). => M3-deviation endpoint is not usable.

HUMAN-NORM REFERENCE (replacement, fully applicable, no OOD): the PATH-TYPE-conditioned pure HV-HV IPV
distribution (Phase 2f). Per path type {CP,HO,MP,F} the human center ≈ 0, spread varies by geometry.
deviation_abs = |candidate IPV|; deviation_norm = |IPV| / (path-type HV central-90 half-width).

TWO SCHEMES (both executed):
- Scheme 1 (future-only): IPV over [t*, t*+w] of real candidate future × real counterpart future; N=75 (≥2s future).
- Scheme 2 (history+future combined, INTERSECTION window, ≥1s future): IPV over ([cp_first_obs,cp_last_obs] ∩ ego)
  of (real ego past ++ candidate future) × (real counterpart past ++ real future); N=98. Discrimination requires
  post-t* counterpart overlap (3-candidate spread grows with future length; ≥2s subset N=66 = strong-discrimination robustness).
  Big-N (349) recovery is illusory: with no counterpart future the 3 candidate IPVs collapse to identical (zero spread).

ENDPOINT (per scheme): candidate quantity = IPV + deviation from the path-type HV norm.
Within-segment MAIN test: H1 "smaller deviation_abs ↔ higher rating" via within-segment rank corr
(Spearman/Kendall) + Bradley–Terry; wild cluster bootstrap by segment; + top-rated=min-deviation hit-rate;
+ raw-IPV signed direction + inverted-U shape.
IPV-SPECIFICITY controls (mandatory): kinematics_only, shuffled_ipv, max-statistic permutation.
PASS requires (a) non-trivial within-segment effect, (b) beats ALL controls, (c) permutation significant; else bounded/null.
Ratings are joined ONLY at this test. Report BOTH schemes (triangulation). N ceiling for the per-candidate
preference test is ~75–98 (counterpart post-t* FoV limited); the only real N lever is 360° cameras (not done).
