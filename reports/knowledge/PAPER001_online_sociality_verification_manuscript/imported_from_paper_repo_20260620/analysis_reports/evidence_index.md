# Evidence & Decisions Index

**Purpose.** One compact pointer to (1) the canonical evidence files and what each
owns, and (2) the still-valid conclusions and red-lines distilled from superseded
long-form notes removed on 2026-06-20. Git history retains every original. If anything
here conflicts with `../manuscript_structure.md` (v3), **v3 wins**.

---

## Canonical sources (authoritative — keep)

- `../manuscript_structure.md` — v3 self-anchor narrative; single source of truth for
  sections, claims, and the "removed as superseded" list.
- `methods_revision_memo_online_verifier.md` — guard/floor/abstain Methods revision
  (situational floor + moderate-Δ abstention; guarded Algorithm 1/2 patches; the
  "why deliberately standard tools" defense). **Not yet fully landed in `main.tex`.**
- `nsfc_ipv_validation_plan_v2.md` — frozen confirmatory spec for the NSFC real-vehicle
  external validation; also flags two still-open `main.tex` issues (deviation-sign /
  one-sided soft cost; rolling-vs-full-window).
- `../drafts/` (`main_remote_github_before_sync_2026-06-19_113002.tex` v2 draft +
  `structure_remote_v2_before_sync_2026-06-19.md` v2 outline +
  `main_tex_version_note.md`) — retained for a deliberate merge pass (see *Pending merge*).

---

## Distilled conclusions from removed notes

### Hard red-lines / forbidden claims (never state — unsupported)
- No ">95% verification accuracy", no "38% collision reduction", no "12% efficiency gain".
- No closed-loop planner improvement; no real-vehicle deployment. The planner link is an
  **interface demo only**.
- "Cross-source robust / Waymo held-out 0.902" holds only on the **balanced lane-referenced
  locked slice** (5k cases, sigma=0.1), not unconditionally.
- ~74% deployable (needs lane/route reference); integrated leave-Waymo-out still <0.90 →
  soft-cost / warning / monitor, with hard constraints only after target-domain recalibration.
- *(sources: NMI_论文撰写思路, narrative_adjustment_v2; also in v3 "Removed as superseded")*

### Core pivot — why self-anchor, not risk (the one idea to preserve)
Two distinct estimation targets:
(a) **where the population norm sits** (its median) moves with risk — R1, +0.058→−0.034, a
population-level fact; (b) **an individual's online reasonable band** is dominated by
between-driver variance and is best predicted by the driver's own causal early-window IPV,
not by risk. Risk shifts the center (~0.09) but does not narrow the spread (~0.5–0.86); the
verifier needs the spread → self-anchor, **PET excluded**. Bonus: excluding PET/risk makes
the verifier orthogonal to collision risk, so it cannot be dismissed as a relabelled safety
check. *(source: narrative_adjustment_v2_self_anchor)*

### Norm-laundering validation (E1–E5) — verdict and what it forced
- **Risk tested:** conditioning the band on the agent's own early window could degrade into
  "self-consistency" — a uniformly aggressive, internally self-consistent agent always passes.
- **Verdict: MUST REVISE → hybrid** = self-anchor (sharpness) + situational floor +
  out-of-support abstention (anti-laundering). Pure self-anchor is insufficient but must
  **not** be replaced — it carries real signal.
- **Numbers now relied on by `methods_revision_memo`:** E5 high-risk (PET≤1) self-anchor
  flag-lift 0.850 < situational 1.129, and the "self-anchor passes but situation flags"
  subset is enriched **1.507×** for bad outcomes; E3 residual laundering window Δ≈0.4–0.6;
  E4 situational-only R² on early IPV 0.044, disposition-residual incremental R² 0.45,
  M2/M1 width ratio 1.34.
- Pre-registered PASS/FAIL design (E1 non-triviality + no leakage; E2 cross-individual
  coverage ≥0.88; E3 consistent-deviator stress, *decisive*; E4 situation-vs-disposition
  decomposition; E5 external-outcome adjudication) survives in git if a re-run or citation
  is needed. *(sources: validation_plan_self_anchor_group_norm + codex_prompt_self_anchor_validation)*

### Locked verifier numbers (balanced lane_ids slice)
| method | TEST cov/width | Leave-Waymo-Out cov/width |
|---|---|---|
| FLOOR (global) | 0.889 / 0.898 | 0.868 / 0.871 |
| oracle PET (ceiling) | 0.889 / 0.867 | 0.860 / 0.840 (under-covers) |
| no-roll causal kinematics | 0.896 / 0.738 | 0.857 / 0.743 |
| **causal-roll self-anchor (recommended)** | **0.899 / 0.591** | **0.902 / 0.628 (only ≥0.90)** |

Verifier A/B (same deviation interface), LWO: online self-anchor 0.823/0.488/false-flag 0.114
beats PET-bin 0.786/0.678/0.142. Leakage settled: map-lane-centerline reference corr **0.993**
(MAE 0.027) vs observed-prefix 0.281. *(source: narrative_adjustment_v2; also in v3 R2/R3)*

### Demoted-but-keep-for-Extended-Data numbers (v2 evidence line)
Central in v2, demoted in v3, still citable as ED/sensitivity. Full prose lives in the
retained v2 draft.
- **Baseline ladder (LODO, demoted Claim 2):** full-state vs scalar width −2.1% (0.684 vs
  0.699), pinball −3.5% (0.0598 vs 0.0619), deviation AUC 0.814 vs 0.797 (fold SD≈0.038);
  coverage NOT improved (0.759 vs 0.768); shuffled-state control ≈ scalar (0.507).
- **Early warning (proof of concept):** 5% FPR ROC-AUC 0.648; recall 12.8% / 7.4% / 1.1%
  at lead ≥0/1/2 s; median horizon 1.1 s.
- **Online calibration drift:** Waymo held-out coverage 0.706; Lyft online FPR 19.8%.
- **Envelope state space:** 432 cells / 79 reliable (n≥200) / 16 main-text; hierarchical
  partial-pooling LODO median AE 0.142. *(sources: structure_remote_v2 + v2 main.tex — both retained)*

### Reviewer-defense framing (strategic)
- No single gold standard for social compliance → **construct-validity triangulation**:
  predictive + criterion (NSFC ranking) + convergent (optional small human-rating study) +
  discriminant (beyond safety).
- The three reviewer "knives" the NSFC data answers: (1) no ground truth; (2) no real subject
  (InterHub is Waymo-dominated replay); (3) no deviation→consequence loop.
- Division of labour: **InterHub establishes the norm; NSFC tests it against an independent
  world.** *(source: NMI_论文撰写思路 — superseded on method, but this framing is still used by
  nsfc_ipv_validation_plan_v2)*

### Known `main.tex` issues still open (do not lose)
- **Deviation-sign / one-sided soft cost:** with θ>0=prosocial, δ=(θ−m)/w makes the positive
  side *over-yielding*, but current `main.tex` fires a one-sided "more competitive" cost on the
  positive side → sign inverted. Fix with two-tailed D_comp / D_yield.
  *(nsfc_ipv_validation_plan_v2 §11.2; visible in the retained v2 main.tex deviation block)*
- **E1 label-overlap:** ensure the scoring target is a non-overlapping post-early window
  (t > anchor end) before any strict no-leakage claim. *(methods_revision_memo §3)*

---

## Pending merge (do not delete the v2 draft yet)
`drafts/main_remote_github_before_sync_2026-06-19_113002.tex` (v2, 563 lines) is **not** a
subset of the active v3 `main.tex`. Mine before any deletion: full written Intro/Discussion
prose; the complete conditional-criterion notation and equations (band B, signed deviation δ,
EWMA, Mondrian split-conformal κ_c); two full algorithm blocks (old formulation); and the
longer state-space / envelope Methods prose. Do a deliberate merge pass, then this index plus
git history make it safe to drop.

---

## Removed on 2026-06-20 (recoverable via git)
`knowledge/archive/` subtree — `legacy_paper_notes/` (NMI_论文撰写思路, narrative_adjustment_v2,
validation_plan_self_anchor_group_norm, codex_prompt_self_anchor_validation),
`agent_rules/cursor-paper-drafting.mdc`, `main_repo_removed/nsfc_open_exploration_plan.md`,
`overleaf_fragments/` (Authorea/Nature-Communications template scraps + Feynman example bib),
and stray `.DS_Store` files. Unique conclusions are distilled above.

`../../methods_online_verification.tex` was also removed on 2026-06-20 because it had become
a superseded pointer stub. Use this index and `methods_revision_memo_online_verifier.md`
instead.
