# Manuscript Structure Outline (v3 — self-anchor narrative)

**Purpose:** Single source of truth for every section. Aligned to *Nature Machine
Intelligence* (IMRaD; abstract ≤150 words; main text ≤4,000 words). Organised directly
around the strongest current evidence. Superseded results are removed, not hedged.

**Title:** Online runtime verification of socially compliant autonomous driving

**Running head:** Verifying AV social compliance online against human reasonable intervals

---

## One-sentence argument
Social compliance is contextual, and the right online handle on it is not the situation's
risk but the driver's own early-window preference: a self-anchored, conformally calibrated
verifier is sharp, nominally covered and transfers across sources, whereas risk-based
envelope lookup does not.

## Narrative spine
1. **The norm is contextual (R1, strong).** IPV right-of-way preference reverses with risk;
   a scalar sociality score is ill-posed.
2. **What to read (R2, counterintuitive core).** Predicting risk barely narrows the
   reasonable interval; the driver's own causal early-window IPV does. Self-anchor +
   conformal.
3. **A working verifier (R3).** Calibrated (nominal 90%), sharp (−42% width), and the only
   method that transfers across a held-out source (Waymo 0.902); beats envelope lookup in
   A/B.
4. **Actionability (R4).** Deviation → planner-facing soft cost / warning / fallback; paired
   joint check; interface demo.
5. **External test (R5/R6, planned, NSFC).** Criterion validity, consequence chain, and
   discriminant value beyond safety verification.

---

## Abstract (≤150 words)
Context (AVs certified for safety but operate in human social traffic) → reframe (social
compliance as online runtime verification) → R1 (state-dependent; priority reverses with
risk, robust; scalar ill-posed) → R2/R3 counterintuitive core (risk/PET barely narrows the
interval; self-anchored causal early-window IPV + conformal narrows 42%, nominal coverage,
Waymo held-out 0.902) → significance (sharp, calibrated, source-transferable verifier that
excludes risk and is complementary to safety). **Do not** state predictive-superiority,
deployed-warning, closed-loop or real-vehicle-deployment claims.

---

## Introduction (≤1,500 words)
- Field stake: social competence is part of AV safety in mixed traffic (Shirado et al., 2023).
- Gap: offline aggregate sociality scores vs formal safety verification (Pek et al., 2020 —
  itself NMI) answer adjacent questions, not "is this behaviour appropriate in this state?".
- Reframe: runtime verification of a data-driven empirical norm via IPV; pluggable monitor.
- Distinction from offline evaluation / formal RV / socially compliant planning.
- Contributions: (i) formalisation + verifier; (ii) state-dependence; (iii) **the verifier
  should not estimate risk — self-anchored conformal interval is sharp, calibrated and
  source-transferable**; (iv) **[PLANNED]** NSFC external validation + discriminant value.

---

## Results

### R1 — Social compliance is a state, not a score *(InterHub; strong)* → Fig 2
38,228 cases / 4 sources / 3.7M frames. Priority−non-priority IPV gap +0.058 [0.050,0.067] →
+0.001 → −0.034 [−0.045,−0.023] across PET≤1 / 1–2 / >2 s. Coarse geometry prosocial 4/4
sources. Hierarchical LODO median AE 0.142. Robust to drop-Waymo, 3/5/quantile bins, geometry
coarsening.

### R2 — The reasonable interval is set by the driver, not by risk *(core)* → Fig 3
Oracle PET narrows interval only ~3% vs global floor (width 0.833 vs 0.857 at ~90% coverage).
Self-anchored causal rolling-IPV (prefix + map-lane reference) narrows to 0.485 (−42%) at
coverage 0.901; no-roll kinematics 0.627. Split-conformal needed (raw quantiles under-cover
~0.86). Dominant uncertainty is between-driver; the driver's own preference is the signal.

### R3 — A calibrated, source-transferable verifier *(core)* → Fig 4, Fig 5
Leave-Waymo-Out (locked balanced slice): causal-roll is the only method ≥0.90 (0.902, CI
[0.894,0.910], width 0.628); FLOOR 0.868, oracle PET 0.860, no-roll 0.857 all under-cover.
Causality settled (Extended Data): observed-prefix reference corr 0.281 vs map-lane reference
corr 0.993 (MAE 0.027). Verifier A/B vs envelope lookup, same interface: TEST 0.901/0.485/FF
0.052 vs 0.900/0.833/0.051; LWO 0.823/0.488/0.114 vs 0.786/0.678/0.142. Integrated LWO still
<0.90 → soft cost / warning / monitor; hard constraints after target recalibration.

### R4 — Verifier output as a planner-facing channel *(interface demo)* → Fig 6
Deviation → soft cost / warning / fallback / monitor. Paired joint check (pair-sum, pair-diff)
as a gate (paired-only adds <0.5 pp coverage, wider). Counterfactual injection: fallback
3.0%→12.9%, median cost lift +0.252; compliant streams largely unpenalised (mean 0.134, FF
3.0%). Interface, not closed-loop.

### R5 — External validation on a real-vehicle challenge **[PLANNED, NSFC]** → Fig 7
Criterion validity (rank ↔ closeness to human interval); consequence chain (deviation ↔
safety events / scores).

### R6 — Discriminant value beyond safety **[PLANNED, NSFC]** → Fig 8
Cases that pass safety yet are flagged socially; incremental prediction of scores/events
controlling for safety/rule metrics ⇒ complementary, not a relabelling.

---

## Discussion
Reframes social behaviour as online runtime verification and identifies what to read. Norm
location is contextual; individual online verification is best self-anchored. Verifier
excludes risk/PET ⇒ cannot be a relabelled safety check. **Limitations:** ~74% deployable
(lane/route reference); strong transfer holds on the balanced lane-referenced locked slice,
not unconditionally; integrated verifier under-covers under shift ⇒ hard constraints need
target recalibration; rival explanation (IPV re-encodes kinematics) needs kinematic-only /
IPV-ablated controls. Broader impact: template for monitoring agents against human
normative ranges online.

---

## Methods
Data & extraction · IPV + **causal self-anchor** (prefix + map-lane reference; reference
choice settles leakage: corr 0.281→0.993) · **reasonable-interval estimation** (conditional
quantile regression on self-anchor + context, PET excluded; split-conformal; source-guard;
lane fallback) · paired joint reasonableness (gate) · leakage contract · evaluation
(coverage/width/Winkler, leave-Waymo-out, verifier A/B; planned kinematic-only / IPV-ablation)
· **[PLANNED]** NSFC protocol. Two algorithms: offline calibration, online step (O(1)).

---

## Main figures (6 supported + 2 planned)
| Fig | Content | Source | Boundary |
|---|---|---|---|
| 1 | Verifier architecture (self-anchor path; PET excluded; source-guard + lane fallback) | schematic | not a formal proof |
| 2 | State dependence: PET-gating + geometry prior | InterHub R1 | descriptive, population-level |
| 3 | Risk barely narrows interval; self-anchor does | fig1_reframe | held-out human data |
| 4 | Cross-source transfer: only self-anchor ≥0.90 (Waymo held-out 0.902) | fig2_cross_dataset | balanced lane-referenced slice |
| 5 | Verifier A/B vs envelope lookup | fig4_verifier_ab | integrated LWO still <0.90 |
| 6 | Planner-facing interface demo | planner demo | interface, not closed-loop |
| 7 | **[PLANNED]** NSFC: deviation vs ranking / scores / events | NSFC | external set, not global transfer |
| 8 | **[PLANNED]** NSFC: discriminant value vs safety checks | NSFC + InterHub | complementary, not replacement |

## Extended Data
Data inventory & health · full state-space + sparse fallback (R1) · causality reconstruction
(map-lane reference corr 0.993) · source-guard / calibration-mode detail · paired-IPV
coverage tables · descriptive pre-conflict-signal note · **[PLANNED]** NSFC per-algorithm /
per-scenario tables.

---

## Removed as superseded (do not reintroduce)
- PET-bin empirical-envelope **as the verifier mechanism** → replaced by self-anchor.
- "State-conditioned envelope barely beats scalar / Claim 2 downgraded" baseline ladder →
  superseded by R2/R3.
- "Runtime gap: online risk recovers ~55%" as a central limitation → verifier does not need
  risk online.
- Early-warning as a main result → at most a brief Extended Data note.
- P(IPV_i|IPV_j) conditional criterion as the primary check → repositioned as paired gate.
- 95% accuracy / 38% collision reduction / real-vehicle deployment → never (unsupported).
