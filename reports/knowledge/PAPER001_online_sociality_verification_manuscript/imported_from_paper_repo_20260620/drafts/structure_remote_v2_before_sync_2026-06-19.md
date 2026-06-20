# Manuscript Structure Outline (v2 — evidence-scoped)

**Purpose:** Single source of truth for every section draft. Aligned to *Nature Machine
Intelligence* (IMRaD; abstract ≤150 words; main text ≤4,000 words). Revised so that every
headline claim is matched to the evidence actually frozen in the analysis reports. Claims
that the data do not yet support are demoted or marked **[PLANNED]**.

**Title:** Online runtime verification of socially compliant autonomous driving
*(alt: "Social compliance is a state, not a score: runtime verification of human-aligned autonomy")*

**Running head:** Verifying AV social compliance online against human normative envelopes

**Author list & affiliations:** *(to be completed)*

---

## Narrative spine (three acts)

1. **The norm exists and is estimable (Discovery, strong).** Social preference (IPV) is
   quantifiable, online-estimable, and is *not a single average* but a state-conditioned
   empirical envelope over `risk × geometry × role × time`.
2. **The verifier (Method).** A four-layer online verifier whose monitored social signal —
   a dynamic IPV estimated causally from past trajectory — is online by construction; the
   residual runtime gap is confined to risk-state indexing and is reported, not hidden.
3. **External test & closing the loop (Validation) — [PLANNED, NSFC].** Run the verifier on
   award-winning algorithms of an independent real-vehicle challenge; use official rankings,
   per-scenario scores and safety events as outcome ground truth to establish criterion
   validity and discriminant value beyond safety verification.

> Reframing in one line: from "inject synthetic violations and report an AUC" to "real
> algorithms judged against real, independent outcomes." This manuscript delivers acts 1–2
> in full and scopes act 3 as planned validation.

---

## Abstract (≤150 words)
Background (AVs are certified for safety but operate in human social traffic) → Problem
(no online, state-conditioned compliance test) → Method (state-conditioned empirical
normative envelopes + online verifier with causal dynamic IPV) → Results (strong state
dependence: priority gap reverses with risk, +0.058→−0.034, robust; social signal is
causal/online; runtime gap confined to risk-state recovery) → Significance (an auditable
monitoring layer complementary to collision-safety verification). **Do not** state 95%
accuracy / 38% collision reduction / real-vehicle deployment — unsupported by current data.

---

## Introduction (≤1,500 words)
- **Background & motivation.** Social intelligence as a missing layer of AV safety in mixed
  traffic; machine agents can suppress human social norms (Shirado et al., 2023).
- **Gap.** Offline evaluation reports a single aggregate sociality score; formal safety
  verification (Pek et al., 2020 — itself NMI; RSS; reachability; CBFs) answers "will it
  collide?", not "is this behaviour appropriate in this state?". The needed question is
  local, conditional and graded.
- **Reframe.** Runtime verification of a *data-driven, state-conditioned empirical norm*
  (not a hand-written specification); an online monitor that feeds the planner.
- **Distinction from three paradigms.** vs offline AV evaluation (online, per-timestep,
  state-conditioned); vs formal runtime verification (empirical distribution, not logic);
  vs socially compliant planning (a pluggable monitor, not a planner objective).
- **Contributions.** (i) formalisation + four-layer verifier; (ii) strong state dependence;
  (iii) causal online social signal + honest runtime gap; (iv) **[PLANNED]** external
  validation on a real-vehicle challenge + discriminant value beyond safety.

---

## Results

### R1 — Social compliance is a state, not a score *(InterHub; **strong**)* → Fig 2, 3
- Data: 38,228 cases, 4 sources (Waymo 23,218 / nuPlan 7,499 / Lyft 5,105 / AV2 2,406),
  3,695,981 frames; IPV θ∈[−π/2,π/2], θ>0 prosocial.
- Risk-gated right-of-way: priority−non-priority IPV gap +0.058 [0.050,0.067] → +0.001
  [−0.004,0.006] → −0.034 [−0.045,−0.023] across PET≤1 / 1–2 / >2 s.
- Envelope gradient (16 main-text cells of 432; 79 reliable at n≥200): median 0.137 /
  prosocial share 0.738 (high-risk priority-conflict) → median −0.003 / share 0.476
  (low-risk priority-pre). Coarse geometry prosocial in 4/4 sources. Hierarchical
  partial-pooling LODO median AE 0.142 (< 0.145 dataset-specific, < 0.150 pooled).
- Robust under drop-Waymo, 3/5/quantile PET bins, geometry coarsening, n≥100/200/300.

### R2 — An online verifier with a causal social signal *(InterHub; method)* → Fig 1, 4
- Four layers: state recogniser → empirical normative envelope → deviation scorer →
  planner interface; offline calibration vs online monitoring explicitly separated.
- **Causal/online by construction:** dynamic IPV from a trailing window (ego + counterpart),
  no future info ⇒ strict-online; full-window IPV is an offline label only.
- **Runtime gap, re-scoped:** only risk-state indexing is incomplete online — hybrid risk
  proxy agrees with offline PET bin 52.1% (balanced)/57.8% (weighted), high-risk recall
  0.67 / precision 0.48; rolling phase alignment 72.2%, median 0.6 s early, IoU 0.30. Gap
  framed as concept-level rationale for runtime verification.

### R3 — State-conditioned envelopes are interpretable references *(InterHub; **demoted**)* → Fig 5
- LODO: full-state vs scalar — width −2.1% (0.684 vs 0.699), pinball −3.5% (0.0598 vs
  0.0619), deviation AUC 0.814 vs 0.797 (fold SD≈0.038); coverage **not** improved (0.759 vs
  0.768); shuffled-state control ≈ scalar (0.507).
- Wording: interpretable monitoring reference, **not** predictive superiority. (Demote
  Claim 2.)

### R4 — Social-deviation signal can precede the conflict window *(InterHub; **proof of concept**)* → Fig 6
- Descriptive: non-zero IPV before onset in 71.4% of cases.
- Controlled 5% FPR detector: ROC-AUC 0.648; recall 12.8% (≥0 s) / 7.4% (≥1 s) / 1.1%
  (≥2 s); median horizon 1.1 s [0.2,1.9]; full-state not better than scalar.
- Wording: proof of concept; descriptive precedence ≠ high-recall online alarm.

### R5 — Verifier output as a planner-facing channel *(InterHub; **interface demo**)* → Fig 7
- Deviation → soft cost / warning / fallback / monitor. Counterfactual injection: fallback
  3.0%→12.9%, median pre-onset cost lift +0.252; compliant streams largely unpenalised
  (mean 0.134, false-trigger 3.0%).
- Cross-dataset: offline deviation AUC transfers moderately (0.830 vs 0.817); online
  calibration source-dependent (Waymo coverage 0.706; Lyft online FPR 19.8%).
- Wording: interface demonstration, not closed-loop performance or safety guarantee.

### R6 — External validation on an independent real-vehicle challenge **[PLANNED, NSFC]** → Fig 8
- Criterion validity: higher-ranked algorithms sit closer to the human envelope?
- Consequence chain: higher deviation ↔ more safety events / lower per-scenario scores?

### R7 — Discriminant value beyond collision-safety verification **[PLANNED, NSFC]** → Fig 9
- Cases that pass safety verification yet are flagged socially; incremental prediction of
  scores/events controlling for conventional safety/rule metrics ⇒ complementary, not a
  relabelling.

---

## Discussion
- Reframes AV social behaviour from offline metric to online runtime-verification property.
- **Bounded claims (Limitations):** modest, not better-calibrated predictive gain →
  interpretable reference; early warning a proof of concept (no causal event timestamps;
  synthetic labels); local generalisation (Waymo 61% of cases / 85% of AV cases; source
  drift; only coarse high-support states transfer); construct validity, not a single gold
  label; full closed-loop is future work. Planned NSFC validation targets these directly.
- **Broader impact:** template for monitoring autonomous agents against human normative
  envelopes online under uncertainty and distribution shift.

---

## Methods
Data & interaction extraction · IPV estimation and **causal online inference** (trailing
window, ego + counterpart; *to specify: window length, use of predicted intentions,
rolling-vs-full-window convergence/bias*) · State space & normative envelopes (432 cells /
79 reliable / 16 main-text; hierarchical partial-pooling; **dynamic envelopes matched to the
online signal**) · Deviation scoring + P(IPV_i|IPV_j) conditional criterion · **Leakage
contract** (dynamic IPV = strict-online; full-window IPV / observed PET / phase = offline) ·
Evaluation protocols (LODO; controlled-FPR early warning; counterfactual interface) ·
**[PLANNED]** NSFC external-validation protocol (scenario alignment, deviation vs ranking /
scores / events, incremental regression vs safety metrics, human reference comparison).

---

## Main figures (7 supported + 2 planned)
| Fig | Content | Source | Boundary |
|---|---|---|---|
| 1 | Verifier framework (offline/online layers + runtime-gap annotation) | round5/fig1_architecture | not a formal proof |
| 2 | State-space support + sample sizes | round2/fig2_state_space_map | observed PET offline |
| 3 | State dependence: PET-gating + envelopes + geometry prior | round1&2 | descriptive |
| 4 | Causal online deviation + risk-state feasibility (55%, IoU 0.30) | round2&4 | gap = risk indexing only |
| 5 | Baseline ladder (demoted Claim 2) | round3/fig6_baseline_ladder | no strong predictive edge |
| 6 | Early-warning performance (proof of concept) | round4/fig5_early_warning | replay, not online alarm |
| 7 | Planner-facing interface demo | round5/fig7_planner_demo | interface, not closed-loop |
| 8 | **[PLANNED]** NSFC: deviation vs ranking / scores / events | NSFC | external set, not global transfer |
| 9 | **[PLANNED]** NSFC: discriminant value vs safety checks | NSFC + InterHub | complementary, not replacement |

---

## Extended Data
Data inventory & health audit · full state-space support matrix + sparse fallback ·
baseline ladder + negative controls (shuffled-state ≈ scalar) · early-warning threshold /
lead-time sensitivity · cross-source LODO + online FPR drift · online-proxy feasibility ·
planner signal spec + over-conservatism control · **[PLANNED]** NSFC per-algorithm /
per-scenario tables · rejected claims & future-work evidence requirements.

---

## Data / Code Availability · Acknowledgements · Author Contributions · Competing Interests
*(standard; competing interests: none)*

---

## What changed from v1 (over-ambitious outline)
| v1 claim | Problem | v2 |
|---|---|---|
| ">95% verification accuracy" | no gold label | construct validity + [PLANNED] NSFC external agreement |
| "38% collision-risk / 12% efficiency" | from non-existent closed loop | removed; only as [PLANNED] NSFC counterfactual, strictly bounded |
| "Real-vehicle deployment (commercial AV)" | replay / external data, not deployment | "independent real-vehicle challenge external validation" |
| Trajectory reconstruction as headline result | synthetic demo only | demoted to interface demo + future work |
| Results end on reconstruction/real-vehicle | weakest evidence in strongest slot | end on **[PLANNED]** NSFC external validation + discriminant value |
