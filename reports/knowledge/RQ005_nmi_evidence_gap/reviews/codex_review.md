# Codex Review: RQ005 NMI Evidence Gap

Status: review-complete, not yet frozen in `decision.md`.

Review date: 2026-06-21.

## Scope

Reviewed study package:

- `reports/studies/RQ005_nmi_evidence_gap/RQ005_1_online_verification_gap_review_20260618/`

Primary evidence read:

- `00_entry/index.html`
- `02_process/agent_H/round6_memo.md`
- `02_process/agent_H/evidence_map.csv`
- `02_process/coordination/round5_synthesis.md`
- selected verifier outputs under `02_process/agent_D_verifier/`

## Overall Verdict

RQ005 is best used as a governance and evidence-boundary review for the online
social-compliance verification manuscript. It supports a framework paper only if
the claims are carefully downgraded. It does not support a deployed verifier,
formal guarantee, strong predictive superiority, global transfer or closed-loop
planner-performance claim.

Paper-safe phrasing:

> The current evidence supports a state-conditioned empirical runtime-monitoring
> framework and a leakage-aware verifier interface. Deployment-grade online
> warning, nominal calibration under source shift, external validation and
> planner-performance benefits remain unproven.

## Claims That Can Be Carried Forward

1. **Runtime verification framing is defensible as an empirical monitor.**
   The four-layer architecture of state recognizer, normative envelope,
   deviation score and planner-facing output is a useful manuscript spine.
   It must be described as empirical/probabilistic monitoring, not formal proof.

2. **Social-compliance norms are state-dependent.**
   The report supports risk x geometry x role x time envelopes as the right
   object of analysis. This aligns with RQ004's state-space result.

3. **The feature/leakage contract is a major contribution.**
   Observed PET, actual order, post-hoc phase, full-window IPV, closest frame
   and minimum distance must be treated as offline/replay labels, not deployed
   online inputs. Runtime candidates need causal provenance.

4. **Planner-facing output can be shown as an interface demo.**
   Soft cost, warning, fallback, hard-constraint candidate and monitor-record
   channels are reasonable design outputs. The current demo is small (`n=6`)
   and should not be converted into a planner-performance result.

## Claims That Must Be Downgraded

1. **State-conditioned verifier superiority.**
   The report's verifier comparison does not show strong predictive advantage:
   scalar coverage is `0.855`, full-state observed-PET oracle coverage is
   `0.850`, and full-state split-conformal coverage is `0.823`, below the
   nominal `0.90` target. Use "norm-reference/interface advantage", not
   "better predictor".

2. **Online early warning.**
   Strict-context replay reports a signal but with limited operational quality
   (`precision=0.335`, `recall=0.442`, `FPR=0.129`, `FNR=0.558`,
   `F1=0.381`). This is replay/pre-conflict evidence only. It lacks causal
   alarm timestamps and a causal phase/conflict detector.

3. **Cross-dataset transfer.**
   Evidence supports only local/coarse/MP-heavy stability. It does not support
   global cross-dataset verifier generalization.

4. **Calibration.**
   Conformal envelopes are a candidate path but do not yet restore nominal
   coverage under source shift.

## Claims To Reject For Current Manuscript Use

- The verifier is deployment-ready.
- The verifier provides a formal safety or social-compliance guarantee.
- The verifier improves planner safety, comfort, progress or collision risk.
- A hard planner constraint is validated.
- AV/HV sociality has a stable global direction.
- Human drivers systematically "bully" AVs.
- Priority is a static social norm label.
- One average IPV value represents social compliance.

## Required Evidence Before Upgrading Claims

The report's own P0/P1 list is correct:

- replace observed PET with predicted TTC/APET-like causal risk proxies;
- build a causal phase/conflict detector and explicit alarm/event timestamps;
- add source-adaptive calibration or mixed-effects state-space calibration;
- validate externally beyond InterHub replay;
- run closed-loop or counterfactual planner evaluations before any planner
  benefit claim.

## Knowledge-Layer Action

Update `synthesis.md` as a claim-demotion record: framework and leakage contract
can move forward; deployed warning, predictor superiority, transfer and planner
benefit claims must remain blocked. Do not mark RQ005 accepted until
`decision.md` explicitly freezes this downgraded claim slate.
