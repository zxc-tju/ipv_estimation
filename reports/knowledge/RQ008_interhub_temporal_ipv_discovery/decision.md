# RQ008 InterHub Temporal IPV Discovery Decision

RUN_ID: `RQ008_1_temporal_ipv_discovery_20260622T234914+0800_3e3e776a`
Plan SHA-256: `3a94986587c3c30afa18b7a7c2151e272ac1382b24c43f8cbbf9a06b015f77db`
Decision date: 2026-06-23
Status: RQ008A discovery decided; confirmation pending

## Decision Summary

RQ008A is a negative exploratory discovery result. It accepts one knowledge-layer boundary claim: on InterHub sigma=0.1 discovery cases, no fixed candidate temporal IPV structure survived the mechanical, compositional, uncertainty, estimability, and source-balance controls. It rejects all positive temporal IPV motif, feedback, reciprocity, lead-lag, risk-coupling, and IPV-mean temporal-law claims from this data. It keeps the frozen primary null and two falsification targets pending for Wave B only.

Manuscript use is limited to an exploratory negative/boundary/limitation claim scoped to directional temporal structure. It is not a confirmed temporal law and not evidence for a positive temporal IPV process.

## Accepted Claims

### RQ008-KC-NEGATIVE-DISCOVERY-BOUNDARY

- Type: accept
- Evidence ID: `RQ008-EV-NEGATIVE-0OF24`
- Scope: InterHub sigma=0.1 RQ008A discovery split; 24 fixed discovery candidates consisting of 18 motifs, 2 alignments, and 4 role/uncertainty effects (3 role/phase candidates plus 1 uncertainty/estimability candidate); directional temporal IPV structure only.
- Claim: On InterHub sigma=0.1 discovery cases, no candidate temporal IPV structure survives the RQ008A mechanical and compositional negative-control gate. The Phase-7 control matrix reports 0/24 survivors, so apparent temporal "dynamics" are explained by time-reversal invariance, source composition, duration, and estimation uncertainty rather than by a supported directional temporal IPV law.
- Figure / table pointer: Fig. 2 control survival matrix; `02_process/08_controls/survival_summary.csv`.
- Strength: Discovery-side robust. The claim survived independent ratification, final review, red-team status `PASS_WITH_FIXES`, and partial independent replication. The replication reproduced the discovery/confirmation split exactly and agreed on 4/5 checks; the known change-point divergence does not change the 0/24 survival result because change-point candidates were already rejected by controls.
- Manuscript usability: yes, exploratory only. Usable only as a negative discovery boundary, limitation, or program-planning result. It must be scoped to directional temporal IPV structure and to the InterHub sigma=0.1 discovery setting. It is not a confirmed law and cannot be used as evidence for a positive temporal IPV process.

## Rejected Claims

RQ008A rejects the following as unsupported by the discovery package:

- Any positive temporal IPV motif from the 18-row motif catalogue.
- Any continuous feedback-adjustment process inferred from this discovery data.
- Any reciprocity, complementarity, lead-lag, or role-negotiation law.
- Any risk-IPV coupling or hysteresis law.
- Any IPV-mean temporal law derived from descriptive atlas curves, alignment concentration, role/phase summaries, or motif labels.
- Any claim that uncertainty/estimability dynamics rescue an IPV-mean temporal claim.

Reason: every proposed candidate failed at least one matched control in Phase 7. Most motifs failed `reversed_time`; alignments failed `source_balancing` and `uncertainty_only_clustering`; role/phase candidates failed combinations of time shuffle, reversed time, pseudo-pair, duration, random-alignment, uncertainty-only, and estimability-matched controls; the uncertainty candidate failed pseudo-pair, duration-matched, and estimability-matched controls. These failures disqualify positive temporal-law language from RQ008A.

Manuscript usability: no. Rejected claims may be mentioned only as rejected or unsupported exploratory attempts, with their control failures attached.

## Pending / Conditional Claims

### RQ008-KC-PENDING-PRIMARY-NULL

- Type: pending
- Evidence ID: `RQ008-EV-FROZEN-NULL-PRIMARY`
- Status: `FROZEN_RQ008A`; not decided.
- Conditional claim: Held-out Wave B confirmation should again produce zero fixed catalogue rows surviving all matched controls and Holm correction.
- Wave-B gate: only explicit user authorization may open `confirmation_PROTECTED`. Confirmation was not run in RQ008A.
- Manuscript usability: pending. Not manuscript-usable as a decided claim until Wave B is authorized, run under the frozen protocol, and interpreted in the knowledge layer.

### RQ008-KC-PENDING-MOTIF-008-FALSIFICATION

- Type: pending
- Evidence ID: `RQ008-EV-FROZEN-FALS-MOTIF-008`
- Status: `FROZEN_RQ008A`; falsification-only target, not a positive claim.
- Conditional claim: Fixed `MOTIF-008` may falsify the primary null only if Wave B shows the joint-upward IPV-mean endpoint passing every required control, source check, material threshold, and Holm rule.
- Boundary: discovery statistic `0.6519597398`; killed by `reversed_time` in discovery.
- Manuscript usability: pending. It cannot be cited as a positive temporal motif from RQ008A.

### RQ008-KC-PENDING-UNCERTAINTY-FALSIFICATION

- Type: pending
- Evidence ID: `RQ008-EV-FROZEN-FALS-UNCERTAINTY`
- Status: `FROZEN_RQ008A`; falsification-only target, not an IPV-mean claim.
- Conditional claim: Fixed early-minus-late uncertainty/error decline may falsify the uncertainty endpoint only if Wave B passes the required matched controls and source checks.
- Boundary: discovery statistic `0.06560057061`; failed `pseudo_pair`, `duration_matched_null`, and `estimability_matched_controls` in discovery.
- Manuscript usability: pending. It cannot confirm IPV-mean dynamics.

## Wave-B Governance

The protected confirmation hold-out was never opened during RQ008A. Wave B requires explicit user authorization and the pre-Wave-B Attack-10 amendment before any confirmation data are read.

The required amendment is direction-sensitive: before Wave B, either add or pre-ratify a companion motif statistic that can test signed temporal direction, or explicitly scope the primary null to directional temporal IPV structure that must exceed reverse-invariant shape. Without that amendment, the current reversed-time/SNR pairing can over-kill reverse-invariant motif structure and must not be interpreted as ruling out all non-directional temporal shape.

Only Wave B, run after that amendment and fixed-code review, can elevate or overturn the RQ008A negative discovery boundary.

## Boundaries And Limitations

- Dataset/sigma scope: single dataset family and single sigma setting, InterHub sigma=0.1.
- Source and geometry imbalance: discovery data are dominated by `waymo_train` at 60.7% of cases and `MP` geometry at 90.3%.
- Estimability scope: estimability is proxy/error-based; no frozen RQ007 estimability or valid-window contract was available.
- Change-point sensitivity: independent replication found change-point-bin divergence, so change-point rows remain implementation-sensitive and exploratory.
- Reversed-time/SNR scope: one-vs-rest separation SNR can be direction-insensitive; the accepted negative boundary is about directional temporal structure, not all possible reverse-invariant shape.
- Confirmation scope: Wave B was not authorized or run. The accepted claim is discovery-layer only.

## Decision And Next Step

Decision: accept the RQ008A negative discovery boundary into the knowledge base. Do not open confirmation without explicit user authorization. Do not elevate any positive temporal IPV claim from RQ008A.

Recommended next step: either run Wave B after the Attack-10 amendment and code review are complete, or cite RQ008A only as an exploratory negative boundary in program planning and manuscript limitation language.

## Claims Table

| claim_id | type | evidence_id | status | manuscript_usable | scope |
|---|---|---|---|---|---|
| RQ008-KC-NEGATIVE-DISCOVERY-BOUNDARY | accept | `RQ008-EV-NEGATIVE-0OF24` | accepted | yes-exploratory | InterHub sigma=0.1 discovery split; 0/24 candidate directional temporal structures survived controls; boundary/limitation only |
| RQ008-KC-REJECT-POSITIVE-MOTIFS | reject | `RQ008-EV-NEGATIVE-0OF24` | rejected | no | No positive motif, alignment, role/phase, risk-coupling, lead-lag, reciprocity, feedback-adjustment, or IPV-mean temporal-law claim from RQ008A |
| RQ008-KC-PENDING-PRIMARY-NULL | pending | `RQ008-EV-FROZEN-NULL-PRIMARY` | pending: `FROZEN_RQ008A` | pending | Held-out Wave B primary null; zero survivors expected among the fixed 24-row catalogue |
| RQ008-KC-PENDING-MOTIF-008-FALSIFICATION | pending | `RQ008-EV-FROZEN-FALS-MOTIF-008` | pending: `FROZEN_RQ008A` | pending | Falsification-only `MOTIF-008` joint-upward endpoint; not a positive discovery claim |
| RQ008-KC-PENDING-UNCERTAINTY-FALSIFICATION | pending | `RQ008-EV-FROZEN-FALS-UNCERTAINTY` | pending: `FROZEN_RQ008A` | pending | Falsification-only uncertainty/estimability early-decline endpoint; not an IPV-mean claim |
