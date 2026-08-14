# Backlog delta — round 2 (NEW items only; B01–B17 live in ../round1/BACKLOG_draft.md)

Round-2 reviewers re-demanded, without new facts, the existing backlog items B01 (construct validation), B02 (kinematic-baseline incremental control), B03 (consequence landmark redesign), B04 (sequential/persistence calibration + run-length structure), B05 (cluster-aware conformal + stratified coverage; now also carries B-q11's subgroup-disparity audit), B06 (reproducibility spec; now also carries: quantile-model class docs, interval-clipping behaviour, per-source frame-rate/preprocessing table, principal-counterpart rule, situation-cell variable list, non-lane-referenced-slice handling), B07 (estimator sensitivity incl. candidate-grid endpoints/boundary bias), B08 (non-inferiority margins + multiplicity), B09 (uncertainty for descriptive panels; now also Fig. 4c R² CIs and Fig. 2a paired condition contrasts), B11–B12 (C7 completion + crossed-design inference), B13 (online implementation study), B15 (leakage-audit note), B16 (benchmark documentation pack), B17 (abstention–kinematics dependence). Do not re-log those. The genuinely new items:

## B18 — Over-yielding flag class: outcome characterisation
- Needs: efficiency/delay endpoints (time-to-clear, throughput loss), follower-pressure and counterpart-confusion proxies computable from the same benchmark logs, evaluated at Over-Yielding moments under the same frozen protocol style (pre-specified battery, run-clustered inference), plus the human-arm mirror once C7 real data exist.
- Resolves: D-M2, D-q5, D-P4 (majority flag class — 869/1,388 = 63% — currently has only the null assertive-style signature).
- Upgrades if done: the hesitation/efficiency planner mapping from interface specification to an evidence-backed verdict class; without it the over-yielding verdict stays exploratory (which the manuscript's current wording already tolerates).
- Constraint memo: new endpoints = endpoint-set change for the human arm → E2/E10 ruling first for the cross-arm part; AV-side-only analysis needs no C7 gate but does need a frozen battery before looking.

## B19 — Benchmark missingness audit
- Needs: reason-coded accounting for (i) the 18 never-produced runs (285 possible vs 267 observed), (ii) the 65 non-scripted runs without recorded outcomes (240 vs 175), (iii) the 47/519 flagged moments without a defined post-verdict margin and the 50 flagged moments outside the counterpart-record set (519 vs 469), with tests of whether each missingness channel correlates with flag status or outcome severity. Items (i)–(ii) additionally need benchmark-organiser records that are not in the workspace (→ E6/B16 for the facts; the correlation analysis is ours once logs are in hand).
- Resolves: D-M4, D-q3; B-m11 (missingness dependence); A-m7.
- Upgrades: the run/denominator accounting from "frozen universes disclosed" (round-1 P17) to "missingness shown ignorable or bounded"; the near-balanced undefined-margin rates already frozen (9.06% vs 8.20%, RQ018 B4 — being surfaced now as plan Q18) become one row of this audit rather than its whole answer.

## B20 — Alternative-denominator sensitivity for flag rates
- Needs: recompute the population flag contrasts under prespecified alternative estimands — flags per run, flags per active-interaction second, per-scenario equal weighting, and first-flag-per-interaction — for the AV arm now and the human arm after C7; report how the headline contrast moves across estimands.
- Resolves: D-M4 (per-frame length bias), C-M3 (estimand prespecification: random frame vs run vs scenario), B-M6-part.
- Upgrades: the per-moment flag-rate comparison from a single frame-weighted estimand to an estimand-robust statement; protects the abstract's contrast (whatever E2 decides) against the length-bias objection.
- Constraint memo: touching the human-arm side is endpoint work → E2/E10 first; AV-only sensitivity is ungated but must be frozen before computation.

## B21 — Oracle-counterpart ablation and equivalence-grade reporting (extends B10)
- Needs: (i) confidence intervals and a minimum detectable effect for the frozen counterpart-IPV and self-history ablation contrasts (B10 already covers the TOST framing); (ii) a NEW oracle variant conditioning on the counterpart's offline-estimated (full-trajectory) IPV, to separate "counterpart preference adds nothing" from "the online counterpart estimate is too noisy to add anything".
- Resolves: D-M7, D-q6 (attenuation critique of the null ablation).
- Upgrades: the headline event-vs-case tension ("partners' preferences co-vary, yet the online range needs only the situation") from a p=0.863 null to a bounded equivalence claim robust to the measurement-noise counter-explanation; also the strongest available defence of the Discussion's "no reading of hidden intention" framing.
- Constraint memo: the offline counterpart estimate is anticipatory — the oracle ablation is a diagnostic, never a deployable channel; write-up must keep it out of the monitor definition (freeze-order discipline).

## B22 — Per-situation-cell human-arm calibration diagnostics + two-sided acceptance band (E-gated)
- Needs: once C7 real data exist — per-situation-cell outside rates in the matched human arm (15,102 judgeable moments make cells feasible), a PRE-STATED two-sided acceptance band for the on-course human outside rate (against both alarm inflation and insensitivity/over-coverage), and the reconciliation narrative against the LOSO bounds.
- Resolves: D-M6, D-q7; C-q7 (why is 4.7% vs nominal 10% "success"); A-q14; B-M6-part.
- Upgrades: "the instrument survives the move" from a one-sided no-inflation statement to a calibration statement with a declared rejection region — the single change most likely to convert reviewer D's M6 and C's M4 on resubmission.
- Constraint memo: this ADDS endpoints to the pre-registered human-arm set → requires the E2/E10 PI ruling BEFORE any band is chosen; choosing the band after seeing the real rate would void it. Log here so the band is fixed at swap time, not after.
