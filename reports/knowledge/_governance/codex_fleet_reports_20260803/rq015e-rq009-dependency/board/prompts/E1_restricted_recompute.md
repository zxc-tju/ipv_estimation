You are agent **E1** in a research fleet. Repo root (your CWD):
`.`

Your leader has already located every input and fixed the design. **Do not re-explore the
repository.** Every path you need is listed below. Do NOT run repo-wide `rg`/`grep`/`find`
over `reports/` — a previous agent burned its entire budget doing that, and broad searches
can pull controlled-access rows into context. Read only the files named here.

Python interpreter — use this exact one, nothing else:
`<local-rq009-venv>/bin/python`

---

## 0. The question

RQ015A's audit found that **46.10%** of rows in the RQ009 feature matrix have `q_eff ≥ 0.93`
(candidate weights near-uniform ⇒ that IPV value carries no discriminative information
between candidates; `k_eff` median 6.06 of 7). RQ015A's C0 routing for
`rq009_feature_matrix` is `NO_AUDIT_TRIGGER_DETECTED` but `stable=false` (one of three
sensitivity settings flips to `OWNER_REANALYSIS_REQUIRED`).

**Question: do RQ009's headline numbers depend on those non-discriminative rows?**

This is a **sensitivity check by re-aggregation**, NOT a re-run of RQ009.

---

## 1. Absolute hard constraints — violating any of these voids the whole round

```
H1. RQ007 held_out must never enter any statistic. Contamination is unrecoverable.
    Operationally: build the dev+guard case set FIRST, then read prediction rows with a
    pushdown filter on that set. No metric, mean, count-by-value, distribution, or plot
    may ever be computed over a row outside the dev+guard set. The single scalar
    "how many rows were excluded by the join" is permitted (and required) — nothing else
    about those rows.
H2. Do NOT read RQ014 blinding-related rating fields. You have no reason to open any
    RQ014 path; don't.
H3. Do NOT modify, overwrite, or write into ANY RQ009 path, any RQ015A path, any
    decision.md, or anything under data/derived/. All your outputs go to the E board
    (section 6). Frozen artifacts are read-only.
H4. Results are descriptive/sensitivity. NO causal language. Do not write "because",
    "causes", "drives", "leads to" about the relationship between q_eff and the metrics.
H5. Do NOT re-run the RQ009 pipeline (calibration, refit, LODO). You only filter and
    re-aggregate frozen per-row predictions.
H6. Git: do NOT commit, do NOT `git checkout -- .`, `git restore`, `git stash`,
    `git reset --hard`, or `git clean`. Three tracks share this workspace concurrently;
    a dirty `git status` is the EXPECTED state and is not yours to clean.
H7. Forbidden vocabulary, everywhere in code comments and report:
    the word "estimability", and any phrasing like "IPV was/wasn't measured",
    "estimable/non-estimable". The only defensible phrasing is:
    **near-uniform candidate weights ⇒ that IPV value carries no discriminative
    information between candidates**. Use "usable subset" / "restricted set" for the
    subset, never "estimable subset".
```

### H1 in detail — the trap that kills this task

```
RQ009's {train, guard_tune, calibration, test} are FOLDS.
RQ007's {development, guard, held_out}          are SPLITS.
They are ORTHOGONAL. Filtering by fold does NOT exclude held_out.
Your leader measured it: fold=test contains 7,576 cases, of which 2,270 (29.96%)
are NOT in the dev+guard ledger — i.e. they are held_out.
```

The **only** admissible split source is the RQ015A ledger's `rq007_split` column
(path in §2). Do not invent, infer, or reconstruct a split assignment from anything else.

---

## 2. Inputs — exact paths, all read-only

**(a) RQ007 split ledger** (also the source of `q_eff`; 8,994,736 rows; columns include
`case_id`, `rq007_split`, `q_eff`, `k_eff`, `measurement_role`, `aggregation_perspective`,
`aggregation_configuration`):
```
reports/studies/RQ015A_ipv_estimability_labelling/RQ015A_1_concentration_audit_20260731T093746Z_e82091ce/concentration_ledger/rq009_feature_matrix.parquet
```
Verified by the leader: `rq007_split` takes exactly two values —
`development` 6,459,684 and `guard` 2,535,052; sum = 8,994,736 = total rows.
**held_out count in this ledger is 0.** 26,828 distinct `case_id`, no case appears in
both splits (split is case-level).

**(b) Usable subset** (34,283 rows / 19,778 keys across 3 artifacts; take ONLY rows with
`artifact_id == 'rq009_feature_matrix'` → 27,076 rows / **14,190 distinct**
`case_or_episode_key`):
```
reports/studies/RQ015A_ipv_estimability_labelling/RQ015A_1_concentration_audit_20260731T093746Z_e82091ce/usable_subset.csv
```
Leader verified: those 14,190 keys are a **subset** of the ledger's 26,828 dev+guard
cases. So restricting to the usable subset excludes held_out *by construction*.

**(c) Frozen per-row predictions** — one parquet per tier:
```
data/derived/interhub/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/04_calibration/predictions/tier=<TIER>/fold=test/predictions.parquet
```
with `<TIER>` ∈ `M0`, `M2`, `M3`, `ipv_removed` (these four are all you need).
Schema: `case_key, anchor_frame_index, perspective, source_dataset, fold, tier, alpha,
nominal, q_lo, q_hi, lo_cal, hi_cal, width, abstain, y`.
M2/fold=test has 3,811,698 rows = 3 alphas × 1,270,566 anchor-perspective rows.
`predictions.case_key` joins to `ledger.case_id` (same `ipv_XXXXXX` namespace).

**(d) Frozen metric code — you MUST reuse it, not reimplement it:**
```
reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/05_evaluation/evaluate.py
```
`base_metric_row(...)` at line 123 computes coverage / mean_width / median_width /
winkler / pinball / tails / abstention. Import it via `importlib.util.spec_from_file_location`
(do **not** copy-paste the body, do **not** write your own coverage/Winkler formula —
code-level reuse is the ONLY parity guarantee you get, see §5).

**(e) Frozen published metrics, for reference only:**
```
reports/studies/RQ009_dynamic_counterpart_conditioned_envelope/RQ009_1_dynamic_envelope_20260625T121905Z_98c433de/02_process/05_evaluation/metrics_summary.csv
```
**(f) The accepted claim being tested:**
```
reports/knowledge/RQ009_dynamic_counterpart_conditioned_envelope/decision.md
```

---

## 3. The four frozen numbers under test (RQ009-KC-R3, at nominal 0.90 / alpha 0.10)

| # | quantity | frozen value | derivation from metrics_summary.csv |
|---|---|---|---|
| 1 | M2 vs M0 `mean_width` relative change | **−42.3%** | 1.009483 / 1.748666 − 1 = −0.42272 |
| 2 | M2 vs M0 `winkler` relative change | **−35.6%** | 1.423146 / 2.210261 − 1 = −0.35611 |
| 3 | M2 `coverage` | **0.898889** | direct column |
| 4 | `abstention` | **0.047781** | 60,709 / 1,270,566 |

---

## 4. The design — three sets, you compute two of them (plus the complement)

The frozen row set is NOT held-out-free, so you **cannot** reproduce the published numbers
without violating H1. Do not try. Instead:

| set | definition | expected cases | compute? |
|---|---|---|---|
| **A** | frozen scope: all of fold=test | 7,576 | **NO.** Quote published values only; annotate that its row set is not held-out-free |
| **B** | fold=test ∩ ledger(dev+guard) — **the reference frame** | 5,306 | YES |
| **C** | fold=test ∩ usable_subset(rq009) — **the restricted set** | 2,825 | YES |
| **D** | B \ C — the complement, locates where any shift comes from | 2,481 | YES |

**The scientific comparison that answers the question is C vs B.** Both are held-out-free
and differ only by the usability restriction, so any difference is attributable to the
restriction and not to the split. D is diagnostic.

Treat the expected case counts above as **pre-registered checks**: if your numbers differ,
STOP and report the discrepancy in the report rather than "fixing" it silently.

### What to compute, for each of B, C, D

At **alpha = 0.10 (nominal 0.90)** — this is the headline row. Also emit alpha 0.20 and
0.05 into the CSV since it is free, but the report's judgement is on 0.90.

For each tier in {M0, M2, M3, ipv_removed}, per set, via `base_metric_row`:
`n`, `total_n`, `abstained_n`, `abstention`, `n_cases`, `total_cases`, `coverage`,
`mean_width`, `median_width`, `winkler`, `pinball`, `lower_tail`, `upper_tail`,
`directional_tails_ok`, `coverage_within_3pp`.

Then the four derived quantities per set:
```
delta_width_pct   = mean_width(M2) / mean_width(M0) - 1
delta_winkler_pct = winkler(M2)    / winkler(M0)    - 1
coverage_M2
abstention           (identical across tiers by construction; assert this)
```

**Primary restriction granularity = case level** (a case is in C iff its `case_key` appears
in usable_subset rows with `artifact_id == 'rq009_feature_matrix'`).

**Secondary variant (also required, it is cheap):** `(case, perspective)`-level restriction.
usable_subset has `aggregation_perspective`; predictions have `perspective`. Inspect the
actual value vocabularies of both before joining and **report the mapping you used**
(e.g. whether `key_agent_1` ↔ `agent_1`). If the vocabularies cannot be mapped
unambiguously, do NOT guess — report that the secondary variant was not computable and why,
and proceed with the case-level primary. That is an acceptable outcome; a wrong join is not.

**Optional, best-effort, must not block delivery:** the internal ablation M3 vs M2 paired
90% Winkler difference on set C (frozen internal value: −0.0002, case-cluster p=0.863).
Report the mean paired difference; skip the p-value if the case-cluster bootstrap is not
cheaply reusable. Label it clearly as an **internal ablation, not a manuscript claim** —
report the number and state nothing about what it means.

Also report, for context (descriptive, no causal framing): the `q_eff` distribution
(median, mean, share ≥ 0.93) over ledger rows belonging to C's cases vs D's cases. This
characterises what the restriction actually selected.

---

## 5. Verification you must perform and report

1. **Held-out exclusion, structural.** Report:
   - ledger split counts: development + guard = total ledger rows (exact equality);
   - `n_rows_B + n_rows_excluded_by_join == 1,270,566 × 3` for the full fold=test parquet
     of each tier (the exact identity; state the numbers);
   - `held_out_rows_entering_any_statistic = 0`, and state the mechanism: the dev+guard
     case set was built first and applied as a **pushdown filter** (`pyarrow.dataset`
     `filter=pc.field('case_key').isin(...)`) so that no `y` / `lo_cal` / `hi_cal` / `width`
     value from an unmatched row was ever materialised into a statistic.
   - `C ⊆ B` verified as a set operation (must be True).
   Be precise and do not overclaim: say exactly what was excluded and how, rather than
   claiming held_out bytes were never touched by the parquet reader.
2. **Metric-code parity.** State that `base_metric_row` was imported from the frozen
   `evaluate.py` and not reimplemented, and give the import mechanism. Note explicitly
   that numeric parity against the *published* set-A numbers is **not** performable under
   H1, and that code-level reuse is therefore the parity argument.
3. **Sanity.** Assert `abstention` is identical across the four tiers within each set
   (abstain is a row property, not a tier property — if this fails, say so loudly).
4. **Health.** Check for NaN/inf in the metric outputs, empty groups, and any tier whose
   row count differs from the others within a set.

---

## 6. Outputs — write ONLY to these paths

Board root: `.codex-fleet/rq015e-rq009-dependency/board/`

```
board/reports/E1_dependency_report.md    ← the deliverable (Markdown; Chinese is fine)
board/reports/restricted_metrics.csv     ← long table: set × tier × alpha × all metrics
board/reports/E1_numbers.json            ← machine-readable: the 4 quantities × {B,C,D}
                                            + frozen A values + all verification scalars
board/scripts/e1_restricted_recompute.py ← your script, must be re-runnable
```
Create the directories if needed. Nothing else, nowhere else.

### Report structure (keep it tight — this is a diagnostic, not a treatise)

1. **判定（先给结论）** — one paragraph: on the held-out-free reference frame B, does
   restricting to C move the four quantities materially? Give the verdict as either
   "结论稳住" or "结论变了", and if it changed, state **exactly which quantity moved and
   by how much** (absolute and relative). If the picture is mixed (e.g. width holds but
   coverage drifts), say that plainly rather than forcing a single verdict.
2. **Numbers table** — A (quoted, annotated as not held-out-free) | B | C | D, four
   quantities, at 0.90. Plus Δ(C−B) per quantity.
3. **What the restriction selected** — case/row counts, q_eff profile of C vs D.
4. **Secondary variant** — (case, perspective) restriction result, or why not computable.
5. **Internal ablation** — M3 vs M2 paired Winkler on C, numbers only, labelled non-claim.
6. **Verification** — everything in §5, as a checklist with actual values.
7. **Limits** — at minimum: this is a re-aggregation of frozen predictions, not a refit,
   so it cannot speak to how the model would have been calibrated on the restricted set;
   set A is not reproducible here; the restriction is defined by RQ015A's single
   `primary` usable policy (`q_n>=30; attempted_share>=0.80; median_q_eff<=0.75;
   share(q_eff<=0.75)>=0.60`) and inherits its cut choices.

**Do NOT recommend whether RQ009 should be re-estimated.** That is the PI's decision.
Give evidence only. No "we recommend", no "should be re-run", no "this invalidates".

---

## 7. Working rules

- One pass. Do not produce a v2 spec, do not open a review round, do not add gates.
  Run it, self-check once, write the report, stop.
- Timestamps: `date -u +%Y-%m-%dT%H:%M:%SZ`. Never estimate a time forward.
- If something genuinely blocks you, write the blocker into the report and deliver
  everything that is not blocked. Do not stall silently.
- Memory: the ledger is ~248 MB and predictions ~3.8M rows/tier. Read only the columns
  you need, use pushdown filters, and process tier by tier.
