# Codex Review: RQ007 Interaction-Conditioned IPV Estimability

Status: review-complete; study final review PASS; knowledge-layer `decision.md` is missing.
Review date: 2026-06-24.

## Scope

Reviewed study package:

- `reports/studies/RQ007_interaction_conditioned_ipv_estimability/RQ007_1_ipv_estimability_20260622T155229Z_289d9a99/`

Reviewed knowledge-layer files:

- `reports/knowledge/RQ007_interaction_conditioned_ipv_estimability/README.md`
- `reports/knowledge/RQ007_interaction_conditioned_ipv_estimability/report_index.md`
- `reports/knowledge/RQ007_interaction_conditioned_ipv_estimability/synthesis.md`

Primary study evidence read:

- `00_entry/index.html`
- `conclusions.md`
- `final_review.md`
- `final_no_blocker_review_status.json`
- `conclusion_freeze_status.json`
- `claim_evidence_matrix.csv`
- `reviews/independent_review/review.md`
- `reviews/red_team/red_team.md`
- `reviews/replication/replication_compare.md`

## Overall Verdict

RQ007 is strong enough to carry forward as a development/guard evidence package
for an interaction-conditioned IPV estimability contract. It supports the claim
that IPV estimates become more concentrated in selected interaction/opportunity
contexts and that this concentration is not merely a plotting artifact. It does
not yet support a final held-out scientific law, a causal timing claim, or a
claim that high estimability means the social state has stopped changing.

Paper-safe phrasing:

> In development and guard splits, selected interaction windows show lower IPV
> dispersion than matched non-opportunity or perturbed controls, but most of
> the gross contrast is shared with proximity/history structure. The remaining
> conflict-conditioned residual is modest and should be treated as an
> estimability/measurement boundary, not as a causal behavioral law.

## Claims That Can Be Carried Forward

1. Interaction-conditioned windows show lower per-frame IPV concentration index
   in development and guard splits. The main total contrast is about -0.13 index
   units in both splits, with replication values close to the primary run.
2. The total contrast is not wholly conflict-specific. Nearby nonconflicting
   controls explain a large share of the contrast, while the residual
   conflict-conditioned component is about -0.032 to -0.036 after proximity and
   history controls.
3. Perturbation checks are directionally supportive. The time-shift,
   counterpart-permutation, and re-estimated counterpart-switch controls weaken
   or remove the observed concentration pattern.
4. Estimability and behavioral settling are separate constructs. Low-index
   windows can still have substantial current-IPV movement, and high
   concentration should not be rewritten as `IPV = 0` or behavioral stasis.
5. Episode-level summaries are definition-sensitive. All-valid and
   interaction-active summaries can differ by about 0.26 rad on average and can
   flip sign in roughly one-fifth of cases; estimability weighting reduces but
   does not remove this sensitivity.
6. The correct downstream use is a validity guard: state when an IPV estimate is
   measurement-supported, where an episode summary is stable, and where a claim
   should abstain.

## Claims To Reject Or Defer

- Do not claim the full -0.13 concentration contrast is caused by conflict. The
  red-team review showed that proximity/history explains most of the gross gap.
- Do not claim temporal precedence or causal onset. The lifecycle result is
  descriptive only, with a substantial fraction of estimator changes appearing
  before the annotated opportunity point.
- Do not claim held-out confirmation. The package explicitly preserves the
  held-out split as sealed.
- Do not claim latent IPV ground truth, planner performance, or normative
  behavior validation. RQ007 is an estimator-validity package, not a behavioral
  truth package.
- Do not use the estimator-input recomputation check as a full robustness proof;
  the review evidence treats it as a sanity check only.

## Quality And Compliance Notes

The final study gate is clean: the no-blocker review passes, the conclusion
freeze status passes, evidence rows are registered, links and figure provenance
were checked, and the independent review plus red-team review were reconciled.
The replication packet reports a mixed-but-passing result because minor
implementation divergences remain while the main opportunity-mask and
concentration conclusions are stable.

The largest remaining risk is claim inflation. Any future manuscript text should
keep the phrase "interaction-conditioned estimability" instead of rewriting the
result as "conflict causes IPV identifiability" or "IPV stabilizes during
negotiation."

## Knowledge-Layer Action

The existing knowledge README and synthesis say the interpretation is accepted
or frozen in `decision.md`, but no `decision.md` exists in this knowledge
folder at review time. Before paper use, create the missing decision file or
revise the knowledge README/synthesis so the layer is internally consistent.

Recommended decision state: accept as a development/guard estimability boundary,
hold out final confirmation, and require proximity/history-bounded wording for
all manuscript-facing claims.
