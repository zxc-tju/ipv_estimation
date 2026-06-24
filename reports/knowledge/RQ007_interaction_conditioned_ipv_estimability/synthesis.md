# RQ007 Synthesis: Interaction-Conditioned IPV Estimability

Status: study final review PASS; knowledge-layer `decision.md` missing. Development/guard only; held-out sealed.

Run: `RQ007_1_ipv_estimability_20260622T155229Z_289d9a99` (overall `COMPLETE`; all gates PASS).
Report: `reports/studies/RQ007_interaction_conditioned_ipv_estimability/RQ007_1_ipv_estimability_20260622T155229Z_289d9a99/90_report/index.html`.
Candidate carry-forward claims: C1, C2, and C3 below. Treat them as not paper-frozen until a
knowledge-layer `decision.md` exists.

## Research question

Is the IPV equally interpretable at every timestamp, or is per-frame IPV identifiability
interaction-conditioned? RQ007 tests whether, within causal interaction-opportunity windows,
the per-frame IPV estimate is more identifiable, and separates that estimability from
behavioural settling and from episode-summary choice. It underwrites the v4.1 estimability
contract: the verifier must separate interaction opportunity, IPV estimability,
human-reference support, and social deviation, and must not read high estimation uncertainty
as a neutral IPV.

## What The Reports Establish

**C1 (MODERATE, strong boundaries) — estimability is interaction-conditioned, but mostly via
proximity.** Within causal opportunity windows (`cv_cpa_conflict`), the per-frame estimator
concentration index is lower (IPV more identifiable) than history-length expectation by a
total of about 0.13 index units (development -0.132, guard -0.129; independently replicated
-0.134 / -0.133). The effect is time-locked (time-shift control collapses the gap to +0.006)
and counterpart-specific (counterpart permutation +0.021; a re-estimated counterpart switch
reverses it to +0.122), so it is not a history-length, arbitrary-pairing, or alignment
artifact. However, the majority of the gap (about -0.096) is reproduced by a nearby
non-conflicting actor, i.e. it is spatial proximity. The conflict-geometry-specific increment
beyond proximity is small but nonzero: about -0.032 to -0.036, with case-clustered CIs
excluding zero (dev -0.034 [-0.038, -0.031]; guard -0.032 [-0.038, -0.026]). Distant
no-opportunity controls sit near/above zero (+0.027 / +0.026). 26/26 analysis-level
perturbations robust.

**C2 (SUPPORTED) — estimability is not behavioural settling.** Under low (estimable) index the
current-IPV estimate keeps changing: mean |dθ| ≈ 0.30 ego / 0.31 counterpart in event-window
low-index transitions (≈0.34 / 0.36 across all development+guard transitions). High index is
not IPV = 0 (higher-index mean |dθ| still ≈ 0.17). Concentration index, the current IPV
estimate, and any episode summary are therefore distinct constructs and must not be conflated.

**C3 (SUPPORTED) — episode IPV summary is definition-dependent.** All-valid-frame mean versus
interaction-active mean differ by ≈0.26 rad on average and flip strict sign in ≈22% of cases;
an estimability-weighted summary is closer to all-valid (≈7% sign flips, Spearman ≈0.91). An
"episode IPV" is not a definition-free quantity; the summary rule must be stated.

**Descriptive lifecycle (not a standalone claim).** Concentration-index minimum occurs near
resolution (ego 0.374 pre → 0.275 resolution → 0.466 post; counterpart similar), but onset
timing is roughly 44–46% before / ~54% after the first opportunity frame (median Δ ≈ 0.2 s),
so this is descriptive lifecycle evidence, not clean causal precedence.

## What The Reports Do Not Establish

- Not that the full -0.13 gap is interaction-specific; proximity/history explains the majority.
- No IPV "truth" / latent-preference recovery, no planner-performance, and no causal-effect claim.
- No held-out confirmation: the sealed split is untouched; all claims are development/guard only.
- Estimator-input reruns are sanity checks only (recompute mismatch mean ≈0.11, p95 ≈0.45), not rigorous proof.
- No map/lane, observed PET, intensity, passing order, priority, or outcome fields were used.
- The post-opportunity concentration rise is descriptive, not temporal precedence (cf. RQ008's negative temporal-discovery boundary).

## Boundary Conditions

- Splits: 19,258 development / 7,628 guard / 11,342 sealed cases (38,228 case/episode units; 3,695,981 rows scanned for case-id-only assignment). Outcome-blind, case-level split.
- Opportunity definition `cv_cpa_conflict` (both agents active, closing, projected close approach within a constant-velocity horizon); frame coverage ≈0.04, case coverage ≈0.245.
- Single InterHub setting; "estimability" = estimator concentration index (an identifiability proxy), not a standard deviation.
- Case-clustered intervals throughout; independent replication PASS/MIXED (headline gaps reproduce; minor window/bin and rounded-τ differences).
- Independent review PASS (0 blocking, 0 major, 2 minor); red team PASS (claim shrunk to a proximity-bounded residual statement).

## Manuscript-Safe Language

- "IPV is not equally estimable at every timestamp; within causal interaction-opportunity windows the per-frame estimate is more identifiable, though most of this concentration reflects spatial proximity, with a small but nonzero conflict-geometry-specific increment (development/guard only, held-out sealed)."
- "Estimability is distinct from behavioural settling: under estimable frames the current IPV estimate still changes, and high estimability does not mean IPV ≈ 0."
- "Episode-level IPV summaries are definition-dependent (strict sign flips in ≈22% of cases between all-valid and interaction-active means), so an episode IPV must state its summary rule."
- Supports the v4.1 estimability contract (separate opportunity / estimability / support / deviation). Do not phrase as a temporal IPV law or as causal precedence, and do not interpret high uncertainty as neutral IPV.
- Keep all wording provisional until held-out confirmation under the frozen contract.

## Relation to other RQs

Provides the measurement/estimability foundation for the v4.1 layer in
`PAPER001_online_sociality_verification_manuscript` and the planned RQ009 dynamic
counterpart-conditioned envelope (which consumes the valid-window/estimability contract).
Consistent with RQ008's negative temporal-discovery boundary: RQ007's lifecycle is explicitly
descriptive, so it does not assert the directional temporal IPV law that RQ008A failed to find.
Independent of the RQ002 self-anchor / norm-laundering line.
