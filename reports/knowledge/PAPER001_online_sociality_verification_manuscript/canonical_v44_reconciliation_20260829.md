# PAPER001 canonical manuscript reconciliation — v4.4

Date: 2026-08-29  
Paper repository merge: `2dab39d60216633bb6b1a9713d4ff2585eae2024` (PR #11)

## Decision

The paper repository `main` is now the sole canonical manuscript baseline. It reconciles the 23 August semantic draft with the 28 August estimator-scope decision.

## Canonical content

- The published T-ITS IPV estimator remains a fixed method component; detailed validation and implementation stay in Supplementary Note 1.
- The accepted two-sided human-interaction evidence is restored: partner complementarity, early role legibility, turning-yield shift, geometry prior and the source-transfer boundary.
- Waymo, nuPlan, Lyft and Argoverse-2 are used only for human-human interaction structure and human-reference construction/audit.
- The public-data AV/fleet comparison and its figure are removed.
- All AV-human comparisons come from the matched OnSite automated and human arms.
- Results are organised into five semantic sections and five main figures.
- The prospective subjective-evaluation experiment remains outside the manuscript Results until completed and accepted.

## Synchronized files

- `main.tex`
- `structure.md` — v4.4
- `claims_register.md` — v4.4
- `CLAUDE.md` — v4.4 collaboration contract

## Verification

The merged paper branch passed source lint, full LaTeX compilation, bibliography processing, undefined-reference/citation checks and PDF artifact generation before merge.
