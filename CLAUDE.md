# Project memory — Online Social-Compliance Verification for AVs (NMI)

## Overleaf commit rule (IMPORTANT)
Only **manuscript-content** files may be committed/pushed to the Overleaf repo
(`git.overleaf.com/68387d43a6513a514e161a2f`, branch `main`). Do **not** push
research-process or analysis materials to Overleaf.

**Push to Overleaf (paper content only):**
- `main.tex`, `bibliography/biblio.bib`, `structure.md` (manuscript outline, already in repo),
  and figure assets that the manuscript actually `\includegraphics`.

**Keep local only — never push to Overleaf** (live in the workspace folder, e.g. `paper_draft/`):
- strategy / 思路 docs, narrative-adjustment notes
- experiment plans, codex prompts, validation plans
- analysis memos, extracted/`uploaded` reports (InterHub, NSFC, etc.)
- superseded drafts, scratch files

When pushing, copy only the allowed files into the clone before `git add`; do not `git add -A`
blindly if non-manuscript files are present in the clone.

## Context
- Paper: online runtime verification of socially compliant autonomous driving.
- Core: self-anchored, conformally calibrated IPV reasonable-interval verifier (PET/risk
  excluded from the online path); InterHub defines the human norm; NSFC onsite challenge is
  the external validation set.
- Working copies live in `paper/`. Workspace is OneDrive-synced (bash may not delete
  files; use Read/Write/Edit, overwrite rather than delete).
