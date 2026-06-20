# `main.tex` Version Note

On 2026-06-20, three `main.tex` copies were compared:

| Copy | Timestamp / commit | Size | Lines | SHA-256 | Interpretation |
|---|---:|---:|---:|---|---|
| Parent project `paper/main.tex` | 2026-06-19 22:58 +0800 | 28,616 bytes | 464 | `e62eca4b786d66500b6d4902bf0f03efc7ba76fd534f0d199d77898ba721ae40` | Newer v3 self-anchor narrative; promoted to active `main.tex`. |
| GitHub remote `origin/main:main.tex` | 2026-06-19 11:30 +0800 | 34,476 bytes | 563 | `d7dfa96750d3755410f31bc0fdd850eecc69f2d7ae5142d94b6eb752fc283fd6` | Longer v2 evidence-scoped draft; archived here before sync. |
| Existing local Overleaf clone `../9_overleaf/nmi_sociality_verfication/main.tex` | 2025-08-09 10:05 +0800 | 14,538 bytes | not used | not used | Older local working copy; not used as active source. |

Decision: use the parent project's v3 self-anchor file as active `main.tex`, but preserve the GitHub remote v2 draft at `main_remote_github_before_sync_2026-06-19_113002.tex`.

Rationale: the parent project copy has the newer timestamp and matches `knowledge/manuscript_structure.md` v3. The GitHub copy is not a subset, so it is preserved for a deliberate merge pass.
