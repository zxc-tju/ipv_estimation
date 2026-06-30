# Knowledge Folder Conventions

Authoritative naming and lifecycle rules for `reports/knowledge/`. This is the
standing spec; the one-off migration that introduced it is recorded in
`KNOWLEDGE_GOVERNANCE_PLAN_20260630.md`. New RQ folders are created with
`cp -r _template <id>_<short_topic>`.

## Folder model

```text
reports/knowledge/
  README.md                       # index + source-of-truth pointers
  CONVENTIONS.md                  # this file
  RQ_PROGRESS_DASHBOARD.md        # program-level board
  rq_progress_registry.csv        # machine-readable status
  _template/                      # canonical RQ skeleton
  _analysis/                      # cross-RQ engineering / analysis notes
  _governance/                    # lint + governance tooling
  <RQ###[_B]>_<topic>/            # one folder per registered research id
  PAPER###_<topic>/               # one folder per registered paper id
```

Infrastructure folders are prefixed `_` (`_template`, `_analysis`,
`_governance`). Everything else is a registered id.

## Standard files (per RQ / paper folder)

| File | Meaning | Required from |
|---|---|---|
| `README.md` | question, scope, current state | folder creation |
| `report_index.md` | index of execution reports | folder creation |
| `synthesis.md` | consolidated interpretation across reports | stage `review` onward |
| `decision.md` | accepted / rejected / deferred claims | folder creation |

`plan.md` does **not** live here. Plans are in `reports/plans/`; the RQ `README.md`
links to the relevant plan. (Folders at registry status `planning` only need
`README.md`.)

## Reviews (`reviews/`)

Lowercase, role-based filenames:

| File | Role |
|---|---|
| `chatgpt_review.md` | ChatGPT review of results/implementation |
| `claude_review.md` | Claude review |
| `codex_review.md` | Codex review |
| `codex_response.md` | Codex's reply to the above reviews (optional — only when a response exists) |

Rules:

- **One role, one file.** A second review round is a new `## Round 2 — YYYY-MM-DD`
  section *inside* the file. Dates go in the body, never in the filename.
- **Sub-question variants** that must be filed alongside a parent review use the
  suffix `_rq<NNN><letter>` (lowercase, matching the registered id), e.g.
  `claude_review_rq012b.md`. No date in the suffix.
- Filenames must match `^(chatgpt|claude|codex)_(review|response)(_[a-z][a-z0-9]*)*\.md$`
  — a role prefix plus optional lowercase suffix segments that each start with a
  letter (so `_rq012b` and `_wave_b` are fine), and **no bare date stamps**.

## Sub-questions (B-series)

- A B-series id gets its **own folder** only if it is a separate row in
  `rq_progress_registry.csv` (e.g. `RQ011B`). Its reviews use the standard role
  names inside its own `reviews/`.
- A B-series that is **not** separately registered (e.g. `RQ012B`) stays inside its
  parent folder; its artifacts use the `_rq<NNN>b` suffix. The canonical claim
  record remains the parent `decision.md`.

## Handoffs

Cross-RQ / paper handoff notes go in `<folder>/handoffs/`, named
`to_rq<NNN>_handoff.md` or `paper_handoff.md`. They are not standard files and are
not claim ledgers.

## `_analysis/`

Cross-RQ engineering or analysis notes that do not belong to a single question.
Name `<topic>.md`, optionally with a same-stem `<topic>.json` config. Provenance and
orphan status are tracked in `_analysis/README.md`. Raw machine artifacts ideally
belong in `reports/studies/`; `_analysis/` is the interim home for notes that have
no single RQ owner.

## Lifecycle

Status is expressed by `rq_progress_registry.csv` (`program_status`) plus each
`decision.md`. **Folders do not move when status changes** — an `archived-review`
or `closed-out` RQ keeps its folder in place.

## The 1:1 invariant and its exceptions

Each registered research RQ has one knowledge folder; for active research RQs the
stem matches the execution folder in `reports/studies/`. Documented exceptions:

- **Papers** (`PAPER###`) live in knowledge but may have no `studies/` counterpart.
- **Registered B-series** (`RQ011B`) have a knowledge folder but no separate
  `studies/` stem.
- **`PAPER001`** is imported manuscript context only and is exempt from the
  standard-file requirement.

## Pre-commit checklist (enforced by `_governance/lint_knowledge.py`)

1. No `.DS_Store` anywhere under `knowledge/`.
2. No loose `.md`/`.json` at the knowledge root except: `README.md`,
   `CONVENTIONS.md`, `RQ_PROGRESS_DASHBOARD.md`, `rq_progress_registry.csv`,
   `KNOWLEDGE_GOVERNANCE_PLAN_*.md`.
3. Every `reviews/*.md` matches the naming regex above.
4. Every RQ/paper folder has its required standard files (per registry status).
5. Every registry id (research + paper, including registered B-series) has a folder;
   every non-`_` folder is a registry id.
