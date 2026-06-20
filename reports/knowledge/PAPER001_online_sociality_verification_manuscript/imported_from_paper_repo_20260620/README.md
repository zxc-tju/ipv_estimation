# Imported Manuscript Knowledge Base

This directory was moved from the standalone paper repository into the research
repository on 2026-06-20. It remains the cross-agent manuscript memory, but the
paper repository should not recreate a local `knowledge/` directory.

## Directory Map

- `manuscript_structure.md` - current narrative spine, section plan, and claim boundaries.
- `analysis_reports/` - evidence memos, validation plans, and analysis summaries that support manuscript claims; start from `analysis_reports/evidence_index.md`.
- `drafts/` - alternate or superseded manuscript drafts and version-comparison notes.
- `agent_handoff.md` - chronological cross-agent handoff log.

## Rules

- Put durable evidence summaries here, not in chat transcripts.
- Preserve superseded drafts when they contain unique text or claims.
- Keep claims traceable: every major manuscript claim should point to either
  `main.tex`, an RQ decision under `reports/knowledge/RQxxx_topic/`, this
  `manuscript_structure.md`, or a note in `analysis_reports/`.
- Do not store raw datasets or heavy generated report folders here.
