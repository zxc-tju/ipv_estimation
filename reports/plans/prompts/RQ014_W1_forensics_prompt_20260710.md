# W1-forensics — recover residue of the lost WOD-E2E rating↔IPV-deviation study

ROLE: experimenter (digital forensics). READ-ONLY everywhere except your report directory.

## Context (self-contained)
A prior study on the WOD-E2E dataset showed: higher human rating → lower deviation of the
candidate trajectory from a human "IPV envelope" (strong, significant, negative
correlation). Its data, code, and configuration are lost. The current repo
(`~/Library/CloudStorage/OneDrive-个人/Desktop/Projects/1_Codes/2_sociality_estimation`)
contains only the LATER RQ010B pipeline (2026-06-22 onward), whose registered result is a
bounded NULL — so any WOD/rating artifact dated BEFORE 2026-06-22, or any variant config
differing from RQ010B (trajectory frequency, IPV window history/future/combined, envelope
definition), is a high-value target. Repo-internal search was already done and found
nothing; do NOT re-search inside the repo except `.codex-fleet/` and untracked/ignored
paths.

## Objective
Find any fragment of the lost study: scripts, result tables, plots, transcripts
discussing it, Slurm jobs, sbatch files, or config values (frequency/resampling, IPV
window, envelope/quantile band, deviation metric, counterpart source).

## Search surfaces (in order)
1. Run `bash reports/plans/prompts/RQ014_forensics_mac_local_20260710.sh` (Mac stores:
   Cowork session transcripts, Claude Code transcripts, codex sessions, sibling projects,
   ~/.rq009_codex_fleet, Spotlight).
2. Run the HPC scan:
   `ssh -o BatchMode=yes -o ConnectTimeout=12 tongji-hpc 'bash -s' < reports/plans/prompts/RQ014_forensics_hpc_readonly_20260710.sh`
3. For every hit from 1–2: open the file/transcript, extract the exact experimental
   settings and numbers mentioned. For transcripts, quote the relevant exchanges verbatim.
4. Follow one level of indirection (a hit that references another path → check that path).

## Hard constraints
- Read-only: no rm/mv/edit anywhere; HPC login node = light commands only, no compute.
- Write outputs ONLY under
  `reports/studies/RQ014_wod_e2e_rating_recovery/00_forensics/` (create it).
- Save the two raw scan outputs there, plus your report as `forensics_report.md`.

## Required bounded final report (≤120 lines)
1. VERDICT: FOUND_CONFIG / FOUND_FRAGMENTS / NOTHING — one line.
2. Findings table: path | date | what it is | which config axis it pins down | confidence.
3. Verbatim key fragments (trimmed).
4. Reconstructed config hypotheses, ranked (frequency / IPV window / envelope /
   counterpart / deviation metric / unit — mark unknown axes as UNKNOWN).
5. Dead ends checked (one line each).
6. Manual follow-ups only a human can do (OneDrive web history, old Windows machine).
