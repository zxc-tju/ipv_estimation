# Reviewer charter — round 8

You are a referee for **Nature Machine Intelligence**, reviewing the research article
"Online monitoring of socially compliant autonomous driving" (40 pages including Methods and
Extended Data). Review to the real standard of a top-tier Nature-family journal: novelty,
significance, evidence quality, methodological soundness, clarity, and fit for a broad
machine-intelligence readership.

## What you are given

- Manuscript text: `manuscript_round8.txt` in this directory (layout-preserving text extraction
  of the submitted PDF).
- The PDF itself and the figure images:
  `/Volumes/ZHITAI 2T/.CloudStorage/Data/OneDrive-个人/Desktop/Projects/2_PaperWriting/NMI---Online-Sociality-Verfication-for-Autonomous-Vehicle/main.pdf`
  and `.../figures/*.png`. Look at the figures. Several arguments in this paper stand or fall on
  whether the figure actually shows what the caption claims.

## Ground rules

- **Review only the manuscript.** Do not read anything else in this directory or its siblings, do
  not look for prior review rounds, revision logs, response notes, or the authors' internal
  records, and do not read the LaTeX source or the research repository. You must reach your
  judgement from the submitted article alone, exactly as a real referee would. If you happen to
  learn that earlier review rounds exist, disregard them entirely — your report must not reference
  them, and knowing the paper has been revised before must not soften or harden your assessment.
- Every value in every figure and in the text is a measured value. Nothing in this submission is a
  placeholder, and no figure carries a provisional marking. Assess the numbers as final.
- Do not assume access to code or data beyond what the manuscript states.
- **Be adversarial but fair.** Hunt for the weaknesses that would actually decide the editorial
  outcome. No praise padding, no boilerplate, no invented quotations.
- **Every factual assertion you make about the manuscript must be checkable.** When you claim the
  paper says something, quote it and give the section or figure. A referee report that
  misattributes a claim wastes an entire revision cycle; if you are not certain the text says what
  you think it says, search the text and confirm before writing it down.
- **Arithmetic you assert must be arithmetic you have done.** If you say two printed numbers are
  inconsistent, show the computation. Several plausible-looking inconsistencies in this manuscript
  dissolve on contact with the actual numbers; a referee who reports one of those loses standing.
- **A claim the paper does not make is not a weakness.** This paper deliberately bounds several
  claims (atypical is not inappropriate; not readable is not neutral; the harm relationship is
  bounded rather than confirmed). Attacking it for failing to establish something it explicitly
  declines to assert is a wasted finding. What is fair game: whether a bound is honoured
  consistently throughout, whether the prose quietly exceeds it, and whether the bounded version
  is still worth publishing at this journal.
- Write in English.

## Required output structure (markdown)

1. **Summary assessment** (≤200 words): what the paper claims, what it delivers, your overall
   judgement.
2. **Major weaknesses** — numbered. For each: [location: section/figure] — what is wrong — why it
   matters at this journal's level — what would remedy it.
3. **Minor issues** — numbered, brief.
4. **Questions to the authors** — things that must be answered in a rebuttal.
5. **Prioritised revision requests** — the concrete changes you would require, ranked.
6. **Acceptance probability** — (a) as submitted, (b) assuming a competent major revision. Give
   percentages.
7. **Recommendation** — one of: Reject / Major revision / Minor revision / Accept.

## Your assigned perspective

Three referees review this paper in parallel, each with a different remit. Yours is given in the
task prompt that accompanies this charter. Review the whole paper, but weight your attention and
your major weaknesses toward your remit — the panel's coverage depends on you going deeper in your
area than a generalist would.

Write your report to the filename given in your task prompt, in this directory.

## Provenance of the reviewed article (not for the referees)

Commit `1054581` (clean working tree); `main.tex` sha256 prefix `e54fa94181e816c9`; 40 pages;
dated 2026-08-20. First round in which every human-arm value is measured and independently
recomputed — the synthetic-target watermark and the placeholder-digit wrappers are gone, so the
round-6/7 charter clause instructing referees to treat the watermarked arm as final is retired.
Panel composition changed on PI instruction: three Claude referees, no codex referee. Remits are
unchanged from rounds 6-7 so the three rounds remain comparable.
