# Nature Skill Capability Check

Worker: `RQ003_phase0B_inventory_001`  
Checked: 2026-06-20T16:37:49+08:00

## Status

`FOUND_USABLE`

## Found Skill

- Skill name: `nature-figure`
- Skill path: `/Users/xiaocong/.claude/skills/nature-figure/SKILL.md`
- Manifest path: `/Users/xiaocong/.claude/skills/nature-figure/manifest.yaml`
- Version: `2.0.0` from skill front matter and manifest
- Skill SHA-256: `7cddee9337bc2bb6fe6b72e1cd89cf05eb00f26d947f4c1e37b52bdd566f3781`
- Manifest SHA-256: `279d77c4a087aa96aacbad0d29ffb4d1258d79518d5943494f0a82799158a153`

## Invocation / Routing

The Claude skill is invoked as `nature-figure` when creating, revising, auditing, or polishing publication-grade scientific figures. The skill requires a backend gate before plotting: the later plotting phase must explicitly choose `python` or `r`; if not chosen, the skill asks exactly `Python or R?` and stops.

For a valid invocation, the skill loads `manifest.yaml`, then always-loads `static/core/contract.md` and `static/core/stance.md`, then loads only the selected backend fragment. It supports manuscript-grade SVG/PDF/TIFF-style figure workflows and visual QA.

## Usability for Later RQ003 Figures

Usable, with one constraint: later plotting/HTML workers must not silently fall back to ad-hoc plotting. They must invoke `nature-figure`, resolve the backend explicitly, and follow its figure-contract and QA gate.
