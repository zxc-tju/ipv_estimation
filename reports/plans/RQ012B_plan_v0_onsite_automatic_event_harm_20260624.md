# RQ012B Plan v0 — OnSite Automatic-Event Harm Analysis (no human labels)

Status: `approved` (PI authorized 2026-06-24; human blind annotation deprecated) · Wave: B · Work group: Group 6B · Date: 2026-06-24

## 1. Research question

> Using automatic interaction events (no human labels) plus OnSite official outcomes, does frozen-M3 IPV
> deviation align with realised interaction events / harm (the consequence half of manuscript R5)?

## 2. Scope decision (PI 2026-06-24)

Human two-annotator blind labelling is **deprecated**. The behaviour/consequence reference is the frozen
**automatic event extractor** (9 events, precedence/identity guards) + **official collisions/deductions**.
Human-judgment alignment is carried elsewhere (WOD preference + InterHub), not here.

## 3. Frozen inputs / contracts

RQ012 revised decision (automatic events + official outcomes; no human labels; construct-proximal labels are
secondary, never primary); RQ011 frozen universe (clean_285 replay / full_300 outcomes); RQ009 frozen M3;
RQ005 leakage contract.

## 4. Denylist

Human-only / construct-proximal labels as primary endpoints; event thresholds tuned on outcomes; circular
event–IPV definitions (events derived from the IPV being tested); observed PET online.

## 5. Method / work packages

- W0: run the frozen automatic event extractor on `clean_285` replay (extractor-health checks: computability,
  precedence suppression, identity stability, sampling-rate sensitivity).
- W1: align M3 deviation to automatic events and to official collisions/deductions.
- W2: test deviation → event/harm association with incremental regression over kinematic+safety and the same
  negative-control battery as RQ011B.

## 6. Endpoints + PASS/FAIL

Supported = deviation→event/harm association that is IPV-specific (beats negative controls) and survives the
kinematic+safety baseline. Otherwise bounded/null. Automatic event counts are extractor evidence, never
scientific outcomes on their own.

## 7. Deliverables, stop conditions, claim boundaries

Deliverables: event-aligned harm result (or null); `decision.md`. Claims: automatic-event + objective-outcome
based; no human-judgment convergent claim from OnSite. Feeds RQ013.

## 8. Dependencies

Upstream: RQ012 (frozen, revised), RQ011 (frozen), RQ009 (M3). Feeds RQ013.
