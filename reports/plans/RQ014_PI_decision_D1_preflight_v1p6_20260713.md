# RQ014 PI scoped decision — v1.6 D1 contract-preflight authorization loop

Date: 2026-07-13. Source: user/PI decision captured via the interactive prompt in the Lead session.

## D1 decision record

The Lead session recorded these D1 facts:

- Export: `ACCEPTED`.
- Preflight authorization loop (§9): `AUTHORIZED TO START` (`scoped decision → allowlist commit → candidate manifest → fresh dual review → new Formal G1 → final bundle → publish/sync → immutable spec → validate-only`).
- The Lead's binding promise from the selected option is one more explicit user confirmation **before** the preflight submit.

The accepted evidence is the `PASS_RATING_BLIND_EXPORT_READY_FOR_PI_DECISION` bounded report at
`reports/studies/RQ014_wod_e2e_rating_recovery/RQ014_1_declassification_export_20260712T165224Z_0e6ca130/`.

## Exactly one operation

This decision authorizes starting the §8.1 authority-change loop for exactly one operation:

```text
rq014_g2_contract_preflight
```

It authorizes adding only `rq014_g2_contract_preflight` to the canonical central allowlist alongside the already allowed
`rq014_g2_declassification_export`. It does not authorize adding any other operation.

## This is not machine authorization to submit

The central machine conclusion for contract preflight remains `DENY` until the complete §8.1 chain is satisfied:

1. this versioned scoped decision and the single-operation central allowlist change;
2. one candidate manifest binding the entrypoint, tests, schemas, contract, decision, and authority bytes;
3. fresh statistics and fresh execution/governance reviews of that same manifest, by distinct reviewers, both
   `NO_BLOCKER` with no unresolved major;
4. a new Formal G1 and final checksum bundle;
5. publication of the exact commit, safe synchronization, an immutable preflight run spec, and validate-only PASS;
6. a fresh validator finding no drift; and
7. the promised explicit user confirmation before the single preflight submit.

## Explicitly not authorized

This decision does not authorize the resource pilot, G2R rating-blind feature build, G3R rating join/rank screen, G4R
clean replay, or any access to ratings, preferences, ranks, rating-bearing inputs, or rating-derived statistics. Those
operations remain centrally denied and retain their own later scoped-decision and §8.1 authorization loops.
