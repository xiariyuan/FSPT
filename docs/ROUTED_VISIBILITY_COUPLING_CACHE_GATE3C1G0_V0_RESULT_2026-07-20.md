# Route-D Visibility-Coupling Cache Gate 3C1G0 v0 Result — 2026-07-20

## Decision

```text
COMPLETED_PASS
AUTHORIZE_GATE3C1G1_NESTED_VIDEO_OOF_PREREGISTRATION
```

The formally preregistered richer causal cache completed on all 44 exposed Gate 3C1F2 action videos. Every source video first reproduced the sealed entry, candidate, shortlist, action, coordinate, and visibility decisions before any new feature row was emitted.

## Frozen support

```text
videos:       44
actions:      89
frame rows:   801
feature dim:  66
frames/action: 15--23 inclusive
```

The 66-D feature tensor contains only causal trajectory, visibility-confidence, DINO identity, four-level CoTracker identity/memory consistency, and sealed entry/top-1 evidence. GT visibility, coordinate errors, threshold-hit labels, action categories, and future-derived labels remain separate evaluation tensors and are not feature channels.

## Integrity

```text
cache index file SHA256:
5c690647e8fee4ae812e24f132622f3941fa12682b5e000f7ce08851d2493b45

cache index payload SHA256:
a990040bcd8a86808451c8e77c6bed3b38ec2032522b9ef3bd1372f08dc1ea72

combined sidecar digest:
c6edcef339c5deaa81a235a4f634aeb45e2c5e1878bc1e0b18fae9dfc7e8cf82

combined tensor digest:
8b7a61cb111ec3bb76d54161558d5c24c2e89cfbb636cb30488c088b7e7ab464
```

Independent reload verified every sidecar file SHA, sidecar payload, tensor hash, tensor digest, 66-D shape, source identity, sealed decision check, and final index payload.

## Claim boundary

This is an exposed-development cache result. It authorizes only a separately committed nested source-video OOF model-selection protocol. It does not authorize threshold tuning on a new population, a raw-record-disjoint confirmation, DAVIS, Kinetics, final holdout, or official Kinetics 1,144.
