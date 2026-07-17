# Route-D CMCP pairwise token-cache result — 2026-07-17

## Scope

This milestone freezes local candidate evidence for P0h comparator-only training.
It is not learned comparator performance and does not read calibration, final
holdout, DAVIS, or Kinetics.

## Cache contract

Each native-plus-five-candidate token is 88-D:

```text
static generator/local evidence: 84
online causal decision summary:   4
```

The cache stores the 84 static fields plus four zero placeholders. During
comparator training, only the final four fields are replaced by the preceding
causal comparator decision. The frozen CMCP generator is never rerun inside an
optimizer step and receives no gradients.

## Completed partitions

| Partition | Videos | Status | Candidate-coordinate combined SHA-256 |
|---|---:|---|---|
| fit | 48 / 48 | complete | `3f1be12649a8c4757c933c3f6d8c99b9f3d6bcd146262406557e139377ca57c2` |
| model-validation | 16 / 16 | complete | `edf8ebe25cc11b04406efe27203a7d2bea4da8ce075160a27a11d8edda1fb0a1` |

The complete cache occupies approximately 207 MB.

## Integrity

- generator model state is fixed at
  `fc3044eb6daa1fb2416164fc0afe58ff5fa4eb4442811ce9c0b4917fbbdb57c0`;
- candidate 0 equals native for every sidecar;
- static token construction uses no GT;
- candidate coordinates are not modified;
- the dynamic summary placeholder is exactly zero;
- replay anchors 0, 47, 48, and 63 are bit-identical;
- fit index 0 reproduces the P0h interface candidate hash
  `c4cc84161d6522eb889c48c80cb5fb54673e4f446f54c697079ab5e4a9979300`.

## Decision

```text
ALLOW_PAIRWISE_COMPARATOR_ONLY_TRAINING
```

Only the local comparator may now receive fit gradients. Candidate generation,
NMS, EMA, top-K, backbone, MUSR state writeback, calibration, final holdout,
DAVIS, and Kinetics remain locked.
