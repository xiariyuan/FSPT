# Route-D strong-backbone MUSR ablation v0 plan — 2026-07-17

## 1. Purpose

P0i is the first exact-replay strong-backbone result to pass every frozen gate.
Before introducing state writeback, the paper needs evidence that the gain is not
an opaque consequence of one jointly trained block. P0j therefore separates the
contribution of metric adaptation, dense proposal learning, and local safety
selection under one fixed fit/model-validation protocol.

## 2. Frozen evidence and data boundary

The following remain fixed:

- Kubric fit/model-validation identities and all base/native tensors;
- native CoTracker trajectory, visibility, confidence, and candidate 0;
- proposal top-K 5, one-cell NMS, EMA 0.9, seed 17, and safety-first checkpoint rule;
- P0i optimizer budgets and loss definitions;
- calibration, final holdout, DAVIS, and official Kinetics locks.

No layer, rank, learning-rate, threshold, NMS, EMA, top-K, or abstention sweep is
allowed.

## 3. Preregistered component matrix

Use the same initialization and training budget as P0i:

```text
A: P0h reference — frozen metric, frozen P0g CMCP, trained comparator
B: joint CMCP + comparator — LMRA fixed at exact identity
C: LMRA + comparator — CMCP frozen at formal P0g checkpoint
D: full P0i — LMRA + CMCP + comparator (already completed)
```

Variants B and C are scientific ablations, not hyperparameter candidates. Each
uses fit gradients only and complete model-validation checkpoint selection once.

## 4. Non-redundancy gates

For a component to be claimed as useful, its inclusion must improve direct AJ by
at least `+0.10` point over the matched ablation while preserving:

```text
paired-video AJ CI lower bound > 0
harmful non-native rate <= 1%
16px severe rate not worse
exact seed-17 replay
```

The full P0i route must remain the best safety-feasible model. If B or C equals or
exceeds D, revise the architecture claim rather than tuning D.

## 5. Bounded state-write authorization

Only after the component matrix is complete may one bounded state-write ablation
be implemented. It must initialize as coordinate-only P0i, use fit gradients
only, preserve candidate-0/native fallback, and pass a stricter no-regression
comparison against P0i. Calibration and locked datasets remain unavailable until
that separate gate passes.
