# Route-D late metric residual adapter interface result — 2026-07-17

## 1. Scope

This is a zero-step interface audit on authorized Kubric fit video 0. It
initializes the exact formal P0g CMCP and P0h comparator checkpoints and inserts
a zero-initialized rank-32 Late Metric Residual Adapter (LMRA) in the
proposal/evidence feature branch. It is not learned LMRA performance.

No model-validation, calibration, final-holdout, DAVIS, or Kinetics data are
read.

## 2. Adapter contract

```text
input/output feature channels: 128
adapter rank:                  32
trainable LMRA parameters:     8,352
insertion: after fnet.conv3, before correlation normalization
up projection:                 zero initialized
```

The adapter computes a residual through `128 -> 32 -> 128` 1x1 projections.
Correlation construction performs the final normalization. At zero
initialization, the adapter preserves the cached feature map byte-for-byte,
including signed-zero bit patterns.

```text
frozen and adapted feature-map SHA-256:
c00a882aa360692953fe8a809dae1df9d0feec1e41cadad20bc06bc708b0978b
```

## 3. Initialized joint system

```text
formal CMCP state SHA-256:
fc3044eb6daa1fb2416164fc0afe58ff5fa4eb4442811ce9c0b4917fbbdb57c0

formal comparator state SHA-256:
46ba80fbb217fd760e009d3bb5a03d43c55c6e3a8dca58e3fb247a00f51e1bcd

trainable parameters:
LMRA:        8,352
CMCP:      263,747
comparator:409,224
total:     681,323
```

The original CoTracker native trajectory, visibility, confidence, candidate 0,
updateformer, and fnet parameters remain frozen.

## 4. Real zero-step equivalence audit

On fit video 0 (`64` points, `24` frames), the online LMRA path is compared with
the frozen P0h token-cache path. All checks are exact:

- adapted feature maps equal frozen feature maps byte-for-byte;
- native plus five candidate coordinates equal the frozen P0h cache;
- raw 88-D candidate tokens equal the frozen cache;
- online four-field causal decision summaries produce the same decisions;
- selected candidate indices and selected coordinates equal formal P0h.

Key hashes:

```text
candidate-coordinate SHA-256:
c4cc84161d6522eb889c48c80cb5fb54673e4f446f54c697079ab5e4a9979300

candidate-token SHA-256:
82dfbabca0d8cede98c82c781e48a61a5017ccfd23d1dd029fbc2109de95064b

selected-coordinate SHA-256:
46d63f0fc9218f54221d075e209bd3fada3a76ad448d7d5fd584a3841716a8e4
```

The full online reconstruction is repeated twice in-process and in two
independent processes. Every versioned tensor hash is identical.

## 5. Decision

```text
ALLOW_LMRA_FIT_ONLY_JOINT_TRAINING
```

P0i-b may train LMRA, the initialized CMCP proposal generator, and the
initialized P0h comparator on fit only. The native tracker branch stays frozen.
No rank, insertion layer, threshold, NMS, EMA, or top-K sweep is authorized.
