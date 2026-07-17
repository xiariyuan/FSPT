# Route-D CMCP local pairwise safety training result — 2026-07-17

## 1. Scope and claim boundary

This is the formal seed-17 comparator-only result on the frozen Kubric
fit/model-validation protocol. Gradients use only the 48 fit videos; checkpoint
selection and gates use the complete 16-video model-validation split. The
formal P0g epoch-1 CMCP generator, all proposal coordinates, NMS, EMA, top-K,
and the native trajectory are frozen.

Calibration, final holdout, DAVIS, and the official 1,144-sample Kinetics
protocol were not read or rerun.

The result is **safe, statistically positive, and below the preregistered
magnitude gate**. It is not a formal pass and does not reopen MUSR or state
writeback.

## 2. Frozen candidate contract

```text
P0g generator model-state SHA-256:
fc3044eb6daa1fb2416164fc0afe58ff5fa4eb4442811ce9c0b4917fbbdb57c0

fit candidate-coordinate combined SHA-256:
3f1be12649a8c4757c933c3f6d8c99b9f3d6bcd146262406557e139377ca57c2

model-validation candidate-coordinate combined SHA-256:
edf8ebe25cc11b04406efe27203a7d2bea4da8ce075160a27a11d8edda1fb0a1
```

The candidate pool and its coordinate oracle are exactly unchanged from P0g-c:

```text
oracle AJ gain:            +20.0375 points
oracle delta-average gain: +22.6947 points
```

## 3. Training and checkpoint rule

```text
comparator parameters:       409,224
local token dimension:       88
fit-only static normalization: first 84 dimensions
online causal summary:       final 4 dimensions
optimizer:                   AdamW
learning rate:               3e-4
requested epochs:            10
executed epochs:             8
patience:                    4
best epoch:                  3
```

Checkpoint selection is safety-first. A checkpoint is eligible only when:

```text
harmful non-native rate <= 1%
16px severe-error rate is not worse than native
```

Among eligible checkpoints, direct AJ is maximized. Exact native-safe epoch
`-1` remains a legal baseline.

## 4. Exact reproducibility

Independent seed-17 runs reproduce normalization, optimizer metadata, all eight
epochs, best epoch, candidate hashes, final metrics, gates, and model state
exactly.

```text
model-state SHA-256:
46ba80fbb217fd760e009d3bb5a03d43c55c6e3a8dca58e3fb247a00f51e1bcd

primary checkpoint SHA-256:
97ec32f064b5fef30ed50043035971de6bb4f4bad07055e2f5def615efb58825

replay checkpoint SHA-256:
97ec32f064b5fef30ed50043035971de6bb4f4bad07055e2f5def615efb58825

primary metrics SHA-256:
9db660e5a14c6a841224a9dad42402f4c12f72806f37be866d08b4e9f5c7c6ae

replay metrics SHA-256:
47a96edc36f5bbf85a44bc8541b67c96d507c926e2beabdb37928980f25d2eb8
```

The primary and replay checkpoint files are byte-identical.

## 5. Best complete model-validation result

| Metric | Native | P0h comparator | Gain |
|---|---:|---:|---:|
| AJ | 25.0135 | 25.2584 | **+0.2449 points** |
| Delta average | 37.1739 | 37.6701 | **+0.4962 points** |

Paired-video AJ gain:

```text
mean:   +0.2420 points
95% CI: [+0.1324, +0.3693]
positive videos: 15 / 16
negative videos: 1 / 16
```

Threshold gains:

| Threshold | Gain |
|---|---:|
| 1px | -0.0553 points |
| 2px | +0.1167 points |
| 4px | +0.6509 points |
| 8px | +1.1361 points |
| 16px | +0.6325 points |

The only pooled threshold regression is a very small 1px decrease. Gains become
larger from 4px onward.

## 6. Safety and behavior

```text
selected non-native rate:          2.5669%
harmful non-native rate:           0.3132%
beneficial candidate availability: 57.0621%
beneficial candidate recall:       3.3362%
```

Severe 16px error rate:

```text
native:   25.8782%
selected: 25.2518%
oracle:   6.2208%
change:   -0.6264 percentage points
```

Relative to direct P0g top-1, harmful selection falls from approximately 28% to
0.3132%. The remaining limitation is under-recall: only 3.3362% of rows with a
beneficial frozen candidate are safely selected.

## 7. Formal gate

| Gate | Result |
|---|---|
| Candidate-coordinate hash exact | **PASS** |
| Candidate oracle AJ >= +3.0 | **PASS** |
| Direct AJ >= +0.5 | **FAIL** — +0.2449 |
| Paired AJ CI lower > 0 | **PASS** |
| Delta gain > 0 | **PASS** |
| Severe 16px rate not worse | **PASS** |
| Harmful non-native rate <= 1% | **PASS** |
| Exact seed-17 replay | **PASS** |

Formal decision:

```text
STOP_PAIRWISE_COMPARATOR_AND_CONSIDER_LATE_BACKBONE_FINETUNING
```

P0h therefore closes as a positive but insufficient-magnitude result. No
threshold, abstention, loss-weight, candidate, NMS, EMA, or top-K sweep is
permitted on model validation.

## 8. Next controlled route

P0i introduces one zero-initialized rank-32 late feature metric residual adapter
after the frozen CoTracker `fnet.conv3` output. The original native trajectory
branch remains frozen; the adapter affects only proposal correlations and local
safety evidence. It contains 8,352 parameters and is jointly trained with the
initialized CMCP and comparator on fit only.

See `docs/ROUTED_CMCP_LATE_METRIC_ADAPTER_V0_PLAN_2026-07-17.md`.
