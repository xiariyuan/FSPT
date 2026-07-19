# Route-D CMCP LMRA joint-training result — 2026-07-17

## 1. Scope and claim boundary

This is the formal seed-17 P0i result on the frozen Kubric fit/model-validation
protocol. Gradients use only the 48 fit videos. Checkpoint selection and every
formal gate use the complete 16-video model-validation split.

The native CoTracker trajectory, candidate 0, visibility, confidence, query
times, updateformer, and original feature encoder are frozen. The trainable
proposal/evidence branch contains:

```text
rank-32 LMRA:                 8,352 parameters
CMCP proposal generator:   263,747 parameters
local safety comparator:   409,224 parameters
------------------------------------------------
total:                     681,323 parameters
```

Calibration, final holdout, DAVIS, and the official 1,144-video Kinetics
protocol were not read or rerun.

The result **passes every preregistered P0i gate**. It authorizes controlled
strong-backbone MUSR ablation, but does not by itself authorize calibration,
final-holdout evaluation, or external benchmark claims.

## 2. Zero-step formal P0h equality

Before any gradient update, the online LMRA -> CMCP -> comparator path exactly
reproduces the formal P0h result over all 16 model-validation videos:

- native, selected, and oracle TAP metrics;
- paired-video confidence intervals;
- behavior and severe-tail statistics;
- every per-video result;
- candidate-coordinate combined SHA-256;
- exact candidate-0/native parity.

The LMRA is zero-initialized and preserves signed-zero bytes, so the adapted
feature map is byte-identical to the frozen map at initialization.

## 3. Frozen training configuration

```text
seed:                       17
optimizer:                  AdamW
LMRA learning rate:         3e-4
CMCP learning rate:         1e-4
comparator learning rate:   1e-4
weight decay:               1e-4
requested epochs:           8
executed epochs:            6
patience:                   3
point batch:                4
optimizer step:             once per video after point-batch accumulation
checkpoint rule:            safety-feasible, then maximum direct AJ
```

Joint loss weights:

```text
dense CMCP proposal loss:   1.0
local comparator loss:      1.0
feature distortion:         0.1
```

Hard NMS coordinates and local candidate sampling are detached. Query,
previous-native, and EMA memory samples are stop-gradient to retain strict CUDA
determinism. The current dense feature map remains differentiable, so LMRA and
CMCP receive proposal-supervision gradients.

## 4. Training trajectory and safety-first selection

| Epoch | AJ gain | Oracle AJ gain | Harmful rate | Beneficial recall | Safe | Saved |
|---:|---:|---:|---:|---:|:---:|:---:|
| 0 | +0.0243 | +20.1696 | 0.1351% | 0.7121% | yes | no |
| 1 | +0.0331 | +20.4139 | 0.2579% | 1.4045% | yes | no |
| 2 | **+0.5945** | +20.2743 | **0.8229%** | **7.4796%** | **yes** | **yes** |
| 3 | +1.2245 | +20.3056 | 1.3879% | 13.1053% | no | no |
| 4 | +0.4173 | +20.4397 | 0.6694% | 6.1213% | yes | no |
| 5 | +0.4158 | +20.3818 | 0.6387% | 5.9327% | yes | no |

Epoch 3 demonstrates the real recall/safety trade-off: it has the largest raw AJ
improvement but violates the fixed 1% harmful-selection ceiling. The
safety-first rule therefore correctly retains epoch 2 rather than reporting the
unsafe peak.

## 5. Exact reproducibility

Independent seed-17 runs reproduce all six epochs, best epoch, initialization,
normalization, optimizer metadata, final metrics, gates, and all three component
states exactly.

```text
combined model-state SHA-256:
bff4bb67f7b60538cd818ab0ee5d0c5a41af065f8611d44c53fdc44023066ddb

LMRA state SHA-256:
6804546725ea4b206dcabcff8a9b8cdd0b2d9a98fbdea2b40ab18a3be0a5a8eb

CMCP state SHA-256:
cfaf672938fc66c1e645755dd70fbcd637eebfed32aa8321ce6cd601e05da11c

comparator state SHA-256:
bc97bd76581f2979545695a7562f184827a44a11e312a9621f56dca417fdecaf

primary checkpoint SHA-256:
ada17585b9ef25272d30e5ae7eb4a6761e1318f7d765f8aa43913331dffdb610

replay checkpoint SHA-256:
ada17585b9ef25272d30e5ae7eb4a6761e1318f7d765f8aa43913331dffdb610
```

The checkpoint files are byte-identical. During replay monitoring, an accidental
second replay process briefly opened the same log path and was terminated before
checkpoint creation. The shared `run.log` is therefore excluded from evidence;
the independent byte-identical checkpoint, structured metrics, complete history,
and model-state hashes are the reproducibility evidence.

## 6. Best complete model-validation result

| Metric | Native | P0i LMRA route | Gain |
|---|---:|---:|---:|
| AJ | 25.0135 | 25.6080 | **+0.5945 points** |
| Delta average | 37.1739 | 38.0484 | **+0.8745 points** |

Paired-video AJ gain:

```text
mean:   +0.5890 points
95% CI: [+0.3910, +0.8171]
positive videos: 16 / 16
negative videos: 0 / 16
```

Threshold gains:

| Threshold | Selected gain | Oracle gain |
|---|---:|---:|
| 1px | -0.2886 | +4.3724 |
| 2px | +0.0246 | +16.5070 |
| 4px | +1.3449 | +38.2093 |
| 8px | +2.2046 | +35.0774 |
| 16px | +1.0870 | +19.9889 |

The method improves medium and large-error thresholds strongly while retaining a
small 1px regression. This boundary must remain explicit in paper claims.

## 7. Safety and candidate behavior

```text
selected non-native rate:          5.4778%
harmful non-native rate:           0.8229%
beneficial candidate availability: 53.5311%
beneficial candidate recall:       7.4796%
```

Compared with P0h, beneficial recall rises from `3.3362%` to `7.4796%` while
remaining below the fixed 1% harmful-selection ceiling.

Severe 16px error rate:

```text
native:   25.8782%
selected: 24.7912%
oracle:    5.9690%
change:   -1.0870 percentage points
```

The learned candidate pool remains strong:

```text
oracle AJ gain:            +20.2743 points
oracle delta-average gain: +22.8310 points
candidate-coordinate combined SHA-256:
a1349c40d342b9da9d8030ddefd3c099a7ff8f7c48de4a512375261e22ca6645
```

## 8. Formal gate

| Gate | Result |
|---|---|
| Zero-step equality to formal P0h | **PASS** |
| Native candidate parity | **PASS** |
| Fixed rank/layer identity | **PASS** |
| Candidate oracle AJ >= +3.0 | **PASS** — +20.2743 |
| Direct AJ >= +0.5 | **PASS** — +0.5945 |
| Paired AJ CI lower > 0 | **PASS** — +0.3910 |
| Delta gain > 0 | **PASS** — +0.8745 |
| Severe 16px rate not worse | **PASS** |
| Harmful non-native rate <= 1% | **PASS** — 0.8229% |
| Exact seed-17 replay | **PASS** |

Formal decision:

```text
ALLOW_LMRA_STRONG_BACKBONE_ROUTE_AND_REOPEN_MUSR_ABLATION
```

## 9. Authorized next step

P0j may run a preregistered fit/model-validation-only strong-backbone MUSR
ablation. It must separate the contribution of LMRA, joint proposal learning,
and local safety selection before testing bounded state writeback.

The following remain locked:

- calibration;
- final holdout;
- DAVIS;
- official Kinetics 1,144-video rerun;
- validation-selected layer, rank, threshold, NMS, EMA, or top-K sweeps.

## 10. Post-P0j claim revision — 2026-07-19

The preregistered P0j component matrix supersedes the interpretation that full
joint LMRA+CMCP+comparator training is the preferred architecture. Variant C,
which freezes the formal P0g CMCP core and trains LMRA plus the comparator,
reaches `+0.8861` AJ versus `+0.5945` for full P0i. The paired C-minus-D
advantage is `+0.2650` AJ with 95% CI `[+0.1151,+0.4168]`.

Formal revision:

```text
REVISE_TO_FROZEN_CMCP_LMRA_COMPARATOR
```

P0i remains a valid passing result and a required ablation, but it is no longer
the recommended final strong-backbone configuration. See
`docs/ROUTED_STRONG_BACKBONE_MUSR_ABLATION_RESULT_2026-07-19.md`.
