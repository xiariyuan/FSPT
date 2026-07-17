# Route-D CMCP proposal-only training result — 2026-07-17

## 1. Scope and claim boundary

This is the formal seed-17 CMCP proposal-only result on the frozen Kubric
fit/model-validation protocol. Gradients use only the 48 fit videos; checkpoint
selection and gates use the complete 16-video model-validation split.
Calibration, final holdout, DAVIS, and the official 1,144-sample Kinetics
protocol were not read or rerun.

The result is a **partial mechanism success and formal gate failure**. CMCP
learns a substantially stronger candidate generator, but its direct
native-vs-proposal decision is not statistically consistent or safe enough.

## 2. Corrected formal implementation

The preregistered nine-channel recurrent input contains:

```text
three causal correlation fields:                 3
all three pairwise correlation differences:      3
native-centered motion prior:                    1
two previous proposal-evidence maps:             2
---------------------------------------------------
total recurrent input channels:                  9
```

Formal model:

```text
hidden channels:       64
trainable parameters:  263,747
proposal candidates:   native + five learned peaks
NMS radius:            one feature cell
EMA alpha:             0.9
best epoch:            1
epochs executed:       5
seed:                  17
```

Zero-step native parity is exact. The zero-step model is included as checkpoint
`epoch -1`, so training can never force selection of a model worse than the
native-safe initialization.

## 3. Exact reproducibility

Independent seed-17 runs reproduce the complete history, best epoch, final
validation metrics, gates, and model state exactly.

```text
model-state SHA-256:
fc3044eb6daa1fb2416164fc0afe58ff5fa4eb4442811ce9c0b4917fbbdb57c0

primary checkpoint SHA-256:
08df7e01583addb4ed3bf2ddd86e87f04b806a3df0fb2f6e0ed38dbaf821aee3

primary metrics SHA-256:
45a096f5eac9ec12b058944ce43cb31b37ced613bc1eb7ceafe854628dec4096

replay checkpoint SHA-256:
08df7e01583addb4ed3bf2ddd86e87f04b806a3df0fb2f6e0ed38dbaf821aee3

replay metrics SHA-256:
44c12e79012d2a60e1d3f47548e5c74a87c317c8ccb82189a69b2890125667cc
```

The primary and replay checkpoint files are byte-identical.

## 4. Candidate-pool mechanism result

| Metric | Native | CMCP coordinate oracle | Gain |
|---|---:|---:|---:|
| AJ | 25.0135 | 45.0510 | **+20.0375 points** |
| Delta average | 37.1739 | 59.8686 | **+22.6947 points** |

Paired-video oracle AJ gain:

```text
mean:  +19.8941 points
95% CI: [+18.0040, +21.7391]
```

All 16 videos have positive oracle AJ gain:

```text
minimum: +10.8301
median:  +20.3092
mean:    +19.8941
maximum: +26.2119
```

The learned proposal pool therefore passes the candidate-generation gate by a
large margin. Candidate quality is not the current bottleneck.

## 5. Direct proposal top-1 result

| Metric | Native | Direct CMCP top-1 | Gain |
|---|---:|---:|---:|
| AJ | 25.0135 | 26.5978 | **+1.5843 points** |
| Delta average | 37.1739 | 39.0604 | **+1.8865 points** |

Paired-video direct AJ gain:

```text
mean:  +1.1788 points
95% CI: [-2.5666, +4.5731]
positive videos: 11 / 16
negative videos: 5 / 16
```

Threshold gains expose the trade-off:

| Threshold | Direct top-1 gain | Oracle gain |
|---|---:|---:|
| 1px | -5.5392 | +4.6672 |
| 2px | -5.7357 | +17.6492 |
| 4px | +1.3510 | +37.2451 |
| 8px | +11.9749 | +34.1685 |
| 16px | +7.3815 | +19.7433 |

CMCP strongly reduces large errors but frequently abandons already accurate
native coordinates, reducing 1px and 2px precision.

## 6. Safety behavior

```text
selected non-native rate:          90.1253%
harmful non-native rate:           27.9968%
beneficial candidate availability: 57.0621%
beneficial candidate recall:       76.9910%
```

Severe 16px error rate:

```text
native:   25.8782%
direct:   18.5212%
oracle:   6.2208%
```

The direct model selects non-native candidates on about 90% of visible rows and
makes a harmful non-native choice on about 28%. This is incompatible with the
preregistered 1% safety ceiling even though the large-error tail improves.

## 7. Formal gate

| Gate | Result |
|---|---|
| Candidate oracle AJ >= +3.0 | **PASS** |
| Direct top-1 AJ >= +0.5 | **PASS** |
| Paired direct AJ CI lower > 0 | **FAIL** |
| Delta gain > 0 | **PASS** |
| Severe 16px rate not worse | **PASS** |
| Harmful non-native rate <= 1% | **FAIL** |
| Exact seed-17 replay | **PASS** |

Formal decision retained from the frozen runner:

```text
STOP_CMCP_AND_CONSIDER_LATE_BACKBONE_FINETUNING
```

This decision closes P0g-c as a gate failure. It does not authorize MUSR,
state-write training, calibration, final holdout, DAVIS, or Kinetics.

## 8. Scientific diagnosis and next controlled route

The proposal generator should not be discarded: its oracle is strong and
consistent on every validation video. The structurally weak component is the
current native fallback, which globally averages the whole hidden map before
producing one native logit. It cannot compare local native evidence with local
proposal-peak evidence.

The preregistered P0h route freezes the entire best epoch-1 proposal generator,
all proposal coordinates, NMS, EMA, top-K, and dense maps. It trains only a
local pairwise native-vs-candidate safety comparator on fit. No validation-tuned
threshold is allowed. See
`docs/ROUTED_CMCP_LOCAL_PAIRWISE_SAFETY_V0_PLAN_2026-07-17.md`.
