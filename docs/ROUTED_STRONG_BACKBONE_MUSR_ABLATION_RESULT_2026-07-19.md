# Route-D strong-backbone MUSR component ablation result — 2026-07-19

## 1. Formal decision

The preregistered P0j A/B/C/D component matrix is complete on the frozen Kubric
fit/model-validation protocol. Independent seed-17 replays for B and C reproduce
the original checkpoints byte-for-byte and reproduce the complete structured
metrics payload apart from output-path fields.

The preregistered rule states that the architecture claim must be revised if B or
C equals or exceeds the full jointly trained variant D. Variant C not only
exceeds D; its paired 16-video advantage has a strictly positive confidence
interval. The formal decision is:

```text
REVISE_TO_FROZEN_CMCP_LMRA_COMPARATOR
```

The recommended strong-backbone model is:

```text
frozen formal P0g CMCP proposal core
+ trainable rank-32 LMRA metric adapter
+ trainable local safety comparator
+ frozen native CoTracker3 trajectory
```

Jointly updating CMCP with LMRA is not part of the revised main architecture.

## 2. Claim and data boundary

All gradients use only the 48-video Kubric fit partition. Checkpoint selection,
component comparisons, and confidence intervals use the complete 16-video model-
validation partition. The following remain unread and locked:

```text
calibration
final synthetic holdout
TAP-Vid-DAVIS
official TAP-Vid-Kinetics 1,144-video result
```

The completed official Kinetics result was not rerun or used to choose any P0j
variant.

## 3. Frozen component matrix

| Variant | Trainable components | AJ gain | Delta-average gain | Harmful non-native | Beneficial recall | Formal gate |
|---|---|---:|---:|---:|---:|:---:|
| A — P0h reference | comparator only | 0.2449 | 0.4962 | 0.3132% | 3.3362% | FAIL |
| B — identity LMRA | CMCP + comparator | 0.6365 | 0.9187 | 0.8843% | 7.8705% | PASS |
| C — frozen CMCP | LMRA + comparator | 0.8861 | 1.4530 | 0.9396% | 9.5711% | PASS |
| D — full joint P0i | LMRA + CMCP + comparator | 0.5945 | 0.8745 | 0.8229% | 7.4796% | PASS |

Variant C is the best safety-feasible result:

```text
AJ gain:                       +0.8861 points
delta-average gain:            +1.4530 points
paired-video AJ mean:          +0.8541 points
paired-video AJ 95% CI:        [+0.6489, +1.0694]
positive / negative videos:    16 / 0
harmful non-native rate:       0.9396%
beneficial-candidate recall:   9.5711%
severe-16px rate change:       -1.0194 percentage points
candidate-oracle AJ gain:      +20.0484 points
```

The epoch-2 C checkpoint reaches a larger raw AJ gain but violates the frozen 1%
harmful-selection ceiling. Safety-first selection therefore retains epoch 3.

## 4. Paired component attribution

| Comparison | Mean AJ difference | 95% paired-video CI | Positive videos |
|---|---:|---:|---:|
| B − A | 0.3869 | [0.2476, 0.5529] | 16 / 16 |
| C − A | 0.6120 | [0.4639, 0.7583] | 16 / 16 |
| C − B | 0.2251 | [0.0861, 0.3655] | 13 / 16 |
| D − B | -0.0399 | [-0.0856, 0.0081] | 5 / 16 |
| D − C | -0.2650 | [-0.4168, -0.1151] | 4 / 16 |

Interpretation:

1. Updating CMCP with the comparator is useful relative to A.
2. Updating LMRA with the comparator while freezing CMCP is more useful and is
   positive on all 16 videos relative to A.
3. C significantly exceeds B, so metric adaptation contributes beyond CMCP
   retraining.
4. D significantly underperforms C. Simultaneous CMCP and LMRA updates create a
   negative proposal–metric interaction rather than an additive gain.

This is not a post-hoc hyperparameter choice. C and B were preregistered scientific
ablations, and the plan explicitly required revising the architecture claim when
an ablation met or exceeded D.

## 5. Reproducibility

```text
B primary/replay checkpoint SHA-256:
ac76e16e896c030a48105acaae618d1eba782f3dc6e4b41c0c89e6b53f8505c6

B combined model-state SHA-256:
8d7043fc6c9b8ccc0b964485a11990bc00e0bc03a04713aa3b8277abf8e13bf3

C primary/replay checkpoint SHA-256:
7babb76e3407497832d0bc0fca4557df64b2d09450ebc60766e32a70816ab52f

C combined model-state SHA-256:
64c3f6dae6ae34dc0223754aca3c76f78a1f48137d08bab6c3e01f2a806cd074
```

For both B and C:

- checkpoint files are byte-identical;
- best epoch is identical;
- complete epoch histories are identical;
- final per-video and aggregate metrics are identical;
- gate decisions are identical;
- frozen-component state hashes remain exact.

The formal generated summary is:

```text
docs/generated/ROUTED_STRONG_BACKBONE_MUSR_ABLATION_SUMMARY_2026-07-19.json
SHA-256: 52bc0921f1815c66922db263004c81df696acbb608ff73f3051d1907cac34cb6
```

## 6. Scientific implication

The strongest result does not require continuous co-adaptation of proposal
coordinates. The formal P0g CMCP core already provides a strong, universal
candidate pool. The remaining transferable improvement comes from adapting the
feature metric used by the proposal/evidence branch while training a conservative
native-versus-candidate comparator.

The revised contribution is therefore more precise:

> A frozen learned multi-memory proposal core can be made safely actionable by a
> late residual metric adapter and a local risk-aware comparator; jointly moving
> the proposal core and metric is unnecessary and measurably harmful under the
> frozen strong-backbone protocol.

## 7. Next authorized step

One separately preregistered coordinate-only bounded closed-loop writeback
ablation may now compare against variant C on fit/model-validation only. It must:

```text
preserve exact candidate-0/native fallback
initialize exactly as output-only variant C
modify only overlap coordinates inherited by the next streaming window
use one frozen intervention-strength parameterization
use fit gradients only
pass a stricter no-regression comparison against C
```

Calibration, final holdout, DAVIS, and official Kinetics remain locked until that
separate gate passes. No layer, rank, threshold, NMS, EMA, top-K, write strength,
or horizon sweep is authorized on model validation.
