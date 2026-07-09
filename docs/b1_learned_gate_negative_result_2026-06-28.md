# B1 Learned Visibility Gate Negative Result — 2026-06-28

## Decision

The direct frame-level learned visibility gate is a negative result and should not replace the current heuristic B1 fusion rule.

Best learned gate found:

```text
name = learned_l8_C1p0_p0p2
true_AJ_RD_256 = 0.5953
```

This is below:

```text
current B1 quick heuristic = 0.6264
current B1 refine heuristic = 0.6279
old 3-teacher gated baseline = 0.6189
```

## What was attempted

Script:

```text
scripts/train_b1_learned_visibility_gate.py
```

The script builds frame-level features from the 4-teacher outputs, trains a grouped logistic regression visibility classifier, and replaces the final predicted visibility with:

```text
pred_visibility = predicted_probability >= threshold
```

It uses GroupKFold over videos and labels frames as positive when:

```text
gt_visible and all_median4_error_256 < label_thr
```

## Best result

From:

```text
outputs/paper_discovery_2026-06-27/teacher_expansion/b1_learned_gate/summary.json
```

| Method | label_thr | C | prob_thr | true_AJ_RD_256 | true_AJ_RD | proxy | dmin1 | dmin4 | dmin16 | delta vs 0.6264 | delta vs 0.6189 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| learned_l8_C1p0_p0p2 | 8 | 1.0 | 0.2 | 0.5953 | 0.4167 | 0.4654 | 0.5917 | 0.5763 | 0.5243 | -0.0311 | -0.0236 |

## Interpretation

The failure is informative:

1. Directly replacing the strong heuristic visibility rule is unsafe.
2. Frame-level visibility classification is not equivalent to maximizing post-reappearance AJ_RD.
3. The current heuristic rules are strong baselines, not weak baselines.
4. Future learned modules must preserve the current best rule by default and override only when predicted gain is high.

## Next learned route

Do not continue this direct replacement approach as the main route.

Recommended next learned route:

```text
safe action/delta router
```

Default action:

```text
B0 = vis4_gated288
true_AJ_RD_256 = 0.6279
```

Training target:

```text
predicted_delta(action) = predicted_score(action) - predicted_score(B0)
```

Inference policy:

```text
use B0 by default
if max(predicted_delta) > margin:
    use predicted best action
else:
    use B0
```

Before training this router, compute candidate-action oracle to verify that the current action set has enough recoverable headroom.
