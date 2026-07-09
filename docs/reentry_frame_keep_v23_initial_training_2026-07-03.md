# ReEntry-VisCalibrator V2.3 Frame-Keep Initial Training — 2026-07-03

## 1. Goal

V2.3 tries to learn a second-stage frame-keep / risk decoder on top of V1 proposed recovery frames.

Unlike V1, it does not propose new visible frames. It only decides whether to keep or drop a frame that V1 already proposed for recovery.

---

## 2. Code added / restored

```text
utils/reentry_frame_keep_features.py
scripts/build_reentry_frame_keep_dataset.py
scripts/train_reentry_frame_keep_decoder.py
scripts/eval_reentry_viscalibrator_v23_frame_keep.py
scripts/eval_reentry_viscalibrator_v23_oracle_frame_keep.py
```

Important note:

```text
During one patch attempt, two files were accidentally overwritten with patch fragments:
  utils/reentry_frame_keep_features.py
  scripts/build_reentry_frame_keep_dataset.py
Both were fully rewritten and restored.
```

All current scripts passed py_compile.

---

## 3. V2.3 label-complete datasets

New datasets with complete labels:

```text
outputs/paper_discovery_2026-06-27/reentry_frame_keep_v23/datasets/rgb_dev0_6_frame_keep_v1thr010_labels_v2.npz
outputs/paper_discovery_2026-06-27/reentry_frame_keep_v23/datasets/rgb_dev7_9_frame_keep_v1thr010_labels_v2.npz
```

### dev0-6 train

```text
n_samples = 95925
feature_dim = 49
positive_rate_gt_visible = 0.7235
positive_rate_safe16 = 0.6761
positive_rate_utility = 0.4061
positive_rate_safe8 = 0.5560
positive_rate_safe4 = 0.4061
mean_utility = -0.1694
candidate_windows = 8666
candidate_frames = 129422
raw_recovery_frames = 125489
changed_recovery_frames = 95925
```

### dev7-9 validation

```text
n_samples = 46041
feature_dim = 49
positive_rate_gt_visible = 0.8209
positive_rate_safe16 = 0.7710
positive_rate_utility = 0.3312
positive_rate_safe8 = 0.6080
positive_rate_safe4 = 0.3312
mean_utility = -0.2429
candidate_windows = 3441
candidate_frames = 53092
raw_recovery_frames = 50427
changed_recovery_frames = 46041
```

Important distribution observation:

```text
safe16 / gt_visible positive rates are higher on dev7-9 than dev0-6.
utility / safe4 positive rates are lower on dev7-9.
This suggests split-level distribution shift.
```

---

## 4. Oracle upper-bound on dev7-9

Baseline V1 on dev7-9:

```text
AJ_RD = 0.3213
AJ    = 79.7517
OA    = 94.0018
```

Oracle results:

| Oracle mode | AJ_RD | AJ | OA | Keep rate over raw changed |
|---|---:|---:|---:|---:|
| v1_keep_all | 0.3213 | 79.7517 | 94.0018 | 1.0000 |
| gt_visible | 0.3478 | 80.3801 | 94.8082 | 0.8209 |
| safe16 | 0.3479 | 80.5493 | 94.5941 | 0.7710 |
| safe8 | 0.3429 | 80.8431 | 93.8484 | 0.6080 |
| safe4 / utility | 0.2805 | 80.8451 | 92.6328 | 0.3312 |
| safe2 | 0.2395 | 80.5340 | 91.7953 | 0.1446 |

Interpretation:

```text
V2.3 has a high theoretical upper bound.
The best first label target is safe16 or gt_visible, not utility/safe4.
Utility/safe4 is too conservative and destroys AJ_RD.
```

---

## 5. Initial trained models

Trained models:

```text
outputs/paper_discovery_2026-06-27/reentry_frame_keep_v23/models/reentry_frame_keep_v23_safe16_dev0_6.pt
outputs/paper_discovery_2026-06-27/reentry_frame_keep_v23/models/reentry_frame_keep_v23_gtvisible_dev0_6.pt
outputs/paper_discovery_2026-06-27/reentry_frame_keep_v23/models/reentry_frame_keep_v23_safe16_weighted_dev0_6.pt
```

Model:

```text
MLP, input_dim=49, hidden=128, dropout=0.1
```

Training finding:

```text
For safe16 and gt_visible, the best validation sample-accuracy checkpoint was epoch 1.
Later epochs overfit or become poorly calibrated on dev7-9.
```

This is a warning sign:

```text
Sample-level accuracy is not a sufficient selection metric.
Need downstream AJ_RD/AJ selection or checkpoint snapshots.
```

---

## 6. Dev7-9 downstream threshold sweep: unweighted safe16

| keep threshold | AJ_RD | AJ | OA | Keep rate | Kept frames |
|---:|---:|---:|---:|---:|---:|
| 0.001 | 0.3213 | 79.7517 | 94.0018 | 1.0000 | 46041 |
| 0.005 | 0.3213 | 79.7517 | 94.0018 | 1.0000 | 46041 |
| 0.010 | 0.3213 | 79.7517 | 94.0018 | 1.0000 | 46041 |
| 0.020 | 0.3213 | 79.7517 | 94.0018 | 1.0000 | 46041 |
| 0.050 | 0.3213 | 79.7517 | 94.0018 | 1.0000 | 46041 |
| 0.100 | 0.3210 | 79.7513 | 94.0001 | 0.9996 | 46023 |
| 0.200 | 0.3210 | 79.8065 | 93.9673 | 0.9823 | 45225 |
| 0.300 | 0.3214 | 79.8638 | 93.9660 | 0.9634 | 44357 |
| 0.500 | 0.3035 | 80.2273 | 93.3607 | 0.7013 | 32289 |

Best constrained observation:

```text
threshold 0.30 preserves AJ_RD and improves AJ to 79.8638.
Still below V2.1/V22Q AJ on dev7-9.
```

---

## 7. Dev7-9 downstream threshold sweep: gt_visible

| keep threshold | AJ_RD | AJ | OA | Keep rate | Kept frames |
|---:|---:|---:|---:|---:|---:|
| 0.001--0.300 | 0.3213 | 79.7517 | 94.0018 | 1.0000 | 46041 |
| 0.500 | 0.3195 | 79.8721 | 93.9957 | 0.9369 | 43138 |

Interpretation:

```text
gt_visible model is too close to keeping everything at low/mid thresholds.
It does not learn a useful enough risk filter in this first training setup.
```

---

## 8. Dev7-9 downstream threshold sweep: utility-weighted safe16

| keep threshold | AJ_RD | AJ | OA | Keep rate | Kept frames |
|---:|---:|---:|---:|---:|---:|
| 0.001 | 0.3213 | 79.7517 | 94.0018 | 1.0000 | 46041 |
| 0.005 | 0.3213 | 79.7517 | 94.0018 | 1.0000 | 46041 |
| 0.010 | 0.3209 | 79.7509 | 93.9994 | 0.9994 | 46015 |
| 0.020 | 0.3208 | 79.7545 | 94.0029 | 0.9982 | 45958 |
| 0.050 | 0.3210 | 79.7670 | 94.0161 | 0.9945 | 45789 |
| 0.100 | 0.3213 | 79.8458 | 93.9594 | 0.9693 | 44627 |
| 0.150 | 0.3207 | 79.8888 | 93.9497 | 0.9500 | 43739 |
| 0.200 | 0.3185 | 79.9531 | 93.9084 | 0.9128 | 42024 |
| 0.300 | 0.2958 | 80.2014 | 93.3736 | 0.7007 | 32260 |
| 0.500 | 0.2652 | 80.4180 | 92.6948 | 0.4428 | 20385 |

Best constrained observation:

```text
threshold 0.10 preserves AJ_RD = 0.3213 and improves AJ to 79.8458.
Still below V2.1/V22Q dev AJ.
```

---

## 9. Current conclusion

The V2.3 oracle is very strong, but the first learned MLP does not approach it.

Current learned frame-keep models are weaker than V2.1/V22Q on dev7-9:

```text
V1 dev7-9:    AJ_RD 0.3213, AJ 79.7517, OA 94.0018
V2.1 dev7-9:  AJ_RD 0.3216, AJ 79.9493, OA 94.0191
V22Q dev7-9:  AJ_RD 0.3212, AJ 79.9763, OA 94.0033
V2.3 safe16-weighted best constrained: AJ_RD 0.3213, AJ 79.8458, OA 93.9594
V2.3 safe16 unweighted best constrained: AJ_RD 0.3214, AJ 79.8638, OA 93.9660
```

Interpretation:

```text
The V2.3 task has high upper bound, but the current model/training setup is insufficient.
The issue is not the idea; it is the learning setup.
```

Likely causes:

```text
1. Train/val distribution shift between dev0-6 and dev7-9.
2. Sample-level training objective does not align with downstream AJ_RD/AJ.
3. Epoch selection by sample accuracy is poor.
4. Features may not separate hard false-visible frames from true re-entry frames enough.
5. Current MLP lacks temporal context beyond handcrafted segment stats.
```

---

## 10. Recommended next step

Do not run fresh with the current V2.3 MLP because it is not competitive on dev7-9.

Next step should be diagnostic, not more fresh testing:

```text
V2.3 diagnostic: feature separability / oracle gap analysis.
```

Specifically:

```text
1. Analyze safe16-positive vs safe16-negative distributions for key features:
   - v1_prob
   - segment position
   - segment length
   - event_gate_prob
   - base_override_dist
   - speed/acceleration
   - border distance

2. Compute simple AUC / AP for each individual feature.

3. Train a simpler calibrated model:
   - logistic regression or shallow MLP
   - class-balanced BCE
   - maybe one epoch / early stopping by downstream proxy

4. If feature separability is weak, move to appearance verifier / external teacher.
```

Practical next code target:

```text
scripts/analyze_reentry_frame_keep_features.py
```

This should report feature AUCs and reveal whether the 49-dimensional numeric features are sufficient.
