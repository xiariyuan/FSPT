# ReEntry-VisCalibrator V2.3 RiskScore Conservative Sweep — 2026-07-03

## 1. Goal

The first V23-RiskScore policy:

```text
event001_or_dist025:
  drop if event_gate_prob < 0.01 OR frame_base_override_dist_norm >= 0.25
```

improved AJ beyond V22Q on fresh20-49, but paid a small AJ_RD/OA cost. This sweep tested more conservative variants to recover most of the AJ gain while reducing AJ_RD/OA cost.

---

## 2. New policies added

Added to:

```text
scripts/eval_reentry_viscalibrator_v23_risk_score.py
```

Policies:

```text
event001_or_dist050:
  drop if event_gate_prob < 0.01 OR frame_base_override_dist_norm >= 0.50

event001_or_dist025_and_v1lt050:
  drop if event_gate_prob < 0.01 OR (frame_base_override_dist_norm >= 0.25 AND v1_prob < 0.50)

event001_or_dist025_and_segmaxlt050:
  drop if event_gate_prob < 0.01 OR (frame_base_override_dist_norm >= 0.25 AND segment_prob_max < 0.50)

event001_or_dist025_and_v1lt050_and_segmaxlt050:
  drop if event_gate_prob < 0.01 OR (frame_base_override_dist_norm >= 0.25 AND v1_prob < 0.50 AND segment_prob_max < 0.50)
```

Script passed py_compile after the update.

---

## 3. Dev7-9 conservative sweep

Reference:

```text
V1 dev7-9:    AJ_RD 0.3213, AJ 79.7517, OA 94.0018
V2.1 dev7-9:  AJ_RD 0.3216, AJ 79.9493, OA 94.0191
V22Q dev7-9:  AJ_RD 0.3212, AJ 79.9763, OA 94.0033
Aggressive RiskScore event001_or_dist025:
  AJ_RD 0.3215, AJ 79.9917, OA 94.0135
```

Conservative policies:

| Policy | AJ_RD | AJ | OA | Keep rate | Dropped changed frames |
|---|---:|---:|---:|---:|---:|
| event001_or_dist050 | 0.3218 | 79.9708 | 94.0020 | 0.9229 | 3551 |
| event001_or_dist025_and_v1lt050 | 0.3215 | 79.9863 | 94.0101 | 0.9174 | 3804 |
| event001_or_dist025_and_segmaxlt050 | 0.3215 | 79.9782 | 94.0000 | 0.9198 | 3692 |
| event001_or_dist025_and_v1lt050_and_segmaxlt050 | 0.3215 | 79.9782 | 94.0000 | 0.9198 | 3692 |

Selected conservative candidate for fresh sanity:

```text
event001_or_dist025_and_v1lt050
```

Reason:

```text
It satisfies AJ_RD >= 0.3212 and OA >= 94.00, while having the highest AJ among conservative policies except the aggressive policy.
```

---

## 4. Fresh20-49 conservative policy result

Policy:

```text
event001_or_dist025_and_v1lt050:
  drop if event_gate_prob < 0.01 OR (frame_base_override_dist_norm >= 0.25 AND v1_prob < 0.50)
```

### Query-weighted result

| Setting | V22Q AJ_RD | Conservative AJ_RD | ΔAJ_RD | V22Q AJ | Conservative AJ | ΔAJ | V22Q OA | Conservative OA | ΔOA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| natural | 0.4520 | 0.4520 | +0.0000 | 79.2775 | 79.2787 | +0.0012 | 92.9233 | 92.9013 | -0.0220 |
| translate_L16 | 0.5350 | 0.5349 | -0.0001 | 74.9919 | 74.9990 | +0.0071 | 92.1943 | 92.1720 | -0.0223 |
| occluder_L16 | 0.6669 | 0.6669 | +0.0000 | 77.2218 | 77.2302 | +0.0084 | 92.2326 | 92.2230 | -0.0096 |

### Conservative vs aggressive RiskScore

| Setting | ΔAJ_RD | ΔAJ | ΔOA |
|---|---:|---:|---:|
| natural | +0.0006 | -0.0210 | -0.0063 |
| translate_L16 | +0.0004 | -0.0314 | +0.0058 |
| occluder_L16 | +0.0005 | -0.0234 | -0.0092 |

---

## 5. Interpretation

The conservative policy did what it was supposed to do in part:

```text
It recovers most of the AJ_RD lost by the aggressive policy.
```

However, it does not improve the overall stability tradeoff versus V22Q:

```text
AJ gain over V22Q is tiny: +0.0012 / +0.0071 / +0.0084.
OA is consistently lower than V22Q.
```

Therefore:

```text
Conservative RiskScore is not worth promoting as a new method.
```

The earlier aggressive policy remains a useful diagnostic because it showed interpretable signals can raise AJ, but the conservative sweep shows that simple threshold/rule policies cannot simultaneously keep V22Q-level OA and produce meaningful additional AJ.

---

## 6. Current conclusion

Best current method positioning remains:

```text
V1 Learned = AJ_RD-oriented main method.
V22Q = best current stability-oriented extension.
V23-RiskScore = useful diagnostic, not final replacement.
```

RiskScore has reached a clear tradeoff boundary:

```text
Aggressive RiskScore: more AJ, but small AJ_RD/OA cost.
Conservative RiskScore: preserves AJ_RD, but AJ gain over V22Q nearly disappears and OA still drops.
```

This suggests that the current numeric risk signals are useful but insufficient to close the oracle gap.

---

## 7. Recommended next step

Do not continue hand-tuning RiskScore policies.

The next meaningful direction is to add a new information source:

```text
appearance verifier / patch-consistency signal
```

Rationale:

```text
V2.3 safe16/gt-visible oracle is very strong.
Numeric features and transparent rules recover only a small part of that oracle gap.
The missing decision likely requires checking whether the candidate point still matches the previous reliable appearance.
```

Recommended first appearance-verifier step:

```text
Build a lightweight RGB patch-similarity diagnostic for V1 proposed recovery frames:
  - last reliable base-visible patch around the point
  - candidate recovery patch around base coordinate
  - simple similarity features: NCC / L2 / mean-color distance / gradient magnitude difference
  - analyze AUC for safe16 / gt_visible
```

If patch similarity adds meaningful AUC over numeric features, train a small frame-keep model with numeric + appearance features.
If it does not, move to DINO / external tracker teacher.
