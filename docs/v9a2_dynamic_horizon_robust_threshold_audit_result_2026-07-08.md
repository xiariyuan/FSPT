# V9-A2.3c-R Robust Threshold and Stability Audit Result

Date: 2026-07-08

---

## 1. Purpose

V9-A2.3c initially found a learned dynamic-horizon policy that improved over W16 accept-all, but the best row came from a broad trajectory-level threshold sweep.

This robust audit uses a much smaller and more conservative threshold set:

```text
- OOF-F1 thresholds from classifier validation;
- fixed thresholds such as 0.01, 0.02, 0.05, 0.30, 0.36, 0.50;
- mostly event_max mode, with a few event_mean / frame controls.
```

The goal is to check whether the improvement is not just an artifact of large threshold sweeping.

---

## 2. Artifacts

Script:

```text
scripts/v9a2_dynamic_horizon_robust_audit.py
```

Output:

```text
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_dynamic_horizon_robust_threshold_audit.json
```

Input:

```text
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_dynamic_horizon_controller_report.json
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_joint_w8_common_plus_w16_extension_v3.npz
```

---

## 3. Main baselines

| Variant | AJ Δ | OA Δ | AJ_RD_256 Δ | Accepted | Ext accepted | Ext good precision | Ext good recall | Pos/Neg/Zero |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| W8 preserve common only | +0.0964 | +0.9462 | +0.0274 | 1456 | 0 | - | 0.000 | 16/4/5 |
| W16 accept all joint | +0.0810 | +0.9701 | +0.0286 | 2013 | 557 | 0.212 | 1.000 | 15/5/5 |
| Oracle ext candidate_good | +0.1955 | +1.0204 | +0.0321 | 1574 | 118 | 1.000 | 1.000 | 17/3/5 |
| Oracle ext good_not_worse | +0.1966 | +1.0106 | +0.0321 | 1570 | 114 | 1.000 | 0.966 | 17/3/5 |

---

## 4. Best robust learned rows

| Variant | AJ Δ | OA Δ | AJ_RD_256 Δ | Accepted | Ext accepted | Ext good precision | Ext good recall | Pos/Neg/Zero |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| all_logreg_event_max_fixed_0.05 | +0.1025 | +0.9711 | +0.0294 | 1937 | 481 | 0.227 | 0.924 | 17/3/5 |
| all_logreg_event_mean_fixed_0.005 | +0.0976 | +0.9746 | +0.0293 | 1977 | 521 | 0.225 | 0.992 | 17/3/5 |
| all_logreg_event_max_oof_f1 | +0.0975 | +0.9746 | +0.0293 | 1971 | 515 | 0.225 | 0.983 | 17/3/5 |
| all_logreg_event_max_fixed_0.02 | +0.0970 | +0.9711 | +0.0293 | 1954 | 498 | 0.225 | 0.949 | 17/3/5 |
| anchor_logreg_event_max_fixed_0.01 | +0.0986 | +0.9746 | +0.0293 | 1972 | 516 | 0.229 | 1.000 | 17/3/5 |
| anchor_logreg_event_max_oof_f1 | +0.0979 | +0.9746 | +0.0292 | 1976 | 520 | 0.227 | 1.000 | 17/3/5 |
| anchor_logreg_event_max_fixed_0.02 | +0.0996 | +0.9746 | +0.0291 | 1962 | 506 | 0.227 | 0.975 | 17/3/5 |

---

## 5. Interpretation

The robust audit confirms that the V9-A2.3c improvement is not only a broad-sweep artifact.

A simple fixed threshold works:

```text
all_logreg_event_max_fixed_0.05:
AJ_RD_256 Δ = +0.0294
```

This is above:

```text
W8 accept-all:  +0.0274
W16 accept-all: +0.0286
```

It also improves video stability:

```text
W16 accept-all: 15/5/5 positive/negative/zero videos
V9-A2 robust:  17/3/5 positive/negative/zero videos
```

Therefore, the learned event-level dynamic horizon controller passes a stronger robustness check than the original threshold sweep.

---

## 6. Important caveat

The learned controller improves mostly by accepting many extension rows while dropping some harmful rows:

```text
W16 extension total: 557
fixed_0.05 accepted extension rows: 481
extension good precision: 0.227
extension good recall: 0.924
```

This means the score is not a high-precision selector. It is closer to:

```text
high-recall extension policy with partial risk trimming.
```

So V9-A2.3c is useful, but not yet a strong identity verifier.

---

## 7. Feature-level reflection

The robust winners are logistic regression variants using `all` or `anchor` features with event-level aggregation.

However, previous OOF classifier summaries showed that RGB patch anchor features alone are not very strong. Therefore:

```text
V9-A2.3c validates the dynamic horizon action space.
It does not yet validate RGB patch identity as a strong appearance model.
```

If this line is extended toward a stronger paper contribution, the next identity feature should likely be:

```text
DINO / TrackOn2 internal descriptor / learned candidate memory feature
```

rather than raw RGB patch statistics alone.

---

## 8. Decision

V9-A2.3c-R passes.

The dynamic horizon controller is now stronger than both W8 and W16 under a conservative fixed-threshold audit.

Current best robust method:

```text
all_logreg_event_max_fixed_0.05
```

Current best robust result:

```text
AJ Δ        = +0.1025
OA Δ        = +0.9711
AJ_RD_256 Δ = +0.0294
Pos/Neg/Zero = 17/3/5
```

Next:

```text
V9-A2.4 Paper-Ready Dynamic Horizon Ablation
```

The next step should package:

```text
1. W8 baseline;
2. W16 accept-all;
3. oracle upper bound;
4. V9-A2 learned robust controller;
5. per-video stability;
6. feature ablation: base / anchor / all;
7. action ablation: frame vs event_max vs event_mean.
```
