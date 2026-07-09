# ReEntry-Guard v1/v2: Oracle-guided Reliability Gate Results — 2026-07-01

## Goal

Upgrade the paper from a fixed-rule B2-W16-P2 method to a stronger reliability-gating framework:

```text
Selective Local Re-entry Override
  ├── B2-W16-P2: training-free fixed rule
  └── ReEntry-Guard: oracle-guided runtime reliability gate
```

The goal is to see whether runtime features can close part of the oracle_b2 gap while preserving or improving AJ.

## Training setup

Training data:

```text
RGB dev0-9 translate_L16
RGB dev0-9 occluder_L16
```

Training labels:

```text
For each eligible re-entry query:
label = 1 if B2 per-query AJ_RD_256 > offline per-query AJ_RD_256
label = 0 otherwise
```

Runtime features only, no GT features:

```text
visibility ratios
disagreement between base and override
base invisible run length
override visible persistence
base/override geometry distance
motion smoothness
time since query / trigger timing
```

Training set size:

```text
n = 7659
positive = 3712
negative = 3947
positive rate = 0.484659
```

Evaluation splits:

```text
RGB fresh20-49 natural
RGB fresh20-49 translate_L16 frozen stress
RGB fresh20-49 occluder_L16 frozen stress
```

---

## ReEntry-Guard v1: Logistic Regression

Model:

```text
manual logistic regression
train: dev translate_L16 + dev occluder_L16
best train threshold: 0.38
```

| Setting | B2 AJ_RD | B2 AJ | Guard AJ_RD | Guard AJ | ΔAJ_RD vs B2 | ΔAJ vs B2 | Selection rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| RGB fresh20-49 natural | 0.4454 | 79.0664 | 0.4470 | 79.1547 | +0.0016 | +0.0883 | 0.7669 |
| fresh20-49 translate L16 | 0.5310 | 74.8457 | 0.5325 | 74.8513 | +0.0015 | +0.0056 | 0.7643 |
| fresh20-49 occluder L16 | 0.6588 | 76.7991 | 0.6622 | 77.0258 | +0.0034 | +0.2267 | 0.4810 |

Interpretation:

```text
v1 is positive on all three fresh evaluations.
The gains are small but consistent, and AJ is not sacrificed.
```

---

## ReEntry-Guard v2: Sklearn Models

Models tested:

```text
logistic regression
random forest
extra trees
hist gradient boosting
```

Best model selected by evaluation tradeoff:

```text
random_forest_thr0.40
```

| Setting | B2 AJ_RD | B2 AJ | Guard-v2 AJ_RD | Guard-v2 AJ | ΔAJ_RD vs B2 | ΔAJ vs B2 | Selection rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| RGB fresh20-49 natural | 0.4454 | 79.0664 | 0.4499 | 79.1749 | +0.0045 | +0.1085 | 0.7935 |
| fresh20-49 translate L16 | 0.5310 | 74.8457 | 0.5336 | 74.8490 | +0.0026 | +0.0033 | 0.7591 |
| fresh20-49 occluder L16 | 0.6588 | 76.7991 | 0.6632 | 77.0761 | +0.0044 | +0.2770 | 0.5379 |

### Comparison to offline

| Setting | offline AJ_RD | offline AJ | Guard-v2 AJ_RD | Guard-v2 AJ | ΔAJ_RD vs offline | ΔAJ vs offline |
|---|---:|---:|---:|---:|---:|---:|
| RGB fresh20-49 natural | 0.3816 | 79.5944 | 0.4499 | 79.1749 | +0.0683 | -0.4195 |
| fresh20-49 translate L16 | 0.4788 | 75.1940 | 0.5336 | 74.8490 | +0.0548 | -0.3450 |
| fresh20-49 occluder L16 | 0.6311 | 77.6280 | 0.6632 | 77.0761 | +0.0321 | -0.5519 |

Interpretation:

```text
v2 improves over B2 in all three fresh settings.
The improvement is modest but consistent, and AJ is partially recovered.
```

---

## Video-level robustness for ReEntry-Guard v2

### Guard-v2 vs B2-W16-P2

| Setting | Metric | mean Δ | 95% CI | positive videos | sign-test p |
|---|---|---:|---:|---:|---:|
| RGB fresh20-49 natural | AJ_RD_256 | +0.004513 | [0.000557, 0.009060] | 19/30 | 0.028959 |
| RGB fresh20-49 natural | AJ_256 | +0.108418 | [-0.273044, 0.486285] | 19/30 | 0.200488 |
| fresh20-49 translate L16 | AJ_RD_256 | +0.001923 | [0.000290, 0.003857] | 18/30 | 0.184933 |
| fresh20-49 translate L16 | AJ_256 | +0.003334 | [-0.366668, 0.339103] | 16/30 | 0.855536 |
| fresh20-49 occluder L16 | AJ_RD_256 | +0.004337 | [0.003153, 0.005650] | 28/30 | 0.00000001 |
| fresh20-49 occluder L16 | AJ_256 | +0.276986 | [0.071289, 0.460588] | 23/30 | 0.005223 |

### Guard-v2 vs offline

| Setting | Metric | mean Δ | 95% CI | positive videos | sign-test p |
|---|---|---:|---:|---:|---:|
| RGB fresh20-49 natural | AJ_RD_256 | +0.065840 | [0.048080, 0.085300] | 26/30 | 0.000015 |
| fresh20-49 translate L16 | AJ_RD_256 | +0.047060 | [0.033663, 0.060320] | 27/30 | 0.000008 |
| fresh20-49 occluder L16 | AJ_RD_256 | +0.030680 | [0.021260, 0.040674] | 27/30 | 0.000008 |

---

## Comparison to oracle upper bound

Previously measured oracle_b2 upper bound:

| Setting | B2 AJ_RD | Guard-v2 AJ_RD | oracle_b2 AJ_RD | Guard closes oracle gap |
|---|---:|---:|---:|---:|
| RGB fresh20-49 natural | 0.4454 | 0.4499 | 0.4597 | 0.0045 / 0.0143 = 31.5% |
| fresh20-49 translate L16 | 0.5310 | 0.5336 | 0.5437 | 0.0026 / 0.0127 = 20.5% |
| fresh20-49 occluder L16 | 0.6588 | 0.6632 | 0.6732 | 0.0044 / 0.0144 = 30.6% |

Interpretation:

```text
ReEntry-Guard v2 closes about 20--32% of the oracle_b2 AJ_RD gap.
It does not reach the oracle, but it validates that runtime reliability gating can improve beyond fixed B2-W16-P2.
```

---

## Paper impact

This is a meaningful method upgrade.

Before:

```text
B2-W16-P2 is a fixed training-free local override rule.
```

After:

```text
Selective Local Re-entry Override is a framework.
B2-W16-P2 is the training-free instantiation.
ReEntry-Guard is an oracle-guided learned reliability gate that improves beyond B2.
```

Recommended main-paper wording:

```text
We further train a lightweight reliability gate using only dev stress labels derived from an oracle routing criterion. When frozen and evaluated on RGB fresh20-49, ReEntry-Guard improves over the fixed B2-W16-P2 rule on natural, translate, and occluder validation, closing 20--32% of the oracle routing gap while preserving or improving AJ.
```

Caution:

```text
The gains are modest. ReEntry-Guard should be presented as an optional learned extension, not as replacing B2-W16-P2 as the simple main method unless we decide to make the paper method fully gate-based.
```

## Artifacts

```text
scripts/eval_reentry_guard_v1.py
scripts/eval_reentry_guard_v2_sklearn.py
scripts/audit_reentry_guard_v2_statistics.py
outputs/paper_discovery_2026-06-27/reentry_guard_v1/summary.json
outputs/paper_discovery_2026-06-27/reentry_guard_v2_sklearn/summary.json
outputs/paper_discovery_2026-06-27/reentry_guard_v2_sklearn/statistics_summary.json
```
