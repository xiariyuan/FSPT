# Strict Official Protocol Completion Plan — 2026-07-04

## 1. What “strict official protocol” means for TAP-Vid

A result should only be described as strict official-protocol local evaluation if it satisfies all items below:

```text
1. Dataset is an official TAP-Vid subset.
2. Query mode is an official TAP-Vid query mode: first or strided.
3. Metrics are computed using the official TAP-Vid metric implementation or an exactly ported equivalent.
4. Reported metrics include standard TAP-Vid metrics: AJ, OA, and δ_avg / average_pts_within_thresh.
5. Resolution is explicitly stated and matches the intended reporting mode, typically 256x256 for common reported checkpoints or original/input resolution if explicitly audited.
6. Aggregation is video-level mean when matching official benchmark reporting.
7. The paper clearly separates official-protocol local evaluation from local bridge / diagnostic evaluation.
8. The paper does not claim official leaderboard submission unless an official server returns scores.
```

Official TAP-Vid provides benchmark data and evaluation metrics. Public pages/repo do not show a clear online prediction-upload leaderboard server. Therefore use:

```text
official-protocol local evaluation
```

not:

```text
official leaderboard submission
```

---

## 2. Current status

### Already close to strict official protocol

```text
TAPVid-DAVIS strided + original local evaluation
```

Completed artifacts:

```text
docs/official_protocol_audit_2026-07-04.md
docs/official_protocol_v25_threshold_sweep_2026-07-04.md
```

This result uses:

```text
Dataset: TAPVid-DAVIS
Query mode: strided
Metric resolution: original
Metrics: AJ / OA / δ_avg / AJ_RD
```

### Not yet strict official protocol

```text
TAPVid RGB-Stacking full50
```

Reason:

```text
It is a full50 local standard evaluation, but still needs explicit official query-mode audit/rescore or rerun under official first/strided query protocol.
```

### Diagnostic only

```text
fresh20-49
translate_L16
occluder_L16
```

These are useful re-entry/stress diagnostics, not official strict protocol results.

---

## 3. What to supplement for strict official protocol credibility

### A. Add an Evaluation Protocol table to the paper

Columns:

```text
Dataset | Videos | Query mode | Resolution | Metrics | Evaluation type
```

Rows should include:

```text
TAPVid-DAVIS | 30 | strided | original | AJ/OA/δ_avg/AJ_RD | official-protocol local evaluation
TAPVid-DAVIS | 30 | first/query-first | 256 or original | AJ/OA/δ_avg/AJ_RD | to run
TAPVid RGB-Stacking | 50 | first or strided | 256 | AJ/OA/δ_avg/AJ_RD | to audit/rerun
RGB fresh20-49 | 30 | local re-entry split | 256 | AJ/OA/AJ_RD | diagnostic local evaluation
Stress variants | 30 each | synthetic stress | 256 | AJ/OA/AJ_RD | diagnostic stress evaluation
```

### B. Run DAVIS first-query official-style evaluation

Why:

```text
Recent TAP-Vid method pages often report query-first performance.
```

Target:

```text
Dataset: TAPVid-DAVIS
Query mode: first
Metrics: AJ / OA / δ_avg
Add AJ_RD as diagnostic
```

### C. Audit / rerun RGB-Stacking full50 under official query protocol

Target:

```text
Dataset: TAPVid RGB-Stacking
Videos: 50
Query mode: first or strided
Metrics: AJ / OA / δ_avg
Resolution: 256x256, unless explicitly using original/input and stated
```

### D. Keep AJ_RD but label it correctly

AJ_RD should be described as:

```text
re-entry / re-detection diagnostic metric
```

not as an official TAP-Vid leaderboard metric.

---

## 4. Minimal strict-protocol supplement package

If time/compute is limited, the minimum credible supplement is:

```text
1. DAVIS strided+original official-protocol table — already done.
2. DAVIS first/query-first official-style local table — run next.
3. Evaluation Protocol table — write next.
4. Clear statement: no official server submission, official-protocol local evaluation only.
```

This is enough to make the paper protocol-safe.

---

## 5. Stronger strict-protocol supplement package

For stronger paper comparison:

```text
1. DAVIS first and strided.
2. RGB-Stacking full50 first or strided.
3. Report AJ/OA/δ_avg plus AJ_RD.
4. Include V25-safe threshold=0.80 as official-safe variant.
5. Keep bridge/fresh/stress as supplementary diagnostics.
```

---

## 6. Recommended next action

Run:

```text
DAVIS first-query official-style local evaluation
```

Then write:

```text
docs/evaluation_protocol_section_2026-07-04.md
```

Do not claim:

```text
official leaderboard submission
strict full TAP-Vid benchmark result
```

Claim:

```text
official-protocol local evaluation on TAPVid-DAVIS
full50 local evaluation on TAPVid RGB-Stacking
re-entry diagnostic evaluation on fresh/stress variants
```
