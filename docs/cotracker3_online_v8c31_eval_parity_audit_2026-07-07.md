# CoTracker3 Online V8-C3.1 Evaluator Parity Audit

Date: 2026-07-07

Script:

```text
scripts/audit_v8c31_eval_parity.py
```

Report:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v8c31_eval_parity/v8c31_eval_parity_report.json
```

---

## 1. Purpose

This audit checks whether the current DAVIS standard metrics are compatible with the local CoTracker/TAP-Vid evaluator metric implementation.

It does **not** claim full original-paper reproduction across every dataset. It checks metric and query-mode parity for the cached DAVIS records used in the CVRRM experiments.

---

## 2. Metric parity result

Compared metric paths:

```text
Project wrapper:
datasets.metrics.compute_tapvid_metrics

CoTracker backup official evaluator:
cotracker.cotracker_eval_backup.core.eval_utils.compute_tapvid_metrics
```

Variants checked:

```text
native CoTracker3 online
TrackOn2 standalone
CVRRM + TrackOn2, dist<=64, W=8
```

Result:

```text
max_abs_diff_all_variants = 0.0
metric_parity_pass = true
```

Per-variant differences:

```text
native:             AJ/OA/delta_avg/delta_4px diff = 0.0
TrackOn2:           AJ/OA/delta_avg/delta_4px diff = 0.0
CVRRM dist<=64 w8:  AJ/OA/delta_avg/delta_4px diff = 0.0
```

Native standard metrics are exactly reproduced by both paths:

```text
AJ        65.2365867368
OA        90.8185839365
delta_avg 77.9458210655
delta_4px 85.8670098125
```

Conclusion:

```text
The current standard DAVIS metrics are TAP-Vid metric-compatible at the evaluator-function level.
```

---

## 3. Query-first audit

Audited 650 query points from the cached DAVIS records.

Result:

```text
n_queries = 650
bad_query_not_visible = 0
bad_prior_visible_before_query = 0
query_audit_pass = true
```

Conclusion:

```text
The cached records satisfy first-visible query semantics.
```

---

## 4. What this proves

This proves:

```text
1. Our standard metrics AJ / OA / delta_avg / delta_4px match the local CoTracker official-style evaluator on the same cached DAVIS records.
2. The cached query points satisfy first-query semantics.
3. CVRRM's standard metric gains are not due to a custom metric implementation mismatch.
```

Therefore it is safe to describe the standard table as:

```text
TAP-Vid-DAVIS-first metric-compatible on the cached true-streaming DAVIS reproduction.
```

---

## 5. What this does not prove

This does **not** yet prove:

```text
1. Full reproduction of the original CoTracker3 paper Table 1 across all datasets.
2. Equivalence to a one-query-at-a-time plus support-points protocol if the paper used that exact setting.
3. Kinetics / RGB-Stacking / RoboTAP / Dynamic Replica performance.
4. That CVRRM is a single-model CoTracker3 improvement.
```

CVRRM + TrackOn2 remains a two-source output-level recovery system:

```text
CoTracker3 online native + TrackOn2 candidate + CVRRM policy
```

---

## 6. Position of AJ_RD / AJ_RD_256

AJ_RD and AJ_RD_256 remain re-entry-focused failure-mode metrics.

They are valid as supplementary metrics because:

```text
1. All baselines are evaluated with the same definition.
2. They are not used as GT-oracle selectors for the main CVRRM policy.
3. They directly measure the failure mode the method targets.
```

But they should not be presented as original CoTracker3 main-table metrics.

Correct framing:

```text
Standard TAP-Vid-like metrics show that overall tracking does not collapse and slightly improves.
AJ_RD / AJ_RD_256 show that the improvement is concentrated in re-entry recovery.
```

---

## 7. Decision

Use dual tables:

```text
Table A: standard metrics
AJ / OA / delta_avg / delta_4px

Table B: re-entry recovery metrics
AJ_RD / AJ_RD_256 / repair-damage / robustness
```

Main method remains:

```text
CVRRM + TrackOn2 bridge, dist<=64, W=8
```

Its standard metric deltas are evaluator-parity safe on the cached DAVIS reproduction:

```text
AJ        +0.0964
OA        +0.9462
delta_avg +0.3971
delta_4px +0.5085
```

Its re-entry deltas remain:

```text
AJ_RD     +0.0153
AJ_RD_256 +0.0274
```

---

## 8. Next step

Next step depends on paper ambition.

If targeting a re-entry-focused paper/report:

```text
Proceed to V8-C4 final writing package with protocol caveats.
```

If targeting direct original CoTracker3 Table-1-style claims:

```text
Run a separate paper-protocol reproduction audit:
- exact checkpoint mapping
- exact official evaluator config
- support-grid / one-query settings
- Kinetics / RGB-Stacking expansion
```

Recommended immediate next step:

```text
V8-C4 final paper-style package for DAVIS dual-table result,
while marking broader official-dataset expansion as next-stage validation.
```
