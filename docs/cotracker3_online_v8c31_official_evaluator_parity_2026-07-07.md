# CoTracker3 Online V8-C3.1 Official Evaluator Parity Audit

Date: 2026-07-07

## 1. Judgement

```text
Label: TAP-Vid-DAVIS-first metric-compatible reproduction
Metric formula parity pass: True
Query-first pass: True
Local evaluator protocol-close pass: True
Max metric diff: 0.0
```

## 2. Metric parity

The current project wrapper and the local CoTracker backup official evaluator give the same AJ/OA/delta metrics on the native cache and CVRRM default cache. This confirms metric formula parity for the current DAVIS records.

### native

```text
standard_and_ajrd: {'AJ': 65.23658673675799, 'OA': 90.81858393645052, 'delta_avg': 77.94582106551431, 'delta_4px': 85.86700981252287}
project wrapper: {'AJ': 65.23658673675799, 'OA': 90.81858393645052, 'delta_avg': 77.94582106551431, 'delta_4px': 85.86700981252287}
backup official: {'AJ': 65.23658673675799, 'OA': 90.81858393645052, 'delta_avg': 77.94582106551431, 'delta_4px': 85.86700981252287}
diff project vs backup: {'AJ': 0.0, 'OA': 0.0, 'delta_avg': 0.0, 'delta_4px': 0.0}
```

### CVRRM_default_w8

```text
standard_and_ajrd: {'AJ': 65.32061366864887, 'OA': 91.74276067873016, 'delta_avg': 78.34289460565951, 'delta_4px': 86.37548077230663}
project wrapper: {'AJ': 65.32061366864887, 'OA': 91.74276067873016, 'delta_avg': 78.34289460565951, 'delta_4px': 86.37548077230663}
backup official: {'AJ': 65.32061366864887, 'OA': 91.74276067873016, 'delta_avg': 78.34289460565951, 'delta_4px': 86.37548077230663}
diff project vs backup: {'AJ': 0.0, 'OA': 0.0, 'delta_avg': 0.0, 'delta_4px': 0.0}
```

## 3. Query protocol

```text
{
  "total_queries_with_visible": 650,
  "bad_query_time_count": 0,
  "bad_query_position_count": 0,
  "max_query_position_err_px": 0.0,
  "examples": []
}
```

## 4. Local evaluator protocol checks

```text
{
  "evaluator": {
    "path": "/gemini/code/FSPT/baselines/cotracker/cotracker/cotracker_eval_backup/core/evaluator.py",
    "contains_queried_first_true": false,
    "contains_resize_256_default": false,
    "contains_add_support_grid_false": true,
    "contains_grid_size_zero": true,
    "contains_query_yx_to_xy_stack": true
  },
  "dataset": {
    "path": "/gemini/code/FSPT/baselines/cotracker/cotracker/datasets/tap_vid_datasets.py",
    "contains_queried_first_true": true,
    "contains_resize_256_default": true,
    "contains_add_support_grid_false": false,
    "contains_grid_size_zero": false,
    "contains_query_yx_to_xy_stack": false
  },
  "export": {
    "path": "/gemini/code/FSPT/scripts/export_cotracker3_online_v7a4_raw_visconf_components.py",
    "contains_queried_first_true": true,
    "contains_resize_256_default": true,
    "contains_add_support_grid_false": true,
    "contains_grid_size_zero": true,
    "contains_query_yx_to_xy_stack": false
  }
}
```

## 5. Interpretation

```text
The DAVIS standard metrics can be described as TAP-Vid-DAVIS-first metric-compatible under the local CoTracker evaluator formula.
This still should not be overclaimed as full original-paper Table-1 parity across Kinetics/RGB-Stacking or all checkpoint/support settings.
AJ_RD / AJ_RD_256 remain re-entry-focused supplementary metrics computed under the same records/baselines.
```

## 6. Next step

```text
Proceed to V8-C4 final paper-style packaging with two tables: standard TAP-Vid-like metrics and re-entry recovery metrics.
Optional later: run Kinetics/RGB-Stacking if claiming broad benchmark generalization.
```
