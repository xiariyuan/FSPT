# ReEntry-TAP Reproducibility Notes

This document lists the current reproducibility flow for the ReEntry-TAP experiments.

## 1. Natural RGB validation

Existing artifacts:

```text
outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_aggregate/summary.json
outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_aggregate/offline_rgb_stacking_fresh20_49.pt
outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_aggregate/online_rgb_stacking_fresh20_49.pt
outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_aggregate/b2_w16_p2_rgb_stacking_fresh20_49.pt
```

## 2. Build controlled stress datasets on RGB dev0-9

Translate stress:

```bash
python scripts/build_reentry_stress_rgb_dev10.py \
  --length 16 --amplitude-frac 0.4 --t0 40 --ramp 8 \
  --fill-mode frame_mean \
  --out-dir outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/translate_L16
```

Moving-occluder stress:

```bash
python scripts/build_reentry_occluder_rgb_dev10.py \
  --length 16 --t0 40 --speed-px 4.0 --fill-mode frame_mean \
  --out-dir outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/occluder_L16
```

## 3. Run CoTracker3 on stress datasets

```bash
python scripts/export_cotracker_reentry_stress_cache.py \
  --stress-dataset <stress_dataset.pt> \
  --model offline \
  --out-cache <offline_cache.pt> \
  --out-report <offline_report.json> \
  --query-batch-size 128

python scripts/export_cotracker_reentry_stress_cache.py \
  --stress-dataset <stress_dataset.pt> \
  --model online \
  --out-cache <online_cache.pt> \
  --out-report <online_report.json> \
  --query-batch-size 128
```

## 4. Evaluate severity curves

```bash
python scripts/eval_reentry_translate_severity.py
python scripts/eval_reentry_occluder_severity.py
```

Outputs:

```text
outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/translate_severity_summary.json
outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/occluder_severity_summary.json
```

## 5. Event provenance and stress-induced-only AJ_RD

```bash
python scripts/audit_reentry_stress_event_provenance.py \
  --families translate,occluder \
  --lengths 8,16,32
```

Output:

```text
outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/event_provenance_combined_summary.json
```

## 6. Strictness supplements

```bash
python scripts/eval_reentry_intersection_severity.py
python scripts/eval_reentry_stress_ablation_l16.py
python scripts/audit_reentry_stress_statistical_robustness.py
```

Outputs:

```text
outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/intersection_query_severity_summary.json
outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/stress_ablation_L16_summary.json
outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/stress_statistical_robustness_summary.json
```

## 7. Frozen RGB fresh20-29 validation

Build stress datasets from the frozen natural fresh20-29 cache:

```bash
python scripts/build_reentry_stress_rgb_from_cache.py \
  --source-cache outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_29/cotracker3_offline_rgb_stacking_fresh20_29.pt \
  --out-dir outputs/paper_discovery_2026-06-27/reentry_stress_rgb_fresh20_29/translate_L16 \
  --stress-type translate_exit_reenter \
  --start-index 20 --num-videos 10 \
  --length 16 --t0 40 --ramp 8 --amplitude-frac 0.4 --fill-mode frame_mean

python scripts/build_reentry_stress_rgb_from_cache.py \
  --source-cache outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_29/cotracker3_offline_rgb_stacking_fresh20_29.pt \
  --out-dir outputs/paper_discovery_2026-06-27/reentry_stress_rgb_fresh20_29/occluder_L16 \
  --stress-type moving_occluder \
  --start-index 20 --num-videos 10 \
  --length 16 --t0 40 --speed-px 4.0 --fill-mode frame_mean
```

Evaluate:

```bash
python scripts/eval_reentry_fresh20_29_frozen_validation.py
python scripts/audit_reentry_fresh20_29_statistical_robustness.py
```

## 8. Trigger taxonomy

```bash
python scripts/audit_reentry_stress_trigger_taxonomy.py
```

Outputs:

```text
outputs/paper_discovery_2026-06-27/reentry_stress_rgb_dev10/trigger_taxonomy_stress_summary.json
outputs/paper_discovery_2026-06-27/reentry_stress_rgb_fresh20_29/trigger_taxonomy_stress_summary.json
```

## 9. Generate paper tables and figures

```bash
python scripts/make_reentry_tap_paper_tables_and_figures.py
```

Outputs:

```text
docs/reentry_tap_paper_tables_2026-07-01.md
docs/reentry_tap_figures_manifest_2026-07-01.md
docs/figures/reentry_tap/*.png
```

## Main caution

Do not interpret ReEntry-TAP as an official leaderboard replacement. It is a controlled stress-test protocol. Internal comparisons within each stress variant are strictly comparable because all methods share the same videos, queries, GT tracks, visibility labels, and evaluator.
