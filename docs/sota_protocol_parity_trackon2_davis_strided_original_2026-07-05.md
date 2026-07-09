# SOTA Protocol Parity Audit — TrackOn2 DAVIS strided/original — 2026-07-05

## Decision

TrackOn2 DAVIS strided/original should **not** proceed to ReEntry plug-in or 30-video main-table evaluation under the current exporter/evaluator path.

The new exporter successfully reproduces the existing TrackOn2 strided/original behavior, which means the old cache was not simply stale or corrupted. The query-frame geometry is clean, but long-horizon coordinate quality collapses under this protocol. Because ReEntry is coordinate-preserving and only changes visibility, applying ReEntry on top of this TrackOn2 strided/original cache cannot produce a credible SOTA-base main result.

Recommended immediate action:

```text
Stop TrackOn2 strided/original rescue under the current path.
Keep TrackOn2 first/input plug-in evidence as appendix/supplemental.
Move SOTA rescue attention to TAPNext++ acquisition / 1-video smoke, or to a different parity-valid SOTA base.
```

## Artifacts

Exporter added:

```text
scripts/export_trackon2_davis_strided_original_cache.py
```

1-video smoke outputs:

```text
outputs/paper_discovery_2026-07-05/trackon2_strict_parity/trackon2_davis_strided_original_v1_1video.pt
outputs/paper_discovery_2026-07-05/trackon2_strict_parity/trackon2_davis_strided_original_v1_1video_report.json
outputs/paper_discovery_2026-07-05/trackon2_strict_parity/trackon2_davis_strided_original_v1_1video_parity.json
outputs/paper_discovery_2026-07-05/trackon2_strict_parity/trackon2_davis_strided_original_v1_1video_geometry.json
outputs/paper_discovery_2026-07-05/trackon2_strict_parity/trackon2_davis_strided_original_v1_1video_metrics_compare.json
outputs/paper_discovery_2026-07-05/trackon2_strict_parity/trackon2_davis_strided_original_v1_1video_visibility_variants.json
```

Environment checks:

```text
outputs/paper_discovery_2026-06-27/external_baseline_smoke/trackon2_env_check.json
outputs/paper_discovery_2026-06-27/external_baseline_smoke/dino_backbone_check.json
```

## Environment status

Command:

```bash
python scripts/check_trackon2_environment.py
```

Result:

```text
mmcv_ops_ok = true
trackon2_import_ok = true
torch = 2.2.2+cu121
cuda_available = true
```

Local TrackOn2 checkpoints:

```text
baselines/track_on/checkpoints_trackon2_dinov2.pt  # 90M
baselines/track_on/checkpoints_trackon2_dinov3.pt  # 90M
```

DINOv3 local directory used by the new exporter:

```text
third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m
```

Note:

```text
scripts/check_trackon_dino_backbone.py reports no local HF DINOv2 directory, but this does not block the DINOv3 TrackOn2 path. The DINOv3 local directory exists and was used through DINOV3_LOCAL_DIR.
```

## Export command

```bash
python scripts/export_trackon2_davis_strided_original_cache.py \
  --out-cache outputs/paper_discovery_2026-07-05/trackon2_strict_parity/trackon2_davis_strided_original_v1_1video.pt \
  --out-report outputs/paper_discovery_2026-07-05/trackon2_strict_parity/trackon2_davis_strided_original_v1_1video_report.json \
  --out-parity outputs/paper_discovery_2026-07-05/trackon2_strict_parity/trackon2_davis_strided_original_v1_1video_parity.json \
  --max-videos 1 \
  --start-index 0 \
  --query-mode strided \
  --query-stride 5 \
  --input-scale uint8 \
  --memory-policy unconditional
```

## 1-video smoke result

Video:

```text
video_id = bike-packing
frames = 69
queries = 219
size_hw = [480, 910]
```

Export summary:

```text
pred_vis_rate = 0.422937
gt_vis_rate   = 0.726160
anchor_err_px_mean = 0.299290
anchor_err_px_max  = 0.826536
peak_mem_mb = 1013.4
```

Interpretation:

```text
The query-frame anchor is clean. The cache is not broken at the query point.
The predicted visibility is much lower than GT visibility, suggesting conservative visibility, but visibility is not the whole problem.
```

## Geometry audit

Command:

```bash
python scripts/audit_trackon2_cache_geometry.py \
  --caches outputs/paper_discovery_2026-07-05/trackon2_strict_parity/trackon2_davis_strided_original_v1_1video.pt \
  --out-json outputs/paper_discovery_2026-07-05/trackon2_strict_parity/trackon2_davis_strided_original_v1_1video_geometry.json
```

Aggregate:

```text
anchor_err_px_mean      = 0.2993
anchor_err_px_median    = 0.2803
anchor_err_px_p95       = 0.5838
pred_in_range_rate_pm005 = 1.0
pred_vis_rate_mean      = 0.4229
err_px_all_median       = 371.52
err_px_all_p75          = 586.65
err_px_all_p95          = 727.30
err_px_gt_visible_median = 21.90
err_px_gt_visible_p75    = 567.19
err_px_gt_visible_p95    = 735.74
```

Interpretation:

```text
The query frame is aligned, but long-horizon coordinate errors are huge. The GT-visible p75/p95 errors indicate serious trajectory quality collapse after the query frame. This is not a small visibility-threshold issue.
```

## Standard / AJ_RD comparison

Command:

```bash
python scripts/eval_external_baseline_smoke_cache.py \
  --items \
    new_trackon2=outputs/paper_discovery_2026-07-05/trackon2_strict_parity/trackon2_davis_strided_original_v1_1video.pt \
    old_trackon2=caches/trackon2_strided_original.pt \
    cotracker_offline=outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt \
  --max-records 1 \
  --out-json outputs/paper_discovery_2026-07-05/trackon2_strict_parity/trackon2_davis_strided_original_v1_1video_metrics_compare.json
```

Results on first DAVIS video:

| Method | AJ_RD_256 | AJ_RD | First re-entry proxy | AJ_256 | OA_256 | delta_avg_256 | Queries |
|---|---:|---:|---:|---:|---:|---:|---:|
| New TrackOn2 export | 0.4504 | 0.2581 | 0.2677 | 31.0007 | 57.5208 | 38.0565 | 219 |
| Old TrackOn2 cache | 0.4515 | 0.2591 | 0.2613 | 30.9450 | 57.2925 | 38.0305 | 219 |
| CoTracker3 offline | 0.3372 | 0.2012 | 0.2097 | 53.6969 | 86.7983 | 71.7221 | 219 |

Interpretation:

```text
The new exporter reproduces the old TrackOn2 strided/original behavior almost exactly.
Therefore, the old TrackOn2 cache was not merely stale/corrupt.
TrackOn2 has higher AJ_RD_256 than CoTracker3 offline on this video, but standard AJ/OA/delta_avg are far worse.
This is not a useful SOTA-base main-table result.
```

## Visibility oracle variants

Command:

```bash
python scripts/eval_cache_visibility_variants.py \
  --cache outputs/paper_discovery_2026-07-05/trackon2_strict_parity/trackon2_davis_strided_original_v1_1video.pt \
  --out-json outputs/paper_discovery_2026-07-05/trackon2_strict_parity/trackon2_davis_strided_original_v1_1video_visibility_variants.json
```

Results:

| Variant | AJ_RD_256 | First re-entry proxy | AJ_256 | OA_256 | delta_avg_256 |
|---|---:|---:|---:|---:|---:|
| original | 0.4504 | 0.2677 | 31.0007 | 57.5208 | 38.0565 |
| GT visibility | 0.5508 | 0.3774 | 24.1730 | 100.0000 | 38.0565 |
| all visible | 0.4482 | 0.3774 | 19.4135 | 72.2133 | 38.0565 |
| query visible fill | 0.4504 | 0.2677 | 31.0007 | 57.5208 | 38.0565 |

Interpretation:

```text
GT visibility raises AJ_RD_256 by +0.1004, so there is visibility/re-entry diagnostic headroom.
However, AJ_256 drops and delta_avg_256 stays fixed at 38.0565 because coordinate quality is the bottleneck.
A coordinate-preserving ReEntry module cannot fix this.
```

## Why not proceed to ReEntry on this cache

ReEntry's design constraint is:

```text
final coordinates = base coordinates
only visibility is calibrated
```

For this TrackOn2 strided/original cache:

```text
base coordinate quality is too weak under the audited protocol.
GT visibility does not rescue standard AJ because delta_avg is fixed and low.
Therefore, any ReEntry variant that preserves TrackOn2 coordinates would at best improve AJ_RD while leaving the standard-metric collapse unresolved.
```

This is not enough for the SOTA rescue route, whose success criterion requires:

```text
AJ >= SOTA base - 0.05
OA >= SOTA base - 0.05
AJ_RD >= SOTA base + 0.01
paired AJ CI must not be entirely negative
protocol must be parity-valid
```

Here the base itself is not a usable SOTA baseline under strided/original.

## Stop / go decision

### Stop

Stop the current TrackOn2 strided/original path because:

```text
1. New export reproduces old weak cache.
2. Query anchor is clean, so simple query conversion is not the issue.
3. Long-horizon coordinate errors are too large.
4. GT visibility improves AJ_RD but does not rescue standard AJ/delta_avg.
5. ReEntry cannot fix coordinates by design.
```

### Preserve

Preserve the following as appendix/supplemental evidence:

```text
docs/first_input_trackon2_b2w_plugin_experiment_2026-06-29.md
```

This remains valuable because first/input TrackOn2 is parity-valid and ReEntry/B2-W16 gives small positive gains there.

### Next

Move to TAPNext++ or another SOTA base only through the same gates:

```text
1. checkpoint/code acquisition
2. 1-video smoke
3. protocol parity
4. visibility/re-entry oracle headroom
5. plug-in ReEntry only if oracle headroom exists and coordinate quality is usable
```

Do not train V26 yet.

## Claim impact

No change to the current paper claim.

Current paper remains:

```text
diagnostic / workshop-first ReEntry paper
```

The TrackOn2 strided/original result should be used only as a negative parity/audit finding, not as a main comparison.
