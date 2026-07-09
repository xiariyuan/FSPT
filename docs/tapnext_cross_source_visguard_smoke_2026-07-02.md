# TAPNext Cross-Source ReEntry-VisGuard Smoke — 2026-07-02

## Purpose

Test whether a non-CoTracker visibility source can improve CoTracker offline/base coordinates under the ReEntry-VisGuard channel-wise framework.

This is a small RGB dev0 smoke using existing caches:

```text
base cache:     outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/cotracker3_offline_rgb_stacking_1video.pt
override cache: outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/tapnext_rgb_stacking_1video.pt
```

Command:

```bash
python scripts/eval_reentry_channel_factorial.py \
  --setting rgb_dev0_tapnext_cross_source \
  --base-cache outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/cotracker3_offline_rgb_stacking_1video.pt \
  --override-cache outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/tapnext_rgb_stacking_1video.pt \
  --out-dir outputs/paper_discovery_2026-06-27/reentry_channel_factorial/rgb_dev0_tapnext_cross_source
```

## TAPNext cache state

TAPNext RGB dev0 cache report:

```text
n_records = 1
n_queries = 1148
mean_vis_rate = 0.1551
export_sec = 22.898
```

The visibility rate is very low for this RGB dev0 cache, suggesting under-visible calibration or protocol mismatch.

## Factorial results

| Method | Coordinates | Visibility | AJ_RD_256 | AJ_256 | OA_256 | ΔAJ_RD vs base | ΔAJ vs base |
|---|---|---|---:|---:|---:|---:|---:|
| base_coord_base_vis | CoTracker offline | CoTracker offline | 0.6169 | 78.8736 | 88.2859 | - | - |
| override_coord_override_vis | TAPNext | TAPNext | 0.0647 | 15.0884 | 30.6603 | -0.5522 | -63.7852 |
| override_coord_base_vis | TAPNext | CoTracker offline | 0.1737 | 12.8401 | 88.2859 | -0.4432 | -66.0335 |
| base_coord_override_vis_global | CoTracker offline | TAPNext global | 0.0732 | 16.9479 | 30.6603 | -0.5437 | -61.9257 |
| ReEntry-VisGuard-W8P2 with TAPNext vis | CoTracker offline | local TAPNext vis | 0.6111 | 78.9388 | 88.3657 | -0.0058 | +0.0652 |

## Conclusion

TAPNext is **not** a useful visibility source in this RGB dev0 smoke under the existing cache / adapter:

```text
local TAPNext visibility does not improve AJ_RD over the CoTracker offline base.
```

This is not a failure of the ReEntry-VisGuard main method. It is an important source-quality result:

```text
ReEntry-VisGuard requires a responsive and reasonably calibrated visibility source. Not every external tracker visibility channel is suitable.
```

## Paper usage

Do not use this as a main negative baseline against TAPNext.

Use as appendix / feasibility evidence only:

```text
A cross-source TAPNext smoke shows that the existing TAPNext RGB cache is heavily under-visible and does not provide useful visibility transfer. This motivates treating external visibility sources with parity and calibration checks before main-table use.
```

## Next step

The best next cross-source target remains LocoTrack, but it requires a local checkpoint:

```text
locotrack_base.ckpt
```

Exporter is ready:

```text
scripts/export_locotrack_rgb_stacking_cache.py
```
