# Next Step After SOTA Plan Review — 2026-07-05

## Decision

The immediate next step is **TrackOn2 strict strided/original protocol parity**, not TAPNext++ and not new V26 training.

Reason:

```text
1. TrackOn2 local environment is now runnable enough for the next audit:
   - mmcv.ops_ok = true
   - trackon2_import_ok = true
   - CUDA available
   - local checkpoints exist:
     baselines/track_on/checkpoints_trackon2_dinov2.pt
     baselines/track_on/checkpoints_trackon2_dinov3.pt

2. TrackOn2 already has a parity-valid first/input bridge and a small positive plug-in ReEntry result.

3. Existing TrackOn2 strided/original cache is not parity-valid, but the blocker is now likely export/protocol/convention, not missing mmcv.ops.

4. TAPNext++ is conceptually ideal but still requires checkpoint/code acquisition and 1-video smoke; it should run in parallel only after TrackOn2 parity work is underway.

5. Current V26 distance-filter result is insufficient and should not be extended until SOTA oracle headroom is known.
```

## Evidence checked

### TrackOn2 environment

Command run:

```bash
python scripts/check_trackon2_environment.py
```

Result:

```text
outputs/paper_discovery_2026-06-27/external_baseline_smoke/trackon2_env_check.json
mmcv_ops_ok = true
trackon2_import_ok = true
torch = 2.2.2+cu121
cuda_available = true
```

### TrackOn2 checkpoints

Local files:

```text
baselines/track_on/checkpoints_trackon2_dinov2.pt  # 90M
baselines/track_on/checkpoints_trackon2_dinov3.pt  # 90M
```

### Existing parity audit

Command run:

```bash
python scripts/audit_baseline_cache_parity.py
```

Result summary:

```text
cotracker3_offline_main:
  queries = 5882
  anchor_mean_256 = 0.00001
  pred_visibility_mean = 0.740838

trackon2_strided_original_existing:
  queries = 5882
  anchor_mean_256 = 0.117206
  pred_visibility_mean = 0.410425

trackon2_first_input_bridge:
  queries = 650
  anchor_mean_256 = 0.114648
  pred_visibility_mean = 0.695027

tapnext_strided_original_existing:
  queries = 5882
  anchor_mean_256 = 0.270904
  pred_visibility_mean = 0.500984
```

Interpretation:

```text
Existing TrackOn2 strided cache has clean query anchors, so the problem is not query-frame anchor corruption. The low visibility rate / poor standard metrics likely come from inference protocol, memory policy, thresholding, frame scaling, coordinate convention, or evaluation mismatch.
```

### DINO backbone note

Command run:

```bash
python scripts/check_trackon_dino_backbone.py
```

Result:

```text
ok = false
No local HF dinov2-base directory found under expected paths.
```

Interpretation:

```text
Do not overinterpret this as TrackOn2 failure. The configured TrackOn2 test path uses vit_backbone = dinov3_s_plus and local TrackOn2 checkpoints exist. The missing HF DINOv2 directory matters only if a path requires external HF DINOv2 assets.
```

## Next executable milestone

Create a **new TrackOn2 DAVIS strided/original parity exporter** instead of reusing the old cache blindly.

Suggested script:

```text
scripts/export_trackon2_davis_strided_original_cache.py
```

It should reuse logic from:

```text
scripts/export_trackon2_reentry_stress_cache.py
scripts/audit_trackon2_cache_geometry.py
scripts/audit_baseline_cache_parity.py
datasets/tapvid_davis.py
```

Required output:

```text
outputs/paper_discovery_2026-07-05/trackon2_strict_parity/trackon2_davis_strided_original_v1.pt
outputs/paper_discovery_2026-07-05/trackon2_strict_parity/trackon2_davis_strided_original_v1_report.json
outputs/paper_discovery_2026-07-05/trackon2_strict_parity/trackon2_davis_strided_original_v1_parity.json
```

Required doc:

```text
docs/sota_protocol_parity_trackon2_davis_strided_original_2026-07-05.md
```

## Exporter requirements

The exporter must:

```text
1. Load TAPVidDAVISDataset with query_mode = strided and query_stride = 5.
2. Preserve original resolution and original frame count.
3. Use the same query_points as current CoTracker3 offline strided/original cache.
4. Convert FSPT query format [t, y_norm, x_norm] into TrackOn2 expected [t, x_px, y_px].
5. Convert TrackOn2 output xy pixel into normalized yx using original_size [H-1, W-1].
6. Store pred_tracks as [N, T, 2] yx normalized.
7. Store pred_visibility as [N, T] bool.
8. Copy gt_tracks, gt_visibility, query_points, original_size into the standard cache schema.
```

## Audit matrix to run after export

Run at least these audits:

```bash
python scripts/audit_trackon2_cache_geometry.py \
  --caches outputs/paper_discovery_2026-07-05/trackon2_strict_parity/trackon2_davis_strided_original_v1.pt \
  --out-json outputs/paper_discovery_2026-07-05/trackon2_strict_parity/trackon2_davis_strided_original_v1_geometry.json

python scripts/audit_baseline_cache_parity.py
```

Then evaluate standard TAP metrics and AJ_RD using the same evaluator used for current paper tables.

## Ablations to isolate the current failure

Run small controlled exports first, not full 30 videos immediately.

Recommended smoke matrix:

```text
Video count: 1 -> 3 -> 30 only if previous step is sane.

Variants:
A. checkpoint = trackon2_dinov3, input_scale = uint8, memory_policy = unconditional, delta_v = default 0.8
B. checkpoint = trackon2_dinov3, input_scale = unit,  memory_policy = unconditional, delta_v = default 0.8
C. checkpoint = trackon2_dinov3, input_scale = uint8, memory_policy = visibility_selective, delta_v = default 0.8
D. checkpoint = trackon2_dinov3, input_scale = uint8, memory_policy = unconditional, delta_v = 0.5
E. checkpoint = trackon2_dinov2 only if DINOv3 path is unstable
```

Primary diagnostics:

```text
query_anchor_mean_256
query_anchor_max_256
pred_visibility_mean vs gt_visibility_mean
AJ
OA
delta_avg
AJ_RD
err_px_gt_visible_median/p75/p95
```

## Stop / go gates

### Stop immediately if

```text
1. Query anchor mean > 1 px or max > 3 px after convention correction.
2. Output coordinate range is invalid or delta_avg collapses for every variant.
3. TrackOn2 model cannot run full original DAVIS frames without OOM and no chunking strategy is available.
4. The best parity attempt still has implausible AJ/delta_avg far below TrackOn2 first/input and there is no clear protocol explanation.
```

### Continue to oracle headroom if

```text
1. Query anchor is clean.
2. Standard metrics are plausible.
3. delta_avg does not collapse.
4. Visibility rate is not pathologically low unless threshold explains it.
```

## After parity passes

Only then run oracle headroom:

```text
base original
base + GT visibility
base + GT re-entry-only visibility
base + candidate-window oracle visibility
base + all-visible stress
base + query-visible-fill sanity
```

Go to ReEntry plug-in only if:

```text
candidate-window oracle or GT re-entry-only gives AJ_RD gain >= +0.01
and AJ/OA are not catastrophically worse.
```

## Final next action

Implement `scripts/export_trackon2_davis_strided_original_cache.py` and run a 1-video smoke export under variant A.

Do **not** train V26 yet.
Do **not** run TAPNext++ before TrackOn2 parity exporter reaches at least a 1-video smoke result.
Do **not** rewrite the paper claim until TrackOn2 parity + oracle results exist.
