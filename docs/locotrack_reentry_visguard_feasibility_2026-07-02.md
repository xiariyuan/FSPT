# LocoTrack Cross-Source Feasibility for ReEntry-VisGuard — 2026-07-02

## Goal

Upgrade ReEntry-VisGuard from a CoTracker-family result into a stronger plug-in TAP re-detection improvement method by testing a non-CoTracker source/base.

Primary question:

```text
Can LocoTrack provide a useful non-CoTracker coordinate or visibility channel for ReEntry-VisGuard?
```

## Current repository state

LocoTrack code exists:

```text
baselines/track_on/ensemble/locotrack/locotrack_model.py
baselines/track_on/ensemble/locotrack/locotrack_predictor.py
baselines/track_on/ensemble/locotrack/nets.py
baselines/track_on/ensemble/locotrack/utils.py
references/papers/locotrack_eccv2024.pdf
```

The lightweight `baselines/locotrack/` directory is empty, but the Track-On ensemble wrapper contains a usable `LocoTrackPredictor`.

## Current blocker

No local LocoTrack checkpoint was found.

Search result:

```text
find . -iname '*loco*.pt' -o -iname '*loco*.pth' -o -iname '*locotrack*.ckpt'
# no checkpoint found
```

The ensemble README says the optional LocoTrack checkpoint can be downloaded with:

```bash
wget -P path/to/ckpt https://huggingface.co/datasets/hamacojr/LocoTrack-pytorch-weights/resolve/main/locotrack_base.ckpt
```

Anthro-LocoTrack checkpoint is also documented, but for a clean first smoke the plain LocoTrack base checkpoint is preferred.

## New exporter added

```text
scripts/export_locotrack_rgb_stacking_cache.py
```

Status:

```text
python -m py_compile scripts/export_locotrack_rgb_stacking_cache.py  # pass
python scripts/export_locotrack_rgb_stacking_cache.py --help          # pass
```

The script converts LocoTrack output to the same unified cache schema used by the CoTracker RGB-Stacking exporters:

```text
pred_tracks: normalized yx, shape (N,T,2)
pred_visibility: bool, shape (N,T)
gt_tracks: normalized yx
gt_visibility: bool
query_points: [t,y,x] normalized
original_size: [H,W]
```

## Smoke command after checkpoint is available

Example:

```bash
python scripts/export_locotrack_rgb_stacking_cache.py \
  --checkpoint /path/to/locotrack_base.ckpt \
  --start-index 0 \
  --num-videos 1 \
  --query-batch-size 64 \
  --out-cache outputs/paper_discovery_2026-06-27/locotrack_rgb_smoke/locotrack_rgb_stacking_dev0.pt \
  --out-report outputs/paper_discovery_2026-06-27/locotrack_rgb_smoke/locotrack_rgb_stacking_dev0_report.json
```

Then evaluate:

```bash
python scripts/eval_aj_rd_from_cache.py \
  --cache-path outputs/paper_discovery_2026-06-27/locotrack_rgb_smoke/locotrack_rgb_stacking_dev0.pt \
  --output-json outputs/paper_discovery_2026-06-27/locotrack_rgb_smoke/locotrack_rgb_stacking_dev0_ajrd.json
```

Standard metrics can be obtained via existing `eval_one` helper or a small wrapper.

## Planned cross-source tests

### Stage 1 — LocoTrack standalone smoke

```text
LocoTrack coord + LocoTrack vis
```

Need to verify:

```text
query-frame anchor error
coordinate range
visibility rate
AJ_RD_256 / AJ_256 / OA_256
```

### Stage 2 — LocoTrack visibility as source

Use CoTracker offline/base coordinates and LocoTrack visibility.

Candidate combinations:

```text
base_coord + locotrack_global_vis
base_coord + local_locotrack_vis
```

The local version should use the same ReEntry-VisGuard trigger if possible, but if LocoTrack visibility semantics differ, first run a simpler local predicted-window compatibility audit.

### Stage 3 — LocoTrack coordinate as base

If LocoTrack standalone coordinates look strong:

```text
locotrack_coord + locotrack_vis
locotrack_coord + cotracker_online_local_vis
```

## Go / no-go gates

Proceed from 1-video to dev0-9 if:

```text
AJ_RD gain >= +0.01
AJ loss <= 1.0
query anchor sanity passes
visibility convention is clear
```

Proceed from dev0-9 to fresh split if:

```text
AJ_RD gain >= +0.02
AJ loss <= 1.0
positive trend is not driven by only 1-2 videos
```

Stop / appendix if:

```text
gain < +0.005
or AJ loss > 2.0
or coordinate/visibility convention cannot be made parity-clean
```

## Paper role

If successful, LocoTrack becomes the critical non-CoTracker generality evidence:

```text
ReEntry-VisGuard is not only a CoTracker offline/online heuristic.
```

If unsuccessful, it is still useful as an appendix limitation:

```text
Not every visibility source is useful; ReEntry-VisGuard requires a responsive and reasonably calibrated visibility channel.
```
