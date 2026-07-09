# RGB-Stacking 10-Video B2 Gate Ablation — 2026-06-29

## Decision

The lightweight RGB gate ablation found a better RGB B2 variant: `post16`.

Instead of overriding the full post-trigger segment, `post16` only applies the online override for a short 16-frame window after each trigger. This fixes most of the `rgb_stacking_000008` failure while improving the 10-video aggregate.

## Main result

| method | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 | rgb000008 AJ_RD | trigger recall | trigger precision |
|---|---:|---:|---:|---:|---:|---:|---:|
| CoTracker3 offline | 0.3763 | 80.0721 | - | - | 0.3989 | - | - |
| CoTracker3 online | 0.4522 | 45.4239 | - | - | 0.2315 | - | - |
| baseline_full | 0.466 | 79.7562 | 92.9475 | 88.9484 | 0.2378 | 0.947303 | 0.635225 |
| post16 | 0.4865 | 79.903 | 93.0717 | 89.0994 | 0.3821 | 0.947303 | 0.635225 |
| post32 | 0.4818 | 79.8718 | 93.046 | 89.065 | 0.3236 | 0.947303 | 0.635225 |
| base_invis_t_post32 | 0.4864 | 79.885 | 93.0377 | 89.0777 | 0.3562 | 0.861411 | 0.62871 |
| base_invis_t_persist2_full | 0.4777 | 79.7963 | 92.9573 | 88.9829 | 0.3319 | 0.83112 | 0.633059 |
| persist4_full | 0.474 | 79.7798 | 92.9557 | 88.9714 | 0.2515 | 0.915353 | 0.652856 |
| k4_full | 0.4739 | 79.6902 | 92.7874 | 88.9733 | 0.2517 | 0.902075 | 0.665442 |

Best row:

```text
name = post16
true_AJ_RD_256 = 0.4865
AJ_256_pct = 79.903
OA_256_pct = 93.0717
delta_avg_256_pct = 89.0994
rgb000008_AJ_RD_256 = 0.3821
rgb000008_AJ_256_pct = 78.925
trigger_precision_track = 0.635225
trigger_recall_track = 0.947303
tracks_with_trigger = 3594
triggered_reentry_tracks = 2283
triggered_nonreentry_tracks = 1311
missed_reentry_tracks = 127
```

## Gains over previous RGB B2 baseline

```text
baseline_full AJ_RD_256 = 0.466
post16 AJ_RD_256 = 0.4865
AJ_RD_256 gain = +0.0205

baseline_full AJ_256 = 79.7562
post16 AJ_256 = 79.903
AJ_256 gain = +0.1468

baseline_full rgb000008 AJ_RD_256 = 0.2378
post16 rgb000008 AJ_RD_256 = 0.3821
rgb000008 AJ_RD_256 gain = +0.1443
```

## Interpretation

The `rgb_stacking_000008` failure audit showed that full-post override can be harmful when the override branch is worse than the base after true re-entry. `post16` addresses this directly: it still uses online to help around the re-entry moment, but it returns control to the offline base instead of replacing the whole future trajectory.

This is important because it improves both objectives on RGB 10-video:

```text
AJ_RD_256 improves over baseline_full
standard AJ also improves over baseline_full
rgb000008 failure is largely reduced
```

Compared with the offline base:

```text
post16 vs offline AJ_RD_256 = +0.1102
post16 vs offline AJ_256 = -0.1691
```

Compared with the online override:

```text
post16 vs online AJ_RD_256 = +0.0343
post16 vs online AJ_256 = +34.4791
```

## What did not work as well

- Pure persistence gates (`persist2/3/4_full`) improve over baseline, but less than `post16`.
- Requiring longer base invisibility (`k2/k4_full`) also helps, but less than `post16`.
- Base-invisible guards help `rgb_stacking_000008`, but lose too much recall when used with full-post override.
- Simple max-disagreement guards do almost nothing in this setting.

## Paper-facing implication

For RGB-Stacking, the safest B2 form may be a localized finite-window override rather than a full-post override. This reinforces the paper story: B2 should be a local re-entry intervention, not a permanent tracker replacement.

## Artifacts

```text
scripts/sweep_rgb_b2_gate_ablation.py
outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/rgb_b2_gate_ablation_10video/summary.json
outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/rgb_b2_gate_ablation_10video/post16.pt
```
