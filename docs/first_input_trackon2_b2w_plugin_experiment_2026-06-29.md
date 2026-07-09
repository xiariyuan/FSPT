# First-input TrackOn2 B2-W Plug-in Experiment — 2026-06-29

## Decision

This is a supplemental plug-in generality experiment under a parity-valid first-query/input-resolution DAVIS protocol. It is not the main B2-W strided-original protocol. The goal is to test whether B2-W can improve a stronger reproduced baseline, TrackOn2, without relying only on the CoTracker3 offline / global-B1 setup.

Result: B2-W gives a small but positive improvement over TrackOn2 when TrackOn2 is used as the base and CoTracker3 offline is used as the local override. The gain is modest but important because it shows B2-W can act as a plug-in local routing mechanism on a strong reproduced baseline.

## Protocol

```text
Dataset: DAVIS
Protocol: first-query/input-resolution bridge
Queries: 650
Reason: TrackOn2 parity is validated under this protocol
Not comparable directly to main strided-original DAVIS/RGB tables
```

## Main query-weighted results

| method | base | override | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 | trigger precision | trigger recall |
|---|---|---|---:|---:|---:|---:|---:|---:|
| CoTracker3 offline first-input | - | - | 0.4525 | 62.1660 | 89.0107 | 77.2244 | - | - |
| TrackOn2 first-input | - | - | 0.5444 | 67.0406 | 93.0615 | 79.8418 | - | - |
| B2-W16 | TrackOn2 | CoTracker3 offline | 0.5513 | 67.1750 | 93.2513 | 79.8543 | 0.7700 | 0.6332 |
| B2-W16-P2 | TrackOn2 | CoTracker3 offline | 0.5509 | 67.1260 | 93.2224 | 79.8202 | 0.7892 | 0.6216 |
| B2-W16 | CoTracker3 offline | TrackOn2 | 0.5492 | 64.4149 | 92.6774 | 77.8309 | 0.6071 | 0.8533 |
| B2-W16-P2 | CoTracker3 offline | TrackOn2 | 0.5495 | 64.4295 | 92.7140 | 77.8145 | 0.6132 | 0.8263 |

## Key gains

TrackOn2 as base, CoTracker3 as override:

```text
B2-W16 vs TrackOn2:
AJ_RD_256 +0.0069
AJ_256 +0.1344
OA_256 +0.1898
delta_avg_256 +0.0125

B2-W16-P2 vs TrackOn2:
AJ_RD_256 +0.0065
AJ_256 +0.0854
OA_256 +0.1609
delta_avg_256 -0.0216
```

CoTracker3 as base, TrackOn2 as override:

```text
B2-W16-P2 vs CoTracker3 first-input:
AJ_RD_256 +0.0970
AJ_256 +2.2635

B2-W16-P2 vs TrackOn2 first-input:
AJ_RD_256 +0.0051
AJ_256 -2.6111
```

Interpretation: when TrackOn2 is used as the override for a weaker CoTracker3 base, re-entry improves strongly but standard AJ remains below using TrackOn2 globally. When TrackOn2 is used as the base, B2-W produces a small improvement over the strong TrackOn2 baseline itself.

## Video-weighted means

| method | video-mean AJ_RD_256 | video-mean AJ_256 | video-mean OA_256 | video-mean delta_avg_256 |
|---|---:|---:|---:|---:|
| CoTracker3 offline first-input | 0.409808 | 62.16605 | 89.01066 | 77.22442 |
| TrackOn2 first-input | 0.497728 | 67.04064 | 93.06147 | 79.84184 |
| B2-W16, TrackOn2 base | 0.508632 | 67.17504 | 93.25126 | 79.854317 |
| B2-W16-P2, TrackOn2 base | 0.508408 | 67.126047 | 93.222423 | 79.82024 |
| B2-W16, CoTracker base | 0.509392 | 64.41494 | 92.677363 | 77.83087 |
| B2-W16-P2, CoTracker base | 0.509472 | 64.42954 | 92.713983 | 77.81451 |

## Per-video stability

```text
n_videos = 30
p2_trackon_base_improves_AJRD_vs_trackon = 12
p2_trackon_base_improves_AJ_vs_trackon = 13
p2_trackon_base_improves_both_vs_trackon = 9
p2_trackon_base_AJRD_drop_gt_0p01 = 6
p2_trackon_base_AJ_drop_gt_1 = 3
p2_cotracker_base_improves_AJRD_vs_cotracker = 20
p2_cotracker_base_improves_AJ_vs_cotracker = 20
```

Interpretation: improvements over TrackOn2 are modest and not universal, but the aggregate is positive. This is sufficient for a supplemental plug-in generality claim, not a main superiority claim.

## How to use in the paper

Safe claim:

```text
Under a parity-valid first-query/input-resolution DAVIS protocol, B2-W can also provide small positive gains when applied on top of a reproduced TrackOn2 baseline.
```

Unsafe claim:

```text
Do not claim B2-W generally beats TrackOn2 under all protocols.
Do not mix these first-input numbers directly with the main strided-original DAVIS/RGB tables.
```

Recommended placement:

```text
Appendix or supplemental experiment: Plug-in generality on a reproduced TrackOn2 first-input protocol.
Use as CCF-B/Q2 credibility booster, not as the central result.
```

## Artifacts

```text
outputs/paper_discovery_2026-06-27/first_input_trackon2_b2w/
outputs/paper_discovery_2026-06-27/first_input_trackon2_b2w/first_input_trackon2_b2w_per_video_summary.json
outputs/paper_discovery_2026-06-27/first_input_trackon2_b2w/first_input_trackon2_b2w_per_video_rows.jsonl
```
