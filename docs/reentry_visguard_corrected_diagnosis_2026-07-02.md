# ReEntry-VisGuard Corrected Diagnosis and Improvement Evidence — 2026-07-02

## Why this document exists

During the improvement-paper sprint, we re-ran the coordinate-vs-visibility diagnosis with an explicit no-leak sanity audit. This found an important correction:

```text
Older coordinate-vs-visibility diagnostics compared normalized [y,x] coordinate distances directly against pixel thresholds such as 8 and 16.
Because normalized coordinates live near [0,1], this made coord8_rate spuriously saturate near 1.0.
```

The corrected scripts now convert normalized coordinates to TAPNext++ / AJ_RD 256-space pixels before applying thresholds.

Corrected files / outputs:

```text
scripts/audit_reentry_segment_coord_visibility_final.py
scripts/audit_reentry_visguard_no_leak.py
outputs/paper_discovery_2026-06-27/reentry_visibility_bottleneck_final/summary.json
outputs/paper_discovery_2026-06-27/reentry_visguard_no_leak_sanity/rgb_fresh20_49_natural/summary.json
```

## Main correction to the paper story

Old over-strong story:

```text
Coordinates are fully saturated; re-entry failure is only visibility failure.
```

Corrected story:

```text
Coordinates are not perfect, but the offline/base coordinate channel is consistently stronger than the override coordinate channel. The most effective improvement is to preserve the strong base coordinates and locally recover visibility in predicted re-entry windows.
```

Therefore, the paper should be framed as:

```text
coordinate-preserving visibility recovery
```

not as:

```text
coordinate is never a bottleneck
```

---

## 1. No-leak sanity audit on RGB fresh20-49 natural

Command:

```bash
python scripts/audit_reentry_visguard_no_leak.py \
  --cache-path outputs/paper_discovery_2026-06-27/reentry_visguard_w8p2_repro/rgb_fresh20_49_natural/reentry_visguard_w8p2.pt \
  --out-dir outputs/paper_discovery_2026-06-27/reentry_visguard_no_leak_sanity/rgb_fresh20_49_natural \
  --make-baselines
```

Key result:

```text
exact_close_frame_rate_atol1e-8 = 0.001969
```

Interpretation:

```text
The predictions are not exact copies of GT. This argues against obvious GT leakage.
```

Sanity baselines:

| Method | AJ_RD_256 | AJ_256 | OA_256 |
|---|---:|---:|---:|
| original ReEntry-VisGuard-W8P2 | 0.4510 | 79.1110 | 92.8853 |
| visibility shuffled | 0.3987 | 67.1319 | 82.0698 |
| coordinate perturb sigma=8px | 0.1340 | 19.9635 | 92.8853 |

Interpretation:

```text
Shuffling visibility hurts AJ_RD and AJ.
Perturbing coordinates heavily destroys AJ_RD and AJ.
Therefore both coordinate and visibility channels matter, and the metric path is sensitive to both.
```

Corrected coordinate/visibility rates for ReEntry-VisGuard-W8P2 on natural:

| Metric | Value |
|---|---:|
| segment coord1_rate | 0.3263 |
| segment coord2_rate | 0.5130 |
| segment coord4_rate | 0.6665 |
| segment coord8_rate | 0.7703 |
| segment coord16_rate | 0.8555 |
| first coord8_rate | 0.7578 |
| predvis_recall on GT-visible re-entry frames | 0.8111 |
| first_predvis_rate | 0.5917 |

---

## 2. Corrected coordinate-vs-visibility diagnosis

### RGB fresh20-49 natural

| Method | coord8 | coord16 | predvis | joint8 | first_coord8 | first_predvis | first_joint8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| offline | 0.7815 | 0.8640 | 0.6240 | 0.5617 | 0.7578 | 0.2538 | 0.2216 |
| full_w8_p2 | 0.7748 | 0.8585 | 0.8350 | 0.6863 | 0.7336 | 0.5917 | 0.4489 |
| vis_w8_p2 | 0.7815 | 0.8640 | 0.8350 | 0.6916 | 0.7578 | 0.5917 | 0.4678 |

Interpretation:

```text
Base/offline coordinates are stronger than full override coordinates.
VisGuard preserves offline coordinate rates and recovers the visibility rates of the full local override.
```

### fresh20-49 translate_L16

| Method | coord8 | coord16 | predvis | joint8 | first_coord8 | first_predvis | first_joint8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| offline | 0.8472 | 0.9117 | 0.7593 | 0.7068 | 0.7494 | 0.3473 | 0.3037 |
| full_w8_p2 | 0.8452 | 0.9087 | 0.8989 | 0.7934 | 0.7307 | 0.6474 | 0.4996 |
| vis_w8_p2 | 0.8472 | 0.9117 | 0.8989 | 0.7947 | 0.7494 | 0.6474 | 0.5140 |

Interpretation:

```text
Again, visibility recovery drives joint recovery, while coordinate preservation avoids the coordinate degradation seen in full override.
```

### fresh20-49 occluder_L16

| Method | coord8 | coord16 | predvis | joint8 | first_coord8 | first_predvis | first_joint8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| offline | 0.9057 | 0.9517 | 0.8390 | 0.8099 | 0.8574 | 0.4180 | 0.4004 |
| full_w8_p2 | 0.9008 | 0.9488 | 0.9365 | 0.8676 | 0.8060 | 0.7122 | 0.5974 |
| vis_w8_p2 | 0.9057 | 0.9517 | 0.9365 | 0.8718 | 0.8574 | 0.7122 | 0.6436 |

Interpretation:

```text
Occluder is the cleanest setting for the final story: VisGuard keeps the best coordinate channel and obtains the improved visibility channel, producing the strongest joint first-reentry gain.
```

---

## 3. Channel-wise factorial intervention evidence

### RGB fresh20-49 natural

| Method | Coordinates | Visibility | AJ_RD_256 | AJ_256 | OA_256 | ΔAJ_RD vs base | ΔAJ vs base |
|---|---|---|---:|---:|---:|---:|---:|
| base_coord_base_vis | base | base | 0.3816 | 79.5944 | 91.4636 | - | - |
| override_coord_override_vis | override | override | 0.4121 | 44.6933 | 55.7585 | +0.0305 | -34.9011 |
| override_coord_base_vis | override | base | 0.3494 | 60.8419 | 91.4636 | -0.0322 | -18.7525 |
| base_coord_override_vis_global | base | override global | 0.4282 | 45.7904 | 55.7585 | +0.0466 | -33.8040 |
| ReEntry-VisGuard-W8P2 | base | local override vis | 0.4510 | 79.1110 | 92.8853 | +0.0694 | -0.4834 |

### fresh20-49 translate_L16

| Method | Coordinates | Visibility | AJ_RD_256 | AJ_256 | OA_256 | ΔAJ_RD vs base | ΔAJ vs base |
|---|---|---|---:|---:|---:|---:|---:|
| base_coord_base_vis | base | base | 0.4788 | 75.1940 | 90.7805 | - | - |
| override_coord_override_vis | override | override | 0.4981 | 43.0978 | 56.7847 | +0.0193 | -32.0962 |
| override_coord_base_vis | override | base | 0.4499 | 52.8525 | 90.7805 | -0.0289 | -22.3415 |
| base_coord_override_vis_global | base | override global | 0.5133 | 44.0643 | 56.7847 | +0.0345 | -31.1297 |
| ReEntry-VisGuard-W8P2 | base | local override vis | 0.5336 | 74.7705 | 92.2245 | +0.0548 | -0.4235 |

### fresh20-49 occluder_L16

| Method | Coordinates | Visibility | AJ_RD_256 | AJ_256 | OA_256 | ΔAJ_RD vs base | ΔAJ vs base |
|---|---|---|---:|---:|---:|---:|---:|
| base_coord_base_vis | base | base | 0.6311 | 77.6280 | 90.8918 | - | - |
| override_coord_override_vis | override | override | 0.6146 | 43.1490 | 58.0983 | -0.0165 | -34.4790 |
| override_coord_base_vis | override | base | 0.5858 | 59.6368 | 90.8918 | -0.0453 | -17.9912 |
| base_coord_override_vis_global | base | override global | 0.6469 | 44.7880 | 58.0983 | +0.0158 | -32.8400 |
| ReEntry-VisGuard-W8P2 | base | local override vis | 0.6659 | 77.0436 | 92.1748 | +0.0348 | -0.5844 |

## 4. Corrected core thesis

The strongest paper claim is now:

```text
ReEntry-VisGuard is a coordinate-preserving visibility recovery layer.
It works because the base coordinate channel is stronger than the override coordinate channel, while the override visibility channel contains useful re-entry recovery signal.
Localizing this visibility transfer to predicted re-entry windows converts that signal into AJ_RD gains without the severe standard-AJ damage of global override.
```

Do not claim:

```text
coord8_rate = 1.0000
coordinate is irrelevant
re-entry failure is only visibility failure
```

Allowed claim:

```text
In our final settings, coordinate and visibility both matter, but the best deployable improvement is channel-selective: preserve the stronger base coordinates and locally recover visibility.
```

## 5. Paper impact of the correction

This correction does not kill the paper. It makes it more credible.

Why:

```text
1. The old claim was too strong and vulnerable to reviewer attack.
2. The new no-leak audit shows there is no obvious GT copy.
3. The channel-wise factorial results now provide stronger improvement-paper evidence than the old coord8=1.0 claim.
4. The method is still clearly effective across natural, translate, and occluder settings.
```

Updated story:

```text
Existing trackers expose complementary channels:
- base/offline coordinates are reliable;
- override/online visibility is responsive but globally harmful;
- local visibility transfer preserves AJ while improving AJ_RD.
```
