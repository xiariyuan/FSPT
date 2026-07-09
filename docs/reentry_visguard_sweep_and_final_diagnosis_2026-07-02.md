# ReEntry-VisGuard Sweep and Final Diagnosis — 2026-07-02

## 1. Visibility-only sweep

We evaluated visibility-only variants:

```text
coordinates = offline/base coordinates
visibility  = full B2 variant visibility
```

Variants:

```text
fullpost_p1
w8_p2
w16_p1
w16_p2
w32_p2
```

Selection rule for reporting best variant:

```text
maximize AJ_RD_256 subject to AJ loss <= 1 point relative to offline
```

### dev_translate_L16

| Method | AJ_RD_256 | AJ_256 | OA_256 |
|---|---:|---:|---:|
| offline | 0.4708 | 75.8656 | 90.7947 |
| full_w8_p2 | 0.5498 | 75.8454 | 92.6814 |
| vis_w8_p2 | **0.5511** | 75.5652 | **92.6814** |
| full_w16_p2 | 0.5485 | 75.8344 | 92.6729 |
| vis_w16_p2 | 0.5506 | 75.5666 | 92.6729 |

Best: `vis_w8_p2`.

### dev_occluder_L16

| Method | AJ_RD_256 | AJ_256 | OA_256 |
|---|---:|---:|---:|
| offline | 0.6422 | 77.6880 | 90.6437 |
| full_w8_p2 | 0.6897 | 77.2316 | **92.2945** |
| vis_w8_p2 | **0.6943** | **77.2555** | **92.2945** |
| full_w16_p2 | 0.6873 | 77.1682 | 92.2855 |
| vis_w16_p2 | 0.6940 | 77.2521 | 92.2855 |

Best: `vis_w8_p2`.

### RGB fresh20-49 natural

| Method | AJ_RD_256 | AJ_256 | OA_256 |
|---|---:|---:|---:|
| offline | 0.3816 | 79.5944 | 91.4636 |
| full_w8_p2 | 0.4462 | 79.0753 | 92.8853 |
| vis_w8_p2 | **0.4510** | 79.1110 | **92.8853** |
| full_w16_p2 | 0.4454 | 79.0664 | 92.8804 |
| vis_w16_p2 | 0.4505 | **79.1111** | 92.8804 |

Best: `vis_w8_p2`, though `vis_w16_p2` is almost tied and has marginally higher AJ.

### fresh20-49 translate_L16

| Method | AJ_RD_256 | AJ_256 | OA_256 |
|---|---:|---:|---:|
| offline | 0.4788 | 75.1940 | 90.7805 |
| full_w8_p2 | 0.5309 | 74.8456 | **92.2245** |
| vis_w8_p2 | **0.5336** | 74.7705 | **92.2245** |
| full_w16_p2 | 0.5310 | 74.8457 | 92.2166 |
| vis_w16_p2 | 0.5333 | **74.7709** | 92.2166 |

Best: `vis_w8_p2`.

### fresh20-49 occluder_L16

| Method | AJ_RD_256 | AJ_256 | OA_256 |
|---|---:|---:|---:|
| offline | 0.6311 | 77.6280 | 90.8918 |
| full_w8_p2 | 0.6607 | 76.8550 | **92.1748** |
| vis_w8_p2 | **0.6659** | **77.0436** | **92.1748** |
| full_w16_p2 | 0.6588 | 76.7991 | 92.1682 |
| vis_w16_p2 | 0.6658 | 77.0422 | 92.1682 |

Best: `vis_w8_p2`, nearly tied with `vis_w16_p2`.

## Sweep conclusion

```text
vis_w8_p2 is the best or tied-best visibility-only configuration across dev, natural, fresh translate, and fresh occluder.
```

Recommended clean main method:

```text
ReEntry-VisGuard-W8P2:
  coordinates = offline/base coordinates
  visibility  = B2-W8-P2 predicted visibility
```

W16-P2 remains a very close conservative alternative and is useful for comparison with previous B2-W16-P2 mainline.

---

## 2. Final split coordinate-vs-visibility diagnosis

We evaluated re-entry GT-visible segments in the final settings. Metrics:

```text
coord8_rate: coordinate within 8 px on GT-visible re-entry-segment frames, ignoring predicted visibility
predvis_recall: predicted visible on GT-visible re-entry-segment frames
joint8_rate: coordinate within 8 px AND predicted visible
first_coord8_rate: coordinate within 8 px at first re-entry frame
first_predvis_rate: predicted visible at first re-entry frame
```

### RGB fresh20-49 natural

```text
events = 10,818
GT-visible re-entry segment frames = 434,438
```

| Method | coord8_rate | predvis_recall | joint8_rate | first_coord8 | first_predvis |
|---|---:|---:|---:|---:|---:|
| offline | 1.0000 | 0.6240 | 0.6240 | 1.0000 | 0.2538 |
| full_w8_p2 | 1.0000 | 0.8350 | 0.8350 | 1.0000 | 0.5917 |
| vis_w8_p2 | 1.0000 | 0.8350 | 0.8350 | 1.0000 | 0.5917 |
| full_w16_p2 | 1.0000 | 0.8340 | 0.8340 | 1.0000 | 0.5908 |
| vis_w16_p2 | 1.0000 | 0.8340 | 0.8340 | 1.0000 | 0.5908 |

### fresh20-49 translate_L16

```text
events = 12,728
GT-visible re-entry segment frames = 760,737
```

| Method | coord8_rate | predvis_recall | joint8_rate | first_coord8 | first_predvis |
|---|---:|---:|---:|---:|---:|
| offline | 1.0000 | 0.7593 | 0.7593 | 1.0000 | 0.3473 |
| full_w8_p2 | 1.0000 | 0.8989 | 0.8989 | 1.0000 | 0.6474 |
| vis_w8_p2 | 1.0000 | 0.8989 | 0.8989 | 1.0000 | 0.6474 |
| full_w16_p2 | 1.0000 | 0.8982 | 0.8982 | 1.0000 | 0.6468 |
| vis_w16_p2 | 1.0000 | 0.8982 | 0.8982 | 1.0000 | 0.6468 |

### fresh20-49 occluder_L16

```text
events = 19,948
GT-visible re-entry segment frames = 1,709,050
```

| Method | coord8_rate | predvis_recall | joint8_rate | first_coord8 | first_predvis |
|---|---:|---:|---:|---:|---:|
| offline | 1.0000 | 0.8390 | 0.8390 | 1.0000 | 0.4180 |
| full_w8_p2 | 1.0000 | 0.9365 | 0.9365 | 1.0000 | 0.7122 |
| vis_w8_p2 | 1.0000 | 0.9365 | 0.9365 | 1.0000 | 0.7122 |
| full_w16_p2 | 1.0000 | 0.9362 | 0.9362 | 1.0000 | 0.7086 |
| vis_w16_p2 | 1.0000 | 0.9362 | 0.9362 | 1.0000 | 0.7086 |

## Final diagnosis conclusion

Across natural, translate, and occluder final splits:

```text
coord8_rate = 1.0000 for offline, full B2, and VisGuard variants.
```

Therefore, coordinate localization is not the measured bottleneck in these re-entry segments. The difference is predicted visibility recovery:

```text
natural first_predvis:
offline 0.2538 -> vis_w8_p2 0.5917

translate first_predvis:
offline 0.3473 -> vis_w8_p2 0.6474

occluder first_predvis:
offline 0.4180 -> vis_w8_p2 0.7122
```

This strongly supports the reframed paper claim:

```text
TAP re-entry failures in our setting are primarily visible-state recovery failures, not coordinate re-localization failures.
```

## Updated method recommendation

Main method:

```text
ReEntry-VisGuard-W8P2
```

Definition:

```text
Trigger: base predicted invisible after query and override predicted visible for P=2 consecutive frames
Window: t-1 through t+8
Coordinates: offline/base
Visibility: override/B2-W8-P2 visibility in the window
```

Close conservative baseline:

```text
ReEntry-VisGuard-W16P2
```

Use W16P2 when we want direct continuity with earlier B2-W16-P2 mainline; use W8P2 as the best dev-selected visibility-only method.
