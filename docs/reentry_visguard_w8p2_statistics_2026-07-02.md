# ReEntry-VisGuard-W8P2 Statistics — 2026-07-02

## Method

**ReEntry-VisGuard-W8P2**:

```text
coordinates = offline/base coordinates
visibility  = B2-W8-P2 predicted visibility in predicted re-entry windows
```

Runtime trigger:

```text
base predicted invisible after query
and override predicted visible for P=2 consecutive frames
```

Window:

```text
t-1 through t+8
```

This is a clean prediction-only method. It does not use GT at inference.

## Main query-weighted results

### RGB fresh20-49 natural

| Method | AJ_RD_256 | AJ_256 | OA_256 |
|---|---:|---:|---:|
| offline | 0.3816 | 79.5944 | 91.4636 |
| full_w8_p2 | 0.4462 | 79.0753 | 92.8853 |
| ReEntry-VisGuard-W8P2 | **0.4510** | 79.1110 | **92.8853** |
| full_w16_p2 | 0.4454 | 79.0664 | 92.8804 |
| ReEntry-VisGuard-W16P2 | 0.4505 | **79.1111** | 92.8804 |
| ReEntry-Guard RF | 0.4499 | 79.1749 | 92.0864 |
| Guard-Vis | 0.4513 | 79.2045 | 92.0864 |

### fresh20-49 translate L16

| Method | AJ_RD_256 | AJ_256 | OA_256 |
|---|---:|---:|---:|
| offline | 0.4788 | 75.1940 | 90.7805 |
| full_w8_p2 | 0.5309 | 74.8456 | 92.2245 |
| ReEntry-VisGuard-W8P2 | **0.5336** | 74.7705 | **92.2245** |
| full_w16_p2 | 0.5310 | 74.8457 | 92.2166 |
| ReEntry-VisGuard-W16P2 | 0.5333 | **74.7709** | 92.2166 |
| ReEntry-Guard RF | 0.5336 | 74.8490 | 91.4654 |
| Guard-Vis | 0.5336 | 74.8618 | 91.4654 |

### fresh20-49 occluder L16

| Method | AJ_RD_256 | AJ_256 | OA_256 |
|---|---:|---:|---:|
| offline | 0.6311 | 77.6280 | 90.8918 |
| full_w8_p2 | 0.6607 | 76.8550 | 92.1748 |
| ReEntry-VisGuard-W8P2 | **0.6659** | **77.0436** | **92.1748** |
| full_w16_p2 | 0.6588 | 76.7991 | 92.1682 |
| ReEntry-VisGuard-W16P2 | 0.6658 | 77.0422 | 92.1682 |
| ReEntry-Guard RF | 0.6632 | 77.0761 | 92.0632 |
| Guard-Vis | 0.6655 | 77.1760 | 92.0632 |

---

## Video-level paired statistics

Statistics below use per-video paired deltas, method A minus method B. These are video-mean statistics, so the aggregate AJ_RD numbers differ from query-weighted main tables.

### ReEntry-VisGuard-W8P2 vs offline

| Setting | mean ΔAJ_RD | 95% bootstrap CI | + videos | - videos | sign p |
|---|---:|---:|---:|---:|---:|
| natural | +0.0677 | [0.0512, 0.0848] | 28 | 1 | 1.1e-7 |
| translate L16 | +0.0475 | [0.0343, 0.0608] | 28 | 2 | 8.7e-7 |
| occluder L16 | +0.0337 | [0.0242, 0.0442] | 28 | 2 | 8.7e-7 |

Interpretation:

```text
ReEntry-VisGuard-W8P2 robustly improves AJ_RD over offline across natural, translate, and occluder settings.
```

### ReEntry-VisGuard-W8P2 vs full B2-W8P2

| Setting | mean ΔAJ_RD | 95% bootstrap CI | + videos | - videos | sign p |
|---|---:|---:|---:|---:|---:|
| natural | +0.0048 | [-0.0008, 0.0106] | 18 | 11 | 0.2649 |
| translate L16 | +0.0025 | [-0.0015, 0.0065] | 18 | 12 | 0.3616 |
| occluder L16 | +0.0054 | [0.0021, 0.0089] | 20 | 10 | 0.0987 |

Interpretation:

```text
Visibility-only coordinate preservation gives a small improvement over full B2-W8P2. It is consistently positive in mean, strongest on occluder, but not uniformly significant by sign test.
```

### ReEntry-VisGuard-W8P2 vs full B2-W16P2

| Setting | mean ΔAJ_RD | 95% bootstrap CI | + videos | - videos | sign p |
|---|---:|---:|---:|---:|---:|
| natural | +0.0063 | [-0.0008, 0.0138] | 17 | 12 | 0.4583 |
| translate L16 | +0.0024 | [-0.0028, 0.0075] | 16 | 14 | 0.8555 |
| occluder L16 | +0.0074 | [0.0033, 0.0115] | 21 | 9 | 0.0428 |

Interpretation:

```text
ReEntry-VisGuard-W8P2 is clearly better than the old full B2-W16P2 on occluder, and slightly better on natural/translate, but the latter two should be described as small, not dramatic.
```

---

## Final conclusion

The final main method should be:

```text
ReEntry-VisGuard-W8P2
```

Main paper claim:

```text
Re-entry failures in our TAP setting are primarily visible-state recovery failures. ReEntry-VisGuard preserves the base coordinates and locally recovers visibility, yielding robust AJ_RD gains over the offline base while preserving standard AJ.
```

Important wording constraint:

```text
Do not claim a large universal improvement over full B2. The robust claim is against offline; the methodological claim is that visibility-only recovery is cleaner and often better than coordinate+visibility override, especially under occlusion stress.
```

Recommended role of methods:

```text
Main method: ReEntry-VisGuard-W8P2
Conservative baseline: full B2-W16P2 and full B2-W8P2
Close comparison: ReEntry-VisGuard-W16P2
Learned extension: ReEntry-Guard / Guard-Vis
Appendix: candidate-pool oracle, TrackOn2/TAPNext feasibility
```
