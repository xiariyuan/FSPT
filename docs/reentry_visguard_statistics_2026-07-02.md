# ReEntry-VisGuard Statistics — 2026-07-02

## Method definition

**B2-Vis / ReEntry-VisGuard**:

```text
coordinates = offline/base coordinates
visibility  = B2-W16-P2 predicted visibility
```

**Guard-Vis**:

```text
coordinates = offline/base coordinates
visibility  = ReEntry-Guard predicted visibility
```

Both are visibility-only hybrids. They do not use GT at inference. B2-Vis is the cleaner deployable variant because it uses only the fixed B2 prediction/trigger pipeline.

## Important metric note

Two aggregation modes appear in our artifacts:

```text
query-weighted AJ_RD: used in main tables, e.g. offline natural = 0.3816
video-mean AJ_RD: used in paired video-level statistics, e.g. offline natural = 0.3330
```

The two differ because videos have different numbers of eligible re-entry queries. For paper main tables we keep query-weighted AJ_RD, while statistical robustness is assessed with video-level paired deltas.

---

## Main query-weighted results

### RGB fresh20-49 natural

| Method | AJ_RD_256 | AJ_256 | OA_256 |
|---|---:|---:|---:|
| offline | 0.3816 | 79.5944 | 91.4636 |
| B2-W16-P2 | 0.4454 | 79.0664 | 92.8804 |
| ReEntry-Guard RF | 0.4499 | 79.1749 | 92.0864 |
| B2-Vis | **0.4505** | 79.1111 | **92.8804** |
| Guard-Vis | **0.4513** | **79.2045** | 92.0864 |

### fresh20-49 translate L16

| Method | AJ_RD_256 | AJ_256 | OA_256 |
|---|---:|---:|---:|
| offline | 0.4788 | 75.1940 | 90.7805 |
| B2-W16-P2 | 0.5310 | 74.8457 | 92.2166 |
| ReEntry-Guard RF | **0.5336** | 74.8490 | 91.4654 |
| B2-Vis | 0.5333 | 74.7709 | **92.2166** |
| Guard-Vis | **0.5336** | **74.8618** | 91.4654 |

### fresh20-49 occluder L16

| Method | AJ_RD_256 | AJ_256 | OA_256 |
|---|---:|---:|---:|
| offline | 0.6311 | 77.6280 | 90.8918 |
| B2-W16-P2 | 0.6588 | 76.7991 | **92.1682** |
| ReEntry-Guard RF | 0.6632 | 77.0761 | 92.0632 |
| B2-Vis | **0.6658** | 77.0422 | **92.1682** |
| Guard-Vis | 0.6655 | **77.1760** | 92.0632 |

---

## Video-level paired robustness

The following statistics use per-video paired deltas. Values below are method A minus method B.

### B2-Vis vs offline

| Setting | mean ΔAJ_RD | 95% bootstrap CI | + videos | - videos | sign p |
|---|---:|---:|---:|---:|---:|
| natural | +0.0671 | [0.0505, 0.0844] | 28 | 1 | 1.1e-7 |
| translate L16 | +0.0473 | [0.0340, 0.0606] | 28 | 2 | 8.7e-7 |
| occluder L16 | +0.0336 | [0.0237, 0.0444] | 28 | 2 | 8.7e-7 |

Interpretation:

```text
B2-Vis is robustly better than offline on AJ_RD across all three settings.
```

### B2-Vis vs B2-W16-P2

| Setting | mean ΔAJ_RD | 95% bootstrap CI | + videos | - videos | sign p |
|---|---:|---:|---:|---:|---:|
| natural | +0.0057 | [-0.0010, 0.0128] | 16 | 12 | 0.5716 |
| translate L16 | +0.0022 | [-0.0028, 0.0071] | 17 | 13 | 0.5847 |
| occluder L16 | +0.0073 | [0.0033, 0.0112] | 23 | 7 | 0.0052 |

Interpretation:

```text
B2-Vis is a small improvement over full B2 in natural/translate but not statistically significant there.
In occluder L16, B2-Vis significantly improves over B2, suggesting coordinate preservation helps most when visibility recovery is the key factor.
```

### Guard-Vis vs ReEntry-Guard

| Setting | mean ΔAJ_RD | 95% bootstrap CI | + videos | - videos | sign p |
|---|---:|---:|---:|---:|---:|
| natural | +0.0017 | [-0.0042, 0.0076] | 16 | 12 | 0.5716 |
| translate L16 | +0.0000 | [-0.0049, 0.0047] | 15 | 15 | 1.0000 |
| occluder L16 | +0.0025 | [-0.0012, 0.0061] | 17 | 12 | 0.4583 |

Interpretation:

```text
Guard-Vis is comparable to ReEntry-Guard. It does not provide a statistically reliable improvement over Guard, but it confirms that keeping coordinates fixed is safe and often slightly beneficial.
```

---

## Runtime cleanliness check

B2-W16-P2 / B2-Vis is the clean deployable variant. Code inspection confirms the B2 trigger is based on predicted visibility only:

```text
trigger condition = base invisible run >= 1 and override visible for P consecutive frames
```

Ground-truth visibility is read by the evaluation scripts for metric computation and alignment checks, but not for the B2 trigger itself.

ReEntry-Guard / Guard-Vis remains more sensitive in interpretation because previous learned-gate datasets were built over eligible re-entry queries. Therefore:

```text
B2-Vis should be the main clean method variant.
Guard-Vis should be reported as a learned/diagnostic extension unless a fully runtime query-level gate is implemented.
```

---

## Final interpretation

The strongest corrected story is:

```text
TAP re-entry recovery failure is often a visible-state recovery failure, not a coordinate localization failure.
B2-W16-P2 works primarily because it recovers visibility after re-entry.
B2-Vis makes this explicit: preserve offline coordinates, recover visibility locally.
```

This reframes the main method as:

```text
ReEntry-VisGuard: Visibility-Calibrated Re-entry Recovery for TAP
```

rather than a full coordinate override method.

## Recommended paper positioning

Main method:

```text
B2-Vis / ReEntry-VisGuard
```

Main claim:

```text
A simple prediction-only re-entry visibility calibrator robustly improves AJ_RD while preserving standard AJ.
```

Supporting methods:

```text
B2-W16-P2: original full local override baseline
Guard / Guard-Vis: learned diagnostic extension
candidate-pool oracle: headroom analysis
TrackOn2/TAPNext: appendix feasibility only
```
