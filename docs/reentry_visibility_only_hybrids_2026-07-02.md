# ReEntry Visibility-Only Hybrids — 2026-07-02

## Motivation

The coordinate-vs-visibility diagnosis showed that, under the current ReEntry-TAP stress settings, re-entry coordinates are often already correct while predicted visibility lags behind.

Therefore we tested a simple visibility-only variant:

```text
Keep offline/base coordinates unchanged.
Replace predicted visibility with B2 / Guard / online visibility.
```

This tests whether re-entry recovery is primarily a visible-state recovery problem rather than a coordinate re-localization problem.

## Method variants

For each setting, we evaluated:

```text
offline
online_global
B2-W16-P2
ReEntry-Guard RF thr0.40, where available
offline_coord_b2_vis
offline_coord_guard_vis
offline_coord_online_vis
```

Key hybrids:

```text
offline_coord_b2_vis:
  coordinates = offline
  visibility  = B2-W16-P2

offline_coord_guard_vis:
  coordinates = offline
  visibility  = ReEntry-Guard
```

These are not oracle methods; they use predictions only.

---

## RGB fresh20-49 natural

| Method | AJ_RD_256 | AJ_256 | OA_256 |
|---|---:|---:|---:|
| offline | 0.3816 | 79.5944 | 91.4636 |
| online_global | 0.4121 | 44.6933 | 55.7585 |
| B2-W16-P2 | 0.4454 | 79.0664 | 92.8804 |
| ReEntry-Guard RF | 0.4499 | 79.1749 | 92.0864 |
| offline_coord_b2_vis | 0.4505 | 79.1111 | 92.8804 |
| offline_coord_guard_vis | **0.4513** | **79.2045** | 92.0864 |
| offline_coord_online_vis | 0.4282 | 45.7904 | 55.7585 |

Best under AJ-loss <= 1 point:

```text
offline_coord_guard_vis: AJ_RD=0.4513, AJ=79.2045
```

This improves over ReEntry-Guard:

```text
AJ_RD +0.0014
AJ    +0.0296
```

and over B2:

```text
AJ_RD +0.0059
AJ    +0.1381
```

---

## fresh20-49 translate_L16

| Method | AJ_RD_256 | AJ_256 | OA_256 |
|---|---:|---:|---:|
| offline | 0.4788 | 75.1940 | 90.7805 |
| online_global | 0.4981 | 43.0978 | 56.7847 |
| B2-W16-P2 | 0.5310 | 74.8457 | 92.2166 |
| ReEntry-Guard RF | 0.5336 | 74.8490 | 91.4654 |
| offline_coord_b2_vis | 0.5333 | 74.7709 | 92.2166 |
| offline_coord_guard_vis | **0.5336** | **74.8618** | 91.4654 |
| offline_coord_online_vis | 0.5133 | 44.0643 | 56.7847 |

Best under AJ-loss <= 1 point:

```text
offline_coord_guard_vis: AJ_RD=0.5336, AJ=74.8618
```

This matches ReEntry-Guard AJ_RD and slightly improves AJ:

```text
AJ_RD +0.0000
AJ    +0.0128
```

---

## fresh20-49 occluder_L16

| Method | AJ_RD_256 | AJ_256 | OA_256 |
|---|---:|---:|---:|
| offline | 0.6311 | 77.6280 | 90.8918 |
| online_global | 0.6146 | 43.1490 | 58.0983 |
| B2-W16-P2 | 0.6588 | 76.7991 | 92.1682 |
| ReEntry-Guard RF | 0.6632 | 77.0761 | 92.0632 |
| offline_coord_b2_vis | **0.6658** | 77.0422 | 92.1682 |
| offline_coord_guard_vis | 0.6655 | **77.1760** | 92.0632 |
| offline_coord_online_vis | 0.6469 | 44.7880 | 58.0983 |

Best under AJ-loss <= 1 point:

```text
offline_coord_b2_vis: AJ_RD=0.6658, AJ=77.0422
```

This improves over B2:

```text
AJ_RD +0.0070
AJ    +0.2431
```

and improves over ReEntry-Guard in AJ_RD:

```text
AJ_RD +0.0026
AJ    -0.0339
```

---

## dev translate_L16

| Method | AJ_RD_256 | AJ_256 | OA_256 |
|---|---:|---:|---:|
| offline | 0.4708 | 75.8656 | 90.7947 |
| online_global | 0.5194 | 43.9771 | 57.0539 |
| B2-W16-P2 | 0.5485 | 75.8344 | 92.6729 |
| offline_coord_b2_vis | **0.5506** | 75.5666 | 92.6729 |
| offline_coord_online_vis | 0.5298 | 44.6282 | 57.0539 |

## dev occluder_L16

| Method | AJ_RD_256 | AJ_256 | OA_256 |
|---|---:|---:|---:|
| offline | 0.6422 | 77.6880 | 90.6437 |
| online_global | 0.6477 | 43.6800 | 58.2391 |
| B2-W16-P2 | 0.6873 | 77.1682 | 92.2855 |
| offline_coord_b2_vis | **0.6940** | 77.2521 | 92.2855 |
| offline_coord_online_vis | 0.6774 | 45.2512 | 58.2391 |

---

## Interpretation

This is a strong diagnostic and method result.

The best visibility-only hybrids equal or exceed ReEntry-Guard while keeping coordinates untouched:

```text
RGB natural:
ReEntry-Guard          0.4499 / 79.1749
offline_coord_guard_vis 0.4513 / 79.2045

fresh translate:
ReEntry-Guard          0.5336 / 74.8490
offline_coord_guard_vis 0.5336 / 74.8618

fresh occluder:
ReEntry-Guard          0.6632 / 77.0761
offline_coord_b2_vis    0.6658 / 77.0422
offline_coord_guard_vis 0.6655 / 77.1760
```

The key conclusion:

```text
Re-entry recovery in this setting is largely a visibility-state recovery problem.
Coordinates from the offline/base branch are often already good enough; replacing coordinates is unnecessary and sometimes harmful.
```

## Method implication

The stronger method should be reframed as:

```text
Visibility-Calibrated Re-entry Recovery
```

or:

```text
ReEntry-Visibility Guard
```

Core principle:

```text
Preserve base coordinates.
Recover / calibrate visibility only in re-entry windows.
```

This gives a cleaner and more defensible structure than full coordinate override.

## Recommended new main method variant

Candidate method name:

```text
ReEntry-VisGuard
```

Instantiations:

```text
B2-Vis:
  coordinates = offline
  visibility  = B2-W16-P2

Guard-Vis:
  coordinates = offline
  visibility  = ReEntry-Guard
```

Current best choices:

```text
natural / translate: Guard-Vis
occluder: B2-Vis or Guard-Vis depending AJ_RD vs AJ preference
```

## Next step

Train or design a runtime visibility calibrator that predicts re-entry visibility directly instead of selecting full trajectory candidates.

This should use:

```text
base visibility
online/B2 visibility
visibility persistence
trigger timing
coordinate stability
base/online disagreement
optional appearance identity evidence
```

Output:

```text
calibrated visibility sequence
```

Coordinates should remain offline/base unless a separate coordinate-failure detector fires.
