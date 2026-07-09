# Re-entry Recovery Diagnosis: Coordinates vs Visibility — 2026-07-02

## Goal

Before designing a heavier ReEntry-ID / memory-retrieval module, diagnose what actually limits AJ_RD under the current ReEntry-TAP dev stress protocol:

```text
1. coordinate recovery after re-entry
2. predicted visibility recovery after re-entry
```

## Settings

```text
dev_translate_L16
dev_occluder_L16
```

Compared methods:

```text
offline
online
B2-W16-P2
```

For each eligible re-entry event, we measured over the GT-visible re-entry segment:

```text
coord8_rate: fraction of GT-visible re-entry-segment frames whose predicted coordinate is within 8 px of GT, ignoring predicted visibility
predvis_recall: fraction of GT-visible re-entry-segment frames predicted visible
joint8_rate: fraction where predicted visible AND coordinate within 8 px
first_coord8_rate: same coordinate test at first re-entry frame
first_predvis_rate: predicted visible at first re-entry frame
```

Since GT is visible throughout the re-entry segment, joint8 is effectively the AJ-relevant visibility-and-coordinate success proxy.

---

## dev_translate_L16

```text
events = 5,139
frames = 277,186 GT-visible re-entry-segment frames
```

| Method | coord8_rate | predvis_recall | joint8_rate | first_coord8 | first_predvis |
|---|---:|---:|---:|---:|---:|
| offline | 1.0000 | 0.7180 | 0.7180 | 1.0000 | 0.3563 |
| online | 1.0000 | 0.8729 | 0.8729 | 1.0000 | 0.6649 |
| B2-W16-P2 | 1.0000 | 0.9010 | 0.9010 | 1.0000 | 0.6869 |

## dev_occluder_L16

```text
events = 7,523
frames = 591,554 GT-visible re-entry-segment frames
```

| Method | coord8_rate | predvis_recall | joint8_rate | first_coord8 | first_predvis |
|---|---:|---:|---:|---:|---:|
| offline | 1.0000 | 0.8356 | 0.8356 | 1.0000 | 0.4232 |
| online | 1.0000 | 0.9223 | 0.9223 | 1.0000 | 0.6870 |
| B2-W16-P2 | 1.0000 | 0.9428 | 0.9428 | 1.0000 | 0.7161 |

---

## Key finding

Under these dev stress settings, coordinate recovery at re-entry is not the bottleneck:

```text
coord8_rate = 1.0000 for offline, online, and B2
first_coord8_rate = 1.0000 for offline, online, and B2
```

The bottleneck is predicted visibility recovery:

```text
translate first_predvis:
offline 0.3563 -> B2 0.6869

occluder first_predvis:
offline 0.4232 -> B2 0.7161

translate segment predvis_recall:
offline 0.7180 -> B2 0.9010

occluder segment predvis_recall:
offline 0.8356 -> B2 0.9428
```

Therefore, much of the AJ_RD improvement from B2 is not from finding better coordinates at the first re-entry frame, but from recovering visibility / visible-state confidence over the re-entry segment.

## Implication for stronger innovation

The next stronger module should prioritize:

```text
Re-entry Visibility Recovery / Visibility Calibration
```

rather than immediately building a heavy global point-identity retrieval system.

A better next method may be:

```text
ReEntry-Visibility Guard:
  detect missing -> re-entry transition
  calibrate visibility using override branch persistence, base/override disagreement, and temporal consistency
  keep coordinates mostly from base/offline when already accurate
  only modify coordinates when a true coordinate failure is detected
```

This reframes the current method:

```text
B2-W16-P2 is mainly a visibility-recovery local override, with coordinate preservation.
```

## Design consequence

Strong innovation path should be adjusted:

```text
Old next idea:
  Point Identity Memory + dense re-detection + coordinate retrieval

Updated near-term idea:
  Re-entry state / visibility recovery module first
  Appearance identity only as auxiliary evidence for visibility calibration and false-visible prevention
```

This is more aligned with the measured bottleneck.
