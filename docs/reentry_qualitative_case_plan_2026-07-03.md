# ReEntry Qualitative Case Plan — 2026-07-03

## Purpose

This document selects representative qualitative cases from the mined case lists. These cases should be rendered into paper figures later.

Generated case mining files:

```text
scripts/mine_reentry_qualitative_cases.py
outputs/paper_discovery_2026-06-27/reentry_qualitative_cases/rgb_fresh20_49_natural.json
docs/reentry_qualitative_cases_rgb_fresh20_49_natural_2026-07-03.md
outputs/paper_discovery_2026-06-27/reentry_qualitative_cases/fresh20_49_translate_L16.json
docs/reentry_qualitative_cases_fresh20_49_translate_L16_2026-07-03.md
outputs/paper_discovery_2026-06-27/reentry_qualitative_cases/fresh20_49_occluder_L16.json
docs/reentry_qualitative_cases_fresh20_49_occluder_L16_2026-07-03.md
```

The case miner reports:

```text
natural:       6461 re-entry query items
translate_L16: 8104 re-entry query items
occluder_L16:  14020 re-entry query items
```

---

## Recommended paper figure layout

A strong qualitative figure should show 4 rows:

```text
Row 1: Base failure and Ours-Learned recovery
Row 2: Ours-Det over-recovers, Ours-Learned is more stable
Row 3: Stress case success, preferably translate or occluder
Row 4: Failure case / limitation
```

Each row should show 5--7 frames around re-entry:

```text
query frame
last visible / disappearance frame
just before re-entry
re-entry frame
post re-entry frames
```

For each method, overlay:

```text
GT point / GT visibility
Base prediction
Ours-Det prediction
Ours-Learned prediction
```

If visual clutter is high, use separate small panels per method.

---

## Case A: Natural base-fail, learned-success

Recommended candidate:

```text
video_id: rgb_stacking_000031
query_idx: 180
query_t: 30
reentry_t: 51
occ_length: 12
segment frames: [39, 63]
```

Metrics:

```text
Base AJ_RD:       0.0000
Rule W8P2 AJ_RD: 0.8333
Ours-Det AJ_RD:  0.8333
Ours-Learned:    1.0000
Δ Learned - Base: +1.0000
Δ Learned - Det:  +0.1667
```

Why this is useful:

```text
It shows the ideal story: Base misses re-entry; deterministic recovery helps; learned calibration is best.
```

Potential caption:

```text
Base remains conservative after re-entry, while the learned calibrator recovers visibility and preserves the base coordinates, producing perfect re-entry AJ on this query.
```

---

## Case B: Natural Ours-Det over-recovers, learned is more stable

Recommended candidate:

```text
video_id: rgb_stacking_000031
query_idx: 96
query_t: 15
reentry_t: 51
occ_length: 12
segment frames: [39, 63]
```

Metrics:

```text
Base AJ_RD:       0.0000
Rule W8P2 AJ_RD: 0.1897
Ours-Det AJ_RD:  0.1897
Ours-Learned:    0.7417
Δ Learned - Base: +0.7417
Δ Learned - Det:  +0.5520
```

Segment diagnostic:

```text
Ours-Det false-visible frames: 16
Ours-Learned false-visible frames: 12
Ours-Det missed-visible frames: 0
Ours-Learned missed-visible frames: 0
```

Why this is useful:

```text
This is the clearest visual story for the learned module: the deterministic recovery is too permissive, while learned calibration reduces false-visible frames and improves AJ_RD.
```

Potential caption:

```text
The deterministic variant opens too many frames in the candidate window. The learned calibrator suppresses part of this over-recovery while preserving the useful re-entry recovery.
```

---

## Case C: Translate stress success

Recommended candidate:

```text
video_id: rgb_stacking_000041_translate_L16
query_idx: 120
query_t: 20
reentry_t: 69
occ_length: 27
segment frames: [57, 81]
```

Metrics:

```text
Base AJ_RD:       0.0000
Rule W8P2 AJ_RD: 0.9803
Ours-Det AJ_RD:  0.9803
Ours-Learned:    0.9803
Δ Learned - Base: +0.9803
```

Why this is useful:

```text
It shows the stress-setting story: the point leaves/returns under translation, and visibility recovery restores re-detection.
```

Alternative translate learned-vs-det candidate:

```text
video_id: rgb_stacking_000035_translate_L16
query_idx: 49
query_t: 5
reentry_t: 65
occ_length: 51
Base AJ_RD: 0.9085
Ours-Det AJ_RD: 0.5309
Ours-Learned AJ_RD: 0.8356
```

This alternative shows learned calibration recovering from a bad deterministic intervention, but the Base is still higher. Use it only if the figure is about avoiding over-recovery rather than beating Base.

---

## Case D: Occluder stress success

Recommended candidate:

```text
video_id: rgb_stacking_000031_occluder_L16
query_idx: 209
query_t: 35
reentry_t: 48
occ_length: 8
segment frames: [36, 60]
```

Metrics:

```text
Base AJ_RD:       0.0000
Rule W8P2 AJ_RD: 0.8889
Ours-Det AJ_RD:  0.8889
Ours-Learned:    1.0000
Δ Learned - Base: +1.0000
Δ Learned - Det:  +0.1111
```

Segment diagnostic:

```text
Ours-Det false-visible frames: 1
Ours-Learned false-visible frames: 0
Ours-Det missed-visible frames: 2
Ours-Learned missed-visible frames: 2
```

Why this is useful:

```text
It shows an occlusion-driven re-entry where the learned module is best and slightly cleaner than deterministic recovery.
```

---

## Case E: Occluder deterministic over-recovery / learned stability

Recommended candidate:

```text
video_id: rgb_stacking_000026_occluder_L16
query_idx: 183
query_t: 30
reentry_t: 92
occ_length: 15
segment frames: [80, 104]
```

Metrics:

```text
Base AJ_RD:       0.9145
Rule W8P2 AJ_RD: 0.3134
Ours-Det AJ_RD:  0.3134
Ours-Learned:    0.9145
Δ Learned - Det: +0.6011
```

Why this is useful:

```text
This is a strong limitation/benefit case: deterministic recovery badly hurts a case where Base was already good, while learned calibration avoids that damage and preserves the Base-quality result.
```

Potential caption:

```text
When the base stream is already correct, deterministic recovery can introduce unnecessary visibility changes. The learned calibrator preserves the base behavior and avoids the degradation.
```

---

## Case F: Natural failure case

Recommended candidate:

```text
video_id: rgb_stacking_000047
query_idx: 338
query_t: 60
reentry_t: 63
occ_length: 1
segment frames: [51, 75]
```

Metrics:

```text
Base AJ_RD:       1.0000
Rule W8P2 AJ_RD: 0.2143
Ours-Det AJ_RD:  0.2118
Ours-Learned:    0.2118
Δ Learned - Base: -0.7882
```

Why this is useful:

```text
This shows a failure mode: very short occlusion / ambiguous re-entry where recovery logic harms an already-correct Base prediction.
```

Potential discussion:

```text
Short occlusion events remain difficult because the candidate trigger can fire even when the base stream already handled the event correctly. Future work can add a stronger window-level confidence gate.
```

---

## Case G: Occluder failure family

Recommended candidate:

```text
video_id: rgb_stacking_000024_occluder_L16
query_idx: 309
query_t: 55
reentry_t: 75
occ_length: 16
segment frames: [63, 87]
```

Metrics:

```text
Base AJ_RD:       1.0000
Rule W8P2 AJ_RD: 0.1311
Ours-Det AJ_RD:  0.1270
Ours-Learned:    0.1270
Δ Learned - Base: -0.8730
```

Why this is useful:

```text
This is a stronger failure example under occluder stress. The base stream is already excellent, while recovery hurts. It supports the limitation that candidate-level gating remains imperfect.
```

---

## Failure taxonomy from mined cases

The mined cases suggest at least four failure types:

### 1. Base already correct, recovery unnecessary

Examples:

```text
rgb_stacking_000047 q=338
rgb_stacking_000024_occluder_L16 q=309
```

Symptom:

```text
Base AJ_RD is high, but recovery versions reduce AJ_RD.
```

Likely cause:

```text
candidate trigger fires in short or ambiguous events even when base handled the event.
```

Possible fix:

```text
add window-level confidence / no-action classifier.
```

### 2. Deterministic over-recovery

Examples:

```text
rgb_stacking_000031 q=96
rgb_stacking_000026_occluder_L16 q=183
```

Symptom:

```text
Ours-Det opens too many frames; learned calibration is much better.
```

Possible fix already partly handled by learned V1:

```text
learned frame-level recovery suppresses some harmful frames.
```

### 3. Learned too conservative

Examples:

```text
rgb_stacking_000025 q=691
rgb_stacking_000041_occluder_L16 q=159
```

Symptom:

```text
Ours-Det recovers, learned remains low or misses visible frames.
```

Possible fix:

```text
train with metric-aware reward that values true re-entry recovery more strongly.
```

### 4. Very short occlusion instability

Examples:

```text
occ_length = 1 cases in natural and occluder failure lists.
```

Symptom:

```text
short invisible events trigger recovery but may not need intervention.
```

Possible fix:

```text
explicitly condition on occlusion duration / require stronger trigger for short events.
```

---

## Recommended next rendering actions

1. Try to render Case B first:

```text
rgb_stacking_000031, query_idx=96, frames 39--63
```

Reason: it best shows learned-vs-deterministic value.

2. Render Case D:

```text
rgb_stacking_000031_occluder_L16, query_idx=209, frames 36--60
```

Reason: clean occluder success, learned best.

3. Render Case E:

```text
rgb_stacking_000026_occluder_L16, query_idx=183, frames 80--104
```

Reason: learned avoids deterministic degradation.

4. Render Case F or G as a limitation figure.

---

## Paper integration suggestion

Add one qualitative figure in the main paper:

```text
Figure X: Qualitative re-entry visibility recovery.
Rows: Base failure success, deterministic over-recovery, occluder success, learned failure.
```

Add failure taxonomy in discussion or appendix.

The case miner provides video/query/frame IDs for reproducible figure generation.
