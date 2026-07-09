# ReEntry Timeline Figures Generated — 2026-07-03

## Purpose

Timeline figures were generated to provide qualitative evidence without requiring raw RGB frames. Each figure shows:

```text
GT visibility
Base visibility
Rule W8P2 visibility
Ours-Det visibility
Ours-Learned visibility
preserved base coordinate error in 256-space
```

Color coding:

```text
green = correct visible
red   = false visible
blue  = missed visible
gray  = invisible / true negative
```

Renderer:

```text
scripts/render_reentry_timeline_panels.py
```

Output directory:

```text
paper/reentry_viscalibrator_tex/figures/
```

## Generated figures

### 1. Base failure, learned success

```text
paper/reentry_viscalibrator_tex/figures/qual_timeline_natural_base_fail_learned_success.png
paper/reentry_viscalibrator_tex/figures/qual_timeline_natural_base_fail_learned_success.json
```

Case:

```text
video_id: rgb_stacking_000031
query_idx: 180
query_t: 30
reentry_t: 51
frames: 39--63
```

AJ_RD:

```text
Base:       0.0000
Rule W8P2: 0.8333
Ours-Det:  0.8333
Ours-Learned: 1.0000
```

### 2. Deterministic over-recovery, learned stability

```text
paper/reentry_viscalibrator_tex/figures/qual_timeline_natural_det_over_recovery.png
paper/reentry_viscalibrator_tex/figures/qual_timeline_natural_det_over_recovery.json
```

Case:

```text
video_id: rgb_stacking_000031
query_idx: 96
query_t: 15
reentry_t: 51
frames: 39--63
```

AJ_RD:

```text
Base:       0.0000
Rule W8P2: 0.1897
Ours-Det:  0.1897
Ours-Learned: 0.7417
```

This is the most important learned-vs-deterministic qualitative case.

### 3. Natural failure case

```text
paper/reentry_viscalibrator_tex/figures/qual_timeline_natural_failure_short_occ.png
paper/reentry_viscalibrator_tex/figures/qual_timeline_natural_failure_short_occ.json
```

Case:

```text
video_id: rgb_stacking_000047
query_idx: 338
query_t: 60
reentry_t: 63
frames: 51--75
```

AJ_RD:

```text
Base:       1.0000
Rule W8P2: 0.2143
Ours-Det:  0.2118
Ours-Learned: 0.2118
```

This shows a short-occlusion failure where recovery is unnecessary.

### 4. Occluder success

```text
paper/reentry_viscalibrator_tex/figures/qual_timeline_occluder_success.png
paper/reentry_viscalibrator_tex/figures/qual_timeline_occluder_success.json
```

Case:

```text
video_id: rgb_stacking_000031_occluder_L16
query_idx: 209
query_t: 35
reentry_t: 48
frames: 36--60
```

AJ_RD:

```text
Base:       0.0000
Rule W8P2: 0.8889
Ours-Det:  0.8889
Ours-Learned: 1.0000
```

### 5. Learned preserves base / avoids deterministic degradation

```text
paper/reentry_viscalibrator_tex/figures/qual_timeline_occluder_learned_preserves_base.png
paper/reentry_viscalibrator_tex/figures/qual_timeline_occluder_learned_preserves_base.json
```

Case:

```text
video_id: rgb_stacking_000026_occluder_L16
query_idx: 183
query_t: 30
reentry_t: 92
frames: 80--104
```

AJ_RD:

```text
Base:       0.9145
Rule W8P2: 0.3134
Ours-Det:  0.3134
Ours-Learned: 0.9145
```

This shows that learned calibration can avoid unnecessary recovery and preserve a strong base prediction.

## LaTeX integration

Updated:

```text
paper/reentry_viscalibrator_tex/sections/06_qualitative_analysis.tex
```

The section now includes:

```text
Figure: qualitative timeline panels using three generated PNGs
Figure: failure case using one generated PNG
```

The TeX draft references:

```text
figures/qual_timeline_natural_det_over_recovery.png
figures/qual_timeline_occluder_success.png
figures/qual_timeline_occluder_learned_preserves_base.png
figures/qual_timeline_natural_failure_short_occ.png
```

The base-failure success figure is generated but not yet inserted into the main figure; it can be used in appendix or if the main figure needs replacement.
