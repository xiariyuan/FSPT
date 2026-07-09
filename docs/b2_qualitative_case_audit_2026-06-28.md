# B2 Qualitative Case Export — 2026-06-28

## Decision

Qualitative case export is working. DAVIS raw frames are available from:

```text
/gemini/code/datasets/tapvid_davis/tapvid_davis.pkl
```

The export script renders panels with four trajectories:

```text
GT = green
fixed_offline = yellow
global B1 vis4 = cyan
B2 predicted = magenta
```

## Script

```text
scripts/export_b2_qualitative_cases.py
```

## Output directory

```text
outputs/paper_discovery_2026-06-27/teacher_expansion/b2_qualitative_cases/
```

Files:

```text
manifest.json
README.md
targeted_failure_manifest.json
images/*.png
```

## Exported case categories

### 1. B2 success / re-entry recovery

```text
success_reentry_recovery__car-roundabout__q1.png
success_reentry_recovery__bmx-trees__q0.png
success_reentry_recovery__dogs-jump__q13.png
```

Purpose: show fixed_offline weakness and B2 recovery behavior.

### 2. B2 avoids B1 global damage

```text
b2_avoids_b1_global_damage__drift-straight__q83.png
b2_avoids_b1_global_damage__soapbox__q267.png
b2_avoids_b1_global_damage__drift-straight__q80.png
```

Purpose: show why B2 is preferable to global B1 despite nearly matching B1 AJ_RD.

### 3. Automatically selected harmful false triggers

```text
harmful_false_trigger__dog__q15.png
harmful_false_trigger__bike-packing__q66.png
harmful_false_trigger__breakdance__q78.png
```

Purpose: show false-trigger failures selected by worst per-query B2-minus-fixed AJ drop.

### 4. Targeted harmful false-trigger videos from taxonomy

```text
targeted_harmful_false_trigger__drift-chicane__q81.png
targeted_harmful_false_trigger__pigs__q43.png
targeted_harmful_false_trigger__shooting__q47.png
targeted_harmful_false_trigger__soapbox__q105.png
targeted_harmful_false_trigger__drift-straight__q92.png
```

Purpose: focus on videos responsible for the largest standard-AJ drops in the false-trigger taxonomy.

## Interpretation

This export converts the B2 evidence from tables into inspectable visual cases. These cases should be used for:

```text
1. paper qualitative figures,
2. debugging harmful false triggers,
3. selecting representative success/failure examples,
4. explaining the B2 method story visually.
```

## Next step

Review exported PNGs and select 3-4 final paper figures:

```text
Figure A: fixed misses re-entry, B2 recovers.
Figure B: global B1 hurts ordinary tracking, B2 avoids it.
Figure C: harmful false trigger failure.
Figure D optional: trigger timing panel.
```

After figure selection, move to a clean B2 mainline runner and RGB-Stacking 1-video smoke.
