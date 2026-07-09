# Comparability Audit — 2026-06-27

## Verdict

The repository contains enough data and evaluation infrastructure to compare against TAP-Vid SOTA **if** we use the correct baseline/evaluation protocol.

However, the recent ensemble-distillation/student experiments are **not** SOTA-comparable experiments. They are DAVIS smoke/distillation runs and should not be used as the project-level performance number.

## Available local datasets

- TAP-Vid DAVIS: `/gemini/code/datasets/tapvid_davis/tapvid_davis.pkl`, 2.311 GB
- TAP-Vid Kinetics: `/gemini/code/datasets/tapvid_kinetics`, 10 shards, 1144 train samples, about 23 GB
- TAP-Vid RGB-Stacking: `/gemini/code/datasets/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl`, 2.295 GB
- TAP-Vid Kubric TFDS/preprocessed: `/gemini/code/datasets/tapvid_kubric`, about 287 GB

## Evaluation infrastructure

- `datasets/tapvid_official_eval.py` vendors the official TAP-Vid metric core.
- `datasets/metrics.py` wraps it and converts internal `[y,x]` normalized coordinates to official raster `[x,y]` coordinates.
- GT-as-prediction sanity check passed: AJ=1.0 / OA=1.0.

## Protocols found

### Comparable route already present

`outputs/attempt0_2026-06-15_recovery/final_decision.md` reports a working first-query + input-space bridge:

| Baseline | DAVIS AJ | DAVIS delta_avg | DAVIS OA |
|---|---:|---:|---:|
| Track-On2 DINOv3 | 67.04 | 79.84 | 92.09 |
| CoTracker3 online | 64.89 | 77.36 | 91.80 |
| CoTracker3 offline | 62.66 | 77.22 | 88.15 |

It also states the unified bridge has 0.00 diff vs repo-native for the three runnable baselines.

### Not-yet-complete route

The same document states that true `strided + original` Attempt 0 main protocol is still pending.

### Recent non-comparable student route

The recent configs under `fspt_distill_ensemble_masked_median_*` use:

- train dataset: TAP-Vid DAVIS
- val dataset: TAP-Vid DAVIS
- 30-video DAVIS smoke setting
- low-memory 256x256 student training
- teacher/base tracks used as supervision or cache, not as official SOTA baseline

These are useful for diagnosis, not for SOTA comparison.

## Main conclusion

Do not report the 9.8% student AJ as the repository's SOTA-comparable result.

For SOTA comparison, use the Attempt 0 baseline bridge / repo-native baseline path, then complete the missing `strided + original` protocol if that is the intended paper protocol.

## Next required action

1. Freeze recent student-distillation results as diagnostic only.
2. Promote `outputs/attempt0_2026-06-15_recovery` as the SOTA-comparability baseline evidence.
3. Run or finish true `strided + original` baseline evaluation for Track-On2 / CoTracker3.
4. Only then compare our proposed module against the strongest runnable baseline.
