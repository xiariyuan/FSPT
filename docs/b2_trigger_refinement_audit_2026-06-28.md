# B2 Trigger Refinement Audit — 2026-06-28

## Decision

The 648-config trigger refinement sweep does not replace the current B2 mainline. The current mainline remains the best AJ_RD point.

Current B2 mainline:

```text
name = refine_k1_pre1_post9999_cnt1_pers1_tau0
AJ_RD_256 = 0.6278
AJ_256 = 68.9702
trigger_precision_track = 0.529996
trigger_recall_track = 0.963177
```

Best near-mainline conservative alternative:

```text
name = refine_k1_pre1_post9999_cnt1_pers2_tau192
AJ_RD_256 = 0.6274
AJ_256 = 69.0245
trigger_precision_track = 0.542164
trigger_recall_track = 0.951625
```

More conservative ablation:

```text
name = refine_k1_pre1_post9999_cnt1_pers4_tau192
AJ_RD_256 = 0.6250
AJ_256 = 69.0996
trigger_precision_track = 0.555363
trigger_recall_track = 0.927076
```

## Interpretation

Persistence improves precision and standard AJ, but the gains are small and come with lower re-entry recall and lower AJ_RD. Existing output-only gates cannot produce a large second jump.

The current B2 mainline should remain the deployable method. `persist=2` and `persist=4` are useful Pareto / ablation variants, not replacements.

## Next step

Move from rule search to evidence consolidation: per-video stability, occlusion-length buckets, and trigger taxonomy.
