# RGB-Stacking rgb_stacking_000008 Failure Audit — 2026-06-29

## Decision

The main `rgb_stacking_000008` failure is not simply that false triggers are expensive. The deeper issue is that the online override is often worse than the offline base on true re-entry tracks. B2 correctly triggers many re-entry tracks, but then switches to a weaker override for this video.

This explains why `rgb_stacking_000008` is the only RGB 10-video case where B2 loses large AJ_RD versus offline.

## Video-level facts

From the RGB 10-video per-video audit:

```text
offline AJ_RD_256 = 0.3989
online AJ_RD_256 = 0.2315
B2 AJ_RD_256 = 0.2378
B2 - offline AJ_RD_256 = -0.1611
B2 - offline AJ_256 = -1.0629
```

Trigger behavior:

```text
n_tracks = 1244
n_reentry = 57
n_triggers = 335
n_true_triggers = 56
n_false_triggers = 279
n_missed_reentry = 1
trigger_precision = 0.167164
trigger_recall = 0.982456
false_trigger_rate = 0.224277
```

## Key finding

```text
mean_delta_b2_offline_full_false = 0.009113
mean_delta_b2_offline_full_true = -0.162908
mean_delta_b2_offline_reentry_true = -0.16895
```

Interpretation:

- False triggers are numerous, but their average full-track delta is slightly positive in this video.
- True triggers are the damaging part: B2 switches true re-entry tracks to online, but online is worse than offline on many of those re-entry segments.
- Therefore the failure is an override-quality failure, not only a trigger-precision failure.

## Worst true re-entry override cases

| query | query_t | reentry_t | occ_len | offline reAJ | online reAJ | B2 reAJ | B2-offline reAJ | offline full AJ | B2 full AJ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 871 | 165 | 186 | 12 | 0.633333 | 0.226235 | 0.226235 | -0.407099 | 0.8433 | 0.6723 |
| 848 | 160 | 186 | 12 | 0.637736 | 0.231905 | 0.231905 | -0.405831 | 0.8406 | 0.6685 |
| 779 | 145 | 186 | 12 | 0.822841 | 0.41913 | 0.41913 | -0.40371 | 0.8613 | 0.6959 |
| 376 | 60 | 186 | 12 | 0.6 | 0.203226 | 0.203226 | -0.396774 | 0.8054 | 0.6220 |
| 756 | 140 | 186 | 12 | 0.810909 | 0.417241 | 0.417241 | -0.393668 | 0.8611 | 0.6976 |

These cases share the same pattern: offline is already good after re-entry, but B2 overwrites with online and loses re-entry AJ.

## Worst false triggers

| query | query_t | trigger_t | GT visible at trigger | offline full AJ | online full AJ | B2 full AJ | B2-offline full AJ |
|---:|---:|---:|---|---:|---:|---:|---:|
| 313 | 50 | 56 | True | 0.9643 | 0.0600 | 0.3733 | -0.5910 |
| 340 | 55 | 58 | False | 0.7946 | 0.0485 | 0.2777 | -0.5169 |
| 14 | 0 | 50 | True | 0.8276 | 0.4067 | 0.4067 | -0.4209 |
| 388 | 65 | 82 | True | 0.7273 | 0.1098 | 0.3536 | -0.3737 |
| 44 | 5 | 49 | True | 0.8103 | 0.4761 | 0.4761 | -0.3342 |

False-trigger failures exist and should be visualized, but their mean effect is not the main video-level loss.

## Rendered qualitative cases

Images exported to:

```text
outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/rgb_000008_failure_audit/images/
```

Representative images:

- `bad_true_reentry_override` q=871: `outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/rgb_000008_failure_audit/images/bad_true_reentry_override__rgb_stacking_000008__q871.png`
- `bad_true_reentry_override` q=848: `outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/rgb_000008_failure_audit/images/bad_true_reentry_override__rgb_stacking_000008__q848.png`
- `bad_true_reentry_override` q=779: `outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/rgb_000008_failure_audit/images/bad_true_reentry_override__rgb_stacking_000008__q779.png`
- `bad_false_trigger` q=313: `outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/rgb_000008_failure_audit/images/bad_false_trigger__rgb_stacking_000008__q313.png`
- `bad_false_trigger` q=340: `outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/rgb_000008_failure_audit/images/bad_false_trigger__rgb_stacking_000008__q340.png`
- `bad_false_trigger` q=14: `outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/rgb_000008_failure_audit/images/bad_false_trigger__rgb_stacking_000008__q14.png`
- `good_true_reentry_recovery` q=308: `outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/rgb_000008_failure_audit/images/good_true_reentry_recovery__rgb_stacking_000008__q308.png`
- `good_true_reentry_recovery` q=34: `outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/rgb_000008_failure_audit/images/good_true_reentry_recovery__rgb_stacking_000008__q34.png`
- `good_true_reentry_recovery` q=4: `outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/rgb_000008_failure_audit/images/good_true_reentry_recovery__rgb_stacking_000008__q4.png`

## Implication for next method step

A simple trigger based only on base invisible + override visible is not enough for full RGB scaling. The next improvement should be a lightweight override-quality gate, not another blind trigger sweep.

Possible low-risk gates to test before scaling to 20/50 videos:

```text
1. Only override if online has higher local visibility confidence / persistence than base.
2. Require online and offline disagreement not to exceed a loose threshold when base remains stable.
3. Add a per-video or per-track guard: if offline remains visible around the re-entry segment, avoid full post override.
4. Use B2 persist=2-like conservative trigger for RGB only and compare Pareto.
```

## Artifacts

```text
scripts/audit_rgb_stacking_000008_failure.py
outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/rgb_000008_failure_audit/summary.json
outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/rgb_000008_failure_audit/track_rows.jsonl
```
