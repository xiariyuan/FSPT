# RGB-Stacking 1-Video B2 Smoke — 2026-06-29

## Decision

The first RGB-Stacking cross-dataset smoke is positive. Using CoTracker3-offline as the base and CoTracker3-online as the override, the clean B2 runner improves both re-entry AJ_RD and standard TAP AJ on RGB-Stacking video 0.

This is not yet full cross-dataset validation, but it is an important proof that the B2 mechanism transfers beyond DAVIS in a 1-video smoke setting.

## 1-video teacher metrics

| method | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 | note |
|---|---:|---:|---:|---:|---|
| TAPNext | 0.0647 | 15.0884 | 30.6603 | 22.8275 | weak standalone on this RGB video |
| CoTracker3 offline | 0.6169 | 78.8736 | 88.2859 | 89.253 | strong standard base |
| CoTracker3 online | 0.6759 | 46.9765 | 59.7907 | 72.7586 | strong re-entry but weak standard AJ |
| B2 offline base + online override | 0.6949 | 80.2522 | 91.1507 | 89.626 | localized override |

## Key comparison

```text
CoTracker3 offline base:
  AJ_RD_256 = 0.6169
  AJ_256 = 78.8736

CoTracker3 online override:
  AJ_RD_256 = 0.6759
  AJ_256 = 46.9765

B2 localized override:
  AJ_RD_256 = 0.6949
  AJ_256 = 80.2522
```

B2 improves over the offline base by:

```text
AJ_RD_256 gain = 0.0780
AJ_256 gain = 1.3786
```

B2 also exceeds the online override AJ_RD while avoiding its standard-AJ collapse.

## B2 trigger behavior on RGB video 0

```text
tracks_with_trigger = 433
triggered_reentry_tracks = 372
triggered_nonreentry_tracks = 61
missed_reentry_tracks = 32
trigger_precision_track = 0.859122
trigger_recall_track = 0.920792
```

Compared with DAVIS, this RGB video has a much cleaner trigger regime: precision is high while recall remains high.

## Artifacts

```text
datasets/tapvid_rgb_stacking.py
scripts/export_tapnext_rgb_stacking_cache.py
scripts/export_cotracker_rgb_stacking_cache.py
outputs/paper_discovery_2026-06-27/rgb_stacking_teacher_smoke/
```

## Caveat

This is only one RGB-Stacking video. It is strong evidence that the pipeline works and that the B2 mechanism can transfer, but it is not enough for paper-level cross-dataset claims. Next step is 5-video RGB smoke with the same base/override pair.
