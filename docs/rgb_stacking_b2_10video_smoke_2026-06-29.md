# RGB-Stacking 10-Video B2 Smoke — 2026-06-29

## Decision

The RGB-Stacking 10-video smoke is positive. B2 improves re-entry AJ_RD over the CoTracker3-offline base while preserving nearly all standard AJ. This strengthens the cross-dataset evidence beyond the earlier 1-video and 5-video smoke tests.

## Metrics

| method | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 | note |
|---|---:|---:|---:|---:|---|
| CoTracker3 offline | 0.3763 | 80.0721 | 90.9956 | 88.9997 | standard-strong base |
| CoTracker3 online | 0.4522 | 45.4239 | 55.9164 | 73.1913 | re-entry stronger, standard weaker |
| B2 offline base + online override | 0.466 | 79.7562 | 92.9475 | 88.9484 | localized override |

## Gains

```text
B2 vs offline AJ_RD_256 = +0.0897
B2 vs offline AJ_256 = -0.3159
B2 vs online AJ_RD_256 = +0.0138
B2 vs online AJ_256 = +34.3323
```

## Trigger behavior

```text
tracks_with_trigger = 3594
triggered_reentry_tracks = 2283
triggered_nonreentry_tracks = 1311
missed_reentry_tracks = 127
trigger_precision = 0.635225
trigger_recall = 0.947303
```

## 1 / 5 / 10 video trend

| n videos | offline AJ_RD | offline AJ | online AJ_RD | online AJ | B2 AJ_RD | B2 AJ | B2 gain vs offline AJ_RD | B2 delta vs offline AJ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6169 | 78.8736 | 0.6759 | 46.9765 | 0.6949 | 80.2522 | +0.0780 | +1.3786 |
| 5 | 0.4088 | 79.5438 | 0.4799 | 44.7452 | 0.4968 | 79.0152 | +0.0880 | -0.5286 |
| 10 | 0.3763 | 80.0721 | 0.4522 | 45.4239 | 0.466 | 79.7562 | +0.0897 | -0.3159 |

## Interpretation

The cross-dataset trend is consistent:

```text
offline base: high standard AJ, weaker re-entry
online override: stronger re-entry, poor standard AJ
B2: improves re-entry over offline while keeping standard AJ close to offline
```

At 10 videos, B2 improves AJ_RD by +0.0897 over offline while losing only -0.3159 standard AJ. It also exceeds online AJ_RD and beats online standard AJ by +34.3323 points.

## Caveat

This is still a smoke validation over 10 / 50 RGB-Stacking videos and uses CoTracker3-online as the override rather than the full four-teacher DAVIS-style B1 branch. It is strong cross-dataset evidence for the B2 mechanism, but not final full-dataset validation.
