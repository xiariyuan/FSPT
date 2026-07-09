# RGB-Stacking 5-Video B2 Smoke — 2026-06-29

## Decision

The RGB-Stacking 5-video smoke is positive. B2 improves re-entry AJ_RD over the CoTracker3-offline base while preserving almost all standard AJ.

This is stronger than the 1-video smoke because it covers five RGB-Stacking videos and 5752 strided queries.

## Metrics

| method | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 | note |
|---|---:|---:|---:|---:|---|
| CoTracker3 offline | 0.4088 | 79.5438 | 89.9776 | 89.0096 | standard-strong base |
| CoTracker3 online | 0.4799 | 44.7452 | 56.7666 | 72.5164 | re-entry stronger, standard weaker |
| B2 offline base + online override | 0.4968 | 79.0152 | 91.69 | 88.9884 | localized override |

## Gains

```text
B2 vs offline AJ_RD_256 = +0.0880
B2 vs offline AJ_256 = -0.5286
B2 vs online AJ_RD_256 = +0.0169
B2 vs online AJ_256 = +34.2700
```

## Trigger behavior

```text
tracks_with_trigger = 2067
triggered_reentry_tracks = 1453
triggered_nonreentry_tracks = 614
missed_reentry_tracks = 88
trigger_precision = 0.702951
trigger_recall = 0.942894
```

## Interpretation

The 5-video result supports the main B2 story beyond DAVIS:

```text
offline base: high standard AJ, weaker re-entry
online override: better re-entry, poor standard AJ
B2: localized override improves re-entry while preserving base standard AJ
```

B2 does not improve standard AJ over offline on the 5-video aggregate, but the AJ drop is small while AJ_RD improves substantially.

## Caveat

This is still a smoke validation, not full RGB-Stacking validation. It covers 5 / 50 videos. The next step is either 10-video / 20-video scaling or adding TrackOn2 / B1 fusion for a closer DAVIS-like teacher pool.
