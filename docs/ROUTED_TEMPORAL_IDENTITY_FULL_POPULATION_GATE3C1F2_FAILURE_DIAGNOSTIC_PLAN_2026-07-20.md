# Route-D Gate 3C1F2 full-population failure diagnostic plan — 2026-07-20

## Status

```text
PREREGISTERED_EXPOSED_DIAGNOSTIC_NOT_RUN
no new external or raw-record-disjoint confirmation data
```

## Authorized evidence

Gate 3C1F2 completed on the now-exposed third 128-video Kubric population with fresh-process exact replay and the formal decision:

```text
STOP_BEFORE_OFFICIAL_TAPVID
```

Complete-video gains are approximately `+0.0014` AJ points, `+0.0937` delta_avg points, and `+0.1591` OA points. Delta_avg and OA paired-video CIs are positive; AJ is null. Among 51 actions with visible future GT, mean future error reduction is `+15.268 px`, `98.04%` are positive, and none is harmful by more than 4 px.

The diagnostic may use only the exposed Gate 3C1F2 128-video population and committed result records. It cannot authorize an external benchmark or a tracking claim.

## Primary question

Why do strong and safe action-level coordinate improvements produce nearly zero complete-video AJ gain?

The diagnostic tests two nonexclusive hypotheses:

1. **Coverage dilution:** useful actions affect too few complete-video evaluation terms.
2. **Coordinate–visibility coupling:** modified coordinates improve, but modified visibility/confidence changes Jaccard denominators or suppresses otherwise accurate positions.

## Frozen diagnostic population

Only the 44 Gate 3C1F2 videos with at least one sealed top-1 action are rerun. Their source indices are read from the committed exact-replay result. No no-action video is newly selected or excluded based on a new metric.

For every rerun video, the diagnostic must exactly reproduce the sealed Gate 3C1F2 digests for:

- eligible point indices;
- entry features, probabilities, and mask;
- candidate coordinates and temporal/static features;
- shortlist and candidate features;
- selected slot, top-1 action, and output candidate;
- native and modified coordinates;
- native and modified binary visibility.

Any mismatch stops the diagnostic before interpretation.

## Frozen trajectory views

The following complete-video views are evaluated without changing coordinates or logits:

```text
native:
  native coordinates + native predicted visibility

actual_modified:
  modified coordinates + modified predicted visibility

modified_coordinates_native_visibility:
  modified coordinates + native predicted visibility

native_coordinates_modified_visibility:
  native coordinates + modified predicted visibility

native_coordinates_gt_visibility_oracle:
  native coordinates + GT visibility

modified_coordinates_gt_visibility_oracle:
  modified coordinates + GT visibility
```

The two GT-visibility views are diagnostic oracles only. They cannot become runtime policies.

## Per-action frame decomposition

For every sealed action and affected frame 15--23, record:

- GT visibility;
- native and modified visibility probability, confidence probability, joint probability, and binary visibility;
- native and modified coordinate error when GT is visible;
- threshold hits at 1, 2, 4, 8, and 16 px;
- visibility transitions:
  - recovered GT-visible false negative;
  - newly introduced GT-visible false negative;
  - removed GT-occluded false positive;
  - newly introduced GT-occluded false positive;
- per-threshold Jaccard numerator and denominator contributions for native, actual modified, and the two coordinate/visibility cross views.

The query frame is excluded exactly as in the first-query TAP evaluator. Because all entry-eligible queries are before frame 8, all affected frames 15--23 are post-query.

## Frozen summaries

Report equal-video AJ, delta_avg, and OA for all six views, paired differences between:

```text
actual_modified - native
modified_coordinates_native_visibility - native
native_coordinates_modified_visibility - native
modified_coordinates_gt_visibility_oracle - native_coordinates_gt_visibility_oracle
```

Also report action/frame decompositions by:

- GT-defined category: failure, ambiguous, other;
- positive/negative/zero actual AJ video;
- future-visible versus no-future-visible action;
- native-to-modified visibility transition type.

No threshold sweep, model retraining, or post-hoc policy selection is part of this diagnostic.

## Interpretation boundary

A finding that visibility causes AJ loss may motivate a new visibility-coupled architecture or safety guard. A finding that coverage dominates may motivate a new entry objective. Either redesign must be trained and selected only on exposed data and then confirmed on a newly preregistered raw-record-disjoint population.

DAVIS, Kinetics, final holdout, and official Kinetics 1,144 remain locked.
