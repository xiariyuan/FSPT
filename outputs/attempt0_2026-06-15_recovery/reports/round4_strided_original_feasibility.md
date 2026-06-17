# Round 4: strided + original Feasibility Assessment (2026-06-15)

## Question 1: Can current .npz caches support true strided + original export?

**Answer: No.**

The current repo-native `.npz` caches were generated with `queried_first=True` and `resize_to_256=True`. This means:

1. **Query protocol**: first-query (one query per track at the first visible frame)
2. **Resolution space**: 256×256 input space (all coordinates are in 0–255 pixel range)

True Attempt 0 main protocol requires:

1. **Query protocol**: strided (queries sampled every `query_stride=5` frames)
2. **Resolution space**: original video resolution (e.g., 480×854 for DAVIS)

The `.npz` caches store only the model predictions — they do NOT contain the query protocol or resolution semantics. The predictions were generated under the `first + input` conditions:

- The model received queries at frame 0 (first visible frame only)
- The model processed 256×256 video frames
- The model outputs predictions in 256×256 pixel space

To generate predictions for `strided + original`:
1. The dataloader must sample strided queries (every 5 frames)
2. The model must process original-resolution video (or at least be re-informed of the original resolution)
3. The model predictions must be in original-resolution pixel space

**Current `.npz` caches cannot support this** — they contain only `first-query + 256-input` predictions.

## Question 2: Must we re-run the predictor?

**Answer: Yes, for true strided + original.**

The predictor must be re-run with:
- `queried_first=False` (strided queries)
- `resize_to_256=None` or `resize_to_256=False` (original resolution)

This requires:
1. GPU inference time (estimated ~30-60s per model per dataset for DAVIS)
2. Correct dataloader configuration for each predictor
3. Proper handling of variable-resolution inputs

## Recommendation

For Round 4, the **first + input** bridge is the correct stopping point. It demonstrates that:
1. The unified rescoring pipeline works correctly (Track-On2 shows 0.0 diff)
2. The normalization convention is consistent
3. The long-occ sub-metrics are computable

The `strided + original` upgrade should be a separate effort in a future round, requiring dedicated GPU time and predictor re-runs.

## Status

- [x] `first + input` bridge completed for all 3 baselines
- [ ] `strided + original` Attempt 0 main protocol — **NOT STARTED** (requires predictor re-runs)
