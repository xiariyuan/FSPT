# CoTracker3 Online V5 Candidate-Search Oracle Plan — 2026-07-06

## Decision after V3/V4

V3/V4 visibility/state-writeback is structurally correct but has too little headroom:

```text
current re-entry-adjacent candidate pool perfect oracle: AJ_RD +0.0011 only
```

Therefore, do not keep tuning visibility writeback.

## V5 question

Can active re-detection candidate search provide enough headroom on CoTracker3 true-streaming online outputs?

## Diagnostic only

This stage may use GT to define/evaluate re-entry events and oracle-select candidates. It is not a deployable method and cannot be used as a main result.

## Smoke scope

Run first 3 DAVIS videos only.

Baseline:

```text
outputs/paper_discovery_2026-07-05/cotracker3_true_streaming_v3_decoupled_subset_eval/full30_overlap_soft_tau055_w2_j4_min1_confirm4_p081/cotracker3_true_streaming_native_subset.pt
```

## Candidate search design for smoke

For each GT-visible frame where native CoTracker3 predicts invisible or has low score near a re-entry phase:

1. Build support memory from prior high-confidence native-visible frames.
2. Extract a DINO patch descriptor for the support frame.
3. Search a local grid around the native predicted coordinate in the candidate frame.
4. Score each grid point by DINO cosine similarity to support.
5. Keep top-k candidates.
6. Use GT only for oracle audit:
   - top-k recall@4/8/16 px
   - oracle replacement metric upper bound

## Search parameters

First smoke:

```text
videos = first 3
radius = 32 px
stride = 4 px
topk = 10
crop_size = 33
support score threshold = 0.80
max events per video = capped for speed
```

## Success gate

Continue only if:

```text
top5 recall@8px clearly beats native at these events
oracle AJ_RD gain >= +0.005 on smoke
no catastrophic AJ/OA drop in oracle replacement
```

If not, CoTracker3 online active re-detection should not be prioritized.
