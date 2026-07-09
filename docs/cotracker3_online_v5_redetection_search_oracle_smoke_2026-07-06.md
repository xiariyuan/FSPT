# CoTracker3 Online V5 Re-Detection Search Oracle Smoke — 2026-07-06

## Purpose

After V3/V4 showed that visibility-only state writeback has too little headroom, this smoke tests whether an active re-detection candidate search has usable upper-bound signal.

This is not a method result. It is a candidate-pool diagnostic.

## Script

```text
scripts/audit_cotracker3_online_redetection_search_oracle.py
```

Output:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_redetection_search_oracle/smoke3_stride8.json
```

## Setting

- Base cache: true-streaming CoTracker3 native full30 cache.
- Dataset: TAP-Vid DAVIS.
- First 3 records, up to 80 GT re-entry events.
- Candidate search: DINO patch embedding dense grid search.
- Grid stride: 8 px.
- Patch size: 33 px.
- References: query patch, last native-visible patch, last strict high-confidence visible patch.
- Candidate score: max DINO cosine similarity to references.
- GT is used only for event selection and recall audit, not for candidate scoring.

## Key result

All 80 evaluated GT re-entry events:

| Metric | Top1 | Top5 | Top10 | Top20 | Top50 |
|---|---:|---:|---:|---:|---:|
| recall@4px | 22.5% | 36.25% | 50.0% | 55.0% | 63.75% |
| recall@8px | 42.5% | 68.75% | 81.25% | 87.5% | 91.25% |
| recall@16px | 73.75% | 87.5% | 90.0% | 92.5% | 97.5% |

Native-invisible re-entry events only:

```text
n = 6
native visible rate = 0
native error mean = 1.99 px
nearest grid recall@8 = 100%
```

| Metric | Top1 | Top5 | Top10 | Top20 | Top50 |
|---|---:|---:|---:|---:|---:|
| recall@4px | 16.67% | 50.0% | 50.0% | 66.67% | 83.33% |
| recall@8px | 16.67% | 66.67% | 66.67% | 83.33% | 83.33% |
| recall@16px | 66.67% | 83.33% | 83.33% | 83.33% | 83.33% |

## Interpretation

This is the first positive signal after V3/V4:

```text
Active DINO candidate search has meaningful top-k recall on re-entry frames.
```

But this is not yet an end-to-end improvement:

```text
1. The smoke used GT to select re-entry event frames.
2. It did not yet implement an online trigger.
3. It did not yet replace coordinates or retrack the tail.
4. It did not yet train a verifier to choose among top-k candidates.
5. The first 3 videos are not enough for a paper result.
```

## Design implication

Continue CoTracker3 online only if the method is reframed as:

```text
support-conditioned online re-detection + verifier + retracking
```

Do not continue tuning visibility-only state writeback.

## Next experiment

Run a larger candidate-search oracle:

```text
1. Use 10 or 30 videos.
2. Keep grid stride 8 if runtime is acceptable; otherwise compare stride 8 vs 16.
3. Report top-k recall@4/8/16 separately for:
   - all GT re-entry events
   - native-invisible events
   - native-visible but high-error events
   - long-occlusion events
4. If top5/top10 recall@8 remains strong, build candidate replacement oracle.
```

Success gate for continuing:

```text
top5 recall@8 >= 50% on native-invisible or high-error re-entry events
and enough events exist outside the first 3 videos.
```
