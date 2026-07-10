# V9-A5C.0 Candidate-Recall / Correlation-Map Oracle Audit Design

Date: 2026-07-10

## 1. Motivation

V9-A3.0 showed substantial oracle headroom inside TrackOn2's top-K candidate set, but V9-A3.1 through V9-A4.5 showed that every tested fixed-topK reranking route fails to transfer across videos or synthetic sequences.

V9-A4.5 therefore closes the fixed-topK reranking-adaptation route. The next question is upstream:

```text
Does the correlation map contain useful GT-near locations outside the current top16,
or is candidate recall already saturated and the remaining problem purely ordering?
```

## 2. TrackOn2 candidate path

```text
normalized query-feature correlation at c4/c8/c16/c32
-> bilinear upsample each scale to stride-4 grid
-> concatenate four maps
-> learned 1x1 ms_corr_proj
-> fused correlation map
-> top-K spatial selection
-> local decoder / reranker
```

V9-A5C.0 audits the maps before local reranking. It does not train or change the model.

## 3. Repository and provenance

Run only in:

```text
/gemini/code/FSPT_v9a45_clean
branch: v9a45-conservative-residual-20260710
base branch HEAD: 7c24367cf8aeb1467946d2ced3b294299032538e
```

TrackOn2 Python code and config come from the clean worktree. Immutable checkpoint, DINOv3 weights, datasets, and frozen pools are read from `/gemini/code/FSPT` after hash verification.

## 4. Stage A: PointOdyssey full audit

Canonical pool:

```text
4,878 balanced hard/easy rows
9 clips
sequences: ani / animal3 / r4_new_f
32 deterministic queries per clip
```

Use the exact V9-A4.3 repo-native replay:

```text
clip length: 96
queries selected by stable_seed(sequence,start)
support grid: 20x20
memory policy: unconditional
state updated at every frame
```

The diagnostic forward must exactly match official TrackOn2 `p/v/q` tensors on a smoke frame and must reproduce the canonical fused C1 top16 candidate coordinates.

## 5. Correlation maps

For every selected row compute:

```text
c4 native map, stride 4
c8 native map, stride 8
c16 native map, stride 16
c32 native map, stride 32
fused map after ms_corr_proj, stride 4
```

Also construct a raw-cosine union-of-scales diagnostic set:

```text
union_raw_equal_budget(K): take top ceil(K/4) candidates from each native scale
```

The union has at most `4 * ceil(K/4)` candidates and does not use GT for selection. It is diagnostic only: native maps are raw normalized cosine maps, while `ms_corr_proj` contains signed per-scale weights (including a negative c4 weight), so this raw union is not used by the primary gate.

## 6. Coordinate convention

Grid coordinates are patch centers from `indices_to_coords` in internal model space:

```text
model H/W = 384/512
```

They are mapped to the unified output space used by existing pools:

```text
x_256 = x_model / 512 * 256
y_256 = y_model / 384 * 256
```

Candidate-to-GT error is measured in the existing unified metric:

```text
error_px = ||candidate_yx_norm - gt_yx_norm|| * 255
```

## 7. Predeclared budgets and radii

```text
K = 1, 4, 8, 16, 32, 64
radius = 1, 2, 4, 8 px
```

No K/radius sweep outside this table is allowed.

## 8. Required metrics

For fused, raw c4/c8/c16/c32, and union_raw_equal_budget:

```text
recall@radius for each K
mean/median minimum candidate error for each K
per-sequence recall
hard/easy recall
```

For fused and each native scale:

```text
rank of the grid cell nearest to GT
nearest-grid distance
correlation score at nearest-GT grid cell
correlation top1 score
nearest-GT minus top1 correlation margin
fraction nearest-GT rank <= 16 / 32 / 64
```

Additional integrity:

```text
fused map recomputation max_abs versus model.multiscale_correlation
fused top16 ordered/set coordinate parity versus canonical pool C1 candidates
all values finite
```

## 9. PointOdyssey headroom gate

Primary metric:

```text
recall@4px
```

For each sequence define:

```text
current = fused top16 recall@4px
expanded_fused = fused top64 recall@4px
headroom = expanded_fused - current

The raw-scale union top64 recall is reported only as a diagnostic and is not part of the gate.
```

Stage A passes only if all three sequences satisfy:

```text
headroom >= 0.02
```

Interpretation:

```text
PASS:
  at least two percentage points of sequence-consistent fused-map GT-near recall
  lies between current top16 and expanded top64. Freeze this as synthetic upstream headroom and
  combine it with V9-A5.0 temporal-state evidence before selecting the next
  full-stream synthetic experiment. Do not automatically run DAVIS or train.

FAIL:
  candidate recall is effectively saturated or inconsistent across synthetic
  sequences; stop the current TrackOn2 correlation-adapter branch. The remaining
  gap is ordering/representation, already exhausted by V9-A3/A4.
```

## 10. Deferred DAVIS diagnostic

Stage A does not automatically authorize a DAVIS run. A separate written gate is required after combining:

```text
V9-A5.0 sampled-causal temporal-state result
V9-A5C.0 candidate-recall result
V9-A5.1 full-stream beam/reachability result
```

Any later DAVIS audit must remain diagnostic and may not select K, radius, weights, or model settings.

## 11. Outputs

Stage A:

```text
scripts/v9a5c0_candidate_recall_correlation_oracle.py
outputs/paper_discovery_2026-07-05/v9a5c0_candidate_recall/v9a5c0_pointodyssey_candidate_recall.json
outputs/paper_discovery_2026-07-05/v9a5c0_candidate_recall/v9a5c0_pointodyssey_candidate_recall.npz
docs/v9a5c0_pointodyssey_candidate_recall_result_2026-07-10.md
```

Deferred DAVIS artifacts, only after a separate written gate:

```text
outputs/paper_discovery_2026-07-05/v9a5c0_candidate_recall/v9a5c0_davis_candidate_recall.json
outputs/paper_discovery_2026-07-05/v9a5c0_candidate_recall/v9a5c0_davis_candidate_recall.npz
docs/v9a5c0_davis_candidate_recall_result_2026-07-10.md
```
