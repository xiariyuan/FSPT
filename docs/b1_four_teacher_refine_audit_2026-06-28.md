# B1 Four-Teacher Refine Audit — 2026-06-28

## Decision

The current best deployable re-entry fusion rule is:

```text
vis4_gated288
tracks = visiblemedian_0_1_2_3
visibility = gated_288_0_1_2_3
true_AJ_RD_256 = 0.6279
```

This supersedes the previous 4-teacher quick result (`b1_all_median4_gated192 = 0.6264`) and the older 3-teacher gated result (`old3_all_median_gated144 = 0.6189`).

## Teacher pool

| id | teacher |
|---:|---|
| 0 | `cotracker3_online` |
| 1 | `cotracker3_offline` |
| 2 | `trackon2` |
| 3 | `tapnext_bootstapnext` |

Input caches:

```text
outputs/redetection_ladder_2026-06-17/caches/cotracker3_online_strided_original.pt
outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt
caches/trackon2_strided_original.pt
outputs/paper_discovery_2026-06-27/teacher_expansion/tapnext_bootstapnext_strided_original.pt
```

## Main result ladder

| Method | Teacher pool | true_AJ_RD_256 | true_AJ_RD | proxy | dmin1 | dmin4 | dmin16 | long20 <4px | long20 <8px |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed best `cotracker3_offline` | 3 | 0.5546 | | | | | | | |
| old masked median majority | 3 | 0.5871 | 0.4107 | 0.4315 | 0.5847 | 0.5648 | 0.4994 | 0.2945 | 0.5959 |
| old3 all-median gated144 | 3 | 0.6189 | 0.4235 | 0.4884 | 0.6148 | 0.6039 | 0.5578 | 0.3425 | 0.6301 |
| B1 quick all-median4 gated192 | 4 | 0.6264 | 0.4300 | 0.4998 | 0.6225 | 0.6117 | 0.5706 | 0.3767 | 0.6712 |
| **B1 refine vis4 gated288** | 4 | **0.6279** | **0.4302** | **0.4969** | **0.6240** | **0.6132** | **0.5715** | **0.3973** | **0.6438** |
| 3-teacher oracle upper bound | 3 | 0.6509 | 0.4586 | | | | | | |

## Best rule definition

The best rule is implemented in `scripts/sweep_b1_4teacher_refine.py` as:

```text
track rule: visiblemedian_0_1_2_3
visibility rule: gated_288_0_1_2_3
```

Track construction:

```text
tracks = visible-masked median over all four teachers, with finite-median fallback
```

Visibility construction:

```text
if mean pairwise 4-teacher disagreement <= 288 px:
    visibility = union of four teacher visibility signals
else:
    visibility = at least 2 visible teachers
```

## Interpretation

The key finding is not that TAPNext is the best standalone teacher. It is not.

```text
tapnext_single true_AJ_RD_256 = 0.5184
```

Despite this weak standalone score, adding TAPNext into the fusion pool improves the best deployable result:

```text
old 3-teacher gated best: 0.6189
4-teacher quick best:     0.6264
4-teacher refine best:    0.6279
```

This supports the project hypothesis:

```text
Teacher value under re-entry is not equal to standalone score.
A weak standalone teacher can still provide complementary visibility / coordinate signal for fusion.
```

## Caveats

- The 4-teacher oracle upper bound has not yet been computed.
- Standard TAP metrics, OA, delta_avg, false-visible rate, and missed-visible rate still need a dedicated audit.
- Current evidence is DAVIS-only.
- `CURRENT_MAINLINE.md` must be kept updated so downstream agents do not revert to the older `0.5871` baseline.

## Next required experiments

1. Compute 4-teacher oracle and teacher usage.
2. Build candidate action-value oracle over the strongest B1 rules.
3. Attempt a baseline-preserving safe router only if action oracle shows usable headroom.
4. Run standard metric / visibility-error audit for `vis4_gated288`.
5. Validate on RGB-Stacking or Kinetics subset.
