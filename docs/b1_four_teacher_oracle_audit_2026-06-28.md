# B1 Four-Teacher Oracle Audit — 2026-06-28

## Decision

Adding TAPNext increases the oracle upper bound.

```text
3-teacher oracle true_AJ_RD_256: 0.6509
4-teacher oracle true_AJ_RD_256: 0.6746
current best deployable fusion: 0.6279
```

The 4-teacher oracle result confirms that TAPNext contributes real complementary events even though its standalone performance is weak.

## Teacher pool

| teacher | median re-entry error px |
|---|---:|
| cotracker3_online | 3.86 |
| cotracker3_offline | 3.71 |
| trackon2 | 3.91 |
| tapnext | 3.99 |

Standalone note:

```text
tapnext_single true_AJ_RD_256 = 0.5184
```

Despite this weak standalone AJ_RD_256, TAPNext is selected by the oracle on a substantial number of events.

## Main metrics

| Quantity | Value |
|---|---:|
| n_events | 1385 |
| fixed_best_teacher | cotracker3_offline |
| fixed_best true_AJ_RD_256 | 0.5546 |
| 3-teacher oracle true_AJ_RD_256 | 0.6509 |
| 4-teacher oracle true_AJ_RD_256 | 0.6746 |
| 4-teacher oracle delta vs fixed | 0.1200 / +12.0pp |
| current best deployable `vis4_gated288` | 0.6279 |
| current gap to 4-teacher oracle | 0.0467 |
| oracle gap used by current best | 61.1% |

Original-resolution diagnostic:

| Quantity | Value |
|---|---:|
| fixed_best true_AJ_RD | 0.3870 |
| 4-teacher oracle true_AJ_RD | 0.4832 |
| delta vs fixed | 0.0962 |

## Oracle teacher usage

| teacher | events | share |
|---|---:|---:|
| cotracker3_online | 314 | 22.7% |
| cotracker3_offline | 368 | 26.6% |
| trackon2 | 325 | 23.5% |
| tapnext | 378 | 27.3% |


TAPNext usage:

```text
378 / 1385 = 27.3%
```

This is the key finding: TAPNext is not a strong standalone teacher on this metric, but it is oracle-best for many re-entry events.

## Interpretation

The current best deployable fusion uses 4 teachers and reaches:

```text
vis4_gated288 true_AJ_RD_256 = 0.6279
```

The new 4-teacher oracle is:

```text
oracle true_AJ_RD_256 = 0.6746
```

Therefore the remaining deployable gap is:

```text
0.6746 - 0.6279 = 0.0467
```

This means there is still measurable headroom after the rule-based B1 refine result. The next question is whether that gap is recoverable by a deployable action router or whether it requires new visual evidence / a stronger teacher.

## Next required experiment

Build a candidate action-value oracle over the strongest deployable B1 actions:

```text
B0 = vis4_gated288 = 0.6279
```

If the candidate-action oracle is only slightly above B0, then the current rule set is saturated and the remaining oracle gap requires new actions, visual evidence, or new teachers.

If the candidate-action oracle is meaningfully above B0, then a baseline-preserving safe router is justified.
