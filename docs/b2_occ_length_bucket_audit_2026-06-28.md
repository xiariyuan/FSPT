# B2 Occlusion-Length Bucket Audit — 2026-06-28

## Decision

B2 gains are consistent across occlusion-length buckets and are especially important for longer occlusion / re-entry cases. B2 mainline nearly matches global B1 in every bucket while preserving standard TAP AJ globally.

## Bucket comparison

Mean post-reappearance AJ segment in 256-space:

| bucket | n events | fixed_offline | global_b1_vis4 | b2_mainline | b2_gt_oracle | B2 - fixed | B2 - B1 | B2 - oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| occ 1-4 | 810 | 0.624646 | 0.662390 | 0.661936 | 0.667453 | +0.037290 | -0.000454 | -0.005517 |
| occ 5-8 | 340 | 0.626069 | 0.639688 | 0.639648 | 0.642052 | +0.013579 | -0.000040 | -0.002404 |
| occ 9-16 | 351 | 0.449964 | 0.552900 | 0.553275 | 0.551655 | +0.103311 | +0.000375 | +0.001620 |
| occ 17-32 | 330 | 0.492585 | 0.598044 | 0.597964 | 0.597900 | +0.105379 | -0.000080 | +0.000064 |
| occ 33+ | 32 | 0.291353 | 0.409669 | 0.409669 | 0.409669 | +0.118316 | +0.000000 | +0.000000 |

## Interpretation

The bucket evidence supports the B2 method story:

```text
shorter occlusion: B2 stays close to B1 and improves over fixed.
longer occlusion: B2 strongly improves over fixed and essentially matches B1 / oracle.
```

The largest relative gains over fixed occur in longer occlusion buckets:

```text
occ 9-16: +0.103311
occ 17-32: +0.105379
occ 33+: +0.118316
```

This confirms that B2 is not merely improving ordinary tracking. It is recovering post-reappearance trajectories where fixed_offline loses visibility or re-detection reliability.

## Relation to B1

B2 matches global B1 within tiny differences in every bucket:

```text
occ 1-4: -0.000454
occ 5-8: -0.000040
occ 9-16: +0.000375
occ 17-32: -0.000080
occ 33+: +0.000000
```

This is strong evidence that the predicted localized override captures essentially all of the B1 re-entry benefit while avoiding B1's global standard-AJ collapse.

## Relation to GT-window oracle

B2 is also close to the GT-window oracle. The only visible gap is in short occlusion bins, especially occ 1-4. For long occlusion bins, B2 reaches the oracle.

## Next step

Run false-trigger taxonomy. The main remaining question is why B2 still loses about 1.08 standard AJ points versus fixed_offline and which false triggers are responsible.
