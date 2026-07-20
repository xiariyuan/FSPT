# Route-D Visibility Activation Oracle — Gate 3C1G1 Post-Failure Result — 2026-07-20

## Interpretation

```text
COMPLETED_EXPOSED_ORACLE_DIAGNOSTIC
REJECT_ACTIVATION_ONLY_AS_SUFFICIENT_MAINLINE
```

The diagnostic changes only `native=occluded, modified=visible` action-frame transitions. All other Gate 3C1F2 modified visibility states and coordinates remain fixed.

```text
activation rows:       454
activation videos:     36
GT-visible activations:268
GT-occluded activations:186
within-16px activations:193
```

| Policy | AJ | Gain over actual modified |
|---|---:|---:|
| Actual modified | 0.252827 | 0 |
| GT-visible activation oracle | 0.253597 | +0.0770 points |
| Utility16 activation oracle | 0.253934 | +0.1107 points |

The preregistered action-video requirement was `+0.15` points. Even the best fixed activation oracle reaches only `+0.1107` points. Therefore activation calibration can be a supporting safety component but cannot by itself recover the complete-population AJ target. The next mainline must also increase safe action coverage.

No new raw population or external benchmark was read.
