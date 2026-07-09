# TAPNext++ Online Controlled Visibility-Lag Audit — 2026-07-06

## Decision

This is the next strongest SOTA-compatible diagnostic result.

Unlike the previous TAPNext++ offline_w8 and TrackOn2 offline_batch stress baselines, this audit keeps **normal TAPNext++ online coordinates** fixed and only degrades visibility. This removes the biggest weakness of the prior cross-family table: weak/degraded coordinates.

Result:

```text
Strong TAPNext++ online coordinates are preserved.
Only re-entry visibility is artificially delayed.
Restoring original online visibility recovers AJ/OA/AJ_RD while all coordinate thresholds stay unchanged.
```

This is not yet a deployable ReEntry method result, because the recovery target is original online visibility. It is a controlled headroom / diagnostic result showing that visibility lag alone can materially hurt metrics even when coordinates are strong.

## Artifacts

Script:

```text
scripts/eval_tapnextpp_controlled_visibility_lag.py
```

Output:

```text
outputs/paper_discovery_2026-07-05/tapnextpp_smoke/tapnextpp_online_controlled_visibility_lag_audit.json
```

Input cache:

```text
outputs/paper_discovery_2026-07-05/tapnextpp_smoke/tapnextpp_davis_first_input_cache_v4.pt
```

Important cache note:

```text
The current TAPNext++ online cache stores binary pred_visibility but not pred_vis_conf/logits.
Therefore this audit uses binary visibility lag, not confidence-threshold degradation.
```

## Protocol

Base:

```text
TAPNext++ normal online first/input DAVIS cache.
30 videos.
Coordinates are TAPNext++ online coordinates.
```

Degradation:

```text
For every predicted invisible -> visible re-entry segment, set the first L visible frames to invisible.
L in {1, 2, 4, 8}.
```

Recovery:

```text
Restore the original TAPNext++ online binary visibility.
```

Coordinates:

```text
Never changed.
```

## Baseline online metrics

Original TAPNext++ online cache:

| Metric | Value |
|---|---:|
| AJ | 65.85 |
| OA | 92.32 |
| δ@1px | 44.74 |
| δ@4px | 87.27 |
| δ@16px | 98.07 |
| δ_avg | 79.09 |
| J@4px | 74.09 |
| J@8px | 83.91 |
| J@16px | 87.53 |
| AJ_RD | 0.5961 |
| AJ_RD @D4 | 0.6995 |
| AJ_RD @D16 | 0.8046 |

## Controlled visibility-lag results

### Lag = 1 frame

| Metric | Degraded | Restored gain |
|---|---:|---:|
| AJ | 65.58 | +0.27 |
| OA | 91.95 | +0.37 |
| δ_avg | 79.09 | 0.00 |
| J@4px | 73.76 | +0.33 |
| J@8px | 83.43 | +0.47 |
| J@16px | 86.99 | +0.54 |
| AJ_RD | 0.5547 | +0.0415 |
| AJ_RD @D4 | 0.6522 | +0.0473 |
| AJ_RD @D16 | 0.7519 | +0.0527 |
| visible frames suppressed | 750 | - |

### Lag = 2 frames

| Metric | Degraded | Restored gain |
|---|---:|---:|
| AJ | 64.72 | +1.13 |
| OA | 91.04 | +1.28 |
| δ_avg | 79.09 | 0.00 |
| J@4px | 72.72 | +1.36 |
| J@8px | 82.24 | +1.67 |
| J@16px | 85.70 | +1.82 |
| AJ_RD | 0.5147 | +0.0815 |
| AJ_RD @D4 | 0.6022 | +0.0973 |
| AJ_RD @D16 | 0.6975 | +0.1071 |
| visible frames suppressed | 1404 | - |

### Lag = 4 frames

| Metric | Degraded | Restored gain |
|---|---:|---:|
| AJ | 63.09 | +2.76 |
| OA | 89.25 | +3.07 |
| δ_avg | 79.09 | 0.00 |
| J@4px | 70.72 | +3.36 |
| J@8px | 79.90 | +4.01 |
| J@16px | 83.17 | +4.36 |
| AJ_RD | 0.4450 | +0.1512 |
| AJ_RD @D4 | 0.5168 | +0.1827 |
| AJ_RD @D16 | 0.6015 | +0.2031 |
| visible frames suppressed | 2490 | - |

### Lag = 8 frames

| Metric | Degraded | Restored gain |
|---|---:|---:|
| AJ | 60.53 | +5.32 |
| OA | 86.47 | +5.85 |
| δ_avg | 79.09 | 0.00 |
| J@4px | 67.75 | +6.34 |
| J@8px | 76.23 | +7.68 |
| J@16px | 79.21 | +8.32 |
| AJ_RD | 0.3319 | +0.2642 |
| AJ_RD @D4 | 0.3942 | +0.3053 |
| AJ_RD @D16 | 0.4597 | +0.3449 |
| visible frames suppressed | 4083 | - |

## Interpretation

### What this proves

```text
1. With strong TAPNext++ online coordinates fixed, visibility lag alone can produce large metric loss.
2. Restoring visibility recovers AJ/OA/AJ_RD while leaving δ metrics unchanged.
3. This is cleaner evidence than TAPNext++ offline_w8 and TrackOn2 offline_batch, because coordinate quality is not degraded.
4. The result supports the core ReEntry thesis: re-entry visibility is a separable failure axis.
```

### What this does not prove

```text
1. This is not a deployable ReEntry method yet.
2. Recovery uses original online visibility, so it is an oracle/control target, not a learned predictor.
3. The degradation is artificial binary lag, not a naturally occurring TAPNext++ failure distribution.
4. It should not be reported as ReEntry improving TAPNext++ online performance.
```

## Claim boundary

Safe claim:

```text
A controlled TAPNext++ online-coordinate audit shows that re-entry visibility lag alone can substantially reduce AJ/OA/AJ_RD while leaving coordinate thresholds unchanged; restoring the visibility stream recovers these metrics.
```

Unsafe claim:

```text
ReEntry improves TAPNext++ online.
ReEntry beats TAPNext++.
This is a deployable TAPNext++ ReEntry result.
```

## Next step

The best next step is **not V26 training** yet.

Immediate next technical action:

```text
Rerun / modify TAPNext++ online export to save raw visibility logits or confidence.
```

Reason:

```text
The current cache only has binary visibility. Without confidence/logits, we cannot build a realistic confidence-threshold degradation or calibrator.
```

After logits/confidence are available:

```text
1. Evaluate TAPNext++ online visibility threshold sweeps.
2. Identify whether real confidence-threshold lag exists without coordinate degradation.
3. Build a simple calibration/repair rule that uses only available inference signals, not GT or original online visibility as oracle.
4. Only then consider a deployable TAPNext++ ReEntry adapter.
```

Recommended experiments after saving logits:

```text
A. threshold sweep:
   pred_visible = sigmoid(logit) >= tau, tau in {0.3, 0.5, 0.7, 0.8, 0.9}

B. lag-risk audit:
   measure how often high-coordinate-quality points are hidden by overly strict visibility.

C. confidence-based recovery:
   restore only frames where coordinate continuity is good and visibility confidence is near threshold.

D. compare to controlled binary lag:
   use this audit as upper-bound / sanity reference.
```

Paper placement:

```text
Diagnostic appendix or ablation:
TAPNext++ Online Controlled Visibility-Lag Audit.
```
