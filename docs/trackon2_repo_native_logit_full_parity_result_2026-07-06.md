# TrackOn2 Repo-Native Logit Full Export Parity Result — 2026-07-06

## Purpose

Build a full 30-video TrackOn2 first/input true-base confidence/logit cache and verify that adding logits/confidence does not materially change the original TrackOn2 base.

This is not a ReEntry method experiment yet.

## Outputs

Repo-native `.npz` files with logits/confidence:

```text
outputs/paper_discovery_2026-07-05/trackon2_true_base_conf/repo_native_logit_full/davis/trackon2/*.npz
```

Unified bridge cache:

```text
outputs/paper_discovery_2026-07-05/trackon2_true_base_conf/trackon2_davis_first_input_repo_native_conf_full.pt
```

Report:

```text
outputs/paper_discovery_2026-07-05/trackon2_true_base_conf/repo_native_logit_full_report.json
```

Metric comparison:

```text
outputs/paper_discovery_2026-07-05/trackon2_true_base_conf/trackon2_davis_first_input_repo_native_conf_full_metric_compare.json
```

## What was exported

Each `.npz` contains:

```text
tracks
visibility
visibility_logit
visibility_conf
```

The unified `.pt` cache contains the old TrackOn2 query/GT/schema and adds:

```text
pred_vis_logit
pred_vis_conf
```

## Full parity result

Compared old TrackOn2 first/input bridge:

```text
outputs/attempt0_2026-06-15_recovery/prediction_caches/trackon2_dinov3_davis_first_input_bridge.pt
```

against new confidence bridge:

```text
outputs/paper_discovery_2026-07-05/trackon2_true_base_conf/trackon2_davis_first_input_repo_native_conf_full.pt
```

Metrics from the same parity evaluator:

| Cache | AJ_256 | OA_256 | delta_avg_256 | AJ_RD_256 | AJ_RD |
|---|---:|---:|---:|---:|---:|
| old TrackOn2 | 67.0406 | 93.0615 | 79.8418 | 0.5444 | 0.3714 |
| new conf bridge | 67.0340 | 93.0385 | 79.8783 | 0.5439 | 0.3709 |
| new - old | -0.0066 | -0.0230 | +0.0365 | -0.0005 | -0.0005 |

Preferred full metric parity gate:

```text
AJ diff <= 0.10 pp
OA diff <= 0.10 pp
delta_avg diff <= 0.10 pp
AJ_RD_256 diff <= 0.005
```

Result:

```text
pass_full_metric_parity_preferred_gate = true
```

## Raw visibility differences

During export, small binary visibility differences from old `.npz` were observed on some videos, usually near the confidence threshold. However full metric parity remains well within gate.

Important handling rule:

```text
For native TrackOn2 base, use saved binary visibility.
For threshold/recovery sweeps, use visibility_conf as the continuous confidence source.
```

## Decision

This step succeeds.

Meaning:

```text
We now have a parity-validated TrackOn2 true-base confidence/logit cache.
```

It does not yet mean:

```text
ReEntry improves TrackOn2.
```

## Next step

Only now is it valid to run TrackOn2 true-base module audits:

```text
1. Global confidence threshold sweep on TrackOn2 true base.
2. Local re-entry confidence recovery audit on TrackOn2 true base.
```

Success gate for any TrackOn2 method claim:

```text
AJ >= base - 0.10
OA >= base - 0.10
AJ_RD >= base + 0.01
```

If the module audit fails:

```text
TrackOn2 remains appendix/negative evidence, not a main improvement baseline.
```

If it passes:

```text
TrackOn2 can become a second true-base improvement result.
```
