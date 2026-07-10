# V9-A2.4 Paper-Ready Ablation Design

Date: 2026-07-10

## Frozen policy

```text
all features + logistic regression + event_max + fixed threshold 0.05
preserve W8 common rows; control W16-extension rows only
```

OOF-F1 is the calibration-free threshold comparison. No new dense trajectory threshold search is allowed.

## Required analyses

1. Unified paper table: Native, W8, W16, V9-A1 diagnostic, V9-A2 fixed0.05, V9-A2 OOF-F1, oracle.
2. Feature-set ablation: base / anchor / all with the same logistic model.
3. Controller-mode ablation: frame / event_max / event_mean at fixed0.05.
4. Threshold protocol: W16 accept-all / fixed0.05 / OOF-F1.
5. Anchor-group ablation: generic-only, query-only, last-only, preocc-only, all anchors.
6. Paired video-level uncertainty for fixed0.05 versus W16 and W8 using AJ_RD_256.
7. CPU reference efficiency: feature extraction, controller batch inference, event activation.

## Statistical protocol

- Unit: DAVIS video with defined AJ_RD_256.
- Report mean, median, better/worse/equal, percentile bootstrap 95% CI.
- Report exact sign-flip permutation p-value and exact sign-test p-value on nonzero pairs.
- Aggregate metric gain and unweighted video-paired gain must be reported separately.

## Decision rule

- If fixed0.05 versus W16 has robust positive paired evidence, freeze V9-A2 as paper-ready prototype.
- If CI crosses zero / evidence is concentrated, retain V9-A2 as positive diagnostic and proceed to semantic/internal identity features.
