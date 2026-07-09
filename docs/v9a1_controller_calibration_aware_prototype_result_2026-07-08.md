# V9-A1 Controller Calibration-Aware Prototype Result

## Tabular group-heldout OOF summary

| Model | AP | AUC | Brier | Risk@50 loss | Threshold |
|---|---:|---:|---:|---:|---:|
| logreg | 0.6934 | 0.7574 | 0.2055 | 0.3516 | 0.3824 |
| hgb | 0.7257 | 0.7493 | 0.2279 | 0.3640 | 0.3508 |
| extratrees | 0.7442 | 0.7696 | 0.1924 | 0.3462 | 0.4039 |

## Trajectory apply-back summary

| Variant | AJ Δ | OA Δ | δavg Δ | δ4px Δ | AJ_RD Δ | AJ_RD_256 Δ | Accepted | Pos/Neg/Zero |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| native_reject_all | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0 | 0/0/25 |
| cvrrm_accept_all_touched | +0.0964 | +0.9462 | +0.3971 | +0.5085 | +0.0153 | +0.0274 | 1456 | 16/4/5 |
| v9a1_logreg_oof_f1 | +0.1678 | +1.0239 | +0.3104 | +0.3955 | +0.0071 | +0.0152 | 786 | 12/1/12 |
| v9a1_hgb_oof_f1 | +0.2490 | +0.9697 | +0.3028 | +0.4132 | +0.0095 | +0.0190 | 680 | 10/4/11 |
| v9a1_extratrees_oof_f1 | +0.1707 | +1.0252 | +0.3494 | +0.4861 | +0.0103 | +0.0212 | 804 | 10/2/13 |

## Decision

V9-A1 tabular models are learnable, but the current OOF-F1 apply-back does not beat the CVRRM accept-all baseline on AJ_RD_256. Do not scale to full ReEntryBeliefTrack from this result alone.
## Threshold sweep audit

A 31-quantile threshold sweep was run over OOF scores for each V9-A1 model.

Artifact:

```text
outputs/paper_discovery_2026-07-05/v9a1_controller_calibration_aware_prototype/v9a1_threshold_sweep.json
```

Key finding:

```text
The best AJ_RD_256 rows are the thresholds that accept all touched frames, which exactly reproduce CVRRM accept-all.
Once the controller actually filters candidate frames, AJ/OA can improve and negative videos can decrease, but AJ_RD_256 drops below CVRRM.
```

Interpretation:

```text
V9-A1 confirms that candidate-good / risk is learnable, but a simple tabular learned gate is not enough to replace the hard CVRRM rule as the main method.
The learned controller is useful as a diagnostic/risk-filtering ablation, not as evidence to scale immediately into full ReEntryBeliefTrack.
```

Updated decision:

```text
Do not scale directly from V9-A1 controller-only to full ReEntryBeliefTrack.
The next model step must add genuinely new information or structure, such as uncertainty-aware candidate consistency, anchor similarity, or internal candidate generation.
A learned gate over existing V8-C0.2 features alone is insufficient.
```
