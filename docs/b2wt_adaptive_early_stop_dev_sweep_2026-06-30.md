# B2-WT Adaptive Early-stop Dev Sweep — 2026-06-30

## Goal

Test whether the fixed B2-W16-P2 override window can be improved by temporal / geometry risk-controlled early stopping.

This is a development-only sweep on:

```text
DAVIS
RGB dev0-9
```

No RGB fresh20-49 validation data was used.

## Compared baseline

Main baseline remains:

```text
B2-W16-P2
```

Reference values used by the sweep:

```text
RGB dev10 B2-W16-P2:
AJ_RD_256 = 0.4863
AJ_256    = 79.9035

DAVIS B2-W16-P2:
AJ_RD_256 = 0.6251
AJ_256    = 69.0119
```

## Sweep families

The sweep tested 22 variants:

```text
fixed shorter windows: W4, W8
stop if override loses visibility
stop if override speed spikes
stop if base becomes visible
stop if base/override visible disagreement is large
risk-shortening based on trigger-time base/override distance
distance-threshold early stop
relative distance-growth early stop
combined conservative variants
```

Outputs:

```text
scripts/sweep_b2wt_adaptive_early_stop_dev.py
outputs/paper_discovery_2026-06-27/b2wt_adaptive_dev/summary.json
```

## Main result

B2-WT did **not** produce a large enough improvement to replace B2-W16-P2.

The best retention candidates are positive but very small:

| variant | RGB ΔAJ_RD | RGB ΔAJ | DAVIS ΔAJ_RD | DAVIS ΔAJ | mean ΔAJ |
|---|---:|---:|---:|---:|---:|
| p2_disttau8_min4 | +0.0013 | +0.0096 | +0.0000 | +0.0218 | +0.0157 |
| p2_combined_conservative | +0.0023 | +0.0064 | +0.0009 | +0.0204 | +0.0134 |
| p2_riskdist4_post4 | +0.0020 | +0.0164 | -0.0003 | +0.0060 | +0.0112 |
| p2_disttau16_min4 | +0.0014 | +0.0073 | +0.0000 | +0.0137 | +0.0105 |
| p2_combined_risk_short | +0.0014 | +0.0013 | +0.0009 | +0.0166 | +0.0090 |

These are far below the target success criterion:

```text
AJ_RD drop <= 0.003
AJ gain >= +0.2
```

## Useful observations

### 1. Direction is not wrong

Several variants preserve or slightly improve AJ_RD while mildly improving AJ. For example:

```text
p2_combined_conservative:
RGB:   AJ_RD +0.0023, AJ +0.0064
DAVIS: AJ_RD +0.0009, AJ +0.0204
```

So temporal/geometry early-stop is not harmful when conservative.

### 2. But the effect is too small

Even the best mean AJ gain is only:

```text
+0.0157 AJ points
```

This is not enough to justify replacing B2-W16-P2.

### 3. Aggressive early-stop is domain-dependent

Variants such as `p2_basevis1`, `p2_basevis2`, and `p2_w4` improve RGB dev10 AJ more, but hurt DAVIS AJ_RD or AJ:

```text
p2_basevis2:
RGB:   AJ_RD +0.0044, AJ +0.0281
DAVIS: AJ_RD -0.0044, AJ -0.0443

p2_w4:
RGB:   AJ_RD +0.0020, AJ +0.0260
DAVIS: AJ_RD -0.0040, AJ -0.0101
```

Thus aggressive shortening risks dataset-specific tuning.

## Decision

Do not promote B2-WT as the main method.

Current status:

```text
B2-W16-P2 remains the main method.
B2-WT is a small positive exploratory refinement.
B2-WT supports the failure analysis but does not provide a strong new method result.
```

Safe paper use:

```text
Appendix / exploratory analysis:
Adaptive temporal early-stop gives only marginal gains, suggesting that B2-W16-P2 is already close to a robust operating point for the tested rule family.
```

Unsafe claim:

```text
B2-WT significantly improves B2-W16-P2.
```

## Next recommended direction

Stop rule-level tinkering for now. The current explored upgrades show:

```text
runtime learned verifier: insufficient
simple appearance: insufficient
CLIP crop: insufficient
ResNet dense: insufficient
B2-WT early-stop rules: only marginal gains
```

For the current paper, consolidate B2-W16-P2 as the final method and use these explorations as evidence that the final operating point is robust.

If still pursuing CCF-A-level innovation, the next attempt must be qualitatively different: true tracker-native correspondence features, stronger baseline parity, or an additional benchmark—not more W/P/early-stop threshold sweeps.
