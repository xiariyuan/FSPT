# Paper Method V2 — ReEntry-VisCalibrator

Date: 2026-07-03

## 1. Method position

The final method should be presented as:

```text
ReEntry-VisCalibrator:
A learned coordinate-preserving visibility recovery module for TAP re-detection.
```

The core idea is not to build a new tracker and not to replace coordinates. Instead:

```text
Keep the stronger base/offline coordinates.
Recover visibility only in predicted re-entry regions.
```

This avoids the main failure of global override:

```text
global override contains useful re-entry visibility signal,
but it catastrophically damages standard AJ.
```

So the method is a plug-in visibility calibration layer:

```text
input:  base tracker output + override tracker output
output: base coordinates + calibrated visibility
```

---

## 2. Notation

For each query point and frame `t`, assume two prediction streams:

```text
base stream:
  coordinates: p_b(t)
  visibility:  v_b(t)

override stream:
  coordinates: p_o(t)
  visibility:  v_o(t)
```

The final method always preserves base coordinates:

```text
p_hat(t) = p_b(t)
```

Only visibility is modified:

```text
v_hat(t) = calibrated visibility
```

This is the central constraint of the method.

---

## 3. Why coordinate preservation matters

The corrected channel-wise diagnosis shows:

```text
base coordinates are stronger than override coordinates;
override visibility contains useful re-entry response;
global override visibility damages standard AJ;
local visibility recovery gives AJ_RD gains with small AJ cost.
```

Therefore, the method should not be described as a tracker fusion method.

It is better described as:

```text
coordinate-preserving visibility recovery
```

or:

```text
learned visibility calibration for re-entry windows
```

---

## 4. Candidate re-entry regions

The learned module does not scan the whole video blindly. It first receives high-recall candidate regions produced from prediction-only signals.

A candidate trigger occurs when:

```text
base has been invisible after query,
and override is visible for P consecutive frames.
```

Current setting:

```text
P = 2
candidate context = trigger_t - 16 ... trigger_t + 16
```

This trigger uses only predictions. It does not use GT at inference.

The candidate stage is deliberately high-recall. The learned module is responsible for deciding which candidate frames are safe to recover.

---

## 5. Deterministic non-learned variant: Ours-Det

Before introducing the learned module, the paper should define a deterministic variant.

Recommended paper name:

```text
Ours-Deterministic
```

or short:

```text
Ours-Det
```

It should not be described as an external strong rule baseline. It is the non-learned version of the proposed framework.

Definition:

```text
p_hat(t) = p_b(t)

inside predicted re-entry windows:
  v_hat(t) = v_b(t) OR v_o(t)

outside predicted re-entry windows:
  v_hat(t) = v_b(t)
```

This is positive-only recovery:

```text
it can turn invisible into visible;
it never turns base-visible frames off.
```

Why include it:

```text
1. It proves that coordinate-preserving local visibility recovery is already strong.
2. It shows that most AJ_RD recovery comes from the core principle.
3. It gives a transparent non-learned variant.
4. It prevents reviewers from claiming the paper omitted a simple rule baseline.
```

Current best deterministic setting:

```text
W16P2
```

---

## 6. Learned variant: ReEntry-VisCalibrator

The learned version replaces the fixed deterministic decision with a small temporal calibrator.

For each candidate window, extract a sequence of frame-level features:

```text
base visibility
override visibility
visibility disagreement
base and override coordinates
base/override distance
base and override speed / acceleration
base invisible run length
override visible run length
distance to image border
override future visible rate
base past invisible rate
```

The current feature dimension is:

```text
28 features per frame
33 frames per candidate window
```

The model outputs a recovery score for each frame:

```text
s(t) in [0, 1]
```

Inference:

```text
if s(t) >= tau and override is visible:
    recover visible
else:
    keep base visibility
```

Coordinates remain unchanged:

```text
p_hat(t) = p_b(t)
```

---

## 7. Training target

The training label is not raw GT visibility.

The label asks:

```text
Is it safe to mark this frame visible while keeping base coordinates?
```

So the positive label requires:

```text
GT visible AND base coordinate close to GT
```

Soft label schedule:

```text
base coordinate error < 1 px  -> 1.00
base coordinate error < 2 px  -> 0.90
base coordinate error < 4 px  -> 0.75
base coordinate error < 8 px  -> 0.55
base coordinate error <16 px  -> 0.25
otherwise                     -> 0.00
```

This label is important because the method keeps base coordinates. If base coordinates are wrong, recovering visibility may hurt AJ.

---

## 8. Clean train / validation / test protocol

The learned module uses a clean protocol:

```text
train: dev0-6
threshold selection: dev7-9
frozen test: fresh20-49
```

Current model:

```text
outputs/paper_discovery_2026-06-27/reentry_viscalibrator/models/reentry_viscalibrator_v1_dev0_6_clean.pt
```

Threshold selected on dev7-9:

```text
tau = 0.10
```

The threshold is selected before fresh20-49 testing.

This should be stated clearly in the paper to avoid test-set tuning concerns.

---

## 9. Main result summary

Query-weighted final results:

| Setting | Method | AJ_RD_256 | AJ_256 | OA_256 |
|---|---|---:|---:|---:|
| natural | Base | 0.3816 | 79.5944 | 91.4636 |
| natural | Ours-Det | 0.4524 | 79.1098 | 92.8919 |
| natural | ReEntry-VisCalibrator | 0.4533 | 79.1867 | 92.9234 |
| translate_L16 | Base | 0.4788 | 75.1940 | 90.7805 |
| translate_L16 | Ours-Det | 0.5348 | 74.7679 | 92.2329 |
| translate_L16 | ReEntry-VisCalibrator | 0.5354 | 74.8798 | 92.2390 |
| occluder_L16 | Base | 0.6311 | 77.6280 | 90.8918 |
| occluder_L16 | Ours-Det | 0.6667 | 77.0443 | 92.1858 |
| occluder_L16 | ReEntry-VisCalibrator | 0.6671 | 77.1320 | 92.2273 |

Main gains over Base:

```text
natural:       AJ_RD +0.0717, AJ -0.4077, OA +1.4598
translate_L16: AJ_RD +0.0566, AJ -0.3142, OA +1.4585
occluder_L16:  AJ_RD +0.0360, AJ -0.4960, OA +1.3355
```

Gains over Ours-Det:

```text
natural:       AJ_RD +0.0009, AJ +0.0769, OA +0.0315
translate_L16: AJ_RD +0.0006, AJ +0.1119, OA +0.0061
occluder_L16:  AJ_RD +0.0004, AJ +0.0877, OA +0.0415
```

---

## 10. Video-level stability

Paired video-level statistics show:

```text
Learned vs Base AJ_RD improves on:
  natural:   29 / 30 videos
  translate: 29 / 30 videos
  occluder:  28 / 30 videos
```

Compared with Ours-Det, the learned module has its clearest advantage on standard AJ:

```text
Learned vs Ours-Det AJ improves on:
  natural:   26 / 30 videos
  translate: 28 / 30 videos
  occluder:  27 / 30 videos
```

This supports the claim:

```text
The learned module improves the recovery-stability tradeoff.
```

---

## 11. Recommended paper wording

Strong and safe wording:

```text
ReEntry-VisCalibrator improves TAP re-detection by preserving the reliable base coordinate channel and learning to recover visibility in predicted re-entry windows. Compared with the offline/base tracker, it improves AJ_RD_256 by +0.0717, +0.0566, and +0.0360 across natural, translate_L16, and occluder_L16 settings. A deterministic variant already captures most of the re-entry recovery, while the learned calibrator further improves the recovery-stability tradeoff, matching or slightly improving AJ_RD and consistently improving standard AJ.
```

中文：

```text
ReEntry-VisCalibrator 通过保留可靠的 base 坐标通道，并学习在预测重入窗口中恢复可见性，从而提升 TAP 重入再检测能力。相比 offline/base，它在 natural、translate_L16、occluder_L16 上分别提升 AJ_RD_256 +0.0717、+0.0566、+0.0360。无学习确定性版本已经捕获了大部分重入恢复收益，而学习校准器进一步改善恢复和普通追踪稳定性之间的平衡：它保持或略微提升 AJ_RD，同时稳定提高 standard AJ。
```

---

## 12. Claims to avoid

Do not write:

```text
The learned module dramatically improves AJ_RD over all rule variants.
```

Because compared with Ours-Det, the extra AJ_RD gain is small.

Do not write:

```text
This proves universal cross-tracker generalization.
```

The current evidence is still CoTracker-family.

Do not write:

```text
Coordinates are irrelevant.
```

Corrected diagnosis shows coordinates and visibility both matter; the key is that base coordinates are stronger than override coordinates.

---

## 13. Next steps after method V2

1. Add these V2 tables to the paper outline.
2. Produce qualitative figures showing:
   - base visibility too conservative;
   - Ours-Det recovery;
   - learned calibrator preserving stability;
   - failure cases.
3. Run LocoTrack or another non-CoTracker validation if checkpoint becomes available.
4. Optionally develop V2 model with window-level confidence and metric-aware objective.
