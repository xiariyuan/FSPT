# ReEntry-VisGuard Paper Tables v1 — 2026-07-02

This document consolidates the current paper-ready evidence after the corrected coordinate-space audit.

## Authoritative corrected thesis

```text
ReEntry-VisGuard is a coordinate-preserving visibility recovery layer.
It preserves the stronger base/offline coordinate channel and locally transfers the responsive override visibility channel in predicted re-entry windows.
The paper must not claim coord8 saturation or that coordinates are irrelevant.
```

---

## Table 1. Main query-weighted results

| Setting | Method | Coordinates | Visibility | AJ_RD_256 ↑ | AJ_256 ↑ | OA_256 ↑ | ΔAJ_RD vs base | ΔAJ vs base |
|---|---|---|---|---:|---:|---:|---:|---:|
| RGB fresh20-49 natural | offline/base | base | base | 0.3816 | 79.5944 | 91.4636 | — | — |
| RGB fresh20-49 natural | online/override global | override | override | 0.4121 | 44.6933 | 55.7585 | 0.0305 | -34.9011 |
| RGB fresh20-49 natural | base coord + override vis global | base | override global | 0.4282 | 45.7904 | 55.7585 | 0.0466 | -33.8040 |
| RGB fresh20-49 natural | ReEntry-VisGuard-W8P2 | base | local override vis | 0.4510 | 79.1110 | 92.8853 | 0.0694 | -0.4834 |
| fresh20-49 translate_L16 | offline/base | base | base | 0.4788 | 75.1940 | 90.7805 | — | — |
| fresh20-49 translate_L16 | online/override global | override | override | 0.4981 | 43.0978 | 56.7847 | 0.0193 | -32.0962 |
| fresh20-49 translate_L16 | base coord + override vis global | base | override global | 0.5133 | 44.0643 | 56.7847 | 0.0345 | -31.1297 |
| fresh20-49 translate_L16 | ReEntry-VisGuard-W8P2 | base | local override vis | 0.5336 | 74.7705 | 92.2245 | 0.0548 | -0.4235 |
| fresh20-49 occluder_L16 | offline/base | base | base | 0.6311 | 77.6280 | 90.8918 | — | — |
| fresh20-49 occluder_L16 | online/override global | override | override | 0.6146 | 43.1490 | 58.0983 | -0.0165 | -34.4790 |
| fresh20-49 occluder_L16 | base coord + override vis global | base | override global | 0.6469 | 44.7880 | 58.0983 | 0.0158 | -32.8400 |
| fresh20-49 occluder_L16 | ReEntry-VisGuard-W8P2 | base | local override vis | 0.6659 | 77.0436 | 92.1748 | 0.0348 | -0.5844 |

**Key result:** ReEntry-VisGuard-W8P2 improves AJ_RD_256 by `+0.0348` to `+0.0694` over the offline/base method while keeping AJ_256 loss below `0.6` points in all three final settings.

---

## Table 2. Channel-wise factorial intervention

This is the core mechanism evidence. It separates coordinate and visibility channels to show that local visibility transfer, not full trajectory replacement, is the useful improvement.

### RGB fresh20-49 natural

| Method | Coordinates | Visibility | AJ_RD_256 ↑ | AJ_256 ↑ | OA_256 ↑ | ΔAJ_RD vs base | ΔAJ vs base |
|---|---|---|---:|---:|---:|---:|---:|
| base_coord_base_vis | base | base | 0.3816 | 79.5944 | 91.4636 | — | — |
| override_coord_override_vis | override | override | 0.4121 | 44.6933 | 55.7585 | 0.0305 | -34.9011 |
| override_coord_base_vis | override | base | 0.3494 | 60.8419 | 91.4636 | -0.0322 | -18.7525 |
| base_coord_override_vis_global | base | override global | 0.4282 | 45.7904 | 55.7585 | 0.0466 | -33.8040 |
| ReEntry-VisGuard-W8P2 | base | local override vis | 0.4510 | 79.1110 | 92.8853 | 0.0694 | -0.4834 |

### fresh20-49 translate_L16

| Method | Coordinates | Visibility | AJ_RD_256 ↑ | AJ_256 ↑ | OA_256 ↑ | ΔAJ_RD vs base | ΔAJ vs base |
|---|---|---|---:|---:|---:|---:|---:|
| base_coord_base_vis | base | base | 0.4788 | 75.1940 | 90.7805 | — | — |
| override_coord_override_vis | override | override | 0.4981 | 43.0978 | 56.7847 | 0.0193 | -32.0962 |
| override_coord_base_vis | override | base | 0.4499 | 52.8525 | 90.7805 | -0.0289 | -22.3415 |
| base_coord_override_vis_global | base | override global | 0.5133 | 44.0643 | 56.7847 | 0.0345 | -31.1297 |
| ReEntry-VisGuard-W8P2 | base | local override vis | 0.5336 | 74.7705 | 92.2245 | 0.0548 | -0.4235 |

### fresh20-49 occluder_L16

| Method | Coordinates | Visibility | AJ_RD_256 ↑ | AJ_256 ↑ | OA_256 ↑ | ΔAJ_RD vs base | ΔAJ vs base |
|---|---|---|---:|---:|---:|---:|---:|
| base_coord_base_vis | base | base | 0.6311 | 77.6280 | 90.8918 | — | — |
| override_coord_override_vis | override | override | 0.6146 | 43.1490 | 58.0983 | -0.0165 | -34.4790 |
| override_coord_base_vis | override | base | 0.5858 | 59.6368 | 90.8918 | -0.0453 | -17.9912 |
| base_coord_override_vis_global | base | override global | 0.6469 | 44.7880 | 58.0983 | 0.0158 | -32.8400 |
| ReEntry-VisGuard-W8P2 | base | local override vis | 0.6659 | 77.0436 | 92.1748 | 0.0348 | -0.5844 |

**Mechanism readout:** global override visibility carries AJ_RD signal but collapses standard AJ by roughly `31–34` points. Local visibility transfer converts that signal into AJ_RD gains with less than `0.6` AJ loss.

---

## Table 3. Corrected coordinate-vs-visibility diagnosis

Coordinates are evaluated in 256-space pixels. Earlier normalized-distance coord8 outputs are invalid.

### RGB fresh20-49 natural

| Method | coord8 | coord16 | predvis | joint8 | first_coord8 | first_predvis | first_joint8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| offline/base | 0.7815 | 0.8640 | 0.6240 | 0.5617 | 0.7578 | 0.2538 | 0.2216 |
| full_w8_p2 | 0.7748 | 0.8585 | 0.8350 | 0.6863 | 0.7336 | 0.5917 | 0.4489 |
| ReEntry-VisGuard-W8P2 | 0.7815 | 0.8640 | 0.8350 | 0.6916 | 0.7578 | 0.5917 | 0.4678 |

### fresh20-49 translate_L16

| Method | coord8 | coord16 | predvis | joint8 | first_coord8 | first_predvis | first_joint8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| offline/base | 0.8472 | 0.9117 | 0.7593 | 0.7068 | 0.7494 | 0.3473 | 0.3037 |
| full_w8_p2 | 0.8452 | 0.9087 | 0.8989 | 0.7934 | 0.7307 | 0.6474 | 0.4996 |
| ReEntry-VisGuard-W8P2 | 0.8472 | 0.9117 | 0.8989 | 0.7947 | 0.7494 | 0.6474 | 0.5140 |

### fresh20-49 occluder_L16

| Method | coord8 | coord16 | predvis | joint8 | first_coord8 | first_predvis | first_joint8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| offline/base | 0.9057 | 0.9517 | 0.8390 | 0.8099 | 0.8574 | 0.4180 | 0.4004 |
| full_w8_p2 | 0.9008 | 0.9488 | 0.9365 | 0.8676 | 0.8060 | 0.7122 | 0.5974 |
| ReEntry-VisGuard-W8P2 | 0.9057 | 0.9517 | 0.9365 | 0.8718 | 0.8574 | 0.7122 | 0.6436 |

**Corrected diagnosis:** base/offline coordinates are not perfect, but they are generally stronger than full override coordinates. ReEntry-VisGuard preserves those coordinates and recovers local visibility, improving joint coordinate+visibility recovery.

---

## Table 4. No-leak / metric sanity

| Check | Value | Interpretation |
|---|---:|---|
| exact_close_frame_rate_atol1e-8 | 0.001969 | Predictions are not exact GT copies. |
| segment coord8_rate | 0.7703 | Corrected 256-space coord8, not saturated. |
| first coord8_rate | 0.7578 | First re-entry coordinate accuracy is useful but not perfect. |
| predvis recall | 0.8111 | Visibility recovery is a major contributor. |
| first predvis rate | 0.5917 | First re-entry visibility remains a key difficulty. |

| Baseline | AJ_RD_256 | AJ_256 | OA_256 |
|---|---:|---:|---:|
| original | 0.4510 | 79.1110 | 92.8853 |
| visibility_shuffled | 0.3987 | 67.1319 | 82.0698 |
| coord_perturb_sigma8px | 0.1340 | 19.9635 | 92.8853 |

**Sanity conclusion:** shuffling visibility hurts AJ_RD and AJ; perturbing coordinates heavily destroys AJ_RD and AJ. Both channels matter, and the method works by combining the stronger base coordinate channel with local override visibility.

---

## Main paper wording

Allowed wording:

```text
ReEntry-VisGuard-W8P2 improves AJ_RD_256 over the offline base by +0.0694 on natural RGB fresh20-49, +0.0548 on translate-L16 stress, and +0.0348 on occluder-L16 stress, while keeping AJ_256 loss below 0.6 points. Channel-wise factorial experiments show that this gain comes from preserving the stronger base coordinate channel and locally transferring the responsive override visibility channel, avoiding the severe AJ collapse of global override.
```

Do not write:

```text
coordinates are perfect / coord8 is saturated / re-entry failure is only visibility failure
```
