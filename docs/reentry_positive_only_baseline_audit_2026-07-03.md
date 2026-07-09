# ReEntry Positive-Only Baseline Audit — 2026-07-03

## Why this audit matters

The learned `ReEntry-VisCalibrator V1` slightly improved over the original rule-based `ReEntry-VisGuard-W8P2`. However, a key concern remained:

```text
Is the learned model really learning a better decision, or is the improvement mostly from a simpler rule change: only turn visibility on, never turn it off?
```

This document evaluates that control.

## Positive-only rule

Original rule W8P2:

```text
inside predicted re-entry window:
    pred_visibility = override_visibility
```

Positive-only rule:

```text
inside predicted re-entry window:
    pred_visibility = base_visibility OR override_visibility
```

So it only recovers visible frames. It never turns a base-visible frame into invisible.

New script:

```text
scripts/eval_reentry_positive_only_baseline.py
```

Status:

```text
python -m py_compile scripts/eval_reentry_positive_only_baseline.py  # pass
```

## Results

### RGB fresh20-49 natural

| Method | AJ_RD_256 | AJ_256 | OA_256 |
|---|---:|---:|---:|
| offline/base | 0.3816 | 79.5944 | 91.4636 |
| original rule W8P2 | 0.4510 | 79.1110 | 92.8853 |
| learned V1 thr0.15 | 0.4521 | 79.2356 | 92.8896 |
| positive-only W8P2 | 0.4517 | 79.1109 | 92.8895 |
| positive-only W16P2 | 0.4524 | 79.1098 | 92.8919 |

Key points:

```text
positive-only W16P2 gives the best AJ_RD on natural.
learned V1 gives higher AJ than positive-only and original rule.
```

### fresh20-49 translate_L16

| Method | AJ_RD_256 | AJ_256 | OA_256 |
|---|---:|---:|---:|
| offline/base | 0.4788 | 75.1940 | 90.7805 |
| original rule W8P2 | 0.5336 | 74.7705 | 92.2245 |
| learned V1 thr0.15 | 0.5349 | 74.9201 | 92.2000 |
| positive-only W8P2 | 0.5342 | 74.7699 | 92.2317 |
| positive-only W16P2 | 0.5348 | 74.7679 | 92.2329 |

Key points:

```text
learned V1 has the best AJ_RD, but only by +0.0001 over positive-only W16P2.
learned V1 has clearly better AJ than both positive-only variants.
```

### fresh20-49 occluder_L16

| Method | AJ_RD_256 | AJ_256 | OA_256 |
|---|---:|---:|---:|
| offline/base | 0.6311 | 77.6280 | 90.8918 |
| original rule W8P2 | 0.6659 | 77.0436 | 92.1748 |
| learned V1 thr0.15 | 0.6667 | 77.1780 | 92.1795 |
| positive-only W8P2 | 0.6664 | 77.0461 | 92.1841 |
| positive-only W16P2 | 0.6667 | 77.0443 | 92.1858 |

Key points:

```text
learned V1 and positive-only W16P2 tie on AJ_RD.
learned V1 has substantially higher AJ than positive-only W16P2.
```

## Learned V1 vs positive-only W16P2

| Setting | ΔAJ_RD learned - positive W16 | ΔAJ learned - positive W16 | ΔOA learned - positive W16 |
|---|---:|---:|---:|
| natural | -0.0003 | +0.1258 | -0.0023 |
| translate_L16 | +0.0001 | +0.1522 | -0.0329 |
| occluder_L16 | +0.0000 | +0.1337 | -0.0063 |

## Interpretation

This audit changes how the learned module should be presented.

### What positive-only explains

A meaningful part of the gain over the original rule comes from a simpler principle:

```text
recover visibility, but do not turn base-visible frames off.
```

This means the original W8P2 rule was slightly too aggressive because it copied override visibility wholesale inside the window, including possible override-invisible frames.

### What learned V1 still contributes

The learned module does not strongly beat positive-only W16P2 on AJ_RD. But it consistently gives higher standard AJ:

```text
natural:   +0.1258 AJ over positive-only W16
translate: +0.1522 AJ over positive-only W16
occluder:  +0.1337 AJ over positive-only W16
```

So the learned V1 is best interpreted as:

```text
a learned tradeoff calibrator that matches positive-only AJ_RD while improving standard AJ.
```

not as:

```text
a large AJ_RD improvement over a strong simple rule.
```

## Implication for the paper

Do not claim:

```text
The learned module is the main source of AJ_RD improvement.
```

Claim instead:

```text
A positive-only coordinate-preserving visibility recovery rule is a stronger simple baseline than the original full visibility replacement. ReEntry-VisCalibrator V1 further learns a slightly better AJ tradeoff, matching or slightly improving AJ_RD while improving standard AJ.
```

## Recommended paper hierarchy after this audit

### Main simple method

```text
ReEntry-PositiveOnly-W16P2
```

or, if we want to keep the original method name:

```text
ReEntry-VisGuard+ : positive-only local visibility recovery
```

### Learned extension

```text
ReEntry-VisCalibrator V1
```

Role:

```text
learned tradeoff extension, not yet a strong standalone replacement
```

## Remaining caveats

1. Threshold 0.15 for learned V1 still needs a clean dev-selected protocol.
2. Positive-only W16P2 itself is still rule-based, so it does not solve the rule-based criticism by itself.
3. The learned module's AJ advantage is real but modest.
4. Both simple and learned versions remain CoTracker-family until non-CoTracker evidence is added.

## Next required experiment

Clean protocol for learned threshold:

```text
train: dev0-6
threshold selection: dev7-9 by metric-level AJ_RD/AJ tradeoff
test: fresh20-49 natural / translate / occluder
```

Also report:

```text
original W8P2
positive-only W8P2
positive-only W16P2
learned V1
```

This will decide whether the learned module should be promoted to the main method or kept as an extension.
