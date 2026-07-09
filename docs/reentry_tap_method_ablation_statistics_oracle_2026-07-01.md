# ReEntry-TAP Method-paper Supplements: Natural Ablation, Natural Statistics, and Oracle Upper Bound — 2026-07-01

## Goal

These supplements support the method-paper framing:

```text
Problem: TAP re-entry recovery is weak.
Method: B2-W16-P2 / Selective Local Re-entry Override.
Primary metric: AJ_RD_256.
Constraint metric: AJ_256.
```

The goal is to show not only that B2-W16-P2 improves AJ_RD, but also why its design choices are reasonable and how much headroom remains.

---

## 1. RGB fresh20-49 natural ablation

Source caches:

```text
outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_aggregate/offline_rgb_stacking_fresh20_49.pt
outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_aggregate/online_rgb_stacking_fresh20_49.pt
```

| Method | AJ_RD_256 | AJ_256 | OA_256 | ΔAJ_RD vs offline | ΔAJ vs offline |
|---|---:|---:|---:|---:|---:|
| offline | 0.3816 | 79.5944 | 91.4636 | — | — |
| online_global | 0.4121 | 44.6933 | 55.7585 | +0.0305 | -34.9011 |
| b2_fullpost_p1 | 0.4219 | 78.9590 | 92.7706 | +0.0403 | -0.6354 |
| b2_w8_p2 | 0.4462 | 79.0753 | 92.8853 | +0.0646 | -0.5191 |
| b2_w16_p1 | 0.4456 | 79.0634 | 92.8891 | +0.0640 | -0.5310 |
| b2_w16_p2 | 0.4454 | 79.0664 | 92.8804 | +0.0638 | -0.5280 |
| b2_w32_p2 | 0.4421 | 79.0451 | 92.8736 | +0.0605 | -0.5493 |

### Interpretation

This ablation is important for the method-paper framing.

```text
1. online_global improves AJ_RD only modestly (+0.0305) but collapses standard AJ by -34.9011.
2. full-post override avoids global collapse but underperforms windowed override in AJ_RD.
3. W8/W16 windowed variants are the best operating points.
4. B2-W16-P2 is essentially tied with W8-P2/W16-P1 on AJ_RD while keeping the conservative persistent trigger used in natural false-trigger audits.
```

Safe wording:

```text
Windowed local override is the key mechanism; global use of the re-entry branch is not a viable tradeoff.
```

---

## 2. RGB fresh20-49 natural statistical robustness

B2-W16-P2 vs offline:

| Metric | mean delta | 95% CI | positive videos | sign-test p | notes |
|---|---:|---:|---:|---:|---|
| AJ_RD_256 | +0.0613 | [0.0419, 0.0805] | 24/30 | 0.000546 | primary gain |
| AJ_256 | -0.5280 | [-0.9716, -0.0668] | 7/30 positive | 0.005223 | 3 videos drop > 2 AJ |

Additional AJ cost counts:

```text
AJ drop > 1 point: 10 / 30 videos
AJ drop > 2 points: 3 / 30 videos
```

B2-W16-P2 vs online:

| Metric | mean delta | 95% CI | positive videos | sign-test p |
|---|---:|---:|---:|---:|
| AJ_RD_256 | +0.0390 | [0.0166, 0.0669] | 23/30 | 0.002316 |
| AJ_256 | +34.3731 | [32.9299, 35.7316] | 30/30 | 0.000000 |

Online vs offline:

| Metric | mean delta | 95% CI | positive videos | sign-test p |
|---|---:|---:|---:|---:|
| AJ_RD_256 | +0.0223 | [-0.0185, 0.0560] | 22/30 | 0.008130 |
| AJ_256 | -34.9011 | [-36.3717, -33.4044] | 0/30 | 0.000000 |

### Interpretation

The main natural gain is video-consistent:

```text
B2-W16-P2 improves AJ_RD on 24/30 fresh videos.
The video-level mean AJ_RD gain is +0.0613 with 95% CI [0.0419, 0.0805].
```

The AJ cost is measurable but controlled:

```text
Mean AJ cost is -0.5280.
Only 3/30 videos drop by more than 2 AJ points.
```

This supports the central method claim:

```text
B2-W16-P2 improves re-entry recovery while preserving most standard tracking performance.
```

---

## 3. Oracle upper-bound analysis

Protocol:

```text
For each eligible re-entry query, choose the candidate branch over offline iff the candidate has higher per-query AJ_RD_256.
Non-reentry queries remain offline.
```

This is a per-query oracle, not a deployable method. It estimates headroom for a learned or stronger re-entry gate.

### RGB fresh20-49 natural

| Method | AJ_RD_256 | AJ_256 | OA_256 |
|---|---:|---:|---:|
| offline | 0.3816 | 79.5944 | 91.4636 |
| online_global | 0.4121 | 44.6933 | 55.7585 |
| B2-W16-P2 | 0.4454 | 79.0664 | 92.8804 |
| oracle_b2 | 0.4597 | 79.5633 | — |
| oracle_online | 0.4677 | 78.2940 | — |

Oracle selection:

```text
oracle_b2 selected 3205 / 6461 eligible re-entry queries = 49.61%
oracle_online selected 3103 / 6461 eligible re-entry queries = 48.03%
```

Headroom:

```text
oracle_b2 vs B2-W16-P2: +0.0143 AJ_RD, +0.4969 AJ
oracle_online vs B2-W16-P2: +0.0223 AJ_RD, -0.7724 AJ
```

### Frozen fresh20-49 translate L16

| Method | AJ_RD_256 | AJ_256 |
|---|---:|---:|
| offline | 0.4788 | 75.1940 |
| online_global | 0.4981 | 43.0978 |
| B2-W16-P2 | 0.5310 | 74.8457 |
| oracle_b2 | 0.5437 | 75.3044 |
| oracle_online | 0.5545 | 74.1730 |

Oracle selection:

```text
oracle_b2 selected 3840 / 8104 eligible re-entry queries = 47.38%
oracle_online selected 3676 / 8104 eligible re-entry queries = 45.36%
```

Headroom:

```text
oracle_b2 vs B2-W16-P2: +0.0127 AJ_RD, +0.4587 AJ
oracle_online vs B2-W16-P2: +0.0235 AJ_RD, -0.6727 AJ
```

### Frozen fresh20-49 occluder L16

| Method | AJ_RD_256 | AJ_256 |
|---|---:|---:|
| offline | 0.6311 | 77.6280 |
| online_global | 0.6146 | 43.1490 |
| B2-W16-P2 | 0.6588 | 76.7991 |
| oracle_b2 | 0.6732 | 77.6702 |
| oracle_online | 0.6820 | 76.0199 |

Oracle selection:

```text
oracle_b2 selected 5722 / 14020 eligible re-entry queries = 40.81%
oracle_online selected 5652 / 14020 eligible re-entry queries = 40.31%
```

Headroom:

```text
oracle_b2 vs B2-W16-P2: +0.0144 AJ_RD, +0.8711 AJ
oracle_online vs B2-W16-P2: +0.0232 AJ_RD, -0.7792 AJ
```

### Interpretation

The oracle analysis gives a balanced message.

```text
1. B2-W16-P2 captures most of the available rule-level gain while keeping AJ high.
2. There remains consistent but modest headroom of about +0.013 to +0.014 AJ_RD for a better gate over the current B2 windows.
3. An oracle over the online branch can reach higher AJ_RD, but with more AJ cost, showing the same re-entry/standard-tracking tradeoff.
```

Recommended wording:

```text
Oracle routing suggests that better re-entry gating could further improve AJ_RD by roughly 1--1.5 points without sacrificing AJ, motivating future learned reliability gates. However, the current training-free B2-W16-P2 already realizes most of the attainable gain under a simple local override design.
```

## Artifacts

```text
scripts/eval_rgb_fresh20_49_natural_ablation.py
scripts/audit_rgb_fresh20_49_natural_statistics.py
scripts/eval_b2_oracle_upper_bound.py
outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_natural_ablation/summary.json
outputs/paper_discovery_2026-06-27/rgb_stacking_fresh20_49_natural_statistics/summary.json
outputs/paper_discovery_2026-06-27/b2_oracle_upper_bound/summary.json
```
