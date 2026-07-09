# Candidate-aware ReEntry Selector v0 — 2026-07-02

## Goal

Turn the candidate-pool oracle headroom into a deployable runtime selector.

Training data:

```text
dev_translate_L16 + dev_occluder_L16
n = 23,666 examples
```

Label = per-query candidate with best AJ_RD among:

```text
offline
online_global
b2_fullpost_p1
b2_w8_p2
b2_w16_p1
b2_w16_p2
b2_w32_p2
```

Class distribution:

```text
offline:        19,142
online_global:   2,901
b2_w8_p2:          891
b2_w16_p1:         312
b2_w32_p2:         229
b2_fullpost_p1:    105
b2_w16_p2:          86
```

Evaluation target:

```text
RGB fresh20-49 natural
```

## Baselines

```text
offline:          AJ_RD=0.3816, AJ=79.5944, OA=91.4636
B2-W16-P2:        AJ_RD=0.4454, AJ=79.0664, OA=92.8804
ReEntry-Guard RF: AJ_RD=0.4499, AJ=79.1749, OA=92.0864
```

Oracle headroom:

```text
candidate-pool oracle:          AJ_RD=0.4719, AJ=78.6150, OA=90.7152
expanded candidate-pool oracle: AJ_RD=0.4741, AJ=78.5977, OA=90.8134
```

## Runtime selector v0 result

Best selector under AJ-preservation preference:

```text
logreg_conf0.50: AJ_RD=0.4070, AJ=79.1556, OA=91.6772
```

Gain vs ReEntry-Guard:

```text
AJ_RD -0.0429
AJ    -0.0193
OA    -0.4092
```

Most aggressive selector variants by AJ_RD:

```text
logreg_conf0.00: AJ_RD=0.4401, AJ=78.0478, OA=91.4442
logreg_conf0.20: AJ_RD=0.4400, AJ=78.0497, OA=91.4469
logreg_conf0.30: AJ_RD=0.4329, AJ=78.3930, OA=91.6838
random_forest_conf0.00: AJ_RD=0.4319, AJ=76.4620, OA=89.2229
extra_trees_conf0.20: AJ_RD=0.4314, AJ=76.2116, OA=89.0265
```

None beats B2 or ReEntry-Guard.

## Diagnosis

The key problem is not candidate-pool headroom. The headroom exists:

```text
ReEntry-Guard:              AJ_RD=0.4499
candidate-pool oracle:      AJ_RD=0.4719
expanded candidate oracle:  AJ_RD=0.4741
```

The key problem is oracle-to-runtime transfer:

```text
The selector trained on dev stress labels cannot reliably identify which candidate will be best on natural fresh20-49 using the current runtime features.
```

The most likely causes are:

```text
1. Domain gap: dev synthetic stress labels do not transfer cleanly to natural RGB fresh20-49.
2. Feature weakness: current runtime features describe visibility/disagreement/motion but not true point identity or appearance consistency.
3. Severe class imbalance: offline dominates; rare candidates such as W16-P2/fullpost are underrepresented.
4. Candidate utility ambiguity: argmax AJ_RD labels are too aggressive and do not encode AJ/OA no-harm constraints well.
5. Candidate quality gap: expanded coordinate/visibility hybrids only add +0.0022 oracle AJ_RD, so simple visibility swapping is not a major breakthrough.
```

## Decision

Do not continue ordinary tabular multi-class selector tuning as the main path.

Next stronger options:

```text
A. Improve selector supervision:
   use AJ-budget utility labels instead of pure argmax AJ_RD.

B. Improve features:
   add appearance/identity consistency around the re-entry window.

C. Improve candidate quality:
   build local coordinate correction / proposal repair instead of only selecting among existing branches.

D. Use a two-stage selector:
   detect whether to override first, then choose window/candidate only among high-confidence re-entry events.
```

Current biggest blocker:

```text
The candidate-pool oracle shows a better answer exists, but the runtime selector cannot recognize it from current features.
```
