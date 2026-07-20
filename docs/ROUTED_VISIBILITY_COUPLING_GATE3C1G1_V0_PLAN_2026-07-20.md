# Route-D Visibility-Coupling Gate 3C1G1 v0 Plan — 2026-07-20

## Purpose

Gate 3C1F2 showed that the frozen coordinate-plus-memory action is accurate and safe on action rows, but complete-video AJ is cancelled by post-writeback visibility coupling. Gate 3C1G0 built a 66-D causal cache on the 44 already exposed action videos. Gate 3C1G1 asks whether a post-writeback visibility model can recover enough complete-video AJ under strictly nested source-video OOF evaluation to justify a new raw-record-disjoint confirmation.

This is an exposed-development design gate. It cannot authorize DAVIS, Kinetics, final holdout, official Kinetics 1,144, or a paper-table claim.

## Frozen data

```text
videos:      44 exposed Gate 3C1F2 action videos
actions:     89 sealed actions
frame rows:  801, frames 15--23
features:    66 causal channels
group:       source video
```

The cache index authority is:

```text
file SHA256:
5c690647e8fee4ae812e24f132622f3941fa12682b5e000f7ce08851d2493b45

payload SHA256:
a990040bcd8a86808451c8e77c6bed3b38ec2032522b9ef3bd1372f08dc1ea72
```

GT visibility, coordinate errors, threshold hits, and categories are labels/evaluation tensors only. They are prohibited from the 66-D model input.

## Nested OOF protocol

```text
outer folds: 5 GroupKFold by source video
inner folds: 4 GroupKFold by source video
model candidates: 10
thresholds: 0.05--0.95 in 0.025 increments
```

For every outer fold, model family, target, regularization, and binary visibility threshold are selected only from inner OOF predictions on the outer-training videos. The selected model is then refit on the outer-training rows and evaluated once on held-out outer videos. All 801 outer predictions are assembled exactly once.

Candidate targets are:

```text
GT-visible classification
GT-visible-and-within-16px utility classification
mean correctness across 1/2/4/8/16px regression
```

Candidate models are regularized logistic regression and deterministic histogram gradient boosting. Per-row training weights make every source video contribute equally.

## Selection objective

Inner ranking first requires:

```text
OA change vs actual modified >= -0.10 points
GT-visible recall >= 65%
occluded false-positive rate <= 50%
```

Feasible candidates are ranked by complete-video AJ gain over the actual Gate 3C1F2 modified output, then OA preservation, lower false-positive rate, higher recall, and lower predicted-visible support. Candidate and threshold order are final deterministic tie-breaks.

## Formal gates

Nested outer OOF must satisfy all of:

```text
GT-visible AUC >= 0.70
GT-visible AP >= 0.65
GT-visible recall >= 70%
occluded false-positive rate <= 40%
AJ gain over actual modified >= +0.15 points
paired-video AJ CI lower bound > 0
AJ gain over native >= +0.15 points
OA change vs actual modified >= -0.05 points
OA change vs native >= 0
delta_avg unchanged exactly
fresh-process scientific replay exact
```

The +0.15 action-video AJ target is chosen because only 44/128 Gate 3C1F2 videos contain actions; it is the minimum scale plausibly capable of producing the preregistered +0.05-point complete-population AJ target after dilution.

## Decision

Pass:

```text
AUTHORIZE_GATE3C1G2_RAW_DISJOINT_VISIBILITY_CONFIRMATION_DATA
```

Fail:

```text
STOP_GATE3C1G1_VISIBILITY_MODEL
```

No threshold, feature, target, or model-family change is allowed after primary results are read. A pass authorizes only a separately preregistered new raw-record-disjoint confirmation.
