# V9-A2.5 DINOv3 Identity Feature Pilot Design

Date: 2026-07-10

## Motivation

V9-A2.4 shows that the event-level dynamic-horizon action space is useful, but RGB patch anchor features provide weak ranking and the paired video-level gain versus W16 is not statistically established.

V9-A2.5 tests whether a semantic dense descriptor adds complementary identity evidence on the exact current DAVIS W16-extension rows.

## Frozen action protocol

```text
Preserve all W8 common rows.
Control W16-extension rows only.
Use event_max aggregation.
No dense trajectory threshold search.
```

## Semantic feature source

```text
Local DINOv3 ViT-S/16+ checkpoint:
third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m
```

For each W16-extension row, sample dense descriptors at:

```text
query anchor
last causal anchor
strict pre-occlusion anchor
TrackOn2 candidate at tau
native CoTracker point at tau
local one/two-token negatives around the candidate
```

## Feature families

```text
anchor-candidate cosine/L2
anchor-native cosine/L2
multi-anchor memory-candidate/native cosine/L2
anchor consistency
candidate-native semantic agreement
candidate-vs-native semantic advantage
local-negative margins / distinctiveness
```

## OOF models

```text
DINO-only
base + DINO
all existing + DINO
```

Use video-group-heldout OOF. Primary target remains `candidate_good`.

## Apply-back protocols

```text
event_max + OOF-F1
logistic event_max + fixed 0.05 (calibration comparison only)
```

Compare against:

```text
W8
W16
frozen V9-A2 fixed0.05
oracle
```

## Decision

Continue semantic identity only if it improves heldout ranking and/or trajectory apply-back over frozen V9-A2 without relying on a dense threshold sweep.
