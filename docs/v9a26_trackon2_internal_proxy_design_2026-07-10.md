# V9-A2.6 TrackOn2 Internal-State Proxy Design

Date: 2026-07-10

## Motivation

External RGB and DINOv3 identity features provide heldout row-level signal but do not robustly improve the frozen V9-A2 trajectory policy. The next evidence source should be closer to the candidate generator itself.

## Important provenance limitation

The original TrackOn2 first-input cache contains only tracks and visibility. A repo-native rerun cannot reproduce every old coordinate exactly. The closest audited path is:

```text
256x256 first-input bridge
TrackOn2 DINOv3 checkpoint
M_i = 24
support_grid_size = 20
```

Therefore the exported tensors are called **internal-state proxy features**, not exact latent states of the old cached trajectory.

## Causal internal signals

At each current W16-extension row:

```text
C1/C2 correlation peak, margin, entropy
correlation at old TrackOn2 candidate and native CoTracker coordinates
top-K rerank certainty/score logits and spatial agreement
visibility logit/confidence and uncertainty logit
prediction-head offset magnitude
query-state update and memory consistency
proxy-position distance to old cached candidate
```

## Validation

The copied diagnostic forward must match official `Track_On2.track_frame` on identical state/input for:

```text
position
visibility logit
updated query feature
```

## Evaluation

Use video-group-heldout OOF, event_max and OOF-F1, preserving W8 common rows. No dense trajectory threshold search.
