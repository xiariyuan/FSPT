# Route-D counterfactual tracker-state restoration Gate 0 plan — 2026-07-19

## 1. Why a new route is necessary

The frozen CoTracker3 extension passes identity-disjoint Kubric evaluation but
fails zero-shot DAVIS on all 30 videos. Two simultaneous failures appear:

```text
candidate-oracle AJ gain on DAVIS: +0.4936
selected AJ gain on DAVIS:         -1.9170
harmful non-native rate:            2.2148%
```

The result closes the current `candidate generation -> candidate selection`
family as an external-transfer story. Changing a candidate feature, adding
another selector, or sweeping abstention cannot solve the weak oracle headroom.

Historical audits also close or strongly weaken:

```text
DINO / DINOv3 identity features;
TAPNext patch identity aggregation;
TrackOn2 internal proxies;
short-window learned verifiers;
fixed and learned candidate reranking;
beam search and independent state branches;
coordinate-only state writeback;
fresh-query respawn without reliable identity localization.
```

A scientifically distinct route must change the intervention target and the
training unit, not only the candidate score.

## 2. Proposed mechanism

We define **Counterfactual Tracker-State Restoration** as recovery of the state
that the frozen online tracker will actually consume at its next window.

Let `S_t` be CoTracker3's commit-time online state after the current window:

```text
online_ind;
online_coords_predicted;
online_vis_predicted;
online_conf_predicted;
online_track_support at every correlation level;
online_track_feat at every correlation level;
predictor queries and point count.
```

A corruption operator produces `C(S_t)`. A future learned restorer would predict

```text
R_theta(C(S_t), current visual evidence) -> S_hat_t
```

and be trained through the frozen next-window rollout:

```text
future trajectory distillation to the clean branch;
future visibility/confidence distillation;
state reconstruction where identifiable;
strict no-op consistency on uncorrupted states;
bounded residual and sparse repair regularization.
```

This is not a candidate selector. The supervision target is the future behavior
of the tracker after repairing its native recurrent state.

## 3. Difference from prior MegaDepth and recovery work

MegaDepth has already been used in this project. Existing checkpoints train the
full FSPT model on two-frame position and occlusion supervision. Their validation
AJ is approximately zero, and the available Kubric-finetuning smoke does not
show a reliable advantage over its matched no-MegaDepth initialization.

Those runs do **not**:

```text
snapshot CoTracker online state;
corrupt the state consumed by the next online window;
restore visibility, confidence, or correlation support memory;
optimize a future-window rollout against a clean teacher;
enforce exact no-op behavior on clean tracker state.
```

Likewise, prior state-write experiments edit output coordinates or a narrow
coordinate state. P0k proves that finalized output decisions are unavailable at
the provisional commit point. The new route starts from the true commit-time
state and treats the whole state as the intervention object.

MegaDepth may later supply real two-view appearance invariance, but it cannot be
the contribution by itself and is not used in Gate 0.

## 4. Gate 0 purpose

Before building a network, establish three causal facts on one frozen fit-only
Kubric sample:

1. The complete online state can be snapshotted and restored exactly.
2. A fixed composite state corruption materially changes the next-window future.
3. Restoring coordinates alone does not recover the clean future, while restoring
   the complete state does.

If fact 3 is false, there is no evidence that a full-state restorer adds a model
mechanism beyond coordinate repair, so the route stops before training.

## 5. Frozen sample and timing

```text
backbone:       CoTracker3 scaled-online true-streaming
checkpoint SHA: 205d34789f19699d64b22cf93f9b697f15f28d4025240e31532e504109837218
fit manifest:   Kubric train index
manifest SHA:   bedc1678fccfeb89ea081bf006dc28b4fb37deae1f226d39cb37a68195d0eb14
source index:   0
video name:     1680
frames:         24
points:         64
window / step:  16 / 8
snapshot:       after processing global frames 0--15
next window:    chunk beginning at frame 8
overlap state:  frames 8--15
future audit:   newly exposed frames 16--23
```

No ground truth metric is used. The clean frozen tracker continuation is the
counterfactual teacher.

## 6. Frozen corruption

The corruption is fixed before any result:

```text
coordinate state:
  add [+16,-12] input-raster pixels to overlap coordinates;

visibility and confidence state:
  subtract 4.0 from overlap logits;

track-support memory:
  retain exactly 50% of channels at every correlation level;
  choose retained channels by ascending SHA-256(seed, level, channel);
  set all other channels to zero;

active points:
  only points whose query frame is earlier than frame 16.
```

The corruption is deliberately structured and deterministic. It is not intended
to model the final training distribution; it tests whether the relevant state
fields exert causal influence on the future rollout.

## 7. Frozen variants

```text
A. clean
B. exact no-op restore
C. composite corruption, no repair
D. composite corruption, restore coordinates only
E. composite corruption, restore the complete online state
```

All variants begin from the same exact first-window snapshot. The model weights,
queries, video, and second-window input are identical.

## 8. Measurements and gates

For active post-query rows in frames 16--23, compare every variant with clean:

```text
coordinate mean and maximum L2 difference at raster 256;
fraction of rows above 1px;
visibility/confidence probability differences;
SHA-256 and exact equality for every nested state tensor;
independent-process exact replay.
```

Required gates:

```text
clean independent replay: exact
snapshot round-trip: exact
no-op restore future: exact
full-state restore future: exact

composite corruption:
  mean future coordinate difference >= 2px
  fraction above 1px >= 25%

coordinate-only restore:
  mean future coordinate difference >= 0.5px
  fraction above 1px >= 10%

full restore must be strictly better than both corrupted variants.
```

## 9. Decision boundary

Pass:

```text
AUTHORIZE_SEPARATE_FIT_ONLY_LEARNED_STATE_RESTORER_PREREGISTRATION
```

A pass authorizes only a new fit-only training protocol. It does not authorize
model validation, final holdout, DAVIS, Kinetics, or a paper claim.

Fail:

```text
STOP_COUNTERFACTUAL_STATE_RESTORATION_BEFORE_TRAINING
```

No corruption-strength or state-field sweep is allowed after observing Gate 0.

## 10. Locked data

Gate 0 must not read:

```text
Kubric model validation;
Kubric calibration;
Kubric final holdout;
DAVIS;
PointOdyssey locked partitions;
official Kinetics 1,144 or any Kinetics result used for tuning.
```
