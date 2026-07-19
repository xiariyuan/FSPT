# Route-D oracle state transplant Gate 1 plan — 2026-07-19

## 1. Motivation after Gate 0

Gate 0 proves that CoTracker3's full commit-time state is causally meaningful:
a fixed composite corruption changes future coordinates by `17.29px` on average,
coordinate-only restoration leaves `1.50px` average error and a large tail, while
full-state restoration exactly reproduces the clean future.

That result is necessary but insufficient. A learned denoiser trained only to
recover an uncorrupted native state cannot improve beyond native tracking. It
would learn robustness to artificial perturbation, not recovery from a natural
tracking failure.

Gate 1 therefore asks a stronger question:

> Does there exist a causal latent state, constructible with GT only during
> training, whose transplant improves the future of a naturally failed online
> track?

## 2. Oracle latent-state teacher

For each selected natural failure, create an independent frozen CoTracker3
fresh-query branch. Its query is placed at the exact GT coordinate on the latest
GT-visible overlap frame in `8--15`. The fresh branch processes the same first
16 frames and exposes:

```text
fresh overlap coordinates;
fresh visibility/confidence logits;
fresh four-level center track features;
fresh four-level 7x7 correlation support memory.
```

These tensors are copied into the corresponding point slot of the original
online state. Original predictor queries and query times remain unchanged. GT is
used only to create the training teacher and to score fit-only future behavior.

This is not the same as the earlier GT fresh-query oracle. The earlier audit ran
a separate fresh tracker. Gate 1 tests whether its latent state can be
**transplanted into an existing online trajectory** and decomposes which state
fields are responsible for future improvement.

## 3. Frozen fit-only sample set

```text
manifest: Kubric train index
source indices: 0--7
videos: 8
frames per video: 24
points per video: 64
window / step: 16 / 8
commit snapshot: after frames 0--15
future: frames 16--23
```

No model-validation, calibration, final-holdout, DAVIS, or Kinetics data may be
read.

## 4. Natural failure selection

A point is eligible only if:

```text
its original query frame is before frame 8;
GT is visible in at least one overlap frame 8--15;
GT is visible in at least four future frames 16--23;
native mean future error on those visible frames is at least 16px.
```

At most eight points are retained per video, ordered by descending native future
mean error and then point index. There is no fallback to easier points if fewer
than 16 natural failures exist across the eight videos.

This rule is frozen before result generation and prevents selecting teacher-friendly
points after observing transplant behavior.

## 5. Frozen variants

```text
N — native continuation
C — copy oracle fresh overlap coordinates only
P — copy oracle coordinates plus visibility/confidence logits
F — copy coordinates, probabilities, track features, and track support
```

For each point, fresh overlap tensors are copied only from its oracle query frame
through frame 15. Earlier overlap state remains native. This keeps the transplant
causal and avoids inserting undefined pre-query fresh predictions.

## 6. Evaluation

Only GT-visible future rows in frames 16--23 are evaluated. Per point:

```text
mean L2 error in 256px raster;
16px severe-error rate;
average correctness utility at 1/2/4/8/16px.
```

The full-state support delta is also decomposed by SVD at ranks 1, 2, 4, 8, and
16. This does not select a rank; it determines whether a future restorer can use
a compact low-rank state action or would need to predict the full 7x7x128 tensor.

## 7. Scientific gates

The oracle teacher is useful only if all are true:

```text
selected points >= 16 across at least 4 videos;

F versus native:
  mean error reduction >= 4px;
  threshold-utility gain >= 0.10;
  positive point fraction >= 70%;
  severe-16px rate not worse;

F versus coordinate-only:
  mean error reduction >= 0.50px;
  threshold-utility gain >= 0.02;
  better on at least 60% of videos with selected events;

F versus coordinate+probability:
  mean error reduction >= 0.25px;
  threshold-utility gain >= 0.01.
```

The final comparison is essential. If full state does not beat coordinate and
probability transplantation, there is no justification for learning the large
support-memory action.

## 8. What a pass would authorize

```text
AUTHORIZE_FIT_ONLY_COUNTERFACTUAL_STATE_RESTORER_TRAINING_PROTOCOL
```

A pass authorizes a separately preregistered learned model that imitates the
oracle latent state from causal evidence. The likely action parameterization is
not a dense 1.6-million-value output. It should predict:

```text
bounded overlap coordinate residuals;
visibility/confidence residuals;
a compact low-rank support correction or a coordinate-conditioned support
re-extraction action;
a strict no-op probability for already-clean states.
```

Training must include direct teacher-state reconstruction and future-rollout
validation. The frozen CoTracker model is not fine-tuned.

## 9. Stop rule

Failure yields:

```text
STOP_ORACLE_STATE_TRANSPLANT_BEFORE_LEARNED_RESTORER
```

No event-threshold, oracle-frame, source-index, transplant-field, or future-window
sweep is allowed after observing Gate 1.

## 10. Claim boundary

Even a pass is fit-only oracle evidence. It does not establish inference-time
localization, external transfer, or paper-level improvement. DAVIS remains a
failed frozen audit and cannot be used to train or rescue this branch. Official
Kinetics 1,144 remains frozen and is not rerun.

## 11. Completed result — 2026-07-19

Gate 1 passed under the frozen protocol.

```text
selected points:                         63
selected videos:                         8 / 8
native future mean error:                60.7129 px
coordinate-only future mean error:       42.3839 px
coordinate+probability mean error:       43.2158 px
full-state future mean error:             5.8543 px
full-state vs native error reduction:    54.8586 px
95% CI:                                  [45.9531,63.9536]
full-state positive points:              63 / 63
full-state vs coordinate-only reduction: 36.5296 px
full-state better videos:                 8 / 8
independent replay exact:                true
```

Formal decision:

```text
AUTHORIZE_FIT_ONLY_COUNTERFACTUAL_STATE_RESTORER_TRAINING_PROTOCOL
```

See:

```text
docs/ROUTED_ORACLE_STATE_TRANSPLANT_GATE1_RESULT_2026-07-19.md
docs/generated/ROUTED_ORACLE_STATE_TRANSPLANT_GATE1_SUMMARY_2026-07-19.json
```
