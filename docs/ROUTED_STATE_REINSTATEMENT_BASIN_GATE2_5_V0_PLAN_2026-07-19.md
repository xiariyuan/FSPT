# Route-D state-reinstatement coordinate basin Gate 2.5 v0 — 2026-07-19

## Status

```text
PREREGISTERED_NOT_RUN
```

Frozen implementation hashes at preregistration:

```text
config:  8ede4e88755eaaa5d22b4ef0d7b4e26618227322629466f6549e0fd9b4575942
runner:  259746932185b22929d7ca07ca3d0f502d6d060927b035a3b1fd8ed9c638b077
cache:   67878d5f1aa202f9b743bb724038099989adf993a6719c84c184ff519b04c3bf
helpers: cdd1dcdbcbc9b55fcf9bb8e2e680ced3770a7eec921e3c631b3c9bf6c1538217
```

Gate 2.5 is the next active experiment. It supersedes running Gate 3A v0 as a
route-decision gate. Gate 3A v0 remains frozen and unrun and may later be used
unchanged as the pooled-native-map control inside a broader representation
audit. No result from Gate 3A v0 has been read.

## Mechanistic question

The frozen interface audit already established that memory re-extraction at the
exact commit coordinate reconstructs the four fresh-query CoTracker memory
levels. Gate 2.5 asks the missing quantitative question:

> How far may the commit coordinate be displaced before full four-level memory
> re-extraction loses its future tracking benefit?

Without this measurement, a candidate recall gate at 4 px or 8 px is an
assumption rather than a tracker-specific recovery criterion.

This experiment does not use the learned Gate 2 checkpoint, train a selector,
or test a candidate representation.

## Data boundary

Only the architecture-design-exposed source indices are authorized:

```text
0--7:   Gate 2.5 mechanism audit
8--31:  no outcome read
32--47: no read
48--63: locked and unread
```

The existing Gate 2 cache builder is extended with a `design_exposed`
partition. It applies the already frozen Gate 2 row construction and serializes
the same exact native state, frame-15 feature pyramid, teacher coordinate and
continuation tensors. These rows may not provide gradients, checkpoint
selection, action thresholds or representation hyperparameters.

The design-only cache additionally seals the input-raster teacher commit
coordinate and the teacher visibility/confidence probabilities as float32.
Gate 2's existing float16 normalized model view remains unchanged, but it is not
used for these mechanism controls because its quantization would make “zero” a
small nonzero displacement and would make the teacher-probability upper bound
inexact.

## Frozen offset schedule

For every natural failure row, the cached frame-15 teacher coordinate is
perturbed at radii:

```text
0, 1, 2, 4, 8, 12, 16 input pixels
```

Every nonzero radius uses eight fixed compass directions. Diagonal vectors are
Euclidean-normalized. Coordinates outside the 256 x 256 raster are clipped;
both nominal radius and realized post-clipping error are reported. Radius zero
appears once.

For each candidate coordinate:

1. deterministically re-extract track feature and 7 x 7 support memory at all
   four levels from the frozen frame-15 feature pyramid;
2. write only the frame-15 coordinate;
3. replace all four memory levels;
4. continue the exact cached native state through frames 16--23;
5. measure future error on visible GT rows.

## Probability controls

Two variants are frozen before execution:

```text
native_probability:
  keep native frame-15 visibility and confidence;
  primary coordinate-plus-memory mechanism result.

teacher_probability:
  write the cached fresh-query frame-15 visibility and confidence;
  probability-controlled upper bound only.
```

The difference between the two variants isolates whether an apparent basin
failure is caused by coordinate/memory displacement or by stale probability
state. The teacher-probability variant is not deployable and cannot be reported
as a method.

## Metrics and uncertainty

At every radius the audit reports:

- realized commit error and boundary-clipping fraction;
- future mean L2 error;
- severe error rate above 16 px;
- mean threshold utility at 1/2/4/8/16 px;
- error reduction, utility gain and severe-rate reduction versus native;
- positive point/direction fraction;
- per-direction outcomes;
- equal-video-weight clustered bootstrap intervals.

The old downstream gates are applied only to calibrate the largest empirically
supported candidate radius:

```text
mean error reduction              >= 8 px
error-reduction CI lower          >= 2 px
threshold utility gain            >= 0.12
utility-gain CI lower             >= 0.03
positive row fraction             >= 0.65
severe-rate reduction             >= 0.15
```

Passing or failing these checks does not authorize model validation. The output
is a measurement used to preregister Gate 3A v1 recall thresholds.

## Execution

From the frozen project environment:

```bash
python scripts/build_routeD_counterfactual_state_restorer_cache.py \
  --partition design_exposed \
  --output-root outputs/routeD_counterfactual_state_restorer_cache_20260719 \
  --device cuda \
  --resume

python scripts/audit_routeD_state_reinstantiation_basin_gate2_5.py \
  --config configs/routeD_state_reinstantiation_basin_gate2_5_v0.yaml \
  --cache-root outputs/routeD_counterfactual_state_restorer_cache_20260719 \
  --output outputs/routeD_state_reinstantiation_basin_gate2_5_20260719/primary.json \
  --device cuda
```

Before result interpretation, the execution commit, config hash, design cache
index hash and combined tensor digest must be recorded. An independent replay
must reproduce the JSON rows and aggregate values before Gate 3A v1 is frozen.

## Next decision

```text
Always:
  report the empirical state-reinstatement basin;
  use that basin to define Gate 3A v1 candidate recall radii;
  keep indices 48--63 locked.

If radius zero does not recover strongly:
  stop candidate work and audit state-write/re-extraction parity.

If only teacher_probability recovers strongly:
  solve causal probability reinstatement before candidate selection.

If native_probability has a nontrivial basin:
  preregister the geometry-preserving representation audit;
  retain the original Gate 2 pooled-native logits as its frozen control.
```

No new pretrained weights or external dataset is authorized at this stage.
