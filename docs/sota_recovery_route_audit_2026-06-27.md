# SOTA Recovery Route Audit — 2026-06-27

## Decision

Stop training the current FSPTTracker student for SOTA pursuit.

The next route is **teacher-first**:

1. use existing strong teacher / ensemble outputs as the performance floor;
2. build a deployable teacher selector / gate from the existing oracle labels;
3. only after selector feasibility, consider a base-track-conditioned refiner.

## Why the current student route failed

The evaluator sanity check passed: GT-as-prediction gives AJ=1.0 and OA=1.0.

The direct exported masked-median base tracks are much stronger than the trained student under the same local TAP metric wrapper:

| system | AJ | OA | <4px | median px |
|---|---:|---:|---:|---:|
| exported masked-median base tracks | ~0.3882 | ~0.6961 | ~0.6821 | ~2.85 |
| best current student C2-GTlite-pos0.05 | 0.0980 | 0.7198 | 0.0929 | 24.44 |

Root cause: `models/point_tracker.py::FSPTTracker.forward()` takes only `video + query_points`. It does **not** consume `base_tracks` as an input trajectory. `base_tracks` are only used in loss. Therefore the model is a weak from-scratch student, not a teacher-preserving refiner.

## Existing teacher/ensemble headroom

From `outputs/oracle_teacher_selection_2026-06-26.json`:

- fixed best teacher: `cotracker3_offline`
- fixed best `true_AJ_RD_256`: `0.5546`
- oracle teacher selection `true_AJ_RD_256`: `0.6509`
- oracle gap: `0.0963`

A compact oracle-label table was exported to:

`outputs/route_sota_recovery_2026-06-27/teacher_selector_labels_from_oracle.jsonl`

## Immediate next steps

### Step A — no-training floor

Treat `visibility_masked_median` / existing teachers as the deployment floor. Any learned module must beat this, not the weak student baseline.

### Step B — teacher selector A0

Use oracle labels from `teacher_selector_labels_from_oracle.jsonl` to build a lightweight selector dataset. Candidate features:

- teacher visibility/confidence at re-entry;
- teacher disagreement distances;
- motion smoothness / jump magnitude;
- occlusion length;
- in-bounds flags;
- per-teacher local consistency.

Target: choose among `cotracker3_online`, `cotracker3_offline`, `trackon2`.

### Step C — base-track-conditioned refiner only if selector has signal

If selector features cannot recover enough oracle gap, do not train a refiner yet. If selector shows signal, build a refiner whose forward input is:

`video + query_points + base_tracks + base_visibility + teacher_id/confidence`

and predicts only small deltas / visibility corrections.

## Hard rule

Do not restart the current `FSPTTracker(video, query_points)` student route for SOTA.
