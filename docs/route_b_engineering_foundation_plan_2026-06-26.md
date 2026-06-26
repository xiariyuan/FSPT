# Route B Engineering Foundation Plan (2026-06-26)

## Purpose

Route B is engineering foundation work. It is not a new training route and must not re-open the closed DINO/local-refiner/verifier stack.

The goal is to make FSPT importable, testable, and less path-dependent before Route A teacher evaluation or Route C paper packaging continues.

## Current risks found by scan

### Oversized / monolithic files

- `train.py` (~6.1k lines)
- `models/cotracker_refiner.py` (~5.6k lines)
- root duplicate `cotracker_refiner.py` (~4.8k lines)
- large dataset scripts around TAP-Vid Kubric / Kinetics

### Duplicate module roots

- Root-level modules duplicate package-like files, e.g. `cotracker_refiner.py`, `freq_semantic_fusion.py`, TAP-Vid files.
- `models/`, `datasets/`, and `utils/` are importable but not under a single project package.
- `projects/mmp_tracker/` already has its own nested package and should remain isolated until explicitly promoted.

### Path / environment fragility

- Many scripts use `sys.path.insert(...)`.
- Multiple defaults reference `outputs/`, `datasets/`, `baselines/`, or absolute paths such as `/gemini/code/...`.
- Reproducibility depends on running from the repository root.

## Package target

Introduce `fspt/` as the stable import root.

Initial layout:

```text
fspt/
  __init__.py
  smoke.py
  core/
  metrics/
  data/
  models/
  experiments/
  cli/
```

Do not move everything at once. Use wrappers first, then migrate modules once tests cover the import paths.

## Migration phases

### Phase B0: package smoke and single entry point

Status: started.

- Add `fspt/__init__.py`.
- Add `fspt/smoke.py`.
- Add `setup.py`.
- Add `CURRENT_MAINLINE.md`.
- Add a test that imports `fspt` and runs the smoke entrypoint.

Gate:

```bash
python -m fspt.smoke
python -m py_compile fspt/__init__.py fspt/smoke.py setup.py
```

### Phase B1: centralize pure utilities

Move or wrap pure utility modules first. Low risk because they do not instantiate models.

Priority:

1. `utils/coords.py` -> `fspt/core/coords.py`
2. `utils/reentry_metrics.py` -> `fspt/metrics/reentry.py`
3. `utils/attempt0_schema.py` -> `fspt/io/attempt0_schema.py`
4. `utils/attempt0_predictions.py` -> `fspt/io/attempt0_predictions.py`

Compatibility rule:

- Keep old imports working with thin wrappers until all scripts migrate.
- No behavior change in this phase.

Gate:

- Existing re-entry metric scripts compile.
- Roundtrip coordinate tests still pass.
- `eval_aj_rd_from_cache.py` can import through either path.

### Phase B2: unify path config

Create a small config module:

```text
fspt/core/paths.py
```

Responsibilities:

- Resolve repository root.
- Resolve dataset root from env or explicit CLI args.
- Resolve output root.
- Forbid new hard-coded absolute paths in active code.

Gate:

- New active scripts must accept path args or use `fspt.core.paths`.
- No new `/gemini/code/...` literals in active mainline files.

### Phase B3: split CoTracker refiner monolith

Do not continue method work inside the monolith. Split only for maintainability.

Suggested modules:

```text
fspt/models/cotracker_refiner/
  __init__.py
  base_tracker.py
  feature_cache.py
  refinement_heads.py
  visibility.py
  prior_features.py
  debug_export.py
  config.py
```

Archive rule:

- Closed DINO/local verifier/refiner code should move to `archive/` or remain frozen with clear comments.
- Do not add new experiments to killed code paths.

Gate:

- Import parity: existing public class can still be imported.
- No training run required.
- Minimal forward-construction smoke only if dependencies are available.

### Phase B4: scripts to CLI wrappers

For active scripts only, prefer:

```bash
python -m fspt.cli.<name>
```

Keep legacy `scripts/*.py` as thin wrappers where needed.

Priority active CLI wrappers:

1. cache validation
2. AJ_RD evaluation
3. external teacher cache export/eval
4. current mainline report generation

### Phase B5: tests

Minimum tests:

- `tests/test_fspt_package.py`: package import + smoke.
- `tests/test_coords_roundtrip.py`: coordinate conversion roundtrip.
- `tests/test_reentry_metrics_minimal.py`: synthetic re-entry event metric sanity.
- `tests/test_attempt0_schema_minimal.py`: load/save small cache payload.

## Non-goals

- No full training.
- No P4 restart.
- No new DINOv2 local refiner/verifier experiment.
- No pseudo-label rollout expansion.
- No MMP-Tracker merge into `fspt/` until Route A/B decision is revisited.

## Immediate next actions

1. Add package smoke test.
2. Add `fspt/core/coords.py` wrapper around `utils.coords`.
3. Add `fspt/metrics/reentry.py` wrapper around `utils.reentry_metrics`.
4. Add path utility skeleton.
5. Compile and run minimal tests.
