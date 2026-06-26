# FSPT Package Path Helper Manifest (2026-06-26)

## What was added

- `fspt/paths.py`
- `fspt/coords.py`
- `fspt/reentry_metrics.py`
- `fspt/core/paths.py`
- `fspt/core/__init__.py` updated to re-export core helpers
- `fspt/scripts/` thin wrappers
- `pyproject.toml` (setuptools.build_meta, Python >= 3.10)
- `fspt/smoke.py` (real assertions, not just version print)

## What was modified

- `verify_project.py` now resolves the repository root through `fspt.core.paths.repo_root()` with a safe fallback.
- `tests/test_fspt_package.py` now checks package root, wrapper imports, smoke entrypoint, and top-level `fspt.paths/coords/reentry_metrics` imports.
- `cotracker_refiner.py` `_project_root()` now calls `fspt.paths.repo_root()`.
- `scripts/build_ctoffline_local_refiner_dataset.py` PROJECT_ROOT and pkl/DINO paths now use `fspt.paths`.
- `scripts/build_ctoffline_grid_verifier_dataset.py` PROJECT_ROOT and pkl/DINO paths now use `fspt.paths`.
- `utils/reentry_metrics.py` `aggregate_reappearance_ajrd` now accepts `summary_key` parameter for 256-space aggregation.

## Unified import style

Preferred imports for active code:

```python
from fspt.paths import repo_root, resolve_repo_path, outputs_dir, caches_dir
from fspt.coords import yx_norm_to_xy_pixel, find_reentry_events, pixel_l2_error
from fspt.reentry_metrics import aggregate_reappearance_ajrd, compute_reappearance_segment_aj
from fspt.io.attempt0_schema import load_attempt0_cache
```

Legacy `utils.*` remains for compatibility. Do not copy implementations into new files; re-export from `fspt.*`.

## Hardcoded paths replaced in this pass

- `verify_project.py` root resolution → `fspt.core.paths.repo_root()`
- `cotracker_refiner.py` `_project_root()` → `fspt.paths.repo_root()`
- `scripts/build_ctoffline_local_refiner_dataset.py` PROJECT_ROOT → `fspt.paths.repo_root()`
- `scripts/build_ctoffline_local_refiner_dataset.py` pkl default → `resolve_repo_path()`
- `scripts/build_ctoffline_local_refiner_dataset.py` DINO weights → `resolve_repo_path()`
- `scripts/build_ctoffline_grid_verifier_dataset.py` PROJECT_ROOT → `fspt.paths.repo_root()`
- `scripts/build_ctoffline_grid_verifier_dataset.py` pkl default → `resolve_repo_path()`
- `scripts/build_ctoffline_grid_verifier_dataset.py` DINO weights → `resolve_repo_path()`

## Remaining hardcoded paths

Run `grep -R "/gemini/code/FSPT" -n . --exclude-dir=.git --exclude-dir=outputs` for full list. Main remaining sites are in `scripts/auto_poll_sota.sh` and some `docs/` historical files.

## Canonical metric

Paper-facing canonical: `true_AJ_RD_256` (256-space, TAPNext++ comparable).
Original-resolution `true_AJ_RD` is internal diagnostic only.

## Notes

- No training routes were reopened.
- The closed DINO/local-refiner/verifier stack remains closed.
