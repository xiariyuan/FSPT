# FSPT Package Path Helper Manifest (2026-06-26)

## What was added

- `fspt/paths.py`
- `fspt/coords.py`
- `fspt/reentry_metrics.py`
- `fspt/core/paths.py`
- `fspt/core/__init__.py` updated to re-export core helpers
- `fspt/scripts/` thin wrappers

## What was modified

- `verify_project.py` now resolves the repository root through `fspt.core.paths.repo_root()` with a safe fallback.
- `tests/test_fspt_package.py` now checks both the package root and the wrapper imports.

## Unified import style

Preferred imports for active code:

```python
from fspt.paths import repo_root, resolve_repo_path, outputs_dir, caches_dir
from fspt.coords import yx_norm_to_xy_pixel
from fspt.reentry_metrics import aggregate_reappearance_ajrd
```

Legacy compatibility is still available through `utils.*`, but active code should move to `fspt.*`.

## Hardcoded paths replaced so far

- `verify_project.py` root resolution
- `fspt.paths` now provides repository-relative resolution helpers

## Hardcoded paths still to replace

- `scripts/build_ctoffline_local_refiner_dataset.py`
- `scripts/build_ctoffline_grid_verifier_dataset.py`
- `cotracker_refiner.py` root helper and checkpoint resolution sites

These are left for the next pass because they require careful behavior checks and may depend on model assets.

## Smoke tests run

- `python -c "import fspt"`
- `python -c "from fspt.paths import repo_root"`
- `python -c "from fspt.coords import yx_norm_to_xy_pixel"`
- `python -c "from fspt.reentry_metrics import aggregate_reappearance_ajrd"`
- `python -m py_compile fspt/__init__.py fspt/paths.py fspt/coords.py fspt/reentry_metrics.py fspt/core/paths.py verify_project.py`

## Canonical import policy

Active code should use:

```python
from fspt.paths import repo_root, resolve_repo_path, outputs_dir, caches_dir
from fspt.coords import yx_norm_to_xy_pixel, find_reentry_events, pixel_l2_error
from fspt.reentry_metrics import aggregate_reappearance_ajrd, compute_reappearance_segment_aj
from fspt.io.attempt0_schema import load_attempt0_cache
```

Legacy `utils.*` remains for compatibility. Do not copy implementations into new files; re-export from `fspt.*`.

## Notes

- No training routes were reopened.
- The closed DINO/local-refiner/verifier stack remains closed.
