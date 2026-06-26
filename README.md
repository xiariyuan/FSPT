# FSPT Repository Status

> **Current status as of 2026-06-26:** this repository is in route-closure and engineering-foundation cleanup mode. It is **not** currently a validated SOTA FSPT method repository.

Read first:

1. [`CURRENT_MAINLINE.md`](CURRENT_MAINLINE.md)
2. [`docs/current_mainline_status_2026-06-26.md`](docs/current_mainline_status_2026-06-26.md)
3. [`docs/current_redetection_route_closure_2026-06-26.md`](docs/current_redetection_route_closure_2026-06-26.md)
4. [`docs/fspt_package_path_helper_manifest_2026-06-26.md`](docs/fspt_package_path_helper_manifest_2026-06-26.md)

## What this repository is now

The active research problem is long-occlusion point re-entry / re-detection. The current work is a pivot after multiple negative results, plus Route B engineering cleanup.

Current allowed routes:

| Route | Status | Purpose |
|---|---|---|
| Route A: external stronger teacher | pending | Evaluate TAPNext++, Track-On-R, AllTracker, or similar stronger checkpoints under the unified strided+original protocol. |
| Route B: engineering foundation | active | Build an importable `fspt/` package, centralize paths / coordinates / metrics, remove hardcoded paths, and make the repo auditable. |
| Route C: analysis / benchmark paper | fallback | If Route A does not produce a stronger teacher, write the AJ_RD / null-result ladder / re-entry failure analysis. |

## Closed routes

Do not restart these without a new written gate document:

- DINOv2 whole-frame / multi-anchor retrieval.
- DINOv2 local feature pseudo-label rollout.
- CT-offline-centered local refiner.
- CT-offline-centered grid verifier.
- Learned offset regression on frozen DINOv2 features.
- P4 student training on the old pseudo-label route.

Current labels:

- `P0 = BLOCKED_METRIC_RECONCILIATION`
- `P1 = WEAK_DIAGNOSTIC_ONLY`
- `P2 = PARTIAL_LOCAL_FEATURE_SMOKE`
- `P3A = SMOKE_STOP`
- `P4 = DO_NOT_START`

## What this repository no longer claims

The original frequency-semantic point-tracking proposal is historical. The repository no longer claims:

- a reproducible FSPT SOTA tracker;
- `67%+` TAP-Vid-DAVIS AJ;
- superiority over CoTracker3, TAPNext, or other public trackers;
- an active CoTracker3 + FSPT refinement training route;
- an active DINO/local verifier/refiner training route.

Historical documents may still contain old plans or paper-draft language. Treat `CURRENT_MAINLINE.md` and the 2026-06-26 status documents as authoritative.

## Canonical metric

- Paper-facing canonical re-entry metric: `true_AJ_RD_256`.
- Original-resolution `true_AJ_RD` is internal diagnostic only.

## Package / foundation status

The `fspt/` package scaffolding is now the preferred import root for new code.

Examples:

```python
from fspt.paths import repo_root, resolve_repo_path, outputs_dir, caches_dir
from fspt.coords import yx_norm_to_xy_pixel
from fspt.reentry_metrics import aggregate_reappearance_ajrd
```

Smoke checks:

```bash
python -c "import fspt"
python -c "from fspt.paths import repo_root; print(repo_root())"
python -c "from fspt.coords import yx_norm_to_xy_pixel"
python -c "from fspt.reentry_metrics import aggregate_reappearance_ajrd"
python -m fspt.smoke
```

## Installation smoke

```bash
pip install -e .
python -m fspt.smoke
```

## Repository caveats

This repository still contains legacy code paths and old experiment scaffolds:

- `models/` contains the original frequency-semantic tracker code.
- `cotracker_refiner.py` and `models/cotracker_refiner.py` contain legacy Route A / local refiner code and closed-route branches.
- `projects/mmp_tracker/` is an independent experimental direction and is not yet the current mainline.
- Many `docs/` and `outputs/` files are historical evidence rather than current instructions.

Do not start training from old README-era commands. Any new training route must be reopened by a current gate document.

## Next priorities

1. Finish Route B cleanup enough that active audits are importable and reproducible.
2. Obtain external teacher checkpoints for Route A and evaluate them under the unified protocol.
3. If Route A does not produce a stronger teacher, prepare Route C analysis / benchmark paper artifacts.
