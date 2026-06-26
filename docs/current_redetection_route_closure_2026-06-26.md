# Current Redetection Route Closure (2026-06-26)

## Closed status

The current DINO/local-refiner/verifier route is formally closed.

### Finalized negative result

- `P0 = BLOCKED_METRIC_RECONCILIATION`.
- `P1 = WEAK_DIAGNOSTIC_ONLY`.
- `P2 = PARTIAL_LOCAL_FEATURE_SMOKE`.
- `P3A = SMOKE_STOP`.
- `P4 = DO_NOT_START`.

### Explicitly closed routes

- DINOv2 whole-frame retrieval.
- DINOv2 multi-anchor retrieval.
- DINOv2 local feature pseudo-label rollout.
- CT-offline-centered local refiner.
- CT-offline-centered grid verifier.
- Learned offset regression on frozen DINOv2 features.
- Any continued training on these routes.

### What remains true

- Re-entry / re-detection is still a valid problem statement.
- `CT-offline` raw prediction remains the strongest currently available re-entry signal.
- Oracle headroom exists, but the current public frozen feature family cannot reliably learn to use it.
- The failure is in the current implementation stack, not in the existence of the problem.

### Canonical metric

- Paper canonical: `true_AJ_RD_256`.
- Internal diagnostic only: original-resolution `true_AJ_RD`.
- The canonical metric choice is part of the reconciliation closure for P0.

### Do not do

- Do not keep patching the current DINO/local refiner/verifier stack.
- Do not expand pseudo-label rollout.
- Do not start P4.
- Do not treat this route as a live training mainline.

### Next allowed directions

- External stronger teacher evaluation.
- Engineering foundation cleanup.
- Analysis / benchmark paper planning.
