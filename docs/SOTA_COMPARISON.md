# SOTA Comparison Status

> Status: historical draft, superseded by `CURRENT_MAINLINE.md`.

This document previously contained paper-style SOTA comparison tables and claimed FSPT improvements such as `67.5% AJ`. Those claims are not supported by the current runnable mainline and must not be cited as current results.

## Current authoritative status

Use these files instead:

- [`../CURRENT_MAINLINE.md`](../CURRENT_MAINLINE.md)
- [`current_redetection_route_closure_2026-06-26.md`](current_redetection_route_closure_2026-06-26.md)
- [`route_b_engineering_foundation_plan_2026-06-26.md`](route_b_engineering_foundation_plan_2026-06-26.md)

## Current research position

The current project is in a re-entry / re-detection pivot state:

- The DINOv2 local feature / pseudo-label / verifier / local-refiner route is closed.
- `true_AJ_RD_256` is the paper-facing canonical AJ_RD metric.
- Raw CoTracker3 offline is the strongest currently available re-entry signal in the existing audit chain.
- Route A now depends on evaluating external stronger teachers such as TAPNext++, Track-On-R, or AllTracker.
- Route B is engineering foundation work: package structure, path helpers, metric wrappers, and maintainability.
- Route C is an analysis / benchmark paper if Route A does not produce a stronger teacher.

## What this document no longer asserts

This document no longer asserts:

- FSPT has a reproducible SOTA result.
- FSPT achieves 67.5 AJ on TAP-Vid-DAVIS.
- FSPT achieves 55.2 AJ on TAP-Vid-Kinetics.
- FSPT outperforms CoTracker3, TAPNext, or other public trackers.
- The frequency-semantic tracker is the current live method mainline.

## Future use

If Route A obtains and validates a stronger teacher, this document can be rebuilt from generated evaluation artifacts. Until then, it should be treated only as a placeholder for future comparison tables.
