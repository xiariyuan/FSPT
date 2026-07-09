# Experimental Rigor Rules — 2026-07-06

This project is now treated as a paper-experiment workflow, not an exploratory coding playground.

## Non-negotiable rules

1. No full run before a written plan.
2. Every new exporter or adapter must pass 1-video smoke first.
3. Smoke must check schema, query/GT alignment, and metric parity when an older cache exists.
4. Main paper rows require real base tracker output plus the ReEntry module.
5. Stress variants, window-reset variants, offline-batch variants, and non-parity caches are appendix/diagnostic only.
6. Do not overwrite old caches; use new versioned names.
7. Every paper-relevant result must report AJ, OA, delta metrics, Jaccard metrics, AJ_RD, protocol, cache path, n_videos, n_queries, and whether coordinates changed.
8. Claims must follow result type: real-base success, controlled oracle, stress diagnostic, or negative result.
9. No leaderboard or SOTA language unless the protocol and source are official.
10. Every result that changes paper direction must create a doc in docs/ and append a short status to CURRENT_MAINLINE.md.

## Current TrackOn2 next-step gate

Do not run 30-video TrackOn2 confidence export yet.

First fix 1-video parity by using the old TrackOn2 first/input cache as the query/GT/schema source. Raw DAVIS video may only be used as the frame source.

Parity pass conditions:

- query_points preserved from old cache
- gt_tracks preserved from old cache
- gt_visibility preserved from old cache
- first-video AJ/OA close to old cache
- visibility difference explained and preferably small

Only after parity passes:

- export 30-video TrackOn2 true-base confidence cache
- run threshold sweep
- run local recovery audit
- apply gate: AJ >= base - 0.10, OA >= base - 0.10, AJ_RD >= base + 0.01
