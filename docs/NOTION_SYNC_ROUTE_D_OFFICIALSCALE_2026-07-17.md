# Route-D Corrected Official-Scale Notion Sync

- Synced at: `2026-07-17T09:49:19+09:00` (Asia/Tokyo)
- Branch: `beliefcal-mmp-mvp1-20260713`
- Experiment evidence commit: `7ecd07b4a03ada6b110be19543e70e89c1c2b077`
- Corrected protocol SHA-256: `61f1458ee0e62b865cde85d9f77e765b79da567add8fc277fe10599b0db13155`
- Final audit SHA-256: `8cf2af93c6629542b02062a625fc8c54ac3a14c6c65a6a3fb4111228c450408c`

## Synchronized Notion page

- [点追踪（Point Tracking / TAP）近五年论文详解与整理（2022–2025）](https://app.notion.com/p/2ec9a5d73e624dfa905521f6fc284ad9)
  - Updated the existing Route-D authority page after search and fetch verification; no duplicate project page was created.
  - Synchronized the completed official-scale experiment log, current mainline, method freeze, experiment freeze, and paper-task plan.

## Verified scope

Exact order-preserving local materialization of 1,144 of the 1,147 uniquely annotated video segments in the byte-verified official TAP-Vid-Kinetics release CSV, evaluated with pinned official metric formulas and a frozen controller. Three release-CSV segments were not materialized.

The Notion page was fetched again after mutation. The corrected aggregate metrics, all six paired-bootstrap intervals, finite-pair NaN handling, severe failure cases, claim boundary, artifact hashes, and `Completed / Passed` state were present. Kinetics release scope was not described as a fixed 1,000-video benchmark, and the auxiliary 1,189-ID split-file union was explicitly excluded from the release-CSV count.
