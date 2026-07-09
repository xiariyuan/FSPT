# TAPNext / TrackOn2 Baseline Parity Audit — 2026-06-29

## Decision

Current TAPNext / TrackOn2 artifacts should **not** be added directly to the main B2-W strided-original table as strong baselines. TrackOn2 has a strong validated first-query/input-resolution bridge, but the available strided-original TrackOn2/TAPNext caches are not comparable to the B2-W main protocol: their coordinate tracking quality collapses under the current strided-original export/evaluation setting, even though query anchors are sane.

Therefore, use these baselines carefully:

```text
TrackOn2 first-input bridge: valid baseline for first-query/input-resolution protocol, not directly comparable to B2-W strided-original AJ_RD tables.
TrackOn2/TAPNext strided-original caches: parity not established; do not use in main table.
TAPNext++: blocked, no checkpoint available locally.
```

## Existing official/parity evidence

TrackOn2 status file reports repo-native DAVIS parity under first-query/input-resolution protocol:

```text
repo_native AJ = 67.04
repo_native OA = 92.09
repo_native delta_avg = 79.84
reproduction_status = match
rescoring_note = Round 4: first+input bridge completed. Unified bridge AJ=67.04, delta_avg=79.84, OA=92.09 — PERFECT MATCH vs repo-native (0.00 diff). Long-occ AJ=45.27, delta_avg=68.11. True strided+original Attempt 0 main protocol still pending.
```

This means TrackOn2 itself is not weak; the issue is protocol/adapter comparability for the strided-original setup.

## Anchor sanity

| cache | records | queries | pred-query anchor mean @256 | pred-query anchor max @256 | pred visibility mean | GT visibility mean | conclusion |
|---|---:|---:|---:|---:|---:|---:|---|
| cotracker3_offline_main | 30 | 5882 | 1e-05 | 4.3e-05 | 0.740838 | 0.776625 | reference cache, clean anchor |
| trackon2_strided_original_existing | 30 | 5882 | 0.117206 | 0.797956 | 0.410425 | 0.776625 | anchor clean, but visibility low and protocol not parity-established |
| tapnext_strided_original_existing | 30 | 5882 | 0.270904 | 0.950536 | 0.500984 | 0.776625 | anchor clean, but visibility low and protocol not parity-established |
| trackon2_first_input_bridge | 30 | 650 | 0.114648 | 0.412378 | 0.695027 | 0.702352 | clean anchor and valid first-query bridge |

Anchor sanity indicates the current strided-original caches are not broken at the query anchor. The weakness is downstream tracking/visibility/protocol behavior.

## Visibility-oracle sensitivity

| cache | variant | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 | interpretation |
|---|---|---:|---:|---:|---:|---|
| trackon2_strided_original_existing | original | 0.5383 | 37.5303 | 58.3304 | 42.7415 | not main-table-ready in strided-original protocol |
| trackon2_strided_original_existing | gt_visibility | 0.644 | 27.7615 | 100.0 | 42.7415 | coordinate-only with oracle visibility; not a usable method |
| trackon2_strided_original_existing | all_visible | 0.4651 | 23.2414 | 77.2699 | 42.7415 | visibility stress test |
| trackon2_strided_original_existing | query_visible_fill | 0.5383 | 37.5303 | 58.3304 | 42.7415 | anchor visibility not the main issue |
| tapnext_strided_original_existing | original | 0.5199 | 34.5573 | 65.9795 | 43.495 | not main-table-ready in strided-original protocol |
| tapnext_strided_original_existing | gt_visibility | 0.6644 | 28.4085 | 100.0 | 43.495 | coordinate-only with oracle visibility; not a usable method |
| tapnext_strided_original_existing | all_visible | 0.4817 | 23.7578 | 77.2699 | 43.495 | visibility stress test |
| tapnext_strided_original_existing | query_visible_fill | 0.5199 | 34.5573 | 65.9795 | 43.495 | anchor visibility not the main issue |
| trackon2_first_input_bridge | original | 0.5444 | 67.0406 | 93.0615 | 79.8418 | valid first-query/input bridge; good standard AJ |
| trackon2_first_input_bridge | gt_visibility | 0.6436 | 71.4538 | 100.0 | 79.8418 | coordinate-only with oracle visibility; not a usable method |
| trackon2_first_input_bridge | all_visible | 0.4763 | 51.9331 | 69.7129 | 79.8418 | visibility stress test |
| trackon2_first_input_bridge | query_visible_fill | 0.5444 | 67.0406 | 93.0615 | 79.8418 | anchor visibility not the main issue |

Important observations:

```text
1. TrackOn2 first-input bridge is strong: AJ=67.0406, OA=93.0615, delta_avg=79.8418.
2. TrackOn2 strided-original has reasonable AJ_RD_256=0.5383 but very low AJ=37.5303 and delta_avg=42.7415.
3. TAPNext strided-original has AJ_RD_256=0.5199 but AJ=34.5573 and delta_avg=43.4950.
4. Oracle visibility can raise AJ_RD, but delta_avg remains low for strided-original caches, implying coordinate/protocol degradation beyond visibility threshold alone.
5. Query-visible-fill does not change metrics, so query-frame visibility is not the main problem.
```

## What can be claimed now

Safe:

```text
TrackOn2 was reproduced under its first-query/input-resolution evaluation protocol and is strong there.
Existing TrackOn2/TAPNext strided-original exports are not suitable for main comparison until protocol parity is established.
B2-W main comparisons should remain against CoTracker3 offline/online and global B1 under the same unified strided-original protocol.
```

Unsafe:

```text
Do not claim B2-W beats TrackOn2/TAPNext in general based on the weak strided-original cache numbers.
Do not put current TAPNext/TrackOn2 strided-original results into the main paper table as official baselines.
Do not compare TrackOn2 first-input AJ directly against B2-W strided-original AJ_RD tables without explaining protocol differences.
```

## Recommended strengthening options

### Option A — Paper-safe baseline statement

Use CoTracker3 offline/online and global B1 as same-protocol baselines. Mention TrackOn2/TAPNext parity audit in appendix: strong first-query TrackOn2 parity exists, but strided-original parity for these baselines is unresolved.

### Option B — Proper second-baseline integration

If we want a true stronger baseline comparison, implement a validated strided-original export for TrackOn2 or TAPNext:

```text
1. confirm model officially supports arbitrary strided query times
2. export under original-resolution query protocol
3. validate query anchor and official/repo parity if available
4. only then add to main table
```

### Option C — Separate first-query appendix

Report TrackOn2 first-input bridge separately as protocol reference, but do not mix it with B2-W main strided-original table.

## Artifacts

```text
scripts/audit_baseline_cache_parity.py
scripts/audit_baseline_visibility_oracle.py
outputs/paper_discovery_2026-06-27/baseline_parity_audit/summary.json
outputs/paper_discovery_2026-06-27/baseline_parity_audit/visibility_oracle_summary.json
```
