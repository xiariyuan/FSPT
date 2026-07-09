# Paper Upgrade / Strengthening Plan — 2026-06-29

## Decision

We should continue strengthening the paper, but not by repeatedly tuning B2-W16-P2 on fresh validation data. The correct upgrade path is:

```text
1. strengthen baseline alignment
2. strengthen qualitative figures
3. clarify relation to recent re-detection / verifier papers
4. keep B2-W16-P2 as the main method
5. keep B2-RV as exploratory unless a window-level verifier becomes stable
```

## Why not keep tuning P2?

RGB fresh20-49 is now the primary validation split. Tuning on it would contaminate the paper protocol. Any new method development must use only DAVIS + RGB dev0-9, and must later be evaluated on a new untouched split or reported as exploratory.

## External paper pressure points

Recent TAP papers raise the standard in three ways:

```text
TAPNext++: explicitly targets re-detection / AJ_RD and long sequences.
Track-On2: emphasizes online long-term tracking with memory across multiple benchmarks.
Verifier-guided pseudo-labeling: uses learned reliability verification over multiple pretrained trackers.
AllTracker: emphasizes high-resolution dense point tracking and extensive ablations.
```

Our paper should therefore avoid claiming new tracker SOTA and instead claim a focused inference-time re-entry reliability intervention.

## Current baseline alignment audit

Existing repository assets found:

```text
baselines/track_on/
caches/trackon2_strided_original.pt
checkpoints/tapnext/bootstapnext_ckpt.npz
scripts/export_tapnext_strided_original_cache.py
scripts/export_tapnext_rgb_stacking_cache.py
outputs/paper_discovery_2026-06-27/teacher_expansion/tapnext_bootstapnext_strided_original.pt
```

Initial unified-protocol smoke metrics:

| baseline cache | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 | interpretation |
|---|---:|---:|---:|---:|---|
| TrackOn2 existing DAVIS cache | 0.5383 | 37.5303 | 58.3304 | 42.7415 | too weak under current bridge; not main-table ready |
| TAPNext existing DAVIS cache | 0.5199 | 34.5573 | 65.9795 | 43.495 | too weak under current bridge; needs protocol/parity check before use |

These numbers are far below CoTracker3 offline and B2-W. Therefore they should not be used as strong baseline evidence without additional adapter/protocol validation.

## Upgrade experiments that are worth doing

### U1 — Baseline parity audit for TAPNext / TrackOn2

Goal: determine whether poor metrics are due to model weakness, adapter mismatch, query convention, visibility convention, checkpoint mismatch, or evaluation protocol mismatch.

Tasks:

```text
1. validate coordinate convention on a few DAVIS frames
2. compare official/repo evaluation vs unified evaluation
3. inspect query ordering and point normalization
4. check visibility logits/threshold convention
5. if parity is fixed, rerun TAPNext/TrackOn2 on DAVIS or RGB dev subset
```

Decision gate:

```text
If baseline remains weak after parity fixes, mention as omitted/unsupported rather than main-table baseline.
If baseline becomes strong, add it to DAVIS main table and possibly test B2-W as plug-in over it.
```

### U2 — Qualitative figure pack

Goal: upgrade presentation quality. The paper currently has strong tables but needs visual evidence.

Required figures:

```text
Figure 1: method diagram
Figure 2: DAVIS re-entry recovery success
Figure 3: B2 avoids global/online damage
Figure 4: false-trigger / weak override limitation
```

Priority: very high. This is the best immediate paper-strengthening step.

### U3 — RGB diagnostic 10-49 aggregate appendix

Main RGB validation should remain fresh20-49. But the consumed diagnostic heldout10 can be reported in appendix as diagnostic evidence, clearly labeled as consumed/diagnostic.

### U4 — Window-level verifier prototype

Track-level B2-RV is too coarse. A future upgrade should be window-level:

```text
accept/reject each override window independently
or shrink window length when risk is high
or early-return to base after K frames
```

But this should remain development-only until stable. Do not touch RGB fresh20-49 for tuning.

### U5 — Full paper draft

The skeleton exists, but the next upgrade is a coherent first draft. Tables alone are not enough for CCF-B/Q2; the story must be clean.

## Recommended execution order

```text
P0. Create qualitative figure manifest and select representative images.
P1. Create baseline parity audit doc for TAPNext / TrackOn2.
P2. Write full paper draft using existing tables.
P3. If time remains, implement window-level B2-RV on dev only.
P4. Only after P3 is stable, decide whether a new untouched validation split is needed.
```

## What this means for target level

For CCF-B / CAS-Q2:

```text
current data + qualitative figures + clean draft may be enough for a serious submission
baseline parity audit would strengthen credibility
```

For CCF-A:

```text
need at least one of:
- strong recent baseline parity and comparison
- plug-in generality over a second strong base/override pair
- stable learned/window-level verifier
- additional benchmark beyond DAVIS + RGB
```

## Current conclusion

Do not abandon paper strengthening. The next best step is not more P2 tuning; it is qualitative figure selection plus baseline parity audit. The paper is already strong enough to consolidate, but CCF-A-level strengthening requires stronger baseline alignment or a stable verifier beyond the current track-level B2-RV.


## Correction after baseline parity audit

The initial upgrade-plan smoke table used non-256 AJ_RD values for TrackOn2/TAPNext. The corrected unified AJ_RD_256 values are:

```text
TrackOn2 strided-original existing: AJ_RD_256=0.5383, AJ_256=37.5303, delta_avg_256=42.7415
TAPNext strided-original existing:  AJ_RD_256=0.5199, AJ_256=34.5573, delta_avg_256=43.4950
TrackOn2 first-input bridge:        AJ_RD_256=0.5444, AJ_256=67.0406, delta_avg_256=79.8418
```

Interpretation remains: current TrackOn2/TAPNext strided-original caches are not main-table-ready because standard AJ and coordinate delta_avg collapse under this protocol, while TrackOn2 first-input bridge is strong but not directly comparable to the main B2-W strided-original protocol.
