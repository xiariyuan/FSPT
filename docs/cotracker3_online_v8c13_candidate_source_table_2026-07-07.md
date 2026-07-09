# CoTracker3 Online V8-C1.3 Candidate-Source Final Table

Date: 2026-07-07

## 1. Purpose

Consolidate strict CVRRM results, cross-candidate results, TAPNext++ metadata-bridge diagnostics, and learned-verifier ablation into one candidate-source table.

## 2. Candidate-source table

| Row | Candidate | Protocol | AJ Δ | OA Δ | AJ_RD Δ | AJ_RD_256 Δ | Robustness | Status | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| Native CoTracker3 online | none | strict native | +0.0000 | +0.0000 | +0.0000 | +0.0000 |  | baseline | reference |
| V7-B2 visibility-only | none | strict output-level | -0.0524 | +0.0331 | +0.0020 | +0.0047 | 2/5 folds | baseline | visibility-only gain is small |
| TrackOn2 standalone | TrackOn2 bridge | strict aligned | +1.8041 | +1.2730 | +0.0180 | +0.0111 |  | candidate baseline | strong global candidate but less re-entry-focused than CVRRM |
| CVRRM + TrackOn2 w8 default | TrackOn2 bridge | strict aligned | +0.0964 | +0.9462 | +0.0153 | +0.0274 | 16/4/5 pos/neg/zero | MAIN DEFAULT | recommended main method |
| CVRRM + TrackOn2 w16 high-gain | TrackOn2 bridge | strict aligned | +0.0810 | +0.9701 | +0.0162 | +0.0286 | 15/5/5 pos/neg/zero | high-gain ablation | slightly higher AJ_RD_256, less robust than w8 |
| CVRRM + TAPNext++ online w8 | TAPNext++ online-style | metadata-bridge diagnostic | -0.6909 | +0.5097 | +0.0063 | +0.0204 | 16/6/3 pos/neg/zero | supporting ablation | crosses +0.020 AJ_RD_256 but AJ is negative; diagnostic protocol |
| CVRRM + TAPNext++ online w16 | TAPNext++ online-style | metadata-bridge diagnostic | -0.9359 | +0.4455 | +0.0045 | +0.0209 | 16/6/3 pos/neg/zero | supporting ablation | cross-candidate evidence; not default |
| CVRRM + old CoTracker online w8 | old CoTracker3 online bridge | strict aligned | -0.3241 | +0.3956 | +0.0020 | +0.0065 | 12/2/11 pos/neg/zero | weak candidate evidence | small gain only |
| CVRRM + old CoTracker offline w8 | old CoTracker3 offline bridge | strict aligned | -0.0773 | +0.4463 | +0.0027 | +0.0039 | 8/6/11 pos/neg/zero | weak candidate evidence | small gain only |
| CVRRM + TAPNext++ offline w8 | TAPNext++ offline | metadata-bridge diagnostic | -5.3874 | -3.5318 | -0.0400 | -0.0616 | 0/16/9 pos/neg/zero | negative evidence | harmful; high false-visible/damage |
| OOF verifier ablation | TrackOn2 bridge | OOF learned diagnostic | +0.1183 | +0.9407 | +0.0154 | +0.0275 | 16/4/5 pos/neg/zero | diagnostic only | not default: only +0.0001 AJ_RD_256 over CVRRM default |

## 3. Final interpretation

```text
CVRRM is not arbitrary candidate fusion. It needs a strong online re-entry candidate provider.
TrackOn2 bridge is the default provider.
TAPNext++ online-style metadata bridge is supporting cross-candidate evidence.
Old CoTracker candidates are too weak. Offline TAPNext++ is harmful.
OOF verifier remains diagnostic only.
```

## 4. Recommended method statement

```text
Main method: CVRRM + TrackOn2 bridge, dist<=64, W=8.
High-gain ablation: CVRRM + TrackOn2 bridge, dist<=64, W=16.
Cross-candidate diagnostic: CVRRM + TAPNext++ online-style metadata bridge.
```

## 5. Next step

```text
Option A: freeze output-level method and write final paper-style method section/results.
Option B: start V8-C2 state-level repair, but only with TrackOn2 bridge default first.
Recommended: do a short state-level repair feasibility precheck before committing.
```
