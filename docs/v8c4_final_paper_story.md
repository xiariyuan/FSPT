# V8-C4 Final Paper-Style Story

## 1. Working title

**CVRRM: Causal Candidate-Visible Re-entry Recovery for Online TAP**

## 2. Core claim

```text
On a TAP-Vid-DAVIS-first metric-compatible reproduction, CVRRM improves standard TAP-Vid-like metrics and re-entry-focused metrics.
It is a strict-causal output-level recovery mode, not a single-model CoTracker3 replacement.
```

## 3. Method in one paragraph

CVRRM detects low-confidence or invisible native CoTracker3 online events, opens a finite recovery window, and only replaces a frame when a strong candidate provider is currently visible and passes a native-candidate distance sanity check. The default uses TrackOn2 bridge as the candidate provider, `dist<=64`, and `W=8`.

## 4. Standard TAP-Vid-like metrics

Protocol label: `TAP-Vid-DAVIS-first metric-compatible reproduction`

| Method | Protocol | AJ Δ | OA Δ | δavg Δ | δ4px Δ | Status |
|---|---|---:|---:|---:|---:|---|
| CoTracker3 online native | current DAVIS true-streaming reproduction | +0.0000 | +0.0000 | +0.0000 | +0.0000 | baseline |
| TrackOn2 standalone | strict aligned candidate cache | +1.8041 | +1.2730 | +1.8960 | +1.9562 | candidate baseline |
| V7-B2 visibility-only | strict output-level ablation | -0.0524 | +0.0331 | +0.0000 | +0.0000 | weak baseline |
| CVRRM + TrackOn2 w8 | strict output-level two-source | +0.0964 | +0.9462 | +0.3971 | +0.5085 | MAIN DEFAULT |
| CVRRM + TrackOn2 w16 | strict output-level two-source | +0.0810 | +0.9701 | +0.4339 | +0.5593 | high-gain ablation |
| CVRRM + TAPNext++ online w8 | metadata-bridge diagnostic | -0.6909 | +0.5097 | +0.1055 | +0.1809 | supporting diagnostic |
| OOF verifier ablation | OOF diagnostic | +0.1183 | +0.9407 | +0.3446 | +0.4390 | diagnostic only |
| State writeback p=0.90 | state hook follow-up | +0.0401 | +0.8578 | +0.4103 | +0.4798 | future work only |

## 5. Re-entry recovery metrics

AJ_RD and AJ_RD_256 are supplementary failure-mode metrics, computed for all baselines with the same definition.

| Method | Protocol | AJ_RD Δ | AJ_RD_256 Δ | Repair16 | Damage16 | Robustness | Status |
|---|---|---:|---:|---:|---:|---|---|
| CoTracker3 online native | current DAVIS true-streaming reproduction | +0.0000 | +0.0000 |  |  |  | baseline |
| TrackOn2 standalone | strict aligned candidate cache | +0.0180 | +0.0111 |  |  |  | candidate baseline |
| V7-B2 visibility-only | strict output-level ablation | +0.0020 | +0.0047 |  |  |  | weak baseline |
| CVRRM + TrackOn2 w8 | strict output-level two-source | +0.0153 | +0.0274 | 0.1016 | 0.0238 | 16/4/5 | MAIN DEFAULT |
| CVRRM + TrackOn2 w16 | strict output-level two-source | +0.0162 | +0.0286 | 0.0834 | 0.0171 | 15/5/5 | high-gain ablation |
| CVRRM + TAPNext++ online w8 | metadata-bridge diagnostic | +0.0063 | +0.0204 | 0.1051 | 0.0456 | 16/6/3 | supporting diagnostic |
| CVRRM + old CoTracker online w8 | strict aligned candidate | +0.0020 | +0.0065 |  |  | 12/2/11 | weak candidate evidence |
| CVRRM + old CoTracker offline w8 | strict aligned candidate | +0.0027 | +0.0039 |  |  | 8/6/11 | weak candidate evidence |
| CVRRM + TAPNext++ offline w8 | metadata-bridge diagnostic | -0.0400 | -0.0616 |  |  | 0/16/9 | negative evidence |
| OOF verifier ablation | OOF diagnostic | +0.0154 | +0.0275 | 0.0898 | 0.0236 | 16/4/5 | diagnostic only |
| State writeback p=0.90 | state hook follow-up | +0.0150 | +0.0285 |  |  | 10/7/8 | future work only |

## 6. Candidate-source evidence

| Candidate / Method | Protocol | AJ_RD_256 Δ | Status | Notes |
|---|---|---:|---|---|
| Native CoTracker3 online | strict native | +0.0000 | baseline | reference |
| V7-B2 visibility-only | strict output-level | +0.0047 | baseline | visibility-only gain is small |
| TrackOn2 standalone | strict aligned | +0.0111 | candidate baseline | strong global candidate but less re-entry-focused than CVRRM |
| CVRRM + TrackOn2 w8 default | strict aligned | +0.0274 | MAIN DEFAULT | recommended main method |
| CVRRM + TrackOn2 w16 high-gain | strict aligned | +0.0286 | high-gain ablation | slightly higher AJ_RD_256, less robust than w8 |
| CVRRM + TAPNext++ online w8 | metadata-bridge diagnostic | +0.0204 | supporting ablation | crosses +0.020 AJ_RD_256 but AJ is negative; diagnostic protocol |
| CVRRM + TAPNext++ online w16 | metadata-bridge diagnostic | +0.0209 | supporting ablation | cross-candidate evidence; not default |
| CVRRM + old CoTracker online w8 | strict aligned | +0.0065 | weak candidate evidence | small gain only |
| CVRRM + old CoTracker offline w8 | strict aligned | +0.0039 | weak candidate evidence | small gain only |
| CVRRM + TAPNext++ offline w8 | metadata-bridge diagnostic | -0.0616 | negative evidence | harmful; high false-visible/damage |
| OOF verifier ablation | OOF learned diagnostic | +0.0275 | diagnostic only | not default: only +0.0001 AJ_RD_256 over CVRRM default |

## 7. Negative / neutral results

```text
Learned verifier: not promoted. Best OOF verifier improves AJ_RD_256 by only +0.0001 over default.
State writeback: not promoted. Full30 p=0.90 nearly matches output-level w16 but lowers AJ/OA and is mixed per video.
Weak candidates: old CoTracker candidates produce only small gains; offline TAPNext++ is harmful.
```

## 8. Claim boundaries

Safe claims:

```text
TAP-Vid-DAVIS-first metric-compatible reproduction.
Two-source output-level re-entry recovery system.
AJ_RD_256 is a supplementary re-entry-focused metric.
```

Unsafe claims:

```text
Full original CoTracker3 Table-1 improvement across all datasets.
Single-model CoTracker3 improvement.
Universal candidate-agnostic recovery.
```

## 9. Recommended next step

```text
Write final method/results section from this package.
Then optionally expand to Kinetics/RGB-Stacking, or search stronger strictly aligned candidate providers.
```
