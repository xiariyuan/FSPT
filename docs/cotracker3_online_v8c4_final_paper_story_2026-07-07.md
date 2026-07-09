# V8-C4 Final Paper-Style Story: CVRRM for Online TAP Re-entry Recovery

Date: 2026-07-07

## 1. One-sentence claim

```text
CVRRM is a strict-causal, output-level, finite-horizon recovery mode for online TAP re-entry failures.
On TAP-Vid-DAVIS-first metric-compatible reproduction, it improves both standard TAP-Vid-like metrics and re-entry-focused metrics.
```

## 2. Method definition

```text
CVRRM = Causal Candidate-Visible Re-entry Recovery Mode

Input: native CoTracker3 online outputs and an external candidate provider.
Trigger: native low-score / invisible event rebuilt from native history.
Sanity filter: native-candidate distance <= 64 px at event frame.
Recovery mode: W=8 by default.
Frame-level confirmation: replace a frame only if the candidate is visible at that frame.
Output: candidate coordinate + visible flag for accepted frames; native output otherwise.
```

## 3. Protocol status

```text
Protocol label: TAP-Vid-DAVIS-first metric-compatible reproduction
Metric formula parity pass: True
Query-first pass: True
Local evaluator protocol-close pass: True
Max metric diff: 0.0
```

Safe claim: On TAP-Vid-DAVIS-first metric-compatible reproduction, CVRRM improves standard TAP-Vid-like metrics and re-entry-specific metrics.

Unsafe claims:
- Do not claim full original CoTracker3 multi-dataset Table-1 improvement yet.
- Do not claim single-model CoTracker3 improvement; this is a two-source recovery system.
- Do not treat AJ_RD_256 as a replacement for original TAP-Vid main metrics.

## 4. Standard TAP-Vid-like metrics on DAVIS

These metrics answer whether overall tracking quality is preserved or improved.

| Method | AJ | ΔAJ | OA | ΔOA | δavg | Δδavg | δ4px | Δδ4px | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CoTracker3 online native | 65.2366 | +0.0000 | 90.8186 | +0.0000 | 77.9458 | +0.0000 | 85.8670 | +0.0000 | baseline |
| TrackOn2 standalone | 67.0406 | +1.8041 | 92.0916 | +1.2730 | 79.8418 | +1.8960 | 87.8233 | +1.9562 | candidate baseline |
| V7-B2 visibility-only | 65.1841 | -0.0524 | 90.8517 | +0.0331 | 77.9458 | +0.0000 | 85.8670 | +0.0000 | weak baseline |
| CVRRM + TrackOn2 w8 | 65.3330 | +0.0964 | 91.7648 | +0.9462 | 78.3429 | +0.3971 | 86.3755 | +0.5085 | MAIN DEFAULT |
| CVRRM + TrackOn2 w16 | 65.3176 | +0.0810 | 91.7886 | +0.9701 | 78.3797 | +0.4339 | 86.4263 | +0.5593 | high-gain ablation |
| CVRRM + TAPNext++ online w8 | 64.5457 | -0.6909 | 91.3283 | +0.5097 |  | +0.1055 |  | +0.1809 | supporting diagnostic |
| OOF verifier ablation | 65.3549 | +0.1183 | 91.7593 | +0.9407 | 78.2904 | +0.3446 | 86.3060 | +0.4390 | diagnostic only |
| State writeback p=0.90 | 65.2767 | +0.0401 | 91.6764 | +0.8578 | 78.3561 | +0.4103 | 86.3468 | +0.4798 | future work only |

## 5. Re-entry recovery metrics on the same records

These metrics answer whether the method specifically fixes re-entry failures.

| Method | AJ_RD | ΔAJ_RD | AJ_RD_256 | ΔAJ_RD_256 | Repair16 | Damage16 | Robustness | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| CoTracker3 online native | 0.3534 | +0.0000 | 0.5333 | +0.0000 |  |  |  | baseline |
| TrackOn2 standalone | 0.3714 | +0.0180 | 0.5444 | +0.0111 |  |  |  | candidate baseline |
| V7-B2 visibility-only | 0.3554 | +0.0020 | 0.5380 | +0.0047 |  |  |  | weak baseline |
| CVRRM + TrackOn2 w8 | 0.3687 | +0.0153 | 0.5607 | +0.0274 | 0.1016 | 0.0238 | 16/4/5 pos/neg/zero | MAIN DEFAULT |
| CVRRM + TrackOn2 w16 | 0.3696 | +0.0162 | 0.5619 | +0.0286 | 0.0834 | 0.0171 | 15/5/5 pos/neg/zero | high-gain ablation |
| CVRRM + TAPNext++ online w8 | 0.3597 | +0.0063 | 0.5537 | +0.0204 | 0.1051 | 0.0456 | 16/6/3 pos/neg/zero | supporting diagnostic |
| CVRRM + old CoTracker online w8 |  | +0.0020 |  | +0.0065 |  |  | 12/2/11 pos/neg/zero | weak candidate evidence |
| CVRRM + old CoTracker offline w8 |  | +0.0027 |  | +0.0039 |  |  | 8/6/11 pos/neg/zero | weak candidate evidence |
| CVRRM + TAPNext++ offline w8 | 0.3134 | -0.0400 | 0.4718 | -0.0616 |  |  | 0/16/9 pos/neg/zero | negative evidence |
| OOF verifier ablation | 0.3688 | +0.0154 | 0.5608 | +0.0275 | 0.0898 | 0.0236 | 16/4/5 pos/neg/zero | diagnostic only |
| State writeback p=0.90 | 0.3684 | +0.0150 | 0.5618 | +0.0285 |  |  | 10/7/8 pos/neg/zero | future work only |

## 6. Candidate-source evidence

This table shows CVRRM is candidate-aware: strong online candidate providers help, weak/offline candidates do not.

| Row | Candidate | Protocol | ΔAJ_RD_256 | Status | Notes |
| --- | --- | --- | ---: | --- | --- |
| Native CoTracker3 online / none / strict native / +0.0000 / baseline / reference |
| V7-B2 visibility-only / none / strict output-level / +0.0047 / baseline / visibility-only gain is small |
| TrackOn2 standalone / TrackOn2 bridge / strict aligned / +0.0111 / candidate baseline / strong global candidate but less re-entry-focused than CVRRM |
| CVRRM + TrackOn2 w8 default / TrackOn2 bridge / strict aligned / +0.0274 / MAIN DEFAULT / recommended main method |
| CVRRM + TrackOn2 w16 high-gain / TrackOn2 bridge / strict aligned / +0.0286 / high-gain ablation / slightly higher AJ_RD_256, less robust than w8 |
| CVRRM + TAPNext++ online w8 / TAPNext++ online-style / metadata-bridge diagnostic / +0.0204 / supporting ablation / crosses +0.020 AJ_RD_256 but AJ is negative; diagnostic protocol |
| CVRRM + TAPNext++ online w16 / TAPNext++ online-style / metadata-bridge diagnostic / +0.0209 / supporting ablation / cross-candidate evidence; not default |
| CVRRM + old CoTracker online w8 / old CoTracker3 online bridge / strict aligned / +0.0065 / weak candidate evidence / small gain only |
| CVRRM + old CoTracker offline w8 / old CoTracker3 offline bridge / strict aligned / +0.0039 / weak candidate evidence / small gain only |
| CVRRM + TAPNext++ offline w8 / TAPNext++ offline / metadata-bridge diagnostic / -0.0616 / negative evidence / harmful; high false-visible/damage |
| OOF verifier ablation / TrackOn2 bridge / OOF learned diagnostic / +0.0275 / diagnostic only / not default: only +0.0001 AJ_RD_256 over CVRRM default |

## 7. Negative and neutral follow-ups

### Learned verifier

```text
OOF verifier AJ_RD_256 delta: +0.0275
Default AJ_RD_256 delta: +0.0274
Decision: diagnostic only, not default.
```

### State writeback

```text
State writeback p=0.90 AJ_RD_256 delta: +0.0285
Output-level high-gain w16 AJ_RD_256 delta: +0.0286
Decision: future work only; it does not beat output-level high-gain and lowers AJ/OA.
```

## 8. Main result summary

```text
Base AJ: 65.2366
CVRRM w8 AJ: 65.3330
AJ delta: +0.0964

Base OA: 90.8186
CVRRM w8 OA: 91.7648
OA delta: +0.9462

CVRRM w8 AJ_RD_256 delta: +0.0274
CVRRM w16 AJ_RD_256 delta: +0.0286
```

## 9. Limitations

```text
1. This is a two-source recovery system: CoTracker3 online native + external candidate provider + CVRRM.
2. Current strongest claim is DAVIS-first metric-compatible reproduction, not full original multi-dataset benchmark parity.
3. AJ_RD / AJ_RD_256 are re-entry-focused supplementary metrics, not replacements for AJ/OA/δavg.
4. TrackOn2 bridge is the default candidate provider; TAPNext++ online-style is supporting diagnostic evidence.
```

## 10. Recommended next steps

```text
First: use this V8-C4 package as the paper/story baseline.
Then choose one expansion path:
A. Benchmark expansion: Kinetics subset -> RGB-Stacking smoke -> full Kinetics/RGB-Stacking.
B. Candidate expansion: strictly aligned TAPNext++ / LocoTrack / TAPIR candidate caches.
Avoid further learned-verifier/state-writeback work unless a new full30 signal clearly exceeds output-level CVRRM.
```
