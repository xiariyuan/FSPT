# CoTracker3 Online V8-C3.0 DAVIS Dual-Metric Protocol Audit

Date: 2026-07-07

## 1. Why this audit exists

This audit avoids two mistakes: rejecting re-entry metrics just because they are not the original main-table metrics, and overclaiming AJ_RD_256 as an original-paper benchmark improvement.

## 2. Standard TAP-Vid-like table on current DAVIS reproduction

| Method | Protocol tier | AJ | AJ Δ | OA | OA Δ | δavg | δavg Δ | δ4px | δ4px Δ | Status | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| CoTracker3 online native / current DAVIS true-streaming reproduction / 65.2366 / +0.0000 / 90.8186 / +0.0000 / 77.9458 / +0.0000 / 85.8670 / +0.0000 / baseline / not claimed as original-paper table protocol yet |
| TrackOn2 standalone / strict aligned candidate cache / 67.0406 / +1.8041 / 92.0916 / +1.2730 / 79.8418 / +1.8960 / 87.8233 / +1.9562 / candidate baseline / two-source comparison baseline |
| V7-B2 visibility-only / strict output-level ablation / 65.1841 / -0.0524 / 90.8517 / +0.0331 / 77.9458 / +0.0000 / 85.8670 / +0.0000 / weak baseline / visibility-only helps little |
| CVRRM + TrackOn2 w8 / strict output-level two-source / 65.3330 / +0.0964 / 91.7648 / +0.9462 / 78.3429 / +0.3971 / 86.3755 / +0.5085 / MAIN DEFAULT / best stability/quality tradeoff |
| CVRRM + TrackOn2 w16 / strict output-level two-source / 65.3176 / +0.0810 / 91.7886 / +0.9701 / 78.3797 / +0.4339 / 86.4263 / +0.5593 / high-gain ablation / slightly higher re-entry, less robust |
| CVRRM + TAPNext++ online w8 / metadata-bridge diagnostic / 64.5457 / -0.6909 / 91.3283 / +0.5097 /  / +0.1055 /  / +0.1809 / supporting diagnostic / not strict raw-cache alignment |
| OOF verifier ablation / OOF diagnostic / 65.3549 / +0.1183 / 91.7593 / +0.9407 / 78.2904 / +0.3446 / 86.3060 / +0.4390 / diagnostic only / not default |
| State writeback p=0.90 / state hook follow-up / 65.2767 / +0.0401 / 91.6764 / +0.8578 / 78.3561 / +0.4103 / 86.3468 / +0.4798 / future work only / does not beat output-level w16 and has lower AJ/OA |

## 3. Re-entry recovery table on the same DAVIS records

| Method | Protocol tier | AJ_RD | AJ_RD Δ | AJ_RD_256 | AJ_RD_256 Δ | Repair16 | Damage16 | Robustness | Status | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| CoTracker3 online native / current DAVIS true-streaming reproduction / 0.3534 / +0.0000 / 0.5333 / +0.0000 /  /  /  / baseline / reference |
| TrackOn2 standalone / strict aligned candidate cache / 0.3714 / +0.0180 / 0.5444 / +0.0111 /  /  /  / candidate baseline / standalone improves standard AJ strongly but re-entry less than CVRRM |
| V7-B2 visibility-only / strict output-level ablation / 0.3554 / +0.0020 / 0.5380 / +0.0047 /  /  /  / weak baseline / visibility-only is not enough |
| CVRRM + TrackOn2 w8 / strict output-level two-source / 0.3687 / +0.0153 / 0.5607 / +0.0274 / 0.1016 / 0.0238 / 16/4/5 pos/neg/zero / MAIN DEFAULT / main re-entry recovery result |
| CVRRM + TrackOn2 w16 / strict output-level two-source / 0.3696 / +0.0162 / 0.5619 / +0.0286 / 0.0834 / 0.0171 / 15/5/5 pos/neg/zero / high-gain ablation / highest output-level AJ_RD_256 |
| CVRRM + TAPNext++ online w8 / metadata-bridge diagnostic / 0.3597 / +0.0063 / 0.5537 / +0.0204 / 0.1051 / 0.0456 / 16/6/3 pos/neg/zero / supporting diagnostic / cross-candidate evidence but not strict raw-cache alignment |
| CVRRM + old CoTracker online w8 / strict aligned candidate /  / +0.0020 /  / +0.0065 /  /  / 12/2/11 pos/neg/zero / weak candidate evidence / small gain only |
| CVRRM + old CoTracker offline w8 / strict aligned candidate /  / +0.0027 /  / +0.0039 /  /  / 8/6/11 pos/neg/zero / weak candidate evidence / small gain only |
| CVRRM + TAPNext++ offline w8 / metadata-bridge diagnostic / 0.3134 / -0.0400 / 0.4718 / -0.0616 /  /  / 0/16/9 pos/neg/zero / negative evidence / harmful candidate |
| OOF verifier ablation / OOF diagnostic / 0.3688 / +0.0154 / 0.5608 / +0.0275 / 0.0898 / 0.0236 / 16/4/5 pos/neg/zero / diagnostic only / only +0.0001 over default |
| State writeback p=0.90 / state hook follow-up / 0.3684 / +0.0150 / 0.5618 / +0.0285 /  /  / 10/7/8 pos/neg/zero / future work only / does not exceed output-level w16 |

## 4. Protocol position

```text
AJ_RD / AJ_RD_256 are valid as re-entry-focused failure-mode metrics because all baselines are evaluated with the same definition.
They are not replacements for the original CoTracker3 main-table AJ / δavg / OA.
Current DAVIS cache is a true-streaming first-input reproduction, not yet audited as the original paper one-query-at-a-time + support-points protocol.
CVRRM + TrackOn2 is a two-source recovery system, not a single-model CoTracker3 result.
```

## 5. Decision

```text
Use two tables:
1. Standard TAP-Vid-like table for AJ / OA / δavg / δ4px.
2. Re-entry recovery table for AJ_RD / AJ_RD_256 / repair-damage / robustness.
Main method remains CVRRM + TrackOn2 w8 output-level.
Do not claim original CoTracker3 Table-1 improvement until official-protocol comparability is separately audited.
```

## 6. Next step

```text
If the goal is a paper claim against original CoTracker3 tables: run official-protocol comparability gap audit.
If the goal is a re-entry paper: proceed to final writing with dual tables and clear protocol caveats.
Recommended: do a focused official-protocol gap audit on DAVIS before expanding to Kinetics/RGB-Stacking.
```
