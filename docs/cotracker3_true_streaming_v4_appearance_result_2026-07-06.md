# CoTracker3 True-Streaming V4 Appearance-Verifier Result — 2026-07-06

## Purpose

Test whether an appearance verifier can make CoTracker3 true-streaming state writeback more reliable than V3 visibility/geometric gates alone.

## V4 candidate audit

Candidate audit script:

```text
scripts/audit_cotracker3_online_v4_appearance_candidates.py
```

Outputs:

```text
outputs/paper_discovery_2026-07-05/cotracker3_v4_appearance_audit/davis_true_streaming_v4_appearance_audit.json
outputs/paper_discovery_2026-07-05/cotracker3_v4_appearance_audit/davis_true_streaming_v4_appearance_candidates.npz
```

Audit facts:

```text
raw candidates: 672
numeric-pass candidates: 31
numeric-pass GT-visible precision: 35.48%
```

DINO feature used:

```text
last_strict_candidate_cosine
```

Best useful threshold found in the audit:

```text
last_strict_candidate_cosine >= 0.7204042673110962
```

This improves numeric-pass candidate precision to about:

```text
5 / 7 = 71.4%
```

Interpretation:

```text
DINO appearance similarity can improve candidate precision in audit, but it also greatly reduces candidate count.
```

## V4 integrated full30 run

Run directory:

```text
outputs/paper_discovery_2026-07-05/cotracker3_true_streaming_v4_appearance_subset_eval/full30_dino_laststrict0720/
```

Posthoc summary:

```text
outputs/paper_discovery_2026-07-05/cotracker3_true_streaming_v4_appearance_subset_eval/full30_dino_laststrict0720/cotracker3_true_streaming_v4_appearance_full30_summary.json
```

Appearance gate:

```text
feature = last_strict_candidate_cosine
threshold = 0.7204042673110962
```

Full30 metrics:

| Variant | AJ | OA | delta_avg | delta_4px | AJ_RD | AJ_RD_256 |
|---|---:|---:|---:|---:|---:|---:|
| native | 65.2366 | 90.8186 | 77.9458 | 85.8670 | 0.3534 | 0.5333 |
| output-only | 65.2159 | 90.7955 | 77.9458 | 85.8670 | 0.3535 | 0.5335 |
| V4 state-writeback | 65.2366 | 90.8186 | 77.9458 | 85.8670 | 0.3534 | 0.5333 |
| V4 state+output | 65.2159 | 90.7955 | 77.9458 | 85.8670 | 0.3535 | 0.5335 |

Deltas vs native:

| Variant | dAJ | dOA | dDelta | dAJ_RD | dAJ_RD_256 |
|---|---:|---:|---:|---:|---:|
| output-only | -0.0207 | -0.0231 | +0.0000 | +0.0001 | +0.0002 |
| V4 state-writeback | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| V4 state+output | -0.0207 | -0.0231 | +0.0000 | +0.0001 | +0.0002 |

## Opened-frame audit

```text
output-only opened 31 frames: 11 GT-visible, 20 GT-occluded, precision 35.48%.
V4 state-writeback opened 0 frames.
V4 state+output equals output-only behavior for final visibility.
```

## Interpretation

The appearance verifier is useful diagnostically:

```text
It can raise candidate precision from 35.5% to about 71.4% in the candidate audit.
```

But the integrated state-writeback setting becomes too conservative:

```text
The chosen DINO threshold filters out all state-writeback actions in full30.
```

Therefore V4 does not produce an online state-level improvement.

## Decision

Do not use V4 as a main result.

Safe paper statement:

```text
A DINO appearance verifier improves candidate precision in an audit, but under the conservative overlap-only state-writeback protocol it filters out all writebacks and does not yield a measurable online gain.
```

Unsafe statement:

```text
Appearance-verified online state writeback improves CoTracker3.
```

## Next design implication

The limiting factor is not merely verifier precision. The next version needs better recall while preserving precision:

```text
1. More candidate proposals, not only suffix-before-native-visible.
2. A less brittle appearance feature or learned verifier.
3. Possibly separate thresholds for state-writeback and output-only.
4. If staying with hand-written rules, state-level gains will likely remain too small.
```

Current conclusion:

```text
CoTracker3 online state-writeback is structurally feasible and can be made safe, but current hand-written visibility/geometric/appearance gates do not yield meaningful online re-entry improvement.
```
