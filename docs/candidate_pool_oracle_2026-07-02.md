# Candidate-Pool Oracle — 2026-07-02

## Goal

Test whether the current local-override candidate pool has enough headroom beyond B2-W16-P2 / ReEntry-Guard.

Protocol:

```text
For each eligible re-entry query, choose the candidate with the highest per-query AJ_RD_256.
Non-re-entry queries remain offline.
```

This is an oracle upper bound, not a deployable method.

## Candidate pool

```text
offline
online_global
b2_fullpost_p1
b2_w8_p2
b2_w16_p1
b2_w16_p2
b2_w32_p2
guard_rf_thr0.40
```

Setting:

```text
RGB fresh20-49 natural
30 videos
37,221 total queries
6,461 eligible re-entry queries
```

## Main results

| Method | AJ_RD_256 | AJ_256 | OA_256 |
|---|---:|---:|---:|
| offline | 0.3816 | 79.5944 | 91.4636 |
| online_global | 0.4121 | 44.6933 | 55.7585 |
| b2_fullpost_p1 | 0.4219 | 78.9590 | 92.7706 |
| b2_w8_p2 | 0.4462 | 79.0753 | 92.8853 |
| b2_w16_p1 | 0.4456 | 79.0634 | 92.8891 |
| b2_w16_p2 | 0.4454 | 79.0664 | 92.8804 |
| b2_w32_p2 | 0.4421 | 79.0451 | 92.8736 |
| ReEntry-Guard RF thr0.40 | 0.4499 | 79.1749 | 92.0864 |
| oracle_b2 | 0.4597 | 79.5633 | 92.0193 |
| oracle_online | 0.4677 | 78.2940 | 90.3600 |
| candidate-pool oracle | 0.4719 | 78.6150 | 90.7152 |

## Gains

Candidate-pool oracle vs offline:

```text
AJ_RD_256 +0.0903
AJ_256    -0.9794
OA_256    -0.7484
```

Candidate-pool oracle vs B2-W16-P2:

```text
AJ_RD_256 +0.0265
AJ_256    -0.4514
OA_256    -2.1652
```

Candidate-pool oracle vs ReEntry-Guard RF thr0.40:

```text
AJ_RD_256 +0.0220
AJ_256    -0.5599
OA_256    -1.3712
```

Candidate-pool oracle vs oracle_b2:

```text
AJ_RD_256 +0.0122
AJ_256    -0.9483
OA_256    -1.3041
```

## Candidate selection distribution

Among 6,461 eligible re-entry queries:

| Candidate | Count | Rate |
|---|---:|---:|
| offline | 2847 | 44.0644% |
| online_global | 2375 | 36.7590% |
| b2_w8_p2 | 656 | 10.1532% |
| b2_w16_p1 | 227 | 3.5134% |
| b2_w32_p2 | 180 | 2.7859% |
| b2_fullpost_p1 | 101 | 1.5632% |
| b2_w16_p2 | 75 | 1.1608% |

## Interpretation

This is a good result for the stronger-innovation direction.

The candidate-pool oracle reaches:

```text
AJ_RD_256 = 0.4719
```

This is above the earlier decision threshold of 0.47, meaning the existing candidate pool has meaningful exploitable headroom.

However, this headroom is not free:

```text
AJ drops by about 0.98 vs offline.
OA drops by about 0.75 vs offline.
```

Therefore, the next deployable method should not blindly imitate the oracle. It should optimize for:

```text
candidate selection under an AJ budget
```

or:

```text
Constrained Candidate-Pool Selector, e.g. maximize AJ_RD while keeping AJ loss <= 1 point.
```

## Decision

Proceed to candidate-aware ReEntry-Guard / multi-candidate selector.

Target:

```text
Current ReEntry-Guard: AJ_RD_256 0.4499, AJ 79.1749
Candidate-pool oracle: AJ_RD_256 0.4719, AJ 78.6150

Reasonable selector target:
AJ_RD_256 >= 0.455 -- 0.465
AJ_256 >= 79.0 if possible, or AJ loss <= 1 point.
```

This can substantially strengthen the paper by shifting the contribution from binary local override to candidate-aware re-entry reliability routing.
