# CoTracker3 Online V5-B Hard-Event Candidate-Pool Oracle Result — 2026-07-06

## Purpose

After V3/V4 showed that visibility/state-writeback has very limited headroom, V5-B tests whether active re-detection candidate pools can contain better points when CoTracker3 true-streaming native actually fails.

This is an oracle diagnostic, not a deployable method.

## Script

```text
scripts/eval_cotracker3_online_v5b_hard_candidate_pool_oracle.py
```

Report:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v5b_hard_candidate_pool_oracle/v5b_hard_candidate_pool_oracle_report.json
```

Selective-oracle posthoc report:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v5b_hard_candidate_pool_oracle/selective_oracles/selective_oracle_summary.json
```

## Protocol

Subset:

```text
first 10 DAVIS videos
40 hard re-entry events
```

Hard event definition:

```text
GT re-entry frame
GT visible
native invisible/low-score OR native error >= 8 px
```

Candidate pools:

```text
local64_s8  : local grid around native coordinate, radius 64 px, stride 8 px
local96_s8  : local grid around native coordinate, radius 96 px, stride 8 px
global_s16  : whole-frame coarse grid, stride 16 px
```

Feature:

```text
DINOv3 patch CLS cosine against last reliable visible support patch
```

## Candidate recall audit

Native hard-event error:

```text
median 9.28 px
mean   25.66 px
r4     32.5%
r8     45.0%
r16    60.0%
```

Candidate-pool oracle recall:

| Pool | top-k | median err | r4 | r8 | r16 |
|---|---:|---:|---:|---:|---:|
| local64_s8 | top10 | 7.85 | 25.0% | 55.0% | 77.5% |
| local64_s8 | top20 | 5.07 | 45.0% | 70.0% | 77.5% |
| local96_s8 | top10 | 7.92 | 25.0% | 52.5% | 82.5% |
| local96_s8 | top20 | 4.74 | 45.0% | 72.5% | 85.0% |
| global_s16 | top10 | 7.92 | 10.0% | 52.5% | 82.5% |
| global_s16 | top20 | 7.00 | 12.5% | 65.0% | 100.0% |

Interpretation:

```text
Candidate pools do improve hard-event recall over native, especially local96_s8 top20 at r8/r16.
```

## Forced replacement oracle

Forced replacement means the oracle picks the best candidate within top-k, but still replaces native even when native is better. This can underestimate the realistic upper bound of a verifier that can reject candidates.

Native on first 10 videos:

```text
AJ      71.4896
OA      92.1433
delta   83.0547
d4px    90.5001
AJ_RD   0.4168
AJ_RD_256 0.5975
```

Best forced replacement rows:

| Variant | dAJ | dOA | dDelta | d4px | dAJ_RD | dAJ_RD_256 |
|---|---:|---:|---:|---:|---:|---:|
| local64_s8_top20 | +0.0301 | +0.2038 | +0.0352 | +0.0392 | +0.0030 | +0.0071 |
| local96_s8_top20 | +0.0473 | +0.2038 | +0.0454 | +0.0490 | +0.0035 | +0.0082 |
| global_s16_top20 | +0.0242 | +0.2038 | +0.0195 | -0.0569 | +0.0016 | +0.0104 |

## Selective oracle posthoc

Selective oracle keeps native unless the top-k candidate is closer to GT. It also opens GT-visible hard-event visibility for upper-bound accounting.

Best selective rows:

| Variant | accepted coord replacements | dAJ | dOA | dDelta | d4px | dAJ_RD | dAJ_RD_256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| local64_s8_top20 selective | 28/40 | +0.0662 | +0.2038 | +0.0635 | +0.0873 | +0.0045 | +0.0091 |
| local96_s8_top20 selective | 28/40 | +0.0856 | +0.2038 | +0.0753 | +0.0971 | +0.0050 | +0.0106 |
| global_s16_top20 selective | 25/40 | +0.0753 | +0.2038 | +0.0653 | +0.0291 | +0.0047 | +0.0149 |

## Main finding

V5-B changes the conclusion from V4:

```text
There is real hard-event candidate-pool headroom, but it is still moderate.
```

Compared to V4 visibility/writeback:

```text
V4 state-writeback: no measurable gain.
V5-B hard-event candidate pool oracle: AJ_RD +0.0045 to +0.0050, AJ_RD_256 +0.0091 to +0.0149 on first10 selective oracle.
```

This is not strong enough yet for a deployable method result, but it is enough to justify one more diagnostic step.

## Risks / caveats

1. This is oracle-selected. It uses GT to decide whether a candidate is better.
2. It is first10 only, not full30.
3. The event set is hard-event biased, not a full online trigger policy.
4. DINO top-k ranking itself is not enough; the useful gain comes from selective oracle choosing when to accept.
5. The verifier problem is non-trivial: it must reject harmful top-k candidates and accept only better ones.

## Decision

Do not train a complex verifier yet.

Proceed to a lightweight verifier feasibility audit:

```text
Build a candidate-level dataset from V5-B first10/full30 candidate pools.
Label: candidate better than native by >= 2 px and/or candidate error <= 8 px.
Features: DINO score, rank, distance-to-native, native score, native visibility, support age, candidate pool type, local consistency proxies.
Evaluate leave-video-out / first10 split.
```

Continue only if a simple verifier can recover a meaningful fraction of the selective oracle:

```text
candidate acceptance precision >= 70%
accepted candidate count >= 15 on first10 or proportional full30 count
AJ_RD gain approaches at least half of selective oracle headroom
```

If the verifier cannot recover this, stop CoTracker3 online V5 as a mainline and keep it as appendix/diagnostic.
