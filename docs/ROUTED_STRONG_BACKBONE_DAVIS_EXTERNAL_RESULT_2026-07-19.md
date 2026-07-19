# Route-D strong-backbone DAVIS external transfer result — 2026-07-19

## 1. Formal decision

The preregistered frozen DAVIS transfer gate fails:

```text
STOP_STRONG_BACKBONE_EXTERNAL_CLAIM_AND_RETAIN_SYNTHETIC_EVIDENCE_ONLY
```

Variant C must therefore remain an internal strong-backbone result supported by
Kubric model-validation and the identity-disjoint Kubric final holdout. It does
not support a strong-backbone external-transfer claim.

DAVIS was already historically exposed to earlier project routes, so this result
was never eligible to be called an untouched final test. The exact allowed scope
was a one-time, model-frozen zero-shot transfer audit.

## 2. Implementation integrity before the valid evaluation

Two interface defects were caught before a valid metric file was accepted:

1. The DAVIS cache report omitted the `native_state_hashes` field required by the
   generic CMCP loader. The repair computes the five native tensor hashes from
   each base artifact and requires exact equality with the already-sealed feature
   artifact. All 30 report rows were updated through the resume path; every video
   was `resume_skip`, so CoTracker and variant C were not rerun during repair.
2. The Kubric evaluator concatenated videos along the point axis and therefore
   assumed equal frame counts. DAVIS videos have different lengths. P0m now
   follows the official CoTracker aggregation contract: compute official TAP-Vid
   metrics independently for each video, then take the equal-video mean over all
   30 videos. Candidate generation, model outputs, visibility, and per-video
   metrics were unchanged.

Neither failed attempt wrote a valid performance result. The final primary and
replay evaluations use the corrected sealed-cache contract and are byte-identical.

## 3. Protocol sanity

The frozen native CoTracker3 result is:

```text
P0m native AJ:                    64.4109
independent official replication: 64.4408
absolute difference:              -0.0299 points
```

This close agreement makes a query/raster/evaluator mismatch an implausible
explanation for the negative variant-C result.

Aggregation:

```text
official equal-video mean over the complete 30-video DAVIS set
```

## 4. Main result

| Metric | Native | Variant C | Candidate oracle | Variant-C gain | Oracle gain |
|---|---:|---:|---:|---:|---:|
| AJ | 64.4109 | 62.4940 | 64.9045 | -1.9170 | +0.4936 |
| Delta average | 77.1721 | 75.6130 | 77.8276 | -1.5591 | +0.6555 |
| OA | 90.8488 | 90.8488 | 90.8488 | +0.0000 | +0.0000 |

Paired-video direct AJ:

```text
mean:   -1.9170 points
95% CI: [-2.4472, -1.4449]
positive / negative / zero videos: 0 / 30 / 0
median video gain: -1.5429
range: [-5.3879, -0.2373]
```

The confidence interval is entirely negative and all 30 videos regress. This is
not a marginal failure around the preregistered `+0.30` gate.

## 5. Threshold profile

| Threshold | Selected pts gain | Oracle pts gain | Selected Jaccard gain | Oracle Jaccard gain |
|---|---:|---:|---:|---:|
| 1px | -1.9648 | +0.3701 | -1.6809 | +0.2575 |
| 2px | -2.1853 | +0.8836 | -2.4469 | +0.7830 |
| 4px | -2.0282 | +0.9602 | -2.8607 | +0.9212 |
| 8px | -1.5272 | +0.7077 | -2.4365 | +0.4399 |
| 16px | -0.0902 | +0.3560 | -0.1598 | +0.0664 |

Variant C loses accuracy at every threshold. The loss is largest at 1--8px and
smallest at 16px. The oracle remains positive at every threshold, but its total
AJ headroom is only `+0.4936` point, far below the
preregistered `+3.0` requirement.

## 6. Safety and action behavior

```text
visible post-query rows:             28174
selected non-native rate:            2.3035%
harmful non-native rate:             2.2148%
beneficial candidate availability:   2.1332%
beneficial candidate recall:         1.6639%
```

Approximate row counts implied by the pooled rates:

```text
non-native interventions: about 649
harmful interventions:    about 624
beneficial selections:    about 10
```

Severe 16px error rate:

```text
native:   2.6656%
selected: 2.7508%
change:   +0.0852 percentage points
oracle:   2.4455%
```

The selected intervention rate is only about 2.3%, but almost every accepted
intervention is harmful. Sparsity alone therefore does not provide transfer
safety.

## 7. Per-video extremes

Worst five videos:

| Index | Video | AJ gain | Selection rate | Harmful rate | Beneficial availability |
|---:|---|---:|---:|---:|---:|
| 26 | scooter-black | -5.3879 | 6.0475% | 6.0475% | 0.2160% |
| 21 | kite-surf | -4.9615 | 4.1872% | 4.1872% | 0.2463% |
| 16 | blackswan | -4.7772 | 3.6735% | 3.6735% | 0.0000% |
| 24 | car-shadow | -4.4560 | 4.0053% | 4.0053% | 0.0000% |
| 5 | drift-straight | -3.3589 | 2.8947% | 2.8947% | 0.0000% |

Least-negative five videos:

| Index | Video | AJ gain | Selection rate | Harmful rate | Beneficial availability |
|---:|---|---:|---:|---:|---:|
| 27 | mbike-trick | -0.2373 | 0.7879% | 0.7879% | 2.6921% |
| 10 | india | -0.2921 | 0.3799% | 0.3799% | 0.9119% |
| 12 | cows | -0.4196 | 0.6863% | 0.6863% | 0.0000% |
| 8 | dogs-jump | -0.5705 | 1.0036% | 0.9124% | 1.6423% |
| 3 | breakdance | -0.6024 | 1.2725% | 1.0479% | 3.6677% |

Even the best video remains negative (`-0.2373` AJ point).

## 8. Formal gate

| Gate | Result | Observed value |
|---|---|---:|
| Complete 30-video partition | PASS | 30 / 30 |
| Native candidate parity exact | PASS | exact |
| Candidate oracle AJ >= +3.0 | FAIL | +0.4936 |
| Direct AJ >= +0.30 | FAIL | -1.9170 |
| Paired AJ CI lower > 0 | FAIL | -2.4472 |
| Delta-average gain > 0 | FAIL | -1.5591 |
| Severe 16px not worse | FAIL | +0.0852 pp |
| Harmful non-native <= 1% | FAIL | 2.2148% |
| Positive videos >= 18/30 | FAIL | 0 / 30 |
| Exact primary/replay | PASS | byte-identical |

Only completeness, native parity, and exact replay pass. Every scientific
performance or safety gate fails.

## 9. Why the transfer fails

The result identifies two simultaneous domain shifts:

1. **Candidate-generation shift.** Kubric final-holdout oracle gain was about
   `+16.75 AJ`; on DAVIS it is only `+0.4936`. The
   frozen CMCP proposal pool therefore contains little recoverable external-domain
   headroom.
2. **Action-selection shift.** The comparator still accepts approximately
   `2.30%` of visible rows, but the
   harmful rate is `2.21%`. The
   Kubric utility/risk boundary does not transfer to DAVIS evidence statistics.

Because the oracle itself fails, this cannot be repaired merely by changing an
abstention threshold. Because the selected actions are almost uniformly harmful,
a post-hoc DAVIS threshold or calibration sweep would also violate the frozen
protocol and would not establish zero-shot transfer.

## 10. Claim correction and stop rule

Retain:

```text
- reproducible CoTracker3 strong-backbone improvement on Kubric model-validation;
- reproducible improvement on the identity-disjoint 16-video Kubric final holdout;
- component attribution favoring frozen CMCP + LMRA + comparator;
- output-only execution boundary;
- DAVIS transfer failure as a transparent limitation.
```

Do not claim:

```text
- strong-backbone external transfer;
- tracker-agnostic or domain-general routing;
- empirical safety on DAVIS;
- a closed-loop strong-backbone result;
- SOTA or universal point-tracking improvement.
```

The preregistered stop rule forbids using this DAVIS result for checkpoint,
candidate, threshold, calibration, NMS, EMA, top-K, or writeback rescue. Official
Kinetics 1,144 remains frozen and is not rerun or retuned.

## 11. Exact replay and artifacts

```text
primary result SHA-256:
44b580455e9913d31c654b53051d7b7d76944cb0e34331c6a752bf24a2ccb301

replay result SHA-256:
44b580455e9913d31c654b53051d7b7d76944cb0e34331c6a752bf24a2ccb301

canonical summary SHA-256:
3ed8702252b455c070f0e589165e569df5e4c799a701b5c362f06c788b581b40

sealed cache index SHA-256:
7719b131992de0db8e17f4e6ccf6e5114e6f17e13b058103a2085999ed9277a5

sealed cache payload SHA-256:
0145645d3c02e60f14fa8a7012d85f9a02a19b04c8f17c99bf02c928ac2d52c8

variant-C checkpoint SHA-256:
7babb76e3407497832d0bc0fca4557df64b2d09450ebc60766e32a70816ab52f

combined model-state SHA-256:
64c3f6dae6ae34dc0223754aca3c76f78a1f48137d08bab6c3e01f2a806cd074
```

Canonical summary:

```text
docs/generated/ROUTED_STRONG_BACKBONE_DAVIS_EXTERNAL_SUMMARY_2026-07-19.json
```
