# TAP-Vid-Kinetics Official Protocol and Package Audit — 2026-07-16

## Decision

The local Kinetics package and the corrected project evaluator are now independently tied to pinned official TAP-Vid artifacts.

The corrected Route-D rerun may be reported, if its frozen primary decision rule passes, using the following scope:

> First-query, 256 x 256 evaluation on the complete local materialization of 1,144 of the 1,147 uniquely annotated video segments in the byte-verified official TAP-Vid-Kinetics release CSV, using metric formulas with exact parity to the pinned official evaluator.

It must not be described as:

- a universally fixed 1,000-video dataset;
- an official train-only split result;
- a leaderboard submission;
- an untouched-dataset result without disclosing the prior balanced-50 and superseded full-package evaluations.

The controller, scorer, policy, thresholds, fusion, and guards remain unchanged after all Kinetics observations. The corrected run is a protocol-correction rerun, not a tuned rerun.

## 1. Metric-coordinate defect and correction

The project TAP-Vid wrapper previously mapped normalized coordinates to a 256 x 256 raster with `width - 1` and `height - 1`, producing a scale of 255. The pinned official reader maps normalized coordinates with full `width` and `height`, producing a scale of 256.

This difference changes strict 1 / 2 / 4 / 8 / 16-pixel threshold decisions near their boundaries and therefore affects:

- `pts_within_1/2/4/8/16`;
- `jaccard_1/2/4/8/16`;
- average point-threshold accuracy;
- Average Jaccard;
- all derived differences and paired bootstraps based on those metrics.

Occlusion accuracy is not changed by coordinate scaling when visibility predictions are unchanged.

The corrected project contract is:

```text
normalized [y, x] -> raster [x, y]
x_raster = x_normalized * width
y_raster = y_normalized * height
```

## 2. Pinned official evaluator parity

Official source:

```text
repository: google-deepmind/tapnet
commit: 989a1fd62f7b2a3cf7f1c339bbde38e086e3a0fc
source: tapnet/tapvid/evaluation_datasets.py
source SHA-256: 90cd01e53e23f6d489d3a6cd840cfd93fed4c1a6f4a0dfd373933cc164f0e570
```

Parity audit:

```text
artifact: official_protocol_audit_20260716/metric_parity.json
artifact SHA-256: 61e2da03730de3604687d1641d885fbfa79611f42dae9123cb016f5fdffc9462
```

Results:

- 100 randomized direct official-function comparisons: maximum absolute difference `0.0`;
- 100 randomized project-wrapper comparisons across `first` and `strided` query modes and square/non-square rasters: maximum absolute difference `0.0`;
- deterministic real Kinetics annotation case with threshold-boundary perturbations: maximum absolute difference `0.0`;
- overall parity: pass.

A regression test explicitly rejects the previous 255-scale behavior using a `1.001 / 256` normalized displacement at the strict one-pixel boundary.

## 3. Byte-verified official release package

Official archive:

```text
URL: https://storage.googleapis.com/dm-tapnet/tapvid_kinetics.zip
Content-Length: 25,018,959 bytes
ETag / MD5: c67da40cdb7b08e3d0a375f5fe605a89
Last-Modified: Mon, 19 Dec 2022 23:59:35 GMT
ZIP SHA-256: a0e55b43d4381744221b45fc3fef4f2535b54c152c7902004f49ee87f92f18d0
```

The archive passed full decompression validation. The following local files match the archive members byte-for-byte:

| File | SHA-256 | Exact match |
|---|---|---:|
| `tapvid_kinetics.csv` | `d02e8fafa631ec1929c30172a9b0fe0a17dff9306f5ebd35120e9ed4cdf100e6` | yes |
| `README.md` | `18c90d566bc910a175825a0ac9f267da8d6d07714b8887f789f4c96323e752a5` | yes |
| `train.txt` | `28d64e88bce9b804213c46e1e21c91c21b88d2fa55d3676a5b5ec1790d9042e1` | yes |
| `val.txt` | `51b03ff2811d2e0589f3a6e53489c12fa847b20a8b92b3e686d12474b895ddae` | yes |
| `test.txt` | `8a348ed9c776e87a755f862297035bc5a25dec696b1ec731ecec3aab729d7387` | yes |

Release audit:

```text
artifact: official_protocol_audit_20260716/official_release_package.json
artifact SHA-256: 8cec5d67ec9c1707e5c1da4d4eccf5209cbb316191fe60d6dabed6e1f5653be3
status: pass
```

## 4. Official generator and local shard identity

Pinned generator:

```text
repository: google-deepmind/tapnet
commit: 989a1fd62f7b2a3cf7f1c339bbde38e086e3a0fc
source: tapnet/tapvid/generate_tapvid.py
source SHA-256: 2341343f14a1d2e562bf83a387828448c7d0af39729760011570db47ccf6a53f
```

The official generator:

1. preserves first-occurrence CSV key order;
2. skips a CSV video segment when the source video is unavailable;
3. stores coordinates as `(x * width - 0.5) / width` and `(y * height - 0.5) / height`;
4. writes continuous shards with `ceil(number_of_materialized_videos / 10)` examples per shard.

A full reverse audit loaded all ten local pickle shards and matched every local sample against the ordered release CSV using the exact official coordinate transform and occlusion array.

Results:

```text
release CSV annotation groups: 1,147
local materialized samples: 1,144
exact matched samples: 1,144
skipped CSV segments: 3
unmatched local samples: 0
maximum point absolute difference: 0.0
occlusion arrays: exact
shard count: 10
matched key-sequence SHA-256: bb3126c2b220c2fdc83a4b6da2dacbe307ae8eb112f05713eae88337825f48d1
```

Skipped release CSV segments:

```text
BBSK3Wv0jXM_000112_000122
G4Z1Ug34B5I_000224_000234
pk1yi_HMsAI_000053_000063
```

Identity artifacts:

```text
raw full scan: official_protocol_audit_20260716/package_identity.raw_full_scan.json
raw full scan SHA-256: 55048ce54bcaed622d088e63cc38bee99a7e69fabab46f5bbfe3de8be98e1292
reviewed identity audit: official_protocol_audit_20260716/package_identity.json
reviewed identity audit SHA-256: 909676c1cb519b65ffb93ab22faed8498739fc734a3def314702429d010e4d16
status: pass
```

The raw scan originally failed because it incorrectly assumed that the sum of the auxiliary split text files, 1,189, was the number of CSV annotation groups. The raw matching evidence was preserved unchanged. Reclassification rehashed all ten shards, revalidated the full ordered key partition, and changed only the metadata-authority rule.

## 5. Auxiliary split-file discrepancy

The byte-verified release itself contains metadata that is not set-equal:

```text
release CSV unique IDs: 1,147
train.txt IDs: 475
val.txt IDs: 237
test.txt IDs: 477
split-file union: 1,189
CSV IDs absent from split-file union: 19
split-file IDs absent from CSV: 61
```

The three split files are mutually disjoint, but their union is not an exact membership index for the release CSV. Therefore:

- the release CSV is the annotation identity authority;
- split text files are auxiliary metadata only;
- the local sharded manifest value `split=train` is a loader label, not an official train-only evaluation restriction;
- publication text must not aggregate the split-file row counts into the CSV annotation count.

## 6. Superseded Route-D metrics

All pre-correction Route-D closed-loop TAP position metrics are marked superseded, including DAVIS, RGB-Stacking, Kinetics balanced-50, and the first full-1,144 run.

```text
artifact: official_protocol_audit_20260716/routeD_pre_official_scale_results.SUPERSEDED.json
artifact SHA-256: 7403b943397781d6b8c816b5e0d5cd5dd58ec1e3ad1103e458b3fd8e442e6945
listed artifacts: 15
```

These files are preserved for audit. Their TAP position values must not be quoted as corrected official-scale results.

## 7. Corrected full-package rerun requirements

Before corrected inference begins, a new protocol must record:

- clean Git HEAD and branch;
- corrected metric implementation SHA-256;
- checkpoint, controller, and configuration SHA-256 values;
- official evaluator source commit and hash;
- official generator source commit and hash;
- metric-parity audit hash;
- official release-package audit hash;
- package-identity audit hash;
- global supersession manifest hash;
- all ten source-shard and per-shard manifest hashes;
- query mode `first`;
- input and metric raster `256 x 256`;
- independent baseline;
- unchanged primary comparison and decision rule;
- no controller or policy tuning after the superseded observations.

Every per-shard result must contain the corrected metric-coordinate contract and exact metric-implementation hash. The resumable runner must reject any old 255-scale result even when its row count is complete.

The corrected primary rule remains:

> Both paired-video bootstrap 95% confidence-interval lower bounds for closed-loop versus independent baseline must exceed zero: one for Average Jaccard and one for average point-threshold accuracy.

## 8. Evidence interpretation

A passing corrected run supports frozen-controller transfer on the exact byte-verified release/materialization scope. It does not erase prior Kinetics exposure. The paper must disclose:

- balanced-50 was evaluated before the full package;
- a first full-package run was superseded after independent metric parity found the raster-scale defect;
- no Kinetics result was used to retune the frozen controller or policy;
- the corrected run retained the original primary decision rule.

## 9. Corrected official-scale full evaluation result

The corrected protocol was generated from clean Git commit
`f65e4af10827101de8b44180ff0d14efc6345a8a` before any corrected inference.
Its SHA-256 is:

```text
61f1458ee0e62b865cde85d9f77e765b79da567add8fc277fe10599b0db13155
```

Execution completed all ten source shards sequentially. Every shard completed on
attempt 1, and every result passed the frozen metric-implementation SHA and
coordinate-contract checks.

Protocol scope:

```text
official release CSV annotation groups: 1,147
exact local materialization evaluated: 1,144
CSV segments not materialized: 3
query mode: first
input raster: 256 x 256
metric raster: 256 x 256
normalized-to-raster contract: x * width, y * height
controller/policy changes after superseded run: none
paired bootstrap resamples: 20,000
bootstrap seed: 17
```

Aggregate metrics:

| System | AJ | OA | Delta average |
|---|---:|---:|---:|
| Independent baseline/local | 0.324945 | 0.939843 | 0.420461 |
| Frozen tree, open-loop | 0.337390 | 0.939843 | 0.436503 |
| Frozen tree, closed-loop | 0.347988 | 0.939843 | 0.447991 |

Aggregate gains:

| Comparison | AJ | Delta average |
|---|---:|---:|
| Open-loop vs baseline | +0.012445 | +0.016043 |
| Closed-loop vs baseline | +0.023043 | +0.027530 |
| Closed-loop vs open-loop | +0.010598 | +0.011488 |

Paired-video bootstrap results:

| Comparison | Metric | Videos | Mean | 95% CI |
|---|---|---:|---:|---:|
| Open-loop vs baseline | AJ | 1,138 | +0.012445 | [+0.011200, +0.013715] |
| Open-loop vs baseline | Delta average | 1,137 | +0.016043 | [+0.014671, +0.017425] |
| Closed-loop vs baseline | AJ | 1,138 | +0.023043 | [+0.020567, +0.025569] |
| Closed-loop vs baseline | Delta average | 1,137 | +0.027530 | [+0.024800, +0.030260] |
| Closed-loop vs open-loop | AJ | 1,138 | +0.010598 | [+0.008805, +0.012395] |
| Closed-loop vs open-loop | Delta average | 1,137 | +0.011488 | [+0.009494, +0.013465] |

The preregistered primary decision passes because the lower confidence bounds
for closed-loop AJ and delta average versus the independent baseline are both
strictly greater than zero. The additional closed-loop benefit over open-loop is
also statistically resolved.

Finite-pair and failure accounting:

```text
AJ finite paired videos: 1,138
Delta-average finite paired videos: 1,137
videos with at least one undefined official TAP metric: 7
invalid comparison/metric entries retained in the audit: 57
closed-loop AJ positive / negative / tied: 775 / 199 / 164
closed-loop Delta-average positive / negative / tied: 783 / 188 / 166
```

The seven undefined-metric videos remain in the raw merged data. They are
excluded only from the affected paired finite-value statistic; no NaN is
replaced by zero.

Operational diagnostics:

```text
closed-loop global selection rate: 4.0244%
open-loop global selection rate: 7.9028%
mean closed-loop trajectory difference from local: 2.1264 px
mean closed-versus-open trajectory difference: 1.7413 px
memory-write disagreement rate: 0
```

The most severe corrected closed-loop AJ failures remain material limitations:

1. `kinetics_source_s000_p000113_kinetics_s000_000113`: AJ -0.297626,
   delta average -0.274853.
2. `kinetics_source_s009_p000080_kinetics_s000_000080`: AJ -0.252266,
   delta average -0.264887.
3. `kinetics_source_s008_p000034_kinetics_s000_000034`: AJ -0.159537,
   delta average -0.096077.

They must not be used to tune a Kinetics-specific guard. Any future stability
guard must be developed on Kubric-only partitions and evaluated under a new
external protocol.

Sensitivity to the corrected raster scale:

```text
baseline AJ change from superseded run: -0.000786
baseline Delta-average change: -0.000871
closed-loop AJ change: -0.000812
closed-loop Delta-average change: -0.000906
closed-vs-baseline AJ gain change: -0.000027
closed-vs-baseline Delta-average gain change: -0.000035
```

Thus the coordinate correction changes the exact numerical values but not the
primary statistical decision.

Final artifacts:

```text
kinetics_full1144_officialscale_v2_20260716/full1144.officialscale.protocol.json
  SHA-256: 61f1458ee0e62b865cde85d9f77e765b79da567add8fc277fe10599b0db13155
kinetics_full1144_officialscale_v2_20260716/full1144.officialscale.merged.json
  SHA-256: 4b6d796ed017a86773c67c6447fd12827e9dfc185b0198430eef5f9813bf7b7e
kinetics_full1144_officialscale_v2_20260716/full1144.officialscale.paired.json
  SHA-256: f97da99d8e905d765c1f267521e4995b1c940ec2ae59bdcd751159a2b6e38c9a
kinetics_full1144_officialscale_v2_20260716/full1144.officialscale.final_audit.json
  SHA-256: 8cf2af93c6629542b02062a625fc8c54ac3a14c6c65a6a3fb4111228c450408c
```

Final audit decisions:

```text
primary_pass: true
official_protocol_pass: true
paper_claim_eligible: true
```

Required wording:

> Report this as the exact order-preserving local materialization of 1,144 of
> the 1,147 uniquely annotated video segments in the byte-verified official
> TAP-Vid-Kinetics release CSV (3 CSV segments were not materialized), evaluated
> with the pinned official metric formulas. Do not describe it as a universally
> fixed 1,000-video set or an official train-only split.
