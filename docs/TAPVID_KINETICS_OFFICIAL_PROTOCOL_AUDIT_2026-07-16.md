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
