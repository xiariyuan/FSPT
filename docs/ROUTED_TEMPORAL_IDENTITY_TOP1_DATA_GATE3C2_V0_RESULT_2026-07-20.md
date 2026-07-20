# Route-D top-1 data renewal Gate 3C2 v0 result — 2026-07-20

## Formal status

```text
COMPLETED_PASS
AUTHORIZE_GATE3C1D_V1_TWO_STAGE_TOP1_PREREGISTRATION
```

Gate 3C2 qualifies data identity and partition membership only. It does not
report a top-1 selector or future-rollout result.

## Qualified renewal population

| Quantity | Result |
|---|---:|
| New raw videos | 512 |
| Output shards | 32 |
| Selected source TFRecords | 55 |
| Existing raw identities excluded | 512 |
| Raw-identity overlap | 0 |
| Available source records | 9,749 |
| Output size | approximately 2.3 GiB |

The first selected identity is
`movi_e-train.tfrecord-00053-of-01024:7`; the last is
`movi_e-train.tfrecord-00107-of-01024:4`.

## Integrity

Every preregistered check passed:

- sample and shard counts are exact;
- all selected identities are unique;
- overlap with the existing 512 identities is empty;
- exclusion manifest path, hash, count, and identity digest are exact;
- selected identity digest equals the pre-decoding frozen digest;
- first and last identities are exact;
- TFDS dataset-info authority reports exactly 1,024 train shards and 9,749
  records;
- all 55 selected source TFRecords independently rehash exactly;
- the three renewed partitions cover exactly indices 0--511 and are disjoint;
- all three partition identity digests match their preregistered values;
- calibration, final holdout, DAVIS, Kinetics, and official Kinetics remain
  unread.

## Frozen identity authority

```text
renewal manifest SHA256:
  8937a2ef925b6c97c994da7b755b91b401d2239162c1254b720567f66e028105
selected identity digest:
  ae7f8c4231dc81b52327020c43914def5ec6a4666bc374bd2a5539d1be5bcf37
summary payload SHA256:
  576f3a97f9a878726dda6eca7047c38413c343882decb23b8947c1e70d6db03b
summary file SHA256:
  0aa42f3d8d952dc6ff7fcc600eedbf3d9738609838f439dc34df7200988e5ad8
```

## Renewed partitions

| Partition | Indices | Videos | Raw-identity digest |
|---|---:|---:|---|
| checkpoint selection v1 | 0--255 | 256 | `c4e2abf2cc062ef0fcff8c239397f11dbb41c1096958f16b2615656975c4653d` |
| fit-only audit v1 | 256--383 | 128 | `1c530d9a0ffc0e06ca88ddcb8d338428f3d050c4dd42fd2f5a2265e722e8b96c` |
| model validation v1 | 384--511 | 128 | `5b737ec10091e6744e4f598673cbb2068b8b3fdbdb2b6fa28ceb0b3ffc55f056` |

## Authorized next mechanism

A Gate 3C1D v1 protocol may now be preregistered with two distinct functions:

1. shortlist candidate ranking / expected-distance prediction; and
2. row-level expected action value and harmful-action risk.

The row-level model must be trained from out-of-fold candidate predictions. The
failed v0 design, which reused one candidate-support probability as both ranking
score and action confidence, is forbidden.

No renewed partition may be used for metrics before the v1 model, thresholds,
gates, feature cache protocol, and exact replay requirements are committed.

## Claim boundary

This is a raw-data identity result, not model performance. Final holdout and all
external datasets remain locked.
