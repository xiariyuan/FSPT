# Route-D full-population data Gate 3C1F1 v0 — 2026-07-20

## Status

```text
PREREGISTERED_NOT_MATERIALIZED
new population metrics unread
external benchmarks unread
```

## Purpose

Gate 3C1F0 established a conservative causal entry model, but the complete-video
pilot used 16 previously exposed videos and its AJ confidence interval crossed
zero. Gate 3C1F1 creates a third raw-record-disjoint Kubric population for a
one-shot internal confirmation of the entire frozen pipeline:

```text
causal entry
-> expensive temporal-identity candidate generation only for accepted points
-> frozen Gate 3C1D top-1 support/value/harm policy
-> deterministic four-level memory writeback
-> complete-video TAP metrics
```

This gate qualifies raw identity only. It cannot change any model or threshold.

## Exact exclusion

Both prior 512-video manifests are excluded by raw TFRecord identity:

```text
Gate 3C0 population: 512 identities
Gate 3C2 population: 512 identities
combined exclusions: 1,024 unique identities
combined digest:
6569347faf10aa3189581186b919e94bb39aaac33172552436ec30a0e8e24c89
```

Changing point seeds on an existing raw video does not count as independent.

## Frozen selection

The materializer deterministically takes the first 128 non-excluded train
records in source-file/record order:

```text
first: movi_e-train.tfrecord-00107-of-01024, record 5
last:  movi_e-train.tfrecord-00120-of-01024, record 9
selected source TFRecords: 14
selected identity digest:
85193d0aa6381c7d78442e85bf87750ce72a497995a92016001ca0cf51832f28
```

Materialization:

```text
videos: 128
points per video: 64
output shards: 8 x 16 videos
sampling: uniform
seed: 271828
workers: 8
source-file hashing: required
```

## Qualification gates

The data gate requires:

- exactly 128 unique selected identities;
- exactly 1,024 unique excluded identities;
- zero raw overlap;
- exact first/last identity and selected/excluded digests;
- exact two-manifest exclusion authority;
- exact dataset-info authority;
- 14 selected source TFRecords with independent exact hashes;
- exactly eight output shards;
- all locked metric and external-data flags remain false.

A pass issues only:

```text
AUTHORIZE_GATE3C1F2_FULL_POPULATION_CONFIRMATION_PREREGISTRATION
```

A fail issues:

```text
STOP_GATE3C1F1_AND_REPAIR_DATA_IDENTITY
```

## Downstream boundary

The 128 videos form one frozen `raw_disjoint_full_population_confirmation_v0`
partition. There is no threshold-selection split. Gate 3C1F2 must commit all
implementation hashes, complete-video gates, bootstrap rules, and exact replay
contract before reading any model metric on this population.

DAVIS, Kinetics, final holdout, and official Kinetics 1,144 remain locked.
