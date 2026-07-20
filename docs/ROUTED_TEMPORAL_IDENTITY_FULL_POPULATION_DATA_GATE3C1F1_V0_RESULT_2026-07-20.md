# Route-D full-population renewal data Gate 3C1F1 v0 result — 2026-07-20

## Formal decision

```text
COMPLETED_PASS
AUTHORIZE_GATE3C1F2_FULL_POPULATION_CONFIRMATION_PREREGISTRATION
```

The third Kubric population was materialized exactly as preregistered and independently qualified. This gate validates raw data identity only; no tracking metric from the new population has been read.

## Frozen population

```text
samples:                         128
shards:                          8
excluded raw identities:        1,024
selected source TFRecords:      14
manifest SHA256:                060e0f9de3aeb945932ea15b9a0a7ab70aa88c1d703fa313c9e0db21ad2f3c01
selected identity digest:       85193d0aa6381c7d78442e85bf87750ce72a497995a92016001ca0cf51832f28
first identity:                 movi_e-train.tfrecord-00107-of-01024:5
last identity:                  movi_e-train.tfrecord-00120-of-01024:9
```

The population excludes every raw identity used by the original Gate 3C0 512-video population and the renewed Gate 3C2 512-video population. Point resampling was not treated as independence.

## Qualification checks

All preregistered checks pass:

```text
sample count exact:                 true
shard count exact:                  true
selected identities unique:        true
excluded identities unique:        true
raw identity overlap empty:         true
excluded count and digest exact:    true
selected digest exact:              true
first/last identities exact:        true
source-file count exact:            true
all 14 source TFRecord hashes exact:true
exclusion authorities exact:        true
dataset_info authority exact:       true
locked data unread:                 true
```

Qualification summary payload SHA256:

```text
25688db59aa64a84c1be877ea8c136f5b6d471eb5a064ac2385cd0846beac461
```

Qualification summary file SHA256:

```text
bfdf6a1dc2b513dcf8db6a678573eaf403294d2925f11acc9153e46177655969
```

## Authorized next step

Gate 3C1F2 may preregister one complete-population confirmation using the already frozen pipeline:

```text
causal 130-D entry model
entry probability >= 0.93
native joint probability <= 0.02
Gate 3C1D expected-distance top-1 plus support/value/harm policy
coordinate plus deterministic four-level memory writeback
complete-video AJ, delta_avg, and OA
fresh-process exact replay
```

No threshold, feature, model, candidate, writeback, metric, or visibility rule may be adjusted after reading the new 128-video result.

## Claim boundary

This result is a raw-data qualification, not a tracking result. DAVIS, Kinetics, final holdout, and official Kinetics 1,144 remain locked.
