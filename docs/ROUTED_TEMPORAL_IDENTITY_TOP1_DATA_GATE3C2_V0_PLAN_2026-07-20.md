# Route-D top-1 data renewal Gate 3C2 v0 — 2026-07-20

## Status

```text
PREREGISTERED_NOT_RUN
```

## Why new data is required

Gate 3C1D v0 failed exactly on checkpoint-selection indices `384--447`. Those
results are now observed and cannot independently confirm a redesigned policy.
The fit-only and original-model partitions were already exposed to teacher
coordinates and future rollout by Gates 3C1B/3C1C. Reusing them as fresh v1
confirmation would overstate independence.

The MOVi-E Kubric train source contains 9,749 raw video records across 1,024
TFRecord shards. The existing temporal-identity population uses 512 unique raw
records, leaving 9,237 unused. No new external dataset or pretrained weight is
needed.

## Frozen materialization

Materialize exactly 512 new records after excluding every raw identity in the
existing 512-sample manifest:

```text
output: routeD_temporal_identity_top1v1_kubric512_seed314159_20260720
points per video: 64
output shard size: 16
sampling: uniform, hard fraction 0.5
point seed: 314159
```

Raw identity is `(source_tfrecord, source_record_index)`. A different point seed
on the same video does not count as independent data. Before decoding, the exact
512-member selection is frozen from the dataset-info shard lengths and exclusion
set: first identity `00053:7`, last identity `00107:4`, combined identity digest
`ae7f8c4231dc81b52327020c43914def5ec6a4666bc374bd2a5539d1be5bcf37`.
Each partition identity digest is also pinned in the config.

The historical Gate 3C0 preprocessor remains byte-unchanged. A new wrapper imports
its frozen decoder/track generator and adds exact manifest exclusion. TFRecord
files fully covered by the exclusion set are skipped using the hash-pinned TFDS
`dataset_info.json` shard lengths, avoiding unnecessary reads while preserving
identity order.

## Frozen partitions

```text
new checkpoint selection: 0--255   (256 videos)
new fit-only audit:        256--383 (128 videos)
new model validation:      384--511 (128 videos)
```

All current 512 records are development-exposed after completed gates and may be
used only to design/train the two-stage v1 model. No renewed partition may be
read for v1 metrics before the v1 model, threshold grid, and gates are committed.

## Mandatory data checks

- exactly 512 selected samples and 32 output shards;
- every selected raw identity is unique;
- zero raw-identity overlap with the existing 512;
- exclusion manifest path and SHA256 are exact;
- exclusion identity count and digest are exact;
- TFDS dataset-info path, SHA256, shard count, and record count are exact;
- every TFRecord containing a selected sample is independently rehashed;
- selected identity digest is reproduced from output sidecars;
- the three renewed partitions cover exactly `0--511` without overlap;
- calibration, final holdout, DAVIS, Kinetics, and official Kinetics remain
  unread.

## Authorized v1 mechanism after a pass

A pass authorizes preregistration only of a two-stage top-1 policy:

1. candidate ranker/expected-distance model inside the frozen shortlist;
2. row-level action-value and harmful-action risk model;
3. action-risk training must use out-of-fold candidate predictions;
4. checkpoint threshold selection may use only renewed indices `0--255`;
5. renewed audit and model validation cannot change the policy.

The failed v0 pattern—one candidate-success probability reused directly as both
ranking score and action confidence—is forbidden.

```text
pass -> AUTHORIZE_GATE3C1D_V1_TWO_STAGE_TOP1_PREREGISTRATION
fail -> STOP_GATE3C2_AND_REPAIR_RAW_IDENTITY_RENEWAL
```

## Claim boundary

Gate 3C2 is a data-identity gate. It cannot establish top-1 accuracy, future
rollout improvement, final-holdout gain, or external performance.
