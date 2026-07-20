# Notion deferred synchronization log

This file is the temporary authoritative handoff while the Notion connector is unavailable.
At the end of every execution round, append a dated entry containing:

1. exact server/Git state;
2. completed code, tests, experiments, and decisions;
3. claim boundary and locked-data status;
4. failures or infrastructure blockers;
5. next authorized action;
6. files and commits that must later be synchronized to Notion.

When Notion access is restored, synchronize every entry in order and record the destination page and synchronization timestamp here. Do not delete historical entries after synchronization.

---

## 2026-07-20 — Gate 3C1A finalization and Gate 3C1B preregistration

### Repository state

- repository: `/gemini/code/FSPT_v9a60_clean`
- branch: `beliefcal-mmp-mvp1-20260713`
- Gate 3C1A result commit: `076c8b1 docs(routeD): finalize temporal identity train cache`
- Gate 3C1B preregistration commit: `f8fa88e feat(routeD): preregister query-closure identity selector`
- both commits pushed to `origin/beliefcal-mmp-mvp1-20260713`

### Gate 3C1A completed result

- exact gradient-train membership: source indices `64--383`
- validated sidecars: `320/320`
- videos with failures: `308`
- valid empty videos: `12`
- frozen natural-failure rows: `3,325`
- native 12 px support: `54/3,325 = 1.6241%`
- native plus static M1 top-8 support: `1,725/3,325 = 51.8797%`
- complete native-plus-128 oracle support: `3,185/3,325 = 95.7895%`
- formal decision: `AUTHORIZE_GATE3C1B_SELECTOR_PREREGISTRATION`

Claim boundary: this is training-only candidate-bank headroom, not learned selector performance, future rollout improvement, model-validation evidence, or an external result.

### Gate 3C1B frozen mechanism

The primary selector is a zero-parameter query-closure identity score. Candidate zero is always retained; eight non-native candidates are retained by the fixed score:

```text
score =
  -2 * z(reverse_cycle_error_at_query)
  +1 * z(query_descriptor_cosine_at_query_frame)
  +1 * z(mean_query_descriptor_cosine)
  +1 * z(minimum_query_descriptor_cosine)
```

No learned residual, optimizer, loss, checkpoint, or checkpoint-dependent hyperparameter is permitted. Visibility, confidence, trajectory smoothness, acceleration, jerk, and raw M1 score are excluded from the primary score and retained only as preregistered controls/ablations.

### Gradient-train mechanism diagnosis

- static M1 native plus top-8 12 px support: `51.8797%`
- cycle-only support: `54.7669%`
- identity-only support: `55.2782%`
- primary query-closure identity support: `59.2481%`
- primary absolute gain over static: `+7.3684` percentage points
- 8 px support: `31.6090% -> 38.1955%`
- median minimum teacher distance: `11.4892 px -> 9.8567 px`
- paired-video bootstrap 95% CI for support gain: approximately `[+6.04, +11.10]` percentage points
- full-cache deterministic replay: exact for selected-index, score, metric, and video-record digests

Claim boundary: this is design evidence on gradient-train only. It is not an independent confirmation result and cannot be reported as final tracking improvement.

### Data-isolation protocol

- checkpoint-selection `384--447`: authorized only after Gate 3C1B preregistration
- fit-only audit `448--511`: hard-blocked until checkpoint-selection primary run and exact replay pass
- original model-validation `48--63`: unread
- calibration and final holdout: unread
- TAP-Vid DAVIS and Kinetics: unread
- official Kinetics 1,144 protocol: not rerun

The audit-cache builder validates the checkpoint-selection replay authorization before importing or initializing CoTracker/DINO heavy dependencies.

### Current infrastructure blocker

Both direct-shell and managed-process CUDA smoke tests fail before CUDA context creation with the platform bootstrap error:

```text
Fail to finish initialization. This is most likely caused by insufficient GPU resources.
```

`nvidia-smi` reports the GPU as idle (`24,258 MiB` free), so this is currently treated as an execution-platform bootstrap failure rather than repository code failure. No checkpoint-selection sidecar has been created and no index in `384--447` has been read.

### Next authorized action

Run the frozen checkpoint-selection cache builder when CUDA process initialization becomes available:

```bash
python scripts/build_routeD_temporal_identity_eval_cache_gate3c1b_v0.py \
  --config configs/routeD_temporal_identity_checkpoint_cache_gate3c1b_v0.yaml \
  --device cuda \
  --resume
```

Then run the preregistered selector evaluation twice and require exact replay before opening `448--511`.

### Notion synchronization status

- status: pending
- reason: Notion connector unavailable in the current session
- destination page: unresolved
- synchronized at: not yet synchronized

---

## 2026-07-20 — Gate 3C1B independent confirmation and Gate 3C1C preregistration

### Gate 3C1B checkpoint-selection

- source indices: `384--447`
- videos: `64`, videos with failures: `63`, failure rows: `631`
- static M1 12 px support: `58.7956%`
- query-closure identity 12 px support: `67.5119%`
- absolute gain: `+8.7163` percentage points
- equal-video bootstrap 95% CI: `[+4.1980, +12.4154]` points
- median minimum-distance improvement: `1.4719 px`
- exact fresh-process replay: pass
- decision: `AUTHORIZE_GATE3C1B_FIT_ONLY_AUDIT_448_511`

### Gate 3C1B one-shot fit-only audit

- source indices: `448--511`
- videos: `64`, videos with failures: `63`, failure rows: `753`
- static M1 12 px support: `58.5657%`
- query-closure identity 12 px support: `64.0106%`
- absolute gain: `+5.4449` percentage points
- equal-video bootstrap 95% CI: `[+2.8084, +13.5213]` points
- median minimum-distance improvement: `1.2689 px`
- exact fresh-process replay: pass
- decision: `AUTHORIZE_GATE3C1C_FUTURE_ROLLOUT_PREREGISTRATION`

Cycle-only and identity-only controls are weaker on both independent partitions.
Gate 3C1B proves improved causal shortlist retention, not deployable top-1
selection or future tracking improvement.

### Gate 3C1C frozen protocol

Gate 3C1C is preregistered but not yet run. It uses only the already exposed
fit-only indices `448--511` and freezes the following order:

1. recompute and hash the causal native-plus-eight shortlist;
2. reveal the frame-15 teacher only after shortlist freezing;
3. choose the nearest coordinate inside the shortlist;
4. compare native, coordinate-only, and coordinate-plus-four-level-memory
   continuations over frames `16--23`;
5. require coordinate-plus-memory to beat both native and coordinate-only;
6. require exact native replay and fresh-process scientific replay.

A pass authorizes only preregistration of an original model-validation protocol.
Indices `48--63`, calibration, final holdout, DAVIS, Kinetics, and official
Kinetics 1,144 remain locked.

### Notion synchronization status

- status: pending
- destination: unresolved while connector is unavailable
- synchronized at: not yet synchronized

---

## 2026-07-20 — Gate 3C1C state-action result and original model-validation preregistration

### Gate 3C1C v0 fit-only future rollout

- partition: fit-only internal audit indices `448--511`
- videos: `64`, failure videos: `63`, failure rows: `753`
- teacher selection: nearest coordinate only inside the already frozen
  query-closure native-plus-eight shortlist
- shortlist 12 px support: `64.0106%`
- shortlist median commit error: `9.0194 px`

Future rollout results over frames `16--23`:

| action | mean future error | severe >16 px | threshold utility |
|---|---:|---:|---:|
| native | 34.4136 px | 0.97792 | 0.00501 |
| coordinate only | 33.2747 px | 0.97356 | 0.00624 |
| coordinate + four-level memory | 14.9708 px | 0.30138 | 0.22518 |

Full state versus native:

- mean error reduction: `+19.4428 px`
- video-cluster 95% CI: `[+17.4422, +20.4299] px`
- threshold-utility gain: `+0.22017`
- utility 95% CI: `[+0.20416, +0.24277]`
- positive-point fraction: `98.2736%`
- severe-rate reduction: `+0.67654`

Memory versus coordinate-only:

- incremental error reduction: `+18.3039 px`
- video-cluster 95% CI: `[+16.2614, +18.9950] px`
- incremental utility: `+0.21895`
- memory-better video fraction: `98.4127%`

Native cached/recomputed parity is exact. Every selected-index, teacher-selected,
future-coordinate, point-record, video-record, and scientific-payload digest
replayed exactly.

Formal decision:

```text
AUTHORIZE_GATE3C1C_ORIGINAL_MODEL_VALIDATION_PREREGISTRATION
```

Claim boundary: this is a teacher-nearest shortlist state-action oracle. It is
not deployable top-1 selection or external tracking improvement.

### Gate 3C1C v1 original model-validation protocol

The protocol for original model-validation indices `48--63` is frozen but not
run. It makes no numerical or architectural change relative to Gate 3C1C v0:

- identical query-closure shortlist;
- identical teacher-nearest restriction;
- identical native / coordinate-only / coordinate-plus-memory branches;
- identical metrics, bootstrap seeds, and pass thresholds;
- candidate cache generation requires the exact Gate 3C1C v0 replay result and
  its pinned file hash before index 48 can be opened.

```text
pass -> AUTHORIZE_GATE3C1D_CAUSAL_TOP1_SELECTOR_PREREGISTRATION
fail -> STOP_QUERY_CLOSURE_STATE_ACTION_BEFORE_TOP1
```

At this entry, indices `48--63` remain unread. Calibration, final holdout,
DAVIS, Kinetics, and official Kinetics 1,144 remain locked.

### Notion synchronization status

- status: pending
- destination: unresolved while connector is unavailable
- synchronized at: not yet synchronized

---

## 2026-07-20 — Gate 3C1C v1 original model-validation result

### Cache integrity

- source indices: `48--63`
- videos: `16`
- videos with natural failures: `16`
- failure rows: `192`
- exact source membership and all sidecar hashes: pass
- complete top-128 12 px support: `96.8750%`
- static M1 native-plus-eight support: `64.5833%`
- read state: checkpoint selection, fit-only audit, and original model validation
  read; external data unread

### Frozen query-closure shortlist

- retained 12 px support: `72.9167%`
- median teacher-nearest commit error: `7.8908 px`
- teacher access occurred only after shortlist indices were frozen and hashed

### Future rollout

| action | mean future error | severe >16 px | threshold utility |
|---|---:|---:|---:|
| native | 34.5549 px | 0.97070 | 0.00599 |
| coordinate only | 32.9539 px | 0.95898 | 0.00962 |
| coordinate + four-level memory | 13.8870 px | 0.26321 | 0.25079 |

Full state versus native:

- error reduction: `+20.6680 px`
- video-cluster 95% CI: `[+15.8849, +27.8495] px`
- threshold-utility gain: `+0.24480`
- utility 95% CI: `[+0.20506, +0.29148]`
- positive-point fraction: `98.9583%`
- severe-rate reduction: `+0.70750`

Memory versus coordinate-only:

- incremental error reduction: `+19.0669 px`
- video-cluster 95% CI: `[+15.2600, +22.2483] px`
- incremental utility: `+0.24118`
- memory-better video fraction: `100%`

Every scientific digest and native replay reproduced exactly.

Formal decision:

```text
AUTHORIZE_GATE3C1D_CAUSAL_TOP1_SELECTOR_PREREGISTRATION
```

Claim boundary: teacher-nearest oracle only. Gate 3C1D must replace the teacher
with a causal top-1 decision and native-safe abstention. Calibration, final
holdout, DAVIS, Kinetics, and official Kinetics 1,144 remain locked.

### Notion synchronization status

- status: pending
- destination: unresolved while connector is unavailable
- synchronized at: not yet synchronized


---

## 2026-07-20 — Gate 3C1D causal top-1 selector preregistration

### Training-only mechanism diagnosis

The fixed query-closure energy remains useful for shortlist retention but is not
a valid direct top-1 confidence:

- analytic best non-native 12 px top-1 support: `9.5338%`;
- analytic candidate-vs-native 12 px support: `2.9774%`;
- same native-plus-eight shortlist oracle support: `59.3083%`;
- analytic margin AUC: `0.4815`;
- analytic top-two gap AUC: `0.4994`.

A source-video-held, gradient-train-only HGB mechanism probe produced candidate
AUC/AP `0.8489/0.4926` and raw top-1 12 px support `38.8406%`. The probe used a
temporary float16 feature copy and is design evidence only. Formal execution
rebuilds committed float32 features from source sidecars.

### Frozen Gate 3C1D protocol

- train exactly once on gradient-train indices `64--383`;
- model: fixed `HistGradientBoostingClassifier`, no architecture or epoch sweep;
- features: `102-D` causal shortlist-candidate representation;
- checkpoint indices `384--447` may choose only the lowest passing threshold
  from `[0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60]`;
- low-confidence or native-winning rows abstain to exact native state;
- audit `448--511` and original model-validation `48--63` cannot change model,
  feature contract, or threshold;
- primary/fresh-process replay compares shortlist, feature, probability, output,
  threshold-grid, record, and scientific-payload digests;
- analytic shortlist, feature builder, and runner source hashes are pinned.

Formal claim scope is commit-level top-1 selection only. Future rollout with the
causal policy is reserved for Gate 3C1E. Calibration, final holdout, DAVIS,
Kinetics, and official Kinetics 1,144 remain locked.

### Notion synchronization status

- status: pending
- destination: unresolved while connector is unavailable
- synchronized at: not yet synchronized


---

## 2026-07-20 — Gate 3C1D causal top-1 selector result

### Formal result

```text
COMPLETED_FAIL
EXACT_REPLAY_PASS
STOP_GATE3C1D_TOP1_SELECTOR
```

Checkpoint-selection indices `384--447` contained `631` failure rows. The fixed
HGB produced candidate AUC/AP `0.8274/0.4227` and raw top-1 12 px support
`35.8162%`, but no frozen threshold passed all native-safe policy gates.

| threshold | coverage | action precision | error reduction | CI lower | harmful rate |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 46.12% | 51.55% | +4.663 px | +2.968 px | 4.28% |
| 0.30 | 31.70% | 56.00% | +3.205 px | +2.127 px | 2.22% |
| 0.35 | 23.45% | 60.81% | +2.445 px | +1.498 px | 1.11% |
| 0.40 | 16.64% | 64.76% | +1.849 px | +0.990 px | 0.63% |
| 0.45 | 11.25% | 70.42% | +1.286 px | +0.592 px | 0.48% |
| 0.50 | 7.61% | 75.00% | +0.951 px | +0.634 px | 0.00% |
| 0.55 | 4.91% | 83.87% | +0.591 px | +0.313 px | 0.00% |
| 0.60 | 2.22% | 85.71% | +0.222 px | +0.090 px | 0.00% |

Primary/replay model files are byte-identical. Feature, shortlist, probability,
output-slot, threshold-grid, point-record, and scientific-payload digests are
exact. Per protocol, top-1 audit `448--511` and top-1 original-model validation
`48--63` were not run.

### Mechanism conclusion

The failure is a precision/coverage frontier, not absence of candidate signal.
A redesigned v1 should separate candidate ranking from row-level action value and
harm risk, using out-of-fold candidate predictions for the safety model. Because
`384--447` is now observed, a new raw-record-disjoint Kubric expansion is needed
for independent confirmation. Calibration, final holdout, DAVIS, Kinetics, and
official Kinetics 1,144 remain locked.

### Notion synchronization status

- status: pending
- destination: unresolved while connector is unavailable
- synchronized at: not yet synchronized


---

## 2026-07-20 — Gate 3C2 raw-record-disjoint top-1 data renewal preregistration

### Capacity and necessity

The MOVi-E train source contains `9,749` records across `1,024` TFRecord shards.
The existing temporal-identity population uses exactly `512` raw records. No new
external dataset or pretrained model is required. A new point seed on an existing
video is explicitly not considered independent.

### Frozen renewal membership

- excluded manifest: existing Gate 3C0 512-sample manifest;
- renewed samples: `512`;
- first raw identity: `movi_e-train.tfrecord-00053-of-01024:7`;
- last raw identity: `movi_e-train.tfrecord-00107-of-01024:4`;
- selected source files: `55`;
- full selected identity digest:
  `ae7f8c4231dc81b52327020c43914def5ec6a4666bc374bd2a5539d1be5bcf37`.

Frozen partitions:

```text
checkpoint selection v1: 0--255
fit-only audit v1:       256--383
model validation v1:     384--511
```

The historical Gate 3C0 preprocessor remains byte-unchanged. A new wrapper adds
exact manifest exclusion and precomputes the full selected identity plan before
decoding. Source files containing selected samples are independently hashed.

A data-gate pass authorizes only preregistration of a two-stage v1 policy:
candidate ranking plus an out-of-fold row-level expected-value/harm-risk model.
The failed v0 single candidate-probability threshold is forbidden. Calibration,
final holdout, DAVIS, Kinetics, and official Kinetics 1,144 remain locked.

### Materialization engineering note

The first post-preregistration execution used the frozen serial decoder. It was
stopped after the first output shard because the measured throughput would make
the 512-record run unnecessarily long. No final manifest or Gate 3C2 result was
created. Membership, point seeds, partitioning, and output schema were unchanged.
A separate implementation-only fix parallelizes decoding across eight TFRecord
files, then reassembles and verifies samples in the original frozen identity
order before writing the final manifest.

### Notion synchronization status

- status: pending
- destination: unresolved while connector is unavailable
- synchronized at: not yet synchronized


---

## 2026-07-20 — Gate 3C2 raw-record-disjoint renewal result

```text
COMPLETED_PASS
AUTHORIZE_GATE3C1D_V1_TWO_STAGE_TOP1_PREREGISTRATION
```

- new videos: `512`;
- output shards: `32`;
- selected source TFRecords: `55`, all hashes exact;
- existing identities excluded: `512`;
- raw overlap: `0`;
- selected identity digest:
  `ae7f8c4231dc81b52327020c43914def5ec6a4666bc374bd2a5539d1be5bcf37`;
- renewal manifest SHA256:
  `8937a2ef925b6c97c994da7b755b91b401d2239162c1254b720567f66e028105`;
- summary payload SHA256:
  `576f3a97f9a878726dda6eca7047c38413c343882decb23b8947c1e70d6db03b`.

Renewed partition sizes are 256 checkpoint, 128 fit-only audit, and 128 model
validation videos. No renewed model metric has been read. A two-stage v1 protocol
may now be designed on the already exposed old development population, but must
be committed before any renewed feature-cache or metric access.

Calibration, final holdout, DAVIS, Kinetics, and official Kinetics 1,144 remain
locked.

### Notion synchronization status

- status: pending
- destination: unresolved while connector is unavailable
- synchronized at: not yet synchronized


---

## 2026-07-20 — Gate 3C1D v1 two-stage causal top-1 preregistration

### Status

```text
PREREGISTERED_NOT_RUN
renewed checkpoint metrics unread
```

### Old-development mechanism diagnosis

Only the already exposed four development caches were used: `4,901` failure
rows from `450` videos. A four-fold outer source-video split and three-fold
inner OOF candidate prediction compared candidate-probability argmax with
minimum predicted expected distance.

Candidate discrimination remained strong:

- candidate AUC/AP: `0.8483/0.4549`;
- expected-distance raw top-1 12 px support: `35.6662%`;
- expected-distance raw selected-candidate harmful rate: `8.0800%`.

Value and harm gates alone produced zero passing strategies for both ranking
rules. Adding an independent 12-pixel support condition created a feasible
region. The highest-coverage expected-distance design point was:

```text
support >= 0.30
value   >= 0 px
harm    <= 0.20
coverage:                 31.7282%
action precision 12 px:   65.6592%
mean error reduction:     +3.9970 px
video 95% CI:             [+3.6819,+4.3937] px
all-row harmful rate:     0.7958%
nonnegative videos:       98.0%
```

This is design evidence only and is not copied as the final checkpoint policy.

### Frozen factorization

The v1 bundle contains four fixed models:

1. candidate 12-pixel support classifier;
2. candidate expected-distance regressor used for stable top-1 ranking;
3. row-level expected-value regressor;
4. row-level harmful-action classifier.

Row models are trained only from five-fold source-video OOF candidate
predictions on the old 4,901-row development pool. The renewed 512 videos are
forbidden from model fitting.

### Renewed execution sequence

```text
checkpoint selection v1: 0--255
fit-only audit v1:       256--383
model validation v1:     384--511
```

Checkpoint selects one policy from a frozen `10 x 8 x 6 = 480` grid over support,
value, and harm thresholds. It cannot alter models, features, or ranking. A
checkpoint failure stops before audit cache creation. Audit and validation
require exact replay authorization and use the unchanged hash-pinned primary
bundle and policy.

### Replay and runtime

Checkpoint replay independently retrains all four models and compares OOF
training digests plus every renewed target prediction and scientific digest.
Joblib bytes are not used as the scientific equality criterion because two
scientifically identical sklearn fits produced different pickle bytes. Later
stages may load only the primary bundle with its fixed file SHA256.

Frozen runtime: Python `3.11.8`, NumPy `1.26.4`, PyTorch `2.2.2+cu121`,
scikit-learn `1.4.2`, joblib `1.4.2`.

Calibration, final holdout, DAVIS, Kinetics, and official Kinetics 1,144 remain
locked.

### Notion synchronization status

- status: pending
- destination: unresolved while connector is unavailable
- synchronized at: not yet synchronized


---

## 2026-07-20 — Gate 3C1D v1 two-stage causal top-1 result

```text
COMPLETED_PASS
EXACT_REPLAY_PASS_ON_CHECKPOINT_AUDIT_AND_MODEL_VALIDATION
AUTHORIZE_GATE3C1E_CAUSAL_TOP1_FUTURE_ROLLOUT_PREREGISTRATION
```

The checkpoint-selected policy is frozen at support `>=0.30`, predicted value
`>=1 px`, and predicted harm `<=0.20` after expected-distance top-1 ranking.
Raw-record-disjoint results are:

| Partition | Coverage | Precision <=12 px | Commit reduction | CI lower | All-row harm |
|---|---:|---:|---:|---:|---:|
| checkpoint 0--255 | 29.7258% | 66.6262% | +3.7312 px | +3.4295 px | 0.6854% |
| audit 256--383 | 24.8724% | 62.1701% | +3.2180 px | +2.9634 px | 0.0729% |
| model validation 384--511 | 28.0638% | 62.2739% | +3.4864 px | +3.1016 px | 0.7252% |

On final model validation, native mean commit error is `34.3669 px` and policy
error is `30.8805 px`, a `10.144%` relative reduction. Nonnegative-video
fraction is `97.6%`. Every stage has exact scientific replay.

This is a deployable causal selector result on the frozen natural-failure task,
not a complete-video TAP-Vid result. Gate 3C1E must test coordinate plus
four-level memory future rollout. Only a later official-protocol AJ/delta/OA run
may be compared directly with the CoTracker3 paper baseline.

Calibration, final holdout, DAVIS, Kinetics, and official Kinetics 1,144 remain
locked for the v1 route.

### Notion synchronization status

- status: pending
- destination: unresolved while connector is unavailable
- synchronized at: not yet synchronized


---

## 2026-07-20 — Gate 3C1E deployable causal future-rollout preregistration

### Status

```text
PREREGISTERED_NOT_RUN
future rollout metrics unread
```

Gate 3C1D v1 completed with exact replay on checkpoint, fit-only audit, and
model-validation. The unchanged selector reduces mean frame-15 commit error by
`+3.7312`, `+3.2180`, and `+3.4864` px on the three renewed partitions,
respectively. The model-validation decision authorizes only preregistration of a
deployable future-rollout audit.

Gate 3C1E freezes the Gate 3C1D primary bundle and policy:

```text
support >= 0.30
value   >= 1.0 px
harm    <= 0.20
```

Before CoTracker initialization, all `1,379` model-validation rows must reproduce
the sealed selected slot, action, and output candidate exactly. The runner then
evaluates frames `16--23` under native, coordinate-only, and coordinate plus
deterministic four-level memory actions. The primary new gates require at least
`+1.0 px` all-row future error reduction, a positive video-cluster CI, controlled
future harm, and at least `+0.5 px` memory gain beyond coordinate-only.

A passing exact replay may authorize only a separately preregistered official
TAP-Vid benchmark. It does not itself establish a paper-table AJ/OA gain.
Calibration, final holdout, DAVIS, Kinetics, and official Kinetics 1,144 remain
locked.

### Notion synchronization status

- status: pending
- destination: unresolved while connector is unavailable
- synchronized at: not yet synchronized


---

## 2026-07-20 — Gate 3C1E deployable causal future-rollout result

```text
COMPLETED_PASS
EXACT_REPLAY_PASS
AUTHORIZE_GATE3C1F_OFFICIAL_TAPVID_BENCHMARK_PREREGISTRATION
```

All `1,379` causal selector decisions exactly reproduce the sealed Gate 3C1D
model-validation replay before CoTracker initialization. The frozen policy acts
on `387` rows.

Coordinate plus deterministic four-level memory reduces visible future error on
frames `16--23` by `+3.6673 px` over all failure rows, with video-cluster 95% CI
`[+3.2492,+4.6715]`. On action rows the reduction is `+13.0150 px`; `94.06%` of
action rows improve and future harm above native by more than 4 px is `3.10%`.
All-row harmful rate is `0.87%`, and `96%` of source videos are nonnegative.

Coordinate-only improves by only `+0.0700 px`. Four-level memory contributes
`+3.5973 px` beyond coordinate-only, with CI `[+3.1859,+4.5559]`, confirming that
memory reinstatement is the dominant mechanism.

A paper-table benchmark still requires a causal full-population entry contract.
The natural-failure audit population was defined using future ground truth only
for scientific isolation and cannot be used as a runtime trigger. DAVIS,
Kinetics, final holdout, and official Kinetics 1,144 remain unread by Gate 3C1E.

### Notion synchronization status

- status: pending
- destination: unresolved while connector is unavailable
- synchronized at: not yet synchronized


---

## 2026-07-20 — Gate 3C1F0 causal full-population entry preregistration

```text
PREREGISTERED_NOT_RUN
new raw-record-disjoint population unread
```

The natural-failure population used by Gate 3C1D/E is not a runtime trigger
because its membership uses future ground truth. Gate 3C1F0 freezes a separate
130-D causal HGB entry model using only observed CoTracker trajectory,
visibility/confidence, and four native feature/support memory levels.

Frozen entry rule:

```text
entry probability >= 0.93
native joint probability <= 0.02
```

On old exposed data, five-fold source-video OOF gives `14.63%` failure recall,
`0%` clean false apply, and `100%` precision. Independent fit-validation gives
`15.00%`, `0%`, and `100%`, respectively.

A 16-video design-only complete-population pilot with the unchanged Gate 3C1D
selector gives `+0.0858` AJ point, `+0.4707` delta point, and `+0.5242` OA point.
The delta/OA paired-video confidence intervals are positive, but the AJ interval
crosses zero. Final actions improve future error by `+16.51 px` on average;
`95.65%` improve and `4.35%` are harmful by more than 4 px.

A passing exact replay authorizes only a third raw-record-disjoint Kubric
population excluding all previous 1,024 identities. DAVIS, Kinetics, final
holdout, and official Kinetics 1,144 remain locked.

### Notion synchronization status

- status: pending
- destination: unresolved while connector is unavailable
- synchronized at: not yet synchronized


---

## 2026-07-20 — Gate 3C1F0 causal full-population entry result

```text
COMPLETED_PASS
EXACT_REPLAY_PASS
AUTHORIZE_GATE3C1F0_RAW_DISJOINT_FULL_POPULATION_DATA
```

The frozen 130-D HGB entry rule is:

```text
entry probability >= 0.93
native joint probability <= 0.02
```

Five-fold source-video OOF on 376 rows gives AUC/AP `0.8514/0.8090`, `14.63%`
failure recall, `0%` clean false apply, and `100%` action precision. Independent
fit-validation on 160 rows gives AUC/AP `0.7570/0.7864`, `15.00%` failure recall,
`0%` clean false apply, and `100%` precision.

Primary and fresh-process replay exactly match every feature, label, group, OOF
prediction, validation prediction, point-record, operating-point, and scientific
payload digest. Only the primary bundle is authorized downstream.

The result authorizes a third raw-record-disjoint Kubric complete-population
confirmation after excluding all 1,024 identities used by Gate 3C0 and Gate 3C2.
DAVIS, Kinetics, final holdout, and official Kinetics 1,144 remain locked.

### Notion synchronization status

- status: pending
- destination: unresolved while connector is unavailable
- synchronized at: not yet synchronized


---

## 2026-07-20 — Gate 3C1F1 third raw-record-disjoint population preregistration

```text
PREREGISTERED_NOT_MATERIALIZED
new population metrics unread
```

The data gate excludes both prior 512-video manifests, totaling 1,024 unique raw
identities with combined digest
`6569347faf10aa3189581186b919e94bb39aaac33172552436ec30a0e8e24c89`.

The frozen third population contains exactly 128 videos:

```text
first identity: movi_e-train.tfrecord-00107-of-01024:5
last identity:  movi_e-train.tfrecord-00120-of-01024:9
selected source files: 14
selected identity digest:
85193d0aa6381c7d78442e85bf87750ce72a497995a92016001ca0cf51832f28
```

It will be written as eight 16-video shards using uniform 64-point sampling and
seed `271828`. Every selected source TFRecord must hash exactly. This gate
qualifies identity only; no model metrics may be read until Gate 3C1F2 is
separately preregistered. DAVIS, Kinetics, final holdout, and official Kinetics
1,144 remain locked.

### Notion synchronization status

- status: pending
- destination: unresolved while connector is unavailable
- synchronized at: not yet synchronized


---

## 2026-07-20 — Gate 3C1F1 raw-record-disjoint full-population data result

### Status

```text
COMPLETED_PASS
AUTHORIZE_GATE3C1F2_FULL_POPULATION_CONFIRMATION_PREREGISTRATION
```

The third Kubric population contains exactly 128 videos in eight shards. It
excludes all 1,024 raw identities used by Gate 3C0 and Gate 3C2. The new manifest
SHA256 is `060e0f9de3aeb945932ea15b9a0a7ab70aa88c1d703fa313c9e0db21ad2f3c01`;
the selected identity digest is
`85193d0aa6381c7d78442e85bf87750ce72a497995a92016001ca0cf51832f28`.
All 14 source TFRecords independently rehash exactly and raw overlap is zero.
No tracking metric from this population has been read. DAVIS, Kinetics, final
holdout, and official Kinetics 1,144 remain locked.

### Notion synchronization status

- status: pending
- destination: unresolved while connector is unavailable
- synchronized at: not yet synchronized


---

## 2026-07-20 — Gate 3C1F2 complete-population confirmation preregistration

### Status

```text
PREREGISTERED_NOT_RUN
third-population tracking metrics unread
```

Gate 3C1F2 freezes the complete causal pipeline before reading any metric from
the third raw-record-disjoint 128-video Kubric population. The entry rule is
probability `>=0.93` and native joint probability `<=0.02`. Entry-positive rows
then use the frozen Gate 3C1D expected-distance top-1 selector and Gate 3C1E
coordinate plus four-level memory writeback.

The confirmation requires at least `+0.05` AJ points, `+0.25` delta_avg points,
and `+0.25` OA points, with paired-video 95% CI lower bounds strictly above
zero for all three. It also requires at least 64 evaluable actions, at least
`+8 px` mean future error reduction, at least 85% positive actions, at most 5%
harmful actions, and fresh-process exact replay. A pass authorizes only a later
separately preregistered official TAP-Vid evaluation.

DAVIS, Kinetics, final holdout, and official Kinetics 1,144 remain locked.

### Notion synchronization status

- status: pending
- destination: unresolved while connector is unavailable
- synchronized at: not yet synchronized


---

## 2026-07-20 — Gate 3C1F2 complete-population confirmation result

### Status

```text
COMPLETED_FAIL_WITH_EXACT_REPLAY
STOP_BEFORE_OFFICIAL_TAPVID
```

The frozen causal pipeline completed on all 128 third-population Kubric videos.
Complete-video equal-video gains are `+0.0014` AJ points, `+0.0937` delta_avg
points, and `+0.1591` OA points. AJ CI is `[-0.0340,+0.0435]` points; delta_avg
and OA CIs are strictly positive but below their preregistered magnitude gates.

There are 537 causal entry triggers, 89 final top-1 actions, 44 action-support
videos, and 51 actions with visible future GT. Those 51 actions reduce future
error by `+15.2680 px` on average; `98.04%` improve and `0%` are harmful by more
than 4 px. The remaining bottleneck is complete-population coverage/AJ leverage,
not action quality. Fresh-process replay reproduces every per-video and aggregate
scientific digest exactly.

No official TAP-Vid evaluation is authorized. Any redesigned entry/action policy
requires a newly preregistered raw-record-disjoint confirmation population.
DAVIS, Kinetics, final holdout, and official Kinetics 1,144 remain locked.

### Notion synchronization status

- status: pending
- destination: unresolved while connector is unavailable
- synchronized at: not yet synchronized


---

## 2026-07-20 — Gate 3C1F2 failure diagnostic preregistration

### Status

```text
PREREGISTERED_EXPOSED_DIAGNOSTIC_NOT_RUN
no new confirmation or external data
```

The diagnostic reruns only the 44 Gate 3C1F2 videos with sealed actions. It must
first reproduce every entry, candidate, shortlist, action, coordinate, and
visibility digest from the exact-replay result. It then evaluates actual modified
tracks, modified coordinates with native visibility, native coordinates with
modified visibility, and two GT-visibility localization oracles.

Per-action frames 15--23 record coordinate threshold hits, visibility/confidence,
false-negative recovery, new false negatives, removed false positives, new false
positives, and official Jaccard numerator/denominator contributions. The purpose
is to separate coverage dilution from coordinate--visibility coupling. No sweep,
retraining, or post-hoc policy selection is included.

DAVIS, Kinetics, final holdout, and official Kinetics 1,144 remain locked.

### Notion synchronization status

- status: pending
- destination: unresolved while connector is unavailable
- synchronized at: not yet synchronized


---

## 2026-07-20 — Gate 3C1F2 coordinate--visibility failure diagnostic result

### Status

```text
COMPLETED_EXPOSED_DIAGNOSTIC
SEALED_GATE3C1F2_PIPELINE_REPRODUCED_EXACTLY
```

All 44 action videos and 89 actions reproduce the committed Gate 3C1F2 entry,
candidate, shortlist, action, coordinate, and visibility digests. Under GT
visibility, modified coordinates improve equal-video AJ by `+0.2921` points with
a positive paired CI. Keeping native visibility suppresses the coordinate gain,
while applying modified visibility to native coordinates reduces AJ by `-0.2195`
points with a strictly negative CI.

Visibility transitions identify the structural failure: failure actions recover
220 GT-visible false negatives and create only two new occluded false positives;
ambiguous actions recover 42 and create eight; `other` actions recover only six
but create 176 new occluded false positives. The next redesign should preserve
the frozen coordinate/memory action and learn a causal post-writeback visibility
decision. Any confirmation requires another raw-record-disjoint population.

DAVIS, Kinetics, final holdout, and official Kinetics 1,144 remain locked.

### Notion synchronization status

- status: pending
- destination: unresolved while connector is unavailable
- synchronized at: not yet synchronized


---

## 2026-07-20 — Gate 3C1G0 richer visibility cache preregistration

### Status

```text
PREREGISTERED_EXPOSED_CACHE_NOT_BUILT
formal cache output unread
```

A 19-D grouped source-video OOF probe is rejected as the primary redesign. The
best regularized logistic model reaches GT-visible AUC `0.6233`, AP `0.5662`,
and pooled affected-frame AJ `0.09816` versus the frozen modified-visibility
baseline `0.09245`, while still producing `68.94%` occluded false positives.

Gate 3C1G0 freezes a 66-D causal feature schema over the 44 exposed action videos,
89 sealed actions, and frames 15--23. It adds DINO query/commit identity, four
levels of CoTracker commit/current/previous identity consistency, trajectory
dynamics, native/modified visibility state, and sealed entry/top-1 evidence. GT,
coordinate errors, categories, and future-after-prediction frames are prohibited
from model features. Complete trajectories and labels are stored only for later
nested video-OOF evaluation.

A source-0 smoke build passes with three actions, 27 frame rows, all 66 channels,
exact sealed digests, and tensor reload. The formal 44-video cache remains unbuilt.
No new raw population or external data is read.

### Notion synchronization status

- status: pending
- destination: unresolved while connector is unavailable
- synchronized at: not yet synchronized
