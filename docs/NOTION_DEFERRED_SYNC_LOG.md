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
