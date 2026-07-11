# V9-A5.2 Independent Model-State Branching Feasibility Design

Date: 2026-07-11

## 1. Motivation

V9-A5.0 showed large temporal headroom when the previous state is correct.

V9-A5.1a and V9-A5.1c used one shared TrackOn2 query/memory stream. Their external beams changed coordinate/descriptor histories, but every hypothesis received future candidates from the same official model state.

V9-A5.1c also failed the required same-capacity comparator:

```text
frame-local score-top4 oracle mean: 8.5363 px
temporal B4 beam oracle mean:       8.7879 px
temporal minus frame-local:        +0.2516 px
95% CI:                            [+0.1280,+0.4307]
```

Therefore the remaining untested mechanism is not another selector over the same candidate pool. It is whether an alternative candidate-conditioned query feature, written into an independent memory state, changes future correlation maps and produces useful future outputs unavailable to the shared-state capacity-matched control.

No training and no DAVIS data are allowed.

## 2. Repository isolation

```text
worktree: /gemini/code/FSPT_v9a52_clean
branch: v9a52-independent-state-branching-20260711
base HEAD: 53f475a283c70c49b6bc2167458b520d11d6b3d9
```

TrackOn2 Python code/config must be imported from this worktree. Immutable checkpoint, DINOv3 weights, PointOdyssey annotations and frames must be hash verified.

The committed V9-A5.1b and V9-A5.1c scripts/JSON/NPZ artifacts are frozen replay inputs.

## 3. Structural correction: a branch is the full active tracker state

TrackOn2 applies `query_attention` across all active queries before memory attention.

The experiment uses:

```text
32 evaluated PointOdyssey queries
400 support-grid queries
N = 432 active queries
```

Consequences:

```text
A/B hypotheses must not be concatenated into one N dimension.
Otherwise query_attention lets the hypotheses communicate.

The target query must not be run alone.
Otherwise branch A loses the original 431-query context and cannot match official TrackOn2.
```

Each branch must clone and independently roll the complete active state:

```text
q_init:          (432, 256)
point_memory:    (432, 24, 256)
temporal_mask:   (432, 24)
```

The experiments from V9-A5.1b/c explicitly set `M_i = M`, so the canonical inference memory is 24, not the config-file extension value 72. V9-A5.2 must preserve `M=24` for exact replay parity.

## 4. Event manifest

Frozen event manifest:

```text
docs/v9a52_independent_state_event_manifest_2026-07-11.json
```

Builder:

```text
scripts/v9a52_build_event_manifest.py
```

Source:

```text
V9-A5.1c rows NPZ SHA256:
05b450fb2031ce1d762ed7436bcef34a47dfaf37c10d1c33bd8fd74c1d0381a0
```

The builder reads only:

```text
clip_id
query_idx
frame_tau
risk
```

No GT visibility, position or error field is read for event selection.

Event definition:

```text
first native-risk frame per query
1 <= frame_tau <= 87
```

Within each clip, select the eight eligible events with the lowest stable SHA256 event key.

```text
9 clips
8 events per clip
72 total events
horizons: 1 / 4 / 8
216 event-horizon rows before visibility filtering
```

Every clip has at least 15 eligible events, so no GT-dependent fallback or result-dependent replacement is needed.

## 5. Split timing

The split occurs at the memory write after processing the event frame.

At event frame `t`:

```text
1. Run the normal official forward from the shared pre-event state.
2. Obtain official q_new_A for all 432 active queries.
3. Generate the frozen K64 candidates for the target query.
4. Select the first maximum frozen-score candidate in K64 order.
5. Build its singleton candidate-conditioned q2_B.
6. Clone the complete pre-update q_init / point_memory / temporal_mask.
7. Branch A writes q_new_A for all rows.
8. Branch B writes q_new_A for all non-target rows and q2_B for the target row.
9. Both writes use the original unconditional all-true write mask.
10. The official stream writes q_new_A normally and continues.
```

This timing prevents:

```text
using future information
writing the event frame twice
one-frame offset errors
mixing pre-write and post-write state definitions
```

## 6. Branch definitions

### 6.1 Branch A: official-state clone

```text
q_features_A = official q_new for all 432 rows
```

Branch A must reproduce the uninterrupted official rollout exactly.

### 6.2 Branch B: score-top1 singleton-state alternative

```text
q_features_B = official q_new for all 432 rows
q_features_B[target] = candidate-conditioned q2 of the frozen K64 score-top1 candidate
```

The alternative is score-top1, not “the highest candidate different from branch A.” Branch A is the all-candidate fused official state, not an individual candidate. Score-top1 gives a deterministic, causal and capacity-matched alternative without post-hoc candidate choice.

Ties use the first maximum in the frozen fused-K64 order.

## 7. Independent rollout

After the split, Branch A and Branch B roll forward separately for up to eight frames.

At every future frame:

```text
same frozen frame features are reused
A receives only A q_init/memory/mask
B receives only B q_init/memory/mask
A and B produce their own p/v/u/q_new/C1/C2
A updates only A memory
B updates only B memory
```

The complete 432-query state is updated in each branch. A target-row change may propagate to other evaluated/support queries through later query-attention layers; this is part of the real model-state mechanism and must be measured.

Do not reset non-target rows to official state after the split.

## 8. Capacity-matched controls

### 8.1 Official single-state output

The uninterrupted official final coordinate at the horizon frame.

### 8.2 Shared-state capacity-2 control

At the same future frame, using the uninterrupted official state:

```text
output 1: official final coordinate
output 2: frozen K64 score-top1 candidate-conditioned refined coordinate
```

GT-only shared-state B2 oracle:

```text
min(error(official final), error(shared score-top1 refined))
```

### 8.3 Independent-state capacity-2 output

```text
output 1: Branch A final coordinate
output 2: Branch B final coordinate
```

GT-only independent-state B2 oracle:

```text
min(error(Branch A final), error(Branch B final))
```

The primary comparison is:

```text
independent-state B2 oracle
versus
shared-state capacity-2 oracle
```

Both sides contain two generated outputs. GT is used only after all outputs are generated.

### 8.4 Additional diagnostics

```text
Branch A deterministic final
Branch B deterministic final
shared score-top1 refined final
shared full-K64 refined oracle
Branch B K16/K64 candidate oracles
```

These diagnostics cannot replace the primary capacity-2 comparison.

## 9. Horizon rows and evaluation power

The event manifest is frozen before this power audit.

Visible-valid evaluation counts from the frozen events are:

```text
horizon 1: 28 visible rows  (ani 10 / animal3 10 / r4_new_f 8)
horizon 4: 36 visible rows  (ani 13 / animal3 12 / r4_new_f 11)
horizon 8: 44 visible rows  (ani 15 / animal3 17 / r4_new_f 12)
```

Every clip has at least one visible row at horizons 1 and 4 and at least two visible rows at horizon 8. Clip-block means are therefore defined for all nine clips.

Re-entry counts are sparse:

```text
horizon-8 first re-entry: 9 visible rows
horizon-8 early8:        29 visible rows
```

Therefore first-reentry is descriptive only. It must not be a per-sequence hard gate. Early8 may be used as a pooled secondary gate.

## 10. Mandatory state-divergence diagnostics

At event split:

```text
initial q2_B versus q_new_A target cosine distance / L2
A/B target memory max_abs after write
A/B full-memory max_abs after write
physical storage pointers for q_init/memory/mask
```

At horizons 1/4/8:

```text
target q_new cosine distance / L2
full q_new mean/max divergence
non-target evaluated-query q_new divergence
support-grid q_new divergence
point-memory mean/max divergence
C1 map cosine/L2/max_abs divergence
C2 map divergence
final coordinate divergence
visibility/uncertainty divergence
```

Candidate-set diagnostics for the target query:

```text
C1 top16 exact-set equality
C1 top16 overlap / Jaccard
C1 top16 Hausdorff distance
C1 top64 overlap / Jaccard
shared versus Branch B frozen-score top candidates
```

Useful novelty diagnostics, evaluated with GT only after generation:

```text
Branch B final <= 4 px while shared B2 oracle > 4 px
Branch B final improves shared B2 oracle by > 1 px
Branch B K16/K64 contains a <=4 px candidate absent from shared-state capacity2
```

## 11. Integrity gates

Mandatory before scientific metrics:

```text
all input hashes pass
exact event-manifest hash passes
online row keys and native-risk flags match V9-A5.1c
N = 432 and M = 24
TrackOn code imported from the clean V9-A5.2 worktree
Branch A p/v/u/q_new/C1/C2 parity with uninterrupted official <= 1e-6
Branch A memory/mask parity after every update <= 1e-6
A/B q_init, memory and mask storages are physically independent
mutating Branch B state cannot change Branch A state
A->B versus B->A call order changes outputs by <= 1e-6 in smoke
all state/correlation/candidate/output arrays finite
branch initialization and rollout functions receive no GT/error input
all 72 event keys and all requested horizons persisted exactly once
```

A/B must be invoked as separate full-state calls. Concatenated hypotheses or target-only rollouts are invalid.

## 12. Statistical protocol

Evaluation unit:

```text
event-horizon target row
```

Primary horizon:

```text
horizon 8
```

Secondary summaries:

```text
horizon 1
horizon 4
pooled horizons 1/4/8
pooled early8 rows
```

Use 100,000-resample clip-block paired bootstrap:

```text
cluster: clip_id
seed: 20260716
```

Report per event, clip, sequence and horizon.

## 13. Primary independent-state gate

At horizon 8, the independent-state B2 oracle must beat the shared-state capacity-2 oracle on all of:

```text
mean error lower on ani
mean error lower on animal3
mean error lower on r4_new_f
safe16 not lower on every sequence
global better rows > worse rows
nine-clip bootstrap 95% CI upper bound < 0
clip better count > clip worse count
```

Pooled horizons 1/4/8 must additionally satisfy:

```text
mean difference < 0
better rows > worse rows
clip-block CI upper bound < 0
```

Early8 is a secondary gate only when at least 20 visible rows exist:

```text
independent B2 mean <= shared B2 mean
```

First-reentry is descriptive because the frozen manifest yields only nine horizon-8 first-reentry rows.

## 14. Mechanistic non-degeneracy gate

A statistical improvement is not enough if both states remain numerically identical.

Require:

```text
A/B target q_new L2 > 1e-4 on at least 10% of visible event-horizon rows
C1 top16 exact-set change on at least 10% of visible event-horizon rows
at least one useful-novel Branch B row in every sequence
at least 10 useful-novel rows globally across pooled horizons
```

Useful-novel means:

```text
Branch B final improves the shared-state B2 oracle by > 1 px
```

These thresholds are fixed before implementation results are observed.

## 15. Interpretation

```text
Primary + non-degeneracy pass:
  true independent model state produces future value beyond a shared-state
  same-capacity readout. Proceed to a bounded full-stream B2 state beam.

Primary pass but non-degeneracy fail:
  apparent gain is not supported by meaningful future-state divergence.
  Treat as numerical/selection ambiguity and stop.

Primary fail:
  close the temporal multi-state route. Do not train a branch selector and do
  not read DAVIS.
```

A pass does not authorize training or DAVIS. It only authorizes a later full-stream independent-state beam audit.

## 16. Smoke protocol

Before the 72-event run, use one frozen event from `ani:0` and horizons 1/4/8.

Smoke must prove:

```text
Branch A exact parity for all 432 rows
A/B physical storage independence
mutation isolation
A->B / B->A order invariance
correct split timing
online risk/event-key parity
shared-state B2 formula
independent-state B2 formula
candidate-set divergence metrics finite
no GT enters branch generation
```

No smoke metric is a scientific result.

## 17. Compute budget

Canonical state dimensions:

```text
N = 432
M = 24
D = 256
```

Approximate state storage:

```text
point memory per branch: 10.125 MiB
q_init per branch:        0.422 MiB
mask per branch:          0.010 MiB
two branches per event:  21.114 MiB
worst eight simultaneous events: about 168.9 MiB
```

Expected forward calls:

```text
official calls: 9 * 96 = 864
branch calls:   72 * 2 * 8 = 1152
total:         2016 full-state track_frame calls
```

Frame features must be extracted once per frame and reused by the official and active branch states.

## 18. Immediate execution order

```text
1. Commit this design and frozen 72-event manifest before implementation.
2. Implement a full-state diagnostic forward returning p/v/u/q_new/q_pre/C1/C2.
3. Implement score-top1 singleton q2 state construction with V9-A5.1b parity.
4. Implement full-state clone, split-write and active-event rollout.
5. Run the one-event smoke and independently inspect all parity/isolation outputs.
6. Run the 72-event synthetic audit only if every smoke integrity gate passes.
7. Do not train, tune thresholds or read DAVIS.
```
