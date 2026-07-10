# V9-A5.1 Full-Stream Deterministic Multi-Hypothesis / Beam Reachability Audit Design

Date: 2026-07-10

## 1. Motivation

V9-A5.0 sampled-causal temporal-state audit established:

```text
No sampled self-state policy passes the three-sequence / safe16 / clip-CI gate.
Past-oracle motion+latent reduces mean C1 error from 12.3112 px to 6.0285 px.
Past-GT motion reduces it to 5.3269 px.
```

This is `ORACLE_STATE_HEADROOM_ONLY`: temporal evidence is useful when prior state is correct, while single-state error propagation is the blocker.

V9-A5.1 asks whether a deterministic beam can retain useful hypotheses under the real full stream, without using GT for state updates.

## 2. Repository isolation

Run only in:

```text
/gemini/code/FSPT_v9a51_clean
branch: v9a51-fullstream-beam-20260710
base HEAD: 7c24367cf8aeb1467946d2ced3b294299032538e
```

TrackOn2 Python code/config are loaded from this clean worktree. Immutable checkpoint, DINOv3 weights, PointOdyssey annotations/RGB frames, and frozen pools are read from `/gemini/code/FSPT` after SHA256 verification.

## 3. Dataset and stream

```text
PointOdyssey sequences: ani / animal3 / r4_new_f
clip starts per sequence: 0 / 256 / 512
clip length: 96 frames
queries: 32 deterministic query identities per clip, initialized at frame 0
full stream rows: 9 * 96 * 32 = 27,648 query-frame observations
```

The exact stable-seed query selection and TrackOn2 repo-native memory update path from V9-A4.3 are reused.

Unlike V9-A5.0, beam state is updated on every frame, including GT-occluded/invalid frames.

## 4. Candidate observations

At every frame and query:

```text
candidate coordinates: TrackOn2 C1 top16
teacher emission: reranking_head s_topk
candidate descriptor: normalized 512D pre-fusion latent captured at fusion_layer input
```

All candidates come from the frozen TrackOn2 checkpoint. No candidate generation, feature, score, or threshold is trained or modified.

## 5. Beam path state

Each path stores:

```text
cumulative ordinal cost
current candidate index / coordinate / latent
previous candidate coordinate and time
second-previous candidate coordinate and time
```

Motion prediction:

```text
one state: predicted coordinate = last coordinate
two states: constant velocity extrapolation to current frame
```

Current candidates receive:

```text
teacher rank: descending ordinal s_topk rank
motion rank: ascending distance to the path's predicted coordinate
latent rank: descending cosine similarity to the path's previous latent
```

## 6. Beam expansion and pruning

For every parent path, expand to all 16 current candidates.

Predeclared step costs:

```text
teacher+motion: teacher_rank + motion_rank
teacher+latent: teacher_rank + latent_rank
teacher+motion+latent: teacher_rank + motion_rank + latent_rank
```

Path cost:

```text
new cumulative cost = parent cumulative cost + current step cost
```

Initialization at frame 0:

```text
beam is seeded by teacher-ranked candidates
initial cumulative cost = teacher rank
```

Pruning:

```text
sort by cumulative cost, current teacher rank, current candidate index, parent slot
retain the best path for each distinct current candidate index
keep the first B distinct candidates
```

This prevents the beam from wasting capacity on duplicate current hypotheses while preserving the best history for each retained candidate.

## 7. Predeclared policies

Top1 policies:

```text
teacher_top1
beam1_motion_latent_top1
beam4_motion_top1
beam4_latent_top1
beam4_motion_latent_top1   # primary deterministic policy
beam8_motion_latent_top1
```

GT-only readouts that never influence beam state:

```text
teacher_top4_oracle
teacher_top8_oracle
beam4_motion_oracle
beam4_latent_oracle
beam4_motion_latent_oracle   # primary reachability diagnostic
beam8_motion_latent_oracle
frame_top16_oracle
```

`beam*_oracle` selects the minimum-error candidate only from the already retained deterministic beam at the current frame. GT does not change current or future beam membership.

## 8. Evaluation subsets

Primary evaluation uses GT-visible and valid rows.

Subsets:

```text
all_visible
teacher_hard: teacher error > 4 px
teacher_easy: teacher error <= 4 px
reentry_first: first visible-valid frame after >=1 invisible/invalid frame
reentry_early4: first four visible-valid frames after each re-entry
reentry_early8: first eight visible-valid frames after each re-entry
reentry_after_occ4: reentry_first after >=4 consecutive invisible/invalid frames
```

Report globally, per sequence, and per clip.

## 9. Metrics

For every top1/readout policy:

```text
mean / median error
safe4 / safe8 / safe16
frame-top16 oracle regret
exact-best and within-1px-best rate
better / worse / equal versus the corresponding baseline
```

Beam-set reachability:

```text
beam recall within 1 / 2 / 4 / 8 px
mean / median minimum beam error
unique beam size
```

Uncertainty:

```text
query-track paired bootstrap: 100,000 resamples, seed 20260710
clip-block paired bootstrap: 100,000 resamples, seed 20260711
```

The conservative 9-clip interval is used for gates.

## 10. Integrity gates

Mandatory:

```text
all input hashes pass
clean branch/HEAD and tracked status pass
official TrackOn2 p/v/q parity max_abs <= 1e-6
all 27,648 query-frame rows processed exactly once
candidate coordinates/scores/latents finite
beam candidate indices unique and in [0,15]
no GT value is passed into beam update functions
saved NPZ independently reproduces reported metrics
```

On the 4,878 canonical sampled-pool rows:

```text
C1 candidate set Hausdorff max <= 1e-6
teacher top1 match rate = 1.0
teacher score max_abs <= 0.005
```

## 11. Primary deterministic gate

`beam4_motion_latent_top1` passes only if all hold:

```text
mean visible error lower than teacher_top1 on ani / animal3 / r4_new_f
safe16 not lower on each sequence
global visible better > worse
visible clip-block mean-difference CI upper bound < 0
reentry_early8 mean error lower than teacher_top1
reentry_early8 safe16 not lower
```

If this passes, proceed to synthetic robustness packaging and only then a DAVIS diagnostic.

## 12. Beam reachability gate

If deterministic top1 fails, `beam4_motion_latent_oracle` justifies a learned beam readout only if all hold relative to `teacher_top4_oracle`:

```text
mean visible error improves by >=0.10 px globally
mean visible error is not worse on any sequence
safe16 is not lower on any sequence
global visible better > worse
visible clip-block mean-difference CI upper bound < 0
reentry_early8 mean error improves by >=0.25 px
reentry_early8 safe16 is not lower
```

If this passes, temporal beam membership adds reachability beyond the teacher's raw top4 and the next step is sequence-heldout learned beam scoring/readout.

If neither gate passes, stop the temporal multi-hypothesis branch and return to candidate-recall/correlation-map analysis.

## 13. Outputs

```text
scripts/v9a51_fullstream_beam_reachability.py
docs/v9a51_input_manifest_2026-07-10.json
docs/v9a51_pointodyssey_frame_manifest_2026-07-10.json
outputs/paper_discovery_2026-07-05/v9a51_fullstream_beam/v9a51_fullstream_beam_reachability.json
outputs/paper_discovery_2026-07-05/v9a51_fullstream_beam/v9a51_fullstream_beam_reachability.npz
docs/v9a51_fullstream_beam_reachability_result_2026-07-10.md
```
