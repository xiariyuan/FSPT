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
