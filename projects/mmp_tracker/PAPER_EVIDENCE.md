# Paper Evidence Plan for MMP-Tracker

## Target claim

The paper should not start from a vague claim like "frequency helps tracking".
The evidence chain should support a sharper statement:

> Explicit local matching plus memory-based global re-localization plus posterior fusion
> is a better decomposition for long-occlusion point tracking than residual coordinate refinement.

## External baselines

- `CoTracker3 offline`
- `TAPIR` or `BootsTAP` when protocol-aligned results are available
- optional `LocoTrack` when protocol and code path are fully aligned

The paper-facing main baseline is `CoTracker3`.

## Internal baselines

These are mandatory. Without them the paper has no causal evidence.

1. `Ours-Local`
   - local matching only
   - no global re-localization
   - no posterior fusion
2. `Ours-LocalGlobal`
   - adds visible-memory global re-localization
   - still chooses one observation path naively
3. `Ours-Posterior`
   - adds posterior fusion over prior, local, and global hypotheses
4. `Ours-Full`
   - adds optional frequency-guided evidence reweighting

## Core hypotheses

### H1. Local matching is a better base than coordinate residual refinement

Evidence:

- compare `Ours-Local` vs legacy residual-refiner route
- show improvements in `<1px`, `<2px`, `avg_error_px`
- visualize local heatmaps and subpixel corrections

### H2. Long-occlusion recovery needs explicit global re-localization

Evidence:

- compare `Ours-Local` vs `Ours-LocalGlobal`
- main metric: `AJ_longocc20_delta`, `AJ_longocc30_delta`
- case studies on reappearance frames after long invisibility

### H3. Multi-source posterior fusion reduces harmful wrong corrections

Evidence:

- compare `Ours-LocalGlobal` vs `Ours-Posterior`
- show overall `AJ_delta` and `avg_error_px_delta`
- report failure-rate of catastrophic re-localization jumps

### H4. Frequency-guided evidence reweighting helps only after matching is strong

Evidence:

- compare `Ours-Posterior` vs `Ours-Full`
- apply frequency only as evidence modulation, not as the core tracker
- focus on fast motion, blur, and repetitive texture subsets

## Main metrics

- `AJ`
- `AJ_delta` against `CoTracker3`
- `<1px`, `<2px`, `<4px`, `<8px`
- `average_pts_within_thresh`
- `avg_error_px`
- `OA`
- `AJ_longocc10_delta`, `AJ_longocc20_delta`, `AJ_longocc30_delta`

## Dataset ladder

### Phase A. Engineering validation

- TAP-Vid DAVIS eval256
- train on a controlled subset for rapid iteration only

### Phase B. Real training recipe

- TAP-Vid Kinetics train shards
- real-video pseudo labels if available

### Phase C. Paper-level evaluation

- DAVIS full protocol
- Kinetics protocol when aligned
- long-occlusion subset analysis

## Required tables

### Table 1. External comparison

- CoTracker3
- Ours-Full
- optional other official baselines

### Table 2. Internal ablation ladder

- Ours-Local
- Ours-LocalGlobal
- Ours-Posterior
- Ours-Full

### Table 3. Training recipe ablations

- synthetic only
- synthetic + real pseudo labels
- with and without two-view matching warm start

### Table 4. Hard-case subsets

- long occlusion
- fast motion / blur
- repetitive texture / low texture

## Required figures

- architecture figure with local / global / posterior branches
- failure taxonomy of CoTracker3 on long occlusion
- qualitative reappearance sequences
- posterior hypothesis weight visualization
- optional frequency evidence visualization

## Stop criteria

If `Ours-LocalGlobal` does not clearly beat `Ours-Local` on long-occ metrics,
the global branch design is wrong and should be revised before adding more modules.

If `Ours-Posterior` does not improve `AJ_delta` over `Ours-LocalGlobal`,
posterior fusion is not justified and should be simplified or redesigned.

If `Ours-Full` does not improve over `Ours-Posterior`, frequency should be kept as
an optional appendix experiment rather than a main contribution.

