# MMP-Tracker Experiment Matrix

This file turns the paper evidence chain into executable experiment stages.

## Stage 0. Baseline locking

### E0-1 `baseline_cotracker3_eval256`
- goal: lock the external paper baseline
- must report: `AJ`, `OA`, `<1px`, `<2px`, `<4px`, `avg_error_px`
- stop condition: metrics are stable and protocol is frozen

### E0-2 `baseline_localmatch_smoke`
- goal: verify the new matching-first code path trains and evaluates end-to-end
- model: `Ours-Local`
- data: tiny train subset + DAVIS eval256
- success criterion: loss decreases and predictions stay numerically stable

## Stage 1. Engineering baseline

### E1-1 `localmatch_davis_dev`
- goal: make `Ours-Local` a real internal baseline
- compare against: legacy residual-refiner route
- success criterion: better `<1px` or lower `avg_error_px`
- stop criterion: if `Ours-Local` cannot beat legacy residual refinement on precision, local matcher design is wrong

### E1-2 `localmatch_kinetics_train`
- goal: move from smoke test to a real training split
- compare against: E1-1
- success criterion: stable training on full train shards

## Stage 2. Global re-localization proof

### E2-1 `localglobal_visiblebank`
- goal: prove explicit global re-localization helps long occlusion
- model: `Ours-LocalGlobal`
- compare against: `Ours-Local`
- primary metrics: `AJ_longocc20_delta`, `AJ_longocc30_delta`
- success criterion: positive and repeatable long-occ gains
- failure interpretation: if gains are absent, memory or retrieval design is wrong

### E2-2 `localglobal_memory_ablation`
- vary: memory size, visible-only update, query-only memory, bank stride
- objective: identify which memory policy actually matters

## Stage 3. Posterior fusion proof

### E3-1 `posterior_fusion_base`
- goal: prove fusion reduces harmful wrong corrections
- model: `Ours-Posterior`
- compare against: `Ours-LocalGlobal`
- primary metrics: `AJ_delta`, `avg_error_px_delta`
- success criterion: better global safety without losing long-occ gains

### E3-2 `posterior_candidate_count`
- vary: number of global hypotheses `K`
- objective: show whether multi-hypothesis is truly needed

### E3-3 `posterior_without_prior`
- remove prior branch
- objective: prove prior-observation fusion matters

## Stage 4. Training recipe proof

### E4-1 `synthetic_only`
- goal: establish lower-bound training recipe

### E4-2 `synthetic_plus_real_pseudo`
- goal: test whether real-video pseudo labels are the main performance unlock
- compare against: `synthetic_only`
- expected: biggest gain among recipe changes

### E4-3 `two_view_warmstart`
- goal: test whether matching pretraining improves re-localization

## Stage 5. Frequency module proof

### E5-1 `posterior_plus_freq_reweight`
- goal: test frequency as evidence modulation only
- compare against: `Ours-Posterior`
- expected: small but targeted gain on fast motion / blur / repetitive texture

### E5-2 `freq_subset_analysis`
- goal: determine whether frequency is a main contribution or an auxiliary one
- stop criterion: if gains are inconsistent, keep frequency in appendix only

## Stage 6. Paper-ready comparisons

### E6-1 `paper_main_table`
- compare: `CoTracker3` vs `Ours-Full`
- protocol: fixed and frozen

### E6-2 `paper_ablation_table`
- rows: `Ours-Local`, `Ours-LocalGlobal`, `Ours-Posterior`, `Ours-Full`

### E6-3 `paper_hard_cases`
- subsets: long occlusion, fast motion / blur, repetitive texture

## Hard stop rules

- If `Ours-LocalGlobal` does not improve long-occ metrics, do not add posterior or frequency yet.
- If `Ours-Posterior` does not improve safety metrics, do not add frequency yet.
- If recipe gains are larger than architecture gains, prioritize the recipe section in the paper story.
- If frequency only improves a niche subset, present it as an auxiliary targeted module rather than the paper core.

