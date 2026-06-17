# MMP-Tracker Training Stages

## Stage A: local matching smoke test

- objective: establish a trainable matching-first tracker
- model variant: `Ours-Local`
- data: TAP-Vid Kinetics train subset + DAVIS eval256
- stop rule: if local precision is unstable, fix local matcher before adding memory

## Stage B: engineering baseline

- objective: scale `Ours-Local` to the real train split
- compare against: legacy residual-refiner route

## Stage C: visible-memory global re-localization

- objective: add long-occlusion recovery
- model variant: `Ours-LocalGlobal`
- key metrics: `AJ_longocc20_delta`, `AJ_longocc30_delta`

## Stage D: posterior fusion

- objective: prevent wrong global corrections from harming overall AJ
- model variant: `Ours-Posterior`
- key metrics: `AJ_delta`, `avg_error_px_delta`

## Stage E: matching warm start

- objective: strengthen local and global correspondence quality
- pretraining: MegaDepth / ScanNet style two-view matching

## Stage F: real-video pseudo-label training

- objective: reduce synthetic-to-real gap
- data: real videos with teacher-generated tracks and visibility

## Stage G: optional frequency augmentation

- objective: improve evidence quality in blur / fast motion / repetitive texture
- important rule: frequency remains an evidence modulation module, not the core tracker

