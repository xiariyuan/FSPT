# BeliefCal-MMP MVP-1 preregistration

Status: frozen; Amendment A1 appended on 2026-07-14 before any new formal experiment

Date: 2026-07-13

Base tag: `v9a-route-closure-20260711`

Base commit: `4fd2a1b7891912e3cd34aec8d8e3f2d24a73783e`

Execution branch: `beliefcal-mmp-mvp1-20260713`

## 1. Research question

Can an uncertainty-only head, trained on frozen MMP point predictions, produce better calibrated 2D predictive uncertainty under occlusion and re-entry than post-hoc calibration baselines, without changing the tracker trajectory, visibility decision, candidate routing, memory update, or Average Jaccard score?

MVP-1 is an evaluation and calibration experiment. It is not yet a learned coast model, a multi-hypothesis tracker, or an identity-memory system.

## 2. Frozen tracker contract

The MMP tracker produces the predictive mean `mu = (y, x)` in normalized image coordinates. For strict MVP-1:

- all MMP parameters are frozen;
- `mu` is detached before it is passed to the uncertainty head;
- predicted covariance cannot affect point coordinates, visibility, candidate selection, commit decisions, template updates, or memory writes;
- all compared calibration methods consume the same cached MMP predictions and diagnostic features;
- Average Jaccard, position accuracy, and occlusion accuracy must therefore be numerically unchanged apart from serialization tolerance.

Any implementation that lets the uncertainty loss update the tracker is outside MVP-1 and requires a new preregistration.

## 3. Distribution parameterization

The first learned model is a single diagonal 2D Gaussian in evaluation-pixel coordinates:

`p(z | x) = N(mu_px, diag(sigma_y^2, sigma_x^2))`.

The head predicts `(log_var_y, log_var_x)`. Values are clamped before exponentiation:

- minimum standard deviation: `0.25 px`;
- maximum standard deviation: `256 px`;
- default numerical epsilon: `1e-6`.

Full covariance, mixture components, and correlation coefficients are excluded from MVP-1.

## 4. Input features

Only detached diagnostics already produced by MMP may be used initially:

1. local confidence;
2. local entropy;
3. global confidence;
4. global entropy;
5. tracker confidence;
6. visibility probability;
7. selector probability;
8. local-global disagreement in pixels;
9. coarse-refine disagreement in pixels;
10. selected-global indicator;
11. active-query indicator;
12. elapsed frames since query;
13. consecutive predicted-occluded duration.

Missing branch-specific values are represented by a value plus an explicit availability mask, not by an undocumented sentinel.

## 5. Data partitions and leakage rules

Four roles must remain distinct:

- training split: fits learned uncertainty-head parameters;
- calibration split: fits post-hoc scale, temperature, isotonic, or conformal parameters;
- validation split: selects architecture and fixed hyperparameters;
- test split: final reporting only.

No test item, test residual, test occlusion-duration statistic, or test quantile may be used to fit or select any parameter. Dataset identity and video identity must not cross train/calibration/validation/test boundaries. Exact dataset manifests and hashes must be stored with each result bundle.

Initial development may use a bounded synthetic pilot, but a paper-level claim requires held-out real-video evaluation.

## 6. Baselines

All baselines use identical cached `mu` predictions:

1. global scalar variance fitted by validation NLL;
2. visibility-conditioned scalar variance with visible and occluded bins;
3. occlusion-duration-binned scalar variance, with bins fixed before test evaluation;
4. learned diagonal Gaussian head.

An optional conformal rescaling baseline may be added, but it cannot replace the first three.

## 7. Loss and metrics

Primary proper scoring rule:

- mean 2D Gaussian negative log likelihood in pixel coordinates.

Required secondary metrics:

- 50%, 68%, 90%, and 95% confidence-ellipse empirical coverage;
- absolute coverage gap at each nominal level;
- mean ellipse area or equivalent sharpness statistic;
- risk-coverage curve and AURC using Euclidean point error as risk;
- median and tail Mahalanobis distance;
- unchanged AJ, position accuracy, and occlusion accuracy;
- average and p95 inference latency for the uncertainty head.

For nominal coverage `q`, the ellipse is defined by

`(z - mu)^T Sigma^{-1} (z - mu) <= chi2_ppf(q, df=2)`.

The one-dimensional `1 sigma = 68%` and `2 sigma = 95%` rule must not be used for 2D ellipse coverage.

## 8. Required strata

Metrics are reported overall and for:

- visible frames;
- occluded frames;
- short, medium, and long occlusion, with thresholds fixed from training/calibration data;
- first 1-5 visible frames after re-entry;
- visible frames 6-20 after re-entry;
- local-selected and global-selected frames when applicable.

Buckets with insufficient support must report sample count and confidence intervals rather than being merged after seeing results.

## 9. Seeds and uncertainty

Learned-head experiments use seeds `17`, `29`, and `43`. Report mean, standard deviation, and per-seed values. Bootstrap confidence intervals are computed by video, not by frame, to respect temporal dependence.

## 10. Hard decision gates

MVP-1 passes only if all of the following hold on held-out evaluation data:

1. learned-head NLL improves over every required post-hoc baseline;
2. long-occlusion or re-entry coverage gap improves by at least 20% relative to the strongest post-hoc baseline at two or more preregistered nominal levels;
3. risk-coverage or AURC improves over the strongest post-hoc baseline;
4. improvement direction is consistent across all three seeds and is not caused by unbounded ellipse inflation;
5. AJ change is within `0.1` absolute point and cached coordinate predictions are identical within serialization tolerance;
6. no train/test leakage or failed provenance check is detected.

If the learned head only matches simple post-hoc scaling, MVP-1 is considered negative for a model-paper claim. The calibration protocol may still remain useful as an evaluation contribution.

## 11. Stop rules

- Do not implement learned coast unless MVP-1 passes.
- Do not implement `K=4` hypotheses unless learned coast passes its own preregistered gate.
- Do not add positive/negative identity memory unless multi-hypothesis re-entry improves while false reacquisition, identity switches, and p95 latency remain controlled.
- A failed gate is recorded in the shared Notion page and in a repository result note; it is not silently retuned.

## 12. Phase-0 engineering requirements

Before uncertainty training:

- repair flat-routing selector loss handling for empty rank logits;
- make malformed two-stage routing fail fast with a shape-specific error;
- reject unknown routing-mode strings;
- add unit tests for all three cases;
- run MMP shape and routing tests;
- commit and push the phase-0 repair before adding the uncertainty head.

## 13. Planned files after phase 0

- `projects/mmp_tracker/mmp_tracker/uncertainty_head.py`
- `projects/mmp_tracker/mmp_tracker/calibration_metrics.py`
- `projects/mmp_tracker/configs/beliefcal_mvp1.yaml`
- `tests/test_beliefcal_uncertainty.py`
- `tests/test_beliefcal_calibration_metrics.py`

The training and cache-generation entrypoints will be added only after their data contract and output manifest are specified.

## Amendment A1 - 2026-07-14

This amendment was written after the bounded engineering pilots and before any
new formal multi-seed, conditional-calibration, or conformal experiment. It
clarifies ambiguous protocol text; it does not retroactively convert prior
pilots into confirmatory evidence.

### A1.1 Data-role authority

The four roles have the following non-overlapping purposes:

- the training split fits learned uncertainty-head parameters;
- the calibration split fits every post-hoc parameter, including global or
  grouped variance scales and any predeclared empirical quantile;
- the validation split selects architecture and fixed hyperparameters using the
  predeclared validation objective; it must not fit post-hoc calibration
  parameters;
- the test split is used once for final reporting after all choices are frozen.

Accordingly, Section 6 item 1 is corrected from "fitted by validation NLL" to
"fitted on the calibration split by calibration NLL." The same rule applies to
the visibility-conditioned and duration-binned baselines.

### A1.2 Dataset identity, family, and leakage

`dataset_identity` means the exact role-specific dataset realization, including
its resolved root, split, annotation manifest, and manifest hash when present.
`dataset_family` is the normalized parent corpus name, such as `kubric`,
`davis`, or `rgb_stacking`.

The mandatory leakage rule is exact-item separation: a source video, source
sample, annotation item, or role-specific dataset identity must not cross
training, calibration, validation, and test. Reusing one dataset family across
roles is not by itself leakage when exact source items are disjoint. A
`--require-distinct-dataset-families` audit is an optional cross-family stress
test, not a default validity requirement.

Formal result bundles must record the checkpoint hash, model-config hash,
feature-order hash, role-specific dataset identity, source-file provenance, git
head, and git dirty state. A dirty-worktree cache, an exact identity collision,
a source-file collision, or a feature-order mismatch fails the formal audit.
Legacy manifests that lack these fields may be used only for engineering
diagnostics and cannot satisfy the paper gate.

### A1.3 Feature contract and post-hoc ablations

The 13 value channels plus 13 availability bits remain the preregistered MVP-1
feature contract. The `drop_inert_mmp` profile was introduced after inspecting
pilot feature support and is therefore classified as an exploratory engineering
ablation. It cannot replace the full profile in the preregistered decision gate
without a later amendment written before a new confirmatory run.

A learned conditional-calibration network, calibration-aware training loss, or
other feature-conditioned post-hoc model is outside the current frozen MVP-1
method set. It requires a separate prospective amendment before implementation
or evaluation.

### A1.4 Conformal terminology and guarantees

An empirical residual or Mahalanobis quantile may be reported descriptively.
It may be called split conformal with a finite-sample marginal coverage
guarantee only when calibration and evaluation examples satisfy the required
exchangeability assumptions and the conformity score and nominal levels were
fixed before test inspection.

Cross-dataset or cross-family calibration-to-test stress tests do not establish
exchangeability. Their coverage is empirical only and must not be presented as
distribution-free or finite-sample guaranteed. No conformal-radius experiment
will be added to the formal MVP-1 gate without a prospective amendment that
fixes the score, quantile convention, dependence unit, nominal levels, and
exchangeability claim.

### A1.5 Status of existing pilots

The corrected-v2 run, the RGB-Stacking cache smoke test, the feature-coverage
analysis, and the feature-pruning run are engineering diagnostics only. They use
small role counts, include a one-video DAVIS test pilot, and predate the complete
manifest schema. A lower AURC observed in one seed and one test video is a
numerical observation, not general evidence that uncertainty ranking succeeds.
These pilots cannot pass or fail the scientific MVP-1 gate and cannot be used to
select a new method while retaining confirmatory status.


## Amendment A2 - 2026-07-14: conditional shared-scale calibration extension

This amendment is prospective relative to the implementation and any new
conditional-calibration run. It defines an exploratory extension, `MVP-1C`,
that is evaluated separately from the original MVP-1 decision gate. Existing
pilot test results are not used as confirmatory evidence for this extension.

### A2.1 Frozen components and data roles

The MMP tracker, cached predictive mean, learned uncertainty head, and learned
raw diagonal variance remain frozen. The conditional calibrator may only fit on
the calibration split. The validation split is used once to choose between the
pre-existing shared scalar calibration and the fixed conditional calibrator.
The test split is not loaded by the fitting command and is evaluated only after
the calibration bundle and selection decision have been serialized.

### A2.2 Fixed conditional calibrator

For each cache row, the calibrator input is fixed to:

1. the full preregistered 26-dimensional uncertainty feature vector;
2. log geometric mean of the raw diagonal variance;
3. log variance anisotropy, `log(var_y) - log(var_x)`.

The calibrator is one affine layer that predicts one shared row-wise log scale.
It cannot change the predictive mean or the ratio between `var_y` and `var_x`.
The log scale is clamped to `[log(1/64), log(64)]`, and the final coordinate
variances remain clamped to the existing `[0.25^2, 256^2]` pixel-squared bounds.

The affine weights are initialized to zero and the bias is initialized from the
closed-form shared scalar calibration. Inputs are standardized using
calibration-split statistics only. Optimization is deterministic full-batch
AdamW with:

- 500 steps;
- learning rate `0.01`;
- weight decay `0`;
- L2 penalty `1e-3` on non-bias affine weights;
- the learned-head seed reused as the calibrator seed.

The fitting objective is calibration-split mean diagonal-Gaussian NLL plus the
fixed L2 penalty. The best calibration-NLL state across the fixed 500 steps is
serialized. No early stopping or hyperparameter search uses validation or test
labels.

### A2.3 Predeclared validation selection rule

Both candidates are fitted only on the calibration split:

- shared scalar calibration;
- the fixed conditional affine calibrator.

The conditional candidate is selected only when its validation NLL improves on
the shared scalar candidate by at least 1% relative. Otherwise the shared
scalar candidate remains selected. This selection decision, both validation
reports, all hyperparameters, cache hashes, and the learned-head result hash
must be serialized before any test evaluation.

### A2.4 Evaluation and interpretation

The extension uses the same NLL, coverage, sharpness, AURC, strata, three
learned-head seeds, and video-level bootstrap requirements as MVP-1. A shared
positive scalar usually preserves uncertainty ranking; the conditional scale
may change ranking and therefore AURC must be remeasured rather than assumed.

A result from legacy manifests, a one-video test set, or a command that loaded
test labels before serialization is engineering-only. `MVP-1C` is not a
conformal method and carries no distribution-free coverage guarantee. No
conformal quantile or calibration-aware training loss is authorized by A2.
