# BeliefCal-MMP MVP-1 preregistration

Status: frozen before learned uncertainty experiments

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
