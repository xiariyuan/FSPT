# Route D Frozen Tree Controller Audit — 2026-07-16

## Claim boundary

All results in this note are development or transfer diagnostics. They are not yet eligible as untouched final-test paper claims.

- DAVIS has already been used during Route-D development.
- RGB-Stacking has also appeared in prior repository experiments.
- No DAVIS or RGB-Stacking labels were used to fit or retune the frozen Kubric tree controller described here.
- The next publication-grade requirement is a predeclared evaluation on a genuinely unused dataset or split.

## Why this branch was opened

Earlier neural risk, expected-utility, conformal, video-context, and hierarchical controllers recovered only a small fraction of the frozen candidate oracle gap. The key unresolved question was whether the existing action feature vector lacked sufficient information, or whether the small neural controller was a poor match for the non-smooth tabular decision boundary.

The tree audit isolates this question without changing the tracker, candidate generator, or causal multi-threshold scorer.

## 1. Nested DAVIS tree-family audit

Protocol:

- 30 TAP-Vid-DAVIS videos, first-query, 256 x 256.
- Five outer video folds.
- Each outer-training partition is split into tree fitting, tree-configuration validation, isotonic post-hoc calibration, and policy calibration.
- Outer-test videos are excluded from all fitting, calibration, and policy selection.
- ExtraTrees search: 400 trees, `max_features=sqrt`, `min_samples_leaf` in `{10, 20, 40}`.

OOF ranking:

- AUC: 0.79484.
- Average precision: 0.35846.

OOF frozen-action metrics:

- Global selection rate: 9.165%.
- Beneficial recall: 32.259%.
- Harmful selected-global rate: 16.267%.
- Mean pixel gain over local: +0.16058 px.
- Mean threshold-utility gain: +0.011509.
- Delta gains at 1 / 2 / 4 / 8 / 16 px:
  - +0.003845
  - +0.013314
  - +0.023672
  - +0.016505
  - +0.000209

Paired video bootstrap:

- Pixel gain: +0.19793 px, 95% CI [0.08352, 0.34417].
- Threshold utility: +0.01323, 95% CI [0.00659, 0.02135].
- Delta-1: +0.00460, 95% CI [0.00088, 0.00837].
- Delta-2: +0.01505, 95% CI [0.00746, 0.02393].
- Delta-4: +0.02568, 95% CI [0.01429, 0.03953].
- Delta-8: +0.01927, 95% CI [0.00871, 0.03245].
- Delta-16 remains inconclusive.

Interpretation: the existing action evidence is not exhausted. Model family and decision geometry materially affect recoverable utility.

## 2. Kubric-only controller freeze

All fitting and policy selection use disjoint partitions of the causal Kubric training cache. The resulting controller is frozen before reading Kubric validation or DAVIS evaluation caches.

Frozen policy:

```text
probability_threshold = 0.6298701167
min_p1_margin         = -0.05
min_coarse_gain       = 0.01
min_total_gain        = -0.05
```

Kubric validation, after freeze:

- Selection rate: 6.445%.
- Mean pixel gain: +0.18730 px.
- Mean threshold-utility gain: +0.012748.
- Delta gains at 1 / 2 / 4 / 8 / 16 px:
  - +0.005053
  - +0.017104
  - +0.025327
  - +0.014656
  - +0.001601

DAVIS zero-fit open-loop transfer:

- Selection rate: 3.045%.
- Mean pixel gain: +0.06203 px.
- Mean threshold-utility gain: +0.004457.
- Delta gains at 1 / 2 / 4 / 8 / 16 px:
  - +0.001726
  - +0.005336
  - +0.009207
  - +0.006199
  - -0.000183

Paired video bootstrap on DAVIS transfer:

- Pixel gain: +0.06958 px, 95% CI [0.02842, 0.11680].
- Threshold utility: +0.00492, 95% CI [0.00264, 0.00751].
- Delta-1: +0.00201, 95% CI [0.00094, 0.00332].
- Delta-2: +0.00611, 95% CI [0.00321, 0.00943].
- Delta-4: +0.00954, 95% CI [0.00527, 0.01449].
- Delta-8: +0.00697, 95% CI [0.00343, 0.01094].
- Delta-16 remains inconclusive.

## 3. DAVIS true closed-loop evaluation

The tree controller was integrated into the tracker so each accepted Route-D action updates the subsequent prior and tracker state. The baseline is an independent model instance without a Route-D selector.

Aggregate TAP metrics:

| System | AJ | OA | Delta average |
|---|---:|---:|---:|
| Independent baseline/local | 0.081948 | 0.751327 | 0.159946 |
| Frozen tree, open-loop | 0.084912 | 0.751327 | 0.165760 |
| Frozen tree, closed-loop | 0.094253 | 0.751327 | 0.179634 |

Differences:

- Closed-loop versus baseline:
  - AJ: +0.012305.
  - Delta average: +0.019688.
- Open-loop versus baseline:
  - AJ: +0.002964.
  - Delta average: +0.005814.
- Closed-loop versus open-loop:
  - AJ: +0.009341.
  - Delta average: +0.013874.

Paired 30-video bootstrap, closed-loop versus baseline:

- AJ: +0.012305, 95% CI [0.005524, 0.020205].
- Delta average: +0.019688, 95% CI [0.009858, 0.030062].
- 22 videos improved, 5 declined, and 3 tied.
- Negative videos: 4, 5, 9, 22, and 25.

Closed-loop diagnostics:

- Actual global selection rate: 1.890%.
- Mean trajectory difference from local: 4.168 px.
- Mean closed-versus-open trajectory difference: 4.045 px.
- Memory-write disagreement rate: 0.

Interpretation: sparse corrective actions can have a large downstream trajectory effect. The gain is not explained by open-loop candidate replacement alone; the feedback into the prior/state is a central part of the mechanism.

## 4. RGB-Stacking frozen corroboration protocol

A protocol manifest was written before the full evaluation. It records SHA-256 hashes for the model checkpoint, tree controller, configuration, dataset, and working-tree diff.

Frozen evaluation:

- Dataset: full 50-video TAP-Vid RGB-Stacking pickle.
- Query mode: first.
- Input and metric resolution: 256 x 256.
- Independent baseline model.
- Closed-loop Route-D enabled.
- No parameter or policy changes after seeing smoke or full-run results.

The first smoke attempt exposed a schema-only interface defect: the shared evaluation configuration passed `resolution=[256,256]` into the RGB-Stacking adapter, which does not implement resize-on-load. The stored data is already 256 x 256. The fix removes this inherited argument only for RGB-Stacking; it does not alter the controller, scorer, gate, or metric resolution. A regression test was added.

A two-video runtime smoke then completed successfully. Its numerical values were not used for tuning.

Full 50-video aggregate TAP metrics:

| System | AJ | OA | Delta average |
|---|---:|---:|---:|
| Independent baseline/local | 0.225756 | 0.809503 | 0.378642 |
| Frozen tree, open-loop | 0.250908 | 0.809503 | 0.417432 |
| Frozen tree, closed-loop | 0.270719 | 0.809503 | 0.440232 |

Differences:

- Closed-loop versus baseline:
  - AJ: +0.044964.
  - Delta average: +0.061589.
- Open-loop versus baseline:
  - AJ: +0.025152.
  - Delta average: +0.038790.
- Closed-loop versus open-loop:
  - AJ: +0.019812.
  - Delta average: +0.022800.

Paired 50-video bootstrap:

- Open-loop versus baseline AJ: +0.025152, 95% CI [0.020779, 0.029776]; 50/50 videos positive.
- Open-loop versus baseline delta average: +0.038790, 95% CI [0.033018, 0.044790]; 50/50 videos positive.
- Closed-loop versus baseline AJ: +0.044964, 95% CI [0.037637, 0.052376]; 49 positive and 1 negative video.
- Closed-loop versus baseline delta average: +0.061589, 95% CI [0.053008, 0.070181]; 49 positive and 1 negative video.
- Closed-loop versus open-loop AJ: +0.019812, 95% CI [0.013875, 0.025852]; 41 positive and 9 negative videos.
- Closed-loop versus open-loop delta average: +0.022800, 95% CI [0.015331, 0.030079]; 41 positive and 9 negative videos.

Closed-loop diagnostics:

- Actual global selection rate: 3.312%.
- Open-loop global selection rate: 13.841%.
- Mean trajectory difference from local: 7.495 px.
- Mean closed-versus-open trajectory difference: 7.095 px.
- Memory-write disagreement rate: 0.

Only RGB-Stacking video 12 declined versus baseline: AJ -0.005683 and delta average -0.012025. The same video improved substantially in open-loop mode, so its failure is specifically a closed-loop feedback instability rather than a candidate-quality failure.

Interpretation: the Kubric-only tree controller transfers strongly across domain without label fitting. The positive result is broad rather than concentrated, and closed-loop state feedback adds significant aggregate value. The controller should now be preserved unchanged as the Route-D MVP; RGB-Stacking must not be used for further tuning.

## Reproduction artifacts

Primary scripts:

- `scripts/audit_routeD_nested_tree_risk_gate.py`
- `scripts/train_routeD_kubric_tree_transfer.py`
- `scripts/analyze_routeD_closed_loop_paired.py`
- `scripts/eval_routeD_risk_selector.py`

Inference integration:

- `projects/mmp_tracker/mmp_tracker/routeD_tree_selector.py`
- `projects/mmp_tracker/mmp_tracker/routeD_selector.py`
- `projects/mmp_tracker/mmp_tracker/model.py`

Tests:

- `tests/test_routeD_nested_tree_risk_gate.py`
- `tests/test_routeD_tree_selector.py`
- `tests/test_rgb_stacking_registration.py`
- existing Route-D closed-loop, selector, and multi-threshold tests.

Output directory:

```text
/gemini/code/FSPT/outputs/routeD_causal_v3_protocol_20260715/
```

Important files:

- `nested_tree_risk_gate_audit.json`
- `kubric_tree_transfer_audit.json`
- `kubric_tree_controller.pkl`
- `davis_first_query_256_tree_closed_loop.json`
- `davis_first_query_256_tree_closed_loop_paired.json`
- `rgb_stacking_tree_protocol_manifest.json`
- `rgb_stacking_first_tree_closed_loop_smoke2.json`
- `rgb_stacking_first_tree_closed_loop_full50.json`
- `rgb_stacking_first_tree_closed_loop_full50_paired.json`

## Decision rule after RGB-Stacking

The first decision condition passed. Preserve the exact controller and proceed to a predeclared Kinetics evaluation without tuning. A deterministic subset must cover all ten Kinetics shards rather than taking only the first sequential videos. The one RGB closed-loop failure remains a diagnostic case; it must not be used to tune a new guard before the Kinetics result is observed.

## 5. Kinetics balanced-50 predeclared evaluation

The available TAP-Vid-Kinetics data is stored in ten source shards with 1,144 total videos. A sequential `limit=50` evaluation would over-represent the first shard, so a deterministic shard-balanced subset was materialized before model inference.

Selection protocol:

- Source: all ten `train.index.json` shards.
- Five videos per source shard.
- Each shard is divided into five equal-width source-position bins.
- One fixed-seed uniform draw is made from each bin.
- Seed: 17.
- Total: 50 unique source positions and 50 unique stable video names.
- No labels, candidate scores, tracker outputs, or metrics are used for selection.

The protocol file was written before loading any source sample. Every source shard, output shard, subset manifest, checkpoint, controller, configuration, and working-tree diff has a SHA-256 record.

Primary decision rule:

- paired 50-video bootstrap for closed-loop AJ versus the independent baseline;
- paired 50-video bootstrap for closed-loop delta average versus the independent baseline;
- success requires both 95% confidence intervals to have lower bounds above zero.

Claim boundary:

- Kinetics is treated as a predeclared external cross-domain evaluation;
- no Kinetics result may be used to retune the scorer, tree, policy, fusion, guard, or subset.

The data-lineage audit found:

- checkpoint training dataset: TAP-Vid-Kubric;
- checkpoint validation dataset: TAP-Vid-DAVIS, subset size 2;
- causal scorer training cache: 64 Kubric videos;
- tree-controller training and calibration cache: the same Kubric-only causal cache family;
- no prior Kinetics result or tuning record in historical output text outside the current Route-D output directory.

Therefore, the result is paper-claim eligible with a restricted scope: a predeclared deterministic shard-balanced 50-video TAP-Vid-Kinetics subset. It is not eligible to be described as full 1,144-video TAP-Vid-Kinetics benchmark performance.

Artifacts:

```text
kinetics_balanced50_seed17/balanced50.protocol.json
kinetics_balanced50_seed17/balanced50.index.json
kinetics_balanced50_tree_protocol_manifest.json
kinetics_balanced50_tree_execution_log.json
kinetics_balanced50_tree_closed_loop_smoke1.json
kinetics_balanced50_tree_closed_loop_full50.json
```

The one-video smoke is restricted to runtime and schema validation. The full run uses the unchanged controller and policy.

Full 50-video result:

| System | AJ | OA | Delta average |
|---|---:|---:|---:|
| Independent baseline/local | 0.382284 | 0.958854 | 0.477211 |
| Frozen tree, open-loop | 0.400033 | 0.958854 | 0.498950 |
| Frozen tree, closed-loop | 0.406890 | 0.958854 | 0.505953 |

Aggregate differences:

- Open-loop versus baseline:
  - AJ: +0.017749.
  - Delta average: +0.021739.
- Closed-loop versus baseline:
  - AJ: +0.024607.
  - Delta average: +0.028742.
- Closed-loop versus open-loop:
  - AJ: +0.006857.
  - Delta average: +0.007003.

Paired 50-video bootstrap:

- Open-loop AJ: +0.017749, 95% CI [0.010496, 0.024992].
- Open-loop delta average: +0.021739, 95% CI [0.013918, 0.029768].
- Closed-loop AJ: +0.024607, 95% CI [0.011284, 0.036929].
- Closed-loop delta average: +0.028742, 95% CI [0.015952, 0.041121].
- Closed-loop versus open-loop AJ: +0.006857, 95% CI [-0.000952, 0.014531].
- Closed-loop versus open-loop delta average: +0.007003, 95% CI [-0.000549, 0.014684].

The predeclared primary decision passed because both closed-loop-versus-baseline confidence intervals have lower bounds above zero. Open-loop transfer is also independently positive. The additional mean gain from closed-loop feedback over open-loop is not statistically resolved on this 50-video subset because both intervals cross zero.

Video-level support:

- Closed-loop AJ: 34 positive, 9 negative, and 7 tied videos.
- Closed-loop delta average: 35 positive, 8 negative, and 7 tied videos.
- Open-loop AJ and delta average: 37 positive, 6 negative, and 7 tied videos.
- Actual closed-loop global selection rate: 4.210%.
- Open-loop global selection rate: 9.460%.
- Mean closed-loop trajectory difference from local: 2.020 px.
- Memory-write disagreement rate: 0.

The most severe failure is `kinetics_balanced_s05_p0061_kinetics_source_s005_000061`:

- open-loop AJ change: -0.070572;
- closed-loop AJ change: -0.164836;
- closed-loop delta-average change: -0.118301;
- global selection rate: 19.291%;
- mean trajectory difference: 9.120 px.

This failure is retained as evidence of a high-selection, high-divergence regime. It must not be used to tune a Kinetics-specific guard.

Final decision:

- preserve the exact Kubric-trained tree controller as the current Route-D MVP;
- report Kinetics only as the predeclared balanced-50 subset result;
- do not tune on DAVIS, RGB-Stacking, or Kinetics;
- the next headline-strength experiment is the complete Kinetics set or another predeclared untouched benchmark, using the same frozen controller;
- development of any trajectory-stability guard must return to Kubric-only partitions and receive a new external protocol before evaluation.

Final artifacts:

```text
kinetics_balanced50_tree_closed_loop_full50.json
kinetics_balanced50_tree_closed_loop_full50_paired.json
kinetics_balanced50_tree_final_audit.json
```
