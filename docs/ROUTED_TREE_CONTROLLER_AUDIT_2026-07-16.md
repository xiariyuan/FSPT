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

A two-video runtime smoke then completed successfully. Its numerical values were not used for tuning. The full 50-video evaluation is the next frozen result.

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
- `rgb_stacking_first_tree_closed_loop_full50.json` after completion.

## Decision rule after RGB-Stacking

- If the frozen controller produces positive paired AJ and delta-average intervals, preserve this controller as the current Route-D MVP and move directly to an untouched external evaluation plus ablations.
- If aggregate transfer is positive but highly concentrated, audit sequence-level failure modes and add a predeclared trajectory-stability guard trained only on Kubric.
- If transfer collapses, do not tune on RGB-Stacking. Return to Kubric and add candidate-specific appearance, positive-memory, negative-memory, and cycle-consistency features before one new frozen evaluation.
