# V9-A1 Controller Calibration-Aware Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a V9-A1 prototype that trains video-heldout recovery controllers on the existing V8-C0.2 touched-frame dataset and applies accepted frames back to trajectories to compare against CVRRM.

**Architecture:** Reuse the V8-C0.2 NPZ touched-frame dataset for supervised tabular learning, then reuse the V8-C0.4 accept-mask trajectory application pattern. V9-A1 remains controller-only: no new backbone, no internal candidate head, and no full belief tracker.

**Tech Stack:** Python 3, NumPy, PyTorch cache loading, scikit-learn tabular classifiers, existing V8 metric helpers.

## Global Constraints

- Inputs must remain causal at the evaluated frame; GT is only used for offline labels and metrics.
- Validation must be video-heldout, not random frame split.
- Report trajectory-level metrics, not only AP/AUC.
- Compare against CVRRM w8/w16 and V8-C0.4 verifier rows.
- Output directory: `outputs/paper_discovery_2026-07-05/v9a1_controller_calibration_aware_prototype/`.

---

## Files

- Create `scripts/v9a1_controller_calibration_aware_prototype.py`.
- Create `tests/test_v9a1_controller_utils.py`.
- Create `docs/v9a1_controller_calibration_aware_prototype_result_2026-07-08.md` after running.
- Modify `CURRENT_MAINLINE.md` after result generation.

---

## Task 1: Utility layer

**Produces:**

- `video_from_meta(meta: dict) -> str`
- `risk_coverage_curve(scores, losses, coverages) -> dict`
- `choose_threshold_by_metric(y_true, scores, metric='f1') -> float`

**Steps:**

- [ ] Write unit tests for metadata parsing, risk-coverage ordering, and threshold selection.
- [ ] Run `pytest tests/test_v9a1_controller_utils.py -v` and confirm failure before implementation.
- [ ] Implement minimal utilities in `scripts/v9a1_controller_calibration_aware_prototype.py`.
- [ ] Run the same test file and confirm pass.

---

## Task 2: Dataset loading and video groups

**Produces:**

- `load_v8c02_dataset(path: Path) -> dict`
- `build_video_groups(metas: list[dict]) -> np.ndarray`
- `select_feature_matrix(data: dict) -> tuple[np.ndarray, list[str]]`
- `label_array(data: dict, preferred: list[str]) -> tuple[np.ndarray, str]`

**Steps:**

- [ ] Add tests for video group construction and defensive feature selection.
- [ ] Load default dataset `cotracker3_online_v8c02_fine_risk_verifier_dataset/v8c02_fine_risk_dist_nc_le64_w8.npz`.
- [ ] Parse `meta_json` into Python dictionaries.
- [ ] Detect feature matrix `X` first; fall back to one-dimensional `feature_*` vectors.
- [ ] Detect labels by priority, starting with `candidate_good`, then compatible alternatives.
- [ ] Run utility tests.

---

## Task 3: Video-heldout controller training

**Produces:**

- `train_groupheldout_models(data: dict, label_keys: list[str]) -> dict`
- JSON report `v9a1_tabular_cv_report.json`

**Models:**

- Logistic regression.
- Histogram gradient boosting.
- Random forest or ExtraTrees.

**Metrics:**

- AP.
- AUC where valid.
- Brier score.
- risk-coverage using accept probability as low-risk score.
- fold video names.

**Steps:**

- [ ] Implement `GroupKFold` by video.
- [ ] Train each model on train videos and evaluate on heldout videos.
- [ ] Select validation threshold using F1 or precision-constrained F1.
- [ ] Write fold rows into JSON.
- [ ] Run `python scripts/v9a1_controller_calibration_aware_prototype.py --tabular-only`.

---

## Task 4: Apply controller back to trajectories

**Produces:**

- `make_accept_mask(scores, threshold) -> np.ndarray`
- `apply_controller_to_trajectories(...) -> dict`
- full trajectory metrics against native and CVRRM baselines.

**Steps:**

- [ ] Inspect `scripts/eval_cotracker3_online_v8c04_apply_fine_risk_verifier.py` for the exact `load_dataset` and `apply_touched_mask` helper signatures.
- [ ] Reuse V8-C0.4 apply-back semantics rather than reimplementing trajectory replacement from memory.
- [ ] For each heldout fold, predict candidate accept scores only for heldout touched frames.
- [ ] Select threshold on train/validation videos only.
- [ ] Apply heldout accept mask back to trajectories.
- [ ] Compute AJ, OA, delta_avg, delta_4px, AJ_RD, AJ_RD_256, repair16, damage16, and positive/negative/zero videos.

---

## Task 5: Report generation and decision

**Produces:**

- `docs/v9a1_controller_calibration_aware_prototype_result_2026-07-08.md`
- updated `CURRENT_MAINLINE.md`

**Decision rules:**

V9-A1 passes only if:

- AJ_RD_256 beats CVRRM w8 on heldout trajectory-level evaluation.
- AJ/OA do not materially drop.
- false reacquisition or damage decreases.
- risk-coverage improves over raw confidence proxies.
- gains are not dominated by one video.

V9-A1 fails if:

- AP/AUC improves but trajectory metrics do not.
- thresholds are unstable across folds.
- gains vanish outside the training-like videos.

**Steps:**

- [ ] Generate markdown table for tabular metrics.
- [ ] Generate markdown table for trajectory metrics.
- [ ] Add explicit pass/fail judgement.
- [ ] Append final V9-A1 decision to `CURRENT_MAINLINE.md`.

---

## Self-Review

Spec coverage: dataset loading, video-heldout validation, causal inputs, risk calibration, trajectory application, and stop criteria are covered.

Placeholder scan: no unresolved placeholders remain.

Scope check: this is controller-only and does not include internal candidate generation or full ReEntryBeliefTrack.
