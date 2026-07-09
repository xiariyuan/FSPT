# V9-A2 Anchor Uncertainty Reacquisition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build V9-A2 feature dataset and prototype controller that augments V9-A1 with anchor identity similarity and uncertainty-aware consistency.

**Architecture:** Reuse V8-C0.2 touched-frame rows and V9-A1 apply-back infrastructure. Add deterministic RGB patch descriptors at candidate/query/last-reliable/pre-occlusion coordinates and uncertainty-proxy features. Train video-heldout controllers and apply them back to trajectories.

**Tech Stack:** Python 3, NumPy, scikit-learn, PyTorch cache loading, existing V8/V9 apply-back helpers.

## Global Constraints

- No future frames in default features.
- No GT features as model inputs.
- Video-heldout validation only.
- Apply-back to full trajectories is mandatory.
- V9-A2 remains prototype-level; no full ReEntryBeliefTrack claim.

---

## Task 1: Feature-builder smoke

- [ ] Create `scripts/v9a2_anchor_uncertainty_reacquisition_prototype.py`.
- [ ] Load V8-C0.2 dataset and native/candidate caches.
- [ ] Implement metadata-to-record mapping by `video_id`, `query_idx`, and `frame_tau`.
- [ ] Extract RGB patches for 20 rows at candidate, query, last reliable, and pre-occlusion coordinates.
- [ ] Save smoke JSON with patch validity rates and feature dimensions.

## Task 2: Full feature dataset

- [ ] Build features for all 1456 touched rows.
- [ ] Include V9-A1 original `X` features.
- [ ] Add patch similarities for radii 5, 9, 17.
- [ ] Add uncertainty proxy features: event age, low-run proxy, motion disagreement, normalized distance.
- [ ] Save `v9a2_features_dist_nc_le64_w8.npz`.

## Task 3: Video-heldout training

- [ ] Train logreg, HGB, and ExtraTrees on V9-A2 feature matrix.
- [ ] Use `y_candidate_good`, `y_candidate_bad`, and `y_false_visible` targets.
- [ ] Report AP, AUC, Brier, and risk-coverage.
- [ ] Compare directly with V9-A1 tabular report.

## Task 4: Trajectory apply-back

- [ ] Reuse V9-A1 / V8-C0.4 apply-back functions.
- [ ] Evaluate anchor-trust, uncertainty-normalized, and combined policies.
- [ ] Run threshold sweep.
- [ ] Compare AJ, OA, delta_avg, delta_4px, AJ_RD, AJ_RD_256, repair/damage, and positive/negative/zero videos.

## Task 5: Result document and stop/go decision

- [ ] Write `docs/v9a2_anchor_uncertainty_reacquisition_result_2026-07-08.md`.
- [ ] Append decision to `CURRENT_MAINLINE.md`.
- [ ] If V9-A2 does not beat or usefully match CVRRM, stop controller route and move to internal candidate head / multi-hypothesis design.

## Self-Review

No unresolved placeholders remain. Each task has an independently testable output. The plan intentionally avoids full model training until V9-A2 proves that anchor/uncertainty features add trajectory-level value.
