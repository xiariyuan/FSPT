# SOTA ReEntry Upgrade Plan — 2026-07-05

## 0. Decision

Open a gated SOTA-upgrade route, but do **not** directly replace the base tracker and run ReEntry blindly.

The correct route is:

```text
SOTA protocol parity -> visibility/re-entry oracle headroom -> plug-in ReEntry -> learned metric-aware V26 -> paper decision
```

Current top-tier route remains closed until this route produces evidence that survives the strict gates below.

## 1. Why this route is necessary

The current paper has real diagnostic value, but as a method/improvement paper it has severe weaknesses:

```text
1. The headline +2.19 AJ is vs CoTracker3 offline, not vs stronger CoTracker3 baseline.
2. DAVIS strided/original default ReEntry improves AJ_RD but hurts standard AJ/OA.
3. RGB-Stacking full50 shows systematic AJ trade-off: most videos lose AJ while OA/AJ_RD improve.
4. Evidence is mostly inside the CoTracker3 family.
5. Improvements concentrate on AJ_RD, which is diagnostic rather than leaderboard-standard.
```

Therefore, the only credible way to upgrade the paper is to show that ReEntry is a **SOTA-compatible plug-in re-entry calibration layer**, not merely a patch for a weak/offline CoTracker variant.

## 2. External SOTA context

Relevant current trackers / baselines:

```text
TAPNext++:
  - Directly targets long sequences and re-detection.
  - Introduces / emphasizes AJ_RD-style re-detection evaluation.
  - Highest conceptual fit for ReEntry, but likely smallest remaining headroom.
  - Requires checkpoint/code availability and protocol parity first.

TrackOn2:
  - Strong online long-term point tracker with memory.
  - Already has a parity-valid DAVIS first/input bridge in this repo.
  - Existing strided/original cache is not parity-valid and must not be used in main tables.

CoTracker3:
  - Current main base/override family.
  - Useful for current diagnostic paper, but insufficient alone for strong method claim.

TAPIR / BootsTAP / LocoTrack / TAPNext:
  - Secondary candidates.
  - Use only if protocol parity can be established and checkpoints are accessible.
```

## 3. Existing internal evidence to preserve

### 3.1 TrackOn2 first/input plug-in evidence

Current valid appendix-level result:

```text
TrackOn2 first-input:
  AJ_RD_256 = 0.5444
  AJ_256    = 67.0406
  OA_256    = 93.0615

TrackOn2 base + B2-W16 / ReEntry:
  AJ_RD_256 = 0.5513
  AJ_256    = 67.1750
  OA_256    = 93.2513

Gain:
  AJ_RD_256 +0.0069
  AJ_256    +0.1344
  OA_256    +0.1898
```

Use this as:

```text
Appendix / supplemental evidence that ReEntry can produce small positive plug-in gains on a reproduced strong TrackOn2 first/input baseline.
```

Do **not** use it as:

```text
Main proof that ReEntry beats TrackOn2 generally.
Main strided/original evidence.
Top-tier claim.
```

### 3.2 Existing TrackOn2/TAPNext strided-original warning

Existing strided/original TrackOn2/TAPNext caches are not parity-valid:

```text
TrackOn2 strided/original existing cache:
  AJ is too low; delta_avg collapses.

TAPNext strided/original existing cache:
  AJ is too low; delta_avg collapses.
```

Interpretation:

```text
The current strided-original issue is probably protocol/export/adapter related, not proof that TrackOn2/TAPNext are weak.
```

Rule:

```text
No existing non-parity TrackOn2/TAPNext strided cache can enter the main paper table.
```

## 4. Core hypothesis

The SOTA-upgrade route tests the following hypothesis:

```text
Even for a strong SOTA base tracker, re-entry visibility remains a partially separable failure mode.
A coordinate-preserving visibility calibrator can improve re-entry reliability without harming standard AJ/OA if corrections are selected by a metric-aware safety gate.
```

This hypothesis has two sub-questions:

```text
Q1. Does a SOTA base still have visibility/re-entry headroom when coordinates are held fixed?
Q2. Can a deployable selector approximate that headroom without using ground truth at inference?
```

If Q1 fails, stop. If Q1 passes but Q2 fails, keep diagnostic paper only.

## 5. Gated experimental plan

## Gate 0 — Freeze current diagnostic paper

Before new SOTA work changes the story, preserve the current paper state:

```text
Current paper target = diagnostic / workshop-first.
Current claim = re-entry failure-mode diagnosis + visibility calibration trade-off.
Do not rewrite the main paper as a SOTA method paper unless later gates pass.
```

Required artifacts:

```text
docs/final_paper_tables_2026-07-04.md
docs/paper_full_draft_v0_2026-07-04.md
paper/reentry_viscalibrator_tex/main.pdf
```

## Gate 1 — SOTA candidate acquisition and adapter smoke

For each SOTA candidate, first prove that the model can be loaded and exported into the repository cache schema.

Candidate priority:

```text
Priority 1: TrackOn2 strict protocol parity
Priority 2: TAPNext++ 1-video smoke + checkpoint availability
Priority 3: TAPNext / LocoTrack / TAPIR / BootsTAP only if they are easier to run with parity
```

Adapter output must match this schema:

```text
records[i].video_id
records[i].query_points      # [N, 3], query_t/y/x
records[i].pred_tracks       # [N, T, 2], normalized y/x
records[i].pred_visibility   # [N, T]
records[i].gt_tracks
records[i].gt_visibility
records[i].original_size
```

Stop conditions:

```text
1. Cannot load official/pretrained checkpoint.
2. Cannot export query-aligned predictions.
3. Model requires unsupported compiled ops and no safe install path exists.
4. Output coordinate convention cannot be reconciled.
```

## Gate 2 — Protocol parity audit

No SOTA result is usable until parity is established.

Minimum checks:

```text
1. Query anchor sanity:
   predicted query-frame position must match the query point.

2. Coordinate convention audit:
   yx vs xy, H/W vs H-1/W-1, 256-space vs original-resolution must be explicit.

3. Visibility convention audit:
   visible=True / occluded=False must match the evaluator.

4. Official/repo parity if available:
   first/input result should match the model's reported or repo-native evaluation.

5. No coordinate collapse:
   delta_avg and AJ must be in a plausible range for the claimed SOTA tracker.
```

Suggested thresholds:

```text
query_anchor_mean_256 <= 1.0 px
query_anchor_max_256  <= 3.0 px unless model quantization explains it
first/input AJ within approx. 0.5 pp of repo-native / paper reproduction when available
strided/original must not show unexplained delta_avg collapse
```

Required artifact:

```text
docs/sota_protocol_parity_<tracker>_<protocol>_<date>.md
```

Hard rule:

```text
If Gate 2 fails, do not run ReEntry and do not report the tracker in main tables.
```

## Gate 3 — Oracle headroom audit

Before applying our module, test whether the SOTA base has visibility/re-entry headroom with coordinates fixed.

Evaluate variants:

```text
A. base_original
B. base + GT visibility
C. base + GT re-entry-only visibility correction
D. base + candidate-window oracle visibility correction
E. base + all-visible stress
F. base + query-visible-fill sanity
```

Metrics:

```text
AJ
OA
delta_avg
AJ_RD
AJ_RD_256
paired video statistics
re-entry event buckets by occlusion length
false-visible rate during occlusion
visibility lag at re-entry
```

Interpretation:

```text
If GT visibility barely improves AJ_RD, then the SOTA bottleneck is not visibility. Stop.
If GT visibility improves AJ_RD but hurts AJ/OA, then a safety selector is mandatory.
If candidate-window oracle improves AJ_RD and preserves AJ/OA, then ReEntry has deployable headroom.
```

Minimum go criterion:

```text
candidate-window oracle or GT re-entry-only visibility gives:
  AJ_RD gain >= +0.01
  and AJ/OA not catastrophically worse
```

Preferred go criterion:

```text
candidate-window oracle gives:
  AJ_RD gain >= +0.02
  AJ >= base - 0.05
  OA >= base - 0.05
```

Required artifact:

```text
docs/sota_oracle_headroom_<tracker>_<protocol>_<date>.md
```

Hard rule:

```text
If oracle headroom is absent, do not train V26 for that base. Keep result as diagnostic negative evidence.
```

## Gate 4 — Naive plug-in ReEntry on SOTA base

Only after Gate 3 passes, apply existing ReEntry variants.

Variants:

```text
1. Deterministic B2-W16 / B2-W16-P2
2. V1 learned calibrator
3. V22Q interval/gate post-processing
4. V24 DINOScore filter if feature access is practical
5. V25 conservative threshold sweep
```

Base / override combinations to test:

```text
TrackOn2 base + CoTracker3 offline/online override
TAPNext++ base + CoTracker3 offline/online override
SOTA base + same-family alternative visibility stream, if available
CoTracker3 base + SOTA override, only as diagnostic comparison
```

Important:

```text
Coordinates always come from the base tracker.
Override contributes visibility/candidate signal only.
```

Success criterion for a main-paper upgrade:

```text
AJ >= base - 0.05
OA >= base - 0.05
AJ_RD >= base + 0.01
paired AJ CI must not be entirely negative
paired OA or AJ_RD should be positive / stable
```

If this passes on a strong SOTA base, paper can upgrade from diagnostic-only to plug-in method paper.

If it fails but oracle headroom exists, proceed to Gate 5.

Required artifact:

```text
docs/sota_reentry_plugin_results_<tracker>_<protocol>_<date>.md
```

## Gate 5 — True V26 metric-aware selector

Current distance-filter V26 is not enough. Existing result is essentially V25-level:

```text
Best safe V26 distance-filter point:
  AJ_orig = 51.5671
  OA_orig = 92.2151
  AJ_RD   = 0.3901

Gain vs CoTracker3 offline:
  AJ    +0.0286
  OA    +0.0608
  AJ_RD +0.0031
```

Therefore, future V26 must be a true metric-aware selector, not another distance threshold.

Training target:

```text
For each proposed recovery frame or segment, estimate local utility:
  utility = expected metric contribution if visibility is opened

Positive label:
  keep if expected ΔAJ/ΔOA/ΔAJ_RD utility is positive under the target protocol.

Negative label:
  drop if opening visibility creates false-visible penalty or coordinate-inaccurate visible penalty.
```

Feature families:

```text
1. Existing 28 V1 temporal features.
2. base/override coordinate disagreement.
3. base temporal smoothness / acceleration / jump features.
4. visibility run-length features.
5. candidate segment duration and confidence statistics.
6. base tracker confidence if available.
7. override tracker confidence if available.
8. local appearance similarity if cheap and parity-safe.
9. protocol-specific query timing / occlusion-length features.
```

Model candidates:

```text
Start simple:
  logistic regression / small MLP / gradient-boosted tree on frame or segment features.

Then sequence-aware:
  small temporal Conv/Transformer only if simple models show separability.
```

Data split rule:

```text
Train on dev videos only.
Tune threshold on held-out dev videos.
Freeze once.
Evaluate once on fresh/test videos.
No tuning on final test split.
```

Success criterion:

```text
Strict protocol:
  AJ >= base
  OA >= base
  AJ_RD >= base + 0.01

or at minimum:
  AJ >= base - 0.05
  OA >= base - 0.05
  AJ_RD >= base + 0.015
```

If V26 cannot beat V25 by a meaningful margin, do not promote it.

Required artifact:

```text
docs/sota_v26_metric_aware_selector_<tracker>_<protocol>_<date>.md
```

## 6. Paper decision tree

### Case A — SOTA plug-in success

Condition:

```text
At least one strong SOTA base passes parity and ReEntry/V26 improves AJ_RD while preserving AJ/OA.
```

Paper position:

```text
SOTA-compatible plug-in re-entry calibration method.
```

Possible title:

```text
ReEntry: Plug-in Re-Detection Calibration for State-of-the-Art Point Trackers
```

Possible venue target:

```text
Mid-tier conference; top-tier only if gains are strong, cross-tracker, and protocol-clean.
```

### Case B — only small TrackOn2 first/input gains

Condition:

```text
Only the existing TrackOn2 first/input +0.0069 AJ_RD_256 result holds.
No strict protocol improvement.
```

Paper position:

```text
Diagnostic paper with appendix plug-in generality evidence.
```

Venue target:

```text
Workshop / arXiv / possibly low-mid if writing and analysis are strong.
```

### Case C — SOTA oracle headroom exists but deployable selector fails

Paper position:

```text
Diagnostic + upper-bound analysis paper.
```

Claim:

```text
Strong trackers still have re-entry visibility headroom, but deployable metric-safe selection remains unsolved.
```

Venue target:

```text
Workshop or analysis-oriented submission.
```

### Case D — no SOTA oracle headroom

Paper position:

```text
Failure-mode diagnostic paper only.
```

Claim:

```text
Visibility calibration helps weaker/offline CoTracker-family baselines, but stronger SOTA trackers shift the bottleneck to coordinates, identity, or memory.
```

Venue target:

```text
Workshop / arXiv.
```

## 7. Required scripts / artifacts to create or reuse

Reuse existing scripts where possible:

```text
scripts/audit_baseline_cache_parity.py
scripts/audit_baseline_visibility_oracle.py
scripts/paired_video_stats_reentry_methods.py
scripts/eval_reentry_viscalibrator.py
scripts/eval_reentry_viscalibrator_v22_interval.py
scripts/eval_reentry_v26_metric_aware.py
```

Likely new scripts:

```text
scripts/export_sota_to_reentry_cache.py
scripts/audit_sota_protocol_parity.py
scripts/eval_sota_visibility_oracles.py
scripts/eval_reentry_plugin_on_sota.py
scripts/build_v26_metric_utility_dataset.py
scripts/train_v26_metric_aware_selector.py
scripts/eval_v26_metric_aware_selector.py
```

Every new experiment must produce:

```text
1. result cache path
2. json summary
3. paired-video stats
4. claim-boundary doc
5. CURRENT_MAINLINE pointer
```

## 8. Immediate next actions

### Action 1 — Preserve current paper route

Add this document to CURRENT_MAINLINE and mark:

```text
Current default target = diagnostic / workshop-first.
SOTA upgrade route = gated, not assumed.
```

### Action 2 — Formalize TrackOn2 first/input result

Move existing TrackOn2 first/input plug-in result into the current paper appendix plan.

Required doc update:

```text
docs/paper_full_draft_v0_2026-07-04.md
paper/reentry_viscalibrator_tex/sections/appendix_b_additional_tables.tex
```

Only if space allows; otherwise keep as supplemental.

### Action 3 — Start TrackOn2 strict parity work

Goal:

```text
Produce a parity-valid TrackOn2 strided/original cache or document why it is blocked.
```

Stop if:

```text
mmcv/compiled ops cannot be installed safely.
Coordinate convention cannot be reconciled.
Repo-native strided support does not exist.
```

### Action 4 — TAPNext++ acquisition smoke

Goal:

```text
Find/check checkpoint and run 1-video DAVIS smoke.
```

Stop if:

```text
No accessible checkpoint.
Code cannot export arbitrary query predictions.
Protocol cannot be aligned.
```

### Action 5 — Oracle before module

For whichever SOTA candidate passes parity, run oracle headroom before applying ReEntry.

Rule:

```text
No oracle headroom -> no ReEntry experiment.
```

## 9. Non-negotiable rules

```text
1. Do not report non-parity SOTA caches in main tables.
2. Do not tune on final/fresh split.
3. Do not call local results official leaderboard results.
4. Do not claim universal improvement unless cross-tracker evidence exists.
5. Do not promote V26 unless it beats V25 by a meaningful margin.
6. Do not continue broad threshold sweeps without a written gate question.
7. Do not let the SOTA route block the current diagnostic paper indefinitely.
```

## 10. Final recommendation

Proceed with SOTA integration, but treat it as a gated rescue route:

```text
Primary stable route:
  diagnostic/workshop paper based on current evidence.

High-upside rescue route:
  SOTA-compatible ReEntry if TrackOn2/TAPNext++ parity + oracle + plug-in/V26 succeed.

Do not rewrite the paper as a SOTA method paper until the gates pass.
```
