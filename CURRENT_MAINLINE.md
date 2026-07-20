# CURRENT_MAINLINE

## Active override — 2026-07-19

```text
Gate 2 is complete and formally failed. Gate 2.5 completed with exact replay and
measured a contiguous deterministic state-reinstatement basin through 12 input
pixels. Teacher probability does not enlarge that basin.

Gate 3A v1 is complete with exact candidate-cache and teacher-audit replay. All
three top-8 representations fail candidate support: recall within 12 px is only
0.3750--0.3875 and median nearest error is 15.21--16.73 px. All future-rollout
gates pass under the nearest-candidate teacher oracle, but this cannot rescue
missing top-8 support. Formal decision:
STOP_CURRENT_CANDIDATES_AND_REDESIGN_TEMPORAL_IDENTITY_MATCHING.

Post-gate diagnosis shows M1 native and M2 immutable-query maps are effectively
identical because CoTracker's online support is already query memory. Reverse
cycle, peer geometry, and static DINOv3 controls remain at only 0.333--0.438
top-8 recall and are closed. M1 top-128 oracle recall is 0.9125, so recoverable
support exists but needs candidate-conditioned spatiotemporal identity.

Gate 3C0 completed with exact data-identity qualification: 512 Kubric videos,
32 shards, 54 hashed source TFRecords, and raw-record-disjoint gradient-train
64--383 / checkpoint-selection 384--447 / fit-only audit 448--511 partitions.
Gate 3C1 then fixed the causal candidate-conditioned feature contract with exact
fresh-process replay: 129 candidates, 16 observed frames, nine temporal channels,
14 static channels, and no persisted raw DINO descriptors.

Gate 3C1A completed and passed on exactly gradient-train indices 64--383. All
320 sidecars and tensor/file hashes validate; 12 videos are valid empty members
and the cache contains 3,325 frozen natural-failure rows. Native plus static
top-8 supports 1,725/3,325 rows within 12 px, while the complete native-plus-128
bank supports 3,185/3,325, leaving a 43.9098-point ranking target. This is
training-only candidate headroom, not selector or tracking improvement. Formal
decision: AUTHORIZE_GATE3C1B_SELECTOR_PREREGISTRATION. Checkpoint-selection
384--447, fit-only audit 448--511, original model-validation 48--63, calibration,
final holdout, DAVIS, Kinetics, and official Kinetics 1,144 remain unread.

Gate 3C1B v0 completed and passed with exact replay on both independent
development partitions. On checkpoint-selection 384--447, retained 12-pixel
support improves 58.7956% -> 67.5119% (+8.7163 points), with equal-video 95%
CI [+4.1980,+12.4154]. On the one-shot fit-only audit 448--511, support improves
58.5657% -> 64.0106% (+5.4449 points), with CI [+2.8084,+13.5213]. Cycle-only
and identity-only controls remain weaker. Formal decision:
AUTHORIZE_GATE3C1C_FUTURE_ROLLOUT_PREREGISTRATION.

Gate 3C1C v0 completed and passed with exact replay on fit-only indices
448--511. The teacher-nearest shortlist oracle reduces mean future error from
34.4136 px native to 14.9708 px with coordinate plus deterministic four-level
memory; coordinate-only reaches only 33.2747 px. Full-state reduction is
+19.4428 px with video-cluster 95% CI [+17.4422,+20.4299]. Memory contributes
+18.3039 px beyond coordinate-only and is better on 98.4127% of failure videos.
Formal decision: AUTHORIZE_GATE3C1C_ORIGINAL_MODEL_VALIDATION_PREREGISTRATION.

Gate 3C1C v1 original model-validation completed and passed with exact replay
on indices 48--63. Query-closure shortlist support is 72.9167% within 12 px and
median commit error is 7.8908 px. Full state reduces future error by +20.6680 px
with video-cluster 95% CI [+15.8849,+27.8495]; memory contributes +19.0669 px
beyond coordinate-only and improves all 16 videos. Formal decision:
AUTHORIZE_GATE3C1D_CAUSAL_TOP1_SELECTOR_PREREGISTRATION. The result remains a
teacher-nearest oracle, not deployable top-1 selection. Calibration, final
holdout, and all external datasets remain locked.

Gate 3C1D v0 completed and failed with exact fresh-process replay on
checkpoint-selection 384--447. Candidate discrimination is real (AUC/AP
0.8274/0.4227; raw top-1 12-pixel support 35.8162%), but no frozen threshold
simultaneously meets coverage, precision, aggregate improvement, CI, and harmful
rate gates. Threshold 0.35 retains coverage and improvement but precision is only
60.81%; threshold 0.40 nearly reaches precision but falls below coverage and
improvement gates. Formal decision: STOP_GATE3C1D_TOP1_SELECTOR. Top-1 audit
448--511 and top-1 original-model confirmation 48--63 were not run. The failure
rejects the single-stage candidate probability plus scalar threshold policy, not
the query-closure shortlist or coordinate-plus-memory recovery mechanism. A new
independent data expansion is required before a redesigned top-1 v1 can be
formally confirmed.

Gate 3C2 v0 completed and passed. Exactly 512 additional raw-record-disjoint
Kubric videos were materialized into 32 shards after excluding all existing 512
identities. The frozen combined identity digest is
`ae7f8c4231dc81b52327020c43914def5ec6a4666bc374bd2a5539d1be5bcf37`;
all 55 selected source TFRecords independently rehash exactly and raw overlap is
zero. Renewed partitions are checkpoint 0--255, audit 256--383, and model
validation 384--511. Formal decision:
AUTHORIZE_GATE3C1D_V1_TWO_STAGE_TOP1_PREREGISTRATION. No renewed selector metric
has been read.

Gate 3C1D v1 completed and passed with exact replay on all three renewed,
raw-record-disjoint partitions. The checkpoint-frozen policy is support >=0.30,
predicted value >=1 px, and predicted harm <=0.20 after expected-distance top-1
ranking. Mean commit-error reductions are +3.7312 px on checkpoint 0--255,
+3.2180 px on fit-only audit 256--383, and +3.4864 px on model validation
384--511; all video-cluster CI lower bounds exceed +2.96 px. Model validation
has 28.0638% action coverage, 62.2739% 12-pixel action precision, 0.7252%
all-row harmful rate, and 97.6% nonnegative videos. Formal decision:
AUTHORIZE_GATE3C1E_CAUSAL_TOP1_FUTURE_ROLLOUT_PREREGISTRATION. These remain
failure-row commit metrics, not complete-video AJ/OA or a paper-baseline claim.

Gate 3C1E v0 completed and passed with exact replay. On renewed model-validation
384--511, coordinate plus deterministic four-level memory reduces mean visible
future error on frames 16--23 by +3.6673 px over all 1,379 failure rows, with
video-cluster 95% CI [+3.2492,+4.6715]. On the 387 frozen action rows, reduction
is +13.0150 px. Coordinate-only contributes only +0.0700 px; memory contributes
+3.5973 px beyond coordinate-only and is better on 96% of source videos. Formal
decision: AUTHORIZE_GATE3C1F_OFFICIAL_TAPVID_BENCHMARK_PREREGISTRATION.
A complete-video benchmark still requires a fully causal full-population entry
contract because the isolated natural-failure population used future GT only to
define the scientific audit set.

Gate 3C1F0 v0 completed and passed with exact replay. The 130-D causal HGB
entry model uses only observed CoTracker trajectory, visibility/confidence, and
four native memory/support levels. The frozen operating point is entry
probability >=0.93 and native joint probability <=0.02. Five-fold source-video
OOF gives 14.63% failure recall with 0% clean false apply and 100% precision;
independent fit-only validation gives 15.00%, 0%, and 100%. Formal decision:
AUTHORIZE_GATE3C1F0_RAW_DISJOINT_FULL_POPULATION_DATA. The exposed 16-video
complete-policy pilot remains design-only; its AJ CI crosses zero.

Gate 3C1F1 v0 is preregistered and not materialized. It will create exactly 128
new Kubric videos after excluding all 1,024 raw identities used by Gate 3C0 and
Gate 3C2. The frozen selected identity digest is
`85193d0aa6381c7d78442e85bf87750ce72a497995a92016001ca0cf51832f28`.
This is a data-identity gate only; no new population metric may be read before a
separate Gate 3C1F2 complete-population protocol is committed.
```

Read first:

```text
docs/ROUTED_COUNTERFACTUAL_STATE_RESTORER_GATE2_RESULT_2026-07-19.md
docs/ROUTED_CSRR_GATE2_FAILURE_DIAGNOSTIC_2026-07-19.md
docs/ROUTED_STATE_REINSTATEMENT_BASIN_GATE2_5_V0_PLAN_2026-07-19.md
docs/ROUTED_STATE_REINSTATEMENT_BASIN_GATE2_5_RESULT_2026-07-19.md
docs/ROUTED_GEOMETRY_PRESERVING_RELOCALIZATION_INTERFACE_V0_2026-07-19.md
docs/ROUTED_GEOMETRY_REPRESENTATION_AUDIT_GATE3A_V1_PLAN_2026-07-19.md
docs/ROUTED_GEOMETRY_REPRESENTATION_AUDIT_GATE3A_V1_RESULT_2026-07-19.md
docs/ROUTED_TEMPORAL_IDENTITY_DATA_GATE3C0_V0_PLAN_2026-07-19.md
docs/ROUTED_TEMPORAL_IDENTITY_DATA_GATE3C0_V0_RESULT_2026-07-19.md
docs/ROUTED_TEMPORAL_IDENTITY_FEATURE_PROBE_GATE3C1_V0_PLAN_2026-07-19.md
docs/ROUTED_TEMPORAL_IDENTITY_FEATURE_PROBE_GATE3C1_V0_RESULT_2026-07-19.md
docs/ROUTED_TEMPORAL_IDENTITY_TRAIN_CACHE_GATE3C1A_V0_PLAN_2026-07-19.md
docs/ROUTED_TEMPORAL_IDENTITY_TRAIN_CACHE_GATE3C1A_V0_RESULT_2026-07-19.md
docs/ROUTED_TEMPORAL_IDENTITY_SELECTOR_GATE3C1B_V0_PLAN_2026-07-20.md
docs/ROUTED_TEMPORAL_IDENTITY_SELECTOR_GATE3C1B_V0_RESULT_2026-07-20.md
docs/ROUTED_TEMPORAL_IDENTITY_FUTURE_ROLLOUT_GATE3C1C_V0_PLAN_2026-07-20.md
docs/ROUTED_TEMPORAL_IDENTITY_FUTURE_ROLLOUT_GATE3C1C_V0_RESULT_2026-07-20.md
docs/ROUTED_TEMPORAL_IDENTITY_FUTURE_ROLLOUT_GATE3C1C_V1_MODEL_VALIDATION_PLAN_2026-07-20.md
docs/ROUTED_TEMPORAL_IDENTITY_FUTURE_ROLLOUT_GATE3C1C_V1_MODEL_VALIDATION_RESULT_2026-07-20.md
docs/ROUTED_TEMPORAL_IDENTITY_TOP1_GATE3C1D_V0_PLAN_2026-07-20.md
docs/ROUTED_TEMPORAL_IDENTITY_TOP1_GATE3C1D_V0_RESULT_2026-07-20.md
docs/ROUTED_TEMPORAL_IDENTITY_TOP1_DATA_GATE3C2_V0_PLAN_2026-07-20.md
docs/ROUTED_TEMPORAL_IDENTITY_TOP1_DATA_GATE3C2_V0_RESULT_2026-07-20.md
docs/ROUTED_TEMPORAL_IDENTITY_TOP1_GATE3C1D_V1_PLAN_2026-07-20.md
docs/ROUTED_TEMPORAL_IDENTITY_TOP1_GATE3C1D_V1_RESULT_2026-07-20.md
docs/ROUTED_TEMPORAL_IDENTITY_CAUSAL_FUTURE_ROLLOUT_GATE3C1E_V0_PLAN_2026-07-20.md
docs/ROUTED_TEMPORAL_IDENTITY_CAUSAL_FUTURE_ROLLOUT_GATE3C1E_V0_RESULT_2026-07-20.md
docs/ROUTED_TEMPORAL_IDENTITY_ENTRY_GATE3C1F0_V0_PLAN_2026-07-20.md
docs/ROUTED_TEMPORAL_IDENTITY_ENTRY_GATE3C1F0_V0_RESULT_2026-07-20.md
docs/ROUTED_TEMPORAL_IDENTITY_FULL_POPULATION_DATA_GATE3C1F1_V0_PLAN_2026-07-20.md
```

## Active override — 2026-07-17

```text
Current active research mainline: Route-D x strong-backbone feasibility and prospective validation.
The corrected official-scale TAP-Vid-Kinetics gate is complete and frozen.
Do not quote any pre-correction Route-D closed-loop TAP position metric.
Do not use observed Kinetics results to tune any controller, threshold, fusion rule, candidate source, or guard.
The earlier ReEntry and weak-backbone paper status below is retained as historical project context, not the current execution priority.
```

Completed P0 audits:

```text
- official evaluator pinned to google-deepmind/tapnet commit 989a1fd62f7b2a3cf7f1c339bbde38e086e3a0fc;
- 100 randomized direct parity cases: zero difference;
- 100 randomized wrapper parity cases: zero difference;
- real Kinetics annotation parity case: zero difference;
- official GCS release zip verified by length, ETag/MD5, decompression, and member hashes;
- all 1,144 local materialized samples exactly matched, in order, to 1,144 of 1,147 release-CSV annotation groups;
- three unavailable CSV segments identified;
- auxiliary split-file discrepancy documented and excluded from identity authority;
- 15 pre-correction Route-D result artifacts marked superseded;
- controller and policy remain unchanged.
```

Decisive external gate completed:

```text
- corrected official-scale protocol SHA-256: 61f1458ee0e62b865cde85d9f77e765b79da567add8fc277fe10599b0db13155;
- all 10 shards and 1,144 materialized segments completed sequentially on attempt 1;
- closed-loop vs baseline AJ: +0.023043, paired 95% CI [+0.020567, +0.025569];
- closed-loop vs baseline delta average: +0.027530, paired 95% CI [+0.024800, +0.030260];
- closed-loop vs open-loop AJ: +0.010598, paired 95% CI [+0.008805, +0.012395];
- closed-loop vs open-loop delta average: +0.011488, paired 95% CI [+0.009494, +0.013465];
- primary preregistered decision: pass;
- official protocol and final integrity decision: pass;
- controller, policy, checkpoint, and configuration unchanged.
```

Paper-writing gate completed:

```text
- independent Route-D paper identity separated from the historical ReEntry paper;
- core abstract, introduction, method, corrected results, mechanism interpretation, and limitations drafted;
- novelty positioned as multi-threshold utility modeling, risk-calibrated sparse routing, and explicit closed-loop state feedback;
- current evidence judged promising but not yet sufficient for a confident CCF-B / CAS Zone 2 submission.
```

Absolute-performance decision:

```text
- independent MMP baseline AJ: 32.49;
- Route-D closed-loop AJ: 34.80;
- absolute improvement: +2.30 AJ points;
- relative improvement: approximately +7.09%;
- statistical confidence and mechanism evidence are strong;
- absolute tracker competitiveness remains weak relative to modern strong TAP systems;
- do not package the current 34.8 AJ system as SOTA, a benchmark leader, or a competitive full tracking system.
```

Current next gate:

```text
P0a completed: the new MUSR core model contract and tested implementation are in-tree. MUSR jointly models causal candidates, monotonic multi-threshold utility, catastrophic risk, abstention, and bounded multi-field state writeback; it is not a tree/threshold micro-tuning route.
P0b one-video feasibility passed on frozen Kubric validation video 0: the CoTracker3 true-streaming adapter exports native plus five causal query-memory local-correlation candidates, 64-dimensional candidate features, 32-dimensional state features, and the real online_track_feat state. Routing-disabled candidate-0 parity is exact (max_abs 0). With native visibility frozen, the GT-only coordinate oracle improves AJ from 25.59 to 34.24 (+8.65 points) and delta average from 36.22 to 48.79 (+12.57 points), exceeding the +3.0 one-video oracle gate. This is oracle headroom, not a learned result.
P0c completed: a hash-pinned identity-disjoint Kubric protocol permanently excludes the observed validation pilot (index 0), reserves validation indices 16--31 as final holdout, and qualifies the frozen candidate generator on validation indices 1--15. All 15 videos passed exact candidate-0/native parity; adapter replays at indices 1, 8, and 15 were bit-identical. With native visibility fixed, pooled coordinate-oracle AJ improves 28.59 -> 35.14 (+6.55 points) and delta average 42.03 -> 50.64 (+8.62 points). Every video is positive (minimum +4.22 AJ, median +6.40), and all 1/2/4/8/16-px hit-rate gains are positive. This is oracle headroom, not learned MUSR performance.
P0d completed with a negative selector gate: fit/model-validation/calibration caches are complete at 48/16/32 videos, with coordinate-oracle AJ headroom +6.08/+6.85/+6.84. Calibration was exported but not read. After native-safe initialization and train/deploy-aligned straight-through selection, the formal utility-aligned Stage-A selector (seed 17, exact replay) reaches only +0.017 AJ point on model validation with paired 95% CI [-0.009,+0.054]; 8/16 videos are positive. Delta improves +0.243 and severe-16px rate improves, but the +0.5 AJ selector gate and positive-CI gate fail. State-write training and calibration are forbidden.
P0e completed with a negative raw-representation gate: fit/model-validation raw-v1 caches are complete at 48/16 videos with 601-D candidate and 160-D state tokens. All candidate-coordinate hashes are unchanged and replay anchors 0/47/48/63 are bit-identical. The exact-replay seed-17 selector reaches -0.0149 AJ point with paired 95% CI [-0.0806,+0.0502], delta +0.129, and improved severe-16px rate; only 6/16 videos are positive. It is 0.0322 AJ point worse than structured-64. State-write training and calibration remain forbidden.
P0f completed with a negative temporal gate: the fixed six-frame causal rank-invariant selector passes padding, causality, permutation, zero-step parity, monotonicity, and exact seed-replay audits. Best epoch 2 reaches only +0.0041 AJ point with paired 95% CI [-0.0045,+0.0106], delta +0.0823, improved severe-16px rate, and 0.0798% harmful selection. Beneficial-event recall is only 1.8443%, and only 4/16 videos are positive. State-write training and calibration remain forbidden.
P0g-a completed and corrected before training: the formal 64-channel CMCP uses the preregistered nine-channel input (three causal correlations, all three pairwise differences, motion prior, and two evidence maps), contains 263,747 trainable parameters, and passes exact full-video native-safety and replay audits. The earlier 262,019-parameter interface count is superseded.
P0g-b completed: frozen float16 CoTracker feature-map caches are complete at fit/model-validation 48/16 videos. Native state is exact for every sidecar; float32 replay anchors 0/47/48/63 are bit-identical. Global worst fp16 reconstruction is max-abs 2.36e-4 with minimum cosine 0.99999940; fit-video three-memory correlation error is max 1.24e-4.
P0g-c completed with a partial mechanism success and formal gate failure. Exact-replay seed-17 CMCP produces a strong learned candidate pool: oracle AJ +20.0375, oracle delta +22.6947, paired oracle AJ CI [+18.0040,+21.7391], and all 16 videos positive. Direct top-1 reaches +1.5843 AJ and +1.8865 delta while reducing the 16px severe rate, but paired AJ CI [-2.5666,+4.5731] crosses zero and harmful non-native selection is 27.9968%, far above the 1% gate. MUSR, state writeback, calibration, and locked datasets remain forbidden.
P0h-a completed: the exact epoch-1 CMCP proposal generator is frozen and the 409,224-parameter local pairwise comparator passes its real fit-video interface gate. The full 64-point x 24-frame candidate-coordinate SHA-256 is c4cc84161d6522eb889c48c80cb5fb54673e4f446f54c697079ab5e4a9979300; independent runs reproduce all candidate and selection tensor hashes, and zero-step selection is exactly native.
P0h-b completed: frozen 88-D local-token caches are complete at fit/model-validation 48/16 videos. The 84-D static evidence is GT-free and the four dynamic causal-summary fields are zero placeholders. Candidate-coordinate combined SHA-256 values are 3f1be12649a8c4757c933c3f6d8c99b9f3d6bcd146262406557e139377ca57c2 (fit) and edf8ebe25cc11b04406efe27203a7d2bea4da8ce075160a27a11d8edda1fb0a1 (model-validation); replay anchors 0/47/48/63 are exact.
P0h-c completed with a safe, statistically positive, exact-replay result that fails only the preregistered magnitude gate. Best epoch 3 reaches +0.2449 AJ and +0.4962 delta with paired AJ CI [+0.1324,+0.3693], 15/16 videos positive, harmful non-native selection 0.3132%, and improved 16px tail. The frozen candidate oracle remains +20.0375 AJ. Beneficial-candidate recall is only 3.3362%, so the +0.5 AJ gate fails and MUSR/state writeback/calibration remain forbidden.
P0i-a completed: the 8,352-parameter rank-32 LMRA passes exact fit-video zero-step equivalence. Adapted and frozen feature maps are byte-identical; online reconstruction reproduces the formal P0h candidate-coordinate hash c4cc84161d6522eb889c48c80cb5fb54673e4f446f54c697079ab5e4a9979300, raw token hash, dynamic decisions, and selected coordinates in independent runs. The initialized joint trainable branch has 681,323 parameters while the native tracker remains frozen.
P0i-b completed with a formal exact-replay pass. Safety-first checkpoint selection retains epoch 2 at +0.5945 AJ and +0.8745 delta, paired AJ CI [+0.3910,+0.8171], all 16 videos positive, harmful non-native selection 0.8229%, and a 1.0870-point improvement in the 16px severe rate. Candidate oracle remains +20.2743 AJ. All preregistered gates pass.
P0j paused after the preregistered B/C component runs exposed that the full P0i joint model is not the strongest ablation; those runs remain provisional until exact replay and packaging. No P0j result is used to define the new recovery model.
P0k active: official strong-backbone alignment is now a hard prerequisite. The exact Meta CoTracker3 scaled-online checkpoint reaches 64.4408 AJ / 90.8941 OA / 77.1575 delta average on official DAVIS first-query with single_point=True. The public TAPNext++ AJ_RD implementation is reproduced with zero scalar difference over 100 randomized cases. A native-safe long-occlusion re-detection model is implemented with reliable/episodic appearance memory, global multi-anchor proposals, coordinate+visibility recovery, two-frame confirmation, abstention, and future-only bounded writeback. PointOdyssey is frozen at 24 fit scenes/240 events and 8 model-validation scenes/160 events; 7 internal-holdout and 13 test scenes remain locked. Formal sparse event-cache export is active. Official Kinetics 1,144 remains frozen and is not rerun.
```

Read first:

```text
docs/ROUTED_PAPER_CORE_DRAFT_V0_2026-07-17.md
docs/ROUTED_PAPER_READINESS_AND_EXPERIMENT_PLAN_2026-07-17.md
docs/ROUTED_STRONG_BACKBONE_FEASIBILITY_AUDIT_2026-07-17.md
docs/ROUTED_MUSR_KUBRIC_CACHE_PROTOCOL_V0_2026-07-17.md
docs/ROUTED_MUSR_KUBRIC_QUALIFICATION_RESULT_2026-07-17.md
docs/ROUTED_MUSR_SELECTOR_STAGEA_RESULT_2026-07-17.md
docs/ROUTED_MUSR_RAW_REPRESENTATION_V1_PLAN_2026-07-17.md
docs/ROUTED_MUSR_RAW_V1_STAGEA_RESULT_2026-07-17.md
docs/ROUTED_MUSR_TEMPORAL_REPRESENTATION_V0_PLAN_2026-07-17.md
docs/ROUTED_MUSR_TEMPORAL_V0_STAGEA_RESULT_2026-07-17.md
docs/ROUTED_MUSR_MULTI_MEMORY_PROPOSAL_V0_PLAN_2026-07-17.md
docs/ROUTED_CMCP_INTERFACE_RESULT_2026-07-17.md
docs/ROUTED_CMCP_FEATURE_CACHE_RESULT_2026-07-17.md
docs/ROUTED_CMCP_PROPOSAL_TRAINING_RESULT_2026-07-17.md
docs/ROUTED_CMCP_LOCAL_PAIRWISE_SAFETY_V0_PLAN_2026-07-17.md
docs/ROUTED_CMCP_PAIRWISE_SAFETY_INTERFACE_RESULT_2026-07-17.md
docs/ROUTED_CMCP_PAIRWISE_TOKEN_CACHE_RESULT_2026-07-17.md
docs/ROUTED_CMCP_PAIRWISE_SAFETY_TRAINING_RESULT_2026-07-17.md
docs/ROUTED_CMCP_LATE_METRIC_ADAPTER_V0_PLAN_2026-07-17.md
docs/ROUTED_CMCP_LMRA_INTERFACE_RESULT_2026-07-17.md
docs/ROUTED_CMCP_LMRA_TRAINING_RESULT_2026-07-17.md
docs/ROUTED_STRONG_BACKBONE_MUSR_ABLATION_V0_PLAN_2026-07-17.md
docs/ROUTED_SAFE_REDETECTION_V0_PLAN_2026-07-18.md
docs/generated/OFFICIAL_COTRACKER3_DAVIS_FIRST_ALIGNMENT_2026-07-18.json
docs/generated/OFFICIAL_TAPNEXTPP_AJRD_PARITY_2026-07-18.json
docs/generated/ROUTED_SAFE_REDETECTION_RUNTIME_ALIGNMENT_2026-07-18.json
docs/TAPVID_KINETICS_OFFICIAL_PROTOCOL_AUDIT_2026-07-16.md
docs/ROUTED_TREE_CONTROLLER_AUDIT_2026-07-16.md
```

## Historical status

Current project status as of 2026-07-05:

```text
Primary stable route: diagnostic / workshop-first ReEntry paper.
High-upside rescue route: gated SOTA-compatible ReEntry upgrade.
Do not treat the current evidence as a top-tier SOTA method paper yet.
```

Read first:

```text
docs/sota_reentry_upgrade_plan_2026-07-05.md
docs/paper_full_draft_v0_consistency_audit_2026-07-05.md
docs/paper_full_draft_v0_2026-07-04.md
docs/final_paper_tables_2026-07-04.md
docs/final_protocol_review_and_next_steps_2026-07-04.md
```

## SOTA ReEntry upgrade route opened as a gated rescue plan (2026-07-05)

Artifact:

```text
docs/sota_reentry_upgrade_plan_2026-07-05.md
```

Decision:

```text
Open a SOTA-compatible ReEntry route, but only behind strict gates:
SOTA protocol parity -> oracle headroom -> plug-in ReEntry -> true V26 metric-aware selector.
Do not directly swap in a SOTA tracker and report results without parity/oracle audits.
```

Current paper target remains diagnostic / workshop-first unless the gated route succeeds. The SOTA route is a high-upside rescue path, not the default claim.

Priority order:

```text
1. Preserve current diagnostic paper and claim boundaries.
2. Keep TrackOn2 first/input plug-in evidence as appendix/supplemental only.
3. Establish TrackOn2 strict strided/original parity before any main-table use.
4. Attempt TAPNext++ acquisition + 1-video smoke only if checkpoint/code are accessible.
5. Run visibility/re-entry oracle headroom before applying ReEntry to any SOTA base.
6. Promote V26 only if it beats V25 by a meaningful margin under strict protocol.
```

Non-negotiable rules:

```text
- Existing non-parity TrackOn2/TAPNext strided-original caches remain excluded from main tables.
- No SOTA result enters the paper without a protocol parity document.
- No ReEntry-on-SOTA run is worthwhile unless oracle headroom exists first.
- Current distance-filter V26 is insufficient; future V26 must be metric-utility-aware.
```

## Current paper position

The paper is no longer positioned as a straightforward performance-improvement / SOTA paper.

Correct positioning:

```text
ReEntry diagnoses re-entry visibility failures and shows that coordinate-preserving visibility calibration can improve re-entry-focused metrics, while exposing a systematic trade-off with standard TAP-Vid AJ/OA under stricter protocols.
```

Do not claim:

```text
official leaderboard submission
full TAP-Vid official benchmark result
universal improvement across protocols
V24 beats TrackOn2 on DAVIS
V24 is universally best
```

Use only:

```text
official-style local evaluation
local standard evaluation
diagnostic re-entry metric
coordinate-preserving visibility calibration
re-entry failure-mode analysis
```

## Paper improvement pass completed (2026-07-05)

Categories fixed:

```text
P0-A2: Honest baseline comparison
  - Headline +2.19 AJ is vs CoTracker3 offline, not vs stronger CoTracker3 baseline.
  - Against stronger baseline, V24 is approximately -0.05 AJ.
  - Abstract / Introduction / Results / Discussion / Conclusion were rewritten to avoid overclaiming.

P0-A1: Narrative repositioning
  - Paper reframed from improvement paper to diagnostic + method paper.
  - Contributions reordered around failure-mode diagnosis first.

P1-A3: Failure-mode analysis deepened
  - scripts/analyze_failure_mode_stats.py
  - docs/failure_mode_stats_2026-07-05.json
  - Method section now includes visibility lag / false visibility statistics.
```

Key failure-mode findings:

```text
DAVIS base: 34.5% lag >= 1 frame, 20.0% lag >= 6 frames, 31.6% any false visibility.
DAVIS override: 12.4% lag >= 1 frame, 5.1% lag >= 6 frames, 65.8% any false visibility.
RGB base: 74.6% lag >= 1 frame, 47.7% lag >= 6 frames, 19.2% any false visibility.
RGB override: 43.5% lag >= 1 frame, 26.5% lag >= 6 frames, 55.3% any false visibility.
```

Interpretation:

```text
The override channel reduces visibility lag but introduces false-visible risk. This supports channel-selective visibility calibration, but also explains why aggressive recovery can hurt standard AJ/OA.
```

## Paper full draft / LaTeX status

Artifacts:

```text
docs/paper_full_draft_v0_2026-07-04.md
docs/paper_full_draft_v0_consistency_audit_2026-07-05.md
paper/reentry_viscalibrator_tex/main.tex
paper/reentry_viscalibrator_tex/main.pdf
```

Current paper includes:

```text
Abstract
Introduction
Related Work
Method
Experiments
Results
Qualitative Analysis
Discussion and Limitations
Conclusion
Appendix A: Implementation Details
Appendix B: Additional Tables
```

Figures rendered:

```text
paper/reentry_viscalibrator_tex/figures/fig1_failure_mode.png/pdf
paper/reentry_viscalibrator_tex/figures/fig2_pipeline.png/pdf
paper/reentry_viscalibrator_tex/figures/fig3_davis_first_bar.png/pdf
paper/reentry_viscalibrator_tex/figures/fig4_rgb_full50_tradeoff.png/pdf
paper/reentry_viscalibrator_tex/figures/fig5_v25_threshold_sweep.png/pdf
```

Known submission-time tasks:

```text
1. Verify refs.bib author lists / arXiv IDs before submission.
2. Migrate to target venue template if submitting.
3. Decide whether to include TrackOn2 first/input plug-in evidence in appendix or supplemental.
4. Do not wait indefinitely for the SOTA rescue route if the target is workshop/arXiv.
```

## Frozen paper results

Source:

```text
docs/final_paper_tables_2026-07-04.md
```

### DAVIS first/input official-style local evaluation

```text
Protocol: TAPVid-DAVIS, 30 videos, 650 queries, first/input.

CoTracker3 baseline:
  AJ    64.8949
  OA    91.7983
  AJ_RD 0.3486

CoTracker3 offline:
  AJ    62.6566
  OA    88.1487
  AJ_RD 0.3142

TrackOn2:
  AJ    67.0406
  OA    92.0916
  AJ_RD 0.3714

ReEntry V24-DINOScore:
  AJ    64.8439
  OA    91.8851
  AJ_RD 0.3549
```

Interpretation:

```text
V24 improves over CoTracker3 offline by +2.1874 AJ, +3.7363 OA, +0.0407 AJ_RD, but only matches the stronger CoTracker3 baseline and remains below TrackOn2 on AJ.
```

### DAVIS strided/original official-style local evaluation

```text
Protocol: TAPVid-DAVIS, 30 videos, 5,882 queries, strided/original.

CoTracker3 offline:
  AJ    51.5385
  OA    92.1543
  AJ_RD 0.3870

ReEntry V1 default:
  AJ    50.7303
  OA    91.2730
  AJ_RD 0.4144

ReEntry V22Q default:
  AJ    50.8342
  OA    91.3439
  AJ_RD 0.4136

V25-safe threshold=0.80:
  AJ    51.5648
  OA    92.2106
  AJ_RD 0.3900
```

Interpretation:

```text
Default ReEntry improves AJ_RD but hurts AJ/OA. V25 protects AJ/OA but reduces AJ_RD gain to about +0.003.
```

### RGB-Stacking full50 local standard evaluation

```text
Protocol: TAPVid RGB-Stacking full50, 50 videos, 60,829 queries.

CoTracker3 offline:
  AJ    79.9345
  OA    91.6371
  AJ_RD 0.3617

ReEntry V1:
  AJ    79.4302
  OA    93.0758
  AJ_RD 0.4414

ReEntry V22Q:
  AJ    79.5756
  OA    93.0481
  AJ_RD 0.4400

ReEntry V24-DINOScore:
  AJ    79.5974
  OA    93.0389
  AJ_RD 0.4394
```

Interpretation:

```text
RGB full50 is strong evidence for re-entry/OA improvement, but the AJ decrease is systematic. Paired V24 AJ W/L/T = 11/39/0 and CI is negative.
```

## V26 current status

Distance-filter V26 has already been tried and is insufficient.

Best safe row found in existing outputs:

```text
outputs/paper_discovery_2026-06-27/reentry_v26_metric_aware/davis_strided_2d/v26_davis_strided_sweep_summary.json

label: v10.80_dist0.10
AJ_orig = 51.5671
OA_orig = 92.2151
AJ_RD   = 0.3901
```

Gain vs CoTracker3 offline:

```text
AJ    +0.0286
OA    +0.0608
AJ_RD +0.0031
```

Decision:

```text
Do not promote distance-filter V26 as a method. It is essentially V25-level.
Future V26 must be a true metric-aware utility selector, trained to keep only corrections with positive expected metric contribution.
```

## External baseline / SOTA status

### TrackOn2

Valid evidence:

```text
docs/first_input_trackon2_b2w_plugin_experiment_2026-06-29.md
```

TrackOn2 first/input bridge is parity-valid and strong:

```text
TrackOn2 first-input:
  AJ_RD_256 = 0.5444
  AJ_256    = 67.0406
  OA_256    = 93.0615

TrackOn2 base + B2-W16:
  AJ_RD_256 = 0.5513
  AJ_256    = 67.1750
  OA_256    = 93.2513
```

Use as appendix/supplemental plug-in evidence only.

Invalid for main table:

```text
Existing TrackOn2 strided/original cache is not parity-valid and must not be used as a main result.
```

### TAPNext / TAPNext++

Current status:

```text
TAPNext strided/original existing cache is not parity-valid.
TAPNext++ is a high-priority SOTA candidate but requires checkpoint/code acquisition and 1-video smoke before any claim.
```

## SOTA upgrade gates

See full plan:

```text
docs/sota_reentry_upgrade_plan_2026-07-05.md
```

Short version:

```text
Gate 0: Freeze current diagnostic paper.
Gate 1: Acquire SOTA candidate and export cache schema.
Gate 2: Protocol parity audit.
Gate 3: Visibility/re-entry oracle headroom audit.
Gate 4: Existing ReEntry plug-in on SOTA base.
Gate 5: True V26 metric-aware selector if plug-in fails but oracle headroom exists.
```

Main-paper upgrade success criterion:

```text
AJ >= SOTA base - 0.05
OA >= SOTA base - 0.05
AJ_RD >= SOTA base + 0.01
paired AJ CI must not be entirely negative
paired OA or AJ_RD should be positive / stable
protocol must be parity-valid
```

## Route decisions

### Route A — Current diagnostic paper

Status: active / stable.

```text
This is the default route. It can be submitted as workshop/arXiv once final writing is polished.
```

### Route B — SOTA-compatible rescue route

Status: opened, gated.

```text
Only becomes the main route if TrackOn2/TAPNext++ parity + oracle + plug-in/V26 succeed.
```

### Route C — Old FSPT student / pseudo-label / DINO-local routes

Status: closed.

Do not restart without a new written gate document.

Closed routes:

```text
DINOv2 whole-frame / multi-anchor retrieval
DINOv2 local feature pseudo-label rollout
CT-offline-centered local refiner
CT-offline-centered grid verifier
learned offset regression on frozen DINOv2 features
old FSPTTracker(video, query_points) SOTA pursuit
old P4 pseudo-label student route
broad threshold sweeps without a written audit question
```

## Engineering foundation

Package status:

```text
fspt/ package exists.
pyproject.toml uses setuptools.build_meta.
Path helpers and environment-variable roots exist.
Coordinate conversions and re-entry metrics have package wrappers.
python -m fspt.smoke passes.
```

Recent smoke result:

```text
python -m fspt.smoke -> ok
```

Pytest is currently unavailable in the environment:

```text
No module named pytest
```

This means tests were not run, not that they failed.

## Working rules

```text
1. Do not claim official leaderboard results.
2. Do not claim universal improvement.
3. Do not compare across protocols without explicit caveats.
4. Do not use non-parity SOTA caches in main tables.
5. Do not promote V26 unless it meaningfully beats V25.
6. Do not let the SOTA route block the diagnostic paper indefinitely.
7. Always write a result/audit doc before changing the paper claim.
```

## Immediate next action

If continuing the SOTA rescue route:

```text
1. Start TrackOn2 strict strided/original protocol parity audit.
2. In parallel, check TAPNext++ checkpoint/code availability and run 1-video smoke if possible.
3. For any SOTA candidate that passes parity, run oracle headroom before applying ReEntry.
```

If continuing paper submission:

```text
1. Keep the paper as diagnostic / workshop-first.
2. Add TrackOn2 first/input plug-in result as appendix/supplemental only if space allows.
3. Polish Discussion/Limitations around systematic AJ trade-off and non-leaderboard AJ_RD.
```

## TrackOn2 DAVIS strided/original parity audit completed (2026-07-05)

Artifact:

```text
docs/sota_protocol_parity_trackon2_davis_strided_original_2026-07-05.md
```

Decision:

```text
Stop TrackOn2 DAVIS strided/original rescue under the current exporter/evaluator path.
Do not proceed to ReEntry plug-in, V26 training, or 30-video main-table evaluation on this cache.
```

What was done:

```text
1. Added scripts/export_trackon2_davis_strided_original_cache.py.
2. Exported 1-video DAVIS strided/original TrackOn2 DINOv3 cache.
3. Ran geometry audit, standard/AJ_RD comparison, and visibility-oracle variants.
```

Key result on first DAVIS video (bike-packing, 219 strided queries):

```text
New TrackOn2 export:
  AJ_RD_256     = 0.4504
  AJ_256        = 31.0007
  OA_256        = 57.5208
  delta_avg_256 = 38.0565

Old TrackOn2 cache:
  AJ_RD_256     = 0.4515
  AJ_256        = 30.9450
  OA_256        = 57.2925
  delta_avg_256 = 38.0305

CoTracker3 offline same video:
  AJ_RD_256     = 0.3372
  AJ_256        = 53.6969
  OA_256        = 86.7983
  delta_avg_256 = 71.7221
```

Interpretation:

```text
The new exporter reproduces the old weak TrackOn2 strided/original cache, so the old cache was not merely stale/corrupt.
Query-frame geometry is clean (anchor mean 0.299px, max 0.827px), but long-horizon coordinate quality collapses.
GT visibility raises AJ_RD_256 from 0.4504 to 0.5508, but AJ_256 drops to 24.1730 and delta_avg_256 remains 38.0565.
Therefore the bottleneck is coordinate/long-horizon trajectory quality under this protocol, not only visibility.
```

Consequence:

```text
Because ReEntry preserves base coordinates and only changes visibility, ReEntry cannot rescue this TrackOn2 strided/original cache into a credible SOTA-base main result.
Keep TrackOn2 first/input plug-in evidence as appendix/supplemental only.
Move SOTA rescue attention to TAPNext++ acquisition / 1-video smoke or another parity-valid SOTA base.
Do not train V26 yet.
```

## Three-baseline ReEntry full-metric audit completed (2026-07-06)

Artifact:

```text
docs/three_baseline_reentry_full_metric_audit_2026-07-06.md
outputs/paper_discovery_2026-07-05/tapnextpp_smoke/three_baseline_reentry_recomputed_official_full.json
```

Decision:

```text
The reported three-baseline table is reproducible, but it is a diagnostic visibility-stress result, not a SOTA method result.
```

Clarification:

```text
TAPNext++ numbers are from the W=8 sliding-window offline cache, not the fully per-frame independent offline cache.
TrackOn2 numbers are from predictor.model.forward() offline_batch mode, not normal forward_online().
```

Key interpretation:

```text
CoTracker3: meaningful method evidence because coordinates are strong; ReEntry improves AJ/OA/AJ_RD.
TAPNext++ offline_w8: visibility recovery evidence; coordinates are weak, so not a normal TAPNext++ improvement claim.
TrackOn2 offline_batch: visibility failure evidence; coordinates are extremely weak, so not a usable SOTA-base result.
```

Safe claim:

```text
Coordinate-preserving visibility correction improves visibility-derived metrics across CoTracker3, TAPNext++ offline_w8, and TrackOn2 offline_batch stress baselines while leaving all coordinate thresholds unchanged.
```

Unsafe claim:

```text
ReEntry improves normal online TAPNext++ or TrackOn2 as competitive SOTA trackers.
```

Next technical route:

```text
Use TAPNext++ normal online coordinates with controlled visibility degradation/recovery if we want a stronger SOTA-compatible diagnostic.
Do not train V26 yet.
```

## TAPNext++ online controlled visibility-lag audit completed (2026-07-06)

Artifact:

```text
docs/tapnextpp_online_controlled_visibility_lag_audit_2026-07-06.md
outputs/paper_discovery_2026-07-05/tapnextpp_smoke/tapnextpp_online_controlled_visibility_lag_audit.json
scripts/eval_tapnextpp_controlled_visibility_lag.py
```

Decision:

```text
This is the strongest SOTA-compatible diagnostic so far: TAPNext++ normal online coordinates are fixed, only binary re-entry visibility is artificially delayed, and restoring visibility recovers AJ/OA/AJ_RD while all coordinate thresholds remain unchanged.
```

Key results:

```text
Original TAPNext++ online: AJ 65.85, OA 92.32, delta_avg 79.09, AJ_RD 0.5961.
Lag 2 restored gain: AJ +1.13, OA +1.28, AJ_RD +0.0815, delta_avg +0.00.
Lag 4 restored gain: AJ +2.76, OA +3.07, AJ_RD +0.1512, delta_avg +0.00.
Lag 8 restored gain: AJ +5.32, OA +5.85, AJ_RD +0.2642, delta_avg +0.00.
```

Claim boundary:

```text
Safe: visibility lag alone can materially harm TAPNext++ metrics even when coordinates are strong.
Unsafe: ReEntry already improves normal online TAPNext++ as a deployable method.
```

Next action:

```text
Modify/rerun TAPNext++ online export to save raw visibility logits or confidence. Current cache only stores binary visibility, so realistic confidence-threshold calibration cannot be tested yet.
Do not train V26 yet.
```

## TAPNext++ visibility confidence threshold sweep completed (2026-07-06)

Artifact:

```text
docs/tapnextpp_visibility_confidence_sweep_2026-07-06.md
outputs/paper_discovery_2026-07-05/tapnextpp_smoke/tapnextpp_visibility_threshold_sweep.json
scripts/eval_tapnextpp_visibility_threshold_sweep.py
```

Decision:

```text
Global visibility-threshold tuning is not a safe deployable ReEntry method for TAPNext++.
Lowering tau can improve AJ_RD slightly, but hurts AJ/OA too much.
Native tau=0.5 is near the best standard-metric operating point.
```

Key row:

```text
tau=0.20: AJ_RD +0.0157 vs tau=0.5, but AJ -3.05 and OA -3.46.
tau=0.40: AJ_RD +0.0057, but AJ -0.71 and OA -0.57.
```

Next action:

```text
Try re-entry-local confidence recovery with safety gates, not global threshold tuning.
Do not train V26 yet.
```

## TAPNext++ local confidence recovery audit completed (2026-07-06)

Artifact:

```text
docs/tapnextpp_reentry_local_conf_recovery_audit_2026-07-06.md
outputs/paper_discovery_2026-07-05/tapnextpp_smoke/tapnextpp_reentry_local_conf_recovery_sweep.json
scripts/eval_tapnextpp_reentry_local_conf_recovery.py
```

Decision:

```text
No local confidence-recovery configuration passes the deployable-method success gate.
Do not promote TAPNext++ confidence-based ReEntry as a method result.
```

Key findings:

```text
Success criterion: AJ >= native - 0.10, OA >= native - 0.10, AJ_RD >= native + 0.01.
Success count: 0.
Best under AJ/OA >= native -0.10 gives only AJ_RD +0.0017.
Best AJ_RD overall gives +0.0081 but costs AJ -0.61 and OA -0.50.
```

Consequence:

```text
TAPNext++ controlled visibility-lag remains strong diagnostic evidence, but simple global/local confidence rules do not produce a deployable TAPNext++ improvement.
Keep current paper diagnostic/workshop-first. Do not train V26 yet.
```

## Paper repositioned as improvement-without-SOTA (2026-07-06)

Artifact:

```text
docs/paper_reposition_improvement_not_sota_2026-07-06.md
```

Decision:

```text
The paper does not need to claim SOTA, but it must read as an improvement/advantage paper with non-weak metrics.
This is achievable by framing ReEntry as coordinate-preserving visibility calibration for re-entry failures.
```

Main usable results:

```text
DAVIS first/input CoTracker3 offline:
  AJ +1.94, OA +3.65, AJ_RD +0.0532. Main headline result.

RGB-Stacking full50:
  OA +1.40, AJ_RD +0.0777, AJ -0.34. Secondary robustness/trade-off result.

DAVIS strided/original V25 safe:
  AJ +0.026, OA +0.056, AJ_RD +0.003. Conservative safety result.
```

Main-table exclusion rule:

```text
Do not put rows with AJ below 50 in the main table unless explicitly labeled as stress/diagnostic.
TAPNext++ offline_w8 and TrackOn2 offline_batch go to appendix/diagnostic only.
```

Claim boundary:

```text
Safe: ReEntry improves re-entry visibility calibration under coordinate-preserving settings.
Unsafe: ReEntry improves normal online SOTA TAPNext++ / TrackOn2 or beats leaderboard SOTA.
```

Next action:

```text
Freeze result set and build writing package: final_main_tables.md, final_claim_boundary.md, paper_outline_improvement_position.md, abstract_and_intro_draft.md.
Do not train V26 unless a new paper branch is opened.
```

## True-base ReEntry requirement recorded (2026-07-06)

Artifact:

```text
docs/true_base_reentry_requirement_and_status_2026-07-06.md
```

Decision:

```text
Main improvement tables require real base tracker outputs plus the ReEntry module.
Stress/window-reset/offline-batch variants may only appear as diagnostic appendix rows.
```

Status:

```text
CoTracker3: PASS as main real-base improvement row.
TAPNext++: true-base confidence/logit module tested; no meaningful deployable gain, appendix/negative only.
TrackOn2: normal first/input cache exists but lacks confidence/logits; true-base module experiment requires new forward_online confidence exporter.
```

Next choice:

```text
Writing-first route: freeze CoTracker3-centered paper.
More-baseline route: export TrackOn2 normal forward_online with confidence/logits, then run same true-base threshold/local recovery audit.
```

## TrackOn2 true-base confidence smoke started (2026-07-06)

Artifact:

```text
docs/trackon2_true_base_conf_smoke_status_2026-07-06.md
scripts/export_trackon2_davis_first_input_conf_smoke.py
outputs/paper_discovery_2026-07-05/trackon2_true_base_conf/trackon2_davis_first_input_conf_smoke_1video.pt
```

Status:

```text
TrackOn2 normal forward_online logit/confidence export is technically possible.
The smoke successfully exports pred_vis_logit and pred_vis_conf.
```

But parity with the old TrackOn2 first/input bridge is not exact yet:

```text
old first-video: AJ_256 58.4117, OA_256 90.0588, AJ_RD_256 0.5257
new smoke:       AJ_256 57.0214, OA_256 88.1176, AJ_RD_256 0.4730
query/gt max_abs_diff 0.00382; visibility diff_rate 0.0551
```

Next action:

```text
Do not run 30-video export yet.
Fix exporter parity by using the old TrackOn2 first/input cache as query/GT/schema source and raw DAVIS only as video source.
Then rerun 1-video parity smoke.
```

## Experimental rigor rules recorded (2026-07-06)

Artifact:

```text
docs/experimental_rigor_rules_2026-07-06.md
```

Decision:

```text
No new full experiment may run before written plan + 1-video smoke + schema/alignment/parity checks.
Main paper rows require real base tracker output plus ReEntry module.
Stress/window-reset/offline-batch/non-parity caches are appendix/diagnostic only.
```

Current TrackOn2 rule:

```text
Do not run 30-video TrackOn2 confidence export yet.
First fix 1-video parity using the old TrackOn2 first/input cache as query/GT/schema source and raw DAVIS only as frame source.
```

## TrackOn2 true-base parity fix plan recorded (2026-07-06)

Artifact:

```text
docs/trackon2_true_base_parity_fix_plan_2026-07-06.md
```

Decision:

```text
The first TrackOn2 confidence smoke failed exact parity because the old bridge is in input256 space while the smoke used original-size video space.
Next smoke must preserve old cache query/GT/schema and run TrackOn2 on resized 256x256 video with model queries reconstructed from old query_points * 255.
```

Next action:

```text
Write scripts/export_trackon2_davis_first_input_conf_parity_smoke.py and run first-video parity only.
No 30-video export until parity passes.
```

## TrackOn2 M24 parity smoke completed but not passed (2026-07-06)

Artifact:

```text
docs/trackon2_true_base_parity_smoke_m24_result_2026-07-06.md
```

Result:

```text
M_i=24 was necessary and improved AJ_RD parity.
Schema parity passed exactly: query_points/gt_tracks/gt_visibility/original_size all match old cache.
But metric parity still does not pass: AJ -1.11 pp, OA -1.18 pp on first video.
```

Decision:

```text
Do not run 30-video TrackOn2 confidence export yet.
Do not run TrackOn2 ReEntry module yet.
Next step is to instrument the actual repo-native evaluation/evaluator path to save V_logit, rather than hand-reimplementing Predictor.forward.
```

## Next TrackOn2 step switched to repo-native logit instrumentation (2026-07-06)

Artifact:

```text
docs/next_step_trackon2_repo_native_logit_instrumentation_2026-07-06.md
```

Decision:

```text
Stop hand-written TrackOn2 forward reimplementation for parity.
The old TrackOn2 .npz cache was generated by repo-native evaluation/evaluator and already has perfect repo-native metric parity.
Next step is to instrument that same repo-native path to save visibility logits/confidence.
```

Gate:

```text
Run 1-video repo-native logit smoke only.
New .npz must contain tracks, visibility, visibility_logit, visibility_conf.
Old 000000.npz tracks/visibility must match new tracks/visibility exactly or near-exactly.
No 30-video export and no ReEntry module until this passes.
```

## TrackOn2 repo-native code review completed (2026-07-06)

Artifact:

```text
docs/trackon2_repo_native_code_review_next_step_2026-07-06.md
```

Code-level finding:

```text
Repo-native TAPVid dataloader resizes DAVIS to 256 and scales points by 256.
prepare_tapvid_data converts [t,y,x] to [t,x,y].
Predictor.forward_frame obtains v_logit but discards it; Predictor.forward returns only tracks and binary visibility.
```

Next action:

```text
Write scripts/export_trackon2_repo_native_logits_smoke.py using the repo-native dataloader/evaluator path, not raw-pkl reconstruction.
Run only first DAVIS sample and compare new .npz against old repo-native 000000.npz.
```

## TrackOn2 repo-native logit smoke completed with evaluation-aware parity (2026-07-06)

Artifact:

```text
docs/trackon2_repo_native_logit_smoke_result_2026-07-06.md
```

Result:

```text
New repo-native logit smoke .npz contains tracks, visibility, visibility_logit, visibility_conf.
Raw all-frame strict equality to old 000000.npz does not pass due a large invisible-frame track outlier and 2 visibility differences.
But bridged metric parity passes on first video: AJ +0.0209 pp, OA +0.0000 pp, delta_avg +0.0192 pp, AJ_RD_256 +0.0020.
GT-visible after-query visibility diff_count = 0.
```

Decision:

```text
Proceed to 30-video repo-native logit export only as a parity-preserving confidence augmentation check.
Do not run TrackOn2 ReEntry audits until 30-video full-cache parity passes.
```

## TrackOn2 repo-native logit full parity passed (2026-07-06)

Artifact:

```text
docs/trackon2_repo_native_logit_full_parity_result_2026-07-06.md
```

Result:

```text
30-video TrackOn2 repo-native confidence/logit export completed.
New unified cache: outputs/paper_discovery_2026-07-05/trackon2_true_base_conf/trackon2_davis_first_input_repo_native_conf_full.pt
Full metric parity passed: AJ -0.0066 pp, OA -0.0230 pp, delta_avg +0.0365 pp, AJ_RD_256 -0.0005 vs old TrackOn2 base.
```

Meaning:

```text
This proves the TrackOn2 true-base confidence/logit cache is valid.
It does not prove ReEntry improves TrackOn2 yet.
```

Next action:

```text
Run TrackOn2 true-base confidence threshold sweep and local re-entry recovery audit.
Only claim TrackOn2 improvement if AJ >= base -0.10, OA >= base -0.10, and AJ_RD >= base +0.01.
```

## CoTracker3 online state-level ReEntry smoke (2026-07-06)

Artifact:

```text
docs/cotracker3_online_state_reentry_smoke_2026-07-06.md
```

Finding:

```text
CoTracker3 online keeps online_vis_predicted and online_conf_predicted and copies them into the next window, so state-level visibility correction is architecturally possible.
```

Protocol caution:

```text
Existing cotracker3_baseline first/input cache was produced by CoTrackerPredictor(offline=False) full-sequence online-architecture evaluation, not by the true streaming CoTrackerOnlinePredictor loop. True streaming native does not parity-match the old cache on 1-video smoke, so state-writeback must be compared against a true-streaming native baseline, not the old baseline cache.
```

1-video smoke result:

```text
Aggressive state writeback worsened AJ/OA/AJ_RD.
Conservative state writeback gave only +0.0017 AJ_RD but -0.389 AJ and -0.532 OA vs streaming native.
```

Decision:

```text
Do not run full 30-video state-writeback yet. First build a clean true-streaming CoTracker3 native baseline and design stronger gates.
No CoTracker3 online state-level improvement claim.
```

## CoTracker3 online state-writeback integration audit (2026-07-06)

Artifact:

```text
docs/cotracker3_online_state_writeback_integration_audit_2026-07-06.md
```

Decision:

```text
Integration is structurally correct: state-writeback edits online_vis_predicted and online_conf_predicted, which CoTracker3 online copies into later vis_init/conf_init.
```

But current simple gate is not good enough:

```text
First-3-video subset:
output-only: AJ -0.0977, OA -0.1844, AJ_RD +0.0036
state-writeback: AJ -0.2713, OA -0.3523, AJ_RD +0.0048
```

Opened-frame audit:

```text
output-only GT-visible precision = 30.77%
state-writeback GT-visible precision = 31.25%
```

Conclusion:

```text
No online state-level improvement claim yet. Do not run 30-video state-writeback until stronger gates and overlap-only writeback pass a subset smoke.
```

## CoTracker3 true-streaming online V3 full30 result (2026-07-06)

Artifact:

```text
docs/cotracker3_true_streaming_v3_online_result_2026-07-06.md
```

Result:

```text
true-streaming native: AJ 65.2366, OA 90.8186, AJ_RD 0.3534
state-writeback V3:  AJ 65.2395, OA 90.8228, AJ_RD 0.3534
```

Delta:

```text
AJ +0.0029
OA +0.0043
delta_avg +0.0005
delta_4px -0.0034
AJ_RD +0.0000
```

Opened-frame audit:

```text
state-writeback opened 5 frames, 3 GT-visible and 2 GT-occluded, precision 60%.
```

Decision:

```text
V3 overlap-only soft state-writeback is safe/non-destructive but not meaningfully improving. Do not use as main result; appendix/diagnostic only. A real online state-level gain likely needs an appearance verifier.
```

## CoTracker3 V4 appearance-verifier audit/result (2026-07-06)

Artifacts:

```text
docs/cotracker3_true_streaming_v4_appearance_result_2026-07-06.md
outputs/paper_discovery_2026-07-05/cotracker3_v4_appearance_audit/davis_true_streaming_v4_appearance_audit.json
outputs/paper_discovery_2026-07-05/cotracker3_true_streaming_v4_appearance_subset_eval/full30_dino_laststrict0720/cotracker3_true_streaming_v4_appearance_full30_summary.json
```

Audit finding:

```text
DINO last_strict_candidate_cosine can improve numeric-pass candidate precision from 35.48% to about 71.4% at threshold 0.7204042673110962, but keeps only 7 candidates in the audit.
```

Integrated full30 result:

```text
V4 state-writeback opened 0 frames and exactly matches native metrics.
```

Decision:

```text
V4 is diagnostic only. It shows appearance verification can improve candidate precision, but the conservative threshold removes all state writebacks. No CoTracker3 online state-level improvement claim.
```

## CoTracker3 Online V5 re-detection search oracle smoke (2026-07-06)

Artifact:

```text
docs/cotracker3_online_v5_redetection_search_oracle_smoke_2026-07-06.md
```

Result:

```text
DINO dense grid search, first 3 DAVIS videos, 80 GT re-entry events, stride=8.
All events: top5 recall@8 = 68.75%, top20 recall@8 = 87.5%.
Native-invisible events: n=6, top5 recall@8 = 66.67%, top20 recall@8 = 83.33%.
```

Decision:

```text
Continue CoTracker3 online only as active re-detection candidate search + verifier + retracking. Do not continue visibility-only state-writeback tuning.
```

## CoTracker3 online V5-B hard-event candidate-pool oracle (2026-07-06)

Artifact:

```text
docs/cotracker3_online_v5b_hard_candidate_pool_oracle_result_2026-07-06.md
```

Result summary:

```text
First10 hard-event diagnostic, 40 events.
Native hard-event r8 = 45.0%.
local96_s8 top20 oracle r8 = 72.5%, r16 = 85.0%.
```

Selective oracle best rows:

```text
local96_s8_top20 selective: AJ +0.0856, OA +0.2038, AJ_RD +0.0050, AJ_RD_256 +0.0106
local64_s8_top20 selective: AJ +0.0662, OA +0.2038, AJ_RD +0.0045, AJ_RD_256 +0.0091
global_s16_top20 selective: AJ +0.0753, OA +0.2038, AJ_RD +0.0047, AJ_RD_256 +0.0149
```

Decision:

```text
There is moderate hard-event candidate-pool headroom, unlike V4. Do not train a complex verifier yet. Next step is lightweight verifier feasibility audit over candidate-level features. Continue only if simple verifier recovers a meaningful fraction of selective oracle.
```

## CoTracker3 online V5-C lightweight verifier feasibility (2026-07-06)

Artifact:

```text
docs/cotracker3_online_v5c_lightweight_verifier_result_2026-07-06.md
```

Dataset:

```text
40 hard events, 800 local96_s8 top20 candidates, 34 positives, positive rate 4.25%.
```

LOOV candidate-level result:

```text
Logistic AP 0.0500, AUC 0.4773
ExtraTrees AP 0.0449, AUC 0.5122
RandomForest AP 0.0442, AUC 0.5129
```

Decision:

```text
The lightweight verifier does not recover V5-B selective oracle headroom. Do not continue CoTracker3 online V5 with current single-frame DINO/top-k features. Keep as appendix/diagnostic unless switching to temporal candidate consistency, CoTracker internal features, or a stronger candidate generator/verifier.
```

## CoTracker3 online V5 posthoc component review (2026-07-06)

Artifact:

```text
docs/cotracker3_online_v5_posthoc_review_and_next_step_2026-07-06.md
```

Critical finding:

```text
V5-B selective oracle headroom is mostly visibility/opening, not coordinate replacement.
```

Component ablation first10:

```text
visibility-only oracle: AJ_RD +0.0027, AJ_RD_256 +0.0061
coordinate-selective only: AJ_RD +0.0004, AJ_RD_256 +0.0004
coordinate-selective + open accepted: AJ_RD +0.0031, AJ_RD_256 +0.0068
coordinate-selective + open all hard: AJ_RD +0.0050, AJ_RD_256 +0.0106
```

Decision:

```text
Do not continue current coordinate-candidate verifier. If continuing CoTracker3 online, the next and final diagnostic should be V5-D event-level visibility/opening verifier, with coordinate replacement only secondary.
```

## CoTracker3 online V5-D visibility/opening verifier (2026-07-06)

Artifact:

```text
docs/cotracker3_online_v5d_visibility_verifier_result_2026-07-06.md
```

Dataset:

```text
3021 event samples, 27 videos, y_safe16 positive rate 21.62%.
```

LOOV classifier result:

```text
Logistic AP 0.3679, AUC 0.6609
ExtraTrees AP 0.4136, AUC 0.6682
RandomForest AP 0.3792, AUC 0.6319
```

Metric finding:

```text
Safe thresholds preserve AJ/OA but yield negligible AJ_RD gain.
Aggressive thresholds improve AJ_RD_256 but damage AJ/OA too much.
```

Decision:

```text
Do not continue CoTracker3 online as a mainline under current V3/V4/V5 hand-engineered designs. Keep as appendix/diagnostic. Resume only with temporal verifier, CoTracker internal correlation, or a trained sequence-level re-detection module.
```

## CoTracker3 online V6-A1 temporal verifier (2026-07-06)

Artifact:

```text
docs/cotracker3_online_v6a1_temporal_verifier_result_2026-07-06.md
```

Dataset:

```text
1686 temporal event samples, 27 videos, t-2...t+4 window, future_safe16 positive rate 30.78%.
```

LOOV classifier result:

```text
Logistic AP 0.5168, AUC 0.6701
ExtraTrees AP 0.5640, AUC 0.6710
RandomForest AP 0.5628, AUC 0.6635
```

Best constrained metric rows:

```text
ExtraTrees first_recovery/scoremax thr 0.6006: AJ -0.0389, OA +0.0474, AJ_RD +0.0003, AJ_RD_256 +0.0012
RandomForest first_recovery/scoremax thr 0.5512: AJ -0.0893, OA -0.0003, AJ_RD +0.0007, AJ_RD_256 +0.0024
```

Raw aggressive rows reach AJ_RD_256 +0.004 to +0.005 but damage AJ/OA too much.

Decision:

```text
V6-A1 improves verifier ranking substantially but does not yet pass strict method success gate. One more targeted iteration is justified: V6-A2 damage-aware temporal verifier / utility objective. Do not add DINO until damage-aware objective is tested.
```

## CoTracker3 online V6-A2/V6-A3 utility and early-reentry review (2026-07-06)

Artifact:

```text
docs/cotracker3_online_v6a2_v6a3_review_result_2026-07-06.md
```

Key findings:

```text
V6-A2 useful-opening oracle: AJ +0.2890, OA +0.8621, AJ_RD +0.0061, AJ_RD_256 +0.0163.
But useful-opening positives are mostly not early re-entry frames: only 21/365 are GT re-entry first frames and 97/365 are within first 4 frames.
```

V6-A3 early-reentry oracle:

```text
early4 oracle: AJ +0.0976, OA +0.2622, AJ_RD +0.0032, AJ_RD_256 +0.0088
early8 oracle: AJ +0.1286, OA +0.3479, AJ_RD +0.0044, AJ_RD_256 +0.0120
```

V6-A3 verifier result:

```text
early4 target: Logistic AP 0.0516/AUC 0.3941; ExtraTrees AP 0.0401/AUC 0.3266; RandomForest AP 0.0482/AUC 0.4404.
early8 also does not yield useful constrained metric gains.
```

Decision:

```text
Stop native temporal score-only verifier sweeps. The AJ_RD-aligned early re-entry subset has oracle headroom, but current native temporal features cannot identify it. Next step, if continuing online, is V7-A: export CoTracker internal correlation/support-memory features for early re-entry verifier.
```

## CoTracker3 online V7-A1 internal correlation first10 result (2026-07-06)

Artifact:

```text
docs/cotracker3_online_v7a1_internal_corr_first10_result_2026-07-06.md
```

Result:

```text
V7-A0 first3 internal correlation optimism did not hold under first10 LOOV.
```

Exact-corr first10 LOOV:

```text
early4: native AP 0.0453/AUC 0.3594; exact_corr AP 0.0348/AUC 0.3439; all AP 0.0332/AUC 0.3028
early8: native AP 0.0640/AUC 0.3455; exact_corr AP 0.0495/AUC 0.3383; all AP 0.0535/AUC 0.3560
useful_open_t: native AP 0.1979/AUC 0.6030; exact_corr AP 0.1013/AUC 0.4017
```

Decision:

```text
Do not proceed to V7-B with handcrafted internal correlation summaries. AJ_RD-aligned early re-entry has oracle headroom, but current frozen handcrafted CoTracker3 features do not recover it across videos. If continuing online, next branch must be a learned sequence-level re-entry module or a stronger online tracker/memory architecture.
```

## CoTracker3 online V7-A1 internal correlation review (2026-07-06)

Artifact:

```text
docs/cotracker3_online_v7a1_internal_corr_review_2026-07-06.md
```

Key result:

```text
V7-A1 first10 no-exact: 688 events, 9 effective videos. Approx internal features did not show stable LOOV early4/early8 signal.
V7-A1 exact-corr stratified early8 smoke: exact corr features show pooled single-feature signal but still poor LOOV generalization.
```

Representative exact stratified early8 result:

```text
native-only LOOV AP 0.2913, AUC 0.2999
approx-internal-only LOOV AP 0.2554, AUC 0.3507
exact-corr-only LOOV AP 0.2509, AUC 0.3388
all LOOV AP 0.3037, AUC 0.3553
```

Decision:

```text
Do not proceed to V7-B verifier yet. Raw internal correlation magnitudes appear video/query distribution-shifted. Next step is V7-A2 normalized internal correlation features: per-video/per-query z-score, temporal rank, and relative/residual features.
```

## CoTracker3 online V7-A3 internal spatial search (2026-07-06)

Artifact:

```text
docs/cotracker3_online_v7a3_spatial_search_result_2026-07-06.md
```

Key result:

```text
r32 spatial search improves candidate recall relative to r64, but early8 positives already have native_r16 = 1.0 by construction. Search top5_r16 = 0.9583 and top5_r8 = 0.6458, but native_r16 = 1.0 and native_r8 = 0.8333.
```

LOOV search-stat classifier remains poor:

```text
early8 r32 search-stat LOOV AP 0.2364, AUC 0.2741.
```

Decision:

```text
Do not continue spatial candidate search as the main branch. Current CoTracker3 early-reentry headroom is visibility calibration at already-good native coordinates, not coordinate re-localization. Next step is V7-A4: export raw vis/conf component temporal features instead of only product score.
```

## CoTracker3 online V7-B0 raw vis/conf verifier (2026-07-07)

Artifact:

```text
docs/cotracker3_online_v7b0_raw_visconf_verifier_result_2026-07-07.md
```

Key result:

```text
Best branch: early4 + all_components.
AP 0.1183, AUC 0.6377.
Best constrained threshold 0.85:
AJ -0.0833, OA -0.0228, AJ_RD +0.0014, AJ_RD_256 +0.0035.
```

Refined threshold sweep over 0.78–0.90 found no successful threshold:

```text
success_count = 0
```

Decision:

```text
V7-B0 is the strongest CoTracker3 online near-miss but not a successful method. Continue only with targeted damage-aware gating: V7-B1 benefit-minus-damage raw vis/conf verifier. Avoid broad feature/model sweeps.
```

## CoTracker3 online V7-B1 damage-aware raw vis/conf verifier (2026-07-07)

Artifact:

```text
docs/cotracker3_online_v7b1_damage_aware_result_2026-07-07.md
```

Key result:

```text
V7-B1 is the first CoTracker3 online branch result that passes the full predefined gate.
Best constrained row:
lambda = 1.25, threshold = 0.0, accept = 135
AJ -0.0786, OA +0.0177, AJ_RD +0.0020, AJ_RD_256 +0.0046
```

OOF quality:

```text
benefit_oof AP 0.1183, AUC 0.6377
damage_oof AP 0.8380, AUC 0.6497
```

Decision:

```text
Promote V7-B1 to the main CoTracker3 online line. Next step is V7-B1-refine: local lambda/threshold refinement around successful regions, then V7-B2 final method variant if stable.
```

## CoTracker3 online V7-B1 refine and group robustness (2026-07-07)

Artifact:

```text
docs/cotracker3_online_v7b1_refine_robustness_2026-07-07.md
```

Key local refine result:

```text
190 local lambda/threshold settings tested.
122 pass the full metric gate.
Best row: lambda 1.35, threshold -0.01:
AJ -0.0576, OA +0.0410, AJ_RD +0.0020, AJ_RD_256 +0.0047.
```

Key group robustness result:

```text
Fixed candidates average around AJ_RD_256 +0.0040 across five event-balanced folds, but only 2/5 folds individually pass the full gate.
```

Decision:

```text
V7-B1 is a real positive local basin, not a single-threshold accident, but it is not uniformly robust across video groups. Next step: V7-B1.1 conservative operating policy, especially per-video top-k / accept-cap gating around lambda 1.35, threshold -0.01.
```

## CoTracker3 online V7-B1.2 damage-ceiling policy (2026-07-07)

Artifact:

```text
docs/cotracker3_online_v7b12_damage_ceiling_policy_result_2026-07-07.md
```

Key result:

```text
V7-B1.2 confirms damage_ceiling=0.7 as the best default operating policy.
Full30: AJ -0.0524, OA +0.0331, AJ_RD +0.0020, AJ_RD_256 +0.0047.
5-fold mean: AJ -0.0396, OA +0.0606, AJ_RD +0.00170, AJ_RD_256 +0.00406.
```

Conservative ablation:

```text
damage_ceiling=0.6 improves safe/bad ratio and AJ/OA but lowers mean AJ_RD_256 to +0.00372.
```

Decision:

```text
Proceed to V7-B2 final method packaging with damage_ceiling=0.7 as default and damage_ceiling=0.6 as conservative ablation. Do not continue per-video top-k as main policy.
```

## CoTracker3 online V8-A large-gain oracle/action audit (2026-07-07)

Artifact:

```text
docs/cotracker3_online_v8a_large_gain_oracle_audit_2026-07-07.md
```

Key result:

```text
V7-B2 learned visibility only gives AJ_RD_256 +0.0047.
Visibility-only oracle over many safe frames gives AJ_RD_256 about +0.030.
Coordinate+visibility oracle gives much larger headroom:
- early4 w4: AJ_RD_256 +0.0597
- early4 w8: AJ_RD_256 +0.0761
- early8 w8: AJ_RD_256 +0.0810
- useful_open_t w16: AJ_RD_256 +0.1055
```

Decision:

```text
Stop V7-B threshold/policy tuning as the main route. Large gains require coordinate + visibility recovery over post-reentry windows. Next step: V8-B candidate coordinate source audit.
```

## CoTracker3 online V8-B candidate coordinate source audit (2026-07-07)

Artifact:

```text
docs/cotracker3_online_v8b_candidate_coord_source_audit_2026-07-07.md
```

Key result:

```text
Best non-GT coordinate source is trackon2_dinov3_bridge.
Standalone: AJ +1.8041, OA +1.2730, AJ_RD +0.0180, AJ_RD_256 +0.0111.
With oracle event/window selection, TrackOn2 coordinate recovery reaches:
- useful_open_t w16: AJ_RD_256 +0.0354
- useful_open_t segment: AJ_RD_256 +0.0335
- early8 w16: AJ_RD_256 +0.0323
- early8 w8: AJ_RD_256 +0.0264
- early4 w8: AJ_RD_256 +0.0254
```

Decision:

```text
A viable non-GT coordinate source exists. Next step: V8-C candidate-aware recovery selector using TrackOn2 bridge coordinates, not further V7 visibility calibration.
```

## CoTracker3 online V8-C / CRRP-v2 experiment design (2026-07-07)

Artifact:

```text
docs/cotracker3_online_v8c_crrp_experiment_design_2026-07-07.md
```

Main decision:

```text
V8-C should not be framed as a gate, plugin, or CoTracker3+TrackOn2 ensemble.
The method is defined as CRRP-v2: Observation-Centric Counterfactual Re-entry Recovery Policy.
It models online TAP re-entry recovery as a finite-option, risk-constrained, causal action-selection problem over native and candidate trajectories.
```

Core algorithmic components:

```text
- causal re-entry event proposal,
- candidate trajectory provider,
- causal state/action feature construction,
- counterfactual action utility/risk labeling,
- utility/risk predictor,
- risk-constrained recovery action selection,
- optional observation-centric bridge,
- optional dual-hypothesis confirmation,
- optional state-level repair after output-level success.
```

Immediate next step:

```text
Implement V8-C0 causal anti-redundancy baseline ladder before training any learned policy.
Script target: scripts/eval_cotracker3_online_v8c0_causal_recovery_baselines.py
```

Stop condition:

```text
If CRRP cannot beat TrackOn2 standalone and best simple causal routing under video-heldout/causal evaluation, stop CRRP as the main route.
```

## CoTracker3 online V8-C causal-feature preflight correction (2026-07-07)

Finding:

```text
The V7-A4 event feature package contains non-causal feature fields: dt1/dt2/dt3/dt4 and post_* aggregates. meta_json also includes future labels such as future_safe16 and best_safe16_offset.
Therefore X/X_aug and existing V7 benefit/damage OOF scores must not be used as strict online policy features for V8-C0/C1.
```

Allowed next-step rule:

```text
V8-C0 causal baselines must be built directly from causal fields in native/candidate caches and event meta history fields: low_run, invis_run, last_visible_t, native_visible_t, score_t, native/candidate visibility at <=t, tracks at <=t, motion and relation features at <=t.
```

Additional observation:

```text
TrackOn2 candidate_visible at the native event_t is only about 23.2% across 1686 event proposals, so candidate_visible-only policies may have limited recall. Low-confidence candidate mining and motion-consistency policies should be tested in V8-C0, but separated from oracle/leaky diagnostics.
```

## CoTracker3 online V8-C0 strict-causal baseline result (2026-07-07)

Artifact:

```text
docs/cotracker3_online_v8c0_causal_baseline_result_2026-07-07.md
```

Script:

```text
scripts/eval_cotracker3_online_v8c0_causal_recovery_baselines.py
```

Report:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v8c0_causal_recovery_baselines/v8c0_causal_baseline_report.json
```

Key result:

```text
Best strict-causal core policy: all_events_w16_candidate_visible.
Meaning: every native low/invisible event enters a 16-frame recovery mode, and each frame is repaired only when TrackOn2 candidate visibility is true at that frame.
No X/X_aug, no V7 OOF, no oracle labels, no GT-visible forcing are used.

Delta vs native:
AJ +0.0753, OA +0.9598, AJ_RD +0.0162, AJ_RD_256 +0.0286.
```

Comparison:

```text
TrackOn2 standalone AJ_RD_256 delta: +0.0111.
V8-C0 best AJ_RD_256 delta: +0.0286.
However TrackOn2 standalone is still much stronger on overall AJ (+1.8041 vs V8-C0 +0.0753), so V8-C0 is a re-entry recovery layer, not a globally stronger tracker.
```

Robustness caution:

```text
Per-video AJ_RD_256 deltas for all_events_w16_candidate_visible: 15 positive, 5 negative, 5 unchanged.
Negative videos include camel, shooting, breakdance.
```

Decision:

```text
Do not jump directly to learned CRRP.
Next: V8-C0.1 robustness and risk-control audit for all_events_w8/w16_candidate_visible, including explicit causal event-proposal rebuild, per-video deltas, and negative-video analysis.
```

## CoTracker3 online V8-C0.1 recovery risk audit result (2026-07-07)

Artifact:

```text
docs/cotracker3_online_v8c01_recovery_risk_audit_result_2026-07-07.md
```

Script:

```text
scripts/eval_cotracker3_online_v8c01_recovery_risk_audit.py
```

Report:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v8c01_recovery_risk_audit/v8c01_recovery_risk_audit_report.json
```

Key causal proposal verification:

```text
Rebuilt event list from native cache exactly matches old meta event list:
n_meta = 1686, n_rebuilt = 1686, intersection = 1686, Jaccard = 1.0.
```

Key result:

```text
dist_nc_le64_w16_candidate_visible:
AJ +0.0810, OA +0.9701, AJ_RD +0.0162, AJ_RD_256 +0.0286.

all_w16_candidate_visible gives the same AJ_RD_256 but slightly lower AJ/OA.
```

Default recommendation after risk audit:

```text
dist_nc_le64_w8_candidate_visible as safer default:
AJ +0.0964, OA +0.9462, AJ_RD +0.0153, AJ_RD_256 +0.0274,
positive/negative/zero videos = 16/4/5.

Use dist_nc_le64_w16_candidate_visible as high-gain ablation:
AJ_RD_256 +0.0286,
positive/negative/zero videos = 15/5/5.
```

Risk audit conclusion:

```text
Simple score/speed/distance filters do not eliminate negative videos.
Negative videos are driven by false candidate visibility and fine-threshold degradation rather than catastrophic 16px damage.
Next step should be V8-C0.2 fine-risk verifier dataset, not another broad threshold sweep.
```

## CoTracker3 online V8-C0.2 / V8-C0.3 fine-risk verifier dataset and learnability (2026-07-07)

Artifact:

```text
docs/cotracker3_online_v8c02_v8c03_fine_risk_verifier_result_2026-07-07.md
```

Dataset script:

```text
scripts/build_cotracker3_online_v8c02_fine_risk_verifier_dataset.py
```

Verifier script:

```text
scripts/train_cotracker3_online_v8c03_fine_risk_verifier.py
```

Dataset:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v8c02_fine_risk_verifier_dataset/v8c02_fine_risk_dist_nc_le64_w8.npz
```

Key dataset stats:

```text
n_samples = 1456 touched candidate-visible frames.
good_rate = 0.4444.
false_visible_rate = 0.2493.
candidate_worse_px_rate = 0.3180.
bad_strict_false_or_worse_rate = 0.5673.
```

Key learnability results under video-level LOOV:

```text
candidate_good: RandomForest AP 0.7763, AUC 0.7829.
false_visible: LogisticRegression AP 0.5182, AUC 0.7979.
bad_strict_false_or_worse: best AUC only about 0.577.
utility regression is weak: ExtraTrees R2 0.0532.
```

Decision:

```text
Useful-frame and false-visible signals are learnable, but strict bad detection and direct utility regression are weak.
Do not claim verifier helps until OOF scores are applied back to trajectories.
Next: V8-C0.4 OOF verifier application to evaluate actual AJ/OA/AJ_RD/AJ_RD_256.
```

## CoTracker3 online V8-C0.4 OOF fine-risk verifier application (2026-07-07)

Artifact:

```text
docs/cotracker3_online_v8c04_apply_fine_risk_verifier_result_2026-07-07.md
```

Script:

```text
scripts/eval_cotracker3_online_v8c04_apply_fine_risk_verifier.py
```

Report:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v8c04_apply_fine_risk_verifier/v8c04_apply_fine_risk_verifier_report.json
```

Sanity check:

```text
accept_all_baseline exactly reproduces the V8-C0.1 default:
AJ +0.0964, OA +0.9462, AJ_RD +0.0153, AJ_RD_256 +0.0274.
```

Best verifier-filtered AJ_RD_256 row:

```text
utility_et_ge_-0.3431:
AJ +0.1183, OA +0.9407, AJ_RD +0.0154, AJ_RD_256 +0.0275,
positive/negative/zero = 16/4/5.
```

Interpretation:

```text
Verifier gives only +0.0001 AJ_RD_256 over the simple baseline. It slightly reduces false-visible and bad rates but does not change negative-video count.
```

Robust verifier tradeoff:

```text
good_rf_minus_false_lr_ge_-0.2709 reduces negative videos to 2 but drops AJ_RD_256 to +0.0162, below the +0.020 target.
```

Decision:

```text
Do not promote learned fine-risk verifier as the default method.
Default remains dist_nc_le64_w8_candidate_visible.
High-gain ablation remains dist_nc_le64_w16_candidate_visible.
Next step: V8-C0.5 method packaging and robustness table.
```


## CoTracker3 online V8-C0.5 final recovery method packaging (2026-07-07)

Artifact:

```text
docs/cotracker3_online_v8c05_final_recovery_method_2026-07-07.md
```

Report:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v8c05_final_recovery_method/v8c05_final_recovery_method_table.json
```

Decision:

```text
Promote CVRRM default: dist_nc_le64_w8_candidate_visible.
Default result: AJ +0.0964, OA +0.9462, AJ_RD +0.0153, AJ_RD_256 +0.0274, positive/negative/zero = 16/4/5.
High-gain ablation: dist_nc_le64_w16_candidate_visible, AJ_RD_256 +0.0286, positive/negative/zero = 15/5/5.
Do not promote learned fine-risk verifier as default.
Next: V8-C1 cross-candidate / cross-native generalization audit; state-level repair only if cross-source evidence holds.
```

## CoTracker3 online V8-C1 cross-candidate generalization audit (2026-07-07)

Artifact:

```text
docs/cotracker3_online_v8c1_cross_candidate_generalization_result_2026-07-07.md
```

Script:

```text
scripts/v8c1_cross_candidate.py
```

Report:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v8c1_cross_candidate_generalization/v8c1_cross_candidate_generalization_report.json
```

Key result:

```text
TrackOn2 bridge remains the only candidate source that reaches the +0.020 AJ_RD_256 target.
CVRRM default with TrackOn2 bridge: AJ_RD_256 +0.0274.
CVRRM high-gain with TrackOn2 bridge: AJ_RD_256 +0.0286.
```

Other candidates:

```text
old_cotracker3_online_bridge default: AJ_RD_256 +0.0065.
old_cotracker3_offline_bridge default: AJ_RD_256 +0.0039.
trackon2_dinov3_davis excluded by strict alignment check.
```

Decision:

```text
CVRRM is not fully candidate-agnostic. It requires a strong re-entry candidate provider.
Do not start state-level repair yet.
Next: V8-C1.1 inspect/generate stronger aligned candidate caches such as TAPIR/LocoTrack/TAPNext if available; otherwise frame the method as TrackOn2-assisted CVRRM for CoTracker3 online re-entry recovery.
```


## CoTracker3 online V8-C1.1 TAPNext++ candidate probe (2026-07-07)

Artifact:

```text
docs/cotracker3_online_v8c11_tapnextpp_candidate_probe_2026-07-07.md
```

Key finding:

```text
Online-style TAPNext++ caches, after native metadata bridge, reach CVRRM w8 AJ_RD_256 +0.0204 and w16 +0.0209. This means CVRRM is not only TrackOn2-specific, but TrackOn2 bridge remains the best candidate provider. Offline TAPNext++ caches are harmful.
```

Decision:

```text
Do not start state-level repair yet. Next: V8-C1.2 formalize TAPNext++ metadata-bridge candidate evaluation with a clean JSON report and final cross-candidate table.
```


## CoTracker3 online V8-C1.2 TAPNext++ metadata-bridge candidate evaluation (2026-07-07)

Artifact:

```text
docs/cotracker3_online_v8c12_tapnextpp_bridge_eval_result_2026-07-07.md
```

Report:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v8c12_tapnextpp_bridge_eval/v8c12_tapnextpp_bridge_eval_report.json
```

Key result:

```text
Online-style TAPNext++ metadata-bridge candidates reach CVRRM w8 AJ_RD_256 +0.0204 and w16 +0.0209, crossing the +0.020 target. TrackOn2 bridge remains stronger: w8 +0.0274, w16 +0.0286. Offline TAPNext++ candidates are harmful.
```

Decision:

```text
CVRRM is not only TrackOn2-specific, but it requires a strong online re-entry candidate provider. Keep TrackOn2 bridge as default; include online-style TAPNext++ metadata-bridge as diagnostic cross-candidate ablation. Next: update final package/candidate-source table.
```


## CoTracker3 online V8-C1.3 candidate-source final table (2026-07-07)

Artifact:

```text
/gemini/code/FSPT/docs/cotracker3_online_v8c13_candidate_source_table_2026-07-07.md
```

Report:

```text
/gemini/code/FSPT/outputs/paper_discovery_2026-07-05/cotracker3_online_v8c13_candidate_source_table/v8c13_candidate_source_table.json
```

Decision:

```text
Default remains CVRRM + TrackOn2 bridge w8. TAPNext++ online-style metadata bridge is added as supporting cross-candidate evidence. Old CoTracker candidates are weak; TAPNext++ offline is harmful; OOF verifier remains diagnostic only. Next: final paper packaging or V8-C2 state-level repair feasibility precheck.
```


## CoTracker3 online V8-C2.0 state writeback feasibility precheck (2026-07-07)

Artifact:

```text
docs/cotracker3_online_v8c20_writeback_precheck_result_2026-07-07.md
```

Key finding:

```text
coords_only writeback is insufficient. coords_visconf_high shows positive AJ_RD_256 signal on 4-video and 8-video subsets, slightly exceeding output-level CVRRM on the 8-video subset (+0.0386 vs +0.0360 AJ_RD_256), but it worsens some negative videos and can reduce AJ.
```

Decision:

```text
Do not promote state writeback yet. Next: V8-C2.1 make full30 writeback script safe with unique output naming, mode selector, per-video deltas, then run full30 coords_visconf_high.
```


## CoTracker3 online V8-C2.1 controlled state writeback full30 (2026-07-07)

Artifact:

```text
docs/cotracker3_online_v8c21_controlled_state_writeback_full30_result_2026-07-07.md
```

Report:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v8c21_full_hook_eval/v8c21_full_report.json
```

Key result:

```text
State writeback prob=0.80: AJ +0.1310, OA +0.9394, AJ_RD +0.0151, AJ_RD_256 +0.0275.
State writeback prob=0.90: AJ +0.0401, OA +0.8578, AJ_RD +0.0150, AJ_RD_256 +0.0285.
Output-level default w8 remains +0.0274 AJ_RD_256; high-gain w16 remains +0.0286.
```

Decision:

```text
Do not promote state writeback as main method. It is fragile, mixed per-video, and does not beat output-level high-gain w16. Main method remains output-level CVRRM + TrackOn2 bridge, dist<=64, W=8. Next: V8-C3 final paper-style packaging.
```


## CoTracker3 online V8-C3.0 DAVIS dual-metric protocol audit (2026-07-07)

Artifact:

```text
/gemini/code/FSPT/docs/cotracker3_online_v8c30_davis_dual_metric_audit_2026-07-07.md
```

Report:

```text
/gemini/code/FSPT/outputs/paper_discovery_2026-07-05/cotracker3_online_v8c30_davis_dual_metric_audit/v8c30_davis_dual_metric_audit_report.json
```

Decision:

```text
Use dual tables: standard TAP-Vid-like metrics and re-entry recovery metrics. AJ_RD_256 is valid as same-baseline failure-mode analysis, but not a substitute for original CoTracker3 main-table protocol. Main method remains CVRRM + TrackOn2 w8. Next: official-protocol gap audit only if making direct original-paper Table-1-style claims.
```


## CoTracker3 online V8-C3.1 evaluator parity audit (2026-07-07)

Artifact:

```text
docs/cotracker3_online_v8c31_eval_parity_audit_2026-07-07.md
```

Report:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v8c31_eval_parity/v8c31_eval_parity_report.json
```

Key result:

```text
Project standard metrics and CoTracker backup official metric entry are exactly identical on native, TrackOn2, and CVRRM dist<=64 w8 records: max_abs_diff_all_variants = 0.0. Query-first audit passes: 650/650 valid, no prior visible frames.
```

Decision:

```text
Current DAVIS standard table is TAP-Vid metric-compatible on cached true-streaming DAVIS records. AJ_RD/AJ_RD_256 remain valid supplementary re-entry metrics, not original main-table replacements. Next: V8-C4 final paper-style package with dual tables and protocol caveats, unless direct Table-1-style claims require broader dataset/protocol reproduction.
```


## CoTracker3 online V8-C3.1 official evaluator parity audit (2026-07-07)

Artifact:

```text
/gemini/code/FSPT/docs/cotracker3_online_v8c31_official_evaluator_parity_2026-07-07.md
```

Report:

```text
/gemini/code/FSPT/outputs/paper_discovery_2026-07-05/cotracker3_online_v8c31_official_evaluator_parity/v8c31_official_evaluator_parity_report.json
```

Decision:

```text
Metric formula parity passes against local CoTracker backup official evaluator; query-first check passes; local evaluator/export both use add_support_grid=False and 256 resize. Current DAVIS standard metrics can be framed as TAP-Vid-DAVIS-first metric-compatible reproduction, but not full original-paper multi-dataset Table-1 parity. Next: V8-C4 final paper-style packaging with dual tables.
```


## CoTracker3 online V8-C4 final paper-style story package (2026-07-07)

Artifact:

```text
/gemini/code/FSPT/docs/cotracker3_online_v8c4_final_paper_story_2026-07-07.md
```

Report:

```text
/gemini/code/FSPT/outputs/paper_discovery_2026-07-05/cotracker3_online_v8c4_final_paper_story/v8c4_final_paper_story.json
```

Decision:

```text
CVRRM final story is ready for paper-style writing. Main method: output-level CVRRM + TrackOn2 bridge, dist<=64, W=8. It improves standard DAVIS-first metric-compatible metrics and re-entry metrics. Learned verifier and state writeback remain diagnostic/future work. Next choose benchmark expansion or candidate-provider expansion.
```


## V8-C4 final paper-style story packaging

Artifact:

```text
/gemini/code/FSPT/docs/v8c4_final_paper_story.md
```

Report:

```text
/gemini/code/FSPT/outputs/paper_discovery_2026-07-05/v8c4_story/v8c4_final_paper_story.json
```

Decision:

```text
Main story is now frozen for DAVIS: CVRRM + TrackOn2 w8 output-level, dual tables, TAP-Vid-DAVIS-first metric-compatible reproduction, AJ_RD_256 as supplementary re-entry metric. Next: write final paper sections or expand to more datasets.
```


## V9-A0 ReEntryBeliefTrack design spec (2026-07-08)

Artifact:

```text
docs/v9a0_reentry_belieftrack_design_spec_2026-07-08.md
```

Summary:

```text
CVRRM is reinterpreted as a hard-coded special case of a belief-based Track/Coast/Reacquire model. The next step is not full tracker training, but V9-A1: a learned recovery controller over existing native/candidate caches to test whether modelizing the rule can beat the strong CVRRM baseline.
```


## V9-A0 ReEntryBeliefTrack design spec (2026-07-08)

Artifact:

```text
docs/v9a0_reentry_belieftrack_design_spec_2026-07-08.md
```

Decision:

```text
CVRRM is frozen as rule-based teacher/baseline. For A-level ambition, continue as ReEntryBeliefTrack: Track/Coast/Reacquire phase, uncertainty-aware belief prediction, identity-aware multi-hypothesis reacquisition, and safe memory write. Next: V9-A1 controller-only prototype; it must beat CVRRM before scaling.
```


## V9-A0 addendum and V9-A1 design (2026-07-08)

Artifacts:

```text
docs/v9a0_reentry_belieftrack_addendum_2026-07-08.md
docs/v9a1_controller_calibration_aware_prototype_design_2026-07-08.md
```

Decision:

```text
V9-A route is narrowed: do not build full ReEntryBeliefTrack immediately. First run V9-A1 controller-only + calibration-aware prototype using existing native/candidate caches. It must beat CVRRM on heldout trajectory-level metrics and improve risk/calibration before scaling.
```


## V9-A1 controller calibration-aware prototype result

Artifacts:

```text
outputs/paper_discovery_2026-07-05/v9a1_controller_calibration_aware_prototype/v9a1_report.json
docs/v9a1_controller_calibration_aware_prototype_result_2026-07-08.md
```

Decision:

```text
See V9-A1 result doc for pass/fail judgement.
```


## V9-A1 threshold sweep audit

Artifact:

```text
outputs/paper_discovery_2026-07-05/v9a1_controller_calibration_aware_prototype/v9a1_threshold_sweep.json
```

Decision:

```text
V9-A1 tabular controller does not beat CVRRM accept-all on AJ_RD_256 after threshold sweep. It can improve AJ/OA and reduce negative videos at stricter thresholds, but loses re-entry recovery. Do not scale full ReEntryBeliefTrack from controller-only evidence; next step requires additional structure/features.
```


## V9-A2 anchor- and uncertainty-aware reacquisition design

Artifacts:

```text
docs/v9a2_anchor_uncertainty_reacquisition_design_2026-07-08.md
docs/superpowers/plans/2026-07-08-v9a2-anchor-uncertainty-reacquisition.md
```

Decision:

```text
V9-A2 will not continue threshold tuning from V9-A1. It adds new information: anchor identity similarity and uncertainty-aware consistency. Start with deterministic RGB patch features aligned to V8-C0.2 touched frames; DINO/CLIP are deferred until patch alignment is proven.
```


## V9-A2.1 stratified feature smoke and W8/W16 action-space audit

Artifacts:

```text
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_feature_smoke_stratified.json
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_w8_w16_action_space_audit.json
docs/v9a2_action_space_audit_result_2026-07-08.md
```

Decision:

```text
Stratified smoke succeeded across 25 DAVIS videos: 49 rows, 69 features, finite_rate=1.0, valid_p05=1.0. W16 strictly extends W8 by 557 rows with 118 candidate-good extension opportunities and very low damage16 rate (~0.18%). V9-A2 should use W8+W16 dynamic horizon/action selection rather than W8-only filtering.
```


## V9-A2.2 action-space oracle upper-bound audit

Artifacts:

```text
scripts/v9a2_action_space_oracle_audit.py
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_action_space_oracle_audit.json
docs/v9a2_action_space_oracle_audit_result_2026-07-08.md
```

Decision:

```text
V9-A2.2 passes. W8+selective W16 oracle reaches AJ_RD_256 Δ +0.0321 versus W8 +0.0274 and W16 +0.0286, with Pos/Neg/Zero 17/3/5. Event-level W16 utility oracle also reaches +0.0320. This validates dynamic horizon/action selection as the next learnable target.
```


## V9-A2.3a feature builder correction and label overlap audit

Artifacts:

```text
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_feature_smoke_stratified_v2.json
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_label_overlap_audit.json
docs/v9a2_feature_builder_correction_result_2026-07-08.md
```

Decision:

```text
V9-A2.3a passes. Feature builder no longer clips coordinates before patch extraction; explicit oob and anchor reliability features are included. Stratified smoke v2: 49 rows, 25 videos, feature_dim=112, finite_rate=1.0. Label overlap audit shows candidate_good and utility_positive are identical under current labels (Jaccard=1.0), so candidate_good should be the main target and utility_positive should not be treated as independent supervision.
```


## V9-A2.3a2 preocc anchor strict-fix

Artifacts:

```text
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_feature_smoke_stratified_v3.json
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_preocc_strict_anchor_audit.json
docs/v9a2_preocc_anchor_strict_fix_result_2026-07-08.md
```

Decision:

```text
V9-A2.3a2 passes. Preocc anchor now searches strictly before first_event_t. Stratified smoke v3: 49 rows, 25 videos, feature_dim=115, finite_rate=1.0, strict_preocc_rate=1.0. Full W8/W16 audit: W8 1456/1456 strict and found; W16 2013/2013 strict and found; no fallback. Proceed to full W8+W16 feature build v3.
```


## V9-A2.3b full W8+W16 feature build

Artifacts:

```text
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_features_dist_nc_le64_w8_v3.npz
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_features_dist_nc_le64_w16_v3.npz
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_joint_w8_common_plus_w16_extension_v3.npz
docs/v9a2_full_feature_build_result_2026-07-08.md
```

Decision:

```text
V9-A2.3b passes after correction. W8 full features: 1456 rows, dim 143, finite_rate 1.0. W16 full features: 2013 rows, dim 143, finite_rate 1.0, with 557 extensions. Common-row consistency audit found 662 mismatched common rows due to different event context in W16, so canonical joint dataset uses W8 common rows plus W16-only extension rows. Next: V9-A2.3c dynamic horizon controller.
```

## V9-A2.3c-R robust threshold and video stability audit

Artifacts:

```text
scripts/v9a2_dynamic_horizon_robust_audit.py
scripts/v9a2_dynamic_horizon_video_stability_audit.py
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_dynamic_horizon_robust_threshold_audit.json
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_dynamic_horizon_video_stability_audit.json
docs/v9a2_dynamic_horizon_robust_threshold_audit_result_2026-07-08.md
docs/v9a2_dynamic_horizon_robust_threshold_audit_result_2026-07-09.md
docs/v9a2_dynamic_horizon_video_stability_audit_result_2026-07-09.md
```

Decision:

```text
V9-A2.3c-R passes under conservative, predeclared thresholds. all_logreg event_max fixed 0.05 reaches AJ_RD_256 Δ +0.0294, above W16 +0.0286 and W8 +0.0274, with AJ Δ +0.1025 and Pos/Neg/Zero 17/3/5. On the 25 videos where AJ_RD_256 is defined, learned fixed 0.05 versus W16 is better on 4, worse on 2, equal on 19, with mean difference +0.000665. The gain is real but modest and concentrated; RGB anchor features remain weak. Next: V9-A2.4 paper-ready ablation and/or stronger semantic/internal identity features.
```

## V9-A2.4 paper-ready ablation and paired uncertainty audit

Artifacts:

```text
scripts/v9a2_paper_ready_ablation.py
outputs/paper_discovery_2026-07-05/v9a2_anchor_uncertainty_reacquisition/v9a2_paper_ready_ablation.json
docs/v9a2_paper_ready_ablation_design_2026-07-10.md
docs/v9a2_paper_ready_ablation_result_2026-07-10.md
```

Decision:

```text
V9-A2.4 completes the paper-ready prototype audit. The frozen all-logreg + event_max + fixed0.05 policy reaches aggregate AJ_RD_256 Δ +0.029379 versus W16 +0.028587 and W8 +0.027411, with AJ Δ +0.102542 and Pos/Neg/Zero 17/3/5. A predeclared 3 feature-set x 3 controller-mode x 2 threshold-protocol grid confirms that event_max is the best controller mode and that all features are required at fixed0.05; pre-occ is the strongest single RGB anchor but remains below the all-anchor result. However, paired video-level fixed0.05-vs-W16 mean is +0.000665 with bootstrap 95% CI [-0.000051, +0.001681], exact sign-flip p=0.15625, and 4 better / 2 worse / 19 equal videos. The aggregate gain is positive but not statistically established across DAVIS videos. The independent efficiency audit exactly reproduces the frozen 557 extension-row anchor matrix (max_abs_diff=0), measures 2.425 s total feature extraction, and measures single-thread controller inference at 0.462/0.472 ms p50/p95 for all 557 rows. Retain V9-A2 as a positive paper-ready prototype/diagnostic, not a final method. Next: V9-A2.5 semantic/internal identity feature audit under the same frozen dynamic-horizon protocol.
```

## V9-A2.5 DINOv3 semantic identity feature pilot

Artifacts:

```text
scripts/v9a25_build_dinov3_identity_features.py
scripts/v9a25_eval_dinov3_identity_features.py
scripts/v9a25_dinov3_late_fusion_audit.py
scripts/v9a25_dinov3_feature_family_audit.py
outputs/paper_discovery_2026-07-05/v9a25_dinov3_identity/v9a25_dinov3_identity_features.npz
outputs/paper_discovery_2026-07-05/v9a25_dinov3_identity/v9a25_dinov3_identity_eval.json
outputs/paper_discovery_2026-07-05/v9a25_dinov3_identity/v9a25_dinov3_late_fusion_audit.json
outputs/paper_discovery_2026-07-05/v9a25_dinov3_identity/v9a25_dinov3_feature_family_audit.json
docs/v9a25_dinov3_identity_feature_pilot_design_2026-07-10.md
docs/v9a25_dinov3_identity_feature_pilot_result_2026-07-10.md
docs/v9a25_dinov3_late_fusion_audit_result_2026-07-10.md
docs/v9a25_dinov3_feature_family_audit_result_2026-07-10.md
```

Decision:

```text
DINOv3 semantic evidence is real but does not become a robust trajectory gain. DINO-only logreg reaches AP/AUC 0.3637/0.6429. The structured feature-family audit shows the signal is not only candidate-native disagreement: history_candidate_identity reaches AP/AUC 0.3703/0.6532 and local_distinctiveness reaches 0.3967/0.6537. Nevertheless, the predeclared primary all+DINO policy changes aggregate AJ_RD_256 Δ only from frozen V9-A2 +0.029379 to +0.029439, with paired mean +0.000035, 95% CI [-0.000227,+0.000277], and exact sign-flip p=0.875. All late-fusion weights fail, and every predeclared feature family has a paired CI crossing zero; even anchor_consistency's aggregate +0.000228 over frozen has video mean -0.000180. Stop DINO concatenation/fusion/family tuning. Next: audit TrackOn2 internal matching/memory features; if unavailable or ineffective, move to V9-A3 internal multi-hypothesis candidate generation.
```

## V9-A2.6 TrackOn2 internal-state proxy audit and selector-route closure

Artifacts:

```text
scripts/v9a26_build_trackon2_internal_proxy_features.py
scripts/v9a26_eval_trackon2_internal_proxy.py
scripts/v9a26_trackon2_internal_late_fusion_audit.py
scripts/v9a26_trackon2_internal_feature_family_audit.py
scripts/v9a26_trackon2_internal_logo_robustness_audit.py
scripts/v9a26_trackon2_internal_proxy_integrity_audit.py
outputs/paper_discovery_2026-07-05/v9a26_trackon2_internal_proxy/v9a26_trackon2_internal_proxy_features.npz
outputs/paper_discovery_2026-07-05/v9a26_trackon2_internal_proxy/v9a26_trackon2_internal_proxy_eval.json
outputs/paper_discovery_2026-07-05/v9a26_trackon2_internal_proxy/v9a26_trackon2_internal_late_fusion_audit.json
outputs/paper_discovery_2026-07-05/v9a26_trackon2_internal_proxy/v9a26_trackon2_internal_feature_family_audit.json
outputs/paper_discovery_2026-07-05/v9a26_trackon2_internal_proxy/v9a26_trackon2_internal_logo_robustness_audit.json
outputs/paper_discovery_2026-07-05/v9a26_trackon2_internal_proxy/v9a26_trackon2_internal_proxy_integrity_audit.json
docs/v9a26_trackon2_internal_proxy_design_2026-07-10.md
docs/v9a26_trackon2_internal_proxy_result_2026-07-10.md
docs/v9a26_trackon2_internal_late_fusion_audit_result_2026-07-10.md
docs/v9a26_trackon2_internal_feature_family_audit_result_2026-07-10.md
docs/v9a26_trackon2_internal_logo_robustness_audit_result_2026-07-10.md
docs/v9a26_trackon2_internal_proxy_integrity_audit_result_2026-07-10.md
```

Decision:

```text
V9-A2.6 closes the selector-only feature route after a full integrity and robustness audit. The 557x64 internal proxy matrix is strictly aligned to the frozen W16-extension rows; old smoke, enhanced long-sequence smoke, and the full prefix are bit-identical. The copied diagnostic forward matches official TrackOn2 position/logit/query tensors with max_abs=0 on all 20 first-active checks and on three long-sequence target frames. The proxy remains explicitly non-identical to the historical cache latent state (proxy-to-old-candidate internal-model distance p95 10.49 px). Internal-only logreg has AP/AUC 0.3262/0.5972 and all+internal OOF-F1 reaches only AJ_RD_256 Δ +0.027958 versus frozen +0.029379; all predeclared late-fusion weights fail. Structured families confirm real internal signal: visibility/uncertainty AP 0.4368 and query-update AP 0.4230. Five-fold memory-consistency reaches aggregate Δ +0.030291, but its paired CI crosses zero; under 20-fold LOGO it falls to +0.028353. LOGO query-update is the strongest trajectory family at +0.030038, yet its paired CI [-0.000499,+0.006168] also crosses zero. No family has a positive paired-CI lower bound. Stop selector threshold/feature/fusion tuning. Next: V9-A3.0 TrackOn2 internal top-K multi-hypothesis oracle/action-space audit, followed by trainable multi-hypothesis reacquisition only if the oracle gap is material.
```

## V9-A3.0 TrackOn2 top-K multi-hypothesis oracle/action-space audit

Artifacts:

```text
scripts/v9a30_trackon2_topk_hypothesis_oracle.py
outputs/paper_discovery_2026-07-05/v9a30_topk_hypothesis/v9a30_trackon2_topk_hypothesis_oracle.json
docs/v9a30_trackon2_topk_hypothesis_oracle_design_2026-07-10.md
docs/v9a30_trackon2_topk_hypothesis_oracle_result_2026-07-10.md
```

Decision:

```text
V9-A3.0 passes strongly. On 490 GT-visible W16-extension rows, the union of old candidate + proxy final + C1 top16 + C2 top16 reduces mean coordinate error from 3.606 px to 1.432 px, improves 365 rows without worsening any visible row, and raises safe16 from 473 to 485. Oracle source attribution is old=125, proxy=60, C1=297, C2=8, showing that C1 top-K contains substantial information discarded by the final single-point decision. Coordinate-only union oracle reaches AJ_RD_256 Δ +0.036312 versus W16 +0.028587 (+0.007725); coordinate+GT-invisible rejection reaches +0.037056 (+0.008469). Naive C1/C2 top1 and existing rerank logits are worse than the old candidate, so the opportunity is candidate ranking, not raw top1 replacement. Next: V9-A3.1 video-heldout trainable multi-hypothesis ranking, with frozen V9-A2 event activation and learned coordinate selection.
```

## V9-A3.1 through V9-A4.5 multi-hypothesis ranking route closure

Canonical review:

```text
docs/v9a4_comprehensive_review_and_next_step_2026-07-10.md
docs/v9a45_route_closure_and_v9a5_next_step_2026-07-10.md
```

Decision:

```text
The TrackOn2 top-K oracle headroom from V9-A3.0 is real, but all tested fixed-topK ranking routes fail to transfer across videos/sequences: exported-summary tree/MLP/listwise models, source/rank factorization, DINO/PointOdyssey pretraining, post-fusion raw latents, pre-fusion small-head adaptation, unconstrained local-decoder adaptation, and conservative residual local-decoder adaptation. Do not repeat these model families or tune them on DAVIS. The remaining allowed direction must change candidate generation/correlation supervision before top-K.
```

## V9-A4.5 conservative residual end-to-end synthetic gate

Artifacts:

```text
scripts/v9a45_conservative_residual_end_to_end.py
docs/v9a45_conservative_residual_end_to_end_design_2026-07-10.md
docs/v9a45_input_manifest_2026-07-10.json
docs/v9a45_pointodyssey_frame_manifest_2026-07-10.json
docs/v9a45_smoke_holdout_ani_e1_seed20260710_result_2026-07-10.md
docs/v9a45_route_closure_and_v9a5_next_step_2026-07-10.md
outputs/paper_discovery_2026-07-05/v9a45_conservative_residual/v9a45_smoke_holdout_ani_e1_seed20260710.json
```

Protocol:

```text
clean worktree / branch: /gemini/code/FSPT_v9a45_clean / v9a45-conservative-residual-20260710
base HEAD: 4f3c01d
train sequences: animal3 + r4_new_f
heldout sequence: ani
train/validation rows: 2936 / 1942
epochs: 1
DAVIS labels/evaluation: none
```

Integrity:

```text
All input hashes pass, including 844 individually verified RGB frames.
PointOdyssey GT error reproduction max_abs = 4.58e-05.
Initial student/teacher scores and candidates are identical.
Two-step gradient audit proves delta-head learning on step 1 and local-decoder/fusion/alpha gradients on step 2.
Frozen base, teacher, and frozen student parameters remain bit-identical.
```

Result:

```text
teacher mean error 14.7458 -> student 15.0387
better/worse/equal = 35/70/1837
safe16 = 1563 -> 1544
oracle regret = 10.2049 -> 10.4979
teacher-good retention = 0.9768 overall, but 0.8966 on hard rows
improvement-opportunity success = 0.0324
alpha = 0.0500 -> 0.0732
synthetic gate = 1/5 passed
```

Decision:

```text
V9-A4.5 fails the predeclared synthetic heldout gate. Do not run the remaining two folds, do not run DAVIS, and do not sweep alpha/learning rate/loss weights on ani. The near-zero-alpha control produces no ranking changes, while alpha=0.05 is independently reproduced and harmful. Close the fixed-topK reranking-adaptation route. Next: V9-A5.0 candidate-recall/correlation-map oracle audit; proceed to a bounded residual correlation adapter only if top-K candidate recall has unsaturated headroom.
```

## V9-A5.0 sampled-causal temporal-state feasibility audit

Artifacts:

```text
scripts/v9a50_temporal_multihypothesis_feasibility.py
docs/v9a50_input_manifest_2026-07-10.json
docs/v9a50_temporal_multihypothesis_feasibility_design_2026-07-10.md
docs/v9a50_temporal_multihypothesis_feasibility_result_2026-07-10.md
outputs/paper_discovery_2026-07-05/v9a50_temporal_feasibility/v9a50_temporal_feasibility.json
outputs/paper_discovery_2026-07-05/v9a50_temporal_feasibility/v9a50_temporal_feasibility_selections.npz
```

Protocol and integrity:

```text
PointOdyssey balanced sampled pool only: 4878 rows, 279 query tracks, 9 clips, 3 sequences.
No training, no DAVIS read, no threshold or weight tuning.
State is causal with respect to sampled observations; gaps >8 reset all histories.
Track and conservative clip-block bootstrap use 100,000 resamples.
Checkpoint score reconstruction from all 4878x16 post-fusion latents has top1 match 1.0 and max_abs 0.001544.
All saved selections independently reproduce the reported metrics; non-temporal rows exactly fall back to teacher.
```

Result:

```text
Teacher mean C1 error: 12.3112 px.
Best sampled self-state policy: causal_teacher_latent, 12.0118 px globally.
However it worsens all three r4_new_f clips, its clip 95% CI is [-1.0067,+0.0903], and it fails the three-sequence/safe16 gate.
No sampled-causal self-state policy passes.

Past-oracle motion+latent: 6.0285 px, better/worse/equal 2278/399/2201.
Past-GT motion diagnostic: 5.3269 px, better/worse/equal 2703/484/1691.
Every past-oracle/past-GT diagnostic improves all three sequences and all 9 clips; their clip-block CIs are strictly negative.
```

Decision:

```text
V9-A5.0 yields ORACLE_STATE_HEADROOM_ONLY. Temporal information is strongly useful when the prior state is correct, while self-state error propagation prevents robust gains. Do not train another single-state reranker. Next: V9-A5.1 full-stream deterministic multi-hypothesis/beam reachability audit. It must update state on every frame, evaluate GT-visible and re-entry subsets, and separate deterministic beam top1 from GT-only beam-oracle readout. DAVIS remains out of scope until synthetic full-stream gates pass.
```

## V9-A5C.0 fused correlation candidate-recall audit

Artifacts:

```text
scripts/v9a5c0_candidate_recall_correlation_oracle.py
docs/v9a5c0_input_manifest_2026-07-10.json
docs/v9a5c0_candidate_recall_correlation_oracle_design_2026-07-10.md
docs/v9a5c0_pointodyssey_candidate_recall_result_2026-07-10.md
outputs/paper_discovery_2026-07-05/v9a5c0_candidate_recall/v9a5c0_pointodyssey_candidate_recall.json
outputs/paper_discovery_2026-07-05/v9a5c0_candidate_recall/v9a5c0_pointodyssey_candidate_recall.npz
```

Protocol and integrity:

```text
PointOdyssey full online replay over 9 clips and 4878 canonical rows.
No training, no DAVIS read, no K/radius/weight tuning.
Predeclared K = 1/4/8/16/32/64 and radius = 1/2/4/8 px.
Official TrackOn2 p/v/q tensors, fused correlation map, and canonical C1 top16 set all reproduce exactly.
Fused top16 oracle error reproduces the frozen pool with max_abs 4.58e-05.
All 844 used frames and immutable inputs are hash verified.
```

Result:

```text
Global fused recall@4: K16 0.7729 -> K64 0.9287.
Per-sequence fused K16 -> K64 headroom:
  ani      0.7199 -> 0.9058, +0.1859
  animal3  0.8555 -> 0.9304, +0.0749
  r4_new_f 0.7697 -> 0.9545, +0.1849
All three sequences pass the predeclared >=0.02 gate.

Hard rows: K16 0.5420 -> K64 0.8561, +0.3142.
Easy rows: K16 1.0000 -> K64 1.0000, +0.0000.
Therefore candidate expansion should be risk/event activated rather than globally increasing K.

Fused K16/K64 exact boundary tie rate is only 0.0205%; the headroom is not a tie-order artifact.
Raw c4 has a negative learned ms_corr_proj coefficient and raw-scale unions are diagnostic only, not part of the primary fused gate.
```

Decision:

```text
V9-A5C.0 passes strongly. The current fused top16 discards substantial GT-near candidate recall on hard rows, while easy rows are already saturated. Combined with V9-A5.0 ORACLE_STATE_HEADROOM_ONLY, the next step is V9-A5.1 full-stream deterministic dynamic-K multi-hypothesis/beam reachability audit: use K16 as the default, expose up to K64 only under a predeclared risk signal, update multiple hypotheses on every frame, and separately report deployable beam-top1 versus GT-only beam-oracle reachability. Do not read DAVIS or train until the synthetic full-stream gate passes.
```

## V9-A5.1a fixed-K16 shared-state beam baseline

Canonical frozen artifacts:

```text
scripts/v9a51a_fixed_k16_shared_state_beam.py
docs/v9a51a_fixed_k16_shared_state_beam_design_2026-07-10.md
docs/v9a51a_fixed_k16_shared_state_beam_result_2026-07-10.md
docs/v9a51a_fixed_k16_baseline_freeze_2026-07-10.md
docs/v9a51_comprehensive_review_and_next_step_2026-07-10.md
outputs/paper_discovery_2026-07-05/v9a51a_fixed_k16_shared_state_beam/v9a51a_fixed_k16_shared_state_beam.json
outputs/paper_discovery_2026-07-05/v9a51a_fixed_k16_shared_state_beam/v9a51a_fixed_k16_shared_state_beam.npz
```

Integrity:

```text
The V9-A5.1a script/JSON/NPZ are byte-identical copies of the independently completed experiment.
Script SHA256 = 4c49df7c743ba14229f30786d13e2776348f9bb66f14aa40b0b6215c94bc5ee8.
The script hash matches the result JSON.
27,648 query-frame rows and 864 RGB frames were audited.
Official p/v/q parity max_abs = 0.
Beam update receives no GT or error input.
```

Result after excluding frame 0:

```text
teacher C1 top1 mean 9.4394 px
beam4 motion+latent top1 9.8841 px, delta +0.4447 px
teacher top4 oracle 6.3934 px
beam4 oracle 7.5775 px, delta +1.1841 px
teacher top8 oracle 5.3604 px
beam8 oracle 6.2353 px, delta +0.8749 px
```

Decision:

```text
V9-A5.1a is a valid negative baseline for the exact fixed-K16 shared-state beam configuration. It does not close risk-gated K64 proposals, candidate-conditioned C2/offset refinement, history-preserving beams, finite-window costs, official-final fallback, or independent per-hypothesis model states. Do not repeat the fixed-K16/raw-C1/current-candidate-dedup implementation. Next: V9-A5.1b candidate-conditioned downstream refinement audit. No new beam is allowed unless its risk-gated refined candidate oracle improves the official final tracker on all synthetic gates.
```

## V9-A5.1b candidate-conditioned downstream refinement

Artifacts:

```text
scripts/v9a51b_candidate_conditioned_refinement_audit.py
docs/v9a51b_candidate_conditioned_refinement_design_2026-07-10.md
docs/v9a51b_candidate_conditioned_refinement_result_2026-07-10.md
docs/v9a51b_candidate_conditioned_refinement_review_2026-07-10.md
docs/v9a51b_input_manifest_2026-07-10.json
docs/v9a51b_pointodyssey_frame_manifest_2026-07-10.json
outputs/paper_discovery_2026-07-05/v9a51b_candidate_refinement/v9a51b_k16_k64_full_parity_preflight.json
outputs/paper_discovery_2026-07-05/v9a51b_candidate_refinement/v9a51b_candidate_conditioned_refinement.json
outputs/paper_discovery_2026-07-05/v9a51b_candidate_refinement/v9a51b_candidate_conditioned_refinement_rows.npz
```

Integrity:

```text
27,648 query-frame rows, 9 clips, 864 hash-verified RGB frames.
Official p/v/q parity max_abs = 0.
K16/K64 candidate-set Hausdorff max = 0.
All-row numerical preflight: score max 5.245e-5, certainty max 4.864e-5, official-top1 mismatch 0/27,648.
Final numerical tolerance fixed at 6e-5 before the formal refinement metrics were run.
Candidate refinement function receives no GT/error input.
Saved NPZ independently reproduces risk, re-entry, readouts, bootstrap CIs and gates.
```

Reachability result:

```text
official final mean:            9.1418 px
hybrid refined oracle mean:     7.8185 px
paired mean difference:        -1.3233 px
better / worse / equal:         4267 / 0 / 15663
clip-block 95% CI:             [-2.1037, -0.7501]
all 9 clip mean differences are negative
all 3 sequence means and first-reentry means improve
```

Selection result:

```text
hybrid refined score-top1 mean: 9.1023 px
paired mean difference:        -0.0394 px
better / worse / equal:         2085 / 2210 / 15635
clip-block 95% CI:             [-0.1990, +0.0713]
ani and animal3 are slightly worse; r4_new_f is better
```

Candidate diversity:

```text
raw K64 oracle:                 2.9606 px
refined K64 oracle:             7.0763 px
mean unique refined C2 locations at K64: 3.44
mean refined 4px clusters at K64:       1.68
```

Decision:

```text
V9-A5.1b passes as candidate-conditioned refined-coordinate reachability, not as a deterministic selector. Singleton downstream refinement strongly compresses candidate diversity and must not replace the raw C1 hypothesis state. Proceed to V9-A5.1c only with raw candidate identity/history as state, refined coordinate as readout, native-risk K16/K64, an eight-frame finite cost window, official-final fallback on non-risk rows, and separate deterministic versus GT-only reachability gates. Do not read DAVIS and do not train yet.
```

## V9-A5.1c history-preserving shared-state beam and incremental-value closure

Formal artifacts:

```text
scripts/v9a51c_history_preserving_beam_audit.py
docs/v9a51c_history_preserving_beam_design_2026-07-10.md
docs/v9a51c_smoke_review_2026-07-10.md
docs/v9a51c_history_preserving_beam_result_2026-07-10.md
docs/v9a51c_input_manifest_2026-07-10.json
docs/v9a51c_pointodyssey_frame_manifest_2026-07-10.json
outputs/paper_discovery_2026-07-05/v9a51c_history_beam/v9a51c_history_preserving_beam.json
outputs/paper_discovery_2026-07-05/v9a51c_history_beam/v9a51c_history_preserving_beam_rows.npz
```

Supplemental fair-comparator review:

```text
scripts/v9a51c_same_capacity_incremental_audit.py
docs/v9a51c_same_capacity_incremental_review_2026-07-10.md
docs/v9a51c_comprehensive_review_and_v9a52_next_step_2026-07-10.md
outputs/paper_discovery_2026-07-05/v9a51c_history_beam/v9a51c_same_capacity_incremental_audit.json
```

Formal integrity:

```text
9 clips x 96 frames x 32 queries = 27,648 rows.
864 RGB frames individually hash verified.
Official p/v/q parity max_abs = 0.
V9-A5.1b row/risk/score/raw-error/refined-error/oracle replay max_abs = 0.
All policy beam widths/shapes are exact.
All last-three-grid signatures are unique within retained beams.
Beam update receives no GT/error input.
Saved NPZ independently reproduces all top1/oracle formulas, bootstrap CIs and gates.
```

Original predeclared system-level result:

```text
official final mean:                 9.1418 px
native_dynamic_B4 top1 mean:         9.1575 px
native_dynamic_B4 top1 delta:       +0.0157 px
native_dynamic_B4 top1 CI:          [-0.0573,+0.1110]

native_dynamic_B4 oracle mean:       8.7879 px
native_dynamic_B4 oracle delta:     -0.3539 px
native_dynamic_B4 oracle CI:        [-0.5043,-0.2343]
all 3 sequence means and all 9 clip means improve versus official final
```

Formal interpretation:

```text
The deterministic top1 gate fails. The GT-only min(official, surviving beam) gate passes, proving that the beam preserves some useful refined alternatives. It retains only 26.74% of the full V9-A5.1b candidate headroom.
```

Mandatory same-capacity correction:

```text
frame-local score-top4 oracle mean:  8.5363 px
temporal B4 oracle mean:             8.7879 px
temporal minus frame-local:         +0.2516 px
95% CI:                              [+0.1280,+0.4307]
temporal B4 is worse on ani / animal3 / r4_new_f and all 9 clips

frame-local score top1 mean:         9.1023 px
temporal B4 top1 mean:               9.1575 px
temporal minus frame-local:         +0.0552 px
95% CI:                              [-0.1054,+0.2999]
```

All predeclared diagnostic policies fail the incremental temporal-value gate. B1 has only a weak, inconsistent oracle mean advantage with a CI crossing zero; B4/B8/fixed16/fixed64 are weaker than their frame-local same-capacity oracle controls.

Effective diversity:

```text
B4 unique history signatures: 4.0 / 4
B4 unique current raw grids on risk-visible rows: 2.81 / 4
B4 raw 4px clusters: 1.45
B4 refined 4px clusters: 1.13
```

Decision:

```text
HISTORY_INCREMENTAL_VALUE_FAIL. Keep the original system-level beam-oracle pass as a valid reachability statement relative to official final, but do not interpret it as temporal incremental value. Close the shared-official-state external history-beam route and do not train a learned readout on the current beam. Fixed-topK selectors/adapters were already closed in V9-A3.1 through V9-A4.5. Next: V9-A5.2 bounded independent model-state branching feasibility, where each hypothesis owns separate point memory/temporal mask/q_new and must beat a shared-state frame-local same-capacity control. No DAVIS and no training.
```

## V9-A5.2 independent model-state branching pre-registration

Pre-registered artifacts:

```text
scripts/v9a52_build_event_manifest.py
docs/v9a52_independent_state_event_manifest_2026-07-11.json
docs/v9a52_independent_state_branching_design_2026-07-11.md
```

Repository:

```text
worktree: /gemini/code/FSPT_v9a52_clean
branch: v9a52-independent-state-branching-20260711
base HEAD: 53f475a283c70c49b6bc2167458b520d11d6b3d9
```

Structural correction:

```text
TrackOn2 query_attention couples all 32 evaluated queries and 400 support-grid queries. A hypothesis is therefore a complete 432-query tracker state, not one target row and not a duplicate query concatenated into the same attention batch. Each A/B branch must separately own q_init, point_memory and temporal_mask and roll all 432 rows independently.
```

Canonical state:

```text
N = 432
M = 24, preserving the V9-A5.1b/c M_i=M execution path
D = 256
```

Frozen event protocol:

```text
source: committed V9-A5.1c NPZ SHA256 05b450fb2031ce1d762ed7436bcef34a47dfaf37c10d1c33bd8fd74c1d0381a0
selection fields: clip_id/query_idx/frame_tau/risk only
first native-risk event per query with 1 <= frame_tau <= 87
stable SHA256 selection of 8 events per clip
9 clips / 72 events / horizons 1,4,8
minimum eligible events in any clip = 15
```

Branch split:

```text
At the event frame, clone the complete pre-update state. Branch A writes official q_new for all rows. Branch B writes official q_new for all non-target rows and the frozen K64 score-top1 singleton candidate-conditioned q2 for the target row. Both then roll forward independently with unconditional memory updates.
```

Primary fair comparator:

```text
independent B2 oracle = min(Branch A final, Branch B final)
shared-state B2 oracle = min(official final, current official-state score-top1 refined coordinate)
```

The primary horizon is 8. Clip-block bootstrap uses 100,000 resamples, seed 20260716. First-reentry is descriptive because only 9 frozen horizon-8 rows are first-reentry; pooled early8 is secondary.

Decision boundary:

```text
Only a sequence-consistent, clip-CI-negative independent B2 improvement over the shared-state capacity-2 control, together with non-degenerate q_new/C1 candidate-set divergence and useful novel rows in every sequence, can authorize a later full-stream independent-state beam. Failure closes the temporal multi-state route. No training, threshold tuning or DAVIS read is allowed.
```

## V9-A5.2 independent-state branching smoke

Artifacts:

```text
scripts/v9a52_independent_state_branching_smoke.py
docs/v9a52_independent_state_smoke_result_2026-07-11.md
docs/v9a52_independent_state_smoke_review_2026-07-11.md
outputs/paper_discovery_2026-07-05/v9a52_independent_state/v9a52_independent_state_smoke.json
outputs/paper_discovery_2026-07-05/v9a52_independent_state/v9a52_independent_state_smoke_rows.npz
```

Frozen smoke event:

```text
ani:0 / query 20 / frame 2
horizons 1 / 4 / 8
432 active queries = 32 evaluated + 400 support
M = 24 / D = 256
```

Integrity:

```text
official diagnostic p/v/q_new parity max_abs = 0
Branch A p/v/u/q_new/q_pre/C1/C2 parity max_abs = 0
Branch A q_init/memory/mask parity exact
A/B/official state storages disjoint
Branch B mutation leaves Branch A unchanged
A->B versus B->A call-order max_abs = 0
online native-risk mismatch = 0
V9-A5.1b candidate replay max_abs = 1.526e-5
shared and independent B2 formulas exact
all values finite
```

The full numerical smoke was repeated and all NPZ arrays, gates, parity values, candidate replay values, horizon diagnostics and split numerical values reproduced exactly.

Mechanistic result:

```text
split q2_B versus official q_new_A L2 = 10.2380
future target q_new L2 at horizons 1/4/8 = 8.9140 / 2.7783 / 2.0466
top16 C1 overlap at horizons 1/4/8 = 14 / 15 / 15
top64 C1 overlap = 54 / 59 / 62
```

The perturbation propagates to non-target evaluated and support-grid queries, confirming that a hypothesis must clone the complete 432-query state.

Compression warning:

```text
final A/B coordinate divergence at horizons 1/4/8 = 0.0963 / 0.1470 / 0.0529 px
```

State and candidate-set divergence do not by themselves establish useful output divergence. The formal run must retain both mechanistic and capacity-matched accuracy gates.

All three smoke horizon rows are GT-invisible/invalid, so the saved coordinate-error values verify formulas only and are not scientific evidence.

Decision:

```text
SMOKE_PASS. The full 72-event audit may be implemented. No training and no DAVIS read are authorized.
```

## V9-A5.2 independent model-state branching formal closure

Formal artifacts:

```text
scripts/v9a52_independent_state_branching_audit.py
docs/v9a52_input_manifest_2026-07-11.json
docs/v9a52_independent_state_branching_result_2026-07-11.md
outputs/paper_discovery_2026-07-05/v9a52_independent_state/v9a52_independent_state_branching.json
outputs/paper_discovery_2026-07-05/v9a52_independent_state/v9a52_independent_state_branching_rows.npz
```

Non-gating candidate-pool review:

```text
scripts/v9a52_candidate_reachability_postaudit.py
docs/v9a52_candidate_reachability_postaudit_2026-07-11.md
docs/v9a52_comprehensive_review_and_route_closure_2026-07-11.md
outputs/paper_discovery_2026-07-05/v9a52_independent_state/v9a52_candidate_reachability_postaudit.json
```

Execution integrity:

```text
9 clips / 72 frozen events / horizons 1,4,8 / 216 unique event-horizon rows
864 RGB frames individually hash verified
288 candidate replay rows = 72 split + 216 horizon
peak active events = 6
official diagnostic p/v/q_new parity max_abs = 0
Branch A p/v/u/q_new/q_pre/C1/C2 parity max_abs = 0
Branch A q_init/memory/mask parity exact
online risk and GT-visibility replay mismatch = 0
candidate score max_abs = 2.0504e-5
refined candidate error max_abs = 4.5538e-5
storage/mutation/cross-event-alias/event-key/first-risk failures = 0
saved NPZ independently reproduces formulas, summaries, bootstraps and gates
```

Primary horizon-8 capacity-matched result:

```text
independent-state B2 oracle mean: 5.4788 px
shared-state capacity-2 oracle:    5.1949 px
difference:                      +0.2839 px
better / worse / equal:           14 / 14 / 16
safe16:                            41 / 42
95% clip-block CI:               [+0.0159,+0.6744]
ani / animal3 / r4_new_f:        +0.0684 / +0.5234 / +0.2141 px
```

Secondary results:

```text
horizon 1: +0.0054 px
horizon 4: -0.1311 px, but r4_new_f and multiple clips are worse
pooled horizons: +0.0734 px, CI [-0.0891,+0.3215]
early8: +0.3054 px, CI [+0.0351,+0.6420]
first re-entry: +0.7384 px
```

Mechanistic result:

```text
target q_new divergence on 100% of visible rows
top16 candidate set changes on 51.85% of visible rows
mean target q_new L2 = 1.6416
mean final-coordinate divergence = 0.5493 px
Branch B final improves shared B2 by >1 px on only 3 rows
useful final novelty by sequence = ani 1 / animal3 2 / r4_new_f 0
```

Candidate-pool supplement:

```text
Branch B K64 versus shared K64:
  mean difference -0.0018 px
  CI [-0.1819,+0.2219]
  ani -0.0882 / animal3 +0.1771 / r4_new_f -0.1210

union(shared K64, Branch B K64) oracle:
  difference -0.1191 px
  CI [-0.2256,-0.0335]
  structurally non-worsening GT-only upper bound

Branch B beats shared K64 by >1 px: 4 / 108 visible rows
by sequence: ani 2 / animal3 0 / r4_new_f 2
Branch B <=4 px while shared K64 >4 px: 0
```

Decision:

```text
INDEPENDENT_STATE_FAIL. True independent full-model state changes future hidden states and candidate sets, but does not beat the shared-state same-capacity final-output control and does not robustly improve the complete K64 candidate pool. Close the tested temporal multi-state route. Do not train a branch selector, sweep alternative branch seeds/widths/horizons/risk thresholds, or read DAVIS.
```

Route scope:

```text
Close sampled causal self-state selection, fixed/shared-state beams, history-preserving external beams, and frozen score-top1 singleton independent-state B2 branching. This is a project route closure under the committed protocols, not a mathematical claim about every recurrent tracker.
```

Remaining distinct direction:

```text
V9-A5C.0 still shows fused top16-to-top64 recall headroom, especially on hard rows. The only unclosed mechanism is to change the fused correlation map before top-K, not to select/branch over a fixed pool. Next: V9-A6.0 no-training bounded correlation rank-shift feasibility. Only if that gate passes may a frozen-backbone sequence-heldout bounded residual correlation adapter be preregistered. No training or DAVIS in V9-A6.0.
```

## V9-A6.0 bounded correlation rank-shift pre-registration

Artifacts:

```text
docs/v9a60_bounded_correlation_rank_shift_design_2026-07-11.md
docs/v9a60_input_manifest_2026-07-11.json
```

Repository:

```text
worktree: /gemini/code/FSPT_v9a60_clean
branch: v9a60-correlation-rank-shift-20260711
base HEAD: bb879183ffff03129cb652824c286f56f9201804
```

Scope correction:

```text
V9-A6.0 is a no-training necessary-condition audit. A GT-directed target boost can measure only the score span required to promote an existing rank-17-to-64 <=4px candidate into top16. It cannot measure target identification, false promotions, easy-row collateral or final TrackOn2 accuracy. A pass authorizes only a separately preregistered V9-A6.1 sequence-heldout bounded residual adapter.
```

Existing evidence is insufficient for exact rank-shift magnitude because V9-A5C.0 does not save the top16 threshold score or the highest-scoring <=4px candidate in ranks 17-64. V9-A6.0 must first export top129 fused/component scores and errors with full V9-A5C.0 parity.

Frozen opportunity expectations:

```text
4,878 source rows
K16 miss / K64 hit at 4px: 760
ani 361 / animal3 98 / r4_new_f 301
```

Risk semantics:

```text
GT-hard = old_error_px > 4 and is reporting-only.
Native risk remains visibility_conf < 0.8 OR uncertainty_sigmoid >= 0.5.
Native risk covers only 325/760 opportunity rows and only 27/301 on r4_new_f, so the primary magnitude audit is global. Risk-gated conversion is diagnostic only.
```

Target and perturbation:

```text
Target = first/highest-scoring <=4px candidate in original fused ranks 17-64.
Score span = rank16_score - target_score.
Positive target-only epsilon = nextafter(rank16_score,+inf) - target_score.
Signed pairwise per-cell L∞ lower bound = score span / 2.
```

Frozen natural reference and budgets:

```text
g_ref = committed all-row fused K16 boundary-gap median = 0.003143310546875
span budgets = 0.5/1/2/4/8 * g_ref
primary span budget = 2*g_ref = 0.00628662109375
```

Primary necessary-condition gate at `span <= 2*g_ref`:

```text
global opportunity conversion >= 0.50
each sequence conversion >= 0.40
at least 7/9 clips conversion >= 0.30
clip-bootstrap 95% lower bound > 0.35
implied global recall@4 oracle gain >= 0.075
implied GT-hard recall@4 oracle gain >= 0.12
median span/fused_std <= 0.02 on every sequence
global p90 span/fused_std <= 0.10
```

No easy-row safety claim is allowed from the target-directed oracle. No training, epsilon sweep, risk-threshold sweep or DAVIS read is allowed.

## V9-A6.0 fused top129 export

Artifacts:

```text
scripts/v9a60_export_fused_top129.py
docs/v9a60_fused_top129_export_result_2026-07-11.md
docs/v9a60_fused_top129_export_review_2026-07-11.md
outputs/paper_discovery_2026-07-05/v9a60_rank_shift/v9a60_fused_top129_export.json
outputs/paper_discovery_2026-07-05/v9a60_rank_shift/v9a60_fused_top129_export.npz
```

Formal export:

```text
4,878 rows / 9 clips / 844 hash-verified RGB frames
top129 fused indices/scores/coords/errors
four upsampled component values per candidate
full fused-map summary statistics
native-risk / GT-hard / reentry metadata
```

Integrity:

```text
fused map / official p/v/q / K16/K64 errors / nearest rank-distance-margin / K16 gap parity = 0
component-weight reconstruction max_abs = 0
official top16 set Hausdorff = 0
pool top16 set Hausdorff = 8.4294e-8
pool ordered max_abs = 0.18823537, diagnostic tie-order only
all top129 rows unique, sorted and finite
```

Frozen hashes:

```text
script 684e7a6243d7fccddd23bef79355356dd78ef2b40d2257d1483f790ac680ed2c
JSON   72fb654381eef236c73b54d8d52d5bfbe4a7b3e4cfaf7ebcf850f1cdf8472cbb
NPZ    0849bb9ab700e8f5224bf751f897aea7a38e0efd6363d6d83297e04ecbfac3ac
```

Opportunity parity:

```text
K16 miss / K64 hit at 4px = 760
ani 361 / animal3 98 / r4_new_f 301
native-risk opportunity rows = 325
GT-hard opportunity rows = 760
```

Decision:

```text
EXPORT_PASS. Freeze this export. The V9-A6.0 rank-shift audit must consume only the saved JSON/NPZ and must not rerun TrackOn2 or change target ranks, radius, budgets or gates.
```

## V9-A6.0 bounded correlation rank-shift formal closure

Artifacts:

```text
scripts/v9a60_bounded_rank_shift_audit.py
docs/v9a60_bounded_rank_shift_audit_result_2026-07-11.md
docs/v9a60_comprehensive_review_and_algorithmic_route_closure_2026-07-11.md
outputs/paper_discovery_2026-07-05/v9a60_rank_shift/v9a60_bounded_rank_shift_audit.json
outputs/paper_discovery_2026-07-05/v9a60_rank_shift/v9a60_bounded_rank_shift_audit_rows.npz
```

Integrity:

```text
760 / 760 K16-miss/K64-hit opportunity rows have a valid first <=4px target in ranks 17-64.
All target ranks/errors, rank16 errors, score spans, positive/signed epsilon formulas and component decompositions independently reproduce from the saved NPZ.
100,000-resample clip-block conversion/global-gain/hard-gain CIs independently reproduce.
No TrackOn2 execution, training, threshold tuning or DAVIS read occurs in the rank-shift audit.
```

Frozen hashes:

```text
script 61dd1ea369353a698a683212dcb2c45d6ba0eb9b1d7a603e6fdbc270eec538c6
JSON   a541d4475ca647a798827ac05a5befff588d33ceccfefb1229b85dff3a4f9370
NPZ    24c3b49d85635aa6879ef084630422ccf63f145951eeae808d72a24d58da30a6
```

Primary `span <= 2*g_ref` result:

```text
promoted 67 / 760 opportunities = 8.82%
ani 8.59% / animal3 22.45% / r4_new_f 4.65%
0 / 9 clips reach 30% conversion
conversion 95% CI [0.0549,0.1421]
implied global recall@4 oracle gain +0.0137
implied GT-hard recall@4 oracle gain +0.0277
native-risk-only implied global gain +0.0074
```

Magnitude:

```text
score-span median 0.03431 = about 10.92*g_ref
score-span p90 0.07756
span/fused-std median 0.1265
span/fused-std p90 0.2744
sequence medians 0.1426 / 0.0815 / 0.1235, all above the preregistered 0.02 bound
```

All 67 primary promotions come from target ranks 17-24. Ranks 25-64 have zero primary conversion.

Even at `8*g_ref`:

```text
conversion 34.74%
implied global gain +0.0541
implied hard gain +0.1091
only 5 / 9 clips reach 30%
r4_new_f conversion 27.57%
```

Component diagnostics do not show one scale that consistently favors the target across all sequences. They are non-gating and may not be used for post-hoc fusion-weight selection.

Decision:

```text
RANK_SHIFT_MAGNITUDE_FAIL. Even a GT-aware target-only oracle with no easy-row collateral cannot recover a meaningful and sequence/clip-consistent fraction of K16-to-K64 headroom under the committed small bounded score span. Close the bounded correlation residual route. Do not train V9-A6.1, add epsilon budgets, tune fusion weights or read DAVIS.
```

Project route decision:

```text
Stop algorithmic expansion on the current rescue tree. Route A diagnostic paper remains active/stable. Consolidate committed evidence, route-closure tables, reproducibility checks, manuscript figures and limitations. Any future model experiment requires a genuinely new written architectural hypothesis, not another selector/state branch/small residual variant.
```

## V9-A6.0 visibility-boundary post-audit

Artifacts:

```text
scripts/v9a60_visibility_stratified_postaudit.py
docs/v9a60_visibility_stratified_postaudit_2026-07-11.md
outputs/paper_discovery_2026-07-05/v9a60_rank_shift/v9a60_visibility_stratified_postaudit.json
```

Result:

```text
canonical V9-A3.8/V9-A5C.0 rows: 4,878 visible / 0 invisible
V9-A6.0 opportunity rows:       760 visible / 0 invisible
```

The V9-A3.8 pool builder skips a row before append when the target is not visible or the trajectory coordinate is non-finite. Therefore V9-A5C.0 and V9-A6.0 are visible-frame candidate audits by construction.

Independent checks reproduce the exact source-index mapping and the formal `2*g_ref`/`8*g_ref` promoted counts of 67 and 264.

Decision:

```text
VISIBILITY_STRATIFICATION_NOT_APPLICABLE. Invisible rows do not drive the V9-A6.0 failure because no invisible row is present in the canonical opportunity set. Preserve the formal gate and route closure. Add a visible-only sampling caveat to the paper and do not claim invisible-frame correlation-recall coverage.
```

## V9-A evidence consolidation and verification

Tooling commits:

```text
f777929 Add V9-A evidence index and verifier tooling
7e8d2f6 Relax evidence verifier float32 mean tolerance
e390c01 Normalize generated evidence document endings
bf4aed1 Stabilize evidence tooling provenance
```

Generated artifacts:

```text
docs/V9A_EVIDENCE_INDEX_2026-07-11.md
docs/V9A_ROUTE_CLOSURE_MATRIX_2026-07-11.md
docs/REPRODUCE_V9A_EVIDENCE.md
docs/V9A_EVIDENCE_VERIFICATION_2026-07-11.md
outputs/paper_discovery_2026-07-05/v9a_evidence/v9a_evidence_index.json
outputs/paper_discovery_2026-07-05/v9a_evidence/v9a_evidence_verification.json
```

Evidence index:

```text
schema: v9a-evidence-index-v1
routes: 8
indexed artifacts: 43
stable tooling head: bf4aed14c8b3dfa656991392bf27e27dd3982c01
index SHA256: 74f2f04d8a56b3c399636ce3243ac4b336a704ba765a9cf512b460f740516eb1
```

CPU-side committed-artifact verification:

```text
PASS
8 route checks passed
43 artifact SHA256/size checks passed
commit ancestry checks passed
key NPZ formulas independently reproduced
verification SHA256: 19a2919aebf475dbd1a6b5ce6f737cda3cfb12f7ac29bf4be4e9f70099df7f20
```

Independent formula coverage includes:

```text
V9-A4.5 student-minus-teacher arithmetic and failed gate
V9-A5.0 sampled-state failure / oracle-state headroom
V9-A5C.0 per-sequence K16/K64 recall and headroom
V9-A5.1a selected-candidate top1/oracle means
V9-A5.1b risk-gated hybrid-oracle formula
V9-A5.1c beam formulas and same-capacity final failure
V9-A5.2 shared/independent B2 formulas
V9-A6.0 span/epsilon/component/budget formulas and visible-only boundary
```

Reproducibility boundary:

```text
The final checkout can verify committed artifacts and saved-array formulas without GPU inference. Exact full GPU recomputation additionally requires immutable external assets, historical execution code states, and the recorded absolute-path layout or an external path-mapping wrapper. Several formal scripts were executed from a historical HEAD before the execution script/result commit; use the recorded execution HEAD plus script SHA256 rather than assuming direct final-checkout rerun.
```

Project state:

```text
Route A diagnostic paper: active/stable
Route B current rescue hypothesis tree: exhausted
Route C old student/pseudo-label/DINO-local routes: closed
Next work: manuscript tables/figures, discussion/limitations, clean-checkout artifact verification, and archival/tagging. No new model experiment without a genuinely new written architectural hypothesis.
```
