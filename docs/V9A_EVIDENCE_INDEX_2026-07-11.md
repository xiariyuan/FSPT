# V9-A Evidence Index

Date: 2026-07-11

Generated from `v9a60-correlation-rank-shift-20260711` at `bf4aed14c8b3dfa656991392bf27e27dd3982c01`.

This index separates the original preregistered result from the final fair interpretation. Artifact hashes are machine-readable in the accompanying JSON.

## V9-A4.5 — Conservative residual reranking

- Commit: `8fe78db05fa23a229efb92329e8fe96a247b7989`
- Status: `closed`
- Question: Can a bounded residual reranker improve the frozen candidate ordering without damaging safe rows?
- Data: PointOdyssey synthetic holdout; ani held out after sequence split
- Fair baseline: Frozen teacher C1 ranking on the same rows/candidates
- Primary gate: Mean improvement, better>worse, safe16 non-decrease, retention, and oracle-regret improvement
- Original result: Heldout student ranking is worse than the frozen teacher.
- Final interpretation: Fixed-topK residual reranking adaptation is closed.

```json
{
  "teacher_mean_error": 14.745758832552221,
  "student_mean_error": 15.038686401482162,
  "student_minus_teacher": 0.2929275689299394,
  "better_worse_equal": [
    35,
    70,
    1837
  ],
  "teacher_safe16": 1563,
  "student_safe16": 1544,
  "gate_pass": false,
  "decision": "Integrity passes, but heldout synthetic ranking collapses materially. Stop before DAVIS and do not scale this objective."
}
```

Artifacts:

- `scripts/v9a45_conservative_residual_end_to_end.py` — execution_script — `28ba42ff99a90158806ffe91b7a3215ebf6748f2a4d05e9ec3c1dd7b4c55b197`
- `outputs/paper_discovery_2026-07-05/v9a45_conservative_residual/v9a45_smoke_holdout_ani_e1_seed20260710.json` — result_json — `a8981bd6b7ee45dabc622f10d96c712b46069ccfad294a5daf1a7d999fdaf91f`
- `docs/v9a45_conservative_residual_end_to_end_design_2026-07-10.md` — design — `2d99ec36afb18304657da9abe45ad3ae0d497bc7287d0dab8a5742e44b46d1d5`
- `docs/v9a45_route_closure_and_v9a5_next_step_2026-07-10.md` — review — `04fa4cee28567b5fbc694424b80fe359ca9118ada3a4176aa7ef7e0d7160909d`

## V9-A5.0 — Sampled-causal temporal-state feasibility

- Commit: `7c24367cf8aeb1467946d2ced3b294299032538e`
- Status: `diagnostic_headroom_only`
- Question: Does causal prior state contain useful temporal information, and is self-state error propagation the blocker?
- Data: PointOdyssey canonical 4,878-row visible candidate pool
- Fair baseline: Frozen teacher/current-frame selection
- Primary gate: Sampled self-state must improve robustly; oracle/past-GT states are diagnostic headroom only
- Original result: Sampled self-state fails; causal past-oracle/past-GT state has large sequence-consistent headroom.
- Final interpretation: Temporal information exists, but single-state error propagation blocks deployable use.

```json
{
  "best_sampled_policy": "causal_teacher_latent",
  "sampled_mean_error": 12.011761665344238,
  "sampled_difference_vs_teacher": -0.2994817793369293,
  "best_oracle_policy": "past_gt_motion",
  "oracle_mean_error": 5.326907634735107,
  "oracle_difference_vs_teacher": -6.984335899353027,
  "sampled_gate_pass": [],
  "oracle_gate_pass": [
    "past_oracle_motion",
    "past_oracle_latent",
    "past_oracle_motion_latent",
    "past_gt_motion",
    "past_gt_motion_plus_teacher"
  ],
  "decision": "ORACLE_STATE_HEADROOM_ONLY: sampled self-state policies fail, but causal past-oracle or past-GT state improves every sequence by at least 0.25 px. Temporal information has headroom while state-error propagation is the blocker. Next build an explicit multi-hypothesis/beam full-stream audit; do not train another single-state reranker."
}
```

Artifacts:

- `scripts/v9a50_temporal_multihypothesis_feasibility.py` — execution_script — `fe92e0470187b4f5b43ebb72fc452a919ab4ab33d98b60aa12bcb891270723f1`
- `outputs/paper_discovery_2026-07-05/v9a50_temporal_feasibility/v9a50_temporal_feasibility.json` — result_json — `fb4afc10673337e0389f2e2c92d987af6f637ac84568220b1029e8b856ea43cd`
- `outputs/paper_discovery_2026-07-05/v9a50_temporal_feasibility/v9a50_temporal_feasibility_selections.npz` — result_npz — `398381f693a1108e936836c1ab276997aaa44cec4e32ab22d1b82eec43c197a8`
- `docs/v9a50_temporal_multihypothesis_feasibility_design_2026-07-10.md` — design — `deb4221a0d81a3de72a3151c0115cddb98918a137dbb905297009411e462fd00`
- `docs/v9a50_temporal_multihypothesis_feasibility_result_2026-07-10.md` — result_doc — `bdcf46f59d60f5f57cce53086c598d2e1d4174ed74bbfc147d7437ed44497f37`

## V9-A5C.0 — Fused correlation candidate-recall oracle

- Commit: `448c095543bf3904898170660ac9944c76963615`
- Status: `oracle_headroom_pass`
- Question: Is useful <=4px candidate recall materially higher at fused K64 than at fused K16?
- Data: PointOdyssey canonical 4,878-row visible candidate pool
- Fair baseline: Fused C1 top16 on the same correlation map
- Primary gate: Every sequence must have fused K64-K16 recall@4 headroom >=0.02
- Original result: All three sequences pass; hard-row headroom is large.
- Final interpretation: Synthetic upstream candidate availability exists, but does not establish deployable selection.

```json
{
  "per_sequence": {
    "ani": {
      "fused_top16_recall4": 0.7198764160659115,
      "fused_top64_recall4": 0.9057672502574665,
      "union_raw_equal_top64_recall4_diagnostic": 0.84346035015448,
      "fused_headroom": 0.18589083419155505,
      "pass_fused_headroom_ge_0.02": true
    },
    "animal3": {
      "fused_top16_recall4": 0.8555045871559633,
      "fused_top64_recall4": 0.9304281345565749,
      "union_raw_equal_top64_recall4_diagnostic": 0.9212538226299695,
      "fused_headroom": 0.07492354740061158,
      "pass_fused_headroom_ge_0.02": true
    },
    "r4_new_f": {
      "fused_top16_recall4": 0.7696560196560197,
      "fused_top64_recall4": 0.9545454545454546,
      "union_raw_equal_top64_recall4_diagnostic": 0.8998771498771498,
      "fused_headroom": 0.18488943488943488,
      "pass_fused_headroom_ge_0.02": true
    }
  },
  "gate_pass": true,
  "hard_fused_k16_recall4": 0.541959487391484,
  "hard_fused_k64_recall4": 0.8561389003720545,
  "decision": "POINTODYSSEY_CANDIDATE_RECALL_HEADROOM_PASS: every sequence has at least two percentage points of fused-map recall@4px between top16 and top64. Freeze this as synthetic upstream headroom. Do not automatically read DAVIS or train an adapter; combine this result with the V9-A5.0 temporal-state audit to choose the next full-stream synthetic experiment."
}
```

Artifacts:

- `scripts/v9a5c0_candidate_recall_correlation_oracle.py` — execution_script — `627ba83754456c9895bc96f68c1360a43337c52e865decac824c49288e4e70aa`
- `outputs/paper_discovery_2026-07-05/v9a5c0_candidate_recall/v9a5c0_pointodyssey_candidate_recall.json` — result_json — `3fd782ac678e0854cbbbba7904447d3a9f4c47fb239a0d6f02388867959332cd`
- `outputs/paper_discovery_2026-07-05/v9a5c0_candidate_recall/v9a5c0_pointodyssey_candidate_recall.npz` — result_npz — `b1df24fd0f8e08dfd52e38a5cca22b9895dd8ed93e5b664157de18d2db9f148b`
- `docs/v9a5c0_candidate_recall_correlation_oracle_design_2026-07-10.md` — design — `759c4be8347e5a6d5850fb919e99e821c0edc7a6426602558954d5cf2346b72e`
- `docs/v9a5c0_pointodyssey_candidate_recall_result_2026-07-10.md` — result_doc — `e04e4510f0f1c6d968fe381229e0d84b89665ad675db4205313903e734d53f35`

## V9-A5.1a — Fixed-K16 shared-state beam baseline

- Commit: `d8e5ca9fb190ce57a3a35725603447a89a177ce3`
- Status: `closed`
- Question: Can fixed-K16 temporal path scoring preserve or improve current-frame candidate reachability?
- Data: 9 PointOdyssey clips; 27,648 query-frame rows
- Fair baseline: Current-frame teacher top1/topB at equal capacity
- Primary gate: Deterministic top1 and same-capacity beam oracle must improve sequence/clip/reentry metrics
- Original result: Both deterministic and same-capacity reachability fail.
- Final interpretation: The exact fixed-K16/raw-C1/shared-state beam configuration is closed.

```json
{
  "teacher_top1_mean": 9.317731857299805,
  "beam4_top1_mean": 9.75613021850586,
  "teacher_top4_oracle_mean": 6.315011501312256,
  "beam4_oracle_mean": 7.4822211265563965,
  "deterministic_pass": false,
  "reachability_pass": false,
  "decision": "FULLSTREAM_BEAM_FAIL: neither deterministic top1 nor GT-only beam reachability passes. Stop the temporal multi-hypothesis branch and return to candidate-recall/correlation-map analysis."
}
```

Artifacts:

- `scripts/v9a51a_fixed_k16_shared_state_beam.py` — execution_script — `4c49df7c743ba14229f30786d13e2776348f9bb66f14aa40b0b6215c94bc5ee8`
- `outputs/paper_discovery_2026-07-05/v9a51a_fixed_k16_shared_state_beam/v9a51a_fixed_k16_shared_state_beam.json` — result_json — `902952433e00ed91f584a4ce74efe6469bd6ee87b23ff1f7a69426e20bf5d0e5`
- `outputs/paper_discovery_2026-07-05/v9a51a_fixed_k16_shared_state_beam/v9a51a_fixed_k16_shared_state_beam.npz` — result_npz — `6606afbebe56b7737bb8b50758cfc206815321b2ae22f958e48ec136f9a519cc`
- `docs/v9a51a_fixed_k16_baseline_freeze_2026-07-10.md` — review — `1fe2e40a6d6e362de643c843ea89fb27c35415beac2a2ca58c783ff1df1869b4`

## V9-A5.1b — Candidate-conditioned downstream refinement

- Commit: `29512f0d95d8b4069b73b9130e60b75d1a892b98`
- Status: `oracle_headroom_pass`
- Question: Can a frozen fused C1 candidate drive a useful singleton-conditioned C2/head refined coordinate?
- Data: 9 PointOdyssey clips; 27,648 query-frame rows
- Fair baseline: Official final TrackOn2 output with risk-gated fallback
- Primary gate: Risk-gated refined candidate oracle must improve all sequences, clip CI, and reentry summaries
- Original result: GT-only refined candidate oracle passes strongly; frozen score-top1 remains unreliable.
- Final interpretation: Candidate refinement has oracle headroom only; it authorizes a temporal candidate-set audit, not deployment.

```json
{
  "official_mean_error": 9.141794939227056,
  "hybrid_oracle_mean_error": 7.81845032557433,
  "oracle_difference_vs_official": -1.3233446136527267,
  "better_worse_equal": [
    4267,
    0,
    15663
  ],
  "gate_pass": true,
  "decision": "CANDIDATE_REFINEMENT_PASS: risk-gated candidate-conditioned refinement has sequence-consistent system-level oracle headroom. Proceed to V9-A5.1c history-preserving beam; do not read DAVIS or train yet."
}
```

Artifacts:

- `scripts/v9a51b_candidate_conditioned_refinement_audit.py` — execution_script — `1a0e4a75db2f48e75fae1d3cab1b11708f77ef1ea96964a26beba545d6e129f5`
- `outputs/paper_discovery_2026-07-05/v9a51b_candidate_refinement/v9a51b_candidate_conditioned_refinement.json` — result_json — `34df580628abacf02fbf45c4475f2eb72572c7753e99c295d53ce95e75d18e86`
- `outputs/paper_discovery_2026-07-05/v9a51b_candidate_refinement/v9a51b_candidate_conditioned_refinement_rows.npz` — result_npz — `eb9736cbe7fc1af08e5bfe23248b333e18ae01c826fd26fc65aa40945746d266`
- `docs/v9a51b_candidate_conditioned_refinement_review_2026-07-10.md` — review — `0a311a983ab844af4aa81a667a297b6317d311c1e0f3c522a6c753660c62a836`

## V9-A5.1c — History-preserving shared-state beam

- Commit: `53f475a283c70c49b6bc2167458b520d11d6b3d9`
- Status: `closed_after_fair_control`
- Question: Does finite-window history-preserving beam pruning add value beyond a same-frame same-capacity candidate set?
- Data: 9 PointOdyssey clips; 27,648 query-frame rows
- Fair baseline: Same-frame frozen-score topB candidate oracle at matched B/K
- Primary gate: Deterministic top1 and capacity-matched temporal oracle must improve across sequence/clip/reentry
- Original result: System-level min(official,beam) oracle passes versus official; deterministic top1 fails.
- Final interpretation: Capacity-matched review reverses the temporal claim: primary B4 is worse than frame-local top4.

```json
{
  "official_mean_error": 9.141794939227056,
  "temporal_top1_mean_error": 9.15751101826647,
  "system_oracle_mean_error": 8.787938573795506,
  "raw_deterministic_pass": false,
  "raw_oracle_pass": true,
  "fair_top1_difference": 0.0551646970121804,
  "fair_top1_ci": [
    -0.10540510300234451,
    0.2998721649794963
  ],
  "fair_oracle_difference": 0.2515971643002531,
  "fair_oracle_ci": [
    0.12796971526112141,
    0.43066073942616556
  ],
  "final_incremental_pass": false,
  "decision": "HISTORY_INCREMENTAL_VALUE_FAIL: the primary history beam is useful relative to official final, but it is weaker than the same-frame, same-capacity score-top4 candidate oracle. The original V9-A5.1c system-level reachability pass is valid, but it is not evidence that temporal history adds incremental candidate reachability. Do not proceed directly to a learned temporal beam readout; first redesign pruning/state so it beats the frame-local capacity-matched baseline."
}
```

Artifacts:

- `scripts/v9a51c_history_preserving_beam_audit.py` — execution_script — `ee96041ce451c0639c94f7558771e6958907dc5030ea1b429228ec97ce3f4da1`
- `scripts/v9a51c_same_capacity_incremental_audit.py` — fair_control_script — `7dca4ed2382d667c58875baf695a9ee4ff0008f2dd519f3bb75f6cbe0c7af505`
- `outputs/paper_discovery_2026-07-05/v9a51c_history_beam/v9a51c_history_preserving_beam.json` — result_json — `ae6bd3eabd36aa8d64014298f48bcc24a31b0611854c8827455a96aa69a6d00b`
- `outputs/paper_discovery_2026-07-05/v9a51c_history_beam/v9a51c_history_preserving_beam_rows.npz` — result_npz — `05b450fb2031ce1d762ed7436bcef34a47dfaf37c10d1c33bd8fd74c1d0381a0`
- `outputs/paper_discovery_2026-07-05/v9a51c_history_beam/v9a51c_same_capacity_incremental_audit.json` — fair_control_json — `38e399cc6d74ec16e9cc3e73bbf8a7976b37febe67e5cfe2dab23d7943a8423f`
- `docs/v9a51c_comprehensive_review_and_v9a52_next_step_2026-07-10.md` — review — `99805ac752eb8b8e01131e56965e1c079a7a18de49a81594b7a34a23c055321c`

## V9-A5.2 — Independent full-model-state branching

- Commit: `bb879183ffff03129cb652824c286f56f9201804`
- Status: `closed`
- Question: Does a complete independent 432-query TrackOn2 state produce useful future outputs beyond a shared-state capacity-2 control?
- Data: 72 frozen first-risk events across 9 PointOdyssey clips; horizons 1/4/8
- Fair baseline: Shared-state capacity-2 oracle: official final plus current-state score-top1 refined output
- Primary gate: Horizon-8 sequence/clip/CI/safe16 plus pooled, early8, and mechanistic novelty gates
- Original result: State and C1 sets diverge, but horizon-8, pooled, early8, and useful-novelty gates fail.
- Final interpretation: The tested independent score-top1 singleton state branch is closed.

```json
{
  "horizon8_independent_mean": 5.47883185304024,
  "horizon8_shared_mean": 5.194916469129649,
  "horizon8_difference": 0.28391538391059096,
  "horizon8_better_worse_equal": [
    14,
    14,
    16
  ],
  "pooled_difference": 0.07336724552981279,
  "q_new_divergence_fraction": 1.0,
  "top16_changed_fraction": 0.5185185185185185,
  "useful_novel_rows": 3,
  "gate_pass": false,
  "decision": "INDEPENDENT_STATE_FAIL: the frozen independent-state branch does not satisfy the capacity-matched statistical and mechanistic gates. Close the temporal multi-state route; do not train or read DAVIS."
}
```

Artifacts:

- `scripts/v9a52_independent_state_branching_audit.py` — execution_script — `d400e8a8b0e713df00620fa113bda1e1bddcf2612821968dd3682811913c7104`
- `scripts/v9a52_candidate_reachability_postaudit.py` — postaudit_script — `db0f8eab66f85ed1acc84f3136a72d299b9e2c7d246c56524b0daaac50c6ae17`
- `outputs/paper_discovery_2026-07-05/v9a52_independent_state/v9a52_independent_state_branching.json` — result_json — `929e87feb67795448cfece53d2d6d3c0eb0cc7174d30f211790e4c2eeac12d74`
- `outputs/paper_discovery_2026-07-05/v9a52_independent_state/v9a52_independent_state_branching_rows.npz` — result_npz — `fd84754a5c5bb8192c9ce8251de2f1bd51b567d6ead6147b6d81849d453a19b5`
- `outputs/paper_discovery_2026-07-05/v9a52_independent_state/v9a52_candidate_reachability_postaudit.json` — postaudit_json — `fae42684d9eae2b92e6a4423d3d5b60bd6a13bbd30e28368a19c86ba970c2ec9`
- `docs/v9a52_comprehensive_review_and_route_closure_2026-07-11.md` — review — `46f668dbd8e22b9df37ec4fb2d4604968276d3da4c3cb11ec1d804e23ed5aea1`

## V9-A6.0 — Bounded fused-correlation rank shift

- Commit: `a1c697de250e1d15f4653cdb77aa85e7e7ce8899`
- Status: `closed`
- Question: Are K16-miss/K64-hit candidates close enough to the K16 score boundary for a small identity-preserving residual?
- Data: 760 visible K16-miss/K64-hit opportunities from the 4,878-row canonical pool
- Fair baseline: Original fused top16 boundary; GT-directed target boost is an optimistic magnitude lower bound
- Primary gate: At span<=2*g_ref: >=50% global conversion, >=40% each sequence, clip/CI/gain and normalized-magnitude gates
- Original result: Only 67/760 opportunities convert at 2*g_ref; every gate fails.
- Final interpretation: Small bounded pre-topK correlation residual promotion is closed; the evaluated pool is visible-only.

```json
{
  "opportunities": 760,
  "primary_promoted": 67,
  "primary_conversion": 0.0881578947368421,
  "primary_global_gain": 0.013735137351373513,
  "primary_hard_gain": 0.027697395618023975,
  "primary_conversion_ci": [
    0.05486725663716814,
    0.14207650273224043
  ],
  "wide_conversion": 0.3473684210526316,
  "span_over_std_median": 0.12653189253198444,
  "span_over_std_p90": 0.2744305452054363,
  "canonical_visible_rows": 4878,
  "canonical_invisible_rows": 0,
  "gate_pass": false,
  "decision": "RANK_SHIFT_MAGNITUDE_FAIL: even the optimistic target-directed oracle does not satisfy the preregistered small-span, sequence/clip-consistency and implied-recall gates. Close the bounded correlation residual route; do not train or read DAVIS.",
  "boundary_decision": "VISIBILITY_STRATIFICATION_NOT_APPLICABLE: all 4,878 canonical candidate-audit rows and all 760 rank-shift opportunity rows are GT-visible by construction. The V9-A6.0 negative result already applies to the complete evaluated opportunity set and is not driven by invisible/occluded rows. Preserve the formal gate and route closure. Add an explicit visible-only sampling caveat to the paper; do not claim invisible-frame correlation recall coverage."
}
```

Artifacts:

- `scripts/v9a60_export_fused_top129.py` — export_script — `684e7a6243d7fccddd23bef79355356dd78ef2b40d2257d1483f790ac680ed2c`
- `scripts/v9a60_bounded_rank_shift_audit.py` — execution_script — `61dd1ea369353a698a683212dcb2c45d6ba0eb9b1d7a603e6fdbc270eec538c6`
- `scripts/v9a60_visibility_stratified_postaudit.py` — boundary_postaudit_script — `0915dfcaca4e1533508b610f1c17293cb3ffe17f6096912d6e4eeb300682b316`
- `outputs/paper_discovery_2026-07-05/v9a60_rank_shift/v9a60_fused_top129_export.json` — export_json — `72fb654381eef236c73b54d8d52d5bfbe4a7b3e4cfaf7ebcf850f1cdf8472cbb`
- `outputs/paper_discovery_2026-07-05/v9a60_rank_shift/v9a60_fused_top129_export.npz` — export_npz — `0849bb9ab700e8f5224bf751f897aea7a38e0efd6363d6d83297e04ecbfac3ac`
- `outputs/paper_discovery_2026-07-05/v9a60_rank_shift/v9a60_bounded_rank_shift_audit.json` — result_json — `a541d4475ca647a798827ac05a5befff588d33ceccfefb1229b85dff3a4f9370`
- `outputs/paper_discovery_2026-07-05/v9a60_rank_shift/v9a60_bounded_rank_shift_audit_rows.npz` — result_npz — `24c3b49d85635aa6879ef084630422ccf63f145951eeae808d72a24d58da30a6`
- `outputs/paper_discovery_2026-07-05/v9a60_rank_shift/v9a60_visibility_stratified_postaudit.json` — boundary_postaudit_json — `f017220b2fcfae00ac01a93391564e7370578aa740c4939ba0f9b70040cb2479`
- `docs/v9a60_comprehensive_review_and_algorithmic_route_closure_2026-07-11.md` — review — `557633cb07cd7f6ca930a6f56899ad43904a57e0030e833240e2d392be5c7e85`
