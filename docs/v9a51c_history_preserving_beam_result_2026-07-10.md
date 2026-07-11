# V9-A5.1c History-Preserving Risk-Gated Beam Result

Date: 2026-07-10

No training and no DAVIS data were used. Official TrackOn2 query/memory state remained unchanged.

## Global visible-frame results

| Readout | Mean | Median | Safe4/8/16 | Better/Worse/Equal | Mean Δ | Clip 95% CI |
|---|---:|---:|---:|---:|---:|---|
| official_final | 9.1418 | 1.1232 | 15169/16537/17515 | 0/0/19930 | +0.0000 | [+0.0000,+0.0000] |
| v9a51b_hybrid_score_top1 | 9.1023 | 1.1263 | 15162/16553/17522 | 2085/2210/15635 | -0.0394 | [-0.1992,+0.0713] |
| v9a51b_full_candidate_oracle | 7.8185 | 1.0183 | 15797/16983/17906 | 4267/0/15663 | -1.3233 | [-2.0984,-0.7537] |
| beam_top1_native_dynamic_B4 | 9.1575 | 1.1261 | 15148/16535/17514 | 2124/2171/15635 | +0.0157 | [-0.0573,+0.1110] |
| beam_oracle_native_dynamic_B4 | 8.7879 | 1.0894 | 15367/16688/17627 | 3107/0/16823 | -0.3539 | [-0.5043,-0.2343] |
| beam_raw_min_native_dynamic_B4 | 9.3065 | 1.3059 | 15252/16367/17333 | 8570/11360/0 | +0.1647 | [-0.6785,+1.2661] |
| beam_top1_native_dynamic_B1 | 9.1118 | 1.1306 | 15141/16529/17513 | 2123/2172/15635 | -0.0300 | [-0.1798,+0.0692] |
| beam_oracle_native_dynamic_B1 | 8.8697 | 1.1094 | 15309/16632/17596 | 2123/0/17807 | -0.2721 | [-0.4309,-0.1533] |
| beam_top1_native_dynamic_B8 | 9.1314 | 1.1242 | 15156/16532/17512 | 2158/2137/15635 | -0.0104 | [-0.0590,+0.0289] |
| beam_oracle_native_dynamic_B8 | 8.7181 | 1.0780 | 15420/16704/17645 | 3434/0/16496 | -0.4237 | [-0.5982,-0.2831] |
| beam_top1_fixed16_B4 | 9.0284 | 1.1299 | 15130/16546/17530 | 2104/2191/15635 | -0.1134 | [-0.3290,+0.0060] |
| beam_oracle_fixed16_B4 | 8.6599 | 1.0883 | 15394/16693/17655 | 3186/0/16744 | -0.4819 | [-0.7677,-0.2723] |
| beam_top1_fixed64_B4 | 9.0772 | 1.1263 | 15153/16536/17528 | 2152/2143/15635 | -0.0646 | [-0.3018,+0.1097] |
| beam_oracle_fixed64_B4 | 8.7038 | 1.0912 | 15371/16688/17636 | 3026/0/16904 | -0.4380 | [-0.7178,-0.2458] |

## Primary beam diagnostics

```json
{
  "width": 4,
  "raw_survival_risk_visible": {
    "1": 0.0731082654249127,
    "2": 0.17462165308498254,
    "4": 0.29383003492433063,
    "8": 0.42770663562281724
  },
  "refined_survival_risk_visible": {
    "1": 0.08661233993015134,
    "2": 0.18160651920838183,
    "4": 0.3168800931315483,
    "8": 0.46216530849825377
  },
  "mean_unique_current_grid_risk_visible": 2.8083818393480793,
  "mean_unique_signature_risk_visible": 4.0,
  "raw_cluster4_mean_risk_visible": 1.4481955762514551,
  "refined_cluster4_mean_risk_visible": 1.1313154831199068,
  "top1_oracle_gap_mean_risk_visible": 1.7149194478988647,
  "raw_state_min_mean_visible": 9.3065185546875,
  "raw_state_min_mean_risk_visible": 29.96604347229004,
  "beam_oracle_gap_to_full_candidate_mean_visible": 0.9694882035255432,
  "full_candidate_headroom_retained_fraction": 0.26739547792832885
}
```

## Gates

```json
{
  "primary_policy": "native_dynamic_B4",
  "deterministic_top1": {
    "sequence_mean_improved": {
      "ani": true,
      "animal3": true,
      "r4_new_f": false
    },
    "sequence_safe16_not_decreased": {
      "ani": false,
      "animal3": true,
      "r4_new_f": false
    },
    "global_better_gt_worse": false,
    "clip_bootstrap_ci_upper_lt_zero": false,
    "first_reentry_mean_improved": false,
    "early8_mean_improved": false,
    "reentry_sequence_nonincrease_when_n_ge10": {
      "ani": false,
      "animal3": false,
      "r4_new_f": false
    },
    "all_clip_mean_differences_nonpositive": false,
    "pass_all": false
  },
  "beam_reachability": {
    "sequence_mean_improved": {
      "ani": true,
      "animal3": true,
      "r4_new_f": true
    },
    "sequence_safe16_not_decreased": {
      "ani": true,
      "animal3": true,
      "r4_new_f": true
    },
    "global_better_gt_worse": true,
    "clip_bootstrap_ci_upper_lt_zero": true,
    "first_reentry_mean_improved": true,
    "early8_mean_improved": true,
    "reentry_sequence_nonincrease_when_n_ge10": {
      "ani": true,
      "animal3": true,
      "r4_new_f": true
    },
    "all_clip_mean_differences_nonpositive": true,
    "pass_all": true
  },
  "pass_all": true,
  "decision": "HISTORY_BEAM_REACHABILITY_ONLY: deterministic path scoring fails, but the surviving beam preserves sequence-consistent refined alternatives. A later sequence-heldout learned beam readout is justified; do not read DAVIS."
}
```

## Decision

HISTORY_BEAM_REACHABILITY_ONLY: deterministic path scoring fails, but the surviving beam preserves sequence-consistent refined alternatives. A later sequence-heldout learned beam readout is justified; do not read DAVIS.

## Replay and integrity

```json
{
  "input_audit": {
    "pass": true,
    "head": "29512f0d95d8b4069b73b9130e60b75d1a892b98",
    "branch": "v9a51c-history-preserving-beam-20260710",
    "tracked_status": "",
    "checked_inputs": [
      {
        "path": "/gemini/code/FSPT/baselines/track_on/checkpoints_trackon2_dinov3.pt",
        "size_bytes": 93901966,
        "sha256": "0e319c279cbdf51a5fc761b47dc1969520e8cfccfb57dc5a019a8c56e1039cd4"
      },
      {
        "path": "/gemini/code/FSPT/third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m/model.safetensors",
        "size_bytes": 114794096,
        "sha256": "208146e499dace99e4c9376ddb8a26f77d64c31c46c4dc4b86ff8bc63b0235e2"
      },
      {
        "path": "/gemini/code/FSPT/third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m/config.json",
        "size_bytes": 742,
        "sha256": "6f4ac67fea1761fe684d2a7db3139bab2d0dfdf94c05063d5992717c4c1da0ac"
      },
      {
        "path": "/gemini/code/FSPT/datasets/pointodyssey/train/ani/anno.npz",
        "size_bytes": 214730314,
        "sha256": "cc20ff10b815d607a09a48895a8d76e9cb8169c108ddb9fdacb2955dc26c2368"
      },
      {
        "path": "/gemini/code/FSPT/datasets/pointodyssey/train/animal3/anno.npz",
        "size_bytes": 197918569,
        "sha256": "84110e093f48da96393ddaddeaad15352b11cfa80a298b56ecdaba4365e89dfa"
      },
      {
        "path": "/gemini/code/FSPT/datasets/pointodyssey/train/r4_new_f/anno.npz",
        "size_bytes": 98527880,
        "sha256": "382d1edca0d6d88279ed1a61ee63c9ba27b7c5418f2269f4592c1658fd7d8e77"
      },
      {
        "path": "/gemini/code/FSPT_v9a51c_clean/baselines/track_on/config/test.yaml",
        "size_bytes": 249,
        "sha256": "34c08c71be511b2cda6aaafb6e2e3af7e21c6fd5197de98a54fa4b9ac7a78c8e"
      },
      {
        "path": "/gemini/code/FSPT_v9a51c_clean/docs/v9a51c_pointodyssey_frame_manifest_2026-07-10.json",
        "size_bytes": 152463,
        "sha256": "b1a92a0c5a627fb9a554cd45201dc172bd2b7edaaf13ce78a40eecaddf7effba"
      },
      {
        "path": "/gemini/code/FSPT_v9a51c_clean/scripts/v9a51b_candidate_conditioned_refinement_audit.py",
        "size_bytes": 57638,
        "sha256": "1a0e4a75db2f48e75fae1d3cab1b11708f77ef1ea96964a26beba545d6e129f5"
      },
      {
        "path": "/gemini/code/FSPT_v9a51c_clean/outputs/paper_discovery_2026-07-05/v9a51b_candidate_refinement/v9a51b_candidate_conditioned_refinement.json",
        "size_bytes": 157272,
        "sha256": "34df580628abacf02fbf45c4475f2eb72572c7753e99c295d53ce95e75d18e86"
      },
      {
        "path": "/gemini/code/FSPT_v9a51c_clean/outputs/paper_discovery_2026-07-05/v9a51b_candidate_refinement/v9a51b_candidate_conditioned_refinement_rows.npz",
        "size_bytes": 27603376,
        "sha256": "eb9736cbe7fc1af08e5bfe23248b333e18ae01c826fd26fc65aa40945746d266"
      }
    ],
    "frame_manifest": {
      "path": "/gemini/code/FSPT_v9a51c_clean/docs/v9a51c_pointodyssey_frame_manifest_2026-07-10.json",
      "file_count": 864,
      "total_bytes": 136082946,
      "aggregate_sha256": "64a73640bebb528177cb641ee21bc7e5eeb2bfd6bd7232ba1436788068bc79d7"
    },
    "imported_trackon_code_paths": {
      "model.trackon_predictor": "/gemini/code/FSPT_v9a51c_clean/baselines/track_on/model/trackon_predictor.py",
      "model.trackon": "/gemini/code/FSPT_v9a51c_clean/baselines/track_on/model/trackon.py",
      "model.reranking": "/gemini/code/FSPT_v9a51c_clean/baselines/track_on/model/reranking.py",
      "model.prediction_head": "/gemini/code/FSPT_v9a51c_clean/baselines/track_on/model/prediction_head.py",
      "utils.coord_utils": "/gemini/code/FSPT_v9a51c_clean/baselines/track_on/utils/coord_utils.py",
      "utils.train_utils": "/gemini/code/FSPT_v9a51c_clean/baselines/track_on/utils/train_utils.py"
    },
    "v9a51b_replay": {
      "script_sha256": "1a0e4a75db2f48e75fae1d3cab1b11708f77ef1ea96964a26beba545d6e129f5",
      "json_sha256": "34df580628abacf02fbf45c4475f2eb72572c7753e99c295d53ce95e75d18e86",
      "npz_sha256": "eb9736cbe7fc1af08e5bfe23248b333e18ae01c826fd26fc65aa40945746d266"
    },
    "script": {
      "path": "/gemini/code/FSPT_v9a51c_clean/scripts/v9a51c_history_preserving_beam_audit.py",
      "sha256": "ee96041ce451c0639c94f7558771e6958907dc5030ea1b429228ec97ce3f4da1"
    }
  },
  "processed_rows": 27648,
  "expected_rows": 27648,
  "clips": 9,
  "frames_per_clip": 96,
  "official_p_max_abs": 0.0,
  "official_v_max_abs": 0.0,
  "official_q_max_abs": 0.0,
  "v9a51b_replay": {
    "row_key_mismatch": 0,
    "official_max_abs": 0.0,
    "risk_mismatch": 0,
    "score_max_abs": 0.0,
    "raw_error_max_abs": 0.0,
    "refined_error_max_abs": 0.0,
    "hybrid_oracle_max_abs": 0.0
  },
  "beam_width_shape_pass": {
    "native_dynamic_B1": true,
    "native_dynamic_B4": true,
    "native_dynamic_B8": true,
    "fixed16_B4": true,
    "fixed64_B4": true
  },
  "signature_unique_violations": {
    "native_dynamic_B1": 0,
    "native_dynamic_B4": 0,
    "native_dynamic_B8": 0,
    "fixed16_B4": 0,
    "fixed64_B4": 0
  },
  "beam_update_signature": [
    "parents",
    "raw_coords",
    "refined_coords",
    "descriptors",
    "scores",
    "grid_indices",
    "candidate_budget",
    "beam_width"
  ],
  "beam_update_uses_gt_or_error": false,
  "all_outputs_finite": true
}
```
