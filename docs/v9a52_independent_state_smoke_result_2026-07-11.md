# V9-A5.2 Independent Model-State Branching Smoke Result

Date: 2026-07-11

This is an integrity smoke only. It is not a scientific result and does not evaluate the 72-event gate.

## Event

```json
{
  "clip_id": "ani:0",
  "query_idx": 20,
  "frame_tau": 2,
  "source_row_index": 84,
  "selection_sha256": "2839633f544d7f29e2ec36784cf32a8d84921083c0cdb2540ba48aaffe9b45eb",
  "smoke_selection": "earliest frozen ani:0 event by frame_tau, query_idx, selection hash",
  "horizons": [
    1,
    4,
    8
  ],
  "active_queries": 432,
  "evaluated_queries": 32,
  "support_queries": 400,
  "memory_size": 24,
  "feature_dim": 256
}
```

## Integrity gates

```json
{
  "official_diagnostic_parity_le_1e6": true,
  "branch_a_full_output_parity_le_1e6": true,
  "branch_a_state_parity": true,
  "storage_disjoint": true,
  "mutation_isolation": true,
  "call_order_invariance_le_1e6": true,
  "online_risk_exact": true,
  "event_candidate_replay": true,
  "shared_and_independent_b2_formulas_exact": true,
  "all_outputs_finite": true,
  "branch_generation_has_no_gt_error_tokens": true,
  "expected_state_dimensions": true
}
```

## Event split

```json
{
  "candidate_score_index": 0,
  "candidate_grid_index": 4512,
  "candidate_score": 5.971920490264893,
  "alternative_q2_vs_official_q_new": {
    "max_abs": 2.3016397953033447,
    "l2": 10.237991333007812,
    "cosine_distance": 0.1103711724281311
  },
  "storage_pointers": {
    "official": {
      "q_init": 139661097304064,
      "point_memory": 139659470438400,
      "temporal_mask": 139664612641792
    },
    "branch_a": {
      "q_init": 139664207036416,
      "point_memory": 139659451564032,
      "temporal_mask": 139664207881728
    },
    "branch_b": {
      "q_init": 139664207904768,
      "point_memory": 139664182738944,
      "temporal_mask": 139664207892480
    }
  },
  "storage_disjoint": true,
  "mutation_isolation": {
    "branch_a_before": -0.8864572644233704,
    "branch_a_after_branch_b_mutation": -0.8864572644233704,
    "branch_a_unchanged": true,
    "branch_b_before": -1.4918824434280396,
    "branch_b_restored": -1.4918824434280396,
    "branch_b_restore_exact": true
  },
  "post_write_target_memory_divergence": {
    "max_abs": 2.3016397953033447,
    "l2": 10.237991333007812,
    "cosine_distance": 0.05439096689224243
  },
  "post_write_full_memory_max_abs": 2.3016397953033447,
  "branch_a_official_post_write": {
    "q_init_max_abs": 0.0,
    "memory_max_abs": 0.0,
    "mask_mismatch": 0
  }
}
```

## Horizons

```json
{
  "1": {
    "frame_tau": 3,
    "gt_visible_valid": false,
    "errors_px": {
      "official": 0.4347270131111145,
      "shared_score_top1": 0.07164596021175385,
      "shared_b2_oracle": 0.07164596021175385,
      "branch_a": 0.4347270131111145,
      "branch_b": 0.5245023965835571,
      "independent_b2_oracle": 0.4347270131111145,
      "independent_minus_shared": 0.36308105289936066
    },
    "useful_novel": false,
    "branch_b_final_le4_while_shared_gt4": false,
    "branch_b_candidate_novel_k16": false,
    "branch_b_candidate_novel_k64": false,
    "target_state_divergence": {
      "q_new": {
        "max_abs": 1.7074559926986694,
        "l2": 8.913959503173828,
        "cosine_distance": 0.057332754135131836
      },
      "q_pre": {
        "max_abs": 0.40040871500968933,
        "l2": 2.150096893310547,
        "cosine_distance": 0.006728529930114746
      },
      "c1": {
        "max_abs": 0.16248512268066406,
        "l2": 6.974567413330078,
        "cosine_distance": 2.294778823852539e-05
      },
      "c2": {
        "max_abs": 0.18459033966064453,
        "l2": 6.930947303771973,
        "cosine_distance": 2.276897430419922e-05
      },
      "final_coordinate_l2_px": 0.09627673774957657,
      "v_logit_abs": 0.36878132820129395,
      "u_logit_abs": 1.0551141500473022
    },
    "full_q_new_row_l2": {
      "mean": 0.047905340790748596,
      "max": 8.913959503173828
    },
    "non_target_evaluated_q_new_row_l2": {
      "mean": 0.030804220587015152,
      "max": 0.13673710823059082
    },
    "support_q_new_row_l2": {
      "mean": 0.027065543457865715,
      "max": 0.5092986226081848
    },
    "pre_update_memory": {
      "target": {
        "max_abs": 2.3016397953033447,
        "l2": 10.237991333007812,
        "cosine_distance": 0.05439096689224243
      },
      "full_max_abs": 2.3016397953033447
    },
    "candidate_sets": {
      "top16": {
        "k": 16,
        "exact_set_equal": false,
        "overlap": 14,
        "jaccard": 0.7777777777777778,
        "hausdorff_px": 6.666666030883789
      },
      "top64": {
        "k": 64,
        "exact_set_equal": false,
        "overlap": 54,
        "jaccard": 0.7297297297297297,
        "hausdorff_px": 26.741567611694336
      }
    },
    "branch_b_refined_oracle_px": {
      "k16": 0.02484189160168171,
      "k64": 0.02484189160168171
    }
  },
  "4": {
    "frame_tau": 6,
    "gt_visible_valid": false,
    "errors_px": {
      "official": 0.13610799610614777,
      "shared_score_top1": 0.2035703957080841,
      "shared_b2_oracle": 0.13610799610614777,
      "branch_a": 0.13610799610614777,
      "branch_b": 0.28309258818626404,
      "independent_b2_oracle": 0.13610799610614777,
      "independent_minus_shared": 0.0
    },
    "useful_novel": false,
    "branch_b_final_le4_while_shared_gt4": false,
    "branch_b_candidate_novel_k16": false,
    "branch_b_candidate_novel_k64": false,
    "target_state_divergence": {
      "q_new": {
        "max_abs": 1.8178119659423828,
        "l2": 2.7783000469207764,
        "cosine_distance": 0.004507184028625488
      },
      "q_pre": {
        "max_abs": 0.865142822265625,
        "l2": 1.5065208673477173,
        "cosine_distance": 0.0024939775466918945
      },
      "c1": {
        "max_abs": 0.17394447326660156,
        "l2": 7.680084228515625,
        "cosine_distance": 2.6524066925048828e-05
      },
      "c2": {
        "max_abs": 0.1645984649658203,
        "l2": 7.353674411773682,
        "cosine_distance": 2.47955322265625e-05
      },
      "final_coordinate_l2_px": 0.14702780544757843,
      "v_logit_abs": 0.059261441230773926,
      "u_logit_abs": 0.048216044902801514
    },
    "full_q_new_row_l2": {
      "mean": 0.0311408843845129,
      "max": 2.7783000469207764
    },
    "non_target_evaluated_q_new_row_l2": {
      "mean": 0.029658230021595955,
      "max": 0.22588831186294556
    },
    "support_q_new_row_l2": {
      "mean": 0.024387892335653305,
      "max": 0.16751526296138763
    },
    "pre_update_memory": {
      "target": {
        "max_abs": 2.3016397953033447,
        "l2": 15.06251049041748,
        "cosine_distance": 0.037605106830596924
      },
      "full_max_abs": 2.3016397953033447
    },
    "candidate_sets": {
      "top16": {
        "k": 16,
        "exact_set_equal": false,
        "overlap": 15,
        "jaccard": 0.8823529411764706,
        "hausdorff_px": 5.3333330154418945
      },
      "top64": {
        "k": 64,
        "exact_set_equal": false,
        "overlap": 59,
        "jaccard": 0.855072463768116,
        "hausdorff_px": 3.3333330154418945
      }
    },
    "branch_b_refined_oracle_px": {
      "k16": 0.2930537760257721,
      "k64": 0.2930537760257721
    }
  },
  "8": {
    "frame_tau": 10,
    "gt_visible_valid": false,
    "errors_px": {
      "official": 6.473275661468506,
      "shared_score_top1": 1.1950976848602295,
      "shared_b2_oracle": 1.1950976848602295,
      "branch_a": 6.473275661468506,
      "branch_b": 6.4788408279418945,
      "independent_b2_oracle": 6.473275661468506,
      "independent_minus_shared": 5.278177976608276
    },
    "useful_novel": false,
    "branch_b_final_le4_while_shared_gt4": false,
    "branch_b_candidate_novel_k16": false,
    "branch_b_candidate_novel_k64": false,
    "target_state_divergence": {
      "q_new": {
        "max_abs": 1.2395591735839844,
        "l2": 2.0466296672821045,
        "cosine_distance": 0.0017580986022949219
      },
      "q_pre": {
        "max_abs": 1.1976709365844727,
        "l2": 1.5876388549804688,
        "cosine_distance": 0.0016151666641235352
      },
      "c1": {
        "max_abs": 0.18647384643554688,
        "l2": 8.718828201293945,
        "cosine_distance": 3.260374069213867e-05
      },
      "c2": {
        "max_abs": 0.13167476654052734,
        "l2": 6.461358547210693,
        "cosine_distance": 1.901388168334961e-05
      },
      "final_coordinate_l2_px": 0.05294344574213028,
      "v_logit_abs": 0.15287017822265625,
      "u_logit_abs": 0.047999173402786255
    },
    "full_q_new_row_l2": {
      "mean": 0.018704064190387726,
      "max": 2.0466296672821045
    },
    "non_target_evaluated_q_new_row_l2": {
      "mean": 0.021417802199721336,
      "max": 0.150778129696846
    },
    "support_q_new_row_l2": {
      "mean": 0.013423933647572994,
      "max": 0.24087563157081604
    },
    "pre_update_memory": {
      "target": {
        "max_abs": 2.3016397953033447,
        "l2": 15.858311653137207,
        "cosine_distance": 0.020664691925048828
      },
      "full_max_abs": 2.3016397953033447
    },
    "candidate_sets": {
      "top16": {
        "k": 16,
        "exact_set_equal": false,
        "overlap": 15,
        "jaccard": 0.8823529411764706,
        "hausdorff_px": 2.6666665077209473
      },
      "top64": {
        "k": 64,
        "exact_set_equal": false,
        "overlap": 62,
        "jaccard": 0.9393939393939394,
        "hausdorff_px": 10.666666030883789
      }
    },
    "branch_b_refined_oracle_px": {
      "k16": 0.7248744964599609,
      "k64": 0.7248744964599609
    }
  }
}
```

## Decision

SMOKE_PASS: full 432-query independent-state splitting, official Branch A parity, storage isolation, call-order invariance, candidate replay and B2 readout formulas all pass. The frozen 72-event audit may be implemented; this smoke is not a scientific result.

## Provenance

```json
{
  "branch": "v9a52-independent-state-branching-20260711",
  "head": "74882d3c8a7ff2539fe7319482c7f236e8ff2a0c",
  "worktree_root": "/gemini/code/FSPT_v9a52_clean",
  "source_asset_root": "/gemini/code/FSPT",
  "device": "cuda",
  "seconds": 28.545969627797604,
  "environment": {
    "head": "74882d3c8a7ff2539fe7319482c7f236e8ff2a0c",
    "branch": "v9a52-independent-state-branching-20260711",
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
        "path": "/gemini/code/FSPT_v9a52_clean/baselines/track_on/config/test.yaml",
        "size_bytes": 249,
        "sha256": "34c08c71be511b2cda6aaafb6e2e3af7e21c6fd5197de98a54fa4b9ac7a78c8e"
      },
      {
        "path": "/gemini/code/FSPT/datasets/pointodyssey/train/ani/anno.npz",
        "size_bytes": 214730314,
        "sha256": "cc20ff10b815d607a09a48895a8d76e9cb8169c108ddb9fdacb2955dc26c2368"
      },
      {
        "path": "/gemini/code/FSPT_v9a52_clean/docs/v9a52_independent_state_event_manifest_2026-07-11.json",
        "size_bytes": 19416,
        "sha256": "e5f5fa34c840e5159726ada1c7752e8bfa273b48ad6cd215e7519ed71721fb4b"
      },
      {
        "path": "/gemini/code/FSPT_v9a52_clean/docs/v9a51c_pointodyssey_frame_manifest_2026-07-10.json",
        "size_bytes": 152463,
        "sha256": "b1a92a0c5a627fb9a554cd45201dc172bd2b7edaaf13ce78a40eecaddf7effba"
      },
      {
        "path": "/gemini/code/FSPT_v9a52_clean/scripts/v9a51b_candidate_conditioned_refinement_audit.py",
        "size_bytes": 57638,
        "sha256": "1a0e4a75db2f48e75fae1d3cab1b11708f77ef1ea96964a26beba545d6e129f5"
      },
      {
        "path": "/gemini/code/FSPT_v9a52_clean/outputs/paper_discovery_2026-07-05/v9a51b_candidate_refinement/v9a51b_candidate_conditioned_refinement_rows.npz",
        "size_bytes": 27603376,
        "sha256": "eb9736cbe7fc1af08e5bfe23248b333e18ae01c826fd26fc65aa40945746d266"
      },
      {
        "path": "/gemini/code/FSPT_v9a52_clean/outputs/paper_discovery_2026-07-05/v9a51c_history_beam/v9a51c_history_preserving_beam_rows.npz",
        "size_bytes": 7237192,
        "sha256": "05b450fb2031ce1d762ed7436bcef34a47dfaf37c10d1c33bd8fd74c1d0381a0"
      }
    ],
    "checked_smoke_frames": [
      {
        "relative_path": "ani/rgbs/rgb_00000.jpg",
        "size_bytes": 152415,
        "sha256": "11a448c7717fd60d09ae6637622256547b95375c4f1bccb4b0578f05be614646"
      },
      {
        "relative_path": "ani/rgbs/rgb_00001.jpg",
        "size_bytes": 152128,
        "sha256": "29fa2878b15afacd135015f965ff9f33d1ceaf1d0f166c2d3be63421b9b1f255"
      },
      {
        "relative_path": "ani/rgbs/rgb_00002.jpg",
        "size_bytes": 148229,
        "sha256": "f28b08bca30aadccfad13686c2acf48300ee8fc91381df93948dee467f09708c"
      },
      {
        "relative_path": "ani/rgbs/rgb_00003.jpg",
        "size_bytes": 147654,
        "sha256": "bc5e997528615d6948be07a6d114b6bd30ec8be9fe5b15faf2ecdc28d02d8fbe"
      },
      {
        "relative_path": "ani/rgbs/rgb_00004.jpg",
        "size_bytes": 145747,
        "sha256": "8f817b8e1eacca4ad445f441d7f9367efcc4b2882981703f4e4f57562df942a1"
      },
      {
        "relative_path": "ani/rgbs/rgb_00005.jpg",
        "size_bytes": 148245,
        "sha256": "f1d585ea6817ecb83b7290b52534e563d2b667ffae4f4ce30262a37d75e48e1e"
      },
      {
        "relative_path": "ani/rgbs/rgb_00006.jpg",
        "size_bytes": 149127,
        "sha256": "9a588a502901a80620e8c2c07d25613d8c3ba50b8392c401bbbaad3a4359b904"
      },
      {
        "relative_path": "ani/rgbs/rgb_00007.jpg",
        "size_bytes": 149918,
        "sha256": "3533d3694b78510095f528e49b5ef3399bcc699929560fdbb7745d6560c103fd"
      },
      {
        "relative_path": "ani/rgbs/rgb_00008.jpg",
        "size_bytes": 150157,
        "sha256": "1277c2655d6d5e703890084693a48eb8b1f10fac8ef2c21915ac88212d88cd01"
      },
      {
        "relative_path": "ani/rgbs/rgb_00009.jpg",
        "size_bytes": 150865,
        "sha256": "075860500e37406e1e4871f60548c4b8701d371c815baaec89011e13fc91034c"
      },
      {
        "relative_path": "ani/rgbs/rgb_00010.jpg",
        "size_bytes": 152368,
        "sha256": "3c350fd415898af9c33db101fa3bedebecbc46a66f0a233fe4acf378507beb62"
      }
    ],
    "imported_trackon_code_paths": {
      "model.trackon_predictor": "/gemini/code/FSPT_v9a52_clean/baselines/track_on/model/trackon_predictor.py",
      "model.trackon": "/gemini/code/FSPT_v9a52_clean/baselines/track_on/model/trackon.py",
      "model.reranking": "/gemini/code/FSPT_v9a52_clean/baselines/track_on/model/reranking.py",
      "model.prediction_head": "/gemini/code/FSPT_v9a52_clean/baselines/track_on/model/prediction_head.py",
      "model.modules": "/gemini/code/FSPT_v9a52_clean/baselines/track_on/model/modules.py",
      "utils.coord_utils": "/gemini/code/FSPT_v9a52_clean/baselines/track_on/utils/coord_utils.py",
      "utils.train_utils": "/gemini/code/FSPT_v9a52_clean/baselines/track_on/utils/train_utils.py"
    },
    "branch_function_forbidden_hits": {
      "diagnostic_track_frame": [],
      "extended_candidates": [],
      "candidate_conditioned_states": [],
      "build_candidate_bundle": [],
      "update_state": [],
      "split_event_states": []
    },
    "script": {
      "path": "/gemini/code/FSPT_v9a52_clean/scripts/v9a52_independent_state_branching_smoke.py",
      "sha256": "1e0a0324bf0170bf7f9db1d91de51075bcd690410f4014b534c834628da3ec37"
    }
  }
}
```
