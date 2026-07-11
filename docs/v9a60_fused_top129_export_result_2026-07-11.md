# V9-A6.0 Fused Top129 Export Result

Date: 2026-07-11

This is a read-only export. It does not train or evaluate a correlation adapter.

## Protocol

```json
{
  "formal_run": true,
  "rows": 4878,
  "clips": [
    "ani:0",
    "ani:256",
    "ani:512",
    "animal3:0",
    "animal3:256",
    "animal3:512",
    "r4_new_f:0",
    "r4_new_f:256",
    "r4_new_f:512"
  ],
  "top_export": 129,
  "component_names": [
    "c4",
    "c8",
    "c16",
    "c32"
  ],
  "training": false,
  "davis_read": false,
  "seconds": 375.4947633855045
}
```

## Integrity

```json
{
  "input_audit": {
    "pass": true,
    "head": "db4f79d8c128a208dcfe636de330fb9309888fba",
    "parent_head": "bb879183ffff03129cb652824c286f56f9201804",
    "branch": "v9a60-correlation-rank-shift-20260711",
    "tracked_status": "",
    "checked_inputs": [
      {
        "path": "/gemini/code/FSPT/outputs/paper_discovery_2026-07-05/v9a38_pointodyssey_pool/v9a38_pointodyssey_hypothesis_pool.npz",
        "size_bytes": 7727883,
        "sha256": "137cbec62d7086f53617acc1541b3df146f1664f70f72bfb6b1f2de0176d964f"
      },
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
        "path": "/gemini/code/FSPT_v9a60_clean/baselines/track_on/config/test.yaml",
        "size_bytes": 249,
        "sha256": "34c08c71be511b2cda6aaafb6e2e3af7e21c6fd5197de98a54fa4b9ac7a78c8e"
      },
      {
        "path": "/gemini/code/FSPT_v9a60_clean/docs/v9a45_pointodyssey_frame_manifest_2026-07-10.json",
        "size_bytes": 148902,
        "sha256": "2cc649b8cd6815c0184e9880ed49a5024996c4d8d4d5eda943ffd7bf49bbde74"
      },
      {
        "path": "/gemini/code/FSPT_v9a60_clean/scripts/v9a5c0_candidate_recall_correlation_oracle.py",
        "size_bytes": 38821,
        "sha256": "627ba83754456c9895bc96f68c1360a43337c52e865decac824c49288e4e70aa"
      },
      {
        "path": "/gemini/code/FSPT_v9a60_clean/docs/v9a5c0_pointodyssey_candidate_recall_result_2026-07-10.md",
        "size_bytes": 7447,
        "sha256": "e04e4510f0f1c6d968fe381229e0d84b89665ad675db4205313903e734d53f35"
      },
      {
        "path": "/gemini/code/FSPT_v9a60_clean/outputs/paper_discovery_2026-07-05/v9a5c0_candidate_recall/v9a5c0_pointodyssey_candidate_recall.json",
        "size_bytes": 152762,
        "sha256": "3fd782ac678e0854cbbbba7904447d3a9f4c47fb239a0d6f02388867959332cd"
      },
      {
        "path": "/gemini/code/FSPT_v9a60_clean/outputs/paper_discovery_2026-07-05/v9a5c0_candidate_recall/v9a5c0_pointodyssey_candidate_recall.npz",
        "size_bytes": 1098050,
        "sha256": "b1df24fd0f8e08dfd52e38a5cca22b9895dd8ed93e5b664157de18d2db9f148b"
      },
      {
        "path": "/gemini/code/FSPT_v9a60_clean/outputs/paper_discovery_2026-07-05/v9a51b_candidate_refinement/v9a51b_candidate_conditioned_refinement_rows.npz",
        "size_bytes": 27603376,
        "sha256": "eb9736cbe7fc1af08e5bfe23248b333e18ae01c826fd26fc65aa40945746d266"
      },
      {
        "path": "/gemini/code/FSPT_v9a60_clean/docs/v9a60_bounded_correlation_rank_shift_design_2026-07-11.md",
        "size_bytes": 11385,
        "sha256": "98893e60568e9788a72af505fd7048bb3cc26f4c2eea0c0923c684369ae82869"
      }
    ],
    "frame_manifest": {
      "path": "/gemini/code/FSPT_v9a60_clean/docs/v9a45_pointodyssey_frame_manifest_2026-07-10.json",
      "file_count": 844,
      "total_bytes": 133075204,
      "aggregate_sha256": "0ba38d4fb90cf5a7afd05794eed95fdecd57e8e80b62b79218e0b46a5af13f37"
    },
    "imported_trackon_code_paths": {
      "model.trackon_predictor": "/gemini/code/FSPT_v9a60_clean/baselines/track_on/model/trackon_predictor.py",
      "model.trackon": "/gemini/code/FSPT_v9a60_clean/baselines/track_on/model/trackon.py",
      "model.reranking": "/gemini/code/FSPT_v9a60_clean/baselines/track_on/model/reranking.py",
      "utils.coord_utils": "/gemini/code/FSPT_v9a60_clean/baselines/track_on/utils/coord_utils.py",
      "utils.train_utils": "/gemini/code/FSPT_v9a60_clean/baselines/track_on/utils/train_utils.py"
    },
    "script": {
      "path": "/gemini/code/FSPT_v9a60_clean/scripts/v9a60_export_fused_top129.py",
      "sha256": "684e7a6243d7fccddd23bef79355356dd78ef2b40d2257d1483f790ac680ed2c"
    }
  },
  "processed_rows": 4878,
  "expected_rows": 4878,
  "row_keys_unique": 4878,
  "top129_unique_every_row": true,
  "top129_nonincreasing_every_row": true,
  "all_outputs_finite": true,
  "parity": {
    "fused_max_abs": 0.0,
    "top129_recompose_max_abs": 0.0,
    "p_max_abs": 0.0,
    "v_max_abs": 0.0,
    "q_max_abs": 0.0,
    "k16_error_max_abs": 0.0,
    "k64_error_max_abs": 0.0,
    "nearest_rank_mismatch": 0,
    "nearest_distance_max_abs": 0.0,
    "nearest_margin_max_abs": 0.0,
    "boundary_gap_k16_max_abs": 0.0,
    "official_top16_set_hausdorff_max": 0.0,
    "pool_top16_ordered_max_abs": 0.18823537230491638,
    "pool_top16_set_hausdorff_max": 8.429369557916289e-08
  },
  "pool_top16_ordered_difference_is_diagnostic_only": true,
  "pool_top16_set_is_hard_gate": true,
  "v9a5c_decision": "POINTODYSSEY_CANDIDATE_RECALL_HEADROOM_PASS: every sequence has at least two percentage points of fused-map recall@4px between top16 and top64. Freeze this as synthetic upstream headroom. Do not automatically read DAVIS or train an adapter; combine this result with the V9-A5.0 temporal-state audit to choose the next full-stream synthetic experiment."
}
```

## Opportunity preview

```json
{
  "rows": 4878,
  "opportunity_rows": 760,
  "opportunity_per_sequence": {
    "ani": 361,
    "animal3": 98,
    "r4_new_f": 301
  },
  "native_risk_rows": 787,
  "native_risk_opportunity_rows": 325,
  "gt_hard_rows": 2419,
  "opportunity_gt_hard_rows": 760
}
```

## Decision

EXPORT_PASS: the formal 4,878-row top129 export reproduces V9-A5C.0 and may be frozen for the no-training rank-shift audit.
