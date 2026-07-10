# V9-A4.5 Conservative Residual End-to-End Smoke Result

Holdout sequence: `ani`; train rows=2936; validation rows=1942; epochs=1.

## Integrity

```json
{
  "partial_vs_original_position_max_abs": 0.0,
  "partial_vs_original_score_max_abs": 0.0,
  "initial_student_teacher_score_max_abs": 0.0,
  "teacher_student_candidate_max_abs": 0.0,
  "historical_ordered_candidate_max_abs": 5.960464477539063e-08,
  "historical_candidate_set_hausdorff_max": 6.664001972467304e-08,
  "initial_alpha": 0.05000000074505806,
  "manifest": {
    "pass": true,
    "clean_head": "4f3c01d15a5d3ed14639971d79bffef7748fc96e",
    "clean_branch": "v9a45-conservative-residual-20260710",
    "tracked_status": "",
    "checked_files": [
      {
        "path": "/gemini/code/FSPT/outputs/paper_discovery_2026-07-05/v9a38_pointodyssey_pool/v9a38_pointodyssey_hypothesis_pool.npz",
        "size_bytes": 7727883,
        "sha256": "137cbec62d7086f53617acc1541b3df146f1664f70f72bfb6b1f2de0176d964f"
      },
      {
        "path": "/gemini/code/FSPT/outputs/paper_discovery_2026-07-05/v9a43_prefusion_latents/v9a43_pointodyssey_c1_prefusion_latents.npz",
        "size_bytes": 115703317,
        "sha256": "307a69ab17c28014b2d15199dd3f84c91b2b8f78d7d18d8b83ad0f4dba54c489"
      },
      {
        "path": "/gemini/code/FSPT/baselines/track_on/checkpoints_trackon2_dinov3.pt",
        "size_bytes": 93901966,
        "sha256": "0e319c279cbdf51a5fc761b47dc1969520e8cfccfb57dc5a019a8c56e1039cd4"
      },
      {
        "path": "/gemini/code/FSPT/baselines/track_on/config/test.yaml",
        "size_bytes": 249,
        "sha256": "34c08c71be511b2cda6aaafb6e2e3af7e21c6fd5197de98a54fa4b9ac7a78c8e"
      },
      {
        "path": "/gemini/code/FSPT/scripts/v9a44_end_to_end_rerank_pilot_corrected.py",
        "size_bytes": 28986,
        "sha256": "654d585de5f1c98fc3d0bb369dbe6abafbd899bd4a2bec0df34ea22a32e980ce"
      },
      {
        "path": "/gemini/code/FSPT/outputs/paper_discovery_2026-07-05/v9a44_end_to_end_rerank/v9a44_end_to_end_rerank_pilot_corrected.json",
        "size_bytes": 230090,
        "sha256": "449a5ecb142c8eb8f789ce0b73b5ea06138987a5967e182fe8636bc251c6fe94"
      },
      {
        "path": "/gemini/code/FSPT/docs/v9a4_comprehensive_review_and_next_step_2026-07-10.md",
        "size_bytes": 7468,
        "sha256": "e3ff5d5b28bb7cbe830a53ef718d18ff3087a20ce49e7d7084d235c1446a8c0a"
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
        "path": "/gemini/code/FSPT_v9a45_clean/baselines/track_on/config/test.yaml",
        "size_bytes": 249,
        "sha256": "34c08c71be511b2cda6aaafb6e2e3af7e21c6fd5197de98a54fa4b9ac7a78c8e"
      },
      {
        "path": "/gemini/code/FSPT_v9a45_clean/docs/v9a45_pointodyssey_frame_manifest_2026-07-10.json",
        "size_bytes": 148902,
        "sha256": "2cc649b8cd6815c0184e9880ed49a5024996c4d8d4d5eda943ffd7bf49bbde74"
      }
    ],
    "frame_manifest": {
      "path": "/gemini/code/FSPT_v9a45_clean/docs/v9a45_pointodyssey_frame_manifest_2026-07-10.json",
      "file_count": 844,
      "total_bytes": 133075204,
      "aggregate_sha256": "0ba38d4fb90cf5a7afd05794eed95fdecd57e8e80b62b79218e0b46a5af13f37"
    }
  },
  "gt_error_reproduction_max_abs": 4.57763671875e-05,
  "pool_prefusion_alignment_pass": true,
  "train_sequences": [
    "animal3",
    "r4_new_f"
  ],
  "validation_sequences": [
    "ani"
  ],
  "train_validation_sequence_overlap": [],
  "device": "cuda",
  "trackon_code_root": "/gemini/code/FSPT_v9a45_clean/baselines/track_on",
  "checkpoint_path": "/gemini/code/FSPT/baselines/track_on/checkpoints_trackon2_dinov3.pt",
  "base_probe_name": "t_embedding",
  "two_step_gradient": {
    "steps": [
      {
        "step": 0,
        "loss": {
          "total": 1.046965479850769,
          "rank": 0.6526268720626831,
          "kl": 3.3760443329811096e-09,
          "retention": 0.39433860778808594,
          "improvement": 0.0,
          "delta": 0.0,
          "anchor": 0.0,
          "teacher_good_rate": 1.0,
          "improvement_rate": 0.0
        },
        "gradient_norm": 0.28297513723373413,
        "gradient_by_group": {
          "local_decoder": {
            "norm": 0.0,
            "nonzero": 0,
            "elements": 2122560
          },
          "fusion_layer": {
            "norm": 0.0,
            "nonzero": 0,
            "elements": 131328
          },
          "delta_head": {
            "norm": 0.28297513723373413,
            "nonzero": 257,
            "elements": 257
          },
          "alpha_logit": {
            "norm": 0.0,
            "nonzero": 0,
            "elements": 1
          }
        },
        "alpha_before_step": 0.05000000074505806
      },
      {
        "step": 1,
        "loss": {
          "total": 1.0467885732650757,
          "rank": 0.6525576114654541,
          "kl": 1.7229467630386353e-08,
          "retention": 0.3942308723926544,
          "improvement": 0.0,
          "delta": 1.4095627776100628e-08,
          "anchor": 2.8029800702711327e-09,
          "teacher_good_rate": 1.0,
          "improvement_rate": 0.0
        },
        "gradient_norm": 0.28276100754737854,
        "gradient_by_group": {
          "local_decoder": {
            "norm": 0.0009582008933648467,
            "nonzero": 1755758,
            "elements": 2122560
          },
          "fusion_layer": {
            "norm": 0.00018595486471895128,
            "nonzero": 131328,
            "elements": 131328
          },
          "delta_head": {
            "norm": 0.28275927901268005,
            "nonzero": 257,
            "elements": 257
          },
          "alpha_logit": {
            "norm": 0.00014157874102238566,
            "nonzero": 1,
            "elements": 1
          }
        },
        "alpha_before_step": 0.05000000447034836
      }
    ],
    "expected_first_step_zero_upstream": true,
    "second_step_upstream_gradient_pass": true,
    "probe_final_alpha": 0.050029780715703964
  }
}
```

## Training

| Epoch | Total | Rank | KL | Retention | Improve | Alpha | Grad p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4.2277 | 2.5117 | 0.0007 | 0.6203 | 2.1881 | 0.073225 | 4.1105 |

## Heldout synthetic sequence

| Model | Mean | Median | Safe4/8/16 | Oracle regret | Exact best | Within1px |
|---|---:|---:|---:|---:|---:|---:|
| teacher | 14.7458 | 3.5132 | 1008/1233/1563 | 10.2049 | 0.3321 | 0.4439 |
| student | 15.0387 | 3.7056 | 990/1216/1544 | 10.4979 | 0.3326 | 0.4444 |

Better/Worse/Equal: 35/70/1837

Teacher-good retention: 0.9768

Improvement-opportunity success: 0.0324

## Hard/easy diagnostics

| Subset | Teacher mean | Student mean | Better/Worse/Equal | Teacher-good retention |
|---|---:|---:|---|---:|
| hard | 28.1539 | 28.5244 | 32/57/862 | 0.8966 |
| easy | 1.8788 | 2.0973 | 3/13/975 | 0.9893 |

## Synthetic gate

```json
{
  "mean_error_improved": false,
  "better_gt_worse": false,
  "safe16_not_decreased": false,
  "teacher_good_retention_ge_095": true,
  "oracle_regret_improved": false,
  "pass_all": false
}
```

## Decision

Integrity passes, but heldout synthetic ranking collapses materially. Stop before DAVIS and do not scale this objective.
