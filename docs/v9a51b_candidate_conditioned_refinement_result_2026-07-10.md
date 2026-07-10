# V9-A5.1b Candidate-Conditioned Refinement Audit Result

Date: 2026-07-10

No training and no DAVIS data were used. Official TrackOn2 memory remained on the original q_new path.

## Global visible-frame results

| Readout | Mean | Median | Safe4/8/16 | Better/Worse/Equal | Mean Δ | Clip 95% CI |
|---|---:|---:|---:|---:|---:|---|
| official_final | 9.1418 | 1.1232 | 15169/16537/17515 | 0/0/19930 | +0.0000 | [+0.0000,+0.0000] |
| raw_score_top1_K16 | 9.4394 | 1.5673 | 14888/16443/17452 | 6009/13921/0 | +0.2976 | [-0.1781,+0.8436] |
| raw_score_top1_K64 | 9.4869 | 1.5671 | 14875/16430/17317 | 5991/13939/0 | +0.3451 | [-0.9361,+1.6634] |
| raw_oracle_K16 | 4.8677 | 1.0461 | 17433/18177/18819 | 11547/8383/0 | -4.2741 | [-6.9404,-2.3255] |
| raw_oracle_K64 | 2.9606 | 0.9847 | 18721/19003/19289 | 11591/8339/0 | -6.1812 | [-9.7444,-3.4521] |
| refined_score_top1_K16 | 9.1412 | 1.1696 | 15084/16530/17522 | 8940/10990/0 | -0.0006 | [-0.1282,+0.1181] |
| refined_score_top1_K64 | 9.1345 | 1.1679 | 15100/16554/17526 | 9009/10921/0 | -0.0073 | [-0.1665,+0.1095] |
| refined_oracle_K16 | 7.4848 | 0.4797 | 16287/17065/17987 | 19103/827/0 | -1.6570 | [-2.2869,-1.1840] |
| refined_oracle_K64 | 7.0763 | 0.4056 | 16442/17184/18093 | 19587/343/0 | -2.0655 | [-2.9473,-1.4223] |
| hybrid_refined_score_top1 | 9.1023 | 1.1263 | 15162/16553/17522 | 2085/2210/15635 | -0.0394 | [-0.1990,+0.0713] |
| hybrid_refined_oracle | 7.8185 | 1.0183 | 15797/16983/17906 | 4267/0/15663 | -1.3233 | [-2.1037,-0.7501] |

## Primary hybrid refined oracle

| Subset | Official mean | Hybrid oracle mean | N |
|---|---:|---:|---:|
| all_visible | 9.1418 | 7.8185 | 19930 |
| risk_visible | 29.8350 | 23.6943 | 4295 |
| nonrisk_visible | 3.4573 | 3.4573 | 15635 |
| hard_official | 34.9131 | 29.6136 | 4761 |
| opportunity | 28.0760 | 19.6440 | 2916 |
| reentry_first | 17.7697 | 15.3962 | 855 |
| reentry_early4 | 16.5143 | 14.1679 | 2453 |
| reentry_early8 | 15.7624 | 13.5300 | 3806 |

## Risk activation

```json
{
  "all_rows": 0.35745804398148145,
  "visibility_trigger_all_rows": 0.3080150462962963,
  "uncertainty_trigger_all_rows": 0.3136574074074074,
  "trigger_overlap_all_rows": 0.2642144097222222,
  "visible_rows": 0.21550426492724536,
  "invisible_rows": 0.752086137281292,
  "reentry_first_rows": 0.5274853801169591,
  "opportunity_rows": 0.5850480109739369,
  "conceptual_mean_candidate_count": 33.338596491228074,
  "actual_audit_candidate_count": 64,
  "delta_v": 0.8,
  "uncertainty_threshold": 0.5,
  "per_sequence": {
    "ani": {
      "activation_rate": 0.3470394736842105,
      "conceptual_mean_candidate_count": 32.6578947368421
    },
    "animal3": {
      "activation_rate": 0.3575657894736842,
      "conceptual_mean_candidate_count": 33.16315789473684
    },
    "r4_new_f": {
      "activation_rate": 0.3790570175438597,
      "conceptual_mean_candidate_count": 34.194736842105264
    }
  }
}
```

## Refinement diagnostics

```json
{
  "raw_to_refined_displacement_px": {
    "mean": 15.269436836242676,
    "median": 10.302753448486328,
    "p95": 44.24074058532717,
    "max": 272.67974853515625
  },
  "raw_K64_oracle_mean_visible": 2.9605969345129792,
  "refined_K64_oracle_mean_visible": 7.076287100922232,
  "refined_score_oracle_gap_K64_mean_visible": 2.058241844177246,
  "refined_unique_c2_K16_mean_visible": 2.9801806322127447,
  "refined_unique_c2_K64_mean_visible": 3.44039136979428,
  "refined_cluster4_K16_mean_visible": 1.4746111389864527,
  "refined_cluster4_K64_mean_visible": 1.6810336176618164,
  "refined_candidate_out_of_range_rate_visible": 0.00011446312092323131,
  "per_sequence": {
    "ani": {
      "raw_K64_oracle_mean": 2.507932692622287,
      "refined_K64_oracle_mean": 7.406100956351511
    },
    "animal3": {
      "raw_K64_oracle_mean": 3.149774824057682,
      "refined_K64_oracle_mean": 5.07751030564436
    },
    "r4_new_f": {
      "raw_K64_oracle_mean": 3.227626859812177,
      "refined_K64_oracle_mean": 8.737339178864394
    }
  }
}
```

## Gates

```json
{
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
  "pass_all": true,
  "decision": "CANDIDATE_REFINEMENT_PASS: risk-gated candidate-conditioned refinement has sequence-consistent system-level oracle headroom. Proceed to V9-A5.1c history-preserving beam; do not read DAVIS or train yet."
}
```

## Decision

CANDIDATE_REFINEMENT_PASS: risk-gated candidate-conditioned refinement has sequence-consistent system-level oracle headroom. Proceed to V9-A5.1c history-preserving beam; do not read DAVIS or train yet.

## Integrity

```json
{
  "input_audit": {
    "pass": true,
    "head": "d8e5ca9fb190ce57a3a35725603447a89a177ce3",
    "branch": "v9a51b-candidate-conditioned-refinement-20260710",
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
        "path": "/gemini/code/FSPT_v9a51b_clean/baselines/track_on/config/test.yaml",
        "size_bytes": 249,
        "sha256": "34c08c71be511b2cda6aaafb6e2e3af7e21c6fd5197de98a54fa4b9ac7a78c8e"
      },
      {
        "path": "/gemini/code/FSPT_v9a51b_clean/docs/v9a51b_pointodyssey_frame_manifest_2026-07-10.json",
        "size_bytes": 152463,
        "sha256": "b1a92a0c5a627fb9a554cd45201dc172bd2b7edaaf13ce78a40eecaddf7effba"
      }
    ],
    "frame_manifest": {
      "path": "/gemini/code/FSPT_v9a51b_clean/docs/v9a51b_pointodyssey_frame_manifest_2026-07-10.json",
      "file_count": 864,
      "total_bytes": 136082946,
      "aggregate_sha256": "64a73640bebb528177cb641ee21bc7e5eeb2bfd6bd7232ba1436788068bc79d7"
    },
    "imported_trackon_code_paths": {
      "model.trackon_predictor": "/gemini/code/FSPT_v9a51b_clean/baselines/track_on/model/trackon_predictor.py",
      "model.trackon": "/gemini/code/FSPT_v9a51b_clean/baselines/track_on/model/trackon.py",
      "model.reranking": "/gemini/code/FSPT_v9a51b_clean/baselines/track_on/model/reranking.py",
      "model.prediction_head": "/gemini/code/FSPT_v9a51b_clean/baselines/track_on/model/prediction_head.py",
      "utils.coord_utils": "/gemini/code/FSPT_v9a51b_clean/baselines/track_on/utils/coord_utils.py",
      "utils.train_utils": "/gemini/code/FSPT_v9a51b_clean/baselines/track_on/utils/train_utils.py"
    },
    "script": {
      "path": "/gemini/code/FSPT_v9a51b_clean/scripts/v9a51b_candidate_conditioned_refinement_audit.py",
      "sha256": "1a0e4a75db2f48e75fae1d3cab1b11708f77ef1ea96964a26beba545d6e129f5"
    }
  },
  "processed_rows": 27648,
  "expected_rows": 27648,
  "clips": 9,
  "frames_per_clip": 96,
  "official_p_max_abs": 0.0,
  "official_v_max_abs": 0.0,
  "official_q_max_abs": 0.0,
  "extended_first16_ordered_position_max_abs": 0.1882353127002716,
  "extended_first16_set_hausdorff_max": 0.0,
  "extended_first16_matched_score_max_abs": 5.245208740234375e-05,
  "extended_first16_matched_certainty_max_abs": 4.863739013671875e-05,
  "candidate_refinement_signature": [
    "model",
    "q_pre",
    "post_fusion",
    "frame_features"
  ],
  "candidate_refinement_uses_gt_or_error": false,
  "all_outputs_finite": true,
  "extended_parity_tolerance": 6e-05
}
```
