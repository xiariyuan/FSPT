# V9-A5.1 Full-Stream Deterministic Beam Reachability Audit Result

Date: 2026-07-10

Rows=27648; GT-visible rows=20218; reentry-early8 rows=3806.

## All-visible top1/readout metrics

| Policy | Mean | Median | Safe4/8/16 | Oracle regret | Exact best | Within1px |
|---|---:|---:|---:|---:|---:|---:|
| teacher_top1 | 9.3177 | 1.5491 | 15176/16731/17740 | 4.5067 | 0.4447 | 0.6187 |
| beam1_motion_latent_top1 | 10.4245 | 2.3590 | 14034/16266/17526 | 5.6134 | 0.2512 | 0.4298 |
| beam4_motion_top1 | 9.5300 | 1.7701 | 14975/16721/17754 | 4.7189 | 0.3852 | 0.5661 |
| beam4_latent_top1 | 9.3788 | 1.5870 | 15064/16694/17743 | 4.5677 | 0.4280 | 0.6033 |
| beam4_motion_latent_top1 | 9.7561 | 1.8790 | 14768/16572/17678 | 4.9451 | 0.3536 | 0.5407 |
| beam8_motion_latent_top1 | 9.6145 | 1.8623 | 14854/16669/17748 | 4.8035 | 0.3563 | 0.5456 |
| teacher_top4_oracle | 6.3150 | 1.1489 | 16599/17666/18550 | 1.5040 | 0.7495 | 0.8168 |
| teacher_top8_oracle | 5.2968 | 1.0836 | 17283/18171/18921 | 0.4857 | 0.8760 | 0.9150 |
| beam4_motion_oracle | 7.0740 | 1.1562 | 16472/17482/18355 | 2.2629 | 0.7301 | 0.7980 |
| beam4_latent_oracle | 6.7545 | 1.1578 | 16479/17531/18399 | 1.9435 | 0.7354 | 0.8008 |
| beam4_motion_latent_oracle | 7.4822 | 1.1659 | 16349/17320/18239 | 2.6712 | 0.7171 | 0.7867 |
| beam8_motion_latent_oracle | 6.1592 | 1.0822 | 17089/17932/18685 | 1.3482 | 0.8579 | 0.8907 |
| frame_top16_oracle | 4.8111 | 1.0421 | 17721/18465/19107 | 0.0000 | 1.0000 | 1.0000 |

## Primary pairwise results

| Comparison | Mean Δ | Better/Worse/Equal | Track CI | Clip CI |
|---|---:|---:|---|---|
| primary_top1_vs_teacher | +0.4384 | 2879/5158/12181 | [+0.1468,+0.8315] | [+0.0194,+1.0484] |
| primary_oracle_vs_teacher_top4 | +1.1672 | 1369/2293/16556 | [+0.7363,+1.9009] | [+0.4317,+2.2056] |

## Per-sequence primary means

| Sequence | Teacher | Beam4 top1 | Teacher top4 oracle | Beam4 oracle |
|---|---:|---:|---:|---:|
| ani | 9.9497 | 10.2256 | 5.3631 | 6.7049 |
| animal3 | 7.3322 | 7.4015 | 5.3128 | 5.8356 |
| r4_new_f | 10.6616 | 11.6316 | 8.2726 | 9.9065 |

## Primary beam reachability

```json
{
  "all_visible": {
    "n": 20218,
    "mean_min_error": 7.4822211265563965,
    "median_min_error": 1.1659044027328491,
    "unique_beam_size_mean": 4.0,
    "unique_beam_size_min": 4,
    "recall": {
      "1": 0.3931645068750618,
      "2": 0.7361756850331388,
      "4": 0.8086358690275992,
      "8": 0.8566623800573746
    }
  },
  "teacher_hard": {
    "n": 5042,
    "mean_min_error": 26.46766471862793,
    "median_min_error": 9.859724044799805,
    "unique_beam_size_mean": 4.0,
    "unique_beam_size_min": 4,
    "recall": {
      "1": 0.03153510511701706,
      "2": 0.11761205870686235,
      "4": 0.26715589051963506,
      "8": 0.446449821499405
    }
  },
  "teacher_easy": {
    "n": 15176,
    "mean_min_error": 1.174589991569519,
    "median_min_error": 0.9868188500404358,
    "unique_beam_size_mean": 4.0,
    "unique_beam_size_min": 4,
    "recall": {
      "1": 0.5133104902477597,
      "2": 0.9416842382709542,
      "4": 0.9885345282024249,
      "8": 0.9929493937796521
    }
  },
  "reentry_first": {
    "n": 855,
    "mean_min_error": 15.990769386291504,
    "median_min_error": 3.01094651222229,
    "unique_beam_size_mean": 4.0,
    "unique_beam_size_min": 4,
    "recall": {
      "1": 0.17192982456140352,
      "2": 0.41637426900584795,
      "4": 0.5637426900584795,
      "8": 0.6701754385964912
    }
  },
  "reentry_early4": {
    "n": 2453,
    "mean_min_error": 15.103102684020996,
    "median_min_error": 2.2989532947540283,
    "unique_beam_size_mean": 4.0,
    "unique_beam_size_min": 4,
    "recall": {
      "1": 0.2189156135344476,
      "2": 0.47207501019160214,
      "4": 0.6078271504280472,
      "8": 0.6991439054219323
    }
  },
  "reentry_early8": {
    "n": 3806,
    "mean_min_error": 14.328056335449219,
    "median_min_error": 1.9857465028762817,
    "unique_beam_size_mean": 4.0,
    "unique_beam_size_min": 4,
    "recall": {
      "1": 0.23147661586967946,
      "2": 0.5021019442984761,
      "4": 0.6371518654755649,
      "8": 0.7186022070415135
    }
  },
  "reentry_after_occ4": {
    "n": 312,
    "mean_min_error": 20.561668395996094,
    "median_min_error": 3.7042200565338135,
    "unique_beam_size_mean": 4.0,
    "unique_beam_size_min": 4,
    "recall": {
      "1": 0.13782051282051283,
      "2": 0.3525641025641026,
      "4": 0.5192307692307693,
      "8": 0.6121794871794872
    }
  },
  "sequence_ani": {
    "n": 6775,
    "mean_min_error": 6.704942226409912,
    "median_min_error": 1.1912299394607544,
    "unique_beam_size_mean": 4.0,
    "unique_beam_size_min": 4,
    "recall": {
      "1": 0.3793357933579336,
      "2": 0.7014022140221402,
      "4": 0.7799261992619926,
      "8": 0.8382287822878228
    }
  },
  "sequence_animal3": {
    "n": 6712,
    "mean_min_error": 5.835641384124756,
    "median_min_error": 1.1237868070602417,
    "unique_beam_size_mean": 4.0,
    "unique_beam_size_min": 4,
    "recall": {
      "1": 0.4171632896305125,
      "2": 0.7759237187127532,
      "4": 0.8514600715137068,
      "8": 0.9031585220500596
    }
  },
  "sequence_r4_new_f": {
    "n": 6731,
    "mean_min_error": 9.906512260437012,
    "median_min_error": 1.1862285137176514,
    "unique_beam_size_mean": 4.0,
    "unique_beam_size_min": 4,
    "recall": {
      "1": 0.38315257762591,
      "2": 0.7315406328925865,
      "4": 0.7948298915465756,
      "8": 0.8288515822314664
    }
  }
}
```

## Gates

```json
{
  "deterministic_top1": {
    "per_sequence": {
      "ani": {
        "mean_improved": false,
        "safe16_not_decreased": false
      },
      "animal3": {
        "mean_improved": false,
        "safe16_not_decreased": true
      },
      "r4_new_f": {
        "mean_improved": false,
        "safe16_not_decreased": false
      }
    },
    "global_better_gt_worse": false,
    "clip_ci_upper_lt_zero": false,
    "reentry_early8_mean_improved": false,
    "reentry_early8_safe16_not_decreased": false,
    "pass_all": false
  },
  "beam_reachability": {
    "global_mean_improvement_ge_0.10px": false,
    "per_sequence": {
      "ani": {
        "mean_difference_vs_teacher_top4": 1.3418617248535156,
        "not_worse": false,
        "safe16_not_decreased": false
      },
      "animal3": {
        "mean_difference_vs_teacher_top4": 0.5228481292724609,
        "not_worse": false,
        "safe16_not_decreased": false
      },
      "r4_new_f": {
        "mean_difference_vs_teacher_top4": 1.633955955505371,
        "not_worse": false,
        "safe16_not_decreased": false
      }
    },
    "global_better_gt_worse": false,
    "clip_ci_upper_lt_zero": false,
    "reentry_early8_mean_improvement_ge_0.25px": false,
    "reentry_early8_safe16_not_decreased": false,
    "pass_all": false
  },
  "decision": "FULLSTREAM_BEAM_FAIL: neither deterministic top1 nor GT-only beam reachability passes. Stop the temporal multi-hypothesis branch and return to candidate-recall/correlation-map analysis."
}
```

## Decision

FULLSTREAM_BEAM_FAIL: neither deterministic top1 nor GT-only beam reachability passes. Stop the temporal multi-hypothesis branch and return to candidate-recall/correlation-map analysis.

## Integrity

```json
{
  "input_audit": {
    "pass": true,
    "head": "7c24367cf8aeb1467946d2ced3b294299032538e",
    "branch": "v9a51-fullstream-beam-20260710",
    "tracked_status": "",
    "checked_inputs": [
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
        "path": "/gemini/code/FSPT_v9a51_clean/outputs/paper_discovery_2026-07-05/v9a50_temporal_feasibility/v9a50_temporal_feasibility.json",
        "size_bytes": 190553,
        "sha256": "fb4afc10673337e0389f2e2c92d987af6f637ac84568220b1029e8b856ea43cd"
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
        "path": "/gemini/code/FSPT_v9a51_clean/baselines/track_on/config/test.yaml",
        "size_bytes": 249,
        "sha256": "34c08c71be511b2cda6aaafb6e2e3af7e21c6fd5197de98a54fa4b9ac7a78c8e"
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
        "path": "/gemini/code/FSPT_v9a51_clean/docs/v9a51_pointodyssey_frame_manifest_2026-07-10.json",
        "size_bytes": 152463,
        "sha256": "b1a92a0c5a627fb9a554cd45201dc172bd2b7edaaf13ce78a40eecaddf7effba"
      }
    ],
    "frame_manifest": {
      "path": "/gemini/code/FSPT_v9a51_clean/docs/v9a51_pointodyssey_frame_manifest_2026-07-10.json",
      "file_count": 864,
      "total_bytes": 136082946,
      "aggregate_sha256": "64a73640bebb528177cb641ee21bc7e5eeb2bfd6bd7232ba1436788068bc79d7"
    },
    "imported_trackon_code_paths": {
      "model.trackon_predictor": "/gemini/code/FSPT_v9a51_clean/baselines/track_on/model/trackon_predictor.py",
      "model.trackon": "/gemini/code/FSPT_v9a51_clean/baselines/track_on/model/trackon.py",
      "model.reranking": "/gemini/code/FSPT_v9a51_clean/baselines/track_on/model/reranking.py",
      "utils.coord_utils": "/gemini/code/FSPT_v9a51_clean/baselines/track_on/utils/coord_utils.py"
    },
    "script": {
      "path": "/gemini/code/FSPT_v9a51_clean/scripts/v9a51_fullstream_beam_reachability.py",
      "sha256": "4c49df7c743ba14229f30786d13e2776348f9bb66f14aa40b0b6215c94bc5ee8"
    }
  },
  "processed_rows": 27648,
  "expected_rows": 27648,
  "official_p_max_abs": 0.0,
  "official_v_max_abs": 0.0,
  "official_q_max_abs": 0.0,
  "sampled_rows_checked": 4878,
  "sampled_candidate_ordered_max_abs": 5.960464477539063e-08,
  "sampled_candidate_set_hausdorff_max": 8.429369557916289e-08,
  "sampled_teacher_score_max_abs": 0.0,
  "sampled_teacher_top1_match_rate": 1.0,
  "beam_unique_violations": 0,
  "beam_update_signature": [
    "previous",
    "coords",
    "latents",
    "teacher_rank",
    "current_t",
    "width",
    "use_motion",
    "use_latent"
  ],
  "beam_update_uses_gt_or_error": false,
  "all_finite": true,
  "saved_npz_primary_mean_max_abs": 0.0
}
```
