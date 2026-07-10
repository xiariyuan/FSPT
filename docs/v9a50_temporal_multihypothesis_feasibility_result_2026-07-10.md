# V9-A5.0 Sampled-Causal Temporal State Feasibility Audit Result

Date: 2026-07-10

No training and no DAVIS data were used. State updates occur only on the balanced sampled PointOdyssey pool rows, not on every online frame. Temporal evidence is enabled only for observation gaps <= 8.

## Global results

| Policy | Mean | Median | Safe4/8/16 | Oracle regret | Exact best | Better/Worse/Equal | Mean Δ vs teacher | Track 95% CI | Clip 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| teacher | 12.3112 | 3.5114 | 2564/3295/3855 | 7.7564 | 0.3194 | 0/0/4878 | +0.0000 | [+0.0000,+0.0000] | [+0.0000,+0.0000] |
| causal_teacher_motion | 12.1288 | 3.4922 | 2575/3294/3890 | 7.5740 | 0.2794 | 733/917/3228 | -0.1824 | [-0.3637,+0.2920] | [-0.6867,+0.0898] |
| causal_teacher_latent | 12.0118 | 3.4475 | 2575/3307/3868 | 7.4569 | 0.3091 | 457/401/4020 | -0.2995 | [-0.4975,+0.2356] | [-1.0067,+0.0903] |
| causal_teacher_motion_latent | 12.4221 | 3.6314 | 2531/3233/3854 | 7.8673 | 0.2560 | 815/1225/2838 | +0.1109 | [-0.2492,+0.7151] | [-0.7839,+0.4546] |
| past_oracle_motion | 7.3399 | 2.3782 | 3048/3771/4270 | 2.7851 | 0.3928 | 1747/416/2715 | -4.9713 | [-3.1216,-1.1536] | [-7.8101,-1.6995] |
| past_oracle_latent | 6.7530 | 2.1253 | 3151/3843/4308 | 2.1981 | 0.4397 | 1793/286/2799 | -5.5583 | [-3.3306,-1.3254] | [-8.2459,-1.8966] |
| past_oracle_motion_latent | 6.0285 | 1.8818 | 3384/3966/4419 | 1.4736 | 0.5051 | 2278/399/2201 | -6.2828 | [-3.8158,-1.7091] | [-8.9659,-2.5049] |
| past_gt_motion | 5.3269 | 1.6109 | 3480/4085/4537 | 0.7721 | 0.7587 | 2703/484/1691 | -6.9843 | [-4.0741,-1.8403] | [-9.5302,-3.1446] |
| past_gt_motion_plus_teacher | 6.7712 | 2.1405 | 3141/3839/4334 | 2.2164 | 0.4432 | 1898/183/2797 | -5.5400 | [-3.5053,-1.4806] | [-8.2921,-2.1578] |
| frame_oracle | 4.5548 | 1.2061 | 3770/4195/4577 | 0.0000 | 1.0000 | 3320/0/1558 | -7.7564 | [-5.2839,-3.0115] | [-10.3432,-3.9614] |

## Per-sequence mean error

| Policy | ani | animal3 | r4_new_f |
|---|---:|---:|---:|
| teacher | 14.7429 | 10.3798 | 10.9623 |
| causal_teacher_motion | 14.6330 | 9.6802 | 11.1091 |
| causal_teacher_latent | 14.5415 | 9.2886 | 11.1820 |
| causal_teacher_motion_latent | 15.1230 | 9.4693 | 11.5726 |
| past_oracle_motion | 6.4188 | 7.5081 | 8.3035 |
| past_oracle_latent | 6.1517 | 7.4549 | 6.9063 |
| past_oracle_motion_latent | 5.7432 | 6.6374 | 5.8795 |
| past_gt_motion | 5.2253 | 6.1173 | 4.8131 |
| past_gt_motion_plus_teacher | 5.8168 | 7.0475 | 7.6878 |
| frame_oracle | 4.5408 | 5.0978 | 4.1353 |

## Temporal-eligible subset

| Policy | N | Mean | Better/Worse/Equal | Mean Δ |
|---|---:|---:|---:|---:|
| teacher | 4112 | 13.7987 | 0/0/4112 | +0.0000 |
| causal_teacher_motion | 4112 | 13.5823 | 733/917/2462 | -0.2164 |
| causal_teacher_latent | 4112 | 13.4434 | 457/401/3254 | -0.3553 |
| causal_teacher_motion_latent | 4112 | 13.9302 | 815/1225/2072 | +0.1315 |
| past_oracle_motion | 4112 | 7.9013 | 1747/416/1949 | -5.8974 |
| past_oracle_latent | 4112 | 7.2050 | 1793/286/2033 | -6.5937 |
| past_oracle_motion_latent | 4112 | 6.3455 | 2278/399/1435 | -7.4532 |
| past_gt_motion | 4112 | 5.5133 | 2703/484/925 | -8.2854 |
| past_gt_motion_plus_teacher | 4112 | 7.2267 | 1898/183/2031 | -6.5720 |
| frame_oracle | 4112 | 5.0083 | 2967/0/1145 | -8.7904 |

## Gates

### Sampled-causal self-state policies

```json
{
  "causal_teacher_motion": {
    "sequence_mean_improved": {
      "ani": true,
      "animal3": true,
      "r4_new_f": false
    },
    "sequence_safe16_not_decreased": {
      "ani": true,
      "animal3": true,
      "r4_new_f": false
    },
    "global_better_gt_worse": false,
    "clip_bootstrap_ci_upper_lt_zero": false,
    "pass_all": false
  },
  "causal_teacher_latent": {
    "sequence_mean_improved": {
      "ani": true,
      "animal3": true,
      "r4_new_f": false
    },
    "sequence_safe16_not_decreased": {
      "ani": true,
      "animal3": true,
      "r4_new_f": false
    },
    "global_better_gt_worse": true,
    "clip_bootstrap_ci_upper_lt_zero": false,
    "pass_all": false
  },
  "causal_teacher_motion_latent": {
    "sequence_mean_improved": {
      "ani": false,
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
    "pass_all": false
  }
}
```

### Causal past-oracle / past-GT state headroom

```json
{
  "past_oracle_motion": {
    "sequence_mean_improvement_px": {
      "ani": 8.324100494384766,
      "animal3": 2.871729850769043,
      "r4_new_f": 2.6587982177734375
    },
    "each_sequence_improves_at_least_0.25px": true,
    "global_better_gt_worse": true,
    "pass_headroom": true
  },
  "past_oracle_latent": {
    "sequence_mean_improvement_px": {
      "ani": 8.591240882873535,
      "animal3": 2.9248948097229004,
      "r4_new_f": 4.056074142456055
    },
    "each_sequence_improves_at_least_0.25px": true,
    "global_better_gt_worse": true,
    "pass_headroom": true
  },
  "past_oracle_motion_latent": {
    "sequence_mean_improvement_px": {
      "ani": 8.99970006942749,
      "animal3": 3.742426872253418,
      "r4_new_f": 5.082868576049805
    },
    "each_sequence_improves_at_least_0.25px": true,
    "global_better_gt_worse": true,
    "pass_headroom": true
  },
  "past_gt_motion": {
    "sequence_mean_improvement_px": {
      "ani": 9.517620086669922,
      "animal3": 4.262559413909912,
      "r4_new_f": 6.149227619171143
    },
    "each_sequence_improves_at_least_0.25px": true,
    "global_better_gt_worse": true,
    "pass_headroom": true
  },
  "past_gt_motion_plus_teacher": {
    "sequence_mean_improvement_px": {
      "ani": 8.926138401031494,
      "animal3": 3.3323049545288086,
      "r4_new_f": 3.2745351791381836
    },
    "each_sequence_improves_at_least_0.25px": true,
    "global_better_gt_worse": true,
    "pass_headroom": true
  }
}
```

## Decision

ORACLE_STATE_HEADROOM_ONLY: sampled self-state policies fail, but causal past-oracle or past-GT state improves every sequence by at least 0.25 px. Temporal information has headroom while state-error propagation is the blocker. Next build an explicit multi-hypothesis/beam full-stream audit; do not train another single-state reranker.

## Integrity and audit counts

```json
{
  "input_audit": {
    "pass": true,
    "head": "8fe78db05fa23a229efb92329e8fe96a247b7989",
    "branch": "v9a45-conservative-residual-20260710",
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
        "path": "/gemini/code/FSPT/baselines/track_on/checkpoints_trackon2_dinov3.pt",
        "size_bytes": 93901966,
        "sha256": "0e319c279cbdf51a5fc761b47dc1969520e8cfccfb57dc5a019a8c56e1039cd4"
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
        "path": "/gemini/code/FSPT/datasets/pointodyssey/train/ani/rgbs/rgb_00000.jpg",
        "size_bytes": 152415,
        "sha256": "11a448c7717fd60d09ae6637622256547b95375c4f1bccb4b0578f05be614646"
      },
      {
        "path": "/gemini/code/FSPT/datasets/pointodyssey/train/ani/rgbs/rgb_00256.jpg",
        "size_bytes": 149345,
        "sha256": "bccdbe0881b424723bd12137a1804e455d9a5c3a23b3c25f214df66662602718"
      },
      {
        "path": "/gemini/code/FSPT/datasets/pointodyssey/train/ani/rgbs/rgb_00512.jpg",
        "size_bytes": 136506,
        "sha256": "36c07e88e5b248c20bed9b63aad3963387e2abd6989cee1384949db87e0bc656"
      },
      {
        "path": "/gemini/code/FSPT/datasets/pointodyssey/train/animal3/rgbs/rgb_00000.jpg",
        "size_bytes": 215199,
        "sha256": "381e86a8a7ef194896e960cdb0c3e86ee96e627d80be1d335d923cf60b3b8b59"
      },
      {
        "path": "/gemini/code/FSPT/datasets/pointodyssey/train/animal3/rgbs/rgb_00256.jpg",
        "size_bytes": 211433,
        "sha256": "8c8308a7a8973bb345b48f3a02d4b440cc9e43333d13b76e772087090f2718e3"
      },
      {
        "path": "/gemini/code/FSPT/datasets/pointodyssey/train/animal3/rgbs/rgb_00512.jpg",
        "size_bytes": 184398,
        "sha256": "e880206ed71d4ea51cb7977b6e6ab9ca085a02ae1cd38b2929d438cdffefb1d1"
      },
      {
        "path": "/gemini/code/FSPT/datasets/pointodyssey/train/r4_new_f/rgbs/rgb_00000.jpg",
        "size_bytes": 112699,
        "sha256": "a9c3450566966515f50f148e0d1057b98eb241923baf569da8762c59e05279d2"
      },
      {
        "path": "/gemini/code/FSPT/datasets/pointodyssey/train/r4_new_f/rgbs/rgb_00256.jpg",
        "size_bytes": 117199,
        "sha256": "dc98014ca1323e8d5cbc92eae8104aaa85788f5abf10f038ed135b35c1b5f102"
      },
      {
        "path": "/gemini/code/FSPT/datasets/pointodyssey/train/r4_new_f/rgbs/rgb_00512.jpg",
        "size_bytes": 110060,
        "sha256": "a5230c5ec9b6f46bed3f8496e920edc9b5b602b46f1ee350e80713ae13b6a7c4"
      }
    ],
    "dimension_frames": [
      {
        "sequence": "ani",
        "frame_id": 0,
        "path": "/gemini/code/FSPT/datasets/pointodyssey/train/ani/rgbs/rgb_00000.jpg",
        "size_bytes": 152415,
        "sha256": "11a448c7717fd60d09ae6637622256547b95375c4f1bccb4b0578f05be614646",
        "width": 960,
        "height": 540
      },
      {
        "sequence": "ani",
        "frame_id": 256,
        "path": "/gemini/code/FSPT/datasets/pointodyssey/train/ani/rgbs/rgb_00256.jpg",
        "size_bytes": 149345,
        "sha256": "bccdbe0881b424723bd12137a1804e455d9a5c3a23b3c25f214df66662602718",
        "width": 960,
        "height": 540
      },
      {
        "sequence": "ani",
        "frame_id": 512,
        "path": "/gemini/code/FSPT/datasets/pointodyssey/train/ani/rgbs/rgb_00512.jpg",
        "size_bytes": 136506,
        "sha256": "36c07e88e5b248c20bed9b63aad3963387e2abd6989cee1384949db87e0bc656",
        "width": 960,
        "height": 540
      },
      {
        "sequence": "animal3",
        "frame_id": 0,
        "path": "/gemini/code/FSPT/datasets/pointodyssey/train/animal3/rgbs/rgb_00000.jpg",
        "size_bytes": 215199,
        "sha256": "381e86a8a7ef194896e960cdb0c3e86ee96e627d80be1d335d923cf60b3b8b59",
        "width": 960,
        "height": 540
      },
      {
        "sequence": "animal3",
        "frame_id": 256,
        "path": "/gemini/code/FSPT/datasets/pointodyssey/train/animal3/rgbs/rgb_00256.jpg",
        "size_bytes": 211433,
        "sha256": "8c8308a7a8973bb345b48f3a02d4b440cc9e43333d13b76e772087090f2718e3",
        "width": 960,
        "height": 540
      },
      {
        "sequence": "animal3",
        "frame_id": 512,
        "path": "/gemini/code/FSPT/datasets/pointodyssey/train/animal3/rgbs/rgb_00512.jpg",
        "size_bytes": 184398,
        "sha256": "e880206ed71d4ea51cb7977b6e6ab9ca085a02ae1cd38b2929d438cdffefb1d1",
        "width": 960,
        "height": 540
      },
      {
        "sequence": "r4_new_f",
        "frame_id": 0,
        "path": "/gemini/code/FSPT/datasets/pointodyssey/train/r4_new_f/rgbs/rgb_00000.jpg",
        "size_bytes": 112699,
        "sha256": "a9c3450566966515f50f148e0d1057b98eb241923baf569da8762c59e05279d2",
        "width": 960,
        "height": 540
      },
      {
        "sequence": "r4_new_f",
        "frame_id": 256,
        "path": "/gemini/code/FSPT/datasets/pointodyssey/train/r4_new_f/rgbs/rgb_00256.jpg",
        "size_bytes": 117199,
        "sha256": "dc98014ca1323e8d5cbc92eae8104aaa85788f5abf10f038ed135b35c1b5f102",
        "width": 960,
        "height": 540
      },
      {
        "sequence": "r4_new_f",
        "frame_id": 512,
        "path": "/gemini/code/FSPT/datasets/pointodyssey/train/r4_new_f/rgbs/rgb_00512.jpg",
        "size_bytes": 110060,
        "sha256": "a5230c5ec9b6f46bed3f8496e920edc9b5b602b46f1ee350e80713ae13b6a7c4",
        "width": 960,
        "height": 540
      }
    ],
    "script": {
      "path": "/gemini/code/FSPT_v9a45_clean/scripts/v9a50_temporal_multihypothesis_feasibility.py",
      "sha256": "fe92e0470187b4f5b43ebb72fc452a919ab4ab33d98b60aa12bcb891270723f1"
    }
  },
  "candidate_latent_alignment": {
    "rows": 4878,
    "candidate_count": 16,
    "postfusion_score_max_abs": 0.0015439987182617188,
    "postfusion_score_mean_abs": 0.0002093230141326785,
    "postfusion_score_p99_abs": 0.00081634521484375,
    "top1_match_rate": 1.0,
    "top1_mismatch_rows": 0,
    "prefusion_latent_norm_min": 17.87347984313965,
    "prefusion_latent_norm_p01": 18.984902572631835,
    "prefusion_latent_norm_mean": 30.58460235595703,
    "finite": true
  },
  "gt_audit": {
    "max_abs_error_reproduction": 4.57763671875e-05,
    "mean_abs_error_reproduction": 3.1956308248481946e-06,
    "clip_stats": [
      {
        "clip_id": "ani:0",
        "rows": 1012,
        "eligible_tracks": 5854,
        "selected_tracks": 32,
        "width": 960,
        "height": 540
      },
      {
        "clip_id": "ani:256",
        "rows": 714,
        "eligible_tracks": 2522,
        "selected_tracks": 32,
        "width": 960,
        "height": 540
      },
      {
        "clip_id": "ani:512",
        "rows": 216,
        "eligible_tracks": 2521,
        "selected_tracks": 32,
        "width": 960,
        "height": 540
      },
      {
        "clip_id": "animal3:0",
        "rows": 548,
        "eligible_tracks": 3332,
        "selected_tracks": 32,
        "width": 960,
        "height": 540
      },
      {
        "clip_id": "animal3:256",
        "rows": 472,
        "eligible_tracks": 2063,
        "selected_tracks": 32,
        "width": 960,
        "height": 540
      },
      {
        "clip_id": "animal3:512",
        "rows": 288,
        "eligible_tracks": 4229,
        "selected_tracks": 32,
        "width": 960,
        "height": 540
      },
      {
        "clip_id": "r4_new_f:0",
        "rows": 592,
        "eligible_tracks": 6678,
        "selected_tracks": 32,
        "width": 960,
        "height": 540
      },
      {
        "clip_id": "r4_new_f:256",
        "rows": 778,
        "eligible_tracks": 4563,
        "selected_tracks": 32,
        "width": 960,
        "height": 540
      },
      {
        "clip_id": "r4_new_f:512",
        "rows": 258,
        "eligible_tracks": 5174,
        "selected_tracks": 32,
        "width": 960,
        "height": 540
      }
    ]
  },
  "simulation_audit": {
    "tracks": 279,
    "first_rows": 279,
    "temporal_eligible_rows": 4112,
    "gap_gt8_rows": 487,
    "gap_histogram": {
      "1": 2614,
      "2": 473,
      "3": 302,
      "4": 252,
      "5": 157,
      "6": 137,
      "7": 105,
      "8": 72,
      "9": 63,
      "10": 60,
      "11": 35,
      "12": 26,
      "13": 31,
      "14": 15,
      "15": 17,
      "16": 17,
      "17": 16,
      "18": 18,
      "19": 15,
      "20": 10,
      "21": 7,
      "22": 13,
      "23": 9,
      "24": 5,
      "25": 7,
      "26": 6,
      "27": 6,
      "28": 2,
      "29": 8,
      "30": 6,
      "31": 6,
      "32": 6,
      "33": 3,
      "34": 5,
      "35": 5,
      "36": 5,
      "37": 2,
      "38": 5,
      "39": 4,
      "40": 4,
      "41": 2,
      "42": 3,
      "43": 3,
      "44": 3,
      "45": 4,
      "47": 2,
      "48": 1,
      "49": 2,
      "50": 1,
      "51": 3,
      "52": 2,
      "53": 1,
      "54": 1,
      "55": 3,
      "57": 2,
      "59": 2,
      "60": 2,
      "61": 2,
      "62": 1,
      "64": 1,
      "67": 1,
      "68": 2,
      "70": 1,
      "72": 2,
      "73": 1,
      "77": 1,
      "80": 1
    }
  }
}
```
