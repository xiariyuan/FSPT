# V9-A5.2 Independent Model-State Branching Result

Date: 2026-07-11

No training and no DAVIS data were used.

## Primary horizon-8 comparison

```json
{
  "n": 44,
  "candidate_mean": 5.47883185304024,
  "baseline_mean": 5.194916469129649,
  "mean_difference": 0.28391538391059096,
  "median_difference": 0.0,
  "candidate_safe4": 32,
  "baseline_safe4": 32,
  "candidate_safe8": 38,
  "baseline_safe8": 38,
  "candidate_safe16": 41,
  "baseline_safe16": 42,
  "better": 14,
  "worse": 14,
  "equal": 16
}
```

## Pooled horizons comparison

```json
{
  "n": 108,
  "candidate_mean": 3.983764873144941,
  "baseline_mean": 3.9103976276151284,
  "mean_difference": 0.07336724552981279,
  "median_difference": 0.0,
  "candidate_safe4": 85,
  "baseline_safe4": 85,
  "candidate_safe8": 95,
  "baseline_safe8": 96,
  "candidate_safe16": 103,
  "baseline_safe16": 103,
  "better": 41,
  "worse": 37,
  "equal": 30
}
```

## Mechanistic diagnostics

```json
{
  "visible_event_horizon_rows": 108,
  "target_q_new_l2_gt_1e4_fraction_visible": 1.0,
  "top16_set_changed_fraction_visible": 0.5185185185185185,
  "target_q_new_l2": {
    "mean_visible": 1.6416473388671875,
    "median_visible": 0.7039914727210999,
    "max_visible": 15.483562469482422
  },
  "top16_overlap_mean_visible": 15.064814814814815,
  "top64_overlap_mean_visible": 61.5462962962963,
  "final_coordinate_divergence_mean_visible": 0.5492970943450928,
  "useful_novel_rows_visible": 3,
  "useful_novel_per_sequence": {
    "ani": 1,
    "animal3": 2,
    "r4_new_f": 0
  },
  "branch_b_final_le4_while_shared_gt4": 0,
  "branch_b_candidate_novel_k16": 12,
  "branch_b_candidate_novel_k64": 14
}
```

## Candidate-reachability supplement (non-gating)

```json
{
  "non_gating": true,
  "interpretation": "Post-formal conservative diagnostic. It compares the complete Branch B refined candidate pool against the complete shared-state candidate pool at matched K. It cannot alter the predeclared gates.",
  "pooled_visible": {
    "branch_b_vs_shared_k16": {
      "n": 108,
      "candidate_mean": 2.68656334975148,
      "baseline_mean": 2.128117856586835,
      "mean_difference": 0.5584454931646448,
      "median_difference": -0.00021331897005438805,
      "candidate_safe4": 97,
      "baseline_safe4": 99,
      "candidate_safe8": 99,
      "baseline_safe8": 101,
      "candidate_safe16": 105,
      "baseline_safe16": 105,
      "better": 56,
      "worse": 52,
      "equal": 0
    },
    "branch_b_vs_shared_k64": {
      "n": 108,
      "candidate_mean": 1.9562982445263684,
      "baseline_mean": 1.9581187413919166,
      "mean_difference": -0.0018204968655481935,
      "median_difference": -0.0008327718824148178,
      "candidate_safe4": 99,
      "baseline_safe4": 99,
      "candidate_safe8": 100,
      "baseline_safe8": 101,
      "candidate_safe16": 106,
      "baseline_safe16": 105,
      "better": 59,
      "worse": 49,
      "equal": 0
    },
    "union_vs_shared_k16": {
      "n": 108,
      "candidate_mean": 2.010442873281944,
      "baseline_mean": 2.128117856586835,
      "mean_difference": -0.11767498330489078,
      "median_difference": -0.00021331897005438805,
      "candidate_safe4": 99,
      "baseline_safe4": 99,
      "candidate_safe8": 101,
      "baseline_safe8": 101,
      "candidate_safe16": 106,
      "baseline_safe16": 105,
      "better": 56,
      "worse": 0,
      "equal": 52
    },
    "union_vs_shared_k64": {
      "n": 108,
      "candidate_mean": 1.839032671892912,
      "baseline_mean": 1.9581187413919166,
      "mean_difference": -0.1190860694990045,
      "median_difference": -0.0008327718824148178,
      "candidate_safe4": 99,
      "baseline_safe4": 99,
      "candidate_safe8": 101,
      "baseline_safe8": 101,
      "candidate_safe16": 106,
      "baseline_safe16": 105,
      "better": 59,
      "worse": 0,
      "equal": 49
    }
  },
  "horizon8_visible": {
    "branch_b_vs_shared_k16": {
      "n": 44,
      "candidate_mean": 4.119757343655113,
      "baseline_mean": 2.7908775077032093,
      "mean_difference": 1.328879835951904,
      "median_difference": 0.0007413774728775024,
      "candidate_safe4": 38,
      "baseline_safe4": 39,
      "candidate_safe8": 39,
      "baseline_safe8": 40,
      "candidate_safe16": 42,
      "baseline_safe16": 43,
      "better": 20,
      "worse": 24,
      "equal": 0
    },
    "branch_b_vs_shared_k64": {
      "n": 44,
      "candidate_mean": 2.627223933898759,
      "baseline_mean": 2.7020364797525955,
      "mean_difference": -0.07481254585383629,
      "median_difference": 0.00032826513051986694,
      "candidate_safe4": 39,
      "baseline_safe4": 39,
      "candidate_safe8": 40,
      "baseline_safe8": 40,
      "candidate_safe16": 43,
      "baseline_safe16": 43,
      "better": 21,
      "worse": 23,
      "equal": 0
    },
    "union_vs_shared_k16": {
      "n": 44,
      "candidate_mean": 2.700709481660107,
      "baseline_mean": 2.7908775077032093,
      "mean_difference": -0.09016802604310215,
      "median_difference": 0.0,
      "candidate_safe4": 39,
      "baseline_safe4": 39,
      "candidate_safe8": 40,
      "baseline_safe8": 40,
      "candidate_safe16": 43,
      "baseline_safe16": 43,
      "better": 20,
      "worse": 0,
      "equal": 24
    },
    "union_vs_shared_k64": {
      "n": 44,
      "candidate_mean": 2.558927813767117,
      "baseline_mean": 2.7020364797525955,
      "mean_difference": -0.14310866598547858,
      "median_difference": 0.0,
      "candidate_safe4": 39,
      "baseline_safe4": 39,
      "candidate_safe8": 40,
      "baseline_safe8": 40,
      "candidate_safe16": 43,
      "baseline_safe16": 43,
      "better": 21,
      "worse": 0,
      "equal": 23
    }
  },
  "per_horizon_visible": {
    "1": {
      "branch_b_vs_shared_k16": {
        "n": 28,
        "candidate_mean": 1.8560482749848493,
        "baseline_mean": 1.6438529192070876,
        "mean_difference": 0.21219535577776177,
        "median_difference": -0.0062264129519462585,
        "candidate_safe4": 25,
        "baseline_safe4": 26,
        "candidate_safe8": 26,
        "baseline_safe8": 27,
        "candidate_safe16": 28,
        "baseline_safe16": 28,
        "better": 14,
        "worse": 14,
        "equal": 0
      },
      "branch_b_vs_shared_k64": {
        "n": 28,
        "candidate_mean": 1.5389133719228474,
        "baseline_mean": 1.2861722795059904,
        "mean_difference": 0.252741092416857,
        "median_difference": 0.005352984182536602,
        "candidate_safe4": 26,
        "baseline_safe4": 26,
        "candidate_safe8": 26,
        "baseline_safe8": 27,
        "candidate_safe16": 28,
        "baseline_safe16": 28,
        "better": 13,
        "worse": 15,
        "equal": 0
      },
      "union_vs_shared_k16": {
        "n": 28,
        "candidate_mean": 1.4897040811899518,
        "baseline_mean": 1.6438529192070876,
        "mean_difference": -0.15414883801713586,
        "median_difference": -0.0062885582447052,
        "candidate_safe4": 26,
        "baseline_safe4": 26,
        "candidate_safe8": 27,
        "baseline_safe8": 27,
        "candidate_safe16": 28,
        "baseline_safe16": 28,
        "better": 14,
        "worse": 0,
        "equal": 14
      },
      "union_vs_shared_k64": {
        "n": 28,
        "candidate_mean": 1.2070767387548196,
        "baseline_mean": 1.2861722795059904,
        "mean_difference": -0.0790955407511709,
        "median_difference": 0.0,
        "candidate_safe4": 26,
        "baseline_safe4": 26,
        "candidate_safe8": 27,
        "baseline_safe8": 27,
        "candidate_safe16": 28,
        "baseline_safe16": 28,
        "better": 13,
        "worse": 0,
        "equal": 15
      }
    },
    "4": {
      "branch_b_vs_shared_k16": {
        "n": 36,
        "candidate_mean": 1.580837970909973,
        "baseline_mean": 1.6947287898510695,
        "mean_difference": -0.11389081894109647,
        "median_difference": -0.0006746649742126465,
        "candidate_safe4": 34,
        "baseline_safe4": 34,
        "candidate_safe8": 34,
        "baseline_safe8": 34,
        "candidate_safe16": 35,
        "baseline_safe16": 34,
        "better": 22,
        "worse": 14,
        "equal": 0
      },
      "branch_b_vs_shared_k64": {
        "n": 36,
        "candidate_mean": 1.4609106362072959,
        "baseline_mean": 1.5715109759734736,
        "mean_difference": -0.11060033976617786,
        "median_difference": -0.0014188596978783607,
        "candidate_safe4": 34,
        "baseline_safe4": 34,
        "candidate_safe8": 34,
        "baseline_safe8": 34,
        "candidate_safe16": 35,
        "baseline_safe16": 34,
        "better": 25,
        "worse": 11,
        "equal": 0
      },
      "union_vs_shared_k16": {
        "n": 36,
        "candidate_mean": 1.5718027457801833,
        "baseline_mean": 1.6947287898510695,
        "mean_difference": -0.12292604407088624,
        "median_difference": -0.0006746649742126465,
        "candidate_safe4": 34,
        "baseline_safe4": 34,
        "candidate_safe8": 34,
        "baseline_safe8": 34,
        "candidate_safe16": 35,
        "baseline_safe16": 34,
        "better": 22,
        "worse": 0,
        "equal": 14
      },
      "union_vs_shared_k64": {
        "n": 36,
        "candidate_mean": 1.450682113154067,
        "baseline_mean": 1.5715109759734736,
        "mean_difference": -0.12082886281940672,
        "median_difference": -0.0014188596978783607,
        "candidate_safe4": 34,
        "baseline_safe4": 34,
        "candidate_safe8": 34,
        "baseline_safe8": 34,
        "candidate_safe16": 35,
        "baseline_safe16": 34,
        "better": 25,
        "worse": 0,
        "equal": 11
      }
    },
    "8": {
      "branch_b_vs_shared_k16": {
        "n": 44,
        "candidate_mean": 4.119757343655113,
        "baseline_mean": 2.7908775077032093,
        "mean_difference": 1.328879835951904,
        "median_difference": 0.0007413774728775024,
        "candidate_safe4": 38,
        "baseline_safe4": 39,
        "candidate_safe8": 39,
        "baseline_safe8": 40,
        "candidate_safe16": 42,
        "baseline_safe16": 43,
        "better": 20,
        "worse": 24,
        "equal": 0
      },
      "branch_b_vs_shared_k64": {
        "n": 44,
        "candidate_mean": 2.627223933898759,
        "baseline_mean": 2.7020364797525955,
        "mean_difference": -0.07481254585383629,
        "median_difference": 0.00032826513051986694,
        "candidate_safe4": 39,
        "baseline_safe4": 39,
        "candidate_safe8": 40,
        "baseline_safe8": 40,
        "candidate_safe16": 43,
        "baseline_safe16": 43,
        "better": 21,
        "worse": 23,
        "equal": 0
      },
      "union_vs_shared_k16": {
        "n": 44,
        "candidate_mean": 2.700709481660107,
        "baseline_mean": 2.7908775077032093,
        "mean_difference": -0.09016802604310215,
        "median_difference": 0.0,
        "candidate_safe4": 39,
        "baseline_safe4": 39,
        "candidate_safe8": 40,
        "baseline_safe8": 40,
        "candidate_safe16": 43,
        "baseline_safe16": 43,
        "better": 20,
        "worse": 0,
        "equal": 24
      },
      "union_vs_shared_k64": {
        "n": 44,
        "candidate_mean": 2.558927813767117,
        "baseline_mean": 2.7020364797525955,
        "mean_difference": -0.14310866598547858,
        "median_difference": 0.0,
        "candidate_safe4": 39,
        "baseline_safe4": 39,
        "candidate_safe8": 40,
        "baseline_safe8": 40,
        "candidate_safe16": 43,
        "baseline_safe16": 43,
        "better": 21,
        "worse": 0,
        "equal": 23
      }
    }
  },
  "per_sequence_pooled_visible": {
    "ani": {
      "branch_b_vs_shared_k16": {
        "n": 38,
        "candidate_mean": 3.7982428457440904,
        "baseline_mean": 3.854493968924017,
        "mean_difference": -0.056251123179926685,
        "median_difference": -0.001301392912864685,
        "candidate_safe4": 31,
        "baseline_safe4": 32,
        "candidate_safe8": 33,
        "baseline_safe8": 33,
        "candidate_safe16": 36,
        "baseline_safe16": 35,
        "better": 25,
        "worse": 13,
        "equal": 0
      },
      "branch_b_vs_shared_k64": {
        "n": 38,
        "candidate_mean": 3.5041852340616875,
        "baseline_mean": 3.592387050255447,
        "mean_difference": -0.08820181619375944,
        "median_difference": -0.0024075545370578766,
        "candidate_safe4": 32,
        "baseline_safe4": 32,
        "candidate_safe8": 33,
        "baseline_safe8": 33,
        "candidate_safe16": 36,
        "baseline_safe16": 35,
        "better": 24,
        "worse": 14,
        "equal": 0
      },
      "union_vs_shared_k16": {
        "n": 38,
        "candidate_mean": 3.6625230798350743,
        "baseline_mean": 3.854493968924017,
        "mean_difference": -0.1919708890889428,
        "median_difference": -0.001301392912864685,
        "candidate_safe4": 32,
        "baseline_safe4": 32,
        "candidate_safe8": 33,
        "baseline_safe8": 33,
        "candidate_safe16": 36,
        "baseline_safe16": 35,
        "better": 25,
        "worse": 0,
        "equal": 13
      },
      "union_vs_shared_k64": {
        "n": 38,
        "candidate_mean": 3.406492334888562,
        "baseline_mean": 3.592387050255447,
        "mean_difference": -0.18589471536688507,
        "median_difference": -0.0024075545370578766,
        "candidate_safe4": 32,
        "baseline_safe4": 32,
        "candidate_safe8": 33,
        "baseline_safe8": 33,
        "candidate_safe16": 36,
        "baseline_safe16": 35,
        "better": 24,
        "worse": 0,
        "equal": 14
      }
    },
    "animal3": {
      "branch_b_vs_shared_k16": {
        "n": 39,
        "candidate_mean": 2.9697204960557895,
        "baseline_mean": 1.3117707832119403,
        "mean_difference": 1.6579497128438492,
        "median_difference": 0.0027324557304382324,
        "candidate_safe4": 35,
        "baseline_safe4": 36,
        "candidate_safe8": 35,
        "baseline_safe8": 37,
        "candidate_safe16": 38,
        "baseline_safe16": 39,
        "better": 17,
        "worse": 22,
        "equal": 0
      },
      "branch_b_vs_shared_k64": {
        "n": 39,
        "candidate_mean": 1.3499510427698111,
        "baseline_mean": 1.172841860459019,
        "mean_difference": 0.17710918231079212,
        "median_difference": -0.0009361207485198975,
        "candidate_safe4": 36,
        "baseline_safe4": 36,
        "candidate_safe8": 36,
        "baseline_safe8": 37,
        "candidate_safe16": 39,
        "baseline_safe16": 39,
        "better": 20,
        "worse": 19,
        "equal": 0
      },
      "union_vs_shared_k16": {
        "n": 39,
        "candidate_mean": 1.2513477942213798,
        "baseline_mean": 1.3117707832119403,
        "mean_difference": -0.060422988990560554,
        "median_difference": 0.0,
        "candidate_safe4": 36,
        "baseline_safe4": 36,
        "candidate_safe8": 37,
        "baseline_safe8": 37,
        "candidate_safe16": 39,
        "baseline_safe16": 39,
        "better": 17,
        "worse": 0,
        "equal": 22
      },
      "union_vs_shared_k64": {
        "n": 39,
        "candidate_mean": 1.1292770899927769,
        "baseline_mean": 1.172841860459019,
        "mean_difference": -0.04356477046624208,
        "median_difference": -0.0009361207485198975,
        "candidate_safe4": 36,
        "baseline_safe4": 36,
        "candidate_safe8": 37,
        "baseline_safe8": 37,
        "candidate_safe16": 39,
        "baseline_safe16": 39,
        "better": 20,
        "worse": 0,
        "equal": 19
      }
    },
    "r4_new_f": {
      "branch_b_vs_shared_k16": {
        "n": 31,
        "candidate_mean": 0.9676294931841474,
        "baseline_mean": 1.0389321660322528,
        "mean_difference": -0.07130267284810543,
        "median_difference": 0.00014887750148773193,
        "candidate_safe4": 31,
        "baseline_safe4": 31,
        "candidate_safe8": 31,
        "baseline_safe8": 31,
        "candidate_safe16": 31,
        "baseline_safe16": 31,
        "better": 14,
        "worse": 17,
        "equal": 0
      },
      "branch_b_vs_shared_k64": {
        "n": 31,
        "candidate_mean": 0.8217122853703557,
        "baseline_mean": 0.9427510839586537,
        "mean_difference": -0.12103879858829802,
        "median_difference": 1.4841556549072266e-05,
        "candidate_safe4": 31,
        "baseline_safe4": 31,
        "candidate_safe8": 31,
        "baseline_safe8": 31,
        "candidate_safe16": 31,
        "baseline_safe16": 31,
        "better": 15,
        "worse": 16,
        "equal": 0
      },
      "union_vs_shared_k16": {
        "n": 31,
        "candidate_mean": 0.9403028808413975,
        "baseline_mean": 1.0389321660322528,
        "mean_difference": -0.0986292851908553,
        "median_difference": 0.0,
        "candidate_safe4": 31,
        "baseline_safe4": 31,
        "candidate_safe8": 31,
        "baseline_safe8": 31,
        "candidate_safe16": 31,
        "baseline_safe16": 31,
        "better": 14,
        "worse": 0,
        "equal": 17
      },
      "union_vs_shared_k64": {
        "n": 31,
        "candidate_mean": 0.8105488170629307,
        "baseline_mean": 0.9427510839586537,
        "mean_difference": -0.13220226689572295,
        "median_difference": 0.0,
        "candidate_safe4": 31,
        "baseline_safe4": 31,
        "candidate_safe8": 31,
        "baseline_safe8": 31,
        "candidate_safe16": 31,
        "baseline_safe16": 31,
        "better": 15,
        "worse": 0,
        "equal": 16
      }
    }
  },
  "novelty_counts_visible": {
    "branch_b_beats_shared_k16_by_gt1": 5,
    "branch_b_beats_shared_k64_by_gt1": 4,
    "branch_b_le4_shared_k16_gt4": 0,
    "branch_b_le4_shared_k64_gt4": 0,
    "branch_b_beats_shared_k64_by_gt1_per_sequence": {
      "ani": 2,
      "animal3": 0,
      "r4_new_f": 2
    },
    "branch_b_le4_shared_k64_gt4_per_sequence": {
      "ani": 0,
      "animal3": 0,
      "r4_new_f": 0
    }
  }
}
```

## Gates

```json
{
  "primary_horizon8": {
    "sequence_mean_improved": {
      "ani": false,
      "animal3": false,
      "r4_new_f": false
    },
    "sequence_safe16_not_decreased": {
      "ani": true,
      "animal3": false,
      "r4_new_f": true
    },
    "global_better_gt_worse": false,
    "clip_bootstrap_ci_upper_lt_zero": false,
    "clip_better_gt_worse": false,
    "pass_all": false
  },
  "pooled_horizons": {
    "mean_difference_lt_zero": false,
    "better_gt_worse": true,
    "clip_bootstrap_ci_upper_lt_zero": false,
    "pass_all": false
  },
  "early8_secondary": {
    "n_at_least_20": true,
    "mean_not_increased": false,
    "pass_all": false
  },
  "mechanistic_nondegeneracy": {
    "q_new_divergent_fraction_ge_0_10": true,
    "top16_changed_fraction_ge_0_10": true,
    "useful_novel_each_sequence": {
      "ani": true,
      "animal3": true,
      "r4_new_f": false
    },
    "useful_novel_global_ge_10": false,
    "pass_all": false
  },
  "pass_all": false,
  "decision": "INDEPENDENT_STATE_FAIL: the frozen independent-state branch does not satisfy the capacity-matched statistical and mechanistic gates. Close the temporal multi-state route; do not train or read DAVIS."
}
```

## Decision

INDEPENDENT_STATE_FAIL: the frozen independent-state branch does not satisfy the capacity-matched statistical and mechanistic gates. Close the temporal multi-state route; do not train or read DAVIS.

## Integrity

```json
{
  "input_audit": {
    "pass": true,
    "head": "ee17e03176c9e0b256cf555feab82b0f9be1f041",
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
        "path": "/gemini/code/FSPT_v9a52_clean/baselines/track_on/config/test.yaml",
        "size_bytes": 249,
        "sha256": "34c08c71be511b2cda6aaafb6e2e3af7e21c6fd5197de98a54fa4b9ac7a78c8e"
      },
      {
        "path": "/gemini/code/FSPT_v9a52_clean/docs/v9a51c_pointodyssey_frame_manifest_2026-07-10.json",
        "size_bytes": 152463,
        "sha256": "b1a92a0c5a627fb9a554cd45201dc172bd2b7edaaf13ce78a40eecaddf7effba"
      },
      {
        "path": "/gemini/code/FSPT_v9a52_clean/docs/v9a52_independent_state_event_manifest_2026-07-11.json",
        "size_bytes": 19416,
        "sha256": "e5f5fa34c840e5159726ada1c7752e8bfa273b48ad6cd215e7519ed71721fb4b"
      },
      {
        "path": "/gemini/code/FSPT_v9a52_clean/docs/v9a52_independent_state_branching_design_2026-07-11.md",
        "size_bytes": 13344,
        "sha256": "cf51938d19efba6c0a3333105927074cdcf0dfdf4414bda893beda1550a03fc1"
      },
      {
        "path": "/gemini/code/FSPT_v9a52_clean/scripts/v9a52_build_event_manifest.py",
        "size_bytes": 6287,
        "sha256": "1e60271f3f76dfba09e9f9df08f5780a23894e5afdc1537fe7766f94a0966946"
      },
      {
        "path": "/gemini/code/FSPT_v9a52_clean/scripts/v9a52_independent_state_branching_smoke.py",
        "size_bytes": 55700,
        "sha256": "1e0a0324bf0170bf7f9db1d91de51075bcd690410f4014b534c834628da3ec37"
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
    "verified_frame_count": 864,
    "verified_frame_total_bytes": 136082946,
    "full_frame_hash": true,
    "imported_trackon_code_paths": {
      "model.trackon_predictor": "/gemini/code/FSPT_v9a52_clean/baselines/track_on/model/trackon_predictor.py",
      "model.trackon": "/gemini/code/FSPT_v9a52_clean/baselines/track_on/model/trackon.py",
      "model.reranking": "/gemini/code/FSPT_v9a52_clean/baselines/track_on/model/reranking.py",
      "model.prediction_head": "/gemini/code/FSPT_v9a52_clean/baselines/track_on/model/prediction_head.py",
      "model.modules": "/gemini/code/FSPT_v9a52_clean/baselines/track_on/model/modules.py",
      "utils.coord_utils": "/gemini/code/FSPT_v9a52_clean/baselines/track_on/utils/coord_utils.py",
      "utils.train_utils": "/gemini/code/FSPT_v9a52_clean/baselines/track_on/utils/train_utils.py"
    },
    "formal_branch_forbidden_hits": {
      "create_active_event": [],
      "forward_active_event": []
    },
    "helper_script": {
      "path": "/gemini/code/FSPT_v9a52_clean/scripts/v9a52_independent_state_branching_smoke.py",
      "sha256": "1e0a0324bf0170bf7f9db1d91de51075bcd690410f4014b534c834628da3ec37"
    },
    "script": {
      "path": "/gemini/code/FSPT_v9a52_clean/scripts/v9a52_independent_state_branching_audit.py",
      "sha256": "d400e8a8b0e713df00620fa113bda1e1bddcf2612821968dd3682811913c7104"
    }
  },
  "formal_run": true,
  "selected_clips": [
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
  "selected_events": 72,
  "expected_events_per_clip": {
    "ani:0": 8,
    "ani:256": 8,
    "ani:512": 8,
    "animal3:0": 8,
    "animal3:256": 8,
    "animal3:512": 8,
    "r4_new_f:0": 8,
    "r4_new_f:256": 8,
    "r4_new_f:512": 8
  },
  "event_horizon_rows": 216,
  "expected_event_horizon_rows": 216,
  "unique_event_horizon_keys": 216,
  "split_rows": 72,
  "active_event_peak": 6,
  "official_track_parity": {
    "p": 0.0,
    "v_logit": 0.0,
    "q_new": 0.0
  },
  "branch_a_output_parity": {
    "p": 0.0,
    "v_logit": 0.0,
    "u_logit": 0.0,
    "q_new": 0.0,
    "q_pre": 0.0,
    "c1": 0.0,
    "c2": 0.0
  },
  "branch_a_state_parity": {
    "q_init": 0.0,
    "point_memory": 0.0,
    "temporal_mask_mismatch": 0
  },
  "online_risk_mismatch": 0,
  "gt_visibility_mismatch": 0,
  "candidate_replay": {
    "rows": 288,
    "score_max_abs": 2.0503997802734375e-05,
    "raw_error_max_abs": 0.0,
    "refined_error_max_abs": 4.553794860839844e-05,
    "official_error_max_abs": 0.0,
    "shared_score_top1_error_max_abs": 1.52587890625e-05,
    "risk_mismatch": 0
  },
  "expected_candidate_replay_rows": 288,
  "storage_disjoint_failures": 0,
  "mutation_isolation_failures": 0,
  "cross_event_storage_alias_violations": 0,
  "event_key_mismatch": 0,
  "first_risk_mismatch": 0,
  "all_outputs_finite": true
}
```
