# V9-A5.1c Same-Capacity Incremental Temporal-Value Review

Date: 2026-07-10

This is a post-formal conservative comparator review. It does not alter the original predeclared V9-A5.1c gates.

## Primary B4 fair comparison

```json
{
  "configuration": {
    "budget_on_risk": 64,
    "width": 4
  },
  "temporal_top1_vs_frame_local_score_top1": {
    "gate": {
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
      "early8_mean_improved": true,
      "all_clip_mean_differences_nonpositive": false,
      "pass_all": false,
      "summaries": {
        "all_visible": {
          "n": 19930,
          "candidate_mean": 9.15751101826647,
          "baseline_mean": 9.10234632125429,
          "mean_difference": 0.0551646970121804,
          "median_difference": 0.0,
          "candidate_safe16": 17514,
          "baseline_safe16": 17522,
          "better": 1482,
          "worse": 1489,
          "equal": 16959
        },
        "reentry_first": {
          "n": 855,
          "candidate_mean": 18.066455659662424,
          "baseline_mean": 17.750960406332077,
          "mean_difference": 0.3154952533303471,
          "median_difference": 0.0,
          "candidate_safe16": 639,
          "baseline_safe16": 643,
          "better": 173,
          "worse": 191,
          "equal": 491
        },
        "reentry_early8": {
          "n": 3806,
          "candidate_mean": 15.835720454270723,
          "baseline_mean": 15.859668959157773,
          "mean_difference": -0.023948504887051445,
          "median_difference": 0.0,
          "candidate_safe16": 2941,
          "baseline_safe16": 2949,
          "better": 676,
          "worse": 687,
          "equal": 2443
        },
        "per_sequence": {
          "ani": {
            "n": 6679,
            "candidate_mean": 9.92150743801087,
            "baseline_mean": 9.952140966619387,
            "mean_difference": -0.030633528608516994,
            "median_difference": 0.0,
            "candidate_safe16": 5803,
            "baseline_safe16": 5811,
            "better": 519,
            "worse": 531,
            "equal": 5629
          },
          "animal3": {
            "n": 6616,
            "candidate_mean": 6.675586637394433,
            "baseline_mean": 6.776443480676155,
            "mean_difference": -0.10085684328172198,
            "median_difference": 0.0,
            "candidate_safe16": 6045,
            "baseline_safe16": 6038,
            "better": 474,
            "worse": 470,
            "equal": 5672
          },
          "r4_new_f": {
            "n": 6635,
            "candidate_mean": 10.863265293530457,
            "baseline_mean": 10.566158643314797,
            "mean_difference": 0.29710665021566135,
            "median_difference": 0.0,
            "candidate_safe16": 5666,
            "baseline_safe16": 5673,
            "better": 489,
            "worse": 488,
            "equal": 5658
          }
        }
      }
    },
    "bootstrap": {
      "per_clip_difference": {
        "ani:0": 0.03748807419373949,
        "ani:256": -0.18360777840778322,
        "ani:512": 0.04468283237350222,
        "animal3:0": -0.16171927710897044,
        "animal3:256": -0.16035038954003902,
        "animal3:512": -0.015445737298271556,
        "r4_new_f:0": 0.9681848923734059,
        "r4_new_f:256": -0.05310728460736483,
        "r4_new_f:512": -0.0019256188477909332
      },
      "mean_difference_95_ci": [
        -0.10540510300234451,
        0.2998721649794963
      ],
      "probability_mean_lt_zero": 0.35316,
      "all_clip_nonpositive": false
    }
  },
  "temporal_oracle_vs_frame_local_same_capacity_oracle": {
    "gate": {
      "sequence_mean_improved": {
        "ani": false,
        "animal3": false,
        "r4_new_f": false
      },
      "sequence_safe16_not_decreased": {
        "ani": false,
        "animal3": false,
        "r4_new_f": false
      },
      "global_better_gt_worse": false,
      "clip_bootstrap_ci_upper_lt_zero": false,
      "first_reentry_mean_improved": false,
      "early8_mean_improved": false,
      "all_clip_mean_differences_nonpositive": false,
      "pass_all": false,
      "summaries": {
        "all_visible": {
          "n": 19930,
          "candidate_mean": 8.787938573795506,
          "baseline_mean": 8.536341409495252,
          "mean_difference": 0.2515971643002531,
          "median_difference": 0.0,
          "candidate_safe16": 17627,
          "baseline_safe16": 17700,
          "better": 734,
          "worse": 1889,
          "equal": 17307
        },
        "reentry_first": {
          "n": 855,
          "candidate_mean": 17.07942041307034,
          "baseline_mean": 16.48497923582313,
          "mean_difference": 0.5944411772472119,
          "median_difference": 0.0,
          "candidate_safe16": 657,
          "baseline_safe16": 666,
          "better": 85,
          "worse": 225,
          "equal": 545
        },
        "reentry_early8": {
          "n": 3806,
          "candidate_mean": 15.007533432649085,
          "baseline_mean": 14.59210991836777,
          "mean_difference": 0.41542351428131585,
          "median_difference": 0.0,
          "candidate_safe16": 2993,
          "baseline_safe16": 3031,
          "better": 305,
          "worse": 848,
          "equal": 2653
        },
        "per_sequence": {
          "ani": {
            "n": 6679,
            "candidate_mean": 9.588723144447005,
            "baseline_mean": 9.3394830632971,
            "mean_difference": 0.24924008114990456,
            "median_difference": 0.0,
            "candidate_safe16": 5846,
            "baseline_safe16": 5887,
            "better": 277,
            "worse": 676,
            "equal": 5726
          },
          "animal3": {
            "n": 6616,
            "candidate_mean": 6.350330039640346,
            "baseline_mean": 6.200641165809546,
            "mean_difference": 0.14968887383079982,
            "median_difference": 0.0,
            "candidate_safe16": 6082,
            "baseline_safe16": 6093,
            "better": 226,
            "worse": 600,
            "equal": 5790
          },
          "r4_new_f": {
            "n": 6635,
            "candidate_mean": 10.412471793778803,
            "baseline_mean": 10.05688544965834,
            "mean_difference": 0.35558634412046114,
            "median_difference": 0.0,
            "candidate_safe16": 5699,
            "baseline_safe16": 5720,
            "better": 231,
            "worse": 613,
            "equal": 5791
          }
        }
      }
    },
    "bootstrap": {
      "per_clip_difference": {
        "ani:0": 0.3647465171265941,
        "ani:256": 0.35100405242435273,
        "ani:512": 0.059606780769842625,
        "animal3:0": 0.13249794287071656,
        "animal3:256": 0.3563349667222794,
        "animal3:512": 0.018174604221606386,
        "r4_new_f:0": 0.8065035400976991,
        "r4_new_f:256": 0.23963907421895486,
        "r4_new_f:512": 0.05110935843229393
      },
      "mean_difference_95_ci": [
        0.12796971526112141,
        0.43066073942616556
      ],
      "probability_mean_lt_zero": 0.0,
      "all_clip_nonpositive": false
    }
  }
}
```

## All policy comparisons

```json
{
  "native_dynamic_B1": {
    "top1_mean_difference": 0.009407728706985129,
    "top1_ci": [
      -0.20844475922989922,
      0.24553731772157283
    ],
    "top1_incremental_pass": false,
    "oracle_mean_difference": -0.036825401277355646,
    "oracle_ci": [
      -0.2110410567128802,
      0.11889261978724901
    ],
    "oracle_incremental_pass": false
  },
  "native_dynamic_B4": {
    "top1_mean_difference": 0.0551646970121804,
    "top1_ci": [
      -0.10540510300234451,
      0.2998721649794963
    ],
    "top1_incremental_pass": false,
    "oracle_mean_difference": 0.2515971643002531,
    "oracle_ci": [
      0.12796971526112141,
      0.43066073942616556
    ],
    "oracle_incremental_pass": false
  },
  "native_dynamic_B8": {
    "top1_mean_difference": 0.02903092443939285,
    "top1_ci": [
      -0.10062004349208713,
      0.20728204297907193
    ],
    "top1_incremental_pass": false,
    "oracle_mean_difference": 0.3975757013154027,
    "oracle_ci": [
      0.19247316360937744,
      0.6653811462255181
    ],
    "oracle_incremental_pass": false
  },
  "fixed16_B4": {
    "top1_mean_difference": -0.08098665700717915,
    "top1_ci": [
      -0.3393747189633352,
      0.10900701792141061
    ],
    "top1_incremental_pass": false,
    "oracle_mean_difference": 0.16687076579427979,
    "oracle_ci": [
      0.06350966043208264,
      0.30177642111042424
    ],
    "oracle_incremental_pass": false
  },
  "fixed64_B4": {
    "top1_mean_difference": -0.025195112551632114,
    "top1_ci": [
      -0.3268444615900178,
      0.2956781671704796
    ],
    "top1_incremental_pass": false,
    "oracle_mean_difference": 0.16742826897101415,
    "oracle_ci": [
      -0.031234850564375694,
      0.386703077641484
    ],
    "oracle_incremental_pass": false
  }
}
```

## Interpretation

HISTORY_INCREMENTAL_VALUE_FAIL: the primary history beam is useful relative to official final, but it is weaker than the same-frame, same-capacity score-top4 candidate oracle. The original V9-A5.1c system-level reachability pass is valid, but it is not evidence that temporal history adds incremental candidate reachability. Do not proceed directly to a learned temporal beam readout; first redesign pruning/state so it beats the frame-local capacity-matched baseline.

## Integrity

```json
{
  "checked_hashes": {
    "/gemini/code/FSPT_v9a51c_clean/outputs/paper_discovery_2026-07-05/v9a51b_candidate_refinement/v9a51b_candidate_conditioned_refinement_rows.npz": "eb9736cbe7fc1af08e5bfe23248b333e18ae01c826fd26fc65aa40945746d266",
    "/gemini/code/FSPT_v9a51c_clean/scripts/v9a51c_history_preserving_beam_audit.py": "ee96041ce451c0639c94f7558771e6958907dc5030ea1b429228ec97ce3f4da1",
    "/gemini/code/FSPT_v9a51c_clean/outputs/paper_discovery_2026-07-05/v9a51c_history_beam/v9a51c_history_preserving_beam.json": "ae6bd3eabd36aa8d64014298f48bcc24a31b0611854c8827455a96aa69a6d00b",
    "/gemini/code/FSPT_v9a51c_clean/outputs/paper_discovery_2026-07-05/v9a51c_history_beam/v9a51c_history_preserving_beam_rows.npz": "05b450fb2031ce1d762ed7436bcef34a47dfaf37c10d1c33bd8fd74c1d0381a0"
  },
  "row_keys_exact": true,
  "unique_rows": 27648,
  "v9a51c_original_replay_zero": true,
  "v9a51c_official_parity_zero": true,
  "script_sha256": "7dca4ed2382d667c58875baf695a9ee4ff0008f2dd519f3bb75f6cbe0c7af505"
}
```
