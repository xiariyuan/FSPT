# V9-A6.0 Bounded Correlation Rank-Shift Audit Result

Date: 2026-07-11

This is a no-training GT-directed score-magnitude necessary-condition audit. It is not an adapter evaluation.

## Primary budget

```json
{
  "opportunity_rows": 760,
  "promoted_rows": 67,
  "opportunity_conversion": 0.0881578947368421,
  "implied_global_recall4_gain": 0.013735137351373513,
  "implied_gt_hard_recall4_gain": 0.027697395618023975,
  "per_sequence": {
    "ani": {
      "opportunities": 361,
      "promoted": 31,
      "conversion": 0.08587257617728532
    },
    "animal3": {
      "opportunities": 98,
      "promoted": 22,
      "conversion": 0.22448979591836735
    },
    "r4_new_f": {
      "opportunities": 301,
      "promoted": 14,
      "conversion": 0.046511627906976744
    }
  },
  "per_clip": {
    "ani:0": {
      "opportunities": 236,
      "promoted": 21,
      "conversion": 0.08898305084745763
    },
    "ani:256": {
      "opportunities": 93,
      "promoted": 9,
      "conversion": 0.0967741935483871
    },
    "ani:512": {
      "opportunities": 32,
      "promoted": 1,
      "conversion": 0.03125
    },
    "animal3:0": {
      "opportunities": 21,
      "promoted": 6,
      "conversion": 0.2857142857142857
    },
    "animal3:256": {
      "opportunities": 68,
      "promoted": 15,
      "conversion": 0.22058823529411764
    },
    "animal3:512": {
      "opportunities": 9,
      "promoted": 1,
      "conversion": 0.1111111111111111
    },
    "r4_new_f:0": {
      "opportunities": 161,
      "promoted": 7,
      "conversion": 0.043478260869565216
    },
    "r4_new_f:256": {
      "opportunities": 116,
      "promoted": 3,
      "conversion": 0.02586206896551724
    },
    "r4_new_f:512": {
      "opportunities": 24,
      "promoted": 4,
      "conversion": 0.16666666666666666
    }
  },
  "native_risk": {
    "opportunities": 325,
    "promoted": 36,
    "conversion_within_risk_opportunities": 0.11076923076923077,
    "implied_global_recall4_gain": 0.007380073800738007,
    "opportunity_coverage_ceiling": 0.4276315789473684
  },
  "reentry": {
    "first": {
      "opportunities": 28,
      "promoted": 3,
      "conversion": 0.10714285714285714
    },
    "early8": {
      "opportunities": 146,
      "promoted": 15,
      "conversion": 0.10273972602739725
    }
  },
  "target_rank_bins": {
    "17-24": {
      "opportunities": 213,
      "promoted": 67,
      "conversion": 0.3145539906103286
    },
    "25-32": {
      "opportunities": 150,
      "promoted": 0,
      "conversion": 0.0
    },
    "33-48": {
      "opportunities": 252,
      "promoted": 0,
      "conversion": 0.0
    },
    "49-64": {
      "opportunities": 145,
      "promoted": 0,
      "conversion": 0.0
    }
  },
  "span_multiplier": 2.0,
  "span_budget": 0.00628662109375,
  "positive_strict_promoted_rows": 67,
  "signed_pairwise_per_cell_epsilon": 0.003143310546875
}
```

## Magnitude distributions

```json
{
  "span": {
    "n": 760,
    "mean": 0.040254483724895276,
    "std": 0.029482083118326997,
    "min": 7.724761962890625e-05,
    "p10": 0.006772708892822266,
    "median": 0.03430986404418945,
    "p75": 0.05616402626037598,
    "p90": 0.07755727767944337,
    "p95": 0.09478883743286132,
    "p99": 0.12660140037536616,
    "max": 0.19609451293945312
  },
  "positive_epsilon": {
    "n": 760,
    "mean": 0.04025448372489705,
    "std": 0.029482083118326997,
    "min": 7.724761963068261e-05,
    "p10": 0.006772708892824043,
    "median": 0.03430986404419123,
    "p75": 0.05616402626037775,
    "p90": 0.07755727767944515,
    "p95": 0.0947888374328631,
    "p99": 0.12660140037536793,
    "max": 0.1960945129394549
  },
  "signed_epsilon": {
    "n": 760,
    "mean": 0.020127241862447638,
    "std": 0.014741041559163498,
    "min": 3.8623809814453125e-05,
    "p10": 0.003386354446411133,
    "median": 0.017154932022094727,
    "p75": 0.02808201313018799,
    "p90": 0.038778638839721685,
    "p95": 0.04739441871643066,
    "p99": 0.06330070018768308,
    "max": 0.09804725646972656
  },
  "span_over_fused_std": {
    "n": 760,
    "mean": 0.13897879285041378,
    "std": 0.09454398204837376,
    "min": 0.0004284248406077895,
    "p10": 0.02585262534762833,
    "median": 0.12653189253198444,
    "p75": 0.19583573164153184,
    "p90": 0.2744305452054363,
    "p95": 0.30925168093046246,
    "p99": 0.39778093960586497,
    "max": 0.482837315986137
  },
  "positive_epsilon_over_fused_std": {
    "n": 760,
    "mean": 0.13897879285042058,
    "std": 0.09454398204837437,
    "min": 0.0004284248406129713,
    "p10": 0.025852625347632108,
    "median": 0.12653189253198976,
    "p75": 0.1958357316415378,
    "p90": 0.27443054520544136,
    "p95": 0.3092516809304698,
    "p99": 0.3977809396058733,
    "max": 0.48283731598614604
  },
  "signed_epsilon_over_fused_std": {
    "n": 760,
    "mean": 0.06948939642520689,
    "std": 0.04727199102418688,
    "min": 0.00021421242030389474,
    "p10": 0.012926312673814165,
    "median": 0.06326594626599222,
    "p75": 0.09791786582076592,
    "p90": 0.13721527260271815,
    "p95": 0.15462584046523123,
    "p99": 0.19889046980293248,
    "max": 0.2414186579930685
  },
  "span_over_g_ref": {
    "n": 760,
    "mean": 12.806397307741442,
    "std": 9.379309705061544,
    "min": 0.0245752427184466,
    "p10": 2.154641990291262,
    "median": 10.915200242718447,
    "p75": 17.867794296116507,
    "p90": 24.67375606796117,
    "p95": 30.155734223300968,
    "p99": 40.27645327669901,
    "max": 62.38470873786408
  },
  "per_sequence": {
    "ani": {
      "span": {
        "n": 361,
        "mean": 0.03597813762125877,
        "std": 0.025028376998661537,
        "min": 0.0001468658447265625,
        "p10": 0.00664520263671875,
        "median": 0.031708717346191406,
        "p75": 0.047507286071777344,
        "p90": 0.07139015197753906,
        "p95": 0.0845041275024414,
        "p99": 0.11293544769287099,
        "max": 0.13344478607177734
      },
      "span_over_fused_std": {
        "n": 361,
        "mean": 0.15476352192553247,
        "std": 0.10137580909400391,
        "min": 0.0004284248406077895,
        "p10": 0.032591118807854695,
        "median": 0.14260169732956696,
        "p75": 0.22508967057360268,
        "p90": 0.2981772895053176,
        "p95": 0.3230692157002647,
        "p99": 0.42381454817479836,
        "max": 0.482837315986137
      },
      "target_rank": {
        "n": 361,
        "mean": 37.29362880886426,
        "std": 13.52015785962567,
        "min": 17.0,
        "p10": 19.0,
        "median": 37.0,
        "p75": 48.0,
        "p90": 56.0,
        "p95": 60.0,
        "p99": 64.0,
        "max": 64.0
      }
    },
    "animal3": {
      "span": {
        "n": 98,
        "mean": 0.028701840614785954,
        "std": 0.02494665891461611,
        "min": 7.724761962890625e-05,
        "p10": 0.0034157752990722663,
        "median": 0.02461528778076172,
        "p75": 0.03981280326843262,
        "p90": 0.06838703155517578,
        "p95": 0.07991943359375,
        "p99": 0.10884581565856934,
        "max": 0.1096200942993164
      },
      "span_over_fused_std": {
        "n": 98,
        "mean": 0.10300824668513896,
        "std": 0.08413668973700275,
        "min": 0.0006110058849390114,
        "p10": 0.016724956611122367,
        "median": 0.081527751225602,
        "p75": 0.15213634844249202,
        "p90": 0.20940875266440112,
        "p95": 0.25416921996297337,
        "p99": 0.34585190875435085,
        "max": 0.46989211169914985
      },
      "target_rank": {
        "n": 98,
        "mean": 30.387755102040817,
        "std": 12.432374338601901,
        "min": 17.0,
        "p10": 18.0,
        "median": 26.0,
        "p75": 37.0,
        "p90": 51.3,
        "p95": 56.449999999999974,
        "p99": 60.03,
        "max": 61.0
      }
    },
    "r4_new_f": {
      "span": {
        "n": 301,
        "mean": 0.04914458328703313,
        "std": 0.03311061536306135,
        "min": 0.00040912628173828125,
        "p10": 0.011824607849121094,
        "median": 0.04629707336425781,
        "p75": 0.0674123764038086,
        "p90": 0.08844661712646484,
        "p95": 0.10903167724609375,
        "p99": 0.15487003326416016,
        "max": 0.19609451293945312
      },
      "span_over_fused_std": {
        "n": 301,
        "mean": 0.1317589467642978,
        "std": 0.08465517053259522,
        "min": 0.0009149066809180977,
        "p10": 0.027886827243682473,
        "median": 0.12348909229420382,
        "p75": 0.1862434409687466,
        "p90": 0.24060081447034382,
        "p95": 0.2863152787701453,
        "p99": 0.3565319564280968,
        "max": 0.42750788963259373
      },
      "target_rank": {
        "n": 301,
        "mean": 34.17275747508306,
        "std": 11.974643283417754,
        "min": 17.0,
        "p10": 20.0,
        "median": 33.0,
        "p75": 42.0,
        "p90": 52.0,
        "p95": 57.0,
        "p99": 62.0,
        "max": 64.0
      }
    }
  },
  "target_rank": {
    "n": 760,
    "mean": 35.16710526315789,
    "std": 13.001304470563255,
    "min": 17.0,
    "p10": 19.0,
    "median": 33.0,
    "p75": 45.0,
    "p90": 54.10000000000002,
    "p95": 59.0,
    "p99": 63.0,
    "max": 64.0
  },
  "target_error": {
    "n": 760,
    "mean": 2.992998675511856,
    "std": 0.7646357299650713,
    "min": 0.2327088862657547,
    "p10": 1.9378141522407533,
    "median": 3.1416386365890503,
    "p75": 3.6094087958335876,
    "p90": 3.855616497993469,
    "p95": 3.9331898212432863,
    "p99": 3.989363090991974,
    "max": 3.999765634536743
  },
  "rank16_error": {
    "n": 760,
    "mean": 19.499613162090903,
    "std": 17.63580048034598,
    "min": 4.043671131134033,
    "p10": 7.371456146240234,
    "median": 15.061184883117676,
    "p75": 22.157865524291992,
    "p90": 34.69601478576661,
    "p95": 45.48350334167479,
    "p99": 65.90935768127441,
    "max": 180.60693359375
  }
}
```

## Gates

```json
{
  "formal_run": true,
  "checks": {
    "global_conversion_ge_0_50": false,
    "sequence_conversion_ge_0_40": {
      "ani": false,
      "animal3": false,
      "r4_new_f": false
    },
    "clips_ge_0_30_at_least_7": false,
    "bootstrap_conversion_ci_lower_gt_0_35": false,
    "implied_global_gain_ge_0_075": false,
    "implied_hard_gain_ge_0_12": false,
    "sequence_median_span_over_std_le_0_02": {
      "ani": false,
      "animal3": false,
      "r4_new_f": false
    },
    "global_p90_span_over_std_le_0_10": false
  },
  "clips_conversion_ge_0_30": 0,
  "pass_all": false,
  "decision": "RANK_SHIFT_MAGNITUDE_FAIL: even the optimistic target-directed oracle does not satisfy the preregistered small-span, sequence/clip-consistency and implied-recall gates. Close the bounded correlation residual route; do not train or read DAVIS."
}
```

## Decision

RANK_SHIFT_MAGNITUDE_FAIL: even the optimistic target-directed oracle does not satisfy the preregistered small-span, sequence/clip-consistency and implied-recall gates. Close the bounded correlation residual route; do not train or read DAVIS.

## Integrity

```json
{
  "environment": {
    "head": "7a8dc06638630ec83c22f9ea76e04870b28b545b",
    "branch": "v9a60-correlation-rank-shift-20260711",
    "tracked_status": "",
    "checked_hashes": {
      "/gemini/code/FSPT_v9a60_clean/scripts/v9a60_export_fused_top129.py": "684e7a6243d7fccddd23bef79355356dd78ef2b40d2257d1483f790ac680ed2c",
      "/gemini/code/FSPT_v9a60_clean/outputs/paper_discovery_2026-07-05/v9a60_rank_shift/v9a60_fused_top129_export.json": "72fb654381eef236c73b54d8d52d5bfbe4a7b3e4cfaf7ebcf850f1cdf8472cbb",
      "/gemini/code/FSPT_v9a60_clean/outputs/paper_discovery_2026-07-05/v9a60_rank_shift/v9a60_fused_top129_export.npz": "0849bb9ab700e8f5224bf751f897aea7a38e0efd6363d6d83297e04ecbfac3ac",
      "/gemini/code/FSPT_v9a60_clean/docs/v9a60_bounded_correlation_rank_shift_design_2026-07-11.md": "98893e60568e9788a72af505fd7048bb3cc26f4c2eea0c0923c684369ae82869"
    },
    "script": {
      "path": "/gemini/code/FSPT_v9a60_clean/scripts/v9a60_bounded_rank_shift_audit.py",
      "sha256": "61dd1ea369353a698a683212dcb2c45d6ba0eb9b1d7a603e6fdbc270eec538c6"
    }
  },
  "export_decision": "EXPORT_PASS: the formal 4,878-row top129 export reproduces V9-A5C.0 and may be frozen for the no-training rank-shift audit.",
  "target_exists_every_row": true,
  "target_rank_in_17_64": true,
  "target_error_le4": true,
  "rank16_error_gt4": true,
  "target_is_first_good_candidate": true,
  "span_nonnegative": true,
  "component_score_reconstruction_max_abs": 9.838669070560968e-07,
  "signed_formula_max_abs": 0.0,
  "positive_epsilon_ge_span": true,
  "all_outputs_finite": true
}
```
