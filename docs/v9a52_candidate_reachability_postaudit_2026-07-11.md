# V9-A5.2 Candidate-Reachability Post-Audit

Date: 2026-07-11

This is a non-gating post-formal diagnostic. It does not alter the predeclared V9-A5.2 decision.

## Branch B K64 versus shared-state K64

```json
{
  "pooled_visible": {
    "n": 108,
    "candidate_mean": 1.9562982445263684,
    "baseline_mean": 1.9581187413919166,
    "mean_difference": -0.0018204968655481935,
    "median_difference": -0.0008327718824148178,
    "better": 59,
    "worse": 49,
    "equal": 0,
    "candidate_safe4": 99,
    "baseline_safe4": 99,
    "candidate_safe8": 100,
    "baseline_safe8": 101,
    "candidate_safe16": 106,
    "baseline_safe16": 105
  },
  "horizon8_visible": {
    "n": 44,
    "candidate_mean": 2.627223933898759,
    "baseline_mean": 2.7020364797525955,
    "mean_difference": -0.07481254585383629,
    "median_difference": 0.00032826513051986694,
    "better": 21,
    "worse": 23,
    "equal": 0,
    "candidate_safe4": 39,
    "baseline_safe4": 39,
    "candidate_safe8": 40,
    "baseline_safe8": 40,
    "candidate_safe16": 43,
    "baseline_safe16": 43
  },
  "per_horizon_visible": {
    "1": {
      "n": 28,
      "candidate_mean": 1.5389133719228474,
      "baseline_mean": 1.2861722795059904,
      "mean_difference": 0.252741092416857,
      "median_difference": 0.005352984182536602,
      "better": 13,
      "worse": 15,
      "equal": 0,
      "candidate_safe4": 26,
      "baseline_safe4": 26,
      "candidate_safe8": 26,
      "baseline_safe8": 27,
      "candidate_safe16": 28,
      "baseline_safe16": 28
    },
    "4": {
      "n": 36,
      "candidate_mean": 1.4609106362072959,
      "baseline_mean": 1.5715109759734736,
      "mean_difference": -0.11060033976617786,
      "median_difference": -0.0014188596978783607,
      "better": 25,
      "worse": 11,
      "equal": 0,
      "candidate_safe4": 34,
      "baseline_safe4": 34,
      "candidate_safe8": 34,
      "baseline_safe8": 34,
      "candidate_safe16": 35,
      "baseline_safe16": 34
    },
    "8": {
      "n": 44,
      "candidate_mean": 2.627223933898759,
      "baseline_mean": 2.7020364797525955,
      "mean_difference": -0.07481254585383629,
      "median_difference": 0.00032826513051986694,
      "better": 21,
      "worse": 23,
      "equal": 0,
      "candidate_safe4": 39,
      "baseline_safe4": 39,
      "candidate_safe8": 40,
      "baseline_safe8": 40,
      "candidate_safe16": 43,
      "baseline_safe16": 43
    }
  },
  "per_sequence_pooled_visible": {
    "ani": {
      "n": 38,
      "candidate_mean": 3.5041852340616875,
      "baseline_mean": 3.592387050255447,
      "mean_difference": -0.08820181619375944,
      "median_difference": -0.0024075545370578766,
      "better": 24,
      "worse": 14,
      "equal": 0,
      "candidate_safe4": 32,
      "baseline_safe4": 32,
      "candidate_safe8": 33,
      "baseline_safe8": 33,
      "candidate_safe16": 36,
      "baseline_safe16": 35
    },
    "animal3": {
      "n": 39,
      "candidate_mean": 1.3499510427698111,
      "baseline_mean": 1.172841860459019,
      "mean_difference": 0.17710918231079212,
      "median_difference": -0.0009361207485198975,
      "better": 20,
      "worse": 19,
      "equal": 0,
      "candidate_safe4": 36,
      "baseline_safe4": 36,
      "candidate_safe8": 36,
      "baseline_safe8": 37,
      "candidate_safe16": 39,
      "baseline_safe16": 39
    },
    "r4_new_f": {
      "n": 31,
      "candidate_mean": 0.8217122853703557,
      "baseline_mean": 0.9427510839586537,
      "mean_difference": -0.12103879858829802,
      "median_difference": 1.4841556549072266e-05,
      "better": 15,
      "worse": 16,
      "equal": 0,
      "candidate_safe4": 31,
      "baseline_safe4": 31,
      "candidate_safe8": 31,
      "baseline_safe8": 31,
      "candidate_safe16": 31,
      "baseline_safe16": 31
    }
  },
  "pooled_clip_bootstrap": {
    "n_clips": 9,
    "bootstrap_resamples": 100000,
    "bootstrap_seed": 20260717,
    "per_clip_difference": {
      "ani:0": 0.22870218026666686,
      "ani:256": -0.46028271400635795,
      "ani:512": 0.010832785205407576,
      "animal3:0": -0.012523140344354842,
      "animal3:256": 0.7116075415502895,
      "animal3:512": -0.042511399149110445,
      "r4_new_f:0": -0.10357595191282384,
      "r4_new_f:256": 0.00010737351008823939,
      "r4_new_f:512": -0.2845947411842644
    },
    "clip_better": 5,
    "clip_worse": 4,
    "clip_equal": 0,
    "mean_difference_95_ci": [
      -0.18193613811911535,
      0.22190300619348338
    ],
    "probability_mean_lt_zero": 0.49951
  },
  "horizon8_clip_bootstrap": {
    "n_clips": 9,
    "bootstrap_resamples": 100000,
    "bootstrap_seed": 20260717,
    "per_clip_difference": {
      "ani:0": 0.48183572562411425,
      "ani:256": -0.8500965144485235,
      "ani:512": -0.01961088478565216,
      "animal3:0": 0.05124360918998718,
      "animal3:256": -0.010256707295775413,
      "animal3:512": -0.062463157677224705,
      "r4_new_f:0": -0.1734095641544887,
      "r4_new_f:256": -0.016665011644363403,
      "r4_new_f:512": 0.04237030570705732
    },
    "clip_better": 6,
    "clip_worse": 3,
    "clip_equal": 0,
    "mean_difference_95_ci": [
      -0.29668994355335754,
      0.13393765561006685
    ],
    "probability_mean_lt_zero": 0.70201
  }
}
```

## Union K64 versus shared-state K64

```json
{
  "pooled_visible": {
    "n": 108,
    "candidate_mean": 1.839032671892912,
    "baseline_mean": 1.9581187413919166,
    "mean_difference": -0.1190860694990045,
    "median_difference": -0.0008327718824148178,
    "better": 59,
    "worse": 0,
    "equal": 49,
    "candidate_safe4": 99,
    "baseline_safe4": 99,
    "candidate_safe8": 101,
    "baseline_safe8": 101,
    "candidate_safe16": 106,
    "baseline_safe16": 105
  },
  "horizon8_visible": {
    "n": 44,
    "candidate_mean": 2.558927813767117,
    "baseline_mean": 2.7020364797525955,
    "mean_difference": -0.14310866598547858,
    "median_difference": 0.0,
    "better": 21,
    "worse": 0,
    "equal": 23,
    "candidate_safe4": 39,
    "baseline_safe4": 39,
    "candidate_safe8": 40,
    "baseline_safe8": 40,
    "candidate_safe16": 43,
    "baseline_safe16": 43
  },
  "per_horizon_visible": {
    "1": {
      "n": 28,
      "candidate_mean": 1.2070767387548196,
      "baseline_mean": 1.2861722795059904,
      "mean_difference": -0.0790955407511709,
      "median_difference": 0.0,
      "better": 13,
      "worse": 0,
      "equal": 15,
      "candidate_safe4": 26,
      "baseline_safe4": 26,
      "candidate_safe8": 27,
      "baseline_safe8": 27,
      "candidate_safe16": 28,
      "baseline_safe16": 28
    },
    "4": {
      "n": 36,
      "candidate_mean": 1.450682113154067,
      "baseline_mean": 1.5715109759734736,
      "mean_difference": -0.12082886281940672,
      "median_difference": -0.0014188596978783607,
      "better": 25,
      "worse": 0,
      "equal": 11,
      "candidate_safe4": 34,
      "baseline_safe4": 34,
      "candidate_safe8": 34,
      "baseline_safe8": 34,
      "candidate_safe16": 35,
      "baseline_safe16": 34
    },
    "8": {
      "n": 44,
      "candidate_mean": 2.558927813767117,
      "baseline_mean": 2.7020364797525955,
      "mean_difference": -0.14310866598547858,
      "median_difference": 0.0,
      "better": 21,
      "worse": 0,
      "equal": 23,
      "candidate_safe4": 39,
      "baseline_safe4": 39,
      "candidate_safe8": 40,
      "baseline_safe8": 40,
      "candidate_safe16": 43,
      "baseline_safe16": 43
    }
  },
  "per_sequence_pooled_visible": {
    "ani": {
      "n": 38,
      "candidate_mean": 3.406492334888562,
      "baseline_mean": 3.592387050255447,
      "mean_difference": -0.18589471536688507,
      "median_difference": -0.0024075545370578766,
      "better": 24,
      "worse": 0,
      "equal": 14,
      "candidate_safe4": 32,
      "baseline_safe4": 32,
      "candidate_safe8": 33,
      "baseline_safe8": 33,
      "candidate_safe16": 36,
      "baseline_safe16": 35
    },
    "animal3": {
      "n": 39,
      "candidate_mean": 1.1292770899927769,
      "baseline_mean": 1.172841860459019,
      "mean_difference": -0.04356477046624208,
      "median_difference": -0.0009361207485198975,
      "better": 20,
      "worse": 0,
      "equal": 19,
      "candidate_safe4": 36,
      "baseline_safe4": 36,
      "candidate_safe8": 37,
      "baseline_safe8": 37,
      "candidate_safe16": 39,
      "baseline_safe16": 39
    },
    "r4_new_f": {
      "n": 31,
      "candidate_mean": 0.8105488170629307,
      "baseline_mean": 0.9427510839586537,
      "mean_difference": -0.13220226689572295,
      "median_difference": 0.0,
      "better": 15,
      "worse": 0,
      "equal": 16,
      "candidate_safe4": 31,
      "baseline_safe4": 31,
      "candidate_safe8": 31,
      "baseline_safe8": 31,
      "candidate_safe16": 31,
      "baseline_safe16": 31
    }
  },
  "pooled_clip_bootstrap": {
    "n_clips": 9,
    "bootstrap_resamples": 100000,
    "bootstrap_seed": 20260717,
    "per_clip_difference": {
      "ani:0": -0.029828553589490745,
      "ani:256": -0.4614318386052868,
      "ani:512": -0.019652931527657944,
      "animal3:0": -0.04609646317031649,
      "animal3:256": -0.014219699427485466,
      "animal3:512": -0.059354799260434354,
      "r4_new_f:0": -0.1109628826379776,
      "r4_new_f:256": -0.009414000170571464,
      "r4_new_f:512": -0.3065718953896846
    },
    "clip_better": 9,
    "clip_worse": 0,
    "clip_equal": 0,
    "mean_difference_95_ci": [
      -0.22555810673062265,
      -0.033522313228675465
    ],
    "probability_mean_lt_zero": 1.0
  },
  "horizon8_clip_bootstrap": {
    "n_clips": 9,
    "bootstrap_resamples": 100000,
    "bootstrap_seed": 20260717,
    "per_clip_difference": {
      "ani:0": -0.0070939183235168455,
      "ani:256": -0.8518860273063182,
      "ani:512": -0.025494733452796937,
      "animal3:0": 0.0,
      "animal3:256": -0.013444928824901581,
      "animal3:512": -0.06368950621357986,
      "r4_new_f:0": -0.18971809957708632,
      "r4_new_f:256": -0.016665011644363403,
      "r4_new_f:512": 0.0
    },
    "clip_better": 7,
    "clip_worse": 0,
    "clip_equal": 2,
    "mean_difference_95_ci": [
      -0.3194475286513093,
      -0.012989793552292718
    ],
    "probability_mean_lt_zero": 1.0
  }
}
```

## Novelty counts

```json
{
  "visible_rows": 108,
  "branch_b_beats_shared_k16_by_gt1": 5,
  "branch_b_beats_shared_k64_by_gt1": 4,
  "branch_b_le4_shared_k16_gt4": 0,
  "branch_b_le4_shared_k64_gt4": 0,
  "branch_b_beats_shared_k64_by_gt1_per_sequence": {
    "ani": 2,
    "animal3": 0,
    "r4_new_f": 2
  },
  "branch_b_beats_shared_k64_by_gt1_per_horizon": {
    "1": 0,
    "4": 2,
    "8": 2
  },
  "branch_b_le4_shared_k64_gt4_per_sequence": {
    "ani": 0,
    "animal3": 0,
    "r4_new_f": 0
  }
}
```

## Interpretation

CANDIDATE_POOL_INCREMENTAL_FAIL: independent state changes the future candidate pool, but Branch B K64 does not robustly beat the shared-state K64 pool across sequences/clips. The union oracle gain is sparse and structural; it does not rescue the failed capacity-2 final-output gate. Keep INDEPENDENT_STATE_FAIL and close the tested temporal multi-state route.

## Integrity

```json
{
  "checked_hashes": {
    "/gemini/code/FSPT_v9a52_clean/scripts/v9a52_independent_state_branching_audit.py": "d400e8a8b0e713df00620fa113bda1e1bddcf2612821968dd3682811913c7104",
    "/gemini/code/FSPT_v9a52_clean/outputs/paper_discovery_2026-07-05/v9a52_independent_state/v9a52_independent_state_branching.json": "929e87feb67795448cfece53d2d6d3c0eb0cc7174d30f211790e4c2eeac12d74",
    "/gemini/code/FSPT_v9a52_clean/outputs/paper_discovery_2026-07-05/v9a52_independent_state/v9a52_independent_state_branching_rows.npz": "fd84754a5c5bb8192c9ce8251de2f1bd51b567d6ead6147b6d81849d453a19b5"
  },
  "formal_rows": 216,
  "visible_rows": 108,
  "union_k16_formula_exact": true,
  "union_k64_formula_exact": true,
  "formal_gate_still_failed": true,
  "script_sha256": "db0f8eab66f85ed1acc84f3136a72d299b9e2c7d246c56524b0daaac50c6ae17"
}
```
