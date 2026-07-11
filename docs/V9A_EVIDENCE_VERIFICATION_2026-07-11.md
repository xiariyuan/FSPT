# V9-A Evidence Chain Verification

Date: 2026-07-11

Overall status: **PASS**

## Index and artifact checks

```json
{
  "verification_tool_head": "bf4aed14c8b3dfa656991392bf27e27dd3982c01",
  "generated_head": "bf4aed14c8b3dfa656991392bf27e27dd3982c01",
  "checkout_descends_from_generated_head": true,
  "checkout_descends_from_verification_tool_head": true,
  "routes": 8,
  "artifacts_checked": 43,
  "unique_artifacts_checked": 43
}
```

## Independent route checks

```json
{
  "V9-A4.5": {
    "teacher_mean": 14.745758832552221,
    "student_mean": 15.038686401482162,
    "difference": 0.2929275689299394,
    "gate_pass": false
  },
  "V9-A5.0": {
    "best_sampled": "causal_teacher_latent",
    "sampled_difference": -0.2994817793369293,
    "best_oracle": "past_gt_motion",
    "oracle_difference": -6.984335899353027,
    "sampled_pass_count": 0,
    "oracle_pass_count": 5
  },
  "V9-A5C.0": {
    "rows": 4878,
    "global_k16_recall4": 0.7728577285772857,
    "global_k64_recall4": 0.9286592865928659,
    "opportunity_rows": 760,
    "gate_pass": true
  },
  "V9-A5.1a": {
    "rows": 27648,
    "visible_rows": 20218,
    "teacher_top1_mean": 9.317731847230391,
    "beam_top1_mean": 9.75613061732704,
    "teacher_top4_oracle_mean": 6.315012004535443,
    "beam_top4_oracle_mean": 7.482220997140607
  },
  "V9-A5.1b": {
    "rows": 27648,
    "visible_rows": 19930,
    "official_mean": 9.141794939227056,
    "hybrid_oracle_mean": 7.81845032557433,
    "gate_pass": true
  },
  "V9-A5.1c": {
    "raw_oracle_pass": true,
    "deterministic_pass": false,
    "fair_top1_difference": 0.0551646970121804,
    "fair_oracle_difference": 0.2515971643002531,
    "final_incremental_pass": false
  },
  "V9-A5.2": {
    "rows": 216,
    "horizon8_visible_rows": 44,
    "horizon8_difference": 0.2839154005050659,
    "q_new_divergence_fraction": 1.0,
    "useful_novel_rows": 3,
    "gate_pass": false
  },
  "V9-A6.0": {
    "opportunities": 760,
    "primary_promoted": 67,
    "primary_conversion": 0.0881578947368421,
    "wide_promoted": 264,
    "wide_conversion": 0.3473684210526316,
    "all_opportunities_visible": true,
    "gate_pass": false
  }
}
```

## Boundary

This verification checks committed artifacts and independently recomputes key saved-array formulas. It does not rerun TrackOn2 GPU inference or reproduce external datasets/checkpoints.
