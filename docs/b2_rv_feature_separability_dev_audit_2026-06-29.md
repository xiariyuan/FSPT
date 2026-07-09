# B2-RV Feature Separability Audit on Development Data — 2026-06-29

## Decision

This is a development-only audit for a possible B2-RV / Re-entry Verifier. It uses only DAVIS and RGB dev first-10, and does not use RGB fresh20-49. The goal is to check whether runtime-observable trigger features can separate helpful/harmful override decisions before training any verifier.

The result is positive enough to continue: several runtime features show moderate separability. This does not yet prove that a learned verifier will improve final metrics, but it justifies a next-stage lightweight verifier experiment.

## Protocol

```text
Development-only feature separability audit for future B2-RV. Uses DAVIS + RGB dev first-10 only. Does not use RGB fresh20-49.
```

## Dataset summaries

| dataset | triggered tracks | true triggers | false triggers | true-trigger rate | delta-full mean | delta-post mean | harmful-full rate | helpful-post rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| davis | 2395 | 1314 | 1081 | 0.548643 | -0.011611 | 0.104389 | 0.265971 | 0.423382 |
| rgb_dev10 | 3513 | 2250 | 1263 | 0.640478 | 0.027328 | 0.177917 | 0.128665 | 0.574438 |
| combined_dev | 5908 | 3564 | 2344 | 0.60325 | 0.011543 | 0.14811 | 0.184326 | 0.513202 |

## Top feature separability

### Harmful full-track override

| feature | best AUC | direction | AUC if high=positive |
|---|---:|---|---:|
| base_override_dist_max_next16 | 0.663561 | high | 0.663561 |
| base_override_dist_mean_next16 | 0.658532 | high | 0.658532 |
| base_visible_frac_next16 | 0.646949 | low | 0.353051 |
| base_override_dist_mean_next4 | 0.619395 | high | 0.619395 |
| vis_agreement_frac_next16 | 0.618543 | low | 0.381457 |
| base_override_dist_t | 0.600473 | high | 0.600473 |
| override_visibility_transitions_next16 | 0.587214 | high | 0.587214 |
| override_speed_mean_next16 | 0.5869 | high | 0.5869 |
| override_speed_mean_next4 | 0.581345 | high | 0.581345 |
| override_visible_frac_next16 | 0.580987 | low | 0.419013 |

### Helpful post-trigger override

| feature | best AUC | direction | AUC if high=positive |
|---|---:|---|---:|
| vis_agreement_frac_next16 | 0.758095 | low | 0.241905 |
| base_visible_frac_next16 | 0.711612 | low | 0.288388 |
| n_trigger_windows | 0.652404 | high | 0.652404 |
| base_invis_run | 0.649105 | low | 0.350895 |
| base_override_dist_mean_next4 | 0.5814 | high | 0.5814 |
| trigger_t_norm | 0.574642 | high | 0.574642 |
| base_override_dist_mean_next16 | 0.57132 | high | 0.57132 |
| base_override_dist_t | 0.563678 | high | 0.563678 |
| base_visibility_transitions_next16 | 0.554426 | low | 0.445574 |
| override_visibility_transitions_next16 | 0.551853 | low | 0.448147 |

### True re-entry trigger

| feature | best AUC | direction | AUC if high=positive |
|---|---:|---|---:|
| base_invis_run | 0.705624 | high | 0.705624 |
| vis_agreement_frac_next16 | 0.692354 | high | 0.692354 |
| trigger_t_norm | 0.691392 | low | 0.308608 |
| base_visible_frac_next16 | 0.671443 | high | 0.671443 |
| base_override_dist_mean_next4 | 0.610293 | low | 0.389707 |
| base_override_dist_t | 0.593681 | low | 0.406319 |
| base_override_dist_mean_next16 | 0.593187 | low | 0.406813 |
| base_visibility_transitions_next16 | 0.591849 | high | 0.591849 |
| n_trigger_windows | 0.5918 | high | 0.5918 |
| query_age_norm | 0.560819 | low | 0.439181 |

### Bad post-trigger override

| feature | best AUC | direction | AUC if high=positive |
|---|---:|---|---:|
| base_visible_frac_next16 | 0.661488 | high | 0.661488 |
| vis_agreement_frac_next16 | 0.658598 | high | 0.658598 |
| base_visibility_transitions_next16 | 0.647202 | high | 0.647202 |
| override_visibility_transitions_next16 | 0.588614 | high | 0.588614 |
| base_invis_run | 0.579085 | high | 0.579085 |
| override_visible_frac_next16 | 0.557648 | low | 0.442352 |
| override_persist_len_cap16 | 0.556078 | low | 0.443922 |
| base_override_dist_t | 0.549094 | low | 0.450906 |
| n_trigger_windows | 0.548936 | low | 0.451064 |
| base_override_dist_mean_next4 | 0.542967 | low | 0.457033 |

## Interpretation

The strongest signals are not single perfect rules, but several runtime features are moderately informative:

```text
harmful_full: large base/override disagreement over the next 16 frames is informative, best AUC ≈ 0.664
helpful_post: low base/override visibility agreement and low base visibility in the next 16 frames are informative, best AUC ≈ 0.758
true_trigger: long base-invisible run and high later base/override visibility agreement are informative, best AUC ≈ 0.706 / 0.692
bad_post: high base visibility and high base/override visibility agreement are warning signs, best AUC ≈ 0.661
```

This suggests that a B2-RV verifier should not be a single threshold. A small learned model or a constrained score combining disagreement, base visibility, trigger timing, and persistence may be justified.

## Recommended next experiment

Train a lightweight development-only verifier using these runtime features. Start with logistic regression / small MLP or a simple calibrated linear score. Use labels such as `harmful_full` rejection and/or `helpful_post` acceptance. Evaluate only on held-out development folds first; do not tune on RGB fresh20-49.

A safe next gate is:

```text
B2-RV should reduce false-trigger / harmful-full rate while preserving most AJ_RD.
Expected improvement target: AJ drop closer to zero, with AJ_RD loss <= 0.005 versus B2-W16-P2.
```

## Artifacts

```text
scripts/audit_b2_rv_feature_separability_dev.py
outputs/paper_discovery_2026-06-27/b2_rv_feature_audit_dev/summary.json
outputs/paper_discovery_2026-06-27/b2_rv_feature_audit_dev/trigger_rows.jsonl
```
