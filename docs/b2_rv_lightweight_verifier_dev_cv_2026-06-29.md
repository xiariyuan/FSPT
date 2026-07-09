# B2-RV Lightweight Verifier Prototype — Development CV — 2026-06-29

## Decision

A lightweight B2-RV verifier prototype was tested using grouped cross-validation over DAVIS + RGB dev first-10 only. RGB fresh20-49 was not used. The result shows useful signal, but the prototype is **not yet ready as a new main method**: the score-based verifier improves full-track estimated utility but rejects too many true triggers, especially on DAVIS. A conservative harm-only policy is safer and should be the next refinement target.

## Protocol

```text
Development-only B2-RV grouped cross-validation on DAVIS + RGB dev first-10. No RGB fresh20-49 is used.
n_rows = 5908
n_video_groups = 39
model = logistic regression with imputation + standardization
p_harm = P(harmful_full)
p_help = P(helpful_post)
score = p_help - p_harm
thresholds selected only on train folds
```

## Out-of-fold classification quality

```text
AUC harmful_full = 0.654473
AUC helpful_post = 0.789032
AUC true_trigger using score = 0.334976
```

Interpretation: the verifier can predict harmful/helpful local utility moderately well, but `score = P(help)-P(harm)` is not a good true-trigger classifier. This is acceptable if B2-RV is framed as a reliability/utility gate, but dangerous if used as a re-entry detector.

## Trigger-policy estimates

Baseline accept-all B2-W16-P2 over dev triggered rows:

```text
n = 5908
accepted = 5908
full_sum = 68.197061
post_sum = 875.033793
full_mean_all_triggers = 0.011543
post_mean_all_triggers = 0.14811
harmful_total = 1089
helpful_total = 3032
true_total = 3564
```

| policy | accept rate | full Δ vs accept-all | post retention | harmful reject rate | helpful retention | true retention | comment |
|---|---:|---:|---:|---:|---:|---:|---|
| score_safe90 | 0.6366 | +13.6610 | 0.8826 | 0.3664 | 0.7770 | 0.5328 | best full-sum gain, but too many true triggers rejected |
| score_safe85 | 0.5379 | +10.6506 | 0.8191 | 0.4619 | 0.6972 | 0.4366 | aggressive utility gate |
| score_safe80 | 0.4975 | +9.9590 | 0.7837 | 0.4995 | 0.6527 | 0.3959 | aggressive utility gate |
| harm_safe90 | 0.8612 | +8.4807 | 0.8896 | 0.2286 | 0.8486 | 0.8558 | most conservative useful candidate |
| harm_safe85 | 0.8153 | +3.0824 | 0.8359 | 0.2837 | 0.7935 | 0.8106 | harm-only conservative gate |
| harm_safe80 | 0.7747 | +5.8694 | 0.7905 | 0.3434 | 0.7460 | 0.7736 | harm-only conservative gate |

## Important caveat: cross-domain behavior

The conservative `harm_safe90` policy behaves differently on DAVIS and RGB dev:

| dataset | accept rate | full Δ vs accept-all | post retention | harmful reject rate | helpful retention | true retention |
|---|---:|---:|---:|---:|---:|---:|
| davis | 0.8480 | -0.8814 | 0.8272 | 0.1821 | 0.8107 | 0.8280 |
| rgb_dev10 | 0.8702 | +9.3621 | 0.9146 | 0.2942 | 0.8677 | 0.8720 |

On RGB dev, `harm_safe90` improves full-sum estimate; on DAVIS it slightly hurts full-sum estimate while reducing post utility. This means the current verifier is not yet robust enough to replace B2-W16-P2 as the main method.

## Current conclusion

B2-RV is promising as a future enhancement, but the current prototype should be treated as exploratory. It should not be evaluated on RGB fresh20-49 as a tuned method yet. The main paper should still use B2-W16-P2 as the robust method, and B2-RV can be discussed as an extension only after a more stable training/evaluation protocol is developed.

## Recommended next refinement

The next B2-RV step should not be a more complex model immediately. Instead:

```text
1. Separate two decisions: re-entry trigger recall vs harmful-trigger rejection.
2. Prefer harm-only conservative gating over score gating because it preserves true triggers better.
3. Add domain-balanced training or leave-one-dataset-out checks.
4. Evaluate actual cache-level B2-RV on dev folds, not just trigger-row utility estimates.
5. Only if dev-fold cache-level results are stable, move to a new untouched split or report as future work.
```

## Artifacts

```text
scripts/train_b2_rv_dev_cv.py
outputs/paper_discovery_2026-06-27/b2_rv_dev_cv/summary.json
outputs/paper_discovery_2026-06-27/b2_rv_dev_cv/oof_scores.jsonl
```
