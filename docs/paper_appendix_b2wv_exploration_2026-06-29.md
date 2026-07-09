# Appendix Draft: B2-WV Exploration and Claim Boundary — 2026-06-29

## Purpose

This appendix records the learned window-verification exploration. It should not replace the main B2-W16-P2 method. The goal is to show that we investigated stronger learned routing, found meaningful oracle headroom, but also found that runtime-only learned policies fail to preserve AJ_RD.

## B2-WV oracle upper bound

The B2-WV oracle uses ground-truth counterfactual best actions for candidate windows. It is diagnostic only and is not deployable.

| dataset | method | AJ_RD_256 | AJ_256 | delta AJ_RD vs P2 | delta AJ vs P2 |
|---|---|---:|---:|---:|---:|
| davis | B2-W16-P2 | 0.6251 | 69.0119 | - | - |
| davis | B2-WV oracle lambda=1 | 0.6455 | 71.1162 | +0.0204 | +2.1043 |
| rgb_dev10 | B2-W16-P2 | 0.4863 | 79.9035 | - | - |
| rgb_dev10 | B2-WV oracle lambda=1 | 0.4756 | 80.6310 | -0.0107 | +0.7275 |

Interpretation: the oracle shows strong headroom on DAVIS, but the same utility becomes too conservative on RGB dev10 and trades away AJ_RD for standard AJ. Therefore, the oracle supports B2-WV as a future direction, not as a deployable result.

## Learned B2-WV-Lite policies

| dataset | policy | AJ_RD_256 | AJ_256 | delta AJ_RD vs P2 | delta AJ vs P2 |
|---|---|---:|---:|---:|---:|
| davis | B2-W16-P2 | 0.6251 | 69.0119 | - | - |
| davis | argmax | 0.6046 | 69.4980 | -0.0205 | +0.4861 |
| davis | preserve95_dyn | 0.6061 | 69.4896 | -0.0190 | +0.4777 |
| davis | strict95_dyn | 0.5790 | 69.8558 | -0.0461 | +0.8439 |
| rgb_dev10 | B2-W16-P2 | 0.4863 | 79.9035 | - | - |
| rgb_dev10 | argmax | 0.4508 | 79.9760 | -0.0355 | +0.0725 |
| rgb_dev10 | preserve95_dyn | 0.4488 | 79.9619 | -0.0375 | +0.0584 |
| rgb_dev10 | strict95_dyn | 0.3957 | 80.0616 | -0.0906 | +0.1581 |

Interpretation: learned runtime-feature policies recover some standard AJ, but the AJ_RD loss is too large. This means the learned policy fails the main requirement of preserving re-entry reliability.

## P2-safe window shortening / filtering

| dataset | policy | AJ_RD_256 | AJ_256 | delta AJ_RD vs P2 | delta AJ vs P2 |
|---|---|---:|---:|---:|---:|
| davis | B2-W16-P2 | 0.6251 | 69.0119 | - | - |
| davis | p2_short_re99 | 0.6166 | 69.2530 | -0.0085 | +0.2411 |
| davis | p2_short_re97 | 0.6092 | 69.3941 | -0.0159 | +0.3822 |
| davis | p2_veto_re99 | 0.6086 | 69.3638 | -0.0165 | +0.3519 |
| rgb_dev10 | B2-W16-P2 | 0.4863 | 79.9035 | - | - |
| rgb_dev10 | p2_short_re99 | 0.4450 | 79.9942 | -0.0413 | +0.0907 |
| rgb_dev10 | p2_short_re97 | 0.4379 | 80.0413 | -0.0484 | +0.1378 |
| rgb_dev10 | p2_veto_re99 | 0.4191 | 79.9787 | -0.0672 | +0.0752 |

Interpretation: even conservative modifications to B2-W16-P2 recover only modest standard AJ and still lose too much AJ_RD, especially on RGB dev10.

## Claim boundary

Safe statement:

```text
We explored counterfactual-supervised window-level verification. The GT oracle shows substantial headroom, but lightweight runtime-feature policies do not preserve AJ_RD. We therefore keep B2-W16-P2 as the main method and treat appearance-grounded verification as future work.
```

Unsafe statements:

```text
Do not claim B2-WV is the main method.
Do not claim learned B2-WV improves the final result.
Do not report oracle numbers as deployable performance.
Do not claim runtime-only learned verification solves false-trigger cost.
```

## Next method direction

The failure mode suggests that trajectory and visibility features can identify re-entry-like windows but cannot reliably determine whether the override branch re-localizes the same physical point. A stronger B2-WV version should add appearance evidence, such as query-patch, last-visible-patch, and candidate-patch similarity.

## Artifacts

```text
scripts/build_b2wv_counterfactual_windows.py
scripts/eval_b2wv_oracle_cache.py
scripts/audit_b2wv_feature_separability.py
scripts/eval_b2wv_lite_policy_cv.py
scripts/eval_b2wv_p2safe_cv.py
scripts/build_b2wv_appearance_patch_features.py
outputs/paper_discovery_2026-06-27/b2wv_oracle_dev/summary.json
outputs/paper_discovery_2026-06-27/b2wv_lite_policy_cv/summary.json
outputs/paper_discovery_2026-06-27/b2wv_p2safe_cv/summary.json
docs/b2wv_appearance_verifier_plan_2026-06-29.md
```
