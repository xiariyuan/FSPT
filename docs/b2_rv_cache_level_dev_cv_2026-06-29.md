# B2-RV Cache-Level Development CV — 2026-06-29

## Decision

B2-RV was evaluated as a true cache-level verifier under grouped video cross-validation on DAVIS + RGB dev first-10. It was **not** evaluated or tuned on RGB fresh20-49.

The result is informative but not yet good enough to replace B2-W16-P2 as the main method. B2-RV improves standard AJ over B2-W16-P2 but loses too much AJ_RD, especially on DAVIS. It should remain an exploratory extension unless a more stable verifier is developed.

## Protocol

```text
Development-only cache-level grouped CV for B2-RV. Uses DAVIS + RGB dev first-10. No RGB fresh20-49.
track-level OOF verifier: if first trigger accepted, use B2-W16-P2 output for the whole query track; otherwise keep base output.
```

## Aggregate development estimate

| policy | AJ_RD_256 | AJ_256 | ΔAJ_RD vs base | ΔAJ vs base | ΔAJ_RD vs P2 | ΔAJ vs P2 |
|---|---:|---:|---:|---:|---:|---:|
| score_safe90 | 0.515698 | 76.659087 | 0.082023 | -0.188323 | -0.015267 | 0.260395 |
| harm_safe90 | 0.51836 | 76.656558 | 0.084685 | -0.190852 | -0.012604 | 0.257866 |

Both policies improve AJ over B2-W16-P2 but lose AJ_RD. This is the wrong tradeoff for the main method unless the paper wants a very conservative operating point.

## Dataset-level cache metrics

### davis

| method | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 | ΔAJ_RD vs base | ΔAJ vs base | ΔAJ_RD vs P2 | ΔAJ vs P2 | accept rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 0.5546 | 70.051 | 92.1543 | 82.3206 | 0 | 0 | - | - | - |
| B2-W16 | 0.6266 | 68.9512 | 91.1924 | 82.448 | 0.072 | -1.0998 | 0.0015 | -0.0607 | - |
| B2-W16-P2 | 0.6251 | 69.0119 | 91.2382 | 82.4425 | 0.0705 | -1.0391 | 0 | 0 | 1.0 |
| B2-RV score_safe90 | 0.5922 | 69.4704 | 91.5442 | 82.3609 | 0.0376 | -0.5806 | -0.0329 | 0.4585 | 0.523173 |
| B2-RV harm_safe90 | 0.6049 | 69.3873 | 91.3606 | 82.4157 | 0.0503 | -0.6637 | -0.0202 | 0.3754 | 0.848017 |

### rgb_dev10

| method | AJ_RD_256 | AJ_256 | OA_256 | delta_avg_256 | ΔAJ_RD vs base | ΔAJ vs base | ΔAJ_RD vs P2 | ΔAJ vs P2 | accept rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 0.3763 | 80.0721 | 90.9956 | 88.9997 | 0 | 0 | - | - | - |
| B2-W16 | 0.4865 | 79.903 | 93.0717 | 89.0995 | 0.1102 | -0.1691 | 0.0002 | -0.0005 | - |
| B2-W16-P2 | 0.4863 | 79.9035 | 93.0641 | 89.0961 | 0.11 | -0.1686 | 0 | 0 | 1.0 |
| B2-RV score_safe90 | 0.4794 | 80.0699 | 92.9492 | 89.116 | 0.1031 | -0.0022 | -0.0069 | 0.1664 | 0.71392 |
| B2-RV harm_safe90 | 0.4773 | 80.1056 | 92.9926 | 89.0833 | 0.101 | 0.0335 | -0.009 | 0.2021 | 0.870196 |

## Interpretation

The cache-level result confirms that the trigger-row utility estimate was too optimistic. Rejecting full tracks based on a first-trigger verifier recovers some standard AJ, but it also suppresses useful re-entry windows and therefore reduces AJ_RD.

Key observations:

```text
1. On DAVIS, B2-RV harm_safe90 improves AJ over P2 by +0.3754 but loses -0.0202 AJ_RD.
2. On RGB dev10, B2-RV harm_safe90 improves AJ over P2 by +0.2021 but loses -0.0090 AJ_RD.
3. B2-RV score_safe90 is too aggressive; it accepts only 52.3% of DAVIS trigger tracks and loses -0.0329 AJ_RD vs P2 on DAVIS.
4. Track-level gating is probably too coarse; future verifier should be window-level or should use a softer fallback instead of rejecting the entire P2 track.
```

## Recommendation

Do not replace B2-W16-P2 with B2-RV in the main paper. Keep B2-W16-P2 as the main method. B2-RV can be mentioned as a future direction or appendix exploration: runtime features are predictive, but a naive track-level verifier loses too much re-entry benefit.

If continuing B2-RV, the next version should be:

```text
1. window-level verifier instead of track-level verifier
2. reject only high-risk windows, not whole tracks
3. include a soft fallback: shorten W or use base after K frames
4. optimize a direct AJ_RD/AJ tradeoff objective
5. validate only on development folds before touching any fresh data
```

## Artifacts

```text
scripts/eval_b2_rv_cache_dev_cv.py
outputs/paper_discovery_2026-06-27/b2_rv_cache_dev_cv/summary.json
outputs/paper_discovery_2026-06-27/b2_rv_cache_dev_cv/oof_scores.jsonl
outputs/paper_discovery_2026-06-27/b2_rv_cache_dev_cv/davis/b2_rv_harm_safe90_davis.pt
outputs/paper_discovery_2026-06-27/b2_rv_cache_dev_cv/rgb_dev10/b2_rv_harm_safe90_rgb_dev10.pt
```
