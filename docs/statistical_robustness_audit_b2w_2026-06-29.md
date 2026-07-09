# Statistical Robustness Audit for B2-W — 2026-06-29

## Decision

This audit uses existing per-video metrics only; no model inference or server-heavy experiment was run. It performs paired per-video bootstrap confidence intervals and exact sign tests for the main DAVIS result, RGB fresh20-49 validation, and TrackOn2 plug-in supplementary experiment.

The result strengthens the paper: the main DAVIS and RGB AJ_RD gains are robust across videos, while standard-AJ loss is confirmed as a real tradeoff. The TrackOn2 plug-in gain is positive in aggregate but not statistically robust enough to be a main claim, so it should remain supplementary.

## Summary table

| comparison | metric | mean delta | 95% bootstrap CI | positive videos | sign-test p | interpretation |
|---|---|---:|---|---:|---:|---|
| DAVIS B2-W16-P2 vs fixed | AJ_RD_256 | +0.0795 | [+0.0445, +0.1176] | 18 / 25 | 0.02265584 | robust re-entry gain |
| DAVIS B2-W16-P2 vs fixed | AJ_256 | -1.0391 | [-2.0250, -0.2204] | 10 / 30 | 0.13604595 | standard-AJ tradeoff is real |
| DAVIS B2-W16-P2 vs global B1 | AJ_256 | +21.6072 | [+18.5925, +24.5352] | 30 / 30 | 0.0 | recovers standard AJ vs global fusion |
| RGB fresh20-49 B2-W16-P2 vs offline | AJ_RD_256 | +0.0613 | [+0.0423, +0.0804] | 24 / 30 | 0.00054611 | robust cross-domain re-entry gain |
| RGB fresh20-49 B2-W16-P2 vs offline | AJ_256 | -0.5280 | [-0.9730, -0.0715] | 7 / 30 | 0.00522288 | small but real standard-AJ cost |
| RGB fresh20-49 B2-W16-P2 vs online | AJ_RD_256 | +0.0390 | [+0.0164, +0.0671] | 23 / 30 | 0.0023157 | beats online on re-entry |
| RGB fresh20-49 B2-W16-P2 vs online | AJ_256 | +34.3731 | [+32.9499, +35.7175] | 30 / 30 | 0.0 | strongly avoids online standard collapse |
| TrackOn2 first-input B2-W16-P2 vs TrackOn2 | AJ_RD_256 | +0.0107 | [-0.0108, +0.0381] | 12 / 25 | 0.50344467 | positive aggregate, not statistically robust |
| TrackOn2 first-input B2-W16-P2 vs TrackOn2 | AJ_256 | +0.0854 | [-0.3813, +0.5393] | 13 / 30 | 1.0 | positive aggregate, not statistically robust |
| First-input CoTracker base + TrackOn2 override vs CoTracker | AJ_RD_256 | +0.0997 | [+0.0581, +0.1431] | 20 / 25 | 0.00154388 | robust gain over weaker base |
| First-input CoTracker base + TrackOn2 override vs CoTracker | AJ_256 | +2.2635 | [+1.1797, +3.4175] | 20 / 30 | 0.06142835 | robust standard gain over weaker base |

## Main paper implications

### 1. DAVIS re-entry gain is robust

```text
mean per-video delta = +0.079496
95% bootstrap CI = [+0.044460, +0.117616]
positive videos = 18 / 25
sign-test p = 0.02265584
```

This supports the claim that B2-W16-P2 improves DAVIS re-entry reliability across videos, not only through one outlier.

### 2. RGB fresh20-49 re-entry gain is even stronger

```text
mean per-video delta = +0.061327
95% bootstrap CI = [+0.042250, +0.080410]
positive videos = 24 / 30
sign-test p = 0.00054611
```

This is a strong cross-domain validation result. It should be mentioned in the experiments or appendix as evidence that the RGB fresh20-49 gain is stable.

### 3. Standard-AJ cost is real and should be framed honestly

DAVIS:

```text
mean AJ delta = -1.039103
95% bootstrap CI = [-2.025035, -0.220383]
```

RGB fresh20-49:

```text
mean AJ delta = -0.527960
95% bootstrap CI = [-0.972981, -0.071517]
```

Therefore the correct framing remains a favorable re-entry / standard-tracking tradeoff, not universal standard-AJ improvement.

### 4. B2-W strongly avoids global/online collapse

DAVIS standard AJ vs global B1:

```text
mean delta = +21.607247
95% bootstrap CI = [+18.592473, +24.535181]
positive videos = 30 / 30
```

RGB standard AJ vs online override:

```text
mean delta = +34.373147
95% bootstrap CI = [+32.949920, +35.717518]
positive videos = 30 / 30
```

This is one of the strongest statistical supports for the paper: B2-W does not merely improve AJ_RD; it avoids the catastrophic standard-tracking collapse of global/online re-entry branches.

### 5. TrackOn2 plug-in result should remain supplementary

```text
AJ_RD_256 mean delta = +0.010680, 95% CI = [-0.010800, +0.038060]
AJ_256 mean delta = +0.085407, 95% CI = [-0.381336, +0.539262]
```

The aggregate direction is positive, but the confidence intervals cross zero. This result is still useful as plug-in generality evidence, but it should not be phrased as a robust win over TrackOn2.

## Recommended paper wording

Safe wording:

```text
Per-video paired bootstrap analysis confirms that B2-W16-P2 improves AJ_RD over the base on DAVIS and RGB fresh20-49, while incurring a measurable but controlled standard-AJ cost. The same analysis shows that B2-W strongly avoids the standard-AJ collapse of global/online re-entry branches. A supplementary TrackOn2 first-query experiment shows a small positive aggregate plug-in gain, but this result is not statistically robust and is therefore reported as supplemental evidence rather than a main claim.
```

Unsafe wording:

```text
Do not say B2-W significantly improves TrackOn2.
Do not say B2-W improves standard AJ in general.
Do not hide the standard-AJ cost vs offline base.
```

## Artifacts

```text
scripts/audit_statistical_robustness_b2w.py
outputs/paper_discovery_2026-06-27/statistical_robustness_audit/summary.json
```
