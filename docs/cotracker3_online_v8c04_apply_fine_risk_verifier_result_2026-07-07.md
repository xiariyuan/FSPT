# CoTracker3 Online V8-C0.4 OOF Fine-Risk Verifier Application Result

Date: 2026-07-07

Script:

```text
scripts/eval_cotracker3_online_v8c04_apply_fine_risk_verifier.py
```

Report:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v8c04_apply_fine_risk_verifier/v8c04_apply_fine_risk_verifier_report.json
```

---

## 1. Purpose

V8-C0.3 showed that some touched-frame risks are learnable under video-level LOOV:

```text
candidate_good: RandomForest AP 0.7763, AUC 0.7829
false_visible: LogisticRegression AP 0.5182, AUC 0.7979
```

V8-C0.4 applies these OOF verifier scores back to trajectories to answer the real question:

```text
Does verifier-filtered recovery improve AJ/OA/AJ_RD/AJ_RD_256 compared with the simple default recovery mode?
```

Baseline default:

```text
dist_nc_le64_w8_candidate_visible
```

---

## 2. Sanity check

The `accept_all_baseline` variant accepts every touched frame in the V8-C0.2 dataset.

It reproduces the V8-C0.1 default result exactly:

```text
accept_all_baseline:
AJ        +0.0964
OA        +0.9462
AJ_RD     +0.0153
AJ_RD_256 +0.0274
positive/negative/zero videos = 16/4/5
accepted = 1456 frames
```

Conclusion:

```text
Touched-frame mapping and application logic are correct.
```

---

## 3. Best AJ_RD_256 verifier-filtered result

Best AJ_RD_256 row:

```text
utility_et_ge_-0.3431
```

Result:

```text
AJ        +0.1183
OA        +0.9407
AJ_RD     +0.0154
AJ_RD_256 +0.0275
positive/negative/zero videos = 16/4/5
accepted = 1383 / 1456 frames
accept_rate = 0.9499
```

Compared with baseline:

```text
baseline AJ_RD_256: +0.0274
verifier AJ_RD_256: +0.0275
```

Difference:

```text
+0.0001 AJ_RD_256
```

Interpretation:

```text
The verifier can make a tiny improvement, but it is not meaningful enough to become the main method.
```

Risk stats improved slightly:

```text
baseline false_visible_rate = 0.2493
verifier false_visible_rate = 0.2350

baseline candidate_bad_rate = 0.3221
verifier candidate_bad_rate = 0.3116
```

But the per-video robustness is unchanged:

```text
baseline positive/negative/zero = 16/4/5
verifier positive/negative/zero = 16/4/5
```

---

## 4. Robust verifier variants

Some verifier thresholds reduce negative videos from 4 to 2, but the AJ_RD_256 cost is large.

Example:

```text
good_rf_minus_false_lr_ge_-0.2709
```

Result:

```text
AJ        +0.1718
OA        +1.0280
AJ_RD     +0.0082
AJ_RD_256 +0.0162
positive/negative/zero videos = 16/2/7
accepted = 1092 / 1456 frames
accept_rate = 0.7500
false_visible_rate = 0.1630
candidate_bad_rate = 0.2390
```

Interpretation:

```text
This rule reduces false-visible risk and negative videos, but loses too much re-entry recovery gain.
It falls below the +0.020 AJ_RD_256 target.
```

Another false-visible filter:

```text
false_lr_le_0.60:
AJ_RD_256 +0.0114
positive/negative/zero = 16/2/7
```

This is too weak on AJ_RD_256.

---

## 5. Good-score filtering

`good_rf_ge_0.30` enriches accepted frames for useful candidates:

```text
accepted = 922 / 1456
candidate_good_rate = 0.5759
AJ        +0.1917
OA        +0.9750
AJ_RD     +0.0121
AJ_RD_256 +0.0239
positive/negative/zero = 12/4/9
```

Interpretation:

```text
The verifier does select higher-quality candidate frames, but it removes too many useful re-entry frames and reduces video coverage.
Overall AJ_RD_256 drops from +0.0274 to +0.0239.
```

---

## 6. Main conclusion

The learned verifier is useful diagnostically but not strong enough as a default policy.

Summary:

```text
1. Applying OOF verifier scores can slightly increase AJ and reduce false-visible rate.
2. Best AJ_RD_256 improvement over baseline is only +0.0001.
3. Stronger risk filtering reduces negative videos but drops AJ_RD_256 below target.
4. The simple candidate-visible recovery mode remains the best default.
```

Therefore:

```text
Do not promote learned fine-risk verifier as the main method.
```

---

## 7. Recommended current method

Default method:

```text
dist_nc_le64_w8_candidate_visible
```

Result:

```text
AJ        +0.0964
OA        +0.9462
AJ_RD     +0.0153
AJ_RD_256 +0.0274
positive/negative/zero = 16/4/5
```

High-gain ablation:

```text
dist_nc_le64_w16_candidate_visible
```

Result:

```text
AJ        +0.0810
OA        +0.9701
AJ_RD     +0.0162
AJ_RD_256 +0.0286
positive/negative/zero = 15/5/5
```

Optional diagnostic verifier ablation:

```text
utility_et_ge_-0.3431
```

Result:

```text
AJ        +0.1183
OA        +0.9407
AJ_RD     +0.0154
AJ_RD_256 +0.0275
positive/negative/zero = 16/4/5
```

But this should not be the default because the gain is marginal and threshold selection is not yet independently validated.

---

## 8. Reflection

The original CRRP-v2 idea is now substantially simplified by evidence.

Current evidence says:

```text
The strongest contribution is not a learned counterfactual utility model.
The strongest contribution is a strict-causal finite-horizon recovery mode:
  broad native failure event proposal
  + distance sanity filter
  + candidate-visible frame-level confirmation.
```

The learned verifier is not useless, but it is not worth the complexity as the default.

---

## 9. Next step

Run a final packaging / robustness step:

```text
V8-C0.5 method packaging and robustness table
```

Needed outputs:

```text
1. Final method name and exact algorithm.
2. Main result table:
   - native
   - TrackOn2 standalone
   - V7-B2 visibility-only
   - dist_nc_le64_w8_candidate_visible default
   - dist_nc_le64_w16_candidate_visible high-gain ablation
   - optional verifier ablation
3. Per-video robustness table.
4. Method pseudocode.
5. Stop learned verifier as main route unless new cross-source evidence requires it.
```

Suggested default method name:

```text
Causal Candidate-Visible Re-entry Recovery Mode
```

Short code name:

```text
CVRRM
```
