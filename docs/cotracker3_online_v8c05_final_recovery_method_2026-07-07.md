# CoTracker3 Online V8-C0.5 Final Recovery Method Packaging

Date: 2026-07-07

## 1. Final method

```text
CVRRM: Causal Candidate-Visible Re-entry Recovery Mode
因果候选可见确认式重进入恢复模式
```

## 2. Main result table

| Method | AJ Δ | OA Δ | AJ_RD Δ | AJ_RD_256 Δ | AJ_RD_256 abs | Robustness | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| CoTracker3 online native | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0.5333 |  | reference |
| V7-B2 visibility-only | -0.0524 | +0.0331 | +0.0020 | +0.0047 | 0.5380 | 2/5 folds pass | visibility-only gain is small |
| TrackOn2 standalone | +1.8041 | +1.2730 | +0.0180 | +0.0111 | 0.5444 |  | strong global tracker but lower AJ_RD_256 gain than CVRRM |
| all-events w8 candidate-visible | +0.0840 | +0.9242 | +0.0153 | +0.0274 | 0.5607 | 16/4/5 pos/neg/zero | without distance sanity filter |
| CVRRM default: dist<=64, w8 | +0.0964 | +0.9462 | +0.0153 | +0.0274 | 0.5607 | 16/4/5 pos/neg/zero | recommended default |
| CVRRM high-gain: dist<=64, w16 | +0.0810 | +0.9701 | +0.0162 | +0.0286 | 0.5619 | 15/5/5 pos/neg/zero | slightly higher AJ_RD_256, more negative videos |
| OOF verifier ablation | +0.1183 | +0.9407 | +0.0154 | +0.0275 | 0.5608 | 16/4/5 pos/neg/zero | diagnostic only; not default |

## 3. Final decision

```text
Default: dist_nc_le64_w8_candidate_visible
High-gain ablation: dist_nc_le64_w16_candidate_visible
Learned verifier: diagnostic ablation only, not default
```

## 4. Default algorithm

```text
For each query q and frame t:
    low_mask = native_score[q,t] <= 0.60 or not native_visible[q,t]
    if low_mask and trigger/cooldown conditions pass:
        open recovery event e = (q,t)

For each recovery event e=(q,t0):
    if distance(native_coord[q,t0], candidate_coord[q,t0]) > 64 px:
        skip event
    for tau in [t0, t0 + 8]:
        when frame tau arrives:
            if candidate_visible[q,tau]:
                output candidate coord and visible
            else:
                keep native output
```

Strict-causal note: the future-looking loop is a runtime recovery mode; each tau is processed only when that frame arrives.

## 5. Robustness tables

### Default: dist_nc_le64_w8_candidate_visible

| Video | ΔAJ_RD_256 | ΔAJ | ΔOA |
| --- | ---: | ---: | ---: |
| shooting | -0.0533 | -0.2297 | -0.3398 |
| camel | -0.0307 | -0.4026 | -0.3927 |
| breakdance | -0.0087 | -2.5030 | -2.1961 |
| car-shadow | -0.0013 | -0.0775 | +0.0000 |
| cows | +0.0000 | +0.0000 | +0.0000 |
| drift-chicane | +0.0000 | -0.2533 | +0.0000 |
| horsejump-high | +0.0000 | +0.0000 | +0.0000 |
| lab-coat | +0.0000 | -0.0110 | +0.0870 |
| paragliding-launch | +0.0000 | -0.2588 | -0.4571 |
| india | +0.0006 | -0.0478 | +0.0000 |
| libby | +0.0023 | +3.5546 | +7.2600 |
| pigs | +0.0032 | +0.1648 | +0.4600 |
| drift-straight | +0.0033 | +0.2315 | +0.1676 |
| gold-fish | +0.0037 | -0.5429 | -0.5565 |
| dance-twirl | +0.0047 | -1.1528 | -0.5522 |
| soapbox | +0.0049 | +0.2032 | +2.1333 |
| mbike-trick | +0.0092 | +1.1206 | +2.8718 |
| dogs-jump | +0.0510 | +2.1197 | +4.0761 |
| bmx-trees | +0.0519 | +0.3800 | +0.7357 |
| bike-packing | +0.0536 | -1.0797 | -1.0631 |
| judo | +0.0559 | -0.3497 | +0.8663 |
| loading | +0.0778 | +0.1132 | +1.2007 |
| parkour | +0.0953 | +2.8669 | +6.3641 |
| motocross-jump | +0.1640 | +0.1710 | +2.2590 |
| car-roundabout | +0.2400 | -1.8886 | +0.9510 |
| blackswan |  | +0.0000 | +0.0000 |
| dog |  | -0.8243 | +2.2181 |
| goat |  | +0.0000 | +0.0000 |
| kite-surf |  | +0.0000 | +0.0000 |
| scooter-black |  | +1.5887 | +2.2941 |

### High-gain: dist_nc_le64_w16_candidate_visible

| Video | ΔAJ_RD_256 | ΔAJ | ΔOA |
| --- | ---: | ---: | ---: |
| camel | -0.0710 | -0.5923 | -0.3927 |
| shooting | -0.0533 | -0.2297 | -0.3398 |
| breakdance | -0.0090 | -2.6286 | -2.2472 |
| drift-straight | -0.0071 | +0.0330 | +0.0838 |
| car-shadow | -0.0018 | -0.1161 | +0.0000 |
| cows | +0.0000 | +0.0000 | +0.0000 |
| drift-chicane | +0.0000 | -0.2533 | +0.0000 |
| horsejump-high | +0.0000 | +0.0000 | +0.0000 |
| lab-coat | +0.0000 | -0.0110 | +0.0870 |
| paragliding-launch | +0.0000 | -0.5123 | -0.8634 |
| india | +0.0015 | -0.1162 | +0.0000 |
| dance-twirl | +0.0027 | -1.3543 | -0.5522 |
| libby | +0.0028 | +3.7957 | +7.4941 |
| gold-fish | +0.0037 | -0.5429 | -0.5565 |
| soapbox | +0.0049 | +0.2032 | +2.1333 |
| mbike-trick | +0.0092 | +1.1206 | +2.8718 |
| pigs | +0.0105 | +0.1021 | +0.4600 |
| dogs-jump | +0.0491 | +2.0527 | +4.0761 |
| bmx-trees | +0.0555 | +0.2905 | +0.5255 |
| judo | +0.0565 | -0.3246 | +0.8663 |
| loading | +0.0590 | -0.0926 | +1.2007 |
| bike-packing | +0.0757 | -0.8694 | -1.0631 |
| parkour | +0.1099 | +3.2682 | +7.3031 |
| motocross-jump | +0.1640 | +0.1710 | +2.2590 |
| car-roundabout | +0.2400 | -1.9213 | +1.2436 |
| blackswan |  | +0.0000 | +0.0000 |
| dog |  | -0.8243 | +2.2181 |
| goat |  | +0.0000 | +0.0000 |
| kite-surf |  | +0.0000 | +0.0000 |
| scooter-black |  | +1.7819 | +2.2941 |

## 6. Learned verifier decision

The OOF verifier ablation improves AJ_RD_256 by only +0.0001 over the default. Stronger verifier filtering reduces negative videos but drops AJ_RD_256 below the +0.020 target. Therefore it is not promoted as the default.

## 7. Remaining limitations

```text
1. Output-level cached replay, not state-level repair yet.
2. Cross-candidate-source generalization not yet verified.
3. Some negative videos remain.
```

## 8. Next step

```text
V8-C1: cross-candidate / cross-native generalization audit.
Then test state-level repair only if cross-source evidence holds.
```
