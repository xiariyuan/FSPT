# Patch Verifier v1 LOOCV Status (2026-06-12)

## 三句话

1. **v1 原始结果只是拟合能力演示，不是泛化结论。**
2. **v1 原始 overall 指标含 GT gate leakage**（用 `has_positive_candidate` 代替真实 gate）。
3. **只有 sequence-grouped LOOCV + real gate 的结果，才能作为是否继续这条线的正式依据。**

## LOOCV 结果

| Fold (test seq) | train_pos | test_pos | ranker_acc | pos_final_med | pos_oracle_med |
|---|---|---|---|---|---|
| bike-packing | 20 | 0 | — | — | — |
| breakdance | 20 | 0 | — | — | — |
| dance-twirl | 15 | 5 | 0.0 | 27.56 | 5.41 |
| parkour | 9 | 11 | 0.0 | 25.94 | 23.13 |
| pigs | 16 | 4 | 0.0 | 33.70 | 10.68 |
| **Avg fold median** | | | **0.0** | **29.07** | **13.07** |

Baseline positive median: **20.33**

### Pooled 口径（20 个 positive test events 直接合并）

| 指标 | 值 |
|---|---|
| positive pooled final median | **28.46** |
| positive pooled base median | 35.97 |
| positive pooled oracle median | 10.68 |
| exact oracle accuracy | **0.0** (0/20) |
| better_2px_frac | 0.8 |

注：29.07 是每折 median 再平均；pooled median 28.46 稍好但仍明显差于 Stage-2c baseline 20.33。better_2px_frac=0.8 说明 patch verifier 经常选到"比 base 好"的候选，但选不到最优候选（oracle accuracy=0/20）。

## 判定

**Patch verifier 研究线停止，不再做 v2。**

patch expression 有信号（同集拟合 ranker_acc=1.0，pooled better_2px=0.8），但当前 5-seq / 20 positive 数据规模下无法学到可泛化的 ranker（LOOCV oracle accuracy=0/20）。

### 失败根因

1. **数据太少**：20 个 positive events，每折训练集仅 9-20 个正样本
2. **Sequence 间不可泛化**：dance-twirl oracle 全在 rank 3，parkour 分散 rank 0-4，无通用模式
3. **2/5 folds 测试集零 positive**（bike-packing、breakdance）

### 全部已证伪的 ranker 方向

| 方向 | 同集拟合 | LOOCV 泛化 |
|---|---|---|
| Scalar score ranker (BCE/pairwise/listwise) | ranker_acc 0.57 | positive final 20.3（不变） |
| Frozen DINO patch verifier | ranker_acc 1.0 | positive final 28.46（更差） |

## 下一步

**停止 patch verifier 研究线。** 唯一剩余可选项：Stage-2c scalar ranker + gate threshold=0.50 做一次真实 integration smoke，只验证 overall no-harm。不做 integration smoke 则正式结束 online recovery 分支。
