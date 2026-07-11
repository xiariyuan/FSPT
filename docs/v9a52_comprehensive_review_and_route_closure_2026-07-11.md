# V9-A5.2 Comprehensive Review and Temporal Multi-State Route Closure

Date: 2026-07-11

## 1. Executive conclusion

V9-A5.2 successfully implemented true independent TrackOn2 model-state branches and passed every execution-integrity gate.

The branches were real, complete tracker states:

```text
432 active queries = 32 evaluated + 400 support-grid queries
M = 24 memory entries
D = 256 feature dimensions
separate q_init / point_memory / temporal_mask storage
separate full-state forward and unconditional memory updates
```

Independent state clearly changed future hidden state and proposal generation:

```text
target q_new L2 > 1e-4 on 100% of visible event-horizon rows
C1 top16 set changed on 51.85% of visible rows
mean target q_new L2: 1.6416
mean final A/B coordinate divergence: 0.5493 px
```

However, this mechanistic divergence did not produce robust capacity-matched tracking value.

Primary horizon-8 result:

```text
independent-state B2 oracle mean: 5.4788 px
shared-state capacity-2 oracle:    5.1949 px
difference:                      +0.2839 px
95% clip-block CI:               [+0.0159,+0.6744]
better / worse / equal:           14 / 14 / 16
safe16:                            41 / 42
```

All three sequence means are worse:

```text
ani:       +0.0684 px
animal3:   +0.5234 px
r4_new_f:  +0.2141 px
```

The pooled and re-entry-related summaries also fail:

```text
pooled horizons 1/4/8: +0.0734 px, CI [-0.0891,+0.3215]
early8:                 +0.3054 px, CI [+0.0351,+0.6420]
first re-entry:          +0.7384 px
```

Only three visible rows improve the shared-state B2 oracle by more than 1 px, and none occur in `r4_new_f`.

Therefore:

```text
INDEPENDENT_STATE_FAIL
```

Under the committed preregistration, the tested temporal multi-state route is closed. Do not train a branch selector, sweep alternative branch seeds, change the risk threshold, choose horizon 4 after observing results, or read DAVIS.

## 2. Frozen protocol

Repository:

```text
worktree: /gemini/code/FSPT_v9a52_clean
branch: v9a52-independent-state-branching-20260711
execution HEAD: ee17e03176c9e0b256cf555feab82b0f9be1f041
```

Frozen event protocol:

```text
9 PointOdyssey clips
8 first-native-risk events per clip
72 events total
horizons 1 / 4 / 8
216 event-horizon rows
selection fields: clip_id / query_idx / frame_tau / risk only
no GT field used for event selection
```

Primary comparison:

```text
independent-state B2 oracle:
  min(Branch A official-state final, Branch B independent-state final)

shared-state capacity-2 oracle:
  min(official final, official-state K64 frozen-score-top1 singleton refined)
```

Both sides contain two generated outputs. GT is used only after generation.

## 3. Formal artifact hashes

```text
formal script:
d400e8a8b0e713df00620fa113bda1e1bddcf2612821968dd3682811913c7104

input manifest:
61e6cfe418032b7d3466458c1aa7c1e81b59f8bb0c9ab44e5ff286a4ca0086be

formal result JSON:
929e87feb67795448cfece53d2d6d3c0eb0cc7174d30f211790e4c2eeac12d74

formal rows NPZ:
fd84754a5c5bb8192c9ce8251de2f1bd51b567d6ead6147b6d81849d453a19b5
```

Supplemental candidate-pool audit:

```text
script:
db0f8eab66f85ed1acc84f3136a72d299b9e2c7d246c56524b0daaac50c6ae17

JSON:
fae42684d9eae2b92e6a4423d3d5b60bd6a13bbd30e28368a19c86ba970c2ec9
```

## 4. Execution integrity

All mandatory gates pass:

```text
864 RGB frames individually hash verified
72 split events exactly once
216 unique event-horizon keys
288 official candidate replay rows = 72 split + 216 horizon rows
peak simultaneous active events: 6
```

Official and Branch A parity:

```text
official diagnostic p/v/q_new max_abs: 0
Branch A p/v/u/q_new/q_pre/C1/C2 max_abs: 0
Branch A q_init max_abs: 0
Branch A point_memory max_abs: 0
Branch A temporal-mask mismatch: 0
```

Replay:

```text
online risk mismatch: 0
GT visibility replay mismatch: 0
candidate score max_abs: 2.0504e-05
raw candidate error max_abs: 0
refined candidate error max_abs: 4.5538e-05
official error max_abs: 0
shared score-top1 error max_abs: 1.5259e-05
```

State isolation:

```text
within-event storage failures: 0
mutation-isolation failures: 0
cross-event storage aliases: 0
event-key mismatches: 0
first-risk mismatches: 0
all arrays finite
```

Independent recomputation from the saved NPZ reproduces:

```text
all 216 event-horizon keys
shared and independent B2 formulas
primary, pooled and early8 summaries
all clip bootstrap intervals
mechanistic counts
formal gate failure
```

## 5. Horizon profile

### Horizon 1

```text
n = 28
mean difference: +0.0054 px
better / worse / equal: 9 / 13 / 6
```

Essentially neutral and sequence-inconsistent.

### Horizon 4

```text
n = 36
mean difference: -0.1311 px
better / worse / equal: 18 / 10 / 8
```

This is the only average improvement, but it is not robust:

```text
ani:       -0.3641 px
animal3:   -0.1780 px
r4_new_f:  +0.1954 px
```

Clip signs are mixed. Because horizon 8 was preregistered as primary, horizon 4 cannot be selected post hoc as a successful route.

### Horizon 8

```text
n = 44
mean difference: +0.2839 px
95% CI: [+0.0159,+0.6744]
probability bootstrap mean < 0: 0.00481
```

This is statistically consistent with deterioration, not merely an inconclusive null.

The largest clip deterioration is:

```text
animal3:0: +1.8086 px
```

Seven of nine clip means are positive, two are slightly negative and one is exactly zero.

## 6. Re-entry review

Pooled early8 rows:

```text
n = 53
independent B2 mean: 5.1944 px
shared B2 mean:      4.8891 px
difference:         +0.3054 px
95% CI:             [+0.0351,+0.6420]
better / worse / equal: 13 / 19 / 21
```

First-reentry rows are descriptive because the preregistered power audit identified a small sample:

```text
n = 16
difference: +0.7384 px
safe16: 14 versus 15
```

Thus independent state does not solve the re-entry state-error problem under the tested deterministic branch initialization.

## 7. Mechanistic divergence versus useful value

The model states genuinely diverge:

```text
q_new divergence fraction: 100%
top16 set-change fraction: 51.85%
mean top16 overlap: 15.06 / 16
mean top64 overlap: 61.55 / 64
mean final-coordinate divergence: 0.5493 px
```

But useful final-output novelty is rare:

```text
Branch B final improves shared B2 by >1 px: 3 / 108 visible rows
ani:       1
animal3:   2
r4_new_f:  0
Branch B final <=4 px while shared B2 >4 px: 0
```

This demonstrates a critical distinction:

```text
hidden-state divergence != useful candidate-quality divergence
candidate-set difference != robust final-output improvement
```

## 8. Full candidate-pool post-audit

A conservative non-gating supplement compares the complete Branch B refined candidate pool with the complete shared-state refined candidate pool at matched K.

### Branch B K64 versus shared-state K64

```text
n = 108 visible rows
Branch B K64 oracle mean: 1.9563 px
shared K64 oracle mean:   1.9581 px
difference:              -0.0018 px
better / worse / equal:   59 / 49 / 0
95% clip-block CI:       [-0.1819,+0.2219]
```

Per sequence:

```text
ani:       -0.0882 px
animal3:   +0.1771 px
r4_new_f:  -0.1210 px
```

The independent-state K64 pool does not robustly outperform the shared-state K64 pool.

### Union K64 versus shared-state K64

```text
union oracle difference: -0.1191 px
95% CI:                 [-0.2256,-0.0335]
better / worse / equal: 59 / 0 / 49
```

This union result is an oracle upper bound and is structurally non-worsening because it uses `min(shared K64, Branch B K64)`.

Its gain is sparse:

```text
Branch B beats shared K64 by >1 px: 4 / 108
ani:       2
animal3:   0
r4_new_f:  2
Branch B <=4 px while shared K64 >4 px: 0
```

Therefore independent state sometimes contributes a slightly better candidate, but not with the sequence-consistent magnitude needed to justify another selector or full-state beam.

## 9. Why no post-hoc rescue is allowed

Do not reinterpret the following as a pass:

```text
horizon-4 mean improvement
pooled better-row count 41 > 37
top16 set changes on 51.85%
union K64 oracle CI below zero
```

Reasons:

```text
horizon 8 was the preregistered primary horizon
pooled mean and pooled CI do not pass
early8 is significantly worse
candidate-pool union uses GT oracle selection
useful >1 px candidate novelty is only four rows and misses animal3
final-output novelty is only three rows and misses r4_new_f
```

Do not sweep:

```text
alternative singleton candidate seed
second/third frozen-score branch
branch width B4/B8
risk threshold
memory length
split frame
horizon choice
candidate fusion weights
```

Any such work would be a new route requiring a new preregistration and cannot overturn V9-A5.2.

## 10. Route closure scope

The following project route is closed:

```text
native-risk event activation
frozen K64 score-top1 singleton q2 branch initialization
complete 432-query independent TrackOn2 state
independent unconditional memory rollout
capacity-matched shared-state B2 comparison
horizons 1/4/8
```

Combined with earlier results, close the broader tested temporal route family:

```text
sampled causal motion/latent self-state selection
fixed-K16 shared-state beam
risk-gated K16/K64 external history beam
history-preserving finite-window beam
independent full-model-state B2 branching
```

This is a project decision under the committed evidence. It is not a mathematical claim that every conceivable recurrent multi-hypothesis tracker must fail.

## 11. Remaining scientifically distinct direction

V9-A3.1 through V9-A4.5 closed fixed-topK ranking and local-decoder/reranking adaptation.

V9-A5.1 through V9-A5.2 closed the tested shared-state and independent-state temporal selection routes.

V9-A5C.0 still provides one unclosed upstream fact:

```text
fused recall@4:
K16 = 0.7729
K64 = 0.9287
hard rows K16 = 0.5420
hard rows K64 = 0.8561
```

Therefore the only materially different remaining mechanism is to change the fused correlation map before top-K extraction, rather than selecting among a fixed pool or branching memory state.

## 12. Next step: V9-A6.0 bounded correlation rank-shift feasibility

Do not train immediately.

First run a no-training synthetic audit that asks:

```text
How large a bounded additive perturbation to the fused C1 correlation scores is
required to promote a GT-near rank-17-to-64 candidate into top16?
```

Required analysis:

```text
minimal score increase needed to cross the K16 boundary
minimal pairwise score swaps needed
per-sequence and hard/easy distributions
risk-visible and re-entry distributions
effect of fixed epsilon budgets on K16 recall@4
collateral loss of existing K16 GT-near candidates
easy-row retention
K16 boundary tie sensitivity
```

Predeclare epsilon in normalized units from the observed natural K16 boundary-gap distribution. Do not optimize epsilon on final tracking error.

Primary feasibility gate:

```text
all three sequences improve K16 recall@4
hard-row recall gain >= 0.10
existing easy-row recall remains 1.0 or decreases by <=0.005
promoted rows > displaced-good rows by at least 2:1
required perturbation lies within a small, stable fraction of natural fused-score scale
clip-block CI for net recall gain is positive
```

If this no-training gate fails, close the correlation-adapter route and stop algorithmic expansion.

If it passes, only then preregister a bounded sequence-heldout residual correlation adapter with:

```text
frozen backbone / decoder / reranker / prediction head
strictly bounded residual added before top-K
zero-initialized identity behavior
three sequence-heldout folds
primary official-final tracking gate, not oracle-only recall
no DAVIS until all synthetic folds pass
```

## 13. Immediate execution order

```text
1. Commit V9-A5.2 formal execution, post-audit and route closure.
2. Do not reopen temporal branching or train a branch selector.
3. Create a clean V9-A6.0 worktree from the closure commit.
4. Freeze the rank-shift feasibility design and input manifests.
5. Run a saved-pool/no-training smoke before any TrackOn2 rerun.
6. Run the nine-clip rank-shift audit only after the smoke reproduces V9-A5C.0.
7. No training and no DAVIS during V9-A6.0.
```
