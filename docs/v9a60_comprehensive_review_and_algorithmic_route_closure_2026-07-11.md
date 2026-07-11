# V9-A6.0 Comprehensive Review and Algorithmic Route Closure

Date: 2026-07-11

## 1. Executive conclusion

V9-A6.0 completed the preregistered no-training bounded correlation rank-shift audit.

The audit used an optimistic GT-directed oracle:

```text
source rows: 4,878
K16-miss / K64-hit opportunities: 760
target: highest-scoring <=4px candidate in original ranks 17-64
primary score-span budget: 2 * g_ref = 0.00628662109375
g_ref: committed fused K16 boundary-gap median = 0.003143310546875
```

At the primary budget:

```text
promoted opportunities: 67 / 760
opportunity conversion: 8.82%
implied global recall@4 upper-bound gain: +0.0137
implied GT-hard recall@4 upper-bound gain: +0.0277
```

Sequence conversion:

```text
ani:        31 / 361 =  8.59%
animal3:    22 /  98 = 22.45%
r4_new_f:   14 / 301 =  4.65%
```

Clip-block 95% CI for conversion:

```text
[0.0549, 0.1421]
```

No clip reaches the preregistered 30% minimum.

Every primary gate fails.

Therefore:

```text
RANK_SHIFT_MAGNITUDE_FAIL
```

Because this failure occurs under a GT-aware, target-specific, no-collateral oracle, it is strong evidence against the committed small bounded correlation-residual route.

Do not train a V9-A6.1 residual adapter and do not read DAVIS.

## 2. Frozen evidence chain

Repository:

```text
worktree: /gemini/code/FSPT_v9a60_clean
branch: v9a60-correlation-rank-shift-20260711
preregistration commit: db4f79d
export commit: 7a8dc06
```

Preregistered artifacts:

```text
docs/v9a60_bounded_correlation_rank_shift_design_2026-07-11.md
docs/v9a60_input_manifest_2026-07-11.json
```

Frozen export:

```text
scripts/v9a60_export_fused_top129.py
docs/v9a60_fused_top129_export_result_2026-07-11.md
docs/v9a60_fused_top129_export_review_2026-07-11.md
outputs/paper_discovery_2026-07-05/v9a60_rank_shift/v9a60_fused_top129_export.json
outputs/paper_discovery_2026-07-05/v9a60_rank_shift/v9a60_fused_top129_export.npz
```

Formal rank-shift artifacts:

```text
scripts/v9a60_bounded_rank_shift_audit.py
docs/v9a60_bounded_rank_shift_audit_result_2026-07-11.md
outputs/paper_discovery_2026-07-05/v9a60_rank_shift/v9a60_bounded_rank_shift_audit.json
outputs/paper_discovery_2026-07-05/v9a60_rank_shift/v9a60_bounded_rank_shift_audit_rows.npz
```

## 3. Formal hashes

```text
rank-shift script:
61dd1ea369353a698a683212dcb2c45d6ba0eb9b1d7a603e6fdbc270eec538c6

rank-shift JSON:
a541d4475ca647a798827ac05a5befff588d33ceccfefb1229b85dff3a4f9370

rank-shift NPZ:
24c3b49d85635aa6879ef084630422ccf63f145951eeae808d72a24d58da30a6

rank-shift result document:
5e25e3885f2dde37e49d645ce8a564d5a2da7c13fd713427b0544810967d763d
```

Frozen export hashes:

```text
export script:
684e7a6243d7fccddd23bef79355356dd78ef2b40d2257d1483f790ac680ed2c

export JSON:
72fb654381eef236c73b54d8d52d5bfbe4a7b3e4cfaf7ebcf850f1cdf8472cbb

export NPZ:
0849bb9ab700e8f5224bf751f897aea7a38e0efd6363d6d83297e04ecbfac3ac
```

## 4. Export integrity

The formal top129 export passed:

```text
4,878 rows
9 clips
844 RGB frames individually hash verified
top129 fused scores/indices/coordinates/errors
four upsampled scale components per candidate
```

Parity:

```text
fused-map max_abs: 0
top129 component recomposition max_abs: 0
official p/v/q max_abs: 0
K16/K64 minimum-error max_abs: 0
nearest-rank mismatch: 0
nearest-distance/margin max_abs: 0
K16 boundary-gap max_abs: 0
official top16 set Hausdorff: 0
pool top16 set Hausdorff: 8.4294e-08
```

The pool top16 ordered-coordinate maximum difference is `0.18823537`, but this reproduces the previously documented tie-order effect. The candidate set is unchanged.

Independent saved-array checks reproduced:

```text
all 4,878 row keys
all top129 indices unique and score-sorted
four-component fusion scores
GT coordinate errors
K16/K64 minimum errors and recall flags
native-risk formula
K16 boundary gap
760 opportunity rows and sequence/clip counts
```

## 5. Why the target definition is correct

An opportunity row satisfies:

```text
original K16 min error > 4px
original K64 min error <= 4px
```

The target is the first, therefore highest-scoring, candidate in original ranks 17-64 whose error is <=4px.

Formal target integrity:

```text
760 / 760 rows have a valid target
all target ranks are 17-64
all target errors are <=4px
all rank16 boundary errors are >4px
no earlier rank17-to-target candidate is <=4px
all score spans are nonnegative
```

This corrects the earlier possible confusion between:

```text
rank of the spatially nearest grid cell
versus
rank of the highest-scoring <=4px candidate
```

They are not generally the same.

## 6. Perturbation definitions

For each opportunity:

```text
span = rank16 score - target score
positive epsilon = nextafter(rank16 score,+inf) - target score
signed pairwise per-cell epsilon = span / 2
```

The formal component decomposition exactly reproduces the score span:

```text
sum((target component - boundary component) * fusion weight) = -span
max_abs reconstruction error <= 1e-5
```

Positive strict conversion counts are identical to span conversion counts at every frozen budget, so tie handling does not affect the result.

## 7. Frozen-budget results

| Span budget | Opportunities promoted | Conversion | Implied global gain | Implied GT-hard gain |
|---:|---:|---:|---:|---:|
| 0.5 x g_ref | 14 | 1.84% | +0.0029 | +0.0058 |
| 1 x g_ref | 38 | 5.00% | +0.0078 | +0.0157 |
| 2 x g_ref | 67 | 8.82% | +0.0137 | +0.0277 |
| 4 x g_ref | 126 | 16.58% | +0.0258 | +0.0521 |
| 8 x g_ref | 264 | 34.74% | +0.0541 | +0.1091 |

Even at `8*g_ref`, four times the primary budget:

```text
global conversion: 34.74%
ani:       36.01%
animal3:   52.04%
r4_new_f:  27.57%
clips with conversion >=30%: 5 / 9
implied global gain: +0.0541
implied hard gain: +0.1091
```

This still fails the preregistered meaningfulness thresholds.

## 8. Primary clip consistency

Primary `2*g_ref` clip conversion:

```text
ani:0         8.90%
ani:256       9.68%
ani:512       3.12%
animal3:0    28.57%
animal3:256  22.06%
animal3:512  11.11%
r4_new_f:0    4.35%
r4_new_f:256  2.59%
r4_new_f:512 16.67%
```

No clip reaches 30%.

Bootstrap:

```text
conversion 95% CI: [0.0549,0.1421]
global-gain 95% CI: [0.0080,0.0194]
hard-gain 95% CI: [0.0162,0.0390]
P(conversion >= 0.50): 0
```

Leave-one-clip-out conversion remains approximately 7.5%-10.0%, so the failure is not caused by a single adverse clip.

## 9. Required score magnitude

Score span distribution:

```text
median: 0.03431
p90:    0.07756
p95:    0.09479
max:    0.19609
```

In units of the natural K16 boundary reference:

```text
median span / g_ref ~= 10.92
```

Normalized by each row's full fused-map standard deviation:

```text
global median: 0.1265
global p90:    0.2744
```

Per-sequence median span/fused_std:

```text
ani:       0.1426
animal3:   0.0815
r4_new_f:  0.1235
```

All are far above the preregistered `0.02` median bound.

Global p90 `0.2744` is far above the preregistered `0.10` bound.

Thus most opportunities are not just below the K16 boundary by one natural boundary gap. They are separated by a materially larger score span.

## 10. Target-rank concentration

Opportunity targets:

```text
rank 17-24: 213
rank 25-32: 150
rank 33-48: 252
rank 49-64: 145
```

At the primary budget, all 67 promoted rows come from ranks 17-24.

```text
rank 17-24 conversion: 31.46%
rank 25-32 conversion: 0
rank 33-48 conversion: 0
rank 49-64 conversion: 0
```

At `8*g_ref`:

```text
rank 17-24 conversion: 84.04%
rank 25-32 conversion: 44.00%
rank 33-48 conversion:  6.75%
rank 49-64 conversion:  1.38%
```

The majority of deeper K64 headroom cannot be recovered by a small local rank-boundary adjustment.

## 11. Native-risk and re-entry review

Primary native-risk result:

```text
risk opportunities: 325
risk opportunities promoted: 36
conversion within risk opportunities: 11.08%
implied global gain: +0.0074
```

Native risk already covers only 42.76% of all opportunities and only 8.97% on `r4_new_f`. It cannot rescue the global result.

Re-entry:

```text
first-reentry opportunities: 28
promoted: 3
conversion: 10.71%

early8 opportunities: 146
promoted: 15
conversion: 10.27%
```

There is no special small-span concentration in re-entry rows.

## 12. Component diagnostics

Current fused weights:

```text
c4  = -1.29835
c8  = +1.90041
c16 = +1.90918
c32 = +3.73559
```

Global target-minus-boundary medians:

```text
c4:  +0.00047, positive on 51.2%; weighted median -0.00061
c8:  -0.01068, positive on 22.2%; weighted median -0.02029
c16: -0.00346, positive on 36.8%; weighted median -0.00660
c32: -0.00098, positive on 35.7%; weighted median -0.00366
```

No component consistently favors the target across all sequences.

Examples of sequence inconsistency:

```text
c16 median delta:
ani:       +0.00113
animal3:   -0.00495
r4_new_f:  -0.01193

c4 median delta:
ani:       +0.00618
animal3:   +0.00053
r4_new_f:  -0.00787
```

This does not support a simple post-hoc global fusion-weight change. Component diagnostics are non-gating and do not authorize selecting a scale after observing results.

## 13. Why the failure is decisive for this route

V9-A6.0 is intentionally optimistic:

```text
GT identifies the correct candidate
no false target selection
no residual is applied to already-good rows
no easy-row damage is charged
no final reranking/C2/prediction-head compression is charged
```

A real adapter would have to solve all omitted problems.

Despite these advantages, the oracle fails every primary gate by a large margin.

Therefore a learned bounded residual adapter is not justified under the committed small-span hypothesis.

Increasing epsilon until many rows cross would change the hypothesis from a small identity-preserving residual to a large score-map rewrite. That is a new architecture, not a rescue of the bounded route.

## 14. Formal gate outcome

```text
global conversion >=0.50: false
all sequence conversions >=0.40: false
at least 7/9 clips >=0.30: false, actual 0/9
bootstrap lower bound >0.35: false, actual 0.0549
implied global gain >=0.075: false, actual 0.0137
implied hard gain >=0.12: false, actual 0.0277
all sequence median span/std <=0.02: false
p90 span/std <=0.10: false, actual 0.2744
```

Decision:

```text
RANK_SHIFT_MAGNITUDE_FAIL
```

## 15. Route closure scope

Close the following tested route:

```text
existing frozen TrackOn2 fused C1 map
original rank17-64 <=4px candidates
strictly bounded small residual before top16 extraction
identity-preserving magnitude on the order of natural K16 boundary gaps
```

Combined with prior evidence, the following algorithmic rescue families are now closed:

```text
fixed-topK ranking heads and reranking adaptation
candidate-conditioned singleton readout as deterministic selector
shared-state temporal beams
history-preserving external beams
independent full-model-state branching
small bounded fused-correlation residual promotion
```

This is a project evidence closure, not a universal theorem about every possible tracker or a substantially redesigned correlation architecture.

## 16. Prohibited post-hoc rescue

Do not:

```text
add epsilon budgets above 8*g_ref and relabel the route small/bounded
change the 4px target radius
use candidates below original rank64
select a different target candidate
choose a favorable component after seeing diagnostics
fit global fusion weights on these 760 GT-labeled opportunities
train V9-A6.1
switch primary reporting to animal3
use GT-hard as deployment activation
read DAVIS
```

Any unbounded map rewrite, new candidate generator or new architecture requires a new independent research hypothesis and cannot overturn V9-A6.0.

## 17. Project-level next step

Stop algorithmic expansion on the current rescue tree.

Return to the stable project route already recorded in `CURRENT_MAINLINE.md`:

```text
Route A diagnostic paper: active / stable
Route B SOTA-compatible rescue: exhausted under the current hypothesis tree
Route C old student / pseudo-label / DINO-local routes: closed
```

Immediate work should be reproducibility and paper consolidation:

```text
1. Freeze the V9-A evidence chain and branch hashes.
2. Build a compact route-closure matrix: hypothesis, protocol, gate, result, decision.
3. Generate manuscript-ready tables/figures only from committed JSON/NPZ.
4. Update Discussion and Limitations with the distinction between oracle headroom and deployable selection.
5. Keep TrackOn2/DAVIS claims protocol-caveated and non-leaderboard.
6. Run package/import/compile/hash reproducibility checks from a clean checkout.
7. Do not start another model experiment without a genuinely new written architectural hypothesis.
```

The most scientifically useful conclusion is now negative but precise:

```text
Substantial K64 oracle recall exists, but it is neither reachable by the tested
selection/state mechanisms nor close enough to the K16 boundary for a small
identity-preserving correlation residual to recover at meaningful scale.
```
