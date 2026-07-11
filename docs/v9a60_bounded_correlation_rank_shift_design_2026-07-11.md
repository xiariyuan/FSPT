# V9-A6.0 Bounded Correlation Rank-Shift Feasibility Design

Date: 2026-07-11

## 1. Purpose

V9-A5C.0 established upstream fused-correlation candidate headroom:

```text
fused recall@4, K16: 0.7729
fused recall@4, K64: 0.9287
hard-row recall@4, K16: 0.5420
hard-row recall@4, K64: 0.8561
```

V9-A3.1 through V9-A4.5 closed fixed-topK candidate ranking and reranking/local-decoder adaptation.

V9-A5.1 through V9-A5.2 closed the tested shared-state and independent-state temporal selection routes.

The remaining distinct question is whether the fused C1 score map is close enough to a useful top16 ordering that a strictly bounded residual before top-K could plausibly recover a meaningful part of the K16-to-K64 headroom.

V9-A6.0 is a no-training necessary-condition audit. It does not train or evaluate an adapter.

## 2. Repository isolation

```text
worktree: /gemini/code/FSPT_v9a60_clean
branch: v9a60-correlation-rank-shift-20260711
base HEAD: bb879183ffff03129cb652824c286f56f9201804
```

No DAVIS file or label may be read.

## 3. Important limitation

A GT-directed rank-shift oracle can choose which candidate to promote and can apply zero residual on already-good rows.

Therefore it cannot measure:

```text
false promotions on easy rows
collateral displacement under a learned residual
whether a deployable model can identify the target candidate
final TrackOn2 coordinate quality after reranking/C2/prediction head
```

V9-A6.0 measures only the score magnitude required to promote an already-existing useful rank-17-to-64 candidate into top16.

A pass is necessary but not sufficient. It authorizes only a separately preregistered sequence-heldout V9-A6.1 adapter experiment.

A V9-A6.0 pass must not be described as adapter success.

## 4. Existing artifacts are insufficient for the formal audit

The committed V9-A5C.0 NPZ contains:

```text
K-wise minimum errors and recall
nearest spatial grid rank/score margin
top1 score
K-boundary gaps
```

It does not contain:

```text
top16 threshold score per row
the highest-scoring <=4px candidate in ranks 17-64
its exact fused score and rank
top16 candidate errors
top65/top129 candidate identities and component scores
```

`nearest_rank` is the rank of the spatially nearest grid cell. It is not necessarily the rank of the highest-scoring <=4px candidate. On many K16-miss/K64-hit rows these differ.

Therefore V9-A6.0 must first perform a new hash-verified read-only export. No conclusion may be inferred by subtracting the existing `nearest_margin` or K16 boundary gap.

## 5. Frozen source rows

Use exactly the 4,878 committed V9-A5C.0 rows.

The committed `selected_indices` are exactly `0..4877` and row keys match the V9-A3.8 PointOdyssey pool.

Expected prior opportunity counts, reproduced before implementation:

```text
K16 miss and K64 hit at 4px: 760 / 4878
ani:       361
animal3:    98
r4_new_f:  301
```

These counts are integrity expectations, not tunable filters.

## 6. Hard and risk semantics

The V9-A3.8 `is_hard` field is:

```text
old_error_px > 4
```

It uses GT and is allowed only for stratified reporting.

Deployable native risk remains fixed as:

```text
visibility_conf < 0.8 OR uncertainty_sigmoid >= 0.5
```

Prior coverage of the 760 opportunity rows is:

```text
all sequences: 325 / 760 = 42.76%
ani:           252 / 361 = 69.81%
animal3:        46 /  98 = 46.94%
r4_new_f:       27 / 301 =  8.97%
```

Because native-risk opportunity recall is extremely sequence-dependent, especially on `r4_new_f`, the primary magnitude audit is global. Native-risk-gated conversion is a deployment-ceiling diagnostic only and cannot be the primary gate.

No risk threshold sweep is allowed.

## 7. Read-only top129 export

For every row, recompute the official frozen TrackOn2 C1 fused map with full parity to V9-A5C.0.

Save:

```text
row metadata and exact row keys
GT yx coordinate and visibility metadata
native-risk, GT-hard, reentry-first and reentry-early8 flags
full fused-map mean/std/min/max
top129 fused grid indices and scores
top129 grid coordinates and GT errors
top129 c4/c8/c16/c32 component values
top16/top17/top64/top65 scores
K16 and K64 min error / recall@4
```

Top129 is an export/debug margin. The formal target search remains restricted to original fused ranks 17-64.

Storage may use:

```text
indices: int32
scores/components/errors: float32
```

Do not quantize to float16 because the natural K16 boundary gaps are small.

## 8. Export integrity

Mandatory gates:

```text
4,878 rows exported exactly once
row keys and selected indices exactly match V9-A5C.0
all 864 RGB frames hash verified for the formal run
official p/v/q parity <= 1e-6
recomputed fused map parity <= 1e-6
recomputed K16/K64 min-error parity <= 1e-4
recomputed K16/K64 recall@4 exact
recomputed nearest-rank/distance/margin parity within fixed tolerance
recomputed K16 boundary gap parity <= 1e-6
top129 indices unique and scores non-increasing
all exported arrays finite
TrackOn code imported only from the clean V9-A6.0 worktree
```

The formal rank-shift audit must consume only the saved export; it must not rerun TrackOn2.

## 9. Opportunity target definition

A formal opportunity row satisfies:

```text
original fused K16 min error > 4 px
original fused K64 min error <= 4 px
```

Within original ranks 17-64, define the target as:

```text
the highest-scoring candidate with GT error <= 4 px
```

Because candidates are score-sorted, this is the first <=4px candidate in ranks 17-64.

Store:

```text
target rank
target score
target GT error
rank16 score
rank16 GT error
rank16_score - target_score
```

No target outside the original top64 may be used.

## 10. Rank-shift magnitude definitions

Let:

```text
s16 = original rank-16 fused score
st  = target fused score
span = max(0, s16 - st)
```

### 10.1 Positive target-only lower bound

A one-sided oracle raises only the target score.

Required positive residual:

```text
positive_epsilon = nextafter(s16, +infinity) - st
```

This is an optimistic lower bound because it knows the correct target candidate.

### 10.2 Signed pairwise L-infinity lower bound

A signed oracle may raise the target by `epsilon` and lower the current rank-16 boundary candidate by `epsilon`.

Required per-cell L-infinity bound:

```text
signed_epsilon = span / 2
```

This is more optimistic and is secondary only.

### 10.3 Normalization

Report:

```text
span / full fused-map std
positive_epsilon / fused-map std
signed_epsilon / fused-map std
span / g_ref
```

where the frozen natural reference is the committed all-row fused K16 boundary-gap median:

```text
g_ref = 0.003143310546875
```

## 11. Frozen score-span budgets

Use only the following score-span budgets:

```text
0.5 * g_ref = 0.0015716552734375
1.0 * g_ref = 0.003143310546875
2.0 * g_ref = 0.00628662109375
4.0 * g_ref = 0.0125732421875
8.0 * g_ref = 0.025146484375
```

The primary budget is:

```text
2.0 * g_ref = 0.00628662109375
```

No epsilon may be added after inspecting results.

For positive-only implementation, score span and positive epsilon are effectively the same apart from strict tie handling.

For the signed pairwise lower bound, a score span of `2*g_ref` corresponds to per-cell `epsilon <= g_ref`.

## 12. Conversion metrics

For each budget, report:

```text
fraction of the 760 opportunity rows whose required span is within budget
promoted row count
implied global K16 recall@4 gain = promoted / 4878
implied GT-hard recall@4 gain = promoted / number of GT-hard rows
per-sequence conversion
per-clip conversion
native-risk-only promoted count and implied gain
reentry-first / early8 conversion
rank-bin conversion for 17-24 / 25-32 / 33-48 / 49-64
```

The implied gains are oracle upper bounds with no collateral model. Do not call them realized recall gains.

## 13. Statistical protocol

Primary unit:

```text
opportunity row
```

Use 100,000-resample clip-block bootstrap:

```text
cluster: clip_id
seed: 20260718
```

At the primary budget, bootstrap:

```text
opportunity conversion fraction
implied global recall gain
implied GT-hard recall gain
```

Report leave-one-clip-out conversion and all nine clip rates.

## 14. Primary necessary-condition gate

At score span `<= 2*g_ref`, require all of:

```text
global opportunity conversion >= 0.50
ani opportunity conversion >= 0.40
animal3 opportunity conversion >= 0.40
r4_new_f opportunity conversion >= 0.40
at least 7 of 9 clips have conversion >= 0.30
clip-block 95% CI lower bound for conversion > 0.35
implied global recall@4 gain >= 0.075
implied GT-hard recall@4 gain >= 0.12
```

Magnitude stability additionally requires:

```text
median span/fused_std <= 0.02 on every sequence
90th percentile span/fused_std <= 0.10 globally
```

These thresholds are fixed before the top129 export is generated.

## 15. Secondary signed lower-bound report

At per-cell signed `epsilon <= g_ref`, report the same conversion metrics.

Because this is mathematically equivalent to `span <= 2*g_ref`, it does not provide an independent pass. It only translates the primary span into a signed L-infinity interpretation.

## 16. Non-gating component diagnostics

For every opportunity row, compare the target candidate and rank-16 boundary candidate component values:

```text
delta c4
delta c8
delta c16
delta c32
current weighted contribution by component
```

Report sign consistency and medians by sequence.

Do not choose a component or fusion-weight perturbation after seeing these results. Any weight-rebalancing or learned residual is V9-A6.1 and requires a new preregistration.

## 17. Interpretation

### Pass

```text
RANK_SHIFT_MAGNITUDE_PASS
```

Meaning:

```text
a meaningful and sequence-consistent fraction of known K16-to-K64 opportunity rows
can cross the K16 boundary under a small target-directed score span.
```

A pass authorizes only a new V9-A6.1 sequence-heldout bounded residual adapter design.

V9-A6.1 must independently evaluate:

```text
GT-free target identification
collateral displacement and easy-row retention
three sequence-heldout folds
final official TrackOn2 coordinate metrics
identity initialization and strict residual bound
```

### Fail

```text
RANK_SHIFT_MAGNITUDE_FAIL
```

Meaning:

```text
even an optimistic target-directed oracle requires score changes that are too
large or too sequence/clip inconsistent for the committed bounded-residual route.
```

Failure closes the bounded correlation residual route. Do not train or read DAVIS.

## 18. Prohibited post-hoc actions

Do not:

```text
change the 4px target radius
use candidates beyond original rank64
select a different target than the highest-scoring <=4px rank17-64 candidate
change g_ref or add epsilon budgets
replace the global gate with native-risk after seeing results
use GT-hard as deployment activation
claim easy-row safety from the target-only oracle
train a selector or adapter during V9-A6.0
read DAVIS
```

## 19. Immediate execution order

```text
1. Commit this design and input manifest before exporter implementation.
2. Implement a one-clip/top129 export smoke.
3. Verify all V9-A5C.0 parity and top129 ordering gates.
4. Run the formal 4,878-row hash-verified export.
5. Freeze the export hash.
6. Run the saved-export rank-shift audit with no TrackOn2 execution.
7. Independently recompute gates from the saved audit NPZ/JSON.
8. Do not train and do not read DAVIS.
```
