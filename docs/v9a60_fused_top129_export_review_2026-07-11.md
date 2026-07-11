# V9-A6.0 Fused Top129 Export Review

Date: 2026-07-11

## 1. Scope

This review covers the frozen read-only top129 export used by the later no-training rank-shift audit.

It does not evaluate a residual adapter and does not make a route decision.

## 2. Smoke

A 64-row `ani:0` smoke was run before the formal export.

```text
rows: 64
top129 shape: 64 x 129
component shape: 64 x 129 x 4
```

All smoke checks passed:

```text
fused reconstruction max_abs: 0
official p/v/q parity max_abs: 0
K16/K64 error parity max_abs: 0
nearest-rank mismatch: 0
K16 boundary-gap parity max_abs: 0
official top16 set Hausdorff: 0
pool top16 ordered max_abs: 5.9605e-08
pool top16 set Hausdorff: 8.4294e-08
```

Independent smoke-array checks reproduced:

```text
top129 uniqueness and non-increasing score order
four-component weighted fusion reconstruction
coordinate-to-GT error reconstruction
K16/K64 minimum errors
native-risk formula
```

## 3. Ordered-tie gate correction

The first formal run processed all nine clips but was stopped by an overly strict diagnostic gate:

```text
pool top16 ordered max_abs: 0.18823537
pool top16 set Hausdorff:   8.4294e-08
```

V9-A5C.0 had already documented the same ordered-coordinate difference with an exactly matching candidate set. It is caused by tie ordering, not candidate-set drift.

The failed run produced no formal output. Its script and log were archived before correction.

The correction changed only integrity semantics:

```text
ordered difference: diagnostic only
candidate-set Hausdorff: hard gate
```

No model computation, row selection, score export, target definition, budget or scientific gate changed.

## 4. Formal export

```text
rows: 4,878
clips: 9
top candidates per row: 129
RGB frames hash verified: 844
NPZ size: 15,996,639 bytes
```

Per-clip rows:

```text
ani:0       1012
ani:256      714
ani:512      216
animal3:0    548
animal3:256  472
animal3:512  288
r4_new_f:0   592
r4_new_f:256 778
r4_new_f:512 258
```

## 5. Formal parity

```text
fused-map max_abs: 0
top129 component recomposition max_abs: 0
official p max_abs: 0
official v max_abs: 0
official q max_abs: 0
K16 minimum-error max_abs: 0
K64 minimum-error max_abs: 0
nearest-rank mismatch: 0
nearest-distance max_abs: 0
nearest-margin max_abs: 0
K16 boundary-gap max_abs: 0
official top16 set Hausdorff: 0
pool top16 set Hausdorff: 8.4294e-08
pool top16 ordered max_abs: 0.18823537, diagnostic only
```

## 6. Independent saved-array review

All independently checked conditions pass:

```text
4,878 source indices are exactly 0..4877
4,878 unique row keys
top_indices shape = 4878 x 129
top_scores shape = 4878 x 129
top_components shape = 4878 x 129 x 4
top_coords shape = 4878 x 129 x 2
all top129 indices unique within every row
all top129 scores non-increasing
all numeric arrays finite
component dot weight + bias reproduces fused scores <=1e-6
saved coordinates reproduce GT errors <=1e-4
saved K16/K64 minima reproduce top-error minima
saved recall flags reproduce <=4px conditions
saved native-risk reproduces the fixed rule
saved K16 gap equals score[15]-score[16]
```

Fusion parameters:

```text
c4:  -1.2983515263
c8:  +1.9004112482
c16: +1.9091821909
c32: +3.7355890274
bias: 8.8843173981
```

The exported all-row K16 boundary-gap median exactly reproduces the preregistered reference:

```text
g_ref = 0.003143310546875
```

## 7. Opportunity integrity

```text
K16 miss / K64 hit at 4px: 760
ani:       361
animal3:    98
r4_new_f:  301
```

Per clip:

```text
ani:0        236
ani:256       93
ani:512       32
animal3:0     21
animal3:256   68
animal3:512    9
r4_new_f:0   161
r4_new_f:256 116
r4_new_f:512  24
```

Additional frozen counts:

```text
GT-hard rows: 2419
opportunity rows that are GT-hard: 760
native-risk rows: 787
native-risk opportunity rows: 325
```

## 8. Hashes

```text
export script:
684e7a6243d7fccddd23bef79355356dd78ef2b40d2257d1483f790ac680ed2c

formal JSON:
72fb654381eef236c73b54d8d52d5bfbe4a7b3e4cfaf7ebcf850f1cdf8472cbb

formal NPZ:
0849bb9ab700e8f5224bf751f897aea7a38e0efd6363d6d83297e04ecbfac3ac
```

## 9. Decision

```text
EXPORT_PASS
```

The export is frozen and suitable for the preregistered no-training rank-shift audit.

The next audit must consume only the saved NPZ/JSON and must not rerun TrackOn2, alter target ranks/radius, add score budgets or inspect DAVIS.
