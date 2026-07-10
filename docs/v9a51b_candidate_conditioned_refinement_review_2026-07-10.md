# V9-A5.1b Candidate-Conditioned Refinement Review

Date: 2026-07-10

## 1. Integrity conclusion

The formal nine-clip result is reproducible and internally consistent.

Independent recomputation from the saved NPZ verifies:

```text
script SHA256 matches JSON provenance and input audit
27,648 unique query-frame keys
all saved numeric arrays finite
native risk exactly equals v_conf < 0.8 OR u_conf >= 0.5
raw/refined K16 and K64 oracles reproduce from candidate matrices
score-top1 K64 reproduces from saved frozen candidate scores
dynamic-K readouts reproduce from the risk flag
hybrid deterministic and oracle formulas reproduce exactly
re-entry first/early4/early8 and occlusion lengths reproduce from visibility
100,000-resample clip bootstrap intervals reproduce exactly
```

Hashes:

```text
script: 1a0e4a75db2f48e75fae1d3cab1b11708f77ef1ea96964a26beba545d6e129f5
JSON:   34df580628abacf02fbf45c4475f2eb72572c7753e99c295d53ce95e75d18e86
NPZ:    eb9736cbe7fc1af08e5bfe23248b333e18ae01c826fd26fc65aa40945746d266
```

## 2. What passed

The predeclared GT-only system-level viability upper bound passes:

```text
official final mean:           9.1418 px
hybrid refined oracle mean:    7.8185 px
paired mean difference:       -1.3233 px
better / worse / equal:        4267 / 0 / 15663
clip-block 95% CI:            [-2.1037, -0.7501]
```

Every clip has a negative mean difference:

```text
ani:0       -1.8085
ani:256     -3.5518
ani:512     -0.3492
animal3:0   -1.0860
animal3:256 -1.8637
animal3:512 -0.1622
r4_new_f:0  -2.2441
r4_new_f:256 -1.1136
r4_new_f:512 -0.3442
```

Per-sequence visible means:

```text
ani:       9.9477 -> 8.1128
animal3:   6.6867 -> 5.7587
r4_new_f: 10.7786 -> 9.5760
```

First re-entry also improves on every sequence:

```text
ani:      13.9209 -> 11.4408
animal3:  17.9243 -> 15.2783
r4_new_f: 21.7568 -> 19.8338
```

This proves that the frozen candidate-conditioned branch can generate alternative final coordinates with sequence-consistent and re-entry-relevant headroom.

## 3. What did not pass

The existing frozen score does not reliably select the refined alternative:

```text
hybrid refined score-top1 mean: 9.1023 px
paired mean difference:        -0.0394 px
better / worse / equal:         2085 / 2210 / 15635
clip-block 95% CI:             [-0.1990, +0.0713]
```

Per sequence:

```text
ani:       +0.0044 px worse
animal3:   +0.0897 px worse
r4_new_f:  -0.2124 px better
```

Therefore V9-A5.1b is a reachability/selection-separation result, not a deployable deterministic improvement.

## 4. Important oracle interpretation

`hybrid_refined_oracle` is defined as:

```text
non-risk rows: official final
risk rows: min(official final, refined K64 oracle)
```

GT is used after generation to select the better coordinate. Consequently:

```text
worse rows are structurally impossible
safe thresholds cannot decrease
```

The load-bearing evidence is not the zero-worse count by itself. The meaningful evidence is:

```text
all three sequence means improve
all nine clip means improve
conservative clip CI is strictly negative
first-reentry and early8 means improve
headroom appears on 4,267 visible rows
```

## 5. Candidate diversity finding

Singleton candidate conditioning strongly compresses the candidate set:

```text
mean unique refined C2 argmax locations:
K16: 2.98
K64: 3.44

mean refined 4px spatial clusters:
K16: 1.47
K64: 1.68
```

The raw fused candidate pool remains much stronger as a pure coordinate oracle:

```text
raw K64 oracle:      2.9606 px
refined K64 oracle:  7.0763 px
```

Per sequence raw -> refined K64 oracle:

```text
ani:       2.5079 -> 7.4061
animal3:   3.1498 -> 5.0775
r4_new_f:  3.2276 -> 8.7373
```

This means downstream singleton fusion should not replace the raw C1 hypothesis state. It should be treated as a candidate-conditioned readout attached to a raw candidate identity/history.

## 6. Risk audit

```text
all-row activation:        35.75%
visible activation:        21.55%
invisible activation:      75.21%
first-reentry activation:  52.75%
opportunity activation:    58.50%
conceptual mean K:         33.34
```

Risk has the desired concentration on invisible, re-entry and opportunity rows, although it misses about 41.5% of GT-defined opportunities. The threshold remains frozen; no post-hoc calibration is allowed in V9-A5.1c.

## 7. Decision boundary

V9-A5.1b passes only as:

```text
candidate-conditioned refined-coordinate reachability
```

It does not establish:

```text
frozen score selection
beam path scoring
candidate diversity after refinement
deployable system improvement
```

## 8. V9-A5.1c implications

The next deterministic beam must:

```text
maintain raw C1 candidate identity, raw coordinate and candidate descriptor as state
attach candidate-conditioned refined coordinate as the output/readout coordinate
preserve distinct incoming histories even if refined coordinates collapse
use native-risk K16/K64 without threshold tuning
update state every frame
use a finite eight-frame cost window instead of infinite cumulative cost
retain official final as the non-risk output
compare system-level deterministic output against official final
report GT-only min(official final, surviving refined beam states) separately
```

The beam may not deduplicate solely by refined coordinate or current candidate index. A state signature must preserve at least the last three raw candidate identities so constant-velocity-relevant histories remain distinct.

## 9. Next gate

V9-A5.1c should have two separate outcomes:

```text
Deterministic top1 gate:
  risk-gated beam readout beats official final on all sequence/CI/re-entry gates.

Beam reachability gate:
  deterministic top1 fails, but GT-only hybrid beam oracle remains sequence-consistent,
  clip-CI negative and re-entry-positive relative to official final.
```

Only the first supports a deterministic non-trained method. The second supports a later sequence-heldout learned readout, but not DAVIS and not a claim of deployment readiness.
