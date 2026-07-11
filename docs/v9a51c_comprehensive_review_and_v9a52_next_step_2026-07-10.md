# V9-A5.1c Comprehensive Review and V9-A5.2 Next Step

Date: 2026-07-10

## 1. Executive conclusion

V9-A5.1c completed successfully and is internally reproducible.

The original predeclared system-level beam-reachability gate passes:

```text
official final mean:                 9.1418 px
native_dynamic_B4 beam oracle mean:  8.7879 px
paired mean difference:             -0.3539 px
clip-block 95% CI:                  [-0.5043,-0.2343]
all three sequence means improve
all nine clip means improve
first-reentry and early8 means improve
```

The deterministic beam top1 does not pass:

```text
native_dynamic_B4 top1 mean:         9.1575 px
paired mean difference:             +0.0157 px
better / worse / equal:             2124 / 2171 / 15635
clip-block 95% CI:                  [-0.0573,+0.1110]
first-reentry difference:           +0.2967 px
early8 difference:                 +0.0733 px
```

A mandatory post-formal same-capacity review changes the interpretation:

```text
frame-local score-top4 oracle mean:  8.5363 px
temporal B4 beam oracle mean:        8.7879 px
temporal minus frame-local:         +0.2516 px
95% CI:                              [+0.1280,+0.4307]
```

The temporal B4 oracle is worse than the same-frame top4 oracle on all three sequences and all nine clips.

Therefore the defensible conclusion is:

```text
V9-A5.1c proves that the pruned beam still contains alternatives that can improve
on official final, but it does not prove that temporal history adds incremental
candidate reachability beyond a same-frame, same-capacity candidate set.
```

Do not proceed directly to a learned temporal beam readout.

## 2. Formal execution integrity

Worktree and execution base:

```text
worktree: /gemini/code/FSPT_v9a51c_clean
branch: v9a51c-history-preserving-beam-20260710
base HEAD: 29512f0d95d8b4069b73b9130e60b75d1a892b98
```

Formal hashes:

```text
script:
ee96041ce451c0639c94f7558771e6958907dc5030ea1b429228ec97ce3f4da1

result JSON:
ae6bd3eabd36aa8d64014298f48bcc24a31b0611854c8827455a96aa69a6d00b

result NPZ:
05b450fb2031ce1d762ed7436bcef34a47dfaf37c10d1c33bd8fd74c1d0381a0
```

Formal integrity:

```text
9 clips x 96 frames x 32 queries = 27,648 rows
864 RGB frames individually hash verified
official p/v/q parity max_abs = 0
V9-A5.1b row keys exact
V9-A5.1b official/risk/score/raw-error/refined-error/oracle replay max_abs = 0
all numeric outputs finite
all five policy beam widths/shapes exact
all retained last-three-grid signatures unique
beam update receives no GT/error input
official TrackOn2 memory remains on original unconditional q_new path
```

Independent NPZ recomputation reproduces:

```text
all system top1 formulas
all system oracle formulas
all raw-state minima
all signature uniqueness checks
all clip bootstrap intervals
deterministic top1 gate failure
beam reachability gate pass
```

## 3. What the original formal gate validly establishes

The B4 beam does not completely destroy the candidate-conditioned refinement headroom.

```text
full V9-A5.1b candidate oracle gain: 1.3233 px
B4 beam oracle gain:                 0.3539 px
retained fraction:                   26.74%
```

The surviving B4 beam improves official final on:

```text
3,107 visible rows
all three sequences
all nine clips
320 / 855 first-reentry rows
1,239 / 3,806 early8 rows
```

This proves that deterministic, GT-free state updates and pruning can preserve some useful refined alternatives through the full stream.

It does not prove that the temporal component caused those alternatives to be more useful than a current-frame set.

## 4. Structural oracle caveat

The formal beam oracle is defined as:

```text
non-risk: official final
risk: min(official final, surviving refined beam candidates)
```

Consequently:

```text
worse rows are structurally impossible
safe16 cannot decrease
per-clip mean differences cannot be positive
```

These conditions are useful consistency checks but are not independent evidence of temporal value.

The load-bearing formal evidence is the magnitude and distribution of improvements, not zero worse rows by itself.

A capacity-matched frame-local comparator is required to isolate incremental temporal value.

## 5. Same-capacity incremental-value audit

Supplemental artifacts:

```text
scripts/v9a51c_same_capacity_incremental_audit.py
outputs/paper_discovery_2026-07-05/v9a51c_history_beam/
  v9a51c_same_capacity_incremental_audit.json
docs/v9a51c_same_capacity_incremental_review_2026-07-10.md
```

Hashes:

```text
script:
7dca4ed2382d667c58875baf695a9ee4ff0008f2dd519f3bb75f6cbe0c7af505

JSON:
38e399cc6d74ec16e9cc3e73bbf8a7976b37febe67e5cfe2dab23d7943a8423f
```

The comparator uses:

```text
same row keys
same native risk flags
same candidate budget
same beam width
same frozen candidate scores
same candidate-conditioned refined coordinates
same official-final fallback
```

Only the source of the candidate set differs:

```text
temporal: history beam surviving B candidates
frame-local: current frozen-score top B candidates
```

### 5.1 Primary deterministic top1

```text
frame-local score top1 mean:         9.1023 px
temporal B4 top1 mean:               9.1575 px
temporal minus frame-local:         +0.0552 px
95% CI:                              [-0.1054,+0.2999]
first-reentry temporal minus local:  +0.3155 px
```

No deterministic incremental gate passes.

### 5.2 Primary B4 oracle

```text
frame-local score-top4 oracle:       8.5363 px
temporal B4 beam oracle:             8.7879 px
temporal minus frame-local:         +0.2516 px
better / worse / equal:              734 / 1889 / 17307
95% CI:                              [+0.1280,+0.4307]
first-reentry difference:           +0.5944 px
early8 difference:                 +0.4154 px
```

Per sequence temporal minus frame-local:

```text
ani:       +0.2492 px
animal3:   +0.1497 px
r4_new_f:  +0.3556 px
```

All nine clip differences are positive.

This is strong evidence that the current temporal transition/pruning removes more useful candidates than it adds.

## 6. All policy review

No predeclared diagnostic policy passes an incremental temporal-value gate.

```text
native_dynamic_B1 oracle:
  mean difference vs frame-local top1 oracle = -0.0368 px
  CI = [-0.2110,+0.1189]
  r4_new_f is worse and multiple clips are worse

native_dynamic_B4 oracle:
  +0.2516 px
  CI = [+0.1280,+0.4307]

native_dynamic_B8 oracle:
  +0.3976 px
  CI = [+0.1925,+0.6654]

fixed16_B4 oracle:
  +0.1669 px
  CI = [+0.0635,+0.3018]

fixed64_B4 oracle:
  +0.1674 px
  CI = [-0.0312,+0.3867]
```

The weak B1 negative mean is not sequence/clip consistent and its CI crosses zero.

Increasing beam width worsens the same-capacity incremental gap. The failure is not explained only by K64 expansion.

## 7. Deterministic diagnostic review

The strongest post-hoc deterministic diagnostic is fixed16_B4:

```text
mean difference vs official: -0.1134 px
CI: [-0.3290,+0.0060]
```

It still fails because:

```text
animal3 mean is slightly worse
r4_new_f safe16 decreases
first-reentry is worse
CI upper bound remains positive
```

It may not replace the predeclared primary policy after results are observed.

Fixed16_B4 also fails the same-capacity oracle comparison with a strictly positive CI, so it does not establish temporal reachability.

## 8. Effective beam diversity

The history-preserving signature mechanism works mechanically:

```text
native_dynamic_B4 unique last-three-grid signatures: 4.0 / 4
native_dynamic_B8 unique last-three-grid signatures: 8.0 / 8
```

But output diversity collapses:

```text
native_dynamic_B4, risk-visible rows:
  unique current raw grids: 2.81 / 4
  raw 4px clusters:         1.45
  refined 4px clusters:     1.13

native_dynamic_B8:
  unique current raw grids: 4.26 / 8
  raw 4px clusters:         1.89
  refined 4px clusters:     1.20
```

Thus the beam retains distinct historical signatures, but many signatures converge to nearly identical current/refined outputs.

This explains why nominal beam width does not translate into proportional candidate reachability.

## 9. Relationship to earlier route closures

V9-A3.1 through V9-A4.5 already closed:

```text
exported-summary tree/MLP/listwise selectors
source/rank factorization
DINO/PointOdyssey fixed-topK ranking
post-fusion/pre-fusion latent heads
local-decoder adaptation
conservative residual reranking adaptation
```

Therefore the next step must not be another learned selector over the same shared-state, fixed candidate pool.

A learned beam readout on the current V9-A5.1c beam is not justified because:

```text
the candidate set is weaker than frame-local same-capacity candidates
the deterministic history score does not improve selection
the beam output diversity is strongly collapsed
```

## 10. Correct route closure

Close the following route:

```text
shared official query/memory state
external raw-coordinate/descriptor history beam
finite ordinal emission+motion+identity cost
signature-only history preservation
candidate-conditioned refined readout
```

It remains useful as a negative/diagnostic baseline, but it does not justify training a temporal readout.

Do not:

```text
switch post-hoc to fixed16 or B8
sweep cost weights or window length on these nine clips
train a selector on the current beam
read DAVIS
```

## 11. Next experiment: V9-A5.2 independent model-state branching feasibility

The remaining untested mechanism is true hypothesis-specific model state.

V9-A5.0 showed that correct prior state has large causal headroom. V9-A5.1a/c used one shared official TrackOn2 query/memory stream, so every hypothesis received candidates from the same potentially drifted state.

V9-A5.2 must test whether alternative candidate-conditioned states produce genuinely different and useful future correlation maps.

### 11.1 Bounded event-based protocol

Use the same nine PointOdyssey clips, no DAVIS and no training.

For each query-track, select the first native-risk event after frame 0. Events are fixed by:

```text
sigmoid(v_logit) < 0.8 OR sigmoid(u_logit) >= 0.5
```

At the event frame create two independent model states:

```text
branch A:
  original official q_new and cloned official memory

branch B:
  candidate-conditioned q2 of the highest frozen-score alternative candidate
  whose absolute C1 grid index differs from branch A's top candidate
  and a separately cloned memory
```

No GT enters branch creation.

Roll both states forward independently for horizons:

```text
1 / 4 / 8 frames
```

Each branch uses its own point memory, temporal mask and q_new updates. Do not share q_new after the split.

### 11.2 Mandatory controls

```text
official single-state rollout
shared-state frame-local score-top2 refined readout
independent-state B2 branch oracle
independent-state deterministic branch-A/top-score readout
```

The shared-state top2 control is required to separate candidate count from state branching.

### 11.3 Required state-divergence diagnostics

```text
branch q_new cosine/L2 divergence
branch point-memory divergence
next-frame C1 map cosine/L2 divergence
candidate-set Hausdorff and overlap
refined-output divergence
fraction of events where branch B generates a GT-near candidate absent from shared-state top2
```

A branch that only changes readout but not future candidate generation does not count as independent-state evidence.

### 11.4 Integrity gates

```text
official branch parity with normal TrackOn2 <= 1e-6
branch memory tensors are physically independent
changing branch B cannot alter branch A outputs
all state/correlation/candidate arrays finite
no GT/error input to branch initialization or rollout
exact event keys and horizons persisted
frame/annotation hashes verified
```

### 11.5 Primary reachability gate

The independent-state B2 oracle must beat the shared-state frame-local top2 oracle on:

```text
all three sequence means
clip-block 95% CI upper bound < 0
horizon-8 first-reentry/early8 mean
better rows > worse rows
no sequence safe16 decrease
```

It must also show nontrivial future-state divergence:

```text
candidate sets differ on a substantial fraction of events
branch B creates useful future candidates not present in shared-state top2
```

If this gate fails, close the temporal multi-state route.

If it passes, only then consider a full-stream B2/B4 independent-state beam and later sequence-heldout readout learning.

## 12. Immediate execution order

```text
1. Commit V9-A5.1c formal artifacts unchanged.
2. Commit the same-capacity supplemental review separately in the same closure commit.
3. Update CURRENT_MAINLINE with both original formal pass and corrected incremental-value failure.
4. Create a new clean V9-A5.2 worktree from the V9-A5.1c closure commit.
5. Implement a one-clip, few-event branch-isolation smoke.
6. Verify branch-A parity and branch memory independence.
7. Run the nine-clip event-based synthetic audit only if the smoke passes.
8. Do not train and do not read DAVIS.
```
