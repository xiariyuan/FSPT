# CoTracker3 Online V5 Posthoc Review and Next Step — 2026-07-06

## What was rechecked

After V5-C failed to train a useful lightweight candidate verifier, we audited whether V5-B selective oracle gain actually came from coordinate candidate replacement or from oracle visibility opening.

Artifact:

```text
outputs/paper_discovery_2026-07-05/cotracker3_online_v5b_hard_candidate_pool_oracle/oracle_component_ablation/oracle_component_ablation_summary.json
```

## Critical finding

V5-B selective oracle was not a pure coordinate-selection oracle. It contained two oracle operations:

```text
1. Open hard GT-visible event frames as visible.
2. Replace coordinates only when the top-k candidate is closer than native.
```

This matters because V5-C trained only a coordinate candidate verifier, while much of the observed headroom came from visibility/opening decisions.

## Component ablation

Native first10:

```text
AJ      71.4896
OA      92.1433
delta   83.0547
d4px    90.5001
AJ_RD   0.4168
AJ_RD_256 0.5975
```

### 1. Hard visibility-only oracle

Open all 40 hard GT-visible event frames, no coordinate replacement.

```text
opened visibility: 37/40
AJ      -0.0324
OA      +0.2038
AJ_RD   +0.0027
AJ_RD_256 +0.0061
```

### 2. Coordinate selective only, keep native visibility

Use GT to accept better local96_s8 top20 candidate coordinates, but keep native visibility unchanged.

```text
coord accepted: 28/40
AJ      +0.0119
OA      +0.0000
AJ_RD   +0.0004
AJ_RD_256 +0.0004
```

### 3. Coordinate selective + open only accepted candidates

Accept better coordinates and open visibility only for accepted candidate events.

```text
coord accepted: 28/40
opened visibility: 25/40
AJ      +0.0622
OA      +0.1339
AJ_RD   +0.0031
AJ_RD_256 +0.0068
```

### 4. Coordinate selective + open all hard events

This matches the earlier V5-B selective oracle.

```text
coord accepted: 28/40
opened visibility: 37/40
AJ      +0.0856
OA      +0.2038
AJ_RD   +0.0050
AJ_RD_256 +0.0106
```

## Interpretation

The main V5-B headroom is not coordinate replacement alone.

```text
visibility/opening oracle contributes most of AJ_RD/AJ_RD_256 gain.
coordinate replacement alone contributes little to AJ_RD (+0.0004).
```

This explains V5-C failure:

```text
V5-C asked a lightweight verifier to select better coordinates, but the current oracle headroom is primarily a visibility/opening decision problem.
```

## Review of V5-C

V5-C dataset:

```text
800 candidates
34 positive coordinate-good samples
positive rate 4.25%
```

LOOV verifier results:

```text
Logistic AP 0.0500, AUC 0.4773
ExtraTrees AP 0.0449, AUC 0.5122
RandomForest AP 0.0442, AUC 0.5129
```

Univariate audit found no robust single feature:

```text
best single feature AP only around 0.09
DINO rank is not strongly predictive
positive candidates are sparse and spread across ranks
```

## Correct next step

Do not continue coordinate-candidate verifier training in the current form.

The next step should be:

```text
V5-D: hard-event visibility/opening verifier
```

Not:

```text
more local96/global top-k coordinate verifier tuning
```

## V5-D design

Data unit:

```text
one hard re-entry event, not one candidate point
```

Target:

```text
Should we open native visibility at this hard event frame?
```

Candidate coordinate replacement becomes optional secondary output.

Labels:

```text
positive: GT visible and opening native visibility improves or does not harm AJ_RD-style event utility
negative: GT occluded or opening would be harmful
```

Features must be GT-free:

```text
native visibility
native score
score trend around event
support age
occlusion length / invisible run length
native coordinate stability
DINO best/top-k similarity statistics
candidate pool recall proxies without GT, such as score gap, entropy, top1-top20 spread
agreement between native coordinate and top candidate
```

Validation:

```text
video-level split
separate visibility-opening decision from coordinate replacement decision
```

Success gate:

```text
recover at least half of visibility-only oracle: AJ_RD >= +0.0013 and AJ_RD_256 >= +0.003
without AJ/OA collapse
```

## Decision

Continue CoTracker3 online only as V5-D visibility/opening verifier diagnostic.

If V5-D fails, stop CoTracker3 online as a mainline and keep the complete chain as appendix/limitation:

```text
state writeback feasible but limited;
appearance gate improves precision but kills recall;
hard-event candidate pools have moderate oracle headroom;
coordinate verifier cannot recover it;
headroom is mainly visibility-opening, so event-level visibility verifier is the final reasonable diagnostic.
```
