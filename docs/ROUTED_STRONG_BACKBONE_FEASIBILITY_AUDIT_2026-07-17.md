# Route-D Strong-Backbone Feasibility Audit

Date: 2026-07-17
Status: repository-and-artifact audit complete; no new external benchmark opened
Decision: proceed first with CoTracker3 online true-streaming interface/parity audit

## 1. Research decision

The corrected Route-D result is statistically strong relative to its independent MMP baseline, but the absolute system level is not competitive enough for a confident CCF-B / CAS Zone 2 submission:

```text
MMP baseline AJ:             32.49
Route-D closed-loop AJ:      34.80
absolute gain:               +2.30 AJ points
relative gain:               approximately +7.09%
```

The gain is real and mechanism-relevant. The remaining publication risk is whether the gain survives on a modern strong tracker whose native performance is close to its public reference.

The next mainline question is therefore:

> Does Route-D provide an independent, statistically significant, low-risk gain when attached to a strong modern point-tracking backbone?

The completed 1,144-video Kinetics result is frozen external evidence. It must not be used to select a new controller, threshold, candidate source, fusion rule, or stability guard.

---

## 2. Audit method and evidence boundary

This document audits only assets already present in the repository and experiment tree:

```text
- local source trees;
- local checkpoints;
- existing parity reports;
- existing candidate/cache exports;
- prior state-writeback smokes;
- coordinate, visibility, query, and raster contracts;
- known failure records.
```

No new large evaluation was run. In particular, the corrected full-1,144 TAP-Vid-Kinetics evaluation was not rerun.

A backbone is considered feasible only if it can support all of the following without protocol shortcuts:

```text
1. a causal local candidate;
2. at least one distinct global or alternative candidate;
3. prediction-only candidate quality features;
4. an independently instantiated native baseline;
5. deterministic replay under one pinned protocol;
6. open-loop output replacement;
7. a clearly isolated closed-loop state writeback;
8. Kubric-only controller fitting and calibration;
9. frozen external evaluation after preregistration.
```

---

## 3. Ranked candidates

| Rank | Backbone | Local code | Local checkpoint | Existing baseline parity | Candidate exposure | Closed-loop state access | Engineering risk | Decision |
|---:|---|---|---|---|---|---|---|---|
| 1 | CoTracker3 online true-streaming | yes | yes | strong | medium; instrumentation required | strong and already audited | medium | **proceed first** |
| 2 | TAPNext++ / TAPNext family | yes | yes | strong first/input cache | medium-to-hard | recurrent-state integration not yet isolated | medium-high | second candidate |
| 3 | Track-On2 DINOv3 | yes | yes | strong, repo-native parity | rich internal features but contract is complex | memory/state path accessible but invasive | high | fallback / second-backbone candidate |
| 4 | LocoTrack | wrapper code yes | no local checkpoint found | absent | plausible matching features | no clean online state path established | high | blocked pending asset/parity |
| 5 | CoWTracker | no usable local implementation found | no | absent | unknown | unknown | very high | stop for current cycle |

The ranking is based on implementation validity, not only public benchmark strength.

---

## 4. Rank 1 — CoTracker3 online true-streaming

### 4.1 Assets and parity status

Available locally:

```text
baselines/cotracker/
baselines/cotracker/checkpoints/scaled_online.pth
baselines/cotracker/checkpoints/scaled_offline.pth
```

Existing first/input parity evidence for the online checkpoint:

```text
DAVIS AJ:          64.89
Delta average:     77.36
OA:                91.80
repo-native diff:  effectively zero under the validated bridge
```

Existing true-streaming full-30 diagnostic:

```text
AJ:                65.2366
Delta average:     77.9458
OA:                90.8186
```

Important protocol distinction:

```text
CoTracker3 online-architecture full-sequence inference
is not identical to
CoTracker3 true-streaming window-by-window inference.
```

All Route-D closed-loop experiments must use the true-streaming native rerun as their baseline. They must not compare state-writeback output against the older full-sequence cache.

### 4.2 Candidate interface

Proposed causal candidate contract:

```text
local candidate:
  current native CoTracker3 coordinate for the active query/frame

global/alternative candidates:
  top-K coordinates exported from a pinned correlation-pyramid/global-search probe,
  or a separately pinned causal candidate provider aligned to the current frame

candidate quality:
  native visibility logit/probability
  native confidence logit/probability
  correlation peak
  correlation entropy or peak ratio
  local/global displacement
  agreement across correlation levels
  agreement across deterministic candidate probes

prior state:
  online_coords_predicted
  online_vis_predicted
  online_conf_predicted
  online_ind and overlap-window indices
```

The native model does not expose a ready-made semantic “global candidate list” at its public predictor boundary. Candidate extraction therefore must be an explicit, tested instrumentation layer rather than an assumed interface.

### 4.3 Closed-loop state that can be modified

The local code maintains:

```text
online_track_feat
online_track_support
online_coords_predicted
online_vis_predicted
online_conf_predicted
online_ind
```

The next window copies previous predictions into:

```text
coords_prev -> coords_init
vis_prev    -> vis_init
conf_prev   -> conf_init
```

The minimum defensible Route-D closed-loop writeback is:

```text
accepted coordinate
-> overlap region of online_coords_predicted
-> next-window coords_init
```

Visibility/confidence writeback must remain a separate ablation. Track-feature/support mutation is not part of the first implementation because it changes the intervention semantics and raises a much larger parity risk.

### 4.4 Existing feasibility evidence

Repository experiments already establish:

```text
- true streaming can be rerun independently;
- state writeback can affect later windows;
- overly aggressive writeback contaminates future state;
- conservative overlap-only writeback can be made approximately non-destructive;
- candidate verification, not basic state access, is the current bottleneck.
```

This is exactly the setting Route-D is designed to address: sparse candidate routing with calibrated action risk.

### 4.5 Main risks

```text
1. Candidate semantics:
   correlation top-K positions are not automatically independent global hypotheses.

2. Causal leakage:
   candidate features must be captured before current-frame state commit.

3. Window indexing:
   only the inherited overlap can affect the next streaming window.

4. Baseline mismatch:
   full-sequence online architecture and true streaming produce different outputs.

5. Query and scale:
   query format is t,x,y; raw track order is x,y; input/metric raster must be pinned.

6. Support-grid effects:
   evaluation support points must be identical between baseline and Route-D instances.

7. Tail amplification:
   a wrong accepted coordinate can influence a long trajectory suffix.
```

### 4.6 Estimated engineering effort

```text
Candidate-export hook and schema:          medium
1-video parity/determinism harness:        small-to-medium
overlap-only coordinate writeback hook:    small
Kubric candidate cache exporter:           medium-to-large
Route-D feature adapter:                   medium
new preregistration and audit tooling:      medium
```

CoTracker3 is still the lowest-risk first choice because baseline parity and state propagation have already been demonstrated locally.

---

## 5. Rank 2 — TAPNext++ / TAPNext family

### 5.1 Assets

Current local state supersedes the older blocked acquisition record:

```text
code:
  /gemini/code/FSPT/external/tapnextpp/repo/tapnet/tapnextpp

checkpoint:
  /gemini/code/FSPT/checkpoints/tapnextpp/tapnextpp_ckpt.pt

checkpoint size:
  2,532,282,370 bytes

checkpoint SHA-256:
  cb96a43444ccb4fbdb25d800b88c7ba196179a526e78f01b021a16b1c1eff6da
```

A validated 30-video confidence/logit cache also exists. Native first/input metrics in that audit are approximately:

```text
AJ:             65.85
Delta average:  79.09
OA:             92.32
```

### 5.2 Interface opportunity

Potential Route-D candidates:

```text
local candidate:
  native next-token coordinate prediction

alternative candidates:
  top-K coordinate/token hypotheses if logits can be exported before argmax/decoding
  or a pinned external global candidate aligned to the same causal frame

quality features:
  coordinate-token logits
  visibility logits
  entropy/margin
  temporal token disagreement
  local/global displacement
```

### 5.3 Main limitation

Existing TAPNext++ audits show that simple global visibility-threshold and local confidence recovery rules do not convert diagnostic headroom into a reliable gain. This does not rule out Route-D, but it means a new adapter must expose genuine coordinate alternatives and not repackage the failed confidence-only route.

The recurrent token state and exact commit point also need a fresh code audit before any closed-loop claim. Until the state transition is isolated and deterministic, TAPNext++ is the second candidate rather than the first.

### 5.4 Estimated engineering effort

```text
top-K coordinate hypothesis export:        medium-to-large
recurrent-state commit audit:              large
baseline parity/determinism:               medium
Route-D cache adapter:                     medium
closed-loop writeback validation:          large
```

---

## 6. Rank 3 — Track-On2 DINOv3

### 6.1 Assets and parity

Available locally:

```text
baselines/track_on/
baselines/track_on/checkpoints_trackon2_dinov3.pt
```

Existing repo-native first/input parity evidence:

```text
DAVIS AJ:          67.04
Delta average:     79.84
OA:                approximately 92--93 depending on the audited bridge field
preferred full metric parity gate: pass
```

Raw coordinate/query convention is different from CoTracker3:

```text
raw coordinate order: y,x
query order:          t,y,x
```

### 6.2 Interface opportunity and risk

Track-On2 exposes useful confidence/logit and memory-related internals, and repository work already contains internal proxy and feature-family audits. It may provide stronger native alternative evidence than CoTracker3.

However, its memory/query coupling and coordinate convention make a causal candidate interface and clean state writeback more invasive. The first Track-On2 Route-D implementation must not reuse an output-level bridge and call it closed loop.

Estimated effort: large.

---

## 7. Rank 4 — LocoTrack

Repository status:

```text
wrapper code:       available under baselines/track_on/ensemble/locotrack
local checkpoint:   not found
baseline parity:    not established
online state path:  not established
```

LocoTrack may be useful as an open-loop alternative candidate provider, but it is currently blocked as a primary Route-D backbone. A checkpoint, native parity, query-anchor audit, and causal-state analysis are required before promotion.

Estimated effort: large, with asset risk.

---

## 8. Rank 5 — CoWTracker

No usable local CoWTracker implementation, checkpoint, parity artifact, or state interface was found in the audited workspace. It is not a rational target for the current paper cycle.

Decision: stop unless assets are introduced under a separate acquisition task.

---

## 9. Required one-video CoTracker3 smoke

The first smoke is a protocol/interface audit, not a performance experiment.

### 9.1 Required outputs

```text
1. Native baseline replay A.
2. Native baseline replay B.
3. Instrumented local-only replay.
4. Candidate-export replay with routing disabled.
5. Open-loop forced-candidate diagnostic.
6. Closed-loop forced-candidate diagnostic on one predeclared frame only.
```

### 9.2 Mandatory checks

```text
baseline parity:
  native replay A == native replay B
  native replay A == instrumented local-only replay

coordinate direction:
  x,y preserved end to end

raster scale:
  input raster and metric raster explicitly recorded

query mode:
  first

visibility convention:
  native threshold and boolean conversion recorded

deterministic replay:
  candidate coordinates, scores, and selected indices identical

causal timing:
  candidate features captured before current-frame state commit

state effect:
  open-loop leaves future state unchanged
  closed-loop changes only the declared overlap state and downstream suffix
```

Any parity failure is a stop, not a tuning opportunity.

---

## 10. Kubric-only Route-D development plan

Only after the one-video interface gate passes:

```text
1. Export a causal CoTracker3 candidate cache on preregistered Kubric partitions.
2. Verify sample/video disjointness across fit, model-selection, calibration, and holdout.
3. Measure candidate oracle headroom before training a controller.
4. Train the multi-threshold candidate scorer on fit only.
5. Select scorer/model family on model-validation only.
6. Fit isotonic calibration and policy margins on calibration only.
7. Evaluate A0--A7 component ablations on Kubric holdout.
8. Measure open-loop and closed-loop state effects separately.
9. Freeze all choices.
10. Write a new external protocol before opening a new external result.
```

Required component rows:

```text
local only
best predicted global without tree
multi-threshold scorer only
tree without isotonic
tree + isotonic
tree + policy margins
open loop
closed loop
```

---

## 11. Efficiency audit

The strong-backbone integration must report:

```text
native tracker mean/p95 latency
candidate extraction latency
multi-threshold scorer latency
ExtraTrees/isotonic/policy latency
total open-loop latency
total closed-loop latency
CPU/GPU synchronization cost
peak GPU memory
host memory
controller storage
candidate count per decision
global selection rate
per-action downstream trajectory influence
```

The controller overhead must be measured relative to the exact same native streaming baseline.

---

## 12. Proceed gate

Proceed to a new external evaluation only if all conditions pass prospectively:

```text
1. Strong-backbone native performance is close to its validated public/repository reference.
2. Instrumented local-only output matches the independent native baseline.
3. Candidate oracle shows meaningful headroom on Kubric holdout.
4. Route-D AJ gain is at least +1.0 point on the preregistered strong-backbone holdout/external primary evaluation.
5. Paired-video 95% CI lower bound is greater than zero.
6. Delta-average changes in the same positive direction.
7. Closed loop provides additional positive contribution over open loop.
8. Severe tail failures do not materially worsen under predeclared tail metrics.
9. Runtime and memory overhead remain within a predeclared budget.
10. A second backbone or second untouched external domain shows directionally consistent evidence before a broad generality claim.
```

---

## 13. Stop / downgrade gate

Stop the strong-backbone route or downgrade the paper claim when any of the following holds:

```text
- native baseline parity cannot be established;
- no causal candidate set distinct from the native output can be exposed;
- deterministic candidate replay fails;
- closed-loop state mutation cannot be isolated from unrelated memory changes;
- candidate oracle headroom is negligible;
- Route-D gain is below approximately +0.5 AJ point on a strong backbone;
- gain appears only on the weak MMP backbone;
- paired CI includes zero;
- Delta-average moves in the opposite direction;
- closed loop does not add value and has no justified backbone-specific mechanism explanation;
- severe-tail harm materially worsens;
- a valid second backbone/domain cannot be obtained for a broad-method claim.
```

If the gain is between +0.5 and +1.0 AJ point with a positive CI, retain it as component/diagnostic evidence but do not automatically claim a strong general method paper.

---

## 14. Recommended first implementation task

Implement a non-training audit harness:

```text
scripts/audit_routeD_cotracker3_interface.py
```

Proposed invocation after implementation:

```bash
python scripts/audit_routeD_cotracker3_interface.py \
  --checkpoint baselines/cotracker/checkpoints/scaled_online.pth \
  --query-mode first \
  --video-index 0 \
  --input-raster 256 \
  --metric-raster 256 \
  --deterministic-replays 2 \
  --candidate-topk 5 \
  --routing-disabled \
  --output outputs/routeD_strong_backbone_20260717/cotracker3_interface_smoke.json
```

This command is a specification for the next implementation step; the script does not yet exist and no smoke was run in this audit.

The first pass condition is exact native parity with routing disabled. Performance gain is not part of this smoke gate.

---

## 15. Final recommendation

```text
Recommended first backbone: CoTracker3 online true-streaming.

Reason:
  strongest combination of available code/checkpoint,
  already validated native parity,
  directly auditable streaming state,
  demonstrated state inheritance,
  and a tractable candidate-verification bottleneck.

Second candidate:
  TAPNext++ / TAPNext, after top-K coordinate and recurrent-state commit audit.

Kinetics rule:
  keep the completed corrected 1,144-video result frozen;
  do not tune any new strong-backbone component from it.
```
