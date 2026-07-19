# Route-D Paper Readiness and Experiment Plan

Date: 2026-07-17
Goal: determine what is still required for a credible CCF-B / CAS Zone 2 submission

## 1. Current verdict

### Scientific evidence

```text
External protocol rigor:        strong
Package and metric audit:       strong
Paired statistical evidence:    strong
Mechanism evidence:             promising
Failure transparency:           strong
Base-tracker competitiveness:   weak
Cross-backbone generality:      absent
Component ablation completeness:incomplete
Runtime/complexity evidence:     absent
Corrected multi-dataset table:  incomplete
```

### Publication assessment

The corrected full-scale Kinetics result is strong enough to preserve Route-D as a serious research mainline and to support a component-paper draft. It is not yet sufficient for a confident CCF-B / CAS Zone 2 submission, and the current 34.80 AJ system should not be packaged as a competitive full tracker.

The main weakness is not the statistical significance of the current result. The main weakness is **external validity and comparative strength**:

```text
- one integrated backbone;
- a relatively weak base tracker;
- no corrected official-scale comparison table against modern strong trackers;
- no complete paper ablation isolating every Route-D component;
- no runtime or parameter overhead analysis;
- no prospective stability improvement after severe closed-loop failures.
```

A submission now would likely be reviewed as a well-audited routing improvement on a substantially underperforming base tracker. The decisive publication upgrade is no longer wording or another Kinetics analysis; it is prospective validation on a strong backbone.

---

## 2. Novelty audit against current literature

### Existing work already covers

```text
online causal point tracking;
long-term memory;
global receptive-field matching;
re-detection after occlusion;
training on long sequences;
real-video pseudo-labeling;
streaming-memory propagation;
local/global matching architectures.
```

Therefore, Route-D must not claim novelty for those concepts alone.

### Defensible Route-D novelty

1. **Explicit action formulation:** local-versus-global candidate routing is exposed as a separately trained intervention over a frozen tracker.
2. **Multi-threshold utility profile:** candidate quality is represented across 1/2/4/8/16-pixel correctness rather than one confidence score.
3. **Risk-calibrated action gate:** a calibrated beneficial-action model is combined with explicit fine/coarse utility constraints.
4. **Sparse closed-loop feedback:** accepted corrections update future tracker state, and the same controller is evaluated in open and closed loop to isolate feedback value.
5. **Protocol-level evidence:** controller fitting is Kubric-only and the external Kinetics result is prospectively frozen, package-audited, paired, and failure-audited.

### Reviewer attack to anticipate

> “This is only a post-hoc tree selecting between candidates already produced by the tracker.”

Required response must be empirical, not rhetorical:

```text
- show candidate oracle headroom;
- show naive confidence/rule selectors fail or recover much less utility;
- show the multi-threshold scorer matters;
- show tree action modeling matters beyond scorer margins;
- show calibration/policy constraints matter;
- show closed-loop feedback matters beyond open-loop replacement;
- show the effect survives on a stronger or second backbone;
- report negligible or acceptable overhead.
```

---

## 3. Required paper experiment matrix

All new controller or guard selection must use Kubric-only development partitions. Kinetics remains frozen external evidence and cannot be used for selection.

### P0 — Complete component ablation on Kubric-only partitions

Use one fixed nested or train/calibration/validation protocol and report video-level uncertainty.

| Row | Candidate scorer | Action gate | Calibration | Profile constraints | Feedback |
|---|---|---|---|---|---|
| A0 | none | always local | none | none | closed tracker baseline |
| A1 | raw quality | choose best global | none | none | open loop |
| A2 | multi-threshold scorer | scorer margin only | none | none | open loop |
| A3 | multi-threshold scorer | neural action gate | optional | frozen | open loop |
| A4 | multi-threshold scorer | ExtraTrees | none | frozen | open loop |
| A5 | multi-threshold scorer | ExtraTrees | isotonic | none | open loop |
| A6 | multi-threshold scorer | ExtraTrees | isotonic | frozen profile policy | open loop |
| A7 | same as A6 | same | same | same | closed loop |

Required outputs:

```text
mean pixel gain;
mean threshold-utility gain;
Delta-1/2/4/8/16 gains;
selection rate;
harmful selected rate;
beneficial recall;
video-level paired bootstrap;
calibration metrics;
oracle headroom recovered.
```

Hard rule: do not choose the winning ablation using Kinetics.

### P0 — Runtime and complexity

Measure on one fixed device and batch/query configuration:

```text
base tracker latency;
base + multi-threshold scorer latency;
base + full Route-D open-loop latency;
base + full Route-D closed-loop latency;
mean and p95 latency;
controller parameter/storage size;
peak memory;
number of candidate decisions per frame;
```

The ExtraTrees CPU transfer in the current adapter may become a reviewer concern. Profile host-device synchronization and batch-size sensitivity explicitly.

### P0 — Correct paper artifact consistency

Create one machine-readable table source and generate Markdown/LaTeX tables from it. It must include:

```text
exact scope counts 1,147 / 1,144 / 3;
aggregate metrics;
all six paired intervals;
finite sample counts;
positive/negative/tie counts;
selection and trajectory diagnostics;
severe failures;
artifact hashes.
```

No handwritten duplicate result table should become an independent numerical authority.

### P0 — Strong-backbone feasibility and validation

This is the most important and decisive publication gate.

#### Preferred route

Integrate the Route-D action interface into a strong candidate-generating online tracker without changing the controller concept. Repository evidence ranks the candidates:

```text
1. CoTracker3 online true-streaming;
2. TAPNext++ / TAPNext family;
3. Track-On2 DINOv3;
4. LocoTrack;
5. CoWTracker.
```

CoTracker3 is the first choice because local code/checkpoints, native parity, true-streaming replay, and inherited coordinate state are already present. TAPNext++ is second because code/checkpoint and strong coordinates are available, but top-K coordinate hypotheses and the recurrent state commit point still require a clean audit.

Minimum requirements:

```text
- local and at least one global candidate exposed causally;
- prediction-only candidate features available;
- clean local-only independent baseline;
- open-loop and true closed-loop modes;
- training and controller fitting restricted to Kubric-only partitions;
- all choices frozen before a new external dataset is opened.
```

Candidate families and current role:

```text
CoTracker3 online true-streaming: first implementation target;
TAPNext/TAPNext++: second target after coordinate-hypothesis/state audit;
Track-On2 DINOv3: strong parity fallback with higher state-integration complexity;
LocoTrack: blocked until checkpoint and native parity exist;
CoWTracker: stop for the current cycle unless code/checkpoint assets are introduced.
```

Do not force an adapter that violates the published model's coordinate, visibility, query, or online protocol.

#### Minimum success gate

Before opening a new external set, require on Kubric holdout and in the one-video interface audit:

```text
- native strong-backbone performance close to the validated repository/public reference;
- exact native parity with routing disabled;
- deterministic replay of candidate coordinates, features, and selected indices;
- positive candidate oracle headroom;
- Route-D AJ gain >= +1.0 point on the preregistered primary strong-backbone evaluation;
- lower paired-video 95% CI bound > 0;
- Delta-average improves in the same direction;
- closed loop adds a positive contribution over open loop;
- no material worsening of predeclared severe-tail metrics;
- latency and memory overhead below predeclared budgets;
- directionally consistent evidence on a second backbone or second untouched external domain before a broad generality claim.
```

Downgrade/stop rule:

```text
- gain below approximately +0.5 AJ point on a strong backbone;
- gain only on the weak MMP backbone;
- paired CI includes zero;
- no distinct causal candidate interface;
- native parity or deterministic replay fails;
- closed-loop state mutation cannot be isolated;
- severe-tail harm materially worsens.
```

The full interface ranking, code/checkpoint evidence, state paths, risks, engineering estimates, and stop/proceed gates are recorded in:

```text
docs/ROUTED_STRONG_BACKBONE_FEASIBILITY_AUDIT_2026-07-17.md
```

### P1 — New untouched external protocol

After all P0/P1 choices are frozen, preregister one new external dataset not used for controller or stability development.

Candidate datasets should be selected by actual availability and protocol parity, not by expected positive result. RoboTAP or EgoPoints may be appropriate only after verifying licensing, query mode, coordinate convention, and dataset history in the repository.

The protocol must pin:

```text
dataset identity and hashes;
all available sample counts;
query mode;
raster conversion;
checkpoint/controller/config hashes;
primary comparison and metrics;
bootstrap unit/resamples/seed;
undefined-metric policy;
success/failure rule;
no-retuning statement.
```

### P1 — Failure mechanism analysis

Use existing frozen Kinetics output only for descriptive analysis, not method selection.

Required descriptive buckets:

```text
selection rate;
local-global jump magnitude;
pre-action tracker confidence;
coarse/fine profile margins;
occlusion duration if available;
time since query;
distance to image boundary;
trajectory suffix affected after first action;
number of repeated global actions;
open-loop success but closed-loop failure;
closed-loop rescue after open-loop error.
```

Key mechanism figure:

```text
x-axis: frames after accepted intervention
y-axis: mean closed-loop minus open-loop point error or threshold accuracy
strata: successful action / harmful action
```

This would directly visualize the temporal influence horizon of state feedback.

### P2 — Prospective Kubric-only stability guard

The severe Kinetics failures motivate but cannot train a new guard.

Develop only on Kubric-only partitions. Candidate guard signals may include:

```text
predicted switch distance;
profile uncertainty or disagreement;
temporal persistence of the global hypothesis;
prior/global motion consistency;
repeat-action cooldown;
closed-loop influence-risk prediction;
calibrated abstention region.
```

Prospective objective:

```text
reduce tail harm while retaining most aggregate utility gain.
```

Predeclare before any new external evaluation:

```text
primary tail-risk metric;
allowed aggregate-gain loss;
harmful-video or harmful-action ceiling;
selection-rate range;
external pass/fail rule.
```

Do not call the resulting system safe unless a formal guarantee is actually established.

---

## 4. Recommended main-paper tables

### Table 1 — Main external Kinetics result

Use the corrected 1,144-of-1,147 table from the core draft.

### Table 2 — Paired uncertainty and video support

Include all comparisons, finite counts, confidence intervals, and positive/negative/tie counts.

### Table 3 — Kubric-only component ablation

Rows A0--A7 from the P0 matrix.

### Table 4 — Generalization / stronger-backbone result

Only include after prospective protocol completion.

### Table 5 — Efficiency

Latency, p95, memory, and controller size.

### Appendix tables

```text
threshold-specific Delta-1/2/4/8/16 gains;
calibration metrics;
selection-rate strata;
full severe-failure list;
undefined-metric cases;
package and protocol hashes;
metric-correction sensitivity.
```

---

## 5. Recommended figures

1. **Method figure:** local/global candidates -> multi-threshold profiles -> action features -> calibrated gate -> local/global decision -> optional state feedback.
2. **Open-versus-closed causal diagram:** same current-frame correction, different future state path.
3. **Utility-risk plot:** selected coverage or selection rate versus mean gain/harmful rate on Kubric validation.
4. **Feedback influence horizon:** closed-loop minus open-loop performance after accepted actions.
5. **Qualitative success/failure:** at least two rescues, one harmless abstention, one open-loop success/closed-loop failure, and one severe feedback cascade.

---

## 6. Paper claim ladder

### Tier 1 — Current evidence

> A frozen Kubric-trained Route-D controller improves the frozen MMP tracker on the audited 1,144-of-1,147 TAP-Vid-Kinetics release-CSV materialization, and closed-loop feedback adds significant aggregate benefit over open-loop replacement.

### Tier 2 — After complete Kubric ablations and efficiency

> Multi-threshold utility modeling, calibrated action gating, and state feedback each contribute to sparse global correction under a controlled causal protocol.

### Tier 3 — After a stronger or second backbone

> Risk-calibrated sparse feedback routing generalizes beyond one tracker instance or candidate generator.

### Tier 4 — Only after a new untouched external protocol

> The method transfers prospectively across multiple external domains without dataset-specific tuning.

Do not write Tier 3 or Tier 4 claims before the corresponding evidence exists.

---

## 7. Immediate implementation order

```text
1. Preserve the corrected Kinetics result and controller unchanged.
2. Complete and freeze the strong-backbone feasibility audit.
3. Implement a one-video CoTracker3 true-streaming interface/parity harness with routing disabled.
4. Build the single source-of-truth paper result JSON plus generated Markdown/LaTeX tables.
5. Implement the Kubric-only A0--A7 ablation runner without opening Kinetics.
6. Add latency, memory, selection-rate, and downstream-influence measurement.
7. Export a causal CoTracker3 Kubric candidate cache only after the parity gate passes.
8. Freeze a new external protocol only after all adapter/controller choices are complete.
9. Develop any stability guard only on Kubric-only partitions under a separate preregistration.
```

## 8. Stop rules

```text
- Stop claiming a broad method paper if a second/stronger backbone cannot expose a valid candidate interface.
- Do not use Kinetics failures to select a guard or threshold.
- Do not add external result rows with mismatched query/raster/evaluator protocols.
- Do not report superseded DAVIS/RGB/Kinetics position metrics.
- Do not call the method SOTA unless a strict comparable table supports it.
- Do not hide negative videos or undefined metrics.
- Stop or downgrade a broad method claim if strong-backbone gain is below approximately +0.5 AJ point or only the weak MMP backbone benefits.
- Do not open a strong-backbone external evaluation before native parity, deterministic replay, candidate oracle, and preregistration gates pass.
```

## 9. Strong-backbone update — 2026-07-19

The CoTracker3 strong-backbone Kubric model-validation gate is no longer pending.
The preregistered component matrix identifies a reproducible safety-feasible
configuration:

```text
frozen formal CMCP core + LMRA + comparator
AJ gain: +0.8861 points
paired-video AJ 95% CI: [+0.6489,+1.0694]
positive videos: 16 / 16
harmful non-native rate: 0.9396%
```

This supplies internal strong-backbone evidence, but not a new external claim.
Calibration, final synthetic holdout, DAVIS, and official Kinetics remain locked.
The paper may now describe the strong-backbone mechanism and Kubric ablation, but
Tier-3 external generalization language remains unauthorized until a separately
frozen external protocol is completed.

The immediate next step is one coordinate-only bounded writeback ablation against
the frozen-CMCP variant C. Do not reopen candidate, layer, rank, threshold, NMS,
EMA, top-K, or intervention-strength sweeps on model validation.

## 10. Closed-loop boundary update — 2026-07-19

The strong-backbone variant C result remains valid only under its finalized-state
output-only evaluation contract. The preregistered P0k interface audit shows that
the provisional overlap state required for next-window writeback differs from the
finalized state used by formal C on 50% of eligible native rows, with revisions
up to 40.9733px.

Therefore:

```text
strong-backbone output-only evidence: retained
strong-backbone closed-loop claim: not authorized
P0k model-validation evaluation: not run
coordinate-only writeback route: closed for this paper
```

The paper should present this as an explicit limitation and causal-interface
negative result, not hide it or infer a closed-loop benefit from the P0j oracle.

## 11. Final synthetic holdout update — 2026-07-19

Frozen output-only variant C passes the one-time identity-disjoint 16-video
Kubric final holdout:

```text
AJ gain: +0.7679
paired-video AJ CI: [+0.4532,+0.9184]
delta gain: +0.8889
positive videos: 14 / 16
harmful rate: 0.9462%
```

This materially strengthens the paper's internal generalization evidence and
authorizes a frozen external protocol. It does not yet authorize an external
claim. The paper must retain the output-only boundary, the slight 1px regression,
and per-video safety heterogeneity.

## 12. Frozen DAVIS transfer result — 2026-07-19

The preregistered complete 30-video DAVIS transfer audit of frozen output-only
variant C fails every scientific performance and safety gate:

```text
AJ gain:                    -1.9170
paired-video AJ 95% CI:     [-2.4472,-1.4449]
delta-average gain:         -1.5591
candidate-oracle AJ gain:   +0.4936
harmful non-native rate:    2.2148%
positive videos:            0 / 30
exact primary/replay:       yes
```

The P0m native AJ is `64.4109`, within `0.0299` point of the independent official
CoTracker3 replication. The external failure is therefore not dismissed as a
protocol mismatch. Candidate headroom collapses from the synthetic domain, and
the Kubric-trained safety comparator accepts almost exclusively harmful DAVIS
actions.

Paper claim correction:

```text
Tier 3 external generalization: not authorized
strong-backbone evidence: Kubric-only
strong-backbone closed loop: not authorized
DAVIS zero-shot transfer: failed and must be reported
tracker-agnostic/domain-general wording: forbidden
```

The main paper may still use P0j/P0l as a controlled synthetic strong-backbone
mechanism study, but it must present P0m as a central limitation. A broad method
paper claim of external generalization is not supported by the current evidence.
No DAVIS rescue sweep is allowed, and official Kinetics 1,144 remains frozen.
