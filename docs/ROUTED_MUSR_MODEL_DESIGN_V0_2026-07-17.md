# Route-D MUSR model design v0 — 2026-07-17

## 1. Decision

The next Route-D mainline is not another tree-controller threshold sweep and not a generic residual refiner revival. It is a new trainable core module:

> **MUSR: Multi-Hypothesis Utility-Conditioned State Recovery**

A strong tracker such as CoTracker3 supplies causal native tracks, features, and state. MUSR jointly evaluates multiple competing hypotheses, predicts their multi-scale correctness and downstream risk, may abstain, and emits a bounded state transition. The backbone is frozen in stage 0 so that any gain is attributable to the new module.

This is a model-design route, not a micro-tuning route.

## 2. Why a new module is necessary

Historical repository evidence already rules out several weaker designs:

- generic CoTracker3 + residual refinement frequently drifted or collapsed to exact base parity;
- DINOv2 continuous local offset regression underperformed the raw CoTracker3 coordinate;
- local grid verifiers often learned to always retain the centre candidate;
- independent per-candidate scorers cannot model competition, duplicate hypotheses, or candidate-set context;
- threshold-only visibility correction can improve re-entry diagnostics while harming strict global AJ/OA;
- post-hoc tree routing produced statistically real gains on the weak MMP system, but it is not a sufficiently strong final model contribution.

The new module must therefore solve candidate competition, confidence calibration, tail risk, abstention, and state consequences together.

## 3. Core research hypothesis

Given a causal candidate set at time `t`, a tracker should not select a candidate solely by its local confidence. It should estimate:

1. the probability that each candidate lies within `1/2/4/8/16 px` of the latent target;
2. the probability of catastrophic error;
3. whether any candidate is reliably better than the native continuation;
4. how strongly each tracker-state field should be updated;
5. whether the safest action is to abstain.

The central claim to test is:

> Joint candidate-set reasoning plus risk-aware state transition can convert recoverable oracle headroom into causal closed-loop gain without increasing severe-tail failures.

## 4. Model inputs

For every tracked point and causal time step:

- `candidate_features: [B,P,K,F]`
- `candidate_coords_px: [B,P,K,2]`
- `candidate_valid_mask: [B,P,K]`
- `state_features: [B,P,S]`
- `source_ids: [B,P,K]`

Candidate index `0` is always the native backbone continuation. It must be valid. Other candidates are unordered alternatives identified by source embeddings, not by rank-position embeddings.

Initial candidate families:

1. native CoTracker3 online continuation;
2. native local-correlation top-K alternatives;
3. query-memory global top-K alternatives;
4. last-reliable-memory alternatives;
5. causal motion-prior candidate;
6. an optional separately audited external candidate.

No future frame, future visibility label, future backbone state, or external-domain observation may enter candidate generation.

## 5. Architecture

### 5.1 Candidate tokenization

Each candidate token combines:

- candidate evidence features;
- displacement from the native continuation;
- displacement norm and squared norm;
- source identity;
- an explicit native/non-native role embedding.

### 5.2 Joint candidate-set encoder

A state token and all candidate tokens enter a Transformer encoder. There is no positional encoding over non-native candidates. The design is therefore permutation-equivariant over the non-native candidate set while preserving the special role of candidate `0`.

This is materially different from the historical independent scorer: the score of one candidate can depend on disagreement, duplication, or support from all other candidates.

### 5.3 Monotonic multi-threshold utility

For each candidate, MUSR predicts logits for:

```text
P(error <= 1 px)
P(error <= 2 px)
P(error <= 4 px)
P(error <= 8 px)
P(error <= 16 px)
```

The probabilities are monotonic by parameterization. The first logit is free; all later logits add positive `softplus` increments. A monotonicity penalty is therefore unnecessary and cannot be gamed by optimisation.

Expected candidate utility is a preregistered weighted average of the five probabilities. The tight thresholds receive greater weight.

### 5.4 Catastrophic-risk head

A separate head estimates the probability that candidate error exceeds the catastrophic threshold. Selection score is:

```text
expected utility - lambda_risk * catastrophic probability + bounded contextual bias
```

The risk term is not a post-hoc threshold. It is trained jointly and audited for calibration. The contextual bias is `tanh`-bounded so it cannot overwhelm the interpretable utility/risk terms.

### 5.5 Abstention and state writeback

The decision context contains:

- the encoded tracker state;
- the selected/soft-selected candidate context;
- the native candidate context.

It predicts:

- abstention probability;
- write strength for coordinate;
- write strength for visibility;
- write strength for confidence;
- write strength for memory.

All write strengths are multiplied by `(1 - abstention_probability)`.

Stage 0 only applies coordinate writeback to the inherited online coordinate state. Support tokens and template memory remain unchanged until a separate audit passes.

### 5.6 Trust region

The selected coordinate delta is norm-clipped to a fixed maximum and then scaled by the learned coordinate-write strength. Thus a model bug or overconfident candidate cannot produce an unbounded state jump.

## 6. Training objective

The implemented composite objective contains:

- multi-threshold binary cross-entropy;
- best-candidate ranking loss;
- catastrophic-risk binary cross-entropy;
- differentiable coordinate loss on the bounded update;
- no-harm penalty relative to native continuation;
- abstention supervision when no candidate offers sufficient gain;
- severe-tail penalty above a fixed pixel threshold.

Later rollout training may add short closed-loop unrolling, but external-domain metrics must not be used to choose its horizon or weights.

## 7. Data and protocol

Stage-0 development and model selection are Kubric-only with disjoint, sample-identity-frozen partitions. TAP-Vid-Kinetics official-scale and TAP-Vid-DAVIS are locked until:

1. native parity passes;
2. deterministic candidate export passes;
3. causal state writeback passes;
4. Kubric oracle headroom is large enough;
5. Kubric learned gain passes;
6. the external protocol and success gates are preregistered.

The completed 1,144-sample Kinetics evaluation is not rerun and is not used for tuning.

## 8. Stage-0 success gates

Required before any external evaluation:

- exact native parity when only candidate `0` is valid;
- deterministic replay equality;
- exact monotonic threshold probabilities;
- causal candidate provenance audit;
- Kubric oracle AJ headroom at least `+3.0` points;
- learned Kubric AJ gain at least `+1.0` point;
- paired confidence-interval lower bound above zero;
- closed-loop better than open-loop;
- severe-tail rate not worse than native.

If oracle headroom is below the gate, stop candidate-generation work before training. If oracle passes but learned selection repeatedly fails, redesign representation/contrastive supervision rather than tuning a decision threshold.

## 9. Implemented in this step

- `projects/mmp_tracker/mmp_tracker/routeD_recovery_network.py`
  - model configuration;
  - joint candidate/state Transformer;
  - monotonic utility head;
  - catastrophic-risk head;
  - contextual selection;
  - abstention;
  - multi-field write strengths;
  - bounded coordinate update;
  - composite training objective.
- `configs/routeD_musr_cotracker3_stage0.yaml`
  - frozen experiment and gate contract.
- `tests/test_routeD_recovery_network.py`
  - output contract;
  - monotonicity;
  - invalid-candidate masking;
  - trust-region bound;
  - non-native permutation equivariance;
  - exact native-only parity;
  - loss/backpropagation;
  - deterministic hard selection.

## 10. Immediate next implementation

Build the CoTracker3 online stage-0 adapter that exports, for one Kubric video only:

- native online candidate and state features;
- local-correlation top-K candidates;
- source IDs and validity masks;
- exact candidate provenance;
- deterministic replay hashes;
- routing-disabled native parity.

Then compute candidate oracle headroom before training MUSR.

Completed one-video result: exact routing-disabled parity, exact two-pass adapter export replay, and +8.6477 AJ-point coordinate-oracle headroom on frozen Kubric validation video 0. The next allowed step is a preregistered disjoint multi-video Kubric oracle/cache qualification; this one-video oracle does not authorize external evaluation or final training claims.

## 11. Completed multi-video candidate qualification

The frozen identity-disjoint qualification on 15 previously unobserved Kubric
validation videos passed all preregistered gates. Pooled coordinate-oracle AJ
headroom is `+6.5497` points and pooled delta-average headroom is `+8.6178`
points with native visibility fixed. All 15 videos are positive; minimum and
median per-video AJ gains are `+4.2208` and `+6.4035`. Adapter-export replay at
source indices `1`, `8`, and `15` is bit-identical.

This authorizes fit/model-validation/calibration cache export and MUSR training,
but the oracle result is not a learned-model claim. The final synthetic holdout,
DAVIS, and Kinetics remain locked. See
`docs/ROUTED_MUSR_KUBRIC_QUALIFICATION_RESULT_2026-07-17.md`.

## 12. Stage-A selector feasibility result

The complete fit/model-validation/calibration caches contain 48/16/32 videos.
Calibration was not read. Native-safe initialization gives exact zero-step
parity, and utility-aligned selector-only training is exactly reproducible at
seed 17. Nevertheless, the best model-validation result is only `+0.0173` AJ
point with paired 95% CI `[-0.0092,+0.0537]`, despite `+6.8463` AJ oracle
headroom. State-write Stage B is therefore closed.

The active redesign keeps candidate coordinates fixed and replaces the 64-D
compressed summaries with full query/candidate CoTracker feature interactions,
a local correlation patch, and full online track state. See
`docs/ROUTED_MUSR_SELECTOR_STAGEA_RESULT_2026-07-17.md` and
`docs/ROUTED_MUSR_RAW_REPRESENTATION_V1_PLAN_2026-07-17.md`.

## 13. Raw-v1 Stage-A result and temporal pivot

The representation-only raw-v1 experiment preserves every candidate coordinate
and expands candidate/state inputs from 64/32 to 601/160. Complete fit and
model-validation caches pass all frozen-contract and replay checks. The formal
seed-17 selector is exactly reproducible, but best model-validation AJ changes
by only `-0.0149` point with paired 95% CI `[-0.0806,+0.0502]`. It is slightly
worse than the structured selector and does not authorize state-write training.

Fit-to-validation marginal feature shift is small, while beneficial candidate
events are temporally persistent and the single-frame selector recalls only
`8.27%` of them. The active P0f design is therefore a causal, rank-invariant
set-evidence accumulator over a fixed six-frame window. Candidate generation,
raw-v1 features, partitions, optimizer, and gates remain frozen. See
`docs/ROUTED_MUSR_RAW_V1_STAGEA_RESULT_2026-07-17.md` and
`docs/ROUTED_MUSR_TEMPORAL_REPRESENTATION_V0_PLAN_2026-07-17.md`.

## 14. Temporal-v0 result and proposal-generation pivot

The fixed six-frame causal rank-invariant set-evidence accumulator passes exact
zero-step native parity, strict causality, padding isolation, candidate-set
permutation audits, deterministic monotonic utility construction, and exact
seed-17 replay. Nevertheless, the best model-validation result is only
`+0.0041` AJ point with paired 95% CI `[-0.0045,+0.0106]`. Delta average and
the severe 16px tail improve, and harmful global selection is only `0.0798%`,
but beneficial-event recall is just `1.8443%`. Only four of sixteen videos are
positive.

Three post-discretization selector families have now failed despite `+6.8463`
AJ coordinate-oracle headroom: structured single-frame, raw single-frame, and
causal temporal. The next allowed route therefore moves learning before top-K
candidate extraction. A causal multi-memory correlation proposal generator will
train dense query-anchor, previous-native, and fixed-EMA correlation maps on fit
only while preserving the frozen CoTracker3 backbone and exact native fallback.
No selector, state-write, calibration, final-holdout, DAVIS, or Kinetics step is
authorized until the proposal-only gate passes. See
`docs/ROUTED_MUSR_TEMPORAL_V0_STAGEA_RESULT_2026-07-17.md` and
`docs/ROUTED_MUSR_MULTI_MEMORY_PROPOSAL_V0_PLAN_2026-07-17.md`.

## 15. CMCP interface milestone

The causal multi-memory correlation proposal generator moves trainable reasoning
before top-K discretization. Its formal 64-channel ConvGRU proposal core has 263,747
trainable parameters and consumes immutable query-anchor, previous-native, and
fixed-alpha EMA correlations, all three pairwise correlation differences, a
native motion prior, and previous proposal evidence. Native candidate 0 remains frozen.

On authorized fit video 0, the real CoTracker3 run reproduces every previously
qualified native state tensor exactly. All 64 points and 24 frames satisfy exact
candidate-0 parity and zero-step native selection. Correlations, proposal maps,
candidates, and selections are bit-identical both within the run and in a
separate full-video replay. This authorizes feature-map cache export and fit-only
dense proposal training, but it is not learned performance. See
`docs/ROUTED_CMCP_INTERFACE_RESULT_2026-07-17.md`.

## 16. CMCP feature-map cache milestone

Frozen float16 CoTracker feature maps are complete for 48 fit and 16
model-validation videos. Every regenerated native state matches the qualified
stage-0 sidecar exactly. Float32 feature extraction is bit-identical at indices
0, 47, 48, and 63. Global worst fp16 reconstruction error is `2.3586e-4` with
minimum cosine `0.999999404`; the resulting three-memory correlation error on
the fit smoke is at most `1.2365e-4`. CMCP fit-only dense proposal training is
therefore authorized, while all selectors, state writes, calibration, final
holdout, DAVIS, and Kinetics remain locked.


## 17. CMCP proposal-only result and local-safety pivot

Before formal training, CMCP was corrected to match the preregistered
nine-channel input by adding all three pairwise correlation differences. The
formal 64-channel model contains 263,747 trainable parameters and retains exact
zero-step native behavior and independent full-video replay.

Formal fit-only seed-17 training is exactly reproducible. The best epoch-1
proposal pool has `+20.0375` AJ coordinate-oracle gain with paired 95% CI
`[+18.0040,+21.7391]`; every one of the 16 model-validation videos is positive.
Direct top-1 improves AJ by `+1.5843` and delta average by `+1.8865`, and reduces
the 16px severe-error rate from `25.8782%` to `18.5212%`. However, its paired AJ
CI `[-2.5666,+4.5731]` crosses zero and harmful non-native selection is
`27.9968%`, so the formal proposal-only gate fails.

The strong, universal oracle result rules out candidate availability as the
current bottleneck. P0h freezes the exact epoch-1 proposal generator and trains
only a local pairwise native-vs-peak safety comparator. No candidate-coordinate,
NMS, EMA, top-K, threshold, backbone, state-write, calibration, final-holdout,
DAVIS, or Kinetics change is authorized. See
`docs/ROUTED_CMCP_PROPOSAL_TRAINING_RESULT_2026-07-17.md` and
`docs/ROUTED_CMCP_LOCAL_PAIRWISE_SAFETY_V0_PLAN_2026-07-17.md`.


## 18. P0h local pairwise safety interface

The formal epoch-1 CMCP generator is frozen at model-state SHA-256
`fc3044eb6daa1fb2416164fc0afe58ff5fa4eb4442811ce9c0b4917fbbdb57c0`.
A 409,224-parameter local candidate-set comparator consumes 88-D tokens sampled
at native and five frozen proposal peaks. It has monotonic multi-threshold
utility, catastrophic risk, pairwise preference, and explicit native
abstention.

On authorized fit video 0, the complete candidate-coordinate tensor hash is
`c4cc84161d6522eb889c48c80cb5fb54673e4f446f54c697079ab5e4a9979300`.
Independent full-video runs reproduce all candidate and selection hashes, and
zero-step selected coordinates equal native exactly. P0h-b may export frozen
local tokens and train only the comparator on fit. See
`docs/ROUTED_CMCP_PAIRWISE_SAFETY_INTERFACE_RESULT_2026-07-17.md`.


## 19. P0h frozen local-token cache

The local pairwise comparator cache is complete at 48 fit and 16
model-validation videos. Each 88-D token contains 84 frozen GT-free local fields
and four online causal-summary placeholders. Candidate-coordinate combined
SHA-256 values are
`3f1be12649a8c4757c933c3f6d8c99b9f3d6bcd146262406557e139377ca57c2`
for fit and
`edf8ebe25cc11b04406efe27203a7d2bea4da8ce075160a27a11d8edda1fb0a1`
for model validation. P0h-c may train only the comparator on fit; all generator,
backbone, state-write, calibration, and external-data locks remain active.


## 20. P0h comparator result and P0i metric-adapter pivot

The frozen-candidate local pairwise comparator is exactly reproducible at seed
17. Best epoch 3 improves complete model-validation AJ by `+0.2449` and delta
average by `+0.4962`, with paired AJ CI `[+0.1324,+0.3693]`. Fifteen of sixteen
videos are positive, harmful non-native selection is `0.3132%`, and the severe
16px rate improves. Candidate coordinates and the `+20.0375` AJ oracle remain
exactly equal to P0g.

The comparator passes every safety and consistency gate but fails the
preregistered `+0.5` AJ magnitude gate. Its beneficial-candidate recall is only
`3.3362%`, so no MUSR, state-write, calibration, final-holdout, DAVIS, or
Kinetics step is authorized.

P0i adds a single zero-initialized rank-32 residual metric adapter after frozen
CoTracker `fnet.conv3`, affecting only proposal correlations and local evidence.
The native trajectory branch stays byte-frozen. See
`docs/ROUTED_CMCP_PAIRWISE_SAFETY_TRAINING_RESULT_2026-07-17.md` and
`docs/ROUTED_CMCP_LATE_METRIC_ADAPTER_V0_PLAN_2026-07-17.md`.


## 21. P0i LMRA interface

A zero-initialized rank-32 late feature metric residual adapter is inserted after
frozen CoTracker `fnet.conv3` for the proposal/evidence branch only. It contains
8,352 parameters. Together with the initialized formal CMCP and P0h comparator,
the trainable branch contains 681,323 parameters; the native trajectory branch
remains frozen.

On fit video 0, adapted and frozen feature maps are byte-identical, and online
reconstruction reproduces the formal P0h candidate coordinates, raw tokens,
causal decisions, and selected coordinates exactly in independent runs. The
candidate-coordinate hash remains
`c4cc84161d6522eb889c48c80cb5fb54673e4f446f54c697079ab5e4a9979300`.
P0i-b joint fit-only training is therefore authorized. See
`docs/ROUTED_CMCP_LMRA_INTERFACE_RESULT_2026-07-17.md`.
