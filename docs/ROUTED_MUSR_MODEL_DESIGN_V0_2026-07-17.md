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
