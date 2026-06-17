# Route-A Paper Plan (2026-04)

## Positioning

The strongest current asset in this repository is **not** the original "frequency-semantic point tracking" pitch in the top-level README. The most coherent and defensible line is:

**Verifier-guided relocalization for long-occlusion point tracking**

More specifically:

- Start from a strong base tracker.
- Generate relocalization / rematching candidates when the point reappears after long occlusion.
- Learn when to accept a candidate and when to abstain.
- Improve hard reappearance cases without damaging overall TAP-Vid metrics.

This should be the main paper. Frequency priors, semantic priors, anchor banks, and two-view geometry can remain supporting components or future extensions, but they should not define the title unless they are clearly decisive in the final ablations.

## Why This Direction Fits The Current Repo

The repository already contains:

- a CoTracker-based refiner path in `models/cotracker_refiner.py`
- long-occlusion evaluation in `scripts/eval_long_occlusion_subset.py`
- qualitative tooling in `scripts/visualize_longocc_examples.py`
- table-generation tooling in `scripts/make_longocc_comparison_table.py`
- many stage-3 relocalization configs under `configs/fspt_routeA_stage3_relocal_longocc_*`
- real logs showing a large oracle gap on long-occlusion subsets

Representative evidence from the current logs:

- `outputs/mmp_localglobal_nocommit_night_full_seed42_20260310_180342/epoch_metrics.jsonl`
  - `AJ = 0.1225`
  - `AJ_longocc20 = 0.0920`
  - `AJ_longocc30 = 0.0691`
  - `oracle_coarse_AJ_longocc20 = 0.1448`
  - `oracle_coarse_AJ_longocc30 = 0.0941`

Interpretation:

- candidate quality is not the main bottleneck
- decision quality is the main bottleneck
- a paper centered on candidate verification / safe acceptance is better aligned than a paper centered on frequency-semantic fusion

## Proposed Titles

Preferred:

- `Verifier-Guided Relocalization for Long-Occlusion Point Tracking`

Alternatives:

- `Do-No-Harm Relocalization for Tracking Reappearing Points`
- `Recovering Reappearing Points with Verifier-Guided Candidate Acceptance`
- `Safe Relocalization After Long Occlusion for Tracking Any Point`

## Core Story

The paper should tell a narrow and reviewable story:

1. Strong point trackers already work well on easy or moderately difficult cases.
2. A persistent failure mode remains: points that disappear for a long interval and later reappear.
3. In these cases, candidate relocalization is often available, but blindly switching to it hurts global metrics.
4. The real missing piece is a verifier that decides when a candidate should replace the base prediction.
5. A do-no-harm acceptance policy can improve long-occlusion recovery while preserving official benchmark quality.

This is a better story than:

- "we combine frequency, semantics, CLIP, geometry, and occlusion reasoning into one large framework"

That larger story is too broad for the current experimental maturity and too easy for reviewers to attack as under-validated.

## Contribution Set

Target a clean 3-part contribution list:

1. A relocalization pipeline for long-occlusion point tracking built on top of a strong base tracker.
2. A verifier-guided candidate acceptance module with explicit do-no-harm behavior.
3. A focused evaluation protocol for reappearance after long occlusion, including oracle-gap analysis and hard-subset reporting.

If the verifier becomes solid, a fourth contribution is reasonable:

4. A training protocol that uses verifier signals to improve real-video pseudo-label quality.

Do not claim pseudo-labeling as a main contribution unless it is actually implemented and ablated.

## Similar Papers And How We Should Differ

### CoTracker3 (ICCV 2025)

Their story:

- simpler, stronger base tracker
- large-scale pseudo-labeling on real videos
- better general point tracking recipe

Our difference:

- not a universal new base tracker
- not a data-scaling paper
- focus on the failure mode of long occlusion and safe recovery

Takeaway:

- use as the base and as a level-of-quality reference
- do not try to out-story them on general tracking

### LocoTrack (ECCV 2024)

Their story:

- local 4D correlation design
- efficient and accurate matching

Our difference:

- not primarily an efficiency or correlation-architecture paper
- main claim is acceptance and abstention under reappearance uncertainty

Takeaway:

- cite as a strong tracker baseline and as evidence that better matching alone does not fully solve reappearance safety

### Track-On (ICLR 2025)

Their story:

- online point tracking with persistent state or memory

Our difference:

- not a causal-memory paper
- focus on relocalization decisions after long occlusion, especially when the point becomes visible again

Takeaway:

- useful comparison for the "long-term memory is not enough" narrative

### ReTracker (ICCV 2025)

Their story:

- importing image matching ideas into online any-point tracking
- stronger retracking / rematching

Our difference:

- closest overlap and the most important comparator
- they emphasize stronger matching
- we should emphasize verifier-guided acceptance and do-no-harm switching

Takeaway:

- this is the paper we need to be at least as focused and polished as
- avoid claiming generic "retracking is better"; claim "verification makes relocalization safe"

### TAPNext++ (arXiv 2026)

Their story:

- pushes re-detection / reappearance-oriented evaluation and training

Our difference:

- close topic overlap
- we should not compete as a generic re-detection paper
- we should compete as a safe acceptance / verifier paper

Takeaway:

- align our hard-subset metrics with re-detection style reporting
- keep the algorithmic center on verification and abstention

### Real-World Point Tracking with Verifier-Guided Pseudo-Labeling (arXiv 2026)

Their story:

- verifier used for pseudo-label quality control in training

Our difference:

- verifier at inference time for relocalization acceptance
- pseudo-labeling can become a second-stage extension, not the main claim

Takeaway:

- high relevance
- we need to be explicit that our verifier solves a different stage of the pipeline

### TAPIP3D (2025)

Their story:

- persistent 3D geometry for long-term point tracking

Our difference:

- geometry-first vs acceptance-first
- 3D persistence can be a future extension or backup direction

Takeaway:

- good inspiration for a second paper
- too large a pivot for the current primary submission

## Current Gaps To Fill

### Highest Priority

1. Replace heuristic gating with a real verifier.
2. Standardize one canonical candidate bank.
3. Freeze one canonical evaluation protocol.
4. Produce reviewer-friendly ablations that isolate the verifier.

### Concretely Missing

Algorithm:

- a learned score predicting whether candidate error is lower than base error
- explicit abstention threshold
- calibration analysis for the verifier

Experiments:

- 3-seed results for the final model
- official TAP-Vid metrics and long-occlusion subset metrics in the same table
- oracle-gap analysis
- acceptance precision / recall / coverage analysis
- failure-case visualization

Presentation:

- one clean method diagram
- one clean oracle-gap figure
- one clean long-occlusion qualitative figure

## Experiment Roadmap

### Stage 1: Freeze The Evaluation Protocol

Keep only one main setup:

- base tracker: CoTracker3
- datasets: DAVIS and Kinetics first, Kubric as support if needed
- main hard subsets: long-occ-10, 20, 30
- report:
  - AJ
  - OA
  - `<1 / <2 / <4`
  - `AJ_longocc10 / 20 / 30`
  - `reapp_error_longocc10 / 20 / 30`

Also compute:

- candidate oracle AJ on the same subsets
- selected candidate AJ on the same subsets
- acceptance precision / acceptance recall
- protected-easy-case rate

### Stage 2: Build The Real Verifier

Minimal practical design:

- inputs:
  - base confidence / visibility
  - candidate confidence
  - local-vs-global disagreement
  - temporal consistency features
  - correlation or matching quality features
  - last-visible displacement prior
- target:
  - binary label: candidate better than base by margin `m`
- loss:
  - BCE or focal loss
  - optional calibration regularizer

Important:

- separate candidate generation from candidate acceptance
- do not let the story collapse back into "many heuristics"

### Stage 3: Ablation Ladder

Main table should include:

1. Base tracker
2. Base + candidate generator
3. Base + heuristic gate
4. Base + learned verifier
5. Base + learned verifier + do-no-harm constraint

Optional sixth row:

6. Base + learned verifier + do-no-harm + focus mask

This table is the heart of the paper.

### Stage 4: Oracle-Gap Figure

One figure should show:

- base performance
- oracle candidate performance
- heuristic-selected performance
- verifier-selected performance

If the verifier closes a meaningful fraction of the oracle gap while preserving global AJ, the story becomes much stronger.

### Stage 5: Qualitative Evidence

Need 3 types of examples:

1. Base fails after long occlusion, ours recovers.
2. Heuristic would switch incorrectly, verifier abstains.
3. Multiple candidate locations exist, verifier picks the correct one.

Without these, reviewers will not trust the "safe relocalization" claim.

## Paper Structure

Recommended outline:

1. Introduction
2. Related Work
3. Failure Mode Analysis of Long-Occlusion Reappearance
4. Verifier-Guided Relocalization
5. Experimental Setup
6. Main Results
7. Oracle-Gap And Acceptance Analysis
8. Qualitative Results
9. Limitations
10. Conclusion

## Venue Strategy

Primary recommendation:

- `Pattern Recognition`

Backup:

- `IEEE TCSVT`

Why journal-first:

- the current work is stronger as a focused method plus careful analysis paper
- there is room for more ablations, figures, and protocol detail
- rolling submission avoids forcing an immature conference draft

Stretch option:

- if the verifier becomes very clean and results become strong enough, revisit a future top vision conference cycle
- but the current default plan should be journal-first

## Realistic Success Estimates

If we keep the old FSPT frequency-semantic framing:

- low probability of acceptance even at the user's minimum target

If we reframe now but keep mostly heuristic gates:

- moderate chance at a CCF-B / solid journal target

If we build a real verifier and present a clean protocol:

- strong chance to clear the user's minimum bar
- potentially competitive with focused papers like ReTracker in overall polish, even if not in scale

Practical internal estimate:

- old framing: 5% to 10%
- reframed without a real verifier: 20% to 30%
- reframed with a real verifier and complete ablations: 35% to 50%
- reframed with verifier plus a useful pseudo-label extension: 45%+

These numbers assume the final paper is coherent and the results are stable across seeds.

## The Standard We Can Realistically Match

We can realistically match papers of the following style:

- a sharply scoped method paper on a hard failure mode
- a strong base + one novel decision module + targeted evaluation
- a paper whose value comes from analysis and reliability, not from building the biggest new tracker

We are not yet in a position to match:

- a universal new SOTA tracker trained at CoTracker3 scale
- a large data-scaling paper
- a full geometry-plus-semantic-plus-foundation-model omnibus paper

That is not a weakness. It is the correct scope discipline for a strong submission.

## Immediate Next Actions

1. Freeze one canonical base tracker and one canonical evaluation config.
2. Implement a real verifier head for candidate-vs-base comparison.
3. Export one stable ablation table on DAVIS long-occ subsets.
4. Run 3 seeds for the final verifier model.
5. Generate oracle-gap and qualitative figures.
6. Draft the paper around "safe relocalization after long occlusion".
