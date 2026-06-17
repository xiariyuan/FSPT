You are reviewing a research codebase for point tracking under `projects/mmp_tracker/`.

Please read these three files first:
- `projects/mmp_tracker/GPT_PRO_REVIEW_BRIEF.md`
- `projects/mmp_tracker/PAPER_EVIDENCE.md`
- `projects/mmp_tracker/EXPERIMENT_MATRIX.md`

Then inspect the current mainline implementation files:
- `projects/mmp_tracker/train_mmp.py`
- `projects/mmp_tracker/mmp_tracker/model.py`
- `projects/mmp_tracker/mmp_tracker/losses.py`
- `projects/mmp_tracker/mmp_tracker/memory_bank.py`
- `projects/mmp_tracker/mmp_tracker/global_relocator.py`
- `projects/mmp_tracker/mmp_tracker/local_matcher.py`
- `projects/mmp_tracker/configs/localglobal_dev.yaml`

Context:
- We have already abandoned the old FSPT residual-refiner route.
- The current mainline is `local matching + visible-memory global relocalization`.
- Posterior fusion is no longer the mainline and should be treated as secondary.
- The latest package includes recent fixes for protocol drift and selector/global supervision.

What I need from you:

1. Judge whether the current `localglobal_dev` route is structurally worth continuing as the paper mainline.
2. Identify any remaining code-level bugs or train/eval mismatches that could invalidate conclusions.
3. Critically evaluate whether the current losses match the architecture goals, especially:
   - coordinate loss
   - local heatmap loss
   - global heatmap loss
   - global coordinate loss
   - selector loss
   - visibility loss
   - no-harm prior
   - long-occ focus loss
4. Judge whether the selector design is now principled enough, or whether it should be replaced by a candidate classifier / abstain design.
5. Judge whether the memory bank policy is now sufficient for long occlusion, or whether it still needs redesign.
6. Decide what should be the next 3 highest-ROI experiments before burning more training time.
7. Decide whether posterior should stay removed from the mainline.
8. Decide whether frequency should remain auxiliary / appendix-only for now.

Please give the answer in this format:

## 1. Executive judgment
- continue / pause / pivot
- one-sentence reason

## 2. Critical code issues
- only issues that materially affect validity or convergence
- for each issue: file, problem, severity, fix recommendation

## 3. Architecture judgment
- what part is correct
- what part is still structurally weak
- whether this can plausibly become a publishable story

## 4. Loss-function judgment
- which losses are aligned
- which are misaligned
- what should be removed, reduced, or added

## 5. Experiment priority
- top 3 experiments to run next
- exact expected evidence each experiment should produce
- stop rules if the result is negative

## 6. Paper-story recommendation
- what the main contribution should be
- what should be demoted to ablation / appendix
- whether this is currently closer to top-tier / mid-tier / workshop potential, and why

Constraints:
- Be blunt and technical.
- Do not suggest vague “train longer” advice.
- Prioritize structural correctness, protocol validity, and paper evidence quality.
- If you think the route is still wrong, say so directly and propose the most defensible pivot.
