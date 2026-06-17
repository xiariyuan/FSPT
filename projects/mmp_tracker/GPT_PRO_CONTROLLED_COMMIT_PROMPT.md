You are reviewing the current MMP tracker mainline under `projects/mmp_tracker/`.

Please read these files first:

- `projects/mmp_tracker/GPT_PRO_CONTROLLED_COMMIT_BRIEF.md`
- `projects/mmp_tracker/PAPER_EVIDENCE.md`
- `projects/mmp_tracker/EXPERIMENT_MATRIX.md`

Then inspect these implementation files:

- `projects/mmp_tracker/train_mmp.py`
- `projects/mmp_tracker/mmp_tracker/config.py`
- `projects/mmp_tracker/mmp_tracker/model.py`
- `projects/mmp_tracker/mmp_tracker/losses.py`
- `projects/mmp_tracker/mmp_tracker/global_relocator.py`
- `projects/mmp_tracker/mmp_tracker/local_matcher.py`
- `projects/mmp_tracker/mmp_tracker/memory_bank.py`

And these configs:

- `projects/mmp_tracker/configs/local_dev.yaml`
- `projects/mmp_tracker/configs/localglobal_dev.yaml`
- `projects/mmp_tracker/configs/localglobal_top1rematch_dev.yaml`
- `projects/mmp_tracker/configs/localglobal_top1_relaxedcommit_dev.yaml`
- `projects/mmp_tracker/configs/localglobal_commithead_dev.yaml`

Context:

- We have already moved past the old residual-refiner / posterior mainline.
- The current route is explicitly about long-occlusion recovery.
- The strongest new evidence says `top1 rematch` is useful.
- The newest evidence also says commit is crucial, but the `always_selected` version is unstable across seeds.
- The question is no longer whether this research route exists at all.
- The question is now how to design the next controlled-commit mainline in the most defensible way.

What I need from you:

1. Judge whether the paper story is now legitimately centered on:
   - local matching
   - visible-memory global retrieval
   - top1 rematch
   - controlled commit

2. Decide what is already established by the current evidence, and what is still missing.

3. Critically evaluate whether the next controlled-commit design should be:
   - thresholded controlled commit
   - learned binary commit
   - commit as abstention / defer
   - candidate-based commit / ranking
   - or something simpler

4. Judge whether selector is still the main bottleneck, or whether commit has now clearly become the main bottleneck.

5. Identify only the highest-ROI next experiments needed to convert the current evidence into a paper-ready story.

6. Tell me whether the route is currently closer to:
   - workshop
   - mid-tier conference
   - or still has a real top-tier upgrade path

Please answer in this exact structure:

## 1. Executive judgment
- continue / continue-with-redesign / pause / pivot
- one-sentence reason

## 2. What is now established
- only conclusions that the current evidence really supports

## 3. What remains structurally wrong
- only the main bottlenecks
- cite the relevant file(s)

## 4. Best next controlled-commit design
- what to build next
- why it is better than both heuristic commit and always-selected commit
- whether it should be learned, thresholded, abstain-based, or candidate-based

## 5. Minimal next experiment set
- top 3 experiments only
- exact expected evidence from each
- stop rules if negative

## 6. Paper-story recommendation
- the sharpest possible contribution claim now
- what to leave out of the main story
- whether this still has a top-tier path, and under what conditions

Constraints:

- Be blunt and technical.
- Do not say vague things like "train longer".
- Prioritize mechanism validity, evidence quality, and paper defensibility.
- If you think `always_selected` has already proven enough and should now be retired, say so directly.
