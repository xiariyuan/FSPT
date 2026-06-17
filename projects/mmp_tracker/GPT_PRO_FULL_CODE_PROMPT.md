You are reviewing the attached full-code research bundle for the current point-tracking project.

Please start with these files first:

- `projects/mmp_tracker/GPT_PRO_CONTROLLED_COMMIT_BRIEF.md`
- `projects/mmp_tracker/GPT_PRO_FULL_CODE_HANDOFF.md`
- `projects/mmp_tracker/PAPER_EVIDENCE.md`
- `projects/mmp_tracker/EXPERIMENT_MATRIX.md`

Then inspect the current MMP tracker implementation under:

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

You may also inspect the wider codebase if you think there are relevant interactions elsewhere.

Context:

- The route has narrowed to: local matching + visible-memory global retrieval + top1 rematch + controlled commit.
- Top1 rematch already shows real value.
- Commit is clearly important.
- The aggressive always-selected relaxed-commit variant gave useful upside on one seed but was unstable on another.
- We do not want generic advice like "train longer".
- We want you to think deeply about what the most defensible next mainline should be.

What I need from you:

1. Decide what conclusions are already strongly supported by the existing evidence.
2. Identify the main mechanism that should become the next paper center.
3. Judge whether the next controlled-commit design should be:
   - thresholded
   - learned binary commit
   - abstain / defer
   - candidate-based commit / ranking
   - or something simpler and more robust
4. Decide whether selector is still the real bottleneck, or whether commit is now the dominant bottleneck.
5. Tell me what parts of the current code or plan are still conceptually weak.
6. Give the minimum next experiment set needed to convert this into a paper-ready evidence chain.
7. Judge whether this route still has a realistic top-tier path, and what exact conditions must be met for that.

Please answer in this exact structure:

## 1. Executive judgment
- continue / continue-with-redesign / pause / pivot
- one-sentence reason

## 2. What is now established
- only conclusions strongly supported by current code and evidence

## 3. What remains structurally wrong
- only the main bottlenecks
- cite the relevant files when possible

## 4. Best next controlled-commit design
- what to build next
- why it is better than both conservative commit and always-selected commit

## 5. Minimal next experiment set
- top 3 experiments only
- expected evidence from each
- stop rules if negative

## 6. Paper-story recommendation
- sharpest contribution claim now
- what to leave out of the main story
- whether this still has a top-tier path, and under what conditions

Constraints:

- Be blunt and technical.
- Do not give vague optimization or training-duration advice.
- Prioritize structural correctness, evidence quality, and publication defensibility.
- If you think some current experiments have already answered enough and should be retired, say so directly.
