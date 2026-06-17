# Verifier-Guided Pseudo-Labeling and Selective Tracking

## 1. Why this is the new main line

The earlier relocalization/TTA story did not produce a strong enough and stable AJ gain on a strong base tracker. The most useful signal we have now is not a better relocation head, but a better reliability signal.

The new main line is:

**verifier-guided pseudo-label filtering + calibrated selective tracking**

This is stronger than a pure inference-time gate because it does two things:

- It decides which predictions are reliable enough to keep at test time.
- It decides which pseudo-labels are reliable enough to train on.

That gives us a cleaner and more defensible paper story than "we refined relocalization a bit more."

## 2. What the current evidence says

We ran a full DAVIS selective evaluation on three score sources:

- `verifier_scores`
- `pred_visibility`
- `relocal_conf`

The key conclusion is:

- `verifier_scores` is the best learned reliability signal we currently have.
- `pred_visibility` is a strong baseline, but it is not a learned reliability story.
- `relocal_conf` is clearly weaker and too poorly calibrated to anchor the paper.

The important numbers are:

| source | thr | selected_rate | precision | recall | ECE | oracle_gap_closed |
|---|---:|---:|---:|---:|---:|---:|
| verifier_scores | 10 | 0.177 | 0.640 | 0.202 | 0.091 | 0.275 |
| verifier_scores | 20 | 0.392 | 0.660 | 0.440 | 0.068 | 0.503 |
| verifier_scores | 30 | 0.475 | 0.679 | 0.543 | 0.062 | 0.699 |
| pred_visibility | 10 | 0.697 | 0.609 | 0.756 | 0.161 | 0.378 |
| relocal_conf | 10 | 0.295 | 0.516 | 0.271 | 0.548 | 0.191 |

The AJ deltas are still tiny. That means the paper should not claim "we improved the tracker average score." The story should be about reliability, coverage, calibration, and training-time filtering.

## 3. Paper framing

The central question becomes:

**Can a point tracker estimate when its output is trustworthy, and can that trust signal be used to improve both inference and training?**

The three paper claims should be:

1. A learned reliability head can rank candidate tracks better than a naive visibility signal.
2. Reliability can be evaluated with selective metrics, not just average AJ.
3. Reliability can be used to filter pseudo-labels and improve downstream training.

## 4. What to build next

### Stage 1: lock the main signal source

Use:

- base tracker: `CoTracker3`
- learned reliability signal: `verifier_scores`
- baseline signal: `pred_visibility`
- weak reference: `relocal_conf`

### Stage 2: connect verifier to pseudo-label filtering

The project already has a pseudo-label training path in `train.py` and `utils/advanced_training.py`.

We should extend it so that the teacher confidence can come from:

- `verifier_scores`
- `relocal_acceptor`
- `policy_gate`
- fallback: `pred_visibility`

That lets the verifier act as a training-time filter, not only as an inference-time gate.

### Stage 3: evaluate the reliability story properly

For the paper, the required metrics are:

- selective AJ
- coverage vs risk curve
- AURC
- ECE / Brier score
- acceptance precision / recall
- oracle gap closed

### Stage 4: only then consider more ambitious extensions

If the verifier-guided pseudo-label path is stable, then we can look at:

- real-video pseudo-label mining
- domain adaptation
- stronger calibration objectives

Those are later-stage extensions, not prerequisites.

## 5. What not to do anymore

- Do not continue to treat TTA as the main story.
- Do not continue to treat the current relocalization gain as sufficient.
- Do not write the paper as "safety decision" or generic abstention.
- Do not claim meaningful AJ improvement if the data does not show it.

The correct story is:

**verifier-guided pseudo-label filtering and calibrated selective tracking.**

