# GPT Pro File Guide — 2026-03-09

## Read first

1. `projects/mmp_tracker/GPT_PRO_NEXT_STAGE_BRIEF_20260309.md`
2. `projects/mmp_tracker/pro_review_artifacts/20260309/overnight_autopilot_decision.json`
3. `projects/mmp_tracker/pro_review_artifacts/20260309/mmp_overnight_autopilot_20260308_232031.log`

These three files tell you:

- what was tested,
- what failed,
- what the automatic decision logic concluded,
- and what the next architecture question is.

## Then read these code files

### Core implementation

- `projects/mmp_tracker/mmp_tracker/model.py`
- `projects/mmp_tracker/mmp_tracker/losses.py`
- `projects/mmp_tracker/mmp_tracker/config.py`
- `projects/mmp_tracker/mmp_tracker/global_relocator.py`
- `projects/mmp_tracker/mmp_tracker/local_matcher.py`
- `projects/mmp_tracker/mmp_tracker/memory_bank.py`
- `projects/mmp_tracker/train_mmp.py`

### Most relevant configs

- `projects/mmp_tracker/configs/local_dev.yaml`
- `projects/mmp_tracker/configs/localglobal_nocommit_dev.yaml`
- `projects/mmp_tracker/configs/localglobal_top1_deferredcommit_dev.yaml`
- `projects/mmp_tracker/configs/local_night_full.yaml`
- `projects/mmp_tracker/configs/localglobal_top1_deferredcommit_night_full.yaml`

### Automation / experiment orchestration

- `projects/mmp_tracker/run_controlled_commit_matrix.sh`
- `projects/mmp_tracker/run_overnight_autopilot.sh`

## Then inspect raw metrics if needed

### Seed 42

- `projects/mmp_tracker/pro_review_artifacts/20260309/mmp_local_dev_seed42_20260308_225548.epoch_metrics.jsonl`
- `projects/mmp_tracker/pro_review_artifacts/20260309/mmp_localglobal_nocommit_dev_seed42_20260308_225548.epoch_metrics.jsonl`
- `projects/mmp_tracker/pro_review_artifacts/20260309/mmp_localglobal_top1_deferredcommit_dev_seed42_20260308_225548.epoch_metrics.jsonl`

### Seed 43

- `projects/mmp_tracker/pro_review_artifacts/20260309/mmp_local_dev_seed43_20260309_005244.epoch_metrics.jsonl`
- `projects/mmp_tracker/pro_review_artifacts/20260309/mmp_localglobal_nocommit_dev_seed43_20260309_005244.epoch_metrics.jsonl`
- `projects/mmp_tracker/pro_review_artifacts/20260309/mmp_localglobal_top1_deferredcommit_dev_seed43_20260309_005244.epoch_metrics.jsonl`

### Full baseline

- `projects/mmp_tracker/pro_review_artifacts/20260309/mmp_local_night_full_20260308_232033.epoch_metrics.jsonl`

## If you only have limited context budget

Read exactly this subset:

1. `projects/mmp_tracker/GPT_PRO_NEXT_STAGE_BRIEF_20260309.md`
2. `projects/mmp_tracker/pro_review_artifacts/20260309/overnight_autopilot_decision.json`
3. `projects/mmp_tracker/mmp_tracker/model.py`
4. `projects/mmp_tracker/mmp_tracker/losses.py`
5. `projects/mmp_tracker/train_mmp.py`
6. `projects/mmp_tracker/configs/localglobal_top1_deferredcommit_dev.yaml`

That subset is enough to understand:

- the intended mechanism,
- the current negative result,
- and the next decision point.

