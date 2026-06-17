# Attempt 0 Status Template (2026-06-13)

## 0. 用途

这是一份每跑完一个 baseline 就要填写的统一模板。

## 1. 基本信息

- `model_name`:
- `repo_url`:
- `repo_commit`:
- `checkpoint_path`:
- `checkpoint_sha256`:
- `dataset_name`:
- `split`:
- `run_date`:

## 2. Repo-Native Reproduction

- `repo_native_protocol`:
- `repo_native_metric_names`:
- `repo_native_numbers`:
- `official_reference_numbers`:
- `delta_vs_official`:
- `reproduction_status`: `pass | partial | fail`

## 3. Export / Adapter

- `raw_coordinate_format`:
- `raw_visibility_format`:
- `query_format`:
- `adapter_version`:
- `adapter_sanity_status`: `pass | fail`
- `adapter_notes`:

## 4. Unified Rescoring

- `query_mode`:
- `metric_resolution_mode`:
- `AJ`:
- `OA`:
- `<avg`:
- `<4px`:
- `delta_vs_cotracker3_baseline`:
- `rescoring_status`: `pass | partial | fail`

## 5. Long-Occlusion / Re-Entry

- `long_occ_definition`:
- `long_occ_AJ`:
- `long_occ_<avg`:
- `long_occ_<4px`:
- `reentry_first_frame_error`:
- `AJ_RD`:

## 6. Runtime / Memory

- `gpu_type`:
- `batch_size`:
- `peak_memory_gb`:
- `eval_wall_time`:
- `notes`:

## 7. Decision

- `keep_for_main_ranking`: `yes | no`
- `keep_as_teacher_candidate`: `yes | no`
- `known_blockers`:
- `next_action`:
