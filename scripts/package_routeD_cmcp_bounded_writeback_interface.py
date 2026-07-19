#!/usr/bin/env python3
"""Package and independently verify the failed P0k commit-time interface gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import file_sha256

DEFAULT_PRIMARY = REPO_ROOT / "outputs/routeD_cmcp_bounded_writeback_20260719/interface_fit0.json"
DEFAULT_REPLAY = REPO_ROOT / "outputs/routeD_cmcp_bounded_writeback_20260719/interface_fit0_replay.json"
DEFAULT_OUTPUT = REPO_ROOT / "docs/generated/ROUTED_STRONG_BACKBONE_BOUNDED_WRITEBACK_INTERFACE_SUMMARY_2026-07-19.json"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def verify_tensor_tree_exact(left: Any, right: Any, path: str = "root") -> int:
    """Require identical nested tensor/scalar structure; return tensor count."""
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise RuntimeError(f"tensor/type mismatch at {path}")
        if not torch.equal(left, right):
            raise RuntimeError(f"tensor mismatch at {path}")
        return 1
    if type(left) is not type(right):
        raise RuntimeError(f"type mismatch at {path}: {type(left)} != {type(right)}")
    if isinstance(left, dict):
        if left.keys() != right.keys():
            raise RuntimeError(f"dict key mismatch at {path}")
        return sum(
            verify_tensor_tree_exact(left[key], right[key], f"{path}.{key}")
            for key in left
        )
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise RuntimeError(f"sequence length mismatch at {path}")
        return sum(
            verify_tensor_tree_exact(a, b, f"{path}[{index}]")
            for index, (a, b) in enumerate(zip(left, right))
        )
    if left != right:
        raise RuntimeError(f"value mismatch at {path}: {left!r} != {right!r}")
    return 0


def structural_diagnostics(artifact: dict[str, Any]) -> dict[str, Any]:
    formal = artifact["formal"]
    commit = artifact["commit_no_write"]
    if artifact.get("bounded_write") is not None:
        raise RuntimeError("bounded write must not execute after interface failure")
    index_mismatch = (
        commit["selected_candidate_index"] != formal["selected_candidate_index"]
    )
    coordinate_error = torch.linalg.vector_norm(
        commit["selected_coords_xy_px"] - formal["selected_coords_xy_px"], dim=-1
    )
    candidate_error = torch.linalg.vector_norm(
        commit["candidate_coords_xy_px"] - formal["candidate_coords_xy_px"], dim=-1
    )
    first_seen = commit["first_seen_chunk_start"]
    frame_count = int(first_seen.numel())
    step = 8
    eligible = torch.tensor(
        [
            int(first_seen[frame].item()) >= 0
            and int(first_seen[frame].item()) + step <= frame
            and frame < int(first_seen[frame].item()) + 2 * step
            for frame in range(frame_count)
        ],
        dtype=torch.bool,
    )
    native_delta = torch.linalg.vector_norm(
        commit["decision_native_coords_xy_px"] - commit["final_native_coords_xy_px"],
        dim=-1,
    )
    eligible_index = index_mismatch[:, eligible]
    eligible_native_delta = native_delta[:, eligible]
    return {
        "rows": int(index_mismatch.numel()),
        "eligible_rows": int(eligible_index.numel()),
        "selected_index_mismatch_rows": int(index_mismatch.sum().item()),
        "selected_index_mismatch_fraction": float(index_mismatch.float().mean().item()),
        "eligible_selected_index_mismatch_rows": int(
            eligible_index.sum().item()
        ),
        "eligible_selected_index_mismatch_fraction": (
            float(eligible_index.float().mean().item())
            if eligible_index.numel()
            else 0.0
        ),
        "selected_coordinate_mismatch_rows": int(
            (coordinate_error > 1.0e-6).sum().item()
        ),
        "selected_coordinate_mismatch_fraction": float(
            (coordinate_error > 1.0e-6).float().mean().item()
        ),
        "selected_coordinate_max_difference_px": float(coordinate_error.max().item()),
        "candidate_coordinate_max_difference_px": float(candidate_error.max().item()),
        "formal_non_native_rows": int(
            (formal["selected_candidate_index"] > 0).sum().item()
        ),
        "commit_non_native_rows": int(
            (commit["selected_candidate_index"] > 0).sum().item()
        ),
        "formal_non_native_eligible_rows": int(
            (formal["selected_candidate_index"][:, eligible] > 0).sum().item()
        ),
        "commit_non_native_eligible_rows": int(
            (commit["selected_candidate_index"][:, eligible] > 0).sum().item()
        ),
        "commit_native_vs_final_max_difference_px": float(native_delta.max().item()),
        "commit_native_vs_final_nonzero_fraction": float(
            (native_delta > 0).float().mean().item()
        ),
        "eligible_commit_native_vs_final_max_difference_px": (
            float(eligible_native_delta.max().item())
            if eligible_native_delta.numel()
            else 0.0
        ),
        "eligible_commit_native_vs_final_nonzero_fraction": (
            float((eligible_native_delta > 0).float().mean().item())
            if eligible_native_delta.numel()
            else 0.0
        ),
        "eligible_frame_indices": torch.where(eligible)[0].tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", default=str(DEFAULT_PRIMARY))
    ap.add_argument("--replay", default=str(DEFAULT_REPLAY))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = ap.parse_args()
    primary_path = Path(args.primary).resolve()
    replay_path = Path(args.replay).resolve()
    primary = _read(primary_path)
    replay = _read(replay_path)
    expected_decision = "STOP_P0K_BEFORE_MODEL_VALIDATION_COMMIT_STATE_MISMATCH"
    for label, report in (("primary", primary), ("replay", replay)):
        if report["decision"] != expected_decision or report["interface_pass"]:
            raise RuntimeError(f"unexpected {label} P0k decision")
        if report["bounded_write_executed"]:
            raise RuntimeError(f"{label} bounded write unexpectedly executed")
        if any(report["locked_data_read"].values()):
            raise RuntimeError(f"{label} read locked data")
        if not report["no_write_replay_exact"]:
            raise RuntimeError(f"{label} internal no-write replay failed")
        if not all(row["exact"] for row in report["formal_self_check"].values()):
            raise RuntimeError(f"{label} formal framewise self-check failed")
        if not all(row["exact"] for row in report["final_native_parity"].values()):
            raise RuntimeError(f"{label} final native parity failed")
        if all(row["exact"] for row in report["commit_time_vs_formal_C"].values()):
            raise RuntimeError(f"{label} did not reproduce commit-time mismatch")

    report_fields = (
        "decision",
        "interface_pass",
        "source_index",
        "video_name",
        "frames",
        "points",
        "config_sha256",
        "protocol_sha256",
        "variant_C_checkpoint_sha256",
        "feature_index_sha256",
        "formal_self_check",
        "no_write_replay_exact",
        "no_write_replay_hashes",
        "final_native_parity",
        "commit_time_vs_formal_C",
        "commit_native_vs_final_native",
        "bounded_write_executed",
        "locked_data_read",
    )
    replay_checks = {key: primary[key] == replay[key] for key in report_fields}
    if not all(replay_checks.values()):
        raise RuntimeError(f"independent report replay mismatch: {replay_checks}")

    primary_sidecar = torch.load(
        Path(primary["sidecar"]), map_location="cpu", weights_only=False
    )
    replay_sidecar = torch.load(
        Path(replay["sidecar"]), map_location="cpu", weights_only=False
    )
    tensor_count = verify_tensor_tree_exact(primary_sidecar, replay_sidecar)
    diagnostics = structural_diagnostics(primary_sidecar)
    summary = {
        "schema_version": "routeD_strong_backbone_bounded_writeback_interface_summary_v0",
        "date": "2026-07-19",
        "status": "completed_fail_before_model_validation",
        "formal_decision": expected_decision,
        "scientific_interpretation": (
            "P0j variant C is exact under its frozen finalized-state output-only "
            "contract, but the same decisions are not available at the provisional "
            "commit point required for next-window CoTracker writeback."
        ),
        "claim_boundary": (
            "Variant C remains valid as the previously reported finalized-state "
            "output-only Kubric model-validation result. P0k does not establish a "
            "closed-loop gain and model validation was not read."
        ),
        "interface": {
            "source_index": primary["source_index"],
            "video_name": primary["video_name"],
            "frames": primary["frames"],
            "points": primary["points"],
            "formal_framewise_self_check_exact": True,
            "internal_no_write_replay_exact": True,
            "independent_process_replay_exact": True,
            "independent_replay_tensor_count": tensor_count,
            "final_native_state_parity_exact": True,
            "commit_time_formal_C_exact": False,
            "bounded_write_executed": False,
        },
        "commit_time_vs_formal_C": primary["commit_time_vs_formal_C"],
        "structural_diagnostics": diagnostics,
        "replay": {
            "checks": replay_checks,
            "primary_report_sha256": file_sha256(primary_path),
            "replay_report_sha256": file_sha256(replay_path),
            "primary_sidecar_sha256": file_sha256(Path(primary["sidecar"])),
            "replay_sidecar_sha256": file_sha256(Path(replay["sidecar"])),
            "note": (
                "Torch archive byte hashes differ, while every nested tensor and "
                "scalar in the primary/replay artifacts is exact."
            ),
        },
        "provenance": {
            "config_path": primary["config"],
            "config_sha256": primary["config_sha256"],
            "protocol_path": primary["protocol"],
            "protocol_sha256": primary["protocol_sha256"],
            "variant_C_checkpoint": primary["variant_C_checkpoint"],
            "variant_C_checkpoint_sha256": primary[
                "variant_C_checkpoint_sha256"
            ],
            "feature_index": primary["feature_index"],
            "feature_index_sha256": primary["feature_index_sha256"],
        },
        "gate": {
            "formal_framewise_C_reconstruction_exact": True,
            "final_native_state_matches_cache_exact": True,
            "no_write_replay_exact": True,
            "commit_time_candidate_coordinates_exact": primary[
                "commit_time_vs_formal_C"
            ]["candidate_coordinates"]["exact"],
            "commit_time_selected_indices_exact": primary[
                "commit_time_vs_formal_C"
            ]["selected_indices"]["exact"],
            "commit_time_selected_coordinates_exact": primary[
                "commit_time_vs_formal_C"
            ]["selected_coordinates"]["exact"],
            "commit_time_dynamic_summary_exact": primary[
                "commit_time_vs_formal_C"
            ]["dynamic_summary"]["exact"],
            "pass": False,
            "decision": expected_decision,
        },
        "next_step": (
            "Close coordinate-only next-window writeback for the current paper. "
            "Retain variant C as the strong-backbone output-only model and do not "
            "open model validation or sweep state-write timing."
        ),
        "locked_data_read": primary["locked_data_read"],
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": file_sha256(output),
                "decision": expected_decision,
                "selected_index_mismatch_rows": diagnostics[
                    "selected_index_mismatch_rows"
                ],
                "selected_coordinate_mismatch_rows": diagnostics[
                    "selected_coordinate_mismatch_rows"
                ],
                "eligible_native_revision_fraction": diagnostics[
                    "eligible_commit_native_vs_final_nonzero_fraction"
                ],
                "model_validation_read": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
