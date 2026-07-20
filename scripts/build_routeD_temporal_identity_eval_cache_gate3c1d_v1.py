#!/usr/bin/env python3
"""Build frozen raw-disjoint evaluation caches for Route-D Gate 3C1D v1."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = Path("/gemini/code/FSPT")
COTRACKER_ROOT = EXTERNAL_ROOT / "baselines/cotracker"
for path in (COTRACKER_ROOT, EXTERNAL_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

SCHEMA = "routeD_temporal_identity_eval_cache_gate3c1d_v1"
INDEX_SCHEMA = "routeD_temporal_identity_eval_cache_index_gate3c1d_v1"
PARTITIONS = {
    "checkpoint_selection_v1": list(range(0, 256)),
    "fit_only_internal_audit_v1": list(range(256, 384)),
    "model_validation_v1": list(range(384, 512)),
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _validate_parent(config: Mapping[str, Any]) -> None:
    parent = config["authorized_parent"]
    for key, hash_key in (("result", "result_sha256"), ("summary", "summary_sha256")):
        if file_sha256(parent[key]) != parent[hash_key]:
            raise ValueError(f"Gate 3C1D v1 cache parent hash drift: {key}")
    summary = json.loads(Path(parent["summary"]).read_text())
    without_hash = dict(summary)
    payload_hash = without_hash.pop("summary_payload_sha256", None)
    if (
        payload_hash != parent["summary_payload_sha256"]
        or payload_hash != canonical_json_sha256(without_hash)
        or not bool(summary.get("pass"))
        or summary.get("formal_decision") != parent["required_decision"]
    ):
        raise ValueError("Gate 3C2 did not authorize renewed cache preregistration")


def _validate_runtime_authorization(config: Mapping[str, Any], partition: str) -> None:
    if partition == "checkpoint_selection_v1":
        return
    authorization = config.get("runtime_authorization", {})
    result_path = Path(authorization["required_replay_result"])
    if not result_path.is_file():
        raise ValueError(f"{partition} replay authorization is absent")
    result = json.loads(result_path.read_text())
    without_hash = dict(result)
    payload_hash = without_hash.pop("result_payload_sha256", None)
    if payload_hash != canonical_json_sha256(without_hash):
        raise ValueError(f"{partition} authorization payload drift")
    protocol_path = Path(authorization["required_protocol_config"]).resolve()
    result_config = Path(str(result.get("config", ""))).resolve()
    if result_config != protocol_path:
        raise ValueError(f"{partition} authorization protocol path drift")
    if result.get("config_sha256") != file_sha256(protocol_path):
        raise ValueError(f"{partition} authorization protocol hash drift")
    if result.get("partition") != authorization["required_partition"]:
        raise ValueError(f"{partition} authorization partition drift")
    if bool(result.get("exact_replay")) is not bool(authorization["required_exact_replay"]):
        raise ValueError(f"{partition} authorization replay failed")
    gate = result.get("gate", {})
    if bool(gate.get("pass")) is not bool(authorization["required_gate_pass"]):
        raise ValueError(f"{partition} authorization gate failed")
    if gate.get("decision") != authorization["required_decision"]:
        raise ValueError(f"{partition} authorization decision drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    if config.get("schema_version") != SCHEMA:
        raise ValueError("unexpected Gate 3C1D v1 eval-cache config")
    _validate_parent(config)
    for name, authority in config["implementation"].items():
        if file_sha256(authority["path"]) != authority["sha256"]:
            raise ValueError(f"Gate 3C1D v1 cache implementation drift: {name}")
    manifest = Path(config["partition"]["manifest"])
    if file_sha256(manifest) != config["partition"]["manifest_sha256"]:
        raise ValueError("Gate 3C1D v1 eval-cache manifest hash drift")
    for values in config["backbones"].values():
        weight = values.get("checkpoint") or values.get("weights")
        expected = values.get("checkpoint_sha256") or values.get("weights_sha256")
        if file_sha256(weight) != expected:
            raise ValueError("Gate 3C1D v1 frozen weight hash drift")

    partition = str(config["partition"]["name"])
    if partition not in PARTITIONS:
        raise ValueError("Gate 3C1D v1 cache partition is not authorized")
    _validate_runtime_authorization(config, partition)
    bounds = [int(value) for value in config["partition"]["source_indices"]]
    expected_sources = list(range(bounds[0], bounds[1] + 1))
    if expected_sources != PARTITIONS[partition]:
        raise ValueError("Gate 3C1D v1 source membership drift")
    if len(expected_sources) != int(config["partition"]["expected_videos"]):
        raise ValueError("Gate 3C1D v1 expected-video drift")

    # Heavy imports occur only after every authorization and membership check.
    import torch
    from cotracker.predictor import CoTrackerOnlinePredictor
    from transformers import AutoModel
    from scripts.audit_routeD_cotracker3_interface import set_deterministic
    from scripts.build_routeD_temporal_identity_train_cache_gate3c1a_v0 import (
        _atomic_torch_save,
        _build_video,
        verify_temporal_identity_cache_payload,
    )

    read_state = {
        key: bool(value) for key, value in config["partition"]["read_state"].items()
    }
    config_sha256 = file_sha256(config_path)
    device = str(args.device)
    set_deterministic(int(config["determinism_seed"]))
    predictor = CoTrackerOnlinePredictor(
        checkpoint=config["backbones"]["cotracker3"]["checkpoint"]
    ).to(device).eval()
    for parameter in predictor.model.parameters():
        parameter.requires_grad_(False)
    dino = AutoModel.from_pretrained(
        config["backbones"]["dinov3"]["model_dir"], local_files_only=True
    ).to(device).eval()
    for parameter in dino.parameters():
        parameter.requires_grad_(False)

    output_root = Path(args.output_root or config["cache"]["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for source_index in expected_sources:
        sidecar = output_root / f"video_{source_index:05d}.pt"
        if args.resume and sidecar.exists():
            payload = torch.load(sidecar, map_location="cpu", weights_only=False)
            verify_temporal_identity_cache_payload(
                payload,
                expected_partition=partition,
                expected_source_index=source_index,
                expected_config_sha256=config_sha256,
                expected_read_state=read_state,
            )
            stage = "resume_skip"
        else:
            payload = _build_video(
                config=config,
                config_path=config_path,
                config_sha256=config_sha256,
                predictor=predictor,
                dino=dino,
                source_index=source_index,
                device=device,
                partition_name=partition,
                read_state=read_state,
            )
            _atomic_torch_save(payload, sidecar)
            payload = torch.load(sidecar, map_location="cpu", weights_only=False)
            verify_temporal_identity_cache_payload(
                payload,
                expected_partition=partition,
                expected_source_index=source_index,
                expected_config_sha256=config_sha256,
                expected_read_state=read_state,
            )
            stage = "video_complete"
        tensors = payload["tensors"]
        rows = int(tensors["point_indices"].numel())
        if rows:
            distance = tensors["teacher_candidate_distance_px"].clone()
            distance[~tensors["candidate_valid_mask"]] = float("inf")
            oracle = distance.min(dim=1).values
            static = distance[:, :9].min(dim=1).values
            native = distance[:, 0]
        else:
            oracle = static = native = torch.empty(0)
        row = {
            "source_index": source_index,
            "video_name": payload["video_name"],
            "sidecar": str(sidecar),
            "sidecar_sha256": file_sha256(sidecar),
            "failure_rows": rows,
            "oracle_supported_12px_rows": int((oracle <= 12.0).sum()),
            "static_top8_supported_12px_rows": int((static <= 12.0).sum()),
            "native_supported_12px_rows": int((native <= 12.0).sum()),
            "tensor_hash_digest": payload["tensor_hash_digest"],
        }
        index_rows.append(row)
        print(json.dumps({"stage": stage, **row}), flush=True)
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    total_rows = sum(int(row["failure_rows"]) for row in index_rows)
    oracle_supported = sum(int(row["oracle_supported_12px_rows"]) for row in index_rows)
    static_supported = sum(int(row["static_top8_supported_12px_rows"]) for row in index_rows)
    native_supported = sum(int(row["native_supported_12px_rows"]) for row in index_rows)
    completed_sources = [int(row["source_index"]) for row in index_rows]
    checks = {
        "exact_source_membership": completed_sources == expected_sources,
        "exact_video_count": len(index_rows) == len(expected_sources),
        "all_sidecars_hash_verified": all(
            file_sha256(row["sidecar"]) == row["sidecar_sha256"] for row in index_rows
        ),
        "read_state_exact": all(
            bool(config["partition"]["read_state"][key]) is value
            for key, value in read_state.items()
        ),
    }
    passed = all(checks.values())
    index = {
        "schema_version": INDEX_SCHEMA,
        "date": "2026-07-20",
        "status": "completed_pass" if passed else "completed_fail",
        "pass": passed,
        "partition": partition,
        "config": str(config_path),
        "config_sha256": config_sha256,
        "manifest": str(manifest),
        "manifest_sha256": config["partition"]["manifest_sha256"],
        "expected_source_indices": expected_sources,
        "completed_source_indices": completed_sources,
        "videos": len(index_rows),
        "videos_with_failures": sum(int(row["failure_rows"] > 0) for row in index_rows),
        "failure_rows": total_rows,
        "descriptive_support": {
            "top128_oracle_supported_12px_rows": oracle_supported,
            "top128_oracle_recall_12px": 0.0 if total_rows == 0 else oracle_supported / total_rows,
            "static_native_plus_top8_supported_12px_rows": static_supported,
            "static_native_plus_top8_recall_12px": 0.0 if total_rows == 0 else static_supported / total_rows,
            "native_supported_12px_rows": native_supported,
            "native_recall_12px": 0.0 if total_rows == 0 else native_supported / total_rows,
        },
        "checks": checks,
        "combined_sidecar_sha256": canonical_json_sha256(
            [row["sidecar_sha256"] for row in index_rows]
        ),
        "combined_tensor_hash_digest": canonical_json_sha256(
            [row["tensor_hash_digest"] for row in index_rows]
        ),
        "rows": index_rows,
        "read_state": read_state,
    }
    index["index_payload_sha256"] = canonical_json_sha256(index)
    _atomic_json_save(index, output_root / "cache_index.json")
    print(json.dumps(index, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
