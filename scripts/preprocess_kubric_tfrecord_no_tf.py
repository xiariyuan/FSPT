#!/usr/bin/env python3
"""Convert MOVi-E TFRecord shards to MMP Kubric sharded pickle without TensorFlow.

The converter decodes only the fields required by the point-tracking pipeline:
RGB video, forward/backward flow, segmentation masks, and flow ranges. Sparse
tracks are generated with the same helper used by the TensorFlow preprocessing
path, so the output is consumable by TAPVidKubricShardedIterableDataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import struct
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

import cv2
import numpy as np
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.tapvid_kubric import (  # noqa: E402
    _decode_kubric_flow,
    _generate_sparse_tracks_from_kubric,
)


def _add_field(message, name, number, field_type, *, label=1, type_name=None, oneof=None):
    field = message.field.add()
    field.name = name
    field.number = number
    field.type = field_type
    field.label = label
    if type_name is not None:
        field.type_name = type_name
    if oneof is not None:
        field.oneof_index = oneof
    return field


def build_example_class():
    """Build the minimal tensorflow.Example protobuf schema dynamically."""
    file_descriptor = descriptor_pb2.FileDescriptorProto()
    file_descriptor.name = "tensorflow_example.proto"
    file_descriptor.package = "tensorflow"
    file_descriptor.syntax = "proto3"

    def add_message(name):
        message = file_descriptor.message_type.add()
        message.name = name
        return message

    message = add_message("BytesList")
    _add_field(message, "value", 1, descriptor_pb2.FieldDescriptorProto.TYPE_BYTES, label=3)

    message = add_message("FloatList")
    field = _add_field(message, "value", 1, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT, label=3)
    field.options.packed = True

    message = add_message("Int64List")
    field = _add_field(message, "value", 1, descriptor_pb2.FieldDescriptorProto.TYPE_INT64, label=3)
    field.options.packed = True

    message = add_message("Feature")
    message.oneof_decl.add().name = "kind"
    _add_field(
        message,
        "bytes_list",
        1,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=".tensorflow.BytesList",
        oneof=0,
    )
    _add_field(
        message,
        "float_list",
        2,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=".tensorflow.FloatList",
        oneof=0,
    )
    _add_field(
        message,
        "int64_list",
        3,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=".tensorflow.Int64List",
        oneof=0,
    )

    message = add_message("Features")
    entry = message.nested_type.add()
    entry.name = "FeatureEntry"
    entry.options.map_entry = True
    _add_field(entry, "key", 1, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
    _add_field(
        entry,
        "value",
        2,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=".tensorflow.Feature",
    )
    _add_field(
        message,
        "feature",
        1,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        label=3,
        type_name=".tensorflow.Features.FeatureEntry",
    )

    message = add_message("Example")
    _add_field(
        message,
        "features",
        1,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        type_name=".tensorflow.Features",
    )

    pool = descriptor_pool.DescriptorPool()
    pool.Add(file_descriptor)
    descriptor = pool.FindMessageTypeByName("tensorflow.Example")
    return message_factory.MessageFactory(pool).GetPrototype(descriptor)


EXAMPLE_CLASS = build_example_class()


def iter_tfrecord(path: Path) -> Iterator[bytes]:
    """Yield serialized Example records from an uncompressed TFRecord file."""
    with path.open("rb") as handle:
        record_index = 0
        while True:
            header = handle.read(12)
            if not header:
                return
            if len(header) != 12:
                raise IOError(f"Truncated TFRecord header in {path} at record {record_index}.")
            length = struct.unpack("<Q", header[:8])[0]
            payload = handle.read(length)
            data_crc = handle.read(4)
            if len(payload) != length or len(data_crc) != 4:
                raise IOError(f"Truncated TFRecord payload in {path} at record {record_index}.")
            yield payload
            record_index += 1


def feature_values(features: Mapping[str, object], key: str, kind: str):
    if key not in features:
        raise KeyError(f"Missing required TFDS feature: {key}")
    feature = features[key]
    actual_kind = feature.WhichOneof("kind")
    if actual_kind != kind:
        raise TypeError(f"Feature {key!r} has {actual_kind}, expected {kind}.")
    return getattr(feature, kind).value


def decode_png_sequence(encoded_frames: Sequence[bytes], *, color: bool) -> np.ndarray:
    frames: List[np.ndarray] = []
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_UNCHANGED
    for frame_index, encoded in enumerate(encoded_frames):
        frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), flag)
        if frame is None:
            raise ValueError(f"Failed to decode PNG frame {frame_index}.")
        if color:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif frame.ndim == 3 and frame.shape[-1] == 1:
            frame = frame[..., 0]
        frames.append(frame)
    return np.stack(frames, axis=0)


def decode_required_fields(serialized_example: bytes) -> Dict[str, np.ndarray]:
    example = EXAMPLE_CLASS()
    example.ParseFromString(serialized_example)
    features = example.features.feature

    num_frames = int(feature_values(features, "metadata/num_frames", "int64_list")[0])
    height = int(feature_values(features, "metadata/height", "int64_list")[0])
    width = int(feature_values(features, "metadata/width", "int64_list")[0])

    video = decode_png_sequence(feature_values(features, "video", "bytes_list"), color=True)
    segmentations = decode_png_sequence(
        feature_values(features, "segmentations", "bytes_list"), color=False
    )

    expected_flow_values = num_frames * height * width * 2
    forward_values = feature_values(features, "forward_flow", "int64_list")
    backward_values = feature_values(features, "backward_flow", "int64_list")
    if len(forward_values) != expected_flow_values or len(backward_values) != expected_flow_values:
        raise ValueError(
            "Unexpected flow length: "
            f"forward={len(forward_values)}, backward={len(backward_values)}, "
            f"expected={expected_flow_values}."
        )
    forward_flow = np.asarray(forward_values, dtype=np.uint16).reshape(
        num_frames, height, width, 2
    )
    backward_flow = np.asarray(backward_values, dtype=np.uint16).reshape(
        num_frames, height, width, 2
    )

    forward_range = np.asarray(
        feature_values(features, "metadata/forward_flow_range", "float_list"),
        dtype=np.float32,
    )
    backward_range = np.asarray(
        feature_values(features, "metadata/backward_flow_range", "float_list"),
        dtype=np.float32,
    )
    video_name_values = feature_values(features, "metadata/video_name", "bytes_list")
    video_name = video_name_values[0].decode("utf-8", errors="replace") if video_name_values else ""

    if video.shape != (num_frames, height, width, 3):
        raise ValueError(f"Unexpected video shape: {video.shape}")
    if segmentations.shape[:3] != (num_frames, height, width):
        raise ValueError(f"Unexpected segmentation shape: {segmentations.shape}")

    return {
        "video": video.astype(np.uint8, copy=False),
        "segmentations": segmentations,
        "forward_flow": forward_flow,
        "backward_flow": backward_flow,
        "forward_flow_range": forward_range,
        "backward_flow_range": backward_range,
        "video_name": video_name,
    }


def convert_record(
    serialized_example: bytes,
    *,
    num_points: int,
    sampling_strategy: str,
    hard_fraction: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    decoded = decode_required_fields(serialized_example)
    forward_flow = _decode_kubric_flow(
        decoded["forward_flow"], decoded["forward_flow_range"]
    )
    backward_flow = _decode_kubric_flow(
        decoded["backward_flow"], decoded["backward_flow_range"]
    )
    query_points, target_points, occluded = _generate_sparse_tracks_from_kubric(
        forward_flow=forward_flow,
        backward_flow=backward_flow,
        segmentations=decoded["segmentations"],
        num_points=int(num_points),
        rng=np.random.RandomState(int(seed) % (2**32)),
        sampling_strategy=str(sampling_strategy),
        hard_fraction=float(hard_fraction),
    )
    return {
        "video": decoded["video"],
        "query_points": np.asarray(query_points, dtype=np.float32),
        "target_points": np.asarray(target_points, dtype=np.float32),
        "occluded": np.asarray(occluded, dtype=bool),
        "video_name": str(decoded["video_name"]),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess MOVi-E TFRecord shards without TensorFlow."
    )
    parser.add_argument("--tfds-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--num-points", type=int, default=256)
    parser.add_argument("--shard-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sampling-strategy", default="uniform")
    parser.add_argument("--hard-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hash-source-files", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tfds_root = Path(args.tfds_root).resolve()
    builder_dir = tfds_root / "256x256" / "1.0.0"
    if not builder_dir.exists():
        builder_dir = tfds_root
    source_files = sorted(builder_dir.glob(f"movi_e-{args.split}.tfrecord-*"))
    if not source_files:
        raise FileNotFoundError(
            f"No TFRecord shards found for split={args.split} under {builder_dir}."
        )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    max_samples = None if args.max_samples is None else max(0, int(args.max_samples))
    shard_size = max(1, int(args.shard_size))

    shard_items: List[Dict[str, np.ndarray]] = []
    shard_manifest: List[Dict[str, object]] = []
    source_counts: Dict[str, int] = defaultdict(int)
    used_source_files: List[Path] = []
    processed = 0
    output_shard_index = 0

    def flush_shard() -> None:
        nonlocal shard_items, output_shard_index
        if not shard_items:
            return
        name = f"{args.split}_{output_shard_index:05d}.pkl"
        path = output_dir / name
        with path.open("wb") as handle:
            pickle.dump(shard_items, handle, protocol=4)
        shard_manifest.append(
            {
                "path": name,
                "num_samples": len(shard_items),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        shard_items = []
        output_shard_index += 1

    for source_path in source_files:
        if max_samples is not None and processed >= max_samples:
            break
        used_source_files.append(source_path)
        for record_index, serialized in enumerate(iter_tfrecord(source_path)):
            if max_samples is not None and processed >= max_samples:
                break
            sample_seed = (int(args.seed) + processed * 1013) % (2**32)
            converted = convert_record(
                serialized,
                num_points=int(args.num_points),
                sampling_strategy=str(args.sampling_strategy),
                hard_fraction=float(args.hard_fraction),
                seed=sample_seed,
            )
            converted["source_tfrecord"] = source_path.name
            converted["source_record_index"] = int(record_index)
            shard_items.append(converted)
            source_counts[source_path.name] += 1
            processed += 1
            print(
                f"[sample] split={args.split} index={processed - 1} "
                f"source={source_path.name}:{record_index} "
                f"video={converted['video'].shape} points={converted['target_points'].shape}"
            )
            if len(shard_items) >= shard_size:
                flush_shard()

    flush_shard()

    source_manifest = []
    for source_path in used_source_files:
        entry: Dict[str, object] = {
            "path": str(source_path),
            "size_bytes": source_path.stat().st_size,
            "records_used": int(source_counts.get(source_path.name, 0)),
        }
        if args.hash_source_files:
            entry["sha256"] = sha256_file(source_path)
        source_manifest.append(entry)

    manifest = {
        "version": 1,
        "dataset": "tapvid_kubric",
        "source": "movi_e_tfrecord_without_tensorflow",
        "split": str(args.split),
        "num_samples": int(processed),
        "num_shards": len(shard_manifest),
        "num_points": int(args.num_points),
        "num_frames": 24,
        "resolution": [256, 256],
        "sampling_strategy": str(args.sampling_strategy),
        "hard_fraction": float(args.hard_fraction),
        "seed": int(args.seed),
        "flow_component_order": ["delta_row", "delta_col"],
        "coordinate_order": ["y", "x"],
        "coordinate_normalization": "pixel_center_divide_by_size_minus_one",
        "tfrecord_crc_validation": False,
        "source_files": source_manifest,
        "shards": shard_manifest,
    }
    manifest_path = output_dir / f"{args.split}.index.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print(f"[done] samples={processed} shards={len(shard_manifest)}")
    print(f"[done] manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
