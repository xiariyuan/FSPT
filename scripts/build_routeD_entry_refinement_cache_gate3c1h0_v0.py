#!/usr/bin/env python3
"""Build the exposed full-population causal entry-refinement cache for Gate 3C1H0."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXT = Path('/gemini/code/FSPT')
COTRACKER = EXT / 'baselines/cotracker'
for item in (str(COTRACKER), str(EXT)):
    if item not in sys.path:
        sys.path.append(item)
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from cotracker.predictor import CoTrackerOnlinePredictor
from projects.mmp_tracker.mmp_tracker.cotracker3_stage0_adapter import tensor_sha256
from projects.mmp_tracker.mmp_tracker.routeD_counterfactual_state_restorer import build_csrr_trajectory_features
from projects.mmp_tracker.mmp_tracker.routeD_kubric_cache import canonical_json_sha256, file_sha256
from projects.mmp_tracker.mmp_tracker.routeD_temporal_identity_entry_v0 import (
    ENTRY_FEATURE_SCHEMA_VERSION,
    build_entry_features,
    entry_action_mask,
)
from scripts.audit_routeD_cotracker3_interface import load_manifest_sample, prepare_sample, set_deterministic
from scripts.audit_routeD_oracle_state_transplant_gate1 import (
    _continue_second_window,
    _coords_to_input,
    _initialize_and_first_window,
    _original_queries,
)
from scripts.build_routeD_counterfactual_state_restorer_cache import _selected_memory

DEFAULT_CONFIG = ROOT / 'configs/routeD_entry_refinement_cache_gate3c1h0_v0.yaml'
VIDEO_SCHEMA = 'routeD_entry_refinement_cache_video_gate3c1h0_v0'
INDEX_SCHEMA = 'routeD_entry_refinement_cache_index_gate3c1h0_v0'
CATEGORY_TO_CODE = {'other': 0, 'failure': 1, 'clean': 2, 'ambiguous': 3}


def _atomic_torch(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def _validate_config(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[int, dict[str, Any]]]:
    config = yaml.safe_load(path.read_text())
    if config.get('schema_version') != 'routeD_entry_refinement_cache_gate3c1h0_v0':
        raise ValueError('Gate 3C1H0 config schema drift')
    parent = config['authorized_parent']
    replay_path = Path(parent['replay_path'])
    if file_sha256(replay_path) != parent['replay_file_sha256']:
        raise ValueError('Gate 3C1H0 replay file drift')
    replay = json.loads(replay_path.read_text())
    without_hash = dict(replay)
    embedded = without_hash.pop('result_payload_sha256')
    if embedded != canonical_json_sha256(without_hash) or embedded != parent['replay_payload_sha256']:
        raise ValueError('Gate 3C1H0 replay payload drift')
    if not replay.get('exact_replay') or replay.get('gate', {}).get('decision') != 'STOP_BEFORE_OFFICIAL_TAPVID':
        raise ValueError('Gate 3C1H0 requires completed Gate 3C1F2 replay')
    for authority in config['implementation'].values():
        if file_sha256(Path(authority['path'])) != authority['sha256']:
            raise ValueError(f"Gate 3C1H0 implementation drift: {authority['path']}")
    for authority in config['assets'].values():
        if file_sha256(Path(authority['path'])) != authority['sha256']:
            raise ValueError(f"Gate 3C1H0 asset drift: {authority['path']}")
    reference_root = Path(parent['primary_work_root'])
    references: dict[int, dict[str, Any]] = {}
    for source in config['source_indices']:
        sidecar = reference_root / f'video_{int(source):05d}.json'
        payload = json.loads(sidecar.read_text())
        without_sidecar_hash = dict(payload)
        sidecar_hash = without_sidecar_hash.pop('result_payload_sha256')
        if sidecar_hash != canonical_json_sha256(without_sidecar_hash):
            raise ValueError('Gate 3C1H0 reference sidecar payload drift')
        references[int(source)] = payload
    if not all(value is False for value in config['locked_data'].values()):
        raise ValueError('Gate 3C1H0 locked-data drift')
    return config, replay, references


def _labels(prepared: Mapping[str, Any], native_xy: torch.Tensor, eligible: torch.Tensor) -> dict[str, torch.Tensor]:
    gt_xy = prepared['gt_tracks_yx'][..., [1, 0]].float() * 255.0
    occluded = prepared['gt_occluded'].bool()
    future_visible = ~occluded[eligible, 16:24]
    future_error = torch.linalg.vector_norm(native_xy[eligible, 16:24] - gt_xy[eligible, 16:24], dim=-1)
    future_count = future_visible.sum(dim=1)
    future_mean = (future_error * future_visible.float()).sum(dim=1) / future_count.clamp_min(1)
    commit_visible = ~occluded[eligible, 15]
    evaluable = commit_visible & (future_count >= 4)
    failure = evaluable & (future_mean >= 16.0)
    clean = evaluable & (future_mean <= 4.0)
    ambiguous = evaluable & (~failure) & (~clean)
    other = ~evaluable
    category = torch.zeros(len(eligible), dtype=torch.long)
    category[failure] = CATEGORY_TO_CODE['failure']
    category[clean] = CATEGORY_TO_CODE['clean']
    category[ambiguous] = CATEGORY_TO_CODE['ambiguous']
    return {
        'commit_visible': commit_visible.contiguous(),
        'future_visible_count': future_count.long().contiguous(),
        'future_mean_error_px': future_mean.float().contiguous(),
        'evaluable': evaluable.contiguous(),
        'failure_label': failure.contiguous(),
        'clean_label': clean.contiguous(),
        'ambiguous_label': ambiguous.contiguous(),
        'other_label': other.contiguous(),
        'category_code': category.contiguous(),
    }


def _run_video(*, config: Mapping[str, Any], config_path: Path, reference: Mapping[str, Any], entry_bundle: Mapping[str, Any], predictor: CoTrackerOnlinePredictor, source_index: int, device: str) -> dict[str, Any]:
    sample, metadata = load_manifest_sample(Path(config['data']['manifest']), source_index)
    prepared = prepare_sample(sample, int(config['backbone']['input_raster']))
    video = prepared['video'].to(device)
    set_deterministic(int(config['determinism']['seed']) + int(source_index))
    initial = _initialize_and_first_window(predictor, video, _original_queries(prepared, video))
    query_frames = prepared['query_points_tyx'][:, 0].round().long()
    eligible = torch.where(query_frames < int(config['entry_contract']['query_frame_less_than']))[0]
    trajectory = build_csrr_trajectory_features(initial, point_indices=eligible).float().cpu()
    visibility = torch.sigmoid(initial.online_vis_predicted[0, 15, eligible.to(initial.online_vis_predicted.device)]).float().cpu()
    confidence = torch.sigmoid(initial.online_conf_predicted[0, 15, eligible.to(initial.online_conf_predicted.device)]).float().cpu()
    native_features = _selected_memory(initial.online_track_feat, eligible, support=False)
    native_supports = _selected_memory(initial.online_track_support, eligible, support=True)
    features = build_entry_features(
        trajectory_features=trajectory,
        visibility_probability=visibility,
        confidence_probability=confidence,
        native_track_features=native_features,
        native_track_supports=native_supports,
    )
    if entry_bundle.get('feature_schema_version') != ENTRY_FEATURE_SCHEMA_VERSION:
        raise ValueError('Gate 3C1H0 entry schema drift')
    probability = entry_bundle['model'].predict_proba(features)[:, 1]
    joint = (visibility * confidence).numpy()
    operating = config['entry_contract']['operating_point']
    entry_mask = entry_action_mask(
        entry_probability=probability,
        native_joint_probability=joint,
        probability_min=float(operating['entry_probability_min']),
        joint_probability_max=float(operating['native_joint_probability_max']),
    )
    native_final = _continue_second_window(predictor, video, initial)
    native_xy = _coords_to_input(native_final, interp_height=int(predictor.interp_shape[0]), interp_width=int(predictor.interp_shape[1])).float().cpu()
    observed = {
        'eligible_indices': tensor_sha256(eligible.contiguous()),
        'entry_features': tensor_sha256(torch.from_numpy(features)),
        'entry_probability': tensor_sha256(torch.from_numpy(probability)),
        'entry_mask': tensor_sha256(torch.from_numpy(entry_mask)),
        'native_coordinates': tensor_sha256(native_xy),
    }
    expected = reference['scientific']['digests']
    checks = {key: observed[key] == expected[key] for key in observed}
    if not all(checks.values()):
        raise ValueError(f'Gate 3C1H0 sealed digest mismatch source {source_index}: {checks}')
    labels = _labels(prepared, native_xy, eligible)
    tensors = {
        'features': torch.from_numpy(features).float().contiguous(),
        'source_indices': torch.full((len(eligible),), int(source_index), dtype=torch.long),
        'point_indices': eligible.long().contiguous(),
        'query_frames': query_frames[eligible].long().contiguous(),
        'entry_probability': torch.from_numpy(probability).float().contiguous(),
        'native_joint_probability': torch.from_numpy(joint).float().contiguous(),
        'current_entry_mask': torch.from_numpy(entry_mask).bool().contiguous(),
        **labels,
    }
    tensor_hashes = {name: tensor_sha256(value) for name, value in tensors.items()}
    counts = {name: int(labels[f'{name}_label'].sum()) for name in ('failure', 'clean', 'ambiguous', 'other')}
    payload = {
        'schema_version': VIDEO_SCHEMA,
        'date': '2026-07-20',
        'status': 'completed',
        'config': str(config_path),
        'config_sha256': file_sha256(config_path),
        'source_index': int(source_index),
        'video_name': str(prepared['video_name']),
        'rows': int(len(eligible)),
        'feature_dim': int(features.shape[1]),
        'category_counts': counts,
        'current_entry_rows': int(entry_mask.sum()),
        'sealed_digest_checks': checks,
        'tensors': tensors,
        'tensor_hashes': tensor_hashes,
        'tensor_hash_digest': canonical_json_sha256(tensor_hashes),
        'sample_metadata': metadata,
    }
    payload['payload_sha256'] = canonical_json_sha256({key: value for key, value in payload.items() if key != 'tensors'})
    del video
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    return payload


def verify_payload(payload: Mapping[str, Any], *, source_index: int) -> None:
    if payload.get('schema_version') != VIDEO_SCHEMA or int(payload.get('source_index', -1)) != int(source_index):
        raise ValueError('Gate 3C1H0 video identity drift')
    if int(payload.get('feature_dim', -1)) != 130:
        raise ValueError('Gate 3C1H0 feature dimension drift')
    if not all(payload.get('sealed_digest_checks', {}).values()):
        raise ValueError('Gate 3C1H0 sealed digest failure')
    tensors = payload['tensors']
    rows = int(payload['rows'])
    if tensors['features'].shape != (rows, 130):
        raise ValueError('Gate 3C1H0 feature shape drift')
    actual = {name: tensor_sha256(value) for name, value in tensors.items()}
    if actual != payload.get('tensor_hashes') or canonical_json_sha256(actual) != payload.get('tensor_hash_digest'):
        raise ValueError('Gate 3C1H0 tensor hash drift')
    without = {key: value for key, value in payload.items() if key not in ('tensors', 'payload_sha256')}
    if canonical_json_sha256(without) != payload.get('payload_sha256'):
        raise ValueError('Gate 3C1H0 payload drift')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--output-root', default=None)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--source-index', action='append', type=int, default=None)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config, _, references = _validate_config(config_path)
    sources = [int(value) for value in (args.source_index if args.source_index is not None else config['source_indices'])]
    output_root = Path(args.output_root or config['output_root']).resolve(); output_root.mkdir(parents=True, exist_ok=True)
    entry_bundle = joblib.load(config['assets']['entry_bundle']['path'])
    predictor = CoTrackerOnlinePredictor(checkpoint=config['assets']['checkpoint']['path']).to(args.device).eval()
    for parameter in predictor.model.parameters(): parameter.requires_grad_(False)
    rows = []
    for source in sources:
        sidecar = output_root / f'video_{source:05d}.pt'
        if sidecar.exists() and args.resume:
            payload = torch.load(sidecar, map_location='cpu', weights_only=False); verify_payload(payload, source_index=source)
        else:
            payload = _run_video(config=config, config_path=config_path, reference=references[source], entry_bundle=entry_bundle, predictor=predictor, source_index=source, device=args.device)
            _atomic_torch(payload, sidecar); payload = torch.load(sidecar, map_location='cpu', weights_only=False); verify_payload(payload, source_index=source)
        rows.append({
            'source_index': source, 'video_name': payload['video_name'], 'sidecar': str(sidecar),
            'sidecar_sha256': file_sha256(sidecar), 'rows': payload['rows'],
            'category_counts': payload['category_counts'], 'current_entry_rows': payload['current_entry_rows'],
            'tensor_hash_digest': payload['tensor_hash_digest'], 'payload_sha256': payload['payload_sha256'],
        })
        print(json.dumps({'stage':'video_complete','source_index':source,'rows':payload['rows'],'categories':payload['category_counts'],'current_entry_rows':payload['current_entry_rows']}), flush=True)
    category_counts = {name: sum(row['category_counts'][name] for row in rows) for name in CATEGORY_TO_CODE}
    index = {
        'schema_version': INDEX_SCHEMA, 'date':'2026-07-20', 'status':'completed',
        'config':str(config_path), 'config_sha256':file_sha256(config_path),
        'source_indices':sources, 'videos':len(rows), 'rows_total':sum(row['rows'] for row in rows),
        'feature_dim':130, 'category_counts':category_counts,
        'current_entry_rows':sum(row['current_entry_rows'] for row in rows),
        'rows':rows,
        'combined_sidecar_digest':canonical_json_sha256([row['sidecar_sha256'] for row in rows]),
        'combined_tensor_digest':canonical_json_sha256([row['tensor_hash_digest'] for row in rows]),
        'locked_data':config['locked_data'],
        'claim_boundary':config['claim_scope'],
    }
    if args.source_index is None:
        expected=config['expected_support']
        if (index['videos'],index['rows_total'],index['feature_dim']) != (int(expected['videos']),int(expected['rows']),int(expected['feature_dim'])):
            raise ValueError('Gate 3C1H0 aggregate support drift')
    index['index_payload_sha256']=canonical_json_sha256(index)
    _atomic_json(index, output_root/'cache_index.json')
    print(json.dumps(index,indent=2),flush=True)

if __name__ == '__main__':
    main()
