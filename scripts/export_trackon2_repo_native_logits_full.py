#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils import data

ROOT = Path('/gemini/code/FSPT')
TRACKON_DIR = ROOT / 'baselines/track_on'
OLD_NPZ_DIR_DEFAULT = ROOT / 'outputs/trackon2_dinov3_davis_cache/davis/trackon2'
OLD_PT_DEFAULT = ROOT / 'outputs/attempt0_2026-06-15_recovery/prediction_caches/trackon2_dinov3_davis_first_input_bridge.pt'
OUT_NPZ_DIR_DEFAULT = ROOT / 'outputs/paper_discovery_2026-07-05/trackon2_true_base_conf/repo_native_logit_full/davis/trackon2'
OUT_REPORT_DEFAULT = ROOT / 'outputs/paper_discovery_2026-07-05/trackon2_true_base_conf/repo_native_logit_full_report.json'
OUT_BRIDGE_DEFAULT = ROOT / 'outputs/paper_discovery_2026-07-05/trackon2_true_base_conf/trackon2_davis_first_input_repo_native_conf_full.pt'
OUT_METRIC_COMPARE_DEFAULT = ROOT / 'outputs/paper_discovery_2026-07-05/trackon2_true_base_conf/trackon2_davis_first_input_repo_native_conf_full_metric_compare.json'
DAVIS_PKL_DEFAULT = ROOT / 'datasets/tapvid_davis/tapvid_davis.pkl'
CKPT_DEFAULT = TRACKON_DIR / 'checkpoints_trackon2_dinov3.pt'
CONFIG_DEFAULT = TRACKON_DIR / 'config/test.yaml'
DINOV3_LOCAL_DIR_DEFAULT = ROOT / 'third_party_weights/dinov3/facebook_dinov3_vits16plus_pretrain_lvd1689m'

# Repo-native imports require this root on sys.path.
os.chdir(str(TRACKON_DIR))
sys.path.insert(0, str(TRACKON_DIR))
sys.path.append(str(ROOT))

from dataset.tapvid import TAPVid  # noqa: E402
from model.trackon_predictor import Predictor  # noqa: E402
from utils.coord_utils import get_points_on_a_grid  # noqa: E402
from utils.train_utils import load_args_from_yaml  # noqa: E402


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def prepare_tapvid_data_local(trajectory, visibility, query_points_i, device):
    """Local copy of evaluation.evaluator.prepare_tapvid_data to avoid wandb import."""
    queries = query_points_i.clone().float()
    queries = torch.stack([queries[:, :, 0], queries[:, :, 2], queries[:, :, 1]], dim=2).to(device)
    traj = trajectory.clone()
    query_points = query_points_i.clone().cpu().numpy()
    gt_tracks = traj.permute(0, 2, 1, 3).cpu().numpy()
    gt_occluded = torch.logical_not(visibility.clone().permute(0, 2, 1)).cpu().numpy()
    return queries, gt_tracks, gt_occluded, query_points


class LogitPredictor(Predictor):
    """Local-only Predictor subclass that preserves default Predictor behavior.

    The original Predictor class is not modified. This subclass only exposes
    forward_with_logits for the current export script.
    """

    @torch.no_grad()
    def forward_frame_with_logits(self, frame, new_queries=None):
        device = frame.device
        _, _, H, W = frame.shape
        f4_t, f8_t, f16_t, f32_t, f_fused_t = self.model.extract_frame_features(frame)
        if new_queries is not None and new_queries.shape[0] > 0:
            self.init_queries((f_fused_t, device), new_queries, H, W)
        if self.q_init is None or self.N == 0:
            return (
                torch.empty(0, 2, device=device),
                torch.empty(0, dtype=torch.bool, device=device),
                torch.empty(0, dtype=torch.float32, device=device),
            )
        frame_features = (f4_t, f8_t, f16_t, f32_t, f_fused_t)
        p, v_logit, q_new = self.model.track_frame(
            self.q_init[: self.N],
            self.temporal_mask[: self.N],
            self.point_memory[: self.N],
            frame_features,
            H,
            W,
        )
        v_t = (v_logit.sigmoid() >= self.delta_v)
        if new_queries is not None and new_queries.shape[0] > 0:
            new_count = int(new_queries.shape[0])
            new_query_mask = torch.zeros(self.N, dtype=torch.bool, device=device)
            new_query_mask[self.N - new_count : self.N] = True
        else:
            new_query_mask = torch.zeros(self.N, dtype=torch.bool, device=device)
        if self.model.memory_update_policy == 'visibility_selective':
            write_mask = v_t | new_query_mask
        elif self.model.memory_update_policy == 'unconditional':
            write_mask = torch.ones(self.N, dtype=torch.bool, device=device)
        else:
            raise ValueError(f'Unknown memory_update_policy: {self.model.memory_update_policy}')
        updated_memory, updated_mask = self.model._update_point_memory(
            self.point_memory[: self.N],
            self.temporal_mask[: self.N],
            q_new,
            write_mask,
        )
        self.point_memory[: self.N] = updated_memory
        self.temporal_mask[: self.N] = updated_mask
        self.t += 1
        return p, v_t, v_logit.float()

    @torch.no_grad()
    def forward_with_logits(self, video, queries):
        _, T, _, H, W = video.shape
        device = video.device
        queries = queries.squeeze(0)
        N_orig = queries.shape[0]
        query_times = queries[:, 0].long()
        query_coords = queries[:, 1:]
        if self.support_grid_size > 0:
            extra = get_points_on_a_grid(self.support_grid_size, (H, W), device).squeeze(0)
            extra_queries = torch.cat([torch.zeros(extra.shape[0], 1, device=device), extra], dim=1)
            queries = torch.cat([queries, extra_queries], dim=0)
            query_times = queries[:, 0].long()
            query_coords = queries[:, 1:]
        N_total = queries.shape[0]
        self.reset()
        self.initial_capacity = N_total
        pred_trajectory = torch.zeros(1, T, N_orig, 2, device='cpu', dtype=torch.float32)
        pred_visibility = torch.zeros(1, T, N_orig, device='cpu', dtype=torch.bool)
        pred_vis_logit = torch.zeros(1, T, N_orig, device='cpu', dtype=torch.float32)
        tracking_to_original: list[int] = []
        for t in range(T):
            frame = video[0, t].unsqueeze(0)
            new_queries_mask = query_times == t
            new_queries_this_frame = None
            if new_queries_mask.any():
                new_queries_this_frame = query_coords[new_queries_mask]
                new_indices = new_queries_mask.nonzero(as_tuple=True)[0]
                tracking_to_original.extend(new_indices.tolist())
            if self.N > 0 or new_queries_this_frame is not None:
                p_t, v_t, vlogit_t = self.forward_frame_with_logits(frame, new_queries=new_queries_this_frame)
                if self.N > 0:
                    tracking_to_original_tensor = torch.tensor(tracking_to_original, dtype=torch.long, device='cpu')
                    orig_mask = tracking_to_original_tensor < N_orig
                    if orig_mask.any():
                        orig_indices = tracking_to_original_tensor[orig_mask]
                        pred_trajectory[0, t, orig_indices] = p_t[orig_mask].cpu()
                        pred_visibility[0, t, orig_indices] = v_t[orig_mask].cpu()
                        pred_vis_logit[0, t, orig_indices] = vlogit_t[orig_mask].cpu()
                    del p_t, v_t, vlogit_t, tracking_to_original_tensor, orig_mask
                    if new_queries_this_frame is not None:
                        del new_queries_this_frame
            del frame
        pred_vis_conf = torch.sigmoid(pred_vis_logit)
        return pred_trajectory, pred_visibility, pred_vis_logit, pred_vis_conf


def summarize_diff(old: np.ndarray, new: np.ndarray, *, bool_like: bool = False) -> dict:
    if old.shape != new.shape:
        return {'same_shape': False, 'old_shape': list(old.shape), 'new_shape': list(new.shape)}
    if bool_like:
        diff = old.astype(bool) != new.astype(bool)
        return {'same_shape': True, 'diff_count': int(diff.sum()), 'diff_rate': float(diff.mean()) if diff.size else 0.0}
    d = np.abs(old.astype(np.float64) - new.astype(np.float64))
    return {
        'same_shape': True,
        'max_abs_diff': float(d.max()) if d.size else 0.0,
        'mean_abs_diff': float(d.mean()) if d.size else 0.0,
        'p95_abs_diff': float(np.percentile(d, 95)) if d.size else 0.0,
    }


def bridge_npz_to_record(old_record: dict, npz_path: Path) -> dict:
    z = np.load(npz_path)
    rec = dict(old_record)
    rec['pred_tracks'] = z['tracks'][0].transpose(1, 0, 2)[..., [1, 0]].astype(np.float32) / 255.0
    rec['pred_visibility'] = z['visibility'][0].transpose(1, 0).astype(bool)
    rec['pred_vis_logit'] = z['visibility_logit'][0].transpose(1, 0).astype(np.float32)
    rec['pred_vis_conf'] = z['visibility_conf'][0].transpose(1, 0).astype(np.float32)
    rec['adapter_version'] = 'trackon2_repo_native_logit_full_bridge_v1'
    rec['raw_coordinate_note'] = 'Repo-native logit export; tracks from npz input256 xy -> normalized yx/255; old query/GT/schema preserved.'
    return rec


def eval_metric_compare(old_pt: Path, new_pt: Path, out_json: Path) -> None:
    cmd = [
        sys.executable,
        str(ROOT / 'scripts/eval_external_baseline_smoke_cache.py'),
        '--items',
        f'old_trackon2={old_pt}',
        f'repo_native_conf_full={new_pt}',
        '--max-records',
        '0',
        '--out-json',
        str(out_json),
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-videos', type=int, default=0, help='0 means all videos')
    ap.add_argument('--old-npz-dir', default=str(OLD_NPZ_DIR_DEFAULT))
    ap.add_argument('--old-pt', default=str(OLD_PT_DEFAULT))
    ap.add_argument('--out-npz-dir', default=str(OUT_NPZ_DIR_DEFAULT))
    ap.add_argument('--out-report', default=str(OUT_REPORT_DEFAULT))
    ap.add_argument('--out-bridge', default=str(OUT_BRIDGE_DEFAULT))
    ap.add_argument('--out-metric-compare', default=str(OUT_METRIC_COMPARE_DEFAULT))
    ap.add_argument('--davis-pkl', default=str(DAVIS_PKL_DEFAULT))
    ap.add_argument('--checkpoint', default=str(CKPT_DEFAULT))
    ap.add_argument('--config', default=str(CONFIG_DEFAULT))
    ap.add_argument('--dinov3-local-dir', default=str(DINOV3_LOCAL_DIR_DEFAULT))
    args = ap.parse_args()

    dinov3_dir = Path(args.dinov3_local_dir)
    if dinov3_dir.is_dir():
        os.environ['DINOV3_LOCAL_DIR'] = str(dinov3_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    old_npz_dir = Path(args.old_npz_dir)
    out_npz_dir = Path(args.out_npz_dir)
    out_npz_dir.mkdir(parents=True, exist_ok=True)
    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_bridge = Path(args.out_bridge)
    out_bridge.parent.mkdir(parents=True, exist_ok=True)
    out_metric_compare = Path(args.out_metric_compare)
    out_metric_compare.parent.mkdir(parents=True, exist_ok=True)

    model_args = load_args_from_yaml(str(args.config))
    model_args.grad_checkpoint = False
    model_args.M_i = 24  # repo-native DAVIS setting in evaluation/eval.py
    model = LogitPredictor(model_args, checkpoint_path=str(args.checkpoint), support_grid_size=20).to(device).eval()

    dataset = TAPVid(None, data_root=str(args.davis_pkl), dataset_type='davis')
    loader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    old_payload = torch.load(args.old_pt, map_location='cpu', weights_only=False)
    old_records = old_payload['records']
    max_videos = len(dataset) if int(args.max_videos) <= 0 else min(len(dataset), int(args.max_videos))

    per_video = []
    bridge_records = []
    start_all = time.time()

    for j, (video, trajectory, visibility, query_points_i) in enumerate(loader):
        if j >= max_videos:
            break
        t0 = time.time()
        query_points_i = query_points_i.to(device, non_blocking=True)
        trajectory = trajectory.to(device, non_blocking=True)
        visibility = visibility.to(device, non_blocking=True)
        video = video.to(device, non_blocking=True)
        queries, _, _, _ = prepare_tapvid_data_local(trajectory, visibility, query_points_i, device)
        with torch.no_grad():
            if device.type == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    tracks, vis, vlogit, vconf = model.forward_with_logits(video.clone(), queries.clone())
            else:
                tracks, vis, vlogit, vconf = model.forward_with_logits(video.clone(), queries.clone())
        out_npz = out_npz_dir / f'{j:06d}.npz'
        np.savez(
            out_npz,
            tracks=tracks.cpu().numpy(),
            visibility=vis.cpu().numpy(),
            visibility_logit=vlogit.cpu().numpy(),
            visibility_conf=vconf.cpu().numpy(),
        )
        old_npz = old_npz_dir / f'{j:06d}.npz'
        z_old = np.load(old_npz)
        z_new = np.load(out_npz)
        diff_tracks = summarize_diff(z_old['tracks'], z_new['tracks'])
        diff_visibility = summarize_diff(z_old['visibility'], z_new['visibility'], bool_like=True)
        eq_conf = bool(np.all(z_new['visibility'].astype(bool) == (z_new['visibility_conf'] >= 0.8)))
        # Bridge with old schema.
        bridge_records.append(bridge_npz_to_record(old_records[j], out_npz))
        per_video.append({
            'index': int(j),
            'video_id': str(old_records[j].get('video_id', f'{j:06d}')),
            'old_npz': str(old_npz),
            'new_npz': str(out_npz),
            'infer_sec': round(float(time.time() - t0), 3),
            'diff_tracks': diff_tracks,
            'diff_visibility': diff_visibility,
            'visibility_equals_conf_ge_0p8': eq_conf,
            'conf_min': float(z_new['visibility_conf'].min()),
            'conf_max': float(z_new['visibility_conf'].max()),
            'conf_mean': float(z_new['visibility_conf'].mean()),
            'logit_min': float(z_new['visibility_logit'].min()),
            'logit_max': float(z_new['visibility_logit'].max()),
            'logit_mean': float(z_new['visibility_logit'].mean()),
        })
        print(json.dumps({k: per_video[-1][k] for k in ['index', 'video_id', 'infer_sec', 'diff_visibility', 'visibility_equals_conf_ge_0p8']}, ensure_ascii=False), flush=True)

    new_payload = dict(old_payload)
    new_payload['model_name'] = 'trackon2_repo_native_conf_full'
    new_payload['schema_version'] = 'trackon2_repo_native_conf_full_v1'
    new_payload['records'] = bridge_records
    new_payload['logit_export'] = {
        'source': 'repo-native TAPVid dataloader + local LogitPredictor subclass',
        'out_npz_dir': str(out_npz_dir),
        'old_npz_dir': str(old_npz_dir),
        'M_i': 24,
        'delta_v': float(model_args.delta_v),
        'support_grid_size': 20,
        'uses_gt_at_inference': False,
    }
    torch.save(new_payload, out_bridge)

    eval_metric_compare(Path(args.old_pt), out_bridge, out_metric_compare)
    metric_compare = json.loads(out_metric_compare.read_text())

    # Full summary.
    vis_diff_counts = [v['diff_visibility'].get('diff_count', 0) for v in per_video]
    track_max = [v['diff_tracks'].get('max_abs_diff', None) for v in per_video if v['diff_tracks'].get('max_abs_diff', None) is not None]
    conf_mismatch = []
    for v in per_video:
        # Use already computed eq flag for per-video; count exact mismatches for total.
        z = np.load(v['new_npz'])
        conf_mismatch.append(int(np.sum(z['visibility'].astype(bool) != (z['visibility_conf'] >= 0.8))))
    report = {
        'script': 'scripts/export_trackon2_repo_native_logits_full.py',
        'device': str(device),
        'dataset': str(args.davis_pkl),
        'checkpoint': str(args.checkpoint),
        'config': str(args.config),
        'dinov3_local_dir': os.environ.get('DINOV3_LOCAL_DIR', ''),
        'old_npz_dir': str(old_npz_dir),
        'out_npz_dir': str(out_npz_dir),
        'old_pt': str(args.old_pt),
        'out_bridge': str(out_bridge),
        'out_metric_compare': str(out_metric_compare),
        'num_videos': len(per_video),
        'total_sec': round(float(time.time() - start_all), 2),
        'model_args': {
            'M': int(model_args.M),
            'M_i': int(model_args.M_i),
            'delta_v': float(model_args.delta_v),
            'memory_update_policy': str(model_args.memory_update_policy),
            'support_grid_size': 20,
        },
        'aggregate_raw_npz_diff': {
            'visibility_diff_count_total': int(sum(vis_diff_counts)),
            'visibility_diff_rate_video_mean': float(np.mean([v['diff_visibility'].get('diff_rate', 0.0) for v in per_video])) if per_video else None,
            'track_max_abs_diff_max': float(max(track_max)) if track_max else None,
            'track_max_abs_diff_median': float(np.median(track_max)) if track_max else None,
            'visibility_conf_threshold_mismatch_total': int(sum(conf_mismatch)),
        },
        'metric_compare': metric_compare,
        'per_video': per_video,
    }
    rows = metric_compare.get('rows', [])
    if len(rows) >= 2:
        old_row, new_row = rows[0], rows[1]
        report['full_metric_delta_new_minus_old'] = {
            'AJ_256': float(new_row.get('AJ_256')) - float(old_row.get('AJ_256')),
            'OA_256': float(new_row.get('OA_256')) - float(old_row.get('OA_256')),
            'delta_avg_256': float(new_row.get('delta_avg_256')) - float(old_row.get('delta_avg_256')),
            'AJ_RD_256': float(new_row.get('AJ_RD_256')) - float(old_row.get('AJ_RD_256')),
            'AJ_RD': float(new_row.get('AJ_RD')) - float(old_row.get('AJ_RD')),
        }
        d = report['full_metric_delta_new_minus_old']
        report['pass_full_metric_parity_preferred_gate'] = bool(
            abs(d['AJ_256']) <= 0.10
            and abs(d['OA_256']) <= 0.10
            and abs(d['delta_avg_256']) <= 0.10
            and abs(d['AJ_RD_256']) <= 0.005
        )
    out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({
        'out_report': str(out_report),
        'out_bridge': str(out_bridge),
        'out_metric_compare': str(out_metric_compare),
        'num_videos': len(per_video),
        'full_metric_delta_new_minus_old': report.get('full_metric_delta_new_minus_old'),
        'pass_full_metric_parity_preferred_gate': report.get('pass_full_metric_parity_preferred_gate'),
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
