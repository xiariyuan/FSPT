#!/usr/bin/env python3
"""Strided + Original Resolution evaluation for FSPT Attempt 0.

This script runs the true strided query protocol with original video resolution,
producing caches that are NOT derivable from the first+input cache.
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import subprocess
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_model_via_subprocess(
    model_name: str,
    dataset_path: str,
    cache_dir: str,
    output_json: str,
    dinov3_local_dir: str = "",
):
    """Run model evaluation via subprocess (bypasses sys.path conflicts)."""

    # Base imports and setup - these become literal in the generated script
    base_setup = """
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path("/gemini/code/FSPT")
# Insert in priority order: cotracker root first, then track_on, then project
for _p in [str(PROJECT_ROOT), str(PROJECT_ROOT / "baselines" / "track_on"), str(PROJECT_ROOT / "baselines" / "cotracker")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(str(PROJECT_ROOT))

if "{dinov3_local_dir}":
    os.environ["DINOV3_LOCAL_DIR"] = "{dinov3_local_dir}"
"""

    if model_name == "trackon2":
        script_body = base_setup + """
import numpy as np
import torch
from torch.utils import data
from PIL import Image
import io
import json

from utils.train_utils import load_args_from_yaml, fix_random_seeds
from utils.eval_utils import compute_tapvid_metrics

fix_random_seeds(1234)

def sample_queries_strided_fixed(target_occluded, target_points, query_stride=5):
    tracks, occs, queries = [], [], []
    for i in range(0, target_occluded.shape[1], query_stride):
        mask = target_occluded[:, i] == 0
        query = np.stack([i * np.ones(target_occluded.shape[0:1]), target_points[:, i, 1], target_points[:, i, 0]], axis=-1)
        queries.append(query[mask])
        tracks.append(target_points[mask])
        occs.append(target_occluded[mask])
    return {{"query_points": np.concatenate(queries, axis=0), "target_points": np.concatenate(tracks, axis=0), "occluded": np.concatenate(occs, axis=0)}}

class TAPVidStrided:
    def __init__(self, pkl_path):
        import pickle
        with open(pkl_path, "rb") as f:
            self.data = pickle.load(f)
        self.video_names = sorted(list(self.data.keys()))
        self.resize_to_256 = False
        self.queried_first = False

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, index):
        vn = self.video_names[index]
        entry = self.data[vn]
        vf = entry.get("frames") or entry.get("video")
        if isinstance(vf[0], bytes):
            def decode(frame):
                byteio = io.BytesIO(frame)
                return np.array(Image.open(byteio))
            video = np.array([decode(f) for f in vf])
        else:
            video = vf.copy()
        tpr = entry["points"].copy()
        tocc = entry["occluded"].copy()
        h, w = video.shape[1], video.shape[2]
        tpx = tpr * np.array([w, h])
        conv = sample_queries_strided_fixed(tocc, tpx)
        traj = conv["target_points"].transpose(1, 0, 2)
        vis = (~conv["occluded"]).transpose(1, 0)
        qpts = conv["query_points"]
        return video, traj, vis, qpts

model = "{model_name}"
config_path = str(PROJECT_ROOT / "baselines" / "track_on" / "config" / "test.yaml")
checkpoint_path = str(PROJECT_ROOT / "baselines" / "track_on" / "checkpoints_trackon2_dinov3.pt")
dataset_path = "{dataset_path}"
cache_dir = "{cache_dir}"
output_json = "{output_json}"

print("Loading Trackon2 model")
from model.trackon_predictor import Predictor as Trackon_Predictor
trackon2_args = load_args_from_yaml(config_path)
model = Trackon_Predictor(trackon2_args, checkpoint_path=checkpoint_path, support_grid_size=20).cuda()
print("Trackon2 loaded")

dataset = TAPVidStrided(dataset_path)
loader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=False)

results = []
for j, (video, trajectory, visibility, query_points_i) in enumerate(loader):
    if video.ndim != 5:
        raise RuntimeError("Expected video tensor with shape (B, T, H, W, C) from dataloader")
    video = video.permute(0, 1, 4, 2, 3).contiguous().float().cuda(non_blocking=True)
    query_points_i = query_points_i.cuda(non_blocking=True)
    trajectory = trajectory.cuda(non_blocking=True)
    visibility = visibility.cuda(non_blocking=True)

    queries = query_points_i.clone().float()
    queries = torch.stack([queries[:,:,0], queries[:,:,2], queries[:,:,1]], dim=2).cuda()
    traj = trajectory.clone()
    qp = query_points_i.clone().cpu().numpy()
    gt_tracks = traj.permute(0,2,1,3).cpu().numpy()
    gt_occ = torch.logical_not(visibility.clone().permute(0,2,1)).cpu().numpy()

    cache_path = os.path.join(cache_dir, "davis", "{model_name}", "{{0:06d}}.npz".format(j))
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        pt = torch.from_numpy(cached["tracks"]).cuda()
        pv = torch.from_numpy(cached["visibility"]).cuda()
    else:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            pt, pv = model(video, queries)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, tracks=pt.cpu().numpy(), visibility=pv.cpu().numpy())

    po = torch.logical_not(pv.clone().permute(0,2,1)).cpu().numpy()
    ptracks = pt.permute(0,2,1,3).cpu().numpy()

    m = compute_tapvid_metrics(qp, gt_occ, gt_tracks, po, ptracks, "strided")
    results.append(m)
    print("Sample {{0}}/{{1}}".format(j+1, len(dataset)), flush=True)

agg = {{}}
for key in ["occlusion_accuracy","pts_within_1","pts_within_2","pts_within_4","pts_within_8","pts_within_16"]:
    agg[key] = np.mean([r[key] for r in results])
agg["average_pts_within_thresh"] = np.mean([agg[k] for k in ["pts_within_1","pts_within_2","pts_within_4","pts_within_8","pts_within_16"]])
for thr in [1,2,4,8,16]:
    agg["jaccard_{{}}".format(thr)] = np.mean([r["jaccard_{{}}".format(thr)] for r in results])
agg["average_jaccard"] = np.mean([agg["jaccard_{{}}".format(thr)] for thr in [1,2,4,8,16]])

out = {{
    "model_name": "{model_name}",
    "query_mode": "strided",
    "metric_resolution_mode": "original",
    "AJ": float(agg["average_jaccard"]) * 100,
    "OA": float(agg["occlusion_accuracy"]) * 100,
    "delta_avg": float(agg["average_pts_within_thresh"]) * 100,
    "delta_4px": float(agg["pts_within_4"]) * 100,
}}
with open(output_json, "w") as f:
    json.dump(out, f, indent=2)
print("Results: AJ={{0:.2f}}, OA={{1:.2f}}, delta_avg={{2:.2f}}".format(out["AJ"], out["OA"], out["delta_avg"]))
"""

    else:
        # CoTracker3
        script_body = base_setup + """
import numpy as np
import torch
from torch.utils import data
from PIL import Image
import io
import json

from cotracker.predictor import CoTrackerOnlinePredictor, CoTrackerPredictor
from utils.eval_utils import compute_tapvid_metrics

model_name = "{model_name}"
checkpoint_online = str(PROJECT_ROOT / "baselines" / "cotracker" / "checkpoints" / "scaled_online.pth")
checkpoint_offline = str(PROJECT_ROOT / "baselines" / "cotracker" / "checkpoints" / "scaled_offline.pth")
dataset_path = "{dataset_path}"
cache_dir = "{cache_dir}"
output_json = "{output_json}"

print("Loading CoTracker3 model: {{0}}".format(model_name))
if model_name == "cotracker3_video":
    model = CoTrackerOnlinePredictor(checkpoint=checkpoint_online).cuda()
else:
    model = CoTrackerPredictor(checkpoint=checkpoint_offline).cuda()

def sample_queries_strided_fixed(target_occluded, target_points, query_stride=5):
    tracks, occs, queries = [], [], []
    for i in range(0, target_occluded.shape[1], query_stride):
        mask = target_occluded[:, i] == 0
        query = np.stack([i * np.ones(target_occluded.shape[0:1]), target_points[:, i, 1], target_points[:, i, 0]], axis=-1)
        queries.append(query[mask])
        tracks.append(target_points[mask])
        occs.append(target_occluded[mask])
    return {{"query_points": np.concatenate(queries, axis=0), "target_points": np.concatenate(tracks, axis=0), "occluded": np.concatenate(occs, axis=0)}}

class TAPVidSimple:
    def __init__(self, pkl_path):
        import pickle
        with open(pkl_path, "rb") as f:
            self.data = pickle.load(f)
        self.video_names = sorted(list(self.data.keys()))

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, index):
        vn = self.video_names[index]
        entry = self.data[vn]
        vf = entry.get("frames") or entry.get("video")
        if isinstance(vf[0], bytes):
            def decode(frame):
                byteio = io.BytesIO(frame)
                return np.array(Image.open(byteio))
            video = np.array([decode(f) for f in vf])
        else:
            video = vf.copy()
        tpr = entry["points"].copy()
        tocc = entry["occluded"].copy()
        h, w = video.shape[1], video.shape[2]
        tpx = tpr * np.array([w, h])
        conv = sample_queries_strided_fixed(tocc, tpx)
        traj = conv["target_points"].transpose(1, 0, 2)
        vis = (~conv["occluded"]).transpose(1, 0)
        qpts = conv["query_points"]
        return video, traj, vis, qpts

dataset = TAPVidSimple(dataset_path)
loader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=False)

results = []
for j, (video, trajectory, visibility, query_points_i) in enumerate(loader):
    if video.ndim != 5:
        raise RuntimeError("Expected video tensor with shape (B, T, H, W, C) from dataloader")
    video = video.permute(0, 1, 4, 2, 3).contiguous().float().cuda(non_blocking=True)
    query_points_i = query_points_i.cuda(non_blocking=True)
    trajectory = trajectory.cuda(non_blocking=True)
    visibility = visibility.cuda(non_blocking=True)

    queries = query_points_i.clone().float()
    queries = torch.stack([queries[:,:,0], queries[:,:,2], queries[:,:,1]], dim=2).cuda()
    traj = trajectory.clone()
    qp = query_points_i.clone().cpu().numpy()
    gt_tracks = traj.permute(0,2,1,3).cpu().numpy()
    gt_occ = torch.logical_not(visibility.clone().permute(0,2,1)).cpu().numpy()

    cache_path = os.path.join(cache_dir, "davis", "{model_name}", "{{0:06d}}.npz".format(j))
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        pt = torch.from_numpy(cached["tracks"]).cuda()
        pv = torch.from_numpy(cached["visibility"]).cuda()
    else:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            if model_name == "cotracker3_video":
                model(video_chunk=video, is_first_step=True, queries=queries, add_support_grid=False, grid_size=0)
                pt = None
                pv = None
                for ind in range(0, video.shape[1] - model.step, model.step):
                    chunk = video[:, ind : ind + model.step * 2]
                    pt, pv = model(
                        video_chunk=chunk,
                        is_first_step=False,
                        add_support_grid=False,
                        grid_size=0,
                    )
                if pt is None or pv is None:
                    raise RuntimeError("CoTracker3 online predictor produced no chunks; video may be shorter than model.step")
            else:
                pt, pv = model(video, queries=queries, grid_size=0)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, tracks=pt.cpu().numpy(), visibility=pv.cpu().numpy())

    po = torch.logical_not(pv.clone().permute(0,2,1)).cpu().numpy()
    ptracks = pt.permute(0,2,1,3).cpu().numpy()

    m = compute_tapvid_metrics(qp, gt_occ, gt_tracks, po, ptracks, "strided")
    results.append(m)
    print("Sample {{0}}/{{1}}".format(j+1, len(dataset)), flush=True)

agg = {{}}
for key in ["occlusion_accuracy","pts_within_1","pts_within_2","pts_within_4","pts_within_8","pts_within_16"]:
    agg[key] = np.mean([r[key] for r in results])
agg["average_pts_within_thresh"] = np.mean([agg[k] for k in ["pts_within_1","pts_within_2","pts_within_4","pts_within_8","pts_within_16"]])
for thr in [1,2,4,8,16]:
    agg["jaccard_{{}}".format(thr)] = np.mean([r["jaccard_{{}}".format(thr)] for r in results])
agg["average_jaccard"] = np.mean([agg["jaccard_{{}}".format(thr)] for thr in [1,2,4,8,16]])

out = {{
    "model_name": "{model_name}",
    "query_mode": "strided",
    "metric_resolution_mode": "original",
    "AJ": float(agg["average_jaccard"]) * 100,
    "OA": float(agg["occlusion_accuracy"]) * 100,
    "delta_avg": float(agg["average_pts_within_thresh"]) * 100,
    "delta_4px": float(agg["pts_within_4"]) * 100,
}}
with open(output_json, "w") as f:
    json.dump(out, f, indent=2)
print("Results: AJ={{0:.2f}}, OA={{1:.2f}}, delta_avg={{2:.2f}}".format(out["AJ"], out["OA"], out["delta_avg"]))
"""

    # Generate the script with proper variable substitution
    # Note: We use .format() here to substitute the variables INTO the script template.
    # The generated script will NOT use f-strings for the dynamic parts (j, len, etc.)
    # because the outer template uses regular string (not f-string).
    generated_script = script_body.format(
        model_name=model_name,
        dataset_path=dataset_path,
        cache_dir=cache_dir,
        output_json=output_json,
        dinov3_local_dir=dinov3_local_dir,
    )

    with tempfile.NamedTemporaryFile(mode='w', suffix='_eval.py', delete=False, dir='/tmp') as f:
        f.write(generated_script)
        script_path = f.name

    try:
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{PROJECT_ROOT}/baselines/track_on:{PROJECT_ROOT}/baselines/cotracker:{PROJECT_ROOT}"
        result = subprocess.run(
            ['python3', script_path],
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
            cwd=str(PROJECT_ROOT),
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        if result.returncode != 0:
            raise RuntimeError(f"Subprocess failed with code {result.returncode}")
    finally:
        os.unlink(script_path)


def main():
    parser = argparse.ArgumentParser(description="Strided + Original evaluation")
    parser.add_argument('--model_name', type=str, required=True,
                        choices=['trackon2', 'cotracker3_video', 'cotracker3_window'],
                        help='Model to evaluate')
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, required=True)
    parser.add_argument('--output_json', type=str, required=True)
    parser.add_argument('--dinov3_local_dir', type=str, default="")
    args = parser.parse_args()

    if args.model_name == "trackon2":
        dinov3_dir = args.dinov3_local_dir or str(
            PROJECT_ROOT / "third_party_weights" / "dinov3" / "facebook_dinov3_vits16plus_pretrain_lvd1689m"
        )
        if not Path(dinov3_dir).exists():
            print(f"ERROR: DINOV3_LOCAL_DIR not found at {dinov3_dir}")
            print("Track-On2 cannot run strided+original without DINOv3 backbone.")
            print("Track-On2 uses fixed 384x512 input (from config) and cannot handle original video resolution.")
            raise SystemExit(1)
        run_model_via_subprocess(
            args.model_name, args.dataset_path, args.cache_dir, args.output_json,
            dinov3_local_dir=dinov3_dir,
        )
    else:
        run_model_via_subprocess(
            args.model_name, args.dataset_path, args.cache_dir, args.output_json,
        )


if __name__ == "__main__":
    main()
