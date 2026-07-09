#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.coords import find_reentry_events, yx_norm_to_xy_256

DATASET_PKL = Path("/gemini/code/datasets/tapvid_davis/tapvid_davis.pkl")
OUT = Path("outputs/paper_discovery_2026-06-27/teacher_expansion/b2_qualitative_cases")
FIXED = Path("outputs/redetection_ladder_2026-06-17/caches/cotracker3_offline_strided_original.pt")
B1 = Path("outputs/paper_discovery_2026-06-27/teacher_expansion/b1_4teacher_refine/vis4_gated288.pt")
B2 = Path("outputs/paper_discovery_2026-06-27/teacher_expansion/b2_mainline/b2_predicted_mainline.pt")
FALSE_TAX = Path("outputs/paper_discovery_2026-06-27/teacher_expansion/b2_false_trigger_taxonomy/summary.json")

METHOD_COLORS = {
    "GT": (0, 255, 0),
    "fixed": (255, 220, 0),
    "B1": (0, 180, 255),
    "B2": (255, 60, 255),
}


def npy(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    return a.astype(dtype) if dtype is not None else a


def load_cache(path: Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def invisible_run_before(v: np.ndarray, t: int) -> int:
    c = 0
    j = t - 1
    while j >= 0 and not bool(v[j]):
        c += 1
        j -= 1
    return c


def first_b2_trigger(base_v: np.ndarray, over_v: np.ndarray, query_t: int, k: int = 1) -> Optional[int]:
    for t in range(max(1, int(query_t) + 1), len(base_v)):
        if invisible_run_before(base_v, t) >= int(k) and bool(over_v[t]):
            return int(t)
    return None


def yx_norm_to_xy_img(pt_yx: np.ndarray, h: int, w: int) -> Tuple[float, float]:
    y = float(pt_yx[0]) * max(h - 1, 1)
    x = float(pt_yx[1]) * max(w - 1, 1)
    return x, y


def point_aj(pred_tracks: np.ndarray, gt_tracks: np.ndarray, pred_vis: np.ndarray, gt_vis: np.ndarray, query_t: int, start_t: int = 0) -> float:
    t_len = gt_vis.shape[0]
    mask = np.ones(t_len, dtype=bool)
    qt = max(0, min(t_len - 1, int(query_t)))
    mask[qt] = False
    if start_t > 0:
        mask[: int(start_t)] = False
    if not np.any(mask):
        return 0.0
    pred_px = yx_norm_to_xy_256(np.asarray(pred_tracks, dtype=np.float32))
    gt_px = yx_norm_to_xy_256(np.asarray(gt_tracks, dtype=np.float32))
    sq = np.sum((pred_px - gt_px) ** 2, axis=-1)
    vals = []
    for thr in (1, 2, 4, 8, 16):
        within = sq < float(thr) ** 2
        gt_pos = gt_vis.astype(bool)
        pv = pred_vis.astype(bool)
        tp = float(np.sum(mask & within & gt_pos & pv))
        gp = float(np.sum(mask & gt_pos))
        fp = float(np.sum(mask & pv & ((~gt_pos) | (~within))))
        vals.append(tp / (gp + fp) if (gp + fp) > 0 else 0.0)
    return float(np.mean(vals))


def get_font(size: int = 16):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"]:
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def draw_point(draw: ImageDraw.ImageDraw, xy: Tuple[float, float], color: Tuple[int, int, int], label: str, r: int = 6) -> None:
    x, y = xy
    draw.ellipse((x - r, y - r, x + r, y + r), outline=(0, 0, 0), width=4)
    draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=3)
    draw.text((x + r + 2, y - r - 2), label, fill=color, stroke_width=2, stroke_fill=(0, 0, 0), font=get_font(14))


def draw_trail(draw: ImageDraw.ImageDraw, pts: np.ndarray, vis: np.ndarray, h: int, w: int, t: int, color: Tuple[int, int, int], width: int = 3, max_len: int = 12) -> None:
    lo = max(0, int(t) - max_len)
    coords = []
    for tt in range(lo, int(t) + 1):
        if bool(vis[tt]):
            coords.append(yx_norm_to_xy_img(pts[tt], h, w))
    if len(coords) >= 2:
        draw.line(coords, fill=color, width=width)


def render_case(case: Dict[str, Any], data: Dict[str, Any], fixed_rec: Dict[str, Any], b1_rec: Dict[str, Any], b2_rec: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    vid = case["video_id"]
    qi = int(case["query_idx"])
    video = data[vid]["video"]
    h, w = int(video.shape[1]), int(video.shape[2])
    T = int(video.shape[0])
    qt = int(round(float(fixed_rec["query_points"][qi, 0])))
    gt_vis = npy(fixed_rec["gt_visibility"], bool)[qi]
    evs = find_reentry_events(gt_vis, qt)
    trigger = first_b2_trigger(npy(fixed_rec["pred_visibility"], bool)[qi], npy(b1_rec["pred_visibility"], bool)[qi], qt)
    reentry = int(evs[0]["reentry_frame"]) if evs else None
    last_visible = int(evs[0]["last_visible_t"]) if evs else None
    frames = [qt]
    if last_visible is not None:
        frames.append(last_visible)
    if trigger is not None:
        frames.extend([max(0, trigger - 1), trigger, min(T - 1, trigger + 8)])
    if reentry is not None:
        frames.extend([max(0, reentry - 1), reentry, min(T - 1, reentry + 8)])
    frames.append(T - 1)
    frames = sorted(set([int(max(0, min(T - 1, f))) for f in frames]))
    # Keep at most 6 panels by preserving important frames.
    if len(frames) > 6:
        priority = [qt, last_visible, trigger, reentry, (reentry + 8 if reentry is not None else None), T - 1]
        new = []
        for f in priority:
            if f is not None:
                ff = int(max(0, min(T - 1, f)))
                if ff not in new:
                    new.append(ff)
        frames = sorted(new[:6])
    panels = []
    font = get_font(16)
    small = get_font(13)
    for f in frames:
        img = Image.fromarray(video[f]).convert("RGB")
        draw = ImageDraw.Draw(img)
        title = f"{vid} | q={qi} | t={f}"
        flags = []
        if f == qt: flags.append("query")
        if f == trigger: flags.append("trigger")
        if f == reentry: flags.append("GT re-entry")
        if f == last_visible: flags.append("last visible")
        if flags:
            title += " | " + ", ".join(flags)
        draw.rectangle((0, 0, w, 30), fill=(0, 0, 0))
        draw.text((8, 6), title, fill=(255, 255, 255), font=font)
        tracks = {
            "GT": (npy(fixed_rec["gt_tracks"], np.float32)[qi], npy(fixed_rec["gt_visibility"], bool)[qi]),
            "fixed": (npy(fixed_rec["pred_tracks"], np.float32)[qi], npy(fixed_rec["pred_visibility"], bool)[qi]),
            "B1": (npy(b1_rec["pred_tracks"], np.float32)[qi], npy(b1_rec["pred_visibility"], bool)[qi]),
            "B2": (npy(b2_rec["pred_tracks"], np.float32)[qi], npy(b2_rec["pred_visibility"], bool)[qi]),
        }
        for name, (pts, vis) in tracks.items():
            color = METHOD_COLORS[name]
            draw_trail(draw, pts, vis, h, w, f, color)
        for name, (pts, vis) in tracks.items():
            if bool(vis[f]):
                draw_point(draw, yx_norm_to_xy_img(pts[f], h, w), METHOD_COLORS[name], name)
        legend = "GT green | fixed yellow | B1 cyan | B2 magenta"
        draw.rectangle((0, h - 24, w, h), fill=(0, 0, 0))
        draw.text((8, h - 20), legend, fill=(255, 255, 255), font=small)
        panels.append(img)
    gap = 8
    canvas_w = sum(p.width for p in panels) + gap * (len(panels) - 1)
    canvas_h = max(p.height for p in panels) + 130
    canvas = Image.new("RGB", (canvas_w, canvas_h), (20, 20, 20))
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.width + gap
    draw = ImageDraw.Draw(canvas)
    fixed_aj = point_aj(npy(fixed_rec["pred_tracks"], np.float32)[qi], npy(fixed_rec["gt_tracks"], np.float32)[qi], npy(fixed_rec["pred_visibility"], bool)[qi], npy(fixed_rec["gt_visibility"], bool)[qi], qt)
    b1_aj = point_aj(npy(b1_rec["pred_tracks"], np.float32)[qi], npy(fixed_rec["gt_tracks"], np.float32)[qi], npy(b1_rec["pred_visibility"], bool)[qi], npy(fixed_rec["gt_visibility"], bool)[qi], qt)
    b2_aj = point_aj(npy(b2_rec["pred_tracks"], np.float32)[qi], npy(fixed_rec["gt_tracks"], np.float32)[qi], npy(b2_rec["pred_visibility"], bool)[qi], npy(fixed_rec["gt_visibility"], bool)[qi], qt)
    footer = [
        f"category: {case['category']} | video={vid} query={qi} query_t={qt} trigger={trigger} reentry={reentry}",
        f"full-query AJ256: fixed={fixed_aj:.3f}  B1={b1_aj:.3f}  B2={b2_aj:.3f}  B2-fixed={b2_aj-fixed_aj:+.3f}  B2-B1={b2_aj-b1_aj:+.3f}",
        f"reason: {case.get('reason','')}",
    ]
    y = max(p.height for p in panels) + 10
    for line in footer:
        draw.text((12, y), line, fill=(255, 255, 255), font=font)
        y += 26
    safe_name = f"{case['category']}__{vid}__q{qi}.png".replace("/", "_")
    out_path = out_dir / safe_name
    canvas.save(out_path)
    result = dict(case)
    result.update({
        "query_t": qt,
        "trigger_t": trigger,
        "first_gt_reentry_t": reentry,
        "frames": frames,
        "fixed_query_AJ_256": round(fixed_aj, 6),
        "b1_query_AJ_256": round(b1_aj, 6),
        "b2_query_AJ_256": round(b2_aj, 6),
        "image": str(out_path),
    })
    return result


def build_case_candidates(fixed_payload: Dict[str, Any], b1_payload: Dict[str, Any], b2_payload: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    by_vid_idx = {str(r["video_id"]): i for i, r in enumerate(fixed_payload["records"])}
    false_summary = json.load(open(FALSE_TAX)) if FALSE_TAX.exists() else {}
    success_rows = false_summary.get("largest_true_trigger_gain_rows", [])[:20]
    failure_rows = false_summary.get("worst_false_trigger_rows", [])[:20]
    success = []
    for r in success_rows:
        success.append({
            "category": "success_reentry_recovery",
            "video_id": r["video_id"],
            "query_idx": int(r["query_idx"]),
            "reason": f"true trigger with large B2-fixed AJ gain ({r.get('delta_b2_minus_fixed_full_AJ')})",
        })
    failure = []
    seen = set()
    for r in failure_rows:
        key = (r["video_id"], int(r["query_idx"]))
        if key in seen:
            continue
        seen.add(key)
        failure.append({
            "category": "harmful_false_trigger",
            "video_id": r["video_id"],
            "query_idx": int(r["query_idx"]),
            "reason": f"false trigger with B2-fixed AJ drop ({r.get('delta_b2_minus_fixed_full_AJ')})",
        })
        if len(failure) >= 8:
            break
    # B1-vs-B2 contrast: B1 much worse than fixed, B2 close to fixed.
    contrast = []
    for idx, fr in enumerate(fixed_payload["records"]):
        vid = str(fr["video_id"])
        b1r = b1_payload["records"][idx]
        b2r = b2_payload["records"][idx]
        n = npy(fr["gt_visibility"]).shape[0]
        for qi in range(n):
            qt = int(round(float(fr["query_points"][qi, 0])))
            fixed_aj = point_aj(npy(fr["pred_tracks"])[qi], npy(fr["gt_tracks"])[qi], npy(fr["pred_visibility"], bool)[qi], npy(fr["gt_visibility"], bool)[qi], qt)
            b1_aj = point_aj(npy(b1r["pred_tracks"])[qi], npy(fr["gt_tracks"])[qi], npy(b1r["pred_visibility"], bool)[qi], npy(fr["gt_visibility"], bool)[qi], qt)
            b2_aj = point_aj(npy(b2r["pred_tracks"])[qi], npy(fr["gt_tracks"])[qi], npy(b2r["pred_visibility"], bool)[qi], npy(fr["gt_visibility"], bool)[qi], qt)
            score = (b2_aj - b1_aj) - abs(b2_aj - fixed_aj)
            if b2_aj - b1_aj > 0.25 and abs(b2_aj - fixed_aj) < 0.08:
                contrast.append({
                    "category": "b2_avoids_b1_global_damage",
                    "video_id": vid,
                    "query_idx": int(qi),
                    "score": float(score),
                    "reason": f"B1 much worse than B2 while B2 stays near fixed: fixed={fixed_aj:.3f}, B1={b1_aj:.3f}, B2={b2_aj:.3f}",
                })
    contrast = sorted(contrast, key=lambda x: x["score"], reverse=True)[:10]
    return {"success": success[:8], "contrast": contrast[:8], "failure": failure[:8]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--max-per-category", type=int, default=4)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    print("loading caches", flush=True)
    fixed = load_cache(FIXED)
    b1 = load_cache(B1)
    b2 = load_cache(B2)
    print("building case candidates", flush=True)
    candidates = build_case_candidates(fixed, b1, b2)
    selected = []
    for cat, rows in candidates.items():
        selected.extend(rows[: args.max_per_category])
    needed_vids = sorted(set(c["video_id"] for c in selected))
    print("loading DAVIS pkl", flush=True)
    with DATASET_PKL.open("rb") as f:
        davis = pickle.load(f)
    print("rendering", needed_vids, flush=True)
    rec_idx = {str(r["video_id"]): i for i, r in enumerate(fixed["records"])}
    manifest = []
    for case in selected:
        vid = case["video_id"]
        if vid not in davis or vid not in rec_idx:
            case["render_error"] = "missing_video_or_record"
            manifest.append(case)
            continue
        idx = rec_idx[vid]
        try:
            rendered = render_case(case, davis, fixed["records"][idx], b1["records"][idx], b2["records"][idx], img_dir)
            manifest.append(rendered)
            print("rendered", rendered["image"], flush=True)
        except Exception as e:
            case["render_error"] = repr(e)
            manifest.append(case)
            print("ERROR", case, e, flush=True)
    out = {
        "dataset_pkl": str(DATASET_PKL),
        "fixed_cache": str(FIXED),
        "b1_cache": str(B1),
        "b2_cache": str(B2),
        "n_cases": len(manifest),
        "cases": manifest,
    }
    (out_dir / "manifest.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    lines = ["# B2 Qualitative Cases — 2026-06-28", "", "## Exported cases", ""]
    for c in manifest:
        lines.append(f"- `{c.get('category')}` `{c.get('video_id')}` q={c.get('query_idx')} image={c.get('image', c.get('render_error'))}")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("Colors: GT green, fixed yellow, global B1 cyan, B2 magenta.")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"n_cases": len(manifest), "out_dir": str(out_dir), "images": [c.get("image") for c in manifest if c.get("image")]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
