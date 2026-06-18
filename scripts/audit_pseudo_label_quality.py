#!/usr/bin/env python3
"""Audit pseudo-label quality for canonical re-detection P3A.

Reads a JSONL pseudo-label file produced by build_reentry_pseudo_labels.py and
summarizes:
  - pseudo-label error statistics
  - comparison vs CT-offline error
  - occlusion-length distribution
  - per-video concentration
  - availability of FB / flow consistency fields

This is an audit script, not a trainer. It can run in smoke mode on a subset
and in full mode on the complete pseudo-label set.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _stats(values: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray(list(values), dtype=np.float32)
    if arr.size == 0:
        return {
            "n": 0,
            "mean_px": None,
            "median_px": None,
            "p95_px": None,
            "lt4px": None,
            "lt8px": None,
            "lt16px": None,
        }
    return {
        "n": int(arr.size),
        "mean_px": round(float(arr.mean()), 2),
        "median_px": round(float(np.median(arr)), 2),
        "p95_px": round(float(np.percentile(arr, 95)), 2),
        "lt4px": round(float(np.mean(arr < 4.0)), 4),
        "lt8px": round(float(np.mean(arr < 8.0)), 4),
        "lt16px": round(float(np.mean(arr < 16.0)), 4),
    }


def _bucket_hist(values: Sequence[int]) -> Dict[str, int]:
    arr = np.asarray(list(values), dtype=np.int64)
    if arr.size == 0:
        return {"1-4": 0, "5-9": 0, "10-15": 0, "16-31": 0, "32-63": 0, "64+": 0}
    return {
        "1-4": int(np.sum((arr >= 1) & (arr <= 4))),
        "5-9": int(np.sum((arr >= 5) & (arr <= 9))),
        "10-15": int(np.sum((arr >= 10) & (arr <= 15))),
        "16-31": int(np.sum((arr >= 16) & (arr <= 31))),
        "32-63": int(np.sum((arr >= 32) & (arr <= 63))),
        "64+": int(np.sum(arr >= 64)),
    }


def audit_pseudo_labels(
    labels: List[Dict[str, Any]],
    *,
    mode: str = "smoke",
    min_labels: int = 1000,
    min_long_occ_ratio: float = 0.20,
    min_lt16: float = 0.80,
    max_video_share_threshold: float = 0.20,
    fb_threshold: float = 1.0,
    flow_threshold: float = 1.0,
) -> Dict[str, Any]:
    if not labels:
        return {
            "mode": mode,
            "verdict": "STOP",
            "reason": "no pseudo-labels found",
            "n_labels": 0,
        }

    paired_errors = [
        (float(x["pseudo_label_error_px"]), float(x["ct_offline_error_px"]))
        for x in labels
        if "pseudo_label_error_px" in x and "ct_offline_error_px" in x
    ]
    errors = [p[0] for p in paired_errors]
    ct_errors = [p[1] for p in paired_errors]
    teacher_conf = [float(x["teacher_conf"]) for x in labels if "teacher_conf" in x]
    occ_lens = [int(x["occ_run_len"]) for x in labels if "occ_run_len" in x]
    vids = [str(x.get("video_id", "unknown")) for x in labels]
    source_bucket = [str(x.get("source_bucket", "unknown")) for x in labels]
    quality_flags = Counter(flag for x in labels for flag in x.get("quality_flags", []))

    video_counts = Counter(vids)
    n_labels = len(labels)
    n_videos = len(video_counts)
    n_tracks = len({(str(x.get("video_id", "unknown")), int(x.get("track_id", -1))) for x in labels})
    n_reentry_events = sum(1 for x in labels if bool(x.get("is_reentry_frame", False)))
    long_occ_mask = np.asarray(occ_lens, dtype=np.int64) >= 16 if occ_lens else np.array([], dtype=bool)
    long_occ_ratio = float(long_occ_mask.mean()) if long_occ_mask.size else 0.0

    err_arr = np.asarray(errors, dtype=np.float32) if errors else np.array([], dtype=np.float32)
    ct_arr = np.asarray(ct_errors, dtype=np.float32) if ct_errors else np.array([], dtype=np.float32)
    teacher_arr = np.asarray(teacher_conf, dtype=np.float32) if teacher_conf else np.array([], dtype=np.float32)
    occ_arr = np.asarray(occ_lens, dtype=np.int64) if occ_lens else np.array([], dtype=np.int64)

    better_than_ct = float(np.mean(err_arr < ct_arr)) if err_arr.size and ct_arr.size else None
    gain_px = ct_arr - err_arr if err_arr.size and ct_arr.size else np.array([], dtype=np.float32)

    fb_vals = [float(x.get("fb_error", -1.0)) for x in labels if float(x.get("fb_error", -1.0)) >= 0.0]
    flow_vals = [float(x.get("flow_consistency_error", -1.0)) for x in labels if float(x.get("flow_consistency_error", -1.0)) >= 0.0]
    fb_available = len(fb_vals) > 0
    flow_available = len(flow_vals) > 0

    fb_pass_rate = float(np.mean(np.asarray(fb_vals) <= fb_threshold)) if fb_vals else None
    flow_pass_rate = float(np.mean(np.asarray(flow_vals) <= flow_threshold)) if flow_vals else None

    per_video = [
        {
            "video_id": vid,
            "count": int(cnt),
            "share": round(float(cnt / n_labels), 4),
        }
        for vid, cnt in video_counts.most_common()
    ]
    max_video_share = max((item["share"] for item in per_video), default=0.0)

    quality = {
        "n_labels": n_labels,
        "n_videos": n_videos,
        "n_tracks": n_tracks,
        "n_reentry_events": n_reentry_events,
        "occ_length": {
            "histogram": _bucket_hist(occ_lens),
            "stats": _stats(occ_lens),
            "long_occ_ge16_ratio": round(long_occ_ratio, 4),
        },
        "pseudo_label_error_px": _stats(errors),
        "ct_offline_error_px": _stats(ct_errors),
        "gain_vs_ct_offline_px": {
            "n": int(gain_px.size),
            "mean_px": round(float(gain_px.mean()), 2) if gain_px.size else None,
            "median_px": round(float(np.median(gain_px)), 2) if gain_px.size else None,
            "p95_px": round(float(np.percentile(gain_px, 95)), 2) if gain_px.size else None,
            "better_than_ct_offline_frac": round(better_than_ct, 4) if better_than_ct is not None else None,
        },
        "teacher_conf": _stats(teacher_conf),
        "quality_flags": dict(quality_flags),
        "source_bucket": dict(Counter(source_bucket)),
        "per_video": per_video,
        "max_video_share": round(float(max_video_share), 4),
        "fb_consistency": {
            "available": fb_available,
            "pass_rate": round(float(fb_pass_rate), 4) if fb_pass_rate is not None else None,
            "threshold_px": fb_threshold,
        },
        "flow_consistency": {
            "available": flow_available,
            "pass_rate": round(float(flow_pass_rate), 4) if flow_pass_rate is not None else None,
            "threshold_px": flow_threshold,
        },
    }

    if mode not in {"smoke", "full"}:
        raise ValueError(f"Unknown mode: {mode}")

    if mode == "smoke":
        if n_labels == 0:
            verdict = "STOP"
            reason = "no labels were produced"
        elif quality["pseudo_label_error_px"]["lt16px"] is not None and quality["pseudo_label_error_px"]["lt16px"] >= 0.60:
            verdict = "SMOKE_GO"
            reason = "smoke labels have usable precision; proceed to filtering/splitting"
        else:
            verdict = "SMOKE_STOP"
            reason = "smoke quality is too weak to justify full P3A rollout"
    else:
        enough_labels = n_labels >= min_labels
        enough_long_occ = long_occ_ratio >= min_long_occ_ratio
        enough_precision = (quality["pseudo_label_error_px"]["lt16px"] or 0.0) >= min_lt16
        concentrated_ok = max_video_share <= max_video_share_threshold
        fb_ok = (not fb_available) or (fb_pass_rate is not None and fb_pass_rate >= 0.80)
        flow_ok = (not flow_available) or (flow_pass_rate is not None and flow_pass_rate >= 0.80)
        if enough_labels and enough_long_occ and enough_precision and concentrated_ok and fb_ok and flow_ok:
            verdict = "GO"
            reason = "full pseudo-label audit meets the current P3A thresholds"
        else:
            verdict = "STOP"
            reason = "full pseudo-label audit misses one or more P3A thresholds"

    quality["verdict"] = verdict
    quality["reason"] = reason
    quality["mode"] = mode
    quality["checks"] = {
        "enough_labels": n_labels >= min_labels,
        "enough_long_occ_ratio": long_occ_ratio >= min_long_occ_ratio,
        "enough_lt16": (quality["pseudo_label_error_px"]["lt16px"] or 0.0) >= min_lt16,
        "max_video_share_ok": max_video_share <= max_video_share_threshold,
        "fb_ok": fb_available and (fb_pass_rate is not None and fb_pass_rate >= 0.80) if fb_available else True,
        "flow_ok": flow_available and (flow_pass_rate is not None and flow_pass_rate >= 0.80) if flow_available else True,
    }
    return quality


def _render_md(result: Dict[str, Any]) -> str:
    p_err = result["pseudo_label_error_px"]
    ct_err = result["ct_offline_error_px"]
    gain = result["gain_vs_ct_offline_px"]
    occ = result["occ_length"]
    lines = []
    lines.append("# P3A Pseudo-Label Quality Audit")
    lines.append("")
    lines.append(f"Mode: `{result['mode']}`")
    lines.append(f"Verdict: `{result['verdict']}`")
    lines.append(f"Reason: {result['reason']}")
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append(f"- Labels: `{result['n_labels']}`")
    lines.append(f"- Videos: `{result['n_videos']}`")
    lines.append(f"- Tracks: `{result['n_tracks']}`")
    lines.append(f"- Re-entry events: `{result['n_reentry_events']}`")
    lines.append(f"- Long-occ (>=16) ratio: `{occ['long_occ_ge16_ratio']}`")
    lines.append(f"- Max video share: `{result['max_video_share']}`")
    lines.append("")
    lines.append("## Error Statistics")
    lines.append("")
    lines.append("| Metric | Pseudo-label | CT-offline | Delta |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| median px | {p_err['median_px']} | {ct_err['median_px']} | {gain['median_px']} |"
    )
    lines.append(
        f"| mean px | {p_err['mean_px']} | {ct_err['mean_px']} | {gain['mean_px']} |"
    )
    lines.append(
        f"| <4px | {p_err['lt4px']} | {ct_err['lt4px']} | {round((p_err['lt4px'] or 0.0) - (ct_err['lt4px'] or 0.0), 4)} |"
    )
    lines.append(
        f"| <8px | {p_err['lt8px']} | {ct_err['lt8px']} | {round((p_err['lt8px'] or 0.0) - (ct_err['lt8px'] or 0.0), 4)} |"
    )
    lines.append(
        f"| <16px | {p_err['lt16px']} | {ct_err['lt16px']} | {round((p_err['lt16px'] or 0.0) - (ct_err['lt16px'] or 0.0), 4)} |"
    )
    lines.append("")
    lines.append(f"- Better than CT-offline: `{gain['better_than_ct_offline_frac']}`")
    lines.append("")
    lines.append("## Consistency")
    lines.append("")
    lines.append(f"- FB available: `{result['fb_consistency']['available']}`")
    lines.append(f"- FB pass rate: `{result['fb_consistency']['pass_rate']}`")
    lines.append(f"- Flow available: `{result['flow_consistency']['available']}`")
    lines.append(f"- Flow pass rate: `{result['flow_consistency']['pass_rate']}`")
    lines.append("")
    lines.append("## Quality Flags")
    lines.append("")
    if result["quality_flags"]:
        for k, v in sorted(result["quality_flags"].items()):
            lines.append(f"- `{k}`: `{v}`")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit pseudo-label quality.")
    parser.add_argument("--labels-jsonl", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--output-md", type=str, required=True)
    parser.add_argument("--mode", type=str, default="smoke", choices=["smoke", "full"])
    parser.add_argument("--min-labels", type=int, default=1000)
    parser.add_argument("--min-long-occ-ratio", type=float, default=0.20)
    parser.add_argument("--min-lt16", type=float, default=0.80)
    parser.add_argument("--max-video-share", type=float, default=0.20)
    parser.add_argument("--fb-threshold", type=float, default=1.0)
    parser.add_argument("--flow-threshold", type=float, default=1.0)
    args = parser.parse_args()

    labels = _load_jsonl(Path(args.labels_jsonl))
    result = audit_pseudo_labels(
        labels,
        mode=args.mode,
        min_labels=args.min_labels,
        min_long_occ_ratio=args.min_long_occ_ratio,
        min_lt16=args.min_lt16,
        max_video_share_threshold=args.max_video_share,
        fb_threshold=args.fb_threshold,
        flow_threshold=args.flow_threshold,
    )

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(_render_md(result))

    print(json.dumps({k: result[k] for k in ["mode", "verdict", "reason", "n_labels", "n_videos"]}, indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
