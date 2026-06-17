#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.attempt0_schema import load_attempt0_cache


@dataclass
class QueryEval:
    video_id: str
    sequence_index: int
    query_index: int
    query_t: int
    reentry_t: int
    occ_length: int
    errors_px: Dict[str, float]


def _find_first_reentry(gt_visibility: np.ndarray, query_t: int) -> Optional[Tuple[int, int]]:
    """Return (reentry_t, occ_length) for the first visible-after-occlusion event."""
    in_occlusion = False
    occ_len = 0
    for t in range(int(query_t) + 1, int(gt_visibility.shape[0])):
        visible = bool(gt_visibility[t])
        if not visible:
            in_occlusion = True
            occ_len += 1
            continue
        if in_occlusion:
            return t, occ_len
    return None


def _compute_error_px(pred_yx: np.ndarray, gt_yx: np.ndarray, input_hw: Tuple[int, int]) -> float:
    scale = np.array([float(input_hw[0]), float(input_hw[1])], dtype=np.float32)
    return float(np.linalg.norm((pred_yx - gt_yx) * scale, axis=-1))


def _load_records(cache_path: Path) -> Dict[str, Any]:
    payload = load_attempt0_cache(cache_path)
    if not isinstance(payload, dict):
        raise ValueError(f"{cache_path} did not load to dict")
    if "records" not in payload or not isinstance(payload["records"], list):
        raise ValueError(f"{cache_path} missing top-level records list")
    return payload


def _assert_records_aligned(
    reference_records: List[Dict[str, Any]],
    candidate_records: List[Dict[str, Any]],
    *,
    label_ref: str,
    label_cmp: str,
    atol: float = 1e-6,
) -> Dict[str, Any]:
    issues: List[str] = []
    if len(reference_records) != len(candidate_records):
        raise ValueError(
            f"record count mismatch: {label_ref}={len(reference_records)} vs {label_cmp}={len(candidate_records)}"
        )

    compared_queries = 0
    compared_videos = len(reference_records)
    for idx, (ref, cmp_) in enumerate(zip(reference_records, candidate_records)):
        if ref["video_id"] != cmp_["video_id"]:
            issues.append(
                f"record[{idx}] video_id mismatch: {label_ref}={ref['video_id']} vs {label_cmp}={cmp_['video_id']}"
            )
            continue
        if int(ref["sequence_index"]) != int(cmp_["sequence_index"]):
            issues.append(
                f"record[{idx}] sequence_index mismatch: {label_ref}={ref['sequence_index']} vs {label_cmp}={cmp_['sequence_index']}"
            )
        ref_q = np.asarray(ref["query_points"], dtype=np.float32)
        cmp_q = np.asarray(cmp_["query_points"], dtype=np.float32)
        if ref_q.shape != cmp_q.shape:
            issues.append(
                f"record[{idx}] query_points shape mismatch: {label_ref}={ref_q.shape} vs {label_cmp}={cmp_q.shape}"
            )
            continue
        if not np.allclose(ref_q, cmp_q, atol=atol, rtol=0.0):
            max_diff = float(np.max(np.abs(ref_q - cmp_q)))
            issues.append(
                f"record[{idx}] query_points mismatch: max_abs_diff={max_diff:.6f}"
            )
        ref_gt_tracks = np.asarray(ref["gt_tracks"], dtype=np.float32)
        cmp_gt_tracks = np.asarray(cmp_["gt_tracks"], dtype=np.float32)
        if ref_gt_tracks.shape != cmp_gt_tracks.shape:
            issues.append(
                f"record[{idx}] gt_tracks shape mismatch: {label_ref}={ref_gt_tracks.shape} vs {label_cmp}={cmp_gt_tracks.shape}"
            )
        elif not np.allclose(ref_gt_tracks, cmp_gt_tracks, atol=atol, rtol=0.0):
            max_diff = float(np.max(np.abs(ref_gt_tracks - cmp_gt_tracks)))
            issues.append(
                f"record[{idx}] gt_tracks mismatch: max_abs_diff={max_diff:.6f}"
            )
        ref_gt_vis = np.asarray(ref["gt_visibility"], dtype=bool)
        cmp_gt_vis = np.asarray(cmp_["gt_visibility"], dtype=bool)
        if ref_gt_vis.shape != cmp_gt_vis.shape:
            issues.append(
                f"record[{idx}] gt_visibility shape mismatch: {label_ref}={ref_gt_vis.shape} vs {label_cmp}={cmp_gt_vis.shape}"
            )
        elif not np.array_equal(ref_gt_vis, cmp_gt_vis):
            issues.append(f"record[{idx}] gt_visibility mismatch")
        compared_queries += int(ref_q.shape[0])

    if issues:
        raise ValueError("cache alignment failed:\n" + "\n".join(issues[:20]))

    return {
        "reference_label": label_ref,
        "candidate_label": label_cmp,
        "compared_videos": compared_videos,
        "compared_queries": compared_queries,
        "status": "aligned",
    }


def _bucket_name(occ_length: int) -> str:
    if occ_length < 20:
        return "<20"
    if occ_length < 50:
        return "20-49"
    if occ_length < 100:
        return "50-99"
    return "100+"


def _summarize_errors(errors: np.ndarray) -> Dict[str, Any]:
    if errors.size == 0:
        return {
            "n": 0,
            "mean_px": None,
            "median_px": None,
            "lt4px": None,
            "lt8px": None,
            "p95_px": None,
        }
    return {
        "n": int(errors.size),
        "mean_px": float(errors.mean()),
        "median_px": float(np.median(errors)),
        "lt4px": float(np.mean(errors < 4.0)),
        "lt8px": float(np.mean(errors < 8.0)),
        "p95_px": float(np.percentile(errors, 95)),
    }


def _pairwise_summary(
    queries: List[QueryEval],
    model_a: str,
    model_b: str,
) -> Dict[str, Any]:
    err_a = np.array([q.errors_px[model_a] for q in queries], dtype=np.float32)
    err_b = np.array([q.errors_px[model_b] for q in queries], dtype=np.float32)
    diff = err_a - err_b  # negative means A is better

    per_video: Dict[str, List[Tuple[float, float]]] = {}
    for q in queries:
        per_video.setdefault(q.video_id, []).append((q.errors_px[model_a], q.errors_px[model_b]))

    per_video_better_a = 0
    per_video_better_b = 0
    per_video_tie = 0
    for pairs in per_video.values():
        va = np.median([p[0] for p in pairs])
        vb = np.median([p[1] for p in pairs])
        if math.isclose(va, vb, rel_tol=0.0, abs_tol=1e-6):
            per_video_tie += 1
        elif va < vb:
            per_video_better_a += 1
        else:
            per_video_better_b += 1

    return {
        "n_common_queries": int(err_a.size),
        "median_px_model_a": float(np.median(err_a)) if err_a.size else None,
        "median_px_model_b": float(np.median(err_b)) if err_b.size else None,
        "mean_px_model_a": float(err_a.mean()) if err_a.size else None,
        "mean_px_model_b": float(err_b.mean()) if err_b.size else None,
        "median_signed_diff_a_minus_b_px": float(np.median(diff)) if diff.size else None,
        "mean_signed_diff_a_minus_b_px": float(diff.mean()) if diff.size else None,
        "better_frac_model_a": float(np.mean(err_a < err_b)) if err_a.size else None,
        "better_frac_model_b": float(np.mean(err_b < err_a)) if err_a.size else None,
        "tie_frac": float(np.mean(np.isclose(err_a, err_b, atol=1e-6))) if err_a.size else None,
        "per_video_better_model_a": int(per_video_better_a),
        "per_video_better_model_b": int(per_video_better_b),
        "per_video_tie": int(per_video_tie),
        "n_videos": int(len(per_video)),
    }


def _make_summary_markdown(
    *,
    labels: List[str],
    overall: Dict[str, Any],
    long_occ_overall: Dict[str, Any],
    pairwise: Dict[str, Any],
    buckets: Dict[str, Dict[str, Any]],
    alignment: Dict[str, Any],
) -> str:
    lines: List[str] = []
    lines.append("# Re-Entry Head-to-Head Summary")
    lines.append("")
    lines.append("## Cache Alignment")
    lines.append("")
    lines.append(f"- Compared videos: `{alignment['compared_videos']}`")
    lines.append(f"- Compared queries: `{alignment['compared_queries']}`")
    lines.append(f"- Status: `{alignment['status']}`")
    lines.append("")
    lines.append("## Overall Re-Entry Metrics")
    lines.append("")
    lines.append("| Model | n | median px | mean px | <4px | <8px | p95 px |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for label in labels:
        m = overall[label]
        lines.append(
            f"| {label} | {m['n']} | {m['median_px']:.2f} | {m['mean_px']:.2f} | "
            f"{100.0*m['lt4px']:.1f}% | {100.0*m['lt8px']:.1f}% | {m['p95_px']:.2f} |"
        )
    lines.append("")
    lines.append("## Long-Occlusion Re-Entry (occ >= 20)")
    lines.append("")
    lines.append("| Model | n | median px | mean px | <4px | <8px | p95 px |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for label in labels:
        m = long_occ_overall[label]
        if not m["n"]:
            lines.append(f"| {label} | 0 | - | - | - | - | - |")
            continue
        lines.append(
            f"| {label} | {m['n']} | {m['median_px']:.2f} | {m['mean_px']:.2f} | "
            f"{100.0*m['lt4px']:.1f}% | {100.0*m['lt8px']:.1f}% | {m['p95_px']:.2f} |"
        )
    lines.append("")
    lines.append("## Pairwise")
    lines.append("")
    for name, metrics in pairwise.items():
        model_a, model_b = name.split("__vs__")
        lines.append(f"### {model_a} vs {model_b}")
        lines.append("")
        lines.append(f"- Common re-entry queries: `{metrics['n_common_queries']}`")
        lines.append(
            f"- Median signed diff `{model_a} - {model_b}`: "
            f"`{metrics['median_signed_diff_a_minus_b_px']:.2f}px` "
            f"(negative means `{model_a}` better)"
        )
        lines.append(
            f"- Better fraction: `{model_a}` `{100.0*metrics['better_frac_model_a']:.1f}%`, "
            f"`{model_b}` `{100.0*metrics['better_frac_model_b']:.1f}%`, "
            f"`tie` `{100.0*metrics['tie_frac']:.1f}%`"
        )
        lines.append(
            f"- Per-video medians: `{model_a}` better on `{metrics['per_video_better_model_a']}/{metrics['n_videos']}`, "
            f"`{model_b}` better on `{metrics['per_video_better_model_b']}/{metrics['n_videos']}`, "
            f"`tie` `{metrics['per_video_tie']}`"
        )
        lines.append("")
    lines.append("## Occlusion Buckets")
    lines.append("")
    for bucket_name, bucket_info in buckets.items():
        lines.append(f"### {bucket_name}")
        lines.append("")
        lines.append("| Model | n | median px | <4px | <8px |")
        lines.append("|---|---:|---:|---:|---:|")
        for label in labels:
            m = bucket_info[label]
            if not m["n"]:
                lines.append(f"| {label} | 0 | - | - | - |")
                continue
            lines.append(
                f"| {label} | {m['n']} | {m['median_px']:.2f} | "
                f"{100.0*m['lt4px']:.1f}% | {100.0*m['lt8px']:.1f}% |"
            )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate re-entry head-to-head from Attempt 0 unified caches.")
    parser.add_argument(
        "--cache",
        action="append",
        nargs=2,
        metavar=("LABEL", "PATH"),
        required=True,
        help="Model label and path to Attempt 0 unified cache. Repeat this flag for each model.",
    )
    parser.add_argument("--out-json", type=str, required=True, help="Output JSON metrics path.")
    parser.add_argument("--out-md", type=str, required=True, help="Output markdown summary path.")
    args = parser.parse_args()

    label_to_payload: Dict[str, Dict[str, Any]] = {}
    for label, path_str in args.cache:
        path = Path(path_str)
        if label in label_to_payload:
            raise ValueError(f"duplicate label: {label}")
        label_to_payload[label] = _load_records(path)

    labels = list(label_to_payload.keys())
    if not labels:
        raise ValueError("no caches provided")

    ref_label = labels[0]
    ref_payload = label_to_payload[ref_label]
    ref_records = ref_payload["records"]

    alignment_reports: List[Dict[str, Any]] = []
    for label in labels[1:]:
        alignment_reports.append(
            _assert_records_aligned(
                ref_records,
                label_to_payload[label]["records"],
                label_ref=ref_label,
                label_cmp=label,
            )
        )

    # If only one cache is provided, still return a meaningful alignment section.
    if not alignment_reports:
        n_queries = int(sum(np.asarray(r["query_points"]).shape[0] for r in ref_records))
        alignment_reports.append(
            {
                "reference_label": ref_label,
                "candidate_label": ref_label,
                "compared_videos": len(ref_records),
                "compared_queries": n_queries,
                "status": "self_only",
            }
        )

    queries: List[QueryEval] = []
    for record_idx, ref_record in enumerate(ref_records):
        video_id = str(ref_record["video_id"])
        sequence_index = int(ref_record["sequence_index"])
        query_points = np.asarray(ref_record["query_points"], dtype=np.float32)
        gt_tracks = np.asarray(ref_record["gt_tracks"], dtype=np.float32)
        gt_visibility = np.asarray(ref_record["gt_visibility"], dtype=bool)
        input_h, input_w = [int(x) for x in np.asarray(ref_record["model_input_size"]).tolist()]

        pred_tracks_by_label = {
            label: np.asarray(label_to_payload[label]["records"][record_idx]["pred_tracks"], dtype=np.float32)
            for label in labels
        }

        n_queries = int(query_points.shape[0])
        for query_index in range(n_queries):
            query_t = int(np.clip(round(float(query_points[query_index, 0])), 0, gt_visibility.shape[1] - 1))
            reentry = _find_first_reentry(gt_visibility[query_index], query_t)
            if reentry is None:
                continue
            reentry_t, occ_length = reentry
            gt_yx = gt_tracks[query_index, reentry_t]
            errors_px: Dict[str, float] = {}
            for label in labels:
                pred_yx = pred_tracks_by_label[label][query_index, reentry_t]
                errors_px[label] = _compute_error_px(pred_yx, gt_yx, (input_h, input_w))
            queries.append(
                QueryEval(
                    video_id=video_id,
                    sequence_index=sequence_index,
                    query_index=query_index,
                    query_t=query_t,
                    reentry_t=reentry_t,
                    occ_length=int(occ_length),
                    errors_px=errors_px,
                )
            )

    overall: Dict[str, Any] = {}
    for label in labels:
        errs = np.array([q.errors_px[label] for q in queries], dtype=np.float32)
        overall[label] = _summarize_errors(errs)

    pairwise: Dict[str, Any] = {}
    for i, model_a in enumerate(labels):
        for model_b in labels[i + 1 :]:
            pairwise[f"{model_a}__vs__{model_b}"] = _pairwise_summary(queries, model_a, model_b)

    buckets: Dict[str, Dict[str, Any]] = {}
    for bucket in ("<20", "20-49", "50-99", "100+"):
        bucket_queries = [q for q in queries if _bucket_name(q.occ_length) == bucket]
        buckets[bucket] = {}
        for label in labels:
            errs = np.array([q.errors_px[label] for q in bucket_queries], dtype=np.float32)
            buckets[bucket][label] = _summarize_errors(errs)

    long_occ_queries = [q for q in queries if q.occ_length >= 20]
    long_occ_overall: Dict[str, Any] = {}
    for label in labels:
        errs = np.array([q.errors_px[label] for q in long_occ_queries], dtype=np.float32)
        long_occ_overall[label] = _summarize_errors(errs)

    output = {
        "protocol": {
            "source": "attempt0_unified_bridge_cache",
            "dataset_name": ref_payload.get("dataset_name", ""),
            "split": ref_payload.get("split", ""),
            "protocol": ref_payload.get("protocol", ""),
            "metric_resolution": "input256",
            "reentry_definition": "first visible frame after first post-query occlusion run",
        },
        "cache_alignment": alignment_reports,
        "query_summary": {
            "n_videos": int(len(ref_records)),
            "n_queries_with_reentry": int(len(queries)),
            "occ_bucket_counts": {
                bucket: int(sum(1 for q in queries if _bucket_name(q.occ_length) == bucket))
                for bucket in ("<20", "20-49", "50-99", "100+")
            },
        },
        "overall": overall,
        "long_occ_overall": long_occ_overall,
        "pairwise": pairwise,
        "buckets": buckets,
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(output, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    alignment_for_md = alignment_reports[0].copy()
    if len(alignment_reports) > 1:
        compared_queries = min(int(x["compared_queries"]) for x in alignment_reports)
        compared_videos = min(int(x["compared_videos"]) for x in alignment_reports)
        alignment_for_md = {
            "compared_queries": compared_queries,
            "compared_videos": compared_videos,
            "status": "aligned",
        }
    summary_md = _make_summary_markdown(
        labels=labels,
        overall=overall,
        long_occ_overall=long_occ_overall,
        pairwise=pairwise,
        buckets=buckets,
        alignment=alignment_for_md,
    )
    out_md.write_text(summary_md, encoding="utf-8")

    print(json.dumps(output["query_summary"], indent=2, ensure_ascii=True))
    print(f"[ok] wrote metrics to {out_json}")
    print(f"[ok] wrote summary to {out_md}")


if __name__ == "__main__":
    main()
