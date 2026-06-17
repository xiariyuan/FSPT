#!/usr/bin/env python3
"""
Generate all paper figures and tables for the PRT hybrid selector paper.

Outputs:
  outputs/paper_assets/
    - main_table.json
    - feature_auc.json
    - coverage_risk_curve.json
    - stratified_by_type.json
    - stratified_by_occ.json
    - stratified_by_cam.json
    - stratified_2d.json
    - pooling_ablation.json
    - plot_coverage_risk.py  (matplotlib script)
    - plot_feature_auc.py    (matplotlib script)
    - plot_stratified.py     (matplotlib script)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_world_state_stage2_causal_dino import DINOFeatureExtractor
from scripts.eval_prt_hybrid_selector import encode_patch_batch
from sklearn.metrics import roc_auc_score


def main():
    output_dir = Path("outputs/paper_assets")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = DINOFeatureExtractor(
        Path("/gemini/code/FSPT/weights/dinov2/dinov2_vits14_pretrain.pth"), device
    )

    val = np.load("outputs/prt_candidate_val_v2_3seq/dataset_cache.npz", allow_pickle=False)
    n = val["query_patch"].shape[0]
    topk = val["cand_patches"].shape[1]

    # Compute support features
    print("Computing support features...", flush=True)
    support_count = val["support_patches"].shape[1]
    support_flat = val["support_patches"].reshape(
        n * support_count, *val["support_patches"].shape[2:]
    )
    support_feat = encode_patch_batch(extractor, support_flat).numpy().reshape(
        n, support_count, -1
    )
    support_mask = (
        np.arange(support_count, dtype=np.int32)[None, :]
        < val["support_count"][:, None]
    ).astype(np.float32)
    support_mask_sum = np.clip(support_mask.sum(axis=1, keepdims=True), 1.0, None)
    pooled = (support_feat * support_mask[:, :, None]).sum(axis=1) / support_mask_sum
    pooled /= np.linalg.norm(pooled, axis=1, keepdims=True).clip(1e-8, None)

    baseline_feat = encode_patch_batch(extractor, val["baseline_patch"]).numpy()
    cand_flat = val["cand_patches"].reshape(
        n * topk, *val["cand_patches"].shape[2:]
    )
    cand_feat = encode_patch_batch(extractor, cand_flat).numpy().reshape(n, topk, -1)

    baseline_support_sim = np.sum(baseline_feat * pooled, axis=1).astype(np.float32)
    cand_support_sim = np.sum(
        cand_feat * pooled[:, None, :], axis=2
    ).astype(np.float32)
    support_margin = cand_support_sim - baseline_support_sim[:, None]

    cand_score = val["cand_score"].astype(np.float32)
    cand_ncc = val["cand_ncc"].astype(np.float32)
    baseline_err = val["baseline_err"].astype(np.float32)
    cand_err = val["cand_err"].astype(np.float32)
    occ = val["occ_length"].astype(np.float32)
    cam = val["camera_motion"].astype(np.float32)
    reentry_type = val["reentry_type_id"].astype(int)

    # Hybrid predictions
    hybrid = 4.0 * support_margin + 0.5 * cand_score
    best_idx = np.argmax(hybrid, axis=1)
    hybrid_err = np.array([cand_err[i, best_idx[i]] for i in range(n)])
    oracle_err = np.minimum(baseline_err, cand_err.min(axis=1))

    # ============================================================
    # 1. Main Table
    # ============================================================
    main_table = {
        "n_val": int(n),
        "rows": [
            {
                "method": "Baseline only",
                "median_px": round(float(np.median(baseline_err)), 2),
                "lt4px": round(float(np.mean(baseline_err < 4.0)), 3),
                "coverage": 1.0,
                "better_frac": None,
            },
            {
                "method": "Oracle candidate pool",
                "median_px": round(float(np.median(oracle_err)), 2),
                "lt4px": round(float(np.mean(oracle_err < 4.0)), 3),
                "coverage": 1.0,
                "better_frac": None,
            },
            {
                "method": "Hybrid selector (full)",
                "median_px": round(float(np.median(hybrid_err)), 2),
                "lt4px": round(float(np.mean(hybrid_err < 4.0)), 3),
                "coverage": 1.0,
                "better_frac": round(float(np.mean(hybrid_err < baseline_err)), 3),
            },
        ],
    }
    # Add a selective operating point with target output coverage.
    # If we want 75% accepted coverage, the threshold should be the
    # 25th percentile of the score distribution because we accept
    # samples whose score is >= threshold.
    scores = hybrid.max(axis=1)
    for target_coverage in [0.75]:
        thr_quantile = 1.0 - target_coverage
        thr_val = float(np.quantile(scores, thr_quantile))
        accept = scores >= thr_val
        sel_err = baseline_err.copy()
        for i in range(n):
            if accept[i]:
                sel_err[i] = hybrid_err[i]
        main_table["rows"].append(
            {
                "method": f"Hybrid selector ({target_coverage*100:.0f}%)",
                "median_px": round(float(np.median(sel_err)), 2),
                "lt4px": round(float(np.mean(sel_err < 4.0)), 3),
                "coverage": round(float(accept.mean()), 2),
                "better_frac": round(float(np.mean(sel_err < baseline_err)), 3),
            }
        )

    # Adaptive selector: hybrid only for long occlusion (occ >= 100),
    # keep baseline for short occlusion. Threshold from stratified analysis.
    occ_thr = 100
    adaptive_err = baseline_err.copy()
    long_occ = occ >= occ_thr
    for i in range(n):
        if long_occ[i]:
            adaptive_err[i] = hybrid_err[i]
    main_table["rows"].append(
        {
            "method": f"Adaptive (occ>={occ_thr})",
            "median_px": round(float(np.median(adaptive_err)), 2),
            "lt4px": round(float(np.mean(adaptive_err < 4.0)), 3),
            "coverage": round(float(long_occ.mean()), 2),
            "better_frac": round(float(np.mean(adaptive_err < baseline_err)), 3),
        }
    )
    (output_dir / "main_table.json").write_text(json.dumps(main_table, indent=2) + "\n")
    print("main_table.json")

    # ============================================================
    # 2. Feature AUC
    # ============================================================
    labels = []
    margin_flat, score_flat, ncc_flat = [], [], []
    for i in range(n):
        for j in range(topk):
            labels.append(1 if cand_err[i, j] < baseline_err[i] else 0)
            margin_flat.append(float(support_margin[i, j]))
            score_flat.append(float(cand_score[i, j]))
            ncc_flat.append(float(cand_ncc[i, j]))
    labels = np.array(labels)

    feature_auc = {
        "description": "AUC for predicting whether candidate beats baseline",
        "n_samples": int(len(labels)),
        "features": {
            "support_margin": round(float(roc_auc_score(labels, margin_flat)), 4),
            "cand_score": round(float(roc_auc_score(labels, score_flat)), 4),
            "cand_ncc": round(float(roc_auc_score(labels, ncc_flat)), 4),
        },
    }
    (output_dir / "feature_auc.json").write_text(json.dumps(feature_auc, indent=2) + "\n")
    print("feature_auc.json")

    # ============================================================
    # 3. Coverage-Risk Curve
    # ============================================================
    coverage_risk = []
    for quantile in np.linspace(0.0, 1.0, 21):
        thr_val = float(np.quantile(scores, quantile))
        accept = scores >= thr_val
        cov = float(accept.mean())
        sel_err = baseline_err.copy()
        for i in range(n):
            if accept[i]:
                sel_err[i] = hybrid_err[i]
        coverage_risk.append(
            {
                "threshold": round(thr_val, 4),
                "coverage": round(cov, 3),
                "median_px": round(float(np.median(sel_err)), 2),
                "lt4px": round(float(np.mean(sel_err < 4.0)), 3),
                "better_frac": round(float(np.mean(sel_err < baseline_err)), 3),
            }
        )
    (output_dir / "coverage_risk_curve.json").write_text(
        json.dumps(coverage_risk, indent=2) + "\n"
    )
    print("coverage_risk_curve.json")

    # ============================================================
    # 4. Stratified Results
    # ============================================================
    def stratified_table(mask_fn, labels_list):
        rows = []
        for label, mask in labels_list:
            m = mask
            if m.sum() == 0:
                continue
            rows.append(
                {
                    "group": label,
                    "n": int(m.sum()),
                    "baseline_median": round(float(np.median(baseline_err[m])), 2),
                    "hybrid_median": round(float(np.median(hybrid_err[m])), 2),
                    "adaptive_median": round(float(np.median(adaptive_err[m])), 2),
                    "oracle_median": round(float(np.median(oracle_err[m])), 2),
                    "delta_hybrid": round(float(np.median(baseline_err[m]) - np.median(hybrid_err[m])), 2),
                    "delta_adaptive": round(float(np.median(baseline_err[m]) - np.median(adaptive_err[m])), 2),
                    "baseline_lt4px": round(float(np.mean(baseline_err[m] < 4.0)), 3),
                    "hybrid_lt4px": round(float(np.mean(hybrid_err[m] < 4.0)), 3),
                    "adaptive_lt4px": round(float(np.mean(adaptive_err[m] < 4.0)), 3),
                    "better_frac_hybrid": round(float(np.mean(hybrid_err[m] < baseline_err[m])), 3),
                    "better_frac_adaptive": round(float(np.mean(adaptive_err[m] < baseline_err[m])), 3),
                }
            )
        return rows

    # By reentry type
    type_rows = stratified_table(
        None,
        [
            ("In-frame occlusion", reentry_type == 0),
            ("Off-screen return", reentry_type == 1),
        ],
    )
    (output_dir / "stratified_by_type.json").write_text(
        json.dumps(type_rows, indent=2) + "\n"
    )
    print("stratified_by_type.json")

    # By occlusion length
    occ_rows = stratified_table(
        None,
        [
            ("20-50", (occ >= 20) & (occ < 50)),
            ("50-100", (occ >= 50) & (occ < 100)),
            ("100-200", (occ >= 100) & (occ < 200)),
            ("200-500", (occ >= 200) & (occ < 500)),
            ("500+", occ >= 500),
        ],
    )
    (output_dir / "stratified_by_occ.json").write_text(
        json.dumps(occ_rows, indent=2) + "\n"
    )
    print("stratified_by_occ.json")

    # By camera motion
    cam_rows = stratified_table(
        None,
        [
            ("<0.3", (cam >= 0) & (cam < 0.3)),
            ("0.3-0.5", (cam >= 0.3) & (cam < 0.5)),
            ("0.5-1.0", (cam >= 0.5) & (cam < 1.0)),
            (">1.0", cam >= 1.0),
        ],
    )
    (output_dir / "stratified_by_cam.json").write_text(
        json.dumps(cam_rows, indent=2) + "\n"
    )
    print("stratified_by_cam.json")

    # 2D stratification
    rows_2d = []
    for occ_label, occ_mask in [
        ("Short occ (<100)", occ < 100),
        ("Long occ (>=100)", occ >= 100),
    ]:
        for cam_label, cam_mask in [
            ("Low cam (<0.5)", cam < 0.5),
            ("High cam (>=0.5)", cam >= 0.5),
        ]:
            m = occ_mask & cam_mask
            if m.sum() == 0:
                continue
            rows_2d.append(
                {
                    "occ_group": occ_label,
                    "cam_group": cam_label,
                    "n": int(m.sum()),
                    "baseline_median": round(float(np.median(baseline_err[m])), 2),
                    "hybrid_median": round(float(np.median(hybrid_err[m])), 2),
                    "adaptive_median": round(float(np.median(adaptive_err[m])), 2),
                    "oracle_median": round(float(np.median(oracle_err[m])), 2),
                    "delta_hybrid": round(
                        float(np.median(baseline_err[m]) - np.median(hybrid_err[m])), 2
                    ),
                    "delta_adaptive": round(
                        float(np.median(baseline_err[m]) - np.median(adaptive_err[m])), 2
                    ),
                }
            )
    (output_dir / "stratified_2d.json").write_text(
        json.dumps(rows_2d, indent=2) + "\n"
    )
    print("stratified_2d.json")

    # ============================================================
    # 5. Plotting scripts
    # ============================================================

    # Coverage-risk curve plot
    plot_cr = '''#!/usr/bin/env python3
"""Coverage-risk curve for PRT hybrid selector."""
import json
import matplotlib.pyplot as plt
import numpy as np

with open("outputs/paper_assets/coverage_risk_curve.json") as f:
    data = json.load(f)

coverages = [d["coverage"] for d in data]
medians = [d["median_px"] for d in data]

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(coverages, medians, "o-", color="#2196F3", linewidth=2, markersize=5, label="Hybrid Selector")
ax.axhline(y=36.23, color="#F44336", linestyle="--", linewidth=1.5, label="Baseline (36.23 px)")
ax.axhline(y=25.90, color="#4CAF50", linestyle=":", linewidth=1.5, label="Oracle (25.90 px)")
ax.set_xlabel("Coverage", fontsize=12)
ax.set_ylabel("Median Error (px)", fontsize=12)
ax.set_title("Coverage-Risk Trade-off", fontsize=13)
ax.legend(fontsize=10)
ax.set_xlim(0, 1.05)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("outputs/paper_assets/fig_coverage_risk.pdf", dpi=300)
fig.savefig("outputs/paper_assets/fig_coverage_risk.png", dpi=150)
print("Saved fig_coverage_risk.pdf/png")
'''
    (output_dir / "plot_coverage_risk.py").write_text(plot_cr)

    # Feature AUC bar chart
    plot_auc = '''#!/usr/bin/env python3
"""Feature AUC bar chart."""
import json
import matplotlib.pyplot as plt
import numpy as np

with open("outputs/paper_assets/feature_auc.json") as f:
    data = json.load(f)

features = list(data["features"].keys())
aucs = list(data["features"].values())
labels = ["Support Margin", "Cand. Score", "RGB-NCC"]
colors = ["#2196F3", "#FF9800", "#9E9E9E"]

fig, ax = plt.subplots(figsize=(5, 3.5))
bars = ax.barh(labels, aucs, color=colors, height=0.5, edgecolor="white")
ax.axvline(x=0.5, color="red", linestyle="--", linewidth=1, alpha=0.7, label="Random (0.5)")
for bar, auc in zip(bars, aucs):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f"{auc:.3f}", va="center", fontsize=11)
ax.set_xlim(0.4, 0.75)
ax.set_xlabel("AUC", fontsize=12)
ax.set_title("Per-Feature Predictive Power", fontsize=13)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig("outputs/paper_assets/fig_feature_auc.pdf", dpi=300)
fig.savefig("outputs/paper_assets/fig_feature_auc.png", dpi=150)
print("Saved fig_feature_auc.pdf/png")
'''
    (output_dir / "plot_feature_auc.py").write_text(plot_auc)

    # Stratified results plot
    plot_strat = '''#!/usr/bin/env python3
"""Stratified results comparison."""
import json
import matplotlib.pyplot as plt
import numpy as np

with open("outputs/paper_assets/stratified_by_type.json") as f:
    type_data = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left: by reentry type
ax = axes[0]
labels = [d["group"] for d in type_data]
x = np.arange(len(labels))
width = 0.25
ax.bar(x - width, [d["baseline_median"] for d in type_data], width, label="Baseline", color="#F44336", alpha=0.8)
ax.bar(x, [d["hybrid_median"] for d in type_data], width, label="Hybrid", color="#2196F3", alpha=0.8)
ax.bar(x + width, [d["oracle_median"] for d in type_data], width, label="Oracle", color="#4CAF50", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Median Error (px)")
ax.set_title("By Re-entry Type")
ax.legend(fontsize=9)

# Right: by occ length
with open("outputs/paper_assets/stratified_by_occ.json") as f:
    occ_data = json.load(f)
ax = axes[1]
labels = [d["group"] for d in occ_data]
x = np.arange(len(labels))
ax.bar(x - width, [d["baseline_median"] for d in occ_data], width, label="Baseline", color="#F44336", alpha=0.8)
ax.bar(x, [d["hybrid_median"] for d in occ_data], width, label="Hybrid", color="#2196F3", alpha=0.8)
ax.bar(x + width, [d["oracle_median"] for d in occ_data], width, label="Oracle", color="#4CAF50", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8, rotation=15)
ax.set_ylabel("Median Error (px)")
ax.set_title("By Occlusion Length (frames)")

fig.tight_layout()
fig.savefig("outputs/paper_assets/fig_stratified.pdf", dpi=300)
fig.savefig("outputs/paper_assets/fig_stratified.png", dpi=150)
print("Saved fig_stratified.pdf/png")
'''
    (output_dir / "plot_stratified.py").write_text(plot_strat)

    print("\nAll assets generated in outputs/paper_assets/")
    print("Run plot scripts individually to generate figures.")


if __name__ == "__main__":
    main()
