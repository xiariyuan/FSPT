#!/usr/bin/env python3
"""Generate quantitative paper figures (Fig 3-5)."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("paper/reentry_viscalibrator_tex/figures")
OUT.mkdir(parents=True, exist_ok=True)

C_BASE = "#f59e0b"
C_RED = "#ef4444"
C_GREEN = "#10b981"
C_BLUE = "#3b82f6"
C_V1 = "#8b5cf6"
C_V22Q = "#3b82f6"
C_V24 = "#ec4899"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

def make_fig3():
    methods = ["CT3\noffline", "V1", "V22Q", "V24"]
    aj = [62.66, 64.58, 64.73, 64.84]
    oa = [88.15, 91.73, 91.74, 91.89]
    ajrd = [0.3142, 0.3588, 0.3556, 0.3549]
    x = np.arange(4)
    width = 0.28
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5), gridspec_kw={"width_ratios": [2, 1]})
    ax1.bar(x - width/2, aj, width, label="AJ", color=C_BASE, alpha=0.85)
    ax1.bar(x + width/2, oa, width, label="OA", color=C_BLUE, alpha=0.85)
    ax1.set_ylabel("Score")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=9)
    ax1.set_ylim(50, 100)
    ax1.legend(loc="upper left")
    ax1.set_title("DAVIS first/input: AJ and OA")
    ax1.axhline(y=67.04, color=C_GREEN, ls="--", lw=1, alpha=0.7)
    ax1.text(3.3, 67.5, "TrackOn2 AJ", fontsize=7, color=C_GREEN, ha="right")
    colors_rd = [C_BASE, C_V1, C_V22Q, C_V24]
    ax2.bar(x, ajrd, width*1.5, color=colors_rd, alpha=0.85)
    ax2.set_ylabel("AJ_RD")
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=8, rotation=30, ha="right")
    ax2.set_ylim(0.25, 0.40)
    ax2.set_title("AJ_RD")
    fig.suptitle("DAVIS first/input (30 videos, 650 queries)", fontsize=11, y=1.02)
    fig.savefig(OUT / "fig3_davis_first_bar.png")
    fig.savefig(OUT / "fig3_davis_first_bar.pdf")
    plt.close(fig)
    print("Fig 3 done")

def make_fig4():
    methods = ["CT3 offline", "V1", "V22Q", "V24"]
    aj = [79.93, 79.43, 79.58, 79.60]
    oa = [91.64, 93.08, 93.05, 93.04]
    ajrd = [0.3617, 0.4414, 0.4400, 0.4394]
    x = np.arange(4)
    fig, ax1 = plt.subplots(figsize=(8, 3.5))
    ax1.set_xlabel("Method")
    ax1.set_ylabel("AJ_RD", color=C_V1)
    ax1.bar(x - 0.2, ajrd, 0.4, label="AJ_RD", color=C_V1, alpha=0.75)
    ax1.tick_params(axis="y", labelcolor=C_V1)
    ax1.set_ylim(0.30, 0.50)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=9)
    ax2 = ax1.twinx()
    ax2.set_ylabel("OA", color=C_GREEN)
    ax2.bar(x + 0.2, oa, 0.4, label="OA", color=C_GREEN, alpha=0.75)
    ax2.tick_params(axis="y", labelcolor=C_GREEN)
    ax2.set_ylim(90, 95)
    ax1.annotate(f"AJ: {aj[0]:.1f} -> {aj[3]:.1f}\n(dAJ = -0.34)", xy=(3, 0.40), xytext=(4.0, 0.46),
        fontsize=8, color=C_RED, ha="center", arrowprops=dict(arrowstyle="->", color=C_RED, lw=1))
    l1, la1 = ax1.get_legend_handles_labels()
    l2, la2 = ax2.get_legend_handles_labels()
    ax1.legend(l1+l2, la1+la2, loc="upper left", fontsize=8)
    ax1.set_title("RGB-Stacking full50: AJ_RD/OA gain with small AJ trade-off")
    fig.savefig(OUT / "fig4_rgb_full50_tradeoff.png")
    fig.savefig(OUT / "fig4_rgb_full50_tradeoff.pdf")
    plt.close(fig)
    print("Fig 4 done")

def make_fig5():
    thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    aj = [50.76, 50.80, 50.85, 50.91, 51.05, 51.36, 51.56, 51.56, 51.54, 51.54]
    oa = [91.27, 91.28, 91.32, 91.42, 91.65, 92.00, 92.21, 92.19, 92.16, 92.15]
    ajrd = [0.4141, 0.4136, 0.4131, 0.4118, 0.4092, 0.3988, 0.3900, 0.3875, 0.3870, 0.3870]
    fig, ax1 = plt.subplots(figsize=(8, 3.5))
    ax1.set_xlabel("V1 confidence threshold")
    ax1.set_ylabel("AJ / OA")
    ax1.plot(thresholds, aj, "o-", color=C_BASE, lw=2, ms=5, label="AJ")
    ax1.plot(thresholds, oa, "s-", color=C_GREEN, lw=2, ms=5, label="OA")
    ax1.set_ylim(50, 93)
    ax2 = ax1.twinx()
    ax2.set_ylabel("AJ_RD", color=C_V1)
    ax2.plot(thresholds, ajrd, "^--", color=C_V1, lw=2, ms=5, label="AJ_RD")
    ax2.tick_params(axis="y", labelcolor=C_V1)
    ax2.set_ylim(0.38, 0.42)
    ax1.axvline(x=0.80, color=C_RED, ls=":", lw=1.5, alpha=0.7)
    ax1.text(0.81, 51.5, "V25-safe\n(tau=0.80)", fontsize=8, color=C_RED, ha="left")
    l1, la1 = ax1.get_legend_handles_labels()
    l2, la2 = ax2.get_legend_handles_labels()
    ax1.legend(l1+l2, la1+la2, loc="center right", fontsize=8)
    ax1.set_title("V25 threshold sweep on DAVIS strided/original")
    fig.savefig(OUT / "fig5_v25_threshold_sweep.png")
    fig.savefig(OUT / "fig5_v25_threshold_sweep.pdf")
    plt.close(fig)
    print("Fig 5 done")

if __name__ == "__main__":
    make_fig3()
    make_fig4()
    make_fig5()
