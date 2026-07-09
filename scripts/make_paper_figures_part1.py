#!/usr/bin/env python3
"""Generate 5 paper figures for ReEntry paper."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

OUT = Path("paper/reentry_viscalibrator_tex/figures")
OUT.mkdir(parents=True, exist_ok=True)

C_BASE = "#f59e0b"
C_OVERRIDE = "#06b6d4"
C_B2 = "#d946ef"
C_RED = "#ef4444"
C_GREEN = "#10b981"
C_BLUE = "#3b82f6"
C_GRAY = "#9ca3af"
C_DARK = "#111827"
C_MUTED = "#6b7280"
C_V1 = "#8b5cf6"
C_V22Q = "#3b82f6"
C_V24 = "#ec4899"
C_V25 = "#14b8a6"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

def make_fig1():
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.set_xlim(-1.5, 10.5)
    ax.set_ylim(-1.5, 2.8)
    ax.axis("off")
    gt_vis = [1,1,1,1,0,0,0,1,1,1,1]
    base_vis = [1,1,1,1,0,0,0,0,0,1,1]
    reentry_vis = [1,1,1,1,0,0,0,1,1,1,1]
    for i, f in enumerate(range(11)):
        cg = C_GREEN if gt_vis[i] else C_GRAY
        ag = 0.85 if gt_vis[i] else 0.25
        ax.add_patch(patches.FancyBboxPatch((f-0.35, 1.0), 0.7, 0.7, boxstyle="round,pad=0.05", facecolor=cg, edgecolor="none", alpha=ag))
        cb = C_BASE if base_vis[i] else C_GRAY
        ab = 0.85 if base_vis[i] else 0.25
        ax.add_patch(patches.FancyBboxPatch((f-0.35, 0.0), 0.7, 0.7, boxstyle="round,pad=0.05", facecolor=cb, edgecolor="none", alpha=ab))
        cr = C_B2 if reentry_vis[i] else C_GRAY
        ar = 0.85 if reentry_vis[i] else 0.25
        ax.add_patch(patches.FancyBboxPatch((f-0.35, -1.0), 0.7, 0.7, boxstyle="round,pad=0.05", facecolor=cr, edgecolor="none", alpha=ar))
        ax.text(f, 1.85, str(f), ha="center", va="center", fontsize=8, color=C_MUTED)
    ax.text(-1.0, 1.35, "GT", ha="center", va="center", fontsize=11, fontweight="bold", color=C_DARK)
    ax.text(-1.0, 0.35, "Base", ha="center", va="center", fontsize=11, fontweight="bold", color=C_BASE)
    ax.text(-1.0, -0.65, "ReEntry", ha="center", va="center", fontsize=11, fontweight="bold", color=C_B2)
    ax.annotate("", xy=(3.5, 2.3), xytext=(6.5, 2.3), arrowprops=dict(arrowstyle="<->", color=C_RED, lw=1.5))
    ax.text(5, 2.45, "occluded", ha="center", va="bottom", fontsize=9, color=C_RED, fontstyle="italic")
    ax.annotate("re-entry", xy=(7, 0.0), xytext=(8.5, -1.2), arrowprops=dict(arrowstyle="->", color=C_B2, lw=1.5), fontsize=9, color=C_B2, ha="center")
    ax.annotate("visibility lag", xy=(8, 0.35), xytext=(9.5, -0.5), arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.5), fontsize=8, color=C_RED, ha="center")
    ax.set_title("Re-Entry Failure Mode: Base tracker lags visibility after reappearance", fontsize=11, pad=10)
    fig.savefig(OUT / "fig1_failure_mode.png")
    fig.savefig(OUT / "fig1_failure_mode.pdf")
    plt.close(fig)
    print("Fig 1 done")

def make_fig2():
    fig, ax = plt.subplots(figsize=(10, 3.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.2, 4)
    ax.axis("off")
    def box(x, y, w, h, label, color, tc="white", fs=9):
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", facecolor=color, edgecolor="none", alpha=0.9))
        ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=fs, fontweight="bold", color=tc)
    def ar(x1, y1, x2, y2, color=C_DARK):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color=color, lw=1.8))
    box(0.3, 3.0, 2.2, 0.7, "Base tracker\n(coords + vis)", C_BASE)
    box(3.2, 3.0, 2.2, 0.7, "Override source\n(visibility)", C_OVERRIDE)
    box(0.8, 2.0, 4.0, 0.7, "Candidate re-entry windows (B2-W16-P2)", C_DARK)
    ar(1.4, 3.0, 1.4, 2.7)
    ar(4.3, 3.0, 4.3, 2.7)
    box(0.3, 1.0, 1.8, 0.7, "V1\nLearned\ncalibrator", C_V1, fs=8)
    box(2.4, 1.0, 1.8, 0.7, "V22Q\nInterval /\ngate", C_V22Q, fs=8)
    box(4.5, 1.0, 1.8, 0.7, "V24\nDINOScore\nfilter", C_V24, fs=8)
    box(6.6, 1.0, 1.8, 0.7, "V25\nOfficial-safe\nthreshold", C_V25, fs=8)
    ar(2.8, 2.0, 1.2, 1.7)
    ar(2.8, 2.0, 3.3, 1.7)
    ar(2.8, 2.0, 5.4, 1.7)
    ar(2.8, 2.0, 7.5, 1.7)
    box(2.5, 0.0, 4.0, 0.7, "Corrected visibility\n(coordinates = base)", C_B2)
    ar(1.2, 1.0, 3.5, 0.7)
    ar(3.3, 1.0, 4.0, 0.7)
    ar(5.4, 1.0, 5.0, 0.7)
    ar(7.5, 1.0, 5.5, 0.7)
    ax.text(9.0, 0.35, "coords\npreserved", ha="center", va="center", fontsize=8, color=C_BASE, fontstyle="italic")
    ax.annotate("", xy=(7.5, 0.35), xytext=(8.5, 0.35), arrowprops=dict(arrowstyle="<-", color=C_BASE, lw=1.5, ls="--"))
    ax.set_title("ReEntry Pipeline: coordinate-preserving local visibility recovery", fontsize=11, pad=8)
    fig.savefig(OUT / "fig2_pipeline.png")
    fig.savefig(OUT / "fig2_pipeline.pdf")
    plt.close(fig)
    print("Fig 2 done")

if __name__ == "__main__":
    make_fig1()
    make_fig2()
