#!/usr/bin/env python3
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path('docs/figures/reentry_tap_method/reentry_guard_framework.png')
OUT.parent.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(11, 6))
ax.set_xlim(0, 11)
ax.set_ylim(0, 6)
ax.axis('off')

def box(x, y, w, h, text, fs=10):
    p=FancyBboxPatch((x,y), w,h, boxstyle='round,pad=0.08,rounding_size=0.08', linewidth=1.5, facecolor='white', edgecolor='black')
    ax.add_patch(p)
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=fs, wrap=True)

def arrow(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2), arrowstyle='->', mutation_scale=14, linewidth=1.2))

box(0.4, 4.2, 1.8, 0.8, 'Input video\n+ query point')
box(3.0, 4.7, 2.0, 0.75, 'Base branch B\nstandard-strong')
box(3.0, 3.6, 2.0, 0.75, 'Override branch O\nre-entry-strong')
box(5.8, 4.15, 2.0, 0.9, 'Persistent visibility\ntrigger P=2')
box(8.3, 4.15, 2.0, 0.9, 'Local override\nwindow W=16')
box(5.8, 2.7, 2.0, 0.85, 'Runtime features\nvisibility + geometry\n+ motion')
box(8.3, 2.7, 2.0, 0.85, 'ReEntry-Guard\naccept / reject')
box(8.3, 1.2, 2.0, 0.85, 'B2-W16-P2\nfixed accept')
box(4.4, 0.55, 2.4, 0.85, 'Final output Y\nO inside accepted window;\nB otherwise')

arrow(2.2,4.6,3.0,5.05)
arrow(2.2,4.6,3.0,3.95)
arrow(5.0,5.05,5.8,4.65)
arrow(5.0,3.95,5.8,4.45)
arrow(7.8,4.6,8.3,4.6)
arrow(5.0,5.05,5.8,3.15)
arrow(5.0,3.95,5.8,3.15)
arrow(7.8,3.12,8.3,3.12)
arrow(9.3,4.15,9.3,3.55)
arrow(9.3,2.7,9.3,2.05)
arrow(8.3,1.63,6.8,0.98)
arrow(9.3,2.7,6.8,0.98)
arrow(9.3,4.15,6.8,0.98)
arrow(5.0,5.05,5.1,1.4)
arrow(5.1,1.4,5.2,0.98)

ax.text(0.5,5.65,'Selective Local Re-entry Override Framework',fontsize=16,weight='bold')
ax.text(0.5,0.15,'B2-W16-P2: training-free fixed local override.  ReEntry-Guard: learned reliability gate over the same candidate windows.',fontsize=9)

plt.tight_layout()
plt.savefig(OUT, dpi=220)
plt.close()
print(OUT)
