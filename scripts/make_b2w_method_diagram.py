#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path('outputs/paper_discovery_2026-06-27/paper_figures_selected')
OUT.mkdir(parents=True, exist_ok=True)
SVG = OUT / 'figure1_b2w_method_diagram.svg'
PNG = OUT / 'figure1_b2w_method_diagram.png'
PDF = OUT / 'figure1_b2w_method_diagram.pdf'

W, H = 1800, 980

# Colors are chosen for paper readability and match qualitative legend where possible.
COL = {
    'bg': '#ffffff',
    'ink': '#111827',
    'muted': '#6b7280',
    'base': '#f59e0b',       # yellow/orange = fixed/base
    'override': '#06b6d4',   # cyan = override/global branch
    'b2': '#d946ef',         # magenta = B2 output
    'gt': '#22c55e',
    'light_base': '#fff7ed',
    'light_override': '#ecfeff',
    'light_b2': '#fdf4ff',
    'red': '#ef4444',
    'green': '#10b981',
    'blue': '#3b82f6',
    'gray': '#e5e7eb',
    'darkgray': '#374151',
}


def T(x, y, text, size=32, color=None, weight='500', anchor='middle'):
    color = color or COL['ink']
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}">{escape(text)}</text>'


def rect(x, y, w, h, fill, stroke=None, sw=3, rx=18, opacity=1):
    stroke = stroke or fill
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" fill-opacity="{opacity}" stroke="{stroke}" stroke-width="{sw}" />'


def line(x1, y1, x2, y2, color, sw=5, dash=None, marker=True, opacity=1):
    dash_s = f' stroke-dasharray="{dash}"' if dash else ''
    marker_s = ' marker-end="url(#arrow)"' if marker else ''
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}" stroke-linecap="round" opacity="{opacity}"{dash_s}{marker_s}/>'


def path(d, color, sw=5, fill='none', dash=None, marker=True, opacity=1):
    dash_s = f' stroke-dasharray="{dash}"' if dash else ''
    marker_s = ' marker-end="url(#arrow)"' if marker else ''
    return f'<path d="{d}" fill="{fill}" stroke="{color}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round" opacity="{opacity}"{dash_s}{marker_s}/>'


def circle(cx, cy, r, fill, stroke=None, sw=3):
    stroke = stroke or fill
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" />'


parts = []
parts.append(f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<defs>
  <marker id="arrow" markerWidth="14" markerHeight="14" refX="11" refY="4" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L12,4 L0,8 Z" fill="#374151" />
  </marker>
  <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
    <feDropShadow dx="0" dy="8" stdDeviation="8" flood-color="#000000" flood-opacity="0.12"/>
  </filter>
</defs>
<rect width="100%" height="100%" fill="{COL['bg']}"/>
''')

# Title
parts.append(T(W/2, 58, 'B2-W: Predicted Re-entry Windowed Override', size=42, weight='700'))
parts.append(T(W/2, 100, 'Use the stable base by default; locally activate a re-entry branch only inside predicted windows.', size=24, color=COL['muted']))

# Left: Inputs
parts.append(rect(70, 155, 330, 205, '#f9fafb', COL['gray'], sw=2, rx=24))
parts.append(T(235, 195, 'Inputs', size=30, weight='700'))
parts.append(rect(105, 225, 260, 45, COL['light_base'], COL['base'], sw=3, rx=12))
parts.append(T(235, 255, 'Base tracker B', size=22, color=COL['ink']))
parts.append(T(235, 290, 'standard-strong', size=18, color=COL['muted']))
parts.append(rect(105, 302, 260, 45, COL['light_override'], COL['override'], sw=3, rx=12))
parts.append(T(235, 332, 'Override branch O', size=22, color=COL['ink']))
parts.append(T(235, 367, 're-entry-strong', size=18, color=COL['muted']))

# Middle: trigger detector
parts.append(rect(485, 150, 395, 230, '#f9fafb', COL['gray'], sw=2, rx=24))
parts.append(T(682, 190, 'Predicted re-entry trigger', size=30, weight='700'))
parts.append(T(682, 235, 'base invisible run ≥ 1', size=24, color=COL['base']))
parts.append(T(682, 273, '+ override visible', size=24, color=COL['override']))
parts.append(T(682, 313, '+ P2: visible for 2 frames', size=21, color=COL['b2']))
parts.append(rect(585, 330, 195, 34, COL['light_b2'], COL['b2'], sw=2, rx=16))
parts.append(T(682, 354, 'trigger at t', size=20, color=COL['ink']))

# arrows from inputs to detector
parts.append(line(400, 247, 485, 247, COL['darkgray'], sw=4))
parts.append(line(400, 325, 485, 325, COL['darkgray'], sw=4))

# Right: Windowed override block
parts.append(rect(970, 150, 430, 230, '#f9fafb', COL['gray'], sw=2, rx=24))
parts.append(T(1185, 190, 'Windowed local override', size=30, weight='700'))
parts.append(T(1185, 238, 'copy O only in [t-pre, t+W]', size=25, color=COL['b2']))
parts.append(T(1185, 278, 'W = 16, pre = 1', size=22, color=COL['muted']))
parts.append(T(1185, 318, 'then return control to B', size=24, color=COL['base']))
parts.append(rect(1080, 335, 210, 34, COL['light_b2'], COL['b2'], sw=2, rx=16))
parts.append(T(1185, 359, 'B2-W16-P2 output', size=18, color=COL['ink']))
parts.append(line(880, 265, 970, 265, COL['darkgray'], sw=5))

# Output block
parts.append(rect(1480, 195, 250, 140, COL['light_b2'], COL['b2'], sw=4, rx=26))
parts.append(T(1605, 248, 'Output Y', size=31, weight='700', color=COL['ink']))
parts.append(T(1605, 288, 'mostly base', size=22, color=COL['base']))
parts.append(T(1605, 318, '+ local override', size=22, color=COL['b2']))
parts.append(line(1400, 265, 1480, 265, COL['darkgray'], sw=5))

# Timeline panel
panel_y = 455
parts.append(rect(70, panel_y, 1660, 400, '#ffffff', COL['gray'], sw=2, rx=24))
parts.append(T(900, panel_y+43, 'Temporal behavior for one query track', size=31, weight='700'))
parts.append(T(900, panel_y+78, 'B2-W leaves ordinary frames untouched and intervenes only near predicted re-entry.', size=21, color=COL['muted']))

# Timeline axis
x0, x1 = 170, 1620
y_base = panel_y + 205
y_over = panel_y + 280
y_out = panel_y + 345
parts.append(line(x0, y_base, x1, y_base, COL['base'], sw=8, marker=False))
parts.append(line(x0, y_over, x1, y_over, COL['override'], sw=8, marker=False, dash='16 14', opacity=0.85))
parts.append(line(x0, y_out, x1, y_out, COL['base'], sw=8, marker=False))

# B2 output magenta local windows overlay on output timeline
# Approx positions for q, disappearance, trigger, re-entry, window end.
q_x = 245
dis_x = 635
trig_x = 760
reentry_x = 825
win_end_x = 1020
return_x = 1070

# shaded occlusion region and override window region
parts.append(rect(dis_x, y_base-75, reentry_x-dis_x, 225, '#f3f4f6', '#d1d5db', sw=1, rx=8, opacity=0.75))
parts.append(T((dis_x+reentry_x)/2, y_base-88, 'base invisible / occlusion', size=18, color=COL['muted']))
parts.append(rect(trig_x-30, y_base-105, win_end_x-(trig_x-30), 270, COL['light_b2'], COL['b2'], sw=2, rx=12, opacity=0.55))
parts.append(T((trig_x+win_end_x)/2, y_base-120, 'local override window', size=19, color=COL['b2'], weight='700'))

# output overlay in magenta
parts.append(line(trig_x-30, y_out, win_end_x, y_out, COL['b2'], sw=12, marker=False))

# event markers
for x, label, col in [(q_x, 'query q', COL['darkgray']), (dis_x, 'B invisible', COL['base']), (trig_x, 'trigger t', COL['b2']), (reentry_x, 'GT re-entry', COL['green']), (win_end_x, 'return to B', COL['base'])]:
    parts.append(line(x, y_base-130, x, y_out+45, col, sw=3, dash='8 8', marker=False, opacity=0.9))
    parts.append(T(x, y_out+80, label, size=18, color=col))

# track labels
parts.append(T(120, y_base+8, 'Base B', size=22, color=COL['base'], anchor='start', weight='700'))
parts.append(T(120, y_over+8, 'Override O', size=22, color=COL['override'], anchor='start', weight='700'))
parts.append(T(120, y_out+8, 'B2-W output', size=22, color=COL['b2'], anchor='start', weight='700'))

# qualitative symbols on timelines
# base points mostly ok, missing during occlusion
for x in [q_x, 390, 520, 1120, 1320, 1500]:
    parts.append(circle(x, y_base, 10, COL['base'], '#92400e', sw=2))
for x in [dis_x+35, dis_x+95, dis_x+145]:
    parts.append(circle(x, y_base, 9, '#ffffff', COL['base'], sw=3))
    parts.append(line(x-7, y_base-7, x+7, y_base+7, COL['base'], sw=3, marker=False))
    parts.append(line(x-7, y_base+7, x+7, y_base-7, COL['base'], sw=3, marker=False))

# override appears near trigger/reentry but can be noisy elsewhere
for x in [trig_x, reentry_x, 910, 980]:
    parts.append(circle(x, y_over, 10, COL['override'], '#0e7490', sw=2))
for x in [360, 500, 1260, 1450]:
    parts.append(circle(x, y_over, 8, '#ffffff', COL['override'], sw=3))

# output points: base then magenta window then base
for x in [q_x, 390, 520, 1120, 1320, 1500]:
    parts.append(circle(x, y_out, 10, COL['base'], '#92400e', sw=2))
for x in [trig_x, reentry_x, 910, 980]:
    parts.append(circle(x, y_out, 10, COL['b2'], '#86198f', sw=2))

# Formula callouts
parts.append(rect(100, 875, 515, 70, '#f9fafb', COL['gray'], sw=2, rx=18))
parts.append(T(125, 918, 'Trigger: invisible_run(B) ≥ 1  ∧  visible(O)', size=22, anchor='start'))
parts.append(rect(655, 875, 450, 70, '#f9fafb', COL['gray'], sw=2, rx=18))
parts.append(T(680, 918, 'Override: Y[t-pre : t+W] ← O', size=22, anchor='start'))
parts.append(rect(1145, 875, 515, 70, '#f9fafb', COL['gray'], sw=2, rx=18))
parts.append(T(1170, 918, 'Outside window: Y ← B', size=22, anchor='start'))

parts.append('</svg>')
SVG.write_text('\n'.join(parts))
print(f'wrote {SVG}')

# Convert to PNG/PDF if cairosvg is available; otherwise keep SVG.
try:
    import cairosvg
    cairosvg.svg2png(url=str(SVG), write_to=str(PNG), output_width=W, output_height=H)
    cairosvg.svg2pdf(url=str(SVG), write_to=str(PDF))
    print(f'wrote {PNG}')
    print(f'wrote {PDF}')
except Exception as e:
    print(f'conversion skipped: {e}')
