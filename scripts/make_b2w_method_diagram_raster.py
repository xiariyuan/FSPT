#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path('outputs/paper_discovery_2026-06-27/paper_figures_selected')
OUT.mkdir(parents=True, exist_ok=True)
PNG = OUT / 'figure1_b2w_method_diagram.png'
PDF = OUT / 'figure1_b2w_method_diagram.pdf'

W, H = 1800, 980
COL = {
    'bg': (255,255,255), 'ink': (17,24,39), 'muted': (107,114,128),
    'base': (245,158,11), 'override': (6,182,212), 'b2': (217,70,239),
    'green': (34,197,94), 'gray': (229,231,235), 'darkgray': (55,65,81),
    'light_base': (255,247,237), 'light_override': (236,254,255), 'light_b2': (253,244,255),
    'panel': (249,250,251), 'red': (239,68,68)
}

def font(size, bold=False):
    candidates = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()

def text(draw, xy, s, size=28, fill='ink', bold=False, anchor='mm'):
    draw.text(xy, s, font=font(size, bold), fill=COL[fill] if isinstance(fill, str) else fill, anchor=anchor)

def round_rect(draw, xy, fill, outline='gray', width=3, radius=24):
    draw.rounded_rectangle(xy, radius=radius, fill=COL[fill] if isinstance(fill, str) else fill, outline=COL[outline] if isinstance(outline, str) else outline, width=width)

def arrow(draw, start, end, fill='darkgray', width=5):
    draw.line([start, end], fill=COL[fill], width=width)
    # simple arrow head
    x1,y1=start; x2,y2=end
    import math
    ang=math.atan2(y2-y1,x2-x1)
    L=18
    a1=ang+math.pi*0.82; a2=ang-math.pi*0.82
    pts=[(x2,y2),(x2+L*math.cos(a1),y2+L*math.sin(a1)),(x2+L*math.cos(a2),y2+L*math.sin(a2))]
    draw.polygon(pts, fill=COL[fill])

def dashed_line(draw, xy1, xy2, fill='muted', width=3, dash=12, gap=10):
    import math
    x1,y1=xy1; x2,y2=xy2
    dist=math.hypot(x2-x1,y2-y1)
    if dist == 0: return
    dx=(x2-x1)/dist; dy=(y2-y1)/dist
    t=0
    while t < dist:
        t2=min(t+dash, dist)
        draw.line([(x1+dx*t,y1+dy*t),(x1+dx*t2,y1+dy*t2)], fill=COL[fill], width=width)
        t += dash + gap

img = Image.new('RGB', (W,H), COL['bg'])
d = ImageDraw.Draw(img)

# Title
text(d,(W//2,58),'B2-W: Predicted Re-entry Windowed Override',42,'ink',True)
text(d,(W//2,100),'Use the stable base by default; locally activate a re-entry branch only inside predicted windows.',24,'muted')

# Input block
round_rect(d,(70,155,400,360),'panel','gray',2)
text(d,(235,195),'Inputs',30,'ink',True)
round_rect(d,(105,225,365,270),'light_base','base',3,14)
text(d,(235,254),'Base tracker B',22,'ink')
text(d,(235,290),'standard-strong',18,'muted')
round_rect(d,(105,302,365,347),'light_override','override',3,14)
text(d,(235,331),'Override branch O',22,'ink')
text(d,(235,367),'re-entry-strong',18,'muted')

# Trigger block
round_rect(d,(485,150,880,380),'panel','gray',2)
text(d,(682,190),'Predicted re-entry trigger',30,'ink',True)
text(d,(682,235),'base invisible run ≥ 1',24,'base')
text(d,(682,273),'+ override visible',24,'override')
text(d,(682,313),'+ P2: visible for 2 frames',21,'b2')
round_rect(d,(585,330,780,364),'light_b2','b2',2,16)
text(d,(682,353),'trigger at t',20,'ink')
arrow(d,(400,247),(485,247)); arrow(d,(400,325),(485,325))

# Window block
round_rect(d,(970,150,1400,380),'panel','gray',2)
text(d,(1185,190),'Windowed local override',30,'ink',True)
text(d,(1185,238),'copy O only in [t-pre, t+W]',25,'b2')
text(d,(1185,278),'W = 16, pre = 1',22,'muted')
text(d,(1185,318),'then return control to B',24,'base')
round_rect(d,(1080,335,1290,369),'light_b2','b2',2,16)
text(d,(1185,358),'B2-W16-P2 output',18,'ink')
arrow(d,(880,265),(970,265))

# Output block
round_rect(d,(1480,195,1730,335),'light_b2','b2',4,26)
text(d,(1605,248),'Output Y',31,'ink',True)
text(d,(1605,288),'mostly base',22,'base')
text(d,(1605,318),'+ local override',22,'b2')
arrow(d,(1400,265),(1480,265))

# Timeline panel
panel_y=455
round_rect(d,(70,panel_y,1730,855),'bg','gray',2)
text(d,(900,panel_y+43),'Temporal behavior for one query track',31,'ink',True)
text(d,(900,panel_y+78),'B2-W leaves ordinary frames untouched and intervenes only near predicted re-entry.',21,'muted')

x0,x1=170,1620
y_base=panel_y+205; y_over=panel_y+280; y_out=panel_y+345
# labels
text(d,(120,y_base),'Base B',22,'base',True,'lm')
text(d,(120,y_over),'Override O',22,'override',True,'lm')
text(d,(120,y_out),'B2-W output',22,'b2',True,'lm')
# timeline lines
d.line([(x0,y_base),(x1,y_base)], fill=COL['base'], width=8)
# dashed override
for t in range(x0,x1,30):
    d.line([(t,y_over),(min(t+16,x1),y_over)], fill=COL['override'], width=8)
d.line([(x0,y_out),(x1,y_out)], fill=COL['base'], width=8)

q_x, dis_x, trig_x, reentry_x, win_end_x = 245, 635, 760, 825, 1020
# shaded regions
d.rounded_rectangle((dis_x,y_base-75,reentry_x,y_base+150), radius=8, fill=(243,244,246), outline=(209,213,219), width=1)
text(d,((dis_x+reentry_x)//2,y_base-88),'base invisible / occlusion',18,'muted')
d.rounded_rectangle((trig_x-30,y_base-105,win_end_x,y_base+165), radius=12, fill=(253,244,255), outline=COL['b2'], width=2)
text(d,((trig_x+win_end_x)//2,y_base-120),'local override window',19,'b2',True)
d.line([(trig_x-30,y_out),(win_end_x,y_out)], fill=COL['b2'], width=12)
# events
for x,label,col in [(q_x,'query q','darkgray'),(dis_x,'B invisible','base'),(trig_x,'trigger t','b2'),(reentry_x,'GT re-entry','green'),(win_end_x,'return to B','base')]:
    dashed_line(d,(x,y_base-130),(x,y_out+45),col,3,8,8)
    text(d,(x,y_out+80),label,18,col)
# points
for x in [q_x,390,520,1120,1320,1500]:
    d.ellipse((x-10,y_base-10,x+10,y_base+10), fill=COL['base'], outline=(146,64,14), width=2)
for x in [dis_x+35, dis_x+95, dis_x+145]:
    d.ellipse((x-9,y_base-9,x+9,y_base+9), fill=(255,255,255), outline=COL['base'], width=3)
    d.line([(x-7,y_base-7),(x+7,y_base+7)], fill=COL['base'], width=3)
    d.line([(x-7,y_base+7),(x+7,y_base-7)], fill=COL['base'], width=3)
for x in [trig_x,reentry_x,910,980]:
    d.ellipse((x-10,y_over-10,x+10,y_over+10), fill=COL['override'], outline=(14,116,144), width=2)
for x in [360,500,1260,1450]:
    d.ellipse((x-8,y_over-8,x+8,y_over+8), fill=(255,255,255), outline=COL['override'], width=3)
for x in [q_x,390,520,1120,1320,1500]:
    d.ellipse((x-10,y_out-10,x+10,y_out+10), fill=COL['base'], outline=(146,64,14), width=2)
for x in [trig_x,reentry_x,910,980]:
    d.ellipse((x-10,y_out-10,x+10,y_out+10), fill=COL['b2'], outline=(134,25,143), width=2)

# Formula callouts
round_rect(d,(100,875,615,945),'panel','gray',2,18)
text(d,(125,918),'Trigger: invisible_run(B) ≥ 1  ∧  visible(O)',22,'ink',False,'lm')
round_rect(d,(655,875,1105,945),'panel','gray',2,18)
text(d,(680,918),'Override: Y[t-pre : t+W] ← O',22,'ink',False,'lm')
round_rect(d,(1145,875,1660,945),'panel','gray',2,18)
text(d,(1170,918),'Outside window: Y ← B',22,'ink',False,'lm')

img.save(PNG)
img.save(PDF, 'PDF', resolution=300.0)
print(f'wrote {PNG}')
print(f'wrote {PDF}')
