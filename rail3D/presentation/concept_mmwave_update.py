"""presentation/concept_mmwave_update.py <original.png> <out.png> (e.g. data/figures/concept_original.png
-> data/figures/concept_mmwave_detector_only.png).

Redraw the red mm-wave part of the concept figure for the detector-only system:
one horn on the gauge side at ~55 deg, eight detectors overhead, no metasurface.
The blue optical subsystem, the rail and the right-hand text are left as they are."""
import math
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

src, out = sys.argv[1], sys.argv[2]
im = Image.open(src).convert("RGB")
W, H = im.size
a = np.asarray(im).astype(int)
R, G, B = a[..., 0], a[..., 1], a[..., 2]

# 1. erase every red / pink pixel left of the text column (lines, arcs, horns, hatching,
#    detector squares) and the old red labels (sub-pixel text has coloured fringes)
red = (R - np.maximum(G, B) > 6) & (R > 150)
red[:, 760:] = False
a[red] = 255
for x0, y0, x1, y1 in ((535, 8, 650, 62), (595, 115, 705, 172), (235, 158, 465, 186)):
    a[y0:y1, x0:x1] = 255
base = Image.fromarray(a.astype(np.uint8))

# 2. draw the new red layer, supersampled
S = 4
LINE = (252, 28, 25)
TEXT = (190, 22, 20)
ov = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
d = ImageDraw.Draw(ov)
font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 21 * S)
small = ImageFont.truetype("C:/Windows/Fonts/ariali.ttf", 15 * S)


def P(x, y):
    return (x * S, y * S)


def line(p, q, w=1.3, col=LINE, alpha=255):
    d.line([P(*p), P(*q)], fill=col + (alpha,), width=max(1, int(w * S)))


def arrow(p, q, w=1.3, head=7.0, col=LINE, alpha=255):
    line(p, q, w, col, alpha)
    ang = math.atan2(q[1] - p[1], q[0] - p[0])
    for s in (+1, -1):
        b = ang + math.pi + s * math.radians(24)
        line(q, (q[0] + head * math.cos(b), q[1] + head * math.sin(b)), w, col, alpha)


def dashed(p, q, w=1.6, on=7.0, off=5.0, col=LINE):
    L = math.dist(p, q)
    ux, uy = (q[0] - p[0]) / L, (q[1] - p[1]) / L
    t = 0.0
    while t < L:
        t2 = min(t + on, L)
        line((p[0] + ux * t, p[1] + uy * t), (p[0] + ux * t2, p[1] + uy * t2), w, col)
        t += on + off


def arc(c, r, a0, a1, w=1.4, n=90, fade=0.25):
    """Arc (angles in degrees, image coordinates: 0 = +x, 90 = down) with faded ends."""
    th = np.radians(np.linspace(a0, a1, n))
    pts = [(c[0] + r * math.cos(t), c[1] + r * math.sin(t)) for t in th]
    for k in range(n - 1):
        u = k / (n - 2)
        al = min(1.0, u / fade, (1 - u) / fade)
        line(pts[k], pts[k + 1], w, LINE, int(255 * max(al, 0.0)))


# geometry (image px): crown top centre, gauge corner on the horn side
crown = (349.0, 305.0)
gauge = (394.0, 313.0)
theta = math.radians(55.0)                    # horn axis from vertical
u_in = (math.sin(theta), -math.cos(theta))    # crown -> horn (up and right)

# 2a. the horn: a pyramidal horn on the gauge side, axis at 55 deg through the crown
ap_c = (crown[0] + 260 * u_in[0], crown[1] + 260 * u_in[1])   # aperture centre
back = (ap_c[0] + 40 * u_in[0], ap_c[1] + 40 * u_in[1])        # feed end
nrm = (-u_in[1], u_in[0])
A2, b2 = 20.0, 5.5
horn = [(ap_c[0] + A2 * nrm[0], ap_c[1] + A2 * nrm[1]), (back[0] + b2 * nrm[0], back[1] + b2 * nrm[1]),
        (back[0] - b2 * nrm[0], back[1] - b2 * nrm[1]), (ap_c[0] - A2 * nrm[0], ap_c[1] - A2 * nrm[1])]
d.polygon([P(*p) for p in horn], fill=(232, 30, 28, 255), outline=(150, 10, 10, 255))
feed = [(back[0] + b2 * nrm[0], back[1] + b2 * nrm[1]),
        (back[0] + 9 * u_in[0] + b2 * nrm[0], back[1] + 9 * u_in[1] + b2 * nrm[1]),
        (back[0] + 9 * u_in[0] - b2 * nrm[0], back[1] + 9 * u_in[1] - b2 * nrm[1]),
        (back[0] - b2 * nrm[0], back[1] - b2 * nrm[1])]
d.polygon([P(*p) for p in feed], fill=(170, 16, 14, 255))
d.text(P(ap_c[0] + 50, ap_c[1] - 22), "Mm-wave\nsource", font=font, fill=TEXT + (255,), spacing=2 * S)

# 2b. the illuminating beam: wavefronts centred on the horn, an arrow down its axis
for r in (55, 95, 135, 175):
    phi = math.degrees(math.atan2(-u_in[1] * -1, -u_in[0]))           # direction horn -> crown
    phi = math.degrees(math.atan2(crown[1] - back[1], crown[0] - back[0]))
    arc(back, r, phi - 17, phi + 17, w=1.3)
arrow((ap_c[0] - 14 * u_in[0], ap_c[1] - 14 * u_in[1]), (crown[0] + 24 * u_in[0], crown[1] + 24 * u_in[1]),
      w=1.4, head=8)

# 2c. the detectors: eight windows on the overhead plate (what pruning keeps)
dets = [182, 226, 262, 300, 340, 384, 436, 500]
for x in dets:
    d.rectangle([P(x - 6.5, 31), P(x + 6.5, 44)], fill=(252, 130, 130, 255), outline=(200, 30, 28, 255),
                width=int(1.2 * S))
d.text(P(545, 12), "Mm-wave\ndetectors", font=font, fill=TEXT + (255,), spacing=2 * S)

# 2d. light scattered by the railhead (and its defects) rising straight to the detectors
for src_pt in ((371.0, 306.0),):
    for x in dets:
        arrow(src_pt, (x, 47), w=1.2, head=7)
for r in (40, 70, 100):
    arc((crown[0], crown[1] + 12), r, 205, 335, w=1.3, fade=0.3)

# 2e. the crown's mirror reflection leaves on the far side, past the detectors (dark field)
u_sp = (-math.sin(theta), -math.cos(theta))
end = (crown[0] + 290 * u_sp[0], crown[1] + 290 * u_sp[1])
dashed((crown[0] + 6 * u_sp[0], crown[1] + 6 * u_sp[1]), end, w=1.5)
arrow((end[0] - 1 * u_sp[0], end[1] - 1 * u_sp[1]), end, w=1.5, head=8)
d.text(P(end[0] - 6, end[1] - 44), "crown glint:\nmisses the detectors", font=small, fill=TEXT + (255,),
       spacing=1 * S)

ov = ov.resize((W, H), Image.LANCZOS)
res = base.convert("RGBA")
res.alpha_composite(ov)
res.convert("RGB").save(out)
print("wrote", out, "horn aperture", [round(v) for v in ap_c], "glint end", [round(v) for v in end])
