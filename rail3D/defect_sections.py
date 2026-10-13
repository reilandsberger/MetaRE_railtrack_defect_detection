"""Rail-head cross-sections of dataset defects, removed material in red.

    python defect_sections.py                 # both figures below
    python defect_sections.py --wear 24       # wear only: the first 24 wear samples
    python defect_sections.py --classes 4     # 4 above-average samples per class

Every panel is a REAL dataset sample: its parameters come from the same seed
the generator uses (config.sample_seed(cls, i)), rendered by the same
mesh3d.render_depth_field, on the reference section without the per-sample
roll/jitter augmentation. The section is cut at the y where that defect is
deepest, so a crack is cut through its lines and a dent through its centre.

  data/figures/wear_sections.png      wear seeds 0..N-1 in order (no selection)
  data/figures/defect_sections.png    each class, ABOVE-AVERAGE samples only:
                                      depth above its class mean, and for crack
                                      / dent / shell also width or footprint
                                      above the mean -- picked so small defects
                                      read on a slide; zoom insets for those.
                                      Not representative of the size spread.
"""

from __future__ import annotations

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from rail3d import config, mesh3d, sections

ARC_DS = 0.1        # plotting arc step (mm): crack walls ramp over ~0.16 mm
Y_DS = 0.25         # y step for finding the deepest slice (mm)
VIEW = dict(xlim=(-42, 45), ylim=(-46, 6))
RED = "tab:red"


def _geom(section):
    return mesh3d.arc_geometry(section, mesh3d.default_arc_count(section, ARC_DS))


def _params(section, geom, cls, i):
    p, _ = mesh3d.defect_params_for_sample(section, cls, None,
                                           config.sample_seed(cls, i),
                                           n_arc=geom["n_arc"])
    return p


def _deepest_slice(p, geom):
    """(y, d(s)) at the y where this defect is deepest."""
    y = np.arange(p["y0"] - 60, p["y0"] + 60 + Y_DS / 2, Y_DS)
    d = mesh3d.render_depth_field(p, geom, y)
    k = int(np.argmax(d.max(axis=1)))
    return float(y[k]), d[k]


def _draw(ax, geom, d, zoom=None):
    pts, n = geom["pts"], geom["normal"]
    x, z = pts[:, 0], pts[:, 1] - config.RAIL_HEIGHT
    w = pts - d[:, None] * n
    wx, wz = w[:, 0], w[:, 1] - config.RAIL_HEIGHT
    for a in [ax] + ([zoom] if zoom is not None else []):
        a.fill(np.r_[x, wx[::-1]], np.r_[z, wz[::-1]], color=RED, alpha=0.35, lw=0)
        a.plot(x, z, "k-", lw=0.9)
        a.plot(wx, wz, "-", color=RED, lw=1.3)
        a.set_aspect("equal")
    ax.set(**VIEW)
    ax.tick_params(labelsize=7)


def _zoom(ax, geom, d, min_half=6.0):
    """Inset framing the whole worn region (cracks are 1.5-3 mm wide)."""
    on = np.flatnonzero(d > 0.02 * d.max())
    xs = geom["pts"][on, 0]
    zs = geom["pts"][on, 1] - config.RAIL_HEIGHT
    half = max(min_half, 0.5 * (xs.max() - xs.min()) + 2.0, 0.5 * d.max() + 2.0)
    cx, cz = 0.5 * (xs.min() + xs.max()), zs.max() - 0.4 * half
    ins = ax.inset_axes([0.03, 0.04, 0.42, 0.52])
    ins.set_xlim(cx - half, cx + half)
    ins.set_ylim(cz - 1.4 * half, cz + 0.6 * half)
    ins.set_xticks([]), ins.set_yticks([])
    for sp in ins.spines.values():
        sp.set_edgecolor("0.35")
    ax.add_patch(Rectangle((cx - half, cz - 1.4 * half), 2 * half, 2 * half,
                           fill=False, lw=0.7, ec="0.35"))
    return ins


def _title(cls, p, y):
    if cls == "wear":
        t = f"gauge {p['depth']:.1f} mm @ {p['gauge_angle_deg']:.0f}°, R {p['gauge_radius']:.0f}"
        return t + (f"\n+ crown {p['crown_depth']:.1f} mm, R {p['crown_radius']:.0f}"
                    if p["crown"] else "\ngauge corner only")
    if cls == "crack":
        th = math.degrees(p["theta"])
        kind = "longitudinal" if abs(th) < 1 else "transverse" if abs(th - 90) < 1 else "oblique"
        along = "\nsection runs ALONG one line" if kind == "transverse" else ""
        return (f"crack, {kind} (θ {th:.0f}°): {p['depth']:.1f} mm deep\n"
                f"3 lines × {p['width']:.1f} mm, gap {p['line_gap']:.1f}, L {p['L']:.0f}; "
                f"cut at y {y:+.1f}{along}")
    if cls == "dent":
        return (f"dent: {p['depth']:.1f} mm deep, FWHM {p['fw_s']:.0f} (across)"
                f" × {p['fw_y']:.0f} (along)\n{len(p['pits'])} pit(s); cut at y {y:+.1f}")
    return (f"shell: {p['depth']:.1f} mm deep, {2 * p['r_s']:.0f} × {2 * p['r_y']:.0f} mm"
            f"\n{len(p['lobes'])} lobe(s), ragged; cut at y {y:+.1f}")


def wear_figure(section, geom, n, out):
    ncol = 6 if n > 16 else 4
    nrow = math.ceil(n / ncol)
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.7 * nrow), squeeze=False)
    n_crown = 0
    for i, ax in enumerate(axs.flat):
        if i >= n:
            ax.axis("off")
            continue
        p = _params(section, geom, "wear", i)
        y, d = _deepest_slice(p, geom)
        n_crown += p["crown"]
        _draw(ax, geom, d)
        ax.set_title(f"wear #{i}: " + _title("wear", p, y), fontsize=7.5)
    fig.suptitle(f"Wear cross-sections, dataset samples 0–{n - 1} in order "
                 f"({n_crown} with crown wear; configured share {config.WEAR_CROWN_PROB:.0%}). "
                 "Black = intact head, red = removed material; x (mm) across the head, "
                 "horn on +x (gauge side)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return n_crown


def _above_average(section, geom, cls, k, scan=400):
    """k above-average samples, scanning dataset seeds in order."""
    ps = [(i, _params(section, geom, cls, i)) for i in range(scan)]
    mean = lambda key: np.mean([p[key] for _, p in ps])  # noqa: E731
    dm = mean("depth")
    if cls == "crack":
        wm = mean("width")
        ok = [(i, p) for i, p in ps if p["depth"] > dm and p["width"] > wm]
        # show each orientation: cross-sections of a transverse crack look
        # nothing like a longitudinal one
        kinds = {"long": lambda t: abs(t) < 1e-6, "trans": lambda t: abs(t - math.pi / 2) < 1e-6,
                 "obl": lambda t: 1e-6 < abs(t) < math.pi / 2 - 1e-6}
        want = ["long", "long", "obl", "trans"][:k] + ["long"] * max(0, k - 4)
        out = []
        for w in want:
            out.append(next(ip for ip in ok if kinds[w](ip[1]["theta"]) and ip not in out))
        return out
    if cls == "dent":
        fm = mean("fw_s")
        ok = [(i, p) for i, p in ps if p["depth"] > dm and p["fw_s"] > fm]
    elif cls == "shell":
        rm = mean("r_s")
        ok = [(i, p) for i, p in ps if p["depth"] > dm and p["r_s"] > rm]
    else:   # wear: half with crown wear, half gauge-only
        ok_c = [(i, p) for i, p in ps if p["depth"] > dm and p["crown"]]
        ok_g = [(i, p) for i, p in ps if p["depth"] > dm and not p["crown"]]
        return [x for pair in zip(ok_c, ok_g) for x in pair][:k]
    return ok[:k]


def classes_figure(section, geom, k, out):
    classes = list(config.CLASS_NAMES)
    fig, axs = plt.subplots(len(classes), k, figsize=(3.9 * k, 2.95 * len(classes)),
                            squeeze=False)
    for r, cls in enumerate(classes):
        for c, (i, p) in enumerate(_above_average(section, geom, cls, k)):
            ax = axs[r][c]
            y, d = _deepest_slice(p, geom)
            ins = _zoom(ax, geom, d) \
                if cls in ("crack", "shell") else None
            _draw(ax, geom, d, ins)
            ax.set_title(f"#{i} " + _title(cls, p, y), fontsize=7.5)
        axs[r][0].set_ylabel(cls.upper(), fontsize=12, fontweight="bold")
    fig.suptitle("Defect cross-sections by class: ABOVE-AVERAGE dataset samples (depth above "
                 "the class mean; crack width / dent and shell footprint above the mean), cut "
                 "where each is deepest.\nBlack = intact head, red = removed material; insets "
                 "magnify the crack and shell regions. x (mm) across the head, horn on +x.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wear", type=int, default=None, help="wear panels (default 24)")
    ap.add_argument("--classes", type=int, default=None, help="samples per class (default 4)")
    a = ap.parse_args()
    both = a.wear is None and a.classes is None
    section = sections.load_reference_section()
    geom = _geom(section)
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    if both or a.wear:
        out = config.FIGURE_DIR / "wear_sections.png"
        nc = wear_figure(section, geom, a.wear or 24, out)
        print(f"  -> {out}  ({nc} of {a.wear or 24} with crown wear)")
    if both or a.classes:
        out = config.FIGURE_DIR / "defect_sections.png"
        classes_figure(section, geom, a.classes or 4, out)
        print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
