"""glint_band.py -- Tier 0 of the experimental-design study (DESIGN_STUDY.md).

Ray optics on the intact railhead: which part of the arc reflects the horn
SPECULARLY into a metasurface aperture. That "glint band" is where the system
sees defects best -- the intact reflection is strong there and a defect's
weak field beats against it (|E_i + E_d|^2 - |E_i|^2 ~ 2 Re(E_i* E_d)); off
the band a defect only adds its own |E_d|^2 onto a dark background. Prelim
recall tracked distance to this band (README finding 25).

The rule underneath every number here: a rail point with outward normal tilted
tau from vertical reflects the horn (theta from vertical, on the +x side) into
the direction alpha from vertical (alpha > 0 toward the horn) when
        tau = (theta + alpha) / 2.
An aperture spans a range of alpha, so it sees a WINDOW of surface tilts; the
rail profile turns that window into arc length at ds/dtau = R (local radius).
The crown is nearly flat (tilt within +-5 deg over ~30 mm), the gauge corner
turns 10 -> 60 deg in ~16 mm: a fixed angular window buys ~6x more arc on the
crown than on the corner -- but seeing the crown's 0 deg tilt means alpha =
-theta, i.e. the crown's own mirror reflection (bright field).

No GPU, no fields: seconds per configuration, thousands per sweep.

    python glint_band.py                          # the production setup
    python glint_band.py --theta 45 --x0 -150     # any other
    python glint_band.py --figures                # the design-study figures + JSON
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rail3d import config, mesh3d, sections   # noqa: E402

PRIORITY = (110.0, 170.0)      # crown / running band through the gauge corner (user, 2026-10-29)
HPBW_DEG = 16.5                # H-A75-W20 model, mean of E / H planes (HORN_RUNG-1_TARGETS.md)
LIT_DB = -10.0                 # a rail point counts as lit inside the horn's -10 dB beam
SMOOTH_MM = 1.5                # normals smoothed over this arc length (Gaussian sigma): the
                               # reference profile is a polyline whose flats show as 2-3 deg
                               # tilt steps; 1.5 mm is < lambda/3, below what PO resolves
FIG = config.FIGURE_DIR / "design_2026-10-29"
OUT_JSON = config.GENERATED_DIR / "design_scan" / "glint_band.json"

# colours shared by every design figure
C_RAIL, C_MS, C_DET, C_HORN = "steelblue", "tab:orange", "tab:green", "tab:red"
C_PRI, C_SEEN, C_SEEN2, C_SEEN3 = "#f2b134", "#7b2cbf", "#1b9e77", "#e7298a"


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
_ARC = {}


def arc(n: int = 4000) -> dict:
    """Intact arc: points, arc length s, outward normals, tilt (deg), local R (mm)."""
    if n not in _ARC:
        sec = sections.load_reference_section()
        g = mesh3d.arc_geometry(sec, n)
        ang = np.unwrap(np.arctan2(g["normal"][:, 0], g["normal"][:, 1]))
        ds = float(np.mean(np.diff(g["s"])))
        sig = SMOOTH_MM / ds
        k = np.arange(-int(4 * sig), int(4 * sig) + 1)
        w = np.exp(-0.5 * (k / sig) ** 2)
        w /= w.sum()
        pad = np.pad(ang, len(k) // 2, mode="edge")
        ang = np.convolve(pad, w, mode="valid")
        g["normal"] = np.stack([np.sin(ang), np.cos(ang)], 1)
        with np.errstate(divide="ignore"):
            R = np.abs(np.gradient(g["s"]) / np.gradient(ang))
        g.update(tilt=np.degrees(ang), R=np.minimum(R, 1e4), section=sec.numpy())
        _ARC[n] = g
    return _ARC[n]


def dist_ant(theta_deg: float, h: float) -> float:
    """Horn aperture-centre distance with its lowest edge level with the plane (finding 31)."""
    th = math.radians(theta_deg)
    return (h + (config.SIZE_ANT[0] / 2 + config.HORN_WALL) * math.sin(th)) / math.cos(th)


@dataclass
class Setup:
    theta: float = 55.0            # horn angle from vertical, toward +x (gauge side)
    phi: float = 0.0               # horn azimuth around the vertical (0 = x-z plane)
    H: float = config.H_MS         # metasurface plane height above the crown
    x0: float = -config.WX / 2     # aperture far edge (away from the horn)
    x1: float = config.WX / 2      # aperture near edge (toward the horn)
    wy: float = config.WY          # aperture along the rail
    D: float | None = None         # horn distance; None = lowest edge level with the plane
    label: str = ""

    def __post_init__(self):
        if self.D is None:
            self.D = dist_ant(self.theta, self.H)

    def horn(self) -> np.ndarray:
        th, ph = math.radians(self.theta), math.radians(self.phi)
        return self.D * np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])


@dataclass
class Result:
    setup: dict
    covered_mm: float                    # priority arc inside the glint band (lit)
    covered_frac: float
    intervals: list                      # visible arc intervals (s_lo, s_hi), whole arc
    crown_specular_x: float              # where the flat crown's mirror reflection lands
    crown_margin_mm: float               # its distance outside the aperture (< 0: inside = bright field)
    bright_field: bool
    horn_clearance_mm: float             # horn's lowest edge to the aperture near edge, in x
    roll_shift_mm: float                 # glint shift on the plane for 2 deg of roll
    lo_rel_dB: float                     # mean glint strength (beam gain x R) vs production
    stationary_y_mm: float               # along-rail position of the glint (azimuth moves it)
    vis: np.ndarray = field(repr=False, default=None)
    gain_db: np.ndarray = field(repr=False, default=None)


def landing_x(st: Setup) -> tuple[np.ndarray, np.ndarray]:
    """y = 0 cut: where each arc point's specular ray lands on the plane, and whether it
    is a real, lit, upward reflection (no aperture limit)."""
    wide = Setup(**{**asdict(st), "x0": -1e9, "x1": 1e9})
    vis, _, _ = trace(wide)
    g = arc()
    hp = st.horn()
    P = np.stack([g["x"], g["z"]], 1)
    inc = P - hp[[0, 2]]
    inc /= np.linalg.norm(inc, axis=1, keepdims=True)
    out = inc - 2 * (inc * g["normal"]).sum(1, keepdims=True) * g["normal"]
    with np.errstate(divide="ignore", invalid="ignore"):
        xl = P[:, 0] + (st.H - P[:, 1]) / out[:, 1] * out[:, 0]
    return xl, vis


def trace(st: Setup, n_y: int = 1, y_half: float = 160.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per arc point: visible (bool), horn beam gain at the glint (dB), its rail y.

    n_y = 1 traces the y = 0 cut (exact for an in-plane horn); an azimuthal horn
    needs n_y > 1 because its glint sits along the rail, between horn and plane.
    """
    g = arc()
    hp = st.horn()
    axis = -hp / np.linalg.norm(hp)                        # horn aimed at the crown origin
    ys = np.zeros(1) if n_y == 1 else np.linspace(-y_half, y_half, n_y)
    P = np.zeros((len(ys), len(g["s"]), 3))
    P[..., 0] = g["x"][None]
    P[..., 1] = ys[:, None]
    P[..., 2] = g["z"][None]
    Nrm = np.zeros_like(P)
    Nrm[..., 0] = g["normal"][None, :, 0]
    Nrm[..., 2] = g["normal"][None, :, 1]
    inc = P - hp
    inc /= np.linalg.norm(inc, axis=-1, keepdims=True)
    cosi = (inc * Nrm).sum(-1)
    out = inc - 2 * cosi[..., None] * Nrm
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (st.H - P[..., 2]) / out[..., 2]
    xl = P[..., 0] + t * out[..., 0]
    yl = P[..., 1] + t * out[..., 1]
    ang = np.degrees(np.arccos(np.clip((inc * axis).sum(-1), -1, 1)))
    gain = -12.0 * (ang / HPBW_DEG) ** 2                   # Gaussian beam, -3 dB at HPBW/2
    ok = (out[..., 2] > 0) & (cosi < 0) & (Nrm[..., 2] > 0) & \
        (xl >= st.x0) & (xl <= st.x1) & (np.abs(yl) <= st.wy / 2)
    gmask = np.where(ok, gain, -np.inf)
    best = gmask.argmax(0)
    gbest = gmask.max(0)
    vis = gbest >= LIT_DB
    return vis, gbest, ys[best]


def intervals(s: np.ndarray, vis: np.ndarray) -> list:
    out, start = [], None
    for k, v in enumerate(vis):
        if v and start is None:
            start = k
        if (not v or k == len(vis) - 1) and start is not None:
            stop = k if v else k - 1
            out.append((round(float(s[start]), 1), round(float(s[stop]), 1)))
            start = None
    return out


_LO_REF = {}


def evaluate(st: Setup, n_y: int | None = None) -> Result:
    g = arc()
    if n_y is None:
        n_y = 1 if st.phi == 0 else 161
    vis, gain, ys = trace(st, n_y)
    ds = np.gradient(g["s"])
    pri = (g["s"] >= PRIORITY[0]) & (g["s"] <= PRIORITY[1])
    cov = float((ds * (vis & pri)).sum())
    th = math.radians(st.theta)
    # flat crown: mirror direction; lands at -H tan(theta) (plus the point's own x)
    crown_x = -st.H * math.tan(th) * (1.0 if st.phi == 0 else math.cos(math.radians(st.phi)))
    margin = st.x0 - crown_x if crown_x < st.x0 else (crown_x - st.x1 if crown_x > st.x1 else
                                                       -min(crown_x - st.x0, st.x1 - crown_x))
    crown = np.abs(g["tilt"]) < 3.0
    bright = bool((vis & crown).any())
    lower_x = st.D * math.sin(th) * math.cos(math.radians(st.phi)) - \
        (config.SIZE_ANT[0] / 2 + config.HORN_WALL) * math.cos(th)
    clear = lower_x - st.x1 if abs(st.D * math.sin(th) * math.sin(math.radians(st.phi))) < st.wy / 2 + 15 \
        else float("inf")
    # roll by d rotates normals by d -> reflected ray by 2d -> shift 2d L / cos^2(alpha) on the plane
    if vis.any():
        a = np.radians(2 * g["tilt"][vis] - st.theta)
        L = st.H - g["z"][vis]
        roll = float(np.median(2 * math.radians(2.0) * L / np.cos(a) ** 2))
        lo = float(np.mean(10 ** (gain[vis] / 10) * g["R"][vis]))
        yst = float(np.median(ys[vis]))
    else:
        roll, lo, yst = float("nan"), 0.0, float("nan")
    if "ref" not in _LO_REF:
        _LO_REF["ref"] = 1.0
        _LO_REF["ref"] = evaluate(Setup()).lo_rel_dB
        _LO_REF["ref"] = 10 ** (_LO_REF["ref"] / 10)
    lo_db = 10 * math.log10(lo / _LO_REF["ref"]) if lo > 0 else float("-inf")
    return Result(setup=asdict(st), covered_mm=cov, covered_frac=cov / (PRIORITY[1] - PRIORITY[0]),
                  intervals=intervals(g["s"], vis), crown_specular_x=crown_x, crown_margin_mm=margin,
                  bright_field=bright, horn_clearance_mm=clear, roll_shift_mm=roll, lo_rel_dB=lo_db,
                  stationary_y_mm=yst, vis=vis, gain_db=gain)


def summary(r: Result) -> str:
    st = r.setup
    iv = ", ".join(f"{a:.0f}-{b:.0f}" for a, b in r.intervals) or "none"
    return (f"theta {st['theta']:.0f} phi {st['phi']:.0f} H {st['H']:.0f} aperture x [{st['x0']:.0f}, "
            f"{st['x1']:.0f}] D {st['D']:.1f}: priority arc covered {r.covered_mm:5.1f} mm "
            f"({100 * r.covered_frac:4.1f}%), visible s {iv}; crown reflection at x "
            f"{r.crown_specular_x:+.0f} ({'INSIDE: bright field' if r.bright_field else f'{r.crown_margin_mm:.0f} mm outside'}); "
            f"horn clearance {r.horn_clearance_mm:.0f} mm; roll 2 deg moves the glint {r.roll_shift_mm:.1f} mm; "
            f"glint strength {r.lo_rel_dB:+.1f} dB vs production; glint at y {r.stationary_y_mm:+.0f} mm")


# ---------------------------------------------------------------------------
# strategies (the menu the figures and DESIGN_STUDY.md walk through)
# ---------------------------------------------------------------------------
def strategies() -> dict:
    return {
        "S0 current": [Setup(label="production")],
        "S1 wider metasurface": [Setup(x0=-150, label="far edge -150"),
                                 Setup(x0=-190, label="far edge -190")],
        "S1b near edge toward horn": [Setup(x1=150, label="near edge +150")],
        "S2 lower horn angle": [Setup(theta=45, label="theta 45"), Setup(theta=35, label="theta 35")],
        "S2b higher horn angle": [Setup(theta=65, label="theta 65")],
        "S3 closer metasurface": [Setup(H=100, label="H 100")],
        "S4 bright-field crown aperture": [Setup(x0=-289, x1=-139, label="second MS at x -214")],
        "S5 two horns (time-multiplexed)": [Setup(label="horn A theta 55"),
                                            Setup(theta=35, label="horn B theta 35")],
        "S6 horn azimuth": [Setup(phi=30, label="phi 30"), Setup(phi=60, label="phi 60")],
        "S7 wide MS + two horns": [Setup(x0=-150, label="horn A theta 55"),
                                   Setup(theta=35, x0=-150, label="horn B theta 35")],
    }


def union_cover(results) -> float:
    g = arc()
    vis = np.zeros_like(g["s"], bool)
    for r in results:
        vis |= r.vis
    pri = (g["s"] >= PRIORITY[0]) & (g["s"] <= PRIORITY[1])
    return float((np.gradient(g["s"]) * (vis & pri)).sum())


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold"})
    return plt


def _arc_segment(ax, lo, hi, color, lw=7, alpha=0.85, z=3, **kw):
    g = arc()
    m = (g["s"] >= lo) & (g["s"] <= hi)
    ax.plot(g["x"][m], g["z"][m], color=color, lw=lw, alpha=alpha, solid_capstyle="butt", zorder=z, **kw)


def _draw_scene(ax, st: Setup, color=C_SEEN, rays=7, show_det=True, crown_ray=True,
                vis=None, horn_label=True, ms_label="metasurface"):
    """Side view (x-z) of the rail, the horn, the aperture and rays to the glint band."""
    g = arc()
    sec = g["section"]
    ax.fill(sec[:, 0], sec[:, 1] - config.RAIL_HEIGHT, color="#dfe7f2", zorder=1)
    ax.plot(sec[:, 0], sec[:, 1] - config.RAIL_HEIGHT, color=C_RAIL, lw=1.0, zorder=2)
    ax.plot([st.x0, st.x1], [st.H, st.H], color=C_MS, lw=3, zorder=4, label=ms_label)
    if show_det:
        ax.plot([st.x0, st.x1], [st.H + 100, st.H + 100], color=C_DET, lw=2, zorder=4, alpha=0.7)
    th = math.radians(st.theta)
    hc = st.horn()
    hx, hz = hc[0], hc[2]
    a = (config.SIZE_ANT[0] / 2) * np.array([-math.cos(th), math.sin(th)])
    if st.phi == 0:
        ax.plot([hx - a[0], hx + a[0]], [hz - a[1], hz + a[1]], color=C_HORN, lw=4, zorder=5)
        # horn flare outline
        back = 28.0 * np.array([math.sin(th), math.cos(th)])
        for sgn in (-1, 1):
            ax.plot([hx + sgn * a[0], hx + back[0] + sgn * a[0] * 0.17],
                    [hz + sgn * a[1], hz + back[1] + sgn * a[1] * 0.17], color=C_HORN, lw=1.2, zorder=5)
        if horn_label:
            ax.text(hx + 8, hz + 18, f"horn {st.theta:.0f}°", color=C_HORN, fontsize=9, ha="left")
    if vis is None:
        vis, _, _ = trace(st)
    idx = np.nonzero(vis)[0]
    if len(idx):
        for k in np.linspace(0, len(idx) - 1, rays).astype(int):
            i = idx[k]
            p = np.array([g["x"][i], g["z"][i]])
            n = g["normal"][i]
            inc = p - np.array([hx, hz])
            inc /= np.linalg.norm(inc)
            out = inc - 2 * (inc @ n) * n
            t = (st.H - p[1]) / out[1]
            q = p + t * out
            ax.plot([hx, p[0]], [hz, p[1]], color=color, lw=0.6, alpha=0.55, zorder=3)
            ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=0.9, alpha=0.8, zorder=3)
        for lo, hi in intervals(g["s"], vis):
            _arc_segment(ax, lo, hi, color, lw=6, z=6)
    if crown_ray and st.phi == 0:
        xs = -st.H * math.tan(th)
        ax.plot([hx, 0, xs], [hz, 0, st.H], color="0.45", ls=(0, (4, 3)), lw=0.8, zorder=2)
        ax.plot([xs], [st.H], marker="x", color="0.3", ms=7, zorder=6)
    ax.set_aspect("equal")


def fig_priority(res0: Result):
    plt = _plt()
    g = arc()
    st = Setup()
    fig = plt.figure(figsize=(17, 5.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 1.0, 1.05], wspace=0.18)

    ax = fig.add_subplot(gs[0])
    _draw_scene(ax, st)
    _arc_segment(ax, *PRIORITY, C_PRI, lw=11, alpha=0.6, z=2)
    xs = -st.H * math.tan(math.radians(st.theta))
    ax.annotate("crown's mirror reflection\nx = −H·tan55° = −214 mm\n(outside: dark field)", (xs, st.H),
                xytext=(xs - 25, st.H - 125), fontsize=8.5, color="0.25",
                arrowprops=dict(arrowstyle="->", color="0.4", lw=0.7))
    ax.text(0, st.H + 7, "metasurface 150 × 75 mm (z = 150)", color=C_MS, ha="center", fontsize=9)
    ax.text(0, st.H + 107, "detector plane (z = 250)", color=C_DET, ha="center", fontsize=9)
    ax.annotate("rays that reflect specularly\ninto the metasurface", (40, 95), xytext=(70, 40),
                fontsize=8.5, color=C_SEEN, arrowprops=dict(arrowstyle="->", color=C_SEEN, lw=0.7))
    ax.set_xlim(-250, 305)
    ax.set_ylim(-75, 300)
    ax.set_xlabel("x (mm)  —  across the rail, + = gauge side")
    ax.set_ylabel("z (mm)")
    ax.set_title("A. The production scene, side view (y = 0)")

    ax = fig.add_subplot(gs[1])
    sec = g["section"]
    ax.fill(sec[:, 0], sec[:, 1] - config.RAIL_HEIGHT, color="#dfe7f2", zorder=1)
    ax.plot(sec[:, 0], sec[:, 1] - config.RAIL_HEIGHT, color=C_RAIL, lw=1.0)
    _arc_segment(ax, g["s"][0], PRIORITY[0], "0.65", lw=6, alpha=0.6)
    _arc_segment(ax, *PRIORITY, C_PRI, lw=10, alpha=0.75)
    for lo, hi in res0.intervals:
        _arc_segment(ax, lo, hi, C_SEEN, lw=6, z=5)
    for s_mark in range(90, 200, 10):
        i = np.argmin(np.abs(g["s"] - s_mark))
        p = np.array([g["x"][i], g["z"][i]])
        n = g["normal"][i]
        q = p + 5.5 * n
        ax.plot([p[0], q[0]], [p[1], q[1]], color="k", lw=0.6)
        r = p + 10.5 * n
        ax.text(r[0], r[1], f"{s_mark}", fontsize=7.5, ha="center", va="center")
    ax.text(-8, 19, "crown / running band", color="#9a6a00", fontsize=9, ha="center", fontweight="bold")
    ax.text(30, 17, "gauge corner", color="#9a6a00", fontsize=9, ha="left", fontweight="bold")
    ax.text(-46, -30, "field side\n(deprioritised)", color="0.45", fontsize=8.5, ha="right")
    lo, hi = res0.intervals[0]
    ax.annotate(f"seen today\ns {lo:.0f}–{hi:.0f} ({res0.covered_mm:.0f} mm)", (26, -3), xytext=(-2, -16),
                color=C_SEEN, fontsize=9, ha="center", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_SEEN, lw=0.9))
    ax.set_xlim(-62, 62)
    ax.set_ylim(-45, 25)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_title("B. Railhead: arc length s (mm) along the surface")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_PRI, alpha=0.75, label=f"priority s {PRIORITY[0]:.0f}–{PRIORITY[1]:.0f} (60 mm)"),
                       Patch(color=C_SEEN, label="specular glint band, today"),
                       Patch(color="0.65", alpha=0.6, label="field side, left out")],
              loc="lower center", fontsize=8.5, frameon=False, ncol=1, bbox_to_anchor=(0.5, -0.02))

    ax = fig.add_subplot(gs[2], projection="3d")
    sel = (g["s"] >= 80) & (g["s"] <= 195)
    yy = np.linspace(-45, 45, 40)
    X = np.repeat(g["x"][sel][None], len(yy), 0)
    Z = np.repeat(g["z"][sel][None], len(yy), 0)
    Y = np.repeat(yy[:, None], sel.sum(), 1)
    S = np.repeat(g["s"][sel][None], len(yy), 0)
    col = np.empty(S.shape + (4,))
    import matplotlib.colors as mc
    col[:] = mc.to_rgba("#dfe7f2")
    pri = (S >= PRIORITY[0]) & (S <= PRIORITY[1])
    col[pri] = mc.to_rgba(C_PRI, 0.95)
    seen = np.zeros_like(S, bool)
    for lo, hi in res0.intervals:
        seen |= (S >= lo) & (S <= hi)
    col[seen & (np.abs(Y) <= 20)] = mc.to_rgba(C_SEEN)
    ax.plot_surface(X, Y, Z, facecolors=col, rstride=1, cstride=8, linewidth=0, antialiased=False, shade=True)
    ax.view_init(elev=38, azim=-62)
    ax.set_box_aspect((90, 90, 45))
    ax.set_xlabel("x")
    ax.set_ylabel("y (along rail)")
    ax.set_zlabel("z")
    ax.set_title("C. Where defects matter vs where we look")
    fig.suptitle(f"The design problem: defects matter across s {PRIORITY[0]:.0f}–{PRIORITY[1]:.0f} mm (60 mm of arc); "
                 f"the production geometry's glint band covers {res0.covered_mm:.0f} mm of it "
                 f"({100 * res0.covered_frac:.0f}%)", fontsize=13, fontweight="bold", y=1.02)
    return fig


def fig_curvature(res0: Result):
    plt = _plt()
    g = arc()
    fig, axs = plt.subplots(1, 3, figsize=(17.5, 5.6), gridspec_kw={"wspace": 0.28})
    s = g["s"]
    ax = axs[0]
    m = (s > 80) & (s < 195)
    ax.axvspan(*PRIORITY, color=C_PRI, alpha=0.25, lw=0, label="priority arc")
    ax.plot(s[m], g["tilt"][m], color="k", lw=1.6)
    # current tilt window
    tv = g["tilt"][res0.vis]
    ax.axhspan(tv.min(), tv.max(), color=C_SEEN, alpha=0.15, lw=0)
    for lo, hi in res0.intervals:
        ax.axvspan(lo, hi, color=C_SEEN, alpha=0.35, lw=0, label="seen today")
    ax.annotate(f"tilt window the aperture sees:\n{tv.min():.0f}° to {tv.max():.0f}°  →  only {res0.covered_mm:.0f} mm of arc",
                (153, tv.max()), xytext=(85, 52), fontsize=9, color=C_SEEN,
                arrowprops=dict(arrowstyle="->", color=C_SEEN, lw=0.8))
    ax.annotate("crown: ±5° over ~30 mm\n(nearly flat, R > 100 mm)", (125, 0), xytext=(84, -38), fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.annotate("gauge corner:\n10° → 60° in 16 mm\n(R ≈ 18–35 mm)", (158, 35), xytext=(166, 2), fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.set_xlabel("arc length s (mm)")
    ax.set_ylabel("surface normal tilt τ from vertical (deg, + toward gauge)")
    ax.set_title("A. The railhead's tilt profile")
    ax.set_xlim(80, 195)
    ax.set_ylim(-70, 95)
    ax.grid(alpha=0.25)

    ax = axs[1]
    ax.axvspan(*PRIORITY, color=C_PRI, alpha=0.25, lw=0)
    ax.semilogy(s[m], np.clip(g["R"][m], 5, 1e4), color="k", lw=1.4)
    for lo, hi in res0.intervals:
        ax.axvspan(lo, hi, color=C_SEEN, alpha=0.35, lw=0)
    ax.set_xlabel("arc length s (mm)")
    ax.set_ylabel("local radius R (mm)  =  mm of arc per radian of tilt")
    ax.set_title("B. Curvature: what one degree of view buys")
    ax.text(128, 2.2e3, "crown: 1° of tilt ≈ 2–20 mm of arc", ha="center", fontsize=9)
    ax.text(160, 7, "corner: 1° ≈ 0.3–0.6 mm", ha="center", fontsize=9)
    ax.set_xlim(80, 195)
    ax.grid(alpha=0.25, which="both")

    ax = axs[2]
    win = (s > 100) & (s < 145)
    i_crown = int(np.nonzero(win)[0][np.argmin(np.abs(g["tilt"][win]))])
    for th, c in ((35, "#1b9e77"), (45, "#66a61e"), (55, C_SEEN), (65, "#e7298a")):
        xl, ok = landing_x(Setup(theta=th))
        mm = ok & (s > 100) & (s < 175)
        ax.plot(np.where(mm, xl, np.nan), s, color=c, lw=1.8, label=f"horn {th}°")
        ax.plot([xl[i_crown]], [s[i_crown]], marker="o", color=c, ms=6)
    ax.axvspan(-75, 75, color=C_MS, alpha=0.18, lw=0, label="today's aperture")
    ax.axhspan(*PRIORITY, color=C_PRI, alpha=0.2, lw=0)
    ax.text(-320, 102, "● the flat crown's mirror point:\n   aperture there = bright field", fontsize=8.5, va="bottom")
    ax.set_xlabel("x on the metasurface plane (mm), H = 150")
    ax.set_ylabel("arc position s that reflects there (mm)")
    ax.set_title("C. Which rail point each metasurface x sees")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_xlim(-330, 200)
    ax.set_ylim(100, 175)
    ax.grid(alpha=0.25)
    fig.suptitle("Why only 9 mm: specular reflection maps an angle window onto a TILT window, τ = (θ_horn + α)/2, "
                 "and the gauge corner packs 50° of tilt into 16 mm", fontsize=12.5, fontweight="bold", y=1.0)
    return fig


def sweep_theta_x0(H=config.H_MS, thetas=np.arange(25, 76, 1.0), x0s=np.arange(-300, -74, 5.0), x1=75.0):
    C = np.zeros((len(thetas), len(x0s)))
    B = np.zeros_like(C, bool)
    for i, th in enumerate(thetas):
        for j, x0 in enumerate(x0s):
            r = evaluate(Setup(theta=float(th), x0=float(x0), x1=x1, H=H))
            C[i, j] = r.covered_mm
            B[i, j] = r.bright_field or r.horn_clearance_mm < 0
    return thetas, x0s, C, B


def fig_levers():
    plt = _plt()
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.8), gridspec_kw={"wspace": 0.3})
    ax = axs[0]
    th, x0s, C, B = sweep_theta_x0()
    im = ax.pcolormesh(x0s, th, C, cmap="viridis", shading="nearest", vmin=0, vmax=40)
    ax.contourf(x0s, th, B.astype(float), levels=[0.5, 1.5], colors="none", hatches=["////"])
    ax.contour(x0s, th, B.astype(float), levels=[0.5], colors="w", linewidths=1.2)
    cs = ax.contour(x0s, th, C, levels=[10, 15, 20, 25, 30], colors="w", linewidths=0.6, alpha=0.8)
    ax.clabel(cs, fmt="%d mm", fontsize=8)
    ax.plot([-75], [55], marker="*", color="w", ms=14, mec="k")
    ax.text(-80, 57, "today", color="w", ha="right", fontsize=9, fontweight="bold")
    ax.text(-285, 29, "hatched: crown glint inside the\naperture (bright field) or the\nhorn over the aperture",
            color="k", fontsize=8.5, bbox=dict(fc="w", ec="none", alpha=0.85))
    fig.colorbar(im, ax=ax, label="priority arc in the glint band (mm)")
    ax.set_xlabel("aperture far edge x0 (mm)   [near edge fixed at +75]")
    ax.set_ylabel("horn angle θ from vertical (deg)")
    ax.set_title("A. Horn angle × metasurface width (H = 150)")

    ax = axs[1]
    x0s2 = np.arange(-300, -74, 5.0)
    for H, c in ((100, "#1b9e77"), (150, C_SEEN), (200, "#d95f02")):
        cov, br = [], []
        for x0 in x0s2:
            r = evaluate(Setup(H=H, x0=float(x0)))
            cov.append(r.covered_mm)
            br.append(r.bright_field)
        cov, br = np.array(cov), np.array(br)
        ax.plot(x0s2, np.where(br, np.nan, cov), color=c, lw=2, label=f"H = {H}")
        ax.plot(x0s2, np.where(br, cov, np.nan), color=c, lw=2, ls=":", alpha=0.7)
    ax.axvline(-75, color=C_MS, lw=1, ls="--")
    ax.text(-78, 2, "today", color=C_MS, ha="right", fontsize=9)
    ax.set_xlabel("aperture far edge x0 (mm), horn 55°")
    ax.set_ylabel("priority arc in the glint band (mm)")
    ax.set_title("B. Plane height: closer sees more, but the\ncrown glint arrives sooner (dotted = bright)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax.set_ylim(0, 60)

    ax = axs[2]
    rows = []
    for H in (100, 150, 200):
        r = evaluate(Setup(H=H))
        rows.append((f"H {H}", r.roll_shift_mm))
    for th in (35, 45, 65):
        r = evaluate(Setup(theta=th))
        rows.append((f"θ {th}°", r.roll_shift_mm))
    r = evaluate(Setup(x0=-190))
    rows.append(("x0 −190", r.roll_shift_mm))
    names, vals = zip(*rows)
    ax.barh(names, vals, color=[C_SEEN if n == "H 150" else "0.6" for n in names])
    ax.axvline(config.DET_SIZE[0], color=C_DET, ls="--", lw=1.2)
    ax.text(config.DET_SIZE[0] + 0.3, -0.45, f"detector window {config.DET_SIZE[0]:.1f} mm", color=C_DET,
            fontsize=8.5)
    ax.set_xlabel("glint shift on the plane for 2° of rail roll (mm)")
    ax.set_title("C. The price of distance: pose nuisance")
    ax.grid(alpha=0.25, axis="x")
    fig.suptitle("Design levers, ray optics only: the wider-metasurface direction stays dark-field the longest; "
                 "lower horn angles and closer planes pull the crown glint in", fontsize=12.5,
                 fontweight="bold", y=1.0)
    return fig


def fig_gallery(results: dict):
    plt = _plt()
    g = arc()
    keys = ["S0 current", "S1 wider metasurface", "S2 lower horn angle", "S3 closer metasurface",
            "S4 bright-field crown aperture", "S5 two horns (time-multiplexed)", "S6 horn azimuth",
            "S7 wide MS + two horns"]
    fig = plt.figure(figsize=(20, 8.6))
    gs = fig.add_gridspec(2, 4, hspace=0.42, wspace=0.06, top=0.9, bottom=0.08)
    notes = {
        "S0 current": "dark field: the crown's mirror\nreflection lands 139 mm outside",
        "S1 wider metasurface": "superset of today's view;\nplane cost ×1.5 (−150) / ×1.8 (−190);\nfar edge at 45–50° incidence",
        "S2 lower horn angle": "moves the window toward the crown;\ncrown glint closes in (−150 / −105)",
        "S3 closer metasurface": "same as a wider aperture in angle;\nroll nuisance ×0.8; crown glint at −143",
        "S4 bright-field crown aperture": "sees the flat crown in its own\nmirror beam: +10 dB, but 2° of roll\nmoves it 30 mm (rays arrive at 55°)",
        "S5 two horns (time-multiplexed)": "union of two windows,\none metasurface, frames alternate",
        "S6 horn azimuth": "NOT dark field: acts as a lower angle,\ntan θ_eff = tan θ·cos φ; the glint moves\nalong the rail; horn sits BESIDE the MS",
        "S7 wide MS + two horns": "the two cheapest coverage levers\ncombined; horn B's channel is now\nbright field (crown glint inside)",
    }
    for k, key in enumerate(keys):
        ax = fig.add_subplot(gs[k // 4, k % 4])
        sts = results[key]
        cols = [C_SEEN, C_SEEN2, C_SEEN3]
        if key == "S6 horn azimuth":
            _gallery_azimuth(ax, sts)
        else:
            for j, r in enumerate(sts):
                st = Setup(**{kk: r.setup[kk] for kk in ("theta", "phi", "H", "x0", "x1", "wy", "D", "label")})
                _draw_scene(ax, st, color=cols[j % 3], rays=5, show_det=False, crown_ray=(j == 0),
                            vis=r.vis, horn_label=False, ms_label=None)
                if key in ("S1 wider metasurface",):
                    ax.plot([st.x0, st.x1], [st.H + 4 * j, st.H + 4 * j], color=C_MS, lw=2)
            if key == "S4 bright-field crown aperture":
                r0 = results["S0 current"][0]
                _draw_scene(ax, Setup(), color=C_SEEN, rays=0, show_det=False, crown_ray=False,
                            vis=r0.vis, horn_label=False)
        cov = union_cover(sts) if key != "S6 horn azimuth" else max(r.covered_mm for r in sts)
        lab = " / ".join(f"{r.setup['label']}: {r.covered_mm:.0f} mm" for r in sts)
        if key in ("S5 two horns (time-multiplexed)", "S7 wide MS + two horns"):
            lab += f"  →  union {cov:.0f} mm"
        ax.set_title(f"{key}", loc="left")
        ax.text(0.01, 0.99, lab, transform=ax.transAxes, va="top", fontsize=8.3,
                bbox=dict(fc="w", ec="0.8", alpha=0.9))
        npos = (0.99, 0.02, "bottom") if key == "S6 horn azimuth" else (0.99, 0.86, "top")
        ax.text(npos[0], npos[1], notes[key], transform=ax.transAxes, ha="right", va=npos[2], fontsize=8.3,
                color="0.2", bbox=dict(fc="#fafafa", ec="none", alpha=0.85))
        if key != "S6 horn azimuth":
            ax.set_xlim(-310, 300)
            ax.set_ylim(-60, 300)
        else:
            ax.set_xlim(-210, 400)
            ax.set_ylim(-100, 260)
        ax.set_xticks([])
        ax.set_yticks([])
        # coverage strip under the panel
        ins = ax.inset_axes([0.0, -0.075, 1.0, 0.05])
        ins.axvspan(*PRIORITY, color=C_PRI, alpha=0.5, lw=0)
        for j, r in enumerate(sts):
            for lo, hi in r.intervals:
                ins.axvspan(lo, hi, ymin=0.5 * (j % 2), ymax=0.5 * (j % 2) + 0.5, color=cols[j % 3], lw=0)
        ins.set_xlim(85, 195)
        ins.set_yticks([])
        ins.tick_params(labelsize=7, length=2)
        if k // 4 == 1:
            ins.set_xlabel("arc s (mm): priority (amber) vs seen", fontsize=8)
    fig.suptitle("Strategy menu (ray optics): every option is a different TILT WINDOW. "
                 "Numbers = priority arc (s 110–170) inside the glint band", fontsize=13.5, fontweight="bold",
                 y=0.99)
    return fig


def _gallery_azimuth(ax, results):
    """Top view: the glint of an azimuthal horn sits between horn and plane, along the rail."""
    g = arc()
    ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
        (-40, -160), 80, 320, color="#dfe7f2"))
    ax.plot([-75, 75, 75, -75, -75], [-37.5, -37.5, 37.5, 37.5, -37.5], color=C_MS, lw=2)
    cols = [C_SEEN2, C_SEEN3]
    for j, r in enumerate(results):
        st = Setup(**{kk: r.setup[kk] for kk in ("theta", "phi", "H", "x0", "x1", "wy", "D", "label")})
        hp = st.horn()
        ax.plot(hp[0], hp[1], marker="s", color=C_HORN, ms=8)
        ax.text(hp[0] + 6, hp[1], f"horn φ {st.phi:.0f}°", fontsize=8.5, color=C_HORN, va="center")
        ax.plot([hp[0], 0], [hp[1], 0], color=C_HORN, lw=0.8, ls="--")
        v = r.vis
        if v.any():
            ax.scatter(g["x"][v][::20], np.full(v[::20].sum(), r.stationary_y_mm), s=10, color=cols[j],
                       zorder=5)
            ax.text(-200, r.stationary_y_mm + 12 * (2 * j - 1), f"glint at y ≈ {r.stationary_y_mm:+.0f} mm, beam "
                    f"{np.median(r.gain_db[v]):.0f} dB", fontsize=8.3, color=cols[j], va="center")
    ax.plot(0, 0, marker="+", color="k", ms=10)
    ax.set_aspect("equal")
    ax.text(0.02, 0.97, "top view (x–y), rail runs up the page;\norange = metasurface", transform=ax.transAxes,
            fontsize=8.3, va="top")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--theta", type=float, default=55.0)
    ap.add_argument("--phi", type=float, default=0.0)
    ap.add_argument("--H", type=float, default=config.H_MS)
    ap.add_argument("--x0", type=float, default=-config.WX / 2)
    ap.add_argument("--x1", type=float, default=config.WX / 2)
    ap.add_argument("--figures", action="store_true")
    a = ap.parse_args()
    if not a.figures:
        print(summary(evaluate(Setup(theta=a.theta, phi=a.phi, H=a.H, x0=a.x0, x1=a.x1))))
        return 0
    FIG.mkdir(parents=True, exist_ok=True)
    plt = _plt()
    res = {k: [evaluate(st) for st in v] for k, v in strategies().items()}
    for k, rs in res.items():
        for r in rs:
            print(f"{k:34s} {r.setup['label']:20s} {summary(r)}")
    r0 = res["S0 current"][0]
    for name, fig in (("01_priority_region", fig_priority(r0)), ("02_tilt_window", fig_curvature(r0)),
                      ("04_design_levers", fig_levers()), ("05_strategy_menu", fig_gallery(res))):
        p = FIG / f"{name}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {p}")
    js = {k: [{kk: vv for kk, vv in asdict(r).items() if kk not in ("vis", "gain_db")} for r in rs]
          for k, rs in res.items()}
    for k in ("S5 two horns (time-multiplexed)", "S7 wide MS + two horns"):
        js[k + " | union_mm"] = union_cover(res[k])
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(js, indent=1, default=float))
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
