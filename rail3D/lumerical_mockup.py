"""Draw the TARGET rung-2 setup into a screenshot of the Lumerical FDTD layout.

    python lumerical_mockup.py --screenshot path/to/lumerical_layout.png

An illustration of what the finished setup should look like in the FDTD
2025 R1 Layout window, drawn into the four CAD panes and the Objects Tree of
a real screenshot. It is a MOCKUP, not a run: the title bar and every pane carry
a TARGET label, so no crop of it can pass for a completed simulation.

Every box is taken from the SAME calls that write case.json --
compare_wavefronts.fdtd_plan (region, TFSF, monitor) and defect_bbox (the mesh
override) -- on the geometry the exporter actually writes, so the picture
cannot drift from LUMERICAL.md or from the STL you import. Rung 2 (intact rail)
is drawn WITH the crack's mesh override because LUMERICAL.md 3.5 requires the
identical override in the intact run; rung 3 only swaps the STL.

Pane rectangles are measured for a 1772x868 screenshot of the default
four-view layout; another size is refused rather than drawn in the wrong place.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import compare_wavefronts as cw                      # noqa: E402 (Agg backend)
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                   # noqa: E402
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle   # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection  # noqa: E402
from PIL import Image, ImageDraw, ImageFont          # noqa: E402

from rail3d import config, mesh3d, sections          # noqa: E402

SIZE = (1772, 868)
PANES = {"xy": (152, 214, 959, 520), "persp": (963, 214, 1769, 520),
         "xz": (152, 551, 959, 857), "yz": (963, 551, 1769, 857)}
# colours sampled from the screenshot itself
PML = (153 / 255, 76 / 255, 0)
REGION = (253 / 255, 126 / 255, 0)
HANDLE = (250 / 255, 71 / 255, 71 / 255)
GRID = (0.16, 0.16, 0.16)
SOURCE = "white"
K_ARROW = (0.72, 0.38, 0.95)
E_ARROW = (0.30, 0.60, 1.00)
MONITOR = (1.0, 0.86, 0.0)
OVERRIDE = (0.20, 0.85, 0.90)
RAIL = (0.62, 0.66, 0.72)
PML_MM = 8 * config.WVL / 10          # 8 PML layers at the lambda/10 mesh
GRID_PX = 20                          # the CAD grid is ~20 px in the screenshot


def setup_geometry() -> dict:
    """Region/TFSF/monitor from fdtd_plan, override from defect_bbox, rail outline."""
    sec = sections.load_reference_section()
    g = dict(cw.geom_now(), h_ms=30.0)
    n = mesh3d.default_arc_count(sec, config.MESH_DS)
    v, _ = cw.build_mesh(sec, {"class": "intact"}, config.MESH_DS, n, None)
    v = v.numpy()
    plan = cw.fdtd_plan(v, g, 30.0, None)
    defect = None
    if "crack" in config.DATASET_DIRS:
        files = sections.get_dataset_files("crack")
        defect = sections.match_reference_width(sections.load_vertices_from_csv(files[0]), sec)
    params, _ = mesh3d.defect_params_for_sample(sec, "crack", defect,
                                                config.sample_seed("crack", 0), n_arc=n)
    ov = cw.defect_bbox(sec, params, g, None)["override_box_mm"]
    # the rail's cross-section: the first swept slice, in arc order
    y0 = v[:, 1].min()
    sl = v[np.isclose(v[:, 1], y0, atol=1e-6)]
    return {"sim": plan["simulation_region_mm"], "tfsf": plan["tfsf_source_mm"],
            "mon": plan["monitor_mm"], "ov": ov,
            "rail_xz": sl[:, [0, 2]], "rail_y": (float(v[:, 1].min()), float(v[:, 1].max())),
            "theta": float(np.degrees(config.THETA_INC))}


def view_axes(pane: str, span_v: tuple, centre_h: float):
    """A figure exactly the pane's pixel size, equal aspect, Lumerical black."""
    x0, y0, x1, y1 = PANES[pane]
    w, h = x1 - x0, y1 - y0
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100, facecolor="black")
    ax = fig.add_axes([0, 0, 1, 1], facecolor="black")
    lo, hi = span_v
    half_h = (hi - lo) * (w / h) / 2
    ax.set_xlim(centre_h - half_h, centre_h + half_h)
    ax.set_ylim(lo, hi)
    ax.set_axis_off()
    mm_per_px = (hi - lo) / h
    step = GRID_PX * mm_per_px
    for gx in np.arange(centre_h - half_h, centre_h + half_h, step):
        ax.axvline(gx, color=GRID, lw=0.6, zorder=0)
    for gy in np.arange(lo, hi, step):
        ax.axhline(gy, color=GRID, lw=0.6, zorder=0)
    return fig, ax, mm_per_px


def rect(ax, a, b, **kw):
    ax.add_patch(Rectangle((a[0], b[0]), a[1] - a[0], b[1] - b[0], **kw))


def region(ax, a, b):
    """FDTD region + PML band, as Lumerical draws them."""
    rect(ax, (a[0] - PML_MM, a[1] + PML_MM), (b[0] - PML_MM, b[1] + PML_MM),
         facecolor=PML, edgecolor="none", zorder=1)
    rect(ax, a, b, facecolor="black", edgecolor=REGION, lw=2.0, zorder=2)


def handles(ax, a, b, mm_per_px):
    """Red selection handles at the edge midpoints of the selected object."""
    s = 5 * mm_per_px
    for x, y in [(a[0], sum(b) / 2), (a[1], sum(b) / 2), (sum(a) / 2, b[0]), (sum(a) / 2, b[1])]:
        rect(ax, (x - s / 2, x + s / 2), (y - s / 2, y + s / 2),
             facecolor=HANDLE, edgecolor="none", zorder=9)


def arrow(ax, p, d, color, mm_per_px, length_px=70):
    L = length_px * mm_per_px
    q = (p[0] + d[0] * L, p[1] + d[1] * L)
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=18,
                                 color=color, lw=2.2, zorder=8))


def to_image(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def draw_xy(G):
    s, t, m, o = G["sim"], G["tfsf"], G["mon"], G["ov"]
    fig, ax, mpp = view_axes("xy", (s["y"][0] - 14, s["y"][1] + 14), 0.0)
    region(ax, s["x"], s["y"])
    xr = (G["rail_xz"][:, 0].min(), G["rail_xz"][:, 0].max())
    rect(ax, xr, G["rail_y"], facecolor=RAIL, edgecolor="#d8dde3", lw=1, alpha=0.85, zorder=3)
    ax.plot([0, 0], G["rail_y"], color="#7c848f", lw=0.8, zorder=3)         # crown line
    rect(ax, t["x"], t["y"], facecolor="none", edgecolor=SOURCE, lw=1.4, zorder=5)
    rect(ax, o["x"], o["y"], facecolor="none", edgecolor=OVERRIDE, lw=1.4, ls="--", zorder=6)
    rect(ax, m["x"], m["y"], facecolor="none", edgecolor=MONITOR, lw=1.8, zorder=7)
    handles(ax, m["x"], m["y"], mpp)
    arrow(ax, (30, 0), (-1, 0), K_ARROW, mpp)                                 # k projected
    arrow(ax, (30, 0), (0, 1), E_ARROW, mpp, 45)                              # E along y
    return to_image(fig)


def draw_xz(G):
    s, t, m, o = G["sim"], G["tfsf"], G["mon"], G["ov"]
    fig, ax, mpp = view_axes("xz", (s["z"][0] - 12, s["z"][1] + 12), 0.0)
    region(ax, s["x"], s["z"])
    ax.add_patch(Polygon(G["rail_xz"], closed=True, facecolor=RAIL, edgecolor="#d8dde3",
                         lw=1, alpha=0.9, zorder=3))
    rect(ax, t["x"], t["z"], facecolor="none", edgecolor=SOURCE, lw=1.4, zorder=5)
    rect(ax, o["x"], o["z"], facecolor="none", edgecolor=OVERRIDE, lw=1.4, ls="--", zorder=6)
    ax.plot(m["x"], [m["z"], m["z"]], color=MONITOR, lw=2.4, zorder=7)
    handles(ax, m["x"], (m["z"], m["z"]), mpp)
    th = np.radians(G["theta"])
    start = (t["x"][1] - 6, t["z"][1] - 4)
    arrow(ax, start, (-np.sin(th), -np.cos(th)), K_ARROW, mpp, 80)          # 55 deg, down-left
    ax.plot(*start, marker="o", ms=9, mfc="none", mec=E_ARROW, mew=2, zorder=8)
    ax.plot(*start, marker=".", ms=6, color=E_ARROW, zorder=8)              # E out of page (+y)
    return to_image(fig)


def draw_yz(G):
    s, t, m, o = G["sim"], G["tfsf"], G["mon"], G["ov"]
    fig, ax, mpp = view_axes("yz", (s["z"][0] - 12, s["z"][1] + 12), 0.0)
    region(ax, s["y"], s["z"])
    zr = (G["rail_xz"][:, 1].min(), G["rail_xz"][:, 1].max())
    rect(ax, G["rail_y"], zr, facecolor=RAIL, edgecolor="#d8dde3", lw=1, alpha=0.9, zorder=3)
    rect(ax, t["y"], t["z"], facecolor="none", edgecolor=SOURCE, lw=1.4, zorder=5)
    rect(ax, o["y"], o["z"], facecolor="none", edgecolor=OVERRIDE, lw=1.4, ls="--", zorder=6)
    ax.plot(m["y"], [m["z"], m["z"]], color=MONITOR, lw=2.4, zorder=7)
    handles(ax, m["y"], (m["z"], m["z"]), mpp)
    arrow(ax, (0, t["z"][1] - 4), (0, -1), K_ARROW, mpp, 55)                 # k (its z part)
    arrow(ax, (0, t["z"][1] - 4), (1, 0), E_ARROW, mpp, 45)                  # E along +y
    return to_image(fig)


def box_edges(a, b, c):
    xs, ys, zs = a, b, c
    P = [(x, y, z) for x in xs for y in ys for z in zs]
    E = []
    for i in range(8):
        for j in range(i + 1, 8):
            if sum(P[i][k] != P[j][k] for k in range(3)) == 1:
                E.append((P[i], P[j]))
    return E


def draw_persp(G):
    x0, y0, x1, y1 = PANES["persp"]
    w, h = x1 - x0, y1 - y0
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100, facecolor="black")
    ax = fig.add_axes([0.0, -0.16, 1.0, 1.30], projection="3d", facecolor="black")
    ax.set_axis_off()
    # explicit draw order: like Lumerical's CAD view, object wireframes and
    # arrows show THROUGH structures instead of being depth-sorted behind the
    # rail (matplotlib sorts whole collections, which hid the override + k/E)
    ax.computed_zorder = False
    s, t, m, o = G["sim"], G["tfsf"], G["mon"], G["ov"]
    ax.add_collection3d(Line3DCollection(box_edges(s["x"], s["y"], s["z"]),
                                         colors=[REGION], linewidths=1.6, zorder=2))
    ax.add_collection3d(Line3DCollection(box_edges(t["x"], t["y"], t["z"]),
                                         colors=[SOURCE], linewidths=1.0, zorder=3))
    ax.add_collection3d(Line3DCollection(box_edges(o["x"], o["y"], o["z"]),
                                         colors=[OVERRIDE], linewidths=1.2, linestyles="--",
                                         zorder=4))
    # subsample the outline, then join CONSECUTIVE points: stepping i by 2 while
    # joining i to i+1 skipped every other segment and striped the surface
    xz = G["rail_xz"][::2]
    ya, yb = G["rail_y"]
    quads = [[(xz[i, 0], ya, xz[i, 1]), (xz[i + 1, 0], ya, xz[i + 1, 1]),
              (xz[i + 1, 0], yb, xz[i + 1, 1]), (xz[i, 0], yb, xz[i, 1])]
             for i in range(len(xz) - 1)]
    # Lambertian shading computed here: Poly3DCollection(shade=True) with no
    # edges multiplies by an EMPTY edge-colour array and raises
    seg = xz[1:] - xz[:-1]
    nrm = np.stack([seg[:, 1], np.zeros(len(seg)), -seg[:, 0]], axis=1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True).clip(1e-9)
    light = np.array([0.45, -0.55, 0.70]) / np.linalg.norm([0.45, -0.55, 0.70])
    k = 0.35 + 0.65 * np.abs(nrm @ light)
    ax.add_collection3d(Poly3DCollection(quads, facecolors=[(*(np.array(RAIL) * kk), 0.95)
                                                            for kk in k],
                                         edgecolors="none", zorder=1))
    mz = m["z"]
    ax.add_collection3d(Poly3DCollection(
        [[(m["x"][0], m["y"][0], mz), (m["x"][1], m["y"][0], mz),
          (m["x"][1], m["y"][1], mz), (m["x"][0], m["y"][1], mz)]],
        facecolors=[(*MONITOR, 0.22)], edgecolors=[MONITOR], linewidths=1.4, zorder=5))
    th = np.radians(G["theta"])
    p = np.array([t["x"][1] - 5, 0.0, t["z"][1] - 3])
    d = np.array([-np.sin(th), 0.0, -np.cos(th)]) * 45
    ax.quiver(*p, *d, color=K_ARROW, linewidth=2.4, arrow_length_ratio=0.25, zorder=6)
    ax.quiver(*p, 0, 30, 0, color=E_ARROW, linewidth=2.2, arrow_length_ratio=0.3, zorder=6)
    ax.set_xlim(*s["x"]), ax.set_ylim(*s["y"]), ax.set_zlim(*s["z"])
    ax.set_box_aspect((s["x"][1] - s["x"][0], s["y"][1] - s["y"][0], s["z"][1] - s["z"][0]))
    ax.view_init(elev=24, azim=-58)
    img = to_image(fig)
    # legend, drawn in pixel space so it stays legible at pane size
    d = ImageDraw.Draw(img)
    f = _font(12)
    items = [("FDTD region / PML", REGION), ("TFSF source (s-pol, 55 deg)", (1, 1, 1)),
             ("z = 30 mm monitor", MONITOR), ("mesh override (lambda/20)", OVERRIDE),
             ("rail, PEC (STL)", RAIL), ("k", K_ARROW), ("E (along y)", E_ARROW)]
    x, y = w - 205, h - 16 * len(items) - 8
    d.rectangle([x - 8, y - 6, w - 6, h - 4], fill=(18, 18, 18), outline=(70, 70, 70))
    for i, (lab, col) in enumerate(items):
        c = tuple(int(255 * v) for v in col)
        d.line([x, y + 16 * i + 7, x + 18, y + 16 * i + 7], fill=c, width=3)
        d.text((x + 26, y + 16 * i), lab, fill=(225, 225, 225), font=f)
    return img


def _font(size: int, bold: bool = False):
    for name in (("segoeuib.ttf" if bold else "segoeui.ttf"), "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def tag(img: Image.Image, text: str):
    d = ImageDraw.Draw(img)
    f = _font(12, bold=True)
    tw = d.textlength(text, font=f)
    d.rectangle([6, img.height - 24, 16 + tw, img.height - 5], fill=(40, 0, 0),
                outline=(220, 40, 40))
    d.text((11, img.height - 23), text, fill=(255, 110, 110), font=f)


def objects_tree(shot: Image.Image):
    """Replace the tree rows with the target objects (names fit the narrow pane)."""
    # crop the real icons BEFORE the panel is blanked, or they come out empty
    icons = {"model": shot.crop((46, 307, 66, 325)), "FDTD": shot.crop((72, 330, 92, 348)),
             "source": shot.crop((72, 353, 92, 371))}
    d = ImageDraw.Draw(shot)
    d.rectangle([44, 305, 140, 620], fill=(230, 230, 230))
    f = _font(15)
    rows = [("model", 0, icons["model"], None), ("FDTD", 1, icons["FDTD"], None),
            ("rail", 1, None, RAIL), ("TFSF", 1, icons["source"], None),
            ("mesh", 1, None, OVERRIDE), ("mon30", 1, None, MONITOR)]
    for i, (name, lvl, icon, col) in enumerate(rows):
        y = 307 + 23 * i
        if name == "mon30":                                   # the selected object
            d.rectangle([45, y - 1, 140, y + 19], fill=(167, 206, 235))
        ix = 47 + 26 * lvl
        if icon is not None:
            shot.paste(icon, (ix, y))
        else:
            c = tuple(int(255 * v) for v in col)
            d.rectangle([ix + 2, y + 3, ix + 16, y + 16], fill=c, outline=(60, 60, 60))
        d.text((ix + 23, y - 2), name, fill=(0, 0, 0), font=f)


def title_bar(shot: Image.Image):
    d = ImageDraw.Draw(shot)
    d.rectangle([24, 0, 1150, 21], fill=(243, 243, 243))
    f, fb = _font(15), _font(15, bold=True)
    base = "Ansys Lumerical 2025 R1 Finite Difference IDE - rail3D_rung2_intact.fsp   "
    d.text((27, 1), base, fill=(0, 0, 0), font=f)
    d.text((27 + d.textlength(base, font=f), 1),
           "[TARGET SETUP - illustration, not a completed run]", fill=(200, 0, 0), font=fb)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--screenshot", required=True,
                    help="a 1772x868 screenshot of the FDTD Layout window (four views)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    try:
        shot = Image.open(args.screenshot).convert("RGB")
    except (FileNotFoundError, OSError) as exc:
        print(f"!! cannot read {args.screenshot} as an image: {exc}")
        return 1
    if shot.size != SIZE:
        print(f"!! screenshot is {shot.size}; the pane rectangles are measured for "
              f"{SIZE}. Take it at that size (or update PANES) rather than drawing "
              f"the setup in the wrong place.")
        return 1
    G = setup_geometry()
    print(f"region {G['sim']}\nTFSF   {G['tfsf']}\nmonitor {G['mon']}\noverride {G['ov']}")
    label = "TARGET SETUP (mockup) - rung 2 intact rail - LUMERICAL.md"
    for pane, fn in (("xy", draw_xy), ("xz", draw_xz), ("yz", draw_yz), ("persp", draw_persp)):
        img = fn(G)
        tag(img, label)
        shot.paste(img, PANES[pane][:2])
    objects_tree(shot)
    title_bar(shot)
    out = Path(args.out) if args.out else config.FIGURE_DIR / "lumerical_target_setup.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    shot.save(out)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
