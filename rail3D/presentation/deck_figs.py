"""Figures for the horn / results slides of rail3D_overview.pptx.

Run from rail3D/. Every number drawn is computed here from the code the
simulator runs (field3d, horn_fdtd_case), and printed so it can be quoted:

    python presentation/deck_figs.py data/figures/deck_2026-09-24 <stage_prelim_bundle dir>

Writes horn_drawing / horn_equations / horn_aperture / horn_rail PNGs,
horn_facts.json (every quoted number), and copies/crops the prelim bundle's
analysis figures. README finding 29 explains the physics.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402
from matplotlib.patches import Polygon, Rectangle, FancyArrowPatch   # noqa: E402
from PIL import Image                                 # noqa: E402
from scipy.special import fresnel                     # noqa: E402

sys.path.insert(0, str(Path.cwd()))
from rail3d import config, field3d                    # noqa: E402
import horn_fdtd_case as hf                           # noqa: E402

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
INK, MUTE, ACC, GOOD, BAD = "#1B2733", "#7B8494", "#1F6FEB", "#1A7F5A", "#B42318"
OLDC = "#9AA3B2"
plt.rcParams.update({"font.family": ["Calibri", "DejaVu Sans"], "font.size": 12,
                     "axes.edgecolor": "#C9CFD8", "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.titlecolor": INK,
                     "mathtext.fontset": "stix", "savefig.dpi": 200})

WVL = config.WVL
NEW = tuple(config.SIZE_ANT)
OLD = tuple(v * WVL / 8.0 for v in (27.4, 21.9, 9.3, 6.2, 27.0))   # Face3D horn, x5/8
HORNS = {"old": dict(size=OLD, flare_k="beta_wvg", sampling="linspace",
                     label="previous model (Face3D horn ×5/8)", c=OLDC, ls="--"),
         "new": dict(size=NEW, flare_k="k0", sampling="midpoint",
                     label="RFspin H-A75-W20 (this model)", c=ACC, ls="-")}
facts = {}


def closed_form_D(size, wvl):
    """Nikolova L18 eq. 18.39 (with 18.43-18.44 for the apex distances)."""
    A, B, a, b, L = size
    rh, re = A * L / (A - a), B * L / (B - b)
    t = A ** 2 / (8 * wvl * rh)
    p1, p2 = 2 * np.sqrt(t) * (1 + 1 / (8 * t)), 2 * np.sqrt(t) * (-1 + 1 / (8 * t))
    S1, C1 = fresnel(p1)
    S2, C2 = fresnel(p2)
    eph_h = np.pi ** 2 / (64 * t) * ((C1 - C2) ** 2 + (S1 - S2) ** 2)
    q = B / np.sqrt(2 * wvl * re)
    Sq, Cq = fresnel(q)
    eph_e = (Cq ** 2 + Sq ** 2) / q ** 2
    et = 8 / np.pi ** 2
    D = 4 * np.pi / wvl ** 2 * A * B * et * eph_e * eph_h
    return dict(rho_h=rh, rho_e=re, t=t, q=q, eps_t=et, eps_ph_H=eph_h, eps_ph_E=eph_e,
                eps_ap=et * eph_e * eph_h, D_dBi=10 * np.log10(D))


def aperture_grid(size, flare_k, d=0.05):
    A, B = size[:2]
    x = np.arange(-A / 2 + d / 2, A / 2, d)
    y = np.arange(-B / 2 + d / 2, B / 2, d)
    X, Y = np.meshgrid(x, y, indexing="ij")
    return x, y, field3d.aperture_distribution(X, Y, size, WVL, flare_k)


# ---------------------------------------------------------------- numbers
for key, h in HORNS.items():
    x, y, E = aperture_grid(h["size"], h["flare_k"])
    ff = hf.far_field_cuts(E, x, y)
    h.update(x=x, y=y, E=E, ff=ff)
    cf = closed_form_D(h["size"], WVL)
    fx = {"size_mm": [round(v, 4) for v in h["size"]],
          "hpbw_E_deg": ff["hpbw_E_deg"], "hpbw_H_deg": ff["hpbw_H_deg"],
          "D_numeric_dBi": ff["directivity_dBi"], "closed_form": cf,
          "fraunhofer_mm": 2 * (h["size"][0] ** 2 + h["size"][1] ** 2) / WVL}
    fx["rail_over_fraunhofer"] = config.DIST_ANT / fx["fraunhofer_mm"]
    a, b = h["size"][2:4]
    c = 299.792458   # mm * GHz
    modes = sorted(((c / 2) * np.hypot(m / a, n / b), f"TE{m}{n}")
                   for m, n in ((1, 0), (2, 0), (0, 1), (1, 1), (3, 0)))
    fx["feed_cutoffs_GHz"] = [(nm, round(f, 1)) for f, nm in modes]
    fx["modes_at_60GHz"] = [nm for f, nm in modes if f < 60.0]
    fx["beta_over_k"] = float(np.sqrt(1 - (WVL / (2 * a)) ** 2))
    facts[key] = fx
facts["new"]["band_dBi"] = {f: closed_form_D(NEW, 299.792458 / f)["D_dBi"] for f in (50, 60, 75)}

# rail footprint at crown height: what the horn lights on the rail
dx = 1.0
xs = np.arange(-60, 80 + 1e-6, dx)
ys = np.arange(-90, 90 + 1e-6, dx)
Xt, Yt = torch.meshgrid(torch.as_tensor(xs, dtype=torch.float32),
                        torch.as_tensor(ys, dtype=torch.float32), indexing="ij")
for key, h in HORNS.items():
    psi = field3d.horn_to_plane(Xt[None], Yt[None], 0.0, WVL, config.THETA_INC, h["size"],
                                config.DIST_ANT, config.RESOL_ANT, h["flare_k"],
                                h["sampling"]).reshape(len(xs), len(ys)).numpy()
    h["foot"] = psi
    amp = np.abs(psi) / np.abs(psi).max()
    ix0 = np.argmin(abs(xs))
    ycut = amp[ix0]
    lit = ys[ycut >= np.sqrt(0.5)]
    facts[key]["foot_y_minus3dB_span_mm"] = float(lit.max() - lit.min())
    facts[key]["foot_amp_at_rail_ends"] = float(np.interp(60.0, ys, ycut) / ycut.max())
    xcut = amp[:, np.argmin(abs(ys))]
    litx = xs[xcut >= np.sqrt(0.5)]
    facts[key]["foot_x_minus3dB_span_mm"] = float(litx.max() - litx.min())
box = (abs(xs)[:, None] <= 37) & (abs(ys)[None, :] <= 60)
o, n = HORNS["old"]["foot"][box], HORNS["new"]["foot"][box]
facts["rail_illumination_corr_old_vs_new"] = float(abs(np.vdot(o, n)) / (np.linalg.norm(o) * np.linalg.norm(n)))
(OUT / "horn_facts.json").write_text(json.dumps(facts, indent=2, default=float), encoding="utf-8")
print(json.dumps(facts, indent=1, default=float))


# ---------------------------------------------------------------- 1. drawing
def horn_cut(ax, W, w, L, rho, WG, lab_W, lab_w, plane, dims):
    """One principal-plane cut of the horn: waveguide + flare + apex rays."""
    t = config.SIZE_ANT and 0.5
    ax.add_patch(Polygon([(-WG, -w / 2), (0, -w / 2), (L, -W / 2), (L, W / 2), (0, w / 2),
                          (-WG, w / 2)], closed=True, fc="#EAF1FD", ec=INK, lw=1.6))
    apex = L - rho
    for s in (1, -1):
        ax.plot([apex, 0], [0, s * w / 2], color=MUTE, lw=0.9, ls=":")
    ax.plot([apex, L], [0, 0], color=MUTE, lw=0.8, ls="-.")
    ax.plot(apex, 0, "o", color=MUTE, ms=4)
    ax.annotate("apex", (apex, 0), (apex - 1, -3.2), color=MUTE, fontsize=10, ha="center")
    # dimension lines
    def dim(p0, p1, text, off, rot=0, col=INK):
        ax.annotate("", p0, p1, arrowprops=dict(arrowstyle="<->", color=col, lw=1.1))
        mx, my = (p0[0] + p1[0]) / 2 + off[0], (p0[1] + p1[1]) / 2 + off[1]
        ax.text(mx, my, text, color=col, fontsize=11.5, ha="center", va="center",
                rotation=rot, fontweight="bold",
                bbox=dict(fc="white", ec="none", pad=1.2))
    dim((L + 3, -W / 2), (L + 3, W / 2), lab_W, (3.6, 0), 90, ACC)
    xw = -WG - 2.5
    if w < 3:
        for s_ in (1, -1):
            ax.annotate("", (xw, s_ * w / 2), (xw, s_ * (w / 2 + 3)),
                        arrowprops=dict(arrowstyle="->", color=ACC, lw=1.1))
        ax.plot([xw, xw], [-w / 2, w / 2], color=ACC, lw=1.1)
        ax.text(xw - 3.2, 0, lab_w, color=ACC, fontsize=11.5, ha="center", va="center",
                rotation=90, fontweight="bold")
    else:
        dim((xw, -w / 2), (xw, w / 2), lab_w, (-3.2, 0), 90, ACC)
    dim((0, -W / 2 - 3.5), (L, -W / 2 - 3.5), f"L = {L:g}", (0, -1.8))
    dim((apex, W / 2 + 3.5), (L, W / 2 + 3.5), f"{dims} = {rho:.1f}", (0, 1.8), 0, MUTE)
    ax.set_title(plane, fontsize=13, color=INK, loc="left")
    ax.set_aspect("equal")
    ax.axis("off")


fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw=dict(width_ratios=[62, 62, 34]))
A, B, a, b, L = NEW
fc = facts["new"]["closed_form"]
horn_cut(axs[0], A, a, L, fc["rho_h"], 10, f"A = {A:g}", f"a = {a:.2f}",
         "H-plane cut (x′–z′):  the TE10 cosine lives here", r"$R_{H0}$")
horn_cut(axs[1], B, b, L, fc["rho_e"], 10, f"B = {B:g}", f"b = {b:.2f}",
         "E-plane cut (y′–z′):  E is along y′", r"$R_{E0}$")
for ax in axs[:2]:
    ax.set_xlim(-22, 40)
    ax.set_ylim(-17, 20)
ax = axs[2]
ax.add_patch(Rectangle((-A / 2, -B / 2), A, B, fc="#EAF1FD", ec=INK, lw=1.6))
ax.add_patch(Rectangle((-a / 2, -b / 2), a, b, fc="white", ec=INK, lw=1.2))
xx = np.linspace(-A / 2, A / 2, 200)
ax.plot(xx, -B / 2 - 3 + 2.6 * np.cos(np.pi * xx / A), color=ACC, lw=1.6)
ax.text(0, -B / 2 - 4.3, r"$|E_y| \propto \cos(\pi x'/A)$", color=ACC, ha="center", fontsize=12)
for yy in (-4, 0, 4):
    ax.add_patch(FancyArrowPatch((A / 2 + 2.5, yy - 1.6), (A / 2 + 2.5, yy + 1.6),
                                 arrowstyle="-|>", mutation_scale=12, color=BAD, lw=1.4))
ax.text(A / 2 + 4, 6.5, r"$E_y$", color=BAD, fontsize=13)
ax.text(0, B / 2 + 1.6, f"aperture {A:g} × {B:g} (inner)", ha="center", fontsize=11, color=INK)
ax.annotate("WR-15 feed 3.76 × 1.88", (a / 2, -b / 2), (3.0, -5.2), color=MUTE, fontsize=9.5,
            ha="center", arrowprops=dict(arrowstyle="-", color=MUTE, lw=0.8))
ax.set_xlim(-16, 18)
ax.set_ylim(-17, 20)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("front view (looking into the mouth)", fontsize=13, color=INK, loc="left")
fig.text(0.01, 0.02, "all dimensions in mm  ·  inner walls  ·  L = axial flare length, "
         "R_H0 / R_E0 = apex-to-aperture distances (Nikolova L18 eq. 18.43–18.44)",
         color=MUTE, fontsize=10.5)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "horn_drawing.png", facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------- 2. aperture + patterns
h = HORNS["new"]
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
ext = [h["x"][0], h["x"][-1], h["y"][0], h["y"][-1]]
im = axs[0].imshow(np.abs(h["E"]).T, origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=1)
fig.colorbar(im, ax=axs[0], fraction=0.035, pad=0.03)
axs[0].set(title="|E| across the aperture (eq. 18.38)", xlabel="x′ (mm)", ylabel="y′ (mm)")
cfn = facts["new"]["closed_form"]
Xa, Ya = np.meshgrid(h["x"], h["y"], indexing="ij")
ph = np.degrees(np.pi / WVL * (Xa ** 2 / cfn["rho_h"] + Ya ** 2 / cfn["rho_e"]))
facts["new"]["flare_phase_deg"] = {"H_edge": float(np.degrees(2 * np.pi * cfn["t"])),
                                   "E_edge": float(np.degrees(np.pi / WVL * (NEW[1] / 2) ** 2 / cfn["rho_e"])),
                                   "corner": float(ph.max())}
(OUT / "horn_facts.json").write_text(json.dumps(facts, indent=2, default=float), encoding="utf-8")
im = axs[1].imshow(ph.T, origin="lower", extent=ext, cmap="magma", vmin=0, vmax=ph.max())
cs = axs[1].contour(h["x"], h["y"], ph.T, levels=[45, 90, 135, 180], colors="white", linewidths=0.8)
axs[1].clabel(cs, fmt="%d°", fontsize=9)
cb = fig.colorbar(im, ax=axs[1], fraction=0.035, pad=0.03)
cb.set_label("phase lead (deg)")
axs[1].set(title="phase: the flare's path-length error", xlabel="x′ (mm)", ylabel="y′ (mm)")
for ax in axs[:2]:
    ax.set_aspect("equal")
ax = axs[2]
for key in ("old", "new"):
    hh = HORNS[key]
    ff = hh["ff"]
    for pl, col in (("E_plane", BAD), ("H_plane", ACC)):
        ax.plot(ff["theta_deg"], 20 * np.log10(ff[pl] + 1e-12), color=col if key == "new" else {BAD: "#E8A09A", ACC: "#9DB9F2"}[col],
                ls=hh["ls"], lw=1.8 if key == "new" else 1.3,
                label=f"{pl[0]}-plane, {'H-A75-W20' if key == 'new' else 'previous'}")
ax.axhline(-3, color=MUTE, lw=0.8, ls=":")
f = facts["new"]
ax.text(31, -5.5, f"HPBW  E {f['hpbw_E_deg']:.1f}°  /  H {f['hpbw_H_deg']:.1f}°\n"
        f"D = {f['D_numeric_dBi']:.2f} dBi (aperture integral)", fontsize=11.5, color=INK)
ax.set(xlim=(0, 60), ylim=(-35, 1), xlabel="angle from boresight (deg)", ylabel="dB",
       title="far-field principal planes")
ax.grid(alpha=0.3)
ax.legend(fontsize=9.5, loc="lower left", frameon=False)
fig.tight_layout()
fig.savefig(OUT / "horn_aperture.png", facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------- 3. rail footprint
fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw=dict(width_ratios=[1, 1, 1.3]))
for ax, key in zip(axs[:2], ("old", "new")):
    hh = HORNS[key]
    amp = np.abs(hh["foot"]) / np.abs(hh["foot"]).max()
    ax.imshow(amp.T, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]], cmap="magma",
              vmin=0, vmax=1)
    ax.add_patch(Rectangle((-37, -60), 74, 120, fill=False, ec="white", lw=1.4, ls="--"))
    ax.contour(xs, ys, amp.T, levels=[np.sqrt(0.5)], colors="#7FD1FF", linewidths=1.2)
    ax.set(title=("previous horn" if key == "old" else "H-A75-W20") + "  |E| at crown height",
           xlabel="x (mm, across head)", ylabel="y (mm, along rail)")
    ax.set_aspect("equal")
axs[0].text(-35, 63, "rail head, 120 mm segment", color="white", fontsize=9.5)
ax = axs[2]
for key in ("old", "new"):
    hh = HORNS[key]
    amp = np.abs(hh["foot"]) / np.abs(hh["foot"]).max()
    ax.plot(ys, amp[np.argmin(abs(xs))], color=hh["c"], ls=hh["ls"], lw=2,
            label=("previous" if key == "old" else "H-A75-W20") +
            f":  −3 dB span {facts[key]['foot_y_minus3dB_span_mm']:.0f} mm, "
            f"ends {facts[key]['foot_amp_at_rail_ends']:.2f}")
for s in (-60, 60):
    ax.axvline(s, color=MUTE, lw=1, ls=":")
ax.text(60, 1.02, "rail end", color=MUTE, ha="center", fontsize=10)
ax.text(-60, 1.02, "rail end", color=MUTE, ha="center", fontsize=10)
ax.axhline(np.sqrt(0.5), color=MUTE, lw=0.8, ls=":")
ax.set(xlim=(-90, 90), ylim=(0, 1.08), xlabel="y (mm, along rail) at x = 0",
       ylabel="|E| / peak", title="cut along the rail")
ax.grid(alpha=0.3)
ax.legend(fontsize=10, loc="lower center", frameon=False)
fig.tight_layout()
fig.savefig(OUT / "horn_rail.png", facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------- 4. equations
EQ = [
    ("18.38", r"$E_{ay}(x',y') \;\approx\; E_0\,\cos\!\left(\frac{\pi x'}{A}\right)\,"
              r"\exp\!\left[-j\,\frac{k}{2}\left(\frac{x'^2}{R_{H0}}+\frac{y'^2}{R_{E0}}\right)\right]$",
     "TE$_{10}$ cosine × quadratic phase error. Written for $e^{+j\\omega t}$; rail3D and Lumerical "
     "use $e^{-i\\omega t}$, so the code carries $+j$"),
    ("18.43–44", r"$R_{H0}=\frac{A\,R_H}{A-a}, \qquad R_{E0}=\frac{B\,R_E}{B-b}, \qquad "
                 r"R_H=R_E=L\;\;(18.42)$",
     "apex distances by similar triangles; one flare length L for a realizable horn"),
    ("18.4", r"$\beta_g = k\sqrt{1-\left(\lambda/2a\right)^2}\;=\;0.747\,k$" +
             r"$\quad$ (WR-15, 60 GHz)",
     "guided TE$_{10}$ phase constant in the feed; the textbook aperture phase uses $k$, not $\\beta_g$"),
    ("18.39", r"$D=\frac{4\pi}{\lambda^2}\,A\,B\;\epsilon_t\,\epsilon_{ph}^{E}\,\epsilon_{ph}^{H},"
              r"\qquad \epsilon_t=\frac{8}{\pi^2}$",
     r"$\epsilon_{ph}^{H}=\frac{\pi^2}{64t}\{[C(p_1)-C(p_2)]^2+[S(p_1)-S(p_2)]^2\},\;"
     r"p_{1,2}=2\sqrt{t}\,(\pm 1+\frac{1}{8t}),\;t=\frac{A^2}{8\lambda R_{H0}};\;\;"
     r"\epsilon_{ph}^{E}=\frac{C^2(q)+S^2(q)}{q^2},\;q=\frac{B}{\sqrt{2\lambda R_{E0}}}$"),
    ("18.21", r"$D=\frac{4\pi}{\lambda^2}\,\frac{|\iint_{S_A} E_a\,ds'|^2}{\iint_{S_A}|E_a|^2\,ds'}$",
     "aperture-integral directivity of any sampled field; V9 requires it to equal (18.39)"),
    ("rail3D", r"$\psi(P)=\sum_{n=1}^{N^2} E_a(u_n,v_n)\,\Delta S\;\frac{1}{\lambda}"
               r"\left(\frac{1}{kR_n}-j\right)\frac{\cos\theta_n}{R_n}\,e^{\,jkR_n},"
               r"\qquad \Delta S=\frac{AB}{N^2},\;N=20$",
     "each midpoint sample radiates as a Huygens source through the exact RS-I kernel"),
]
ROWS = {"18.38": (0.62, 0.36), "18.43–44": (0.50, 0.36), "18.4": (0.50, 0.36),
        "18.39": (0.52, 0.40), "18.21": (0.66, 0.36), "rail3D": (0.66, 0.36)}
Htot = sum(a + b + 0.16 for a, b in ROWS.values())
fig = plt.figure(figsize=(9.6, Htot))
yin = Htot
for tag, eq, gloss in EQ:
    eh, gh = ROWS[tag]
    fig.text(0.0, (yin - 0.05) / Htot, tag, fontsize=12.5, color=ACC, fontweight="bold", va="top")
    fig.text(0.12, (yin - eh / 2) / Htot, eq, fontsize=17, color=INK, va="center")
    small = tag == "18.39"
    fig.text(0.12, (yin - eh - gh / 2) / Htot, gloss, fontsize=12.2 if small else 11.8,
             color=INK if small else MUTE, va="center")
    yin -= eh + gh + 0.16
fig.savefig(OUT / "horn_equations.png", facecolor="white", bbox_inches="tight", pad_inches=0.08)
plt.close(fig)

# ---------------------------------------------------------------- 5. result crops
src = Path(sys.argv[2])
im = Image.open(src / "analysis_parameters.png")
W_, H_ = im.size
# rows 1-2 (crack, dent) of the 4x4 panel grid, without the suptitle
im.crop((0, int(0.062 * H_), W_, int(0.527 * H_))).save(OUT / "prelim_crack_dent.png")
Image.open(src / "analysis_performance.png").save(OUT / "prelim_performance.png")
print("figures ->", OUT)
