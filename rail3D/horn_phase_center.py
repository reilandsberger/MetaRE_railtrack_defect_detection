"""Where is the horn's phase centre? (CPU, ~1 min.)

The phase centre is the point the radiated wavefronts look spherical around. It
lies on the horn axis, between the aperture face and the flare apex. This tool
finds it for rail3D's horn model, the same aperture field (Nikolova L18 eq.
18.38, field3d.aperture_distribution) that horn_scan_plan.py uses:

  far field   the far-field phase in each principal plane is fitted with
              const + k d cos(theta), weighted by power over the -10 dB beam;
              d = how far behind the aperture face the centre sits.
  at range z  the phase on a plane at distance z (exact Rayleigh-Sommerfeld
              sum) is fitted with a sphere centred d behind the aperture: the
              centre a scan-plane measurement at that z will see.

    python horn_phase_center.py                          # Eravant SAR-1725-28-KF-R2 at 37.5 GHz
    python horn_phase_center.py --horn eravant --ghz 33.25
    python horn_phase_center.py --horn current           # RFspin H-A75-W20 at 60 GHz

Writes horn_phase_center_<horn>_<GHz>.png and .json to data/generated/<day>/horn_phase_center/.
Scalar aperture model: no rim diffraction, no cross-polarisation, and the flare length is
a fit (the figure shows how far the answer moves if it is off by -20%/+25%).
"""
from __future__ import annotations

import argparse
import json
import math

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection       # noqa: E402

import horn_scan_plan as hs                                    # noqa: E402
from rail3d import config, field3d                             # noqa: E402

C0 = 299.792458                       # mm GHz
HORNS = {
    "eravant": dict(name="Eravant SAR-1725-28-KF-R2", size=hs.ERAVANT, band=(26.5, 40.0), ghz=C0 / 8.0,
                    body=63.5, marks=(33.25,)),
    "current": dict(name=config.HORN_SPEC["part"], size=tuple(config.SIZE_ANT), band=config.HORN_SPEC["band_GHz"],
                    ghz=C0 / config.WVL, body=None, marks=()),
}
RED, BLUE, GREY = "#d62728", "#1f77b4", "0.35"


def rho(size):
    A, B, a, b, L = size
    return A * L / (A - a), B * L / (B - b)


def far_cut(size, wvl, th_deg):
    """Complex far field (no obliquity factor: it is real and does not move the phase) in H and E."""
    U, V, psi, _ = hs.aperture(size, wvl, 80)
    k, t = 2 * math.pi / wvl, torch.tensor(np.radians(th_deg))
    return [(psi[None] * torch.exp(-1j * k * torch.sin(t)[:, None, None] * C[None])).sum((1, 2)).numpy()
            for C in (U, V)]


def far_centre(size, wvl, lvl=-10.0):
    """Per plane: d (mm behind the aperture), rms/max residual (deg), the fit range, the cut."""
    th = np.arange(0.0, 70.01, 0.1)
    k = 2 * math.pi / wvl
    out = []
    for F in far_cut(size, wvl, th):
        p = 20 * np.log10(np.abs(F) / abs(F[0]))
        edge = hs.first_cross(th, p, lvl)
        m = th <= edge
        ph = np.unwrap(np.angle(F / F[0]))
        c, w = np.cos(np.radians(th[m])), np.abs(F[m]) ** 2
        X = np.stack([np.ones_like(c), k * c], 1)
        sol = np.linalg.lstsq(X * np.sqrt(w)[:, None], ph[m] * np.sqrt(w), rcond=None)[0]
        r = ph[m] - X @ sol
        out.append(dict(d=float(sol[1]), rms=float(np.degrees(np.sqrt((w * r ** 2).sum() / w.sum()))),
                        max=float(np.degrees(np.abs(r).max())), edge=float(edge), th=th, ph=ph, p=p))
    return out


def near_centre(size, wvl, z, lvl=-10.0):
    """Apparent centre on a plane at distance z: fit the phase along each principal cut to a sphere."""
    from scipy.optimize import minimize_scalar
    k = 2 * math.pi / wvl
    x = np.linspace(0, z * math.tan(math.radians(35)), 201)
    out = []
    for ax in "HE":
        F = (hs.field(size, wvl, z, x, np.array([0.0]), n=36)[:, 0] if ax == "H"
             else hs.field(size, wvl, z, np.array([0.0]), x, n=36)[0])
        p = 20 * np.log10(np.abs(F) / abs(F[0]))
        m = x <= hs.first_cross(x, p, lvl)
        ph, w = np.unwrap(np.angle(F[m] / F[0])), np.abs(F[m]) ** 2

        def cost(d):
            r = ph - k * (np.sqrt((z + d) ** 2 + x[m] ** 2) - (z + d))
            r = r - (w * r).sum() / w.sum()
            return (w * r ** 2).sum() / w.sum()
        o = minimize_scalar(cost, bounds=(-30.0, 80.0), method="bounded")
        out.append(float(o.x))
    return out


def plane_map(size, wvl, plane, x, z, n=28):
    """psi on the (x, z) grid in the H plane (along A) or the E plane (along B), z > 0 only."""
    U, V, psi, dS = hs.aperture(size, wvl, n)
    k = 2 * math.pi / wvl
    Xg, Zg = np.meshgrid(x, z, indexing="ij")      # rows = x (vertical), cols = z
    X, Z = torch.tensor(Xg), torch.tensor(Zg)
    out = torch.zeros(X.shape, dtype=torch.complex128)
    for uu, vv, p in zip(U.reshape(-1), V.reshape(-1), psi.reshape(-1)):
        du, dv = (X - uu, vv) if plane == "H" else (uu, X - vv)
        R = torch.sqrt(du ** 2 + dv ** 2 + Z ** 2)
        out += p * dS * field3d.rs_kernel(R, Z, wvl, k)
    return out.numpy()


def far_2d(size, wvl, th_max=40.0):
    """Far-field power on (theta, phi), dB re the axis, for colouring the 3D wavefront caps."""
    U, V, psi, _ = hs.aperture(size, wvl, 40)
    k = 2 * math.pi / wvl
    th = np.radians(np.linspace(0, th_max, 41))
    ph = np.radians(np.linspace(0, 360, 73))
    T, P = np.meshgrid(th, ph, indexing="ij")
    kx = torch.tensor((np.sin(T) * np.cos(P)).ravel())
    ky = torch.tensor((np.sin(T) * np.sin(P)).ravel())
    F = torch.cat([(psi[None] * torch.exp(-1j * k * (kx[i:i + 800, None, None] * U[None]
                                                    + ky[i:i + 800, None, None] * V[None]))).sum((1, 2))
                   for i in range(0, len(kx), 800)]).abs().numpy().reshape(T.shape)
    return T, P, 20 * np.log10(F / F[0, 0])


# ---------------------------------------------------------------------------- figure
def draw_3d(ax, h, wvl, d_mean, apex):
    A, B, a, b, L = h["size"]
    t = 1.0                                       # wall thickness (drawn)
    back = h["body"] - L if h["body"] else 22.0   # waveguide/adapter section behind the flare (schematic)

    def box(z0, z1, w0, h0, w1, h1):
        c = lambda z, w, hh: [(-w / 2, -hh / 2, z), (w / 2, -hh / 2, z), (w / 2, hh / 2, z), (-w / 2, hh / 2, z)]  # noqa: E731
        p, q = c(z0, w0, h0), c(z1, w1, h1)
        return [[p[i], p[(i + 1) % 4], q[(i + 1) % 4], q[i]] for i in range(4)]
    flare = box(-L, 0, a + 2 * t, b + 2 * t, A + 2 * t, B + 2 * t)
    stub = box(-L - back, -L, a + 6, b + 6, a + 6, b + 6)
    # plot axes: (x, z, y) so the beam runs left -> right, E (y) up
    sw = lambda faces: [[(p[0], p[2], p[1]) for p in f] for f in faces]  # noqa: E731
    ax.add_collection3d(Poly3DCollection(sw(stub), facecolor="0.55", edgecolor="0.25", lw=0.6, alpha=0.9))
    ax.add_collection3d(Poly3DCollection(sw(flare), facecolor="#c9a227", edgecolor="0.3", lw=0.6, alpha=0.55))
    rim = [(-A / 2, -B / 2), (A / 2, -B / 2), (A / 2, B / 2), (-A / 2, B / 2), (-A / 2, -B / 2)]
    ax.plot([p[0] for p in rim], [0] * 5, [p[1] for p in rim], color="k", lw=1.2)
    # wavefront caps, coloured by the far-field level
    T, P, dB = far_2d(h["size"], wvl, 28.0)
    for r in (25.0, 55.0, 85.0):
        R = r + d_mean
        X = R * np.sin(T) * np.cos(P)
        Y = R * np.sin(T) * np.sin(P)
        Z = R * np.cos(T) - d_mean
        col = plt.cm.inferno(np.clip((dB + 25) / 25, 0, 1))
        col[..., 3] = 0.55
        ax.plot_surface(X, Z, Y, facecolors=col, rstride=2, cstride=3, linewidth=0, shade=False)
    # axis, centre, apex
    ax.plot([0, 0], [-L - back, 140], [0, 0], color="k", lw=0.8, ls=":")
    ax.scatter([0], [-d_mean], [0], color=RED, s=60, depthshade=False, zorder=10)
    ax.scatter([0], [-apex], [0], color="k", marker="x", s=40, depthshade=False)
    ax.text(0, -d_mean - 6, 26, "phase centre", color=RED, fontsize=9, ha="center", fontweight="bold")
    ax.text(0, -apex - 4, -22, "flare apex", color="k", fontsize=8, ha="center")
    ax.set_xlim(-60, 60), ax.set_ylim(-L - back, 120), ax.set_zlim(-60, 60)
    ax.set_box_aspect((120, L + back + 120, 120))
    ax.set_xlabel("x, H plane (mm)", fontsize=8), ax.set_ylabel("z, along the axis (mm)", fontsize=8)
    ax.set_zlabel("y, E plane (mm)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.view_init(elev=22, azim=-58)


def draw_cut(ax, h, wvl, plane, fc, apex):
    A, B, a, b, L = h["size"]
    W, w = (A, a) if plane == "H" else (B, b)
    d = fc["d"]
    k = 2 * math.pi / wvl
    x = np.linspace(-75, 75, 301)
    z = np.linspace(1.0, 120.0, 239)
    F = plane_map(h["size"], wvl, plane, x, z)
    lvl = 20 * np.log10(np.abs(F) / np.abs(F).max())
    img = np.cos(np.angle(F))
    alpha = np.clip((lvl + 30) / 20, 0, 1)
    rgba = plt.cm.RdBu_r((img + 1) / 2)
    rgba[..., 3] = alpha
    ax.imshow(rgba, extent=[z[0], z[-1], x[0], x[-1]], origin="lower", aspect="auto",
              interpolation="bilinear", zorder=1)
    # circles about the fitted centre through the on-axis crests
    ax0 = plane_map(h["size"], wvl, plane, np.array([0.0]), z)[0]
    phz = np.unwrap(np.angle(ax0))
    for n in range(int(phz[0] / (2 * math.pi)) + 1, int(phz[-1] / (2 * math.pi)) + 1, 2):
        zc = np.interp(2 * math.pi * n, phz, z)
        R = zc + d
        tt = np.radians(np.linspace(-38, 38, 120))
        ax.plot(R * np.cos(tt) - d, R * np.sin(tt), color="k", lw=0.9, ls="--", zorder=3)
    # horn walls (extended to the apex) and the aperture
    ax.plot([-L, 0], [w / 2, W / 2], color="k", lw=3, solid_capstyle="butt", zorder=4)
    ax.plot([-L, 0], [-w / 2, -W / 2], color="k", lw=3, solid_capstyle="butt", zorder=4)
    ax.plot([-L - 15, -L], [w / 2, w / 2], color="0.4", lw=3, zorder=4)
    ax.plot([-L - 15, -L], [-w / 2, -w / 2], color="0.4", lw=3, zorder=4)
    ax.plot([-apex, -L], [0, w / 2], color="0.5", lw=0.8, ls=":", zorder=4)
    ax.plot([-apex, -L], [0, -w / 2], color="0.5", lw=0.8, ls=":", zorder=4)
    ax.plot([0, 0], [-W / 2, W / 2], color="#c9a227", lw=2, zorder=4)
    ax.axhline(0, color="0.5", lw=0.6, ls=":", zorder=2)
    ax.plot(-d, 0, "*", color=RED, ms=16, mec="k", mew=0.6, zorder=6)
    ax.plot(-apex, 0, "x", color="k", ms=8, mew=1.6, zorder=6)
    ax.annotate(f"phase centre\n{d:.1f} mm behind\nthe aperture", (-d, 0), (-d - 6, -50), fontsize=9,
                color=RED, ha="center", arrowprops=dict(arrowstyle="->", color=RED), zorder=7)
    ax.annotate(f"flare apex\n{apex:.1f} mm behind", (-apex, 0), (-apex + 16, 46), fontsize=8.5, ha="center",
                arrowprops=dict(arrowstyle="->", color="k"), zorder=7)
    ax.text(1.5, W / 2 + 3, "aperture face", fontsize=8, color="#8a6d00", zorder=7, bbox=dict(fc="w", ec="none", pad=1))
    ax.set_xlim(-L - 15, 120), ax.set_ylim(-75, 75)
    ax.set_aspect("equal")
    ax.set_xlabel("z, along the horn axis (mm; 0 = aperture face)")
    ax.set_ylabel(("x, across the wide side A" if plane == "H" else "y, across the narrow side B") + " (mm)")
    ttl = ("B. Top view: H plane (wide side, A = %g mm)" if plane == "H"
           else "C. Side view: E plane (narrow side, B = %g mm)") % W
    ax.set_title(ttl + "\ncolour: the model's wavefronts; dashed: circles about the ★", loc="left",
                 fontsize=10)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--horn", choices=sorted(HORNS), default="eravant")
    ap.add_argument("--ghz", type=float, default=None, help="frequency of the views (default: 8 mm for the "
                    "Eravant, config.WVL for the current horn)")
    ap.add_argument("--day", default=None)
    args = ap.parse_args()
    h = HORNS[args.horn]
    ghz = args.ghz or h["ghz"]
    wvl = C0 / ghz
    rh, re_ = rho(h["size"])
    fH, fE = far_centre(h["size"], wvl)
    d_mean = 0.5 * (fH["d"] + fE["d"])
    print(f"{h['name']} at {ghz:.2f} GHz (lambda {wvl:.2f} mm)")
    print(f"  flare apex behind the aperture: H {rh:.1f} mm, E {re_:.1f} mm")
    print(f"  far-field phase centre behind the aperture: H {fH['d']:.1f} mm, E {fE['d']:.1f} mm "
          f"(rms residual {fH['rms']:.1f} / {fE['rms']:.1f} deg over the -10 dB beam)")
    freqs = np.linspace(*h["band"], 11)
    fsweep = [[far_centre(h["size"], C0 / f)[i]["d"] for i in (0, 1)] for f in freqs]
    sens = {}
    for s in (0.8, 1.25):
        sz = tuple(h["size"][:4]) + (h["size"][4] * s,)
        sens[s] = [[far_centre(sz, C0 / f)[i]["d"] for i in (0, 1)] for f in freqs]
    zs = np.array([60, 80, 100, 130, 176, 250, 350, 500, 800], float)
    near = np.array([near_centre(h["size"], wvl, z) for z in zs])
    for z, (a_, b_) in zip(zs, near):
        print(f"  seen from a plane at z = {z:4.0f} mm: H {a_:.1f}  E {b_:.1f} mm behind the aperture")

    fig = plt.figure(figsize=(19, 12.5))
    gs = fig.add_gridspec(2, 3, height_ratios=(1.1, 1), hspace=0.32, wspace=0.22,
                          left=0.05, right=0.98, top=0.9, bottom=0.06)
    ax3 = fig.add_subplot(gs[0, 0], projection="3d")
    draw_3d(ax3, h, wvl, d_mean, 0.5 * (rh + re_))
    ax3.set_title("A. 3D view: wavefront shells (coloured by beam level,\n0 to −25 dB) "
                  "centred on the phase centre", loc="left", fontsize=10)
    draw_cut(fig.add_subplot(gs[0, 1]), h, wvl, "H", fH, rh)
    draw_cut(fig.add_subplot(gs[0, 2]), h, wvl, "E", fE, re_)

    # D. proof: residual far-field phase about three candidate centres
    ax = fig.add_subplot(gs[1, 0])
    k = 2 * math.pi / wvl
    for fc, ls, pl, apx in ((fH, "-", "H", rh), (fE, "--", "E", re_)):
        th = fc["th"]
        m = th <= 40
        for dd, col, lab in ((0.0, GREY, "about the aperture centre"), (fc["d"], RED, "about the phase centre ★"),
                             (apx, "k", "about the flare apex")):
            r = np.degrees(fc["ph"] - k * dd * np.cos(np.radians(th)))
            r -= r[0]
            ax.plot(th[m], r[m], ls, color=col, lw=1.8 if col == RED else 1.2,
                    label=f"{lab}, {pl} plane" if True else None)
        ax.axvline(fc["edge"], color="0.6", lw=0.8, ls=ls)
    ax.axhspan(-10, 10, color="0.9", zorder=0)
    ax.set_ylim(-150, 150)
    ax.set_xlim(0, 40)
    ax.set_xlabel("angle off the axis (degrees)")
    ax.set_ylabel("far-field phase left over after\nremoving a sphere about the point (deg)")
    ax.set_title("D. Why this point: only about the ★ is the phase flat (±10° band)\n"
                 f"across the −10 dB beam (grey verticals: H {fH['edge']:.0f}°, E {fE['edge']:.0f}°)",
                 loc="left", fontsize=10)
    ax.legend(fontsize=7.5, ncol=2, loc="lower left")
    ax.grid(alpha=0.3)

    # E. phase centre across the band
    ax = fig.add_subplot(gs[1, 1])
    fs = np.array(fsweep)
    for i, (pl, col) in enumerate((("H", BLUE), ("E", RED))):
        ax.plot(freqs, fs[:, i], "o-", color=col, label=f"{pl} plane")
        lo = np.minimum(np.array(sens[0.8])[:, i], np.array(sens[1.25])[:, i])
        hi = np.maximum(np.array(sens[0.8])[:, i], np.array(sens[1.25])[:, i])
        ax.fill_between(freqs, lo, hi, color=col, alpha=0.12)
    ax.axhline(rh, color=BLUE, ls=":", lw=1)
    ax.axhline(re_, color=RED, ls=":", lw=1)
    ax.text(freqs[0], rh + 0.6, f"H-plane flare apex {rh:.1f} mm", fontsize=8, color=BLUE)
    ax.text(freqs[0], re_ - 2.2, f"E-plane flare apex {re_:.1f} mm", fontsize=8, color=RED)
    ax.axvline(ghz, color="k", lw=0.8)
    ax.text(ghz, 1, f" views: {ghz:.2f} GHz", fontsize=8)
    for mk in h["marks"]:
        ax.axvline(mk, color="0.5", lw=0.8, ls="--")
        ax.text(mk, 4, f" datasheet\n pattern\n {mk:g} GHz", fontsize=7.5, color="0.4")
    ax.set_ylim(0, max(rh, re_) + 4)
    ax.set_xlabel("frequency (GHz)")
    ax.set_ylabel("distance behind the aperture face (mm)")
    ax.set_title("E. It moves back toward the apex as frequency rises\n"
                 "(band = flare length 20% shorter to 25% longer than the fitted "
                 f"{h['size'][4]:g} mm)", loc="left", fontsize=10)
    ax.legend(fontsize=8.5, loc="center right")
    ax.grid(alpha=0.3)

    # F. what a scan plane at distance z sees
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(zs, near[:, 0], "o-", color=BLUE, label="H plane, fitted on the scan plane")
    ax.plot(zs, near[:, 1], "o-", color=RED, label="E plane, fitted on the scan plane")
    ax.axhline(fH["d"], color=BLUE, ls=":", lw=1)
    ax.axhline(fE["d"], color=RED, ls=":", lw=1, label="far-field values (dotted)")
    ax.set_xscale("log")
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([60, 100, 176, 300, 500, 800], ["60", "100", "176", "300", "500", "800"])
    ax.set_xlabel("distance of the scan plane from the aperture, z (mm)")
    ax.set_ylabel("apparent centre, mm behind the aperture face")
    ax.set_title(f"F. The centre a scan plane sees at {ghz:.2f} GHz: farther back up close,\n"
                 "settling to the far-field value", loc="left", fontsize=10)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3, which="both")

    body = (f"; body {h['body']:g} mm long, so ≈ {h['body'] - d_mean:.0f} mm from its back end"
            if h["body"] else "")
    fig.suptitle(f"Phase centre of the {h['name']} at {ghz:.2f} GHz (λ = {wvl:.2f} mm): "
                 f"{fH['d']:.1f} mm (H) / {fE['d']:.1f} mm (E) behind the aperture face{body}",
                 fontsize=13, fontweight="bold", y=0.985)
    fig.text(0.5, 0.945, "rail3D's aperture model (L18 eq. 18.38), A, B, a, b, L = "
             + ", ".join(f"{v:g}" for v in h["size"]) + " mm.  The back section in A is schematic.  "
             "B, C colour = cos(phase) of the field, faded below −30 dB.", ha="center", fontsize=10.5)
    out = config.day_dir(args.day) / "horn_phase_center"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.horn}_{ghz:.2f}GHz".replace(".", "p")
    fig.savefig(out / f"horn_phase_center_{tag}.png", dpi=170)
    plt.close(fig)
    res = dict(horn=h["name"], ghz=ghz, wvl=wvl, size=h["size"], apex_mm=[rh, re_],
               far_field_mm=[fH["d"], fE["d"]], residual_rms_deg=[fH["rms"], fE["rms"]],
               fit_edge_deg=[fH["edge"], fE["edge"]],
               band=dict(ghz=freqs.tolist(), d_mm=fsweep, flare_x0p8=sens[0.8], flare_x1p25=sens[1.25]),
               scan_plane=dict(z=zs.tolist(), d_mm=near.tolist()))
    (out / f"horn_phase_center_{tag}.json").write_text(json.dumps(res, indent=1))
    print(f"wrote {out / f'horn_phase_center_{tag}.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
