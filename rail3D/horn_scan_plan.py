"""Where to put a square scan plane in front of the horn (CPU, ~1 min).

For a horn-validation measurement: a plane of side --side (default 340 mm),
perpendicular to the horn axis, at axial distance z from the aperture. How far
can it sit and still hold the beam? Compares the current horn (config.SIZE_ANT at
config.WVL) with the earlier Eravant SAR-1725-28-KF-R2 (WR-28, used at 8 mm).

    python horn_scan_plan.py                 # -> data/generated/<day>/horn_scan/
    python horn_scan_plan.py --side 300 --z 100 700 25

The field is rail3D's own horn model: field3d.aperture_distribution carried to
the plane by the exact Rayleigh-Sommerfeld kernel (field3d.rs_kernel), so the
maps are what the pipeline assumes the horn radiates -- the thing the
measurement checks. Scalar: it has no cross-polarisation and no rim diffraction.

Distances reported per horn (z from the aperture, on the horn axis):
    z20   the -20 dB edge of the main lobe (wider principal plane) reaches the
          plane edge: the whole main lobe is on the plane. RECOMMENDED.
    z10   the -10 dB edge reaches the plane edge: farther than this the plane
          holds only the beam's core.
    zff   2 D^2 / lambda with D the aperture diagonal (far-field distance).

Writes horn_scan_01_side_view.png, _02_patterns.png, _03_maps.png,
_04_distance.png and horn_scan.json.
"""
from __future__ import annotations

import argparse
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import torch                      # noqa: E402

from rail3d import config, field3d  # noqa: E402

# Eravant SAR-1725-28-KF-R2: 26.5-40 GHz, 17 dBi nominal, half-power beamwidth 23 deg (E) /
# 24 deg (H); body 2.50 x 1.08 x 0.86 in, so the OUTER aperture is 27.4 x 21.9 mm (the
# numbers Face3D used as the aperture). Inner aperture = outer less 1 mm walls, WR-28 feed
# (7.112 x 3.556 mm), flare 27 mm: 17.1 dBi and 25.3 / 23.6 deg at 33 GHz (mid-band).
# Flare 22-45 mm or the outer dimensions move the 37.5 GHz beamwidth by < 2 deg.
ERAVANT = (25.4, 19.9, 7.112, 3.556, 27.0)
ERAVANT_WVL = 8.0
LEVELS = (-3.0, -10.0, -20.0)
COLS = {-3.0: "#1b9e77", -10.0: "#d95f02", -20.0: "#7570b3"}


def horns() -> dict:
    f = lambda w: 299.792458 / w                                  # noqa: E731
    return {"new": {"name": f"{config.HORN_SPEC['part']}, λ = {config.WVL:g} mm ({f(config.WVL):.1f} GHz)",
                    "short": f"current horn, λ {config.WVL:g} mm", "size": tuple(config.SIZE_ANT),
                    "ascii": f"{config.HORN_SPEC['part']} at {f(config.WVL):.1f} GHz ({config.WVL:g} mm)",
                    "wvl": float(config.WVL), "color": "#1f77b4"},
            "old": {"name": f"Eravant SAR-1725-28-KF-R2, λ = {ERAVANT_WVL:g} mm ({f(ERAVANT_WVL):.1f} GHz)",
                    "short": f"Eravant horn, λ {ERAVANT_WVL:g} mm", "size": ERAVANT,
                    "ascii": f"Eravant SAR-1725-28-KF-R2 at {f(ERAVANT_WVL):.1f} GHz ({ERAVANT_WVL:g} mm)",
                    "wvl": ERAVANT_WVL, "color": "#d62728"}}


def aperture(size, wvl, n: int):
    A, B = size[:2]
    u = field3d.aperture_axis(A, n, "midpoint").double()
    v = field3d.aperture_axis(B, n, "midpoint").double()
    U, V = torch.meshgrid(u, v, indexing="ij")
    return U, V, field3d.aperture_distribution(U, V, size, wvl, "k0"), A * B / n ** 2


def directivity(size, wvl) -> float:
    _, _, psi, dS = aperture(size, wvl, 80)
    return float(10 * torch.log10(4 * math.pi / wvl ** 2 * (psi.sum() * dS).abs() ** 2
                                  / ((psi.abs() ** 2).sum() * dS)))


def far_cut(size, wvl, th_deg: np.ndarray):
    """Far-field power pattern (dB re the axis) in the H plane (along A) and the E plane."""
    U, V, psi, _ = aperture(size, wvl, 80)
    k, th = 2 * math.pi / wvl, torch.tensor(np.radians(th_deg))
    out = []
    for C in (U, V):
        f = (psi[None] * torch.exp(-1j * k * torch.sin(th)[:, None, None] * C[None])).sum((1, 2)).abs()
        f = f * (1 + torch.cos(th)) / 2
        out.append((20 * torch.log10(f / f[0])).numpy())
    return out


def field(size, wvl, z: float, x: np.ndarray, y: np.ndarray, n: int = 24) -> np.ndarray:
    """psi on the plane at axial distance z; x along A (H plane), y along B (E plane)."""
    U, V, psi, dS = aperture(size, wvl, n)
    k = 2 * math.pi / wvl
    X, Y = torch.meshgrid(torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64),
                          indexing="ij")
    out = torch.zeros(X.shape, dtype=torch.complex128)
    zt = torch.tensor(float(z), dtype=torch.float64)
    for uu, vv, p in zip(U.reshape(-1), V.reshape(-1), psi.reshape(-1)):
        R = torch.sqrt((X - uu) ** 2 + (Y - vv) ** 2 + z ** 2)
        out += p * dS * field3d.rs_kernel(R, zt, wvl, k)
    return out.numpy()


def first_cross(x, d, lvl) -> float:
    """Where the cut first falls to lvl dB (the main lobe's edge at that level)."""
    i = int(np.argmax(d <= lvl))
    if d[i] > lvl:
        return float("nan")
    return float(np.interp(lvl, [d[i], d[i - 1]], [x[i], x[i - 1]]))


def scan(size, wvl, zs, half: float) -> dict:
    x = np.arange(0.0, 900.0 + 1e-9, 1.0)
    rows = []
    for z in zs:
        h = np.abs(field(size, wvl, z, x, np.array([0.0]))[:, 0]) ** 2
        e = np.abs(field(size, wvl, z, np.array([0.0]), x)[0, :]) ** 2
        h, e = 10 * np.log10(h / h[0]), 10 * np.log10(e / e[0])
        big = max(700.0, 4.0 * z)                                  # quadrant, out to >= 76 deg off axis
        q = np.arange(2.5, big, 5.0)
        I = np.abs(field(size, wvl, z, q, q, 12)) ** 2
        m = q < half
        rows.append({"z": float(z),
                     "H": {str(l): first_cross(x, h, l) for l in LEVELS},
                     "E": {str(l): first_cross(x, e, l) for l in LEVELS},
                     "edge_H": float(np.interp(half, x, h)), "edge_E": float(np.interp(half, x, e)),
                     "inside": float(I[np.ix_(m, m)].sum() / I.sum())})
    return {"rows": rows}


def z_at(rows, lvl: float, half: float) -> float:
    """Largest z at which the main lobe's lvl-dB half-width (wider plane) still fits `half`."""
    z = np.array([r["z"] for r in rows])
    w = np.array([np.nanmax([r["H"][str(lvl)], r["E"][str(lvl)]]) for r in rows])
    ok = np.where(w <= half)[0]
    if len(ok) == 0 or ok[-1] == len(z) - 1:
        return float("nan")
    i = ok[-1]
    return float(np.interp(half, [w[i], w[i + 1]], [z[i], z[i + 1]]))


def at(rows, z: float, key: str) -> float:
    return float(np.interp(z, [r["z"] for r in rows], [r[key] for r in rows]))


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_side(H, half, out):
    """To-scale side view: the beam's -3/-10/-20 dB envelope against the plane."""
    fig, axs = plt.subplots(1, 2, figsize=(16, 7.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.87, bottom=0.17, wspace=0.05)
    zmax = max(r["z"] for r in H["new"]["rows"])
    for ax, h in zip(axs, H.values()):
        z = np.array([r["z"] for r in h["rows"]])
        for lvl in LEVELS:
            for pl, ls in (("H", "-"), ("E", "--")):
                w = np.array([r[pl][str(lvl)] for r in h["rows"]])
                for s in (1, -1):
                    ax.plot(z, s * w, ls, color=COLS[lvl], lw=1.8,
                            label=f"{lvl:g} dB edge, {pl} plane" if s == 1 else None)
        ax.fill_between([0, zmax], half, 320, color="0.9", zorder=0)
        ax.fill_between([0, zmax], -320, -half, color="0.9", zorder=0)
        for zz, lab, c in ((h["z20"], f"RECOMMENDED\nz = {h['z20']:.0f} mm\n(−20 dB edge at the plane edge)", "k"),
                           (h["z10"], f"limit z = {h['z10']:.0f} mm\n(−10 dB edge at\nthe plane edge)", "0.45")):
            ax.plot([zz, zz], [-half, half], color=c, lw=4 if c == "k" else 2.5, solid_capstyle="butt")
            ax.text(zz + 6, -half - 8, lab, va="top", ha="left", fontsize=9, color=c)
        ax.axvline(h["zff"], color="#2ca02c", ls=":", lw=1.5)
        ax.text(h["zff"] + 5, 250, f"far field\n2D²/λ = {h['zff']:.0f} mm", ha="left", va="center",
                fontsize=9, color="#2ca02c")
        A = h["size"][0]
        ax.fill([-h["size"][4], 0, 0, -h["size"][4]], [-h["size"][2] / 2, -A / 2, A / 2, h["size"][2] / 2],
                color="0.3")
        ax.axhline(half, color="0.5", lw=0.8)
        ax.axhline(-half, color="0.5", lw=0.8)
        ax.text(zmax - 5, half + 5, f"plane edge ±{half:g} mm", ha="right", fontsize=8.5, color="0.35")
        ax.set_title(f"{h['name']}  ·  {h['D']:.1f} dBi", loc="left")
        ax.set_xlabel("distance from the horn aperture along the axis, z (mm)")
        ax.set_xlim(-40, zmax)
        ax.set_ylim(-320, 320)
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
    axs[0].set_ylabel("across the beam (mm)")
    hd, lb = axs[0].get_legend_handles_labels()
    fig.legend(hd, lb, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=6, fontsize=9.5, frameon=False)
    fig.suptitle(f"Side view, to scale: how the beam widens against a {2 * half:g} mm plane "
                 "(thick bar = the plane at the recommended distance; grey = outside the plane)",
                 fontweight="bold")
    fig.savefig(out / "horn_scan_01_side_view.png", dpi=200)
    plt.close(fig)


def fig_patterns(H, half, out):
    """Far-field cuts; the top axis turns an angle into the distance where it meets the plane edge."""
    th = np.arange(0.0, 60.01, 0.1)
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.6), sharey=True, constrained_layout=True)
    for ax, (pl, k) in zip(axs, (("H plane (along the wide side A; no sidelobes)", 0),
                                 ("E plane (along the narrow side B; sidelobes)", 1))):
        for h in H.values():
            d = far_cut(h["size"], h["wvl"], th)[k]
            ax.plot(th, d, color=h["color"], lw=2, label=h["short"])
            for lvl in LEVELS:
                t = first_cross(th, d, lvl)
                ax.plot(t, lvl, "o", color=h["color"], ms=6)
                ax.annotate(f"{t:.1f}°", (t, lvl), textcoords="offset points",
                            xytext=(6, 4 if h["color"] == "#d62728" else -13), fontsize=8.5, color=h["color"])
        for lvl in LEVELS:
            ax.axhline(lvl, color=COLS[lvl], lw=0.8, ls="--")
        ax.set_xlim(0, 60)
        ax.set_ylim(-40, 1)
        ax.set_xlabel("angle off the horn axis θ (degrees)")
        ax.set_title(pl, loc="left")
        ax.grid(alpha=0.25)
        top = ax.secondary_xaxis("top", functions=(lambda t: t, lambda t: t))
        tk = np.array([10, 15, 20, 25, 30, 40, 50])
        top.set_xticks(tk, [f"{half / math.tan(math.radians(t)):.0f}" for t in tk])
        top.set_xlabel(f"distance at which this angle reaches the plane edge:  z = {half:g} / tan θ  (mm)")
    axs[0].set_ylabel("power relative to the beam centre (dB)")
    axs[0].legend(fontsize=10, loc="lower left")
    fig.suptitle("Far-field beam shape of each horn: the 8 mm horn's beam is about a third wider in angle, "
                 "so the same plane must sit closer", fontweight="bold")
    fig.savefig(out / "horn_scan_02_patterns.png", dpi=200)
    plt.close(fig)


def fig_maps(H, half, out):
    x = np.arange(-half, half + 1e-9, 2.0)
    fig, axs = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    for row, h in zip(axs, H.values()):
        for ax, (zz, tag) in zip(row, ((0.5 * h["z20"], "too close: plane mostly empty"),
                                       (h["z20"], "RECOMMENDED: main lobe fills the plane"),
                                       (2.0 * h["z20"], "too far: the beam is cut off"))):
            I = np.abs(field(h["size"], h["wvl"], zz, x, x, 16)) ** 2
            d = 10 * np.log10(I / I.max()).T
            im = ax.imshow(d, extent=[-half, half, -half, half], origin="lower", vmin=-40, vmax=0, cmap="magma")
            cs = ax.contour(x, x, d, levels=[-20, -10, -3], colors=["w", "w", "w"],
                            linewidths=[1.0, 1.4, 1.8], linestyles=["dotted", "dashed", "solid"])
            ax.clabel(cs, fmt="%g dB", fontsize=8)
            ax.set_title(f"{h['short']}, z = {zz:.0f} mm\n{tag}\n"
                         f"{100 * at(h['rows'], zz, 'inside'):.0f}% of the radiated power on the plane",
                         fontsize=10)
            ax.set_xlabel("H-plane direction (mm)")
            ax.set_ylabel("E-plane direction (mm)")
    fig.colorbar(im, ax=axs, fraction=0.02, label="|ψ|² relative to the beam centre (dB)")
    fig.suptitle(f"The beam on the {2 * half:g} × {2 * half:g} mm plane at half, one and two times the "
                 "recommended distance", fontweight="bold")
    fig.savefig(out / "horn_scan_03_maps.png", dpi=200)
    plt.close(fig)


def fig_distance(H, half, out):
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
    for h in H.values():
        z = np.array([r["z"] for r in h["rows"]])
        for lvl, ls in ((-20.0, "-"), (-10.0, "--")):
            w = np.array([np.nanmax([r["H"][str(lvl)], r["E"][str(lvl)]]) for r in h["rows"]])
            axs[0].plot(z, w, ls, color=h["color"], lw=2, label=f"{h['short']}: {lvl:g} dB edge")
        axs[0].plot([h["z20"], h["z10"]], [half, half], "o", color=h["color"], ms=7)
        axs[1].plot(z, [100 * r["inside"] for r in h["rows"]], color=h["color"], lw=2, label=h["short"])
        axs[1].plot(h["z20"], 100 * at(h["rows"], h["z20"], "inside"), "o", color=h["color"], ms=7)
        axs[2].plot(z, [max(r["edge_H"], r["edge_E"]) for r in h["rows"]], color=h["color"], lw=2,
                    label=h["short"])
        for ax in axs:
            ax.axvline(h["z20"], color=h["color"], lw=1, ls=":")
    axs[0].axhline(half, color="k", lw=1)
    axs[0].text(axs[0].get_xlim()[0] + 8, half + 4, f"plane edge, {half:g} mm", fontsize=9)
    axs[0].set_ylabel("half-width of the main lobe (wider plane), mm")
    axs[0].set_title("A. Beam half-width against the plane edge", loc="left")
    axs[1].set_ylabel("% of the radiated power landing on the plane")
    axs[1].set_title("B. Power captured by the plane", loc="left")
    axs[2].set_ylabel("level at the middle of the plane's edge (dB re centre)")
    axs[2].set_title("C. How bright the beam still is at the plane edge", loc="left")
    for lvl in (-10, -20):
        axs[2].axhline(lvl, color=COLS[float(lvl)], lw=0.8, ls="--")
    for ax in axs:
        ax.set_xlabel("distance from the horn aperture, z (mm)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8.5)
    fig.suptitle("Both horns against distance (dotted verticals = the recommended distance for each)",
                 fontweight="bold")
    fig.savefig(out / "horn_scan_04_distance.png", dpi=200)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", type=float, default=340.0, help="side of the square scan plane (mm)")
    ap.add_argument("--z", type=float, nargs=3, default=[75.0, 800.0, 25.0], metavar=("LO", "HI", "STEP"))
    ap.add_argument("--day", default=None)
    args = ap.parse_args()
    half = args.side / 2
    zs = np.arange(args.z[0], args.z[1] + 1e-9, args.z[2])
    out = config.day_dir(args.day) / "horn_scan"
    out.mkdir(parents=True, exist_ok=True)
    H = horns()
    for h in H.values():
        A, B = h["size"][:2]
        h.update(scan(h["size"], h["wvl"], zs, half))
        h["D"] = directivity(h["size"], h["wvl"])
        h["zff"] = 2 * (A * A + B * B) / h["wvl"]
        h["z20"], h["z10"] = z_at(h["rows"], -20.0, half), z_at(h["rows"], -10.0, half)
        th = np.arange(0.0, 60.01, 0.05)
        fh, fe = far_cut(h["size"], h["wvl"], th)
        h["hpbw"] = [2 * first_cross(th, fh, -3.0), 2 * first_cross(th, fe, -3.0)]
        h["half_angle"] = {str(l): [first_cross(th, fh, l), first_cross(th, fe, l)] for l in LEVELS}
        print(f"\n{h['ascii']}\n  aperture {A:g} x {B:g} mm, directivity {h['D']:.2f} dBi, half-power beamwidth "
              f"H {h['hpbw'][0]:.1f} / E {h['hpbw'][1]:.1f} deg, far field {h['zff']:.0f} mm")
        print(f"  RECOMMENDED z = {h['z20']:.0f} mm (-20 dB edge at the plane edge; "
              f"{100 * at(h['rows'], h['z20'], 'inside'):.1f}% of the power on the plane)")
        print(f"  limit       z = {h['z10']:.0f} mm (-10 dB edge at the plane edge; "
              f"{100 * at(h['rows'], h['z10'], 'inside'):.1f}%)")
        print("      z   half-width H -3/-10/-20   E -3/-10/-20   edge H/E dB   on plane")
        for r in h["rows"]:
            if r["z"] % 50 == 0:
                print(f"   {r['z']:5.0f}   " + " ".join(f"{r['H'][str(l)]:5.0f}" for l in LEVELS) + "        "
                      + " ".join(f"{r['E'][str(l)]:5.0f}" for l in LEVELS)
                      + f"    {r['edge_H']:6.1f}/{r['edge_E']:6.1f}   {100 * r['inside']:5.1f}%")
    fig_side(H, half, out)
    fig_patterns(H, half, out)
    fig_maps(H, half, out)
    fig_distance(H, half, out)
    (out / "horn_scan.json").write_text(json.dumps({"side": args.side, "horns": H}, indent=1, default=float))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
