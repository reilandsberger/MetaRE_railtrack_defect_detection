"""rail_targets_figs.py -- the figures behind RAIL_TARGETS.md (rail-reflection rungs 1-3).

    python rail_targets_figs.py            # all of them, from the stored JSONs

Inputs (each written by its own script, so these figures are cheap to redraw):
  data/generated/po_canonical_2d.json            po_canonical_2d.py (2D exact vs rail3D's PO)
  data/generated/rail_targets/rung3_*.json       fdtd_agreement.py --diff-target all [--predicted]
Figures go to data/figures/rail_targets/:
  00_approval_ladder.png    what each rung compares, its gates, and why those numbers
  01_crack_physics.png      why a crack is not a mirror: the slot as a waveguide, both
                            polarisations, field maps + depth sweep + far-field patterns
  02_crack_maps.png         rail3D's error over the crack range (width x depth)
  03_class_scorecard.png    every defect class against the rung-3 gates (2D, best..worst)
  04_rung3_summary.png      the 3D rung-3 targets vs predictions, one row per class
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rail3d import config         # noqa: E402

FIG = config.FIGURE_DIR / "rail_targets"
R3 = config.GENERATED_DIR / "rail_targets"
LEVEL_GATE, CORR_GATE = 3.0, 0.90           # = fdtd_agreement.DIFF_CRITERIA
C_EX, C_PO = "#1b1b1b", "#d95f02"
C_SUB = {"longitudinal": "#7570b3", "transverse": "#1b9e77", "x-z": "#e7298a", "y-z": "#66a61e"}
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold"})


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    p = FIG / name
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


def fdtd_minus_rail3d(r: dict, w: str = "ms") -> float:
    """Difference-field level as rung 3 reports it: what the full-wave answer shows
    relative to rail3D (dB). + = rail3D UNDER-predicts the defect. (The 2D JSON
    stores the opposite, 10 log10(rail3D / exact).)"""
    return -r[f"level_dB_{w}"]


def _pattern(r, key):
    a = r["_pattern"][key]
    return np.asarray(a[0]) + 1j * np.asarray(a[1])


# ---------------------------------------------------------------------------
def fig_ladder():
    rungs = [
        ("rung −1", "the horn alone",
         "full-wave horn vs the textbook\naperture model; our propagator\nvs Maxwell at z = 150",
         "aperture corr ≥ 0.95, propagation ≥ 0.98,\nD within ±0.5 dB, Import plane ≥ 0.98",
         "fdtd_agreement --horn\nHORN_RUNG-1_TARGETS.md",
         "lab 2026-10-26: propagator 0.9995 PASS;\naperture model FAILS (D 17.9 vs 20.1 dBi)"),
        ("rung 0", "the empty box",
         "what Lumerical injected vs what\nhorn_source.py wrote; leak into\nthe z = 30 monitor",
         "injection corr ≥ 0.98 as-is,\ncentroid ≤ 5 mm, leak < 1% of peak",
         "fdtd_agreement --injection", "the noise floor under every later rung"),
        ("rung 1", "a flat PEC plate",
         "absolute field at the z = 30 monitor\nand the MS plane (both solvers are\nexact on a plate)",
         "corr ≥ 0.98; ≥ 85% of lit pixels within\n5% / 15°; worst pixel ≤ 15% / 30°;\nMS: corr and barcode cos ≥ 0.98",
         "fdtd_agreement\n--sample plate", "a miss here is SETUP (units, source,\nmonitor), not physics"),
        ("rung 2", "the intact rail",
         "same, on the curved head: physical\noptics, shadowing, the segment's\ncut ends",
         "as rung 1", "fdtd_agreement --target\n(and --external)",
         "expected to pass: smooth, R ≥ 2.6 λ;\ncut-end diffraction cancels in rung 3"),
        ("rung 3", "each defect class",
         "the DIFFERENCE field (defect − intact)\nat z = 30 and the MS plane, one gain\nfitted on the intact run",
         f"diff corr ≥ {CORR_GATE:.2f}, level within ±{LEVEL_GATE:g} dB;\nMS: detector signature cos ≥ 0.95,\nsize within ±3 dB",
         "fdtd_agreement\n--diff-target / --intact-external",
         "where physical optics can fail:\npredicted per class in Figs 03, 04"),
    ]
    fig, ax = plt.subplots(figsize=(20, 9.2))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.03)
    ax.axis("off")
    cols = [0.005, 0.07, 0.175, 0.39, 0.6, 0.79]
    heads = ["", "compares", "where / what", "gate (proposed)", "tool", "status / meaning"]
    for x, h in zip(cols, heads):
        ax.text(x, 1.0, h, fontsize=11, fontweight="bold", va="top", transform=ax.transAxes)
    shade = ["#fde0dd", "#fff7bc", "#e5f5e0", "#e5f5e0", "#ece2f0"]
    step, hgt = 0.162, 0.145
    for i, (r, what, where, gate, tool, note) in enumerate(rungs):
        y = 0.945 - i * step
        ax.add_patch(plt.Rectangle((0, y - hgt + 0.012), 1, hgt, transform=ax.transAxes, color=shade[i],
                                   lw=0, zorder=0))
        ax.text(cols[0], y, r, fontsize=13, fontweight="bold", va="top", transform=ax.transAxes)
        for x, t_ in zip(cols[1:], (what, where, gate, tool, note)):
            ax.text(x, y, t_, fontsize=9.8, va="top", transform=ax.transAxes, linespacing=1.35,
                    family="monospace" if t_ is tool else None)
        if i < len(rungs) - 1:
            ax.annotate("", xy=(0.03, y - hgt + 0.0), xytext=(0.03, y - 0.06), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.5))
    ax.text(0, 0.0,
            "“Good enough” for training (rung 3): the difference field IS the defect's signal. Within ±3 dB, a detection "
            "threshold learned on rail3D is off by at most a factor 2 in power, which keeps\nat least half of the 6 dB "
            "margin DESIGN_STUDY calls robust. At ≥ 0.90 correlation the residual is ≤ 44% of the signature's amplitude. "
            "The 130-detector signature (cos ≥ 0.95) is what a trained\nclassifier actually consumes. A class that FAILS "
            "is not a broken simulation: it says which way, and by how much, rail3D's training data is biased for that class.",
            fontsize=9.8, va="bottom", transform=ax.transAxes, color="#333", linespacing=1.4)
    fig.suptitle("Approval ladder: rail3D vs Lumerical FDTD, each rung gates the next", fontsize=14,
                 fontweight="bold", x=0.02, ha="left")
    _save(fig, "00_approval_ladder.png")


# ---------------------------------------------------------------------------
def fig_crack_physics(res):
    import po_canonical_2d as pc
    sw = res["depth_sweep"]
    w, d = 2.0, 6.0
    maps = {c: pc.near_field_map(w, d, c) for c in ("x-z", "y-z")}
    fig = plt.figure(figsize=(18, 10.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.25], hspace=0.38, wspace=0.28)
    for j, (cutk, title, phys) in enumerate((
            ("x-z", "LONGITUDINAL crack (slot along the rail):\nE_y ∥ the walls",
             f"TE mode, cutoff at width λ/2 = {config.WVL / 2:g} mm:\nnarrower slots are dark below the rim"),
            ("y-z", "TRANSVERSE crack (slot across the head):\nE_y ⊥ the walls",
             "TEM mode, no cutoff: the light reaches the floor\nand standing waves form (a shorted stub)"))):
        m = maps[cutk]
        ax = fig.add_subplot(gs[0, j])
        A = m["abs"] / np.nanmax(m["abs"][m["z"] > 0.5])
        im = ax.pcolormesh(m["x"], m["z"], A, cmap="inferno", vmin=0, vmax=2.0, shading="nearest")
        ax.plot(m["profile_x"], m["profile_z"], color="#4dd0e1", lw=1.4)
        ax.set_xlim(m["x"][0], m["x"][-1])
        ax.set_ylim(m["z"][0], m["z"][-1])
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("across the slot (mm)")
        ax.set_ylabel("z (mm)")
        ax.text(0.02, 0.03, phys, color="w", fontsize=9, transform=ax.transAxes)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="|total field| / incident (exact)")
    ax = fig.add_subplot(gs[0, 2])
    dd = np.asarray(sw["depth"])
    for sub, c in (("longitudinal", C_SUB["longitudinal"]), ("transverse", C_SUB["transverse"])):
        ax.plot(dd, sw[sub]["exact_dB"], "-", color=c, lw=2.2, label=f"{sub}: exact")
        ax.plot(dd, sw[sub]["rail3d_dB"], "--", color=c, lw=1.6, label=f"{sub}: rail3D (PO)")
    ax.axvspan(*config.CRACK_DEPTH_RANGE, color="0.85", lw=0, zorder=0)
    lt = config.WVL / math.cos(math.radians(sw["theta"]))
    for q in (0.25, 0.75, 1.25):
        ax.axvline(q * lt, color=C_SUB["transverse"], lw=0.6, ls=":")
    ax.set_xlabel(f"crack depth (mm), one line {sw['width']:g} mm wide, crown (55°)")
    ax.set_ylabel("defect signal in the metasurface window (dB)")
    ax.set_title("Signal vs depth: rail3D has no resonance\n(grey: the dataset's depth range; dotted: λ_t/4 stub resonances)")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(alpha=0.3)
    # far-field difference patterns: one good and one bad case
    rows = res["cases"]
    pick = [("crack_longitudinal_crown_2.0_6.0_1", "longitudinal, 2 mm × 6 mm, one line"),
            ("crack_transverse_crown_2.0_6.0_1", "transverse, 2 mm × 6 mm, one line"),
            ("crack_transverse_crown_2.5_8.0_3", "transverse, three lines 2.5 × 8 mm")]
    byid = {r["id"]: r for r in rows}
    for j, (cid, lab) in enumerate(pick):
        if cid not in byid:
            continue
        r = byid[cid]
        ax = fig.add_subplot(gs[1, j])
        phi = np.asarray(r["_pattern"]["phi"])
        de, dp = _pattern(r, "d_ex"), _pattern(r, "d_po")
        s = np.abs(de).max()
        ax.plot(phi, np.abs(de) / s, color=C_EX, lw=2, label="exact")
        ax.plot(phi, np.abs(dp) / s, color=C_PO, lw=1.6, ls="--", label="rail3D (PO)")
        half = res["ms_half_angle_deg"][r["cut"]]
        ax.axvspan(-half, half, color="#9ecae1", alpha=0.35, lw=0, label="toward the metasurface")
        ax.set_xlabel("far-field direction (deg from the surface normal)")
        ax.set_ylabel("|difference field| (exact peak = 1)")
        ax.set_title(f"{lab}\nMS window: exact − rail3D {fdtd_minus_rail3d(r):+.1f} dB, corr {r['corr_ms']:.3f};  "
                     f"all directions: corr {r['corr_hemi']:.2f}", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Why a crack is the hard case: at 1.5–3 mm wide (0.3–0.6 λ) it is a waveguide, not a mirror, "
                 "and which waveguide depends on the polarisation", fontsize=13, fontweight="bold", y=0.995)
    _save(fig, "01_crack_physics.png")


# ---------------------------------------------------------------------------
def fig_crack_maps(res):
    rows = [r for r in res["cases"] if r["cls"] == "crack"]
    W = sorted({r["width"] for r in rows})
    D = sorted({r["depth"] for r in rows})
    fig, axs = plt.subplots(2, 4, figsize=(19, 8.4), constrained_layout=True)
    combos = [(sub, where) for sub in ("longitudinal", "transverse") for where in ("crown", "shoulder")]
    for j, (sub, where) in enumerate(combos):
        for i, (key, cmap, lim, lab) in enumerate((
                ("level", "RdBu_r", (-8, 8), "exact − rail3D, difference-field level (dB), MS window\n"
                 "+ = rail3D under-predicts"),
                ("corr_hemi", "viridis", (0.5, 1.0), "difference-field correlation, all directions"))):
            ax = axs[i, j]
            M = np.full((len(D), len(W)), np.nan)
            for r in rows:
                if r["sub"] == sub and r["where"] == where and r["lines"] == 3:
                    M[D.index(r["depth"]), W.index(r["width"])] = \
                        fdtd_minus_rail3d(r) if key == "level" else r[key]
            im = ax.imshow(M, origin="lower", cmap=cmap, vmin=lim[0], vmax=lim[1], aspect="auto")
            for a in range(len(D)):
                for b in range(len(W)):
                    if np.isfinite(M[a, b]):
                        bad = abs(M[a, b]) > LEVEL_GATE if key.startswith("level") else M[a, b] < CORR_GATE
                        ax.text(b, a, f"{M[a, b]:+.1f}" if key.startswith("level") else f"{M[a, b]:.2f}",
                                ha="center", va="center", fontsize=10, fontweight="bold" if bad else None,
                                color="k" if not bad else ("#b00020" if key.startswith("level") else "w"))
            ax.set_xticks(range(len(W)), [f"{w:g}" for w in W])
            ax.set_yticks(range(len(D)), [f"{d:g}" for d in D])
            ax.set_xlabel("line width (mm)")
            ax.set_ylabel("depth (mm)")
            ax.axvline(W.index(2.5) - 0.5 if 2.5 in W else -1, color="k", lw=1, ls=":")
            ax.set_title(f"{sub}, {where} ({'55' if where == 'crown' else '35'}°), 3 lines", fontsize=10)
            if j == 3:
                fig.colorbar(im, ax=axs[i, :], fraction=0.02, pad=0.01, label=lab)
    fig.suptitle(f"Cracks over the dataset's range: bold = outside the rung-3 gate (±{LEVEL_GATE:g} dB, "
                 f"corr ≥ {CORR_GATE}); dotted line = the λ/2 cutoff width", fontsize=13, fontweight="bold")
    _save(fig, "02_crack_maps.png")


# ---------------------------------------------------------------------------
def class_groups(res):
    rows = res["cases"]
    out = {}
    for r in rows:
        if r["cls"] == "crack":
            if r["lines"] != 3:
                continue
            key = f"crack {r['sub']}"
        else:
            key = f"{r['cls']} ({r['sub']} cut)" if r["cls"] != "wear" else "wear"
        out.setdefault(key, []).append(r)
    order = ["wear", "dent (x-z cut)", "dent (y-z cut)", "shell (x-z cut)", "shell (y-z cut)",
             "crack longitudinal", "crack transverse"]
    return [(k, out[k]) for k in order if k in out]


def fig_scorecard(res):
    groups = class_groups(res)
    fig, axs = plt.subplots(1, 3, figsize=(19, 6.8), sharey=True, constrained_layout=True)
    yy = np.arange(len(groups))
    for ax, key, lab, gate in ((axs[0], "level", "difference-field level, exact − rail3D (dB)\n"
                                "metasurface window; + = rail3D under-predicts", ("band", LEVEL_GATE)),
                               (axs[1], "corr_ms", "difference-field correlation\nmetasurface window",
                                ("min", CORR_GATE)),
                               (axs[2], "corr_hemi", "difference-field correlation\nall directions (≈ the z = 30 monitor)",
                                ("min", CORR_GATE))):
        for i, (g, rs) in enumerate(groups):
            v = np.array([fdtd_minus_rail3d(r) if key == "level" else r[key] for r in rs])
            jit = (np.random.default_rng(i).random(len(v)) - 0.5) * 0.35
            ok = (np.abs(v) <= gate[1]) if gate[0] == "band" else (v >= gate[1])
            ax.scatter(v[ok], i + jit[ok], s=16, color="#1a7f37", alpha=0.75)
            ax.scatter(v[~ok], i + jit[~ok], s=16, color="#c62828", alpha=0.85)
            ax.plot([np.median(v)] * 2, [i - 0.3, i + 0.3], color="k", lw=2.2)
            ax.text(1.0, i, f" {100 * ok.mean():.0f}% pass", transform=ax.get_yaxis_transform(),
                    va="center", fontsize=8.5, color="#333")
        if gate[0] == "band":
            ax.axvspan(-gate[1], gate[1], color="#e5f5e0", lw=0, zorder=0)
            ax.axvline(0, color="k", lw=0.8)
        else:
            ax.axvspan(gate[1], 1.0, color="#e5f5e0", lw=0, zorder=0)
            ax.set_xlim(min(0.4, ax.get_xlim()[0]), 1.0)
        ax.set_xlabel(lab)
        ax.grid(alpha=0.3, axis="x")
    axs[0].set_yticks(yy, [g for g, _ in groups])
    axs[0].invert_yaxis()
    fig.suptitle("Predicted rung-3 agreement by defect class (2D exact vs rail3D's physical optics; every case of "
                 "the swept range; green band = the gate; black bar = median)", fontsize=12.5, fontweight="bold")
    _save(fig, "03_class_scorecard.png")


def class_table(res) -> list[dict]:
    out = []
    for g, rs in class_groups(res):
        lv = np.array([fdtd_minus_rail3d(r) for r in rs])
        ch = np.array([r["corr_hemi"] for r in rs])
        cm = np.array([r["corr_ms"] for r in rs])
        ok = (np.abs(lv) <= LEVEL_GATE) & (cm >= CORR_GATE)
        b, w = rs[int(np.argmin(np.abs(lv)))], rs[int(np.argmax(np.abs(lv)))]
        out.append({"class": g, "n": len(rs), "level_median": float(np.median(lv)),
                    "level_best": float(lv[np.argmin(np.abs(lv))]), "level_worst": float(lv[np.argmax(np.abs(lv))]),
                    "corr_ms_min": float(cm.min()), "corr_hemi_median": float(np.median(ch)),
                    "corr_hemi_min": float(ch.min()), "pass_frac": float(ok.mean()),
                    "best_id": b["id"], "worst_id": w["id"]})
    return out


# ---------------------------------------------------------------------------
def fig_rung3_summary():
    paths = {m: R3 / f"rung3_summary_{m}.json" for m in ("target", "predicted")}
    have = {m: json.loads(p.read_text()) for m, p in paths.items() if p.exists()}
    if not have:
        print("no rung-3 summaries yet: run fdtd_agreement.py --diff-target all [--predicted]")
        return
    names = list(next(iter(have.values())).keys())
    metrics = [("diff_corr", "difference agreement (MS)", (0.5, 1.0), ("min", CORR_GATE)),
               ("diff_level_dB", "difference level (MS, dB)", (-10, 10), ("band", LEVEL_GATE)),
               ("diff_barcode_cos", "detector signature cos", (0.5, 1.0), ("min", 0.95)),
               ("diff_barcode_dB", "detector signature size (dB)", (-10, 10), ("band", 3.0))]
    fig, axs = plt.subplots(1, 4, figsize=(19, 5.6), sharey=True, constrained_layout=True)
    yy = np.arange(len(names))
    for ax, (k, lab, lim, gate) in zip(axs, metrics):
        for m, mk, c, off in (("target", "o", "#b00020", -0.15), ("predicted", "D", "#6a1b9a", 0.15)):
            if m not in have:
                continue
            v = np.array([have[m][n]["ms_plane"][k] for n in names])
            ax.scatter(np.clip(v, *lim), yy + off, marker=mk, s=60, color=c, label=m, zorder=3)
        if gate[0] == "band":
            ax.axvspan(-gate[1], gate[1], color="#e5f5e0", lw=0)
            ax.axvline(0, color="k", lw=0.8)
        else:
            ax.axvspan(gate[1], lim[1], color="#e5f5e0", lw=0)
        ax.set_xlim(*lim)
        ax.set_xlabel(lab)
        ax.grid(alpha=0.3, axis="x")
    axs[0].set_yticks(yy, [n.replace("_", " ") for n in names])
    axs[0].invert_yaxis()
    axs[-1].legend(fontsize=9, loc="lower right")
    fig.suptitle("Rung 3 in 3D, the seed-0 sample of each class: TARGET (what passing looks like) vs PREDICTION "
                 "(rail3D re-weighted by the 2D exact/PO ratio).  Green = inside the gate.", fontsize=12.5,
                 fontweight="bold")
    _save(fig, "04_rung3_summary.png")


def po_figures(res):
    fig_ladder()
    fig_crack_physics(res)
    fig_crack_maps(res)
    fig_scorecard(res)
    tab = class_table(res)
    (config.GENERATED_DIR / "po_canonical_2d_table.json").write_text(json.dumps(tab, indent=1))
    for t in tab:
        print(f"  {t['class']:22s} n {t['n']:3d}  level median {t['level_median']:+5.1f} dB "
              f"(best {t['level_best']:+5.1f}, worst {t['level_worst']:+5.1f})  corr MS min {t['corr_ms_min']:.3f}  "
              f"hemi median {t['corr_hemi_median']:.2f} min {t['corr_hemi_min']:.2f}  pass {100 * t['pass_frac']:.0f}%")


def main() -> int:
    import po_canonical_2d as pc
    po_figures(json.loads(pc.OUT_JSON.read_text()))
    fig_rung3_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
