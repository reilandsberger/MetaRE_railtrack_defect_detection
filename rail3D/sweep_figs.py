"""sweep_figs.py -- figures for a setup_sweep.py run (called by sweep_eval.py).

    sweep_01_bounds.png      every setup's floor-ceiling interval; the pruning line
    sweep_02_dials.png       the dials: far edge, horn angle, height
    sweep_03_cells.png       type x position recall: today vs the best setup, and the ceiling
    sweep_04_data.png        data-centric: size strata, trust tags, crack width vs cutoff,
                             learning curve
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold"})
C_FLOOR, C_CEIL, C_DENSE = "#1b9e77", "#7570b3", "#d95f02"


def _name(c):
    return f"θ{c['theta']:g} H{c['H']:g} | {c['label']}"


def _save(fig, root: Path, name: str):
    p = root / name
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


def fig_bounds(res, root):
    cs = sorted(res["configs"], key=lambda c: -c["L8"])
    eta = str(res["eta_prune"])
    n = len(cs)
    fig, ax = plt.subplots(figsize=(13, 0.28 * n + 2.2))
    yy = np.arange(n)
    for i, c in enumerate(cs):
        col = "0.75" if c["pruned"] else C_CEIL
        ax.plot([c["L8"], c["U"][eta]], [i, i], color=col, lw=3, solid_capstyle="butt", zorder=1)
        ax.scatter(c["L8"], i, color="0.6" if c["pruned"] else C_FLOOR, s=30, zorder=3)
        ax.scatter(c["Ldense"], i, marker="|", color=C_DENSE, s=60, zorder=3)
    ax.axvline(res["best_floor"]["L8"], color="k", ls="--", lw=1)
    ax.set_yticks(yy, [_name(c) + ("  ☀" if c["bright"] else "") for c in cs], fontsize=7.5)
    for k, c in enumerate(cs):
        if c["placement"] == "t55_H150" and c["family"] == "today":
            ax.get_yticklabels()[k].set_color("tab:purple")
            ax.get_yticklabels()[k].set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xlabel(f"defect recall at {100 * res['fpr']:.0f}% false alarms, all types and positions")
    ax.scatter([], [], color=C_FLOOR, label="floor L8: no metasurface, 8 detectors (always reachable)")
    ax.plot([], [], color=C_CEIL, lw=3, label=f"up to the ceiling U: ideal matched filter (receiver noise {float(eta):g})")
    ax.scatter([], [], marker="|", color=C_DENSE, s=60, label="all dense detectors (reference)")
    ax.plot([], [], color="0.75", lw=3, label="pruned: ceiling below the best floor")
    ax.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0, -0.16 - 2.0 / max(n, 8)), ncol=2, frameon=False)
    ax.grid(alpha=0.3, axis="x")
    keep = sum(not c["pruned"] for c in cs)
    ax.set_title(f"Bound sandwich: {keep} of {n} setups can still win; the rest cannot, whatever the "
                 f"metasurface is trained to  (☀ = crown glint inside)", loc="left")
    _save(fig, root, "sweep_01_bounds.png")


def fig_dials(res, root):
    cs = res["configs"]
    eta = str(res["eta_prune"])
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.2), constrained_layout=True)
    ax = axs[0]
    thetas = sorted({c["theta"] for c in cs})
    Hs = sorted({c["H"] for c in cs})
    cols = dict(zip(thetas, plt.cm.viridis(np.linspace(0.1, 0.85, len(thetas)))))
    ls = dict(zip(Hs, ["-", "--", ":", "-."][:len(Hs)]))
    for th in thetas:
        for h in Hs:
            e = sorted([c for c in cs if c["theta"] == th and c["H"] == h and c["family"] == "extend"],
                       key=lambda c: c["far_frac"])
            if not e:
                continue
            ax.plot([c["far_frac"] for c in e], [c["L8"] for c in e], ls[h], color=cols[th], marker="o",
                    ms=4, label=f"θ{th:g} H{h:g}")
    ax.axvline(1.0, color="tab:red", lw=1)
    ax.text(1.0, 0.02, " crown glint", color="tab:red", fontsize=8, transform=ax.get_xaxis_transform())
    ax.set_xlabel("far edge / crown mirror point  (−x0 / H tan θ)")
    ax.set_ylabel("floor L8 recall")
    ax.set_title("A. Extending the far edge (near edge +75 mm)")
    ax.legend(fontsize=7.5, ncol=2)
    ax.grid(alpha=0.3)
    for ax, key, lab in ((axs[1], "L8", "floor L8"), (axs[2], "U", f"ceiling U ({float(eta):g})")):
        M = np.full((len(Hs), len(thetas)), np.nan)
        for i, h in enumerate(Hs):
            for j, th in enumerate(thetas):
                v = [c[key][eta] if key == "U" else c[key] for c in cs if c["theta"] == th and c["H"] == h]
                if v:
                    M[i, j] = max(v)
        im = ax.imshow(M, origin="lower", cmap="viridis", vmin=0, vmax=1, aspect="auto")
        for i in range(len(Hs)):
            for j in range(len(thetas)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="w" if M[i, j] < 0.6 else "k")
        ax.set_xticks(range(len(thetas)), [f"{t:g}°" for t in thetas])
        ax.set_yticks(range(len(Hs)), [f"{h:g}" for h in Hs])
        ax.set_xlabel("horn angle θ")
        ax.set_ylabel("metasurface height H (mm)")
        ax.set_title(f"{'B' if key == 'L8' else 'C'}. Best crop per placement: {lab}")
        fig.colorbar(im, ax=ax, fraction=0.05)
    fig.suptitle("The dials", fontsize=13, fontweight="bold")
    _save(fig, root, "sweep_02_dials.png")


def _cell_matrix(res, c, key):
    meta = [m for m in res["meta"][c["placement"]] if m["cls"] != "intact"]
    v = np.asarray(c["_per_sample"][key], float)
    types, pos = res["plan"]["types"], res["plan"]["positions"]
    M = np.full((len(types), len(pos)), np.nan)
    for i, t in enumerate(types):
        for j, s in enumerate(pos):
            m = np.array([mm["type"] == t and mm["s_nom"] == s for mm in meta])
            if m.any():
                M[i, j] = v[m].mean()
    return M


def fig_cells(res, root):
    cs = res["configs"]
    today = next((c for c in cs if c["placement"] == "t55_H150" and c["family"] == "today"), None)
    best = max(cs, key=lambda c: c["L8"])
    panels = [(today, "L8", "today: floor L8"), (best, "L8", "best setup: floor L8"),
              (best, "U", "best setup: ceiling U")]
    panels = [p for p in panels if p[0] is not None]
    fig, axs = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 4.6), constrained_layout=True)
    types, pos = res["plan"]["types"], res["plan"]["positions"]
    for ax, (c, key, lab) in zip(np.atleast_1d(axs), panels):
        M = _cell_matrix(res, c, key)
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        for i in range(len(types)):
            for j in range(len(pos)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                            color="w" if M[i, j] < 0.6 else "k")
        ax.set_xticks(range(len(pos)), [f"{p:g}" for p in pos])
        ax.set_yticks(range(len(types)), [t.replace("_", " ") for t in types])
        ax.set_xlabel("position s (mm): crown ← → gauge corner / face")
        ax.set_title(f"{lab}\n{_name(c)}", fontsize=10)
    fig.colorbar(im, ax=axs, fraction=0.02, label=f"recall at {100 * res['fpr']:.0f}% false alarms")
    _save(fig, root, "sweep_03_cells.png")


def fig_data(res, root):
    best = max(res["configs"], key=lambda c: c["L8"])
    meta = [m for m in res["meta"][best["placement"]] if m["cls"] != "intact"]
    hit = np.asarray(best["_per_sample"]["L8"], float)
    U = np.asarray(best["_per_sample"]["U"], float)
    fig, axs = plt.subplots(1, 4, figsize=(21, 4.8), constrained_layout=True)
    # A: size strata (the first size key of each class is depth or width)
    ax = axs[0]
    types = res["plan"]["types"]
    for t in types:
        idx = [i for i, m in enumerate(meta) if m["type"] == t]
        if not idx:
            continue
        u0 = np.array([meta[i]["u"][1 if meta[i]["cls"] == "crack" else 0] for i in idx])
        b = np.minimum((u0 * 4).astype(int), 3)
        ax.plot(range(4), [hit[idx][b == k].mean() if (b == k).any() else np.nan for k in range(4)],
                "-o", ms=4, label=t.replace("_", " "))
    ax.set_xticks(range(4), ["shallowest\nquarter", "2nd", "3rd", "deepest\nquarter"])
    ax.set_ylabel("floor L8 recall (best setup)")
    ax.set_title("A. Depth strata (Latin-hypercube bins)")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)
    # B: finding-34 trust tags
    ax = axs[1]
    tags = ["high", "medium", "low"]
    for k, tg in enumerate(tags):
        m = np.array([mm["po_trust"] == tg for mm in meta])
        if m.any():
            ax.bar(k - 0.18, hit[m].mean(), 0.36, color=C_FLOOR)
            ax.bar(k + 0.18, U[m].mean(), 0.36, color=C_CEIL)
            ax.text(k, 0.02, f"n={m.sum()}", ha="center", fontsize=8, color="w")
    ax.set_xticks(range(3), [f"{t}\ntrust" for t in tags])
    ax.set_ylim(0, 1.05)
    ax.set_title("B. By rail3D trust (finding 34)")
    ax.bar([], [], color=C_FLOOR, label="floor")
    ax.bar([], [], color=C_CEIL, label="ceiling")
    ax.legend(fontsize=8)
    # C: cracks vs width / cutoff
    ax = axs[2]
    for t, col in (("crack_longitudinal", "#7570b3"), ("crack_transverse", "#1b9e77"), ("crack_oblique", "#e7298a")):
        idx = [i for i, m in enumerate(meta) if m["type"] == t]
        if not idx:
            continue
        w = np.array([meta[i]["width_over_cutoff"] for i in idx])
        bins = np.linspace(0.6, 1.2, 5)
        bc = 0.5 * (bins[1:] + bins[:-1])
        r = [hit[idx][(w >= a) & (w < b)].mean() if ((w >= a) & (w < b)).any() else np.nan
             for a, b in zip(bins[:-1], bins[1:])]
        ax.plot(bc, r, "-o", color=col, ms=4, label=t.replace("_", " "))
    ax.axvline(1.0, color="k", lw=1, ls=":")
    ax.text(1.0, 0.02, " λ/2 cutoff", fontsize=8, transform=ax.get_xaxis_transform())
    ax.set_xlabel("crack width / (λ/2)")
    ax.set_title("C. Cracks vs the waveguide cutoff")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    # D: learning curve
    ax = axs[3]
    lc = res.get("learning_curve")
    if lc:
        ax.plot(np.array(lc["fracs"]) * 100, lc["L8"], "-o", color=C_FLOOR)
        ax.set_xlabel("% of the training folds used (100% = the full plan)")
        ax.set_ylabel("floor L8 recall")
        ax.set_title("D. Learning curve, best setup:\nstill climbing at 100% = more data pays")
        ax.grid(alpha=0.3)
    fig.suptitle(f"Data-centric view of the best setup: {_name(best)}", fontsize=13, fontweight="bold")
    _save(fig, root, "sweep_04_data.png")


def all_figures(res, root: Path):
    fig_bounds(res, root)
    fig_dials(res, root)
    fig_cells(res, root)
    fig_data(res, root)


if __name__ == "__main__":
    import sys
    r = Path(sys.argv[1])
    all_figures(json.loads((r / "sweep_eval_full.json").read_text()), r)
