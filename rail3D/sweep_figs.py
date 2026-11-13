"""sweep_figs.py -- figures for a setup_sweep.py run (called by sweep_eval.py).

    sweep_01_bounds.png      every setup's floor-ceiling interval; the pruning line
    sweep_02_dials.png       the dials: far edge, horn angle, height
    sweep_03_cells.png       type x position recall: today vs the best setup, and the ceiling
    sweep_04_data.png        data-centric: size strata, trust tags, crack width vs cutoff,
                             learning curve
  detection only (defect vs intact, no classification; the 2026-11-05 focus):
    sweep_05_detection.png   setups ranked: AUC, recall and accuracy at 5% false alarms for
                             each detection score, and the average barcode distance
    sweep_05b_detection_slide.png  the same for a slide: top 10 + today, A/B/D, 300 dpi
    sweep_05c_design_story.png     condensed: one row group per dial (angle, height, far edge,
                                   near edge), anchored on the best setup
    sweep_05d_near_edge.png        near edge at +75 vs at the horn's beam limit, every pair
    sweep_06_detection_dials.png   far edge and theta x H, scored by detection AUC
    sweep_07_detection_cells.png   type x position: today vs the best setup, two scores

    python sweep_figs.py <sweep root>      # redraw from sweep_eval_full.json
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


def _save(fig, root: Path, name: str, dpi: int = 140, tight: bool = True):
    p = root / name
    fig.savefig(p, dpi=dpi, bbox_inches="tight" if tight else None)
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
        ax.set_title("D. Learning curve, best setup\n(still rising at 100% would mean more data pays)")
        ax.grid(alpha=0.3)
    fig.suptitle(f"Data-centric view of the best setup: {_name(best)}", fontsize=13, fontweight="bold")
    _save(fig, root, "sweep_04_data.png")


# ---------------------------------------------------------------------------
# detection only (defect vs intact), the 2026-11-05 focus
# ---------------------------------------------------------------------------
DET_STYLE = {   # score -> (colour, marker, short label)
    "cos8":    ("#1b9e77", "o", "total barcode difference, 8 det."),
    "maxz8":   ("#d95f02", "D", "few-detector out-of-bounds, 8 det."),
    "cosall":  ("#66c2a5", "o", "total barcode difference, all det."),
    "maxzall": ("#fc8d62", "D", "few-detector out-of-bounds, all det."),
    "logreg8": ("#7570b3", "s", "trained linear detector, 8 det."),
}
PRIMARY = "cos8"


def _is_today(c):
    return c["placement"] == "t55_H150" and c["family"] == "today"


def _det_rows(res, top=24):
    cs = sorted(res["configs"], key=lambda c: -c["detect"][PRIMARY]["auc"])
    rows = cs[:top]
    today = next((c for c in cs if _is_today(c)), None)
    if today is not None and today not in rows:
        rows.append(today)
    return rows


def _label_today(ax, rows):
    for k, c in enumerate(rows):
        if _is_today(c):
            ax.get_yticklabels()[k].set_color("tab:purple")
            ax.get_yticklabels()[k].set_fontweight("bold")


def fig_detection(res, root):
    """Setup comparison: AUC, recall and accuracy at 5% false alarms, barcode distance."""
    rows = _det_rows(res)
    n = len(rows)
    fpr = 100 * res["fpr"]
    fig, axs = plt.subplots(1, 4, figsize=(22, 0.32 * n + 2.8), sharey=True, constrained_layout=True)
    yy = np.arange(n)
    panels = (("auc", "A. Detection AUC (defect vs intact)", (0.5, 1.0)),
              ("recall", f"B. Defects caught at {fpr:.0f}% false alarms", (0, 1.0)),
              ("best_bal_acc", "C. Accuracy: best (recall + specificity) / 2", (0.5, 1.0)))
    for ax, (key, title, lim) in zip(axs, panels):
        for s, (col, mk, lab) in DET_STYLE.items():
            v = [c["detect"][s][key] for c in rows]
            ax.scatter(v, yy, color=col, marker=mk, s=34 if s == PRIMARY else 20,
                       zorder=3 if s == PRIMARY else 2, label=lab,
                       edgecolor="k" if s == PRIMARY else "none", linewidth=0.6)
        ax.set_xlim(lim[0] - 0.02, lim[1] + 0.01)
        ax.set_title(title, loc="left")
        ax.grid(alpha=0.3, axis="x")
    axs[0].set_yticks(yy, [_name(c) + ("  ☀" if c["bright"] else "") for c in rows], fontsize=7.5)
    axs[0].invert_yaxis()
    _label_today(axs[0], rows)
    ax = axs[3]
    dm = np.array([c["detect"]["distance"]["defect_median"] for c in rows])
    ip = np.array([c["detect"]["distance"]["intact_p95"] for c in rows])
    im = np.array([c["detect"]["distance"]["intact_mean"] for c in rows])
    ax.barh(yy, dm, 0.38, color=DET_STYLE[PRIMARY][0], label="defects: median distance")
    ax.barh(yy + 0.38, ip, 0.38, color="0.65", label=f"intact: {100 - fpr:.0f}th percentile (the alarm line)")
    ax.scatter(im, yy + 0.38, marker="|", color="k", s=40, label="intact: mean")
    for k in range(n):
        ax.text(max(dm[k], ip[k]) * 1.15, yy[k] + 0.19, f"×{dm[k] / ip[k]:.1f}", va="center", fontsize=7)
    ax.set_xscale("log")
    ax.set_title("D. Average barcode distance from intact\n(1 − cos, 8 detectors; ×  = defect / alarm line)",
                 loc="left")
    ax.grid(alpha=0.3, axis="x")
    below = (0, -0.02)
    ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=below, frameon=False)
    axs[0].legend(fontsize=7.5, loc="upper left", bbox_to_anchor=below, ncol=1, frameon=False)
    fig.suptitle(f"Detection only, no metasurface: which setup separates defects from intact rails?  "
                 f"(sorted by {DET_STYLE[PRIMARY][2]} AUC; purple = today; ☀ = crown glint inside; "
                 f"n = {rows[0]['n_samples'] - rows[0]['n_intact']} defects / {rows[0]['n_intact']} intact)",
                 fontsize=12, fontweight="bold")
    _save(fig, root, "sweep_05_detection.png", dpi=250)


SLIDE_LABEL = {"cos8": "total difference, 8 det.", "maxz8": "out-of-bounds, 8 det.",
               "cosall": "total difference, all 130", "maxzall": "out-of-bounds, all 130",
               "logreg8": "trained linear, 8 det."}


def fig_detection_slide(res, root, top: int = 10, size=(9.6, 4.6), dpi: int = 300):
    """sweep_05 for a slide: the top setups + today, panels A, B and D, slide-size type.
    Drawn at its exact placed size (no tight bbox), so the aspect ratio is `size`."""
    rows = _det_rows(res, top)
    n, fpr = len(rows), 100 * res["fpr"]
    fig = plt.figure(figsize=size)
    gs = fig.add_gridspec(1, 3, left=0.215, right=0.985, bottom=0.075, top=0.765, wspace=0.08,
                          width_ratios=(1, 1, 1.05))
    axs = [fig.add_subplot(gs[0])]
    axs += [fig.add_subplot(gs[k], sharey=axs[0]) for k in (1, 2)]
    yy = np.arange(n)
    for ax, (key, title, lim) in zip(axs, (("auc", "A. AUC", (0.6, 1.0)),
                                           ("recall", f"B. Caught at {fpr:.0f}% false alarms", (0.3, 1.0)))):
        for s_, (col, mk, _) in DET_STYLE.items():
            ax.scatter([c["detect"][s_][key] for c in rows], yy, color=col, marker=mk,
                       s=42 if s_ == PRIMARY else 26, zorder=3 if s_ == PRIMARY else 2,
                       edgecolor="k" if s_ == PRIMARY else "none", linewidth=0.6, label=SLIDE_LABEL[s_])
        ax.set_xlim(lim[0] - 0.01, lim[1] + 0.01)
        ax.set_title(title, loc="left", fontsize=9.5)
        ax.grid(alpha=0.3, axis="x")
        ax.tick_params(labelsize=8)
    lab = [f"θ{c['theta']:g} H{c['H']:g}   x {c['x0']:.0f} … {c['x1']:+.0f}"
           + ("  today" if _is_today(c) else "") + ("  ☀" if c["bright"] else "") for c in rows]
    axs[0].set_yticks(yy, lab, fontsize=8)
    axs[0].invert_yaxis()
    _label_today(axs[0], rows)
    for ax in axs[1:]:
        ax.tick_params(labelleft=False)
    ax = axs[2]
    dm = np.array([c["detect"]["distance"]["defect_median"] for c in rows])
    ip = np.array([c["detect"]["distance"]["intact_p95"] for c in rows])
    im = np.array([c["detect"]["distance"]["intact_mean"] for c in rows])
    ax.barh(yy - 0.19, dm, 0.38, color=DET_STYLE[PRIMARY][0], label="defects: median")
    ax.barh(yy + 0.19, ip, 0.38, color="0.65", label=f"intact: {100 - fpr:.0f}th pct (alarm line)")
    ax.scatter(im, yy + 0.19, marker="|", color="k", s=30, label="intact: mean", zorder=3)
    for k in range(n):
        ax.text(max(dm[k], ip[k]) * 1.12, yy[k], f"×{dm[k] / ip[k]:.1f}", va="center", fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlim(right=max(dm.max(), ip.max()) * 3)
    ax.set_title("C. Barcode distance (8 det.)\n× = defect median ÷ alarm line", loc="left", fontsize=9.5)
    ax.grid(alpha=0.3, axis="x")
    ax.tick_params(labelsize=8)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.215, 0.985), ncol=3, fontsize=8,
               frameon=False, columnspacing=1.0, handletextpad=0.3,
               title="A, B:  circle = total barcode difference,  diamond = few-detector out-of-bounds",
               title_fontsize=8, alignment="left")
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", bbox_to_anchor=(0.995, 0.985), ncol=1, fontsize=8,
               frameon=False, handletextpad=0.4, labelspacing=0.3, title="C:", title_fontsize=8,
               alignment="left")
    fig.text(0.005, 0.985, f"top {top} setups + today\nsorted by dark-green AUC\n"
             f"☀ = crown glint in the plane\n{rows[0]['n_samples'] - rows[0]['n_intact']} defects / "
             f"{rows[0]['n_intact']} intact", va="top", fontsize=7.5, color="0.3")
    _save(fig, root, "sweep_05b_detection_slide.png", dpi=dpi, tight=False)


STORY_RULES = ("cos8", "maxz8")


def _story_rows(res):
    """Rows that isolate one design dial at a time, anchored on the best setup."""
    C = res["configs"]
    key = lambda c: c["detect"][PRIMARY]["auc"]                  # noqa: E731
    best = max(C, key=key)
    th0, h0, pl0 = best["theta"], best["H"], best["placement"]
    today = next((c for c in C if _is_today(c)), None)
    groups = []
    if today is not None:
        groups.append(("Today's setup", [today]))
    rows = [max((c for c in C if c["theta"] == t and c["H"] == h0), key=key)
            for t in sorted({c["theta"] for c in C}) if any(c["theta"] == t and c["H"] == h0 for c in C)]
    groups.append((f"1. Horn angle θ\n(H {h0:g} mm, best plane crop each)", rows))
    rows = [max((c for c in C if c["H"] == h and c["theta"] == th0), key=key)
            for h in sorted({c["H"] for c in C}) if any(c["H"] == h and c["theta"] == th0 for c in C)]
    groups.append((f"2. Metasurface height H\n(θ {th0:g}°, best plane crop each)", rows))
    ext = [c for c in C if c["placement"] == pl0 and c["family"] in ("today", "extend")]
    ext = sorted(ext, key=lambda c: (-c["x0"], c["family"] != "today"))
    ext = [c for c in ext if not (c["family"] == "extend" and c["x0"] > -75)]
    groups.append((f"3. Far edge, crown side\n(θ {th0:g}° H {h0:g}, near edge at +75 mm)", ext))
    near = {c["x0"]: c for c in C if c["placement"] == pl0 and c["family"] == "extend_near"}
    pairs = sorted([c for c in C if c["placement"] == pl0 and c["family"] == "extend" and c["x0"] in near],
                   key=key, reverse=True)[:2]
    if pairs:
        rows = []
        for c in sorted(pairs, key=lambda c: -c["x0"]):
            rows += [c, near[c["x0"]]]
        groups.append((f"4. Near edge toward the horn\n(θ {th0:g}° H {h0:g}: +75 mm vs the beam limit)", rows))
    return groups, best


def fig_detection_story(res, root):
    """sweep_05 condensed: one group of rows per design dial, the best setup as the anchor."""
    groups, best = _story_rows(res)
    fpr = 100 * res["fpr"]
    labels, rows, heads, ys = [], [], [], []
    y = 0.0
    for title, rs in groups:
        heads.append((y + 0.25 * title.count("\n"), title))
        y += 1.0 + 0.5 * title.count("\n")
        for c in rs:
            ys.append(y)
            rows.append(c)
            labels.append(f"θ{c['theta']:g}  H{c['H']:g}   x {c['x0']:.0f} … {c['x1']:+.0f}"
                          + ("  today" if _is_today(c) else "") + ("  ☀" if c["bright"] else "")
                          + ("  ★" if c is best else ""))
            y += 1.0
        y += 0.35
    ys = np.array(ys)
    height = 0.34 * y + 2.0
    fig, axs = plt.subplots(1, 4, figsize=(18, height), sharey=True,
                            gridspec_kw={"width_ratios": (1, 1, 1, 1.15)})
    fig.subplots_adjust(left=0.2, right=0.99, top=1 - 1.15 / height, bottom=0.85 / height, wspace=0.06)
    panels = (("auc", "A. Detection AUC", (0.85, 1.0)),
              ("recall", f"B. Defects caught at {fpr:.0f}% false alarms", (0.65, 1.0)),
              ("best_bal_acc", "C. Best balanced accuracy", (0.8, 1.0)))
    for ax, (k, title, lim) in zip(axs, panels):
        for s in STORY_RULES:
            col, mk, lab = DET_STYLE[s]
            v = np.array([c["detect"][s][k] for c in rows])
            ax.scatter(v, ys, color=col, marker=mk, s=55 if s == PRIMARY else 45, zorder=3,
                       edgecolor="k" if s == PRIMARY else "none", linewidth=0.6, label=lab)
            if s == "maxz8":
                for yy, vv in zip(ys, v):
                    ax.text(min(vv, lim[1]) + 0.005, yy, f"{vv:.3f}" if k == "auc" else f"{vv:.2f}",
                            va="center", fontsize=7.5, color=col)
        ax.set_xlim(lim[0], lim[1] + 0.04)
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(alpha=0.3, axis="x")
    ax = axs[3]
    dm = np.array([c["detect"]["distance"]["defect_median"] for c in rows])
    ip = np.array([c["detect"]["distance"]["intact_p95"] for c in rows])
    ax.barh(ys - 0.18, dm, 0.36, color=DET_STYLE[PRIMARY][0], label="defects: median distance")
    ax.barh(ys + 0.18, ip, 0.36, color="0.65", label=f"intact: {100 - fpr:.0f}th percentile (alarm line)")
    for k in range(len(rows)):
        ax.text(max(dm[k], ip[k]) * 1.12, ys[k], f"×{dm[k] / ip[k]:.1f}", va="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(right=max(dm.max(), ip.max()) * 3)
    ax.set_title("D. Barcode distance (total difference, 8 det.)\n× = defect median ÷ alarm line",
                 loc="left", fontsize=11)
    ax.grid(alpha=0.3, axis="x")
    axs[0].set_yticks(ys, labels, fontsize=9)
    for t, c in zip(axs[0].get_yticklabels(), rows):
        if _is_today(c):
            t.set_color("tab:purple")
        if c is best:
            t.set_fontweight("bold")
    axs[0].set_ylim(y - 0.2, -0.8)
    for y0, title in heads:
        n = title.count("\n")
        for a in axs:
            a.axhspan(y0 - 0.45 - 0.25 * n, y0 + 0.45 + 0.25 * n, color="0.92", zorder=0)
        axs[0].text(-0.01, y0, title, transform=axs[0].get_yaxis_transform(), ha="right", va="center",
                    fontsize=9.5, fontweight="bold", color="0.15", linespacing=1.15)
    for a in axs[1:]:
        a.tick_params(labelleft=False)
    h1, l1 = axs[0].get_legend_handles_labels()
    h2, l2 = axs[3].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=4, fontsize=9.5, frameon=False,
               bbox_to_anchor=(0.6, 0.0))
    n_def = rows[0]["n_samples"] - rows[0]["n_intact"]
    fig.suptitle("One design dial at a time: no metasurface, 8 untrained detectors  "
                 f"(★ best setup; ☀ crown glint inside the plane; {n_def} defects / {rows[0]['n_intact']} "
                 "intact per row; printed numbers = out-of-bounds rule)", fontsize=12, fontweight="bold")
    _save(fig, root, "sweep_05c_design_story.png", dpi=250, tight=False)


def fig_near_edge(res, root):
    """Every far edge of every placement solved two ways: near edge at +75 vs at the horn's
    beam limit. On the diagonal = widening toward the horn changes nothing."""
    C = res["configs"]
    near = {(c["placement"], c["x0"]): c for c in C if c["family"] == "extend_near"}
    pairs = [(c, near[(c["placement"], c["x0"])]) for c in C
             if c["family"] == "extend" and (c["placement"], c["x0"]) in near]
    if not pairs:
        return
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.8), constrained_layout=True,
                            gridspec_kw={"width_ratios": (1, 1, 1.5)})
    th = sorted({c["theta"] for c, _ in pairs})
    cols = dict(zip(th, plt.cm.viridis(np.linspace(0, 0.85, len(th)))))
    for ax, (s, k, title) in zip(axs[:2], (("cos8", "auc", "A. AUC, total barcode difference"),
                                           ("maxz8", "recall", f"B. Caught at {100 * res['fpr']:.0f}% false "
                                                               "alarms, out-of-bounds"))):
        a = np.array([p[0]["detect"][s][k] for p in pairs])
        b = np.array([p[1]["detect"][s][k] for p in pairs])
        lo, hi = min(a.min(), b.min()) - 0.01, max(a.max(), b.max()) + 0.01
        tol = 0.01 if k == "auc" else 0.02
        ax.fill_between([lo, hi], [lo - tol, hi - tol], [lo + tol, hi + tol], color="0.88", zorder=0,
                        label=f"±{tol:g} (rough noise band)")
        ax.plot([lo, hi], [lo, hi], "k-", lw=1, label="no change")
        for t in th:
            m = np.array([p[0]["theta"] == t for p in pairs])
            ax.scatter(a[m], b[m], color=cols[t], s=30, label=f"θ {t:g}°", zorder=3)
        d = b - a
        ax.set_title(f"{title}\nwider toward the horn: mean change {d.mean():+.3f}\nbetter in "
                     f"{int((d > 0).sum())} of {len(d)} pairs, worse in {int((d < 0).sum())}; "
                     f"{int((np.abs(d) <= tol).sum())} within ±{tol:g}", loc="left", fontsize=10)
        ax.set_xlabel("near edge at +75 mm")
        ax.set_ylabel("near edge at the horn's beam limit")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    ax = axs[2]
    best_ext = {}
    for c in C:
        if c["family"] in ("extend", "today", "extend_near"):
            k = (c["placement"], "near" if c["family"] == "extend_near" else "75")
            best_ext[k] = max(best_ext.get(k, 0.0), c["detect"]["maxz8"]["recall"])
    pls = sorted({p for p, _ in best_ext}, key=lambda p: (float(p.split("_H")[0][1:]), float(p.split("_H")[1])))
    xs = np.arange(len(pls))
    v75 = np.array([best_ext.get((p, "75"), np.nan) for p in pls])
    vn = np.array([best_ext.get((p, "near"), np.nan) for p in pls])
    ax.bar(xs - 0.2, v75, 0.4, color="0.6", label="near edge at +75 mm")
    ax.bar(xs + 0.2, vn, 0.4, color="#1b9e77", label="near edge at the horn's beam limit")
    for x_, a_, b_ in zip(xs, v75, vn):
        if np.isfinite(b_):
            ax.text(x_ + 0.2, b_ + 0.002, f"{b_ - a_:+.3f}", ha="center", fontsize=7, rotation=90, va="bottom")
        else:
            ax.text(x_ + 0.2, 0.802, "no room: beam\nlimit ≤ +75 mm", ha="center", fontsize=6.5, rotation=90,
                    va="bottom", color="0.4")
    ax.set_xticks(xs, [p.replace("t", "θ").replace("_H", "\nH") for p in pls], fontsize=8)
    ax.set_ylim(0.8, 0.96)
    ax.set_ylabel(f"caught at {100 * res['fpr']:.0f}% false alarms (out-of-bounds)")
    ax.set_title("C. Best far edge per placement, with and without the extra width toward the horn\n"
                 "(label = change in the share caught)", loc="left", fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8.5, loc="upper left")
    fig.suptitle("Does widening the plane toward the horn help? Each point is one far edge of one placement, "
                 "solved both ways on the same rails", fontweight="bold")
    _save(fig, root, "sweep_05d_near_edge.png", dpi=200)


def fig_detection_dials(res, root):
    """The dials, scored by detection AUC: far edge, and the theta x H grid."""
    cs = res["configs"]
    thetas = sorted({c["theta"] for c in cs})
    Hs = sorted({c["H"] for c in cs})
    near = any(c["family"] == "extend_near" for c in cs)
    fig, axs = plt.subplots(1, 4 if near else 3, figsize=(24 if near else 18, 5.2), constrained_layout=True)
    ax = axs[0]
    cols = dict(zip(thetas, plt.cm.viridis(np.linspace(0.1, 0.85, len(thetas)))))
    ls = dict(zip(Hs, ["-", "--", ":", "-."][:len(Hs)]))
    for th in thetas:
        for h in Hs:
            e = sorted([c for c in cs if c["theta"] == th and c["H"] == h and c["family"] == "extend"],
                       key=lambda c: c["far_frac"])
            if e:
                ax.plot([c["far_frac"] for c in e], [c["detect"][PRIMARY]["auc"] for c in e], ls[h],
                        color=cols[th], marker="o", ms=4, label=f"θ{th:g} H{h:g}")
    t = next((c for c in cs if _is_today(c)), None)
    if t is not None:
        ax.axhline(t["detect"][PRIMARY]["auc"], color="tab:purple", lw=1, ls="--")
        ax.text(0.01, t["detect"][PRIMARY]["auc"], " today", color="tab:purple", fontsize=8, va="bottom",
                transform=ax.get_yaxis_transform())
    ax.axvline(1.0, color="tab:red", lw=1)
    ax.text(1.0, 0.02, " crown glint", color="tab:red", fontsize=8, transform=ax.get_xaxis_transform())
    ax.set_xlabel("far edge / crown mirror point  (−x0 / H tan θ)")
    ax.set_ylabel(f"detection AUC ({DET_STYLE[PRIMARY][2]})")
    ax.set_title("A. Extending the far edge (near edge +75 mm)")
    ax.legend(fontsize=7.5, ncol=2)
    ax.grid(alpha=0.3)
    for ax, s, lab in ((axs[1], "cos8", "B"), (axs[2], "maxz8", "C")):
        M = np.full((len(Hs), len(thetas)), np.nan)
        for i, h in enumerate(Hs):
            for j, th in enumerate(thetas):
                v = [c["detect"][s]["auc"] for c in cs if c["theta"] == th and c["H"] == h]
                if v:
                    M[i, j] = max(v)
        im = ax.imshow(M, origin="lower", cmap="viridis", vmin=0.5, vmax=1, aspect="auto")
        for i in range(len(Hs)):
            for j in range(len(thetas)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                            color="w" if M[i, j] < 0.8 else "k")
        ax.set_xticks(range(len(thetas)), [f"{t:g}°" for t in thetas])
        ax.set_yticks(range(len(Hs)), [f"{h:g}" for h in Hs])
        ax.set_xlabel("horn angle θ")
        ax.set_ylabel("metasurface-plane height H (mm)")
        ax.set_title(f"{lab}. Best crop per placement: AUC, {DET_STYLE[s][2]}")
        fig.colorbar(im, ax=ax, fraction=0.05)
    if near:
        # D: what widening toward the horn (to its beam limit) adds over the +75 mm near edge
        ax = axs[3]
        G = np.full((len(Hs), len(thetas)), np.nan)
        xh = {}
        for i, h in enumerate(Hs):
            for j, th in enumerate(thetas):
                a = [c["detect"][PRIMARY]["auc"] for c in cs if c["theta"] == th and c["H"] == h
                     and c["family"] in ("extend", "today")]
                b = [c for c in cs if c["theta"] == th and c["H"] == h and c["family"] == "extend_near"]
                if a and b:
                    G[i, j] = max(c["detect"][PRIMARY]["auc"] for c in b) - max(a)
                    xh[i, j] = b[0]["x1"]
        v = np.nanmax(np.abs(G)) if np.isfinite(G).any() else 0.05
        im = ax.imshow(G, origin="lower", cmap="RdBu", vmin=-max(v, 0.01), vmax=max(v, 0.01), aspect="auto")
        for (i, j), x1 in xh.items():
            ax.text(j, i, f"{G[i, j]:+.3f}\nto x {x1:.0f}", ha="center", va="center", fontsize=9,
                    color="w" if abs(G[i, j]) > 0.6 * max(v, 0.01) else "k")
        ax.set_xticks(range(len(thetas)), [f"{t:g}°" for t in thetas])
        ax.set_yticks(range(len(Hs)), [f"{h:g}" for h in Hs])
        ax.set_xlabel("horn angle θ")
        ax.set_ylabel("metasurface-plane height H (mm)")
        ax.set_title("D. AUC gain: near edge out to the horn's beam limit\n(vs the best crop with "
                     "its near edge at +75; total barcode difference)", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.05)
    fig.suptitle("The dials, scored by detection only", fontsize=13, fontweight="bold")
    _save(fig, root, "sweep_06_detection_dials.png")


def fig_detection_cells(res, root):
    """Where on the rail each score catches defects: today vs the best setup."""
    cs = res["configs"]
    today = next((c for c in cs if _is_today(c)), None)
    best = max(cs, key=lambda c: c["detect"][PRIMARY]["auc"])
    types, pos = res["plan"]["types"], res["plan"]["positions"]
    panels = [(c, s) for c in (today, best) if c is not None for s in ("cos8", "maxz8")]
    fig, axs = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 4.6), constrained_layout=True)
    for ax, (c, s) in zip(np.atleast_1d(axs), panels):
        meta = [m for m in res["meta"][c["placement"]] if m["cls"] != "intact"]
        v = np.asarray(c["_per_sample"]["detect_hits"][s], float)
        M = np.full((len(types), len(pos)), np.nan)
        for i, t in enumerate(types):
            for j, p in enumerate(pos):
                m = np.array([mm["type"] == t and mm["s_nom"] == p for mm in meta])
                if m.any():
                    M[i, j] = v[m].mean()
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        for i in range(len(types)):
            for j in range(len(pos)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                            color="w" if M[i, j] < 0.6 else "k")
        ax.set_xticks(range(len(pos)), [f"{p:g}" for p in pos])
        ax.set_yticks(range(len(types)), [t.replace("_", " ") for t in types])
        ax.set_xlabel("position s (mm): crown ← → gauge corner")
        who = "today" if c is today else "best setup"
        ax.set_title(f"{who}: {DET_STYLE[s][2]}\n{_name(c)}  (recall {c['detect'][s]['recall']:.2f})",
                     fontsize=9.5)
    fig.colorbar(im, ax=axs, fraction=0.02, label=f"defects caught at {100 * res['fpr']:.0f}% false alarms")
    _save(fig, root, "sweep_07_detection_cells.png")


def all_figures(res, root: Path):
    fig_bounds(res, root)
    fig_dials(res, root)
    fig_cells(res, root)
    fig_data(res, root)
    if res["configs"] and "detect" in res["configs"][0]:
        fig_detection(res, root)
        fig_detection_slide(res, root)
        fig_detection_story(res, root)
        fig_near_edge(res, root)
        fig_detection_dials(res, root)
        fig_detection_cells(res, root)


if __name__ == "__main__":
    import sys
    r = Path(sys.argv[1])
    all_figures(json.loads((r / "sweep_eval_full.json").read_text()), r)
