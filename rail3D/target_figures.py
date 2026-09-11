"""Milestone TARGET figures — what a successful rail3D result would look like.

    python target_figures.py                    # reads the real analysis.json
    python target_figures.py --no-current       # targets alone, no measured overlay

These are **specifications, not measurements.** Nothing here is simulated,
trained or observed: every target curve is drawn from a stated analytic model to
hit a stated number. They exist to answer a reviewer's question -- "what would
good look like?" -- with something concrete enough to be held to.

Every panel is labelled TARGET, every figure carries a provenance footer, and
the CURRENT measured value is drawn alongside each target (read from the real
`analysis.json`, not typed in) so the gap is visible in the same frame. Do not
strip that overlay: a target figure that cannot be told apart from a result
figure is how honest work turns into misconduct by accident.

The targets themselves:

  detection AUC >= 0.98 per class      today: crack 0.835, dent 0.870
  recall >= 0.95 at 5% FPR             today: crack 0.39, dent 0.565
  confusion diagonal >= 0.90           today: crack 0.90 (base-rate artifact,
                                       README finding 25), dent 0.205
  detection rate FLAT in s0            today: crack 0.03 -> 0.82 across
                                       s0 107 -> 148 mm (README finding 25)

s0 spans the full railhead arc for crack and dent, which are sampled across it.
Shell is confined to the gauge corner by SHELL_GAUGE_X_RANGE -- a deliberate
modelling choice, so its s0 panel spans only that band; wear has no s0 panel at
all, matching analyze_results.py.

The last one is the real engineering milestone and the one worth defending to a
committee. Wear and shell already exceed the AUC target, so a higher *mean* is
not the goal -- UNIFORMITY across arc position is, because the present system is
not weak, it is blind outside the gauge corner. A flat curve in the s0 column
means the field of view was fixed; a higher average could just mean more defects
were placed where the system already looks.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import torch                             # noqa: E402

from rail3d import config                # noqa: E402

CLASSES = list(config.CLASS_NAMES)                     # crack, dent, wear, shell
TARGET_AUC = {"crack": 0.985, "dent": 0.982, "wear": 0.999, "shell": 0.996}
TARGET_DIAG = {"crack": 0.94, "dent": 0.92, "wear": 0.97, "shell": 0.93}
TARGET_RATE = 0.96                                     # flat detection rate
TARGET_FPR = 0.05
COL = {"crack": "#1f77b4", "dent": "#ff7f0e", "wear": "#2ca02c", "shell": "#d62728"}

# Parameter axes per class, matching analyze_results.py's figure 1 layout.
# Ranges are the sampled operating ranges in config.py.
AXES = {
    "crack": [("depth (mm)", config.CRACK_DEPTH_RANGE), ("L (mm)", config.CRACK_LENGTH_RANGE),
              ("theta (deg)", (-70, 90)), ("s0 (mm)", (105, 160))],
    "dent":  [("depth (mm)", config.DENT_DEPTH_RANGE), ("fw_y (mm)", config.DENT_FOOTPRINT_Y),
              ("fw_s (mm)", config.DENT_FOOTPRINT_S), ("s0 (mm)", (105, 160))],
    "wear":  [("depth (mm)", config.WEAR_DEPTH_RANGE),
              ("L (mm)", config.DEFECT_LENGTH_RANGE["wear"])],
    # shell's s0 spans only where shells are PLACED: SHELL_GAUGE_X_RANGE
    # confines them to the gauge corner, which is the physically realistic
    # location and a deliberate modelling choice. Drawing a shell target across
    # the whole arc would promise detection where the defect model never puts
    # one -- the restricted span is correct, not a limitation.
    "shell": [("depth (mm)", config.SHELL_DEPTH_RANGE), ("r_s (mm)", config.SHELL_RADIUS_RANGE),
              ("r_y (mm)", config.SHELL_RADIUS_RANGE), ("s0 (mm)", (147, 155))],
}
# Measured today, from the prelim analysis figure -- quoted so the s0 panels say
# what is actually being fixed. Only used for annotation, never as data.
MEASURED_S0 = {"crack": (0.03, 0.82), "dent": (0.17, 1.00)}


def phi(x):
    return 0.5 * (1 + torch.erf(torch.as_tensor(x, dtype=torch.float64) / np.sqrt(2))).numpy()


def phi_inv(p):
    p = torch.as_tensor(np.clip(p, 1e-9, 1 - 1e-9), dtype=torch.float64)
    return (np.sqrt(2) * torch.erfinv(2 * p - 1)).numpy()


def binormal_roc(auc, n=400):
    """ROC with EXACTLY this AUC, from the equal-variance binormal model.

    d' = sqrt(2) * Phi^-1(AUC); TPR(FPR) = Phi(Phi^-1(FPR) + d'). Stated openly
    because these curves are drawn, not measured -- the shape is an assumption,
    only the area is the specification.
    """
    d = np.sqrt(2) * phi_inv(auc)
    fpr = np.concatenate([[0], np.logspace(-4, 0, n)])
    return fpr, phi(phi_inv(fpr) + d)


def load_current(path):
    if not path.exists():
        return None
    a = json.loads(path.read_text(encoding="utf-8"))
    return {c: v for c, v in a.get("classes", {}).items()}, a.get("run", "?")


def footer(fig, cur_run, extra=""):
    commit = "?"
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, timeout=5).stdout.strip() or "?"
    except Exception:                                    # noqa: BLE001
        pass
    fig.text(0.5, 0.005,
             f"TARGET SPECIFICATION — drawn from the stated model, NOT measured or simulated. "
             f"Grey = measured today ({cur_run}).  {extra}"
             f"  Generated {datetime.now():%Y-%m-%d} at {commit} by rail3D/target_figures.py",
             ha="center", fontsize=7.5, color="#444")


# ---------------------------------------------------------------------------
def figure_performance(cur, cur_run, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # --- ROC ---------------------------------------------------------------
    for cls in CLASSES:
        f, t = binormal_roc(TARGET_AUC[cls])
        axes[0].plot(f, t, color=COL[cls], lw=2,
                     label=f"{cls} (AUC {TARGET_AUC[cls]:.3f})")
        if cur and cls in cur:
            f0, t0 = binormal_roc(cur[cls]["auc"])
            axes[0].plot(f0, t0, color=COL[cls], lw=1, ls=":", alpha=0.55,
                         label=f"  now {cur[cls]['auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], "k:", lw=0.6)
    axes[0].axvline(TARGET_FPR, color="grey", lw=0.8, ls="--")
    axes[0].annotate(f"operating point\n{TARGET_FPR:.0%} FPR", (TARGET_FPR, 0.55),
                     xytext=(0.26, 0.46), fontsize=7.5, color="grey",
                     arrowprops=dict(arrowstyle="->", color="grey", lw=0.8))
    axes[0].set(xlabel="false positive rate", ylabel="true positive rate",
                xlim=(0, 1), ylim=(0, 1.02),
                title="TARGET: per-class detection ROC\nevery class at or above today's wear/shell")
    axes[0].legend(fontsize=7.2, loc="lower right", ncol=2)

    # --- confusion ---------------------------------------------------------
    n = len(CLASSES)
    cm = np.zeros((n, n))
    for i, cls in enumerate(CLASSES):
        cm[i] = (1 - TARGET_DIAG[cls]) / (n - 1)
        cm[i, i] = TARGET_DIAG[cls]
    im = axes[1].imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            axes[1].text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                         color="white" if cm[i, j] > 0.5 else "black")
    axes[1].set(xticks=range(n), yticks=range(n), xlabel="predicted", ylabel="true",
                title="TARGET: class confusion (row-normalized)\ndiagonal ≥ 0.90, no systematic collapse")
    axes[1].set_xticklabels(CLASSES, rotation=45, fontsize=8)
    axes[1].set_yticklabels(CLASSES, fontsize=8)
    plt.colorbar(im, ax=axes[1], fraction=0.04)
    if cur:
        now = ", ".join(f"{c} {cur[c]['class_acc']:.2f}" for c in CLASSES if c in cur)
        axes[1].set_xlabel(f"predicted\nmeasured today: {now}", fontsize=8)

    # --- score distributions ----------------------------------------------
    rng = np.random.default_rng(0)
    thr = 0.02
    intact = np.abs(rng.normal(0, 0.005, 4000))
    axes[2].hist(intact, bins=np.linspace(0, 0.75, 70), alpha=0.6,
                 label="intact", density=True, color="#7f9fc4")
    centres = {"crack": 0.22, "dent": 0.30, "wear": 0.46, "shell": 0.36}
    for cls in CLASSES:
        s = np.clip(rng.normal(centres[cls], 0.055, 2000), 0.02, None)
        axes[2].hist(s, bins=np.linspace(0, 0.75, 70), histtype="step", lw=1.6,
                     label=cls, density=True, color=COL[cls])
    axes[2].axvline(thr, color="red", ls="--", lw=1, label=f"threshold {thr:.3f}")
    axes[2].set(xlabel="cos gap vs intact", ylabel="density", xlim=(0, 0.75),
                ylim=(0, 14),
                title="TARGET: score distributions\nno defect mass below the threshold")
    axes[2].annotate("intact peak clipped here\n(density ~90, tight at 0)",
                     (0.012, 13.2), xytext=(0.115, 11.4), fontsize=7,
                     color="#4a6a94",
                     arrowprops=dict(arrowstyle="->", color="#4a6a94", lw=0.8))
    axes[2].legend(fontsize=8, loc="upper right")

    fig.tight_layout(rect=(0, 0.035, 1, 1))
    footer(fig, cur_run, "ROC shape is binormal at the specified AUC.")
    fig.savefig(out, dpi=160)
    print("  ->", out)


# ---------------------------------------------------------------------------
def figure_parameters(cur, cur_run, out):
    ncol = max(len(v) for v in AXES.values())
    fig, axes = plt.subplots(len(CLASSES), ncol, figsize=(3.6 * ncol, 2.9 * len(CLASSES)),
                             squeeze=False)
    rng = np.random.default_rng(1)
    for r, cls in enumerate(CLASSES):
        specs = AXES[cls]
        for c in range(ncol):
            ax = axes[r][c]
            if c >= len(specs):
                ax.axis("off")
                continue
            label, (lo, hi) = specs[c]
            x = np.linspace(lo, hi, 7)
            y = np.clip(TARGET_RATE + rng.normal(0, 0.008, x.size), 0, 1)
            err = np.full_like(y, 0.025)
            ax.errorbar(x, y, yerr=err, marker="o", ms=5, lw=1.6, capsize=3,
                        color=COL[cls], label="target")
            ax.axhline(TARGET_RATE, color=COL[cls], ls=":", lw=0.9, alpha=0.6)
            if cur and cls in cur:
                ax.axhline(cur[cls]["recall"], color="grey", ls="--", lw=1.2,
                           label=f"today (class mean {cur[cls]['recall']:.2f})")
            if label.startswith("s0") and cls in MEASURED_S0:
                a, b = MEASURED_S0[cls]
                ax.annotate(f"today this axis runs {a:.2f} → {b:.2f}\n"
                            f"FLATNESS here is the milestone",
                            (0.5, 0.16), xycoords="axes fraction", ha="center",
                            fontsize=7.5, color="#b03030")
            ax.set(xlabel=label, ylim=(0, 1.05))
            if c == 0:
                ax.set_ylabel(f"{cls}\ndetection rate")
            ax.grid(alpha=0.25)
            if r == 0 and c == 0:
                ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("TARGET: detection rate vs defect parameter — flat and high everywhere "
                 "(a downward trend marks a regime the system misses)", fontsize=12)
    fig.tight_layout(rect=(0, 0.028, 1, 0.965))
    footer(fig, cur_run, "Target points are the specification plus 0.008 jitter for legibility.")
    fig.savefig(out, dpi=160)
    print("  ->", out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-current", action="store_true",
                    help="omit the measured overlay (NOT recommended: the overlay is "
                         "what keeps these readable as targets rather than results)")
    ap.add_argument("--analysis", default=str(config.GENERATED_DIR / "analysis.json"))
    args = ap.parse_args()

    from pathlib import Path
    cur, cur_run = None, "no analysis.json found"
    if not args.no_current:
        got = load_current(Path(args.analysis))
        if got:
            cur, cur_run = got
            print(f"measured overlay from {cur_run}")
        else:
            print(f"! {args.analysis} not found - targets will have no measured overlay")

    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    print("writing target figures:")
    figure_performance(cur, cur_run, config.FIGURE_DIR / "target_performance.png")
    figure_parameters(cur, cur_run, config.FIGURE_DIR / "target_parameters.png")
    print("\nThese are SPECIFICATIONS. Keep the footer and the grey measured overlay\n"
          "when they go into a deck -- see the module docstring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
