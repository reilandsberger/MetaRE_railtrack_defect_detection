"""Train from a DENSE detector grid down to several final counts, and compare.

Motivation: the detector layout is a design choice with a real hardware cost —
each detector is a waveguide/receiver. Starting dense (windows tiling most of
the measurement plane) and pruning by redundancy — each detector valued by its
UNIQUE contribution — answers "where do the informative spots actually sit",
while sweeping the final count answers "how few receivers can we get away
with". (--prune-criterion variance restores the Face3D ranking, which is only
valid for sparse layouts.)

Detectors act on the STORED fields, so this needs no regeneration — only
training minutes. Optionally also sweeps the metasurface-to-detector distance,
which is likewise a training-time propagation.

Usage (from rail3D/):
    python sweep_detectors.py                          # counts 4,6,8,10
    python sweep_detectors.py --counts 4 8 --dist 120 160 200
    python sweep_detectors.py --grid 13x10 --epochs 400

Writes data/figures/detector_sweep.png and data/generated/detector_sweep.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

from rail3d import config, train3d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", type=int, nargs="*", default=[4, 6, 8, 10])
    ap.add_argument("--dist", type=float, nargs="*", default=None,
                    help="MS->detector distances to sweep (default: config value)")
    ap.add_argument("--grid", default=None,
                    help="dense starting grid, e.g. 13x10 (default: config.DET_GRID)")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--profile", default="lab")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--surface", default="slm")
    ap.add_argument("--prune-criterion", default="redundancy",
                    choices=["redundancy", "variance"],
                    help="'variance' is the Face3D criterion, only valid for sparse layouts")
    args = ap.parse_args()

    device = config.get_device(args.profile)
    grid = (tuple(int(v) for v in args.grid.split("x"))
            if args.grid else config.DET_GRID)
    n_start = grid[0] * grid[1]
    dists = args.dist or [config.LAYER_DISTANCES[-1]]

    cov = n_start * config.DET_SIZE[0] * config.DET_SIZE[1] / (config.WX * config.WY)
    print(f"[rail3d] detector sweep on {device}")
    print(f"  dense start {grid[0]}x{grid[1]} = {n_start} detectors "
          f"({100 * cov:.0f}% nominal plane coverage), pruning to {args.counts}")
    print(f"  MS->detector distances: {dists}\n")

    rows = []
    for dist in dists:
        for n_final in args.counts:
            name = f"sweep_d{dist:.0f}_n{n_final}"
            cfg = train3d.TrainConfig(
                run_name=name, surface=args.surface, n_epoch=args.epochs,
                batch_size=config.PROFILES[args.profile].train_batch,
                det_grid=grid, n_det_final=n_final,
                prune_schedule="fraction", prune_keep=0.75,
                prune_start=40, prune_end=args.epochs // 2, prune_every=5,
                prune_criterion=args.prune_criterion,
                layer_distances=(dist,),
                data_root=args.data_root,
            )
            t0 = time.time()
            train3d.train(cfg, device=device, verbose=False)
            model, _ = train3d.load_trained(cfg, device, "best")
            data = train3d.load_all_data(cfg, device)
            ev = train3d.full_evaluation(model, data, cfg)
            # how complementary are the surviving detectors? mean |off-diagonal
            # correlation| of their barcodes: high means pruning kept duplicates
            det_te = train3d._hard_dets(model, data["fields"], data["test"])
            dn = det_te - det_te.mean(dim=0, keepdim=True)
            sd = dn.std(dim=0).clamp_min(1e-12)
            corr = ((dn.T @ dn) / (dn.shape[0] * sd[:, None] * sd[None, :])).abs()
            n_d = corr.shape[0]
            off = (corr.sum() - corr.diagonal().sum()) / max(n_d * (n_d - 1), 1)
            with torch.no_grad():
                inten = model.propagate(data["fields"][data["test"][:256]]).abs() ** 2
                dets = model.detector.hard_powers(inten)
                cap = float((dets.sum(dim=1) /
                             ((model.detector.dx ** 2) * inten.sum(dim=(1, 2)))).mean())
            row = {
                "dist": dist, "n_final": n_final,
                "n_det_actual": int(model.detector.n_det),
                "mean_abs_corr": float(off),
                "min_separation_mm": model.detector.min_separation(),
                "capture_fraction": cap,
                "prune_criterion": args.prune_criterion,
                "auc": ev["test"]["auc"],
                "class_acc": ev["test"]["class_acc"],
                "pass_rate": ev["test"]["pass_rate"],
                "false_alarm": ev["test"]["false_alarm"],
                "balanced_acc": ev["test"]["balanced_acc"],
                "roc_auc": ev["roc"]["auc"],
                "tpr_at_1pct": ev["roc"]["tpr_at_1pct_fpr"],
                "per_class_auc": ev["roc"]["per_class_auc"],
                "centers": model.detector.centers().detach().cpu().tolist(),
                "minutes": (time.time() - t0) / 60,
            }
            rows.append(row)
            pc = "  ".join(f"{k[:5]} {v:.3f}" for k, v in row["per_class_auc"].items())
            print(f"  d={dist:4.0f} n={n_final:2d} -> AUC {row['auc']:.3f}  "
                  f"acc {row['class_acc']:.3f}  bal {row['balanced_acc']:.3f}  "
                  f"TPR@1% {row['tpr_at_1pct']:.3f}  [{pc}]  ({row['minutes']:.1f} min)")

    # ---- figure -----------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.2))
    for dist in dists:
        sel = [r for r in rows if r["dist"] == dist]
        xs = [r["n_final"] for r in sel]
        axes[0].plot(xs, [r["auc"] for r in sel], "o-", label=f"d={dist:.0f} mm")
        axes[1].plot(xs, [r["class_acc"] for r in sel], "s-", label=f"d={dist:.0f} mm")
    axes[0].set(xlabel="final detector count", ylabel="detection AUC",
                title="Detection vs detector count")
    axes[1].axhline(1 / len(config.CLASS_NAMES), color="gray", ls=":", lw=0.8)
    axes[1].set(xlabel="final detector count", ylabel="class accuracy",
                title=f"{len(config.CLASS_NAMES)}-class accuracy (dotted = chance)")
    for a in axes[:2]:
        a.legend(fontsize=8)
        a.grid(alpha=0.3)

    # redundancy + throughput: is pruning selecting COMPLEMENTARY detectors?
    ax = axes[2]
    for dist in dists:
        sel = [r for r in rows if r["dist"] == dist]
        xs = [r["n_final"] for r in sel]
        ax.plot(xs, [r["mean_abs_corr"] for r in sel], "o-", label=f"mean |corr| d={dist:.0f}")
        ax.plot(xs, [r["capture_fraction"] for r in sel], "s--",
                label=f"captured power d={dist:.0f}")
    ax.set(xlabel="final detector count", ylim=(0, 1),
           title=f"redundancy & throughput ({args.prune_criterion})")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    best = max(rows, key=lambda r: r["auc"] + r["class_acc"])
    ax = axes[3]
    ax.add_patch(Rectangle((-config.WY / 2, -config.WX / 2), config.WY, config.WX,
                           fill=False, edgecolor="black", lw=1))
    dw, dh = config.DET_SIZE
    for cx, cy in best["centers"]:
        ax.add_patch(Rectangle((cy - dh / 2, cx - dw / 2), dh, dw,
                               fill=False, edgecolor="tab:green", lw=1.4))
    ax.set(xlim=(-config.WY / 2 - 10, config.WY / 2 + 10),
           ylim=(-config.WX / 2 - 10, config.WX / 2 + 10),
           xlabel="y (mm)", ylabel="x (mm)",
           title=f"kept detectors: n={best['n_final']}, d={best['dist']:.0f} mm")
    ax.set_aspect("equal")
    fig.suptitle(f"Detector sweep: dense {grid[0]}x{grid[1]} start pruned to "
                 f"{min(args.counts)}-{max(args.counts)}")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = config.FIGURE_DIR / "detector_sweep.png"
    fig.savefig(p, dpi=170)

    out = config.GENERATED_DIR / "detector_sweep.json"
    out.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(f"\nbest: n={best['n_final']} at d={best['dist']:.0f} mm  "
          f"AUC {best['auc']:.3f}  class acc {best['class_acc']:.3f}")
    print(f"wrote {p}\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
