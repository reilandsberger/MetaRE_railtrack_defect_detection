"""Inspect a generated dataset (or a smoke run) before committing to a long job.

Ties each shard back to the geometry that produced it: for a few samples per
class it re-derives the defect parameters from the stored seed, renders the
depth field d(s, y), and shows it next to the fields actually stored in the
shard - plus the mean defect-minus-intact intensity, which is the signature the
metasurface has to work with.

Usage (from rail3D/):
    python inspect_dataset.py                     # smoke set, 2 samples/class
    python inspect_dataset.py --root data/generated --n 3
    python inspect_dataset.py --classes crack shell

Outputs console statistics plus:
    data/figures/dataset_review.png   geometry -> stored field, per class
    data/figures/dataset_meta.png     parameter distributions per class
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rail3d import config, data3d, mesh3d, sections


def summarize(psi: torch.Tensor, meta: list[dict], cls: str) -> None:
    """Console sanity block: shapes, amplitudes, and parameter ranges."""
    psi1, psi2 = psi[..., 0], psi[..., 1]
    print(f"\n=== {cls}: {psi.shape[0]} samples, psi{tuple(psi.shape[1:])} {psi.dtype}")
    print(f"    |psi1| mean {psi1.abs().mean():.4g}   |psi2| mean {psi2.abs().mean():.4g}"
          f"   (psi2/psi1 = {float(psi2.abs().mean() / psi1.abs().mean()):.1e})")
    finite = bool(torch.isfinite(psi.view(torch.float32)).all())
    print(f"    all finite: {finite}"
          f"   sample-to-sample variation: {float(psi1.abs().std(dim=0).mean()):.4g}")
    if not finite:
        print("    !! non-finite values present - generation is broken")

    keys = [k for k in ("depth", "L", "theta", "y0", "s0", "r_s", "r_y", "fw_y", "fw_s")
            if k in meta[0]]
    for k in keys:
        vals = np.array([float(m[k]) for m in meta])
        unit = "deg" if k == "theta" else "mm"
        if k == "theta":
            vals = np.degrees(vals)
        print(f"    {k:6s}: min {vals.min():7.2f}  mean {vals.mean():7.2f}  "
              f"max {vals.max():7.2f}  ({unit})")


def depth_field_for(meta_row: dict, section, geom, y_grid) -> np.ndarray:
    """Re-derive the depth field of a stored sample from its seed."""
    cls = meta_row["class"]
    defect = None
    if cls in config.DATASET_DIRS:
        files = sections.get_dataset_files(cls)
        defect = sections.match_reference_width(
            sections.load_vertices_from_csv(files[meta_row["csv_index"]]), section)
    params, _ = mesh3d.defect_params_for_sample(
        section, cls, defect, meta_row["seed"], n_arc=geom["n_arc"])
    return mesh3d.render_depth_field(params, geom, y_grid)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(config.GENERATED_DIR / "smoke"),
                    help="shard directory (default: the smoke set)")
    ap.add_argument("--n", type=int, default=2, help="samples shown per class")
    ap.add_argument("--classes", nargs="*", default=list(config.CLASS_NAMES))
    args = ap.parse_args()

    root = Path(args.root)
    config.ensure_dirs()
    print(f"inspecting {root}")

    psi0 = torch.load(data3d.psi0_path(root=root), map_location="cpu", weights_only=False)
    intact_psi, intact_meta = data3d.load_class_fields("intact", root=root)
    intact_tot = data3d.combine_field(intact_psi, "tot", psi0=psi0)
    summarize(intact_psi, intact_meta, "intact")
    intact_mean_I = (intact_tot.abs() ** 2).mean(dim=0)

    section = sections.load_reference_section()
    geom = mesh3d.arc_geometry(section, mesh3d.default_arc_count(section, 1.0))
    y_grid = np.arange(-config.SEG_LEN / 2, config.SEG_LEN / 2 + 0.5, 1.0)
    extent = (-config.WY / 2, config.WY / 2, -config.WX / 2, config.WX / 2)

    rows = []
    for cls in args.classes:
        psi, meta = data3d.load_class_fields(cls, root=root)
        summarize(psi, meta, cls)
        tot = data3d.combine_field(psi, "tot", psi0=psi0)
        rows.append((cls, psi, meta, tot))

    # ---- figure 1: geometry -> stored field ------------------------------
    ncol = args.n + 2
    fig, axes = plt.subplots(len(rows), ncol, figsize=(3.2 * ncol, 3.0 * len(rows)),
                             squeeze=False)
    for r, (cls, psi, meta, tot) in enumerate(rows):
        for c in range(args.n):
            d = depth_field_for(meta[c], section, geom, y_grid)
            ax = axes[r][c]
            im = ax.imshow(d, origin="lower", cmap="hot", aspect="auto",
                           extent=(geom["s"][0], geom["s"][-1], y_grid[0], y_grid[-1]))
            plt.colorbar(im, ax=ax, fraction=0.04)
            ax.set_title(f"{cls} #{c}: d(s,y), max {d.max():.1f} mm", fontsize=8)
            ax.set_xlabel("s (mm)", fontsize=7)
            ax.set_ylabel("y (mm)", fontsize=7)

        ax = axes[r][args.n]
        im = ax.imshow((tot[0].abs() ** 2).numpy(), origin="lower", cmap="inferno",
                       extent=extent, aspect="equal")
        plt.colorbar(im, ax=ax, fraction=0.03)
        ax.set_title(f"{cls} #0: stored |psi_tot|^2", fontsize=8)

        ax = axes[r][args.n + 1]
        diff = ((tot.abs() ** 2).mean(dim=0) - intact_mean_I).numpy()
        lim = np.abs(diff).max()
        im = ax.imshow(diff, origin="lower", cmap="bwr", vmin=-lim, vmax=lim,
                       extent=extent, aspect="equal")
        plt.colorbar(im, ax=ax, fraction=0.03)
        rel = lim / float(intact_mean_I.max())
        ax.set_title(f"{cls}: mean - intact ({100 * rel:.1f}% of peak)", fontsize=8)

    fig.suptitle(f"Dataset review: reconstructed geometry vs stored fields ({root.name})")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p1 = config.FIGURE_DIR / "dataset_review.png"
    fig.savefig(p1, dpi=170)
    print(f"\nwrote {p1}")

    # ---- figure 2: parameter distributions -------------------------------
    keys = ["depth", "y0", "L", "theta"]
    fig2, axes2 = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3.2))
    for ax, k in zip(axes2, keys):
        for cls, psi, meta, tot in rows:
            vals = [float(m[k]) for m in meta if k in m]
            if not vals:
                continue
            if k == "theta":
                vals = np.degrees(vals)
            ax.hist(vals, bins=12, alpha=0.55, label=cls)
        ax.set_title(f"{k} ({'deg' if k == 'theta' else 'mm'})", fontsize=9)
        ax.legend(fontsize=7)
    fig2.suptitle("Sampled defect parameters (check ranges match README section 2)")
    fig2.tight_layout(rect=(0, 0, 1, 0.94))
    p2 = config.FIGURE_DIR / "dataset_meta.png"
    fig2.savefig(p2, dpi=170)
    print(f"wrote {p2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
