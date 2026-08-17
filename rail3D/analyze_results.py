"""Where does the trained metasurface succeed, and where does it fail?

Aggregate AUC tells you how well the system does; this tells you WHICH defects
it misses. Every test sample's outcome is joined to the defect parameters stored
in its shard metadata (depth, length, orientation, footprint, across-head
position), so failures can be attributed to physical causes rather than left as
a single number.

Usage (from rail3D/, after training):
    python analyze_results.py                          # best checkpoint of ms3d_slm_v1
    python analyze_results.py --run-name ms3d_metaunit_v1 --surface metaunit
    python analyze_results.py --data-root data/generated/smoke

Outputs:
    data/figures/analysis_parameters.png   detection rate vs each defect parameter
    data/figures/analysis_performance.png  per-class ROC, confusion, score spread
    data/figures/analysis_failures.png     the worst misses, as depth fields
    data/generated/analysis.json           every number behind those figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rail3d import config, data3d, losses3d, mesh3d, sections, train3d

# parameters worth breaking performance down by, per class
PARAMS = {
    "crack": ["depth", "L", "theta", "s0"],
    "dent": ["depth", "fw_y", "fw_s", "s0"],
    "wear": ["depth", "L"],
    "shell": ["depth", "r_s", "r_y", "s0"],
}
UNITS = {"theta": "deg"}


def point_biserial(x: np.ndarray, hit: np.ndarray) -> float:
    """Correlation between a continuous parameter and the hit/miss indicator."""
    if hit.std() == 0 or x.std() == 0:
        return 0.0
    return float(np.corrcoef(x, hit.astype(float))[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="ms3d_slm_v1")
    ap.add_argument("--surface", default="slm")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--which", default="best", choices=["best", "latest"])
    ap.add_argument("--n-fail", type=int, default=8, help="failure montage size")
    ap.add_argument("--profile", default="lab")
    args = ap.parse_args()

    device = config.get_device(args.profile)
    cfg = train3d.TrainConfig(run_name=args.run_name, surface=args.surface,
                              data_root=args.data_root)
    model, state = train3d.load_trained(cfg, device, args.which)
    # score with the config the run was TRAINED with (objective/metric/fpr),
    # not the CLI defaults — a margin run scored as rank/cos reads wrong
    cfg = train3d.effective_config(cfg, state)
    data = train3d.load_all_data(cfg, device)
    metric = cfg.metric if cfg.objective == "rank" else "l2"
    root = Path(cfg.data_root) if cfg.data_root else None

    # --- score the test split -------------------------------------------
    det = train3d._hard_dets(model, data["fields"], data["test"])
    det0 = train3d._hard_dets(model, data["intact"], data["i_test"])
    det0_val = train3d._hard_dets(model, data["intact"], data["i_val"])
    d_ref = det0.mean(dim=0)
    gap = losses3d.gap_score(det, d_ref, metric)
    gap0 = losses3d.gap_score(det0, d_ref, metric)
    thr = losses3d.calibrated_threshold(
        losses3d.gap_score(det0_val, det0_val.mean(dim=0), metric), cfg.target_fpr)

    labels = data["labels"][data["test"]]
    with torch.no_grad():
        logits = model.classify(train3d._classify_input(det, d_ref, cfg.objective))
    pred = logits.argmax(dim=1)

    detected = (gap > thr).cpu().numpy()
    correct = (pred == labels).cpu().numpy()
    gap_np = gap.cpu().numpy()
    lab_np = labels.cpu().numpy()

    # --- join to the defect parameters ----------------------------------
    metas: list[dict] = []
    for cls in config.CLASS_NAMES:
        metas.extend(data3d.load_meta(cls, root=root))
    test_idx = data["test"].cpu().numpy()
    meta_test = [metas[i] for i in test_idx]

    print(f"[rail3d] {args.run_name} ({args.which}) on {device}")
    print(f"  test samples {len(gap_np)}, intact {len(gap0)}, "
          f"threshold {thr:.4f} at target FPR {cfg.target_fpr}")
    print(f"  overall: AUC {losses3d.auc_score(gap, gap0):.3f}  "
          f"recall {detected.mean():.3f}  "
          f"false alarm {(gap0 > thr).float().mean():.3f}  "
          f"class acc {correct.mean():.3f}\n")

    out = {"run": args.run_name, "threshold": thr, "classes": {}}
    print(f"{'class':7s} {'n':>5s} {'recall':>7s} {'clsacc':>7s} {'AUC':>6s}   "
          f"strongest failure predictors")
    for c, cls in enumerate(config.CLASS_NAMES):
        sel = lab_np == c
        if not sel.any():
            continue
        rec = detected[sel].mean()
        acc = correct[sel].mean()
        auc = losses3d.auc_score(gap[torch.from_numpy(sel).to(gap.device)], gap0)
        corrs = []
        for p in PARAMS.get(cls, ["depth"]):
            vals = np.array([m.get(p, np.nan) for m in
                             (meta_test[i] for i in np.flatnonzero(sel))], dtype=float)
            if np.isnan(vals).all():
                continue
            if p == "theta":
                vals = np.degrees(vals)
            corrs.append((p, point_biserial(vals, detected[sel])))
        corrs.sort(key=lambda t: -abs(t[1]))
        txt = "  ".join(f"{p}:{r:+.2f}" for p, r in corrs[:3])
        print(f"{cls:7s} {sel.sum():5d} {rec:7.3f} {acc:7.3f} {auc:6.3f}   {txt}")
        out["classes"][cls] = {"n": int(sel.sum()), "recall": float(rec),
                               "class_acc": float(acc), "auc": auc,
                               "param_corr": {p: r for p, r in corrs}}

    # --- figure 1: detection rate vs each parameter ----------------------
    ncol = max(len(v) for v in PARAMS.values())
    fig, axes = plt.subplots(len(config.CLASS_NAMES), ncol,
                             figsize=(3.6 * ncol, 2.9 * len(config.CLASS_NAMES)),
                             squeeze=False)
    for r, cls in enumerate(config.CLASS_NAMES):
        sel = np.flatnonzero(lab_np == list(config.CLASS_NAMES).index(cls))
        for cix in range(ncol):
            ax = axes[r][cix]
            plist = PARAMS.get(cls, [])
            if cix >= len(plist) or len(sel) == 0:
                ax.axis("off")
                continue
            p = plist[cix]
            vals = np.array([meta_test[i].get(p, np.nan) for i in sel], dtype=float)
            if np.isnan(vals).all():
                ax.axis("off")
                continue
            if p == "theta":
                vals = np.degrees(vals)
            hit = detected[sel]
            edges = np.quantile(vals, np.linspace(0, 1, 7))
            edges = np.unique(edges)
            centers, rates, errs, ns = [], [], [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (vals >= lo) & (vals <= hi)
                if m.sum() < 3:
                    continue
                pr = hit[m].mean()
                centers.append(0.5 * (lo + hi))
                rates.append(pr)
                errs.append(np.sqrt(max(pr * (1 - pr), 1e-9) / m.sum()))
                ns.append(int(m.sum()))
            if centers:
                ax.errorbar(centers, rates, yerr=errs, fmt="o-", capsize=3)
            ax.axhline(hit.mean(), color="gray", ls=":", lw=0.8)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel(f"{p} ({UNITS.get(p, 'mm')})", fontsize=8)
            if cix == 0:
                ax.set_ylabel(f"{cls}\ndetection rate", fontsize=9)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3)
    fig.suptitle("Detection rate vs defect parameter (dotted = class mean); "
                 "a downward trend marks the regime the system misses")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p1 = config.FIGURE_DIR / "analysis_parameters.png"
    fig.savefig(p1, dpi=160)

    # --- figure 2: ROC, confusion, score distributions -------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for c, cls in enumerate(config.CLASS_NAMES):
        sel = torch.from_numpy(lab_np == c).to(gap.device)
        if not bool(sel.any()):
            continue
        fpr, tpr = losses3d.roc_points(gap[sel], gap0)
        auc = losses3d.auc_score(gap[sel], gap0)
        axes[0].plot(fpr, tpr, label=f"{cls} (AUC {auc:.3f})")
    axes[0].plot([0, 1], [0, 1], "k:", lw=0.6)
    axes[0].set(xlabel="false positive rate", ylabel="true positive rate",
                title="Per-class detection ROC")
    axes[0].legend(fontsize=8)

    cm = losses3d.confusion_matrix(pred, labels).numpy()
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    im = axes[1].imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[1].text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                         fontsize=8, color="white" if cmn[i, j] > 0.5 else "black")
    axes[1].set(xticks=range(len(config.CLASS_NAMES)), yticks=range(len(config.CLASS_NAMES)),
                xlabel="predicted", ylabel="true", title="Class confusion (row-normalized)")
    axes[1].set_xticklabels(config.CLASS_NAMES, rotation=45, fontsize=8)
    axes[1].set_yticklabels(config.CLASS_NAMES, fontsize=8)
    plt.colorbar(im, ax=axes[1], fraction=0.04)

    bins = np.linspace(0, float(max(gap.max(), gap0.max())) * 1.05, 40)
    axes[2].hist(gap0.cpu().numpy(), bins=bins, alpha=0.6, label="intact", density=True)
    for c, cls in enumerate(config.CLASS_NAMES):
        s = lab_np == c
        if s.any():
            axes[2].hist(gap_np[s], bins=bins, histtype="step", lw=1.4,
                         label=cls, density=True)
    axes[2].axvline(thr, color="red", ls="--", lw=1, label=f"threshold {thr:.3f}")
    axes[2].set(xlabel=f"{metric} gap vs intact", ylabel="density",
                title="Score distributions")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    p2 = config.FIGURE_DIR / "analysis_performance.png"
    fig.savefig(p2, dpi=160)

    # --- figure 3: the worst misses, as depth fields ---------------------
    missed = np.flatnonzero(~detected)
    order = missed[np.argsort(gap_np[missed])][: args.n_fail]
    if len(order):
        section = sections.load_reference_section()
        geom = mesh3d.arc_geometry(section, mesh3d.default_arc_count(section, 1.0))
        y_grid = np.arange(-config.SEG_LEN / 2, config.SEG_LEN / 2 + 0.5, 1.0)
        ncols = min(4, len(order))
        nrows = int(np.ceil(len(order) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.0 * nrows),
                                 squeeze=False)
        for k, idx in enumerate(order):
            ax = axes[k // ncols][k % ncols]
            m = meta_test[idx]
            try:
                d = _depth_field(m, section, geom, y_grid)
                ax.imshow(d, origin="lower", cmap="hot", aspect="auto",
                          extent=(geom["s"][0], geom["s"][-1], y_grid[0], y_grid[-1]))
            except Exception:  # noqa: BLE001
                ax.axis("off")
            bits = [f"{m['class']}", f"gap {gap_np[idx]:.3f}"]
            if "depth" in m:
                bits.append(f"D={float(m['depth']):.1f}mm")
            if "theta" in m:
                bits.append(f"{np.degrees(float(m['theta'])):.0f}deg")
            ax.set_title("  ".join(bits), fontsize=8)
            ax.tick_params(labelsize=6)
        for k in range(len(order), nrows * ncols):
            axes[k // ncols][k % ncols].axis("off")
        fig.suptitle(f"Worst {len(order)} missed detections (threshold {thr:.3f})")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        p3 = config.FIGURE_DIR / "analysis_failures.png"
        fig.savefig(p3, dpi=160)
        print(f"\nwrote {p3}")

    outp = config.GENERATED_DIR / "analysis.json"
    outp.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"wrote {p1}\nwrote {p2}\nwrote {outp}")
    return 0


def _depth_field(meta_row: dict, section, geom, y_grid) -> np.ndarray:
    cls = meta_row["class"]
    defect = None
    if cls in config.DATASET_DIRS:
        files = sections.get_dataset_files(cls)
        defect = sections.match_reference_width(
            sections.load_vertices_from_csv(files[meta_row["csv_index"]]), section)
    params, _ = mesh3d.defect_params_for_sample(
        section, cls, defect, meta_row["seed"], n_arc=geom["n_arc"])
    return mesh3d.render_depth_field(params, geom, y_grid)


if __name__ == "__main__":
    raise SystemExit(main())
