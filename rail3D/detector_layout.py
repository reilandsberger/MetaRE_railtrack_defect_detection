"""Where the trained detectors ended up -- a wide figure for slides.

    python detector_layout.py                         # the stage's SLM + no-MS runs
    python detector_layout.py --stage prelim --background intact
    python detector_layout.py --run-name ms3d_slm_l5_prelim_9d5878 --which latest

One panel per run, side by side, each the detector plane as the hardware sees
it: 150 x 75 mm, x across the rail (horizontal), y along it. Drawn on it:

  trained windows   the detectors that survived pruning (11.4 x 7.0 mm each,
                    numbered by x), at their TRAINED centres -- read straight
                    from the checkpoint (SoftDetector2D.centers())
  start lattice     the 13 x 10 = 130 windows training started from, faint
  background        --background signal (default): RMS over defect samples of
                    I_defect - mean I_intact at the detector plane, through the
                    run's own metasurface -- where the DEFECT information lands.
                    --background intact: mean intact intensity there -- where
                    the light lands. --background none: plain.

The light is rebuilt from the run's own dataset (the data_root its checkpoint
recorded, else the stage root) and its own trained metasurface, so a run
trained under an earlier horn placement is drawn with the light it was
trained on. Writes data/figures/detector_layout_<tag>.png (16:9-friendly) and
data/generated/detector_layout_<tag>.json (centres in mm, per run).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

from rail3d import config, data3d, train3d

LABEL = {"slm": "trained metasurface (SLM)", "none": "no metasurface (baseline)",
         "metaunit": "trained metasurface (meta-atoms)"}


def run_detectors(run: str, which: str, device) -> dict:
    """Trained centres + what the run was trained on, from its checkpoint."""
    cfg = train3d.TrainConfig(run_name=run)
    model, state = train3d.load_trained(cfg, device, which)
    eff = train3d.effective_config(cfg, state)
    hist = state.get("history") or {}
    be = hist.get("best_epoch")
    val = next((v for v in hist.get("val", []) if v.get("epoch") == be), None)
    c = model.detector.centers().detach().cpu().numpy()
    order = np.argsort(c[:, 0] + 1e-3 * c[:, 1])
    if c.shape[0] > eff.n_det_final:
        # a "best" saved before pruning finished (runs before README finding 22)
        print(f"  ! {run} {which}.pt has {c.shape[0]} detectors, above the budget of "
              f"{eff.n_det_final}: it predates pruning. Try --which latest.")
    return {"run": run, "model": model, "surface": eff.surface, "mode": eff.mode,
            "n_det_final": eff.n_det_final,
            "data_root": (state.get("config") or {}).get("data_root"),
            "epoch": state.get("epoch"), "best_epoch": be,
            "val_auc": None if val is None else val.get("auc"),
            "centres_mm": c[order], "n_det": int(c.shape[0])}


@torch.no_grad()
def background(r: dict, kind: str, root: Path | None, device, max_samples: int):
    """Normalised map at the detector plane through the run's own surface."""
    if kind == "none":
        return None
    root = Path(r["data_root"]) if r["data_root"] else root
    if root is None or not (root / "dataset_config.json").exists():
        print(f"  {r['run']}: no dataset root ({root}) -- background omitted")
        return None
    try:
        fields, _, _, rms = data3d.load_dataset(r["mode"], root=root)
        intact, _ = data3d.load_intact(r["mode"], rms=rms, root=root)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"  {r['run']}: background omitted ({exc})")
        return None
    gen = torch.Generator().manual_seed(0)
    pick = lambda f: f[torch.randperm(len(f), generator=gen)[:max_samples]]  # noqa: E731
    model = r["model"].to(device).eval()

    def intensity(f):
        return torch.cat([model.propagate(b.to(device)).abs().pow(2).cpu()
                          for b in torch.split(f, 256)])

    I_int = intensity(pick(intact)).mean(0)
    if kind == "intact":
        m = I_int
    else:
        I_def = intensity(pick(fields))
        m = (I_def - I_int).pow(2).mean(0).sqrt()
    return (m / m.max().clamp_min(1e-30)).numpy()


def draw(runs: list[dict], maps: list, kind: str, which: str, out_png: Path) -> None:
    X, Y = config.plane_grid("cpu")
    X, Y = X[0].numpy() + config.PLANE_X_CENTER, Y[0].numpy()
    dw, dh = config.DET_SIZE
    start = config.dense_detector_centers().numpy()
    n = len(runs)
    fig, axs = plt.subplots(1, n, figsize=(13.33, 4.3 if n > 1 else 6.4),
                            layout="constrained", squeeze=False)
    btitle = {"signal": "defect signal at the detector plane (RMS of I_defect - mean I_intact)",
              "intact": "mean intact intensity at the detector plane",
              "none": ""}[kind]
    for ax, r, m in zip(axs[0], runs, maps):
        if m is not None:
            im = ax.pcolormesh(X, Y, m, cmap="magma", vmin=0, vmax=1, shading="nearest")
            cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
            cb.set_label("normalised", fontsize=9)
            edge, txt = "#7fe3ff", "white"
        else:
            ax.set_facecolor("#f4f4f4")
            edge, txt = "#1f6fb4", "#1f3b57"
        for cx, cy in start:
            ax.add_patch(Rectangle((cx - dw / 2 + config.PLANE_X_CENTER, cy - dh / 2), dw, dh,
                                   fill=False, lw=0.4, ec="#9a9a9a", alpha=0.55))
        for k, (cx, cy) in enumerate(r["centres_mm"], start=1):
            x0 = cx + config.PLANE_X_CENTER
            ax.add_patch(Rectangle((x0 - dw / 2, cy - dh / 2), dw, dh, fill=False, lw=2.2,
                                   ec=edge))
            ax.text(x0, cy, str(k), ha="center", va="center", fontsize=9,
                    fontweight="bold", color=txt)
        ax.set(xlim=(-config.WX / 2 + config.PLANE_X_CENTER, config.WX / 2 + config.PLANE_X_CENTER),
               ylim=(-config.WY / 2, config.WY / 2),
               xlabel="x (mm) — across the rail", ylabel="y (mm) — along the rail")
        ax.set_aspect("equal")
        auc = f", val AUC {r['val_auc']:.3f}" if r["val_auc"] is not None and which == "best" else ""
        pre = (f"\nPRE-PRUNING: above the {r['n_det_final']}-detector budget"
               if r["n_det"] > r["n_det_final"] else "")
        ax.set_title(f"{LABEL.get(r['surface'], r['surface'])}\n{r['n_det']} trained detectors "
                     f"({which} checkpoint, epoch {r['best_epoch'] if which == 'best' else r['epoch']}"
                     f"{auc}){pre}", fontsize=11)
    z_det = config.H_MS + config.LAYER_DISTANCES[-1]
    fig.suptitle(f"Trained detector windows on the detector plane (z = {z_det:.0f} mm, "
                 f"{config.WX:.0f} × {config.WY:.0f} mm; windows {dw:.1f} × {dh:.1f} mm; "
                 f"grey: the {len(start)}-window start)"
                 + (f"\nbackground: {btitle}" if btitle else ""), fontsize=12)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="prelim", choices=sorted(config.STAGES))
    ap.add_argument("--run-name", nargs="*", default=None,
                    help="checkpoint run name(s); default: the stage's ms3d_slm_<tag> "
                         "and ms3d_none_<tag>, whichever exist")
    ap.add_argument("--which", default="best", choices=["best", "latest"])
    ap.add_argument("--background", default="signal", choices=["signal", "intact", "none"])
    ap.add_argument("--max-samples", type=int, default=1500,
                    help="samples per pool for the background (random, seed 0)")
    ap.add_argument("--profile", default="lab", choices=list(config.PROFILES))
    args = ap.parse_args()

    device = config.get_device(args.profile)
    root = data3d.stage_root(args.stage)
    if args.run_name:
        runs = args.run_name
    else:
        tag = data3d.run_tag(root)
        runs = [r for r in (f"ms3d_slm_{tag}", f"ms3d_none_{tag}")
                if (config.CHECKPOINT_DIR / r / f"{args.which}.pt").exists()]
    missing = [r for r in runs if not (config.CHECKPOINT_DIR / r / f"{args.which}.pt").exists()]
    if missing or not runs:
        print(f"!! no {args.which}.pt for: {', '.join(missing) or '(the stage default runs)'}")
        print(f"   checkpoints in {config.CHECKPOINT_DIR}:")
        for d in sorted(config.CHECKPOINT_DIR.glob("*")) if config.CHECKPOINT_DIR.exists() else []:
            print(f"     {d.name}")
        print("   pass --run-name <one of those>")
        return 1

    rs = [run_detectors(r, args.which, device) for r in runs]
    maps = [background(r, args.background, root, device, args.max_samples) for r in rs]
    tag = rs[0]["run"] if len(rs) == 1 else "_vs_".join(r["surface"] for r in rs) + \
        "_" + rs[0]["run"].split("_", 2)[-1]
    png = config.FIGURE_DIR / f"detector_layout_{tag}.png"
    draw(rs, maps, args.background, args.which, png)
    js = config.GENERATED_DIR / f"detector_layout_{tag}.json"
    js.write_text(json.dumps(data3d.stamp({
        "which": args.which, "background": args.background,
        "plane": {"z_mm": config.H_MS + config.LAYER_DISTANCES[-1], "x_center_mm": config.PLANE_X_CENTER,
                  "size_mm": [config.WX, config.WY], "window_mm": list(config.DET_SIZE)},
        "runs": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()
                  if k != "model"} for r in rs]}), indent=2, default=str), encoding="utf-8")
    for r in rs:
        print(f"  {r['run']} ({r['surface']}): {r['n_det']} detectors at "
              + ", ".join(f"({x:+.1f}, {y:+.1f})" for x, y in r["centres_mm"]) + " mm")
    print(f"  -> {png}\n  -> {js}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
