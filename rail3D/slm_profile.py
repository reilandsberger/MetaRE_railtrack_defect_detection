"""Show the trained metasurface: wrapped phase, amplitude, and the light on it.

    python slm_profile.py                                  # the stage's SLM run
    python slm_profile.py --stage prelim --delta           # + change from init
    python slm_profile.py --run-name abl_slm_shipped_l5_prelim_9d5878 \\
                                     abl_slm_flat_l5_prelim_9d5878  # compare runs

Writes data/figures/slm_profile_<run>.png (one row per run and layer) and
data/generated/slm_profile_<run>.npz (x, y, wrapped phase, amplitude) so the
mask can be handed to a full-wave solver or a fabrication flow as-is.

Panels, left to right:

  Phase      the trained phase, wrapped to (-pi, pi] on a cyclic colormap. The
             raw parameter is unbounded, and wrapping is the only physically
             meaningful view: a pixel at 7.0 rad and one at 0.72 rad are the
             same pixel.
  Amplitude  the transmission |t| the metasurface applies. **surface="slm" is
             PHASE-ONLY** (SLM2D.forward = signal * exp(1j*phase)), so this is
             exactly 1.0 everywhere by construction, not a result. A structured
             amplitude map needs the meta-atom model (surface="metaunit"), which
             is blocked at lambda=5 until a 60 GHz library is fitted. The panel
             is drawn anyway, on the same 0.5-1.0 scale as a meta-atom plot, so
             the two remain comparable once that library exists.
  Incident   RMS |psi| of the intact fields arriving AT the metasurface plane,
             normalized to its peak. Not a property of the mask -- it is the
             light the mask has to work with. Phase in a dark pixel does nothing,
             so read the phase panel through this one. Needs the dataset root;
             skipped with a note when it is absent.
  Delta      (--delta) circular phase change from the epoch-0 mask, which is
             re-created exactly from the stored seed and init_std. Directly
             relevant to README finding 24: if a trained mask sits near its
             random-diffuser initialisation, training barely moved it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from rail3d import config, data3d, optics3d, train3d

TWO_PI = 2 * np.pi
# Checkpoints written before TrainConfig.slm_init_std existed were all trained
# from the pi/2 diffuser. Pinned here rather than taken from the dataclass
# default, so a future change of that default cannot silently re-seed the
# reconstructed initialisation for old runs.
LEGACY_INIT_STD = float(np.pi / 2)


def wrap(phase: torch.Tensor) -> torch.Tensor:
    """Wrap to (-pi, pi]."""
    return torch.remainder(phase + np.pi, TWO_PI) - np.pi


def circ_rms(d: torch.Tensor, w: torch.Tensor | None = None) -> float:
    """RMS of a phase difference measured on the circle (optionally weighted)."""
    d = wrap(d)
    if w is None:
        return float(d.pow(2).mean().sqrt())
    w = w / w.sum()
    return float((w * d.pow(2)).sum().sqrt())


def layer_maps(model: optics3d.ONN3D) -> list[dict]:
    """Phase and amplitude of every metasurface layer, whatever its type."""
    out = []
    for i, layer in enumerate(model.layers):
        if isinstance(layer, optics3d.SLM2D):
            ph = layer.phase.detach().cpu()
            out.append({"layer": i, "kind": "slm", "phase_raw": ph,
                        "amp": torch.ones_like(ph)})
        elif isinstance(layer, optics3d.MetaUnitSoft):
            with torch.no_grad():
                out.append({"layer": i, "kind": "metaunit",
                            "phase_raw": layer.phase().cpu(), "amp": layer.amp().cpu()})
        # nn.Identity (surface="none") has no mask to show
    return out


def initial_phase(state: dict, cfg: train3d.TrainConfig, layer: int) -> torch.Tensor:
    """The exact epoch-0 SLM phase, rebuilt from the run's own seed and init."""
    stored = state.get("config") or {}
    init_std = float(stored.get("slm_init_std", LEGACY_INIT_STD))
    return optics3d.SLM2D(seed=cfg.seed + layer, init_std=init_std).phase.detach()


def incident_amplitude(root: Path | None) -> torch.Tensor | None:
    """RMS |psi_tot| of the intact pool at the metasurface plane, peak = 1."""
    if root is None or not (Path(root) / "dataset_config.json").exists():
        return None
    try:
        psi, _ = data3d.load_class_fields("intact", root=Path(root))
        field = data3d.combine_field(psi, "tot", root=Path(root))
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"  incident panel skipped: {exc}")
        return None
    rms = field.abs().pow(2).mean(dim=0).sqrt()
    return rms / rms.max().clamp_min(1e-30)


def plot_slm_profile(run_names, which: str = "best", data_root=None,
                     delta: bool = False, out: Path | None = None,
                     device: str = "cpu") -> Path:
    """Draw every metasurface layer of every run; returns the PNG path."""
    if isinstance(run_names, str):
        run_names = [run_names]
    root = Path(data_root) if data_root else None
    inc = incident_amplitude(root)
    if inc is None:
        print("  (no dataset root with intact shards: incident-light panel omitted)")

    X, Y = config.plane_grid("cpu")
    X, Y = X[0].numpy(), Y[0].numpy()               # (NX, NY), indexing="ij"

    rows = []
    for rn in run_names:
        cfg = train3d.TrainConfig(run_name=rn, data_root=str(root) if root else None)
        model, state = train3d.load_trained(cfg, torch.device(device), which)
        eff = train3d.effective_config(cfg, state)
        maps = layer_maps(model)
        if not maps:
            print(f"  {rn}: surface={eff.surface!r} has no metasurface layer to show")
            continue
        for m in maps:
            m["run"], m["epoch"], m["surface"] = rn, state.get("epoch"), eff.surface
            m["phase"] = wrap(m["phase_raw"])
            if m["kind"] == "slm":
                ph0 = initial_phase(state, eff, m["layer"])
                m["delta"] = wrap(m["phase_raw"] - ph0)
                m["moved_rms"] = circ_rms(m["phase_raw"] - ph0)
                m["init_rms"] = circ_rms(ph0)
                if inc is not None:
                    m["moved_rms_lit"] = circ_rms(m["phase_raw"] - ph0, inc.pow(2))
            rows.append(m)
    if not rows:
        raise SystemExit("nothing to draw: no run had a metasurface layer")

    cols = ["phase", "amp"] + (["inc"] if inc is not None else []) + \
           (["delta"] if delta else [])
    # One subfigure per row: its title carries the run label, and constrained
    # layout sizes panels around it. A long label placed with ax.annotate is
    # counted by tight_layout and shrank every panel to a sliver.
    fig = plt.figure(figsize=(4.4 * len(cols), 2.9 * len(rows)), layout="constrained")
    subs = np.atleast_1d(fig.subfigures(len(rows), 1))
    for r, m in enumerate(rows):
        axs = subs[r].subplots(1, len(cols), squeeze=False)[0]
        for c, key in enumerate(cols):
            ax = axs[c]
            if key == "phase":
                im = ax.pcolormesh(X, Y, m["phase"].numpy(), cmap="twilight",
                                   vmin=-np.pi, vmax=np.pi, shading="nearest")
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
                cb.set_ticks([-3, -2, -1, 0, 1, 2, 3])
                ax.set_title("Phase (rad, wrapped)")
            elif key == "amp":
                im = ax.pcolormesh(X, Y, m["amp"].numpy(), cmap="viridis",
                                   vmin=0.5, vmax=1.0, shading="nearest")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
                ax.set_title("Amplitude |t|")
                if m["kind"] == "slm":
                    ax.text(0.5, 0.5, "phase-only SLM:\n|t| = 1 by construction",
                            transform=ax.transAxes, ha="center", va="center",
                            fontsize=8, color="#333",
                            bbox=dict(boxstyle="round", fc="white", alpha=0.75))
            elif key == "inc":
                im = ax.pcolormesh(X, Y, inc.numpy(), cmap="viridis",
                                   vmin=0, vmax=1, shading="nearest")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
                ax.set_title("Incident |psi| on MS (intact RMS)")
            elif key == "delta":
                if "delta" in m:
                    im = ax.pcolormesh(X, Y, m["delta"].numpy(), cmap="twilight",
                                       vmin=-np.pi, vmax=np.pi, shading="nearest")
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
                    ax.set_title("Change from epoch-0 phase (rad)")
                else:
                    ax.axis("off")
            ax.set_aspect("equal")
            ax.set_xlabel("x (mm)")
            if c == 0:
                ax.set_ylabel("y (mm)")

        stat = ""
        if "moved_rms" in m:
            stat = (f"   moved {m['moved_rms']:.2f} rad RMS from init "
                    f"(init spread {m['init_rms']:.2f})")
            if "moved_rms_lit" in m:
                stat += f", {m['moved_rms_lit']:.2f} rad where lit"
        subs[r].suptitle(
            f"{m['run']}   [{which} checkpoint, epoch {m['epoch']}, layer {m['layer']}, "
            f"surface={m['surface']}]" + (f"\n{stat.strip()}" if stat else ""),
            fontsize=9.5, x=0.01, ha="left")

    tag = rows[0]["run"] if len(run_names) == 1 else f"{len(run_names)}runs"
    out = out or config.FIGURE_DIR / f"slm_profile_{tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)

    # the mask itself, in the coordinates it was trained on
    for m in rows:
        npz = config.GENERATED_DIR / f"slm_profile_{m['run']}_L{m['layer']}.npz"
        np.savez(npz, x_mm=X[:, 0], y_mm=Y[0, :], phase_wrapped=m["phase"].numpy(),
                 amplitude=m["amp"].numpy(), run=m["run"], which=which,
                 epoch=m["epoch"] if m["epoch"] is not None else -1)
    for m in rows:
        line = f"  {m['run']} L{m['layer']}: phase std {float(m['phase'].std()):.2f} rad"
        if "moved_rms" in m:
            line += (f", moved {m['moved_rms']:.2f} rad RMS from init "
                     f"(init {m['init_rms']:.2f})")
            if "moved_rms_lit" in m:
                line += f", {m['moved_rms_lit']:.2f} where lit"
        print(line)
    print(f"  -> {out}")
    return out


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")            # CLI only: importing must not change a notebook's backend
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="prelim", choices=sorted(config.STAGES))
    ap.add_argument("--run-name", nargs="*", default=None,
                    help="checkpoint run name(s); default ms3d_slm_<stage run tag>")
    ap.add_argument("--which", default="best", choices=["best", "latest"])
    ap.add_argument("--data-root", default=None,
                    help="dataset root for the incident-light panel; "
                         "default: the stage's root")
    ap.add_argument("--delta", action="store_true",
                    help="add the change-from-initialisation panel")
    args = ap.parse_args()

    root = Path(args.data_root) if args.data_root else data3d.stage_root(args.stage)
    runs = args.run_name or [f"ms3d_slm_{data3d.run_tag(root)}"]
    missing = [r for r in runs
               if not (config.CHECKPOINT_DIR / r / f"{args.which}.pt").exists()]
    if missing:
        print(f"!! no {args.which}.pt for: {', '.join(missing)}")
        print(f"   checkpoints present in {config.CHECKPOINT_DIR}:")
        for d in sorted(config.CHECKPOINT_DIR.glob("*")) if config.CHECKPOINT_DIR.exists() else []:
            print(f"     {d.name}")
        return 1
    plot_slm_profile(runs, which=args.which, data_root=root, delta=args.delta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
