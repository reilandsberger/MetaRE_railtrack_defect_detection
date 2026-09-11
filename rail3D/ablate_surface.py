"""Why does the no-metasurface baseline BEAT the trained metasurface?

    python ablate_surface.py --stage prelim            # ~20 min on the 5090
    python ablate_surface.py --stage prelim --long     # adds a 6000-epoch run

The prelim run of 2026-09-11 produced val AUC 0.880 for surface="slm" and
**0.976** for surface="none". That is not a statement about metasurfaces: the
baseline is INSIDE the SLM's hypothesis space exactly -- SLM2D with zero phase
reproduces nn.Identity, verified to float32 zero on the detector powers -- so a
correctly optimized SLM cannot score below it. Something in the optimization,
not the physics, is giving up 0.10 AUC.

Two candidates, each cheap to test now that training is ~3 min:

  init   SLM2D starts at std pi/2 -- a full random diffuser that scrambles the
         defect signature into speckle at epoch 0. The optimizer then has to
         find its way back to (at least) identity through 1800 phases.
  capture w_capture=0.2 rewards power landing on the retained detectors. It was
         added for RECEIVER SNR, but eval runs noiseless (ONN3D.add_noise is
         gated on self.training), so at scoring time capture buys nothing and
         can only trade against contrast. The shipped run pushed capture to
         0.457 vs a 0.062 floor while AUC fell.

The grid is 2x2 over those two, plus the baseline as the control. Every run
uses the SAME dataset, schedule, seed and detector lattice, so the only moving
parts are the two named knobs.

Reading the table: `auc` is the number to compare (detection, the system's job);
`score` is what model selection actually maximizes (auc + class_acc), shown
because a run can win on score while losing on auc. Any SLM row below the
`none` row on auc is still an optimization failure, whatever the knobs say.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from rail3d import config, data3d, train3d

REPORT = config.GENERATED_DIR / "surface_ablation.json"


def best_val_of(hist: dict) -> dict:
    """The validation record of the epoch train() actually selected.

    train() does NOT store a "best_val" key -- it keeps one dict per epoch in
    history["val"] and the winning epoch in history["best_epoch"]. Match on the
    val dict's own "epoch" field rather than trusting list position, so this
    stays correct if validation ever stops running every epoch; fall back to
    position for histories written before that field existed.
    """
    be = hist.get("best_epoch", -1)
    vals = hist.get("val") or []
    if be < 0 or not vals:
        return {}
    for v in vals:
        if v.get("epoch") == be:
            return v
    return vals[be] if be < len(vals) else {}


def fmt(x, spec=".4f") -> str:
    """Format a metric that may be missing, without killing a finished run."""
    return format(x, spec) if isinstance(x, (int, float)) else "  n/a"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="prelim", choices=sorted(config.STAGES))
    ap.add_argument("--long", action="store_true",
                    help="add a 4x-epoch run of the best-scoring SLM setting")
    ap.add_argument("--profile", default="lab")
    args = ap.parse_args()

    spec = config.STAGES[args.stage]
    root = data3d.stage_root(args.stage)
    if not (root / "dataset_config.json").exists():
        print(f"!! no dataset at {root} - run run_stage.py --stage {args.stage} first")
        return 1
    data3d.check_dataset_config(root, strict=True)      # refuse a geometry drift
    device = config.resolve_device(args.profile) if hasattr(config, "resolve_device") \
        else torch.device(config.best_cuda_device())
    tag = data3d.run_tag(root)

    sched = {k: spec[k] for k in ("n_epoch", "tau_anneal_end", "prune_start", "prune_end")}
    base = dict(data_root=str(root), batch_size=config.PROFILES[args.profile].train_batch,
                **sched)
    diffuser = float(np.pi / 2)

    runs = [
        ("none_ctrl",      dict(surface="none")),
        ("slm_shipped",    dict(surface="slm", slm_init_std=diffuser, w_capture=0.2)),
        ("slm_flat",       dict(surface="slm", slm_init_std=0.0,      w_capture=0.2)),
        ("slm_nocap",      dict(surface="slm", slm_init_std=diffuser, w_capture=0.0)),
        ("slm_flat_nocap", dict(surface="slm", slm_init_std=0.0,      w_capture=0.0)),
    ]

    print(f"\ndataset {root.name} | device {device} | {sched['n_epoch']} epochs/run")
    print(f"{len(runs)} runs; the shipped prelim took 174 s/run\n")

    out: dict = {"dataset_root": root.name, "schedule": sched, "runs": {}}
    for name, kw in runs:
        cfg = train3d.TrainConfig(run_name=f"abl_{name}_{tag}", **base, **kw)
        t0 = time.time()
        hist = train3d.train(cfg, device=device, verbose=False)
        best = best_val_of(hist)
        tr = (hist.get("train") or [{}])[-1]
        if not best:
            print(f"  !! {name}: no validation record for best_epoch "
                  f"{hist.get('best_epoch')} ({len(hist.get('val') or [])} val "
                  f"entries) - the run trained but selected nothing; its "
                  f"checkpoint is in place, so re-running resumes it.")
        out["runs"][name] = {
            # read back from cfg, never from kw: an omitted knob takes the
            # TrainConfig default (w_capture is 0.2, not 0) and recording the
            # kw would mislabel the control's objective in the report.
            "surface": cfg.surface,
            "slm_init_std": cfg.slm_init_std if cfg.surface == "slm" else None,
            "w_capture": cfg.w_capture,
            "best_epoch": hist.get("best_epoch"), "n_det_at_best": hist.get("best_n_det"),
            "meets_detector_budget": hist.get("best_n_det") == cfg.n_det_final,
            "auc": best.get("auc"), "class_acc": best.get("class_acc"),
            "score": best.get("score"), "pass_rate": best.get("pass_rate"),
            "capture_frac_last": tr.get("capture_frac"), "seconds": round(time.time() - t0),
        }
        r = out["runs"][name]
        print(f"  {name:16s} auc {fmt(r['auc'])}  class_acc {fmt(r['class_acc'])}  "
              f"score {fmt(r['score'])}  cap {fmt(r['capture_frac_last'], '.3f')}  "
              f"[{r['seconds']}s, best ep {r['best_epoch']}, n_det {r['n_det_at_best']}]")
        # persist AS WE GO: training is the expensive part and a later error
        # must never throw away a run that already finished
        REPORT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    ctrl = out["runs"]["none_ctrl"]["auc"]
    scored = [v for k, v in out["runs"].items()
              if k != "none_ctrl" and isinstance(v.get("auc"), float)]
    if ctrl is None or not scored:
        print("\n  !! not enough scored runs to compare; see surface_ablation.json")
        REPORT.write_text(json.dumps(out, indent=2), encoding="utf-8")
        return 1
    best_slm = max(scored, key=lambda v: v["auc"])
    out["baseline_auc"] = ctrl
    out["best_slm_auc"] = best_slm["auc"]
    out["slm_beats_baseline"] = best_slm["auc"] > ctrl

    print(f"\n  no-MS baseline auc {ctrl:.4f} | best SLM auc {best_slm['auc']:.4f}")
    if out["slm_beats_baseline"]:
        print("  -> the metasurface now EARNS its place; adopt that setting as default.")
    else:
        print("  -> every SLM setting still loses to a piece of empty space, even though\n"
              "     zero phase reproduces it exactly. The optimization, not the surface,\n"
              "     is the blocker: next suspects are lr on the phase, the rank loss's\n"
              "     scale on normalized barcodes, and n_epoch (try --long).")

    if args.long:
        name, kw = "slm_long", dict(
            surface="slm", slm_init_std=best_slm["slm_init_std"],
            w_capture=best_slm["w_capture"])
        print()
        print(f"  --long: repeating the best SLM setting (init "
              f"{kw['slm_init_std']:.3f}, w_capture {kw['w_capture']}) "
              f"at 4x epochs")
        cfg = train3d.TrainConfig(run_name=f"abl_{name}_{tag}",
                                  **{**base, "n_epoch": 4 * sched["n_epoch"]}, **kw)
        t0 = time.time()
        hist = train3d.train(cfg, device=device, verbose=False)
        best = best_val_of(hist)
        out["runs"][name] = {
            "surface": "slm", "slm_init_std": kw["slm_init_std"],
            "w_capture": kw["w_capture"], "n_epoch": cfg.n_epoch,
            "best_epoch": hist.get("best_epoch"), "n_det_at_best": hist.get("best_n_det"),
            "auc": best.get("auc"), "class_acc": best.get("class_acc"),
            "score": best.get("score"), "seconds": round(time.time() - t0)}
        r = out["runs"][name]
        REPORT.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\n  {name:16s} auc {fmt(r['auc'])}  score {fmt(r['score'])}  "
              f"[{r['seconds']}s, best ep {r['best_epoch']} of {cfg.n_epoch}]")
        if r["best_epoch"] >= 0.9 * cfg.n_epoch:
            print("  ! still improving at 4x epochs — the schedule is the binding limit")

    REPORT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nreport -> {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
