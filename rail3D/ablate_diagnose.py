"""Why did the SLM lose? Train vs validation vs test for every ablation run.

    python ablate_diagnose.py --stage prelim          # ~1-2 min on the 5090, no training

Reads the checkpoints ablate_surface.py left behind (abl_<name>_<tag>/best.pt) and
scores each on ALL THREE splits with the same hard-detector evaluate() training uses.
It separates the two explanations ablate_surface cannot tell apart:

  overfitting   SLM train AUC at or above the control's, val/test AUC below it:
                1800 free phases fit the 320 training intact poses (the rank loss
                pulls the TRAIN intact cluster tight) and do not generalise.
  optimisation  SLM train AUC already below the control's: the optimiser never
                found the solution zero phase gives for free.

Two probes per SLM run:
  zero phase    the same trained detectors and classifier with the phase set to 0
                (= the control's optics). If AUC goes UP, the learned phase is
                actively hurting with these detectors.
  intact spread gap0 (intact distance to its own mean) on train vs val. The SLM
                shrinking train gap0 far below val gap0 is the overfitting signature.

Writes data/generated/surface_ablation_diagnosis.json.
"""
from __future__ import annotations

import argparse
import json

import torch

from rail3d import config, data3d, train3d

REPORT = config.GENERATED_DIR / "surface_ablation_diagnosis.json"
NAMES = ("none_ctrl", "slm_shipped", "slm_flat", "slm_nocap", "slm_flat_nocap", "slm_long")
SPLITS = ("train", "val", "test")
KEEP = ("auc", "class_acc", "pass_rate", "gap_p5", "gap_mean", "gap0_mean")


def score(model, data, cfg) -> dict:
    return {s: {k: train3d.evaluate(model, data, s, cfg)[k] for k in KEEP} for s in SPLITS}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="prelim", choices=sorted(config.STAGES))
    ap.add_argument("--root", default=None, help="dataset root (default: the stage root)")
    ap.add_argument("--which", default="best", choices=["best", "latest"])
    ap.add_argument("--profile", default="lab")
    args = ap.parse_args()

    root = data3d.stage_root(args.stage) if args.root is None else config.GENERATED_DIR / args.root
    data3d.check_dataset_config(root, strict=True)
    tag = data3d.run_tag(root)
    device = config.get_device(args.profile)
    print(f"\ndataset {root.name} | device {device} | checkpoints abl_*_{tag}/{args.which}.pt\n")

    out: dict = {"dataset_root": root.name, "which": args.which, "runs": {}}
    data = None
    for name in NAMES:
        cfg = train3d.TrainConfig(run_name=f"abl_{name}_{tag}", data_root=str(root),
                                  surface="none" if name == "none_ctrl" else "slm")
        if not (config.CHECKPOINT_DIR / cfg.run_name / f"{args.which}.pt").exists():
            print(f"  {name:15s} (no checkpoint, skipped)")
            continue
        model, state = train3d.load_trained(cfg, device, args.which)
        cfg = train3d.effective_config(cfg, state)
        if data is None:
            data = train3d.load_all_data(cfg, device)
            print(f"  splits: train {len(data['train'])} / val {len(data['val'])} / "
                  f"test {len(data['test'])} defects; intact {len(data['i_train'])} / "
                  f"{len(data['i_val'])} / {len(data['i_test'])}\n")
        rec = {"surface": cfg.surface, "epoch": state.get("epoch"), "n_det": model.detector.n_det,
               "splits": score(model, data, cfg)}
        if cfg.surface == "slm":
            with torch.no_grad():
                for layer in model.layers:
                    layer.phase.zero_()
            rec["zero_phase"] = score(model, data, cfg)
        out["runs"][name] = rec
        REPORT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    if "none_ctrl" not in out["runs"]:
        print("!! no control checkpoint; run ablate_surface.py first")
        return 1

    hdr = (f"  {'run':15s} {'AUC train':>9s} {'val':>6s} {'test':>6s} | "
           f"{'gap0 train':>10s} {'val':>7s} | {'zero-phase AUC val':>18s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, r in out["runs"].items():
        s = r["splits"]
        zp = r.get("zero_phase", {}).get("val", {}).get("auc")
        print(f"  {name:15s} {s['train']['auc']:9.4f} {s['val']['auc']:6.4f} "
              f"{s['test']['auc']:6.4f} | {s['train']['gap0_mean']:10.5f} "
              f"{s['val']['gap0_mean']:7.5f} | "
              f"{'' if zp is None else format(zp, '18.4f')}")

    ctrl = out["runs"]["none_ctrl"]["splits"]
    slm = {k: v for k, v in out["runs"].items() if v["surface"] == "slm"}
    verdict = {}
    for name, r in slm.items():
        s = r["splits"]
        tr_ahead = s["train"]["auc"] >= ctrl["train"]["auc"] - 0.002
        va_behind = s["val"]["auc"] < ctrl["val"]["auc"] - 0.002
        zp_better = r["zero_phase"]["val"]["auc"] > s["val"]["auc"] + 0.002
        verdict[name] = ("overfitting" if tr_ahead and va_behind else
                         "optimisation" if va_behind else "ok")
        verdict[name + "_zero_phase_better"] = zp_better
    out["verdict"] = verdict
    REPORT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print()
    for name in slm:
        v = verdict[name]
        msg = {"overfitting": "train AUC >= control, val below: OVERFITTING (generalisation gap)",
               "optimisation": "train AUC already below control: OPTIMISATION (never found it)",
               "ok": "val AUC at or above the control"}[v]
        zp = "; zero phase scores HIGHER with its own detectors" if verdict[name + "_zero_phase_better"] else ""
        print(f"  {name:15s} {msg}{zp}")
    print(f"\nreport -> {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
