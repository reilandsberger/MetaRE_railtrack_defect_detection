"""V8 — end-to-end smoke test on the 20/class smoke dataset.

Checks:
  * training runs: loss falls, gradients reach the detector centers (they move),
    pruning executes 18 -> 8, checkpoints round-trip;
  * kill-and-resume: run B trains 15 epochs, "dies", resumes to 30, and its
    epoch-15..29 validation history must match run A's straight 30-epoch run
    (RNG-state checkpointing makes the continuation bit-identical).

Run:  python v8_smoke_test.py [--device cuda:0]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from rail3d import config, train3d

SMOKE_ROOT = config.GENERATED_DIR / "smoke"
REPORT_PATH = config.GENERATED_DIR / "verification_report.json"


def smoke_cfg(run_name: str) -> train3d.TrainConfig:
    return train3d.TrainConfig(
        run_name=run_name, surface="slm", mode="tot",
        n_epoch=30, batch_size=32, b0=16,
        prune_start=6, prune_end=18, prune_every=3, prune_per_step=4,
        tau_anneal_end=20, checkpoint_every=5,
        data_root=str(SMOKE_ROOT),
    )


def fresh(cfg: train3d.TrainConfig) -> None:
    d = config.CHECKPOINT_DIR / cfg.run_name
    if d.exists():
        shutil.rmtree(d)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None,
                        help="e.g. cuda:0; default = RAIL3D_DEVICE, else strongest CUDA card")
    args = parser.parse_args()
    device = torch.device(args.device) if args.device else config.get_device("lab")

    res = {}

    # --- run A: straight 30 epochs
    cfg_a = smoke_cfg("v8_smoke_a")
    fresh(cfg_a)
    model_probe = train3d.build_model(cfg_a, device)
    u_init = model_probe.detector.u.detach().clone()
    hist_a = train3d.train(cfg_a, device=device)

    # --- run B: 15 epochs, "killed", resumed to 30
    cfg_b_half = replace(smoke_cfg("v8_smoke_b"), n_epoch=15)
    fresh(cfg_b_half)
    train3d.train(cfg_b_half, device=device)
    cfg_b_full = smoke_cfg("v8_smoke_b")
    hist_b = train3d.train(cfg_b_full, device=device)

    # --- checks
    # Raw loss is not comparable across tau-anneal/pruning transitions (the
    # objective changes shape); "training works" = validation score improves.
    losses = [t["loss"] for t in hist_a["train"]]
    res["loss_first"] = losses[0]
    res["loss_last"] = losses[-1]
    res["score_epoch0"] = hist_a["val"][0]["score"]
    res["score_best"] = hist_a["best_score"]
    res["score_improves"] = hist_a["best_score"] > hist_a["val"][0]["score"] + 1e-6

    model_a, state_a = train3d.load_trained(cfg_a, device, "latest")
    res["n_det_final"] = model_a.detector.n_det
    res["pruning_ok"] = model_a.detector.n_det == cfg_a.n_det_final

    u_final = model_a.detector.u.detach()
    keep_moved = float((u_final - u_init[: u_final.shape[0]]).abs().max())
    res["detector_centers_moved"] = keep_moved > 1e-4   # any movement at all
    res["u_max_change"] = keep_moved

    # checkpoint roundtrip: best.pt loads and evaluates
    model_best, _ = train3d.load_trained(cfg_a, device, "best")
    data = train3d.load_all_data(cfg_a, device)
    val = train3d.evaluate(model_best, data, "val", cfg_a)
    res["best_val"] = val

    # full_evaluation must run on the *training device* — the notebooks call it and
    # a CPU/GPU mismatch here (e.g. quantile positions built on CPU) would otherwise
    # only surface mid-notebook on the lab GPU.
    ev = train3d.full_evaluation(model_best, data, cfg_a)
    expected = {"test", "confusion", "roc", "noise_curve", "alignment_curve"}
    res["full_eval_keys_ok"] = expected.issubset(ev)
    res["full_eval_auc"] = ev["roc"]["auc"]

    # legacy-objective regression guard: the pre-rev.2 margin loss must still run
    cfg_legacy = replace(smoke_cfg("v8_smoke_legacy"), n_epoch=6,
                         objective="margin", metric="l2")
    fresh(cfg_legacy)
    hist_legacy = train3d.train(cfg_legacy, device=device, verbose=False)
    res["legacy_objective_ok"] = len(hist_legacy["val"]) == 6

    # resume equivalence (epochs 15..29)
    mism = 0.0
    for ea, eb in zip(hist_a["val"][15:], hist_b["val"][15:]):
        for key in ("pass_rate", "false_alarm", "class_acc", "gap_mean"):
            mism = max(mism, abs(ea[key] - eb[key]))
    res["resume_max_metric_mismatch"] = mism
    res["resume_ok"] = mism < 1e-6

    res["pass"] = bool(res["score_improves"] and res["pruning_ok"]
                       and res["detector_centers_moved"] and res["resume_ok"]
                       and res["full_eval_keys_ok"] and res["legacy_objective_ok"])

    report = json.loads(REPORT_PATH.read_text()) if REPORT_PATH.exists() else {}
    report["V8_end_to_end_smoke"] = res
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    status = "PASS" if res["pass"] else "FAIL"
    print(f"[{status}] V8: { {k: v for k, v in res.items() if k != 'best_val'} }")
    print(f"best val: {res['best_val']}")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
