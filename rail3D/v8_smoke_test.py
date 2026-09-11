"""V8 — end-to-end smoke test on the 20/class smoke dataset.

Checks:
  * training works: TRAIN separation (gap - gap0) grows, gradients genuinely
    reach the surviving detector centers (keep-index composition, not a
    stand-in), pruning executes 130 -> 8, no detector collapse
    (min separation > one pixel), the capture fraction stays a fraction;
  * kill-and-resume: run B trains 15 epochs, "dies", resumes to 30, and its
    epoch-15..29 validation history must match run A's straight 30-epoch run
    (RNG-state checkpointing makes the continuation bit-identical);
  * the legacy margin objective and the legacy variance prune criterion still
    train (regression guards for both pre-rev.2 paths).

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

from rail3d import config, data3d, train3d

SMOKE_ROOT = config.GENERATED_DIR / "smoke"
REPORT_PATH = config.GENERATED_DIR / "verification_report.json"


def smoke_cfg(run_name: str) -> train3d.TrainConfig:
    # Exercises the DEFAULT dense start (config.DET_GRID = 13x10 = 130). The
    # geometric schedule needs enough steps to get there: at prune_keep=0.75 it
    # would take 10 steps, more than this 30-epoch run allows, so the smoke test
    # halves each time (5 steps: 130 -> 65 -> 33 -> 17 -> 9 -> 8). Real runs use
    # 0.75 with a 19-step window, which reaches n_det_final comfortably.
    return train3d.TrainConfig(
        run_name=run_name, surface="slm", mode="tot",
        n_epoch=30, batch_size=32, b0=16,
        prune_start=6, prune_end=18, prune_every=3,
        prune_schedule="fraction", prune_keep=0.5, prune_per_step=4,
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
    # "Training works" must be judged on the TRAINING objective, not validation.
    # The smoke split has 8 defect and 3 intact validation samples, so val AUC
    # is quantised to 1/24 and class accuracy to 1/8 (pinned at chance for four
    # classes) -- both are noise at this scale. Train separation has 64 samples
    # behind it and is exactly what the rank objective optimises.
    # Raw loss is not comparable either, since tau annealing and pruning change
    # the objective's shape mid-run.
    losses = [t["loss"] for t in hist_a["train"]]
    res["loss_first"], res["loss_last"] = losses[0], losses[-1]
    sep = [t["gap_mean"] - t["gap0_mean"] for t in hist_a["train"]]
    res["train_separation_first"] = sep[0]
    res["train_separation_last"] = sep[-1]
    res["separation_improves"] = sep[-1] > sep[0] * 1.2      # a real, not marginal, gain
    cap = [t.get("capture_frac") for t in hist_a["train"] if t.get("capture_frac") is not None]
    if cap:
        res["capture_first"], res["capture_last"] = cap[0], cap[-1]
    res["score_epoch0"] = hist_a["val"][0]["score"]
    res["score_best"] = hist_a["best_score"]

    model_a, state_a = train3d.load_trained(cfg_a, device, "latest")
    res["n_det_final"] = model_a.detector.n_det
    res["pruning_ok"] = model_a.detector.n_det == cfg_a.n_det_final

    # Compose the recorded keep lists so surviving detector row i maps back to
    # its ORIGINAL lattice row. (The old check compared the 8 survivors against
    # the FIRST 8 lattice rows — tens of mm apart whether or not any gradient
    # ever reached u, i.e. it could not fail.)
    orig = list(range(u_init.shape[0]))
    for entry in hist_a["prune_epochs"]:
        orig = [orig[i] for i in entry["keep"]]
    u_final = model_a.detector.u.detach()
    res["keep_composition_ok"] = len(orig) == u_final.shape[0]
    keep_moved = float((u_final - u_init[orig].to(u_final.device)).abs().max())
    res["detector_centers_moved"] = keep_moved > 1e-4   # any movement at all
    res["u_max_change"] = keep_moved

    # no collapse: nothing in the loss repels centres, so duplicates are a
    # real failure mode — min separation below one pixel means two windows
    # read the same spot (duplicate barcode entries)
    res["min_separation_mm"] = model_a.detector.min_separation()
    res["min_sep_ok"] = res["min_separation_mm"] > config.DX
    # capture stays a fraction (the clamp guards overlap double-counting)
    res["capture_ok"] = bool(cap) and 0.0 < cap[-1] <= 1.0

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

    # legacy-criterion regression guard: prune_criterion="variance" must still
    # drive a real training run (V0b covers the static function only)
    cfg_var = replace(smoke_cfg("v8_smoke_variance"), n_epoch=10,
                      prune_start=2, prune_end=8, prune_every=2,
                      prune_criterion="variance")
    fresh(cfg_var)
    hist_var = train3d.train(cfg_var, device=device, verbose=False)
    model_var, _ = train3d.load_trained(cfg_var, device, "latest")
    res["variance_criterion_ok"] = (len(hist_var["val"]) == 10
                                    and model_var.detector.n_det < 130)

    # stale-checkpoint refusal, end to end: doctor a REAL checkpoint's
    # geometry stamp and the loader must refuse it (V0c covers the helper on a
    # synthetic payload; this proves the wiring through load_trained)
    stale_dir = config.CHECKPOINT_DIR / "v8_smoke_stale"
    if stale_dir.exists():
        shutil.rmtree(stale_dir)
    shutil.copytree(config.CHECKPOINT_DIR / cfg_a.run_name, stale_dir)
    stale_path = stale_dir / "latest.pt"
    st = torch.load(stale_path, map_location="cpu", weights_only=False)
    st["geometry"]["WVL"] = 999.0
    torch.save(st, stale_path)
    try:
        train3d.load_trained(replace(cfg_a, run_name="v8_smoke_stale"),
                             device, "latest")
        res["stale_ckpt_refused"] = False
    except RuntimeError:
        res["stale_ckpt_refused"] = True
    shutil.rmtree(stale_dir)

    # resume equivalence (epochs 15..29)
    mism = 0.0
    for ea, eb in zip(hist_a["val"][15:], hist_b["val"][15:]):
        for key in ("pass_rate", "false_alarm", "class_acc", "gap_mean"):
            mism = max(mism, abs(ea[key] - eb[key]))
    res["resume_max_metric_mismatch"] = mism
    res["resume_ok"] = mism < 1e-6

    res["pass"] = bool(res["separation_improves"] and res["pruning_ok"]
                       and res["keep_composition_ok"]
                       and res["detector_centers_moved"] and res["min_sep_ok"]
                       and res["capture_ok"] and res["resume_ok"]
                       and res["full_eval_keys_ok"] and res["legacy_objective_ok"]
                       and res["variance_criterion_ok"]
                       and res["stale_ckpt_refused"])

    report = json.loads(REPORT_PATH.read_text()) if REPORT_PATH.exists() else {}
    report["V8_end_to_end_smoke"] = data3d.stamp(res)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    status = "PASS" if res["pass"] else "FAIL"
    print(f"[{status}] V8: { {k: v for k, v in res.items() if k != 'best_val'} }")
    print(f"best val: {res['best_val']}")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
