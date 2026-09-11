"""Plumbing gates: the non-physics assumptions that have actually broken runs.

    python tests_plumbing.py          # ~15 s, CPU, no dataset, no GPU

tests_physics_3d.py guards the solver. This guards the things AROUND it -- how
results are addressed, read back and compared -- which is where the last three
lab failures came from, each one discovered on the 5090 in the middle of a long
run rather than here:

  P0  stage_root() moves to a fresh root on a geometry change and REUSES the
      root when the geometry matches. A generation refusal with no legal next
      move stranded a run (README finding 23).
  P1  the history that train() returns really does carry the fields the
      reporting code reads. An invented history["best_val"] crashed the
      ablation AFTER a full training run had completed.
  P2  SLM2D at zero phase reproduces surface="none" EXACTLY. This is the
      premise of README finding 24 -- if it ever stops holding, the claim that
      the baseline is inside the SLM's hypothesis space stops holding with it.

Every check runs the real code path on tiny synthetic inputs. Add one here
whenever a run dies on something that was not physics.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from rail3d import config, data3d, optics3d, train3d

NX, NY = config.NX, config.NY


# ---------------------------------------------------------------------------
def p0_stage_root() -> dict:
    """A geometry change must move the dataset root, not dead-end the run."""
    root = config.GENERATED_DIR / "L5_prelim"
    created = not root.exists()
    root.mkdir(parents=True, exist_ok=True)
    cfgp = root / "dataset_config.json"
    backup = cfgp.read_text(encoding="utf-8") if cfgp.exists() else None
    try:
        live = {k: data3d._jsonable(getattr(config, k)) for k in data3d.PROVENANCE_KEYS}

        # (a) a root holding a DIFFERENT geometry must not be reused
        stale = dict(live, CRACK_DEPTH_RANGE=[1.0, 99.0])
        cfgp.write_text(json.dumps(stale, indent=2), encoding="utf-8")
        diffs = data3d.check_dataset_config(root, strict=False)
        moved = data3d.stage_root("prelim")
        fresh = not (moved / "dataset_config.json").exists()

        # (b) the SAME geometry must be reused, or every run forks a new root
        cfgp.write_text(json.dumps(live, indent=2), encoding="utf-8")
        same = data3d.stage_root("prelim")

        out = {
            "drift_detected": len(diffs) > 0,
            "moved_to": moved.name,
            "moved_off_base": moved.name != "L5_prelim",
            "target_is_empty": fresh,
            "tag_in_name": data3d.geometry_tag() in moved.name,
            "run_tag_follows_root": data3d.run_tag(moved) == moved.name.lower(),
            "matching_geometry_reused": same.name == "L5_prelim",
        }
        out["pass"] = all(v for k, v in out.items() if isinstance(v, bool))
        return out
    finally:
        if backup is not None:
            cfgp.write_text(backup, encoding="utf-8")
        elif created:
            shutil.rmtree(root, ignore_errors=True)
        else:
            cfgp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
def p1_history_schema() -> dict:
    """Drive the real train() loop and read it the way the reports do."""
    import ablate_surface

    gen = torch.Generator().manual_seed(0)

    def fake_data(cfg, device, verbose=True):
        n, m = 64, 24
        f = torch.randn(n, NX, NY, generator=gen) + 1j * torch.randn(n, NX, NY, generator=gen)
        inb = torch.randn(m, NX, NY, generator=gen) + 1j * torch.randn(m, NX, NY, generator=gen)
        idx, i0 = torch.randperm(n, generator=gen), torch.randperm(m, generator=gen)
        return {"fields": f.to(device), "intact": inb.to(device),
                "labels": (torch.arange(n) % len(config.CLASS_NAMES)).to(device),
                "train": idx[:40].to(device), "val": idx[40:56].to(device),
                "test": idx[56:].to(device), "i_train": i0[:16].to(device),
                "i_val": i0[16:20].to(device), "i_test": i0[20:].to(device)}

    tmp = Path(tempfile.mkdtemp())
    orig_load, orig_ckpt = train3d.load_all_data, config.CHECKPOINT_DIR
    train3d.load_all_data, config.CHECKPOINT_DIR = fake_data, tmp
    try:
        cfg = train3d.TrainConfig(
            run_name="p1", surface="slm", n_epoch=14, batch_size=32, b0=8,
            det_grid=(4, 3), n_det_final=8, tau_anneal_end=4,
            prune_start=2, prune_end=8, prune_every=2, checkpoint_every=4)
        hist = train3d.train(cfg, device=torch.device("cpu"), verbose=False)
        best = ablate_surface.best_val_of(hist)
        tr = (hist.get("train") or [{}])[-1]
        out = {
            # the key I assumed existed, and does not
            "no_best_val_key": "best_val" not in hist,
            "best_epoch_recorded": hist.get("best_epoch", -1) >= 0,
            "val_entries": len(hist.get("val") or []),
            "best_val_found": bool(best),
            "epoch_matches": best.get("epoch") == hist.get("best_epoch"),
            "metrics_numeric": all(isinstance(best.get(k), float)
                                   for k in ("auc", "class_acc", "score")),
            "capture_numeric": isinstance(tr.get("capture_frac"), float),
            # and a genuinely empty history must degrade, not raise
            "empty_history_safe": ablate_surface.best_val_of(
                {"best_epoch": -1, "val": []}) == {},
            "fmt_handles_none": ablate_surface.fmt(None).strip() == "n/a",
        }
        out["pass"] = all(v for k, v in out.items()
                          if isinstance(v, bool)) and out["val_entries"] > 0
        return out
    finally:
        train3d.load_all_data, config.CHECKPOINT_DIR = orig_load, orig_ckpt
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
def p2_zero_phase_is_baseline() -> dict:
    """README finding 24 rests on this identity; assert it rather than assume."""
    dev = torch.device("cpu")
    a = train3d.build_model(train3d.TrainConfig(surface="slm", slm_init_std=0.0), dev).eval()
    b = train3d.build_model(train3d.TrainConfig(surface="none"), dev).eval()
    d = train3d.build_model(train3d.TrainConfig(surface="slm"), dev).eval()
    b.detector.load_state_dict(a.detector.state_dict())
    d.detector.load_state_dict(a.detector.state_dict())
    torch.manual_seed(0)
    psi = torch.randn(4, NX, NY, dtype=torch.complex64)
    with torch.no_grad():
        _, da = a(psi, hard=True)
        _, db = b(psi, hard=True)
        _, dd = d(psi, hard=True)
    out = {
        "zero_phase_vs_none": float((da - db).abs().max()),
        "default_init_std_rad": round(float(d.layers[0].phase.std()), 3),
        "default_differs_from_none": float((dd - db).abs().max()) > 0,
    }
    out["identity_exact"] = out["zero_phase_vs_none"] == 0.0
    out["default_unchanged"] = abs(out["default_init_std_rad"] - np.pi / 2) < 0.05
    out["pass"] = all(v for k, v in out.items() if isinstance(v, bool))
    return out


# ---------------------------------------------------------------------------
def main() -> int:
    ok = True
    for name, fn in [("P0_stage_root", p0_stage_root),
                     ("P1_history_schema", p1_history_schema),
                     ("P2_zero_phase_is_baseline", p2_zero_phase_is_baseline)]:
        t0 = time.time()
        try:
            res = fn()
        except Exception as exc:                       # noqa: BLE001
            res = {"pass": False, "error": f"{type(exc).__name__}: {exc}"}
        res["seconds"] = round(time.time() - t0, 2)
        ok &= bool(res.get("pass"))
        print(f"[{'PASS' if res.get('pass') else 'FAIL'}] {name}: "
              f"{ {k: v for k, v in res.items() if k != 'pass'} }")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
