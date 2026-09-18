"""Plumbing gates: the non-physics assumptions that have actually broken runs.

    python tests_plumbing.py          # ~25 s, CPU, no dataset, no GPU

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
  P3  slm_profile.py reads a REAL trained checkpoint and a dataset root in the
      generator's exact shard format, rebuilds the epoch-0 phase EXACTLY (its
      "moved from init" number is only meaningful if that holds), writes a
      wrapped phase with |t| = 1, and does not change matplotlib's backend on
      import (the notebook plots inline).
  P4  fdtd_agreement.py --external recovers rail3D's own field from a file
      shaped like LUMERICAL.md 5's matlabsave: SI metres, a 4x finer FDTD-like
      grid, CONJUGATED (exp(+jwt)), an arbitrary complex gain. The round trip
      must score ~100% on both planes -- anything less is the tool, not physics,
      and would otherwise first show up as "FDTD disagrees" on a real run.

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
def _synthetic_loader():
    """A stand-in for train3d.load_all_data: tiny random fields, real layout."""
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
    return fake_data


def _tiny_cfg(run_name: str, **kw) -> train3d.TrainConfig:
    """A few-second training run that still exercises pruning and selection."""
    return train3d.TrainConfig(
        run_name=run_name, surface="slm", n_epoch=14, batch_size=32, b0=8,
        det_grid=(4, 3), n_det_final=8, tau_anneal_end=4,
        prune_start=2, prune_end=8, prune_every=2, checkpoint_every=4, **kw)


# ---------------------------------------------------------------------------
def p1_history_schema() -> dict:
    """Drive the real train() loop and read it the way the reports do."""
    import ablate_surface

    tmp = Path(tempfile.mkdtemp())
    orig_load, orig_ckpt = train3d.load_all_data, config.CHECKPOINT_DIR
    train3d.load_all_data, config.CHECKPOINT_DIR = _synthetic_loader(), tmp
    try:
        cfg = _tiny_cfg("p1")
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
def p3_slm_profile() -> dict:
    """The metasurface figure reads what training actually writes."""
    import matplotlib
    backend = matplotlib.get_backend()
    import slm_profile
    backend_kept = matplotlib.get_backend() == backend

    tmp = Path(tempfile.mkdtemp())
    root = tmp / "root"
    root.mkdir()
    saved = (train3d.load_all_data, config.CHECKPOINT_DIR,
             config.FIGURE_DIR, config.GENERATED_DIR)
    train3d.load_all_data = _synthetic_loader()
    config.CHECKPOINT_DIR, config.FIGURE_DIR, config.GENERATED_DIR = tmp, tmp, tmp
    try:
        # the epoch-0 rebuild must be exact, including checkpoints that predate
        # the slm_init_std field (they were trained from the pi/2 diffuser)
        exact = True
        for std in (float(np.pi / 2), 0.0, 0.3):
            c0 = train3d.TrainConfig(slm_init_std=std)
            m0 = train3d.build_model(c0, torch.device("cpu"))
            exact &= torch.equal(slm_profile.initial_phase(
                {"config": {"slm_init_std": std}}, c0, 0), m0.layers[0].phase.detach())
        m_def = train3d.build_model(train3d.TrainConfig(), torch.device("cpu"))
        exact &= torch.equal(slm_profile.initial_phase({"config": {}}, train3d.TrainConfig(), 0),
                             m_def.layers[0].phase.detach())

        train3d.train(_tiny_cfg("p3"), device=torch.device("cpu"), verbose=False)

        # an intact shard in generate_dataset_3d.py's exact on-disk layout
        gen = torch.Generator().manual_seed(1)
        data3d.atomic_save({"psi": torch.randn(8, NX, NY, 2, generator=gen) * 0.05,
                            "meta": [{"class": "intact"}] * 8},
                           data3d.shard_path("intact", 0, root=root))
        X, _ = config.plane_grid("cpu")
        torch.save(torch.exp(-(X[0] / 40.0) ** 2).to(torch.complex64),
                   data3d.psi0_path(root=root))
        (root / "dataset_config.json").write_text("{}")

        png = slm_profile.plot_slm_profile(["p3"], data_root=root, delta=True)
        npz = np.load(tmp / "slm_profile_p3_L0.npz")
        ph = npz["phase_wrapped"]
        out = {
            "backend_untouched_by_import": backend_kept,
            "epoch0_rebuild_exact": bool(exact),
            "figure_written": png.exists() and png.stat().st_size > 20_000,
            "phase_shape_ok": ph.shape == (NX, NY),
            "phase_wrapped": bool((ph > -np.pi - 1e-6).all() and (ph <= np.pi + 1e-6).all()),
            "amplitude_is_one": bool(np.allclose(npz["amplitude"], 1.0)),
            "x_axis_ok": npz["x_mm"].shape == (NX,)
                         and bool(np.isclose(npz["x_mm"][-1] - npz["x_mm"][0], (NX - 1) * config.DX)),
        }
        out["pass"] = all(out.values())
        return out
    finally:
        (train3d.load_all_data, config.CHECKPOINT_DIR,
         config.FIGURE_DIR, config.GENERATED_DIR) = saved
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
def p4_fdtd_agreement_roundtrip() -> dict:
    """A real Lumerical export's bookkeeping must cost nothing."""
    import sys as _sys
    from scipy.interpolate import RegularGridInterpolator
    from scipy.io import savemat
    import compare_wavefronts as cw
    import fdtd_agreement as fa
    from rail3d import sections

    tmp = Path(tempfile.mkdtemp())
    saved = (config.FIGURE_DIR, config.GENERATED_DIR, list(_sys.argv))
    config.FIGURE_DIR = config.GENERATED_DIR = tmp
    try:
        sec = sections.load_reference_section()
        params = fa.sample_params("plate", sec)
        v, _ = cw.build_mesh(sec, params, config.MESH_DS,
                             mesh3d_arc(sec), None)
        plan = cw.fdtd_plan(v.numpy(), dict(cw.geom_now(), h_ms=fa.Z_MON), fa.Z_MON, None)
        gw = fa.window_geom(plan["monitor_mm"])
        dev = torch.device("cpu")
        ref = cw.solve(g=gw, section=sec, params=params, device=dev, chunk=512, seg=None,
                       source="plane", shadow="raycast", min_t=config.SHADOW_MIN_T)
        X, Y = cw.plane_grid(gw, dev)
        xs, ys = X[0, :, 0].numpy(), Y[0, 0, :].numpy()
        # FDTD-like: 4x finer, one extra sample beyond each edge (a real monitor
        # is wider than the window), coordinates in METRES, field conjugated
        fx = np.concatenate([[xs[0] - 0.625], np.arange(xs[0], xs[-1] + 1e-9, 0.625),
                             [xs[-1] + 0.625]])
        fy = np.concatenate([[ys[0] - 0.625], np.arange(ys[0], ys[-1] + 1e-9, 0.625),
                             [ys[-1] + 0.625]])
        FX, FY = np.meshgrid(np.clip(fx, xs[0], xs[-1]), np.clip(fy, ys[0], ys[-1]),
                             indexing="ij")
        r = ref.numpy()
        fine = sum(RegularGridInterpolator((xs, ys), getattr(r, part))(
            np.stack([FX.ravel(), FY.ravel()], -1)).reshape(FX.shape) * unit
            for part, unit in (("real", 1), ("imag", 1j)))
        gain = 2.5e-3 * np.exp(1j * np.radians(30.0))
        mat = tmp / "plate_z30.mat"
        savemat(str(mat), {"Ey": np.conj(fine) * gain,
                           "x": (fx * 1e-3)[:, None], "y": (fy * 1e-3)[:, None]})

        _sys.argv = ["fdtd_agreement.py", "--sample", "plate", "--external", str(mat),
                     "--profile", "cpu"]
        rc = fa.main()
        out = json.loads((tmp / "fdtd_agreement_plate.json").read_text(encoding="utf-8"))
        al, z30, ms = out["alignment"], out["z30"], out["ms_plane"]
        res = {
            "exit_ok": rc == 0,
            "conjugation_detected": bool(al["conjugated"]),
            "gain_recovered": abs(al["gain"] * 2.5e-3 - 1) < 0.01,
            "z30_corr": round(z30["complex_corr"], 6),
            "ms_corr": round(ms["complex_corr"], 6),
            "z30_roundtrip_exact": z30["complex_corr"] > 0.9999,
            "ms_roundtrip_exact": ms["complex_corr"] > 0.9999,
            "all_gates_pass": all(z30["pass"].values()) and all(ms["pass"].values()),
            "figure_written": (tmp / "fdtd_agreement_plate.png").exists(),
            "stamped": "_stamp" in out,
        }
        res["pass"] = all(v for v in res.values() if isinstance(v, bool))
        return res
    finally:
        config.FIGURE_DIR, config.GENERATED_DIR, _sys.argv[:] = saved[0], saved[1], saved[2]
        shutil.rmtree(tmp, ignore_errors=True)


def mesh3d_arc(section) -> int:
    from rail3d import mesh3d
    return mesh3d.default_arc_count(section, config.MESH_DS)


# ---------------------------------------------------------------------------
def main() -> int:
    ok = True
    for name, fn in [("P0_stage_root", p0_stage_root),
                     ("P1_history_schema", p1_history_schema),
                     ("P2_zero_phase_is_baseline", p2_zero_phase_is_baseline),
                     ("P3_slm_profile", p3_slm_profile),
                     ("P4_fdtd_agreement_roundtrip", p4_fdtd_agreement_roundtrip)]:
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
