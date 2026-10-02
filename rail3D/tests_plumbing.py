"""Plumbing gates: the non-physics assumptions that have actually broken runs.

    python tests_plumbing.py          # ~1 min, CPU, no dataset, no GPU

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
  P4  fdtd_agreement.py --external recovers rail3D's own field (horn, plate)
      from a file shaped like LUMERICAL.md's matlabsave: SI metres, a 4x finer
      FDTD-like grid, an arbitrary complex gain, and Lumerical's exp(-i w t)
      convention -- so it must align AS-IS. A conjugated (HFSS/FEKO-style) copy
      must still be detected. The round trip must score ~100% on both planes:
      anything less is the tool, not physics. A v7.3 (HDF5) .mat -- Lumerical's
      plain matlabsave -- is refused with the fix, not a scipy traceback.
  P5  horn_source.py writes the Import-source file in the layout Lumerical's
      own example uses (x, y column vectors in m, scalar z, (nx, ny) complex
      E/H), with E_y equal to rail3D's windowed horn field, flux going DOWN,
      and |E|/|H| = Z0 -- the three things that decide whether FDTD injects the
      horn at all, and in the right direction. After the lab's "cannot be
      opened ... MATLAB v7" error (README finding 30): the .lsf scripts find
      their files by ABSOLUTE path, not Lumerical's working directory; the
      text copy reads back equal to the .mat in readdata's layout; the load
      check written into the script passes on both ways in and FAILS on a
      conjugated, transposed or real-only load; export_horn saves with
      matlabsavelegacy, which scipy can read.
  P6  the three-plane source verification (README finding 32) on synthetic
      exports built from the model in Lumerical's layout (metres, column
      vectors, an arbitrary complex gain): rung -1's aperture, near, z = 150
      slab (3D, resampled through the envelope) and Import-plane scores, and
      rung 0's injected-plane score, must all round-trip to ~1 and ~0 dB.

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
                       source="horn", terms="psi12", shadow="raycast",
                       min_t=config.SHADOW_MIN_T)
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
        savemat(str(mat), {"Ey": fine * gain,          # exp(-i w t): NOT conjugated
                           "x": (fx * 1e-3)[:, None], "y": (fy * 1e-3)[:, None]})

        _sys.argv = ["fdtd_agreement.py", "--sample", "plate", "--external", str(mat),
                     "--profile", "cpu"]
        rc = fa.main()
        out = json.loads((tmp / "fdtd_agreement_plate.json").read_text(encoding="utf-8"))
        al, z30, ms = out["alignment"], out["z30"], out["ms_plane"]
        res = {
            "exit_ok": rc == 0,
            "aligned_as_is": al["convention"] == "as-is",
            # HFSS/FEKO fields arrive conjugated; that path must still be caught
            "conjugate_still_detected": bool(cw.align_external(
                torch.as_tensor(r).conj() * gain, torch.as_tensor(r))[1]["conjugated"]),
            "gain_recovered": abs(al["gain"] * 2.5e-3 - 1) < 0.01,
            "z30_corr": round(z30["complex_corr"], 6),
            "ms_corr": round(ms["complex_corr"], 6),
            "z30_roundtrip_exact": z30["complex_corr"] > 0.9999,
            "ms_roundtrip_exact": ms["complex_corr"] > 0.9999,
            "all_gates_pass": all(z30["pass"].values()) and all(ms["pass"].values()),
            "figure_written": (tmp / "fdtd_agreement_plate.png").exists(),
            "stamped": "_stamp" in out,
        }
        # a v7.3 (HDF5) .mat -- Lumerical's plain matlabsave -- must be refused
        # with the fix, not die in scipy (README finding 30). Header only: the
        # version word 0x0200 at byte 124 is all scipy looks at.
        v73 = tmp / "v73.mat"
        v73.write_bytes(b"MATLAB 7.3 MAT-file, Platform: PCWIN64, HDF5 schema 1.00 .".ljust(116)
                        + bytes(8) + b"\x00\x02IM" + bytes(384))
        try:
            cw.load_external(str(v73), gw)
            res["v73_refused_with_fix"] = False
        except SystemExit as exc:
            res["v73_refused_with_fix"] = "matlabsavelegacy" in str(exc)
        res["pass"] = all(v for v in res.values() if isinstance(v, bool))
        return res
    finally:
        config.FIGURE_DIR, config.GENERATED_DIR, _sys.argv[:] = saved[0], saved[1], saved[2]
        shutil.rmtree(tmp, ignore_errors=True)


def mesh3d_arc(section) -> int:
    from rail3d import mesh3d
    return mesh3d.default_arc_count(section, config.MESH_DS)


# ---------------------------------------------------------------------------
def p5_horn_source() -> dict:
    """The horn Import-source file: layout, E_y fidelity, direction, impedance."""
    from scipy.io import loadmat
    import horn_source as hs

    tmp = Path(tempfile.mkdtemp())
    try:
        dev = torch.device("cpu")
        info = hs.build(tmp, dev, sample=1.0, check=False)       # coarse: seconds
        m = loadmat(str(tmp / "horn_source.mat"))
        nx, ny = info["grid"]
        xs = np.arange(info["window_mm"]["x"][0], info["window_mm"]["x"][1] + 1e-9, 1.0)
        ys = np.arange(info["window_mm"]["y"][0], info["window_mm"]["y"][1] + 1e-9, 1.0)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        want = hs.horn_field(xs, ys, hs.Z_SRC, dev) * hs.taper(X, *info["window_mm"]["x"]) \
            * hs.taper(Y, *info["window_mm"]["y"])
        lsf = (tmp / "load_horn_source.lsf").read_text(encoding="utf-8")
        out = {
            "x_column_m": bool(m["x"].shape == (nx, 1) and abs(m["x"][0, 0] - xs[0] * 1e-3) < 1e-12),
            "y_column_m": m["y"].shape == (ny, 1),
            "z_scalar_m": m["z"].size == 1 and abs(float(m["z"].ravel()[0]) - hs.Z_SRC * 1e-3) < 1e-12,
            "fields_nx_ny_complex": all(m[k].shape == (nx, ny) and np.iscomplexobj(m[k])
                                        for k in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")),
            "ey_is_rail3d_horn": bool(np.allclose(m["Ey"], want, rtol=1e-5, atol=1e-9)),
            "flux_down": info["flux_down_fraction"] > 0.999,
            "impedance_is_Z0": abs(info["impedance_ohm"] / info["Z0_ohm"] - 1) < 0.01,
            "lsf_uses_documented_calls": all(t in lsf for t in (
                'rectilineardataset("EM fields", x, y, z)', 'addattribute("E"',
                'addattribute("H"', "importdataset(EM)", '"wavelength span", 0',
                '"direction", "Backward"')),
            # the .mat is found by ABSOLUTE path, not Lumerical's working
            # directory, and if matlabload cannot read it the text copy is
            # loaded instead -- the lab's "cannot be opened ... MATLAB v7"
            # error (2026-09-29) did not say which (README finding 30)
            "lsf_absolute_bundle_path": f'BUNDLE = "{tmp.resolve().as_posix()}";' in lsf,
            "lsf_no_placeholder_left": "@" not in lsf,
            "lsf_searches_script_dir_then_workdir": all(t in lsf for t in (
                "filedirectory(currentscriptname)", "src_dir = here;", "src_dir = pwd;",
                'mat_file = src_dir + "/horn_source.mat";', "matlabload(mat_file);")),
            "lsf_catches_unreadable_mat": "try {" in lsf and "catch(load_err);" in lsf,
            "lsf_text_fallback_complete": all(
                f'{k} = readdata(txt_dir + "{k}_re.txt") + 1i*readdata(txt_dir + "{k}_im.txt");' in lsf
                for k in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")) and all(
                f'{k} = readdata(txt_dir + "{k}.txt");' in lsf for k in "xyzf"),
            "lsf_verifies_both_ways": (lsf.count("CHECK FAILED: E, H power") == 2
                                       and "verified (matlabload)" in lsf
                                       and "verified (readdata, the text copy)" in lsf),
            "lsf_source_only_if_verified": lsf.rindex("if (good == 1) {") < lsf.index("addimportedsource"),
            "lsf_saves_beside_mat": ('out_file = src_dir + "/horn_EM_dataset.mat";' in lsf
                                     and "matlabsave(out_file, EM);" in lsf),
            "lsf_braces_balanced": lsf.count("{") == lsf.count("}") and lsf.count("(") == lsf.count(")"),
            # MATLAB's default -v7 layout (miCOMPRESSED), which Ansys documents
            # matlabload as reading
            "mat_is_v7_compressed": _mat_top_level_types(tmp / "horn_source.mat") == {15},
            "build_writes_text_copy": (tmp / hs.TXT_DIR / "Ey_re.txt").exists(),
            "flux_down_fraction": round(info["flux_down_fraction"], 6),
            "impedance_ohm": round(info["impedance_ohm"], 2),
        }
        res_txt, got = _p5_text_copy(tmp, m, nx, ny)
        out.update(res_txt)
        # the script's load check, run here in numpy: it must pass on what either
        # way loads, and FAIL on a conjugated, transposed or real-only load
        c = _lsf_constants(lsf)
        mat_arrays = {k: np.asarray(m[k]) for k in ("x", "y", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
        fld = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
        out["lsf_check_passes_mat"] = _emulate_lsf_check(c, mat_arrays)
        out["lsf_check_passes_txt"] = _emulate_lsf_check(c, got)
        out["lsf_check_catches_bad_loads"] = not any(_emulate_lsf_check(c, bad) for bad in (
            dict(mat_arrays, **{k: np.conj(mat_arrays[k]) for k in fld}),
            dict(mat_arrays, **{k: mat_arrays[k].T for k in fld}),
            dict(mat_arrays, **{k: mat_arrays[k].real for k in fld})))
        # rung -1's export script has the same hazard in the WRITE direction,
        # plus a format one: matlabsave writes v7.3, which scipy cannot read
        import horn_fdtd_case as hf
        case = tmp / "horn_case"
        hf.build(case)
        ex = (case / "export_horn.lsf").read_text(encoding="utf-8")
        out["export_lsf_absolute"] = (f'BUNDLE = "{case.resolve().as_posix()}";' in ex
                                      and "@BUNDLE@" not in ex
                                      and "case_dir = here;" in ex and "case_dir = pwd;" in ex
                                      and ex.count("{") == ex.count("}"))
        out["export_lsf_scipy_readable"] = ("matlabsave(" not in ex and ex.count(
            'matlabsavelegacy(case_dir + "/horn_') == 3)              # aperture, near, slab
        out["export_lsf_slab_optional"] = ('getdata("mon_slab", "Ey")' in ex
                                           and "catch(slab_err);" in ex)
        # a rebuild must replace the text copy, never leave the previous one
        (tmp / hs.TXT_DIR / "stale.txt").write_text("1\n")
        hs.build(tmp, dev, sample=1.0, check=False)
        out["rebuild_replaces_text_copy"] = (not (tmp / hs.TXT_DIR / "stale.txt").exists()
                                             and (tmp / hs.TXT_DIR / "Ey_re.txt").exists())
        # np.bool_ is not a bool: a numpy comparison left unwrapped used to drop
        # out of this gate silently (x_column_m did, until 2026-09-29)
        out["pass"] = all(v for v in out.values() if isinstance(v, (bool, np.bool_)))
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def p6_source_verification() -> dict:
    """The three-plane source verification, scored on synthetic exports built
    from rail3D's own model (README finding 32): every comparison must come
    back ~1 -- anything less is the scorer, not physics."""
    import functools
    import sys as _sys
    from scipy.io import savemat
    import fdtd_agreement as fa
    import horn_fdtd_case as hf
    import horn_source as hs
    from rail3d import field3d

    tmp = Path(tempfile.mkdtemp())
    saved = (config.FIGURE_DIR, config.GENERATED_DIR, list(_sys.argv), hs.source_plane_reference)
    config.FIGURE_DIR = config.GENERATED_DIR = tmp
    gain = 3.7e-4 * np.exp(1j * 0.9)                     # an export is in V/m, not model units
    col = lambda v: (np.asarray(v) * 1e-3)[:, None]       # noqa: E731  (Lumerical: metres, columns)
    try:
        p = hf.plan()
        m = p["monitors"]["mon_aperture"]
        xa = np.arange(m["x"][0], m["x"][1] + 1e-9, 0.25)
        ya = np.arange(m["y"][0], m["y"][1] + 1e-9, 0.25)
        savemat(str(tmp / "ap.mat"), {"Ey": gain * hf.model_aperture(xa, ya), "x": col(xa), "y": col(ya)})
        n_ = p["monitors"]["mon_near"]
        xn = np.arange(n_["x"][0], n_["x"][1] + 1e-9, 0.5)
        yn = np.arange(n_["y"][0], n_["y"][1] + 1e-9, 0.5)
        savemat(str(tmp / "near.mat"), {"Ey": gain * hf.model_plane(xn, yn, n_["z"]), "x": col(xn),
                                        "y": col(yn)})
        # a small core of mon_slab (the z = 150 plane passes through it), the
        # model radiated in the horn frame exactly as model_plane does
        A, B = config.SIZE_ANT[:2]
        xs3, ys3, zs3 = (np.arange(-6.0, 6.0 + 1e-9, 0.5), np.arange(-5.0, 5.0 + 1e-9, 0.5),
                         np.arange(6.0, 28.0 + 1e-9, 0.5))
        u = field3d.aperture_axis(A, 40, "midpoint").double().numpy()
        v = field3d.aperture_axis(B, 40, "midpoint").double().numpy()
        U, V = np.meshgrid(u, v, indexing="ij")
        q = np.stack([U.ravel(), V.ravel(), np.zeros(U.size)], 1)
        G = np.stack(np.meshgrid(xs3, ys3, zs3, indexing="ij"), -1).reshape(-1, 3)
        e3 = hf.radiate(field3d.aperture_distribution(U, V, config.SIZE_ANT, config.WVL), q,
                        field3d.aperture_weight(A, B, 40, "midpoint"), np.array([0.0, 0, 1]), G)
        savemat(str(tmp / "slab.mat"), {"Ey": gain * e3.reshape(len(xs3), len(ys3), len(zs3)),
                                        "x": col(xs3), "y": col(ys3), "z": col(zs3)})
        _sys.argv = ["fdtd_agreement.py", "--horn", str(tmp / "ap.mat"), "--horn-near",
                     str(tmp / "near.mat"), "--horn-slab", str(tmp / "slab.mat"), "--profile", "cpu"]
        rc_h = fa.main()
        h = json.loads((tmp / "fdtd_horn_agreement.json").read_text(encoding="utf-8"))

        # rung 0: z = 0 on the scorer's own grid, and the injected plane at 1 mm
        # (the 0.25 mm production grid costs ~20 s here and tests nothing more)
        hs.source_plane_reference = functools.partial(saved[3], sample=1.0)
        g0 = dict(__import__("compare_wavefronts").geom_now(), h_ms=0.0, nx=48, ny=56)
        x0 = (np.arange(48) + 0.5) * g0["dx"] - 48 * g0["dx"] / 2
        y0 = (np.arange(56) + 0.5) * g0["dx"] - 56 * g0["dx"] / 2
        savemat(str(tmp / "z0.mat"), {"Ey": gain * hs.horn_field(x0, y0, 0.0, torch.device("cpu")),
                                      "x": col(x0), "y": col(y0)})
        ref, xs_s, ys_s, _ = hs.source_plane_reference(hs.Z_SRC - 0.5, torch.device("cpu"))
        savemat(str(tmp / "src.mat"), {"Ey": gain * ref, "x": col(xs_s), "y": col(ys_s)})
        _sys.argv = ["fdtd_agreement.py", "--injection", str(tmp / "z0.mat"), "--injection-src",
                     str(tmp / "src.mat"), "--profile", "cpu"]
        rc_0 = fa.main()
        r0 = json.loads((tmp / "fdtd_rung0.json").read_text(encoding="utf-8"))

        slab = h.get("slab", [{}])[0]
        res = {
            "horn_exit_ok": rc_h == 0,
            "aperture_corr": round(h["aperture"]["complex_corr"], 5),
            "near_corr": round(h["near"]["complex_corr"], 5),
            "slab_plane_z": slab.get("global_z_mm"),
            "slab_points": slab.get("points"),
            "slab_fdtd_vs_rsi": slab.get("fdtd_vs_rsi_of_fdtd_aperture"),
            "slab_fdtd_vs_model": slab.get("fdtd_vs_model"),
            "slab_level_dB": slab.get("level_fdtd_over_model_dB"),
            "source_plane_corr": h["source_plane"]["full_wave_horn_vs_source"],
            "source_plane_level_dB": h["source_plane"]["level_fdtd_over_model_dB"],
            "ms_direction_deg": h["aperture"]["metasurface_direction"]["theta_from_boresight_deg"],
            "rung0_exit_ok": rc_0 == 0,
            "rung0_src_corr": round(r0["source_plane"]["complex_corr"], 5),
        }
        res.update({
            "slab_scored_at_ms_height": slab.get("global_z_mm") == config.HORN_LOWER_EDGE_Z
            and (slab.get("points") or 0) > 200,
            "slab_roundtrip": (slab.get("fdtd_vs_rsi_of_fdtd_aperture") or 0) > 0.995
            and (slab.get("fdtd_vs_model") or 0) > 0.995 and abs(slab.get("level_fdtd_over_model_dB", 9)) < 0.2,
            "source_plane_roundtrip": res["source_plane_corr"] > 0.999
            and abs(res["source_plane_level_dB"]) < 0.1,
            "rung0_src_roundtrip": res["rung0_src_corr"] > 0.9999,
        })
        res["pass"] = all(v for v in res.values() if isinstance(v, (bool, np.bool_)))
        return res
    finally:
        config.FIGURE_DIR, config.GENERATED_DIR = saved[0], saved[1]
        _sys.argv[:] = saved[2]
        hs.source_plane_reference = saved[3]
        shutil.rmtree(tmp, ignore_errors=True)


def _lsf_constants(lsf: str) -> dict:
    """The numbers render_lsf wrote into load_horn_source.lsf."""
    import re
    return {k: float(re.search(rf"\b{k} = ([-+0-9.eE]+);", lsf).group(1))
            for k in ("NX", "NY", "P_E", "P_H", "IPK", "JPK", "EY_PK_RE", "EY_PK_IM")}


def _emulate_lsf_check(c: dict, a: dict) -> bool:
    """load_horn_source.lsf's load check, line for line (Lumerical indexes from 1)."""
    ey = a["Ey"]
    if ey.shape != (c["NX"], c["NY"]) or a["x"].size != c["NX"] or a["y"].size != c["NY"]:
        return False
    pe = sum((np.abs(a[k]) ** 2).sum() for k in ("Ex", "Ey", "Ez"))
    ph = sum((np.abs(a[k]) ** 2).sum() for k in ("Hx", "Hy", "Hz"))
    pk = ey[int(c["IPK"]) - 1, int(c["JPK"]) - 1]
    want = c["EY_PK_RE"] + 1j * c["EY_PK_IM"]
    return bool(abs(pe / c["P_E"] - 1) <= 1e-6 and abs(ph / c["P_H"] - 1) <= 1e-6
                and abs(pk - want) <= 1e-6 * abs(want))


def _p5_text_copy(tmp: Path, m: dict, nx: int, ny: int):
    """The text copy: what Lumerical's readdata would load equals the .mat."""
    import horn_source as hs
    d = tmp / hs.TXT_DIR
    got = {k: np.loadtxt(d / f"{k}.txt", ndmin=2) for k in "xyzf"}
    for k in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        got[k] = (np.loadtxt(d / f"{k}_re.txt", ndmin=2)
                  + 1j * np.loadtxt(d / f"{k}_im.txt", ndmin=2))
    # readdata skips any line starting with a letter, and wants equal columns;
    # a doubled line ending (\r\r\n) shows up here as blank rows
    lines_ok, cols_ok = True, True
    for fp in d.glob("*.txt"):
        rows = fp.read_bytes().decode("ascii").splitlines()
        lines_ok &= all(r[:1].isdigit() or r[:1] == "-" for r in rows)
        cols_ok &= len({len(r.split()) for r in rows}) == 1
    return {
        "txt_files_16": len(list(d.glob("*.txt"))) == 16,
        "txt_layout_nx_by_ny": all(got[k].shape == (nx, ny) for k in ("Ex", "Ey", "Hz"))
        and got["x"].shape == (nx, 1) and got["y"].shape == (ny, 1)
        and got["z"].shape == (1, 1) and got["f"].shape == (1, 1),
        "txt_equals_mat": all(np.allclose(got[k], m[k], rtol=1e-8, atol=0)
                              for k in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"))
        and all(np.array_equal(got[k].ravel(), np.asarray(m[k], float).ravel()) for k in "xyzf"),
        "txt_lines_readdata_safe": bool(lines_ok),
        "txt_rows_equal_length": bool(cols_ok),
    }, got


def _mat_top_level_types(path) -> set:
    """Top-level MAT v5 data-element types: 14 = miMATRIX, 15 = miCOMPRESSED."""
    b = Path(path).read_bytes()
    i, types = 128, set()
    while i + 8 <= len(b):
        t = int.from_bytes(b[i:i + 4], "little")
        types.add(t)
        i += 8 + int.from_bytes(b[i + 4:i + 8], "little")
        if t != 15:
            i += (-i) % 8                               # 64-bit alignment (not after miCOMPRESSED)
    return types


# ---------------------------------------------------------------------------
def main() -> int:
    ok = True
    for name, fn in [("P0_stage_root", p0_stage_root),
                     ("P1_history_schema", p1_history_schema),
                     ("P2_zero_phase_is_baseline", p2_zero_phase_is_baseline),
                     ("P3_slm_profile", p3_slm_profile),
                     ("P4_fdtd_agreement_roundtrip", p4_fdtd_agreement_roundtrip),
                     ("P5_horn_source", p5_horn_source),
                     ("P6_source_verification", p6_source_verification)]:
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
