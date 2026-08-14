"""Automated physics verification (V1-V4) for the rail3D pipeline.

Laptop-safe: tiny meshes, CPU by default (set RAIL3D_TEST_DEVICE to override).
Run:  python tests_physics_3d.py
Results are also cached to rail3D/data/generated/verification_report.json.

V1 — chunked/batched solver vs the verbatim Face3D farfield_from_antenna_2nd
V2 — PropagatorRSFFT (FFT) vs the original conv2d formulation
V3 — angular-spectrum propagator vs RS-FFT on a Gaussian beam
V4 — physics sanity: specular lobe, power conservation, mesh orientation
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from rail3d import config, field3d, mesh3d, optics3d, sections

# V0-V4 default to CPU ON PURPOSE: they use tiny meshes, run in ~5 s, and this
# keeps them safe on a 2 GB laptop GPU. That means they show NO NVIDIA activity
# in Task Manager — which is expected, not a misconfiguration. Force the GPU
# with RAIL3D_TEST_DEVICE=cuda:0 if you want to check the CUDA path.
DEVICE = torch.device(os.environ.get("RAIL3D_TEST_DEVICE", "cpu"))
REPORT_PATH = config.GENERATED_DIR / "verification_report.json"


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm())


# ---------------------------------------------------------------------------
# Verbatim reference implementation (Face3D_clean FieldCalculation.py)
# ---------------------------------------------------------------------------
def farfield_from_antenna_2nd_reference(v, f, X, Y, H, wvl, theta_inc, size_ant,
                                        dist_ant, resol_ant):
    """Copied verbatim from Face3D_clean (lower_half branch dropped)."""
    k0 = 2 * np.pi / wvl
    dir_front = torch.tensor([0, 0, 1], dtype=torch.float).reshape(3, -1)
    dir = field3d.incident_direction(theta_inc, 0).reshape(3, -1)

    centerpoint, area, normal = field3d.surface_geometry(v, f)
    area = area.unsqueeze(-1)

    front_idx = (normal @ dir_front > 0).reshape(-1)
    left_idx = (normal @ dir > 0).reshape(-1)
    idx = left_idx * front_idx

    X_obj = centerpoint[:, 0].reshape(-1, 1, 1)
    Y_obj = centerpoint[:, 1].reshape(-1, 1, 1)
    Z_obj = centerpoint[:, 2].reshape(-1, 1, 1)

    psi0_ant, x_ant, y_ant, z_ant, A_aptr, B_aptr = field3d.aperture_field(
        size_ant, dist_ant, resol_ant, theta_inc, wvl, k0
    )
    X_ant = x_ant.reshape(-1, resol_ant, resol_ant)
    Y_ant = y_ant.reshape(-1, resol_ant, resol_ant)
    Z_ant = z_ant.reshape(-1, resol_ant, resol_ant)

    R_antobj = torch.sqrt((X_obj - X_ant) ** 2 + (Y_obj - Y_ant) ** 2 + (Z_obj - Z_ant) ** 2)
    obj2ant_dist = -(X_obj - X_ant) * np.sin(theta_inc) - (Z_obj - Z_ant) * np.cos(theta_inc)

    dS_ant = A_aptr * B_aptr / (resol_ant - 1) ** 2
    kernel = (1 / wvl) * (1 / (k0 * R_antobj) - 1j) * (obj2ant_dist / R_antobj**2) * torch.exp(1j * k0 * R_antobj)
    psi0_face = (psi0_ant * dS_ant * kernel).sum(dim=(1, 2))

    X_obj = centerpoint[idx, 0].reshape(-1, 1, 1)
    Y_obj = centerpoint[idx, 1].reshape(-1, 1, 1)
    Z_obj = centerpoint[idx, 2].reshape(-1, 1, 1)
    R = torch.sqrt((X - X_obj) ** 2 + (Y - Y_obj) ** 2 + (H - Z_obj) ** 2)
    point2plane = ((X - X_obj) * normal[idx, 0].reshape(-1, 1, 1)
                   + (Y - Y_obj) * normal[idx, 1].reshape(-1, 1, 1)
                   + (H - Z_obj) * normal[idx, 2].reshape(-1, 1, 1))
    point2plane = point2plane * (point2plane > 0)
    dS = area[idx].reshape(-1, 1, 1)
    kernel = (1 / wvl) * (1 / (k0 * R) - 1j) * (point2plane / R**2) * torch.exp(1j * k0 * R)
    psi1_ms = -(psi0_face[idx].reshape(-1, 1, 1) * dS * kernel).sum(dim=0)

    X_obj = centerpoint[left_idx, 0].reshape(-1, 1, 1)
    Y_obj = centerpoint[left_idx, 1].reshape(-1, 1, 1)
    Z_obj = centerpoint[left_idx, 2].reshape(-1, 1, 1)
    R = torch.sqrt((X_ant - X_obj) ** 2 + (Y_ant - Y_obj) ** 2 + (Z_ant - Z_obj) ** 2)
    point2plane = ((X_ant - X_obj) * normal[left_idx, 0].reshape(-1, 1, 1)
                   + (Y_ant - Y_obj) * normal[left_idx, 1].reshape(-1, 1, 1)
                   + (Z_ant - Z_obj) * normal[left_idx, 2].reshape(-1, 1, 1))
    point2plane = point2plane * (point2plane > 0)
    dS = area[left_idx].reshape(-1, 1, 1)
    kernel = (1 / wvl) * (1 / (k0 * R) - 1j) * (point2plane / R**2) * torch.exp(1j * k0 * R)
    psi1_ant = -(psi0_face[left_idx].reshape(-1, 1, 1) * dS * kernel).sum(dim=0)

    X_ant = X_ant.reshape(resol_ant, resol_ant, 1, 1)
    Y_ant = Y_ant.reshape(resol_ant, resol_ant, 1, 1)
    Z_ant = Z_ant.reshape(resol_ant, resol_ant, 1, 1)
    resol1, resol2 = X.shape[1], X.shape[2]
    X = X.reshape(1, 1, resol1, resol2)
    Y = Y.reshape(1, 1, resol1, resol2)
    R = torch.sqrt((X - X_ant) ** 2 + (Y - Y_ant) ** 2 + (H - Z_ant) ** 2)
    point2plane = -((X - X_ant) * dir[0, 0] + (Y - Y_ant) * dir[1, 0] + (H - Z_ant) * dir[2, 0])
    point2plane = point2plane * (point2plane > 0)
    kernel = (1 / wvl) * (1 / (k0 * R) - 1j) * (point2plane / R**2) * torch.exp(1j * k0 * R)
    psi2_ms = -(psi1_ant.reshape(resol_ant, resol_ant, 1, 1) * dS_ant * kernel).sum(dim=(0, 1))
    psi0_ms = (psi0_ant.reshape(resol_ant, resol_ant, 1, 1) * dS_ant * kernel).sum(dim=(0, 1))

    return psi0_ms, psi1_ms, psi2_ms


def _small_rail_mesh(seg_len: float = 40.0):
    section = sections.load_reference_section()
    return mesh3d.sweep_rail_mesh(section, seg_len=seg_len)


# ---------------------------------------------------------------------------
# V0 — rev.2 defect geometry unit checks (CPU, no field solves)
# ---------------------------------------------------------------------------
def test_v0_geometry() -> dict:
    """Depth-field constructors behave as specified:
    crack orientation moves the line the right way, bands confine footprints,
    depth fields are resolution independent, and seeds reproduce exactly."""
    import numpy as np

    section = sections.load_reference_section()
    geom = mesh3d.arc_geometry(section, mesh3d.default_arc_count(section, 1.0))
    y = np.arange(-120.0, 120.5, 1.0)
    res = {}

    # (a) crack orientations: extent along y vs s must follow theta
    base = {"class": "crack", "y0": 0.0, "s0": float(geom["s"][len(geom["s"]) // 2]),
            "L": 24.0, "depth": 4.0, "offsets": [0.0], "band": "crack",
            "profile_v": np.linspace(-1.0, 1.0, 9),
            "profile_d": np.maximum(0.0, 1.0 - np.abs(np.linspace(-1, 1, 9)))}

    def extent(theta):
        d = mesh3d.render_depth_field({**base, "theta": theta}, geom, y)
        m = d > 0.5
        ys = np.flatnonzero(m.any(axis=1))
        ss = np.flatnonzero(m.any(axis=0))
        return (int(ys[-1] - ys[0] + 1) if len(ys) else 0,
                float(geom["s"][ss[-1]] - geom["s"][ss[0]]) if len(ss) else 0.0)

    ey_long, es_long = extent(0.0)
    ey_tr, es_tr = extent(np.pi / 2)
    ey_ob, es_ob = extent(np.pi / 4)
    res["long_extent_y_mm"], res["long_extent_s_mm"] = ey_long, es_long
    res["trans_extent_y_mm"], res["trans_extent_s_mm"] = ey_tr, es_tr
    ok_orient = (ey_long > 15 and es_long < 5 and         # line along y, hairline in s
                 ey_tr < 5 and es_tr > 15 and             # line along s
                 ey_ob > 10 and es_ob > 10)               # diagonal spans both
    res["orientations_ok"] = bool(ok_orient)

    # (b) bands: dent footprint on the running band, wear/shell on the horn side,
    #     nothing on downward-facing surface (nz < -0.3)
    band_ok = True
    for cls in ("crack", "dent", "wear", "shell"):
        defect = None
        if cls in config.DATASET_DIRS:
            files = sections.get_dataset_files(cls)
            defect = sections.match_reference_width(
                sections.load_vertices_from_csv(files[0]), section)
        p, _ = mesh3d.defect_params_for_sample(section, cls, defect,
                                               config.sample_seed(cls, 0),
                                               n_arc=geom["n_arc"])
        d = mesh3d.render_depth_field(p, geom, y)
        if d.max() <= 0:
            band_ok = False
            res[f"{cls}_empty"] = True
            continue
        cols = d.max(axis=0) > 0.1 * d.max()
        nz = geom["normal"][cols, 1]
        if (nz < -0.3).any():
            band_ok = False
            res[f"{cls}_on_underside"] = True
        if cls in ("wear", "shell") and geom["x"][cols].min() < config.GAUGE_X_MIN - 5:
            band_ok = False
            res[f"{cls}_off_gauge_band"] = True
    res["bands_ok"] = bool(band_ok)

    # (c) resolution independence + seed reproducibility (shell has the most
    #     internal randomness — exercise it)
    p1, _ = mesh3d.defect_params_for_sample(section, "shell", None,
                                            config.sample_seed("shell", 7),
                                            n_arc=geom["n_arc"])
    p2, _ = mesh3d.defect_params_for_sample(section, "shell", None,
                                            config.sample_seed("shell", 7),
                                            n_arc=geom["n_arc"])
    d_fine = mesh3d.render_depth_field(p1, geom, y)
    d_fine2 = mesh3d.render_depth_field(p2, geom, y)
    res["seed_reproducible"] = bool(np.array_equal(d_fine, d_fine2))

    geom_c = mesh3d.arc_geometry(section, mesh3d.default_arc_count(section, 4.0))
    y_c = np.arange(-120.0, 120.5, 4.0)
    d_coarse = mesh3d.render_depth_field(p1, geom_c, y_c)
    # compare coarse rendering against fine rendering subsampled at nearest pts
    ii = [int(np.argmin(np.abs(geom["s"] - sc))) for sc in geom_c["s"]]
    jj = [int(np.argmin(np.abs(y - yc))) for yc in y_c]
    diff = np.abs(d_coarse - d_fine[np.ix_(jj, ii)]).max()
    res["resolution_consistency_mm"] = float(diff)
    res["resolution_ok"] = bool(diff < 0.35)   # interp differences only, no shape change

    res["pass"] = bool(ok_orient and band_ok and res["seed_reproducible"]
                       and res["resolution_ok"])
    return res


# ---------------------------------------------------------------------------
# V1 — solver equivalence + chunk/batch invariance
# ---------------------------------------------------------------------------
def test_v1_solver_equivalence() -> dict:
    v, f = _small_rail_mesh()
    v, f = v.to(DEVICE), f.to(DEVICE)
    X, Y = config.plane_grid(DEVICE)
    args = (X, Y, config.H_MS, config.WVL, config.THETA_INC,
            config.SIZE_ANT, config.DIST_ANT, config.RESOL_ANT)

    psi0_ref, psi1_ref, psi2_ref = farfield_from_antenna_2nd_reference(v, f, *args)
    psi0_new = field3d.horn_to_plane(*args)
    psi1_new, psi2_new = field3d.scattered_fields(v, f, *args, chunk_faces=256)
    psi1_big, psi2_big = field3d.scattered_fields(v, f, *args, chunk_faces=4096)
    psi1_b, psi2_b = field3d.scattered_fields(torch.stack([v, v]), f, *args, chunk_faces=256)

    res = {
        "psi0": rel_l2(psi0_new, psi0_ref),
        "psi1": rel_l2(psi1_new, psi1_ref),
        "psi2": rel_l2(psi2_new, psi2_ref),
        "chunk_invariance": rel_l2(psi1_big, psi1_new),
        "batch_consistency": max(rel_l2(psi1_b[0], psi1_new), rel_l2(psi1_b[1], psi1_new)),
        "n_faces": int(f.shape[0]),
    }
    res["pass"] = all(res[k] < 1e-5 for k in ("psi0", "psi1", "psi2", "chunk_invariance", "batch_consistency"))
    return res


# ---------------------------------------------------------------------------
# V2 — FFT propagator vs conv2d
# ---------------------------------------------------------------------------
def test_v2_propagator_equivalence() -> dict:
    torch.manual_seed(0)
    prop = optics3d.PropagatorRSFFT(config.LAYER_DISTANCES[-1]).to(DEVICE)
    worst = 0.0
    for _ in range(10):
        sig = (torch.randn(1, config.NX, config.NY) + 1j * torch.randn(1, config.NX, config.NY)).to(DEVICE)
        worst = max(worst, rel_l2(prop(sig), prop.forward_direct(sig)))
    return {"max_rel_l2": worst, "pass": worst < 1e-4}


# ---------------------------------------------------------------------------
# V3 — angular spectrum vs RS-FFT (Gaussian beam, central aperture)
# ---------------------------------------------------------------------------
def test_v3_asm_vs_rs() -> dict:
    X, Y = config.plane_grid(DEVICE)
    X, Y = X[0], Y[0]
    beam = torch.exp(-(X**2 + Y**2) / (2 * 25.0**2)) + 0j
    rs = optics3d.PropagatorRSFFT(config.LAYER_DISTANCES[-1]).to(DEVICE)
    asm = optics3d.PropagatorASM2D(config.LAYER_DISTANCES[-1], pad_factor=4).to(DEVICE)
    out_rs = rs(beam.unsqueeze(0))[0]
    out_asm = asm(beam.unsqueeze(0))[0]
    cx = slice(int(0.2 * config.NX), int(0.8 * config.NX))
    cy = slice(int(0.2 * config.NY), int(0.8 * config.NY))
    err = rel_l2(out_asm[cx, cy], out_rs[cx, cy])
    return {"central_rel_l2": err, "pass": err < 0.02}


# ---------------------------------------------------------------------------
# V4 — physics sanity
# ---------------------------------------------------------------------------
def test_v4_sanity() -> dict:
    res = {}

    # (a) flat plate under normal incidence -> specular peak at plane center
    nx_p, nz = 30, 30
    gx = torch.linspace(-60, 60, nx_p)
    gy = torch.linspace(-60, 60, nz)
    GX, GY = torch.meshgrid(gx, gy, indexing="ij")
    nv = nx_p * nz
    verts = torch.stack([GX.reshape(-1), GY.reshape(-1), torch.zeros(nv)], dim=1)
    ii = torch.arange(nx_p - 1)
    jj = torch.arange(nz - 1)
    J, I = torch.meshgrid(jj, ii, indexing="ij")
    a = I * nz + J
    b = (I + 1) * nz + J
    c = I * nz + J + 1
    d = (I + 1) * nz + J + 1
    f = torch.cat([torch.stack([a, b, c], -1).reshape(-1, 3),
                   torch.stack([b, d, c], -1).reshape(-1, 3)])
    v_, f_ = verts.to(DEVICE), f.to(DEVICE)
    X, Y = config.plane_grid(DEVICE)
    psi1, _ = field3d.scattered_fields(
        v_, f_, X, Y, config.H_MS, config.WVL, 1e-6,   # theta ~ 0: overhead horn
        config.SIZE_ANT, config.DIST_ANT, config.RESOL_ANT,
        chunk_faces=512, compute_psi2=False,
    )
    intensity = psi1.abs() ** 2
    peak = torch.nonzero(intensity == intensity.max())[0]
    center = torch.tensor([config.NX // 2, config.NY // 2], device=DEVICE)
    res["specular_peak_offset_px"] = float((peak - center).abs().max())
    res["specular_ok"] = res["specular_peak_offset_px"] <= 2

    # (b) power conservation of the trainable propagator
    Xg, Yg = config.plane_grid(DEVICE)
    beam = torch.exp(-(Xg[0] ** 2 + Yg[0] ** 2) / (2 * 20.0**2)) + 0j
    prop = optics3d.PropagatorRSFFT(config.LAYER_DISTANCES[-1]).to(DEVICE)
    out = prop(beam.unsqueeze(0))[0]
    ratio = float((out.abs() ** 2).sum() / (beam.abs() ** 2).sum())
    res["power_ratio"] = ratio
    res["power_ok"] = 0.85 < ratio < 1.02

    # (c) mesh orientation: crown faces point up
    v, f = _small_rail_mesh()
    center_face, _, normal = field3d.surface_geometry(v, f)
    crown = (center_face[:, 0].abs() < 20.0) & (center_face[:, 2] > -20.0)
    frac_up = float((normal[crown, 2] > 0).float().mean())
    res["crown_up_fraction"] = frac_up
    res["orientation_ok"] = frac_up > 0.99

    res["pass"] = bool(res["specular_ok"] and res["power_ok"] and res["orientation_ok"])
    return res


# ---------------------------------------------------------------------------
def main() -> int:
    warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")
    config.ensure_dirs()
    print(f"[rail3d] V0-V4 running on {DEVICE}"
          + ("  (CPU by design - set RAIL3D_TEST_DEVICE=cuda:0 to use the GPU)"
             if DEVICE.type == "cpu" else
             f"  ({torch.cuda.get_device_name(DEVICE.index or 0)})"))
    report = {"device": str(DEVICE), "torch": torch.__version__}
    ok = True
    for name, fn in [
        ("V0_geometry", test_v0_geometry),
        ("V1_solver_equivalence", test_v1_solver_equivalence),
        ("V2_propagator_equivalence", test_v2_propagator_equivalence),
        ("V3_asm_vs_rs", test_v3_asm_vs_rs),
        ("V4_sanity", test_v4_sanity),
    ]:
        t0 = time.time()
        try:
            res = fn()
        except Exception as err:  # noqa: BLE001
            res = {"pass": False, "error": repr(err)}
        res["seconds"] = round(time.time() - t0, 2)
        report[name] = res
        ok &= bool(res["pass"])
        status = "PASS" if res["pass"] else "FAIL"
        detail = {k: v for k, v in res.items() if k not in ("pass",)}
        print(f"[{status}] {name}: {detail}")

    with open(REPORT_PATH, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"report -> {REPORT_PATH}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
