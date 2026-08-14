"""V5-V7 physics validation with figures (laptop-safe sizes).

V5 — 2D<->3D consistency: a y-uniform extruded intact rail's central intensity
     profile vs the established 2D Hankel boundary-integral method
     (re-parameterized to the 3D geometry: λ=8 mm, 55°, horn at 224 mm,
     plane 160 mm above the crown).
V6 — shadowing: ray-cast line-of-sight vs back-face culling only, on the
     deepest dent/wear samples.
V7 — mesh convergence: λ/2 vs λ/4 sampling.

Run:  python validation_3d.py [--device cuda:0]
Figures land in rail3D/data/figures/, metrics appended to the verification
report.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rail3d import config, field3d, mesh3d, sections

REPORT_PATH = config.GENERATED_DIR / "verification_report.json"


# ---------------------------------------------------------------------------
# V5 — 2D Hankel boundary-integral reference (adapted from 2Dmesh_from_vertex)
# ---------------------------------------------------------------------------
def hankel(x: torch.Tensor) -> torch.Tensor:
    return torch.special.bessel_j1(x) + 1j * torch.special.bessel_y1(x)


def field_2d_reference(section: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The 2D pipeline's two-stage Kirchhoff integral, re-parameterized to the
    3D geometry (λ=8 mm, horn 224 mm @ 55° on the +x side, MS plane 160 mm
    above the crown, 60 samples at dx=4 mm). Returns (x_ms, psi_ms).

    Kernel, illumination test, and segment-intersection shadowing are copied
    from 2Dmesh_from_vertex.field / compute_line_of_sight_mask_v3 (units mm).
    """
    wvl = config.WVL
    k0 = 2 * np.pi / wvl
    theta = config.THETA_INC

    # horn aperture, central (y=0) cut of the 3D horn: A-axis in the x-z plane
    A_aptr, _, a_wvg, _, l_horn = config.SIZE_ANT
    R_E = A_aptr * l_horn / (A_aptr - a_wvg)
    beta_wvg = k0 * np.sqrt(1 - (wvl / (2 * a_wvg)) ** 2)
    n_src = 50
    dx_src = A_aptr / n_src
    s = torch.arange(-A_aptr / 2, A_aptr / 2, dx_src) + dx_src / 2
    # source center in section coords (x, y2d): crown at y2d = RAIL_HEIGHT
    cx = config.DIST_ANT * np.sin(theta)
    cy = config.RAIL_HEIGHT + config.DIST_ANT * np.cos(theta)
    v_src = torch.stack([cx - s * np.cos(theta), cy + s * np.sin(theta)], dim=1)
    norm_vec_src = torch.tensor([[-np.sin(theta), -np.cos(theta)]], dtype=torch.float32)
    psi_src = torch.cos(np.pi * s / A_aptr) * torch.exp(0.5j * beta_wvg * (s**2 / R_E))

    # metasurface line
    x_ms = torch.arange(-config.WX / 2 + config.DX / 2, config.WX / 2, config.DX)
    v_ms = torch.stack([x_ms, torch.full_like(x_ms, config.RAIL_HEIGHT + config.H_MS)], dim=1)

    boundary = section
    mask = sections.illuminated_mask(section)
    v_rail = boundary[mask]

    center = (v_rail[:-1] + v_rail[1:]) / 2
    dvec = v_rail[1:] - v_rail[:-1]
    seg = torch.linalg.norm(dvec, dim=1)
    norm_vec = torch.stack([dvec[:, 1], -dvec[:, 0]], dim=1)
    norm_vec = norm_vec / torch.linalg.norm(norm_vec, dim=1, keepdim=True)

    def los_mask(origins, targets):
        # segment-vs-boundary-edge intersection (v3 method)
        seg_start = origins[:, None, :]
        seg_end = targets[None, :, :]
        seg_dir = seg_end - seg_start
        edge_start = boundary
        edge_end = torch.roll(boundary, shifts=-1, dims=0)
        edge_dir = edge_end - edge_start
        p = seg_start[:, :, None, :]
        r = seg_dir[:, :, None, :]
        q = edge_start[None, None, :, :]
        sv = edge_dir[None, None, :, :]

        def cross2d(a, b):
            return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

        qp = q - p
        r_cross_s = cross2d(r, sv)
        qp_cross_r = cross2d(qp, r)
        qp_cross_s = cross2d(qp, sv)
        eps = 1e-6
        nonpar = r_cross_s.abs() > eps
        t = torch.where(nonpar, qp_cross_s / r_cross_s, torch.zeros_like(r_cross_s))
        u = torch.where(nonpar, qp_cross_r / r_cross_s, torch.zeros_like(r_cross_s))
        proper = nonpar & (t > eps) & (t < 1 - eps) & (u > eps) & (u < 1 - eps)
        return ~proper.any(dim=2)

    # source -> rail
    R_vec = center.unsqueeze(1) - v_src.reshape(1, -1, 2)
    R = torch.linalg.norm(R_vec, dim=2)
    R_unit = R_vec / R.unsqueeze(2)
    illum = torch.sum(-R_vec * norm_vec.unsqueeze(1), dim=2) < 0
    illum = illum & los_mask(v_src, center).T
    cosf = torch.sum(R_unit * norm_vec_src.reshape(1, 1, 2), dim=2)
    psi_rail = torch.sum(
        (1j * k0 / 4) * hankel(k0 * R) * illum * cosf * psi_src.reshape(1, -1) * dx_src,
        dim=1,
    )

    # rail -> ms
    R_vec = v_ms.reshape(1, -1, 2) - center.unsqueeze(1)
    R = torch.linalg.norm(R_vec, dim=2)
    R_unit = R_vec / R.unsqueeze(2)
    illum = torch.sum(R_vec * norm_vec.unsqueeze(1), dim=2) < 0
    illum = illum & los_mask(center, v_ms)
    cosf = torch.sum(R_unit * norm_vec.unsqueeze(1), dim=2)
    psi_ms = torch.sum(
        (1j * k0 / 4) * hankel(k0 * R) * illum * cosf * (-psi_rail * seg).unsqueeze(1),
        dim=0,
    )
    return x_ms, psi_ms


def v5_2d_3d_consistency(device: torch.device) -> dict:
    section = sections.load_reference_section()
    x_ms, psi_2d = field_2d_reference(section)

    # One-off physics gate: fine mesh (λ/16) so mesh-discretization error does
    # not mask solver disagreement; production-mesh fidelity is V7's job.
    fine = config.WVL / 16
    v, f = mesh3d.sweep_rail_mesh(section, slice_ds=fine, arc_ds=fine)
    v_occ, f_occ = mesh3d.sweep_rail_mesh(section)
    X, Y = config.plane_grid(device)
    psi1, _ = field3d.scattered_fields(
        v.to(device), f.to(device), X, Y, config.H_MS, config.WVL, config.THETA_INC,
        config.SIZE_ANT, config.DIST_ANT, config.RESOL_ANT,
        chunk_faces=2048, compute_psi2=False,
        shadow=config.SHADOW_MODE, shadow_occluders=(v_occ.to(device), f_occ.to(device)),
    )
    prof_3d = (psi1.abs() ** 2)[:, config.NY // 2].cpu().numpy()
    prof_2d = (psi_2d.abs() ** 2).numpy()

    prof_3d_n = prof_3d / prof_3d.max()
    prof_2d_n = prof_2d / prof_2d.max()
    r = float(np.corrcoef(prof_3d_n, prof_2d_n)[0, 1])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_ms.numpy(), prof_2d_n, label="2D Hankel boundary integral")
    ax.plot(x_ms.numpy(), prof_3d_n, label="3D RS surface integral (central row, y=0)")
    ax.set(xlabel="x (mm)", ylabel="normalized |psi|²",
           title=f"V5: 2D vs 3D intact-rail intensity at the MS plane — Pearson r = {r:.3f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "v5_2d_vs_3d.png", dpi=200)
    return {"pearson_r": r, "pass": r > 0.9}


# ---------------------------------------------------------------------------
# V6 — shadowing check
# ---------------------------------------------------------------------------
def _solve_sample(section, class_name, sample_idx, device, args,
                  ds=None, shadow=True, roll=0.0, jit=(0.0, 0.0),
                  params=None):
    """Solve one sample's psi1 with the rev.2 depth-field geometry.

    ``params=None`` draws the defect from the sample seed (CSV classes load
    their CSV); pass params explicitly to re-render the same defect at another
    resolution. roll/jit override the augmentation (0 = unaugmented)."""
    ds = ds or config.MESH_DS
    if params is None:
        defect = None
        if class_name in config.DATASET_DIRS:
            files = sections.get_dataset_files(class_name)
            defect = sections.match_reference_width(
                sections.load_vertices_from_csv(files[sample_idx]), section)
        n_arc_fine = mesh3d.default_arc_count(section, config.MESH_DS)
        params, _ = mesh3d.defect_params_for_sample(
            section, class_name, defect, config.sample_seed(class_name, sample_idx),
            n_arc=n_arc_fine)
    v, f = mesh3d.sweep_rail_mesh(section, defect_params=params,
                                  slice_ds=ds, arc_ds=ds, roll_deg=roll, jitter_xz=jit)
    kw = {}
    if shadow:
        v_occ, f_occ = mesh3d.sweep_rail_mesh(section, defect_params=params,
                                              slice_ds=config.OCCLUDER_DS,
                                              arc_ds=config.OCCLUDER_DS,
                                              roll_deg=roll, jitter_xz=jit)
        kw = {"shadow": "raycast",
              "shadow_occluders": (v_occ.to(device), f_occ.to(device))}
    psi, _ = field3d.scattered_fields(v.to(device), f.to(device), *args,
                                      chunk_faces=2048, compute_psi2=False, **kw)
    return psi, params


def v6_shadowing(device: torch.device, n_per_class: int = 2) -> dict:
    """Shadowing at the generation config (λ/8 mesh, λ/2 occluders, min_t=3).

    * effect size: raycast-vs-none rel L2 on deep samples of all four classes
      (informational — crack craters give a real few-% effect);
    * artifact guard: on AUGMENTED intact meshes the raycast must stay within
      3% of no-shadow (2D LOS ground truth: intact shadowing ~1%).
    """
    section = sections.load_reference_section()
    X, Y = config.plane_grid(device)
    args = (X, Y, config.H_MS, config.WVL, config.THETA_INC,
            config.SIZE_ANT, config.DIST_ANT, config.RESOL_ANT)

    per_sample, worst = [], 0.0
    for cls in config.CLASS_NAMES:
        for k in range(n_per_class):
            a, params = _solve_sample(section, cls, k, device, args, shadow=True)
            b, _ = _solve_sample(section, cls, k, device, args, shadow=False,
                                 params=params)
            rel = float((a - b).norm() / b.norm())
            per_sample.append({"class": cls, "idx": k,
                               "depth_mm": params.get("depth", 0.0), "rel_l2": rel})
            worst = max(worst, rel)

    intact_params = {"class": "intact"}
    artifact_worst = 0.0
    for roll, jit in [(2.0, (4.0, -4.0)), (-2.0, (-4.0, 4.0)), (1.0, (0.0, 0.0))]:
        a, _ = _solve_sample(section, "intact", 0, device, args, shadow=True,
                             roll=roll, jit=jit, params=intact_params)
        b, _ = _solve_sample(section, "intact", 0, device, args, shadow=False,
                             roll=roll, jit=jit, params=intact_params)
        artifact_worst = max(artifact_worst, float((a - b).norm() / b.norm()))

    # Informational: the generation occluder is λ/2, which CANNOT represent a
    # ~2 mm hairline crack — so shadowing is inert for cracks at the production
    # config (worst_rel_l2 = 0). Re-measure one crack against a generation-
    # resolution occluder to record the true magnitude of what we are omitting.
    files = sections.get_dataset_files("crack")
    d0 = sections.match_reference_width(sections.load_vertices_from_csv(files[0]), section)
    n_arc_fine = mesh3d.default_arc_count(section, config.MESH_DS)
    crack_params, _ = mesh3d.defect_params_for_sample(
        section, "crack", d0, config.sample_seed("crack", 0), n_arc=n_arc_fine)
    v_f, f_f = mesh3d.sweep_rail_mesh(section, defect_params=crack_params,
                                      slice_ds=config.MESH_DS, arc_ds=config.MESH_DS)
    no_sh, _ = field3d.scattered_fields(v_f.to(device), f_f.to(device), *args,
                                        chunk_faces=2048, compute_psi2=False)
    with_sh, _ = field3d.scattered_fields(
        v_f.to(device), f_f.to(device), *args, chunk_faces=2048, compute_psi2=False,
        shadow="raycast", shadow_occluders=(v_f.to(device), f_f.to(device)))
    resolved = float((with_sh - no_sh).norm() / no_sh.norm())

    return {"worst_rel_l2": worst, "samples": per_sample,
            "intact_augmented_artifact": artifact_worst,
            "crack_shadow_with_resolved_occluder": resolved,
            "generation_shadow_mode": config.SHADOW_MODE,
            "note": ("lambda/2 occluders cannot resolve hairline cracks, so the "
                     "production shadow test is inert for them; the resolved-occluder "
                     "figure is the magnitude being omitted (accepted if < 0.02)"),
            "pass": artifact_worst < 0.03 and resolved < 0.02}


# ---------------------------------------------------------------------------
# V7 — mesh convergence
# ---------------------------------------------------------------------------
def v7_mesh_convergence(device: torch.device) -> dict:
    """Detector-level convergence of the generation mesh (λ/8) vs λ/16.

    Raw complex-field L2 does not converge at these facet sizes (PO glint
    speckle); the task consumes detector barcodes, so the criterion is the
    defect-signal direction cosine between resolutions. Cracks (2 mm hairlines,
    the sharpest features of the rev.2 geometry) get the most samples; one
    shell + one dent + one wear round out the check.
    """
    from rail3d import optics3d

    section = sections.load_reference_section()
    X, Y = config.plane_grid(device)
    args = (X, Y, config.H_MS, config.WVL, config.THETA_INC,
            config.SIZE_ANT, config.DIST_ANT, config.RESOL_ANT)
    psi0 = field3d.horn_to_plane(*args)
    det = optics3d.SoftDetector2D().to(device)
    fine = config.WVL / 16

    def barcode(cls, idx, ds, params=None):
        if params is None and cls == "intact":
            params = {"class": "intact"}
        psi, params = _solve_sample(section, cls, idx, device, args, ds=ds,
                                    shadow=True, params=params)
        return det.hard_powers(((psi0 + psi).abs() ** 2).unsqueeze(0))[0], params

    b_i8, ip = barcode("intact", 0, config.MESH_DS)
    b_i16, _ = barcode("intact", 0, fine, params=ip)
    intact_err = float((b_i8 - b_i16).norm() / b_i16.norm())

    cases = [("crack", 0), ("crack", 1), ("crack", 2), ("dent", 0),
             ("wear", 0), ("shell", 0)]
    cosines, labels = [], []
    for cls, idx in cases:
        s8, params = barcode(cls, idx, config.MESH_DS)
        s16, _ = barcode(cls, idx, fine, params=params)
        sig8, sig16 = s8 - b_i8, s16 - b_i16
        cosines.append(float(torch.dot(sig8, sig16) / (sig8.norm() * sig16.norm())))
        labels.append(f"{cls}{idx}")

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.bar(labels, cosines)
    ax.axhline(0.95, color="red", ls="--", label="0.95 target")
    ax.set(ylabel="signal cosine (λ/8 vs λ/16)",
           title="V7: defect-signal consistency of the generation mesh", ylim=(0, 1.05))
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "v7_convergence.png", dpi=200)

    mean_cos = float(np.mean(cosines))
    return {"cases": dict(zip(labels, cosines)), "min_cosine": min(cosines),
            "mean_cosine": mean_cos, "intact_barcode_rel_err": intact_err,
            "pass": mean_cos > 0.95 and min(cosines) > 0.90}


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None,
                        help="e.g. cuda:0; default = RAIL3D_DEVICE, else strongest CUDA card")
    args = parser.parse_args()
    device = torch.device(args.device) if args.device else config.get_device("lab")
    config.ensure_dirs()

    report = {}
    if REPORT_PATH.exists():
        report = json.loads(REPORT_PATH.read_text())

    ok = True
    for name, fn in [("V5_2d_3d_consistency", v5_2d_3d_consistency),
                     ("V6_shadowing", v6_shadowing),
                     ("V7_mesh_convergence", v7_mesh_convergence)]:
        t0 = time.time()
        try:
            res = fn(device)
        except Exception as err:  # noqa: BLE001
            res = {"pass": False, "error": repr(err)}
        res["seconds"] = round(time.time() - t0, 2)
        report[name] = res
        ok &= bool(res["pass"])
        status = "PASS" if res["pass"] else "FAIL"
        brief = {k: v for k, v in res.items() if k not in ("pass", "samples")}
        print(f"[{status}] {name}: {brief}")

    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"report -> {REPORT_PATH}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
