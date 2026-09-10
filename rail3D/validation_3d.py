"""V5-V7 physics validation with figures (laptop-safe sizes).

V5 — 2D<->3D consistency: a y-uniform extruded intact rail's central intensity
     profile vs the established 2D Hankel boundary-integral method
     (re-parameterized to the ACTIVE 3D geometry from config: wavelength,
     incidence, horn distance and plane height all read from config.py).
V6 — shadowing: ray-cast line-of-sight vs back-face culling only, on the
     deepest dent/wear samples.
V7 — mesh convergence: the generation mesh (MESH_DS = λ/8) vs a λ/16
     reference, judged at the detector-barcode level.

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
    active 3D geometry (everything — λ, horn, incidence, plane height, grid —
    is read from config below). Returns (x_ms, psi_ms).

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
                  params=None, min_t=None, occ_ds=None, normal_offset=None):
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
        occ = occ_ds or config.OCCLUDER_DS
        v_occ, f_occ = mesh3d.sweep_rail_mesh(section, defect_params=params,
                                              slice_ds=occ, arc_ds=occ,
                                              roll_deg=roll, jitter_xz=jit)
        kw = {"shadow": "raycast",
              "shadow_occluders": (v_occ.to(device), f_occ.to(device)),
              "shadow_min_t": config.SHADOW_MIN_T if min_t is None else min_t,
              "shadow_normal_offset": (config.SHADOW_NORMAL_OFFSET
                                       if normal_offset is None else normal_offset)}
    psi, _ = field3d.scattered_fields(v.to(device), f.to(device), *args,
                                      chunk_faces=2048, compute_psi2=False, **kw)
    return psi, params


def v6_shadowing(device: torch.device, n_per_class: int = 2) -> dict:
    """Shadowing at the generation config (λ/8 mesh, λ/2 occluders, config.SHADOW_MIN_T).

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
        shadow="raycast", shadow_occluders=(v_f.to(device), f_f.to(device)),
        shadow_min_t=config.SHADOW_MIN_T)
    resolved = float((with_sh - no_sh).norm() / no_sh.norm())

    return {"worst_rel_l2": worst, "samples": per_sample,
            "intact_augmented_artifact": artifact_worst,
            "crack_shadow_with_resolved_occluder": resolved,
            "generation_shadow_mode": config.SHADOW_MODE,
            "shadow_min_t": config.SHADOW_MIN_T,
            "note": ("lambda/2 occluders cannot resolve hairline cracks, so the "
                     "production shadow test is inert for them; the resolved-occluder "
                     "figure is the magnitude being omitted (accepted if < 0.02)"),
            "pass": artifact_worst < 0.03 and resolved < 0.02}


# ---------------------------------------------------------------------------
# V6b — min_t sweep (decides config.SHADOW_MIN_T; not a gate)
# ---------------------------------------------------------------------------
def _barcode(psi, device):
    """Detector powers — the level README finding 2 says to judge fidelity at."""
    from rail3d import optics3d
    det = optics3d.SoftDetector2D().to(device)
    return det.hard_powers((psi.abs() ** 2).unsqueeze(0))[0]


def _cos(a, b):
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-12))


def v6b_guard_sweep(device: torch.device, min_ts, offsets, n_intact: int = 5,
                    crack_idxs=(0, 1)) -> dict:
    """Cost/benefit of the ray-cast self-hit guard, over min_t AND normal offset.

    The guard exists to kill a discretization artifact: facet chords sag inside
    the true convex surface, so grazing rays clip their own neighbours. Two
    knobs control it —

      min_t          ignore hits closer than this ALONG THE RAY
      normal_offset  lift the ray origin along the face NORMAL before casting

    The 2026-09-04 sweep showed min_t alone CANNOT work: on augmented intact
    meshes the artifact saturates for any min_t <= 1 facet while crack
    self-shadowing only appears below ~1.5 mm, so the two populations overlap
    in ray distance and no threshold separates them. The normal offset attacks
    the cause instead (the 1 um along-ray nudge has no perpendicular component
    at grazing incidence), so it should suppress the artifact while LEAVING
    distant real occluders untouched. This sweep tests exactly that.

    Reported per configuration:
      artifact_mean / artifact_max  on AUGMENTED INTACT meshes, where a convex
                                    rail cannot shadow itself so every bit of
                                    change is error. Mean over n_intact
                                    augmentations (max-of-3 was a noisy
                                    statistic dominated by one unlucky roll).
      artifact_barcode              the same error THROUGH THE DETECTORS —
                                    finding 2 says field L2 never converges, so
                                    this is the number that reflects impact.
      crack_effect_*                real crater-wall shadowing we want to KEEP,
                                    per crack sample, at the production
                                    (lambda/2) and a generation-resolution
                                    occluder.

    Informational, never asserted: these knobs change the physics, so they are
    provenance keys and a deliberate decision.
    """
    section = sections.load_reference_section()
    X, Y = config.plane_grid(device)
    args = (X, Y, config.H_MS, config.WVL, config.THETA_INC,
            config.SIZE_ANT, config.DIST_ANT, config.RESOL_ANT)

    rng = np.random.default_rng(0)
    augs = [(0.0, (0.0, 0.0))] + [
        (float(rng.uniform(-config.ROLL_DEG_STD, config.ROLL_DEG_STD)),
         (float(rng.normal(0, config.JITTER_XZ_STD)),
          float(rng.normal(0, config.JITTER_XZ_STD))))
        for _ in range(n_intact - 1)]

    intact_params = {"class": "intact"}
    base_intact = [_solve_sample(section, "intact", 0, device, args, shadow=False,
                                 roll=r, jit=j, params=intact_params)[0]
                   for r, j in augs]
    base_bar = [_barcode(b, device) for b in base_intact]

    cracks = {}
    for ci in crack_idxs:
        psi, params = _solve_sample(section, "crack", ci, device, args, shadow=False)
        cracks[ci] = {"base": psi, "params": params,
                      "depth_mm": float(params.get("depth", 0.0))}
    # The benefit columns below are only as meaningful as the samples behind
    # them, so name them. A crack shallower than the occluder facet shows zero
    # effect at every setting, which reads as "shadowing is inert" and is not.
    print("    probing cracks: " + ", ".join(
        f"c{ci} depth {c['depth_mm']:.2f} mm" for ci, c in cracks.items()))

    rows = []
    for mt, off in [(m, o) for o in offsets for m in min_ts]:
        arts, art_bars = [], []
        for (r, j), b, bb in zip(augs, base_intact, base_bar):
            a, _ = _solve_sample(section, "intact", 0, device, args, shadow=True,
                                 roll=r, jit=j, params=intact_params,
                                 min_t=mt, normal_offset=off)
            arts.append(float((a - b).norm() / b.norm()))
            art_bars.append(1.0 - _cos(_barcode(a, device), bb))
        row = {"min_t_mm": float(mt), "normal_offset_mm": float(off),
               "min_t_over_occluder_facet": float(mt / config.OCCLUDER_DS),
               "artifact_mean": float(np.mean(arts)),
               "artifact_max": float(np.max(arts)),
               "artifact_barcode_mean": float(np.mean(art_bars))}
        for ci, c in cracks.items():
            e, _ = _solve_sample(section, "crack", ci, device, args, shadow=True,
                                 params=c["params"], min_t=mt, normal_offset=off)
            row[f"crack{ci}_effect"] = float((e - c["base"]).norm() / c["base"].norm())
        # the resolved-occluder control, on the deepest crack only (it is the
        # expensive solve: occluder facets go from lambda/2 to lambda/8)
        deep = max(cracks, key=lambda k: cracks[k]["depth_mm"])
        rr, _ = _solve_sample(section, "crack", deep, device, args, shadow=True,
                              params=cracks[deep]["params"], min_t=mt,
                              normal_offset=off, occ_ds=config.MESH_DS)
        row["resolved_effect"] = float((rr - cracks[deep]["base"]).norm()
                                       / cracks[deep]["base"].norm())
        # NOTE the resolved column uses a lambda/8 occluder, so min_t is a
        # DIFFERENT multiple of ITS facet — reported explicitly, since comparing
        # the two columns at one "x facet" number was misleading
        row["min_t_over_resolved_facet"] = float(mt / config.MESH_DS)
        row["artifact_ok"] = row["artifact_mean"] < 0.03
        rows.append(row)
        ce = " ".join(f"c{ci} {row[f'crack{ci}_effect']:.4f}" for ci in cracks)
        print(f"    min_t {mt:6.3f} off {off:5.3f} | artifact mean {row['artifact_mean']:.4f} "
              f"max {row['artifact_max']:.4f} barcode {row['artifact_barcode_mean']:.5f} "
              f"{'OK ' if row['artifact_ok'] else 'BAD'} | {ce} resolved {row['resolved_effect']:.4f}")

    # the honest recommendation: the configuration that keeps the MOST real
    # shadowing while staying under the artifact bound. "smallest safe min_t"
    # was the wrong objective -- it is meaningless when every benefit is zero.
    def benefit(r):
        return max([r[k] for k in r if k.startswith("crack") and k.endswith("_effect")] or [0.0])

    safe = [r for r in rows if r["artifact_ok"]]
    useful = [r for r in safe if benefit(r) > 1e-6]
    rec = max(useful, key=benefit) if useful else None

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for a, xkey, xlab, fixed in [
        (ax[0], "min_t_mm", "min_t (mm along the ray)", "normal_offset_mm"),
        (ax[1], "normal_offset_mm", "normal offset (mm along the face normal)", "min_t_mm"),
    ]:
        for val in sorted({r[fixed] for r in rows}):
            sel = sorted([r for r in rows if r[fixed] == val], key=lambda r: r[xkey])
            if len(sel) < 2:
                continue
            xs = [r[xkey] for r in sel]
            a.plot(xs, [r["artifact_mean"] for r in sel], "o-",
                   label=f"artifact ({fixed.split('_')[0]}={val:g})")
            a.plot(xs, [benefit(r) for r in sel], "s--",
                   label=f"crack shadow ({fixed.split('_')[0]}={val:g})")
        a.axhline(0.03, color="tab:red", ls=":", lw=1, label="artifact bound 0.03")
        a.set(xlabel=xlab, ylabel="relative L2 field change")
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    fig.suptitle("V6b: ray-cast guard — artifact suppressed vs real shadowing kept")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "v6b_min_t_sweep.png", dpi=200)

    return {"rows": rows,
            "current_min_t": config.SHADOW_MIN_T,
            "current_normal_offset": config.SHADOW_NORMAL_OFFSET,
            "occluder_ds": config.OCCLUDER_DS,
            "n_intact_augmentations": n_intact,
            "crack_depths_mm": {str(k): v["depth_mm"] for k, v in cracks.items()},
            "recommended": ({"min_t_mm": rec["min_t_mm"],
                             "normal_offset_mm": rec["normal_offset_mm"],
                             "artifact_mean": rec["artifact_mean"],
                             "crack_effect": benefit(rec)} if rec else None),
            "note": ("recommended = most real crack shadowing kept while the mean "
                     "intact artifact stays under 0.03. None means NO tested "
                     "configuration achieves both, which is itself the result."),
            "pass": True}


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
    parser.add_argument("--min-t", type=float, nargs="*", default=None,
                        metavar="MM",
                        help="run the V6b ray-cast guard sweep over these min_t "
                             "values (mm) INSTEAD of the V5-V7 gates, e.g. "
                             "--min-t 3.0 1.0 0.3 0.125 0.05")
    parser.add_argument("--normal-offset", type=float, nargs="*", default=None,
                        metavar="MM",
                        help="sweep the ray-origin lift along the face normal "
                             "(mm), e.g. --normal-offset 0 0.05 0.1 0.15 0.3 0.6. "
                             "Combine with a single --min-t to isolate it.")
    parser.add_argument("--crack-idx", type=int, nargs="*", default=None,
                        metavar="I",
                        help="which crack samples the sweep probes for REAL "
                             "shadowing (default 0 1). The benefit column is the "
                             "whole point of the sweep, and a shallow crack "
                             "shows none at ANY setting -- the pre-2026-09-04 "
                             "harness probed index 0 alone and concluded "
                             "shadowing was inert, while V6 measured 0.0197 on "
                             "index 1 in the same run. Include a DEEP index; "
                             "depths are printed below and stored in the report.")
    args = parser.parse_args()
    device = torch.device(args.device) if args.device else config.get_device("lab")
    print(f"[rail3d] V5-V7 running on {device}"
          + (f"  ({torch.cuda.get_device_name(device.index or 0)})"
             if device.type == "cuda" else "  (CPU - this will be slow)"))
    config.ensure_dirs()

    report = {}
    if REPORT_PATH.exists():
        report = json.loads(REPORT_PATH.read_text())

    if args.min_t or args.normal_offset:
        min_ts = sorted(args.min_t, reverse=True) if args.min_t else [config.SHADOW_MIN_T]
        offs = (sorted(args.normal_offset) if args.normal_offset
                else [config.SHADOW_NORMAL_OFFSET])
        print(f"  V6b guard sweep (occluder facet {config.OCCLUDER_DS} mm; current "
              f"min_t {config.SHADOW_MIN_T} mm, normal offset "
              f"{config.SHADOW_NORMAL_OFFSET} mm)")
        t0 = time.time()
        kw_ci = ({} if args.crack_idx is None
                 else {"crack_idxs": tuple(args.crack_idx)})
        res = v6b_guard_sweep(device, min_ts, offs, **kw_ci)
        res["seconds"] = round(time.time() - t0, 2)
        report["V6b_guard_sweep"] = res
        REPORT_PATH.write_text(json.dumps(report, indent=2))
        rec = res["recommended"]
        print("\n  " + (f"best configuration: min_t {rec['min_t_mm']} mm, normal offset "
                        f"{rec['normal_offset_mm']} mm -> artifact "
                        f"{rec['artifact_mean']:.4f}, crack shadow {rec['crack_effect']:.4f}"
                        if rec else
                        "NO tested configuration keeps the artifact under 0.03 AND any "
                        "real crack shadowing -- that is the result, not a failure"))
        print(f"  figure -> {config.FIGURE_DIR / 'v6b_min_t_sweep.png'}")
        print(f"report -> {REPORT_PATH}")
        return 0

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
