"""Scan measurement-plane geometry BEFORE committing to a full generation.

The observation plane's height, lateral offset and size are baked into the
stored fields, so they cannot be learned end to end — but they can be measured
cheaply. For each candidate geometry this generates a small set of samples and
reports how much scattered energy the plane captures and how separable the
classes are there.

Why lateral offset matters: the horn illuminates at THETA_INC from +x, so the
specular lobe off a flat crown lands at x = -H*tan(theta) — -114 mm at H=80,
-228 mm at H=160. A plane centred at x=0 therefore collects the off-specular
tail (dark-field). That may well be the better choice for defect contrast, but
it should be a measurement, not an inheritance.

Two separability metrics are reported per configuration:
  field  — AUC from the raw intensity map distance (physical information
           available at that plane, independent of optics)
  det    — AUC from detector barcodes through UNTRANED optics with a fixed
           seed (what a simple readout actually sees)
Both compare each defect class against the intact pool; 0.5 = invisible.

Usage (from rail3D/, ~5 s per configuration on the 5090):
    python scan_geometry.py                  # default candidate list
    python scan_geometry.py --n 20           # more samples per class
    python scan_geometry.py --heights 80 160 --centers 0 -114 -228
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from rail3d import config, field3d, losses3d, mesh3d, optics3d, sections


def build_fields(section, cls, n, H, nx, ny, xc, device, chunk):
    """Generate n samples of one class at the given plane geometry."""
    X, Y = config.plane_grid(device, nx=nx, ny=ny, x_center=xc)
    args = (X, Y, H, config.WVL, config.THETA_INC,
            config.SIZE_ANT, config.DIST_ANT, config.RESOL_ANT)
    psi0 = field3d.horn_to_plane(*args)
    n_arc = mesh3d.default_arc_count(section, config.MESH_DS)
    files = sections.get_dataset_files(cls) if cls in config.DATASET_DIRS else None

    out = []
    for i in range(n):
        defect = None
        if files is not None:
            defect = sections.match_reference_width(
                sections.load_vertices_from_csv(files[i]), section)
        params, aug = mesh3d.defect_params_for_sample(
            section, cls, defect, config.sample_seed(cls, i), n_arc=n_arc)
        v, f = mesh3d.sweep_rail_mesh(
            section, defect_params=params, slice_ds=config.MESH_DS,
            arc_ds=config.MESH_DS, roll_deg=aug["roll_deg"],
            jitter_xz=(aug["jitter_x"], aug["jitter_z"]))
        v_occ, f_occ = mesh3d.sweep_rail_mesh(
            section, defect_params=params, roll_deg=aug["roll_deg"],
            jitter_xz=(aug["jitter_x"], aug["jitter_z"]))
        psi1, psi2 = field3d.scattered_fields(
            v.to(device), f.to(device), *args, chunk_faces=chunk,
            shadow=config.SHADOW_MODE,
            shadow_occluders=(v_occ.to(device), f_occ.to(device)))
        out.append(psi0 + psi1 + psi2)
    return torch.stack(out)


def evaluate(fields: dict, nx, ny, device) -> dict:
    """Energy capture + field-level and detector-level separability."""
    intact = fields["intact"]
    I0 = intact.abs() ** 2
    res = {"energy": float(I0.mean()),
           "edge_frac": float(_edge_fraction(I0.mean(dim=0)))}

    # field-level: distance of each sample's intensity map from the intact mean
    ref = I0.mean(dim=0)
    def fdist(t):
        d = (t.abs() ** 2 - ref).reshape(t.shape[0], -1)
        return d.norm(dim=1) / ref.norm()
    neg_f = fdist(intact)
    res["intact_field_spread"] = float(neg_f.mean())

    # detector-level through untrained optics (fixed seed -> fair comparison)
    det = optics3d.SoftDetector2D(
        centers=_scaled_centers(nx, ny), nx=nx, ny=ny,
        wx=nx * config.DX, wy=ny * config.DX).to(device)
    slm_seed = 0
    phase = torch.randn(nx, ny, generator=torch.Generator().manual_seed(slm_seed)) * (np.pi / 2)
    prop = optics3d.PropagatorRSFFT(config.LAYER_DISTANCES[-1], nx=nx, ny=ny).to(device)
    phase = phase.to(device)

    def barcode(t):
        return det.hard_powers(prop(t * torch.exp(1j * phase)).abs() ** 2)

    b0 = barcode(intact)
    r0 = b0.mean(dim=0)
    neg_d = losses3d.cos_gap(b0, r0)
    res["intact_det_spread"] = float(neg_d.mean())

    for cls in config.CLASS_NAMES:
        res[f"{cls}_field_auc"] = losses3d.auc_score(fdist(fields[cls]), neg_f)
        res[f"{cls}_det_auc"] = losses3d.auc_score(
            losses3d.cos_gap(barcode(fields[cls]), r0), neg_d)
    res["field_auc_mean"] = float(np.mean([res[f"{c}_field_auc"] for c in config.CLASS_NAMES]))
    res["det_auc_mean"] = float(np.mean([res[f"{c}_det_auc"] for c in config.CLASS_NAMES]))
    return res


def _edge_fraction(I: torch.Tensor) -> float:
    """Share of intensity in the outer ring — high means energy is escaping."""
    nx, ny = I.shape
    m = torch.zeros_like(I, dtype=torch.bool)
    k = max(1, min(nx, ny) // 10)
    m[:k, :] = m[-k:, :] = True
    m[:, :k] = m[:, -k:] = True
    return float(I[m].sum() / I.sum())


def _scaled_centers(nx, ny):
    """Detector grid scaled to whatever aperture is being tested."""
    gx, gy = config.DET_GRID
    px = nx * config.DX / (gx + 1)
    py = ny * config.DX / (gy + 1)
    cx = (torch.arange(gx) - (gx - 1) / 2) * px
    cy = (torch.arange(gy) - (gy - 1) / 2) * py
    CX, CY = torch.meshgrid(cx, cy, indexing="ij")
    return torch.stack([CX.reshape(-1), CY.reshape(-1)], dim=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    # n=20 gives 400 (defect, intact) pairs per class -> AUC resolution ~0.0025.
    # Below ~10 the all-pairs AUC saturates at 1.000 and cannot rank configs.
    ap.add_argument("--n", type=int, default=20, help="samples per class per config")
    ap.add_argument("--profile", default="lab")
    ap.add_argument("--heights", type=float, nargs="*", default=None)
    ap.add_argument("--centers", type=float, nargs="*", default=None)
    ap.add_argument("--grids", nargs="*", default=None, help="e.g. 60x30 80x40 80x80")
    args = ap.parse_args()

    device = config.get_device(args.profile)
    chunk = config.PROFILES[args.profile].chunk_faces
    section = sections.load_reference_section()
    classes = ("intact",) + config.CLASS_NAMES

    if args.heights or args.centers or args.grids:
        heights = args.heights or [config.H_MS]
        centers = args.centers or [config.PLANE_X_CENTER]
        grids = [tuple(int(v) for v in g.split("x")) for g in (args.grids or ["60x30"])]
        cands = [(h, c, g) for h, c, g in itertools.product(heights, centers, grids)]
    else:
        # default: current setup, specular-centred variants, lower planes, wider apertures
        spec = lambda h: -h * np.tan(config.THETA_INC)
        cands = [
            (160.0, 0.0, (60, 30)),                 # current
            (160.0, spec(160) / 2, (60, 30)),       # half-way to the lobe
            (160.0, spec(160), (60, 30)),           # specular-centred
            (160.0, 0.0, (80, 40)),                 # wider, same centre
            (160.0, 0.0, (80, 80)),                 # Face3D-sized
            (80.0, 0.0, (60, 30)),                  # lower: lobe nearly inside
            (80.0, spec(80), (60, 30)),             # lower + specular-centred
            (240.0, 0.0, (60, 30)),                 # higher
        ]

    print(f"[rail3d] geometry scan on {device}  ({args.n} samples/class/config)")
    if args.n < 10:
        print("  WARNING: n < 10 -- all-pairs AUC saturates and cannot rank configurations")
    print()
    hdr = (f"{'H':>6s} {'x_ctr':>7s} {'grid':>7s} {'s':>5s} | {'energy':>9s} {'edge%':>6s} "
           f"| {'field AUC':>9s} {'det AUC':>8s} | " +
           " ".join(f"{c[:5]:>6s}" for c in config.CLASS_NAMES))
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for H, xc, (nx, ny) in cands:
        t0 = time.time()
        fields = {c: build_fields(section, c, args.n, H, nx, ny, xc, device, chunk)
                  for c in classes}
        r = evaluate(fields, nx, ny, device)
        r.update(H=H, x_center=xc, nx=nx, ny=ny, seconds=time.time() - t0)
        rows.append(r)
        print(f"{H:6.0f} {xc:7.0f} {nx:3d}x{ny:<3d} {r['seconds']:5.0f} | "
              f"{r['energy']:9.2e} {100*r['edge_frac']:5.1f}% | "
              f"{r['field_auc_mean']:9.3f} {r['det_auc_mean']:8.3f} | " +
              " ".join(f"{r[f'{c}_field_auc']:6.3f}" for c in config.CLASS_NAMES))

    best = max(rows, key=lambda r: r["field_auc_mean"])
    print(f"\nbest mean field AUC: H={best['H']:.0f} mm, x_center={best['x_center']:.0f} mm, "
          f"grid {best['nx']}x{best['ny']}  ->  {best['field_auc_mean']:.3f} "
          f"(current setup: {rows[0]['field_auc_mean']:.3f})")
    print("\nper-class detail of the best configuration:")
    for c in config.CLASS_NAMES:
        print(f"  {c:6s} field AUC {best[f'{c}_field_auc']:.3f}   det AUC {best[f'{c}_det_auc']:.3f}")

    out = config.GENERATED_DIR / "geometry_scan.json"
    out.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(f"\n[saved to {out}]  -- paste this table back before running the full generation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
