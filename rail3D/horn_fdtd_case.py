"""Lumerical rung -1: the physical horn itself, full-wave.

    python horn_fdtd_case.py --out data/generated/fdtd_horn_case

Every other FDTD rung injects rail3D's *modelled* horn (horn_source.py), so
they test the rail scattering and take the horn model on trust. This rung
tests the horn model: a WR-15 waveguide feeding the pyramidal flare of the
RFspin H-A75-W20 (config.SIZE_ANT), driven by Lumerical's Mode source,
with frequency-domain monitors in front of the aperture. `fdtd_agreement.py
--horn` then scores FDTD against rail3D's aperture model (README finding 29).

Everything is in the horn's LOCAL frame, in mm:
    aperture centred at the origin in the z' = 0 plane, boresight +z',
    x' along A (the TE10 cosine, H-plane), y' along B (E is along y').
It maps onto rail3D's global frame by the aperture_field convention:
    x' -> (-cos th, 0, sin th),  y' -> (0, 1, 0),  z' -> (-sin th, 0, -cos th),
    origin -> (D sin th, 0, D cos th),  th = THETA_INC, D = DIST_ANT.

Writes:
  horn_body.stl   watertight hollow PEC solid: waveguide section + flare,
                  WALL-thick walls (import in MILLIMETRES)
  horn_case.json  region, Mode source, monitors, mesh options, expected checks
  export_horn.lsf the two monitor exports fdtd_agreement.py --horn reads
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rail3d import config, field3d

WALL = 0.5                      # mm; RFspin's outer shell is 1.0 mm larger (0.5 per side)
WG_LEN = 16.0                   # mm of straight WR-15 before the throat (runs into the PML)
C_MM_GHZ = 299.792458


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def horn_solid(size_ant=None, wall: float = WALL, wg_len: float = WG_LEN):
    """Hollow horn as a closed, consistently oriented triangle mesh.

    Three stations along z' -- waveguide back end, throat, aperture -- each
    with an inner rectangle (the air) and an outer one (inner + 2*wall).
    The solid is the wall between them: outer walls face outward, inner walls
    face the axis, and the two ends are closed by rectangular annuli.
    """
    A, B, a, b, L = config.SIZE_ANT if size_ant is None else size_ant
    stations = [(-L - wg_len, a, b), (-L, a, b), (0.0, A, B)]
    verts, inner, outer = [], [], []

    def ring(z, w, h):
        return [(-w / 2, -h / 2, z), (w / 2, -h / 2, z), (w / 2, h / 2, z), (-w / 2, h / 2, z)]

    for z, w, h in stations:
        inner.append(len(verts))
        verts += ring(z, w, h)
        outer.append(len(verts))
        verts += ring(z, w + 2 * wall, h + 2 * wall)
    faces = []

    def quad(p, q, r, s_):
        faces.extend([(p, q, r), (p, r, s_)])

    for i in range(len(stations) - 1):
        o0, o1, i0, i1 = outer[i], outer[i + 1], inner[i], inner[i + 1]
        for j in range(4):
            jn = (j + 1) % 4
            quad(o0 + j, o0 + jn, o1 + jn, o1 + j)          # outer wall
            quad(i0 + j, i1 + j, i1 + jn, i0 + jn)          # inner wall
    for idx, top in ((0, False), (len(stations) - 1, True)):
        o, ii = outer[idx], inner[idx]
        for j in range(4):
            jn = (j + 1) % 4
            if top:
                quad(o + j, o + jn, ii + jn, ii + j)
            else:
                quad(o + j, ii + j, ii + jn, o + jn)
    v, f = np.array(verts, float), np.array(faces, int)
    return v, f


def solid_checks(v: np.ndarray, f: np.ndarray, size_ant=None, wall: float = WALL,
                 wg_len: float = WG_LEN) -> dict:
    """Closed, consistently oriented, outward, and the right volume."""
    directed = {}
    for t in f:
        for k in range(3):
            e = (int(t[k]), int(t[(k + 1) % 3]))
            directed[e] = directed.get(e, 0) + 1
    closed = all(c == 1 for c in directed.values()) and \
        all(directed.get((b_, a_), 0) == 1 for (a_, b_) in directed)
    a3, b3, c3 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    vol = float(np.einsum("ij,ij->i", a3, np.cross(b3, c3)).sum() / 6)
    A, B, a, b, L = config.SIZE_ANT if size_ant is None else size_ant
    ring = lambda w, h: (w + 2 * wall) * (h + 2 * wall) - w * h          # noqa: E731
    # the flare's cross-section varies linearly in both dims: prismatoid rule
    flare = L / 6 * (ring(a, b) + 4 * ring((a + A) / 2, (b + B) / 2) + ring(A, B))
    expect = wg_len * ring(a, b) + flare
    return {"watertight_and_oriented": bool(closed), "outward": vol > 0,
            "wall_volume_mm3": round(vol, 3), "expected_mm3": round(expect, 3),
            "volume_ok": abs(vol - expect) < 1e-6 * max(expect, 1)}


# ---------------------------------------------------------------------------
# rail3D's side of the comparison, in the local frame
# ---------------------------------------------------------------------------
def model_aperture(x_mm: np.ndarray, y_mm: np.ndarray, size_ant=None, wvl=None) -> np.ndarray:
    """rail3D's aperture field on (x', y'), zero outside the opening -- the
    aperture method's own assumption (Nikolova L18, below eq. 18.8)."""
    size_ant = config.SIZE_ANT if size_ant is None else size_ant
    wvl = config.WVL if wvl is None else wvl
    A, B = size_ant[:2]
    X, Y = np.meshgrid(x_mm, y_mm, indexing="ij")
    psi = field3d.aperture_distribution(X, Y, size_ant, wvl)
    tol = 1e-6                   # mm: a sample ON the edge must not flip on rounding noise
    inside = (np.abs(X) <= A / 2 + tol) & (np.abs(Y) <= B / 2 + tol)
    return np.where(inside, psi, 0).astype(np.complex128)


def radiate(psi: np.ndarray, q: np.ndarray, dS: float, normal: np.ndarray,
            p: np.ndarray, wvl=None, chunk: int = 2048, device=None) -> np.ndarray:
    """RS-I radiation of plane samples psi at points q (N,3) to points p (M,3).

    The same kernel rail3D uses for the horn (field3d.rs_kernel), with the
    projected distance (p - q).n as its cosine term and the back half-space
    clipped exactly as horn_to_plane does.
    """
    wvl = config.WVL if wvl is None else wvl
    k0 = 2 * np.pi / wvl
    dev = torch.device("cpu") if device is None else torch.device(device)
    fdt, cdt = ((torch.float32, torch.complex64) if dev.type == "cuda"
                else (torch.float64, torch.complex128))
    qt = torch.as_tensor(q, dtype=fdt, device=dev)
    w = torch.as_tensor(psi.ravel() * dS, dtype=cdt, device=dev)
    nt = torch.as_tensor(normal, dtype=fdt, device=dev)
    out = np.zeros(len(p), np.complex128)
    for s0 in range(0, len(p), chunk):
        pt = torch.as_tensor(p[s0:s0 + chunk], dtype=fdt, device=dev)
        d = pt[:, None, :] - qt[None, :, :]
        R = d.norm(dim=-1)
        proj = (d * nt).sum(-1)
        proj = proj * (proj > 0)
        ker = field3d.rs_kernel(R, proj, wvl, k0)
        out[s0:s0 + chunk] = (ker * w).sum(-1).cpu().numpy()
    return out


def model_plane(x_mm: np.ndarray, y_mm: np.ndarray, z_mm: float, n: int = 80,
                size_ant=None, wvl=None) -> np.ndarray:
    """rail3D's horn field on the plane z' = z_mm (local frame), radiated from
    an n x n midpoint sampling of the aperture -- what the rail sees, before
    it has travelled to the rail."""
    size_ant = config.SIZE_ANT if size_ant is None else size_ant
    wvl = config.WVL if wvl is None else wvl
    A, B = size_ant[:2]
    u = field3d.aperture_axis(A, n, "midpoint").double().numpy()
    v = field3d.aperture_axis(B, n, "midpoint").double().numpy()
    U, V = np.meshgrid(u, v, indexing="ij")
    psi = field3d.aperture_distribution(U, V, size_ant, wvl)
    q = np.stack([U.ravel(), V.ravel(), np.zeros(U.size)], 1)
    X, Y = np.meshgrid(x_mm, y_mm, indexing="ij")
    p = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z_mm)], 1)
    dS = field3d.aperture_weight(A, B, n, "midpoint")
    return radiate(psi, q, dS, np.array([0.0, 0.0, 1.0]), p, wvl).reshape(X.shape)


def local_to_global(x, y, z):
    """Local horn coordinates -> rail3D's global frame (aperture_field's own)."""
    th, D = config.THETA_INC, config.DIST_ANT
    ex = np.array([-np.cos(th), 0.0, np.sin(th)])
    ey = np.array([0.0, 1.0, 0.0])
    ez = np.array([-np.sin(th), 0.0, -np.cos(th)])
    c = np.array([D * np.sin(th), 0.0, D * np.cos(th)])
    pts = (c + np.multiply.outer(np.asarray(x, float), ex) + np.multiply.outer(np.asarray(y, float), ey)
           + np.multiply.outer(np.asarray(z, float), ez))
    return pts, ez


def far_field_cuts(E: np.ndarray, x_mm: np.ndarray, y_mm: np.ndarray, wvl=None,
                   theta_max: float = 60.0) -> dict:
    """Principal-plane patterns, HPBW and directivity of a plane field.

    Huygens-aperture far field: F(theta) ~ (1+cos)/2 |sum E exp(-j k sin(theta) s)|
    (Nikolova L18 eq. 18.15); D from eq. 18.21. The same formula for both
    sides, so a disagreement is in the field, not in the post-processing.
    """
    wvl = config.WVL if wvl is None else wvl
    k = 2 * np.pi / wvl
    X, Y = np.meshgrid(x_mm, y_mm, indexing="ij")
    dA = (x_mm[1] - x_mm[0]) * (y_mm[1] - y_mm[0])
    th = np.radians(np.linspace(0, theta_max, 1201))
    H = np.array([abs((E * np.exp(-1j * k * np.sin(t) * X)).sum()) * (1 + np.cos(t)) / 2 for t in th])
    Ep = np.array([abs((E * np.exp(-1j * k * np.sin(t) * Y)).sum()) * (1 + np.cos(t)) / 2 for t in th])
    hp = lambda P: float(2 * np.degrees(th[np.argmax(P / P[0] < np.sqrt(0.5))]))   # noqa: E731
    D = 4 * np.pi / wvl ** 2 * abs(E.sum()) ** 2 * dA / (np.abs(E) ** 2).sum()
    return {"theta_deg": np.degrees(th), "H_plane": H / H[0], "E_plane": Ep / Ep[0],
            "hpbw_H_deg": hp(H), "hpbw_E_deg": hp(Ep), "directivity_dBi": float(10 * np.log10(D))}


# ---------------------------------------------------------------------------
# the FDTD plan
# ---------------------------------------------------------------------------
LSF_EXPORT = """# export_horn.lsf -- written by rail3D/horn_fdtd_case.py
# Run after the rung -1 simulation finishes. E_y is the TE10 polarisation.
E = getresult("mon_aperture", "E");
Ey = pinch(E.Ey);  x = E.x;  y = E.y;
matlabsave("horn_aperture.mat", Ey, x, y);
E = getresult("mon_near", "E");
Ey = pinch(E.Ey);  x = E.x;  y = E.y;
matlabsave("horn_near.mat", Ey, x, y);
?"wrote horn_aperture.mat and horn_near.mat";
"""


def plan(size_ant=None, wall: float = WALL, wg_len: float = WG_LEN) -> dict:
    A, B, a, b, L = config.SIZE_ANT if size_ant is None else size_ant
    lam = config.WVL
    z_near = round(3 * lam, 3)
    mon_ap = {"x": [-(A / 2 + 2 * lam), A / 2 + 2 * lam], "y": [-(B / 2 + 2 * lam), B / 2 + 2 * lam],
              "z": 0.5}
    mon_nr = {"x": [-(A / 2 + 4 * lam), A / 2 + 4 * lam], "y": [-(B / 2 + 4 * lam), B / 2 + 4 * lam],
              "z": z_near}
    pad = 1.6 * lam
    region = {"x": [round(mon_nr["x"][0] - pad, 2), round(mon_nr["x"][1] + pad, 2)],
              "y": [round(mon_nr["y"][0] - pad, 2), round(mon_nr["y"][1] + pad, 2)],
              "z": [round(-L - 10.0, 2), round(z_near + pad, 2)]}
    ext = np.array([region[k][1] - region[k][0] for k in ("x", "y", "z")])
    meshes = {}
    for div in (15, 20):
        dx = lam / div
        cells = float(np.ceil(ext / dx).prod())
        meshes[f"lambda_over_{div}"] = {"dx_mm": round(dx, 4), "cells_M": round(cells / 1e6, 1),
                                        "est_RAM_GB": round(cells * 100 / 1e9, 1),
                                        "wall_cells": round(wall / dx, 1)}
    neff = float(np.sqrt(1 - (lam / (2 * a)) ** 2))
    src_z = round(-L - 6.0, 2)
    return {
        "purpose": ("rung -1: the physical RFspin H-A75-W20 horn (config.SIZE_ANT) "
                    "full-wave, to test rail3D's aperture model (README finding 29)"),
        "frame": ("LOCAL: aperture centred at the origin in z'=0, boresight +z', x' along A "
                  "(TE10 cosine), y' along B (E along y'); see horn_fdtd_case.local_to_global"),
        "units": "mm (set File > Units > Length = mm BEFORE importing the STL)",
        "frequency_Hz": C_MM_GHZ / lam * 1e9,
        "horn": {"A": A, "B": B, "feed_a": a, "feed_b": b, "flare_L": L, "wall": wall,
                 "waveguide_section": wg_len, "stl": "horn_body.stl",
                 "material": "PEC (Perfect Electrical Conductor)"},
        "simulation_region_mm": region,
        "boundaries": "PML on all six faces; the waveguide runs through the z-min PML",
        "mesh_options": meshes,
        "recommended_mesh": ("uniform lambda/20 (0.25 mm, the walls are 2 cells) with mesh "
                             "refinement 'conformal variant 1' (applies to PEC)"),
        "mode_source": {
            "name": "wg_mode", "injection_axis": "z", "direction": "Forward",
            "z_mm": src_z,
            "x_mm": [-(a / 2 + wall + 1.0), a / 2 + wall + 1.0],
            "y_mm": [-(b / 2 + wall + 1.0), b / 2 + wall + 1.0],
            "mode_selection": "fundamental mode",
            "why_fundamental_is_TE10": ("WR-15 is single-mode at 60 GHz (TE10 cutoff 39.9 GHz; "
                                        "TE20/TE01 79.7 GHz), so the largest-effective-index "
                                        "mode IS TE10 -- no TE/TM naming convention needed"),
            "check_effective_index": round(neff, 4),
            "check_profile": "E along y', |E| ~ cos(pi x'/a) across the broad wall, uniform in y'",
        },
        "monitors": {
            "mon_aperture": dict(mon_ap, type="frequency-domain field, 2D Z-normal",
                                 purpose="the aperture field itself (0.1 lambda in front)"),
            "mon_near": dict(mon_nr, type="frequency-domain field, 2D Z-normal",
                             purpose="the radiated field 3 lambda out -- what travels to the rail"),
        },
        "export": "run export_horn.lsf -> horn_aperture.mat, horn_near.mat",
        "then": "python fdtd_agreement.py --horn horn_aperture.mat --horn-near horn_near.mat",
    }


def build(out: Path) -> dict:
    from compare_wavefronts import write_stl
    out.mkdir(parents=True, exist_ok=True)
    v, f = horn_solid()
    chk = solid_checks(v, f)
    if not (chk["watertight_and_oriented"] and chk["outward"] and chk["volume_ok"]):
        raise RuntimeError(f"horn solid failed its checks: {chk}")
    write_stl(out / "horn_body.stl", v, f, f"rail3D {config.HORN_SPEC['part']} horn body (mm)")
    p = plan()
    p["solid_checks"] = chk
    (out / "horn_case.json").write_text(json.dumps(p, indent=2), encoding="utf-8")
    (out / "export_horn.lsf").write_text(LSF_EXPORT, encoding="utf-8")
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    p = build(Path(args.out))
    r, m = p["simulation_region_mm"], p["mesh_options"]
    print(f"{config.HORN_SPEC['part']}: A {p['horn']['A']} x B {p['horn']['B']} mm, "
          f"WR-15 feed, flare {p['horn']['flare_L']} mm, walls {p['horn']['wall']} mm")
    print(f"  solid: {p['solid_checks']}")
    print(f"  region x {r['x']} y {r['y']} z {r['z']} mm")
    for k_, v_ in m.items():
        print(f"  {k_}: {v_['cells_M']} M cells, ~{v_['est_RAM_GB']} GB, walls {v_['wall_cells']} cells")
    print(f"  Mode source at z' = {p['mode_source']['z_mm']} mm: expect neff "
          f"{p['mode_source']['check_effective_index']} (TE10)")
    print(f"  -> {Path(args.out)}  (horn_body.stl, horn_case.json, export_horn.lsf)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
