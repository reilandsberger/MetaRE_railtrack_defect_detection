"""po_canonical_2d.py -- how wrong is rail3D's surface model on each defect? (2D, exact)

rail3D computes the rail's reflection with scalar physical optics (PO): every lit
facet re-radiates minus the incident field through a Rayleigh-Sommerfeld-I kernel
(field3d.scattered_fields), lit = facing the horn, facing up, and in its line of
sight (ray cast, min_t 0.05 / offset 0.3). That is exact for a flat plate and good
on smooth surfaces whose radii are many wavelengths. It has no multiple bounces,
no waveguide modes, no polarisation, and no edge diffraction. Lumerical FDTD has all
of them, so rung 3 (LUMERICAL.md section 5) will measure exactly this error. This
script predicts it per defect class, on canonical 2D cross-sections where an exact
answer is cheap.

Each case is one defect cut across, on a locally flat PEC surface, under a
Gaussian beam (an exact Helmholtz solution, tapered so the profile's ends are dark),
solved twice:

  exact   boundary integral equations, pulse basis, point matching
          Dirichlet (E along the invariant axis):   int G sigma = -psi_inc
          Neumann   (E in the cross-section plane): psi/2 - PV int psi dG/dn' = psi_inc
  rail3D  the same PO rail3D uses, reduced to 2D: lit segments radiate
          2 (-psi_inc) dG/dn', front-clamped, with a 2D ray-cast shadow test

The polarisation of each cut is fixed by the horn (E along the rail, y):
  x-z cut (longitudinal crack, dent/shell/wear across the head): E_y is along the
          invariant axis: Dirichlet, incidence in-plane at the local angle
  y-z cut (transverse crack, dent/shell along the rail): E_y lies in the plane:
          Neumann. The horn's k_x is along the invariant axis, so the reduced 2D
          problem runs at k_t = k cos(theta_loc), normal incidence (separable)
rail3D is scalar Dirichlet in both: in the y-z cut it gets the polarisation wrong
by construction, and that is part of what is measured.

What is scored is the DIFFERENCE field, defect minus the same surface without it.
That is the signal the detectors see. Its far-field pattern is compared over the
directions that reach today's metasurface, and over the whole upper half-space:
  corr      complex correlation of the two difference patterns, rail3D vs exact
  level dB  10 log10(rail3D / exact) difference-field energy, after the complex gain
            is fitted on the flat surface (as fdtd_agreement aligns the absolute field)

Validation runs first (--validate): flat plate vs the analytic reflected beam, and a
PEC circular cylinder vs its exact Bessel series, for both polarisations.

    python po_canonical_2d.py --validate      # seconds
    python po_canonical_2d.py                 # the study: JSON + figures (~10 min, CPU)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import hankel1, jv, jvp, h1vp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rail3d import config   # noqa: E402

K0 = 2 * np.pi / config.WVL
EULER_GAMMA = 0.5772156649015329
DS = config.WVL / 20                 # BIE segment length (lambda/20)
WAIST = 6 * config.WVL               # Gaussian beam waist (mm): footprint ~waist / cos(theta)
OUT_JSON = config.GENERATED_DIR / "po_canonical_2d.json"
FIG = config.FIGURE_DIR / "rail_targets"
# Directions that reach today's 150 x 75 mm metasurface at H 150, as half-angles
# from the local normal: across the head (x, the Dirichlet cuts) +-26.6 deg, along
# the rail (y, the Neumann cuts) +-14 deg.
MS_HALF_ANGLE = {"x-z": math.degrees(math.atan(75 / 150)), "y-z": math.degrees(math.atan(37.5 / 150))}


# ---------------------------------------------------------------------------
# geometry: open polylines, air above, traversed left -> right
# ---------------------------------------------------------------------------
def refine(P: np.ndarray, ds: float = DS) -> np.ndarray:
    """Resample a polyline so no segment is longer than ds (keeps its corners)."""
    out = [P[0]]
    for a, b in zip(P[:-1], P[1:]):
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / ds)))
        t = np.linspace(0, 1, n + 1)[1:, None]
        out.extend(a + t * (b - a))
    return np.array(out)


def segs(P: np.ndarray) -> dict:
    d = P[1:] - P[:-1]
    L = np.linalg.norm(d, axis=1)
    t = d / L[:, None]
    n = np.stack([-t[:, 1], t[:, 0]], 1)          # left normal: into the air
    return {"a": P[:-1], "b": P[1:], "c": 0.5 * (P[:-1] + P[1:]), "L": L, "t": t, "n": n}


def surface(depth_fn, half: float, ds: float = DS, edge: float | None = None,
            breaks: tuple = ()) -> np.ndarray:
    """Flat surface z = -depth(x) on [-half, half], sampled at ds, with extra
    knots at `breaks` (crack wall feet) so steep walls are resolved exactly."""
    x = np.union1d(np.arange(-half, half + ds / 2, ds), np.asarray(breaks, float))
    x = x[(x >= -half) & (x <= half)]
    P = np.stack([x, -depth_fn(x)], 1)
    return refine(P, ds)


def slots(width: float, depth: float, n: int = 1, gap_factor: float = 1.9,
          edge: float = 0.25 * config.MESH_DS):
    """rail3D's crack cross-section: n box slots (flat bottom, walls ramped over a
    quarter mesh cell, mesh3d.sample_defect_params), centre-to-centre gap
    gap_factor x width. Returns (depth_fn, knots)."""
    half = width / 2
    offs = [(i - (n - 1) / 2) * gap_factor * width for i in range(n)]
    v = np.array([-half - edge, -half, half, half + edge])
    prof = np.array([0.0, 1.0, 1.0, 0.0])

    def fn(x):
        return np.max([depth * np.interp(x - o, v, prof, left=0, right=0) for o in offs], 0)
    knots = [o + vv for o in offs for vv in v]
    return fn, knots


def gaussian(depth: float, fwhm: float):
    k = 4 * math.log(2)
    return lambda x: depth * np.exp(-k * (x / fwhm) ** 2)


def shell(depth: float, radius: float, seed: int = 0):
    """A cut through mesh3d's shell: clip(1 - |x|/r)^0.7 with +-20% roughness on a
    4 mm lattice (render_depth_field)."""
    rng = np.random.default_rng(seed)
    lat = rng.standard_normal(33)
    lx = np.linspace(-64.0, 64.0, 33)

    def fn(x):
        base = np.clip(1 - np.abs(x) / radius, 0, None) ** 0.7
        rough = np.interp(x, lx, lat)
        return depth * base * np.clip(1 + 0.2 * rough, 0.6, 1.4)
    return fn


def wear(depth: float, radius: float, soft: float = config.WEAR_EDGE_SOFT):
    """A circular cut of the given depth and radius (a wheel wears the rail),
    edges rounded over `soft` mm (mesh3d._wear_cut, flattened to a plane)."""
    half = math.sqrt(max(2 * radius * depth - depth ** 2, 0.0))

    def fn(x):
        raw = np.clip(depth - (radius - np.sqrt(np.clip(radius ** 2 - x ** 2, 0, None))), 0, None)
        raw = np.where(np.abs(x) < half, raw, 0.0)
        # C1 rounding of the rim: smooth-max with 0 over `soft`
        return 0.5 * (raw + np.sqrt(raw ** 2 + (soft * 0.25) ** 2)) - 0.5 * soft * 0.25
    return fn


# ---------------------------------------------------------------------------
# the incident Gaussian beam: an exact plane-wave superposition
# ---------------------------------------------------------------------------
def beam(theta_deg: float, k: float = K0, waist: float = WAIST, n: int = 401):
    """Down-going 2D Gaussian beam centred on the origin, arriving from the +x
    side at theta from the normal. Returns psi(x, z) and grad psi(x, z)."""
    th = math.radians(theta_deg)
    u = np.linspace(-1, 1, n) * 4 * 2 / (k * waist)            # +-4 sigma in angle
    ang = th + u
    keep = np.abs(ang) < math.pi / 2 - 1e-3
    ang, w = ang[keep], np.exp(-(u[keep] * k * waist / 2) ** 2)
    w = w / w.sum()
    kx, kz = -k * np.sin(ang), -k * np.cos(ang)                 # travelling down, toward -x

    def psi(x, z):
        ph = np.exp(1j * (np.multiply.outer(x, kx) + np.multiply.outer(z, kz)))
        return ph @ w

    def grad(x, z):
        ph = np.exp(1j * (np.multiply.outer(x, kx) + np.multiply.outer(z, kz)))
        return (ph @ (1j * kx * w)), (ph @ (1j * kz * w))
    return psi, grad, (kx, kz, w)


# ---------------------------------------------------------------------------
# kernels and quadrature
# ---------------------------------------------------------------------------
GL4 = np.polynomial.legendre.leggauss(4)
GL16 = np.polynomial.legendre.leggauss(16)


def _quad_pts(S, gl, idx=None):
    """Gauss points on segments idx (or all): (n, q, 2) points and (n, q) weights."""
    a, b, L = S["a"], S["b"], S["L"]
    if idx is not None:
        a, b, L = a[idx], b[idx], L[idx]
    x, w = gl
    t = 0.5 * (x + 1)
    pts = a[:, None, :] + t[None, :, None] * (b - a)[:, None, :]
    return pts, 0.5 * L[:, None] * w[None, :]


def _G(R, k):
    return 0.25j * hankel1(0, k * R)


def _dGdn(r, rp, n, k):
    """dG/dn' at source rp (normal n) for observation r: (ik/4) H1(kR) (r-rp).n/R."""
    d = r - rp
    R = np.linalg.norm(d, axis=-1)
    return 0.25j * k * hankel1(1, k * R) * np.sum(d * n, -1) / R


def matrices(S: dict, k: float, near: float = 3.0):
    """Collocation matrices at segment midpoints: Gm[m, n] = int_n G(c_m, r') dl',
    Dm[m, n] = PV int_n dG/dn'(c_m, r') dl'. 4-point Gauss; 16-point within `near`
    segment lengths; the self terms analytically (straight segments)."""
    c, N = S["c"], len(S["L"])
    pts, wq = _quad_pts(S, GL4)
    Gm = np.empty((N, N), complex)
    Dm = np.empty((N, N), complex)
    nrm = S["n"]
    for i0 in range(0, N, 256):
        i1 = min(N, i0 + 256)
        d = c[i0:i1, None, None, :] - pts[None]                       # (m, n, q, 2)
        R = np.linalg.norm(d, axis=-1)
        R = np.where(R < 1e-12, 1e-12, R)
        H0 = hankel1(0, k * R)
        H1 = hankel1(1, k * R)
        Gm[i0:i1] = 0.25j * np.sum(H0 * wq[None], -1)
        Dm[i0:i1] = np.sum(0.25j * k * H1 * np.sum(d * nrm[None, :, None, :], -1) / R * wq[None], -1)
    # near pairs: 16-point Gauss
    dist = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)
    mm, nn = np.nonzero((dist < near * np.maximum(S["L"][:, None], S["L"][None, :])) & (dist > 0))
    if len(mm):
        p16, w16 = _quad_pts(S, GL16, nn)
        d = c[mm, None, :] - p16
        R = np.linalg.norm(d, axis=-1)
        Gm[mm, nn] = 0.25j * np.sum(hankel1(0, k * R) * w16, -1)
        Dm[mm, nn] = np.sum(0.25j * k * hankel1(1, k * R) * np.sum(d * S["n"][nn, None, :], -1) / R * w16, -1)
    # self terms: int_{-L/2}^{L/2} (i/4) H0(k|t|) dt, small-argument expansion
    L = S["L"]
    ii = np.arange(N)
    Gm[ii, ii] = 0.25j * L * (1 + (2j / np.pi) * (np.log(EULER_GAMMA_EXP * k * L / 4) - 1))
    Dm[ii, ii] = 0.0                                                   # straight segment
    return Gm, Dm


EULER_GAMMA_EXP = math.exp(EULER_GAMMA)   # 1.781...


def solve_exact(S: dict, k: float, psi_inc, grad_inc, bc: str) -> np.ndarray:
    """Surface unknowns: Dirichlet -> sigma (= -dpsi/dn); Neumann -> total psi."""
    Gm, Dm = matrices(S, k)
    c = S["c"]
    pin = psi_inc(c[:, 0], c[:, 1])
    if bc == "dirichlet":
        return np.linalg.solve(Gm, -pin)
    A = 0.5 * np.eye(len(c)) - Dm
    return np.linalg.solve(A, pin)


def lit_mask(S: dict, src_dir: np.ndarray, min_t: float = config.SHADOW_MIN_T,
             offset: float = config.SHADOW_NORMAL_OFFSET) -> np.ndarray:
    """rail3D's visibility in 2D: facing the source, facing up, and a clear ray
    toward the source (origin lifted `offset` along the normal, hits closer than
    min_t ignored), as field3d.scattered_fields + raycast_shadow_mask."""
    n, c = S["n"], S["c"]
    vis = (n @ src_dir > 0) & (n[:, 1] > 0)
    o = c + offset * n
    a, b = S["a"], S["b"]
    e = b - a
    for i in np.nonzero(vis)[0]:
        # ray o + t * src_dir vs every segment a + s e
        den = src_dir[0] * e[:, 1] - src_dir[1] * e[:, 0]
        ok = np.abs(den) > 1e-12
        w = a - o[i]
        t = np.where(ok, (w[:, 0] * e[:, 1] - w[:, 1] * e[:, 0]) / np.where(ok, den, 1), -1)
        s = np.where(ok, (w[:, 0] * src_dir[1] - w[:, 1] * src_dir[0]) / np.where(ok, den, 1), -1)
        hit = ok & (t > min_t) & (s >= 0) & (s <= 1)
        hit[i] = False
        if hit.any():
            vis[i] = False
    return vis


def far_field(S: dict, k: float, dens: np.ndarray, kind: str, phi_deg: np.ndarray) -> np.ndarray:
    """Far-field pattern F(phi) (phi from the +z normal, positive toward +x), common
    factor sqrt(2/(pi k r)) e^{i(kr - pi/4)} dropped. kind: 'single' (density sigma,
    kernel G) or 'double' (density psi, kernel dG/dn')."""
    ph = np.radians(phi_deg)
    rh = np.stack([np.sin(ph), np.cos(ph)], 1)                       # (P, 2)
    pts, wq = _quad_pts(S, GL4)
    e = np.exp(-1j * k * np.einsum("pd,nqd->pnq", rh, pts))           # (P, n, q)
    if kind == "single":
        return 0.25j * np.einsum("pnq,nq,n->p", e, wq, dens)
    # dG/dn' -> (ik/4) H1 cos ~ (ik/4) (-i) rh.n e^{...} relative to H0's asymptote
    cosn = rh @ S["n"].T                                              # (P, n)
    return 0.25j * k * (-1j) * np.einsum("pnq,nq,pn,n->p", e, wq, cosn, dens)


def solve_case(P: np.ndarray, theta_deg: float, bc: str, k: float = K0,
               phi=np.linspace(-80, 80, 321)) -> dict:
    """Exact and rail3D-PO far-field patterns for one surface."""
    S = segs(P)
    th = math.radians(theta_deg)
    psi, grad, _ = beam(theta_deg, k)
    dens = solve_exact(S, k, psi, grad, bc)
    F_ex = far_field(S, k, dens, "single" if bc == "dirichlet" else "double", phi)
    src = np.array([math.sin(th), math.cos(th)])                     # toward the source
    lit = lit_mask(S, src)
    c = S["c"]
    po = np.where(lit, -2 * psi(c[:, 0], c[:, 1]), 0)                 # 2 (-psi_inc) on lit
    # front clamp of rail3D's kernel (point2plane > 0): in the far field rh.n > 0
    ph = np.radians(phi)
    rh = np.stack([np.sin(ph), np.cos(ph)], 1)
    front = (rh @ S["n"].T) > 0
    F_po = far_field(S, k, po, "double", phi) if front.all() else _far_field_clamped(S, k, po, phi, front)
    return {"phi": phi, "exact": F_ex, "po": F_po, "S": S, "dens": dens, "lit": lit}


def _far_field_clamped(S, k, dens, phi_deg, front):
    ph = np.radians(phi_deg)
    rh = np.stack([np.sin(ph), np.cos(ph)], 1)
    pts, wq = _quad_pts(S, GL4)
    e = np.exp(-1j * k * np.einsum("pd,nqd->pnq", rh, pts))
    cosn = (rh @ S["n"].T) * front
    return 0.25j * k * (-1j) * np.einsum("pnq,nq,pn,n->p", e, wq, cosn, dens)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def mie_cylinder(a: float, k: float, phi_deg: np.ndarray, bc: str, nmax: int = 60) -> np.ndarray:
    """Exact far-field pattern of a PEC circular cylinder radius a under a unit plane
    wave travelling in -z (from +z), same normalisation as far_field: psi_s ~
    sqrt(2/(pi k r)) e^{i(kr - pi/4)} F."""
    # incident e^{-ikz} = sum i^{-n} J_n(kr) e^{in(angle from +z... )}; use angle from -z
    psi_ = np.radians(phi_deg)          # observation angle from +z
    ang = psi_ - math.pi                # angle from the propagation direction (-z)
    F = np.zeros_like(ang, dtype=complex)
    for n in range(-nmax, nmax + 1):
        if bc == "dirichlet":
            an = -jv(n, k * a) / hankel1(n, k * a)
        else:
            an = -jvp(n, k * a) / h1vp(n, k * a)
        # H_n(kr) ~ sqrt(2/(pi k r)) e^{i(kr - pi/4)} (-i)^n
        F += (1j ** n) * an * ((-1j) ** n) * np.exp(1j * n * ang)
    return F


def validate() -> dict:
    out = {}
    k = K0
    phi = np.linspace(-80, 80, 161)
    # 1. flat plate under the beam: exact reflected beam is the mirrored spectrum
    for bc, sign in (("dirichlet", -1), ("neumann", +1)):
        for th in (0.0, 55.0):
            P = surface(lambda x: 0 * x, 160.0)
            r = solve_case(P, th, bc, k, phi)
            _, _, (kx, kz, w) = beam(th, k)
            # reflected plane waves (kx, -kz) with amplitude sign*w: far field of a
            # plane-wave superposition is not a 2D cylindrical wave; compare instead
            # exact vs PO (identical for a flat plate up to the polarisation sign)
            cc = abs(np.vdot(r["po"], r["exact"])) / (np.linalg.norm(r["po"]) * np.linalg.norm(r["exact"]))
            lvl = 10 * np.log10(np.sum(abs(r["po"]) ** 2) / np.sum(abs(r["exact"]) ** 2))
            ph = np.angle(np.vdot(r["po"], r["exact"]), deg=True)
            out[f"plate_{bc}_theta{th:g}"] = {"corr": float(cc), "level_dB": float(lvl),
                                              "phase_deg": float(ph), "expected_phase_deg": 0 if sign < 0 else 180}
    # 2. closed PEC cylinder, plane wave from +z: exact BIE vs the Bessel series
    a = 4.0
    n = int(math.ceil(2 * math.pi * a / DS))
    t = np.linspace(0, 2 * math.pi, n + 1)[::-1]                       # clockwise: left normal outward
    P = np.stack([a * np.cos(t), a * np.sin(t)], 1)
    S = segs(P)
    pw = (lambda x, z: np.exp(-1j * k * z))
    for bc in ("dirichlet", "neumann"):
        dens = solve_exact(S, k, pw, None, bc)
        F = far_field(S, k, dens, "single" if bc == "dirichlet" else "double", phi)
        Fm = mie_cylinder(a, k, phi, bc)
        cc = abs(np.vdot(F, Fm)) / (np.linalg.norm(F) * np.linalg.norm(Fm))
        rel = np.linalg.norm(F - Fm) / np.linalg.norm(Fm)
        out[f"cylinder_ka{k*a:.1f}_{bc}"] = {"corr": float(cc), "rel_err": float(rel)}
    return out


# ---------------------------------------------------------------------------
# the study
# ---------------------------------------------------------------------------
HALF = 160.0                          # profile half-length (mm): the beam is < 1e-3 at the ends
CRACK_W = (1.5, 2.0, 2.5, 3.0)        # config.CRACK_WIDTH_RANGE, end points + inside
CRACK_D = (4.0, 6.0, 8.0)             # config.CRACK_DEPTH_RANGE
THETA_LOC = {"crown": 55.0, "shoulder": 35.0, "gauge corner": 20.0}


def cut(kind: str, theta_loc: float) -> tuple[str, float, float]:
    """(bc, incidence angle in the cut, wavenumber) for a cut of the horn's beam.
    x-z: E_y along the invariant axis -> Dirichlet, in-plane incidence at theta.
    y-z: E_y in the plane -> Neumann; k_x lies along the invariant axis, so the
         reduced problem is normal incidence at k_t = k cos(theta)."""
    if kind == "x-z":
        return "dirichlet", theta_loc, K0
    return "neumann", 0.0, K0 * math.cos(math.radians(theta_loc))


def case_list(quick: bool = False) -> list[dict]:
    W = CRACK_W[::3] if quick else CRACK_W
    D = CRACK_D[::2] if quick else CRACK_D
    out = []
    for where in ("crown", "shoulder"):
        th = THETA_LOC[where]
        for w in W:
            for d in D:
                for n in (1, 3):
                    out.append(dict(cls="crack", sub="longitudinal", cut="x-z", where=where, theta=th,
                                    width=w, depth=d, lines=n))
                    out.append(dict(cls="crack", sub="transverse", cut="y-z", where=where, theta=th,
                                    width=w, depth=d, lines=n))
        for d in ((1.5, 8.0) if quick else (1.5, 4.0, 8.0)):
            for fw in ((10.0, 30.0) if quick else (10.0, 20.0, 30.0)):
                for c in ("x-z", "y-z"):
                    out.append(dict(cls="dent", sub=c, cut=c, where=where, theta=th, depth=d, fwhm=fw))
    th = THETA_LOC["gauge corner"]
    for d in ((1.0, 5.0) if quick else (1.0, 3.0, 5.0)):
        for r in ((4.0, 10.0) if quick else (4.0, 7.0, 10.0)):
            for c in ("x-z", "y-z"):
                out.append(dict(cls="shell", sub=c, cut=c, where="gauge corner", theta=th, depth=d, radius=r))
    for th, where in ((20.0, "gauge corner"), (40.0, "shoulder")):
        for d in ((2.0, 8.0) if quick else (2.0, 5.0, 8.0)):
            for R in ((150.0,) if quick else (150.0, 300.0)):
                out.append(dict(cls="wear", sub="x-z", cut="x-z", where=where, theta=th, depth=d, radius=R))
    return out


def depth_profile(c: dict):
    if c["cls"] == "crack":
        return slots(c["width"], c["depth"], c.get("lines", 1))
    if c["cls"] == "dent":
        return gaussian(c["depth"], c["fwhm"]), ()
    if c["cls"] == "shell":
        return shell(c["depth"], c["radius"]), ()
    if c["cls"] == "wear":
        return wear(c["depth"], c["radius"]), ()
    raise ValueError(c["cls"])


def case_id(c: dict) -> str:
    keys = [k for k in ("cls", "sub", "where", "width", "depth", "lines", "fwhm", "radius") if k in c]
    return "_".join(str(c[k]).replace(" ", "") for k in keys)


def ms_centre(c: dict) -> float:
    """Direction of the metasurface's centre in the cut's local frame (deg from the
    local normal, + toward the horn). A point whose normal is tilted tau toward the
    horn sees global vertical at -tau; tau = 55 - theta_loc for the x-z cuts."""
    return -(math.degrees(config.THETA_INC) - c["theta"]) if c["cut"] == "x-z" else 0.0


def _corr(a, b):
    return float(abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


def score_case(c: dict, flat_cache: dict, phi: np.ndarray) -> dict:
    bc, th, k = cut(c["cut"], c["theta"])
    fn, knots = depth_profile(c)
    key = (bc, th, round(k, 9))
    if key not in flat_cache:
        flat_cache[key] = solve_case(surface(lambda x: 0 * x, HALF), th, bc, k, phi)
    fl = flat_cache[key]
    r = solve_case(surface(fn, HALF, breaks=knots), th, bc, k, phi)
    # complex gain fitted on the flat surface, over the whole half-space (as
    # fdtd_agreement aligns the absolute field): removes rail3D's Neumann sign
    g = np.vdot(fl["po"], fl["exact"]) / np.vdot(fl["po"], fl["po"])
    d_ex = r["exact"] - fl["exact"]
    d_po = g * (r["po"] - fl["po"])
    ms = np.abs(phi - ms_centre(c)) <= MS_HALF_ANGLE[c["cut"]]
    ref = np.sum(abs(fl["exact"]) ** 2)
    out = dict(c, id=case_id(c), bc=bc, k=k, theta_cut=th)
    for wname, m in (("ms", ms), ("hemi", np.ones_like(ms))):
        out[f"corr_{wname}"] = _corr(d_po[m], d_ex[m])
        out[f"level_dB_{wname}"] = float(10 * np.log10(np.sum(abs(d_po[m]) ** 2) / np.sum(abs(d_ex[m]) ** 2)))
        out[f"signal_dB_{wname}"] = float(10 * np.log10(np.sum(abs(d_ex[m]) ** 2) / ref))
    out["lit_frac_defect"] = float(r["lit"][np.abs(r["S"]["c"][:, 1]) > 1e-6].mean()) \
        if np.any(np.abs(r["S"]["c"][:, 1]) > 1e-6) else 1.0
    out["_pattern"] = {"phi": phi.tolist(), "d_ex": [d_ex.real.tolist(), d_ex.imag.tolist()],
                       "d_po": [d_po.real.tolist(), d_po.imag.tolist()]}
    return out


def depth_sweep(width: float = 2.0, theta: float = 55.0, phi=None) -> dict:
    """Difference-field level in the metasurface window vs crack depth, exact and
    rail3D, both cuts: the waveguide physics in one picture."""
    phi = np.linspace(-80, 80, 321) if phi is None else phi
    depths = np.arange(0.5, 10.01, 0.5)
    out = {"depth": depths.tolist(), "width": width, "theta": theta}
    for sub, cutk in (("longitudinal", "x-z"), ("transverse", "y-z")):
        bc, th, k = cut(cutk, theta)
        fl = solve_case(surface(lambda x: 0 * x, HALF), th, bc, k, phi)
        g = np.vdot(fl["po"], fl["exact"]) / np.vdot(fl["po"], fl["po"])
        ms = np.abs(phi) <= MS_HALF_ANGLE[cutk]
        ref = np.sum(abs(fl["exact"]) ** 2)
        ex, po, cc = [], [], []
        for d in depths:
            fn, kn = slots(width, d)
            r = solve_case(surface(fn, HALF, breaks=kn), th, bc, k, phi)
            de, dp = (r["exact"] - fl["exact"])[ms], (g * (r["po"] - fl["po"]))[ms]
            ex.append(10 * np.log10(np.sum(abs(de) ** 2) / ref))
            po.append(10 * np.log10(np.sum(abs(dp) ** 2) / ref))
            cc.append(_corr(dp, de))
        out[sub] = {"exact_dB": ex, "rail3d_dB": po, "corr": cc, "k": k}
        print(f"  depth sweep {sub}: done")
    return out


def near_field_map(width: float, depth: float, cutk: str, theta: float = 55.0,
                   box=None, h: float = 0.12) -> dict:
    """|total field| around one slot from the exact solution (16-point quadrature
    on segments near each grid point)."""
    box = (-8.0, 8.0, -depth - 1.5, 5.0) if box is None else box
    bc, th, k = cut(cutk, theta)
    fn, kn = slots(width, depth)
    S = segs(surface(fn, HALF, breaks=kn))
    psi, grad, _ = beam(th, k)
    dens = solve_exact(S, k, psi, grad, bc)
    xs = np.arange(box[0], box[1] + h / 2, h)
    zs = np.arange(box[2], box[3] + h / 2, h)
    X, Z = np.meshgrid(xs, zs, indexing="xy")
    P = np.stack([X.ravel(), Z.ravel()], 1)
    inside = fn(P[:, 0]) * -1 < P[:, 1] - 1e-9            # above the surface: air
    tot = psi(P[:, 0], P[:, 1]).astype(complex)
    near = np.nonzero(np.abs(S["c"][:, 0]) < 30)[0]
    far = np.setdiff1d(np.arange(len(S["L"])), near)
    for idx, gl in ((far, GL4), (near, GL16)):
        pts, wq = _quad_pts(S, gl, idx)
        for i0 in range(0, len(P), 2048):
            p = P[i0:i0 + 2048, None, None, :]
            d = p - pts[None]
            R = np.maximum(np.linalg.norm(d, axis=-1), 1e-6)
            if bc == "dirichlet":
                ker = 0.25j * hankel1(0, k * R)
            else:
                ker = 0.25j * k * hankel1(1, k * R) * np.sum(d * S["n"][idx][None, :, None, :], -1) / R
            tot[i0:i0 + 2048] += np.einsum("pnq,nq,n->p", ker, wq, dens[idx])
    A = np.where(inside, np.abs(tot), np.nan).reshape(X.shape)
    return {"x": xs, "z": zs, "abs": A, "profile_x": S["a"][:, 0], "profile_z": S["a"][:, 1],
            "bc": bc, "k": k}


def run_study(quick: bool = False) -> dict:
    phi = np.linspace(-80, 80, 321)
    cases = case_list(quick)
    flat: dict = {}
    rows = []
    t0 = time.time()
    for i, c in enumerate(cases):
        rows.append(score_case(c, flat, phi))
        r = rows[-1]
        print(f"  [{i + 1:3d}/{len(cases)}] {r['id']:52s} MS: corr {r['corr_ms']:.3f} "
              f"level {r['level_dB_ms']:+6.2f} dB  signal {r['signal_dB_ms']:+6.1f} dB   "
              f"({(time.time() - t0) / (i + 1):.1f} s/case)", flush=True)
    sweep = depth_sweep(phi=phi)
    res = {"generated": time.strftime("%Y-%m-%d %H:%M"), "wvl": config.WVL, "ds": DS, "waist": WAIST,
           "half_length": HALF, "ms_half_angle_deg": MS_HALF_ANGLE, "validation": validate(),
           "cases": rows, "depth_sweep": sweep}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, indent=1))
    print(f"wrote {OUT_JSON}")
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--quick", action="store_true", help="a thinned case list (smoke test)")
    ap.add_argument("--figures", action="store_true", help="figures from the stored JSON only")
    a = ap.parse_args()
    if a.validate:
        t0 = time.time()
        v = validate()
        for kk, vv in v.items():
            print(f"  {kk:32s} " + "  ".join(f"{q} {x:.5g}" for q, x in vv.items()))
        print(f"  ({time.time() - t0:.1f} s)")
        return 0
    if not a.figures:
        run_study(a.quick)
    import rail_targets_figs
    rail_targets_figs.po_figures(json.loads(OUT_JSON.read_text()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
