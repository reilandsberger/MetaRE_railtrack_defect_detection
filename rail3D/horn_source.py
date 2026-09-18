"""The horn's wavefront as a Lumerical FDTD Import source.

    python horn_source.py --out data/generated/fdtd_horn          # build + self-check
    python horn_source.py --out DIR --sample 0.25 --z-src 15      # explicit settings

rail3D's horn is Face3D's pyramidal-aperture model (field3d.aperture_field):
17.1 x 13.7 mm aperture, TE10 cosine along the plane of incidence, so E is along
y -- s-polarised -- with its phase centre 140 mm from the crown at 55 deg. FDTD
cannot hold the horn itself (the box would be ~390 Mcells), so the horn enters
FDTD the way Lumerical documents for arbitrary beams: an **Import source**, a
z-normal plane of E and H injected downward just above the rail.

What this writes (all SI, as Lumerical requires):

  horn_source.mat        x, y (column vectors, m), z (m), f (Hz), and
                         Ex, Ey, Ez, Hx, Hy, Hz as (nx, ny) complex matrices --
                         the layout of Lumerical's own usr_custom_source.lsf
  load_horn_source.lsf   builds the rectilineardataset("EM fields", x, y, z),
                         adds E and H, loads it into an Import source with
                         importdataset(), sets a single wavelength, and saves
                         horn_EM_dataset.mat for the GUI's "Import Source" button
  horn_source.json       the plane, window, sampling and every self-check number

Design decisions, each measured rather than assumed (README finding 28):

  PLANE HEIGHT  z = 3 lambda (15 mm): above the crown and the crack's mesh
      override (top 9.9 mm), below the z = 30 monitor. The monitor then sits on
      the far side of the source from the rail and records ONLY the up-going
      reflected field -- the separation TFSF used to provide, obtained without it.
  WINDOW  the horn beam is far wider than the rail (at crown height its -20 dB
      contour spans y = +/-128 mm), so the plane covers the rays that can REACH
      the rail -- every lit facet projected toward the phase centre -- plus
      6 lambda, with a 2 lambda raised-cosine edge taper. Checked by ASM-
      propagating the windowed field down in free space and comparing it with
      rail3D's own horn field on the rail footprint: complex corr 0.9989 at the
      crown, 1.0000 at z = -40, 0.9991 at z = -80, while carrying 72% of the
      horn's power (the rest misses the rail).
  H FIELD  supplied, not left to Lumerical. Its docs: without H the source
      "makes certain assumptions about the change of phase ... [that] may lead
      to significant errors for ... more complex field profiles". A 55 deg beam
      through a horizontal plane is exactly that. E_y is rail3D's scalar field
      unchanged; each plane-wave component gets E ~ y - (y.k^)k^ (y projected
      transverse to its own k -- the least cross-polarisation consistent with
      Maxwell, and the far field of a y-directed source); H = (k x E)/(omega
      mu0), the exp(-i omega t) relation, which is
      Lumerical's convention as well as rail3D's (Lumerical: "P(omega) = int
      e^{i omega t} P(t) dt").
  SAMPLING  lambda/20 (0.25 mm): the phase ramp across the plane is
      k0 sin(55 deg) = 0.26 rad per sample, so interpolating onto a lambda/10-20
      FDTD mesh costs < 1% amplitude.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rail3d import config, field3d, mesh3d, sections

Z_SRC = 3 * config.WVL            # 15 mm
MARGIN = 6 * config.WVL           # 30 mm beyond the rail's ray bundle
TAPER = 2 * config.WVL            # raised-cosine edge
SAMPLE = config.WVL / 20          # 0.25 mm
MU0 = 4e-7 * np.pi
C0 = 299_792_458.0


def horn_centre() -> np.ndarray:
    th = config.THETA_INC
    return np.array([config.DIST_ANT * np.sin(th), 0.0, config.DIST_ANT * np.cos(th)])


def ray_bundle(z_src: float = Z_SRC) -> dict:
    """Where rays that can reach the rail cross z = z_src.

    Every rail facet whose outward normal faces the horn, projected along the
    line to the horn's phase centre. Always the INTACT rail, so one source file
    serves every rung (plate, intact, crack) and the difference field in rung 3
    is taken between runs with the identical source.
    """
    sec = sections.load_reference_section()
    ds = config.OCCLUDER_DS
    v, f = mesh3d.sweep_rail_mesh(sec, defect_params=None, slice_ds=ds, arc_ds=ds,
                                  n_arc=mesh3d.default_arc_count(sec, ds))
    v, f = v.numpy(), f.numpy()
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cen = (a + b + c) / 3
    nrm = np.cross(b - a, c - a)
    A = horn_centre()
    lit = (nrm * (A - cen)).sum(1) > 0
    top = np.argmax(cen[:, 2] - 1e-3 * np.abs(cen[:, 0]))       # the crown facet
    if not lit[top]:
        raise RuntimeError("the crown does not face the horn: mesh normals point "
                           "inward, so the lit set would be the shadowed side")
    P = cen[lit]
    t = (z_src - P[:, 2]) / (A[2] - P[:, 2])
    hit = P + t[:, None] * (A - P)
    return {"x": [float(hit[:, 0].min()), float(hit[:, 0].max())],
            "y": [float(hit[:, 1].min()), float(hit[:, 1].max())],
            "lit_facets": int(lit.sum()), "facets": int(len(lit))}


def window(bundle: dict, margin: float = MARGIN, sample: float = SAMPLE) -> dict:
    """Bundle + margin, snapped outward to the sample grid."""
    snap = lambda u, up: (np.ceil if up else np.floor)(u / sample) * sample   # noqa: E731
    return {"x": [float(snap(bundle["x"][0] - margin, False)),
                  float(snap(bundle["x"][1] + margin, True))],
            "y": [float(snap(bundle["y"][0] - margin, False)),
                  float(snap(bundle["y"][1] + margin, True))]}


def taper(u: np.ndarray, lo: float, hi: float, T: float = TAPER) -> np.ndarray:
    """1 inside, raised-cosine to exactly 0 at the window boundary."""
    w = np.clip(np.minimum(u - lo, hi - u) / T, 0.0, 1.0)
    return 0.5 * (1 - np.cos(np.pi * w))


def horn_field(xs: np.ndarray, ys: np.ndarray, z: float, device, rows: int = 32) -> np.ndarray:
    """rail3D's free-space horn field on a z-plane, in row chunks (memory-safe)."""
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Xt = torch.tensor(X, dtype=torch.float32, device=device)[None]
    Yt = torch.tensor(Y, dtype=torch.float32, device=device)[None]
    out = []
    for i in range(0, X.shape[0], rows):
        out.append(field3d.horn_to_plane(
            Xt[:, i:i + rows], Yt[:, i:i + rows], z, config.WVL, config.THETA_INC,
            config.SIZE_ANT, config.DIST_ANT, config.RESOL_ANT).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.complex128)


def _k_grid(nx, ny, sample_mm):
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=sample_mm * 1e-3)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=sample_mm * 1e-3)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    k0 = 2 * np.pi / (config.WVL * 1e-3)
    # DOWNWARD: kz < 0 for propagating waves, -i|kz| (decaying downward) otherwise
    kz = -np.sqrt((k0 ** 2 - KX ** 2 - KY ** 2).astype(complex))
    return KX, KY, kz, k0


def vector_fields(ey: np.ndarray, sample_mm: float):
    """Complete a scalar E_y into a transverse, downward-propagating E and H.

    Returns (E, H) as tuples of (nx, ny) complex arrays in consistent SI units
    (H = E / Z0 for a plane wave). E_y is returned unchanged.
    """
    KX, KY, kz, k0 = _k_grid(*ey.shape, sample_mm)
    # Each plane-wave component carries y-hat projected transverse to its own
    # k: E ~ y - (y.k^)k^, the far field of a y-directed source. Scaled so E_y
    # equals rail3D's scalar exactly. Forcing E_x = 0 instead dumped all of the
    # transversality into E_z and gave Ez/Ey = 0.35 rms for this steep, wide
    # beam -- far more cross-polarisation than a y-polarised horn radiates, and
    # further from the scalar s-pol model FDTD is being compared against.
    ux, uy, uz = KX / k0, KY / k0, kz / k0
    den = 1 - uy ** 2
    den = np.where(np.abs(den) < 0.02, 0.02, den)                       # grazing-in-y guard
    Ey_k = np.fft.fft2(ey)
    Ex_k = -(ux * uy / den) * Ey_k
    Ez_k = -(uz * uy / den) * Ey_k
    w_mu = 2 * np.pi * (C0 / (config.WVL * 1e-3)) * MU0
    Hx_k = (KY * Ez_k - kz * Ey_k) / w_mu
    Hy_k = (kz * Ex_k - KX * Ez_k) / w_mu
    Hz_k = (KX * Ey_k - KY * Ex_k) / w_mu
    inv = np.fft.ifft2
    return ((inv(Ex_k), ey, inv(Ez_k)), (inv(Hx_k), inv(Hy_k), inv(Hz_k)))


def flux_down_fraction(E, H) -> float:
    """Fraction of the plane's Poynting flux that goes DOWN (-z). Want ~1."""
    Sz = 0.5 * np.real(E[0] * np.conj(H[1]) - E[1] * np.conj(H[0]))
    return float(-Sz.sum() / np.abs(Sz).sum())


def illumination_fidelity(win: dict, z_src: float, device, dx: float = 1.0,
                          L: float = 640.0, planes=(0.0, -40.0, -80.0)) -> dict:
    """Free-space check: windowed + tapered source, ASM-propagated DOWN, vs the
    full horn field on the rail's footprint (|x| <= 40, |y| <= 60)."""
    xs = np.arange(-L / 2, L / 2, dx) + dx / 2
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    full = horn_field(xs, xs, z_src, device)
    W = taper(X, *win["x"]) * taper(Y, *win["y"])
    S = np.fft.fft2(full * W)
    _, _, kz, _ = _k_grid(len(xs), len(xs), dx)
    foot = (np.abs(X) <= 40) & (np.abs(Y) <= 60)
    out = {}
    for z in planes:
        prop = np.fft.ifft2(S * np.exp(-1j * kz * (z_src - z) * 1e-3))
        tgt = horn_field(xs, xs, z, device)
        p, q = prop[foot], tgt[foot]
        out[f"z={z:g}"] = float(np.abs(np.vdot(p, q)) / (np.linalg.norm(p) * np.linalg.norm(q)))
    out["power_captured"] = float((np.abs(full * W) ** 2).sum() / (np.abs(full) ** 2).sum())
    return out


LSF = """# load_horn_source.lsf -- written by rail3D/horn_source.py. Do not edit the data.
#
# Builds Lumerical's EM dataset from horn_source.mat (SI units: m, Hz, V/m, A/m),
# using exactly the format of Lumerical's own usr_custom_source.lsf example:
#   rectilineardataset("EM fields", x, y, z) + addattribute("E"/"H").
# Run it from the Script File Editor with this folder as the working directory
# (File > Working Directory). It then either creates the Import source for you
# (CREATE_SOURCE = 1) or only saves horn_EM_dataset.mat for the GUI button.

CREATE_SOURCE = 1;              # 0: just write horn_EM_dataset.mat
SOURCE_NAME = "horn_source";

matlabload("horn_source.mat");  # x, y, z, f, Ex, Ey, Ez, Hx, Hy, Hz
EM = rectilineardataset("EM fields", x, y, z);
EM.addparameter("lambda", c/f, "f", f);
EM.addattribute("E", Ex, Ey, Ez);
EM.addattribute("H", Hx, Hy, Hz);
matlabsave("horn_EM_dataset.mat", EM);
?"wrote horn_EM_dataset.mat";

if (CREATE_SOURCE == 1) {
    addimportedsource;
    set("name", SOURCE_NAME);
    importdataset(EM);
    set("direction", "Backward");       # inject DOWN, toward the rail (-z)
    set("center wavelength", c/f);
    set("wavelength span", 0);          # single frequency
    ?"created Import source '" + SOURCE_NAME + "' at z = " + num2str(z*1e3) + " mm";
}
"""


def build(out: Path, device, z_src: float = Z_SRC, margin: float = MARGIN,
          sample: float = SAMPLE, check: bool = True, conjugate: bool = False) -> dict:
    from scipy.io import savemat
    out.mkdir(parents=True, exist_ok=True)
    bundle = ray_bundle(z_src)
    win = window(bundle, margin, sample)
    xs = np.arange(win["x"][0], win["x"][1] + 1e-9, sample)
    ys = np.arange(win["y"][0], win["y"][1] + 1e-9, sample)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    ey = horn_field(xs, ys, z_src, device) * taper(X, *win["x"]) * taper(Y, *win["y"])
    E, H = vector_fields(ey, sample)
    if conjugate:
        # ONLY for the rung-0 case in LUMERICAL.md section 6: Lumerical documents
        # conjugating BEAM profiles for backward injection ("Source Rotations in
        # 3D FDTD"); if it turns out to do the same to imported data, writing the
        # conjugate here undoes it. Never the default -- rung 0 decides.
        E = tuple(np.conj(c) for c in E)
        H = tuple(np.conj(c) for c in H)
    f_hz = C0 / (config.WVL * 1e-3)
    savemat(str(out / "horn_source.mat"),
            {"x": xs * 1e-3, "y": ys * 1e-3, "z": np.array(z_src * 1e-3), "f": np.array(f_hz),
             "Ex": E[0], "Ey": E[1], "Ez": E[2], "Hx": H[0], "Hy": H[1], "Hz": H[2]},
            oned_as="column", do_compression=True)
    (out / "load_horn_source.lsf").write_text(LSF, encoding="utf-8")

    eh = float(np.sqrt((np.abs(E[0]) ** 2 + np.abs(E[1]) ** 2 + np.abs(E[2]) ** 2).sum()
                       / (np.abs(H[0]) ** 2 + np.abs(H[1]) ** 2 + np.abs(H[2]) ** 2).sum()))
    info = {
        "plane_z_mm": z_src, "window_mm": win, "ray_bundle_mm": bundle,
        "margin_mm": margin, "taper_mm": TAPER, "sample_mm": sample,
        "grid": [len(xs), len(ys)], "frequency_Hz": f_hz, "conjugated": conjugate,
        "polarisation": "E along y (s-pol, TE10 of the horn)",
        "injection": "z-normal Import source, direction Backward (-z)",
        "flux_down_fraction": flux_down_fraction(E, H),
        "impedance_ohm": eh, "Z0_ohm": float(MU0 * C0),
        "ex_over_ey_rms": float(np.sqrt((np.abs(E[0]) ** 2).sum() / (np.abs(E[1]) ** 2).sum())),
        "ez_over_ey_rms": float(np.sqrt((np.abs(E[2]) ** 2).sum() / (np.abs(E[1]) ** 2).sum())),
        "files": ["horn_source.mat", "load_horn_source.lsf"],
    }
    if check:
        info["illumination_fidelity"] = illumination_fidelity(win, z_src, device)
    (out / "horn_source.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--z-src", type=float, default=Z_SRC)
    ap.add_argument("--margin", type=float, default=MARGIN)
    ap.add_argument("--sample", type=float, default=SAMPLE)
    ap.add_argument("--profile", default="laptop", choices=list(config.PROFILES))
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--conjugate", action="store_true",
                    help="write conj(E), conj(H): ONLY if rung 0 matched conjugated "
                         "with Direction = Backward confirmed (LUMERICAL.md section 6)")
    args = ap.parse_args()
    info = build(Path(args.out), config.get_device(args.profile), args.z_src,
                 args.margin, args.sample, check=not args.no_check, conjugate=args.conjugate)
    w = info["window_mm"]
    print(f"horn Import source at z = {info['plane_z_mm']:g} mm: window x {w['x']} y {w['y']} "
          f"mm, {info['grid'][0]}x{info['grid'][1]} samples at {info['sample_mm']} mm")
    print(f"  flux downward {info['flux_down_fraction']*100:.2f}%   |E|/|H| "
          f"{info['impedance_ohm']:.1f} ohm (Z0 {info['Z0_ohm']:.1f})   "
          f"cross-pol Ex/Ey {info['ex_over_ey_rms']:.3f}, Ez/Ey {info['ez_over_ey_rms']:.3f} rms")
    if "illumination_fidelity" in info:
        fid = info["illumination_fidelity"]
        print("  rail illumination vs rail3D's full horn: " + "  ".join(
            f"{k} {v:.4f}" for k, v in fid.items() if k.startswith("z=")) +
            f"   (power on plane {fid['power_captured']*100:.0f}%)")
    print(f"  -> {Path(args.out) / 'horn_source.mat'}  + load_horn_source.lsf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
