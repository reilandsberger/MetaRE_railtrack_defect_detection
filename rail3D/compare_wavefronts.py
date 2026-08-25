"""Compare every simulation VERSION's predicted wavefront on the measurement plane.

One physical rail sample, solved many ways, plotted and scored side by side —
the figure set for "how much does each modelling choice actually change the
answer", and the harness for dropping an external full-wave (FDTD/MoM)
reference in beside them.

Versions covered (pick with --variants, default = all):

  lam8 / lam5        the SAME physical rail at the pre- and post-migration
                     geometry. Both planes are 60x30 covering the same angular
                     extent, so the arrays are directly comparable and the
                     comparison IS the scaled-replica claim.
  psi0/psi1/psi2/tot the physics terms: direct horn, single bounce, double
                     bounce, and the sum the dataset actually stores.
  noshadow / mint*   the ray-cast self-shadow guard at several min_t values --
                     the field-level view of what config.SHADOW_MIN_T costs.
  mesh16             the same sample on a lambda/16 mesh (V7's reference).
  asm                the detector-plane field via the angular-spectrum
                     propagator instead of RS-FFT (V3's cross-check).

Outputs (data/figures/ + data/generated/):
  wavefront_maps.png       |psi| and arg(psi) maps per version, plus |delta|
  wavefront_cuts.png       central-cut amplitude and phase, all versions overlaid
  wavefront_metrics.png    rel-L2 / complex correlation / barcode cosine bars
  wavefront_comparison.json    every number behind the figures
  wavefront_fields.npz     all complex fields + metadata (presentation source,
                           and the file an external solver's result joins)

FDTD / full-wave comparison:
  --export-case DIR   writes the geometry (OBJ), source and plane definition so
                      an external solver can be set up on the IDENTICAL problem
  --external ref.npz  loads {field: complex (NX,NY), label: str} and includes it
                      as another version in every figure and metric

Usage (from rail3D/):
    python compare_wavefronts.py --profile lab
    python compare_wavefronts.py --profile laptop --seg 40 --sample crack
    python compare_wavefronts.py --export-case data/generated/fdtd_case
    python compare_wavefronts.py --external fdtd_result.npz
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

from rail3d import config, field3d, mesh3d, optics3d, sections

FIG = config.FIGURE_DIR
GEN = config.GENERATED_DIR

# The pre-migration geometry, written out explicitly so this script can solve
# it without touching config (field3d takes every physical quantity as an
# argument -- it is the one fully wavelength-parametric module).
LAM8 = dict(wvl=8.0, dx=4.0, nx=60, ny=30, h_ms=240.0, dist_ant=28 * 8.0,
            size_ant=(27.4, 21.9, 9.3, 6.2, 27.0), mesh_ds=1.0, occ_ds=4.0,
            layer=160.0)


def geom_now() -> dict:
    return dict(wvl=config.WVL, dx=config.DX, nx=config.NX, ny=config.NY,
                h_ms=config.H_MS, dist_ant=config.DIST_ANT,
                size_ant=tuple(config.SIZE_ANT), mesh_ds=config.MESH_DS,
                occ_ds=config.OCCLUDER_DS, layer=config.LAYER_DISTANCES[-1])


def plane_grid(g: dict, device):
    """Cell-centred (1, nx, ny) grids for an arbitrary geometry dict."""
    wx, wy = g["nx"] * g["dx"], g["ny"] * g["dx"]
    x = torch.arange(-wx / 2 + g["dx"] / 2, wx / 2, g["dx"], device=device)
    y = torch.arange(-wy / 2 + g["dx"] / 2, wy / 2, g["dx"], device=device)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    return X.reshape(1, g["nx"], g["ny"]), Y.reshape(1, g["nx"], g["ny"])


def solve(section, params, g, device, chunk, *, mesh_ds=None, shadow=None,
          min_t=None, terms="psi1", seg=None):
    """One wavefront on the measurement plane of geometry ``g``.

    terms: 'psi0' | 'psi1' | 'psi12' | 'tot'
    shadow: None (back-face culling only) or 'raycast'
    """
    ds = mesh_ds or g["mesh_ds"]
    X, Y = plane_grid(g, device)
    args = (X, Y, g["h_ms"], g["wvl"], config.THETA_INC, g["size_ant"],
            g["dist_ant"], config.RESOL_ANT)

    if terms == "psi0":
        return field3d.horn_to_plane(*args)          # (nx, ny), no batch dim

    n_arc = mesh3d.default_arc_count(section, ds)
    kw_mesh = {} if seg is None else {"seg_len": seg}
    v, f = mesh3d.sweep_rail_mesh(section, defect_params=params, slice_ds=ds,
                                  arc_ds=ds, n_arc=n_arc, **kw_mesh)
    kw = {}
    if shadow == "raycast":
        n_occ = mesh3d.default_arc_count(section, g["occ_ds"])
        v_o, f_o = mesh3d.sweep_rail_mesh(section, defect_params=params,
                                          slice_ds=g["occ_ds"], arc_ds=g["occ_ds"],
                                          n_arc=n_occ, **kw_mesh)
        kw = {"shadow": "raycast",
              "shadow_occluders": (v_o.to(device), f_o.to(device)),
              "shadow_min_t": min_t}
    want2 = terms in ("psi12", "tot")
    # a 2-D vertex tensor makes scattered_fields squeeze the batch dim itself,
    # so psi1/psi2 already come back as (nx, ny)
    psi1, psi2 = field3d.scattered_fields(v.to(device), f.to(device), *args,
                                          chunk_faces=chunk,
                                          compute_psi2=want2, **kw)
    out = psi1 + (psi2 if want2 else 0)
    if terms == "tot":
        out = out + field3d.horn_to_plane(*args)
    return out


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def rel_l2(a, b):
    return float((a - b).norm() / b.norm())


def complex_corr(a, b):
    num = torch.vdot(a.reshape(-1), b.reshape(-1)).abs()
    return float(num / (a.norm() * b.norm()))


def amp_corr(a, b):
    x, y = a.abs().reshape(-1), b.abs().reshape(-1)
    x, y = x - x.mean(), y - y.mean()
    return float((x @ y) / (x.norm() * y.norm()))


def barcode(psi, g, device):
    """Detector powers through the dense lattice — the level README finding 2
    says to judge fidelity at (raw complex-field L2 never converges: speckle)."""
    det = optics3d.SoftDetector2D(
        centers=config.dense_detector_centers(nx=g["nx"], ny=g["ny"]),
        det_size=config.DET_SIZE, dx=g["dx"], nx=g["nx"], ny=g["ny"],
        wx=g["nx"] * g["dx"], wy=g["ny"] * g["dx"]).to(device)
    return det.hard_powers((psi.abs() ** 2).unsqueeze(0))[0]


def cos_sim(a, b):
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-12))


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_maps(fields, ref_key, path):
    keys = list(fields)
    n = len(keys)
    fig, axes = plt.subplots(n, 3, figsize=(11.5, 2.5 * n), squeeze=False)
    ref = fields[ref_key]["psi"]
    for r, k in enumerate(keys):
        psi = fields[k]["psi"]
        a = psi.abs().cpu().numpy()
        ext = (-0.5, 0.5, -0.5, 0.5)          # normalized aperture coords
        im = axes[r][0].imshow(a / a.max(), extent=ext, origin="lower",
                               cmap="inferno", aspect="auto")
        plt.colorbar(im, ax=axes[r][0], fraction=0.04)
        axes[r][0].set_ylabel(f"{k}\nx / W", fontsize=8)

        im = axes[r][1].imshow(np.angle(psi.cpu().numpy()), extent=ext,
                               origin="lower", cmap="twilight", aspect="auto",
                               vmin=-np.pi, vmax=np.pi)
        plt.colorbar(im, ax=axes[r][1], fraction=0.04)

        if psi.shape == ref.shape:
            d = (psi - ref).abs().cpu().numpy() / ref.abs().max().item()
            im = axes[r][2].imshow(d, extent=ext, origin="lower", cmap="viridis",
                                   aspect="auto")
            plt.colorbar(im, ax=axes[r][2], fraction=0.04)
            axes[r][2].set_title(f"rel L2 {rel_l2(psi, ref):.3f}", fontsize=7)
        else:
            axes[r][2].axis("off")
        for c in range(3):
            axes[r][c].set_xticks([]) if r < n - 1 else axes[r][c].set_xlabel("y / W", fontsize=7)
            axes[r][c].tick_params(labelsize=6)
    for c, t in enumerate([r"$|\psi|$ (each normalized)", r"$\arg\psi$",
                           f"$|\\psi-\\psi_{{ref}}|$  (ref = {ref_key})"]):
        axes[0][c].set_title(t, fontsize=9)
    fig.suptitle("Predicted wavefront at the measurement plane, per simulation version",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=190)
    plt.close(fig)


def fig_cuts(fields, ref_key, path):
    """THE comparison graph: central-cut amplitude and phase, all versions.

    Raw phase is dominated by the common propagation ramp (tens of radians
    across the aperture), which hides every difference between versions — so
    the right column shows each version RELATIVE to the reference, which is
    what actually distinguishes the modelling choices.
    """
    ref = fields[ref_key]["psi"].cpu().numpy()
    nxr, nyr = ref.shape
    ref_cut = ref[:, nyr // 2]
    fig, ax = plt.subplots(2, 2, figsize=(14.5, 8), sharex=True)

    amp_dev, ph_dev = [], []
    for k, d in fields.items():
        psi = d["psi"].cpu().numpy()
        nx, ny = psi.shape
        cut = psi[:, ny // 2]
        xn = (np.arange(nx) - (nx - 1) / 2) / nx      # normalized aperture coord
        a = np.abs(cut)
        # a version that lands EXACTLY on the reference would hide under it —
        # which is itself a result worth reading off the legend (e.g. a shadow
        # guard so large it changes nothing), so say so and dash it
        same = psi.shape == ref.shape and rel_l2(d["psi"], fields[ref_key]["psi"]) < 1e-9
        lab = k + ("   [identical to ref]" if same and k != ref_key else "")
        style = dict(lw=1.4, label=lab)
        if same and k != ref_key:
            style.update(lw=2.6, ls=(0, (2, 3)), zorder=6)
        if k == ref_key:
            style.update(lw=2.4, color="k", zorder=5)
        ax[0][0].plot(xn, a / a.max(), **style)
        ph = np.unwrap(np.angle(cut))
        ax[0][1].plot(xn, ph - ph[nx // 2], **style)

        if psi.shape != ref.shape:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.abs(cut) / np.abs(ref_cut)
        dphi = np.angle(cut / np.where(np.abs(ref_cut) > 0, ref_cut, 1))
        ax[1][0].plot(xn, ratio, **style)
        ax[1][1].plot(xn, dphi, **style)
        if k != ref_key and "psi0" not in k:
            amp_dev.append(ratio[np.isfinite(ratio)])
            ph_dev.append(dphi)

    ax[0][0].set(ylabel=r"$|\psi|\;/\;\max$", title="Amplitude — central cut (y = 0)")
    ax[0][1].set(ylabel=r"unwrapped $\arg\psi$ − centre (rad)",
                 title="Phase — raw (the common propagation ramp dominates)")
    ax[1][0].set(ylabel=r"$|\psi|\;/\;|\psi_{ref}|$",
                 xlabel="x / aperture width  (normalized, so wavelengths overlay)",
                 title=f"Amplitude RELATIVE to {ref_key}")
    ax[1][1].set(ylabel=r"$\arg(\psi/\psi_{ref})$ (rad)",
                 xlabel="x / aperture width  (normalized, so wavelengths overlay)",
                 title=f"Phase RELATIVE to {ref_key}")
    # keep the difference panels readable: psi0 is not comparable to a
    # scattered field, so scale to the versions that are
    if amp_dev:
        v = np.concatenate(amp_dev)
        lo, hi = np.percentile(v, [1, 99])
        pad = max(0.05, 0.25 * (hi - lo))
        ax[1][0].set_ylim(max(0.0, lo - pad), hi + pad)
    if ph_dev:
        m = float(np.percentile(np.abs(np.concatenate(ph_dev)), 99))
        ax[1][1].set_ylim(-max(0.05, 1.3 * m), max(0.05, 1.3 * m))
    for a in ax.ravel():
        a.grid(alpha=0.3)
        a.axhline(0 if a in (ax[0][1], ax[1][1]) else 1, color="gray", lw=0.6, ls=":")
    ax[0][0].legend(fontsize=7, ncol=2)
    fig.suptitle("Predicted wavefront by simulation version — amplitude and phase", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=190)
    plt.close(fig)


def fig_metrics(rows, ref_key, path):
    ks = [r["version"] for r in rows]
    x = np.arange(len(ks))
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for a, key, title, lo in [
        (ax[0], "rel_l2", f"relative L2 vs {ref_key}\n(lower = closer)", 0),
        (ax[1], "complex_corr", "complex correlation\n(1 = identical field)", 0),
        (ax[2], "barcode_cos", "barcode cosine\n(what the detector actually sees)", 0),
    ]:
        vals = [r.get(key, np.nan) for r in rows]
        a.bar(x, vals, color="tab:blue")
        a.set_xticks(x)
        a.set_xticklabels(ks, rotation=45, ha="right", fontsize=7)
        a.set_title(title, fontsize=9)
        a.grid(alpha=0.3, axis="y")
        a.set_ylim(bottom=lo)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


# ---------------------------------------------------------------------------
# FDTD-ready case export
# ---------------------------------------------------------------------------
def export_case(section, params, g, out: Path, seg):
    out.mkdir(parents=True, exist_ok=True)
    ds = g["mesh_ds"]
    n_arc = mesh3d.default_arc_count(section, ds)
    kw = {} if seg is None else {"seg_len": seg}
    v, f = mesh3d.sweep_rail_mesh(section, defect_params=params, slice_ds=ds,
                                  arc_ds=ds, n_arc=n_arc, **kw)
    obj = out / "rail_surface.obj"
    with open(obj, "w") as fh:
        fh.write(f"# rail3D surface, mm, lambda={g['wvl']} mm, mesh {ds} mm\n")
        for p in v.numpy():
            fh.write(f"v {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")
        for t in f.numpy():
            fh.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")
    spec = {
        "units": "mm",
        "frequency_GHz": 299.792458 / g["wvl"],
        "wavelength_mm": g["wvl"],
        "geometry_file": obj.name,
        "geometry_note": ("swept rail head, crown at z=0; PEC assumed by the PO "
                          "solver (reflection coefficient -1, no receive-cosine)"),
        "source": {
            "type": "pyramidal horn",
            "aperture_A_B_mm": [g["size_ant"][0], g["size_ant"][1]],
            "waveguide_a_b_mm": [g["size_ant"][2], g["size_ant"][3]],
            "horn_length_mm": g["size_ant"][4],
            "distance_from_origin_mm": g["dist_ant"],
            "incidence_deg_from_z_in_xz": float(np.degrees(config.THETA_INC)),
            "center_xyz_mm": [float(g["dist_ant"] * np.sin(config.THETA_INC)), 0.0,
                              float(g["dist_ant"] * np.cos(config.THETA_INC))],
            "aperture_field": ("cosine taper along A x quadratic phase "
                               "exp(i*beta*(s^2)/(2*R_E)); scalar, single TE10 mode"),
        },
        "observation_plane": {
            "z_mm": g["h_ms"], "nx": g["nx"], "ny": g["ny"], "dx_mm": g["dx"],
            "x_extent_mm": [-g["nx"] * g["dx"] / 2, g["nx"] * g["dx"] / 2],
            "y_extent_mm": [-g["ny"] * g["dx"] / 2, g["ny"] * g["dx"] / 2],
            "sampling": "cell-centred",
        },
        "compare_back": {
            "save_as": "npz with keys: field (complex64 (nx,ny)), label (str)",
            "then_run": "python compare_wavefronts.py --external <file>.npz",
        },
    }
    (out / "case.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    (out / "README.md").write_text(
        "# rail3D full-wave comparison case\n\n"
        f"Everything needed to reproduce this exact scattering problem in an\n"
        f"external solver (FDTD / MoM / FEM) at {spec['frequency_GHz']:.1f} GHz.\n\n"
        "- `rail_surface.obj` — the rail surface, millimetres, crown at z=0.\n"
        "- `case.json` — source, incidence and observation-plane definition.\n\n"
        "Return the complex field sampled on the SAME cell-centred plane grid as\n"
        "`field` in an .npz, then:\n\n"
        "    python compare_wavefronts.py --external your_result.npz\n\n"
        "and it joins every figure and metric alongside the PO versions.\n\n"
        "## Expect PO and full-wave to differ where PO is known to be weak\n"
        "- grazing/terminator faces (this PO formulation applies no receive-cosine)\n"
        "- edge diffraction from the rail's cut ends\n"
        "- multiple scattering beyond the second bounce\n"
        "- features at or below the wavelength (hairline cracks)\n\n"
        "Those are the interesting comparisons, not failures to hide.\n",
        encoding="utf-8")
    print(f"  wrote FDTD case -> {out}  ({v.shape[0]} verts, {f.shape[0]} tris)")


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="lab", choices=list(config.PROFILES))
    ap.add_argument("--sample", default="crack",
                    choices=list(config.CLASS_NAMES) + ["intact"])
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--seg", type=float, default=None,
                    help="segment length override (mm); smaller = faster, for a laptop")
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--external", nargs="*", default=None,
                    help="npz files with keys 'field' (complex (nx,ny)) and 'label'")
    ap.add_argument("--export-case", default=None, metavar="DIR")
    ap.add_argument("--prefix", default="wavefront")
    args = ap.parse_args()

    device = config.get_device(args.profile)
    chunk = config.PROFILES[args.profile].chunk_faces
    config.ensure_dirs()
    section = sections.load_reference_section()
    g5, g8 = geom_now(), LAM8

    print(f"[rail3d] wavefront comparison on {device}   sample={args.sample}[{args.idx}]"
          f"   seg={args.seg or config.SEG_LEN} mm")

    params = None
    if args.sample != "intact":
        defect = None
        if args.sample in config.DATASET_DIRS:
            files = sections.get_dataset_files(args.sample)
            defect = sections.match_reference_width(
                sections.load_vertices_from_csv(files[args.idx]), section)
        params, _ = mesh3d.defect_params_for_sample(
            section, args.sample, defect, config.sample_seed(args.sample, args.idx),
            n_arc=mesh3d.default_arc_count(section, config.MESH_DS))
    else:
        params = {"class": "intact"}

    if args.export_case:
        export_case(section, params, g5, Path(args.export_case), args.seg)

    # resolution-independent params are reused across EVERY variant, so all of
    # them describe the same physical defect (that is the whole point)
    common = dict(section=section, params=params, device=device, chunk=chunk,
                  seg=args.seg)
    ALL = {
        "lam5 psi1 (raycast 3.0)": lambda: solve(g=g5, shadow="raycast", min_t=3.0, **common),
        "lam5 psi1 (raycast 0.125)": lambda: solve(g=g5, shadow="raycast", min_t=0.125, **common),
        "lam5 psi1 (no shadow)": lambda: solve(g=g5, **common),
        "lam5 psi0 (horn only)": lambda: solve(g=g5, terms="psi0", **common),
        "lam5 psi1+psi2": lambda: solve(g=g5, terms="psi12", **common),
        "lam5 tot (stored)": lambda: solve(g=g5, terms="tot", **common),
        "lam5 mesh lambda/16": lambda: solve(g=g5, mesh_ds=g5["wvl"] / 16, **common),
        "lam8 psi1 (raycast 3.0)": lambda: solve(g=g8, shadow="raycast", min_t=3.0, **common),
        "lam8 psi1 (no shadow)": lambda: solve(g=g8, **common),
    }
    names = args.variants or list(ALL)

    fields, t0 = {}, time.time()
    for k in names:
        if k not in ALL:
            print(f"  ! unknown variant {k!r}; choose from {list(ALL)}")
            continue
        t = time.time()
        psi = ALL[k]()
        fields[k] = {"psi": psi, "geom": g8 if k.startswith("lam8") else g5}
        print(f"  {k:32s} {tuple(psi.shape)}  {time.time()-t:6.1f} s")

    for path in args.external or []:
        z = np.load(path, allow_pickle=True)
        lab = str(z["label"]) if "label" in z else Path(path).stem
        psi = torch.as_tensor(z["field"]).to(torch.complex64).to(device)
        fields[f"EXT {lab}"] = {"psi": psi, "geom": g5}
        print(f"  {'EXT ' + lab:32s} {tuple(psi.shape)}  (external reference)")

    if not fields:
        print("no variants solved")
        return 1

    ref_key = ("lam5 psi1 (no shadow)" if "lam5 psi1 (no shadow)" in fields
               else list(fields)[0])
    ref = fields[ref_key]["psi"]
    ref_bar = barcode(ref, fields[ref_key]["geom"], device)

    rows = []
    for k, d in fields.items():
        psi, geo = d["psi"], d["geom"]
        row = {"version": k, "shape": list(psi.shape),
               "mean_abs": float(psi.abs().mean()),
               "peak_abs": float(psi.abs().max())}
        if psi.shape == ref.shape:
            row.update(rel_l2=rel_l2(psi, ref), complex_corr=complex_corr(psi, ref),
                       amp_corr=amp_corr(psi, ref),
                       barcode_cos=cos_sim(barcode(psi, geo, device), ref_bar))
        rows.append(row)

    fig_maps(fields, ref_key, FIG / f"{args.prefix}_maps.png")
    fig_cuts(fields, ref_key, FIG / f"{args.prefix}_cuts.png")
    fig_metrics(rows, ref_key, FIG / f"{args.prefix}_metrics.png")

    np.savez_compressed(
        GEN / f"{args.prefix}_fields.npz",
        **{k: v["psi"].cpu().numpy() for k, v in fields.items()},
        meta=json.dumps({"reference": ref_key, "sample": args.sample,
                         "idx": args.idx, "seg": args.seg or config.SEG_LEN,
                         "lam5": g5, "lam8": g8}))
    out = {"reference": ref_key, "sample": f"{args.sample}[{args.idx}]",
           "seg_mm": args.seg or config.SEG_LEN, "seconds": round(time.time() - t0, 1),
           "versions": rows}
    (GEN / f"{args.prefix}_comparison.json").write_text(json.dumps(out, indent=2),
                                                        encoding="utf-8")

    print(f"\n  {'version':32s} {'relL2':>8s} {'cplx corr':>10s} {'amp corr':>9s} {'barcode cos':>12s}")
    for r in rows:
        print(f"  {r['version']:32s} {r.get('rel_l2', float('nan')):8.4f} "
              f"{r.get('complex_corr', float('nan')):10.4f} "
              f"{r.get('amp_corr', float('nan')):9.4f} "
              f"{r.get('barcode_cos', float('nan')):12.4f}")
    print(f"\n  reference = {ref_key}")
    for n in (f"{args.prefix}_maps.png", f"{args.prefix}_cuts.png",
              f"{args.prefix}_metrics.png"):
        print(f"  figure -> {FIG / n}")
    print(f"  fields -> {GEN / f'{args.prefix}_fields.npz'}")
    print(f"  metrics -> {GEN / f'{args.prefix}_comparison.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
