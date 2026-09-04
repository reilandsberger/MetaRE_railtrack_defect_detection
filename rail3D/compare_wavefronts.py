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
  lam5 pw ...        the same rail under a unit PLANE WAVE instead of the horn
                     (--source plane). No psi0, no psi2 -- there is no horn to
                     radiate them. This is the variant to compare a full-wave
                     solver against, because it removes the horn aperture model
                     as a confound.

Outputs (data/figures/ + data/generated/):
  wavefront_maps.png       |psi| and arg(psi) maps per version, plus |delta|
  wavefront_cuts.png       central-cut amplitude and phase, all versions overlaid
  wavefront_metrics.png    rel-L2 / complex correlation / barcode cosine bars
  wavefront_comparison.json    every number behind the figures
  wavefront_fields.npz     all complex fields + metadata (presentation source,
                           and the file an external solver's result joins)

Full-wave comparison (MoM preferred -- see the exported README):
  --export-case DIR   writes STL + OBJ geometry, the source, the observation
                      plane and the sign/units conventions, so an external
                      solver is set up on the IDENTICAL problem
  --source plane      export (and compare against) plane-wave illumination
  --export-closed     cap the swept shell into a closed body first: MoM puts
                      current on BOTH faces of an open sheet, which is not what
                      an opaque rail does
  --external ref.npz  loads {field: complex (NX,NY), label: str}, aligns it to
                      our sign convention and amplitude, then includes it as
                      another version in every figure and metric

The rail is a PEC surface in an open region, so the natural reference is an
integral-equation solver (Ansys HFSS-IE with the IE licence, or Altair FEKO),
NOT Zemax (whose POP shares this code's scalar-Kirchhoff assumptions and would
only confirm itself) and not a full-scene FDTD (~390 Mcells at lambda/20).

Usage (from rail3D/):
    python compare_wavefronts.py --profile lab
    python compare_wavefronts.py --profile laptop --seg 40 --sample crack
    python compare_wavefronts.py --export-case data/generated/mom_case \
        --source plane --export-closed --sample intact --seg 30
    python compare_wavefronts.py --source plane --external mom_result.npz
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
          min_t=None, terms="psi1", seg=None, source="horn"):
    """One wavefront on the measurement plane of geometry ``g``.

    terms: 'psi0' | 'psi1' | 'psi12' | 'tot'
    shadow: None (back-face culling only) or 'raycast'
    source: 'horn' (the dataset's physical source) or 'plane' (unit plane wave
            along the same direction -- the full-wave comparison mode, which
            drops psi0/psi2 because there is no horn to radiate them)
    """
    ds = mesh_ds or g["mesh_ds"]
    X, Y = plane_grid(g, device)
    args = (X, Y, g["h_ms"], g["wvl"], config.THETA_INC, g["size_ant"],
            g["dist_ant"], config.RESOL_ANT)

    if terms == "psi0":
        return (field3d.plane_wave_incident(X, Y, g["h_ms"], g["wvl"], config.THETA_INC)
                if source == "plane" else field3d.horn_to_plane(*args))

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
    want2 = terms in ("psi12", "tot") and source != "plane"
    # a 2-D vertex tensor makes scattered_fields squeeze the batch dim itself,
    # so psi1/psi2 already come back as (nx, ny)
    psi1, psi2 = field3d.scattered_fields(v.to(device), f.to(device), *args,
                                          chunk_faces=chunk, source=source,
                                          compute_psi2=want2, **kw)
    out = psi1 + (psi2 if want2 else 0)
    if terms == "tot":
        out = out + (field3d.plane_wave_incident(X, Y, g["h_ms"], g["wvl"],
                                                 config.THETA_INC)
                     if source == "plane" else field3d.horn_to_plane(*args))
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


def align_external(psi, ref):
    """Put an external solver's field into rail3D's convention, and SAY what
    that took.

    Two mismatches here are pure bookkeeping, and both would otherwise be
    reported as a physics disagreement:

    (a) TIME CONVENTION. rail3D uses exp(-i w t), so outgoing waves carry
        exp(+i k0 R) (see field3d.rs_kernel). HFSS, FEKO and Lumerical use the
        engineering convention exp(+j w t) and therefore exp(-j k0 R): their
        fields arrive CONJUGATED. Conjugating the wrong one turns a perfect
        match into an apparent total failure, so test both and report which won.
    (b) ABSOLUTE AMPLITUDE. rail3D fields are unnormalised scalars; a full-wave
        solver returns V/m for whatever source power it was given. One complex
        gain alpha (least squares over the whole plane) absorbs the scale and
        any constant phase offset.

    Returns (aligned_field, info). The residual after the fit is
    sqrt(1 - complex_corr^2) by construction, so complex_corr is the number to
    quote; the gain and phase are diagnostics, not fitted-away evidence.
    """
    cands = {"as-is": psi, "conjugated": psi.conj()}
    scored = {k: complex_corr(c, ref) for k, c in cands.items()}
    lab = max(scored, key=scored.get)
    cand = cands[lab]
    x, y = cand.reshape(-1), ref.reshape(-1)
    alpha = torch.vdot(x, y) / (torch.vdot(x, x) + 1e-30)
    aligned = alpha * cand
    return aligned, {
        "convention": lab,
        "conjugated": lab == "conjugated",
        "corr_as_is": scored["as-is"],
        "corr_conjugated": scored["conjugated"],
        "ambiguous": abs(scored["as-is"] - scored["conjugated"]) < 0.05,
        "gain": float(alpha.abs()),
        "phase_deg": float(torch.angle(alpha) * 180 / np.pi),
        "rel_l2_after_fit": rel_l2(aligned, ref),
    }


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
# ---------------------------------------------------------------------------
# external-solver geometry export
# ---------------------------------------------------------------------------
def write_stl(path: Path, v: np.ndarray, f: np.ndarray, name: str) -> None:
    """Binary STL. OBJ is not reliably importable into HFSS/FEKO; STL is the
    lowest common denominator every EM package reads. STL carries no units --
    import as MILLIMETRES (case.json says so too)."""
    import io
    import struct
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    n = np.cross(b - a, c - a)
    n = n / np.linalg.norm(n, axis=1, keepdims=True).clip(1e-12)
    rec = np.zeros((len(f), 12), dtype="<f4")
    rec[:, 0:3], rec[:, 3:6], rec[:, 6:9], rec[:, 9:12] = n, a, b, c
    buf = io.BytesIO()
    buf.write(name.encode()[:80].ljust(80, b" "))
    buf.write(struct.pack("<I", len(f)))
    pad = struct.pack("<H", 0)
    for row in rec:
        buf.write(row.tobytes())
        buf.write(pad)
    path.write_bytes(buf.getvalue())


def close_swept_shell(v: np.ndarray, f: np.ndarray, n_arc: int):
    """Cap the open swept shell into a closed PEC body.

    Why it matters: a MoM solver treats an open surface as an infinitely thin
    sheet and puts current on BOTH faces, which is different physics from our
    opaque rail (the PO solver culls back faces). A closed body has zero
    interior field and matches what we model. The rail is a rectangular grid of
    (n_slices x n_arc) vertices, so its single boundary loop splits into three
    trivially-cappable pieces: the two end cross-sections and the underside.
    """
    n_slices = len(v) // n_arc
    if n_slices * n_arc != len(v):
        raise ValueError("vertex count is not a clean sweep grid")
    idx = np.arange(len(v)).reshape(n_slices, n_arc)
    body_c = v.mean(axis=0)
    tris, v_out = [f], [v]
    nv = [len(v)]

    def add_fan(ring, outward):
        ctr = v[ring].mean(axis=0)
        v_out.append(ctr[None, :])
        ci = nv[0]
        nv[0] += 1
        # np.roll closes the fan: the last triangle spans the chord between the
        # ring's two ends, which is exactly the edge the underside strip lands
        # on. An OPEN fan leaves 3 boundary edges per cap and the body is then
        # not watertight -- a MoM solver would still treat it as a sheet.
        t = np.stack([np.full(len(ring), ci), ring, np.roll(ring, -1)], axis=1)
        a = v[t[:, 1]] - ctr
        b = v[t[:, 2]] - ctr
        nrm = np.cross(a, b)
        if nrm.sum(axis=0) @ outward < 0:
            t = t[:, [0, 2, 1]]
            nrm = -nrm
        # star-shapedness: a fan is only valid if every triangle faces outward
        bad = int((nrm @ outward < 0).sum())
        if bad:
            print(f"    ! cap fan: {bad}/{len(t)} triangles face inward -- the "
                  "cross-section is not star-shaped about its centroid; "
                  "check the exported geometry in the solver's viewer")
        tris.append(t)

    add_fan(idx[0], np.array([0.0, -1.0, 0.0]))     # end cap, y = -L/2
    add_fan(idx[-1], np.array([0.0, 1.0, 0.0]))     # end cap, y = +L/2

    # underside: ruled strip between the two long boundary columns
    l, r = idx[:, 0], idx[:, -1]
    strip = np.concatenate([
        np.stack([l[:-1], l[1:], r[1:]], axis=1),
        np.stack([l[:-1], r[1:], r[:-1]], axis=1)])
    a, b, c = v[strip[:, 0]], v[strip[:, 1]], v[strip[:, 2]]
    mid = v[np.concatenate([l, r])].mean(axis=0)
    if np.cross(b - a, c - a).sum(axis=0) @ (mid - body_c) < 0:
        strip = strip[:, [0, 2, 1]]
    tris.append(strip)

    v2 = np.concatenate(v_out)
    f2 = np.concatenate(tris)
    # 2-manifold check, then fix global orientation from the signed volume
    e = np.sort(np.concatenate([f2[:, [0, 1]], f2[:, [1, 2]], f2[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    watertight = bool((cnt == 2).all())
    a, b, c = v2[f2[:, 0]], v2[f2[:, 1]], v2[f2[:, 2]]
    vol = float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
    if vol < 0:
        f2 = f2[:, [0, 2, 1]]
    return v2, f2, watertight, abs(vol)


def cut_end_illumination(v: np.ndarray, f: np.ndarray, g: dict) -> dict:
    """How brightly does the source light the rail's artificial cut ends?

    This decides whether a full-wave comparison is even meaningful. Our PO
    solver has NO edge diffraction; a MoM/FDTD reference has plenty. If the cut
    end is lit, the reference will show diffraction off a truncation that does
    not exist in the real rail, and the disagreement gets misread as "PO fails
    on the defect". Measured directly on the mesh being exported, so it stays
    honest if the geometry or wavelength changes.
    """
    vt = torch.as_tensor(v, dtype=torch.float32)
    ft = torch.as_tensor(f, dtype=torch.long)
    c, area, nrm = field3d.surface_geometry(vt.unsqueeze(0), ft)
    c, area, nrm = c[0], area[0], nrm[0]
    k0 = 2 * np.pi / g["wvl"]
    d = field3d.incident_direction(config.THETA_INC)
    psi0_ant, xa, ya, za, a_ap, b_ap = field3d.aperture_field(
        g["size_ant"], g["dist_ant"], config.RESOL_ANT, config.THETA_INC,
        g["wvl"], k0)
    dS = a_ap * b_ap / (config.RESOL_ANT - 1) ** 2

    idx = torch.nonzero((nrm @ d) > 0).squeeze(1)          # faces the horn sees
    amp = torch.zeros(len(c))
    for s0 in range(0, len(idx), 4000):
        j = idx[s0:s0 + 4000]
        Xo, Yo, Zo = (c[j, i].reshape(-1, 1, 1) for i in range(3))
        R = torch.sqrt((Xo - xa) ** 2 + (Yo - ya) ** 2 + (Zo - za) ** 2)
        cosw = (-(Xo - xa) * np.sin(config.THETA_INC)
                - (Zo - za) * np.cos(config.THETA_INC))
        amp[j] = (psi0_ant * dS * field3d.rs_kernel(R, cosw, g["wvl"], k0)
                  ).sum(dim=(1, 2)).abs()

    y = c[:, 1]
    y_cut = float(y.abs().max())
    w = (amp ** 2 * area)
    # Illuminated power per unit RAIL LENGTH in each band. Normalising by y
    # extent (not by surface area) keeps this independent of whether the body
    # was closed: the end caps carry large area but zero illumination, and
    # would otherwise deflate the edge band by ~4x.
    m_e = y.abs() > y_cut - g["wvl"]        # within 1 lambda of a cut, BOTH ends
    m_m = y.abs() < g["wvl"] / 2            # mid-span
    edge = w[m_e].sum() / (2 * g["wvl"])
    band = w[m_m].sum() / g["wvl"]
    ratio = float(edge / band.clamp_min(1e-30))
    return {"segment_len_mm": round(2 * y_cut, 2),
            "cut_end_vs_midspan_illumination": round(ratio, 4),
            "lit_faces": int(len(idx)),
            "verdict": ("cut ends are BRIGHTLY LIT -- a full-wave reference will "
                        "show edge diffraction off a truncation the real rail "
                        "does not have. Compare the DEFECT DIFFERENCE field "
                        "(cracked minus intact at the same segment length), "
                        "where the common edge contribution cancels."
                        if ratio > 0.2 else
                        "cut ends are weakly lit; absolute-field comparison is "
                        "defensible, but the difference field is still cleaner.")}


def export_case(section, params, g, out: Path, seg, source="horn", closed=False):
    out.mkdir(parents=True, exist_ok=True)
    ds = g["mesh_ds"]
    n_arc = mesh3d.default_arc_count(section, ds)
    kw = {} if seg is None else {"seg_len": seg}
    v, f = mesh3d.sweep_rail_mesh(section, defect_params=params, slice_ds=ds,
                                  arc_ds=ds, n_arc=n_arc, **kw)
    v, f = v.numpy(), f.numpy()
    watertight, vol = False, 0.0
    if closed:
        v, f, watertight, vol = close_swept_shell(v, f, n_arc)

    obj = out / "rail_surface.obj"
    with open(obj, "w") as fh:
        fh.write(f"# rail3D surface, mm, lambda={g['wvl']} mm, mesh {ds} mm\n")
        for pt in v:
            fh.write(f"v {pt[0]:.5f} {pt[1]:.5f} {pt[2]:.5f}\n")
        for t in f:
            fh.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")
    write_stl(out / "rail_surface.stl", v, f, "rail3D railhead (mm)")

    # MoM sizing: RWG unknowns ~ 1.5 x triangles at the CURRENT density, plus
    # what a lambda/10 remesh would cost (the usual MoM rule of thumb).
    a3, b3, c3 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = float(0.5 * np.linalg.norm(np.cross(b3 - a3, c3 - a3), axis=1).sum())
    tri_l10 = area / (np.sqrt(3) / 4 * (g["wvl"] / 10) ** 2)

    trunc = cut_end_illumination(v, f, g)
    freq = 299.792458 / g["wvl"]
    if source == "plane":
        src = {
            "type": "plane wave",
            "amplitude": 1.0,
            "propagation_dir_xyz": [-float(np.sin(config.THETA_INC)), 0.0,
                                    -float(np.cos(config.THETA_INC))],
            "incidence_deg_from_z_in_xz": float(np.degrees(config.THETA_INC)),
            "note": ("arrives along -d, where d points from the rail towards the "
                     "horn position; phase reference is 1+0j at the crown origin"),
        }
    else:
        src = {
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
        }
    spec = {
        "units": "mm",
        "frequency_GHz": freq,
        "wavelength_mm": g["wvl"],
        "geometry_files": {"obj": obj.name, "stl": "rail_surface.stl"},
        "geometry_note": ("swept rail head, crown at z=0; PEC assumed by the PO "
                          "solver (reflection coefficient -1, no receive-cosine)"),
        "mesh": {
            "vertices": int(len(v)), "triangles": int(len(f)),
            "edge_target_mm": ds, "surface_area_mm2": round(area, 1),
            "closed_body": bool(closed), "watertight": watertight,
            "enclosed_volume_mm3": round(vol, 1) if closed else None,
        },
        "solver_sizing": {
            "MoM_RWG_unknowns_this_mesh": int(1.5 * len(f)),
            "MoM_triangles_at_lambda_over_10": int(tri_l10),
            "MoM_RWG_unknowns_at_lambda_over_10": int(1.5 * tri_l10),
            "dense_MoM_matrix_TB": round((1.5 * tri_l10) ** 2 * 16 / 1e12, 2),
            "note": ("use MLFMM: the dense matrix above is out of the question. "
                     "Closing the body (--export-closed) roughly doubles the "
                     "unknowns, because the unlit caps and underside are meshed "
                     "too -- that is the price of modelling an opaque rail "
                     "rather than an infinitely thin sheet."),
        },
        "truncation": trunc,
        "source": src,
        "observation_plane": {
            "z_mm": g["h_ms"], "nx": g["nx"], "ny": g["ny"], "dx_mm": g["dx"],
            "x_extent_mm": [-g["nx"] * g["dx"] / 2, g["nx"] * g["dx"] / 2],
            "y_extent_mm": [-g["ny"] * g["dx"] / 2, g["ny"] * g["dx"] / 2],
            "sampling": "cell-centred",
            "note": ("cell CENTRES: x_i = -Wx/2 + (i+0.5)*dx. Sample the solver "
                     "on exactly these points, not on a node-centred grid."),
        },
        "conventions": {
            "rail3D_time_convention": "exp(-i w t); outgoing waves carry exp(+i k0 R)",
            "engineering_convention": "exp(+j w t); outgoing waves carry exp(-j k0 R)",
            "consequence": ("HFSS / FEKO / Lumerical fields arrive CONJUGATED "
                            "relative to rail3D. compare_wavefronts.py --external "
                            "detects this and reports which convention matched; "
                            "it does not silently fix it."),
            "amplitude": ("rail3D fields are unnormalised scalars. The comparison "
                          "fits one complex gain alpha over the whole plane before "
                          "differencing, so absolute units do not matter."),
            "polarisation": ("rail3D is SCALAR. Export ONE component from the "
                             "vector solver -- E_y (E along the rail axis, TE) is "
                             "the cleanest match; set the incident polarisation "
                             "to match it."),
        },
        "compare_back": {
            "save_as": "npz with keys: field (complex64 (nx,ny)), label (str)",
            "then_run": "python compare_wavefronts.py --external <file>.npz",
        },
    }
    (out / "case.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")

    body = (
        "# rail3D full-wave comparison case\n\n"
        "Everything needed to reproduce this exact scattering problem in an\n"
        f"external solver at {freq:.1f} GHz. Source: **{src['type']}**; geometry is\n"
        f"{'a CLOSED body' if closed else 'an OPEN shell'} "
        f"({len(f)} triangles, {area:.0f} mm^2).\n\n"
        f"**Cut-end illumination: {trunc['cut_end_vs_midspan_illumination']:.2f}x "
        f"mid-span** over a {trunc['segment_len_mm']:.0f} mm segment.\n"
        f"{trunc['verdict']}\n\n"
        "- `rail_surface.stl` - import this (binary STL, **units = mm**).\n"
        "- `rail_surface.obj` - same mesh, for viewers that prefer OBJ.\n"
        "- `case.json` - source, incidence, observation plane, conventions.\n\n"
        "## Which solver\n\n"
        "This is a PEC surface in an open region: an **integral-equation (MoM)**\n"
        "problem. Ansys **HFSS-IE** (needs the IE solver licensed, not just the\n"
        "FEM seat) or **Altair FEKO**, both with **MLFMM** enabled.\n\n"
        f"At lambda/10 this surface is ~{tri_l10 / 1e3:.0f}k triangles, about "
        f"{1.5 * tri_l10 / 1e3:.0f}k RWG\nunknowns - routine for MLFMM.\n\n"
        "- **Zemax OpticStudio is not a valid reference.** Its POP is a scalar\n"
        "  Fresnel/Kirchhoff propagator, the same approximation family as this\n"
        "  code, so agreement would prove nothing.\n"
        "- **HFSS SBR+** is shooting-bounce-ray PO with PTD edge corrections: an\n"
        "  upgrade on our model, not an independent check. A useful third point,\n"
        "  not the validator.\n"
        "- **Lumerical FDTD** is valid physics but volumetric. Do NOT box the\n"
        "  whole scene out to z=150 mm (~390 Mcells at lambda/20, ~39 GB, which\n"
        "  does not fit a 32 GB GPU). Box the rail only, record a near-field\n"
        "  monitor, and project analytically to the plane.\n\n"
        "## Returning the result\n\n"
        "Sample the scattered field on the cell-CENTRED plane grid defined in\n"
        "`case.json` (x_i = -Wx/2 + (i+0.5)*dx), then:\n\n"
        "```python\n"
        "import numpy as np\n"
        "np.savez('mom_result.npz', field=E.astype(np.complex64), label='HFSS-IE')\n"
        "```\n\n"
        "```\n"
        "python compare_wavefronts.py --external mom_result.npz\n"
        "```\n\n"
        "It joins every figure and metric. Conjugation and complex gain are\n"
        "detected and reported, so a convention mismatch cannot be mistaken for a\n"
        "physics disagreement.\n\n"
        "## Suggested ladder - do not start at the bottom\n\n"
        "1. **Flat PEC plate**, plane wave, same incidence. If this disagrees,\n"
        "   the setup is wrong, not the physics.\n"
        "2. **Intact rail**, plane wave. Tests PO currents on a curved surface.\n"
        "3. **Cracked rail**, plane wave. This is the real question.\n"
        "4. **Real horn**, only after 1-3 agree.\n\n"
        "## Expect PO and full-wave to differ where PO is known to be weak\n\n"
        "- features at or below the wavelength (crack widths are 2-5 mm =\n"
        "  0.4-1 lambda at 60 GHz - this is the point of the exercise)\n"
        "- grazing/terminator faces (this PO formulation applies no receive-cosine)\n"
        "- edge diffraction from the rail's cut ends\n"
        "- multiple scattering beyond the second bounce\n"
        "- polarisation coupling, which a scalar model cannot represent at all\n\n"
        "Those are the interesting comparisons, not failures to hide.\n"
    )
    (out / "README.md").write_text(body, encoding="utf-8")
    extra = f", watertight={watertight}" if closed else ""
    print(f"  wrote case -> {out}  ({len(v)} verts, {len(f)} tris, "
          f"source={source}, closed={closed}{extra})")
    r = trunc["cut_end_vs_midspan_illumination"]
    print(f"    cut-end illumination {r:.2f}x mid-span over "
          f"{trunc['segment_len_mm']:.0f} mm"
          + ("  <-- compare the DEFECT DIFFERENCE field, not absolute fields"
             if r > 0.2 else ""))


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
    ap.add_argument("--source", default="horn", choices=("horn", "plane"),
                    help="illumination for the exported case AND the reference "
                         "variant: 'horn' (what the dataset uses) or 'plane' "
                         "(unit plane wave -- removes the horn aperture model as "
                         "a confound, so use it for full-wave comparisons)")
    ap.add_argument("--export-only", action="store_true",
                    help="write the case bundle and stop -- exporting is instant, "
                         "solving every variant is minutes")
    ap.add_argument("--export-closed", action="store_true",
                    help="cap the swept shell into a closed body before export. "
                         "MoM puts current on BOTH sides of an open sheet, which "
                         "is not what an opaque rail does -- use this for HFSS-IE "
                         "or FEKO")
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
        export_case(section, params, g5, Path(args.export_case), args.seg,
                    source=args.source, closed=args.export_closed)
        if args.export_only:
            return 0

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
        # plane-wave illumination: no horn, so no psi0 and no psi2. These are
        # the variants a full-wave reference should be compared against.
        "lam5 pw psi1 (no shadow)": lambda: solve(g=g5, source="plane", **common),
        "lam5 pw psi1 (raycast 3.0)": lambda: solve(
            g=g5, source="plane", shadow="raycast", min_t=3.0, **common),
    }
    PLANE = [k for k in ALL if k.startswith("lam5 pw")]
    if args.variants:
        names = args.variants
    elif args.source == "plane":
        # the horn no-shadow field rides along as context, not as the reference
        names = PLANE + ["lam5 psi1 (no shadow)"]
    else:
        names = [k for k in ALL if k not in PLANE]

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
        fields[f"EXT {lab}"] = {"psi": psi, "geom": g5, "external": True}
        print(f"  {'EXT ' + lab:32s} {tuple(psi.shape)}  (external reference)")

    if not fields:
        print("no variants solved")
        return 1

    pref = ("lam5 pw psi1 (no shadow)" if args.source == "plane"
            else "lam5 psi1 (no shadow)")
    ref_key = pref if pref in fields else list(fields)[0]
    ref = fields[ref_key]["psi"]

    # Align external fields BEFORE any metric or figure, so what the plots show
    # is a physics difference rather than a time-convention or units difference.
    for k, d in fields.items():
        if not d.get("external"):
            continue
        if d["psi"].shape != ref.shape:
            print(f"  ! {k}: shape {tuple(d['psi'].shape)} != reference "
                  f"{tuple(ref.shape)} -- not aligned or scored. Resample onto "
                  f"the cell-centred grid in case.json.")
            continue
        d["psi"], info = align_external(d["psi"], ref)
        d["align"] = info
        print(f"  aligned {k}: {info['convention']} "
              f"(corr as-is {info['corr_as_is']:.4f} / conj "
              f"{info['corr_conjugated']:.4f}), gain {info['gain']:.4g}, "
              f"phase {info['phase_deg']:+.1f} deg")
        if info["ambiguous"]:
            print("    ! both conventions score alike -- the fields may be "
                  "uncorrelated; do not read the alignment as confirmation.")

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
        if "align" in d:
            row["alignment"] = d["align"]
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
