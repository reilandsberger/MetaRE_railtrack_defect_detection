"""Lumerical FDTD vs rail3D: how well do they agree, pixel by pixel?

    python fdtd_agreement.py --horn horn_aperture.mat --horn-near horn_near.mat  # rung -1
    python fdtd_agreement.py --injection empty_z0.mat --leak empty_z30.mat   # rung 0
    python fdtd_agreement.py --sample plate --external plate_z30.mat        # rung 1
    python fdtd_agreement.py --external intact_z30.mat      # a REAL Lumerical run
    python fdtd_agreement.py --target                       # the TARGET figure

Both modes run the SAME metrics and draw the SAME figure, so the target is a
literal preview of what a passing Lumerical run will produce. Only the source of
the "FDTD" field differs:

  --external  E_y from the Lumerical monitor (LUMERICAL.md section 5), resampled
              onto our grid, time convention and complex gain aligned by
              compare_wavefronts.align_external (it SAYS which it applied)
  --target    rail3D's own field passed through a STATED perturbation model at
              the level a passing run should show. Every such figure is titled
              TARGET and footnoted as synthesized. It is a specification, not a
              result. Do not strip that label when it goes into a deck.

WHERE the comparison happens (LUMERICAL.md section 1). FDTD accumulates
numerical phase error with distance (~125 deg to H_MS at lambda/10), so FDTD is
recorded at a z = 30 mm near-field monitor and never propagated by FDTD itself.
Two rows:

  z = 30 mm   the monitor window itself, where FDTD is recorded. Primary.
  MS plane    both fields carried to H_MS by the IDENTICAL angular-spectrum
              propagator (V3: 0.13%) from the same window, then cropped to the
              metasurface aperture. Identical propagation means any disagreement
              here was already in the near field, mapped to where the metasurface
              sees it. The window's own truncation limit is measured every run
              (the "window ceiling"): rail3D's ASM-carried field vs rail3D solved
              directly at H_MS.

WHAT is compared (--source horn, the default and the setup LUMERICAL.md builds):
the field the rail sends UP under rail3D's own horn, injected into FDTD as an
Import source at z = 15 mm (horn_source.py). The z = 30 monitor sits above that
plane, so it records only the up-going field -- reflected + scattered, every
bounce -- and rail3D's side is psi1 + psi2 under the horn. The horn's DIRECT
field to the metasurface (psi0) is analytic and identical in both by
construction, so including it would only inflate agreement.
(--source plane keeps the older plane-wave comparison: psi1 under a unit
s-polarised plane wave, for a plane-wave source in the external solver.)

RUNG 0 (--injection): before any rail, the empty box shows whether Lumerical
injects what rail3D thinks the horn delivers. The field recorded at z = 0 is
scored against rail3D's horn field there, over the rail footprint only (where
the windowed source reproduces the full horn to 0.999), and --leak measures how
much up-going field the z = 30 monitor sees with nothing to reflect it -- the
noise floor every later rung sits on.

Metrics (per row; thresholds in CRITERIA, one block, all proposed).

GATES DIFFER BY ROW, and that was measured, not assumed. Sweeping the synthetic
disagreement level (2026-09-18, intact rail): at 99.4% MS-plane correlation only
73% of lit MS pixels sit inside 5% / 15 deg, while the 130-detector barcode stays
at 0.997. The MS field is speckle, so per-pixel error there is harsh and says
little about what the detectors see -- README finding 2 already records that
raw complex error never converges on speckle and fidelity is judged at the
barcode. So: the z = 30 row (specular-dominated, primary) is gated per pixel;
the MS row is gated on complex_corr and barcode_cos, and its per-pixel numbers
are REPORTED, not gated.

  complex_corr        |<a,b>| / (|a||b|) after alignment. THE number to quote --
                      the post-fit residual is sqrt(1 - corr^2) by construction.
                      Reported as "% agreement". Gated at 0.98 -- tighter than
                      LUMERICAL.md's 0.95, which only says the SETUP works.
  amplitude_corr      Pearson on |psi|
  in_tolerance        % of LIT pixels (|psi| >= 25% of peak) with |d|psi|| <= 5% of
                      peak AND |d phase| <= 15 deg (~lambda/24). Phase is undefined
                      where there is no light, so unlit pixels are not scored.
  max amp residual    largest |d|psi||/peak over the plane, WITH its (x, y)
  max phase residual  largest |d phase| over lit pixels, WITH its (x, y)
  rms phase (|psi|^2 weighted)  phase error where the energy actually is
  barcode cosine      MS plane only: dense 130-detector powers, the level README
                      finding 2 says fidelity should be judged at (raw complex
                      L2 never converges; speckle)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import compare_wavefronts as cw           # noqa: E402  (sets the Agg backend)
import matplotlib.pyplot as plt           # noqa: E402
import numpy as np                        # noqa: E402
import torch                              # noqa: E402

from rail3d import config, data3d, mesh3d, optics3d, sections   # noqa: E402

Z_MON = 30.0                                  # near-field monitor (LUMERICAL.md)
CRITERIA = {                                  # proposed pass criteria -- one place
    # 0.95 in LUMERICAL.md is the "setup is not broken" gate for rung 1; an
    # agreement TARGET has to be tighter than that.
    "complex_corr": 0.98,                     # both rows
    "barcode_cos": 0.98,                      # MS row, dense 130 detectors
    "in_tolerance": 0.85,                     # z = 30 row: fraction of lit pixels
    "max_amp_resid": 0.15,                    # z = 30 row: worst pixel, fraction of peak
    "max_phase_resid_deg": 30.0,              # z = 30 row: worst lit pixel
    "amp_tol": 0.05,                          # per-pixel: 5% of peak |psi|
    "phase_tol_deg": 15.0,                    # per-pixel: ~lambda/24 of path
    # phase is only meaningful where there is light; 10% of peak admitted
    # near-nulls of the speckle, where any phase is noise
    "lit_frac": 0.25,
}
GATED = {"z30": ("complex_corr", "in_tolerance", "max_amp_resid", "max_phase_resid_deg"),
         "ms": ("complex_corr", "barcode_cos")}
# The perturbation that turns rail3D's field into a TARGET "FDTD" field: the
# level of disagreement a passing run should show, not a prediction of it.
# Level chosen so the example passes every gate with visible, non-trivial
# residuals: the 2%/3 deg level read 99.85% and looked like a promise; 4%/6 deg
# fails the z = 30 in-tolerance gate. Sweep: scratch sweep_target.py, 2026-09-18.
TARGET_MODEL = {
    "amp_rms": 0.03,          # smooth multiplicative amplitude error, corr 15 mm
    "phase_rms_deg": 4.0,     # smooth phase error, corr 25 mm
    "phase_ramp_deg": 5.0,    # residual dispersion ramp across the window in x
    "additive_rms": 0.025,    # PO-missing diffraction, fraction of peak, corr 10 mm
    "gain": 1.2e-3,           # source normalisation (V/m vs unitless)
    "gain_phase_deg": -14.0,  # constant phase offset
    # Lumerical uses exp(-i w t) like rail3D ("P(w) = int e^{i w t} P(t) dt",
    # Ansys Optics KB), so a real export aligns AS-IS. An earlier version set
    # this True on the belief that Lumerical used exp(+j w t); it does not.
    "conjugate": False,
    "seed": 0,
}


def window_geom(plan_mon: dict) -> dict:
    """Our cell-centred grid over the FDTD monitor window at z = Z_MON."""
    nx = int(round((plan_mon["x"][1] - plan_mon["x"][0]) / config.DX))
    ny = int(round((plan_mon["y"][1] - plan_mon["y"][0]) / config.DX))
    return dict(cw.geom_now(), h_ms=Z_MON, nx=nx - nx % 2, ny=ny - ny % 2)


def sample_params(sample: str, section):
    if sample == "plate":
        return {"class": "plate", "half": 7.5 * config.WVL}
    if sample == "intact":
        return {"class": "intact"}
    defect = None
    if sample in config.DATASET_DIRS:
        files = sections.get_dataset_files(sample)
        defect = sections.match_reference_width(
            sections.load_vertices_from_csv(files[0]), section)
    params, _ = mesh3d.defect_params_for_sample(
        section, sample, defect, config.sample_seed(sample, 0),
        n_arc=mesh3d.default_arc_count(section, config.MESH_DS))
    return params


def to_ms_plane(psi_w: torch.Tensor, gw: dict) -> torch.Tensor:
    """Carry a z = Z_MON window to H_MS, crop to the metasurface aperture."""
    prop = optics3d.PropagatorASM2D(config.H_MS - Z_MON, nx=gw["nx"], ny=gw["ny"])
    up = prop.to(psi_w.device)(psi_w)[0]
    ox, oy = (gw["nx"] - config.NX) // 2, (gw["ny"] - config.NY) // 2
    return up[ox: ox + config.NX, oy: oy + config.NY]


def smooth_field(shape, corr_mm: float, gen: torch.Generator) -> torch.Tensor:
    """Unit-RMS Gaussian-correlated random field on the grid."""
    n = torch.randn(shape, generator=gen, dtype=torch.float64)
    kx = torch.fft.fftfreq(shape[0], d=config.DX)
    ky = torch.fft.fftfreq(shape[1], d=config.DX)
    KX, KY = torch.meshgrid(kx, ky, indexing="ij")
    filt = torch.exp(-2 * (np.pi * corr_mm) ** 2 * (KX ** 2 + KY ** 2))
    out = torch.fft.ifft2(torch.fft.fft2(n) * filt).real
    return (out / out.pow(2).mean().sqrt()).float()


def synthesize_target(ref_w: torch.Tensor, gw: dict, m: dict = TARGET_MODEL) -> torch.Tensor:
    """rail3D's field + the stated disagreement + a real export's bookkeeping."""
    gen = torch.Generator().manual_seed(m["seed"])
    shp = tuple(ref_w.shape)
    X, _ = cw.plane_grid(gw, torch.device("cpu"))
    x = X[0] / (gw["nx"] * gw["dx"] / 2)                     # -1 .. 1 across the window
    amp = 1 + m["amp_rms"] * smooth_field(shp, 15.0, gen)
    ph = np.radians(m["phase_rms_deg"]) * smooth_field(shp, 25.0, gen) \
        + np.radians(m["phase_ramp_deg"]) * x
    peak = float(ref_w.abs().max())
    add = m["additive_rms"] * peak * (smooth_field(shp, 10.0, gen)
                                      + 1j * smooth_field(shp, 10.0, gen)) / np.sqrt(2)
    r = ref_w.cpu()
    psi = r * amp * torch.exp(1j * ph) + add.to(torch.complex64)
    if m["conjugate"]:
        psi = psi.conj()
    return (psi * m["gain"] * np.exp(1j * np.radians(m["gain_phase_deg"]))).to(ref_w.device)


def score(ref: torch.Tensor, ext: torch.Tensor, g: dict, device, ms_plane: bool) -> dict:
    X, Y = cw.plane_grid(g, torch.device("cpu"))
    X, Y = X[0].numpy(), Y[0].numpy()
    a, b = ref.detach().cpu(), ext.detach().cpu()
    peak = float(a.abs().max())
    lit = (a.abs() >= CRITERIA["lit_frac"] * peak).numpy()
    d_amp = ((b.abs() - a.abs()) / peak).numpy()
    d_ph = np.degrees(np.angle((b * a.conj()).numpy()))           # wrapped difference
    d_ph_lit = np.where(lit, d_ph, np.nan)
    ia = np.unravel_index(np.nanargmax(np.abs(d_amp)), d_amp.shape)
    ip = np.unravel_index(np.nanargmax(np.abs(d_ph_lit)), d_ph.shape)
    w = (a.abs() ** 2).numpy() * lit
    in_tol = lit & (np.abs(d_amp) <= CRITERIA["amp_tol"]) \
        & (np.abs(d_ph) <= CRITERIA["phase_tol_deg"])
    out = {
        "complex_corr": cw.complex_corr(b, a),
        "amplitude_corr": cw.amp_corr(b, a),
        "rel_l2_after_fit": cw.rel_l2(b, a),
        "lit_pixels": int(lit.sum()), "pixels": int(lit.size),
        "in_tolerance": float(in_tol.sum() / max(lit.sum(), 1)),
        "max_amp_resid": float(abs(d_amp[ia])),
        "max_amp_resid_xy_mm": [round(float(X[ia]), 2), round(float(Y[ia]), 2)],
        "max_phase_resid_deg": float(abs(d_ph[ip])),
        "max_phase_resid_xy_mm": [round(float(X[ip]), 2), round(float(Y[ip]), 2)],
        "rms_phase_resid_deg_weighted": float(np.sqrt((w * d_ph ** 2).sum() / w.sum())),
        "_maps": {"d_amp": d_amp, "d_ph": d_ph_lit, "ia": ia, "ip": ip, "X": X, "Y": Y},
    }
    if ms_plane:
        out["barcode_cos"] = cw.cos_sim(cw.barcode(a.to(device), g, device).cpu(),
                                        cw.barcode(b.to(device), g, device).cpu())
    checks = {
        "complex_corr": out["complex_corr"] >= CRITERIA["complex_corr"],
        "in_tolerance": out["in_tolerance"] >= CRITERIA["in_tolerance"],
        "max_amp_resid": out["max_amp_resid"] <= CRITERIA["max_amp_resid"],
        "max_phase_resid_deg": out["max_phase_resid_deg"] <= CRITERIA["max_phase_resid_deg"],
    }
    if ms_plane:
        checks["barcode_cos"] = out["barcode_cos"] >= CRITERIA["barcode_cos"]
    out["pass"] = {k: checks[k] for k in GATED["ms" if ms_plane else "z30"]}
    out["reported_not_gated"] = sorted(set(checks) - set(out["pass"]))
    return out


def draw(rows, target: bool, meta: dict, path: Path) -> None:
    fig = plt.figure(figsize=(27, 8.4), layout="constrained")
    subs = fig.subfigures(len(rows), 1)
    for sub, (label, ref, ext, m, g) in zip(np.atleast_1d(subs), rows):
        axs = sub.subplots(1, 7, gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 1, 1.55]})
        X, Y = m["_maps"]["X"], m["_maps"]["Y"]
        a, b = ref.detach().cpu(), ext.detach().cpu()
        vmax = float(a.abs().max())
        src = "FDTD target" if target else "Lumerical FDTD"
        panels = [
            (a.abs().numpy(), "viridis", (0, vmax), "|psi| rail3D"),
            (b.abs().numpy(), "viridis", (0, vmax), f"|psi| {src} (aligned)"),
            (m["_maps"]["d_amp"], "RdBu_r", (-0.15, 0.15), "d|psi| / peak"),
            (np.angle(a.numpy()), "twilight", (-np.pi, np.pi), "arg psi rail3D"),
            (np.angle(b.numpy()), "twilight", (-np.pi, np.pi), f"arg psi {src}"),
            (m["_maps"]["d_ph"], "RdBu_r", (-45, 45), "d phase (deg), lit pixels"),
        ]
        for k, (arr, cmap, (lo, hi), title) in enumerate(panels):
            ax = axs[k]
            cm = plt.get_cmap(cmap).copy()
            cm.set_bad("#bdbdbd")
            im = ax.pcolormesh(X, Y, arr, cmap=cm, vmin=lo, vmax=hi, shading="nearest")
            fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
            ax.set_title(title, fontsize=10)
            ax.set_aspect("equal")
            ax.set_xlabel("x (mm)")
            if k == 0:
                ax.set_ylabel("y (mm)")
            if k in (2, 5):
                i = m["_maps"]["ia" if k == 2 else "ip"]
                ax.plot(X[i], Y[i], marker="o", ms=11, mfc="none", mec="k", mew=2)
                ax.plot(X[i], Y[i], marker="o", ms=11, mfc="none", mec="yellow", mew=1)
        # metric table
        ax = axs[6]
        ax.axis("off")
        ok = lambda k: ("PASS" if m["pass"][k] else "FAIL") if k in m["pass"] \
            else "report"                                              # noqa: E731
        xy = lambda k: (f"({m[k][0]:+.1f}, {m[k][1]:+.1f})")          # noqa: E731
        # (label, value, gate key or None, criterion text) -- fixed columns, so
        # a long value can never run into its verdict
        lines = [
            ("agreement (cplx corr)", f"{100*m['complex_corr']:.2f}%", "complex_corr",
             f">= {100*CRITERIA['complex_corr']:.0f}%"),
            ("pixels in tolerance", f"{100*m['in_tolerance']:.1f}%", "in_tolerance",
             f">= {100*CRITERIA['in_tolerance']:.0f}% of lit"),
            ("max |dA| / peak", f"{100*m['max_amp_resid']:.1f}%", "max_amp_resid",
             f"<= {100*CRITERIA['max_amp_resid']:.0f}%"),
            ("   at (x, y) mm", xy("max_amp_resid_xy_mm"), None, ""),
            ("max |dphi| (lit)", f"{m['max_phase_resid_deg']:.1f} deg",
             "max_phase_resid_deg", f"<= {CRITERIA['max_phase_resid_deg']:.0f} deg"),
            ("   at (x, y) mm", xy("max_phase_resid_xy_mm"), None, ""),
            ("rms dphi, |psi|^2 wtd", f"{m['rms_phase_resid_deg_weighted']:.1f} deg",
             None, ""),
            ("amplitude corr", f"{m['amplitude_corr']:.4f}", None, ""),
        ]
        if "barcode_cos" in m:
            lines.append(("barcode cos (130 det)", f"{m['barcode_cos']:.4f}",
                          "barcode_cos", f">= {CRITERIA['barcode_cos']}"))
        for j, (lab_, val, key, crit) in enumerate(lines):
            yy = 0.97 - j * 0.092
            ax.text(0.00, yy, lab_, fontsize=9, va="top", transform=ax.transAxes)
            ax.text(0.50, yy, val, fontsize=9, family="monospace", va="top",
                    transform=ax.transAxes)
            if key is None:
                continue
            verdict = ok(key)
            col = {"PASS": "#1a7f37", "FAIL": "#c62828"}.get(verdict, "#777")
            ax.text(0.80, yy, verdict, fontsize=9, fontweight="bold", va="top",
                    color=col, transform=ax.transAxes)
            ax.text(0.80, yy - 0.040, crit if verdict != "report" else "not gated",
                    fontsize=7, va="top", color="#555", transform=ax.transAxes)
        note = (f"lit = |psi| >= {100*CRITERIA['lit_frac']:.0f}% of peak "
                f"({m['lit_pixels']}/{m['pixels']} px)\nin tolerance = "
                f"|dA| <= {100*CRITERIA['amp_tol']:.0f}% of peak & |dphi| <= "
                f"{CRITERIA['phase_tol_deg']:g} deg")
        if m["reported_not_gated"]:
            note += "\nper-pixel stats not gated here: speckle (README finding 2)"
        ax.text(0.0, 0.97 - len(lines) * 0.092 - 0.01, note, fontsize=7.5,
                color="#555", va="top", transform=ax.transAxes)
        sub.suptitle(label.split("\n")[0], fontsize=12, x=0.01, ha="left")

    illum = ("horn (Import source at z = 15 mm), up-going field psi1+psi2"
             if meta.get("source") == "horn" else "s-pol plane wave at 55 deg, psi1")
    head = ("TARGET: " if target else "") + (
        f"Lumerical FDTD vs rail3D -- {meta['sample']} rail, {illum}, {config.WVL:g} mm")
    fig.suptitle(head, fontsize=14, fontweight="bold",
                 color="#b00020" if target else "black")
    foot = (f"alignment: {meta['align']['convention']} (corr as-is "
            f"{meta['align']['corr_as_is']:.3f} / conj {meta['align']['corr_conjugated']:.3f}), "
            f"fitted gain {meta['align']['gain']:.3g} at {meta['align']['phase_deg']:+.1f} deg.   "
            f"MS-plane row: both fields ASM-propagated {config.H_MS - Z_MON:g} mm from the same "
            f"z={Z_MON:g} window ({meta['window']}); that window reproduces rail3D's directly-"
            f"solved MS field at complex corr {meta['window_ceiling']:.3f} (the ceiling for "
            f"this row).")
    if target:
        tm = TARGET_MODEL
        foot = ("TARGET SPECIFICATION -- the FDTD field is SYNTHESIZED from rail3D plus a stated "
                f"perturbation (amp {100*tm['amp_rms']:.0f}% rms, phase {tm['phase_rms_deg']:g} deg "
                f"rms + {tm['phase_ramp_deg']:g} deg ramp, additive {100*tm['additive_rms']:.1f}% "
                f"of peak, conjugated, gain {tm['gain']:g} at {tm['gain_phase_deg']:+g} deg). "
                "NOT an FDTD result.\n" + foot)
    foot += (f"\nGenerated {datetime.now():%Y-%m-%d} at {meta['commit']} by "
             f"rail3D/fdtd_agreement.py")
    fig.text(0.5, -0.005, foot, ha="center", va="top", fontsize=8.5,
             color="#b00020" if target else "#333", wrap=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def rung0(args, device) -> int:
    """Empty box: does the Import source deliver rail3D's horn field at z = 0?"""
    import horn_source
    g0 = dict(cw.geom_now(), h_ms=0.0, nx=48, ny=56)            # x +/-60, y +/-70
    arr, lab, info = cw.load_external(args.injection, g0)
    X, Y = cw.plane_grid(g0, torch.device("cpu"))
    X, Y = X[0].numpy(), Y[0].numpy()
    ref = horn_source.horn_field(X[:, 0], Y[0, :], 0.0, device)
    foot = (np.abs(X) <= 40) & (np.abs(Y) <= 60)
    a = torch.as_tensor(ref[foot])
    b = torch.as_tensor(np.asarray(arr)[foot])
    aligned, al = cw.align_external(b, a)
    corr = cw.complex_corr(aligned, a)
    # centroids over the FOOTPRINT: outside it the windowed source is meant to
    # differ from the full horn, which shifted a whole-monitor centroid ~3 mm
    I_ext = np.abs(arr) ** 2 * foot
    I_ref = np.abs(ref) ** 2 * foot
    cx_ext = float((I_ext * X).sum() / I_ext.sum())
    cx_ref = float((I_ref * X).sum() / I_ref.sum())
    print(f"RUNG 0 injection ({lab}): complex corr over the rail footprint {corr:.4f}"
          f"  [{al['convention']}: as-is {al['corr_as_is']:.3f} / conj "
          f"{al['corr_conjugated']:.3f}]")
    print(f"  beam centroid x at z=0: FDTD {cx_ext:+.1f} mm, rail3D {cx_ref:+.1f} mm")
    ok = (corr >= 0.98 and (al["convention"] == "as-is" or al["ambiguous"])
          and abs(cx_ext - cx_ref) < 5)
    if al["convention"] != "as-is" and not al["ambiguous"]:
        print("  ! matched only when CONJUGATED. Lumerical is exp(-i w t) like rail3D, "
              "so check the source Direction (must be Backward) before anything else.")
    if abs(cx_ext - cx_ref) >= 5:
        print("  ! the beam lands in the wrong place: a mirrored centroid means the "
              "profile was injected the wrong way (Direction, or the E/H pair).")
    out = {"injection_corr_footprint": corr, "alignment": al,
           "centroid_x_mm": {"fdtd": cx_ext, "rail3d": cx_ref}, "pass": bool(ok)}
    if args.leak:
        g30 = dict(cw.geom_now(), h_ms=Z_MON, nx=64, ny=56)
        leak, _, _ = cw.load_external(args.leak, g30)
        ratio = float(np.abs(leak).max() / np.abs(arr).max())
        out["leak_peak_over_injected_peak"] = ratio
        print(f"  leakage at the z=30 monitor (nothing to reflect): {ratio:.2e} of the "
              f"injected peak -- every later rung's floor")
        ok = ok and ratio < 1e-2
        out["pass"] = bool(ok)
    js = config.GENERATED_DIR / "fdtd_rung0.json"
    js.write_text(json.dumps(data3d.stamp(out), indent=2, default=str), encoding="utf-8")
    print(f"  -> {'PASS' if ok else 'FAIL'}   ({js})")
    return 0 if ok else 1


HORN_CRITERIA = {                  # rung -1, proposed -- revisit after the first real run
    "aperture_corr": 0.95,         # the aperture method ignores edge currents and
                                   # higher-order modes (Nikolova L18 p.15): not ~1
    "near_corr": 0.98,             # 3 lambda out, which is what reaches the rail
    "hpbw_deg": 1.5,               # |FDTD - model| in each principal plane
    "directivity_dB": 0.5,         # |FDTD - model|
}


def horn_check(args, device) -> int:
    """RUNG -1: the full-wave horn (horn_fdtd_case.py) vs rail3D's aperture model."""
    import horn_fdtd_case as hf
    lam = config.WVL
    p = hf.plan()
    out, rows = {"part": config.HORN_SPEC["part"], "criteria": HORN_CRITERIA}, []

    def load(path, mon, dx):
        m = p["monitors"][mon]
        nx = int((m["x"][1] - m["x"][0]) / dx) // 2 * 2
        ny = int((m["y"][1] - m["y"][0]) / dx) // 2 * 2
        g = dict(cw.geom_now(), nx=nx, ny=ny, dx=dx, h_ms=m["z"])
        arr, lab, info = cw.load_external(path, g)
        X, Y = cw.plane_grid(g, torch.device("cpu"))
        return arr, X[0, :, 0].numpy().astype(float), Y[0, 0, :].numpy().astype(float), info

    # --- aperture plane: the aperture method's assumption, tested directly ---
    ext, xs, ys, info = load(args.horn, "mon_aperture", lam / 20)
    mod = hf.model_aperture(xs, ys)
    al_ext, al = cw.align_external(torch.as_tensor(ext), torch.as_tensor(mod))
    al_ext = al_ext.numpy()
    ff_m, ff_x = hf.far_field_cuts(mod, xs, ys), hf.far_field_cuts(al_ext, xs, ys)
    ap_row = {
        "complex_corr": cw.complex_corr(torch.as_tensor(al_ext), torch.as_tensor(mod)),
        "amplitude_corr": cw.amp_corr(torch.as_tensor(al_ext), torch.as_tensor(mod)),
        "alignment": al, "coverage": info.get("plane_coverage", 1.0),
        "model": {k: ff_m[k] for k in ("hpbw_E_deg", "hpbw_H_deg", "directivity_dBi")},
        "fdtd": {k: ff_x[k] for k in ("hpbw_E_deg", "hpbw_H_deg", "directivity_dBi")},
    }
    lo, hi = config.HORN_SPEC["gain_dBi"]
    ap_row["pass"] = {
        "aperture_corr": ap_row["complex_corr"] >= HORN_CRITERIA["aperture_corr"],
        "hpbw_E": abs(ff_x["hpbw_E_deg"] - ff_m["hpbw_E_deg"]) <= HORN_CRITERIA["hpbw_deg"],
        "hpbw_H": abs(ff_x["hpbw_H_deg"] - ff_m["hpbw_H_deg"]) <= HORN_CRITERIA["hpbw_deg"],
        "directivity": abs(ff_x["directivity_dBi"] - ff_m["directivity_dBi"]) <= HORN_CRITERIA["directivity_dB"],
        "fdtd_gain_in_part_spec": lo - 0.3 <= ff_x["directivity_dBi"] <= hi + 0.3,
    }
    out["aperture"] = ap_row
    rows.append(("aperture plane (z' = +0.5 mm)", xs, ys, mod, al_ext))
    print(f"RUNG -1 aperture ({al['convention']}): complex corr {ap_row['complex_corr']:.4f}, "
          f"amplitude corr {ap_row['amplitude_corr']:.4f}")
    print(f"  model: D {ff_m['directivity_dBi']:.2f} dBi, HPBW E {ff_m['hpbw_E_deg']:.1f} / H "
          f"{ff_m['hpbw_H_deg']:.1f} deg   FDTD: D {ff_x['directivity_dBi']:.2f} dBi, HPBW E "
          f"{ff_x['hpbw_E_deg']:.1f} / H {ff_x['hpbw_H_deg']:.1f} deg   "
          f"(part spec {lo:g}-{hi:g} dBi)")

    # --- 3 lambda out: what actually travels towards the rail ----------------
    if args.horn_near:
        ext_n, xn, yn, info_n = load(args.horn_near, "mon_near", lam / 10)
        mod_n = hf.model_plane(xn, yn, p["monitors"]["mon_near"]["z"])
        al_n, aln = cw.align_external(torch.as_tensor(ext_n), torch.as_tensor(mod_n))
        corr_n = cw.complex_corr(al_n, torch.as_tensor(mod_n))
        out["near"] = {"complex_corr": corr_n, "alignment": aln,
                       "coverage": info_n.get("plane_coverage", 1.0),
                       "pass": {"near_corr": corr_n >= HORN_CRITERIA["near_corr"]}}
        rows.append((f"z' = {p['monitors']['mon_near']['z']:g} mm (3 lambda out)",
                     xn, yn, mod_n, al_n.numpy()))
        print(f"  near plane ({aln['convention']}): complex corr {corr_n:.4f}")

    checks = dict(out["aperture"]["pass"], **(out.get("near", {}).get("pass", {})))
    out["pass"] = bool(all(checks.values()))
    for k, v in checks.items():
        if not v:
            print(f"  ! {k} outside its criterion")
    if al["convention"] != "as-is" and not al["ambiguous"]:
        print("  ! matched only when CONJUGATED: Lumerical is exp(-i w t) like rail3D; check the "
              "export used E (not its conjugate) and the Mode source direction (Forward).")

    fig = plt.figure(figsize=(22, 5.2 * len(rows)), layout="constrained")
    subs = np.atleast_1d(fig.subfigures(len(rows), 1))
    for sub, (label, x, y, m, e) in zip(subs, rows):
        axs = sub.subplots(1, 5)
        vmax = float(np.abs(m).max())
        lit = lambda z: np.where(np.abs(z) > 0.05 * np.abs(z).max(), np.angle(z), np.nan)  # noqa: E731
        for ax, arr, cmap, lim, title in (
                (axs[0], np.abs(m), "viridis", (0, vmax), "|E_y| rail3D aperture model"),
                (axs[1], np.abs(e), "viridis", (0, vmax), "|E_y| FDTD (aligned)"),
                (axs[2], lit(m), "twilight", (-np.pi, np.pi), "arg rail3D (|E| > 5%)"),
                (axs[3], lit(e), "twilight", (-np.pi, np.pi), "arg FDTD (|E| > 5%)")):
            cm = plt.get_cmap(cmap).copy()
            cm.set_bad("#bdbdbd")
            im = ax.pcolormesh(x, y, arr.T, cmap=cm, vmin=lim[0], vmax=lim[1], shading="nearest")
            fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
            ax.set(title=title, xlabel="x' (mm, along A)", ylabel="y' (mm, along B, E)")
            ax.set_aspect("equal")
        ax = axs[4]
        if label.startswith("aperture"):
            for ff, ls, who in ((ff_m, "-", "model"), (ff_x, "--", "FDTD")):
                ax.plot(ff["theta_deg"], 20 * np.log10(ff["E_plane"] + 1e-12), "C0" + ls, label=f"E-plane {who}")
                ax.plot(ff["theta_deg"], 20 * np.log10(ff["H_plane"] + 1e-12), "C3" + ls, label=f"H-plane {who}")
            ax.set(ylim=(-40, 1), xlabel="angle from boresight (deg)", ylabel="dB",
                   title="far-field principal planes")
        else:
            jy = len(y) // 2
            ax.plot(x, np.abs(m[:, jy]) / vmax, "k-", label="rail3D")
            ax.plot(x, np.abs(e[:, jy]) / vmax, "C1--", label="FDTD")
            ax.set(xlabel="x' (mm)", title="|E_y| cut at y' = 0")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        sub.suptitle(label, x=0.01, ha="left", fontsize=12)
    fig.suptitle(f"Rung -1: {config.HORN_SPEC['part']} full-wave vs rail3D aperture model -- "
                 f"{'PASS' if out['pass'] else 'FAIL'}", fontsize=14, fontweight="bold")
    config.ensure_dirs()
    fig_path = config.FIGURE_DIR / "fdtd_horn_agreement.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    strip = lambda d: {k: v for k, v in d.items()}                       # noqa: E731
    js = config.GENERATED_DIR / "fdtd_horn_agreement.json"
    js.write_text(json.dumps(data3d.stamp(strip(out)), indent=2, default=str), encoding="utf-8")
    print(f"  -> {'PASS' if out['pass'] else 'FAIL'}   ({fig_path.name}, {js.name})")
    return 0 if out["pass"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", default="intact", choices=["plate", "intact", "crack",
                                                            "dent", "wear", "shell"])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--target", action="store_true",
                      help="synthesize the FDTD side from the stated model (a TARGET)")
    mode.add_argument("--external", default=None,
                      help="Lumerical E_y export (.mat/.npz with x, y) at z = 30 mm")
    mode.add_argument("--injection", default=None,
                      help="RUNG 0: E_y at z = 0 from the EMPTY-box run")
    mode.add_argument("--horn", default=None,
                      help="RUNG -1: E_y on mon_aperture from the horn_fdtd_case.py run")
    ap.add_argument("--horn-near", default=None,
                    help="RUNG -1: E_y on mon_near (3 lambda out) from the same run")
    ap.add_argument("--leak", default=None,
                    help="RUNG 0: E_y at the z = 30 monitor from the EMPTY-box run")
    ap.add_argument("--source", default="horn", choices=("horn", "plane"),
                    help="illumination rail3D is scored under (default horn: the "
                         "Import-source setup in LUMERICAL.md)")
    ap.add_argument("--profile", default="lab", choices=list(config.PROFILES))
    args = ap.parse_args()

    device = config.get_device(args.profile)
    if args.horn:
        return horn_check(args, device)
    if args.injection:
        return rung0(args, device)
    chunk = config.PROFILES[args.profile].chunk_faces
    config.ensure_dirs()
    section = sections.load_reference_section()
    params = sample_params(args.sample, section)

    # the monitor window comes from the same plan that writes case.json
    ds = config.MESH_DS
    v, _ = cw.build_mesh(section, params, ds, mesh3d.default_arc_count(section, ds), None)
    plan = cw.fdtd_plan(v.numpy(), dict(cw.geom_now(), h_ms=Z_MON), Z_MON, None)
    gw = window_geom(plan["monitor_mm"])
    win_txt = f"x +/-{gw['nx']*gw['dx']/2:g} y +/-{gw['ny']*gw['dx']/2:g} mm"
    # horn: every bounce (FDTD has them all); plane: single bounce, as before
    terms = "psi12" if args.source == "horn" else "psi1"
    common = dict(section=section, params=params, device=device, chunk=chunk,
                  seg=None, source=args.source, shadow="raycast",
                  min_t=config.SHADOW_MIN_T, terms=terms)

    t = time.time()
    ref_w = cw.solve(g=gw, **common)
    direct_ms = cw.solve(g=dict(cw.geom_now(), h_ms=config.H_MS), **common)
    print(f"rail3D {terms} under {args.source}: window {gw['nx']}x{gw['ny']} at "
          f"z={Z_MON:g} and direct at H_MS={config.H_MS:g} ({time.time()-t:.1f} s on {device})")

    if args.target:
        ext_raw = synthesize_target(ref_w, gw)
        print("FDTD side: SYNTHESIZED target (see TARGET_MODEL)")
    else:
        arr, lab, info = cw.load_external(args.external, gw)
        if info.get("plane_coverage", 1.0) < 1.0:
            print("  ! the monitor does not cover the window this comparison needs "
                  f"({win_txt}); widen it per LUMERICAL.md 3.3 before trusting the MS row")
        ext_raw = torch.as_tensor(arr).to(torch.complex64).to(device)
    ext_w, align = cw.align_external(ext_raw, ref_w)
    print(f"aligned: {align['convention']} (as-is {align['corr_as_is']:.4f} / conj "
          f"{align['corr_conjugated']:.4f}), gain {align['gain']:.4g}, "
          f"phase {align['phase_deg']:+.1f} deg")
    if align["ambiguous"]:
        print("  ! both conventions score alike: the fields may be uncorrelated. "
              "Go back to rung 1 before reading anything below.")

    ref_ms, ext_ms = to_ms_plane(ref_w, gw), to_ms_plane(ext_w, gw)
    ceiling = cw.complex_corr(ref_ms, direct_ms)
    g_ms = cw.geom_now()
    m30 = score(ref_w, ext_w, gw, device, ms_plane=False)
    mms = score(ref_ms, ext_ms, g_ms, device, ms_plane=True)

    tag = f"{args.sample}{'_plane' if args.source == 'plane' else ''}" \
          f"{'_target' if args.target else ''}"
    commit = data3d._git_commit()
    meta = {"sample": args.sample, "align": align, "window": win_txt,
            "window_ceiling": ceiling, "commit": commit, "source": args.source}
    rows = [(f"z = {Z_MON:g} mm near-field monitor ({win_txt}) -- where FDTD is recorded",
             ref_w, ext_w, m30, gw),
            (f"MS plane, z = {config.H_MS:g} mm ({config.NX}x{config.NY} aperture) -- "
             f"both carried by the same ASM", ref_ms, ext_ms, mms, g_ms)]
    fig_path = config.FIGURE_DIR / f"fdtd_agreement_{tag}.png"
    draw(rows, args.target, meta, fig_path)

    strip = lambda m: {k: v for k, v in m.items() if k != "_maps"}   # noqa: E731
    out = data3d.stamp({
        "mode": "TARGET (synthesized FDTD side)" if args.target else f"external {args.external}",
        "sample": args.sample, "source": args.source, "terms": terms,
        "window": win_txt, "window_ceiling_complex_corr": ceiling,
        "alignment": align, "criteria": CRITERIA,
        "target_model": TARGET_MODEL if args.target else None,
        "z30": strip(m30), "ms_plane": strip(mms)})
    js = config.GENERATED_DIR / f"fdtd_agreement_{tag}.json"
    js.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")

    for name, m in (("z=30 ", m30), ("MS   ", mms)):
        verdict = "PASS" if all(m["pass"].values()) else "FAIL: " + ", ".join(
            k for k, v in m["pass"].items() if not v)
        print(f"  {name} agreement {100*m['complex_corr']:.2f}%  in-tol "
              f"{100*m['in_tolerance']:.1f}%  max|dA| {100*m['max_amp_resid']:.1f}% at "
              f"{m['max_amp_resid_xy_mm']}  max|dphi| {m['max_phase_resid_deg']:.1f} deg at "
              f"{m['max_phase_resid_xy_mm']}" + (f"  barcode {m['barcode_cos']:.4f}"
                                                 if "barcode_cos" in m else "")
              + f"   -> {verdict}")
    print(f"  window ceiling (MS row): {ceiling:.4f}")
    print(f"  -> {fig_path}\n  -> {js}")
    if args.target:
        print("\nThis is a TARGET. Keep its title and footer when it is shown anywhere.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
