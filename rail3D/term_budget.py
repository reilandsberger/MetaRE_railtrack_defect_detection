"""Where the light at the metasurface comes from -- the rail (psi1, the signal)
vs the horn's re-radiation (psi2) and its own field (psi0) -- for the
configured horn placement and a reference placement (README finding 31).

    python term_budget.py --profile lab                  # 5090: production shadowing, 3/class
    python term_budget.py --profile lab --n 10           # tighter class means
    python term_budget.py --no-shadow --n 1 --profile laptop   # quick look

Every number is a power (sum of |psi|^2 over the 60 x 30 metasurface
aperture) relative to psi1 of the INTACT rail, in dB. Per geometry:

  INTO the metasurface (light crossing H_MS going up -- the model's input):
      P1 rail, P2 horn re-radiation, P0 horn direct.  A horn at or above the
      plane contributes NOTHING here (field3d upward_only).
  DEFECT SIGNAL: |psi1(defect) - psi1(intact)|^2 against the same difference
      of psi2 -- same pose, no augmentation, so only the defect differs.
  SIDEWAYS: horn light that reaches the plane travelling sideways or down --
      outside the model (it never passes up through the metasurface), but a
      physical stray-light problem for edges, frames and baffles.
  DIRECT TO DETECTORS: the horn's field on the detector plane (H_MS + layer)
      against intact psi1 carried there by the ASM with no metasurface. With
      the horn above H_MS this path never crosses the metasurface plane.
  ILLUMINATION: intact psi1 vs the reference geometry (signal level per unit
      horn drive), cut-end illumination (compare_wavefronts), and how much the
      intact field moves when the segment doubles (SEG_LEN 120 -> 240 mm).

Samples are unaugmented (no roll/jitter) so defect minus intact isolates the
defect. Writes data/generated/term_budget.json.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import numpy as np
import torch

from rail3d import config, data3d, field3d, mesh3d, optics3d, sections
import compare_wavefronts as cw

CLASSES = ("crack", "dent", "wear", "shell")


def db(x: float) -> float | None:
    return None if x <= 0 else round(10 * np.log10(x), 2)


def power(psi: torch.Tensor) -> float:
    return float((psi.abs() ** 2).sum())


def sample_params(cls: str, i: int, section) -> dict:
    if cls == "intact":
        return {"class": "intact"}
    defect = None
    if cls in config.DATASET_DIRS:
        files = sections.get_dataset_files(cls)
        defect = sections.match_reference_width(
            sections.load_vertices_from_csv(files[i]), section)
    params, _ = mesh3d.defect_params_for_sample(
        section, cls, defect, config.sample_seed(cls, i),
        n_arc=mesh3d.default_arc_count(section, config.MESH_DS))
    return params


def aperture_below(g: dict) -> float:
    """Fraction of the horn's aperture samples below the plane at g['h_ms']."""
    k0 = 2 * np.pi / g["wvl"]
    _, _, _, z, _, _ = field3d.aperture_field(g["size_ant"], g["dist_ant"], config.RESOL_ANT,
                                              config.THETA_INC, g["wvl"], k0)
    return float((z < g["h_ms"]).float().mean())


def horn_args(g: dict, device, h=None):
    X, Y = cw.plane_grid(g, device)
    return (X, Y, g["h_ms"] if h is None else h, g["wvl"], config.THETA_INC,
            g["size_ant"], g["dist_ant"], config.RESOL_ANT)


def rail_terms(section, params, g, device, chunk, shadow, seg=None):
    """psi1, psi2 (all directions) and psi2 (upward only) on the plane g."""
    ds = g["mesh_ds"]
    v, f = cw.build_mesh(section, params, ds, mesh3d.default_arc_count(section, ds), seg)
    kw = {}
    if shadow:
        n_occ = mesh3d.default_arc_count(section, g["occ_ds"])
        v_o, f_o = cw.build_mesh(section, params, g["occ_ds"], n_occ, seg)
        kw = {"shadow": "raycast", "shadow_occluders": (v_o.to(device), f_o.to(device)),
              "shadow_min_t": config.SHADOW_MIN_T,
              "shadow_normal_offset": config.SHADOW_NORMAL_OFFSET}
    args = horn_args(g, device)
    psi1, psi2 = field3d.scattered_fields(v.to(device), f.to(device), *args,
                                          chunk_faces=chunk, **kw)
    below = aperture_below(g)
    if below == 1.0:
        psi2_up = psi2
    elif below == 0.0:
        psi2_up = torch.zeros_like(psi2)
    else:
        _, psi2_up = field3d.scattered_fields(v.to(device), f.to(device), *args,
                                              chunk_faces=chunk, upward_only=True, **kw)
    return psi1.cpu(), psi2.cpu(), psi2_up.cpu()


def budget(g: dict, label: str, section, n: int, device, chunk, shadow, seg_check) -> dict:
    t0 = time.time()
    args = horn_args(g, device)
    psi0_all = field3d.horn_to_plane(*args).cpu()
    psi0_up = field3d.horn_to_plane(*args, upward_only=True).cpu()

    p1_i, p2_i, p2u_i = rail_terms(section, {"class": "intact"}, g, device, chunk, shadow)
    P1 = power(p1_i)
    rel = lambda p: db(p / P1)                                     # noqa: E731

    th = config.THETA_INC
    zc = float(g["dist_ant"] * np.cos(th))
    out = {
        "label": label, "dist_ant_mm": round(float(g["dist_ant"]), 3),
        "aperture_centre_xz_mm": [round(float(g["dist_ant"] * np.sin(th)), 2), round(zc, 2)],
        "horn_lowest_z_mm": round(zc - float((g["size_ant"][0] / 2 + config.HORN_WALL)
                                             * np.sin(th)), 2),
        "aperture_fraction_below_ms": aperture_below(g),
        "P1_intact_abs": P1,
        "into_metasurface_dB_vs_P1": {"psi2": rel(power(p2u_i)), "psi0": rel(power(psi0_up))},
        "sideways_dB_vs_P1": {"psi2": rel(power(p2_i - p2u_i)),
                              "psi0": rel(power(psi0_all - psi0_up))},
    }

    # defect signal: same pose as the intact reference, only the defect differs
    sig = {}
    for cls in CLASSES:
        s1, s2 = [], []
        for i in range(n):
            q1, _, q2u = rail_terms(section, sample_params(cls, i, section), g, device,
                                    chunk, shadow)
            s1.append(power(q1 - p1_i))
            s2.append(power(q2u - p2u_i))
        S1, S2 = float(np.mean(s1)), float(np.mean(s2))
        sig[cls] = {"signal_dB_vs_P1": rel(S1),
                    "psi2_signal_dB_vs_psi1_signal": db(S2 / S1) if S1 > 0 else None,
                    "S1_abs": S1}
    out["defect_signal"] = sig

    # the horn straight onto the detector plane, against the rail's light there
    gd = dict(g)
    h_det = g["h_ms"] + g["layer"]
    horn_det = field3d.horn_to_plane(*horn_args(gd, device, h=h_det), upward_only=True).cpu()
    prop = optics3d.PropagatorASM2D(g["layer"], nx=g["nx"], ny=g["ny"])
    rail_det = prop(p1_i.to(torch.complex64).unsqueeze(0))[0]
    out["detector_plane"] = {
        "z_mm": h_det,
        "horn_direct_dB_vs_rail": db(power(horn_det) / power(rail_det)),
        "horn_line_of_sight_crosses_ms_plane": bool(out["horn_lowest_z_mm"] < g["h_ms"]),
    }

    # illumination: cut ends, and the intact field's sensitivity to segment length
    v, f = cw.build_mesh(section, {"class": "intact"}, g["mesh_ds"],
                         mesh3d.default_arc_count(section, g["mesh_ds"]), None)
    cut = cw.cut_end_illumination(v.numpy(), f.numpy(), g)
    out["cut_end_vs_midspan_illumination"] = float(cut["cut_end_vs_midspan_illumination"])
    if seg_check:
        p1_long, _, _ = rail_terms(section, {"class": "intact"}, g, device, chunk, shadow,
                                   seg=2 * config.SEG_LEN)
        out["intact_field_change_seg_120_to_240"] = {
            "rel_l2": round(float((p1_long - p1_i).norm() / p1_long.norm()), 4),
            "complex_corr": round(cw.complex_corr(p1_i, p1_long), 4)}
    out["seconds"] = round(time.time() - t0, 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=3, help="samples per defect class")
    ap.add_argument("--ref-dist", type=float, default=140.0,
                    help="reference horn distance (mm) to compare against; 0 = none. "
                         "140 = the placement before 2026-10-02 (aperture at z 71-90)")
    ap.add_argument("--no-shadow", action="store_true",
                    help="skip ray-cast shadowing (quick look; datasets use it)")
    ap.add_argument("--no-seg-check", action="store_true",
                    help="skip the 240 mm segment solve")
    ap.add_argument("--profile", default="lab", choices=list(config.PROFILES))
    args = ap.parse_args()

    device = config.get_device(args.profile)
    chunk = config.PROFILES[args.profile].chunk_faces
    config.ensure_dirs()
    section = sections.load_reference_section()
    shadow = not args.no_shadow
    print(f"[rail3d] term budget on {device}: {args.n}/class, shadowing "
          f"{'raycast' if shadow else 'OFF'}")

    geoms = [(cw.geom_now(), f"configured (DIST_ANT {config.DIST_ANT:.1f} mm)")]
    if args.ref_dist > 0:
        geoms.append((dict(cw.geom_now(), dist_ant=args.ref_dist),
                      f"reference (DIST_ANT {args.ref_dist:.1f} mm)"))
    res = [budget(g, lab, section, args.n, device, chunk, shadow, not args.no_seg_check)
           for g, lab in geoms]
    if len(res) == 2:
        res[0]["P1_intact_vs_reference_dB"] = db(res[0]["P1_intact_abs"] / res[1]["P1_intact_abs"])
        res[0]["defect_signal_vs_reference_dB"] = {
            c: db(res[0]["defect_signal"][c]["S1_abs"] / res[1]["defect_signal"][c]["S1_abs"])
            for c in CLASSES}

    def show(x):
        return "  none " if x is None else f"{x:+7.1f}"

    for r in res:
        print(f"\n{r['label']}: aperture centre x,z = {r['aperture_centre_xz_mm']} mm, "
              f"lowest point z = {r['horn_lowest_z_mm']} mm, "
              f"{100 * r['aperture_fraction_below_ms']:.0f}% of the aperture below the metasurface")
        print(f"  into the metasurface, dB vs intact psi1:  psi2 {show(r['into_metasurface_dB_vs_P1']['psi2'])}"
              f"   psi0 {show(r['into_metasurface_dB_vs_P1']['psi0'])}")
        print(f"  sideways/down at the plane (not input):   psi2 {show(r['sideways_dB_vs_P1']['psi2'])}"
              f"   psi0 {show(r['sideways_dB_vs_P1']['psi0'])}")
        for c, s in r["defect_signal"].items():
            print(f"  {c:6s} defect signal {show(s['signal_dB_vs_P1'])} dB vs intact psi1;"
                  f"  its psi2 part {show(s['psi2_signal_dB_vs_psi1_signal'])} dB vs its psi1 part")
        d = r["detector_plane"]
        print(f"  horn straight onto the detector plane (z={d['z_mm']:g}): "
              f"{show(d['horn_direct_dB_vs_rail'])} dB vs the rail's light there"
              f"  (line of sight crosses the metasurface plane: {d['horn_line_of_sight_crosses_ms_plane']})")
        print(f"  cut-end illumination {r['cut_end_vs_midspan_illumination']:.3f}x mid-span"
              + (f";  intact field moves rel L2 {r['intact_field_change_seg_120_to_240']['rel_l2']:.3f}"
                 f" (corr {r['intact_field_change_seg_120_to_240']['complex_corr']:.4f}) when the"
                 f" segment doubles" if "intact_field_change_seg_120_to_240" in r else ""))
    if len(res) == 2:
        print(f"\nsignal level, configured vs reference: intact psi1 "
              f"{show(res[0]['P1_intact_vs_reference_dB'])} dB; defect signal "
              + ", ".join(f"{c} {show(v_)}" for c, v_ in res[0]["defect_signal_vs_reference_dB"].items())
              + " dB")

    out = {"results": res, "n_per_class": args.n, "shadow": shadow,
           "_stamp": {"at": datetime.now().isoformat(timespec="seconds"),
                      "commit": data3d._git_commit(), "device": str(device)}}
    path = config.GENERATED_DIR / "term_budget.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
