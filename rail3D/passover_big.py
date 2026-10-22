"""The big pass-over set: 1205 passes, 130 detectors, narrowed defect ranges.

    python passover_big.py --profile lab            # solve + assemble (resumable)
    python passover_big.py --profile lab --assemble-only
    python passover_big.py --smoke                  # 1 short pass per subtype (laptop check)

Same physics and pass mechanics as passover.py (each pass slides one defect
through the fixed beam, one field solve per frame, the generator's own code
path; pose fixed within a pass; no metasurface), with:

  passes     5 intact + 200 each of crack_transverse, crack_oblique,
             crack_longitudinal, dent, wear, shell
  ranges     every defect centred at arc s0 = 140-160 mm (the well-seen gauge
             shoulder; wear's centre follows its 30-40 deg cut: s0 ~ 157-160)
             crack: 3 lines, 3-4 mm wide, 20-30 mm long, 4-6 mm deep
             dent:  4-6 mm deep, 18-22 mm FWHM both ways, single pit
             wear:  7-10 mm deep at the gauge corner (crown wear as usual, 50%)
             shell: 5-7 mm deep (shape as usual)
  readout    the 130 tiled windows (13 x 10, 92% of the plane), d001..d130

Everything else is the dataset's model. The narrowing is applied by drawing
each defect with these ranges in place of config's (RANGES) and then setting
s0, the crack orientation and the dent pit count (draw_defect); nothing here
changes config, datasets or provenance.

Outputs in data/generated/passover130/:
  passes/<pass_id>.npz            raw metasurface-plane field per pass (resume unit)
  passover130_<noise>.csv / .npz  noise in {clean, low, high}
  passover130_labels.json         every pass's parameters + definitions
  passover130_overview.png        one pass per subtype, all 130 detectors
  fields/<pass_id>.npz            only with --fields: frame-matched fields at the
                                  detector and metasurface planes (~2.7 GB)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import generate_dataset_3d as gen
import passover as po
from rail3d import config, data3d, mesh3d, optics3d, sections

SUBTYPES = [  # (subtype, class, passes, dataset-index offset)
    ("intact", "intact", 5, 0),
    ("crack_transverse", "crack", 200, 0),
    ("crack_oblique", "crack", 200, 200),
    ("crack_longitudinal", "crack", 200, 400),
    ("dent", "dent", 200, 0),
    ("wear", "wear", 200, 0),
    ("shell", "shell", 200, 0),
]
SUBTYPE_INDEX = {s[0]: k for k, s in enumerate(SUBTYPES)}
RANGES = {
    "CRACK_WIDTH_RANGE": (3.0, 4.0), "CRACK_LENGTH_RANGE": (20.0, 30.0),
    "CRACK_DEPTH_RANGE": (4.0, 6.0),
    "DENT_DEPTH_RANGE": (4.0, 6.0), "DENT_FOOTPRINT_S": (18.0, 22.0),
    "DENT_FOOTPRINT_Y": (18.0, 22.0),
    "WEAR_DEPTH_RANGE": (7.0, 10.0), "WEAR_GAUGE_ANGLE_RANGE": (30.0, 40.0),
    "SHELL_DEPTH_RANGE": (5.0, 7.0),
}
S0_RANGE = (140.0, 160.0)
OBLIQUE_DEG = (20.0, 70.0)            # |theta| for oblique cracks, either sign
OUT_NAME = "passover130"


@contextlib.contextmanager
def narrowed():
    """Draw defects with RANGES in place of config's (restored afterwards)."""
    old = {k: getattr(config, k) for k in RANGES}
    try:
        for k, v in RANGES.items():
            setattr(config, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(config, k, v)


def draw_defect(spec, section, n_fine):
    """(params, aug) for one pass: the seed's draw under RANGES, then s0,
    orientation and pit count set from a second, seed-derived generator."""
    cls = spec["class"]
    with narrowed():
        p, aug = mesh3d.defect_params_for_sample(section, cls, None, spec["seed"], n_arc=n_fine)
    rng = np.random.default_rng(spec["seed"])
    if cls in ("crack", "dent", "shell"):
        p["s0"] = float(rng.uniform(*S0_RANGE))
    if cls == "crack":
        st = spec["subtype"]
        p["theta"] = (math.pi / 2 if st == "crack_transverse" else 0.0 if st == "crack_longitudinal"
                      else float(rng.choice([-1.0, 1.0]) * math.radians(rng.uniform(*OBLIQUE_DEG))))
    if cls == "dent":
        p["pits"] = [(0.0, 0.0, 1.0)]
    return p, aug


def pass_list() -> list[dict]:
    per = {s: [] for s, *_ in SUBTYPES}
    for st, cls, n, off in SUBTYPES:
        for k in range(n):
            i = off + k
            per[st].append({"pass_id": f"{st}_{k:03d}", "subtype": st, "class": cls,
                            "dataset_index": i, "seed": config.sample_seed(cls, i)})
    out = []                                         # round robin: partial runs hold every subtype
    for k in range(max(len(v) for v in per.values())):
        out += [per[st][k] for st, *_ in SUBTYPES if k < len(per[st])]
    return out


def run_pass(spec, section, n_fine, n_coarse, device, chunk, batch, out_dir, step_scale):
    path = out_dir / "passes" / f"{spec['pass_id']}.npz"
    if path.exists():
        return None
    cls = spec["class"]
    p, aug = draw_defect(spec, section, n_fine)
    ys = po.frame_positions(cls, p, step_scale)
    psi2 = config.HORN_LOWER_EDGE_Z < config.H_MS   # 0 while the horn is above the plane
    faces, psis = [None, None], []
    for b in range(0, len(ys), batch):
        vs, vos = [], []
        for y0 in ys[b:b + batch]:
            v, faces[0], vo, faces[1], _ = gen.sample_meshes(
                section, cls, spec["dataset_index"], None, n_fine, n_coarse, faces[0], faces[1],
                y0=None if cls == "intact" else y0, params_aug=(p, aug))
            vs.append(v)
            vos.append(vo)
        p1, p2 = gen.solve_fields(torch.stack(vs).to(device), faces[0], torch.stack(vos).to(device),
                                  faces[1], device, chunk, compute_psi2=psi2)
        psis.append((p1 + p2).cpu().numpy().astype(np.complex64))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, psi=np.concatenate(psis), y0=ys, spec=json.dumps(spec),
                        params=json.dumps(p, default=po._jsonable), aug=json.dumps(aug))
    tmp.replace(path)
    return len(ys)


# ---------------------------------------------------------------------------
# assembly (streams pass by pass)
# ---------------------------------------------------------------------------
def _params_from_file(z):
    p = json.loads(str(z["params"]))
    for k in ("profile_v", "profile_d"):
        if k in p:
            p[k] = np.asarray(p[k], float)
    return p


@torch.no_grad()
def _readout(psi, centres, chunk=128):
    return np.concatenate([po.readout(np.ascontiguousarray(psi[i:i + chunk]), centres)
                           for i in range(0, len(psi), chunk)])


def assemble(out_dir: Path, save_fields: bool) -> None:
    section = sections.load_reference_section()
    geom = mesh3d.arc_geometry(section, mesh3d.default_arc_count(section, config.MESH_DS))
    dense = config.dense_detector_centers()
    dn = dense.numpy()
    mirror = [int(np.argmin(np.abs(dn - [c[0], -c[1]]).sum(1))) for c in dn]
    assert np.allclose(dn[mirror], dn * [1, -1])
    order = {s: k for k, (s, *_) in enumerate(SUBTYPES)}
    files = sorted((out_dir / "passes").glob("*.npz"),
                   key=lambda f: (order[f.stem.rsplit("_", 1)[0]], f.stem))
    if not files:
        print("!! no finished passes")
        return
    intact = [f for f in files if f.stem.startswith("intact_")]
    if not intact:
        print("!! no intact pass finished yet: cannot set the power unit")
        return
    ref_win = np.mean([_readout(np.load(f)["psi"], dense).mean(0) for f in intact], 0)
    ref = float(ref_win.mean())
    names = [f"d{k + 1:03d}" for k in range(len(dn))]
    onn = optics3d.ONN3D(surface="none", noise=False).eval() if save_fields else None
    if save_fields:
        (out_dir / "fields").mkdir(exist_ok=True)

    rng = np.random.default_rng(config.SEED)
    fh = {k: open(out_dir / f"{OUT_NAME}_{k}.csv", "w", encoding="utf-8", newline="")
          for k in po.NOISE}
    head = "pass_id,subtype,class,class_index,subtype_index,frame,y_mm,depth_under_beam_mm," + ",".join(names)
    for f in fh.values():
        f.write(head + "\n")
    cols = {k: [] for k in po.NOISE}
    meta_cols = {"pass_id": [], "subtype_index": [], "class_index": [], "frame": [], "y_mm": [],
                 "depth_under_beam_mm": []}
    passes, overview, t0 = [], {}, time.time()
    for n_done, f in enumerate(files):
        z = np.load(f)
        spec, p, aug = json.loads(str(z["spec"])), _params_from_file(z), json.loads(str(z["aug"]))
        psi, y0 = z["psi"], z["y0"]
        cls, st = spec["class"], spec["subtype"]
        d = _readout(psi, dense)
        if cls == "intact":
            n = len(np.arange(-po.HALF_SPAN, po.HALF_SPAN + 1e-9, po.STEP))
            y0 = np.arange(-po.HALF_SPAN, po.HALF_SPAN + 1e-9, po.STEP)
            d, psi = np.repeat(d, n, 0), np.repeat(psi, n, 0)
            depth = np.zeros(n)
        else:
            depth = np.array([mesh3d.render_depth_field({**p, "y0": float(y)}, geom,
                                                        np.array([0.0])).max() for y in y0])
        if cls == "wear":                   # same direction of travel as every pass (passover.py)
            y0 = -(y0 - p["L"] / 2)
            d = d[:, mirror]
            psi = psi[:, :, ::-1]
        base = d / ref
        if save_fields:
            fms = np.ascontiguousarray(psi)
            fdet = np.concatenate([onn.propagate(torch.from_numpy(fms[i:i + 128])).numpy()
                                   for i in range(0, len(fms), 128)]).astype(np.complex64)
            np.savez_compressed(out_dir / "fields" / f"{spec['pass_id']}.npz", y_mm=y0,
                                field_metasurface=fms.astype(np.complex64), field_detector=fdet)
        ci, si = po.CLASSES.index(cls), SUBTYPE_INDEX[st]
        for name, (a, b) in po.NOISE.items():
            v = base * (1 + a * rng.standard_normal(base.shape)) + b * rng.standard_normal(base.shape)
            cols[name].append(v.astype(np.float32))
            w = fh[name]
            for k in range(len(y0)):
                w.write(f"{spec['pass_id']},{st},{cls},{ci},{si},{k},{y0[k]:.2f},{depth[k]:.4f},"
                        + ",".join(f"{x:.6g}" for x in v[k]) + "\n")
        meta_cols["pass_id"] += [spec["pass_id"]] * len(y0)
        meta_cols["subtype_index"] += [si] * len(y0)
        meta_cols["class_index"] += [ci] * len(y0)
        meta_cols["frame"] += list(range(len(y0)))
        meta_cols["y_mm"] += list(y0)
        meta_cols["depth_under_beam_mm"] += list(depth)
        if st not in overview:
            overview[st] = (spec["pass_id"], y0, base / (ref_win / ref))
        passes.append({
            **spec, "class_index": ci, "subtype_index": si, "n_frames": int(len(y0)),
            "description": po.describe(cls, p, geom),
            "parameters": {k: v for k, v in p.items()
                           if k not in ("offsets", "profile_v", "profile_d", "fourier", "lobes",
                                        "pits", "rough_seed", "band", "class", "y0")},
            "pose": {"roll_deg": aug["roll_deg"], "lateral_mm": aug["jitter_x"],
                     "vertical_mm": aug["jitter_z"]},
            "y_mm_range": [float(y0[0]), float(y0[-1])],
            "peak_change_vs_intact": float((np.abs(base - ref_win / ref) / (ref_win / ref)).max())})
        if (n_done + 1) % 100 == 0:
            print(f"  assembled {n_done + 1}/{len(files)} passes ({(time.time() - t0) / 60:.1f} min)",
                  flush=True)
    for f in fh.values():
        f.close()
    for name in po.NOISE:
        np.savez_compressed(out_dir / f"{OUT_NAME}_{name}.npz",
                            detectors=np.concatenate(cols[name]),
                            **{k: np.asarray(v) for k, v in meta_cols.items()})

    g = {"wavelength_mm": config.WVL, "frequency_GHz": round(299.792458 / config.WVL, 4),
         "horn": config.HORN_SPEC.get("part"), "horn_incidence_deg": 55.0,
         "horn_distance_mm": round(config.DIST_ANT, 2),
         "metasurface_plane": {"z_mm": config.H_MS, "size_mm": [config.WX, config.WY]},
         "metasurface": "none (free propagation)",
         "detector_plane_z_mm": config.H_MS + config.LAYER_DISTANCES[-1],
         "rail_segment_mm": config.SEG_LEN, "geometry_tag": data3d.geometry_tag()}
    labels = {
        "about": ("Simulated detector outputs as a train passes over rail-head defects "
                  "(rail3D physical-optics model, NOT measurements), for one specific, possibly "
                  "non-optimal geometry: single horn at 55 deg, 278.5 mm out, no metasurface. "
                  "Defects deliberately restricted to the well-seen gauge shoulder and to narrow "
                  "size ranges."),
        "geometry": g,
        "detectors": {"count": len(dn), "window_mm": list(config.DET_SIZE),
                      "layout": "13 x 10 tiling of the 150 x 75 mm plane (92% covered), not optimised",
                      "order": ("d001..d130, x-major: index = 10 * ix + iy + 1, ix = 0..12 from -x "
                                "to +x, iy = 0..9 from -y to +y"),
                      "centres_mm_x_y": dn.tolist(),
                      "x_axis": "across the rail, +x = gauge side (the horn side)"},
        "frames": {"axis": "y_mm = position of the defect relative to the beam centre along the rail",
                   "direction": ("every pass moves toward +y: rows are in time order, "
                                 "t = (y_mm - y_mm[first frame]) / train speed"),
                   "wear_axis": ("for wear passes y_mm is the worn stretch's LEADING EDGE, -60 -> "
                                 "+60 (worn rail lies behind it): -60 = unworn, +60 = fully worn"),
                   "defect_step_mm": po.STEP, "wear_step_mm": po.WEAR_STEP,
                   "depth_under_beam_mm": "defect depth at y = 0 (the beam centre) per frame"},
        "units": ("relative power: 1.0 = the mean intact window power (over all 130 windows and "
                  f"all intact passes; absolute model value {ref:.4e}). Windows differ a lot: "
                  "see intact_window_level"),
        "intact_window_level": (ref_win / ref).tolist(),
        "noise": {k: {"multiplicative_std": a, "additive_std": b,
                      "model": "P_noisy = P (1 + mult n1) + add n2, n ~ N(0,1), per window per frame"}
                  for k, (a, b) in po.NOISE.items()},
        "defect_ranges": {**{k: list(v) for k, v in RANGES.items()}, "s0_mm": list(S0_RANGE),
                          "oblique_abs_theta_deg": list(OBLIQUE_DEG), "dent_pits": 1,
                          "wear_centre_s0_mm": "157-160 (follows the 30-40 deg gauge cut)"},
        "pose": "rail roll / lateral / vertical offset fixed within a pass, different per pass",
        "peak_change_vs_intact": ("per pass: max over frames and windows of |P - P_intact| / P_intact, "
                                  "P_intact = that window's mean over the intact passes (clean)"),
        "segment_note": ("the modelled rail is 120 mm long and the beam lights most of it, so a "
                         "defect at y = +-40 mm is still partly seen: passes start and end lit"),
        "class_index": {c: k for k, c in enumerate(po.CLASSES)},
        "subtype_index": SUBTYPE_INDEX,
        "passes": passes,
    }
    (out_dir / f"{OUT_NAME}_labels.json").write_text(json.dumps(labels, indent=1, default=po._jsonable),
                                                    encoding="utf-8")
    plot_overview(overview, out_dir / f"{OUT_NAME}_overview.png")
    write_pass_list(labels, out_dir)
    plot_layout(labels, out_dir)
    write_readme(labels, out_dir, n_rows=len(meta_cols["y_mm"]), fields=save_fields)
    import shutil
    shutil.copy(Path(__file__).with_name("load_passover.py"), out_dir / "load_passover.py")
    if (config.FIGURE_DIR / "setup_diagram.png").exists():
        shutil.copy(config.FIGURE_DIR / "setup_diagram.png", out_dir / "setup_diagram.png")
    print(f"  assembled {len(files)} passes -> {out_dir}")


# ---------------------------------------------------------------------------
# human-readable pass list, layout figure, README
# ---------------------------------------------------------------------------
PASS_LIST_COLS = ["pass_id", "subtype", "class_index", "subtype_index", "n_frames", "depth_mm",
                  "width_mm", "length_mm", "orientation_deg", "footprint_mm", "s0_mm", "x_mm",
                  "extra", "roll_deg", "lateral_mm", "vertical_mm", "peak_change_pct"]


def pass_row(p, geom=None) -> dict:
    m, st = p["parameters"], p["subtype"]
    r = {"pass_id": p["pass_id"], "subtype": st, "class_index": p["class_index"],
         "subtype_index": p["subtype_index"], "n_frames": p["n_frames"],
         "depth_mm": "", "width_mm": "", "length_mm": "", "orientation_deg": "", "footprint_mm": "",
         "s0_mm": "", "x_mm": "", "extra": "",
         "roll_deg": f"{p['pose']['roll_deg']:+.2f}", "lateral_mm": f"{p['pose']['lateral_mm']:+.2f}",
         "vertical_mm": f"{p['pose']['vertical_mm']:+.2f}",
         "peak_change_pct": f"{100 * p['peak_change_vs_intact']:.0f}"}
    if st == "intact":
        return r
    r["depth_mm"] = f"{m['depth']:.2f}"
    if "s0" in m:
        r["s0_mm"] = f"{m['s0']:.1f}"
        if geom is not None:
            r["x_mm"] = f"{float(np.interp(m['s0'], geom['s'], geom['x'])):+.1f}"
    if p["class"] == "crack":
        r.update(width_mm=f"{m['width']:.2f}", length_mm=f"{m['L']:.1f}",
                 orientation_deg=f"{math.degrees(m['theta']):.0f}",
                 extra=f"3 lines, {m['line_gap']:.2f} mm apart")
    elif st == "dent":
        r["footprint_mm"] = f"{m['fw_s']:.1f} x {m['fw_y']:.1f}"
    elif st == "shell":
        r["footprint_mm"] = f"{2 * m['r_s']:.1f} x {2 * m['r_y']:.1f}"
    elif st == "wear":
        r.update(length_mm=f"{m['L']:.0f}", orientation_deg=f"{m['gauge_angle_deg']:.0f}",
                 extra=(f"crown wear {m['crown_depth']:.2f} mm" if m.get("crown") else "no crown wear"))
    return r


def write_pass_list(labels, out_dir: Path) -> None:
    section = sections.load_reference_section()
    geom = mesh3d.arc_geometry(section, mesh3d.default_arc_count(section, config.MESH_DS))
    rows = [pass_row(p, geom) for p in labels["passes"]]
    with open(out_dir / f"{OUT_NAME}_pass_list.csv", "w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(PASS_LIST_COLS) + "\n")
        for r in rows:
            fh.write(",".join(f'"{r[c]}"' if "," in str(r[c]) else str(r[c]) for c in PASS_LIST_COLS) + "\n")
    md = [f"# {OUT_NAME}: every pass ({len(rows)})", "",
          "Same columns as `passover130_pass_list.csv`; see README §4 for what each means.", ""]
    md += _md_table(rows)
    (out_dir / f"{OUT_NAME}_pass_list.md").write_text("\n".join(md) + "\n", encoding="utf-8")


MD_COLS = [("pass_id", "pass"), ("subtype", "subtype"), ("depth_mm", "depth"), ("width_mm", "width"),
           ("length_mm", "length"), ("orientation_deg", "angle °"), ("footprint_mm", "footprint"),
           ("s0_mm", "s0"), ("x_mm", "x"), ("extra", "extra"), ("roll_deg", "roll °"),
           ("lateral_mm", "lat."), ("vertical_mm", "vert."), ("peak_change_pct", "peak %")]


def _md_table(rows):
    out = ["| " + " | ".join(h for _, h in MD_COLS) + " |", "|" + "---|" * len(MD_COLS)]
    for r in rows:
        out.append("| " + " | ".join(f"`{r[c]}`" if c == "pass_id" else (str(r[c]) or "—")
                                     for c, _ in MD_COLS) + " |")
    return out


def plot_layout(labels, out_dir: Path) -> None:
    """The CSV layout (real rows) and the 130-window map, for the README."""
    import csv as _csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    with open(out_dir / f"{OUT_NAME}_low.csv", encoding="utf-8") as fh:
        rd = _csv.reader(fh)
        hdr = next(rd)
        rows = []
        for r in rd:
            rows.append(r)
            if len(rows) >= 90:
                break
    keep = list(range(8)) + [8, 9, 10, len(hdr) - 1]      # ids, y, depth, d001-d003, d130
    show = [rows[0], rows[1], None, rows[81], rows[82], None]
    fig = plt.figure(figsize=(17, 6.6))
    ax = fig.add_axes([0.01, 0.08, 0.64, 0.8])
    ax.axis("off")
    nc = len(keep) + 1
    ax.set_xlim(0, nc)
    ax.set_ylim(-1.2, 9.6)
    for a, b, col, txt in ((0, 6, "#dbe9f6", "which pass / which snapshot"),
                           (6, 8, "#fde8cf", "where the defect is"),
                           (8, nc, "#dff3df", "130 window readings")):
        ax.add_patch(Rectangle((a, -0.3), b - a, 9.3, color=col, zorder=0))
        ax.text((a + b) / 2, 9.15, txt, ha="center", va="bottom", fontsize=10, fontweight="bold")
    heads = [hdr[j] for j in keep[:11]] + ["…", hdr[-1]]
    wrap = {"class_index": "class_\nindex", "subtype_index": "subtype_\nindex",
            "depth_under_beam_mm": "depth_under\n_beam_mm", "subtype": "subtype"}
    for j, h in enumerate(heads):
        ax.text(j + 0.5, 8.2, wrap.get(h, h), ha="center", va="center", fontsize=8,
                fontweight="bold", family="monospace")
    y = 7.2
    for r in show:
        if r is None:
            ax.text(nc / 2, y, "⋮", ha="center", va="center", fontsize=14)
            y -= 1.0
            continue
        vals = [r[j] for j in keep[:11]] + ["…", r[-1]]
        for j, v in enumerate(vals):
            if j in (0, 1):                 # long crack names over two lines
                v = v.replace("crack_", "crack_\n")
            elif j >= 6 and v not in ("…",):
                v = f"{float(v):.3g}"
            ax.text(j + 0.5, y, v, ha="center", va="center", fontsize=7.5, family="monospace")
        y -= 1.0
    ax.text(nc / 2, -0.85, "every row = ONE snapshot of all 130 windows; a pass = consecutive rows "
            "with the same pass_id (81 rows; wear 61), in time order", ha="center", fontsize=9.5,
            style="italic")
    ax.set_title(f"{OUT_NAME}_<clean|low|high>.csv - 1 header row + one row per snapshot",
                 fontsize=12, loc="left")
    bx = fig.add_axes([0.67, 0.14, 0.32, 0.7])
    W, H = config.WX, config.WY
    dw, dh = config.DET_SIZE
    bx.add_patch(Rectangle((-W / 2, -H / 2), W, H, fill=False, lw=1.5))
    lvl = np.asarray(labels["intact_window_level"])
    cmap = plt.get_cmap("Greens")
    for k, (cx, cy) in enumerate(labels["detectors"]["centres_mm_x_y"]):
        bx.add_patch(Rectangle((cx - dw / 2, cy - dh / 2), dw, dh, ec="0.3", lw=0.4,
                               fc=cmap(0.15 + 0.75 * min(lvl[k] / lvl.max(), 1))))
        if k % 10 in (0, 9) or k // 10 in (0, 12):
            bx.text(cx, cy, f"{k + 1:03d}", ha="center", va="center", fontsize=5.5)
    bx.annotate("", xy=(-W / 2 - 4, H / 2 + 6), xytext=(-W / 2 - 4, -H / 2 - 6),
                arrowprops=dict(arrowstyle="->", lw=2, color="#c0392b"))
    bx.text(-W / 2 - 3, H / 2 + 8, "defect travels +y (= time)", color="#c0392b", fontsize=8)
    bx.set_xlim(-W / 2 - 10, W / 2 + 4)
    bx.set_ylim(-H / 2 - 12, H / 2 + 14)
    bx.set_aspect("equal")
    bx.set_xlabel("x (mm), across the rail (+x = gauge / horn side)")
    bx.set_ylabel("y (mm), along the rail")
    bx.set_title("the 130 windows (13 x 10), numbered x-major:\nd001 bottom-left, d010 top-left, "
                 "d130 top-right; shade = intact brightness", fontsize=9)
    fig.savefig(out_dir / "data_layout.png", dpi=140)
    plt.close(fig)


def write_readme(labels, out_dir: Path, n_rows: int, fields: bool) -> None:
    g, rng_ = labels["geometry"], labels["defect_ranges"]
    lvl = np.asarray(labels["intact_window_level"])
    counts = {}
    for p in labels["passes"]:
        counts[p["subtype"]] = counts.get(p["subtype"], 0) + 1
    section = sections.load_reference_section()
    geom = mesh3d.arc_geometry(section, mesh3d.default_arc_count(section, config.MESH_DS))
    snap, seen = [], {}
    for p in labels["passes"]:
        if seen.get(p["subtype"], 0) < (5 if p["subtype"] == "intact" else 3):
            snap.append(pass_row(p, geom))
            seen[p["subtype"]] = seen.get(p["subtype"], 0) + 1
    noise = "\n".join(f"| `{OUT_NAME}_{k}.csv` | {v['multiplicative_std'] * 100:g}% | "
                      f"{v['additive_std'] * 100:g}% |" for k, v in labels["noise"].items())
    count_rows = "\n".join(f"| {st} | {SUBTYPE_INDEX[st]} | {n} |" for st, n in counts.items())
    fields_txt = (f"`fields/<pass_id>.npz` (written with `--fields`) holds, per pass, the complex "
                  f"field frame for frame with the CSV: `field_detector` (z = "
                  f"{g['detector_plane_z_mm']:.0f} mm) and `field_metasurface` (z = "
                  f"{g['metasurface_plane']['z_mm']:.0f} mm), each (frames, 60, 30), plus `y_mm`. "
                  "Read it with `load_fields` / `readout` (§7) to try other detector layouts."
                  if fields else
                  "This build has no raw fields (the full set is ~2.7 GB). Re-run the assembly with "
                  "`--fields` to add `fields/<pass_id>.npz`, or ask for them: with them, any other "
                  "detector layout can be read out without re-simulating (§7).")
    text = f"""# Rail pass-over detector data, 130-window set — README

Simulated readings from **130 microwave power detectors** tiling the plane
above a rail, recorded as a train carries the sensor over a defect. There are
**{len(labels['passes'])} passes**: {', '.join(f'{n} {st}' for st, n in counts.items())}.
Each pass is a time series on 130 channels. Made with the rail3D
physical-optics simulator for reservoir-computing hardware tests.

> **Caveat.** This is simulation, not measurement, of ONE setup that has not
> been optimised: a single 60 GHz horn at {g['horn_incidence_deg']:.0f}° from vertical, no
> metasurface. The 130 windows simply tile the plane (not optimised).

> **This set is deliberately easy to see.** Every defect sits on the gauge
> shoulder, the one part of the rail head this geometry sees well (s0 =
> {rng_['s0_mm'][0]:.0f}–{rng_['s0_mm'][1]:.0f} mm; §4), and sizes vary only within narrow ranges.

## 1. The setup

![setup](setup_diagram.png)

- **Source:** {g['horn']} horn, {g['frequency_GHz']:.2f} GHz (wavelength {g['wavelength_mm']:g} mm),
  {g['horn_distance_mm']:.1f} mm from the rail top, aimed from {g['horn_incidence_deg']:.0f}° off vertical on the
  gauge (+x) side.
- **Rail:** {g['rail_segment_mm']:.0f} mm of rail head modelled as a perfect mirror; the defect
  is cut into its surface.
- **Detectors:** the reflected light rises {g['metasurface_plane']['z_mm']:.0f} mm to where a metasurface
  would sit (none here), travels 100 mm further, and lands on **130 windows**
  ({config.DET_SIZE[0]:.1f} × {config.DET_SIZE[1]:.1f} mm, a 13 × 10 tiling covering 92% of the
  150 × 75 mm plane) at z = {g['detector_plane_z_mm']:.0f} mm.
- **The pass:** the sensor stays put and the rail moves under it (along y);
  one snapshot every **1 mm of travel** (wear: 2 mm).

## 2. Files

| file | what it is |
|---|---|
| `{OUT_NAME}_clean.csv`, `_low.csv`, `_high.csv` | **the data**: identical rows, three noise levels (§6) |
| `{OUT_NAME}_*.npz` | the same, as NumPy arrays (`detectors` (rows, 130), `pass_id`, `y_mm`, …) |
| `{OUT_NAME}_labels.json` | every pass's parameters, a written description, pose, units, window centres, definitions |
| **`{OUT_NAME}_pass_list.csv`** / **`.md`** | **one line per pass** (the full version of §4's table): CSV to index, Markdown to read |
| `load_passover.py` | loads a CSV + JSON into a Python `sensor_data` dictionary (§7) |
| `{OUT_NAME}_overview.png` | one pass of each subtype, all 130 windows |
| `fields/` | optional raw fields, see §7 |

## 3. The CSV: one row = one snapshot of all 130 windows

![layout](data_layout.png)

{n_rows} rows + 1 header. **All rows with the same `pass_id` form one pass,
in time order** (`frame` 0, 1, 2, …).

| column | meaning |
|---|---|
| `pass_id` | which pass, e.g. `crack_oblique_017` (subtype + number) |
| `subtype` | intact, crack_transverse, crack_oblique, crack_longitudinal, dent, wear, shell |
| `class` | intact, crack, dent, wear, shell (the three crack subtypes share `crack`) |
| `class_index` | {', '.join(f'{k} = {v}' for k, v in labels['class_index'].items())} |
| `subtype_index` | {', '.join(f'{k} = {v}' for k, v in labels['subtype_index'].items())} |
| `frame` | snapshot number within the pass, 0 … 80 (wear 0 … 60) |
| `y_mm` | defect position along the rail relative to the sensor centre line, −40 → +40 mm. **Wear:** where the worn stretch *starts*, −60 → +60 mm (rail behind it is worn) |
| `depth_under_beam_mm` | defect depth directly under the beam centre at that moment (0 = nothing under the beam) |
| `d001` … `d130` | the 130 window readings (relative power, §5) |

**Window numbering** (x-major): `d(10·ix + iy + 1)` with ix = 0…12 from −x to
+x (across the rail; +x is the horn's side) and iy = 0…9 from −y to +y. So
d001–d010 are the column at x = {labels['detectors']['centres_mm_x_y'][0][0]:.0f} mm, d121–d130 the column at
x = {labels['detectors']['centres_mm_x_y'][-1][0]:+.0f} mm. Every centre is in `{OUT_NAME}_labels.json` →
`detectors.centres_mm_x_y` (entry k is d(k+1)).

## 4. The passes

| subtype | subtype_index | passes |
|---|---|---|
{count_rows}

**Parameter ranges** (each pass draws uniformly within them):

| subtype | ranges |
|---|---|
| cracks (all three) | group of 3 parallel lines; width {rng_['CRACK_WIDTH_RANGE'][0]:g}–{rng_['CRACK_WIDTH_RANGE'][1]:g} mm, length {rng_['CRACK_LENGTH_RANGE'][0]:g}–{rng_['CRACK_LENGTH_RANGE'][1]:g} mm, depth {rng_['CRACK_DEPTH_RANGE'][0]:g}–{rng_['CRACK_DEPTH_RANGE'][1]:g} mm; transverse = across the rail (90°), longitudinal = along it (0°), oblique = ±{rng_['oblique_abs_theta_deg'][0]:g}–{rng_['oblique_abs_theta_deg'][1]:g}° |
| dent | depth {rng_['DENT_DEPTH_RANGE'][0]:g}–{rng_['DENT_DEPTH_RANGE'][1]:g} mm, footprint {rng_['DENT_FOOTPRINT_S'][0]:g}–{rng_['DENT_FOOTPRINT_S'][1]:g} mm (FWHM) across and along, one pit |
| wear | gauge-corner depth {rng_['WEAR_DEPTH_RANGE'][0]:g}–{rng_['WEAR_DEPTH_RANGE'][1]:g} mm, cut at {rng_['WEAR_GAUGE_ANGLE_RANGE'][0]:g}–{rng_['WEAR_GAUGE_ANGLE_RANGE'][1]:g}° from vertical; half also have crown wear; stretch 300–900 mm long |
| shell | depth {rng_['SHELL_DEPTH_RANGE'][0]:g}–{rng_['SHELL_DEPTH_RANGE'][1]:g} mm, ragged 8–20 mm patch |
| all but wear | centred at s0 = {rng_['s0_mm'][0]:g}–{rng_['s0_mm'][1]:g} mm (wear: {rng_['wear_centre_s0_mm']}) |

`s0` = position across the rail head measured along its curved surface (the
gauge shoulder; `x` = the same point as a plain across-rail coordinate). Pose
= how the rail sat in that pass (roll / sideways / vertical, mm). Peak % = the
largest change of any window from its own intact level during the pass.

**A snapshot of the pass list** (the first passes of each subtype; **all
{len(labels['passes'])} are in `{OUT_NAME}_pass_list.md`, and in `{OUT_NAME}_pass_list.csv`
for indexing**):

{chr(10).join(_md_table(snap))}

## 5. Units and time

- **Relative power:** 1.0 = the average intact reading over all 130 windows and
  the 5 intact passes. Windows differ a lot in brightness (intact level
  {lvl.min():.2f}–{lvl.max():.2f}; per window in `labels.json` → `intact_window_level`).
  Dividing a window by its own intact level gives "change vs intact".
- **Time:** every pass moves the same way, so `t = (y_mm − first y_mm) / speed`.
  At 20 m/s, 1 mm = 50 µs (an 80 mm pass is 4 ms, 81 snapshots, 20 kHz).
  The readings are smooth at 1 mm spacing, so interpolating to another rate is
  accurate (`load_passover.resample`).

## 6. Noise

Independently per window and per snapshot: `reading × (1 + a·n₁) + b·n₂`, with
n₁, n₂ standard normal and b in units of the mean intact reading.

| file | a (proportional) | b (constant) |
|---|---|---|
{noise}

Noise is drawn per 1 mm snapshot; to control it after resampling, resample
the clean file and add your own.

## 7. Loading the data

```python
from load_passover import load_passover, to_time, resample

sensor_data = load_passover(".", noise="low")     # finds passover130_*.csv itself
p = sensor_data["passes"]["crack_oblique_017"]
p["detectors"]          # (81, 130): rows = snapshots, columns = d001..d130
p["subtype"], p["y_mm"], p["depth_under_beam_mm"], p["description"], p["parameters"]
t_new, det_new = resample(p, speed_m_s=20.0, rate_hz=100e3)
```

The CSVs are large (one row per snapshot, 130 values), so for speed load the
`.npz` instead: `z = np.load("{OUT_NAME}_low.npz")`, then `z["detectors"]`
(rows × 130) with `z["pass_id"]`, `z["y_mm"]`, `z["subtype_index"]`,
`z["frame"]` per row. Or with pandas: `pd.read_csv(...)`, then
`.groupby("pass_id")`.

**Raw fields for other detector layouts.** {fields_txt}

## 8. Things to know when reading the traces

- **Intact passes are flat:** uniform rail, so all snapshots are identical
  apart from noise. Intact levels still differ between passes, because the
  rail sits slightly differently each time; that is the realistic nuisance.
- **Passes start and end partly lit:** the beam covers most of the 120 mm of
  rail modelled, so traces do not begin at the intact level.
- **Wear is a slow step:** the sensor enters a long worn stretch, so readings
  drift to a new worn level and stay there.
- **Many windows are dim:** with no metasurface the reflected light is
  concentrated in part of the plane, and the dimmest windows carry little
  signal (most visible in the high-noise set).

Generated by `rail3D/passover_big.py` (geometry tag {g['geometry_tag']}).
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def plot_overview(overview, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(overview), figsize=(3.0 * len(overview), 4.6), squeeze=False)
    for ax, (st, (pid, y, rel)) in zip(axs[0], overview.items()):
        im = ax.imshow(np.log2(np.clip(rel.T, 1 / 8, 8)), aspect="auto", origin="lower", cmap="RdBu_r",
                       vmin=-3, vmax=3, extent=(y[0], y[-1], 0.5, rel.shape[1] + 0.5))
        ax.set_title(pid, fontsize=9)
        ax.set_xlabel("y_mm (wear: leading edge)", fontsize=8)
        ax.tick_params(labelsize=7)
    axs[0][0].set_ylabel("window d001..d130 (x-major)", fontsize=8)
    cb = fig.colorbar(im, ax=axs[0].tolist(), fraction=0.02)
    cb.set_label("log2(power / that window's intact level), clean", fontsize=8)
    fig.suptitle("One pass per subtype: all 130 windows over the pass", fontsize=11)
    fig.savefig(png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    ap.add_argument("--profile", default="laptop", choices=list(config.PROFILES))
    ap.add_argument("--smoke", action="store_true",
                    help="1 pass per subtype, every 10th frame, into passover130_smoke/")
    ap.add_argument("--assemble-only", action="store_true")
    ap.add_argument("--fields", action="store_true",
                    help="also write frame-matched fields at both planes (~2.7 GB for the full set)")
    a = ap.parse_args()
    out = Path(a.out) if a.out else config.GENERATED_DIR / (OUT_NAME + ("_smoke" if a.smoke else ""))
    out.mkdir(parents=True, exist_ok=True)
    specs = pass_list()
    if a.smoke:
        specs = [s for s in specs if s["pass_id"].endswith("_000")]
    step = 10.0 if a.smoke else 1.0

    def frames(s):                       # frames a pass will solve (wear's count does not depend on L)
        return len(po.frame_positions(s["class"], {"L": 600.0}, step))
    n_frames = sum(frames(s) for s in specs)
    print(f"[rail3d] passover_big: {len(specs)} passes, ~{n_frames} frames, geometry "
          f"{data3d.geometry_tag()} -> {out}", flush=True)
    if not a.assemble_only:
        if sys.platform == "win32":             # no idle sleep while this runs (passover.py)
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        device = config.get_device(a.profile)
        prof = config.PROFILES[a.profile]
        batch = max(1, min(prof.batch_meshes, 32))
        chunk = max(2048, prof.chunk_faces)
        section = sections.load_reference_section()
        n_fine = mesh3d.default_arc_count(section, config.MESH_DS)
        n_coarse = mesh3d.default_arc_count(section, config.OCCLUDER_DS)
        t0, done = time.time(), 0
        todo = [s for s in specs if not (out / "passes" / f"{s['pass_id']}.npz").exists()]
        print(f"  {len(specs) - len(todo)} passes already done, {len(todo)} to go "
              f"(device {device}, {batch} frames per solve)", flush=True)
        for k, s in enumerate(todo):
            nf = run_pass(s, section, n_fine, n_coarse, device, chunk, batch, out, step)
            done += nf or 0
            el = time.time() - t0
            left = sum(frames(t) for t in todo[k + 1:])
            print(f"  {s['pass_id']}: {nf} frames | {done / el:.2f} frames/s | "
                  f"~{left / max(done / el, 1e-9) / 3600:.1f} h left", flush=True)
    assemble(out, a.fields)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
