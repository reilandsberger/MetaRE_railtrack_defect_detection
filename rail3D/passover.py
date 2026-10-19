"""Detector outputs over time as the train passes over sample defects.

    python passover.py --smoke              # 2 short passes -> data/generated/passover_smoke/
    python passover.py                      # the full set (resumable, one file per pass)
    python passover.py --assemble-only      # rebuild the npz / CSV / JSON from finished passes

Example data for a reservoir-computing hardware test. The rail is uniform along
y, so the train moving over a defect is the defect moving through the fixed
beam: each pass takes ONE dataset sample (its seed fixes the defect shape and
the rail pose) and slides it along y, one field solve per frame, through the
generator's own code path (generate_dataset_3d.sample_meshes / solve_fields:
λ/8 mesh, ray-cast shadowing, upward-only metasurface input).

  defect passes   y0 from -40 to +40 mm in 1 mm steps (81 frames)
  wear passes     the train ENTERS a worn stretch: the stretch's leading edge
                  sweeps across the 120 mm segment, +60 -> -60 mm, 2 mm steps
                  (61 frames; the wear envelope is a slow edge, see WEAR_STEP)
  intact passes   uniform rail: every frame is the same field (1 solve), only
                  the noise differs frame to frame

The rail pose (roll, lateral and vertical offset) is held FIXED within a pass:
an 80 mm pass at 20 m/s lasts 4 ms, far shorter than wheelset motion, while a
random pose per frame would flicker more than most defect signals. Each pass
has its own pose (its dataset seed's).

Readout: NO metasurface (README finding 24: the trained phase mask is currently
worse than none), free propagation 100 mm to the detector plane, 8 fixed,
untrained windows on a 4 x 2 grid (DET8). The raw field at the metasurface
plane is kept per frame, and the 130-window dense readout too, so any other
layout or a metasurface can be applied later without re-solving.

Outputs in --out:
  passes/<pass_id>.npz          raw per-pass fields (resume unit)
  passover_<noise>.npz / .csv   detector traces, noise in {clean, low, high}
  passover_labels.json          per-pass parameters + descriptions, per-frame
                                position and depth-under-beam, all definitions
  passover_traces.png           one panel per pass, the 8 traces
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import generate_dataset_3d as gen
from rail3d import config, data3d, mesh3d, optics3d, sections

CLASSES = ("intact",) + tuple(config.CLASS_NAMES)
N_PASS = 5
HALF_SPAN, STEP = 40.0, 1.0           # defect passes, mm
WEAR_HALF, WEAR_STEP = 60.0, 2.0      # wear: leading-edge sweep across the segment
SCAN = 400                            # dataset seeds scanned when picking samples
# 8 fixed windows, a 4 x 2 grid at the cell centres of the 150 x 75 mm plane
DET8 = torch.tensor([[x, y] for y in (-config.WY / 4, config.WY / 4)
                     for x in np.linspace(-3, 3, 4) * config.WX / 8])
# detector-power noise, per detector per frame, in units of the mean intact power:
# P_noisy = P (1 + mult * n1) + add * n2,   n1, n2 ~ N(0, 1)
NOISE = {"clean": (0.0, 0.0), "low": (0.005, 0.005), "high": (0.02, 0.05)}


# ---------------------------------------------------------------------------
# which samples
# ---------------------------------------------------------------------------
def glint_band(geom) -> tuple[float, float]:
    """Arc range whose specular reflection lands in the aperture (ray optics)."""
    th, D, H = config.THETA_INC, config.DIST_ANT, config.H_MS
    horn = np.array([D * math.sin(th), D * math.cos(th)])
    P = np.stack([geom["x"], geom["z"]], 1)
    N = geom["normal"]
    inc = P - horn
    inc /= np.linalg.norm(inc, axis=1, keepdims=True)
    out = inc - 2 * (inc * N).sum(1, keepdims=True) * N
    with np.errstate(divide="ignore", invalid="ignore"):
        xp = P[:, 0] + (H - P[:, 1]) / out[:, 1] * out[:, 0]
    ok = (out[:, 1] > 0) & ((inc * N).sum(1) < 0) & \
        (np.abs(xp - config.PLANE_X_CENTER) <= config.WX / 2)
    s = geom["s"][ok]
    return float(s.min()), float(s.max())


def pick_passes(section, geom) -> list[dict]:
    """5 passes per class. crack / dent: 3 near the visible band + 2 on the
    crown, where this geometry is nearly blind (README finding 25)."""
    lo, hi = glint_band(geom)
    picks = {c: [] for c in CLASSES}
    picks["intact"] = [(i, "n/a") for i in range(N_PASS)]
    for cls in config.CLASS_NAMES:
        near, far = [], []
        for i in range(SCAN):
            p, _ = mesh3d.defect_params_for_sample(section, cls, None, config.sample_seed(cls, i),
                                                   n_arc=geom["n_arc"])
            s0 = p.get("s0")
            if cls in ("wear", "shell") or lo - 10 <= s0 <= hi + 10:
                near.append(i)
            elif s0 < lo - 20:
                far.append(i)
        if cls in ("crack", "dent"):
            picks[cls] = [(i, "near_visible_band") for i in near[:3]] + \
                         [(i, "crown_outside_band") for i in far[:2]]
        else:
            picks[cls] = [(i, "near_visible_band") for i in near[:N_PASS]]
    # round robin, so a partial run already holds every class
    out = []
    for k in range(N_PASS):
        for cls in CLASSES:
            i, vis = picks[cls][k]
            out.append({"pass_id": f"{cls}_{k}", "class": cls, "dataset_index": i,
                        "seed": config.sample_seed(cls, i), "visibility": vis})
    return out, (lo, hi)


def frame_positions(cls: str, params: dict, step_scale: float = 1.0) -> np.ndarray:
    if cls == "intact":
        return np.zeros(1)
    if cls == "wear":       # y0 such that the leading edge y0 - L/2 runs +60 -> -60
        edge = np.arange(WEAR_HALF, -WEAR_HALF - 1e-9, -WEAR_STEP * step_scale)
        return edge + params["L"] / 2
    return np.arange(-HALF_SPAN, HALF_SPAN + 1e-9, STEP * step_scale)


# ---------------------------------------------------------------------------
# solve
# ---------------------------------------------------------------------------
def run_pass(spec, section, n_fine, n_coarse, device, chunk, out_dir, step_scale):
    path = out_dir / "passes" / f"{spec['pass_id']}.npz"
    if path.exists():
        return "exists"
    cls, i = spec["class"], spec["dataset_index"]
    files = sections.get_dataset_files(cls) if cls in config.DATASET_DIRS else None
    _, _, _, _, meta = gen.sample_meshes(section, cls, i, files, n_fine, n_coarse)
    ys = frame_positions(cls, meta, step_scale)
    # psi2 is exactly 0 at the metasurface while the horn sits wholly above it
    # (upward-only, README finding 31); skip computing it then
    psi2 = config.HORN_LOWER_EDGE_Z < config.H_MS
    psis, t0 = [], time.time()
    faces = [None, None]
    for k, y0 in enumerate(ys):
        v, faces[0], vo, faces[1], _ = gen.sample_meshes(
            section, cls, i, files, n_fine, n_coarse, faces[0], faces[1],
            y0=None if cls == "intact" else y0)
        p1, p2 = gen.solve_fields(v[None].to(device), faces[0], vo[None].to(device), faces[1],
                                  device, chunk, compute_psi2=psi2)
        psis.append((p1 + p2)[0].cpu().numpy().astype(np.complex64))
        if k == 0 or (k + 1) % 10 == 0 or k + 1 == len(ys):
            el = time.time() - t0
            print(f"  {spec['pass_id']}: frame {k + 1}/{len(ys)}  "
                  f"({el / (k + 1):.1f} s/frame, {el / 60:.1f} min)", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, psi=np.stack(psis), y0=ys,
                        meta=json.dumps(meta, default=_jsonable), spec=json.dumps(spec))
    tmp.replace(path)
    return f"{len(ys)} frames in {(time.time() - t0) / 60:.1f} min"


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.tolist()
    return str(o)


# ---------------------------------------------------------------------------
# readout + files
# ---------------------------------------------------------------------------
@torch.no_grad()
def readout(psi: np.ndarray, centers: torch.Tensor) -> np.ndarray:
    det = optics3d.SoftDetector2D(centers=centers.clone().float())
    onn = optics3d.ONN3D(surface="none", noise=False, detector=det).eval()
    _, p = onn(torch.from_numpy(psi), hard=True)
    return p.numpy().astype(np.float64)


def describe(cls, m, geom) -> str:
    if cls == "intact":
        return "intact rail, no defect"
    x0 = float(np.interp(m["s0"], geom["s"], geom["x"])) if m.get("s0") is not None else None
    at = f"centred at arc s0 = {m['s0']:.1f} mm (x = {x0:+.1f} mm across the head; gauge side +x)"
    if cls == "crack":
        th = math.degrees(m["theta"])
        kind = "longitudinal" if abs(th) < 1 else "transverse" if abs(abs(th) - 90) < 1 else "oblique"
        return (f"group of 3 parallel box cracks, {kind} (theta {th:.0f} deg from the rail axis): "
                f"{m['depth']:.2f} mm deep, {m['width']:.2f} mm wide, {m['L']:.1f} mm long, "
                f"lines {m['line_gap']:.2f} mm apart centre to centre; {at}")
    if cls == "dent":
        return (f"dent (Gaussian pit{'s' if len(m['pits']) > 1 else ''}, {len(m['pits'])}): "
                f"{m['depth']:.2f} mm deep, FWHM {m['fw_s']:.1f} mm across x {m['fw_y']:.1f} mm "
                f"along the rail; {at}")
    if cls == "shell":
        return (f"shelling patch (ragged, {len(m['lobes'])} lobe(s)) on the gauge corner: "
                f"{m['depth']:.2f} mm deep, about {2 * m['r_s']:.1f} mm across x "
                f"{2 * m['r_y']:.1f} mm along; {at}")
    crown = (f" plus crown wear {m['crown_depth']:.2f} mm deep (near-flat, R {m['crown_radius']:.0f} mm)"
             if m.get("crown") else ", no crown wear")
    return (f"worn stretch {m['L']:.0f} mm long: gauge-corner wear {m['depth']:.2f} mm deep, "
            f"flattened facet at {m['gauge_angle_deg']:.0f} deg from vertical (R "
            f"{m['gauge_radius']:.0f} mm){crown}; the pass ENTERS the stretch")


def assemble(out_dir: Path, band) -> None:
    section = sections.load_reference_section()
    geom = mesh3d.arc_geometry(section, mesh3d.default_arc_count(section, config.MESH_DS))
    files = sorted((out_dir / "passes").glob("*.npz"))
    if not files:
        print("!! no finished passes")
        return
    order = {c: k for k, c in enumerate(CLASSES)}
    rows = []
    for f in files:
        z = np.load(f)
        spec, meta = json.loads(str(z["spec"])), json.loads(str(z["meta"]))
        rows.append((spec, meta, z["y0"], z["psi"]))
    rows.sort(key=lambda r: (order[r[0]["class"]], r[0]["pass_id"]))

    dense = config.dense_detector_centers()
    reads = []
    for spec, meta, y0, psi in rows:
        d8, d130 = readout(psi, DET8), readout(psi, dense)
        n = 81 if spec["class"] == "intact" else len(y0)     # intact: repeat the one field
        if spec["class"] == "intact":
            y0 = np.arange(-HALF_SPAN, HALF_SPAN + 1e-9, STEP)
            d8, d130 = np.repeat(d8, n, 0), np.repeat(d130, n, 0)
        depth = np.zeros(n)
        if spec["class"] != "intact":
            for k, y in enumerate(y0):
                p, _ = mesh3d.defect_params_for_sample(section, spec["class"], None, spec["seed"],
                                                       n_arc=geom["n_arc"])
                p["y0"] = float(y)
                depth[k] = mesh3d.render_depth_field(p, geom, np.array([0.0])).max()
        reads.append((spec, meta, y0, d8, d130, depth))
    intact = [r for r in reads if r[0]["class"] == "intact"]
    ref = float(np.mean([r[3].mean() for r in intact])) if intact else float(reads[0][3].mean())
    ref130 = float(np.mean([r[4].mean() for r in intact])) if intact else float(reads[0][4].mean())
    # each detector's own intact level (mean over the intact passes), for the change figure
    ref_det = (np.mean([r[3].mean(0) for r in intact], 0) if intact else reads[0][3].mean(0)) / ref

    labels = {
        "about": ("Simulated detector outputs as a train passes over rail-head defects "
                  "(rail3D physical-optics model, NOT measurements). CAVEAT: one specific, "
                  "possibly non-optimal geometry -- single horn at 55 deg, 278.5 mm out, no "
                  "metasurface, 8 fixed untrained detectors."),
        "geometry": {"wavelength_mm": config.WVL, "frequency_GHz": round(299.792458 / config.WVL, 4),
                     "horn": config.HORN_SPEC.get("part"), "horn_incidence_deg": 55.0,
                     "horn_distance_mm": round(config.DIST_ANT, 2),
                     "metasurface_plane": {"z_mm": config.H_MS, "size_mm": [config.WX, config.WY]},
                     "metasurface": "none (free propagation)",
                     "detector_plane_z_mm": config.H_MS + config.LAYER_DISTANCES[-1],
                     "rail_segment_mm": config.SEG_LEN, "geometry_tag": data3d.geometry_tag(),
                     "visible_band_arc_s_mm": [round(band[0], 1), round(band[1], 1)],
                     "visible_band_note": ("arc positions whose mirror reflection reaches the "
                                           "aperture (ray optics); defects far from it scatter "
                                           "only weakly into the detectors (README finding 25)")},
        "detectors": {"count": 8, "window_mm": list(config.DET_SIZE),
                      "centres_mm_x_y": DET8.tolist(),
                      "order": "d1..d8 = row y < 0 then y > 0, each row x from -x to +x",
                      "x_axis": "across the rail, +x = gauge side (the horn side)"},
        "frames": {"axis": ("y_mm = position of the defect relative to the beam centre "
                            "along the rail; time t = (y_end - y) / train speed"),
                   "defect_step_mm": STEP, "wear_step_mm": WEAR_STEP,
                   "sampling_note": ("fields vary smoothly at this step (see the trace "
                                     "spectra); interpolate to the frame rate you need"),
                   "depth_under_beam_mm": "defect depth at y = 0 (the beam centre) per frame"},
        "units": ("relative power: 1.0 = the mean intact detector power (over all 8 detectors "
                  f"and all intact passes; absolute model value {ref:.4e})"),
        "noise": {k: {"multiplicative_std": a, "additive_std": b,
                      "model": "P_noisy = P (1 + mult n1) + add n2, n ~ N(0,1), per detector per frame"}
                  for k, (a, b) in NOISE.items()},
        "pose": "rail roll / lateral / vertical offset fixed within a pass, different per pass",
        "peak_change_vs_intact": ("per pass: max over frames and detectors of |P - P_intact| / "
                                  "P_intact, P_intact = that detector's mean over the intact passes "
                                  "(clean values; includes the pose difference between passes)"),
        "segment_note": ("the modelled rail is 120 mm long and the beam lights most of it, so a "
                         "defect at y = +-40 mm is still partly seen: passes start and end lit"),
        "class_index": {c: k for k, c in enumerate(CLASSES)},
        "passes": [],
    }
    rng = np.random.default_rng(config.SEED)
    sets = {k: [] for k in NOISE}
    for spec, meta, y0, d8, d130, depth in reads:
        cls = spec["class"]
        base = d8 / ref
        labels["passes"].append({
            **spec, "class_index": CLASSES.index(cls), "n_frames": int(len(y0)),
            "description": describe(cls, meta, geom),
            "parameters": {k: v for k, v in meta.items()
                           if k not in ("offsets", "profile_v", "profile_d", "fourier", "lobes",
                                        "pits", "rough_seed", "band", "csv_index", "seed", "class",
                                        "roll_deg", "jitter_x", "jitter_z")},
            "pose": {"roll_deg": meta.get("roll_deg"), "lateral_mm": meta.get("jitter_x"),
                     "vertical_mm": meta.get("jitter_z")},
            "y_mm_range": [float(y0[0]), float(y0[-1])],
            "peak_change_vs_intact": float((np.abs(base - ref_det) / ref_det).max())})
        for name, (a, b) in NOISE.items():
            noisy = base * (1 + a * rng.standard_normal(base.shape)) + b * rng.standard_normal(base.shape)
            sets[name].append((spec, y0, noisy, depth, d130 / ref130))

    for name, rs in sets.items():
        y = np.concatenate([r[1] for r in rs])
        np.savez_compressed(
            out_dir / f"passover_{name}.npz",
            detectors=np.concatenate([r[2] for r in rs]).astype(np.float32),
            detectors_dense130_clean=np.concatenate([r[4] for r in rs]).astype(np.float32),
            y_mm=y, depth_under_beam_mm=np.concatenate([r[3] for r in rs]),
            pass_id=np.concatenate([[r[0]["pass_id"]] * len(r[1]) for r in rs]),
            class_index=np.concatenate([[CLASSES.index(r[0]["class"])] * len(r[1]) for r in rs]),
            frame=np.concatenate([np.arange(len(r[1])) for r in rs]))
        with open(out_dir / f"passover_{name}.csv", "w", encoding="utf-8", newline="") as fh:
            fh.write("pass_id,class,class_index,frame,y_mm,depth_under_beam_mm,"
                     + ",".join(f"d{k + 1}" for k in range(8)) + "\n")
            for spec, y0, det, depth, _ in rs:
                for k in range(len(y0)):
                    fh.write(f"{spec['pass_id']},{spec['class']},{CLASSES.index(spec['class'])},{k},"
                             f"{y0[k]:.2f},{depth[k]:.4f}," + ",".join(f"{v:.6f}" for v in det[k]) + "\n")
    np.savez_compressed(out_dir / "passover_fields.npz",
                        **{r[0]["pass_id"]: r[3] for r in rows},
                        **{r[0]["pass_id"] + "_y_mm": r[2] for r in rows})
    (out_dir / "passover_labels.json").write_text(json.dumps(labels, indent=2, default=_jsonable),
                                                  encoding="utf-8")
    plot(sets["clean"], sets["high"], labels, out_dir / "passover_traces.png")
    print(f"  assembled {len(rows)} passes -> {out_dir}")


def plot(clean, noisy, labels, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    by = {}
    for (spec, y0, det, depth, _), (_, _, dn, _, _) in zip(clean, noisy):
        by.setdefault(spec["class"], []).append((spec, y0, det, dn, depth))
    ncol = max(len(v) for v in by.values())
    fig, axs = plt.subplots(len(by), ncol, figsize=(3.4 * ncol, 2.5 * len(by)), squeeze=False)
    desc = {p["pass_id"]: p for p in labels["passes"]}
    cmap = plt.get_cmap("tab10")
    for r, cls in enumerate(c for c in CLASSES if c in by):
        for c in range(ncol):
            ax = axs[r][c]
            if c >= len(by[cls]):
                ax.axis("off")
                continue
            spec, y0, det, dn, depth = by[cls][c]
            for k in range(8):
                ax.plot(y0, dn[:, k], color=cmap(k), lw=0.5, alpha=0.35)
                ax.plot(y0, det[:, k], color=cmap(k), lw=1.2, label=f"d{k + 1}" if r == c == 0 else None)
            if cls != "intact":
                ax2 = ax.twinx()
                ax2.fill_between(y0, 0, depth, color="0.6", alpha=0.25, lw=0)
                ax2.set_yticks([])
            if cls == "wear":
                ax.invert_xaxis()
            d = desc[spec["pass_id"]]
            ax.set_title(f"{spec['pass_id']} ({d['visibility'].replace('_', ' ')})\n"
                         f"peak change {100 * d['peak_change_vs_intact']:.0f}% vs intact", fontsize=7.5)
            ax.tick_params(labelsize=7)
            ax.set_xlabel("y (mm), wear: leading-edge position" if cls == "wear" else "defect y (mm)",
                          fontsize=7)
    axs[0][0].legend(fontsize=6, ncol=2)
    fig.suptitle("Detector traces over a pass (bold: clean; faint: high-noise set; grey: defect depth "
                 "under the beam centre). Units: mean intact detector power = 1", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(png, dpi=140)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    ap.add_argument("--profile", default="laptop", choices=list(config.PROFILES))
    ap.add_argument("--smoke", action="store_true",
                    help="2 passes (intact + one crack), every 10th frame, into passover_smoke/")
    ap.add_argument("--assemble-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out) if a.out else config.GENERATED_DIR / ("passover_smoke" if a.smoke else "passover")
    out.mkdir(parents=True, exist_ok=True)
    section = sections.load_reference_section()
    n_fine = mesh3d.default_arc_count(section, config.MESH_DS)
    n_coarse = mesh3d.default_arc_count(section, config.OCCLUDER_DS)
    geom = mesh3d.arc_geometry(section, n_fine)
    specs, band = pick_passes(section, geom)
    if a.smoke:
        specs = [s for s in specs if s["pass_id"] in ("intact_0", "crack_0")]
    print(f"[rail3d] passover: {len(specs)} passes, geometry {data3d.geometry_tag()}, "
          f"visible band s {band[0]:.1f}-{band[1]:.1f} mm -> {out}")
    if not a.assemble_only:
        if sys.platform == "win32":
            # ask Windows not to idle-sleep while this process runs (ES_CONTINUOUS |
            # ES_SYSTEM_REQUIRED); released automatically when the process exits,
            # no power setting is changed, a closed lid still sleeps
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        device = config.get_device(a.profile)
        chunk = max(2048, config.PROFILES[a.profile].chunk_faces)
        for spec in specs:
            print(f"{spec['pass_id']} (dataset #{spec['dataset_index']}, {spec['visibility']}): "
                  + run_pass(spec, section, n_fine, n_coarse, device, chunk, out,
                             10.0 if a.smoke else 1.0), flush=True)
    assemble(out, band)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
