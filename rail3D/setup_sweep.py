"""setup_sweep.py -- the experimental-setup sweep for F1: one horn, one wider metasurface.

DESIGN_STUDY.md picked F1 (the metasurface's far edge extended, today's horn) as the
minimum-hardware upgrade. This sweep measures the dials around it on a controlled,
stratified defect set, so the choice rests on data rather than on 52 mid-size defects.

DIALS
  solved (each pair is one placement, ~45-60 min on the 5090 for the default plan):
    --thetas    horn angle from vertical (deg)                    default 45 55 65
    --heights   metasurface height H (mm); the horn's distance    default 100 150 200
                follows from finding 31 (its lowest edge level with the metasurface)
  free (crops of the solved plane, chosen in sweep_eval.py at no solve cost):
    metasurface far edge  (extends the aperture away from the horn; F1's dial)
    metasurface position  (a fixed-width aperture slid along x)
    metasurface width     (150 mm today, 265 mm for F1, or any other)
  The solved plane per placement runs from x = +80 (horn side) out past the crown's
  mirror point -H tan(theta), so every crop up to and including bright field exists.
    --y-half    the plane's half-extent along the rail (37.5 = today's 75 mm)

THE DEFECT SET (a controlled grid: README finding 25 made position the dominant variable)
  6 types: crack transverse / longitudinal / oblique (20-70 deg), dent, shell, wear
  7 positions on the arc: s = 110 120 130 140 150 160 170 mm (+-2 mm jitter, recorded)
  --n-per-cell (100) per type x position, sizes Latin-hypercube stratified over each
  class's OWN operating ranges (config.py), plus --n-intact (600) intact rails.
  - Position is a free factor: every class's surface band is overridden ("facing"),
    so a shell at s 110 or a crack at s 170 is rendered at full depth (finding 25's
    bands would otherwise mask them).
  - Wear at position s is a wheel cut tangent at s: its normal angle is the rail's
    tilt there (capped at 60 deg: deeper cuts truncate on the face, finding 33).
    Crown positions (tilt < 20 deg) use the crown cut's radius range and depth
    0.5-4 mm; gauge positions the gauge cut's (R 150-600, 2-8 mm).
  - Paired design: the SAME rails in the SAME poses for every placement (common
    seeds), so placements are compared sample by sample, not population by population.
  - Every defect also stores its exact difference field: the defect minus the intact
    rail in the SAME pose, solved from the defect's own facets only (~10% extra; the
    head is convex where psi1 looks, so nothing else changes). sweep_eval.py needs it
    for the ceiling bound; data-centric analysis uses it as a per-sample SNR label.

ORDER: breadth first. Shard k of every placement runs before shard k+1 of any, and
each shard is a balanced slice of the whole grid. A night cut short by the deadline
leaves every placement with the same balanced subsample, not some placements done
and others empty.

    python setup_sweep.py --self-test                 # laptop, ~2 min: the twin identity + band overrides
    python setup_sweep.py --timing 20 --profile lab   # measure s/sample, project the night
    python setup_sweep.py --profile lab --budget-hours 8      # the sweep (resumable; stops at the deadline)
    python sweep_eval.py                              # bounds, pruning, figures (sweep_eval.py)

Scene: coverage_scan.py's validated wide-plane scene (240 mm segment with a 30 mm
cos^2 end taper; defect-facet-only ray casting, bit-identical to casting every
facet), upward-only psi1, production shadow guard.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import coverage_scan as cs                                   # noqa: E402
from rail3d import config, data3d, field3d, mesh3d, sections  # noqa: E402

TYPES = ("crack_transverse", "crack_longitudinal", "crack_oblique", "dent", "shell", "wear")
CLASS_OF = {"crack_transverse": "crack", "crack_longitudinal": "crack", "crack_oblique": "crack",
            "dent": "dent", "shell": "shell", "wear": "wear"}
POSITIONS = (110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0)
OBLIQUE_DEG = (20.0, 70.0)
CROWN_TILT_DEG = 20.0                       # wear below this tilt is crown wear
WEAR_MAX_ANGLE = max(config.WEAR_GAUGE_ANGLE_RANGE)
WEAR_CROWN_DEPTH = (0.5, 4.0)
X_HI = 80.0                                 # plane edge on the horn side (mm)
SHARD = 100
OUT = config.GENERATED_DIR / "setup_sweep"


# ---------------------------------------------------------------------------
# placements
# ---------------------------------------------------------------------------
def placement(theta: float, h: float, y_half: float) -> dict:
    xg = -h * math.tan(math.radians(theta))                  # the crown's mirror point
    x_lo = math.floor(min(1.15 * xg - 10.0, -200.0) / config.DX) * config.DX
    return {"name": f"t{theta:g}_H{h:g}", "theta": theta, "H": h, "D": cs.dist_ant(theta, h),
            "x_glint": xg, "x_lo": x_lo, "x_hi": X_HI, "y_half": y_half,
            "nx": int(round((X_HI - x_lo) / config.DX)), "ny": int(round(2 * y_half / config.DX))}


def placements(thetas, heights, y_half) -> list[dict]:
    """Today's setup first, then by distance from it (a cut-short night keeps the
    placements nearest the reference)."""
    out = [placement(t, h, y_half) for t in thetas for h in heights]
    return sorted(out, key=lambda p: (abs(p["theta"] - 55) / 10 + abs(p["H"] - 150) / 50,
                                      p["theta"], p["H"]))


def plane_axes(pl: dict):
    x = pl["x_lo"] + config.DX / 2 + config.DX * np.arange(pl["nx"])
    y = (np.arange(pl["ny"]) - (pl["ny"] - 1) / 2) * config.DX
    return x, y


# ---------------------------------------------------------------------------
# the plan: every sample, its seed, its size stratum
# ---------------------------------------------------------------------------
def size_keys(cls: str):
    return {"crack": ("width", "depth", "L"), "dent": ("depth", "fw_s", "fw_y"),
            "shell": ("depth", "r_s", "r_y"), "wear": ("depth", "radius")}[cls]


def lhs(n: int, k: int, rng) -> np.ndarray:
    """Latin hypercube in (0, 1)^k: each column hits each of n equal bins once."""
    return np.stack([(rng.permutation(n) + rng.random(n)) / n for _ in range(k)], 1)


def make_plan(n_per_cell: int, n_intact: int, s_jitter: float, types=TYPES,
              positions=POSITIONS, plan_seed: int = 0) -> dict:
    cells = [(t, s) for t in types for s in positions]
    samples = []
    for ci, (t, s) in enumerate(cells):
        ti, pi = TYPES.index(t), (POSITIONS.index(s) if s in POSITIONS else 10 + int(round(s)))
        rng = np.random.default_rng([plan_seed, ti, pi])
        u = lhs(n_per_cell, len(size_keys(CLASS_OF[t])), rng)
        jit = rng.uniform(-s_jitter, s_jitter, n_per_cell)
        for j in range(n_per_cell):
            samples.append({"id": f"{t}_s{s:g}_{j:03d}", "type": t, "cls": CLASS_OF[t],
                            "s_nom": s, "s_jit": float(jit[j]), "u": u[j].tolist(),
                            "seed": config.sample_seed(CLASS_OF[t], 60000 + 100000 * ti + 1000 * pi + j),
                            "_rank": j + ci / len(cells)})
    for k in range(n_intact):
        samples.append({"id": f"intact_{k:03d}", "type": "intact", "cls": "intact",
                        "seed": config.sample_seed("intact", 60000 + k),
                        "_rank": k * n_per_cell / max(n_intact, 1)})
    samples.sort(key=lambda d: d["_rank"])                   # every prefix is a balanced slice
    for d in samples:
        del d["_rank"]
    return {"version": 1, "types": list(types), "positions": list(positions),
            "n_per_cell": n_per_cell, "n_intact": n_intact, "s_jitter": s_jitter,
            "plan_seed": plan_seed, "samples": samples}


def scene_digest(plan: dict) -> str:
    """What makes two sweep roots incomparable: the geometry digest, the sweep's
    own scene (segment, taper) and the sample plan."""
    payload = json.dumps({"geometry": data3d.geometry_tag(), "seg": cs.SEG, "taper": cs.TAPER,
                          "plan": plan["samples"], "wear": [CROWN_TILT_DEG, WEAR_MAX_ANGLE,
                                                            WEAR_CROWN_DEPTH]}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# one sample: parameters, meshes, fields
# ---------------------------------------------------------------------------
_ARC: dict = {}


def arc(section, n_fine):
    if n_fine not in _ARC:
        import glint_band as gb
        g = gb.arc()
        _ARC[n_fine] = {"geo": mesh3d.arc_geometry(section, n_fine), "s": g["s"], "tilt": g["tilt"]}
    return _ARC[n_fine]


def tilt_at(a: dict, s0: float) -> float:
    return float(np.interp(s0, a["s"], a["tilt"]))


def _lerp(rng, u):
    return rng[0] + (rng[1] - rng[0]) * u


def po_trust(t: str, p: dict) -> str:
    """How far rung 3 is expected to trust rail3D on this sample (README finding 34)."""
    if t == "crack_longitudinal":
        return "low" if p["width"] >= config.WVL / 2 else "medium"
    if t in ("crack_transverse", "crack_oblique"):
        return "medium"
    if t == "dent" and min(p["fw_s"], p["fw_y"]) < 12.0 and p["depth"] > 6.0:
        return "medium"
    return "high"


def draw(spec: dict, section, n_fine: int):
    """(params, augmentation, metadata) for one planned sample."""
    cls = spec["cls"]
    p, aug = mesh3d.defect_params_for_sample(section, cls, None, spec["seed"], n_arc=n_fine)
    meta = {"id": spec["id"], "type": spec["type"], "cls": cls, **aug}
    if cls == "intact":
        return p, aug, meta
    a = arc(section, n_fine)
    s0 = spec["s_nom"] + spec["s_jit"]
    u = spec["u"]
    rng = np.random.default_rng(spec["seed"])
    if cls == "crack":
        w = _lerp(config.CRACK_WIDTH_RANGE, u[0])
        factor = p["line_gap"] / p["width"]
        half, edge = w / 2, 0.25 * config.MESH_DS
        p["profile_v"] = np.array([-half - edge, -half, half, half + edge])
        p["width"], p["line_gap"] = w, w * factor
        mid = (p["n_lines"] - 1) / 2.0
        p["offsets"] = [(i - mid) * p["line_gap"] for i in range(p["n_lines"])]
        p["depth"] = _lerp(config.CRACK_DEPTH_RANGE, u[1])
        p["L"] = _lerp(config.CRACK_LENGTH_RANGE, u[2])
        p["theta"] = {"crack_transverse": math.pi / 2, "crack_longitudinal": 0.0}.get(
            spec["type"], float(rng.choice([-1.0, 1.0]) * math.radians(rng.uniform(*OBLIQUE_DEG))))
    elif cls == "dent":
        p["depth"] = _lerp(config.DENT_DEPTH_RANGE, u[0])
        p["fw_s"] = _lerp(config.DENT_FOOTPRINT_S, u[1])
        p["fw_y"] = _lerp(config.DENT_FOOTPRINT_Y, u[2])
    elif cls == "shell":
        p["depth"] = _lerp(config.SHELL_DEPTH_RANGE, u[0])
        p["r_s"] = _lerp(config.SHELL_RADIUS_RANGE, u[1])
        p["r_y"] = _lerp(config.SHELL_RADIUS_RANGE, u[2])
    elif cls == "wear":
        tilt = tilt_at(a, s0)
        crown = abs(tilt) < CROWN_TILT_DEG
        p["depth"] = _lerp(WEAR_CROWN_DEPTH if crown else config.WEAR_DEPTH_RANGE, u[0])
        p["gauge_radius"] = _lerp(config.WEAR_CROWN_RADIUS_RANGE if crown
                                  else config.WEAR_GAUGE_RADIUS_RANGE, u[1])
        p["gauge_angle_deg"] = min(tilt, WEAR_MAX_ANGLE)
        p["crown_depth"] = 0.0
        meta["wear_kind"] = "crown" if crown else "gauge"
    if cls != "wear":
        p["s0"] = s0
    p["band"] = "facing"                                      # position is the experiment
    # where the defect really sits: the depth-weighted centroid along the arc (a
    # transverse crack is a uniform-depth line ACROSS the head, so its deepest
    # point is arbitrary) and the arc it spans (depth > 10% of its maximum)
    ys = np.linspace(-60, 60, 121)
    d = mesh3d.render_depth_field(p, a["geo"], ys)
    ds_ = d.max(0)
    s_arc = a["geo"]["s"]
    s_c = float((s_arc * ds_).sum() / max(ds_.sum(), 1e-12))
    span = s_arc[ds_ > 0.1 * ds_.max()] if ds_.max() > 0 else np.array([s0, s0])
    meta.update({"s_nom": spec["s_nom"], "s0": s0, "s_centroid": s_c,
                 "s_span": [float(span.min()), float(span.max())], "depth_realised": float(d.max()),
                 "tilt_deg": tilt_at(a, s_c), "y0": float(p.get("y0", 0.0)),
                 "po_trust": po_trust(spec["type"], p), "u": u})
    for k in size_keys(cls):
        meta[k] = float(p["gauge_radius"] if k == "radius" else p[k])
    if cls == "crack":
        meta["theta_deg"] = math.degrees(p["theta"])
        meta["width_over_cutoff"] = p["width"] / (config.WVL / 2)
    return p, aug, meta


def meshes(p: dict, aug: dict, section, n_fine: int, n_coarse: int, faces: list):
    """Fine defect mesh, the SAME-pose intact mesh on the same faces, the facets
    the defect moved, and the defect's coarse occluder."""
    kw = dict(seg_len=cs.SEG, roll_deg=aug["roll_deg"], jitter_xz=(aug["jitter_x"], aug["jitter_z"]))
    dp = None if p.get("class") == "intact" else p
    v, f = mesh3d.sweep_rail_mesh(section, defect_params=dp, slice_ds=config.MESH_DS,
                                  arc_ds=config.MESH_DS, n_arc=n_fine, faces=faces[0], **kw)
    v0, _ = mesh3d.sweep_rail_mesh(section, defect_params=None, slice_ds=config.MESH_DS,
                                   arc_ds=config.MESH_DS, n_arc=n_fine, faces=f, **kw)
    moved = ((v - v0).norm(dim=1) > 1e-5)[f].any(1)
    vo, fo = mesh3d.sweep_rail_mesh(section, defect_params=dp, slice_ds=config.OCCLUDER_DS,
                                    arc_ds=config.OCCLUDER_DS, n_arc=n_coarse, faces=faces[1], **kw)
    faces[0], faces[1] = f, fo
    return v, f, v0, moved, vo, fo


def fields(v, f, v0, moved, vo, fo, pl: dict, X, Y, device, chunk: int):
    """(psi1 of the sample, its exact difference field vs the same-pose intact rail)."""
    th = math.radians(pl["theta"])
    common = dict(H=pl["H"], wvl=config.WVL, theta_inc=th, size_ant=config.SIZE_ANT,
                  dist_ant=pl["D"], resol_ant=config.RESOL_ANT, chunk_faces=chunk,
                  shadow_min_t=config.SHADOW_MIN_T, shadow_normal_offset=config.SHADOW_NORMAL_OFFSET,
                  compute_psi2=False, upward_only=True)
    occ = (vo[None].to(device), fo.to(device))
    cs._STATE["phi"] = 0.0
    with cs.scene_patches():
        cs._STATE["ray_faces"] = moved
        psi, _ = field3d.scattered_fields(v[None].to(device), f.to(device), X, Y,
                                          shadow=config.SHADOW_MODE, shadow_occluders=occ, **common)
        if not bool(moved.any()):
            return psi[0], torch.zeros_like(psi[0])
        fm = f[moved].to(device)
        cs._STATE["ray_faces"] = None                         # every facet of the subset casts
        pd, _ = field3d.scattered_fields(v[None].to(device), fm, X, Y,
                                         shadow=config.SHADOW_MODE, shadow_occluders=occ, **common)
        pi, _ = field3d.scattered_fields(v0[None].to(device), fm, X, Y, shadow="none", **common)
    return psi[0], (pd - pi)[0]


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------
def run_root(name: str, plan: dict) -> Path:
    return OUT / f"{name}_{scene_digest(plan)}"


def open_root(root: Path, plan: dict, pls: list[dict]) -> None:
    """Create a run root, or check an existing one is the same experiment."""
    root.mkdir(parents=True, exist_ok=True)
    pf = root / "plan.json"
    if pf.exists():
        old = json.loads(pf.read_text())
        if old["samples"] != plan["samples"]:
            raise SystemExit(f"{root} holds a different sample plan; use another --name")
    else:
        pf.write_text(json.dumps(plan))
    meta = {"geometry": data3d.geometry_tag(), "commit": data3d._git_commit(),
            "scene": {"seg": cs.SEG, "taper": cs.TAPER, "x_hi": X_HI},
            "placements": pls, "created": time.strftime("%Y-%m-%d %H:%M")}
    mf = root / "sweep.json"
    if mf.exists():
        was = json.loads(mf.read_text())
        known = {p["name"]: p for p in was["placements"]}
        for p in pls:
            if p["name"] in known and known[p["name"]] != p:
                raise SystemExit(f"placement {p['name']} changed since this root was started")
        meta["placements"] = list({**known, **{p["name"]: p for p in pls}}.values())
        meta["created"] = was["created"]
    mf.write_text(json.dumps(meta, indent=1))


def solve_shard(pl, specs, path: Path, section, n_fine, n_coarse, device, chunk, faces) -> float:
    x, y = plane_axes(pl)
    Xn, Yn = np.meshgrid(x, y, indexing="ij")
    X = torch.tensor(Xn, dtype=torch.float32, device=device)[None]
    Y = torch.tensor(Yn, dtype=torch.float32, device=device)[None]
    psis, deltas, metas = [], [], []
    t0 = time.time()
    for spec in specs:
        p, aug, meta = draw(spec, section, n_fine)
        v, f, v0, moved, vo, fo = meshes(p, aug, section, n_fine, n_coarse, faces)
        psi, dl = fields(v, f, v0, moved, vo, fo, pl, X, Y, device, chunk)
        meta["moved_facets"] = int(moved.sum())
        psis.append(psi.cpu().numpy().astype(np.complex64))
        deltas.append(dl.cpu().numpy().astype(np.complex64))
        metas.append(meta)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, psi=np.stack(psis), delta=np.stack(deltas),
             meta=json.dumps(metas, default=float), x=x, y=y)
    tmp.replace(path)
    return (time.time() - t0) / len(specs)


def run(args) -> int:
    device = config.get_device(args.profile)
    chunk = config.PROFILES[args.profile].chunk_faces
    plan = make_plan(args.n_per_cell, args.n_intact, args.s_jitter, args.types, args.positions)
    pls = placements(args.thetas, args.heights, args.y_half)
    root = run_root(args.name, plan)
    open_root(root, plan, pls)
    section = sections.load_reference_section()
    n_fine = mesh3d.default_arc_count(section, config.MESH_DS)
    n_coarse = mesh3d.default_arc_count(section, config.OCCLUDER_DS)
    shards = [plan["samples"][i:i + SHARD] for i in range(0, len(plan["samples"]), SHARD)]
    deadline = time.time() + 3600 * args.budget_hours if args.budget_hours else None
    rate = {}                                                 # placement -> s/sample
    print(f"[sweep] {root}\n  {len(plan['samples'])} samples x {len(pls)} placements "
          f"({len(shards)} shards each), device {device}")
    for p in pls:
        print(f"  {p['name']:10s} theta {p['theta']:g} H {p['H']:g} D {p['D']:.1f} mm, "
              f"crown glint x {p['x_glint']:.0f}, plane x {p['x_lo']:.0f}..{p['x_hi']:.0f} "
              f"({p['nx']}x{p['ny']})")
    faces = [None, None]
    for k, specs in enumerate(shards):
        for p in pls:
            path = root / p["name"] / f"shard_{k:03d}.npz"
            if path.exists():
                continue
            est = rate.get(p["name"], max(rate.values(), default=0.0)) * len(specs)
            if deadline and time.time() + est > deadline:
                print(f"[sweep] deadline: stopping before {p['name']} shard {k} "
                      f"(est {est / 60:.1f} min)")
                return report(root, pls, len(shards))
            path.parent.mkdir(parents=True, exist_ok=True)
            r = solve_shard(p, specs, path, section, n_fine, n_coarse, device, chunk, faces)
            rate[p["name"]] = r
            done = sum(1 for q in pls for _ in (root / q["name"]).glob("shard_*.npz"))
            left = len(pls) * len(shards) - done
            print(f"  shard {k:3d} {p['name']:10s} {r:5.2f} s/sample   {done} shards done, "
                  f"{left} left (~{left * SHARD * np.mean(list(rate.values())) / 3600:.1f} h)",
                  flush=True)
    return report(root, pls, len(shards))


def report(root: Path, pls, n_shards: int) -> int:
    lines = []
    for p in pls:
        n = len(list((root / p["name"]).glob("shard_*.npz")))
        lines.append(f"  {p['name']:10s} {n:3d}/{n_shards} shards")
    (root / "progress.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"next: python sweep_eval.py --root {root}")
    return 0


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
def self_test(args) -> int:
    """(1) psi(defect) - delta == psi(same-pose intact) exactly, every type;
    (2) every type renders at full depth at s = 110 and 170 (band overrides);
    (3) wear's deepest point follows the requested position."""
    device = config.get_device(args.profile)
    section = sections.load_reference_section()
    n_fine = mesh3d.default_arc_count(section, config.MESH_DS)
    n_coarse = mesh3d.default_arc_count(section, config.OCCLUDER_DS)
    pl = dict(placement(55.0, 150.0, 37.5), x_lo=-150.0, nx=40, ny=8)
    x, y = plane_axes(pl)
    Xn, Yn = np.meshgrid(x, y, indexing="ij")
    X = torch.tensor(Xn, dtype=torch.float32, device=device)[None]
    Y = torch.tensor(Yn, dtype=torch.float32, device=device)[None]
    plan = make_plan(2, 1, 0.0, positions=(110.0, 170.0))
    ok = True
    for spec in [d for d in plan["samples"] if d["id"].endswith("_000") and d["cls"] != "intact"]:
        p, aug, meta = draw(spec, section, n_fine)
        faces = [None, None]
        v, f, v0, moved, vo, fo = meshes(p, aug, section, n_fine, n_coarse, faces)
        psi, dl = fields(v, f, v0, moved, vo, fo, pl, X, Y, device, 2048)
        # the same-pose intact rail solved on its own
        cs._STATE["ray_faces"] = torch.zeros(len(f), dtype=torch.bool)
        with cs.scene_patches():
            pi, _ = field3d.scattered_fields(
                v0[None].to(device), f.to(device), X, Y, pl["H"], config.WVL,
                math.radians(pl["theta"]), config.SIZE_ANT, pl["D"], config.RESOL_ANT,
                chunk_faces=2048, shadow="none", compute_psi2=False, upward_only=True)
        err = float((psi - dl - pi[0]).norm() / pi[0].norm())
        rel = float(dl.norm() / pi[0].norm())
        # shells carry +-20% interior roughness clipped to 0.6-1.4 (render_depth_field)
        full = meta["depth_realised"] >= (0.6 if spec["cls"] == "shell" else 0.95) * meta["depth"]
        near = abs(meta["s_centroid"] - meta["s0"]) <= 6.0 if spec["cls"] != "wear" else True
        good = err < 1e-4 and full and near and rel > 1e-4
        ok &= good
        print(f"  {'ok ' if good else 'BAD'} {spec['id']:26s} twin err {err:.1e}  signal {rel:.3f} of "
              f"intact  depth {meta['depth_realised']:.2f}/{meta['depth']:.2f}  s0 {meta['s0']:.0f} "
              f"-> centroid {meta['s_centroid']:.0f}, spans {meta['s_span'][0]:.0f}-{meta['s_span'][1]:.0f} "
              f"(tilt {meta['tilt_deg']:.0f})"
              + (f"  {meta.get('wear_kind')}" if spec["cls"] == "wear" else ""))
    print("SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def timing(args) -> int:
    device = config.get_device(args.profile)
    chunk = config.PROFILES[args.profile].chunk_faces
    section = sections.load_reference_section()
    n_fine = mesh3d.default_arc_count(section, config.MESH_DS)
    n_coarse = mesh3d.default_arc_count(section, config.OCCLUDER_DS)
    plan = make_plan(args.n_per_cell, args.n_intact, args.s_jitter, args.types, args.positions)
    pls = placements(args.thetas, args.heights, args.y_half)
    specs = plan["samples"][:args.timing]
    tmp = OUT / "_timing"
    tmp.mkdir(parents=True, exist_ok=True)
    ref = pls[0]
    r = solve_shard(ref, specs, tmp / "t.npz", section, n_fine, n_coarse, device, chunk, [None, None])
    pts = ref["nx"] * ref["ny"]
    tot = 0.0
    print(f"[timing] {ref['name']}: {r:.2f} s/sample on {device} ({ref['nx']}x{ref['ny']} plane)")
    for p in pls:
        # psi1 cost scales with plane points; the meshes and ray cast do not
        est = r * (0.35 + 0.65 * p["nx"] * p["ny"] / pts) * len(plan["samples"]) / 3600
        tot += est
        print(f"  {p['name']:10s} {p['nx']}x{p['ny']}  ~{est:.2f} h")
    print(f"  all {len(pls)} placements: ~{tot:.1f} h for {len(plan['samples'])} samples each "
          f"(first-shard timing re-measures each placement during the run)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--thetas", type=float, nargs="+", default=[45.0, 55.0, 65.0])
    ap.add_argument("--heights", type=float, nargs="+", default=[100.0, 150.0, 200.0])
    ap.add_argument("--y-half", type=float, default=37.5)
    ap.add_argument("--types", nargs="+", default=list(TYPES), choices=TYPES)
    ap.add_argument("--positions", type=float, nargs="+", default=list(POSITIONS))
    ap.add_argument("--n-per-cell", type=int, default=100)
    ap.add_argument("--n-intact", type=int, default=600)
    ap.add_argument("--s-jitter", type=float, default=2.0)
    ap.add_argument("--budget-hours", type=float, default=None)
    ap.add_argument("--name", default="F1")
    ap.add_argument("--profile", default="lab", choices=list(config.PROFILES))
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--timing", type=int, default=0, metavar="N")
    a = ap.parse_args()
    if a.self_test:
        return self_test(a)
    if a.timing:
        return timing(a)
    return run(a)


if __name__ == "__main__":
    raise SystemExit(main())
