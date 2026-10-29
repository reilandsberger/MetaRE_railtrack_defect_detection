"""coverage_scan.py -- Tier 1 of the experimental-design study (DESIGN_STUDY.md).

Field-level coverage of the priority arc s = 110-170 mm (crown/running band
through the gauge corner) for alternative horn placements and metasurface
apertures, with NO metasurface training in the loop.

Per horn placement, one wide observation plane at z = H (x -300..+100 mm,
y +-37.5) is solved for
    - 16 intact rails in random poses (roll +-2 deg, 4 mm jitter: the nuisance),
    - cracks (transverse, longitudinal, 45 deg oblique) and single dents of
      FIXED mid-range size (passover_big.py's narrowed ranges) placed on an
      s0 grid 110, 115, ..., 170 mm at y0 = 0, each in its own random pose.
Every candidate aperture is then a CROP of that plane (free): the current
150 mm one, wider ones that extend the edge away from the horn, and a
bright-field one centred on the crown's specular reflection.

Scene deviations from production, all deliberate and applied to every
configuration alike:
    - the rail segment is 240 mm with a 30 mm cos^2 current taper at each end
      (production: 120 mm, hard cut). A horn offset ALONG the rail (azimuth
      phi) puts the intact rail's specular point tens of mm along y; a hard
      120 mm cut would delete it and fake a dark field.
    - azimuth: the horn sits at azimuth phi around the vertical (phi = 0 is
      production, in the x-z plane). Implemented by rotating the rail and the
      plane about z by -phi, so the solver's horn stays in its x-z plane.
    - the horn's lowest edge stays level with the plane (finding 31), so
      DIST_ANT follows theta: D = (H + (A/2 + wall) sin th) / cos th.
    - psi1 only (upward-only input; psi0 = psi2 = 0 with the horn above).

Score (per sample, per aperture): the plane is read by every detector-sized
window (4 x 3 px = 10 x 7.5 mm, stride 1 px); a window's z-score is its power
change vs the intact mean over sqrt(intact pose variance + receiver noise^2).
The sample's score is the max |z| over windows: what the best placed detector
could see. Receiver noise is ABSOLUTE: eta x the mean window power of the
current setup (eta = 1% by default), identical for every configuration, so a
dark aperture cannot win by dividing by ~0. "Detected" = score above the
largest leave-one-out intact score (a ~1/16 false-alarm rate).

    python coverage_scan.py                    # solve every horn (resumable)
    python coverage_scan.py --horns h55 h45    # a subset
    python coverage_scan.py --smoke            # 2 intact + 2 defects, h55 only
    python coverage_scan.py --analyze          # figures + coverage_scan.json
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rail3d import config, field3d, mesh3d, sections   # noqa: E402
import passover_big as pb                               # noqa: E402

H = config.H_MS
PLANE_X = (-300.0, 100.0)
NY = config.NY
SEG, TAPER = 240.0, 30.0
S0_GRID = np.arange(110.0, 170.1, 5.0)
VARIANTS = [("crack_transverse", "crack"), ("crack_longitudinal", "crack"),
            ("crack_oblique", "crack"), ("dent", "dent")]
N_INTACT = 16
OBLIQUE = math.radians(45.0)
HORNS = {   # name: (theta_deg from vertical, azimuth phi_deg around the rail, plane height H)
    "h55": (55.0, 0.0, H),     # production
    "h45": (45.0, 0.0, H),
    "h35": (35.0, 0.0, H),
    "h65": (65.0, 0.0, H),
    "h55_az30": (55.0, 30.0, H),
    "h55_az60": (55.0, 60.0, H),
    "h55_H100": (55.0, 0.0, 100.0),
}
WIN = (4, 3)                   # detector-sized window, pixels (10 x 7.5 mm)
OUT = config.GENERATED_DIR / "design_scan"
FIG = config.FIGURE_DIR / "design_2026-10-29"


def dist_ant(theta_deg: float, h: float = H) -> float:
    """Aperture-centre distance that keeps the horn's lowest edge at z = h."""
    th = math.radians(theta_deg)
    return (h + (config.SIZE_ANT[0] / 2 + config.HORN_WALL) * math.sin(th)) / math.cos(th)


def plane_axes():
    x = np.arange(PLANE_X[0] + config.DX / 2, PLANE_X[1], config.DX)
    y = (np.arange(NY) - (NY - 1) / 2) * config.DX
    return x, y


def rotate(xy: torch.Tensor, phi: float, inverse: bool = False) -> torch.Tensor:
    """Rail frame -> solver frame (rotation about z by -phi); inverse=True back."""
    c, s = math.cos(phi), math.sin(phi)
    if inverse:
        s = -s
    x, y = xy[..., 0], xy[..., 1]
    return torch.stack([c * x + s * y, -s * x + c * y], -1)


# ---------------------------------------------------------------------------
# the end taper: weight each facet's area by its along-rail position
# ---------------------------------------------------------------------------
_STATE = {"phi": 0.0}
_surface_geometry = field3d.surface_geometry


def _tapered_surface_geometry(v, f):
    center, area, normal = _surface_geometry(v, f)
    y = rotate(center[..., :2], _STATE["phi"], inverse=True)[..., 1].abs()
    flat = SEG / 2 - TAPER
    u = ((y - flat) / TAPER).clamp(0, 1)
    return center, area * torch.cos(0.5 * math.pi * u) ** 2, normal




# ---------------------------------------------------------------------------
# rays only from the defect's own facets. The intact head is convex wherever
# psi1 looks (n_z > 0, facing the horn), and a segment from a convex surface
# to an outside source never re-enters the body, so no facet off the defect
# can be shadowed (a depression lies INSIDE the intact hull). Same answer as
# casting every facet -- checked against it in --check-shadow -- at a
# fraction of the O(rays x tris) cost that dominates this scan.
# ---------------------------------------------------------------------------
_raycast = field3d.raycast_shadow_mask


def _raycast_defect_only(v_occ, f_occ, center, src_point, ray_mask=None, **kw):
    m = _STATE.get("ray_faces")
    if m is not None:
        m = m.to(center.device)
        ray_mask = m if ray_mask is None else ray_mask & m
    return _raycast(v_occ, f_occ, center, src_point, ray_mask=ray_mask, **kw)


@contextlib.contextmanager
def scene_patches():
    """The taper and the defect-only ray casting, applied ONLY while this scan solves
    (importing the module leaves field3d untouched)."""
    old = field3d.surface_geometry, field3d.raycast_shadow_mask
    field3d.surface_geometry = _tapered_surface_geometry
    field3d.raycast_shadow_mask = _raycast_defect_only
    try:
        yield
    finally:
        field3d.surface_geometry, field3d.raycast_shadow_mask = old


# ---------------------------------------------------------------------------
# samples
# ---------------------------------------------------------------------------
def sample_list(smoke: bool = False) -> list[dict]:
    out = [{"id": f"intact_{k:02d}", "subtype": "intact", "class": "intact", "s0": None,
            "seed": config.sample_seed("intact", 7000 + k)} for k in range(N_INTACT)]
    for vi, (st, cls) in enumerate(VARIANTS):
        for j, s0 in enumerate(S0_GRID):
            out.append({"id": f"{st}_s{int(s0)}", "subtype": st, "class": cls, "s0": float(s0),
                        "seed": config.sample_seed(cls, 7000 + 100 * vi + j)})
    if smoke:
        out = out[:2] + [d for d in out if d["id"] in ("crack_transverse_s150", "dent_s120")]
    return out


TWINS = ("crack_transverse_s150", "crack_transverse_s120", "dent_s150", "dent_s120")


def twin_specs() -> list[dict]:
    """Pose twins: the intact rail in EXACTLY the pose of a defect sample, so a figure can
    show the defect's own signal (defect minus same-pose intact) without pose motion."""
    by = {s["id"]: s for s in sample_list()}
    return [{"id": f"twin_{i}", "subtype": "pose_twin", "class": "intact", "s0": None,
             "seed": by[i]["seed"], "twin_of": by[i]} for i in TWINS]


def draw(spec, section, n_fine):
    if "twin_of" in spec:
        _, aug = draw(spec["twin_of"], section, n_fine)
        p, _ = pb.draw_defect({"class": "intact", "seed": spec["seed"], "subtype": "intact"},
                              section, n_fine)
        return p, aug
    p, aug = pb.draw_defect(spec, section, n_fine)
    if spec["class"] != "intact":
        p["s0"] = spec["s0"]
        p["y0"] = 0.0
    if spec["subtype"] == "crack_oblique":
        p["theta"] = OBLIQUE
    return p, aug


def meshes(spec, section, n_fine, n_coarse, faces):
    """Fine + occluder meshes, and the fine facets the defect moved (ray sources)."""
    p, aug = draw(spec, section, n_fine)
    kw = dict(defect_params=p, seg_len=SEG, roll_deg=aug["roll_deg"],
              jitter_xz=(aug["jitter_x"], aug["jitter_z"]))
    v, f = mesh3d.sweep_rail_mesh(section, slice_ds=config.MESH_DS, arc_ds=config.MESH_DS,
                                  n_arc=n_fine, faces=faces[0], **kw)
    v0, _ = mesh3d.sweep_rail_mesh(section, slice_ds=config.MESH_DS, arc_ds=config.MESH_DS,
                                   n_arc=n_fine, faces=f, **dict(kw, defect_params=None))
    _STATE["ray_faces"] = ((v - v0).norm(dim=1) > 1e-5)[f].any(1)
    vo, fo = mesh3d.sweep_rail_mesh(section, slice_ds=config.OCCLUDER_DS,
                                    arc_ds=config.OCCLUDER_DS, n_arc=n_coarse,
                                    faces=faces[1], **kw)
    faces[0], faces[1] = f, fo
    meta = {k: v_ for k, v_ in p.items() if k not in ("profile_v", "profile_d", "csv_dev", "csv_s")}
    meta.update(aug)
    return v, f, vo, fo, meta


def solve(name, specs, device, chunk, full_shadow: bool = False):
    th_deg, phi_deg, h = HORNS[name]
    th, phi = math.radians(th_deg), math.radians(phi_deg)
    D = dist_ant(th_deg, h)
    _STATE["phi"] = phi
    section = sections.load_reference_section()
    n_fine = mesh3d.default_arc_count(section, config.MESH_DS)
    n_coarse = mesh3d.default_arc_count(section, config.OCCLUDER_DS)
    x, y = plane_axes()
    Xr, Yr = np.meshgrid(x, y, indexing="ij")
    P = rotate(torch.tensor(np.stack([Xr, Yr], -1), dtype=torch.float32), phi)
    X, Y = P[None, ..., 0].to(device), P[None, ..., 1].to(device)
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    faces = [None, None]
    todo = [s for s in specs if not (out / f"{s['id']}.npz").exists()]
    print(f"[coverage] {name}: theta {th_deg} phi {phi_deg} H {h} D {D:.1f} mm -- "
          f"{len(specs) - len(todo)} done, {len(todo)} to go", flush=True)
    t0 = time.time()
    for k, spec in enumerate(todo):
        v, f, vo, fo, meta = meshes(spec, section, n_fine, n_coarse, faces)
        if full_shadow:
            _STATE["ray_faces"] = None
        v = v.clone()
        vo = vo.clone()
        v[:, :2] = rotate(v[:, :2], phi)
        vo[:, :2] = rotate(vo[:, :2], phi)
        with scene_patches():
            psi1, _ = field3d.scattered_fields(
                v[None].to(device), f.to(device), X, Y, h, config.WVL, th, config.SIZE_ANT, D,
                config.RESOL_ANT, chunk_faces=chunk, shadow=config.SHADOW_MODE,
                shadow_occluders=(vo[None].to(device), fo.to(device)),
                shadow_min_t=config.SHADOW_MIN_T, shadow_normal_offset=config.SHADOW_NORMAL_OFFSET,
                compute_psi2=False, upward_only=True)
        tmp = out / f"{spec['id']}.tmp.npz"
        np.savez_compressed(tmp, psi=psi1[0].cpu().numpy().astype(np.complex64),
                            spec=json.dumps(spec), meta=json.dumps(meta, default=float))
        tmp.replace(out / f"{spec['id']}.npz")
        dt = (time.time() - t0) / (k + 1)
        print(f"  {spec['id']:28s} {dt:5.1f} s/sample, ~{dt * (len(todo) - k - 1) / 60:5.1f} min left",
              flush=True)
    return time.time() - t0


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------
def crown_mirror_x(theta_deg: float, phi_deg: float = 0.0, h: float = H) -> float:
    """Where the flat crown's mirror reflection lands: tan(theta_eff) = tan(theta) cos(phi)."""
    return -h * math.tan(math.radians(theta_deg)) * math.cos(math.radians(phi_deg))


def apertures(theta_deg: float, h: float = H, phi_deg: float = 0.0) -> dict:
    """Candidate apertures as x ranges (mm) on the plane, all 75 mm along y."""
    spec_x = crown_mirror_x(theta_deg, phi_deg, h)
    out = {"current [-75,75]": (-75.0, 75.0),
           "far edge -110": (-110.0, 75.0),
           "far edge -150": (-150.0, 75.0),
           "far edge -190": (-190.0, 75.0)}
    if spec_x - 75 >= PLANE_X[0]:
        out["bright-field crown"] = (spec_x - 75.0, spec_x + 75.0)
    return out


def load(name):
    return load_dir(OUT / name)


def load_dir(d):
    rows = []
    for p in sorted(d.glob("*.npz")):
        if p.name.endswith(".tmp.npz"):
            continue
        z = np.load(p)
        rows.append((json.loads(str(z["spec"])), np.abs(z["psi"]) ** 2))
    return rows


def window_power(I: np.ndarray) -> np.ndarray:
    """(n, nx, ny) intensity -> (n, nx-wx+1, ny-wy+1) detector-window powers."""
    wx, wy = WIN
    c = np.cumsum(np.cumsum(np.pad(I, ((0, 0), (1, 0), (1, 0))), 1), 2)
    return (c[:, wx:, wy:] - c[:, :-wx, wy:] - c[:, wx:, :-wy] + c[:, :-wx, :-wy]) * config.DX ** 2


def scores(rows, xr, noise_abs):
    """Max-window |z| per sample inside x range xr; intact scored leave-one-out."""
    x, _ = plane_axes()
    m = (x >= xr[0]) & (x <= xr[1])
    I = np.stack([r[1][m] for r in rows])
    W = window_power(I)
    intact = np.array([r[0]["class"] == "intact" for r in rows])
    Wi = W[intact]
    mu, var = Wi.mean(0), Wi.var(0, ddof=1)
    z = np.abs(W - mu) / np.sqrt(var + noise_abs ** 2)
    sc = z.reshape(len(rows), -1).max(1)
    loo = []
    n = intact.sum()
    for k in range(n):
        rest = np.delete(Wi, k, 0)
        zk = np.abs(Wi[k] - rest.mean(0)) / np.sqrt(rest.var(0, ddof=1) + noise_abs ** 2)
        loo.append(zk.max())
    sc[intact] = loo
    return sc, float(max(loo)), float(Wi.mean())


ETAS = (0.001, 0.01, 0.1)
# The mid-size defects here clear the bare threshold almost everywhere (each sample gets its
# own best window: an oracle layout), so binary recall saturates near 0.9-1.0. Coverage is
# ranked by "robust recall": score at least MARGIN x the threshold (6 dB), and by the margin.
MARGIN = 2.0


def all_scores():
    """{(horn, aperture, eta): {sample id: score / threshold}} plus bookkeeping."""
    data = {n: load(n) for n in HORNS if (OUT / n).exists()}
    full = len(sample_list())
    data = {n: r for n, r in data.items() if len(r) == full}     # finished horns only
    if "h55" not in data:
        raise SystemExit("h55 (the production horn) must be solved first: it sets the noise unit")
    ref = scores(data["h55"], (-75.0, 75.0), 0.0)[2]          # mean window power, current setup
    ns, info = {}, {}
    for n, rows in data.items():
        th, phi, h = HORNS[n]
        for an, xr in apertures(th, h, phi).items():
            for eta in ETAS:
                sc, thr, pw = scores(rows, xr, eta * ref)
                ns[(n, an, eta)] = {r[0]["id"]: float(v / thr) for r, v in zip(rows, sc)}
                info[(n, an)] = {"x_range": xr, "power_dB": 10 * math.log10(pw / ref),
                                 "bright": xr[0] <= crown_mirror_x(th, phi, h) <= xr[1],
                                 "n_intact": int(sum(r[0]["class"] == "intact" for r in rows))}
    specs = {r[0]["id"]: r[0] for r in data["h55"]}
    return ns, info, specs, ref


def fused(ns, keys, specs):
    """OR-fusion of channels (time-multiplexed horns and/or separate metasurfaces): a sample's
    score is its largest per-channel score/threshold; the fused threshold is re-calibrated on
    the intact samples (same ~1/16 false-alarm rate as one channel)."""
    ids = set.intersection(*[set(ns[k]) for k in keys])
    f = {i: max(ns[k][i] for k in keys) for i in ids}
    thr = max(v for i, v in f.items() if specs[i]["class"] == "intact")
    return {i: v / thr for i, v in f.items()}


OVERLAP_MAX = 12.0    # mm: two metasurfaces may share at most ~one detector window of x


def pair_kind(ka, kb, info):
    """What hardware a two-channel pair needs, by PHYSICAL metasurface (plane height + x range;
    the same aperture name at another theta or H is another surface). None when the pair is not
    buildable: the same surface twice, or two surfaces overlapping in x (one would sit in the
    other's light, or the pair is just the wider aperture with extra steps)."""
    (na, aa), (nb, ab) = ka, kb
    ha, hb = HORNS[na][2], HORNS[nb][2]
    xa, xb = info[ka]["x_range"], info[kb]["x_range"]
    same_ms = ha == hb and tuple(xa) == tuple(xb)
    if na == nb:
        if same_ms:
            return None
    elif same_ms:
        return "2 horns, 1 MS"
    if min(xa[1], xb[1]) - max(xa[0], xb[0]) > OVERLAP_MAX:
        return None
    return "1 horn, 2 MS" if na == nb else "2 horns, 2 MS"


def recall(f, specs, lo=None, hi=None, m=MARGIN):
    v = [f[i] > m for i in f if specs[i]["class"] != "intact"
         and (lo is None or specs[i]["s0"] >= lo) and (hi is None or specs[i]["s0"] <= hi)]
    return float(np.mean(v)) if v else float("nan")


def per_s0(f, specs, m=MARGIN):
    """Fraction of the 4 defect variants detected (score > m x threshold) at each s0."""
    return np.array([np.mean([f[i] > m for i in f if specs[i]["s0"] == s0]) if
                     any(specs[i]["s0"] == s0 for i in f) else np.nan for s0 in S0_GRID])


def per_s0_db(f, specs):
    """Geometric-mean detection margin over the 4 variants at each s0, in dB (20 log10)."""
    return np.array([20 * np.mean([np.log10(max(f[i], 1e-3)) for i in f if specs[i]["s0"] == s0]) if
                     any(specs[i]["s0"] == s0 for i in f) else np.nan for s0 in S0_GRID])


def margin_db(f, specs, lo=None, hi=None):
    v = [20 * np.log10(max(f[i], 1e-3)) for i in f if specs[i]["class"] != "intact"
         and (lo is None or specs[i]["s0"] >= lo) and (hi is None or specs[i]["s0"] <= hi)]
    return float(np.mean(v)) if v else float("nan")


def analyze():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import glint_band as gb
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold"})
    FIG.mkdir(parents=True, exist_ok=True)
    ns, info, specs, ref = all_scores()
    eta = 0.01
    single = {(n, an): {i: v for i, v in ns[(n, an, eta)].items()} for (n, an) in info}
    rows = []
    for (n, an), f in single.items():
        rows.append({"horn": n, "aperture": an, "channels": [[n, an]],
                     "recall": recall(f, specs), "recall_crown_110_140": recall(f, specs, 110, 140),
                     "recall_corner_145_170": recall(f, specs, 145, 170),
                     "recall_at_threshold": recall(f, specs, m=1.0),
                     "margin_dB": margin_db(f, specs), "margin_crown_dB": margin_db(f, specs, 110, 140),
                     "margin_corner_dB": margin_db(f, specs, 145, 170),
                     "recall_eta": {str(e): recall(ns[(n, an, e)], specs) for e in ETAS},
                     "power_dB": info[(n, an)]["power_dB"], "bright": info[(n, an)]["bright"],
                     "per_s0": per_s0(f, specs).tolist(), "per_s0_dB": per_s0_db(f, specs).tolist()})
    rows.sort(key=lambda r: (-r["recall"], -r["margin_dB"]))
    # pairs: two channels, OR-fused
    keys = list(single)
    pairs = []
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            ka, kb = keys[a], keys[b]
            kind = pair_kind(ka, kb, info)
            if kind is None:
                continue
            f = fused(ns, [(ka[0], ka[1], eta), (kb[0], kb[1], eta)], specs)
            pairs.append({"channels": [list(ka), list(kb)], "kind": kind, "recall": recall(f, specs),
                          "recall_crown_110_140": recall(f, specs, 110, 140),
                          "recall_corner_145_170": recall(f, specs, 145, 170),
                          "margin_dB": margin_db(f, specs), "margin_crown_dB": margin_db(f, specs, 110, 140),
                          "margin_corner_dB": margin_db(f, specs, 145, 170),
                          "per_s0": per_s0(f, specs).tolist(), "per_s0_dB": per_s0_db(f, specs).tolist()})
    pairs.sort(key=lambda r: (-r["recall"], -r["margin_dB"]))
    res = {"noise_unit_window_power": ref, "eta_main": eta, "robust_margin": MARGIN,
           "singles": rows, "pairs_top": pairs[:25],
           "pairs_best_by_kind": {k: [p for p in pairs if p["kind"] == k][:5]
                                  for k in ("1 horn, 2 MS", "2 horns, 1 MS", "2 horns, 2 MS")},
           "n_defects": int(sum(s["class"] != "intact" for s in specs.values())),
           "n_intact": int(sum(s["class"] == "intact" for s in specs.values()))}
    js = OUT / "coverage_scan.json"
    js.write_text(json.dumps(res, indent=1))
    print(f"wrote {js}")
    for r in rows:
        print(f"  {r['horn']:9s} | {r['aperture']:20s} robust recall {r['recall']:.2f} (crown {r['recall_crown_110_140']:.2f}, "
              f"corner {r['recall_corner_145_170']:.2f}; at threshold {r['recall_at_threshold']:.2f}) margin "
              f"{r['margin_dB']:.1f} dB (crown {r['margin_crown_dB']:.1f}, corner {r['margin_corner_dB']:.1f})  eta 0.1/1/10%: "
              + " / ".join(f"{v:.2f}" for v in r["recall_eta"].values())
              + f"  power {r['power_dB']:+.1f} dB{'  BRIGHT' if r['bright'] else ''}")
    print("  best pairs:")
    for r in pairs[:10]:
        print(f"    {r['recall']:.2f} (crown {r['recall_crown_110_140']:.2f}, corner {r['recall_corner_145_170']:.2f}) "
              f"{r['margin_dB']:4.1f} dB"
              f"  {r['kind']:14s} " + "  +  ".join(f"{a}|{b}" for a, b in r["channels"]))

    fig_evidence(plt, gb, ns, specs, ref)
    fig_scan(plt, rows)
    fig_combos(plt, gb, rows, pairs, single, specs)


def _band(gb, **kw):
    r = gb.evaluate(gb.Setup(**kw))
    return r.intervals


def fig_evidence(plt, gb, ns, specs, ref):
    rows = load("h55")
    x, y = plane_axes()
    ext = [x[0] - 1.25, x[-1] + 1.25, y[0] - 1.25, y[-1] + 1.25]
    byid = {r[0]["id"]: r[1] for r in rows}
    twins = {r[0]["twin_of"]["id"]: r[1] for r in load_dir(OUT / "_twins" / "h55")}
    Ii = np.mean([I for s, I in rows if s["class"] == "intact"], 0)
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(4, 2, width_ratios=[2.3, 1.0], height_ratios=[1, 1, 1, 1], hspace=0.55, wspace=0.18)
    ax = fig.add_subplot(gs[0, 0])
    dB = 10 * np.log10(Ii / Ii.max())
    im = ax.imshow(dB.T, origin="lower", extent=ext, cmap="magma", vmin=-10, vmax=0, aspect="auto")
    for an, xr in apertures(55.0).items():
        ls = "-" if an.startswith("current") else ("--" if "far" in an else ":")
        c = "tab:red" if "bright" in an else "w"
        ax.add_patch(plt.Rectangle((xr[0], -37.5), xr[1] - xr[0], 75, fill=False, ec=c, ls=ls, lw=1.6))
    ax.text(0, 30, "today's metasurface", color="w", ha="center", fontsize=9)
    for xe in (-110, -150, -190):
        ax.text(xe + 2, -33, f"{xe}", color="w", ha="left", fontsize=8)
    ax.text(-150, 28, "far-edge options", color="w", ha="center", fontsize=9,
            bbox=dict(fc="k", alpha=0.35, lw=0, pad=1.5))
    ax.text(-250, -32, "bright-field crown aperture", color="tab:red", ha="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="dB vs peak", pad=0.01)
    ax.set_ylabel("y (mm)")
    ax.set_title("A. Mean intact |ψ1|² at z = 150, production horn: within ~7 dB everywhere; "
                 "the crown glint peaks near x ≈ −200", loc="left")

    ax = fig.add_subplot(gs[0, 1])
    prof = Ii[:, NY // 2 - 2:NY // 2 + 2].mean(1)
    ax.plot(x, 10 * np.log10(prof / prof.max()), color="k")
    ax.axvspan(-75, 75, color="tab:orange", alpha=0.2, label="today")
    ax.axvspan(-289, -139, color="tab:red", alpha=0.1, label="bright-field aperture")
    ax.set_xlabel("x (mm), y ≈ 0")
    ax.set_ylabel("dB")
    ax.legend(fontsize=8)
    ax.set_title("B. Cut through A at y ≈ 0")
    ax.grid(alpha=0.3)

    maps = []
    for sid, lab in (("crack_transverse_s150", "transverse crack, s0 = 150 (INSIDE today's glint band)"),
                     ("crack_transverse_s120", "transverse crack, s0 = 120 (crown, outside it)"),
                     ("dent_s120", "dent, s0 = 120 (crown)")):
        if sid in twins:
            maps.append(((byid[sid] - twins[sid]) / Ii.max(), lab + " − same-pose intact"))
        else:
            maps.append(((byid[sid] - Ii) / Ii.max(), lab + " − mean intact (pose NOT removed)"))
    v = max(np.abs(m).max() for m, _ in maps)
    for k, (dI, lab) in enumerate(maps):
        ax = fig.add_subplot(gs[1 + k, 0])
        im = ax.imshow(dI.T, origin="lower", extent=ext, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
        ax.add_patch(plt.Rectangle((-75, -37.5), 150, 75, fill=False, ec="tab:orange", lw=1.6))
        ax.add_patch(plt.Rectangle((-289, -37.5), 150, 75, fill=False, ec="tab:red", ls=":", lw=1.4))
        ax.set_title(f"{'CDE'[k]}. ΔI: {lab}", loc="left", fontsize=10)
        ax.set_ylabel("y (mm)")
        fig.colorbar(im, ax=ax, pad=0.01, label="rel. intact peak")
    ax.set_xlabel("x on the plane (mm);  orange = today's aperture, red dotted = bright-field crown aperture")

    ax = fig.add_subplot(gs[1:, 1])
    f = ns[("h55", "current [-75,75]", 0.01)]
    cols = {"crack_transverse": "#1b9e77", "crack_longitudinal": "#d95f02", "crack_oblique": "#7570b3",
            "dent": "#e7298a"}
    for st, c in cols.items():
        vv = [f.get(f"{st}_s{int(s0)}", np.nan) for s0 in S0_GRID]
        ax.semilogy(S0_GRID, vv, "-o", color=c, ms=4, label=st.replace("_", " "))
    for lo, hi in _band(gb):
        ax.axvspan(lo, hi, color=gb.C_SEEN, alpha=0.25, lw=0)
    ax.axhline(1, color="k", lw=1)
    ax.axhline(MARGIN, color="k", lw=0.8, ls="--")
    ax.text(111, 1.06, "threshold (intact, ~1/16 false alarms)", fontsize=8)
    ax.text(111, MARGIN * 1.06, f"robust: {MARGIN:g}× threshold (6 dB)", fontsize=8)
    ax.set_xlabel("defect position s0 (mm)")
    ax.set_ylabel("best-window score / threshold")
    ax.set_title("F. Today's aperture: detectability vs s0\n(purple = ray-traced glint band)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, which="both")
    fig.suptitle("Field-level evidence, production horn (16 intact poses, receiver noise 1% of today's mean window "
                 "power): a defect's signal lands where ITS part of the rail reflects to", fontsize=13,
                 fontweight="bold", y=0.995)
    p = FIG / "03_field_evidence.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


def _label(n, an):
    th, phi, h = HORNS[n]
    hl = f"θ{th:.0f}" + (f" φ{phi:.0f}" if phi else "") + (f" H{h:.0f}" if h != H else "")
    return f"{hl} | {an}"


def fig_scan(plt, rows):
    from matplotlib.colors import ListedColormap, BoundaryNorm
    n = len(rows)
    fig = plt.figure(figsize=(17, 0.33 * n + 2.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.3, 1.0, 0.7], wspace=0.06, top=0.94)
    ax = fig.add_subplot(gs[0])
    M = np.array([r["per_s0_dB"] for r in rows])
    C = np.array([r["per_s0"] for r in rows]) * 4
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=20,
                   extent=[S0_GRID[0] - 2.5, S0_GRID[-1] + 2.5, n - 0.5, -0.5])
    for i in range(n):
        for j, s0 in enumerate(S0_GRID):
            if np.isfinite(C[i, j]):
                ax.text(s0, i, f"{C[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                        color="w" if M[i, j] < 11 else "k")
    ax.set_yticks(range(n))
    ax.set_yticklabels([_label(r["horn"], r["aperture"]) + ("  ☀" if r["bright"] else "") for r in rows],
                       fontsize=8.5)
    for k, r in enumerate(rows):
        if r["horn"] == "h55" and r["aperture"].startswith("current"):
            ax.get_yticklabels()[k].set_fontweight("bold")
            ax.get_yticklabels()[k].set_color("tab:purple")
    ax.set_xticks(S0_GRID)
    ax.set_xlabel("defect position s0 (mm) — crown/running band ← → gauge corner")
    cb = fig.colorbar(im, cax=ax.inset_axes([0.0, -0.075 * 38 / n, 1.0, 0.012 * 38 / n]), orientation="horizontal")
    cb.set_label("detection margin over threshold (dB, mean of 4 defects; 6 dB = robust)")
    ax.set_title("A. Coverage map: colour = margin, number = of 4 defects ≥ 6 dB  (☀ = crown glint inside)")

    ax = fig.add_subplot(gs[1])
    yy = np.arange(n)
    for e, mk, c in (("0.001", "<", "0.6"), ("0.01", "o", "k"), ("0.1", ">", "0.6")):
        ax.scatter([r["recall_eta"][e] for r in rows], yy, marker=mk, color=c, s=28,
                   label=f"noise {float(e) * 100:g}%", zorder=3)
    ax.barh(yy, [r["recall"] for r in rows], color="#9ecae1", height=0.6)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_xlabel("robust recall (≥ 6 dB) over s 110–170\n(52 defects)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, axis="x")
    ax.set_title("B. Recall vs receiver noise")

    ax = fig.add_subplot(gs[2])
    ax.barh(yy, [r["power_dB"] for r in rows], color=["#de2d26" if r["bright"] else "0.6" for r in rows], height=0.6)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks([])
    ax.set_xlabel("intact power per window\nvs today (dB)")
    ax.set_title("C. Light collected")
    ax.grid(alpha=0.3, axis="x")
    fig.suptitle("Tier 1 scan: every horn placement × every aperture, sorted by recall over the priority arc",
                 fontsize=13, fontweight="bold", y=0.985)
    p = FIG / "06_coverage_scan.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


def single_bright(n, an):
    th, phi, h = HORNS[n]
    xr = apertures(th, h, phi)[an]
    return xr[0] <= crown_mirror_x(th, phi, h) <= xr[1]


def fig_combos(plt, gb, rows, pairs, single, specs):
    fig, axs = plt.subplots(1, 2, figsize=(19, 7.0), gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.62})
    ax = axs[0]
    cur = next(r for r in rows if r["horn"] == "h55" and r["aperture"].startswith("current"))
    best1 = rows[0]
    dark1 = next((r for r in rows if not r["bright"]), None)
    lines = [(cur, "today: " + _label(cur["horn"], cur["aperture"]), "tab:purple", "-")]
    if dark1 is not None and dark1 is not cur:
        lines.append((dark1, "best single, dark field: " + _label(dark1["horn"], dark1["aperture"]), "#1b9e77", "-"))
    if best1 is not dark1:
        lines.append((best1, "best single: " + _label(best1["horn"], best1["aperture"]), "#d95f02", "-"))
    for kind, c in (("2 horns, 1 MS", "#e7298a"), ("1 horn, 2 MS", "#7570b3"), ("2 horns, 2 MS", "k")):
        b = next((p for p in pairs if p["kind"] == kind), None)
        if b is not None:
            lab = f"best {kind}: " + "  +  ".join(_label(a, an) for a, an in b["channels"])
            lines.append((b, lab, c, "--"))
    for r, lab, c, ls in lines:
        ax.plot(S0_GRID, r["per_s0_dB"], ls, color=c, marker="o", ms=4, lw=1.8,
                label=f"{lab}  [robust recall {r['recall']:.2f}]")
    for lo, hi in _band(gb):
        ax.axvspan(lo, hi, color=gb.C_SEEN, alpha=0.15, lw=0)
    ax.axhline(20 * np.log10(MARGIN), color="k", ls="--", lw=0.9)
    ax.axhline(0, color="k", lw=0.9)
    ax.text(S0_GRID[0], 20 * np.log10(MARGIN) + 0.3, "robust (6 dB)", fontsize=8)
    ax.set_xlabel("defect position s0 (mm)   (purple band: today's ray-traced glint band)")
    ax.set_ylabel("detection margin, mean of 4 defects (dB)")
    ax.legend(fontsize=7.8, loc="upper left", bbox_to_anchor=(0, -0.13), ncol=1, frameon=False)
    ax.grid(alpha=0.3)
    ax.set_title("A. Margin along the arc: today vs the best options")

    ax = axs[1]
    ladder = [("today", [cur])]
    ladder.append(("1 horn, 1 MS", rows[:3]))
    for kind in ("1 horn, 2 MS", "2 horns, 1 MS", "2 horns, 2 MS"):
        ladder.append((kind, [p for p in pairs if p["kind"] == kind][:3]))
    labs, crown, corner, rec, grp = [], [], [], [], []
    for kind, items in ladder:
        for r in items:
            labs.append(" + ".join(_label(a, an).replace(" | ", "|") + ("☀" if single_bright(a, an) else "")
                                   for a, an in r["channels"]))
            crown.append(r["margin_crown_dB"])
            corner.append(r["margin_corner_dB"])
            rec.append(r["recall"])
            grp.append(kind)
    yy = np.arange(len(labs), dtype=float)
    gap = 0.0
    for k in range(len(yy)):
        if k and grp[k] != grp[k - 1]:
            gap += 0.7
        yy[k] += gap
    ax.barh(yy - 0.19, crown, height=0.36, color="#f2b134", label="crown / running band, s0 110–140")
    ax.barh(yy + 0.19, corner, height=0.36, color="#7b2cbf", label="gauge corner, s0 145–170")
    for y, r, c1, c2 in zip(yy, rec, crown, corner):
        ax.text(max(c1, c2) + 0.3, y, f"{r:.2f}", va="center", fontsize=8)
    ax.axvline(20 * np.log10(MARGIN), color="k", ls="--", lw=0.9)
    ax.set_yticks(yy)
    ax.set_yticklabels(labs, fontsize=7.6)
    for k, g in enumerate(grp):
        if k == 0 or grp[k - 1] != g:
            ax.text(0.2, yy[k] - 0.62, g.upper(), fontsize=8.5, fontweight="bold", ha="left",
                    color="0.3")
    ax.set_ylim(yy[-1] + 0.7, -1.0)
    ax.set_xlim(0, 21)
    ax.set_xlabel("mean detection margin (dB);  number = robust recall over the whole arc")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.1), ncol=2, frameon=False)
    ax.grid(alpha=0.3, axis="x")
    ax.set_title("B. The ladder: best 3 of each hardware kind (☀ = bright field)")
    fig.suptitle("Combinations: a second channel (time-multiplexed horn or a second metasurface) fills the arc "
                 "the first cannot see", fontsize=13, fontweight="bold", y=1.0)
    p = FIG / "07_combinations.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--horns", nargs="*", default=list(HORNS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--check-shadow", action="store_true",
                    help="re-solve the smoke defects casting rays from EVERY facet and compare")
    ap.add_argument("--profile", default="laptop")
    ap.add_argument("--twins", action="store_true",
                    help="solve the pose twins of the evidence figure's defects (production horn)")
    a = ap.parse_args()
    if a.analyze:
        analyze()
        return 0
    device = config.get_device(a.profile)
    chunk = config.PROFILES[a.profile].chunk_faces
    specs = sample_list(a.smoke)
    if a.twins:
        global OUT
        OUT = OUT / "_twins"
        solve("h55", twin_specs(), device, chunk)
        return 0
    if a.check_shadow:
        ref = sample_list(True)[2:]
        fast = {}
        for s in ref:
            fast[s["id"]] = np.load(OUT / "h55" / f"{s['id']}.npz")["psi"]
        OUT = OUT / "_full_shadow"
        solve("h55", ref, device, chunk, full_shadow=True)
        for s in ref:
            full = np.load(OUT / "h55" / f"{s['id']}.npz")["psi"]
            err = np.linalg.norm(fast[s["id"]] - full) / np.linalg.norm(full)
            print(f"  {s['id']}: defect-facet rays vs every facet: rel. field difference {err:.2e}")
        return 0
    for n in (["h55"] if a.smoke else a.horns):
        solve(n, specs, device, chunk)
    return 0


if __name__ == "__main__":
    sys.exit(main())
