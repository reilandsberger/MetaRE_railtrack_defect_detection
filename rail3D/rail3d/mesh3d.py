"""3D rail-segment mesh construction: swept intact cross-section + per-point
defect depth fields.

Geometry model (rev. 2 — replaces the per-slice cross-section blend):

    displaced(s, y) = intact(s) - d(s, y) * n_hat(s)

where ``s`` is the arc-length coordinate along the illuminated cross-section
arc, ``y`` the rail axis, ``n_hat(s)`` the outward 2D normal of the intact
section (in the x-z plane), and ``d(s, y) >= 0`` a per-class scalar depth
field in mm. This represents defects localized in BOTH directions: oriented
hairline cracks (longitudinal / transverse / oblique line-divots), compact
2D-Gaussian dents, side wear confined to the horn-facing shoulder, and ragged
shelling patches.

Defects are described by resolution-independent parameter dicts
(``sample_defect_params``) rendered onto any (s, y) grid
(``render_depth_field``) — so the same defect instance lands identically on
the fine simulation mesh and the coarse ray-cast occluder mesh, and every
sample is reproducible from its seed.

Parameter ranges are taken from laser-scanned defect measurements:
Ye et al. 2018 (Proc IMechE F, Table 1 / Figs 14, 23, 24) and
Ye et al. 2023 (IEEE TIM, Figs 7, 9). See README.md section 2.

Every slice keeps the same number of arc points, so all meshes share one face
array — fixed topology is what lets the field solver batch meshes.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from . import config, sections


# ---------------------------------------------------------------------------
# Arc geometry of the intact cross-section
# ---------------------------------------------------------------------------
def arc_geometry(section_intact: torch.Tensor, n_arc: int | None = None) -> dict:
    """Geometry of the illuminated arc, shared by all samples.

    Returns dict with (all numpy, length n_arc):
      pts    (n_arc, 2)  intact arc points in section coords (x, y2d)
      s      (n_arc,)    arc-length coordinate from the arc start (mm)
      normal (n_arc, 2)  outward unit normal in the section plane
      x, z   (n_arc,)    convenience coords (z = y2d - RAIL_HEIGHT)
    """
    mask = sections.illuminated_mask(section_intact)
    idx = _contiguous_mask_indices(mask)
    if n_arc is None:
        n_arc = default_arc_count(section_intact)
    pts = sections.resample_open_curve(section_intact.numpy()[idx].astype(np.float64), n_arc)

    seg = np.diff(pts, axis=0)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(seg, axis=1))])

    # outward normal: the loop is CCW, arc traversed with increasing index ->
    # rotate tangent by -90 deg gives (ty, -tx), pointing out of the material.
    tang = np.empty_like(pts)
    tang[1:-1] = pts[2:] - pts[:-2]
    tang[0] = pts[1] - pts[0]
    tang[-1] = pts[-1] - pts[-2]
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    normal = np.stack([tang[:, 1], -tang[:, 0]], axis=1)
    # orient outward: on the crown (|x| small) outward means +z (larger y2d)
    crown = np.abs(pts[:, 0]) < 20
    if crown.any() and normal[crown, 1].mean() < 0:
        normal = -normal

    return {"pts": pts, "s": s, "normal": normal,
            "x": pts[:, 0], "z": pts[:, 1] - config.RAIL_HEIGHT, "n_arc": n_arc}


def _smooth_window(values: np.ndarray, lo: float, hi: float, roll: float) -> np.ndarray:
    """1 inside [lo, hi], cosine rolloff to 0 over ``roll`` outside."""
    w = np.ones_like(values, dtype=np.float64)
    below = values < lo
    above = values > hi
    w[below] = 0.5 * (1 + np.cos(np.pi * np.clip((lo - values[below]) / roll, 0, 1)))
    w[above] = 0.5 * (1 + np.cos(np.pi * np.clip((values[above] - hi) / roll, 0, 1)))
    return w


def region_band(geom: dict, band: str) -> np.ndarray:
    """Smooth 0..1 mask over arc points for a named surface region.

    'crack'  — running band + gauge corner: upward-facing surface (n_z high),
               excluding the steep sides ("approximately the top", user spec).
    'gauge'  — horn-facing (+x) shoulder/corner band for wear & shelling.
    """
    x, nz = geom["x"], geom["normal"][:, 1]
    if band == "crack":
        w = _smooth_window(x, -config.CROWN_HALF_WIDTH, config.GAUGE_X_MAX, 4.0)
        return w * np.clip((nz - 0.25) / 0.15, 0, 1)     # fade out on steep sides
    if band == "gauge":
        # x >= GAUGE_X_MIN alone also catches the DOWNWARD-facing under-head
        # fillet (the illuminated arc wraps around the head side, nz ~ -0.9
        # there); the nz factor keeps only crown-right + the horn-facing head
        # side (nz >= ~0) — "the side of the railhead that faces the horn".
        w = _smooth_window(x, config.GAUGE_X_MIN, config.GAUGE_X_MAX, 4.0)
        return w * np.clip((nz + 0.15) / 0.15, 0, 1)
    raise ValueError(f"Unknown band: {band}")


# ---------------------------------------------------------------------------
# CSV cross-section -> measured depth profile
# ---------------------------------------------------------------------------
def extract_csv_profile(section_intact: torch.Tensor, section_defect: torch.Tensor,
                        n_arc: int | None = None) -> dict:
    """Signed depth-of-material-removed along the arc from a defect CSV.

    Uses the index correspondence of the two width-matched loops:
    dev_i = (intact_i - defect_i) . n_hat_i  (positive = material removed).
    Returns the deviation on the arc grid plus the dominant-notch summary
    (center s, width, max depth) used to re-center crack/dent profiles.
    """
    geom = arc_geometry(section_intact, n_arc)
    mask = sections.illuminated_mask(section_intact)
    idx = _contiguous_mask_indices(mask)
    intact_arc = geom["pts"]
    defect_arc = sections.resample_open_curve(
        section_defect.numpy()[idx].astype(np.float64), geom["n_arc"])
    dev = np.einsum("ij,ij->i", intact_arc - defect_arc, geom["normal"])
    # Baseline correction: match_reference_width rescales the whole defect loop
    # by a few %, which shifts the deviation globally (wear CSVs otherwise read
    # as depth 0 — the inflation cancels the removal). Assume >=20% of the arc
    # is undamaged and subtract that baseline before keeping removal only.
    dev = dev - np.percentile(dev, 20.0)
    dev = np.clip(dev, 0.0, None)

    depth = float(dev.max())
    if depth < 1e-6:
        return {"s_dev": geom["s"], "dev": dev, "depth": 0.0,
                "center_s": float(geom["s"].mean()), "width": 2.0}
    peak = int(np.argmax(dev))
    above = dev > 0.1 * depth
    lo = peak
    while lo > 0 and above[lo - 1]:
        lo -= 1
    hi = peak
    while hi < len(dev) - 1 and above[hi + 1]:
        hi += 1
    return {"s_dev": geom["s"], "dev": dev, "depth": depth,
            "center_s": float(geom["s"][peak]),
            "width": float(max(geom["s"][hi] - geom["s"][lo], 1.0)),
            "support": (int(lo), int(hi))}


def _notch_profile(profile: dict) -> tuple[np.ndarray, np.ndarray]:
    """(v, depth(v)) of the dominant notch, re-centered on its peak (v in mm)."""
    if profile["depth"] <= 1e-6 or "support" not in profile:
        v = np.linspace(-1.0, 1.0, 9)
        return v, np.maximum(0.0, 1.0 - np.abs(v))     # unit V-notch fallback
    lo, hi = profile["support"]
    v = profile["s_dev"][lo:hi + 1] - profile["center_s"]
    return v, profile["dev"][lo:hi + 1] / profile["depth"]


# ---------------------------------------------------------------------------
# Per-class defect parameter sampling (resolution independent)
# ---------------------------------------------------------------------------
def _u(gen, lo, hi):
    return float(torch.empty(1).uniform_(lo, hi, generator=gen))


def sample_defect_params(class_name: str, gen: torch.Generator,
                         geom: dict, csv_profile: dict | None = None) -> dict:
    """Draw one defect instance. All lengths in mm; angles in radians.

    crack : line-divot(s). theta measured from the y (rail) axis in the (y, s)
            surface plane: 0 = longitudinal, pi/2 = transverse; obliques mixed
            in. Across-line profile = measured CSV notch (unit-normalized);
            depth from the CSV, clipped to the measured 2-6.9 mm range.
    dent  : super-Gaussian pit(s); footprint/depth per Ye 2018 Table 1 rows 4-5.
    wear  : CSV cross-section deviation masked to the gauge band, long
            super-Gaussian y-envelope (unchanged philosophy).
    shell : parametric ragged patch on the gauge shoulder (no CSVs exist);
            Fourier-modulated ellipse + interior roughness per Ye 2023 Fig 9.
    """
    y0 = _u(gen, *config.DEFECT_CENTER_RANGE)
    # sampling band must match the render band: cracks and dents live on the
    # running band ("crack"), wear and shelling on the horn-facing shoulder
    band_name = "crack" if class_name in ("crack", "dent", "intact") else "gauge"
    band = region_band(geom, band_name)
    s_band = geom["s"][band > 0.5]
    s_lo, s_hi = (float(s_band.min()), float(s_band.max())) if len(s_band) else \
                 (float(geom["s"][0]), float(geom["s"][-1]))

    p: dict = {"class": class_name, "y0": y0}

    if class_name == "crack":
        mode = _u(gen, 0, 1)
        if mode < 0.3:
            theta = 0.0                       # longitudinal (runs along y)
        elif mode < 0.6:
            theta = math.pi / 2               # transverse (runs across the head)
        else:
            sign = 1.0 if _u(gen, 0, 1) < 0.5 else -1.0
            theta = sign * math.radians(_u(gen, 20, 70))
        L = _u(gen, *config.CRACK_LENGTH_RANGE)
        v, prof = _notch_profile(csv_profile) if csv_profile else _notch_profile({"depth": 0})
        # Hairline character (Ye 2018 Table 1: surface width ~2 mm at 45°):
        # the CSV notches are chunky cross-section features — keep their measured
        # depth SHAPE but compress the across-crack support to hairline width.
        w_hair = _u(gen, *config.CRACK_WIDTH_RANGE)
        v_span = float(v.max() - v.min())
        if v_span > 1e-6:
            v = v * (w_hair / v_span)
        depth = float(np.clip(csv_profile["depth"] if csv_profile else 3.0,
                              *config.CRACK_DEPTH_RANGE))
        n_lines = 1
        if _u(gen, 0, 1) < 0.3:
            n_lines = 2 if _u(gen, 0, 1) < 0.7 else 3
        offsets = [0.0] + [(-1) ** k * _u(gen, 5, 15) for k in range(1, n_lines)]
        p.update(theta=theta, L=L, depth=depth, s0=_u(gen, s_lo + 3, s_hi - 3),
                 profile_v=v, profile_d=prof, offsets=offsets, band="crack")

    elif class_name == "dent":
        depth = float(np.clip(csv_profile["depth"] if csv_profile else 2.0,
                              *config.DENT_DEPTH_RANGE))
        fw_y = _u(gen, *config.DENT_FOOTPRINT_Y)      # FWHM along the rail
        fw_s = _u(gen, *config.DENT_FOOTPRINT_S)      # FWHM across the head
        pits = [(0.0, 0.0, 1.0)]
        if _u(gen, 0, 1) < 0.2:                       # chain of pits (Fig 7 col 4)
            for k in range(1 + int(_u(gen, 0, 1) < 0.5)):
                pits.append((_u(gen, 8, 15) * (k + 1), _u(gen, -3, 3),
                             _u(gen, 0.6, 1.0)))
        p.update(depth=depth, fw_y=fw_y, fw_s=fw_s, s0=_u(gen, s_lo + 5, s_hi - 5),
                 pits=pits, band="crack")             # dents live on the running band

    elif class_name == "wear":
        L = _u(gen, *config.DEFECT_LENGTH_RANGE["wear"])
        p.update(L=L, band="gauge",
                 csv_dev=csv_profile["dev"] if csv_profile else None,
                 csv_s=csv_profile["s_dev"] if csv_profile else None,
                 depth=float(csv_profile["depth"]) if csv_profile else 1.5)

    elif class_name == "shell":
        r_s = _u(gen, *config.SHELL_RADIUS_RANGE)     # semi-axes (mm)
        r_y = _u(gen, *config.SHELL_RADIUS_RANGE)
        depth = _u(gen, *config.SHELL_DEPTH_RANGE)
        four = [(_u(gen, 0, 0.25), _u(gen, 0, 2 * math.pi)) for _ in range(2, 6)]
        lobes = [(0.0, 0.0, 1.0)]
        if _u(gen, 0, 1) < 0.3:
            lobes.append((_u(gen, 0.6, 1.0) * r_y, _u(gen, -0.7, 0.7) * r_s,
                          _u(gen, 0.5, 0.9)))
        rough_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=gen))
        p.update(r_s=r_s, r_y=r_y, depth=depth, fourier=four, lobes=lobes,
                 s0=_u(gen, s_lo + 4, s_hi - 4), rough_seed=rough_seed, band="gauge")

    elif class_name == "intact":
        pass
    else:
        raise ValueError(f"Unknown defect class: {class_name}")
    return p


# ---------------------------------------------------------------------------
# Depth-field rendering d(s, y) — works on any grid resolution
# ---------------------------------------------------------------------------
def render_depth_field(p: dict, geom: dict, y_slices: np.ndarray) -> np.ndarray:
    """Render one defect's depth field on (n_slices, n_arc). Depth in mm >= 0."""
    s = geom["s"]
    n_s, n_y = len(s), len(y_slices)
    d = np.zeros((n_y, n_s))
    cls = p["class"]
    if cls == "intact":
        return d

    S = s[None, :] - p.get("s0", 0.0)          # (1, n_s) across-head offset
    Y = y_slices[:, None] - p["y0"]            # (n_y, 1) along-rail offset

    if cls == "crack":
        ct, st = math.cos(p["theta"]), math.sin(p["theta"])
        for off in p["offsets"]:
            u = Y * ct + (S - off) * st        # along the crack line
            v = -Y * st + (S - off) * ct       # across the crack line
            across = np.interp(v, p["profile_v"], p["profile_d"], left=0.0, right=0.0)
            un = np.abs(u) / (p["L"] / 2)
            along = np.where(un <= 0.6, 1.0,
                             np.where(un < 1.0, 0.5 * (1 + np.cos(np.pi * (un - 0.6) / 0.4)), 0.0))
            d = np.maximum(d, p["depth"] * across * along)

    elif cls == "dent":
        k = 4 * math.log(2)
        for dy, ds, scale in p["pits"]:
            r2 = ((Y - dy) / p["fw_y"]) ** 2 + ((S - ds) / p["fw_s"]) ** 2
            d = np.maximum(d, scale * p["depth"] * np.exp(-k * r2))

    elif cls == "wear":
        if p.get("csv_dev") is not None:
            dev = np.interp(s, p["csv_s"], p["csv_dev"])
        else:
            dev = p["depth"] * np.ones_like(s)
        g_y = np.exp(-math.log(2) * (2 * Y[:, 0] / p["L"]) ** 8)
        d = g_y[:, None] * dev[None, :]

    elif cls == "shell":
        rng = np.random.default_rng(p["rough_seed"])
        for dy, ds, scale in p["lobes"]:
            yy = (Y - dy) / p["r_y"]
            ss = (S - ds) / p["r_s"]
            rho = np.sqrt(yy**2 + ss**2)
            phi = np.arctan2(ss, yy)
            edge = np.ones_like(phi)
            for k4, (a, ph) in enumerate(p["fourier"], start=2):
                edge += a * np.cos(k4 * phi + ph)
            base = np.clip(1.0 - rho / np.clip(edge, 0.5, 1.5), 0.0, None) ** 0.7
            d = np.maximum(d, scale * p["depth"] * base)
        if d.max() > 0:
            # ragged interior (Fig 9): seeded roughness on a fixed PHYSICAL
            # 4 mm lattice around the defect center, bilinearly interpolated —
            # resolution independent, so fine mesh and coarse occluder carry
            # the same roughness pattern.
            lat = rng.standard_normal((17, 17))
            ly = np.linspace(-32.0, 32.0, 17)
            iy = np.clip((Y[:, 0] - ly[0]) / 4.0, 0, 15.999)
            is_ = np.clip((S[0, :] - ly[0]) / 4.0, 0, 15.999)
            y0i, s0i = iy.astype(int), is_.astype(int)
            fy, fs = iy - y0i, is_ - s0i
            rough = ((1 - fy)[:, None] * (1 - fs)[None, :] * lat[y0i][:, s0i]
                     + fy[:, None] * (1 - fs)[None, :] * lat[y0i + 1][:, s0i]
                     + (1 - fy)[:, None] * fs[None, :] * lat[y0i][:, s0i + 1]
                     + fy[:, None] * fs[None, :] * lat[y0i + 1][:, s0i + 1])
            d = d * np.clip(1.0 + 0.2 * rough, 0.6, 1.4)

    # confine to the class's surface band (no defects on the steep sides)
    d *= region_band(geom, p["band"])[None, :]
    return d


# ---------------------------------------------------------------------------
# Mesh assembly
# ---------------------------------------------------------------------------
def _contiguous_mask_indices(mask: torch.Tensor) -> np.ndarray:
    idx = np.flatnonzero(mask.numpy())
    if idx.size == 0:
        raise ValueError("Illuminated mask is empty")
    if idx[-1] - idx[0] + 1 != idx.size:
        raise ValueError("Illuminated region is not a contiguous index run on the loop")
    return idx


def default_arc_count(section_intact: torch.Tensor, arc_ds: float = config.SLICE_DS) -> int:
    mask = sections.illuminated_mask(section_intact)
    idx = _contiguous_mask_indices(mask)
    pts = section_intact.numpy()[idx]
    arc_len = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
    return int(round(arc_len / arc_ds)) + 1


def strip_faces(n_arc: int, n_slices: int) -> torch.Tensor:
    i = np.arange(n_arc - 1)
    j = np.arange(n_slices - 1)
    J, I = np.meshgrid(j, i, indexing="ij")
    a = J * n_arc + I
    b = J * n_arc + I + 1
    c = (J + 1) * n_arc + I
    dd = (J + 1) * n_arc + I + 1
    tri1 = np.stack([a, b, c], axis=-1).reshape(-1, 3)
    tri2 = np.stack([b, dd, c], axis=-1).reshape(-1, 3)
    return torch.tensor(np.concatenate([tri1, tri2], axis=0), dtype=torch.long)


def sweep_rail_mesh(
    section_intact: torch.Tensor,
    defect_params: dict | None = None,
    seg_len: float = config.SEG_LEN,
    slice_ds: float = config.SLICE_DS,
    arc_ds: float | None = None,
    n_arc: int | None = None,
    roll_deg: float = 0.0,
    jitter_xz: tuple[float, float] = (0.0, 0.0),
    faces: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one rail-segment surface mesh (intact, or displaced by a defect).

    ``defect_params`` comes from sample_defect_params (None -> intact). The
    depth field is rendered on THIS mesh's (s, y) grid, so the same params
    reproduce the same physical defect at any resolution (fine sim mesh and
    coarse occluder mesh stay geometrically consistent).
    """
    if arc_ds is None:
        arc_ds = slice_ds
    if n_arc is None:
        n_arc = default_arc_count(section_intact, arc_ds)
    geom = arc_geometry(section_intact, n_arc)

    y_slices = np.arange(-seg_len / 2, seg_len / 2 + slice_ds / 2, slice_ds)
    n_slices = len(y_slices)

    pts = geom["pts"][None, :, :].repeat(n_slices, axis=0)     # (n_y, n_arc, 2)
    if defect_params is not None and defect_params.get("class", "intact") != "intact":
        d = render_depth_field(defect_params, geom, y_slices)
        pts = pts - d[:, :, None] * geom["normal"][None, :, :]

    v = np.empty((n_slices, n_arc, 3))
    v[:, :, 0] = pts[:, :, 0]
    v[:, :, 1] = y_slices[:, None]
    v[:, :, 2] = pts[:, :, 1] - config.RAIL_HEIGHT

    if roll_deg != 0.0:
        ang = math.radians(roll_deg)
        c, sn = math.cos(ang), math.sin(ang)
        x, z = v[:, :, 0].copy(), v[:, :, 2].copy()
        v[:, :, 0] = c * x - sn * z
        v[:, :, 2] = sn * x + c * z
    v[:, :, 0] += jitter_xz[0]
    v[:, :, 2] += jitter_xz[1]

    v = torch.tensor(v.reshape(-1, 3), dtype=torch.float32)
    f = faces if faces is not None else strip_faces(n_arc, n_slices)
    f = _ensure_outward(v, f)
    return v, f


def _ensure_outward(v: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    points = v[f]
    ab = points[:, 1] - points[:, 0]
    ac = points[:, 2] - points[:, 0]
    normal_z = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
    center_x = points[:, :, 0].mean(dim=1)
    center_z = points[:, :, 2].mean(dim=1)
    crown = (center_x.abs() < 20.0) & (center_z > -20.0)
    if crown.any() and normal_z[crown].mean() < 0:
        f = f[:, [0, 2, 1]]
    return f


def sample_augmentation(gen: torch.Generator) -> dict:
    roll = _u(gen, -config.ROLL_DEG_STD, config.ROLL_DEG_STD)
    jitter = torch.randn(2, generator=gen) * config.JITTER_XZ_STD
    return {"roll_deg": roll, "jitter_x": float(jitter[0]), "jitter_z": float(jitter[1])}


def build_sample_mesh(
    section_intact: torch.Tensor,
    class_name: str,
    section_defect: torch.Tensor | None,
    seed: int,
    n_arc: int | None = None,
    faces: torch.Tensor | None = None,
    augment: bool = True,
    slice_ds: float = config.SLICE_DS,
    arc_ds: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """One deterministic sample: defect params + augmentation from the seed.

    Returns (v, f, meta); meta carries the full defect parameter dict (minus
    the bulky profile arrays) so shards stay self-describing.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    geom = arc_geometry(section_intact, n_arc if n_arc is not None
                        else default_arc_count(section_intact, arc_ds or slice_ds))
    csv_profile = None
    if section_defect is not None and class_name in ("crack", "dent", "wear"):
        csv_profile = extract_csv_profile(section_intact, section_defect, geom["n_arc"])

    params = sample_defect_params(class_name, gen, geom, csv_profile)
    aug = sample_augmentation(gen) if augment else \
        {"roll_deg": 0.0, "jitter_x": 0.0, "jitter_z": 0.0}

    v, f = sweep_rail_mesh(
        section_intact, defect_params=params,
        slice_ds=slice_ds, arc_ds=arc_ds, n_arc=n_arc,
        roll_deg=aug["roll_deg"], jitter_xz=(aug["jitter_x"], aug["jitter_z"]),
        faces=faces)

    meta = {k: val for k, val in params.items()
            if k not in ("profile_v", "profile_d", "csv_dev", "csv_s")}
    meta.update(aug)
    meta["seed"] = seed
    return v, f, meta


def defect_params_for_sample(
    section_intact: torch.Tensor,
    class_name: str,
    section_defect: torch.Tensor | None,
    seed: int,
    n_arc: int | None = None,
) -> tuple[dict, dict]:
    """(defect params, augmentation) for a seed WITHOUT building the mesh —
    used by the generator to render the same defect on the coarse occluder."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    geom = arc_geometry(section_intact, n_arc)
    csv_profile = None
    if section_defect is not None and class_name in ("crack", "dent", "wear"):
        csv_profile = extract_csv_profile(section_intact, section_defect, geom["n_arc"])
    params = sample_defect_params(class_name, gen, geom, csv_profile)
    aug = sample_augmentation(gen)
    return params, aug
