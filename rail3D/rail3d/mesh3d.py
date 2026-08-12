"""3D rail-segment mesh construction by sweeping 2D cross-sections.

A rail segment is built by sweeping the intact cross-section loop along the
rail axis (y) and blending a defect cross-section in with a class-specific
longitudinal envelope g(y) in [0, 1]:

    loop(y) = (1 - g(y)) * intact + g(y) * defect        (index-wise blend)

Index-wise blending is valid because both loops come from the same
arc-length-uniform resampling anchored at the bottom-center point and are
width-matched (exactly the correspondence the 2D pipeline already relies on).

Only the illuminated upper region (z > Z_CUT) is meshed; every slice is
resampled to the same number of arc points so all meshes share one face
array — that fixed topology is what lets the field solver batch meshes.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from . import config, sections


# ---------------------------------------------------------------------------
# Longitudinal defect envelopes
# ---------------------------------------------------------------------------
def defect_envelope(class_name: str, gen: torch.Generator) -> tuple:
    """Sample a longitudinal envelope for one defect instance.

    Returns (g, params) where ``g(y_mm) -> [0, 1]`` accepts a numpy array.
      crack: short, sharp-edged (plateau + raised-cosine taper)
      dent:  medium Gaussian (FWHM = L)
      wear:  long, near-uniform super-Gaussian (FWHM = L)
    """
    lo, hi = config.DEFECT_LENGTH_RANGE[class_name]
    L = float(torch.empty(1).uniform_(lo, hi, generator=gen))
    y0_lo, y0_hi = config.DEFECT_CENTER_RANGE
    y0 = float(torch.empty(1).uniform_(y0_lo, y0_hi, generator=gen))

    if class_name == "crack":
        # plateau over the central 60% of L, cosine taper to zero at +/- L/2
        def g(y: np.ndarray) -> np.ndarray:
            u = np.abs(np.asarray(y, dtype=np.float64) - y0) / (L / 2)
            out = np.zeros_like(u)
            out[u <= 0.6] = 1.0
            taper = (u > 0.6) & (u < 1.0)
            out[taper] = 0.5 * (1 + np.cos(np.pi * (u[taper] - 0.6) / 0.4))
            return out
    elif class_name == "dent":
        def g(y: np.ndarray) -> np.ndarray:
            return np.exp(-4 * math.log(2) * ((np.asarray(y, dtype=np.float64) - y0) / L) ** 2)
    elif class_name == "wear":
        def g(y: np.ndarray) -> np.ndarray:
            return np.exp(-math.log(2) * (2 * (np.asarray(y, dtype=np.float64) - y0) / L) ** 8)
    else:
        raise ValueError(f"Unknown defect class: {class_name}")

    return g, {"class": class_name, "y0": y0, "L": L}


# ---------------------------------------------------------------------------
# Mesh assembly
# ---------------------------------------------------------------------------
def _contiguous_mask_indices(mask: torch.Tensor) -> np.ndarray:
    """Loop indices of the illuminated region; must form one contiguous run."""
    idx = np.flatnonzero(mask.numpy())
    if idx.size == 0:
        raise ValueError("Illuminated mask is empty")
    if idx[-1] - idx[0] + 1 != idx.size:
        raise ValueError("Illuminated region is not a contiguous index run on the loop")
    return idx


def default_arc_count(section_intact: torch.Tensor, arc_ds: float = config.SLICE_DS) -> int:
    """Number of arc samples per slice: illuminated arc length / arc_ds."""
    mask = sections.illuminated_mask(section_intact)
    idx = _contiguous_mask_indices(mask)
    pts = section_intact.numpy()[idx]
    arc_len = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
    return int(round(arc_len / arc_ds)) + 1


def strip_faces(n_arc: int, n_slices: int) -> torch.Tensor:
    """Triangle strip connecting consecutive rings; shared by all samples.

    Winding is chosen so that (with the CCW loop orientation of the 2D
    sections and slices ordered along +y) face normals point out of the rail
    (+z on the crown). ``sweep_rail_mesh`` asserts this and flips if needed.
    """
    i = np.arange(n_arc - 1)
    j = np.arange(n_slices - 1)
    J, I = np.meshgrid(j, i, indexing="ij")
    a = J * n_arc + I            # (j, i)
    b = J * n_arc + I + 1        # (j, i+1)
    c = (J + 1) * n_arc + I      # (j+1, i)
    d = (J + 1) * n_arc + I + 1  # (j+1, i+1)
    tri1 = np.stack([a, b, c], axis=-1).reshape(-1, 3)
    tri2 = np.stack([b, d, c], axis=-1).reshape(-1, 3)
    return torch.tensor(np.concatenate([tri1, tri2], axis=0), dtype=torch.long)


def sweep_rail_mesh(
    section_intact: torch.Tensor,
    section_defect: torch.Tensor | None = None,
    envelope=None,
    seg_len: float = config.SEG_LEN,
    slice_ds: float = config.SLICE_DS,
    arc_ds: float = config.SLICE_DS,
    n_arc: int | None = None,
    roll_deg: float = 0.0,
    jitter_xz: tuple[float, float] = (0.0, 0.0),
    faces: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one rail-segment surface mesh.

    Returns (v, f): vertices (Nv, 3) float32 in mm (crown at z=0), faces
    (Nf, 3) int64 with outward (+z on crown) winding.
    """
    mask = sections.illuminated_mask(section_intact)
    idx = _contiguous_mask_indices(mask)
    if n_arc is None:
        n_arc = default_arc_count(section_intact, arc_ds)

    intact_np = section_intact.numpy().astype(np.float64)
    defect_np = None if section_defect is None else section_defect.numpy().astype(np.float64)

    y_slices = np.arange(-seg_len / 2, seg_len / 2 + slice_ds / 2, slice_ds)
    n_slices = len(y_slices)
    if envelope is not None and defect_np is not None:
        g_vals = np.clip(envelope(y_slices), 0.0, 1.0)
    else:
        g_vals = np.zeros(n_slices)

    rings = np.empty((n_slices, n_arc, 2), dtype=np.float64)
    for j, g in enumerate(g_vals):
        loop = intact_np if (g == 0.0 or defect_np is None) else (1 - g) * intact_np + g * defect_np
        rings[j] = sections.resample_open_curve(loop[idx], n_arc)

    # Assemble (x, y, z): x = section x, z = section y - RAIL_HEIGHT
    v = np.empty((n_slices, n_arc, 3), dtype=np.float64)
    v[:, :, 0] = rings[:, :, 0]
    v[:, :, 1] = y_slices[:, None]
    v[:, :, 2] = rings[:, :, 1] - config.RAIL_HEIGHT

    # Rigid augmentation: roll about the y axis (around the crown origin), then x/z jitter
    if roll_deg != 0.0:
        ang = math.radians(roll_deg)
        c, s = math.cos(ang), math.sin(ang)
        x, z = v[:, :, 0].copy(), v[:, :, 2].copy()
        v[:, :, 0] = c * x - s * z
        v[:, :, 2] = s * x + c * z
    v[:, :, 0] += jitter_xz[0]
    v[:, :, 2] += jitter_xz[1]

    v = torch.tensor(v.reshape(-1, 3), dtype=torch.float32)
    f = faces if faces is not None else strip_faces(n_arc, n_slices)

    # Orientation check on crown faces (|x| small): outward means n_z > 0
    f = _ensure_outward(v, f)
    return v, f


def _ensure_outward(v: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    points = v[f]
    ab = points[:, 1] - points[:, 0]
    ac = points[:, 2] - points[:, 0]
    normal_z = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]  # z of cross product
    center_x = points[:, :, 0].mean(dim=1)
    center_z = points[:, :, 2].mean(dim=1)
    # true crown top only — web and head-underside faces rightly point down/sideways
    crown = (center_x.abs() < 20.0) & (center_z > -20.0)
    if crown.any() and normal_z[crown].mean() < 0:
        f = f[:, [0, 2, 1]]
    return f


def sample_augmentation(gen: torch.Generator) -> dict:
    """Draw the per-sample rigid augmentation (roll + x/z jitter)."""
    roll = float(torch.empty(1).uniform_(-config.ROLL_DEG_STD, config.ROLL_DEG_STD, generator=gen))
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
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """One deterministic sample: envelope + augmentation from the seed.

    ``class_name='intact'`` (with section_defect=None) builds an intact
    segment with the same augmentation distribution.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    if class_name == "intact" or section_defect is None:
        envelope, params = None, {"class": "intact", "y0": 0.0, "L": 0.0}
    else:
        envelope, params = defect_envelope(class_name, gen)

    aug = sample_augmentation(gen) if augment else {"roll_deg": 0.0, "jitter_x": 0.0, "jitter_z": 0.0}
    v, f = sweep_rail_mesh(
        section_intact,
        section_defect=section_defect,
        envelope=envelope,
        n_arc=n_arc,
        roll_deg=aug["roll_deg"],
        jitter_xz=(aug["jitter_x"], aug["jitter_z"]),
        faces=faces,
    )
    meta = {**params, **aug, "seed": seed}
    return v, f, meta
