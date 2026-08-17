"""Central configuration for the rail3D pipeline.

All physical constants, grid definitions, paths, and device profiles live here.
Every length is in **millimetres** (the Face3D_clean convention), unlike the 2D
rail scripts which used micrometres internally.

Coordinate convention (fixed across the whole package):
    x — across the railhead (the 2D cross-section's horizontal axis)
    y — rail longitudinal axis (the sweep direction)
    z — height; the rail crown sits at z = 0, so a 2D cross-section point
        (x2d, y2d) maps to (x = x2d, z = y2d - RAIL_HEIGHT).
The horn antenna and the metasurface plane sit at z > 0 above the crown,
with the incidence tilt in the x-z plane (matching Face3D's
``_incident_direction(theta, 0)``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent            # .../rail3D/rail3d
RAIL3D_DIR = PACKAGE_DIR.parent                          # .../rail3D
REPO_ROOT = RAIL3D_DIR.parent                            # repo root
DATA_DIR = RAIL3D_DIR / "data"
GENERATED_DIR = DATA_DIR / "generated"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
FIGURE_DIR = DATA_DIR / "figures"

CROSSSECTION_IMAGE = REPO_ROOT / "crosssection.png"

# Defect CSV folders live in the external RailDefect folder; mirrors
# raildefect_paths.py at the repo root (duplicated here so the rail3d package
# is importable without sys.path tricks).
# The default is only a convenience for the original laptop; it is used *only*
# when RAILDEFECT_DATA_DIR is unset, so there is no need to edit it on another
# machine — export the variable instead. check_data_dir() below turns a wrong or
# missing value into an actionable error rather than a silent "0 CSVs".
RAILDEFECT_DIR = Path(
    os.environ.get("RAILDEFECT_DATA_DIR", r"C:\Users\Rei\Downloads\RailDefect\RailDefect")
)
DATASET_DIRS = {
    "crack": RAILDEFECT_DIR / "data_defect_crack2",
    "dent": RAILDEFECT_DIR / "data_defect_dent2",
    "wear": RAILDEFECT_DIR / "data_defect_wear2",
}
# Label order: crack=0, dent=1, wear=2, shell=3. "shell" (shelling/spalling) is
# fully parametric — no CSV folder; its cross-section statistics come from
# Ye et al. 2023 Fig 9 (see SHELL_* below).
CLASS_NAMES = ("crack", "dent", "wear", "shell")

# ---------------------------------------------------------------------------
# Physical constants — λ=5 mm (60 GHz) SCALED REPLICA of the validated λ=8
# Face3D "config 55" scene (2026-08-17). Every rig length that Face3D chose in
# units of λ scales with λ (distances, horn, detector windows), so Fresnel
# numbers, speckle statistics and the dark-field geometry are preserved
# EXACTLY; the rail and its defects keep their physical mm sizes, so the
# defect/λ ratio grows by 8/5 = 1.6x — which is the point of the migration.
# LIBRARY_WVL records the band the meta-atom library was fitted at: the fits
# do NOT transfer across wavelength, so MetaUnitSoft refuses WVL != LIBRARY_WVL
# (surface="slm" and "none" are wavelength-agnostic and unaffected).
# ---------------------------------------------------------------------------
WVL = 5.0                       # wavelength in mm (60 GHz)
LIBRARY_WVL = 8.0               # meta-atom fits (library_*.npy) are 8 mm-band
K0 = 2 * np.pi / WVL
DX = WVL / 2                    # plane sampling: 2.5 mm at λ=5

# Rectangular metasurface / observation grid: 60 cells across the railhead
# (x), 30 cells along the rail (y). The aperture (NX·DX x NY·DX) scales with
# λ, which matches the physics: scattered-lobe widths (θ ≈ λ/d) and the
# speckle grain (λL/D) shrink by the same factor, so the smaller plane
# collects the same angular content the 240x120 mm plane did at λ=8.
NX, NY = 60, 30
WX = NX * DX                    # 150 mm aperture across the railhead (240 at λ=8)
WY = NY * DX                    # 75 mm aperture along the rail (120 at λ=8)

# Crown -> metasurface plane distance, in wavelengths. 30λ reproduces the
# measured λ=8 optimum (H=240 mm) exactly in the scaled replica. That optimum
# came from scan_geometry.py (20 samples/class/config) AT λ=8, absolute mm:
#     H     energy   intact det-spread  field AUC  det AUC   crack field AUC
#      80  1.43e-2        0.0203          0.567     0.419        0.595
#     160  1.06e-3        0.0184          0.867     0.819        0.777
#     240  5.16e-4        0.0052          0.899     0.845        0.803   <- chosen
#     320  3.17e-4        0.0083          0.890     0.842        0.767
#     400  2.20e-4        0.0042          0.872     0.856        0.688
#     480  1.60e-4        0.0022          0.869     0.887        0.688
# H=10λ puts the specular lobe INSIDE the aperture (13x more energy) and
# performs below chance: this system works because it is dark-field. Beyond
# 30λ the field AUC (physical information at the plane) falls while the det
# AUC through an UNTRAINED random SLM keeps rising -- trust the field metric.
# The rig scales with λ but the DEFECTS do not, so RE-RUN scan_geometry.py on
# the lab GPU at λ=5 before the full generation to confirm 30λ still wins.
H_MS = 30 * WVL                 # 150 mm at λ=5 (was 240 at λ=8)
# Lateral offset of the plane centre. The horn illuminates at 55 deg from +x,
# so the specular lobe off a flat crown lands at x = -H*tan(55 deg) = -2.14*H
# -- far outside the +-WX/2 aperture at H=30λ. With PLANE_X_CENTER=0 the
# aperture collects the OFF-SPECULAR tail: dark-field, measured (see table)
# to be the reason the system separates classes at all.
PLANE_X_CENTER = 0.0
LAYER_DISTANCES = (20 * WVL,)   # MS -> detector plane: 100 mm at λ=5 (160 at λ=8)

# Horn antenna (pyramidal): Face3D config-55 scaled by WVL/LIBRARY_WVL. The
# λ-scaling keeps the feeding waveguide single-mode-identical (a/λ fixed, so
# the same TE10-only modal content the validated aperture model assumes) and
# preserves the far-field ratio 2A²/(λ·DIST_ANT) — an unscaled 27.4 mm horn at
# 60 GHz would put the rail deep in its radiating near field.
SIZE_ANT = tuple(v * WVL / LIBRARY_WVL
                 for v in (27.4, 21.9, 9.3, 6.2, 27.0))
                                # A, B aperture; a, b waveguide; horn length
DIST_ANT = 28 * WVL             # 140 mm from the crown origin at λ=5
THETA_INC = 55 * np.pi / 180    # incidence angle in the x-z plane
RESOL_ANT = 20                  # 20x20 aperture samples

# Rail geometry
RAIL_HEIGHT = 180.0             # cross-section normalized height (mm)
Z_CUT = -80.0                   # illuminated region: z > -80 mm (2D's y2d > 100 mm)
# Swept segment length along y — PHYSICAL rail, deliberately NOT λ-scaled:
# the defects it must contain keep their mm sizes (50 mm cracks + y0 = ±10 mm
# need ±35 of the ±60 available). Measured truncation study at λ=8 (λ/8 mesh,
# ray-cast shadowing, defect-signal cosine vs a 240 mm reference):
#     240 mm  120480 faces  6.99 s/sample   reference
#     160 mm   80320 faces  3.54 s/sample   cosines 0.9987-0.9998
#     120 mm   60240 faces  2.25 s/sample   cosines 0.9969-0.9990   <- chosen
#      80 mm   40160 faces  1.22 s/sample   cosines 0.959-0.995     <- too short
# Truncation shifts the intact field ~5% at 120 mm, but that is common mode
# (intact and defect samples share the segment), so the defect SIGNATURE is
# preserved better than the mesh-resolution error we already accept. At λ=5
# the illuminated footprint shrinks ∝λ, so 120 mm is SAFER than it was when
# measured. Note it no longer equals the y-aperture (WY = 75 mm at λ=5); the
# mesh extending past the plane is fine — oblique scattering still lands on it.
SEG_LEN = 120.0
SLICE_DS = WVL / 2              # coarse (lambda/2) sampling: occluder meshes, quick tests
N_BOUNDARY_VERTICES = 2400      # matches the 2D pipeline's loop resampling

# Generation mesh fidelity (validated in V6/V7 at λ=8; V7 re-checks at each λ):
#   - lambda/2 meshes are NOT converged (defect-signal cosine 0.66 vs lambda/12);
#   - the rev.2 geometry has ~2 mm-wide hairline cracks, under-resolved by the
#     old lambda/4 mesh -> generation default is lambda/8 (0.625 mm facets at
#     λ=5); the V7 gate re-checks crack samples at lambda/8 vs lambda/16;
#   - ray-cast shadowing changes psi1 by up to 27% on deep defects -> required.
#     Casting against the lambda/2 occluder mesh keeps it cheap.
MESH_DS = WVL / 8               # slice/arc sampling for dataset generation
OCCLUDER_DS = WVL / 2           # coarse occluder mesh for the ray-cast shadow test
SHADOW_MODE = "raycast"

# --- Defect geometry parameters (rev. 2: per-point depth fields d(s, y)) ---
# Baseline ranges came from laser-scan measurements -- Ye et al. 2018 Table 1
# (cracks 27-31 mm long x ~2 mm wide x 3-4.3 mm deep at 45 deg; squats 16-20 mm
# long x 1.9-2.4 mm deep; deep notch 10.3 x 3.1 x 6.9 mm) and Ye et al. 2023
# Fig 9 (shelling ~10 x 12 mm ragged patch, ~2 mm deep, on the head shoulder).
# The values below are the USER'S operating ranges, widened from those
# measurements to cover the defect severities this system must handle; see
# README section 2 for the per-parameter provenance table.
#
# All depths are SAMPLED uniformly from these ranges, never clipped: clipping
# raw CSV depths used to pin 66% of cracks and 49% of dents at the cap. The CSV
# supplies the across-defect profile SHAPE, the sampled value sets its scale.

# y0: the defect's along-track position. The horn boresights the crown at y=0
# and the sensor rides the train along y, so every defect passes through the
# beam center at some frame -- y0 is near 0 by construction, not uniform over
# the aperture. The residual spread models finite capture rate / trigger jitter.
# (This argument does NOT apply to the across-head position s0, which the train
# cannot change and which stays broadly sampled.)
DEFECT_CENTER_RANGE = (-10.0, 10.0)

# Wear: gauge-corner wear develops over long stretches (curves, older rail), so
# the envelope is longer than the modelled segment (SEG_LEN) -- the whole
# modelled rail carries the worn cross-section, with only a slight taper at
# the ends.
DEFECT_LENGTH_RANGE = {"wear": (300.0, 900.0)}

CRACK_LENGTH_RANGE = (10.0, 50.0)     # along the crack line (mm)
CRACK_WIDTH_RANGE = (2.0, 5.0)        # across the line (mm)
CRACK_DEPTH_RANGE = (2.0, 10.0)       # sampled (mm)
DENT_DEPTH_RANGE = (1.5, 8.0)         # sampled (mm)
DENT_FOOTPRINT_Y = (10.0, 30.0)       # FWHM along the rail (mm)
DENT_FOOTPRINT_S = (10.0, 30.0)       # FWHM across the head (mm)
SHELL_RADIUS_RANGE = (4.0, 10.0)      # semi-axes (mm) -> 8-20 mm footprints
SHELL_DEPTH_RANGE = (1.0, 5.0)        # sampled (mm)
# Wear depth is now sampled too (was: raw CSV, which reached 12.5 mm). Deeper
# wear only increases the barcode distance, so training does not need it.
WEAR_DEPTH_RANGE = (2.0, 8.0)         # sampled (mm)

# Surface region bands (arc positions, mm in section coords)
CROWN_HALF_WIDTH = 25.0               # running band: |x| <= 25
GAUGE_X_MIN = 15.0                    # horn-facing shoulder band starts here
GAUGE_X_MAX = 38.0                    # ... and ends at the gauge corner edge

# Augmentation
ROLL_DEG_STD = 2.0              # roll about the y axis (deg, uniform +/-)
JITTER_XZ_STD = 4.0             # rigid x/z jitter (mm, Gaussian)

# Detectors. The window is the Face3D 8 mm-band receiver aperture scaled by
# WVL/LIBRARY_WVL — a band-appropriate 60 GHz receiver — which preserves
# speckle-grains-per-window and therefore the whole detector-design study.
# The starting layout is DENSE (tiling) and gets pruned to N_DET_FINAL, so
# pruning selects from a rich candidate set rather than a handful of fixed
# spots. DET_GRID is DERIVED as the tiling bound floor(aperture/window):
# window-sized pitch is the densest USEFUL start — closer spacing only makes
# duplicates, wider spacing leaves dead zones position gradients cannot cross.
#
#   layout                       n   pitch (mm)   gap (mm)       coverage
#   Face3D 6x6 (λ=8)            36   48.0 x 48.0  +29.8/+36.8       7.2%  (sparse)
#   rail3D old 6x3 (λ=8)        18   36.0 x 36.0  +17.8/+24.8      12.7%  (sparse)
#   rail3D 13x10 (λ=5, now)    130   10.7 x  6.8  -0.66/-0.18      92.0%  (tiling)
#
# (identical n / coverage to the λ=8 dense start — exact scaled replica.)
# A dense start only works with prune_criterion="redundancy": overlapping
# windows have near-identical variance, so the Face3D variance ranking cannot
# tell a duplicate from a uniquely informative detector (see optics3d).
DET_SIZE = (18.2 * WVL / LIBRARY_WVL,
            11.2 * WVL / LIBRARY_WVL)   # 11.375 x 7.0 mm at λ=5
DET_GRID = (int(WX // DET_SIZE[0]),
            int(WY // DET_SIZE[1]))     # tiling bound: (13, 10) = 130 windows
N_DET_FINAL = 8

# Noise model (Face3D values; multiplicative/additive noise is RELATIVE to the
# RMS-normalized field, so it is λ-invariant by construction)
SNR_ADD = 1e-5
SNR_MULTIPLE = 0.005
# Detector-center jitter: a physical mounting tolerance, deliberately NOT
# λ-scaled. Against the smaller λ=5 windows it is now 26%/43% of the window
# (was 16%/27% at λ=8) — training sees a harder, more honest placement task.
DET_JITTER_MM = 3.0

SEED = 0

# Dataset sizes (single source: generator, lab_report and the docs all read these)
FULL_PER_CLASS = 5000
FULL_INTACT = 512
SMOKE_PER_CLASS = 20


def smoke_intact_count(per_class: int = SMOKE_PER_CLASS) -> int:
    """Intact count for a smoke set, keeping the historical 32:20 ratio."""
    return round(per_class * 32 / 20)


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------
def plane_grid(device: torch.device | str = "cpu",
               nx: int | None = None, ny: int | None = None,
               x_center: float | None = None
               ) -> tuple[torch.Tensor, torch.Tensor]:
    """Cell-centered observation-grid coordinates X, Y of shape (1, nx, ny).

    The leading singleton dim matches what the Face3D field functions expect.
    nx/ny/x_center default to the module constants; scan_geometry.py overrides
    them to compare aperture sizes and lateral plane offsets.
    """
    nx = NX if nx is None else nx
    ny = NY if ny is None else ny
    xc = PLANE_X_CENTER if x_center is None else x_center
    wx, wy = nx * DX, ny * DX
    x = torch.arange(-wx / 2 + DX / 2, wx / 2, DX, device=device) + xc
    y = torch.arange(-wy / 2 + DX / 2, wy / 2, DX, device=device)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    return X.reshape(1, nx, ny), Y.reshape(1, nx, ny)


def dense_detector_centers(grid: tuple[int, int] | None = None,
                           nx: int | None = None, ny: int | None = None) -> torch.Tensor:
    """Detector centres on a gx x gy lattice spanning the whole aperture.

    THE single source for detector layouts — every construction of a detector
    grid (training, validation probes, figures, geometry scans) must come
    through here. Its predecessor (`detector_grid_centers`, a 36 mm-pitch
    helper) silently emitted centres outside the aperture once DET_GRID went
    dense; SoftDetector2D then clamped 130 windows onto 54 unique spots.

    Spacing is aperture/(g+1) so windows sit inside the plane rather than on
    its edge. `nx`/`ny` override the aperture cell counts (scan_geometry
    compares alternate planes); they default to the module constants.
    """
    gx, gy = grid if grid is not None else DET_GRID
    nx = NX if nx is None else nx
    ny = NY if ny is None else ny
    px = (nx * DX) / (gx + 1)
    py = (ny * DX) / (gy + 1)
    cx = (torch.arange(gx) - (gx - 1) / 2) * px
    cy = (torch.arange(gy) - (gy - 1) / 2) * py
    CX, CY = torch.meshgrid(cx, cy, indexing="ij")
    return torch.stack([CX.reshape(-1), CY.reshape(-1)], dim=1)


# ---------------------------------------------------------------------------
# Devices / profiles
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Profile:
    name: str
    device: str
    batch_meshes: int      # meshes solved simultaneously in generation
    chunk_faces: int       # face-chunk size in the surface integral
    train_batch: int

PROFILES = {
    # This laptop: GeForce MX250, 2 GB VRAM — keep every intermediate small.
    "laptop": Profile("laptop", "cuda:0", batch_meshes=2, chunk_faces=1024, train_batch=256),
    # Lab workstation: the big card (RTX 5090). Its index differs per machine,
    # so "cuda:auto" picks the strongest CUDA device instead of guessing.
    "lab": Profile("lab", "cuda:auto", batch_meshes=64, chunk_faces=8192, train_batch=1024),
    # CPU fallback used by the verification suite.
    "cpu": Profile("cpu", "cpu", batch_meshes=1, chunk_faces=512, train_batch=64),
}


def best_cuda_device() -> torch.device:
    """The strongest visible CUDA device: highest compute capability, then most VRAM.

    Multi-GPU boxes do not agree on which index holds the big card (on the lab
    workstation the 5090 is cuda:0, elsewhere it may be cuda:1), so ranking by
    capability beats hard-coding an index. Set RAIL3D_DEVICE to override.
    """
    n = torch.cuda.device_count()
    if n == 0:
        raise RuntimeError("no CUDA devices visible")
    ranked = sorted(
        range(n),
        key=lambda i: (torch.cuda.get_device_capability(i),
                       torch.cuda.get_device_properties(i).total_memory),
        reverse=True,
    )
    idx = ranked[0]
    if n > 1:
        print(f"[rail3d] cuda:auto -> cuda:{idx} ({torch.cuda.get_device_name(idx)}, "
              f"sm_{''.join(map(str, torch.cuda.get_device_capability(idx)))}, "
              f"{torch.cuda.get_device_properties(idx).total_memory / 1e9:.0f} GB) "
              f"out of {n} devices")
    return torch.device(f"cuda:{idx}")


def get_device(profile: str | None = None) -> torch.device:
    """Resolve the compute device.

    Priority: RAIL3D_DEVICE env var > profile's device > cpu. Accepts
    "cuda:auto" (strongest card, see best_cuda_device). Never returns a bare
    "cuda" so multi-GPU boxes behave predictably. Falls back to CPU with a
    warning if CUDA is unavailable.
    """
    requested = os.environ.get("RAIL3D_DEVICE")
    if requested is None and profile is not None:
        requested = PROFILES[profile].device
    if requested is None:
        requested = "cpu"

    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            print(f"[rail3d] {requested} requested but CUDA is unavailable; using CPU")
            return torch.device("cpu")
        device = best_cuda_device() if requested in ("cuda:auto", "cuda") else torch.device(requested)
        check_cuda_build(device)
        return device
    return torch.device(requested)


def check_cuda_build(device: torch.device) -> None:
    """Fail fast with a clear message when the torch build can't drive the GPU.

    The lab RTX 5090 is Blackwell (sm_120): a +cu118 wheel raises cryptic
    "no kernel image" errors at first use. Surface that as an actionable error.
    """
    idx = device.index if device.index is not None else 0
    major, minor = torch.cuda.get_device_capability(idx)
    name = torch.cuda.get_device_name(idx)
    try:
        (torch.zeros(1, device=device) + 1).item()
    except RuntimeError as err:
        raise RuntimeError(
            f"torch {torch.__version__} cannot execute on {name} (sm_{major}{minor}). "
            f"On the RTX 5090 install a cu128 build:\n"
            f"  pip install torch --index-url https://download.pytorch.org/whl/cu128\n"
            f"(see rail3D/SETUP_LAB.md). Original error: {err}"
        ) from err


def seeded_generator(seed: int, device: torch.device | str = "cpu") -> torch.Generator:
    gen = torch.Generator(device=str(device) if torch.device(device).type == "cpu" else device)
    gen.manual_seed(seed)
    return gen


def sample_seed(class_name: str, csv_index: int) -> int:
    """Deterministic per-sample seed so interrupted generation regenerates
    bit-identical samples on any machine."""
    base = {"intact": 0, "crack": 1, "dent": 2, "wear": 3, "shell": 4}[class_name]
    return (base * 1_000_003 + csv_index * 7919 + SEED) % (2**31 - 1)


def ensure_dirs() -> None:
    for d in (GENERATED_DIR, CHECKPOINT_DIR, FIGURE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def check_data_dir(class_name: str | None = None) -> None:
    """Validate RAILDEFECT_DATA_DIR and raise something actionable if it is wrong.

    Called by ``sections.get_dataset_files`` when a defect folder yields no CSVs,
    so a mis-set path surfaces as an explanation instead of a downstream
    IndexError or "requested N but only 0 CSVs".
    """
    raw = os.environ.get("RAILDEFECT_DATA_DIR")
    lines = [
        f"Defect CSVs not found under RAILDEFECT_DIR = {RAILDEFECT_DIR}",
        f"  RAILDEFECT_DATA_DIR = {raw!r}" if raw else
        "  RAILDEFECT_DATA_DIR is UNSET — falling back to the original laptop path.",
    ]
    if raw and not Path(raw).is_absolute():
        # 'C:Users\...' (no separator after the drive) is drive-RELATIVE on Windows
        lines.append("  -> that path is not absolute. A Windows drive needs a separator: "
                     "'C:/Users/...' or 'C:\\Users\\...', not 'C:Users\\...'.")
    if raw and raw.startswith("/") and ":" not in raw:
        lines.append("  -> looks like an MSYS path (/c/Users/...). Git Bash does not "
                     "translate it for exported variables; use 'C:/Users/...'.")
    if not RAILDEFECT_DIR.exists():
        lines.append("  -> that directory does not exist.")
    else:
        missing = [n for n, d in DATASET_DIRS.items() if not d.is_dir()]
        present = [n for n, d in DATASET_DIRS.items() if d.is_dir()]
        if missing:
            lines.append(f"  -> directory exists but is missing: "
                         f"{', '.join(f'data_defect_{n}2' for n in missing)}"
                         + (f" (found: {', '.join(present)})" if present else ""))
            lines.append("  -> RAILDEFECT_DATA_DIR must point at the PARENT of the three "
                         "data_defect_*2 folders, not at one of them.")
        elif class_name:
            lines.append(f"  -> data_defect_{class_name}2 exists but contains no *.csv "
                         "(copy still running?).")
    lines.append("  See rail3D/SETUP_LAB.md section 3 for per-shell syntax "
                 "(export / $env: / set).")
    raise FileNotFoundError("\n".join(lines))
