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
# Physical constants (Face3D "config 55" standard, so the meta-atom library
# fits transfer unchanged)
# ---------------------------------------------------------------------------
WVL = 8.0                       # wavelength in mm (37.5 GHz)
K0 = 2 * np.pi / WVL
DX = WVL / 2                    # 4 mm sampling on the metasurface plane

# Rectangular metasurface / observation grid: 60 cells across the railhead
# (x), 30 cells along the rail (y). Defect features are 1-10 mm, so the
# smaller aperture keeps the simulation fast while covering the scattered
# lobes.
NX, NY = 60, 30
WX = NX * DX                    # 240 mm aperture across the railhead
WY = NY * DX                    # 120 mm aperture along the rail

H_MS = 160.0                    # crown -> metasurface plane distance (20 wvl)
LAYER_DISTANCES = (160.0,)      # MS -> detector plane; extend for 2-layer runs

# Horn antenna (pyramidal), Face3D config-55 verbatim
SIZE_ANT = (27.4, 21.9, 9.3, 6.2, 27.0)   # A, B aperture; a, b waveguide; horn length
DIST_ANT = 28 * WVL             # 224 mm from the crown origin
THETA_INC = 55 * np.pi / 180    # incidence angle in the x-z plane
RESOL_ANT = 20                  # 20x20 aperture samples

# Rail geometry
RAIL_HEIGHT = 180.0             # cross-section normalized height (mm)
Z_CUT = -80.0                   # illuminated region: z > -80 mm (2D's y2d > 100 mm)
SEG_LEN = 240.0                 # swept segment length along y (2x the y-aperture)
SLICE_DS = WVL / 2              # coarse (lambda/2) sampling: occluder meshes, quick tests
N_BOUNDARY_VERTICES = 2400      # matches the 2D pipeline's loop resampling

# Generation mesh fidelity (validated in V6/V7):
#   - lambda/2 meshes are NOT converged (defect-signal cosine 0.66 vs lambda/12);
#   - the rev.2 geometry has ~2 mm-wide hairline cracks, under-resolved by the
#     old lambda/4 mesh -> generation default is now lambda/8 (1 mm facets);
#     the V7 convergence gate re-checks crack samples at lambda/8 vs lambda/16;
#   - ray-cast shadowing changes psi1 by up to 27% on deep defects -> required.
#     Casting against the lambda/2 occluder mesh keeps it cheap.
MESH_DS = WVL / 8               # 1 mm slice/arc sampling for dataset generation
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
# the envelope is longer than the 240 mm segment -- the whole modelled rail
# carries the worn cross-section, with only a slight taper at the ends.
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

# Detectors: 6x3 grid over the 240x120 mm aperture, pruned to N_DET_FINAL
DET_SIZE = (18.2, 11.2)         # window size in mm (Face3D waveguide aperture)
DET_GRID = (6, 3)
DET_PITCH = (36.0, 36.0)        # grid pitch in mm (x, y)
N_DET_FINAL = 8

# Noise model (Face3D values)
SNR_ADD = 1e-5
SNR_MULTIPLE = 0.005
DET_JITTER_MM = 3.0             # detector-center jitter during training

SEED = 0


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------
def plane_grid(device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Cell-centered observation-grid coordinates X, Y of shape (1, NX, NY).

    The leading singleton dim matches what the Face3D field functions expect.
    """
    x = torch.arange(-WX / 2 + DX / 2, WX / 2, DX, device=device)
    y = torch.arange(-WY / 2 + DX / 2, WY / 2, DX, device=device)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    return X.reshape(1, NX, NY), Y.reshape(1, NX, NY)


def detector_grid_centers() -> torch.Tensor:
    """Initial detector centers, shape (N_det, 2) in mm: 6x3 grid, 36 mm pitch."""
    nx, ny = DET_GRID
    px, py = DET_PITCH
    cx = (torch.arange(nx) - (nx - 1) / 2) * px
    cy = (torch.arange(ny) - (ny - 1) / 2) * py
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
