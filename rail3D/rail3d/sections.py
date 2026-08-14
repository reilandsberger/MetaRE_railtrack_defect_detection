"""2D cross-section utilities, adapted from ``2Dmesh_from_vertex.py``.

Differences from the original module:
  * everything is in **mm** (the original worked in micrometres),
  * nothing executes at import time (the original loaded the reference image
    and ran the solver on import),
  * only the boundary/CSV geometry helpers are kept — the 2D wave solver and
    shadow masks stay in the original script (used by the V5 cross-check).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from . import config


# ---------------------------------------------------------------------------
# Image / curve helpers (logic identical to 2Dmesh_from_vertex.py)
# ---------------------------------------------------------------------------
def read_binary_image(path: str | Path, foreground: str = "light") -> np.ndarray:
    """Convert the intact rail reference image into a boolean mask.

    Uses plt.imread exactly like the 2D pipeline: crosssection.png is RGBA and
    matplotlib resolves transparent pixels to white, which the dark-foreground
    threshold relies on (PIL would return black there and invert the mask).
    """
    import matplotlib.pyplot as plt

    image = plt.imread(Path(path))
    if image.ndim == 3:
        image = image[..., :3].mean(axis=2)
    if image.dtype != np.uint8:
        image = (255 * image).astype(np.uint8)

    if foreground == "light":
        return image > 127
    if foreground == "dark":
        return image < 127
    raise ValueError(f"Unsupported foreground mode: {foreground}")


def polygon_area(vertices: np.ndarray) -> float:
    x, y = vertices[:, 0], vertices[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def extract_boundary(mask: np.ndarray) -> np.ndarray:
    """Extract the longest closed contour from a binary mask (matplotlib contour)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    contour_set = ax.contour(mask.astype(float), levels=[0.5])
    plt.close(fig)

    boundaries = []
    if hasattr(contour_set, "collections") and contour_set.collections:
        for collection in contour_set.collections:
            for path in collection.get_paths():
                if path.vertices.shape[0] > 0:
                    boundaries.append(path.vertices)
    elif hasattr(contour_set, "allsegs") and contour_set.allsegs:
        for level_segments in contour_set.allsegs:
            for segment in level_segments:
                if len(segment) > 0:
                    boundaries.append(np.asarray(segment))

    if not boundaries:
        raise ValueError("No foreground boundary found in image")

    boundary = max(boundaries, key=lambda v: v.shape[0])
    if np.allclose(boundary[0], boundary[-1]):
        boundary = boundary[:-1]
    if polygon_area(boundary) < 0:
        boundary = boundary[::-1]
    return boundary


def resample_closed_curve(vertices: np.ndarray, target_count: int) -> np.ndarray:
    """Redistribute samples uniformly along arc length on a closed curve."""
    closed = np.vstack([vertices, vertices[:1]])
    seg_len = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_len)])
    perimeter = cumulative[-1]

    samples = np.linspace(0.0, perimeter, target_count + 1)[:-1]
    x = np.interp(samples, cumulative, closed[:, 0])
    y = np.interp(samples, cumulative, closed[:, 1])
    return np.column_stack([x, y])


def resample_open_curve(vertices: np.ndarray, target_count: int) -> np.ndarray:
    """Uniform arc-length resampling of an open polyline (endpoints kept)."""
    seg_len = np.linalg.norm(np.diff(vertices, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_len)])
    samples = np.linspace(0.0, cumulative[-1], target_count)
    x = np.interp(samples, cumulative, vertices[:, 0])
    y = np.interp(samples, cumulative, vertices[:, 1])
    return np.column_stack([x, y])


def smooth_closed_curve(vertices: np.ndarray, window: int) -> np.ndarray:
    """Circular moving-average smoothing preserving closed-curve continuity."""
    if window <= 1:
        return vertices
    if window % 2 == 0:
        window += 1
    radius = window // 2
    padded = np.pad(vertices, ((radius, radius), (0, 0)), mode="wrap")
    kernel = np.ones(window, dtype=float) / window
    smoothed = np.empty_like(vertices, dtype=float)
    for dim in range(vertices.shape[1]):
        smoothed[:, dim] = np.convolve(padded[:, dim], kernel, mode="valid")
    return smoothed


def reorder_boundary_start(vertices: np.ndarray) -> np.ndarray:
    """Rotate the loop so indexing starts near the centered bottom point."""
    y, x = vertices[:, 1], vertices[:, 0]
    candidates = np.flatnonzero(np.isclose(y, y.min()))
    start = candidates[np.argmin(np.abs(x[candidates]))]
    return np.roll(vertices, -start, axis=0)


def normalize_boundary(
    vertices_px: np.ndarray,
    target_height: float = config.RAIL_HEIGHT,
    smooth_window: int = 9,
) -> torch.Tensor:
    """Center, flip, scale (to mm), and smooth image/CSV-space vertices.

    Output: (N, 2) float32 tensor, x centered, y in [0, target_height] mm.
    """
    x = vertices_px[:, 0] - 0.5 * (vertices_px[:, 0].min() + vertices_px[:, 0].max())
    y = vertices_px[:, 1].max() - vertices_px[:, 1]
    y = y - y.min()

    vertices = np.column_stack([x, y])
    vertices = vertices * (target_height / vertices[:, 1].max())
    vertices = reorder_boundary_start(vertices)
    vertices = smooth_closed_curve(vertices, window=smooth_window)
    vertices = reorder_boundary_start(vertices)
    return torch.tensor(vertices, dtype=torch.float32)


def match_reference_width(vertices: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Rescale a defect boundary so its width matches the intact reference."""
    width = vertices[:, 0].max() - vertices[:, 0].min()
    ref_width = reference[:, 0].max() - reference[:, 0].min()
    return vertices * (ref_width / width)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def read_vertices_csv(path: str | Path) -> np.ndarray:
    """Read ordered rail-surface vertices from a defect CSV (x/y columns)."""
    path = Path(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = {name.strip().lower(): name for name in (reader.fieldnames or [])}
        if "x" not in fieldnames or "y" not in fieldnames:
            raise ValueError(f"{path} must contain x and y columns")
        vertices = [
            [float(row[fieldnames["x"]]), float(row[fieldnames["y"]])] for row in reader
        ]
    if not vertices:
        raise ValueError(f"{path} does not contain any vertices")

    boundary = np.asarray(vertices, dtype=np.float64)
    if np.allclose(boundary[0], boundary[-1]):
        boundary = boundary[:-1]
    if polygon_area(boundary) < 0:
        boundary = boundary[::-1]
    return boundary


def load_vertices_from_csv(
    path: str | Path,
    target_count: int = config.N_BOUNDARY_VERTICES,
    smooth_window: int = 9,
) -> torch.Tensor:
    boundary = read_vertices_csv(path)
    boundary = resample_closed_curve(boundary, target_count=target_count)
    return normalize_boundary(boundary, smooth_window=smooth_window)


def load_reference_section(
    image_path: str | Path = config.CROSSSECTION_IMAGE,
    target_count: int = config.N_BOUNDARY_VERTICES,
    smooth_window: int = 31,
) -> torch.Tensor:
    """The intact rail cross-section loop, (2400, 2) float32 in mm.

    Same parameters as the 2D pipeline (smooth_window=31 for the reference).
    """
    mask = read_binary_image(image_path, foreground="dark")
    boundary = extract_boundary(mask)
    boundary = resample_closed_curve(boundary, target_count=target_count)
    return normalize_boundary(boundary, smooth_window=smooth_window)


def illuminated_mask(section: torch.Tensor, z_cut: float = config.Z_CUT) -> torch.Tensor:
    """Boolean mask over loop indices for the illuminated upper region.

    z_cut is in crown coordinates (z = y2d - RAIL_HEIGHT); the 2D pipeline's
    ``y2d > 100 mm`` corresponds to z_cut = -80 mm.
    """
    return section[:, 1] - config.RAIL_HEIGHT > z_cut


def get_dataset_files(class_name: str) -> list[Path]:
    """Sorted defect-CSV paths for one class.

    Empty result means RAILDEFECT_DATA_DIR is unset/wrong (or the copy is still
    running); check_data_dir explains which, instead of letting the caller fail
    later with an opaque IndexError.
    """
    if class_name not in config.DATASET_DIRS:
        raise ValueError(f"Unknown dataset: {class_name}")
    files = sorted(config.DATASET_DIRS[class_name].glob("*.csv"))
    if not files:
        config.check_data_dir(class_name)
    return files
