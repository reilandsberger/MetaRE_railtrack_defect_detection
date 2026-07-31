#%%
from pathlib import Path
import csv
from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import numpy as np
import plot_setting
import torch
from tqdm import tqdm

Hankel = lambda x: torch.special.bessel_j1(x) + 1j * torch.special.bessel_y1(x)


mm = 1e3
height = 180 * mm
wvl = 12 * mm
k0 = 2 * np.pi / wvl

sim_size = (360 * mm, 300 * mm, 5 * wvl)
N_BOUNDARY_VERTICES = 2400
DEFAULT_SMOOTH_WINDOW = 9
SHADOW_SAMPLES = 15

# Defect CSV folders live in the external RailDefect folder (override with RAILDEFECT_DATA_DIR).
from raildefect_paths import RAILDEFECT_DIR

DATASET_DIRS = {
    "crack": RAILDEFECT_DIR / "data_defect_crack2",
    "dent": RAILDEFECT_DIR / "data_defect_dent2",
    "wear": RAILDEFECT_DIR / "data_defect_wear2",
}


def get_compute_device(preferred: str | torch.device | None = None) -> torch.device:
    # Resolve the requested accelerator, defaulting to MPS when it is available.
    if preferred is None:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(preferred)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available on this machine")
    return device


def read_binary_image(path: str | Path, foreground: str = "light") -> np.ndarray:
    # Convert the intact rail reference image into a boolean mask.
    path = Path(path)
    image = plt.imread(path)
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
    # Compute signed polygon area to determine curve orientation.
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def extract_boundary(mask: np.ndarray) -> np.ndarray:
    # Extract the longest closed contour from a binary mask as ordered boundary vertices.
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

    boundary = max(boundaries, key=lambda vertices: vertices.shape[0])

    if np.allclose(boundary[0], boundary[-1]):
        boundary = boundary[:-1]

    if polygon_area(boundary) < 0:
        boundary = boundary[::-1]

    return boundary


def resample_closed_curve(vertices: np.ndarray, target_count: int) -> np.ndarray:
    # Redistribute boundary samples uniformly along arc length on a closed curve.
    closed_vertices = np.vstack([vertices, vertices[:1]])
    segment_lengths = np.linalg.norm(np.diff(closed_vertices, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    perimeter = cumulative[-1]

    sample_points = np.linspace(0.0, perimeter, target_count + 1)[:-1]
    x = np.interp(sample_points, cumulative, closed_vertices[:, 0])
    y = np.interp(sample_points, cumulative, closed_vertices[:, 1])
    return np.column_stack([x, y])


def smooth_closed_curve(vertices: np.ndarray, window: int) -> np.ndarray:
    # Apply circular moving-average smoothing while preserving closed-curve continuity.
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
    # Rotate the closed boundary so indexing starts near the centered bottom point.
    y = vertices[:, 1]
    x = vertices[:, 0]
    candidate_idx = np.flatnonzero(np.isclose(y, y.min()))
    start_idx = candidate_idx[np.argmin(np.abs(x[candidate_idx]))]
    return np.roll(vertices, -start_idx, axis=0)


def normalize_boundary(
    vertices_px: np.ndarray,
    target_height: float = height,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
) -> torch.Tensor:
    # Center, flip, scale, and smooth image-space vertices into the rail coordinate system.
    x = vertices_px[:, 0] - 0.5 * (vertices_px[:, 0].min() + vertices_px[:, 0].max())
    y = vertices_px[:, 1].max() - vertices_px[:, 1]
    y = y - y.min()

    vertices = np.column_stack([x, y])
    vertices = vertices * (target_height / vertices[:, 1].max())
    vertices = reorder_boundary_start(vertices)
    vertices = smooth_closed_curve(vertices, window=smooth_window)
    vertices = reorder_boundary_start(vertices)

    return torch.tensor(vertices, dtype=torch.float32)


def match_reference_width(vertices: torch.Tensor, reference_vertices: torch.Tensor) -> torch.Tensor:
    # Rescale a defect boundary so its overall width matches the intact reference width.
    width = vertices[:, 0].max() - vertices[:, 0].min()
    reference_width = reference_vertices[:, 0].max() - reference_vertices[:, 0].min()
    return vertices * (reference_width / width)


def load_vertices_from_image(
    path: str | Path,
    foreground: str,
    target_count: int = N_BOUNDARY_VERTICES,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
) -> torch.Tensor:
    # Load one image and convert its binary silhouette into a normalized boundary vertex loop.
    mask = read_binary_image(path, foreground=foreground)
    boundary = extract_boundary(mask)
    boundary = resample_closed_curve(boundary, target_count=target_count)
    return normalize_boundary(boundary, smooth_window=smooth_window)


def read_vertices_csv(path: str | Path) -> np.ndarray:
    # Read ordered rail-surface vertices from CSV and return them as x/y pairs.
    path = Path(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = {name.strip().lower(): name for name in (reader.fieldnames or [])}
        if "x" not in fieldnames or "y" not in fieldnames:
            raise ValueError(f"{path} must contain x and y columns")

        vertices = [
            [float(row[fieldnames["x"]]), float(row[fieldnames["y"]])]
            for row in reader
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
    target_count: int = N_BOUNDARY_VERTICES,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
) -> torch.Tensor:
    # Load one CSV boundary and normalize it into the rail coordinate system.
    boundary = read_vertices_csv(path)
    boundary = resample_closed_curve(boundary, target_count=target_count)
    return normalize_boundary(boundary, smooth_window=smooth_window)


def compute_line_of_sight_mask(
    boundary_vertices: torch.Tensor,
    origins: torch.Tensor,
    targets: torch.Tensor,
    samples: int = SHADOW_SAMPLES,
) -> torch.Tensor:
    # A ray is visible only if the open segment between the two points stays outside the rail polygon.
    if samples < 2:
        raise ValueError("samples must be at least 2")

    boundary_np = boundary_vertices.detach().cpu().numpy()
    origins_np = origins.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    polygon = MplPath(boundary_np, closed=True)

    t = np.linspace(0.0, 1.0, samples + 2, dtype=np.float32)[1:-1]
    segment_samples = (
        origins_np[:, None, None, :]
        + (targets_np[None, :, None, :] - origins_np[:, None, None, :]) * t[None, None, :, None]
    )
    blocked = polygon.contains_points(segment_samples.reshape(-1, 2))
    blocked = blocked.reshape(origins_np.shape[0], targets_np.shape[0], samples).any(axis=2)
    return torch.from_numpy(~blocked).to(device=boundary_vertices.device)


def compute_line_of_sight_mask_v2(
    boundary_vertices: torch.Tensor,
    origins: torch.Tensor,
    targets: torch.Tensor,
    samples: int = SHADOW_SAMPLES,
) -> torch.Tensor:
    # Torch-native point-in-polygon visibility test so the shadow mask can stay on-device.
    if samples < 2:
        raise ValueError("samples must be at least 2")

    dtype = boundary_vertices.dtype
    device = boundary_vertices.device
    origins = origins.to(device=device, dtype=dtype)
    targets = targets.to(device=device, dtype=dtype)

    t = torch.linspace(0.0, 1.0, samples + 2, device=device, dtype=dtype)[1:-1]
    segment_samples = (
        origins[:, None, None, :]
        + (targets[None, :, None, :] - origins[:, None, None, :]) * t[None, None, :, None]
    )

    points = segment_samples.reshape(-1, 2)
    x = points[:, 0:1]
    y = points[:, 1:2]

    x0 = boundary_vertices[:, 0]
    y0 = boundary_vertices[:, 1]
    x1 = torch.roll(x0, shifts=-1, dims=0)
    y1 = torch.roll(y0, shifts=-1, dims=0)

    # Ray casting against all polygon edges at once.
    intersects = (y0.unsqueeze(0) > y) != (y1.unsqueeze(0) > y)
    denom = y1 - y0
    safe_denom = torch.where(
        torch.abs(denom) < torch.finfo(dtype).eps,
        torch.ones_like(denom),
        denom,
    )
    x_intersections = x0.unsqueeze(0) + (x1 - x0).unsqueeze(0) * (y - y0.unsqueeze(0)) / safe_denom.unsqueeze(0)
    crossings = intersects & (x < x_intersections)
    blocked = crossings.sum(dim=1).remainder(2).to(torch.bool)
    blocked = blocked.reshape(origins.shape[0], targets.shape[0], samples).any(dim=2)
    return ~blocked


def compute_line_of_sight_mask_v3(
    boundary_vertices: torch.Tensor,
    origins: torch.Tensor,
    targets: torch.Tensor,
    samples: int = SHADOW_SAMPLES,
) -> torch.Tensor:
    # Segment-edge intersection test that avoids per-ray interior sampling.
    del samples

    dtype = boundary_vertices.dtype
    device = boundary_vertices.device
    origins = origins.to(device=device, dtype=dtype)
    targets = targets.to(device=device, dtype=dtype)

    seg_start = origins[:, None, :]
    seg_end = targets[None, :, :]
    seg_dir = seg_end - seg_start

    edge_start = boundary_vertices
    edge_end = torch.roll(boundary_vertices, shifts=-1, dims=0)
    edge_dir = edge_end - edge_start

    p = seg_start[:, :, None, :]
    r = seg_dir[:, :, None, :]
    q = edge_start[None, None, :, :]
    s = edge_dir[None, None, :, :]

    def cross2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    qp = q - p
    r_cross_s = cross2d(r, s)
    qp_cross_r = cross2d(qp, r)
    qp_cross_s = cross2d(qp, s)

    eps = torch.tensor(1e-6, device=device, dtype=dtype)
    non_parallel = torch.abs(r_cross_s) > eps

    t = torch.where(non_parallel, qp_cross_s / r_cross_s, torch.zeros_like(r_cross_s))
    u = torch.where(non_parallel, qp_cross_r / r_cross_s, torch.zeros_like(r_cross_s))

    # Ignore the origin and target endpoints so touching the boundary there is still considered visible.
    proper_intersection = (
        non_parallel
        & (t > eps)
        & (t < 1 - eps)
        & (u > eps)
        & (u < 1 - eps)
    )
    blocked = proper_intersection.any(dim=2)
    return ~blocked


def get_dataset_files(dataset_name: str) -> list[Path]:
    # Return the sorted CSV paths for one defect dataset.
    if dataset_name not in DATASET_DIRS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return sorted(DATASET_DIRS[dataset_name].glob("*.csv"))


def normalize_dataset_names(dataset_name: str | Sequence[str]) -> tuple[str, ...]:
    # Convert dataset input into a validated tuple of dataset names.
    if isinstance(dataset_name, str):
        dataset_names = (dataset_name,)
    else:
        dataset_names = tuple(dataset_name)
    if len(dataset_names) == 0:
        raise ValueError("At least one dataset name is required")
    for name in dataset_names:
        if name not in DATASET_DIRS:
            raise ValueError(f"Unknown dataset: {name}")
    return dataset_names


def load_dataset_vertices(
    dataset_name: str | Sequence[str],
    limit: int | None = None,
    target_count: int = N_BOUNDARY_VERTICES,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
    rail_mask: torch.Tensor | None = None,
    reference_vertices: torch.Tensor | None = None,
) -> tuple[list[Path] | list[tuple[Path, ...]], torch.Tensor, torch.Tensor]:
    # Convert one or more defect datasets into boundary loops and the upper rail segment used in simulation.
    dataset_names = normalize_dataset_names(dataset_name)
    file_lists = [get_dataset_files(name) for name in dataset_names]
    num_samples = min(len(files) for files in file_lists)
    if limit is not None:
        num_samples = min(num_samples, limit)

    if len(dataset_names) != 1:
        raise ValueError("CSV vertex datasets must be loaded one dataset at a time")

    file_groups = file_lists[0][:num_samples]

    vertices_list = [
        load_vertices_from_csv(
            file_group,
            target_count=target_count,
            smooth_window=smooth_window,
        )
        for file_group in tqdm(file_groups, desc=f"{'+'.join(dataset_names)} boundaries")
    ]
    if reference_vertices is not None:
        vertices_list = [match_reference_width(vertices, reference_vertices) for vertices in vertices_list]
    vertices_batch = torch.stack(vertices_list, dim=0)
    if rail_mask is None:
        rail_mask = REFERENCE_RAIL_MASK if "REFERENCE_RAIL_MASK" in globals() else (vertices_batch[0, :, 1] > 100 * mm)
    v_rail_batch = vertices_batch[:, rail_mask]
    return file_groups, vertices_batch, v_rail_batch


def generate_dataset_fields(
    dataset_name: str | Sequence[str],
    limit: int | None = None,
    batch_size: int = 64,
    device: str | torch.device | None = None,
) -> tuple[list[Path] | list[tuple[Path, ...]], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Run the wave solver over one or more aligned defect datasets in mini-batches and collect the resulting fields.
    dataset_names = normalize_dataset_names(dataset_name)
    compute_device = get_compute_device(device)
    files, vertices_batch, v_rail_batch = load_dataset_vertices(
        dataset_names,
        limit=limit,
        reference_vertices=vertices,
    )

    psi_rail_list = []
    psi_ms_list = []
    for start in tqdm(
        range(0, len(v_rail_batch), batch_size),
        desc=f"{'+'.join(dataset_names)} field batches",
    ):
        stop = start + batch_size
        boundary_batch = vertices_batch[start:stop].to(device=compute_device)
        psi_rail_batch, psi_ms_batch = field(boundary_batch)
        psi_rail_list.append(psi_rail_batch.cpu())
        psi_ms_list.append(psi_ms_batch.cpu())

    psi_rail = torch.cat(psi_rail_list, dim=0)
    psi_ms = torch.cat(psi_ms_list, dim=0)
    return files, vertices_batch, v_rail_batch, psi_rail, psi_ms


## Original railroad shape
vertices = load_vertices_from_image(
    "crosssection.png",
    foreground="dark",
    target_count=N_BOUNDARY_VERTICES,
    smooth_window=31,
)
v_rail = vertices[vertices[:, 1] > 100 * mm]  ## Counter clock-wise
REFERENCE_RAIL_MASK = vertices[:, 1] > 100 * mm


## source
A_aptr, B_aptr, a_wvg, b_wvg, l_horn = 27.4 * mm, 21.9 * mm, 9.3 * mm, 6.2 * mm, 27 * mm
R_E = A_aptr * l_horn / (A_aptr - a_wvg)
beta_wvg = k0 * np.sqrt(1 - (wvl / (2 * a_wvg)) ** 2)
dist = 160 * mm
theta = 60 * np.pi / 180
norm_vec_src = torch.tensor([[np.sin(theta, dtype=np.float32), -np.cos(theta, dtype=np.float32)]])
dx_src = A_aptr / 50
s = torch.arange(-A_aptr / 2, A_aptr / 2, dx_src) + dx_src / 2
v_src_x = -dist * np.sin(theta) + np.cos(theta) * s
v_src_y = 150 * mm + dist * np.cos(theta) + np.sin(theta) * s
v_src = torch.stack([v_src_x, v_src_y], dim=1)
amp = torch.cos(np.pi * s / A_aptr)
psi_src = amp * torch.exp(0.5j * beta_wvg * (s**2 / R_E))

## metasurface plane:
y_ms = 250 * mm
W = 20 * wvl
dx_ms = wvl / 10
v_ms_x = torch.arange(-W / 2, W / 2, dx_ms) + dx_ms / 2
v_ms_y = torch.zeros_like(v_ms_x) + y_ms
v_ms = torch.stack([v_ms_x, v_ms_y], dim=1)


def field(
    boundary_vertices: torch.Tensor,
    rail_mask: torch.Tensor | None = None,
    apply_shadow: bool = True,
    shadow_samples: int = SHADOW_SAMPLES,
    shadow_method: str = "segment",
) -> tuple[torch.Tensor, torch.Tensor]:
    # Propagate the source field to the visible rail boundary and then from the visible rail to the metasurface plane.
    single_input = boundary_vertices.dim() == 2
    if single_input:
        boundary_vertices = boundary_vertices.unsqueeze(dim=0)  # batch * points * xy

    if rail_mask is None:
        rail_mask = REFERENCE_RAIL_MASK
    rail_mask = rail_mask.to(device=boundary_vertices.device)

    local_v_src = v_src.to(device=boundary_vertices.device, dtype=boundary_vertices.dtype)
    local_v_ms = v_ms.to(device=boundary_vertices.device, dtype=boundary_vertices.dtype)
    local_psi_src = psi_src.to(device=boundary_vertices.device)
    local_norm_vec_src = norm_vec_src.to(device=boundary_vertices.device, dtype=boundary_vertices.dtype)

    if shadow_method == "mpl":
        shadow_fn = compute_line_of_sight_mask
    elif shadow_method == "torch":
        shadow_fn = compute_line_of_sight_mask_v2
    elif shadow_method == "segment":
        shadow_fn = compute_line_of_sight_mask_v3
    else:
        raise ValueError(f"Unsupported shadow_method: {shadow_method}")

    v_rail = boundary_vertices[:, rail_mask]

    center = (v_rail[:, :-1] + v_rail[:, 1:]) / 2
    dx_vec = v_rail[:, 1:] - v_rail[:, :-1]
    dx = torch.linalg.norm(dx_vec, dim=2)
    norm_vec = torch.stack([dx_vec[:, :, 1], -dx_vec[:, :, 0]], dim=2)
    norm_vec = norm_vec / torch.linalg.norm(norm_vec, dim=2, keepdim=True)

    ## source -> rail
    R_vec = center.unsqueeze(dim=2) - local_v_src.reshape(1, 1, *local_v_src.shape)
    R_scalar = torch.linalg.norm(R_vec, dim=3)
    R_unit = R_vec / R_scalar.unsqueeze(dim=3)
    illum_factor = torch.sum(-R_vec * norm_vec.unsqueeze(dim=2), dim=3) < 0
    if apply_shadow:
        shadow_mask = torch.stack(
            [
                shadow_fn(
                    boundary_vertices[idx],
                    local_v_src,
                    center[idx],
                    samples=shadow_samples,
                ).T
                for idx in range(boundary_vertices.shape[0])
            ],
            dim=0,
        )
        illum_factor = illum_factor & shadow_mask
    cos_factor = torch.sum(R_unit * local_norm_vec_src.reshape(1, 1, *local_norm_vec_src.shape), dim=3)
    psi_rail = torch.sum(
        (1j * k0 / 4) * Hankel(k0 * R_scalar) * illum_factor * cos_factor
        * local_psi_src.reshape(1, 1, -1) * dx_src,
        dim=2,
    )

    ## rail -> ms
    R_vec = local_v_ms.reshape(1, 1, *local_v_ms.shape) - center.unsqueeze(dim=2)
    R_scalar = torch.linalg.norm(R_vec, dim=3)
    R_unit = R_vec / R_scalar.unsqueeze(dim=3)
    illum_factor = torch.sum(R_vec * norm_vec.unsqueeze(dim=2), dim=3) < 0
    if apply_shadow:
        shadow_mask = torch.stack(
            [
                shadow_fn(
                    boundary_vertices[idx],
                    center[idx],
                    local_v_ms,
                    samples=shadow_samples,
                )
                for idx in range(boundary_vertices.shape[0])
            ],
            dim=0,
        )
        illum_factor = illum_factor & shadow_mask
    cos_factor = torch.sum(R_unit * norm_vec.unsqueeze(dim=2), dim=3)
    psi_ms = torch.sum(
        (1j * k0 / 4) * Hankel(k0 * R_scalar) * illum_factor * cos_factor
        * (-psi_rail * dx).unsqueeze(dim=2),
        dim=1,
    )

    if single_input:
        return psi_rail.squeeze(0), psi_ms.squeeze(0)
    return psi_rail, psi_ms


psi_rail_nodefect, psi_ms_nodefect = field(vertices)


sample_vertices = {}
for dataset_name in DATASET_DIRS:
    sample_path = get_dataset_files(dataset_name)[-1]
    sample = load_vertices_from_csv(
        sample_path,
        target_count=N_BOUNDARY_VERTICES,
        smooth_window=DEFAULT_SMOOTH_WINDOW,
    )
    sample_vertices[dataset_name] = match_reference_width(sample, vertices)

sample_fields = {}
for dataset_name, sample in sample_vertices.items():
    sample_v_rail = sample[REFERENCE_RAIL_MASK]
    sample_psi_rail, sample_psi_ms = field(sample)
    sample_fields[dataset_name] = (sample_v_rail, sample_psi_rail, sample_psi_ms)


## Schematic
markersize = 10
fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, tight_layout=True, figsize=(10, 4.8))
ax[0].plot(*vertices.T / mm, "-", color="gray", label="rail", zorder=-100)
ax[0].plot(*v_src.T / mm, "-", color="gray", label="source aperture", zorder=-100)
ax[0].plot(*v_ms.T / mm, "-", color="gray", label="metasurface", zorder=-100)
ax[0].set(xlim=(-160, 160), ylim=(0, 320), xlabel=r"$x$ (mm)", ylabel=r"$y$ (mm)")

int_color = ax[0].scatter(*v_src.T / mm, c=torch.abs(psi_src) ** 2, s=markersize, vmin=0, cmap=plt.cm.inferno)
ax[0].scatter(*v_rail[1:].T / mm, c=torch.abs(psi_rail_nodefect) ** 2, s=markersize, vmin=0, cmap=plt.cm.inferno)
ax[0].scatter(*v_ms.T / mm, c=torch.abs(psi_ms_nodefect) ** 2, vmin=0, s=markersize, cmap=plt.cm.inferno)

ax[1].plot(*vertices.T / mm, "-", color="gray", label="rail", zorder=-100)
ax[1].plot(*v_src.T / mm, "-", color="gray", label="source aperture", zorder=-100)
ax[1].plot(*v_ms.T / mm, "-", color="gray", label="metasurface", zorder=-100)
ax[1].set(xlim=(-160, 160), ylim=(0, 320), xlabel=r"$x$ (mm)")

phase_color = ax[1].scatter(
    *v_src.T / mm,
    c=torch.angle(psi_src),
    s=markersize,
    vmin=-np.pi,
    vmax=np.pi,
    cmap=plt.cm.twilight,
)
ax[1].scatter(
    *v_rail[1:].T / mm,
    c=torch.angle(psi_rail_nodefect),
    s=markersize,
    vmin=-np.pi,
    vmax=np.pi,
    cmap=plt.cm.twilight,
)
ax[1].scatter(
    *v_ms.T / mm,
    c=torch.angle(psi_ms_nodefect),
    s=markersize,
    vmin=-np.pi,
    vmax=np.pi,
    cmap=plt.cm.twilight,
)

ax[0].set_title("Intensity")
ax[1].set_title("Phase")

plt.colorbar(int_color)
plt.colorbar(phase_color)


fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(8.5, 3), tight_layout=True)
for axis, (dataset_name, sample) in zip(ax, sample_vertices.items()):
    axis.plot(*vertices.T / mm, color="gray", lw=0.7, ls="--")
    axis.plot(*sample.T / mm, color="black", lw=0.8)
    axis.set_title(dataset_name)
    axis.set(xlim=(-100, 100), ylim=(0, 200), xlabel=r"$x$ (mm)")
ax[0].set_ylabel(r"$y$ (mm)")


fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(8.5, 3), tight_layout=True)
for axis, (dataset_name, sample) in zip(ax, sample_vertices.items()):
    sample_v_rail, sample_psi_rail, sample_psi_ms = sample_fields[dataset_name]
    axis.plot(*vertices.T / mm, color="gray", lw=0.6, ls="--", zorder=-100)
    axis.scatter(
        *sample_v_rail[1:].T / mm,
        c=torch.abs(sample_psi_rail) ** 2,
        s=1,
        vmin=0,
        cmap=plt.cm.inferno,
    )
    axis.scatter(
        *v_ms.T / mm,
        c=torch.abs(sample_psi_ms) ** 2,
        s=1,
        vmin=0,
        cmap=plt.cm.inferno,
    )
    axis.set_title(dataset_name)
    axis.set(xlim=(-125, 125), ylim=(50, 300), xlabel=r"$x$ (mm)")
ax[0].set_ylabel(r"$y$ (mm)")


#%%
# Example:
# files, vertices_batch, v_rail_batch = load_dataset_vertices("crack", limit=128)
dataset_name = "wear"
limit = 5000
batch_size = 25
dataset_device = get_compute_device()
print(f"Generating {dataset_name} dataset on {dataset_device}")
files, vertices_batch, v_rail_batch, psi_rail, psi_ms = generate_dataset_fields(
    dataset_name,
    limit=limit,
    batch_size=batch_size,
    device=dataset_device,
)

# Save the generated tensors for later dataset loading in deep learning workflows.
dataset_output_path = RAILDEFECT_DIR / "generated_datasets2" / f"{dataset_name}_fields_limit{limit}.pt"
dataset_output_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(
    {
        # "files": files,
        "vertices_batch": vertices_batch,
        "v_rail_batch": v_rail_batch,
        "psi_rail": psi_rail,
        "psi_ms": psi_ms,
    },
    dataset_output_path,
)
print(f"Saved generated dataset to {dataset_output_path.resolve()}")
