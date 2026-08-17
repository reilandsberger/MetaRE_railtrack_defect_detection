"""Trainable optical stack for the 3D rail system.

Ported from Face3D_clean's ``utils_optics.py`` with the fixes agreed in the
plan: FFT propagation (same exact RS kernel, ~10^3x faster than the original
spatial conv2d), registered buffers (no per-forward host->device copies),
noise gated on ``self.training``, initialized parameters, a soft-clamped
meta-unit parameterization, and differentiable 2D soft detectors with
redundancy-aware pruning (default; the Face3D variance criterion is kept
for sparse layouts).

All modules work on the rectangular (NX, NY) grid.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.fft as fft
from torch import nn

from . import config

_LIB_DIR = Path(__file__).resolve().parent
# The Face3D meta-atom library fits are per-pixel on an 80x80 aperture; the
# rail grid is 60x30, so we use the central crop. Rows 10:70 (x), cols 25:55
# (y). Explicit assumption — reviewed in design_review_notebook.ipynb.
_LIB_CROP = (slice(10, 70), slice(25, 55))


# ---------------------------------------------------------------------------
# Propagators
# ---------------------------------------------------------------------------
def _rs_conv_kernel(distance, wavelength, dx, nx, ny, ref_index=1.0) -> torch.Tensor:
    """Exact RS-I convolution kernel on the doubled support, verbatim prop3d.

    Shape (2*nx-1, 2*ny-1), centered at index (nx-1, ny-1).
    """
    k0 = 2 * np.pi * ref_index / wavelength
    gx = dx * torch.arange(-nx + 1, nx, dtype=torch.float32)
    gy = dx * torch.arange(-ny + 1, ny, dtype=torch.float32)
    X_g, Y_g = torch.meshgrid(gx, gy, indexing="ij")
    R_g = torch.sqrt(X_g**2 + Y_g**2 + distance**2)
    return ((dx**2 / wavelength) * (1 / (k0 * R_g) - 1j)
            * (distance / R_g**2) * torch.exp(1j * k0 * R_g))


class PropagatorRSFFT(nn.Module):
    """Free-space propagation: Face3D's exact RS kernel applied by FFT.

    Machine-precision equivalent to ``prop3d``'s conv2d (the kernel is
    symmetric, so correlation == convolution); verified in V2.
    """

    def __init__(
        self,
        distance: float,
        wavelength: float = config.WVL,
        dx: float = config.DX,
        nx: int = config.NX,
        ny: int = config.NY,
        ref_index: float = 1.0,
    ):
        super().__init__()
        self.D = distance
        self.nx, self.ny = nx, ny
        kernel = _rs_conv_kernel(distance, wavelength, dx, nx, ny, ref_index)
        # linear-convolution FFT size: signal (nx) + kernel (2nx-1) - 1
        self.p1, self.p2 = 3 * nx - 2, 3 * ny - 2
        self.register_buffer("kernel_spec", fft.fft2(kernel, s=(self.p1, self.p2)), persistent=False)
        self.register_buffer("kernel", kernel, persistent=False)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        signal = signal.reshape(-1, self.nx, self.ny) + 0j
        out = fft.ifft2(fft.fft2(signal, s=(self.p1, self.p2)) * self.kernel_spec)
        return out[:, self.nx - 1: 2 * self.nx - 1, self.ny - 1: 2 * self.ny - 1]

    def forward_direct(self, signal: torch.Tensor) -> torch.Tensor:
        """Reference path: the original conv2d formulation (V2 check)."""
        signal = signal.reshape(-1, 1, self.nx, self.ny) + 0j
        weight = self.kernel.reshape(1, 1, 2 * self.nx - 1, 2 * self.ny - 1)
        out = nn.functional.conv2d(signal, weight, padding=(self.nx - 1, self.ny - 1))
        return out.reshape(-1, self.nx, self.ny)


class PropagatorASM2D(nn.Module):
    """Angular-spectrum propagator (cross-validation / config switch).

    Fixes over the BarcodeCalculation version: centered zero-padding and a
    principal-branch kz so evanescent components decay as exp(-|kz| D).
    """

    def __init__(
        self,
        distance: float,
        wavelength: float = config.WVL,
        dx: float = config.DX,
        nx: int = config.NX,
        ny: int = config.NY,
        pad_factor: int = 4,
    ):
        super().__init__()
        self.D = distance
        self.nx, self.ny = nx, ny
        self.p1, self.p2 = nx * pad_factor, ny * pad_factor
        k0 = 2 * np.pi / wavelength
        kx = 2 * np.pi * fft.fftfreq(self.p1, d=dx)
        ky = 2 * np.pi * fft.fftfreq(self.p2, d=dx)
        KX, KY = torch.meshgrid(kx, ky, indexing="ij")
        kz = torch.sqrt(k0**2 - KX**2 - KY**2 + 0j)   # Im(kz) >= 0 on the principal branch
        self.register_buffer("transfer", torch.exp(1j * kz * distance), persistent=False)
        self.o1 = (self.p1 - nx) // 2
        self.o2 = (self.p2 - ny) // 2

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        signal = signal.reshape(-1, self.nx, self.ny) + 0j
        padded = torch.zeros(signal.shape[0], self.p1, self.p2,
                             dtype=torch.complex64, device=signal.device)
        padded[:, self.o1: self.o1 + self.nx, self.o2: self.o2 + self.ny] = signal
        out = fft.ifft2(fft.fft2(padded) * self.transfer)
        return out[:, self.o1: self.o1 + self.nx, self.o2: self.o2 + self.ny]


# ---------------------------------------------------------------------------
# Metasurface layers
# ---------------------------------------------------------------------------
class SLM2D(nn.Module):
    """Phase-only mask on the rectangular grid; explicitly initialized."""

    def __init__(self, nx: int = config.NX, ny: int = config.NY, seed: int = 0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        init = torch.randn(nx, ny, generator=gen) * (np.pi / 2)
        self.phase = nn.Parameter(init)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return signal * torch.exp(1j * self.phase)


class MetaUnitSoft(nn.Module):
    """Face3D MetaUnit with a soft width clamp.

    Trainable variable is an unconstrained map p; the physical pillar width is
    w = 1 + 2.8*sigmoid(p) mm, i.e. always inside the library's [1, 3.8] mm
    validity range with nonzero gradient everywhere (the original hard
    ``clip`` froze saturated cells). Amplitude and phase come from the
    per-pixel polynomial fits, cropped to the rail grid.
    """

    def __init__(self, nx: int = config.NX, ny: int = config.NY, seed: int = 0,
                 ampfit_path: Path = _LIB_DIR / "library_amp_fit.npy",
                 phasefit_path: Path = _LIB_DIR / "library_phase_fit.npy"):
        super().__init__()
        if abs(config.WVL - config.LIBRARY_WVL) > 1e-9:
            raise RuntimeError(
                f"surface='metaunit' is blocked at WVL={config.WVL} mm: the "
                f"meta-atom polynomial fits (library_amp_fit.npy / "
                f"library_phase_fit.npy) were measured at "
                f"{config.LIBRARY_WVL} mm and do not transfer across "
                f"wavelength. The pillar widths also stop being physical — up "
                f"to 3.8 mm against a {config.DX} mm unit cell. Refit the "
                f"library for {299.792458 / config.WVL:.1f} GHz, or use "
                f"surface='slm' (idealized phase) or 'none' (no-MS baseline).")
        ampfit = torch.tensor(np.load(ampfit_path), dtype=torch.float32)
        phasefit = torch.tensor(np.load(phasefit_path), dtype=torch.float32)
        ampfit = ampfit.reshape(-1, 80, 80).flip(dims=(0,))[:, _LIB_CROP[0], _LIB_CROP[1]]
        phasefit = phasefit.reshape(-1, 80, 80).flip(dims=(0,))[:, _LIB_CROP[0], _LIB_CROP[1]]
        if ampfit.shape[1:] != (nx, ny):
            raise ValueError(f"Library crop {tuple(ampfit.shape[1:])} != grid ({nx}, {ny})")
        self.register_buffer("ampfit", ampfit)
        self.register_buffer("phasefit", phasefit)
        self.register_buffer("ampdeg", torch.arange(len(ampfit)).reshape(-1, 1, 1).float())
        self.register_buffer("phasedeg", torch.arange(len(phasefit)).reshape(-1, 1, 1).float())

        gen = torch.Generator().manual_seed(seed)
        w0 = torch.empty(nx, ny).uniform_(1.2, 3.6, generator=gen)
        self.p = nn.Parameter(torch.logit((w0 - 1.0) / 2.8))

    @property
    def w_pillar(self) -> torch.Tensor:
        return 1.0 + 2.8 * torch.sigmoid(self.p)

    def amp(self) -> torch.Tensor:
        return (self.ampfit * self.w_pillar.unsqueeze(0) ** self.ampdeg).sum(dim=0)

    def phase(self) -> torch.Tensor:
        return (self.phasefit * self.w_pillar.unsqueeze(0) ** self.phasedeg).sum(dim=0)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return signal * (self.amp() * torch.exp(1j * self.phase()))


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------
class SoftDetector2D(nn.Module):
    """Rectangular detector windows with trainable centers.

    2D extension of the repo's SoftDetectorPlane: separable sigmoid-edged
    windows of fixed size, centers stored as normalized coordinates
    u in [-1, 1]^2. ``tau_scale`` (edge softness as a fraction of the window
    size) is annealed from 1/4 to 1/8 during training, floored at half a
    pixel per axis (see ``_axis_soft``); ``hard_powers`` uses binary masks
    and is what all reported metrics use.
    """

    def __init__(
        self,
        centers: torch.Tensor | None = None,
        det_size: tuple[float, float] = config.DET_SIZE,
        dx: float = config.DX,
        nx: int = config.NX,
        ny: int = config.NY,
        wx: float = config.WX,
        wy: float = config.WY,
        jitter_sigma: float = config.DET_JITTER_MM,
    ):
        super().__init__()
        if centers is None:
            # NOTE: callers with a non-default aperture (nx/ny/wx/wy) must pass
            # centers explicitly — the default lattice spans the config aperture.
            centers = config.dense_detector_centers()
        self.det_size = det_size
        self.dx = dx
        self.half = torch.tensor([wx / 2, wy / 2])
        self.jitter_sigma = jitter_sigma
        self.tau_scale = 0.25

        self.u = nn.Parameter(centers / self.half)
        x = torch.arange(-wx / 2 + dx / 2, wx / 2, dx)
        y = torch.arange(-wy / 2 + dx / 2, wy / 2, dx)
        # persistent=False: these are geometry derived from config, not learned
        # state. Persisting them let a checkpoint from one geometry silently
        # overwrite the grid coordinates of a model built for another.
        self.register_buffer("grid_x", x, persistent=False)
        self.register_buffer("grid_y", y, persistent=False)
        self.register_buffer("half_buf", self.half, persistent=False)

    @property
    def n_det(self) -> int:
        return self.u.shape[0]

    def centers(self) -> torch.Tensor:
        return torch.clamp(self.u, -1.0, 1.0) * self.half_buf

    def _axis_soft(self, grid: torch.Tensor, c: torch.Tensor, w: float) -> torch.Tensor:
        # Floor tau at half a pixel PER AXIS: below the grid pitch the sigmoid
        # edge cannot represent sub-pixel motion and position gradients die
        # (README finding 14). The old 1e-3 floor never bound, so the y-axis
        # window (w/8 = 1.4 mm at dx=4) was silently sub-pixel.
        tau = max(self.tau_scale * w, 0.5 * self.dx)
        lo = torch.sigmoid((grid.reshape(1, -1) - (c.reshape(-1, 1) - w / 2)) / tau)
        hi = torch.sigmoid(((c.reshape(-1, 1) + w / 2) - grid.reshape(1, -1)) / tau)
        return lo * hi

    def soft_masks(self, jitter: bool = False) -> torch.Tensor:
        c = self.centers()
        if jitter and self.jitter_sigma > 0:
            c = c + torch.randn_like(c) * self.jitter_sigma
        mx = self._axis_soft(self.grid_x, c[:, 0], self.det_size[0])
        my = self._axis_soft(self.grid_y, c[:, 1], self.det_size[1])
        return torch.einsum("ni,nj->nij", mx, my)

    def hard_masks(self, shift: tuple[float, float] = (0.0, 0.0)) -> torch.Tensor:
        c = self.centers().detach()
        mx = ((self.grid_x.reshape(1, -1) - (c[:, 0:1] + shift[0])).abs() < self.det_size[0] / 2).float()
        my = ((self.grid_y.reshape(1, -1) - (c[:, 1:2] + shift[1])).abs() < self.det_size[1] / 2).float()
        return torch.einsum("ni,nj->nij", mx, my)

    def _integrate(self, intensity: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        flat = intensity.reshape(intensity.shape[0], -1)
        return (self.dx**2) * flat @ masks.reshape(masks.shape[0], -1).T

    def powers(self, intensity: torch.Tensor, jitter: bool = False) -> torch.Tensor:
        return self._integrate(intensity, self.soft_masks(jitter=jitter))

    def hard_powers(self, intensity: torch.Tensor, shift: tuple[float, float] = (0.0, 0.0)) -> torch.Tensor:
        return self._integrate(intensity, self.hard_masks(shift=shift))

    def min_separation(self) -> float:
        """Smallest distance between any two detector centres, in mm.

        Nothing in the loss repels detectors from each other, so trained centres
        can converge onto the same spot and produce duplicate barcode entries.
        A value below ~1 pixel (dx) means detectors have effectively collapsed.
        """
        c = self.centers().detach()
        if c.shape[0] < 2:
            return float("inf")
        d = torch.cdist(c, c)
        d.fill_diagonal_(float("inf"))
        return float(d.min())

    @staticmethod
    def prune_indices(det_values: torch.Tensor, num_remove: int,
                      criterion: str = "redundancy") -> torch.Tensor:
        """Choose which detectors to KEEP. Returns sorted indices.

        det_values: (N_samples, N_det) hard powers on the validation split.

        "variance" (Face3D): drop the lowest-variance detectors. Valid when the
        layout is SPARSE — Face3D's 6x6 grid covered 7.2% of its aperture with
        30-37 mm gaps, so windows were near-independent and low variance really
        did mean uninformative.

        "redundancy" (default): greedy backward elimination that accounts for
        detectors seeing the SAME light. A dense start (13x10 = 92% coverage)
        has overlapping windows whose variances are nearly identical, so the
        variance ranking cannot tell "duplicate of my neighbour" from "uniquely
        informative" — it will keep several copies of one hotspot and discard a
        quiet but unique detector. Here each detector is valued by its UNIQUE
        contribution, ``std_i * (1 - max_j |corr_ij|)`` over the surviving set:
        signal strength times the share of it not already covered by another
        detector. The least valuable is removed one at a time, so the scores
        always reflect the set that remains. A bright duplicate scores near zero
        (corr -> 1) while a quiet but independent detector keeps its value.
        """
        norm = det_values / det_values.norm(dim=1, keepdim=True).clamp_min(1e-12)
        std = norm.std(dim=0)

        if criterion == "variance" or det_values.shape[0] < 8:
            if criterion == "redundancy":
                # never downgrade silently: variance ranking on a dense layout
                # keeps duplicates (V0b) — the caller should know it happened
                print(f"[rail3d] WARNING: redundancy pruning requested with only "
                      f"{det_values.shape[0]} samples (<8); falling back to the "
                      f"variance ranking, which is only valid for sparse layouts")
            # too few samples for meaningful correlations -> variance ranking
            keep = torch.argsort(std)[num_remove:]
            return torch.sort(keep).values
        if criterion != "redundancy":
            raise ValueError(f"Unknown prune criterion: {criterion}")

        alive = list(range(norm.shape[1]))
        for _ in range(num_remove):
            if len(alive) <= 1:
                break
            sub = norm[:, alive]
            c = sub - sub.mean(dim=0, keepdim=True)
            s = c.std(dim=0).clamp_min(1e-12)
            corr = (c.T @ c) / (c.shape[0] * s[:, None] * s[None, :])
            corr = corr.abs()
            corr.fill_diagonal_(0.0)
            similarity = corr.max(dim=1).values            # most similar neighbour
            value = s * (1.0 - similarity)                 # unique contribution
            alive.pop(int(torch.argmin(value)))            # drop the least useful
        return torch.sort(torch.tensor(alive, dtype=torch.long)).values


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class ONN3D(nn.Module):
    """[metasurface -> RS-FFT propagation] x N_layer -> |.|^2 -> detectors.

    ``layer_distances[i]`` is the propagation distance after layer i; the last
    entry is the MS -> detector-plane distance. A second metasurface layer is
    a pure config change: N_layer=2, layer_distances=(d_between, 160.0).
    The tiny linear head maps (normalized) detector powers to class logits.
    """

    def __init__(
        self,
        n_layer: int = 1,
        layer_distances: tuple = config.LAYER_DISTANCES,
        surface: str = "slm",
        n_classes: int = len(config.CLASS_NAMES),
        noise: bool = True,
        snr_add: float = config.SNR_ADD,
        snr_multiple: float = config.SNR_MULTIPLE,
        seed: int = config.SEED,
        detector: SoftDetector2D | None = None,
    ):
        super().__init__()
        if len(layer_distances) != n_layer:
            raise ValueError("layer_distances must have one entry per layer")
        self.noise = noise
        self.snr_add = snr_add
        self.snr_multiple = snr_multiple

        if surface == "slm":
            self.layers = nn.ModuleList([SLM2D(seed=seed + i) for i in range(n_layer)])
        elif surface == "metaunit":
            self.layers = nn.ModuleList([MetaUnitSoft(seed=seed + i) for i in range(n_layer)])
        elif surface == "none":
            # no-metasurface baseline: free propagation + detectors only
            self.layers = nn.ModuleList([nn.Identity() for _ in range(n_layer)])
        else:
            raise ValueError(f"Unknown surface type: {surface}")
        self.propagators = nn.ModuleList(
            [PropagatorRSFFT(dist) for dist in layer_distances]
        )
        self.detector = detector if detector is not None else SoftDetector2D()
        self.head = nn.Linear(self.detector.n_det, n_classes)

    def add_noise(self, signal: torch.Tensor) -> torch.Tensor:
        if not (self.training and self.noise):
            return signal
        noise_pro = torch.normal(mean=1.0, std=self.snr_multiple * torch.ones_like(signal.real))
        noise_add = (
            torch.normal(mean=0.0, std=self.snr_add * torch.ones_like(signal.real))
            * torch.exp(1j * 2 * np.pi * torch.rand_like(signal.real))
        )
        return signal * noise_pro + noise_add

    def propagate(self, psi: torch.Tensor) -> torch.Tensor:
        """Complex field at the detector plane."""
        signal = self.add_noise(psi)
        for layer, prop in zip(self.layers, self.propagators):
            signal = layer(signal)
            signal = prop(signal)
            signal = self.add_noise(signal)
        return signal

    def forward(self, psi: torch.Tensor, hard: bool = False):
        """Returns (intensity, det): output intensity map and detector powers.

        Soft (differentiable, jittered when training) masks by default; pass
        hard=True for the physical binary-window readout used in evaluation.
        """
        signal = self.propagate(psi)
        intensity = signal.abs() ** 2
        if hard:
            det = self.detector.hard_powers(intensity)
        else:
            det = self.detector.powers(intensity, jitter=self.training)
        return intensity, det

    def classify(self, det_normalized: torch.Tensor) -> torch.Tensor:
        return self.head(det_normalized)

    @torch.no_grad()
    def prune_detectors(self, det_values: torch.Tensor, num_remove: int,
                        criterion: str = "redundancy") -> torch.Tensor:
        """Drop the least useful detectors (see ``SoftDetector2D.prune_indices``
        for the criterion); slims u and the head to match.

        The caller must rebuild the optimizer afterwards (parameter objects
        change). Returns the kept indices (sorted; row i of the new ``u`` is
        row keep[i] of the old one — training records these for provenance).
        """
        keep = SoftDetector2D.prune_indices(det_values, num_remove, criterion=criterion)
        self.detector.u = nn.Parameter(self.detector.u.data[keep].clone())
        old = self.head
        new = nn.Linear(len(keep), old.out_features).to(old.weight.device)
        new.weight.data = old.weight.data[:, keep].clone()
        new.bias.data = old.bias.data.clone()
        self.head = new
        return keep
