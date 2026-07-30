#%%
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plot_setting
import torch
import torch.fft as fft
import torch.nn as nn
import torch.optim as optim
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
# device = torch.device("cpu")


mm = 1e3
wvl = 12 * mm
k0 = 2 * np.pi / wvl
W = 20 * wvl
y_ms = 250 * mm
dx_ms = wvl / 10
v_ms_x = torch.arange(-W / 2, W / 2, dx_ms) + dx_ms / 2
v_ms_y = torch.zeros_like(v_ms_x) + y_ms
v_ms = torch.stack([v_ms_x, v_ms_y], dim=1)

DATASET_DIR = Path("generated_datasets2")
DATASET_FILES = {
    "crack": DATASET_DIR / "crack_fields_limit5000.pt",
    "dent": DATASET_DIR / "dent_fields_limit5000.pt",
    "wear": DATASET_DIR / "wear_fields_limit5000.pt",
}
SUPPORTED_CLASS_NAMES = tuple(DATASET_FILES.keys())
class_names: list[str] = []


def load_intact_field() -> torch.Tensor:
    for candidate in ("psi_ms_nodefect_12mm.pt", "psi_ms_nodefect.pt"):
        path = Path(candidate)
        if path.exists():
            psi_ms0 = torch.load(path).to(torch.complex64)
            return psi_ms0 / torch.sqrt(torch.mean(psi_ms0.abs() ** 2))
    raise FileNotFoundError("No intact metasurface field file found.")


def load_generated_dataset() -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], list[str]]:
    psi_ms_parts = []
    label_parts = []
    metadata_parts = {"vertices_batch": [], "v_rail_batch": [], "psi_rail": []}
    active_class_names = [class_name for class_name in SUPPORTED_CLASS_NAMES if DATASET_FILES[class_name].exists()]

    if not active_class_names:
        expected = ", ".join(str(DATASET_FILES[class_name]) for class_name in SUPPORTED_CLASS_NAMES)
        raise FileNotFoundError(f"No generated datasets found in {DATASET_DIR}. Expected one of: {expected}")

    missing_class_names = [class_name for class_name in SUPPORTED_CLASS_NAMES if class_name not in active_class_names]
    if missing_class_names:
        print(f"Loading available defect datasets only: {active_class_names}. Missing: {missing_class_names}")

    for label, class_name in enumerate(active_class_names):
        path = DATASET_FILES[class_name]
        bundle = torch.load(path)
        psi_ms_parts.append(bundle["psi_ms"].to(torch.complex64))
        label_parts.append(torch.full((len(bundle["psi_ms"]),), label, dtype=torch.long))
        for key in metadata_parts:
            metadata_parts[key].append(bundle[key])

    psi_ms = torch.cat(psi_ms_parts, dim=0)
    psi_ms = psi_ms / torch.sqrt(torch.mean(psi_ms.abs() ** 2))
    labels = torch.cat(label_parts, dim=0)
    metadata = {key: torch.cat(value, dim=0) for key, value in metadata_parts.items()}
    return psi_ms, labels, metadata, active_class_names


def stratified_split(
    labels: torch.Tensor,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    train_idx = []
    val_idx = []
    test_idx = []

    for class_id in range(len(class_names)):
        class_indices = torch.where(labels == class_id)[0]
        perm = class_indices[torch.randperm(len(class_indices), generator=generator)]
        n_train = int(len(perm) * train_ratio)
        n_val = int(len(perm) * val_ratio)
        train_idx.append(perm[:n_train])
        val_idx.append(perm[n_train:n_train + n_val])
        test_idx.append(perm[n_train + n_val:])

    return (
        torch.cat(train_idx),
        torch.cat(val_idx),
        torch.cat(test_idx),
    )


psi_ms, defect_label, metadata, class_names = load_generated_dataset()
psi_ms0 = load_intact_field()
train_idx, val_idx, test_idx = stratified_split(defect_label)


class FieldDataset(Dataset):
    def __init__(
        self,
        field: torch.Tensor,
        label: torch.Tensor,
        vertices: torch.Tensor,
        v_rail: torch.Tensor,
        psi_rail: torch.Tensor,
    ):
        self.field = field
        self.label = label.to(torch.long)
        self.vertices = vertices
        self.v_rail = v_rail
        self.psi_rail = psi_rail

    def __getitem__(self, index):
        return (
            self.field[index],
            self.label[index],
            self.vertices[index],
            self.v_rail[index],
            self.psi_rail[index],
        )

    def __len__(self):
        return len(self.field)


class Propagator(nn.Module):
    def __init__(
        self,
        distance: float,
        width: float = W,
        resol: int = 200,
        wavelength: float = wvl,
        padding_factor: int = 10,
    ):
        super().__init__()
        self.W = width
        self.dx = width / resol
        self.D = distance
        self.k0 = 2 * np.pi / wavelength
        self.resol = resol
        self.N_pad = padding_factor
        self.kx = 2 * np.pi * fft.fftfreq(self.resol * self.N_pad, d=self.dx, dtype=torch.float)
        self.ky = torch.sqrt(self.k0 ** 2 - self.kx ** 2 + 0j)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.dim() == 1:
            signal = signal.reshape(1, -1)

        zeropad = torch.zeros(len(signal), self.resol * self.N_pad, device=signal.device, dtype=torch.complex64)
        zeropad[:, :self.resol] = signal
        signal_f = fft.fft(zeropad)
        signal_f = signal_f * torch.exp(1j * self.ky.to(signal.device) * self.D)
        signal = torch.fft.ifft(signal_f)
        return signal[:, :self.resol]


class Metalayer(nn.Module):
    def __init__(
        self,
        resol: int = 200,
        phase_init: float = 0.0,
        std_phase: float = 90 * np.pi / 180,
        dummy: int = 0,
        sym: bool = False,
    ):
        super().__init__()
        self.sym = sym
        self.N = resol
        generator = torch.Generator().manual_seed(dummy)
        self.phase_init = phase_init + std_phase * torch.randn(self.N, generator=generator)
        self.phase = nn.Parameter(self.phase_init.clone())

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        if self.sym:
            phase = 0.5 * (self.phase + self.phase.flip(dims=(0,)))
        else:
            phase = self.phase
        return signal * torch.exp(1j * phase).to(signal.device)


class ONN(nn.Module):
    def __init__(
        self,
        N_layer: int = 2,
        distance: float = 50 * mm,
        sym: bool = False,
        det_center: np.ndarray = np.array([-W / 6, W / 6]),
        det_width: float = W / 40,
        width: float = W,
        resol: int = 200,
    ):
        super().__init__()
        self.resol = resol
        self.prop = Propagator(distance=distance, resol=resol)
        self.metalayers = nn.ModuleList([Metalayer(dummy=i, sym=sym, resol=resol) for i in range(N_layer)])

        dx = width / resol
        x = torch.arange(-width / 2 + dx / 2, width / 2, dx)
        self.det_width = det_width
        self.det_center = det_center
        self.detector = torch.stack([torch.abs(x - xc) < det_width / 2 for xc in det_center], dim=1)

    def forward(self, signal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        for ml in self.metalayers:
            signal = self.prop(ml(signal))
        intensity = signal.abs() ** 2

        signal0 = psi_ms0.reshape(1, -1).clone().to(signal.device)
        for ml in self.metalayers:
            signal0 = self.prop(ml(signal0))
        intensity0 = signal0.abs() ** 2

        detector = self.detector.to(signal.device)
        det = torch.sum(intensity.unsqueeze(dim=2) * detector.unsqueeze(dim=0), dim=1)
        det0 = torch.sum(intensity0.unsqueeze(dim=2) * detector.unsqueeze(dim=0), dim=1)
        return intensity, intensity0, det, det0


train_ds = FieldDataset(
    field=psi_ms[train_idx],
    label=defect_label[train_idx],
    vertices=metadata["vertices_batch"][train_idx],
    v_rail=metadata["v_rail_batch"][train_idx],
    psi_rail=metadata["psi_rail"][train_idx],
)
val_ds = FieldDataset(
    field=psi_ms[val_idx],
    label=defect_label[val_idx],
    vertices=metadata["vertices_batch"][val_idx],
    v_rail=metadata["v_rail_batch"][val_idx],
    psi_rail=metadata["psi_rail"][val_idx],
)
test_ds = FieldDataset(
    field=psi_ms[test_idx],
    label=defect_label[test_idx],
    vertices=metadata["vertices_batch"][test_idx],
    v_rail=metadata["v_rail_batch"][test_idx],
    psi_rail=metadata["psi_rail"][test_idx],
)


batch_size = 2000
batch_size = 1000
train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_size)
net = ONN(
    N_layer=1,
    distance=50 * mm,
    sym=False,
    det_center=np.array([-W / 4, 0.0, W / 4]),
).to(device)
N_epoch = 400
lr = 1e-3

optimizer = optim.Adam(net.parameters(), lr=lr)
relative_margin_pct = 40.0
relative_margin = relative_margin_pct / 100.0
power_floor_ratio = 2
power_reg_weight = 0.2


def relative_l2_gap(det: torch.Tensor, det0: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.linalg.norm(det - det0, dim=1) / torch.linalg.norm(det0, dim=1).clamp_min(eps)


def detector_total_power(det: torch.Tensor) -> torch.Tensor:
    return det.sum(dim=1)


def relative_separation_loss(det: torch.Tensor, det0: torch.Tensor, margin: float = relative_margin) -> torch.Tensor:
    gap = relative_l2_gap(det, det0)
    return torch.relu(margin - gap).mean()


def power_regularization_loss(
    det: torch.Tensor,
    det0: torch.Tensor,
    power_floor: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    det_power = detector_total_power(det)
    det0_power = detector_total_power(det0)
    det_penalty = torch.relu(power_floor - det_power) / (power_floor + eps)
    det0_penalty = torch.relu(power_floor - det0_power) / (power_floor + eps)
    return det_penalty.mean() + det0_penalty.mean()


def separation_satisfaction(det: torch.Tensor, det0: torch.Tensor, margin: float = relative_margin) -> float:
    gap = relative_l2_gap(det, det0)
    return (gap > margin).float().mean().item()


def mean_relative_gap(det: torch.Tensor, det0: torch.Tensor) -> float:
    return relative_l2_gap(det, det0).mean().item()


def mean_detector_power(det: torch.Tensor) -> float:
    return detector_total_power(det).mean().item()


def class_sample_indices(label_tensor: torch.Tensor, max_per_class: int | None = None) -> dict[int, torch.Tensor]:
    picks = {}
    for class_id in range(len(class_names)):
        matches = torch.where(label_tensor == class_id)[0]
        if max_per_class is not None:
            matches = matches[:max_per_class]
        picks[class_id] = matches
    return picks


def draw_margin_sphere_3d(axis, center: np.ndarray, radius: float, color: str = "k", lw: float = 0.6) -> None:
    u = np.linspace(0, 2 * np.pi, 25)
    v = np.linspace(0, np.pi, 13)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    axis.plot_wireframe(x, y, z, rstride=2, cstride=2, color=color, linewidth=lw, alpha=0.35)


def set_detector_limits_3d(axis, limit: float, Nticks: int=5) -> None:
    axis.set_xlim(0, limit)
    axis.set_ylim(0, limit)
    axis.set_zlim(0, limit)
    axis.set_xticks(np.linspace(0, limit, Nticks))
    axis.set_yticks(np.linspace(0, limit, Nticks))
    axis.set_zticks(np.linspace(0, limit, Nticks))
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=22, azim=-55)


def margin_legend_labels(relative_gap: torch.Tensor, margin: float) -> tuple[str, str]:
    gap_np = relative_gap.detach().cpu().numpy()
    inside_count = int(np.sum(gap_np <= margin))
    outside_count = int(np.sum(gap_np > margin))
    total_count = max(inside_count + outside_count, 1)
    inside_pct = 100.0 * inside_count / total_count
    outside_pct = 100.0 * outside_count / total_count
    return (
        f"Inside margin ({inside_pct:.1f}%)",
        f"Outside margin ({outside_pct:.1f}%)",
    )


def plot_detector_scatter_hollow_circles_3d(
    axis,
    det: torch.Tensor,
    det0: torch.Tensor,
    relative_gap: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
    title: str,
    dot_size: int = 12,
) -> None:
    det_np = det.detach().cpu().numpy()
    det0_np = det0.detach().cpu().numpy()
    gap_np = relative_gap.detach().cpu().numpy()
    circle_radius = margin * np.linalg.norm(det0_np)
    class_colors = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple"]

    draw_margin_sphere_3d(axis, det0_np[:3], circle_radius)

    labels_np = labels.detach().cpu().numpy()
    for class_id, class_name in enumerate(class_names):
        class_mask = labels_np == class_id
        if not np.any(class_mask):
            continue
        class_gap = gap_np[class_mask]
        pass_rate = 100.0 * np.mean(class_gap > margin)
        class_det = det_np[class_mask]
        inside_mask = class_gap <= margin
        outside_mask = class_gap > margin
        color = class_colors[class_id % len(class_colors)]

        axis.scatter(
            class_det[inside_mask, 0],
            class_det[inside_mask, 1],
            class_det[inside_mask, 2],
            s=dot_size,
            c=color,
            marker="o",
            linewidths=0.0,
            alpha=0.4,
            depthshade=False,
        )
        axis.scatter(
            class_det[outside_mask, 0],
            class_det[outside_mask, 1],
            class_det[outside_mask, 2],
            s=dot_size,
            c=color,
            marker="o",
            linewidths=0.0,
            alpha=0.65,
            label=f"{class_name} ({pass_rate:.1f}% pass)",
            depthshade=False,
        )

    axis.scatter([det0_np[0]], [det0_np[1]], [det0_np[2]], s=45, c="k", marker="*", label="Intact", depthshade=False)
    axis.set(
        # xlabel="Detector 0 power",
        # ylabel="Detector 1 power",
        # zlabel="Detector 2 power",
        title=title,
    )
    axis.legend(fontsize=7, loc="best")


def plot_detector_scatter_shape_threshold_3d(
    axis,
    det: torch.Tensor,
    det0: torch.Tensor,
    relative_gap: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
    title: str,
    dot_size: int = 14,
) -> None:
    det_np = det.detach().cpu().numpy()
    det0_np = det0.detach().cpu().numpy()
    gap_np = relative_gap.detach().cpu().numpy()
    circle_radius = margin * np.linalg.norm(det0_np)

    draw_margin_sphere_3d(axis, det0_np[:3], circle_radius)

    labels_np = labels.detach().cpu().numpy()
    class_handles = []
    for class_id, class_name in enumerate(class_names):
        class_mask = labels_np == class_id
        if not np.any(class_mask):
            continue
        class_gap = gap_np[class_mask]
        class_det = det_np[class_mask]

        inside_mask = class_gap <= margin
        outside_mask = class_gap > margin
        axis.scatter(
            class_det[inside_mask, 0],
            class_det[inside_mask, 1],
            class_det[inside_mask, 2],
            s=dot_size,
            c="tab:red",
            marker="o",
            linewidths=0.0,
            alpha=0.4,
            depthshade=False,
        )
        axis.scatter(
            class_det[outside_mask, 0],
            class_det[outside_mask, 1],
            class_det[outside_mask, 2],
            s=dot_size,
            c="tab:blue",
            marker="o",
            linewidths=0.0,
            alpha=0.65,
            depthshade=False,
        )
        class_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor="0.25",
                markeredgecolor="0.25",
                markeredgewidth=0.0,
                markersize=5,
                label=class_name,
            )
        )

    axis.scatter([det0_np[0]], [det0_np[1]], [det0_np[2]], s=45, c="k", marker="*", label="Intact", depthshade=False)
    axis.set(
        # xlabel="Detector 0 power",
        # ylabel="Detector 1 power",
        # zlabel="Detector 2 power",
        title=title,
    )

    inside_label, outside_label = margin_legend_labels(relative_gap, margin)
    threshold_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="tab:red", markeredgecolor="tab:red", markeredgewidth=0.0, markersize=5, label=inside_label),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="tab:blue", markeredgecolor="tab:blue", markeredgewidth=0.0, markersize=5, label=outside_label),
        Line2D([0], [0], marker="*", linestyle="None", color="k", markersize=6, label="Intact"),
    ]
    axis.legend(handles=threshold_handles, fontsize=7, loc="upper right")


def plot_detector_scatter_single_class_3d(
    axis,
    det: torch.Tensor,
    det0: torch.Tensor,
    relative_gap: torch.Tensor,
    labels: torch.Tensor,
    class_id: int,
    margin: float,
    title: str,
    dot_size: int = 14,
) -> None:
    det_np = det.detach().cpu().numpy()
    det0_np = det0.detach().cpu().numpy()
    gap_np = relative_gap.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    circle_radius = margin * np.linalg.norm(det0_np)

    draw_margin_sphere_3d(axis, det0_np[:3], circle_radius)

    class_mask = labels_np == class_id
    class_det = det_np[class_mask]
    class_gap = gap_np[class_mask]
    inside_mask = class_gap <= margin
    outside_mask = class_gap > margin

    axis.scatter(
        class_det[inside_mask, 0],
        class_det[inside_mask, 1],
        class_det[inside_mask, 2],
        s=dot_size,
        c="tab:red",
        marker="o",
        linewidths=0.0,
        alpha=0.4,
        depthshade=False,
    )
    axis.scatter(
        class_det[outside_mask, 0],
        class_det[outside_mask, 1],
        class_det[outside_mask, 2],
        s=dot_size,
        c="tab:blue",
        marker="o",
        linewidths=0.0,
        alpha=0.65,
        depthshade=False,
    )
    axis.scatter([det0_np[0]], [det0_np[1]], [det0_np[2]], s=35, c="k", marker="*", depthshade=False)
    axis.set(title=title, zlabel="Detector 2 power")
    inside_label, outside_label = margin_legend_labels(torch.as_tensor(class_gap), margin)
    threshold_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="tab:red", markeredgecolor="tab:red", markeredgewidth=0.0, markersize=4, label=inside_label),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="tab:blue", markeredgecolor="tab:blue", markeredgewidth=0.0, markersize=4, label=outside_label),
        Line2D([0], [0], marker="*", linestyle="None", color="k", markersize=5, label="Intact"),
    ]
    axis.legend(handles=threshold_handles, fontsize=6, loc="upper right", frameon=True)


#%% training

loss_history = []
val_loss_history = []
train_gap_history = []
val_gap_history = []
train_sat_history = []
val_sat_history = []
train_power_history = []
val_power_history = []

best_val_loss = float("inf")
epoch_opt = -1
state_init = deepcopy(net.state_dict())
state_opt = deepcopy(net.state_dict())

net.eval()
with torch.no_grad():
    _, _, _, det0_init_ref = net(psi_ms[:1].to(device))
initial_detector_power = detector_total_power(det0_init_ref).mean().item()
power_floor = power_floor_ratio * initial_detector_power

for epoch in tqdm(range(N_epoch)):
    net.train()
    running_loss = 0.0
    running_gap = 0.0
    running_sat = 0.0
    running_power = 0.0
    running_total = 0

    for field, _, _, _, _ in train_loader:
        optimizer.zero_grad(set_to_none=True)
        _, _, det, det0 = net(field.to(device))
        sep_loss = relative_separation_loss(det, det0)
        power_loss = power_regularization_loss(det, det0, power_floor)
        loss = sep_loss + power_reg_weight * power_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * len(field)
        running_gap += mean_relative_gap(det.detach(), det0.detach()) * len(field)
        running_sat += separation_satisfaction(det.detach(), det0.detach()) * len(field)
        running_power += mean_detector_power(det.detach()) * len(field)
        running_total += len(field)

    train_loss = running_loss / running_total
    loss_history.append(train_loss)
    train_gap_history.append(running_gap / running_total)
    train_sat_history.append(running_sat / running_total)
    train_power_history.append(running_power / running_total)

    net.eval()
    with torch.no_grad():
        _, _, det_val, det0_val = net(val_ds.field.to(device))
        val_sep_loss = relative_separation_loss(det_val, det0_val)
        val_power_loss = power_regularization_loss(det_val, det0_val, power_floor)
        val_loss = (val_sep_loss + power_reg_weight * val_power_loss).item()
        val_gap = mean_relative_gap(det_val, det0_val)
        val_sat = separation_satisfaction(det_val, det0_val)
        val_power = mean_detector_power(det_val)

    val_loss_history.append(val_loss)
    val_gap_history.append(val_gap)
    val_sat_history.append(val_sat)
    val_power_history.append(val_power)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epoch_opt = epoch
        state_opt = deepcopy(net.state_dict())
        print(
            f"Train loss = {train_loss:.4f} | train mean gap = {train_gap_history[-1]:.3f} "
            f"| train pass = {100 * train_sat_history[-1]:.1f}% | train power = {train_power_history[-1]:.3e} "
            f"| val loss = {val_loss:.4f} | val mean gap = {val_gap:.3f} "
            f"| val pass = {100 * val_sat:.1f}% | val power = {val_power:.3e}"
        )


checkpoint_path = "onn_ms_l2distance_generated_singlelayer_3det_trained.pt"
torch.save(
    {
        "model_state_dict": state_opt,
        "best_val_loss": best_val_loss,
        "best_epoch": epoch_opt,
        "initial_state_dict": state_init,
        "power_floor": power_floor,
        "class_names": class_names,
        "train_indices": train_idx,
        "val_indices": val_idx,
        "test_indices": test_idx,
        "dataset_files": {key: str(value) for key, value in DATASET_FILES.items()},
        "N_layer": len(net.metalayers),
        "distance": net.prop.D,
        "resol": net.resol,
        "det_center": net.det_center,
        "det_width": net.det_width,
    },
    checkpoint_path,
)
print(f"Saved trained ONN checkpoint to {checkpoint_path}")


#%% Test
fig, ax = plt.subplots(2, 1, sharex=True, tight_layout=True, figsize=(4, 3))
ax[0].plot(torch.tensor(loss_history), label="Train loss")
ax[0].plot(torch.tensor(val_loss_history), "--", label="Val loss")
axt = ax[0].twinx()
axt.plot(torch.tensor(train_gap_history) * 100, color="tab:green", label="Train mean gap")
axt.plot(torch.tensor(val_gap_history) * 100, "--", color="tab:red", label="Val mean gap")
ax[0].set(ylabel="Total loss")
axt.set(ylabel="Mean relative gap (%)")
ax[0].legend(loc="upper left", fontsize=7)
axt.legend(loc="lower right", fontsize=7)

ax[1].plot(torch.tensor(train_power_history), label="Train detector power")
ax[1].plot(torch.tensor(val_power_history), "--", label="Val detector power")
ax[1].axhline(power_floor, color="k", ls="--", lw=1, label="Power floor")
ax[1].axhline(initial_detector_power, color="0.5", ls=":", lw=1, label="Initial intact power")
ax[1].set(xlabel="Epoch", ylabel="Mean detector power")
ax[1].legend(fontsize=7)

net.load_state_dict(state_init)
net.eval()

with torch.no_grad():
    intensity_test_init, intensity0_test_init, det_test_init, det0_test_init = net(test_ds.field.to(device))

net.load_state_dict(state_opt)
net.eval()

with torch.no_grad():
    intensity_test, intensity0_test, det_test, det0_test = net(test_ds.field.to(device))

target_test = test_ds.label
test_sep_loss = relative_separation_loss(det_test, det0_test).item()
test_power_loss = power_regularization_loss(det_test, det0_test, power_floor).item()
test_loss = test_sep_loss + power_reg_weight * test_power_loss
test_gap = mean_relative_gap(det_test, det0_test)
test_sat = separation_satisfaction(det_test, det0_test)
test_l2_gap = relative_l2_gap(det_test, det0_test).cpu()
test_l2_gap_init = relative_l2_gap(det_test_init, det0_test_init).cpu()
test_power = mean_detector_power(det_test)
print(
    f"Best val loss = {best_val_loss:.4f} at epoch {epoch_opt} | "
    f"test loss = {test_loss:.4f} | test mean gap = {test_gap:.3f} | "
    f"test pass = {100 * test_sat:.1f}% | test power = {test_power:.3e} "
    f"| power floor = {power_floor:.3e}"
)

class_colors = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple"]
for class_id, class_name in enumerate(class_names):
    class_mask = target_test == class_id
    if not torch.any(class_mask):
        continue
    class_gap = test_l2_gap[class_mask]
    class_sat = (class_gap > relative_margin).float().mean().item()
    class_power = mean_detector_power(det_test[class_mask])
    print(
        f"Test {class_name}: mean gap = {class_gap.mean().item():.3f} | "
        f"pass = {100 * class_sat:.1f}% | power = {class_power:.3e} | "
        f"n = {int(class_mask.sum().item())}"
    )

fig, ax = plt.subplots(figsize=(3, 2), tight_layout=True)
ax.hist(test_l2_gap_init.numpy() * 100, bins=np.linspace(0, 150, 51), alpha=0.5, label="Before training")
ax.hist(test_l2_gap.numpy() * 100, bins=np.linspace(0, 150, 51), alpha=0.5, label="After training")
ax.axvline(relative_margin_pct, color="k", ls="--", lw=1, label="Threshold")
ax.set(
    xlabel="Per-sample relative L2 gap (%)",
    ylabel="Count",
    title="Detector separation from intact reference",
    xlim=(0, 150),
)
ax.legend()

viz_total = min(1500, len(det_test))
viz_per_class = max(1, viz_total // len(class_names))
viz_remainder = viz_total - viz_per_class * len(class_names)
viz_indices_by_class = class_sample_indices(target_test)
viz_indices_parts = []
for class_id in range(len(class_names)):
    class_take = viz_per_class + int(class_id < viz_remainder)
    class_indices = viz_indices_by_class[class_id][:class_take]
    if len(class_indices) > 0:
        viz_indices_parts.append(class_indices)
viz_indices = torch.cat(viz_indices_parts)
viz_batch_size = len(viz_indices)

viz_det_init = det_test_init[viz_indices].detach().cpu()
viz_det0_init = det0_test_init[0].detach().cpu()
viz_l2_gap_init = relative_l2_gap(
    viz_det_init,
    viz_det0_init.unsqueeze(0).expand(viz_batch_size, -1),
).cpu()

viz_det_trained = det_test[viz_indices].detach().cpu()
viz_det0_trained = det0_test[0].detach().cpu()
viz_l2_gap_trained = test_l2_gap[viz_indices]
viz_target_test = target_test[viz_indices].cpu()

all_coords = torch.cat(
    [
        viz_det_init.reshape(-1),
        viz_det_trained.reshape(-1),
        viz_det0_init.reshape(-1),
        viz_det0_trained.reshape(-1),
    ]
).cpu().numpy()
# det_limit_before = float(viz_det_init.max().item())
# det_limit_after = float(viz_det_trained.max().item())
det_limit_before = 4
det_limit_after = 12
det_limit_all = float(all_coords.max())

fig, ax = plt.subplots(1, 2, figsize=(6.6, 3.2), tight_layout=True, subplot_kw={"projection": "3d"})
plot_detector_scatter_hollow_circles_3d(
    ax[0],
    viz_det_init,
    viz_det0_init,
    viz_l2_gap_init,
    viz_target_test,
    relative_margin,
    title=f"Before training ({viz_batch_size} samples)",
)
set_detector_limits_3d(ax[0], det_limit_before)

plot_detector_scatter_hollow_circles_3d(
    ax[1],
    viz_det_trained,
    viz_det0_trained,
    viz_l2_gap_trained,
    viz_target_test,
    relative_margin,
    title=f"After training ({viz_batch_size} samples)",
)
set_detector_limits_3d(ax[1], det_limit_after)

fig, ax = plt.subplots(1, 2, figsize=(6.8, 3.2), tight_layout=True, subplot_kw={"projection": "3d"})
plot_detector_scatter_shape_threshold_3d(
    ax[0],
    viz_det_init,
    viz_det0_init,
    viz_l2_gap_init,
    viz_target_test,
    relative_margin,
    title=f"Before training ({viz_batch_size} samples)",
)
set_detector_limits_3d(ax[0], det_limit_before)

plot_detector_scatter_shape_threshold_3d(
    ax[1],
    viz_det_trained,
    viz_det0_trained,
    viz_l2_gap_trained,
    viz_target_test,
    relative_margin,
    title=f"After training ({viz_batch_size} samples)",
)
set_detector_limits_3d(ax[1], det_limit_after)

fig, ax = plt.subplots(
    2,
    len(class_names),
    figsize=(9, 5.6),
    tight_layout=True,
    subplot_kw={"projection": "3d"},
)
for class_id, class_name in enumerate(class_names):
    plot_detector_scatter_single_class_3d(
        ax[0, class_id],
        viz_det_init,
        viz_det0_init,
        viz_l2_gap_init,
        viz_target_test,
        class_id,
        relative_margin,
        title=class_name.capitalize(),
    )
    plot_detector_scatter_single_class_3d(
        ax[1, class_id],
        viz_det_trained,
        viz_det0_trained,
        viz_l2_gap_trained,
        viz_target_test,
        class_id,
        relative_margin,
        title="",
    )
    set_detector_limits_3d(ax[0, class_id], det_limit_before)
    set_detector_limits_3d(ax[1, class_id], det_limit_after)

for class_id in range(len(class_names)):
    ax[0, class_id].set_title(class_names[class_id].capitalize())
#     ax[1, class_id].set(xlabel="Detector 0 power")
# ax[0, 0].set(ylabel="Detector 1 power")
# ax[1, 0].set(ylabel="Detector 1 power")

x = np.arange(-W / 2 + W / 400, W / 2, W / 200) / mm
fig, ax = plt.subplots(len(net.metalayers), 1, figsize=(3, 2), sharey=True, sharex=True, tight_layout=True)
if len(net.metalayers) == 1:
    ax = [ax]
for ii, ml in enumerate(net.metalayers):
    phase_wrapped = ((ml.phase.detach().cpu().numpy() + np.pi) % (2 * np.pi)) - np.pi
    ax[ii].plot(x, phase_wrapped, lw=0.75)
ax[-1].set(xlim=(-W / 2 / mm, W / 2 / mm), xlabel=r"$x$ (mm)")
ax[0].set_title("Metagrating phase", fontsize=7)
ax[-1].set(
    xlim=(-W / 2 / mm, W / 2 / mm),
    ylim=(-np.pi, np.pi),
    yticks=(-np.pi, 0, np.pi),
    yticklabels=[r"$-\pi$", 0, r"$\pi$"],
)

init_l2_gap_all = relative_l2_gap(det_test_init, det0_test_init).cpu()

fig, ax = plt.subplots(
    2,
    len(class_names) * 2,
    figsize=(4.5 * len(class_names), 4.2),
    tight_layout=True,
    squeeze=False,
    gridspec_kw={"width_ratios": [3.5, 1.2] * len(class_names)},
)

det_x = np.arange(det_test_init.shape[1])
bar_width = 0.38
hard_defect_summaries = []
hard_defect_examples = []
for class_id, class_name in enumerate(class_names):
    class_candidates = torch.where(target_test == class_id)[0]
    if len(class_candidates) == 0:
        continue

    close_defect_idx = class_candidates[torch.argmin(init_l2_gap_all[class_candidates])].item()
    field_col = 2 * class_id
    barcode_col = field_col + 1

    defect_gap_before_pct = 100 * init_l2_gap_all[close_defect_idx].item()
    defect_gap_after_pct = 100 * test_l2_gap[close_defect_idx].item()
    max_depth_proxy_mm = (
        (test_ds.vertices[close_defect_idx, :, 1].max() - test_ds.vertices[close_defect_idx, :, 1]).max().item() / mm
    )
    hard_defect_summaries.append(
        f"{class_name}:"
        f"gap {defect_gap_before_pct:.1f}% -> {defect_gap_after_pct:.1f}%"
    )
    hard_defect_examples.append((class_id, class_name, close_defect_idx))

    ax[0, field_col].plot(x, intensity0_test_init[0].detach().cpu(), color="k", lw=0.9, label="Intact")
    ax[0, field_col].plot(
        x,
        intensity_test_init[close_defect_idx].detach().cpu(),
        color=class_colors[class_id],
        lw=0.9,
        label=class_name.capitalize(),
    )
    ax[0, field_col].set_title(f"Before: {class_name} field", fontsize=7)
    ax[0, barcode_col].bar(
        det_x - bar_width / 2,
        det0_test_init[0].detach().cpu().numpy(),
        width=bar_width,
        color="k",
        alpha=0.75,
        label="Intact",
    )
    ax[0, barcode_col].bar(
        det_x + bar_width / 2,
        det_test_init[close_defect_idx].detach().cpu().numpy(),
        width=bar_width,
        color=class_colors[class_id],
        alpha=0.75,
        label=class_name.capitalize(),
    )
    ax[0, barcode_col].set_xticks(det_x)
    ax[0, barcode_col].set_xticklabels([f"D{i}" for i in det_x], fontsize=7)
    ax[0, barcode_col].set_title(f"Before: {class_name} barcode", fontsize=7)

    ax[1, field_col].plot(x, intensity0_test[0].detach().cpu(), color="k", lw=0.9, label="Intact")
    ax[1, field_col].plot(
        x,
        intensity_test[close_defect_idx].detach().cpu(),
        color=class_colors[class_id],
        lw=0.9,
        label=class_name.capitalize(),
    )
    ax[1, field_col].set_title(f"After: {class_name} field", fontsize=7)
    ax[1, barcode_col].bar(
        det_x - bar_width / 2,
        det0_test[0].detach().cpu().numpy(),
        width=bar_width,
        color="k",
        alpha=0.75,
        label="Intact",
    )
    ax[1, barcode_col].bar(
        det_x + bar_width / 2,
        det_test[close_defect_idx].detach().cpu().numpy(),
        width=bar_width,
        color=class_colors[class_id],
        alpha=0.75,
        label=class_name.capitalize(),
    )
    ax[1, barcode_col].set_xticks(det_x)
    ax[1, barcode_col].set_xticklabels([f"D{i}" for i in det_x], fontsize=7)
    ax[1, barcode_col].set_title(f"After: {class_name} barcode", fontsize=7)

for field_col in range(0, len(class_names) * 2, 2):
    for axis in ax[:, field_col]:
        for xc in net.det_center:
            axis.axvline(x=(xc + net.det_width / 2) / mm, color="gray", ls="--", lw=0.75)
            axis.axvline(x=(xc - net.det_width / 2) / mm, color="gray", ls="--", lw=0.75)
        axis.set(xlim=(-W / 2 / mm, W / 2 / mm))

for class_id in range(len(class_names)):
    field_col = 2 * class_id
    barcode_col = field_col + 1
    ax[1, field_col].legend(loc="upper right", fontsize=7)
    ax[1, field_col].set(xlabel=r"$x$ (mm)", ylabel="Intensity")
    ax[0, field_col].set(ylabel="Intensity")
    ax[1, barcode_col].set(xlabel="Detector")
    ax[0, barcode_col].set_ylabel("Power")
    ax[1, barcode_col].set_ylabel("Power")

fig.suptitle(
    "Hard defect per class chosen by smallest pre-train gap\n" + " | ".join(hard_defect_summaries),
    fontsize=8,
)


#%% Example Schematics
A_aptr, a_wvg, l_horn = 27.4 * mm, 9.3 * mm, 27 * mm
R_E = A_aptr * l_horn / (A_aptr - a_wvg)
beta_wvg = k0 * np.sqrt(1 - (wvl / (2 * a_wvg)) ** 2)
dist = 160 * mm
theta = 60 * np.pi / 180
dx_src = A_aptr / 50
s = torch.arange(-A_aptr / 2, A_aptr / 2, dx_src) + dx_src / 2
v_src_x = -dist * np.sin(theta) + np.cos(theta) * s
v_src_y = 150 * mm + dist * np.cos(theta) + np.sin(theta) * s
v_src = torch.stack([v_src_x, v_src_y], dim=1)
amp = torch.cos(np.pi * s / A_aptr)
psi_src = amp * torch.exp(0.5j * beta_wvg * (s ** 2 / R_E))

fig, ax = plt.subplots(1, len(hard_defect_examples), sharex=True, sharey=True, figsize=(6, 2.6), tight_layout=True)
if len(hard_defect_examples) == 1:
    ax = [ax]

for axis, (class_id, class_name, sample_idx) in zip(ax, hard_defect_examples):
    vertices_sample = test_ds.vertices[sample_idx].detach().cpu()
    v_rail_sample = test_ds.v_rail[sample_idx].detach().cpu()
    psi_rail_sample = test_ds.psi_rail[sample_idx].detach().cpu()
    psi_ms_sample = test_ds.field[sample_idx].detach().cpu()

    axis.plot(*vertices_sample.T.numpy() / mm, color="0.55", lw=0.8, ls="--", zorder=-100)
    axis.plot(*v_src.detach().cpu().T.numpy() / mm, color="0.7", lw=0.8, zorder=-100)
    axis.plot(*v_ms.detach().cpu().T.numpy() / mm, color="0.7", lw=0.8, zorder=-100)
    axis.scatter(
        *v_src.detach().cpu().T.numpy() / mm,
        c=torch.abs(psi_src.detach().cpu()) ** 2,
        s=6,
        vmin=0,
        cmap=plt.cm.inferno,
    )
    axis.scatter(
        *v_rail_sample[1:].T.numpy() / mm,
        c=torch.abs(psi_rail_sample) ** 2,
        s=4,
        vmin=0,
        cmap=plt.cm.inferno,
    )
    axis.scatter(
        *v_ms.detach().cpu().T.numpy() / mm,
        c=torch.abs(psi_ms_sample) ** 2,
        s=4,
        vmin=0,
        cmap=plt.cm.inferno,
    )
    axis.set(
        xlim=(-125, 125),
        ylim=(50, 300),
        xlabel=r"$x$ (mm)",
        title=class_name,
    )

ax[0].set_ylabel(r"$y$ (mm)")
fig.suptitle("Example field schematics for the hard-defect cases", fontsize=8)


fig, ax = plt.subplots(1, len(hard_defect_examples), sharex=True, sharey=True, figsize=(6, 2.6), tight_layout=True)
if len(hard_defect_examples) == 1:
    ax = [ax]

for axis, (class_id, class_name, sample_idx) in zip(ax, hard_defect_examples):
    vertices_sample = test_ds.vertices[sample_idx].detach().cpu()
    axis.plot(*vertices_sample.T.numpy() / mm, color="k", lw=1.0)
    axis.plot(*v_src.detach().cpu().T.numpy() / mm, color="0.5", lw=0.9)
    axis.plot(*v_ms.detach().cpu().T.numpy() / mm, color="0.5", lw=0.9)
    axis.set(
        xlim=(-125, 125),
        ylim=(50, 300),
        xlabel=r"$x$ (mm)",
        title=class_name,
    )

ax[0].set_ylabel(r"$y$ (mm)")
fig.suptitle("Example geometry schematics", fontsize=8)
