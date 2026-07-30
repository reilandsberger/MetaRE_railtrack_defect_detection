#%%
import torch
import torch.nn as nn
import torch.fft as fft
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import plot_setting

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
# device = torch.device("cpu")


mm = 1e3
wvl = 12 * mm
k0 = 2*np.pi/wvl 

# y_ms = 260 * mm
# W = 20 * wvl
# dx_ms = wvl/10
# v_ms_x = torch.arange(-W/2, W/2, dx_ms) + dx_ms/2
# v_ms_y = torch.zeros_like(v_ms_x) + y_ms 
# v_ms = torch.stack([v_ms_x,v_ms_y], dim=1)

y_ms = 250 * mm
W = 20 * wvl
dx_ms = wvl/10
v_ms_x = torch.arange(-W/2, W/2, dx_ms) + dx_ms/2
v_ms_y = torch.zeros_like(v_ms_x) + y_ms 
v_ms = torch.stack([v_ms_x,v_ms_y], dim=1)


range_defect = torch.load("range_dist_12mm.pt")
depth_defect = torch.load("depth_dist_12mm.pt")
width_defect = torch.abs(range_defect[:,0]-range_defect[:,1])*290.0129
defect_size = width_defect * depth_defect

normal_threshold = 2.5e7

nominal_defect_label = torch.arange(len(depth_defect), dtype=torch.long) % 2  # 0=ellipse, 1=triangle
defect_label = nominal_defect_label  # 0=normal, 1=ellipse, 2=triangle
class_names = ["ellipse", "triangle"]

## Defect size distribution
size_scale = 1e6
ellipse_size = (defect_size[nominal_defect_label == 0] / size_scale).cpu().numpy()
triangle_size = (defect_size[nominal_defect_label == 1] / size_scale).cpu().numpy()
hist_max = np.percentile(np.concatenate([ellipse_size, triangle_size]), 99.5)
bins = np.linspace(0, hist_max, 60)

fig, ax = plt.subplots(figsize=(4, 3), tight_layout=True)
ax.hist(ellipse_size, bins=bins, density=True, alpha=0.45, color="tab:blue", label="Ellipse")
ax.hist(triangle_size, bins=bins, density=True, alpha=0.45, color="tab:orange", label="Triangle")
ax.axvline(normal_threshold / size_scale, color="k", ls="--", lw=1, label="Threshold")
ax.set(
    xlabel="Defect size = depth * width (x1e6)",
    ylabel="Density",
    xlim=(0, hist_max),
)
ax.set_title("Defect size distribution", fontsize=8)
ax.legend(frameon=False, fontsize=7)

psi_ms = torch.load("psi_ms_dist_12mm.pt").to(torch.complex64)
psi_ms = psi_ms / torch.sqrt(torch.mean(psi_ms.abs()**2))
psi_ms0 = torch.load("psi_ms_nodefect_12mm.pt").to(torch.complex64)
psi_ms0 = psi_ms0 / torch.sqrt(torch.mean(psi_ms0.abs()**2))
# psi_ms = psi_ms *torch.sqrt(0.5/torch.mean(torch.abs(psi_ms)**2)) # normalize to have the average intensity of 0.5
# psi = psi_ms[:20]


class FieldDataset(Dataset):
    def __init__(self, 
                field=psi_ms, 
                defect=range_defect, 
                depth=depth_defect, 
                width=width_defect,
                label=defect_label,
                def_min=1080, def_max=1460
        ):
        self.field = field
        self.raw_range = defect
        self.norm_range = (def_max-defect)/(def_max-def_min)  # normalize within 950-1465 in reverse direction
        self.depth = depth
        self.width = width
        self.label = label.to(torch.long)
        self.resol = field.shape[1]
        self.x = torch.arange(0, 1, 1/self.resol) + 0.5/self.resol
        self.location_indicator = 0.0+ (self.x.reshape(1,-1)<self.norm_range[:,0].reshape(-1,1)) * (self.x.reshape(1,-1)>self.norm_range[:,1].reshape(-1,1))

    def __getitem__(self, index):
        return self.field[index], self.label[index], self.raw_range[index], self.depth[index], self.width[index]
    def __len__(self):
        return len(self.field)
    
class Propagator(nn.Module):
    def __init__(
        self,
        distance,
        width: float=W,
        resol: int=200,
        wavelength: float=wvl,
        padding_factor: int=10,
        ):
        super(Propagator, self).__init__()
        
        self.W = width
        self.dx = width/resol
        self.D = distance
        self.k0 = 2*np.pi/wavelength
        self.resol = resol
        self.N_pad = padding_factor

        self.kx = 2*np.pi* fft.fftfreq(self.resol*self.N_pad, d=self.dx, dtype=torch.float)
        self.ky = torch.sqrt(self.k0**2 - self.kx**2 + 0j)

    def forward(self, signal):
        # if len(signal.shape)>3:
        #     signal = signal[:,:,:,0] + 1j*signal[:,:,:,1]

        if signal.dim()==1:
            signal = signal.reshape(1,-1)

        zeropad = torch.zeros(len(signal), self.resol*self.N_pad).to(signal.device) + 0j
        zeropad[:, :self.resol] = signal
        signal = zeropad.clone()
        signal_F = fft.fft(signal+0j)
        signal_F = signal_F * torch.exp(1j * self.ky.to(signal.device) * self.D)
        signal = torch.fft.ifft(signal_F)
        return signal[:, :self.resol]


class Metalayer(nn.Module):
    def __init__(
        self,
        resol: int=200,
        phase_only: bool=True,
        phase_init: float=0.0,
        std_phase: float=90*np.pi/180,
        dummy:int=0,
        sym:bool=False,
        ):
        super(Metalayer, self).__init__()
        self.sym = sym
        self.N = resol
        self.dummy=dummy
        generator = torch.Generator()
        generator.manual_seed(dummy)
        self.phase_init = phase_init + std_phase * torch.randn(self.N, generator=generator)
        self.phase = nn.Parameter(self.phase_init.clone())
        # self.amp = torch.ones(self.N) if phase_only else nn.Parameter(torch.ones(self.N, self.N))
    def forward(self, signal):
        # signal = signal * (self.amp*torch.exp(1j*self.phase)).to(signal.device)
        if self.sym:
            signal = signal * torch.exp(0.5j*(self.phase+self.phase.flip(dims=(0,)))).to(signal.device)
        else:
            signal = signal * torch.exp(1j*self.phase).to(signal.device)
        return signal


class ONN(nn.Module):
    def __init__(
            self, 
            N_layer:int=2, 
            distance:float=10*mm, 
            sym:bool=False,
            det_center:float=np.array([-W/6, W/6]),
            det_width:float=W/40,
            width:float=W,
            resol:int=200
        ):
        super(ONN, self).__init__()
        self.resol=resol
        self.prop = Propagator(distance=distance, resol=resol)
        self.metalayers = nn.ModuleList([Metalayer(dummy=i, sym=sym, resol=resol) for i in range(N_layer)])

        dx = width/resol
        x = torch.arange(-width/2+dx/2, width/2, dx)
        self.det_width =det_width
        self.det_center = det_center
        self.detector = torch.stack([torch.abs(x-xc)<det_width/2 for xc in det_center], dim=1)

    def fields_at_metasurface_planes(self, signal):
        # Return the complex field sampled on each metasurface plane before modulation.
        if signal.dim() == 1:
            signal = signal.reshape(1, -1)

        plane_fields = []
        for layer_idx, ml in enumerate(self.metalayers):
            plane_fields.append(signal.clone())
            if layer_idx < len(self.metalayers) - 1:
                signal = self.prop(ml(signal))

        return torch.stack(plane_fields, dim=1)

    def forward(self, signal):
        for ml in self.metalayers:
            signal = self.prop(ml(signal))
        intensity = signal.abs()**2

        signal0 = psi_ms0.reshape(1,-1).clone()
        for ml in self.metalayers:
            signal0 = self.prop(ml(signal0))
        intensity0 = signal0.abs()**2
        # dint = intensity - intensity0.to(signal.device)

        det = torch.sum(intensity.unsqueeze(dim=2)*self.detector.to(device).unsqueeze(dim=0), dim=1)
        det0 = torch.sum(intensity0.to(device).unsqueeze(dim=2)*self.detector.to(device).unsqueeze(dim=0), dim=1)
        return intensity, intensity0, det, det0



train_ds = FieldDataset(
    field=psi_ms[:80000],
    defect=range_defect[:80000],
    depth=depth_defect[:80000],
    width=width_defect[:80000],
    label=defect_label[:80000],
)
val_ds = FieldDataset(
    field=psi_ms[80000:90000],
    defect=range_defect[80000:90000],
    depth=depth_defect[80000:90000],
    width=width_defect[80000:90000],
    label=defect_label[80000:90000],
)
test_ds = FieldDataset(
    field=psi_ms[90000:100000],
    defect=range_defect[90000:100000],
    depth=depth_defect[90000:100000],
    width=width_defect[90000:100000],
    label=defect_label[90000:100000],
)


batch_size = 2000
train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_size)
net = ONN(N_layer=2, distance=50*mm, sym=False).to(device)
N_epoch = 300
lr = 1e-3


optimizer = optim.Adam(net.parameters(), lr=lr)
from copy import deepcopy

## 79%
# relative_margin_pct = 50.0
# relative_margin = relative_margin_pct / 100.0
# power_floor_ratio = 1.5
# power_reg_weight = 0.1

relative_margin_pct = 40.0
relative_margin = relative_margin_pct / 100.0
power_floor_ratio = 1.5
power_reg_weight = 0.1

def relative_l2_gap(det, det0, eps=1e-12):
    return torch.linalg.norm(det - det0, dim=1) / torch.linalg.norm(det0, dim=1).clamp_min(eps)


def detector_total_power(det):
    return det.sum(dim=1)


def relative_separation_loss(det, det0, margin=relative_margin, eps=1e-12):
    # Enforce a minimum normalized L2 separation from the intact detector response.
    gap = relative_l2_gap(det, det0, eps=eps)
    margin_violation = torch.relu(margin - gap)
    return margin_violation.mean()


def power_regularization_loss(det, det0, power_floor, eps=1e-12):
    det_power = detector_total_power(det)
    det0_power = detector_total_power(det0)
    det_penalty = torch.relu(power_floor - det_power) / (power_floor + eps)
    det0_penalty = torch.relu(power_floor - det0_power) / (power_floor + eps)
    return det_penalty.mean() + det0_penalty.mean()


def separation_satisfaction(det, det0, margin=relative_margin, eps=1e-12):
    gap = relative_l2_gap(det, det0, eps=eps)
    return (gap > margin).float().mean().item()


def mean_relative_gap(det, det0, eps=1e-12):
    return relative_l2_gap(det, det0, eps=eps).mean().item()


def mean_detector_power(det):
    return detector_total_power(det).mean().item()


def class_sample_indices(label_tensor, max_per_class=None):
    picks = {}
    for class_id in range(len(class_names)):
        matches = torch.where(label_tensor == class_id)[0]
        if max_per_class is not None:
            matches = matches[:max_per_class]
        picks[class_id] = matches
    return picks


def representative_indices(label_tensor):
    picks = []
    for class_id in range(len(class_names)):
        matches = torch.where(label_tensor == class_id)[0]
        if len(matches) == 0:
            continue
        rand_idx = torch.randint(len(matches), (1,)).item()
        picks.append(matches[rand_idx].item())
    return picks


def plot_detector_panel(axis, detector_power, highlight_idx=None):
    values = detector_power.detach().cpu().numpy()
    labels = [f"D{i}" for i in range(len(values))]
    colors = ["0.75"] * len(values)
    if highlight_idx is not None and 0 <= highlight_idx < len(values):
        colors[highlight_idx] = "tab:orange"
    axis.bar(np.arange(len(values)), values, color=colors, width=0.7)
    axis.set_xticks(np.arange(len(values)))
    axis.set_xticklabels(labels, fontsize=7)
    axis.set_ylim(0, max(values.max() * 1.15, 1e-6))
    axis.tick_params(axis="y", labelsize=6, length=2)
    for spine in ["top", "right"]:
        axis.spines[spine].set_visible(False)


def draw_margin_circle_2d(axis, center, radius, color="k", alpha=0.10, lw=1.0):
    circle = plt.Circle(
        center,
        radius,
        facecolor="none",
        edgecolor=color,
        linestyle="--",
        lw=lw,
    )
    axis.add_patch(circle)


def plot_detector_scatter_2d(axis, det, det0, relative_gap, margin, title):
    det_np = det.detach().cpu().numpy()
    det0_np = det0.detach().cpu().numpy()
    gap_np = relative_gap.detach().cpu().numpy()
    pass_mask = gap_np > margin
    circle_radius = margin * np.linalg.norm(det0_np)

    draw_margin_circle_2d(axis, det0_np[:2], circle_radius, color="k", alpha=0.10, lw=1.0)
    axis.scatter(
        det_np[~pass_mask, 0],
        det_np[~pass_mask, 1],
        s=2,
        c="tab:red",
        alpha=0.65,
        label=f"Inside margin ({np.sum(~pass_mask)})",
    )
    axis.scatter(
        det_np[pass_mask, 0],
        det_np[pass_mask, 1],
        s=2,
        c="tab:blue",
        alpha=0.65,
        label=f"Outside margin ({np.sum(pass_mask)})",
    )
    axis.scatter(
        [det0_np[0]],
        [det0_np[1]],
        s=30,
        c="k",
        marker="*",
        label="Intact",
    )
    axis.set(
        xlabel="Detector 0 power",
        ylabel="Detector 1 power",
        title=title,
    )
    # axis.set_aspect("equal", adjustable="box")
    axis.annotate(
        "L2 margin boundary",
        xy=(det0_np[0], det0_np[1] + circle_radius),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=7,
    )


#%% Detector-separation training
loss_history = []
val_loss_history = []
train_gap_history = []
val_gap_history = []
train_sat_history = []
val_sat_history = []
train_power_history = []
val_power_history = []

best_val_loss = float("inf")
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

    for field, label, _, _, _ in train_loader:
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
        _, _, det_val, det0_val = net(val_ds[:][0].to(device))
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

checkpoint_path = "onn_ms_l2distance_trained.pt"
torch.save(
    {
        "model_state_dict": state_opt,
        "best_val_loss": best_val_loss,
        "best_epoch": epoch_opt,
        "initial_state_dict": state_init,
        "power_floor": power_floor,
        "N_layer": len(net.metalayers),
        "distance": net.prop.D,
        "resol": net.resol,
        "det_center": net.det_center,
        "det_width": net.det_width,
    },
    checkpoint_path,
)
print(f"Saved trained ONN checkpoint to {checkpoint_path}")


#%% diagram

net.load_state_dict(state_opt)
net.eval()

psi_ms_list = torch.load("psi_ms_dist_12mm.pt")
psi_rail_list = torch.load("psi_rail_dist_12mm.pt")
v_rail_list = torch.load("v_rail_dist_12mm.pt")

idx = 0
psi_ms1 = test_ds[idx][0]
psi_rail = psi_rail_list[90000+idx]
v_rail = v_rail_list[90000+idx]

import cv2
height = 180 * mm
img = cv2.imread("crosssection.png", cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnt = max(contours, key=cv2.contourArea).squeeze()
cnt_x, cnt_y = cnt[:,0], cnt[:,1]
cnt_x = cnt_x - (cnt_x.max()+cnt_x.min())/2
cnt_y = -cnt_y
cnt_y = cnt_y - cnt_y.min()
cnt = np.column_stack([cnt_x, cnt_y])
cnt = cnt * height/cnt_y.max()
cnt = np.concat([cnt[1180:], cnt[:1180]], axis=0)
cnt_sym = []
for ii in range(len(cnt)):
    x1, y1 = cnt[ii]
    x2, y2 = cnt[-ii]
    cnt_sym.append(((x1-x2)/2, (y1+y2)/2))

vertices = torch.tensor(cnt_sym, dtype=torch.float)

A_aptr, B_aptr, a_wvg, b_wvg, l_horn= 27.4*mm, 21.9*mm, 9.3*mm, 6.2*mm, 27*mm
R_E = A_aptr * l_horn / (A_aptr-a_wvg)
# R_H = B_aptr * l_horn / (B_aptr-b_wvg)
beta_wvg = k0 * np.sqrt(1-(wvl/(2*a_wvg))**2)
dist = 160 * mm
theta = 60 * np.pi/180
norm_vec_src = torch.tensor([[np.sin(theta, dtype=np.float32),-np.cos(theta, dtype=np.float32)]])
dx_src = A_aptr/50
s = torch.arange(-A_aptr/2, A_aptr/2, dx_src) + dx_src/2
v_src_x = -dist*np.sin(theta) + np.cos(theta)*s
v_src_y = 150*mm + dist*np.cos(theta) + np.sin(theta)* s
v_src = torch.stack([v_src_x,v_src_y], dim=1)
amp = torch.cos(np.pi*s/A_aptr)
psi_src = amp * torch.exp(0.5j*beta_wvg*(s**2/R_E))
intensity_src = torch.abs(psi_src)**2
intensity_src = intensity_src / torch.mean(intensity_src)

with torch.no_grad():
    intensity_det, _, det, _ = net(test_ds[[idx]][0].to(device))
    field_ms2 = net.fields_at_metasurface_planes(test_ds[[idx]][0].to(device))
    intensity_det = intensity_det[0] / torch.mean(intensity_det)

    intensity_ms1 = torch.abs(psi_ms1.cpu())**2
    intensity_ms1 = intensity_ms1 / torch.mean(intensity_ms1)

    intensity_ms2 = torch.abs(field_ms2[0, 1].cpu())**2
    intensity_ms2 = intensity_ms2 / torch.mean(intensity_ms2)

    intensity_rail = torch.abs(psi_rail)**2
    intensity_rail = intensity_rail / torch.mean(intensity_rail)

fig = plt.figure(tight_layout=True, figsize=(4,4))
ax = fig.add_subplot(projection="3d")

detector_colors = ["purple", "purple"]
detector_masks = net.detector.cpu().T


ax.plot(v_src[:,0]/mm, v_src[:,1]/mm, v_src[:,0]*0, 'k')
ax.fill_between(
    v_src[:,0]/mm, v_src[:,1]/mm, intensity_src, 
    v_src[:,0]/mm, v_src[:,1]/mm, 0, 
    linewidth=0.5, alpha=0.2,
)

ax.plot(vertices[:,0]/mm, vertices[:,1]/mm, vertices[:,0]*0, 'k--', lw=0.5)
ax.plot(v_rail[:,0]/mm, v_rail[:,1]/mm, v_rail[:,0]*0, 'k')
ax.fill_between(
    v_rail[:-1,0]/mm, v_rail[:-1,1]/mm, intensity_rail, 
    v_rail[:-1,0]/mm, v_rail[:-1,1]/mm, 0, 
    linewidth=0.5, alpha=0.2,
)

x = np.arange(-W/2 + W/400, W/2, W/200) / mm
y_ms1 = y_ms/mm
y_ms2 = y_ms1 +50
y_det = y_ms2 +50
ax.plot(x, y_ms1, 0, 'k')
ax.fill_between(
    x, y_ms1, intensity_ms1, 
    x, y_ms1, 0, 
    linewidth=0.5, alpha=0.2
)


ax.plot(x, y_ms2, 0, 'k')
ax.fill_between(
    x, y_ms2, intensity_ms2, 
    x, y_ms2, 0, 
    linewidth=0.5, alpha=0.2
)

ax.plot(x, y_det, 0, 'k')
ax.fill_between(
    x, y_det, intensity_det.cpu(), 
    x, y_det, 0,
    color="gray", 
    linewidth=0.5, alpha=0.2
)
for det_idx, det_mask in enumerate(detector_masks):
    det_mask_np = det_mask.numpy()
    det_color = detector_colors[det_idx % len(detector_colors)]
    det_x = x[det_mask_np]
    det_z = intensity_det.cpu().numpy()[det_mask_np]
    if len(det_x) == 0:
        continue
    ax.fill_between(
        det_x,
        y_det,
        det_z,
        det_x,
        y_det,
        0,
        color=det_color,
        alpha=0.75,
        linewidth=0,
    )
    x_left = det_x[0]
    x_right = det_x[-1]
    ax.plot([x_left, x_left], [y_det, y_det], [0, det_z.max()], color=det_color, lw=1.0, alpha=0.95)
    ax.plot([x_right, x_right], [y_det, y_det], [0, det_z.max()], color=det_color, lw=1.0, alpha=0.95)
    ax.text(
        0.5 * (x_left + x_right),
        y_det + 6,
        det_z.max() + 0.08,
        f"$D_{det_idx}$",
        color=det_color,
        fontsize=8,
        ha="center",
    )

ax.set(
    xlim=(-125,125),
    ylim=(0,350),
    zlim=(0, max(
        float(torch.max(intensity_rail).item()),
        float(torch.max(intensity_ms1).item()),
        float(torch.max(intensity_ms2).item()),
        float(torch.max(intensity_det).item()),
    ) * 1.08),
    xlabel=r"$x$ (mm)",
    ylabel=r"$y$ (mm)",
)
ax.grid(False)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor("white")
ax.yaxis.pane.set_edgecolor("white")
ax.zaxis.pane.set_edgecolor("white")
ax.tick_params(axis="both", which="major", labelsize=7, pad=1)
ax.view_init(elev=40, azim=-45)
ax.set_box_aspect((1.0, 1.0, 0.5))


##2d
fig, ax = plt.subplots(tight_layout=True, figsize=(2.5,3))

detector_colors = ["purple", "purple"]
detector_masks = net.detector.cpu().T


ax.plot(v_src[:,0]/mm, v_src[:,1]/mm, 'k')
# ax.fill_between(
#     v_src[:,0]/mm, v_src[:,1]/mm, intensity_src, 
#     v_src[:,0]/mm, v_src[:,1]/mm, 0, 
#     linewidth=0.5, alpha=0.2,
# )

ax.plot(vertices[:,0]/mm, vertices[:,1]/mm, 'k--', lw=0.5)
ax.plot(v_rail[:,0]/mm, v_rail[:,1]/mm,  'k')
# ax.fill_between(
#     v_rail[:-1,0]/mm, v_rail[:-1,1]/mm, intensity_rail, 
#     v_rail[:-1,0]/mm, v_rail[:-1,1]/mm, 0, 
#     linewidth=0.5, alpha=0.2,
# )

x = np.arange(-W/2 + W/400, W/2, W/200) / mm
y_ms1 = y_ms/mm
y_ms2 = y_ms1 +50
y_det = y_ms2 +50
ax.plot(x, 0*x+y_ms1, 'k')
# ax.fill_between(
#     x, y_ms1, intensity_ms1, 
#     x, y_ms1, 0, 
#     linewidth=0.5, alpha=0.2
# )


ax.plot(x, 0*x+y_ms2, 'k')
# ax.fill_between(
#     x, y_ms2, intensity_ms2, 
#     x, y_ms2, 0, 
#     linewidth=0.5, alpha=0.2
# )

ax.plot(x, 0*x+y_det, 'k')
# ax.fill_between(
#     x, y_det, intensity_det.cpu(), 
#     x, y_det, 0,
#     color="gray", 
#     linewidth=0.5, alpha=0.2
# )
for det_idx, det_mask in enumerate(detector_masks):
    det_mask_np = det_mask.numpy()
    det_color = detector_colors[det_idx % len(detector_colors)]
    det_x = x[det_mask_np]
    det_z = intensity_det.cpu().numpy()[det_mask_np]
    if len(det_x) == 0:
        continue
    # ax.fill_between(
    #     det_x,
    #     y_det,
    #     det_z,
    #     det_x,
    #     y_det,
    #     0,
    #     color=det_color,
    #     alpha=0.75,
    #     linewidth=0,
    # )
    x_left = det_x[0]
    x_right = det_x[-1]
    ax.plot([x_left, x_right], [y_det, y_det],  color=det_color, lw=2.0)
    ax.text(
        0.5 * (x_left + x_right),
        y_det - 20,
        f"$D_{det_idx}$",
        color=det_color,
        fontsize=8,
        ha="center",
    )

ax.set(
    xlim=(-150,150),
    ylim=(0,355),
    xlabel=r"$x$ (mm)",
    ylabel=r"$y$ (mm)",
)
ax.set_aspect("equal")
# ax.grid(False)
# ax.xaxis.pane.fill = False
# ax.yaxis.pane.fill = False
# ax.zaxis.pane.fill = False
# ax.xaxis.pane.set_edgecolor("white")
# ax.yaxis.pane.set_edgecolor("white")
# ax.zaxis.pane.set_edgecolor("white")
# ax.tick_params(axis="both", which="major", labelsize=7, pad=1)
# ax.view_init(elev=20, azim=-45)
# ax.set_box_aspect((1.0, 1.0, 0.5))






#%% Test
net.load_state_dict(state_init)
net.eval()

with torch.no_grad():
    intensity_test_init, intensity0_test_init, det_test_init, det0_test_init = net(test_ds[:][0].to(device))

net.load_state_dict(state_opt)
net.eval()

with torch.no_grad():
    intensity_test, intensity0_test, det_test, det0_test = net(test_ds[:][0].to(device))

target_test = test_ds[:][1]
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

fig, ax = plt.subplots(figsize=(3,2), tight_layout=True)
ax.hist(test_l2_gap_init.numpy() * 100, bins=np.linspace(0,400,101),  alpha=0.5, label=r"Before training")
ax.hist(test_l2_gap.numpy() * 100, bins=np.linspace(0,400,101), alpha=0.5, label=r"After training")
ax.axvline(relative_margin_pct, color="k", ls="--", lw=1, label="Threshold")
ax.set(
    xlabel="Per-sample relative L2 gap (%)",
    ylabel="Count",
    title="Detector separation from intact reference",
    xlim=(0,400),
)
ax.legend()

viz_batch_size = min(1000, len(det_test))
viz_det_init = det_test_init[:viz_batch_size]
viz_det0_init = det0_test_init[0]
viz_l2_gap_init = relative_l2_gap(viz_det_init, det0_test_init[:1].expand(viz_batch_size, -1))

viz_det_trained = det_test[:viz_batch_size]
viz_det0_trained = det0_test[0]
viz_l2_gap_trained = test_l2_gap[:viz_batch_size]


all_x = torch.cat([viz_det_init[:, 0], viz_det_trained[:, 0], viz_det0_init[0:1], viz_det0_trained[0:1]]).cpu().numpy()
all_y = torch.cat([viz_det_init[:, 1], viz_det_trained[:, 1], viz_det0_init[1:2], viz_det0_trained[1:2]]).cpu().numpy()
x_pad = 0.05 * max(all_x.max() - all_x.min(), 1e-6)
y_pad = 0.05 * max(all_y.max() - all_y.min(), 1e-6)
xymax = max(all_x.max(), all_y.max())/2

fig, ax = plt.subplots(1, 2, figsize=(4.7,2.5), tight_layout=True)
plot_detector_scatter_2d(
    ax[0],
    viz_det_init,
    viz_det0_init,
    viz_l2_gap_init,
    relative_margin,
    title=f"Before training ({viz_batch_size} samples)",
)
ax[0].set_xlim(0, xymax/3/power_floor_ratio)
ax[0].set_ylim(0, xymax/3/power_floor_ratio)
ax[0].legend(loc="upper left", fontsize=7)
ax[0].set(xticks=np.linspace(0,1,6),yticks=np.linspace(0,1,6))

plot_detector_scatter_2d(
    ax[1],
    viz_det_trained,
    viz_det0_trained,
    viz_l2_gap_trained,
    relative_margin,
    title=f"After training ({viz_batch_size} samples)",
)
ax[1].set_xlim(0, xymax/3)
ax[1].set_ylim(0, xymax/3)
ax[1].legend(loc="upper left", fontsize=7)
ax[1].set(xticks=np.linspace(0,1.5,4),yticks=np.linspace(0,1.5,4))

x = np.arange(-W/2 + W/400, W/2, W/200) / mm
fig, ax = plt.subplots(len(net.metalayers), 1, figsize=(3, 2), sharey=True, sharex=True, tight_layout=True)
if len(net.metalayers) == 1:
    ax = [ax]
for ii, ml in enumerate(net.metalayers):
    phase_wrapped = ((ml.phase.detach().cpu().numpy() + np.pi) % (2 * np.pi)) - np.pi
    ax[ii].plot(x, phase_wrapped, lw=0.75)
ax[-1].set(xlim=(-W/2 / mm, W/2 / mm), xlabel=r"$x$ (mm)")
ax[0].set_title("Metagrating phase", fontsize=7)
ax[-1].set(xlim=(-W/2/mm,W/2/mm), ylim=(-np.pi,np.pi), yticks=(-np.pi,0,np.pi), 
           yticklabels=[r"$-\pi$", 0, r"$\pi$"])


defect_candidates = torch.where(target_test != 0)[0]
defect_candidates = torch.arange(len(test_ds))
if len(defect_candidates) > 0:
    init_l2_gap_all = relative_l2_gap(det_test_init, det0_test_init)
    # close_defect_idx = defect_candidates[torch.argmin(init_l2_gap_all[defect_candidates])].item()
    close_defect_idx = defect_candidates[torch.argsort(
        torch.abs(intensity_test_init[defect_candidates].cpu() - intensity0_test_init.cpu()).mean(dim=1)
    )[502]].item()
    # close_defect_idx = 9295
else:
    init_l2_gap_all = relative_l2_gap(det_test_init, det0_test_init)
    # close_defect_idx = torch.argmin(init_l2_gap_all).item()
    close_defect_idx = torch.argmin(torch.abs(intensity_test_init.cpu() - intensity0_test_init.cpu()).mean(dim=1))
defect_label_name = class_names[target_test[close_defect_idx].item()]
defect_depth_mm = test_ds[close_defect_idx][3].item() / mm
defect_width_mm = test_ds[close_defect_idx][4].item() / mm
defect_gap_before_pct = 100 * init_l2_gap_all[close_defect_idx].item()
defect_gap_after_pct = 100 * test_l2_gap[close_defect_idx].item()

fig, ax = plt.subplots(
    2,
    2,
    figsize=(4.4, 3.2),
    tight_layout=True,
    squeeze=False,
    gridspec_kw={"width_ratios": [3.5, 1.2]},
)

ax[0, 0].plot(x, intensity0_test_init[0].cpu(), color="k", lw=0.9, label="Intact")
ax[0, 0].plot(x, intensity_test_init[close_defect_idx].cpu(), color="tab:blue", lw=0.9, label="Defect")
ax[0, 0].set_title("Before training: field", fontsize=7)
det_x = np.arange(det_test_init.shape[1])
bar_width = 0.38
ax[0, 1].bar(
    det_x - bar_width / 2,
    det0_test_init[0].detach().cpu().numpy(),
    width=bar_width,
    color="k",
    alpha=0.75,
    label="Intact",
)
ax[0, 1].bar(
    det_x + bar_width / 2,
    det_test_init[close_defect_idx].detach().cpu().numpy(),
    width=bar_width,
    color="tab:blue",
    alpha=0.75,
    label="Defect",
)
ax[0, 1].set_xticks(det_x)
ax[0, 1].set_xticklabels([f"D{i}" for i in det_x], fontsize=7)
ax[0, 1].set_title("Before training: barcode", fontsize=7)

ax[1, 0].plot(x, intensity0_test[0].cpu(), color="k", lw=0.9, label="Intact")
ax[1, 0].plot(x, intensity_test[close_defect_idx].cpu(), color="tab:blue", lw=0.9, label="Defect")
ax[1, 0].set_title("After training: field", fontsize=7)
ax[1, 1].bar(
    det_x - bar_width / 2,
    det0_test[0].detach().cpu().numpy(),
    width=bar_width,
    color="k",
    alpha=0.75,
    label="Intact",
)
ax[1, 1].bar(
    det_x + bar_width / 2,
    det_test[close_defect_idx].detach().cpu().numpy(),
    width=bar_width,
    color="tab:blue",
    alpha=0.75,
    label="Defect",
)
ax[1, 1].set_xticks(det_x)
ax[1, 1].set_xticklabels([f"D{i}" for i in det_x], fontsize=7)
ax[1, 1].set_title("After training: barcode", fontsize=7)

for axis in ax[:, 0]:
    for xc in net.det_center:
        axis.axvline(x=(xc + net.det_width / 2) / mm, color="gray", ls="--", lw=0.75)
        axis.axvline(x=(xc - net.det_width / 2) / mm, color="gray", ls="--", lw=0.75)
    axis.set(xlim=(-W/2 / mm, W/2 / mm))

ax[1, 0].legend(loc="upper right", fontsize=7)
ax[1, 0].set(
    xlabel=r"$x$ (mm)",
    ylabel="Intensity",
)
ax[0, 0].set(ylabel="Intensity")
ax[1, 1].set(xlabel="Detector")
ax[0, 1].set_ylabel("Power")
ax[1, 1].set_ylabel("Power")
# ax[0, 1].legend(loc="upper right", fontsize=7)
# ax[1, 1].legend(loc="upper right", fontsize=7)
fig.suptitle(
    f"Hard defect chosen with small pre-train gap: {defect_label_name}, "
    f"depth={defect_depth_mm:.2f} mm, width={defect_width_mm:.2f} mm, "
    f"gap {defect_gap_before_pct:.1f}% -> {defect_gap_after_pct:.1f}%",
    fontsize=8,
)


fig, ax = plt.subplots(2, 1, sharex=True, tight_layout=True, figsize=(4, 3))
ax[0].plot(torch.tensor(loss_history), label="Train loss")
ax[0].plot(torch.tensor(val_loss_history), '--', label="Val loss")
axt = ax[0].twinx()
axt.plot(torch.tensor(train_gap_history) * 100, color="tab:green", label="Train mean gap")
axt.plot(torch.tensor(val_gap_history) * 100, '--', color="tab:red", label="Val mean gap")
ax[0].set(ylabel="Total loss")
axt.set(ylabel="Mean relative gap (%)")
ax[0].legend(loc="upper left")
axt.legend(loc="lower right")


ax[1].plot(torch.tensor(train_power_history), label="Train detector power")
ax[1].plot(torch.tensor(val_power_history), '--', label="Val detector power")
ax[1].axhline(power_floor, color="k", ls="--", lw=1, label="Power floor")
ax[1].axhline(initial_detector_power, color="0.5", ls=":", lw=1, label="Initial intact power")
ax[1].set(xlabel="Epoch", ylabel="Mean detector power")
ax[1].legend()



#%% Test (class separation)
net.load_state_dict(state_init)
net.eval()

with torch.no_grad():
    intensity_test_init, intensity0_test_init, det_test_init, det0_test_init = net(test_ds[:][0].to(device))

net.load_state_dict(state_opt)
net.eval()

with torch.no_grad():
    intensity_test, intensity0_test, det_test, det0_test = net(test_ds[:][0].to(device))

target_test = test_ds[:][1]
test_sep_loss = relative_separation_loss(det_test, det0_test).item()
test_power_loss = power_regularization_loss(det_test, det0_test, power_floor).item()
test_loss = test_sep_loss + power_reg_weight * test_power_loss
test_gap = mean_relative_gap(det_test, det0_test)
test_sat = separation_satisfaction(det_test, det0_test)
test_l2_gap = relative_l2_gap(det_test, det0_test).cpu()
test_power = mean_detector_power(det_test)
print(
    f"Best val loss = {best_val_loss:.4f} at epoch {epoch_opt} | "
    f"test loss = {test_loss:.4f} | test mean gap = {test_gap:.3f} | "
    f"test pass = {100 * test_sat:.1f}% | test power = {test_power:.3e} "
    f"| power floor = {power_floor:.3e}"
)

class_colors = ["tab:blue", "tab:orange"]
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

fig, ax = plt.subplots(figsize=(3,2), tight_layout=True)
for class_id, class_name in enumerate(class_names):
    class_mask = target_test == class_id
    if not torch.any(class_mask):
        continue
    ax.hist(
        test_l2_gap[class_mask].numpy() * 100,
        bins=np.linspace(0,500,51),
        alpha=0.50,
        color=class_colors[class_id],
        label=f"{class_name.capitalize()} ({int(class_mask.sum().item())})",
    )
ax.axvline(relative_margin_pct, color="k", ls="--", lw=1)
ax.set(
    xlabel="Per-sample relative L2 gap (%)",
    xlim=(0,500),
    ylabel="Count",
    title="Detector separation from intact reference",
)
ax.legend(frameon=False, fontsize=7)


viz_batch_size = min(1000, len(det_test))
viz_det0_init = det0_test_init[0]
viz_det0_trained = det0_test[0]
viz_indices_by_class = class_sample_indices(target_test, max_per_class=viz_batch_size)

all_x_parts = [viz_det0_init[0:1], viz_det0_trained[0:1]]
all_y_parts = [viz_det0_init[1:2], viz_det0_trained[1:2]]
for class_indices in viz_indices_by_class.values():
    if len(class_indices) == 0:
        continue
    all_x_parts.extend([det_test_init[class_indices, 0], det_test[class_indices, 0]])
    all_y_parts.extend([det_test_init[class_indices, 1], det_test[class_indices, 1]])

all_x = torch.cat(all_x_parts).cpu().numpy()
all_y = torch.cat(all_y_parts).cpu().numpy()
x_pad = 0.05 * max(all_x.max() - all_x.min(), 1e-6)
y_pad = 0.05 * max(all_y.max() - all_y.min(), 1e-6)
xymin = min(all_x.min(), all_y.min())
xymax = max(all_x.max(), all_y.max())/3

fig, ax = plt.subplots(2, len(class_names), figsize=(5, 5), tight_layout=True, squeeze=False)
for class_id, class_name in enumerate(class_names):
    class_indices = viz_indices_by_class[class_id]
    if len(class_indices) == 0:
        ax[0, class_id].set_axis_off()
        ax[1, class_id].set_axis_off()
        continue

    viz_det_init = det_test_init[class_indices]
    viz_l2_gap_init = relative_l2_gap(viz_det_init, det0_test_init[:1].expand(len(class_indices), -1))
    plot_detector_scatter_2d(
        ax[0, class_id],
        viz_det_init,
        viz_det0_init,
        viz_l2_gap_init,
        relative_margin,
        title=f"Before: {class_name} ({len(class_indices)} samples)",
    )
    ax[0, class_id].set_xlim(0, xymax/3/power_floor_ratio)
    ax[0, class_id].set_ylim(0, xymax/3/power_floor_ratio)
    ax[0, class_id].legend(loc="upper left", fontsize=7)
    ax[0, class_id].set(xticks=np.linspace(0,1,6),yticks=np.linspace(0,1,6))

    viz_det_trained = det_test[class_indices]
    viz_l2_gap_trained = test_l2_gap[class_indices]
    plot_detector_scatter_2d(
        ax[1, class_id],
        viz_det_trained,
        viz_det0_trained,
        viz_l2_gap_trained,
        relative_margin,
        title=f"After: {class_name} ({len(class_indices)} samples)",
    )
    ax[1, class_id].set_xlim(0, xymax/3)
    ax[1, class_id].set_ylim(0, xymax/3)
    ax[1, class_id].legend(loc="upper left", fontsize=7)
    ax[1, class_id].set(xticks=np.linspace(0,1.5,4),yticks=np.linspace(0,1.5,4))




init_l2_gap_all = relative_l2_gap(det_test_init, det0_test_init)

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
for class_id, class_name in enumerate(class_names):
    class_candidates = torch.where(target_test == class_id)[0]
    if len(class_candidates) == 0:
        continue

    close_defect_idx = class_candidates[torch.argmin(init_l2_gap_all[class_candidates])].item()
    print(close_defect_idx)
    field_col = 2 * class_id
    barcode_col = field_col + 1

    defect_depth_mm = test_ds[close_defect_idx][3].item() / mm
    defect_width_mm = test_ds[close_defect_idx][4].item() / mm
    defect_gap_before_pct = 100 * init_l2_gap_all[close_defect_idx].item()
    defect_gap_after_pct = 100 * test_l2_gap[close_defect_idx].item()
    hard_defect_summaries.append(
        f"{class_name}: depth={defect_depth_mm:.2f} mm, width={defect_width_mm:.2f} mm, "
        f"gap {defect_gap_before_pct:.1f}% -> {defect_gap_after_pct:.1f}%"
    )

    ax[0, field_col].plot(x, intensity0_test_init[0].cpu(), color="k", lw=0.9, label="Intact")
    ax[0, field_col].plot(
        x,
        intensity_test_init[close_defect_idx].cpu(),
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

    ax[1, field_col].plot(x, intensity0_test[0].cpu(), color="k", lw=0.9, label="Intact")
    ax[1, field_col].plot(
        x,
        intensity_test[close_defect_idx].cpu(),
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
        axis.set(xlim=(-W/2 / mm, W/2 / mm))

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
