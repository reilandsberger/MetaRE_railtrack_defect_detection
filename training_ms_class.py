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


range_defect = torch.load("range_2class.pt")
depth_defect = torch.load("depth_2class.pt")
width_defect = torch.abs(range_defect[:,0]-range_defect[:,1])*290.0129
defect_size = width_defect * depth_defect

normal_threshold = 2.5e7

nominal_defect_label = torch.arange(len(depth_defect), dtype=torch.long) % 2  # 0=ellipse, 1=triangle
defect_label = nominal_defect_label + 1  # 0=normal, 1=ellipse, 2=triangle
defect_label[defect_size < normal_threshold] = 0
class_names = ["normal", "ellipse", "triangle"]

#%% Defect size distribution
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

psi_ms = torch.load("psi_ms_2class.pt").to(torch.complex64)
psi_ms = psi_ms / torch.sqrt(torch.mean(psi_ms.abs()**2))
psi_ms0 = torch.load("psi_ms_nodefect.pt").to(torch.complex64)
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
        std_phase: float=20*np.pi/180,
        dummy:int=0,
        sym:bool=False,
        ):
        super(Metalayer, self).__init__()
        self.sym = sym
        self.N = resol
        self.dummy=dummy
        # self.phase_init = phase_init +torch.normal(0, std_phase*torch.ones(self.N))
        torch.manual_seed(dummy)
        self.phase_init = phase_init + 2*np.pi* torch.rand(self.N)
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
            det_center:float=np.array([0.0, -W/4, W/4]),
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
net = ONN(N_layer=3, distance=50*mm, sym=False).to(device)
N_epoch = 200
lr = 1e-3


optimizer = optim.Adam(net.parameters(), lr=lr)
from copy import deepcopy

train_counts = torch.bincount(train_ds[:][1], minlength=len(class_names)).float()
class_weight = train_counts.sum() / (len(class_names) * train_counts.clamp_min(1))
criterion = nn.CrossEntropyLoss(weight=class_weight.to(device))


def detector_logits(detector_power):
    return torch.log(detector_power + 1e-12)

def confusion_matrix(target, pred, n_class):
    cm = torch.zeros((n_class, n_class), dtype=torch.int64)
    for target_i, pred_i in zip(target, pred):
        cm[target_i, pred_i] += 1
    return cm


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
    # Show detector power in spatial order: ellipse (left), normal (center), triangle (right).
    display_order = [1, 0, 2]
    labels = ["E", "N", "T"]
    values = detector_power.detach().cpu()[display_order].numpy()
    colors = ["0.75"] * len(values)
    if highlight_idx is not None:
        colors[display_order.index(highlight_idx)] = "tab:orange"
    axis.bar(np.arange(len(values)), values, color=colors, width=0.7)
    axis.set_xticks(np.arange(len(values)))
    axis.set_xticklabels(labels, fontsize=7)
    axis.set_ylim(0, max(values.max() * 1.15, 1e-6))
    axis.tick_params(axis="y", labelsize=6, length=2)
    for spine in ["top", "right"]:
        axis.spines[spine].set_visible(False)


#%% Defect-type classification
loss_history = []
train_acc_history = []
val_loss_history = []
val_acc_history = []

best_val_acc = 0.0
state_opt = deepcopy(net.state_dict())

for epoch in tqdm(range(N_epoch)):
    net.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for field, label, _, _, _ in train_loader:
        optimizer.zero_grad(set_to_none=True)

        _, _, det, _ = net(field.to(device))
        logits = detector_logits(det)
        label = label.to(device)

        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * len(field)
        running_correct += (torch.argmax(logits, dim=1) == label).sum().item()
        running_total += len(field)

    train_loss = running_loss / running_total
    train_acc = running_correct / running_total
    loss_history.append(train_loss)
    train_acc_history.append(train_acc)

    net.eval()
    with torch.no_grad():
        _, _, det_val, _ = net(val_ds[:][0].to(device))
        logits_val = detector_logits(det_val)
        label_val = val_ds[:][1].to(device)
        val_loss = criterion(logits_val, label_val).item()
        val_pred = torch.argmax(logits_val, dim=1)
        val_acc = torch.mean((val_pred == label_val).float()).item()

    val_loss_history.append(val_loss)
    val_acc_history.append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epoch_opt = epoch
        state_opt = deepcopy(net.state_dict())
        print(
            f"Train loss = {train_loss:.4f} | train acc = {train_acc:.3f} "
            f"| val loss = {val_loss:.4f} | val acc = {val_acc:.3f}"
        )


#%% Test
net.load_state_dict(state_opt)
net.eval()

with torch.no_grad():
    intensity_test, intensity0_test, det_test, det0_test = net(test_ds[:][0].to(device))
    logits_test = detector_logits(det_test)

target_test = test_ds[:][1]
pred_test = torch.argmax(logits_test, dim=1).cpu()
test_acc = torch.mean((pred_test == target_test).float()).item()
cm = confusion_matrix(target_test, pred_test, len(class_names))
cm_percent = 100 * cm.float() / cm.sum(dim=1, keepdim=True).clamp_min(1)
print(f"Best val acc = {best_val_acc:.3f} at epoch {epoch_opt} | test acc = {test_acc:.3f}")
print("Confusion matrix (rows=true, cols=pred):")
print(cm)


fig, ax = plt.subplots(figsize=(3.2, 3), tight_layout=True)
im = ax.imshow(cm_percent, cmap=plt.cm.Blues, vmin=0, vmax=100)
ax.set(
    xlabel="Predicted class",
    ylabel="True class",
    xticks=np.arange(len(class_names)),
    yticks=np.arange(len(class_names)),
    xticklabels=class_names,
    yticklabels=class_names,
)
for row in range(len(class_names)):
    for col in range(len(class_names)):
        ax.text(col, row, f"{cm_percent[row, col]:.1f}%", ha="center", va="center", color="black", fontsize=7)
cbar = plt.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
cbar.ax.set_ylabel("Row percentage")


x = np.arange(-W/2 + W/400, W/2, W/200) / mm
fig, ax = plt.subplots(len(net.metalayers), 1, figsize=(3, 3), sharex=True, tight_layout=True)
if len(net.metalayers) == 1:
    ax = [ax]
for ii, ml in enumerate(net.metalayers):
    ax[ii].plot(x, ml.phase.cpu().detach(), lw=0.75)
ax[-1].set(xlim=(-W/2 / mm, W/2 / mm), xlabel=r"$x$ (mm)")
ax[0].set_title("Metagrating phase", fontsize=7)


sample_indices = representative_indices(target_test)
fig, ax = plt.subplots(
    len(sample_indices) + 1,
    2,
    figsize=(4.8, 3.8),
    sharex="col",
    tight_layout=True,
    squeeze=False,
    gridspec_kw={"width_ratios": [3.5, 1.2]},
)
ax[0, 0].plot(x, intensity0_test[0].cpu(), color="k", lw=0.75)
ax[0, 0].set_title("No defect reference", fontsize=7)
ax[0, 1].set_title("Detector power", fontsize=7)
plot_detector_panel(ax[0, 1], det0_test[0])

for ii, idx in enumerate(sample_indices, start=1):
    ax[ii, 0].plot(x, intensity_test[idx].cpu(), lw=0.75)
    label_name = class_names[target_test[idx].item()]
    depth_mm = test_ds[idx][3].item() / mm
    width_mm = test_ds[idx][4].item() / mm
    ax[ii, 0].set_title(
        f"{label_name} | depth={depth_mm:.2f} mm | width={width_mm:.2f} mm",
        fontsize=7,
    )
    plot_detector_panel(ax[ii, 1], det_test[idx], highlight_idx=target_test[idx].item())

for axis in ax[:, 0]:
    for xc in net.det_center:
        axis.axvline(x=(xc + net.det_width / 2) / mm, color="gray", ls="--", lw=0.75)
        axis.axvline(x=(xc - net.det_width / 2) / mm, color="gray", ls="--", lw=0.75)
ax[-1, 0].set(xlabel=r"$x$ (mm)", xlim=(-W/2 / mm, W/2 / mm))
ax[-1, 1].set(xlabel="Detector")


fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(torch.tensor(loss_history), label="Train loss")
ax.plot(torch.tensor(val_loss_history), label="Val loss")
axt = ax.twinx()
axt.plot(torch.tensor(train_acc_history) * 100, color="tab:green", label="Train acc")
axt.plot(torch.tensor(val_acc_history) * 100, color="tab:red", label="Val acc")
ax.set(xlabel="Epoch", ylabel="Cross-entropy loss")
axt.set(ylabel="Accuracy (%)")
ax.legend(loc="upper left")
axt.legend(loc="lower right")
