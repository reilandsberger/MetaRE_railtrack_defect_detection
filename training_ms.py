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


mm = 1e0
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


# Large input datasets live in the external RailDefect folder (override with RAILDEFECT_DATA_DIR).
from raildefect_paths import RAILDEFECT_DIR

range_defect = torch.load(RAILDEFECT_DIR / "range.pt")
depth_defect = torch.load(RAILDEFECT_DIR / "depth.pt")
psi_ms = torch.load(RAILDEFECT_DIR / "psi_ms.pt").to(torch.complex64)
psi_ms = psi_ms / torch.sqrt(torch.mean(psi_ms.abs()**2))
psi_ms0 = torch.load(RAILDEFECT_DIR / "psi_ms_nodefect.pt").to(torch.complex64)
psi_ms0 = psi_ms0 / torch.sqrt(torch.mean(psi_ms0.abs()**2))
# psi_ms = psi_ms *torch.sqrt(0.5/torch.mean(torch.abs(psi_ms)**2)) # normalize to have the average intensity of 0.5
# psi = psi_ms[:20]


class FieldDataset(Dataset):
    def __init__(self, 
                field=psi_ms, 
                defect=range_defect, 
                depth=depth_defect, 
                def_min=1080, def_max=1460
        ):
        self.field = field
        self.raw_range = defect
        self.norm_range = (def_max-defect)/(def_max-def_min)  # normalize within 950-1465 in reverse direction
        self.depth = depth
        self.resol = field.shape[1]
        self.x = torch.arange(0, 1, 1/self.resol) + 0.5/self.resol
        self.location_indicator = 0.0+ (self.x.reshape(1,-1)<self.norm_range[:,0].reshape(-1,1)) * (self.x.reshape(1,-1)>self.norm_range[:,1].reshape(-1,1))

    def __getitem__(self, index):
        # return self.field[index], self.location_indicator[index], self.depth[index]
        return self.field[index], self.raw_range[index], self.depth[index]
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
            N_layer:int=5, 
            distance:float=10*mm, 
            sym:bool=False,
            det_center:float=np.linspace(-W/4, W/4, 5),
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



train_ds = FieldDataset(field=psi_ms[:80000], defect=range_defect[:80000], depth=depth_defect[:80000])
val_ds = FieldDataset(field=psi_ms[80000:90000], defect=range_defect[80000:90000], depth=depth_defect[80000:90000])
test_ds = FieldDataset(field=psi_ms[90000:100000], defect=range_defect[90000:100000], depth=depth_defect[90000:100000])


# batch_size = 2000
# train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_size)
# net = ONN(N_layer=1, distance=100*mm, sym=False).to(device)
# N_epoch = 200
# lr = 2e-4

batch_size = 2000
train_loader = DataLoader(train_ds, shuffle=True, batch_size=batch_size)
net = ONN(N_layer=3, distance=50*mm, sym=False).to(device)
N_epoch = 200
lr = 1e-4


optimizer = optim.Adam(net.parameters(), lr=lr)
bce = nn.BCELoss(reduction="mean")

cos = nn.CosineSimilarity(dim=1)

mse = nn.MSELoss()
l1 = nn.L1Loss()

cossim = nn.CosineSimilarity()

#%% Height detection
from copy import deepcopy
loss_history = []
power_history = []
corr_history = []
val_history = []

max_corr_val = 0.0

for epoch in tqdm(range(N_epoch)):
    net.train()
    for field, defrange, defheight in train_loader:
        intensity, intensity0, det, det0 = net(field.to(device))
        
        focus = det.mean()

        cos_dist = 1-cos(det, det0)
        dev_norm = torch.mean(cos_dist*(defheight.to(device)/1e4))
        corr = torch.corrcoef(torch.stack([cos_dist, defheight.to(device)]))[0,1]

        # loss = -focus  - 100*corr
        loss = -focus -corr
        loss.backward()
        optimizer.step()

    power_history.append(focus.item())
    corr_history.append(corr.item())

    net.eval()
    with torch.no_grad():
        _, _, det_val, det0_val = net(val_ds[:][0].to(device))
    cos_dist_val = 1 - cos(det_val, det0_val)
    corr_val = torch.corrcoef(torch.stack([cos_dist_val, val_ds[:][2].to(device)]))[0,1]
    val_history.append(corr_val.item())

    # print(f"focus={focus.item()} val={corr_val.item()}")

    if corr_val.item() > max_corr_val:
        max_corr_val = corr_val.item()
        epoch_opt = epoch
        state_opt = deepcopy(net.state_dict())
        print(f"Train loss = {loss.item()} | val score = {max_corr_val}")


#%% Test

# net.load_state_dict(state_opt)

with torch.no_grad():
    int, int0, det, det0 = net(test_ds[:][0].to(device))

depth_sort_idx = np.argsort(test_ds[:][2])
indices = depth_sort_idx[[1000, 3001,  5000, 9000]]


cos_dist = 1 - cos(det, det0)
corr = torch.corrcoef(torch.stack([cos_dist, test_ds[:][2].to(device)]))[0,1]

test_def_range = test_ds[:][1]
test_width = test_def_range[:,1]-test_def_range[:,0]


fig, ax = plt.subplots(figsize=(3,3), tight_layout=True)
sc = ax.scatter(test_ds[:][2]/1e3, cos_dist.cpu().detach(), c='gray', s=1, alpha=0.5, cmap=plt.cm.berlin)
ax.set(xlabel=r"Depth of dent (mm)", ylabel=r"Cosine distance", xlim=(0,10), ylim=(0,0.5))
ax.set_title(f"Test correlation coefficient = {round(corr.item(),3)}", fontsize=7)
[ax.plot(test_ds[ind][2]/1e3, cos_dist[ind].cpu().detach(), c+'o', ms=5) for ind, c in zip(indices, "rgbc")]
# cax = ax.inset_axes([1.02, 0, 0.03, 1])
# cbar = plt.colorbar(sc, cax=cax)
# cbar.ax.set_ylabel("Width (arb.u.)")



x = np.arange(-W/2 + W/400, W/2, W/200)
fig, ax =plt.subplots(len(net.metalayers), 1, figsize=(3,3), sharex=True, tight_layout=True)
if len(net.metalayers)==1:
    ax = [ax]
for ii, ml in enumerate(net.metalayers):
    ax[ii].plot(x, ml.phase.cpu().detach(), lw=0.75)
ax[-1].set(xlim=(-W/2,W/2), xlabel=r"$x$ (mm)")
ax[0].set_title("Metagrating phase", fontsize=7)



fig, ax = plt.subplots(5,1, figsize=(3,4), sharex=True, tight_layout=True)
ax[0].plot(x, int0[0].cpu(), color='k', lw=0.75)
ax[1].plot(x, int[indices[0]].cpu(), color='r', lw=0.75)
ax[2].plot(x, int[indices[1]].cpu(), color='g', lw=0.75)
ax[3].plot(x, int[indices[2]].cpu(), color='b', lw=0.75)
ax[4].plot(x, int[indices[3]].cpu(), color='c', lw=0.75)
ax[-1].set(xlabel=r"$x$ (mm)", xlim=(-W/2,W/2))
ax[0].set_title("No defect", fontsize=7)
ax[1].set_title(f"Depth = { round(test_ds[indices[0]][2].item()/1e3, 4)} mm", fontsize=7)
ax[2].set_title(f"Depth = { round(test_ds[indices[1]][2].item()/1e3, 4)} mm", fontsize=7)
ax[3].set_title(f"Depth = { round(test_ds[indices[2]][2].item()/1e3, 4)} mm", fontsize=7)
ax[4].set_title(f"Depth = { round(test_ds[indices[3]][2].item()/1e3, 4)} mm", fontsize=7)

[ax[ii].axvline(x=xc+net.det_width/2, color="gray", ls='--', lw=0.75) for xc in net.det_center for ii in range(5)]
[ax[ii].axvline(x=xc-net.det_width/2, color="gray", ls='--', lw=0.75) for xc in net.det_center for ii in range(5)]




#%% range detection
from copy import deepcopy
loss_history = []
power_history = []
val_history = []

det_region = torch.abs(v_ms_x)<W/2

max_val_score = 0.0

for epoch in tqdm(range(N_epoch)):
    net.train()
    for field, target in train_loader:
        intensity, dint = net(field.to(device))
        avg_power = intensity.mean()
        loss = -(dint * (2*target.to(device)-1)).sum(dim=1).mean()
        loss.backward()
        optimizer.step()

    net.eval()
    with torch.no_grad():
        field_val, target_val = val_ds[:]
        _, dint_val = net(field_val.to(device))
        dint_val, target_val = dint_val[:,det_region], target_val[:,det_region]
        max_idx = torch.argmax(dint_val, dim=1)
        max_inside_range = target_val.to(device)[torch.arange(len(val_ds)), max_idx]==1
        val_score = torch.mean(0.0+max_inside_range)

    loss_history.append(loss.item())
    val_history.append(val_score.item())
    # power_history.append(avg_power.data)
    
    if val_score.item() > max_val_score:
        max_val_score = val_score.item()
        epoch_opt = epoch
        state_opt = deepcopy(net.state_dict())
        print(f"Train loss = {loss.item()} | val score = {max_val_score}")


#%% test

net.load_state_dict(state_opt)

with torch.no_grad():
    field_test, target_test = test_ds[:]
    intensity_test, dint_test = net(field_test.to(device))
    intensity_test, dint_test, target_test = intensity_test[:,det_region], dint_test[:,det_region], target_test[:,det_region]
    max_idx = torch.argmax(dint_test, dim=1)
    max_inside_range = target_test.to(device)[torch.arange(len(test_ds)), max_idx]==1
    test_score = torch.mean(0.0+max_inside_range)

print(f"Max val score={max_val_score}, test score={test_score.item()}")


fig, ax = plt.subplots(2,4, figsize=(10,5), sharex=True)

for ii, idx in enumerate([1, 7, 8, 10]):
    ax[0,ii].plot(v_ms_x[det_region]/mm, intensity_test[idx].to("cpu"))
    ax[0,ii].plot(v_ms_x[det_region]/mm, (intensity_test[idx]-dint_test[idx]).to("cpu"), 'gray', lw=0.75)
    ax[1,ii].plot(v_ms_x[det_region]/mm, (dint_test[idx]/torch.abs(dint_test[idx].max())).to("cpu"))
    ax[1,ii].plot(v_ms_x[det_region]/mm, target_test[idx].to("cpu").detach())
    if ii>0:
        ax[0,ii].set(yticklabels=[])
        ax[1,ii].set(yticklabels=[])

    # if ii<3:
    #     ax[0,ii].set_title("Success")
    # else:
    #     ax[0,ii].set_title("Fail")

ax[0,0].set(ylabel="Intensity")
ax[1,0].set(xlim=(-W/2,W/2), xlabel=r"x (mm)", ylabel=r"Intensity difference")


fig, ax = plt.subplots(len(net.metalayers), figsize=(6,4), sharex=True, tight_layout=True)
if len(net.metalayers)==1:
    ax = [ax]
[ax[len(net.metalayers)-1-ii].plot(v_ms_x/mm, np.unwrap(meta.phase.to("cpu").detach())) for ii, meta in enumerate(net.metalayers)]
ax[0].set(xlim=(-W/2,W/2), xticks=np.linspace(-W/2,W/2,13))
ax[-1].set(xlabel=r"x (mm)")
ax[0].set_title("Metagrating")


fig, ax = plt.subplots(figsize=(4,3))

ax.plot(torch.tensor(loss_history))
axt = ax.twinx()
axt.plot(torch.tensor(val_history)*100, color="r")
ax.set(xlabel=r"Epoch", ylabel=r"Loss")
axt.set(ylabel=r"Validation score (%)")

