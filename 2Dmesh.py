#%%
import numpy as np 
from scipy import interpolate
import matplotlib.pyplot as plt
import plot_setting
import torch

Hankel = lambda x: torch.special.bessel_j1(x) + 1j * torch.special.bessel_y1(x)


import cv2
# import trimesh as tm
# import trimesh.path.polygons as pg
# import shapely.geometry as sg

mm = 1e3
height = 180 * mm
wvl = 12 * mm
# wvl = 5 * mm
k0 = 2*np.pi/wvl

sim_size = (360*mm, 300*mm, 5*wvl)
# sim_size = (360*mm, 300*mm, 12*wvl)

## Original railroad shape
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
# Moving average smoothing
def permute(vertices, n):
    return torch.cat([vertices[n:], vertices[:n]], dim=0)
vertices = torch.stack([permute(vertices,n) for n in range(-15,16)], dim=2).mean(dim=2)

v_rail = vertices[vertices[:,1]>100*mm] ## Counter clock-wise

## source 
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

# ## source2 
# A_aptr, B_aptr, a_wvg, b_wvg, l_horn= 27.4*mm, 21.9*mm, 9.3*mm, 6.2*mm, 27*mm
# R_E = A_aptr * l_horn / (A_aptr-a_wvg)
# # R_H = B_aptr * l_horn / (B_aptr-b_wvg)
# beta_wvg = k0 * np.sqrt(1-(wvl/(2*a_wvg))**2)
# dist = 160 * mm
# theta = -60 * np.pi/180
# norm_vec_src2 = torch.tensor([[np.sin(theta, dtype=np.float32),-np.cos(theta, dtype=np.float32)]])
# dx_src = A_aptr/50
# s = torch.arange(-A_aptr/2, A_aptr/2, dx_src) + dx_src/2
# v_src_x = -dist*np.sin(theta) + np.cos(theta)*s
# v_src_y = 150*mm + dist*np.cos(theta) + np.sin(theta)* s
# v_src2 = torch.stack([v_src_x,v_src_y], dim=1)
# amp = torch.cos(np.pi*s/A_aptr)
# psi_src2 = amp * torch.exp(0.5j*beta_wvg*(s**2/R_E))

## metasurface plane:
y_ms = 250 * mm
W = 20 * wvl
dx_ms = wvl/10

# W = 50 * wvl
# dx_ms = wvl/5
v_ms_x = torch.arange(-W/2, W/2, dx_ms) + dx_ms/2
v_ms_y = torch.zeros_like(v_ms_x) + y_ms 
v_ms = torch.stack([v_ms_x,v_ms_y], dim=1)



def field(v_rail):
    if v_rail.dim()==2:
        v_rail = v_rail.unsqueeze(dim=0) # batch * points * xy

    center = (v_rail[:,:-1] + v_rail[:,1:])/2
    dx_vec = v_rail[:,1:] - v_rail[:,:-1]
    dx = torch.linalg.norm(dx_vec, dim=2)
    norm_vec = torch.stack([dx_vec[:,:,1], -dx_vec[:,:,0]], dim=2)
    norm_vec = norm_vec / torch.linalg.norm(norm_vec, dim=2, keepdim=True)

    ## source -> rail
    R_vec = center.unsqueeze(dim=2) - v_src.reshape(1,1, *v_src.shape)
    R_scalar = torch.linalg.norm(R_vec, dim=3)
    R_unit = R_vec / R_scalar.unsqueeze(dim=3)
    illum_factor = torch.sum(-R_vec * norm_vec.unsqueeze(dim=2), dim=3)>0
    cos_factor = torch.sum(R_unit * norm_vec_src.reshape(1,1,*norm_vec_src.shape), dim=3)
    psi_rail = torch.sum((1j*k0/4) * Hankel(k0*R_scalar) * illum_factor * cos_factor * psi_src.reshape(1,1,-1) * dx_src, dim=2)

    # ## source2 -> rail
    # R_vec = center.unsqueeze(dim=2) - v_src2.reshape(1,1, *v_src.shape)
    # R_scalar = torch.linalg.norm(R_vec, dim=3)
    # R_unit = R_vec / R_scalar.unsqueeze(dim=3)
    # illum_factor = torch.sum(-R_vec * norm_vec.unsqueeze(dim=2), dim=3)>0
    # cos_factor = torch.sum(R_unit * norm_vec_src2.reshape(1,1,*norm_vec_src2.shape), dim=3)
    # psi_rail2 = torch.sum((1j*k0/4) * Hankel(k0*R_scalar) * illum_factor * cos_factor * psi_src.reshape(1,1,-1) * dx_src, dim=2)
    # psi_rail += psi_rail2

    ## rail -> ms
    R_vec = v_ms.reshape(1,1, *v_ms.shape) - center.unsqueeze(dim=2)
    R_scalar = torch.linalg.norm(R_vec, dim=3)
    R_unit = R_vec / R_scalar.unsqueeze(dim=3)
    illum_factor = torch.sum(R_vec * norm_vec.unsqueeze(dim=2), dim=3)>0
    cos_factor = torch.sum(R_unit * norm_vec.unsqueeze(dim=2), dim=3)
    psi_ms = torch.sum((1j*k0/4) * Hankel(k0*R_scalar) * illum_factor * cos_factor * (-psi_rail * dx).unsqueeze(dim=2), dim=1)
    v_rail = v_rail.squeeze()

    return psi_rail.squeeze(), psi_ms.squeeze()

psi_rail_nodefect, psi_ms_nodefect = field(v_rail)



## Schematic
markersize=10
fig, ax =plt.subplots(1,2, sharex=True, sharey=True, tight_layout=True, figsize=(10,4.8))
ax[0].plot(*vertices.T/mm, '-',  color='gray', label="rail", zorder=-100)
ax[0].plot(*v_src.T/mm, '-',color='gray', label="source aperture", zorder=-100)
ax[0].plot(*v_ms.T/mm, '-',color='gray', label="metasurface", zorder=-100)
ax[0].set(xlim=(-160, 160), ylim=(0, 320), xlabel=r"$x$ (mm)", ylabel=r"$y$ (mm)")
# ax[0].legend(frameon=False)

int_color= ax[0].scatter(*v_src.T/mm, c=torch.abs(psi_src)**2, s=markersize,vmin=0, cmap=plt.cm.inferno)
ax[0].scatter(*v_rail[1:].T/mm, c=torch.abs(psi_rail_nodefect)**2, s=markersize, vmin=0, cmap=plt.cm.inferno)
ax[0].scatter(*v_ms.T/mm, c=torch.abs(psi_ms_nodefect)**2, vmin=0,s=markersize, cmap=plt.cm.inferno)

ax[1].plot(*vertices.T/mm, '-',  color='gray', label="rail", zorder=-100)
ax[1].plot(*v_src.T/mm, '-',color='gray', label="source aperture", zorder=-100)
ax[1].plot(*v_ms.T/mm, '-',color='gray', label="metasurface", zorder=-100)
ax[1].set(xlim=(-160, 160), ylim=(0, 320), xlabel=r"$x$ (mm)")
# ax[0].legend(frameon=False)

phase_color = ax[1].scatter(*v_src.T/mm, c=torch.angle(psi_src), s=markersize, vmin=-np.pi, vmax=np.pi, cmap=plt.cm.twilight)
ax[1].scatter(*v_rail[1:].T/mm, c=torch.angle(psi_rail_nodefect), s=markersize,vmin=-np.pi, vmax=np.pi, cmap=plt.cm.twilight)
ax[1].scatter(*v_ms.T/mm, c=torch.angle(psi_ms_nodefect), s=markersize, vmin=-np.pi, vmax=np.pi, cmap=plt.cm.twilight)

ax[0].set_title("Intensity")
ax[1].set_title("Phase")

plt.colorbar(int_color)
plt.colorbar(phase_color)


torch.save(psi_ms_nodefect, "psi_ms_nodefect_12mm.pt")

#%% Indentation example

def indentation(
        idx_start, idx_end, vertices:torch.Tensor=vertices,
        height:float=0, 
        denttype="ellipse"
    ):
    if idx_start>idx_end:
        idx_start, idx_end = idx_end, idx_start

    N_point = idx_end - idx_start + 1
    p1, p2 = vertices[[idx_start, idx_end]]
    p12 = torch.linalg.norm(p2-p1)
    b = height

    straight_line = p1.reshape(1,-1) + (p2-p1).reshape(1,-1) * torch.linspace(0,1, N_point).reshape(-1,1)

    tan_vec = p2 - p1
    norm_vec = torch.tensor([-tan_vec[1], tan_vec[0]])
    norm_vec = norm_vec / torch.linalg.norm(norm_vec)

    if denttype=="ellipse":
        displacement = b* torch.sqrt(1- torch.linspace(-1,1,N_point)**2)
    elif denttype=="triangle":
        displacement = b * (1 - torch.abs(torch.linspace(-1, 1, N_point)))
    else:
        raise ValueError(f"Unsupported denttype: {denttype}")
    curved_line = straight_line + norm_vec.reshape(1,-1) * displacement.reshape(-1,1)

    vertices_new = vertices.clone()
    vertices_new[idx_start:idx_end+1] = curved_line

    return vertices_new

#%% elliptical chip example
idx_range = np.arange(1080, 1460)

dl_avg = np.linalg.norm(vertices[:-1]-vertices[1:], axis=1).mean()
width_max, width_min = 16*mm, 8*mm

distanced_pairs1 = np.array([
    (i,j)
    for i in idx_range for j in idx_range 
    if i<j and abs(j-i)<=width_max/dl_avg and abs(j-i)>=width_min/dl_avg
])

idx_start, idx_end = distanced_pairs1[np.random.randint(len(distanced_pairs1))]
height = 4*mm 

vertices_d = indentation(idx_start, idx_end, height=height, denttype="ellipse")
v_rail = vertices_d[vertices_d[:,1]>100*mm]
psi_rail, psi_ms = field(v_rail)

markersize= 1
fig, ax = plt.subplots(figsize=(2,2), tight_layout=True)
ax.set(xlim=(-125,125), ylim=(50,300))
ax.plot(*vertices[idx_range[0]]/mm, 's', ms=2)
ax.plot(*vertices[idx_range[-1]]/mm, 's', ms=2)
ax.plot(*vertices.T/mm, color="gray", ls='--', zorder=-100, lw=0.5)
ax.scatter(*v_rail[1:].T/mm, c=torch.abs(psi_rail)**2, s=markersize, vmin=0, cmap=plt.cm.inferno)
ax.scatter(*v_ms.T/mm, c=torch.abs(psi_ms)**2, vmin=0,s=markersize, cmap=plt.cm.inferno)
ax.set(xlabel=r'$x$ (mm)', ylabel=r'$y$ (mm)')




#%% Triangular chip example

dl_avg = np.linalg.norm(vertices[:-1]-vertices[1:], axis=1).mean()
width_max, width_min = 4*mm, 2*mm

distanced_pairs2 = np.array([
    (i,j)
    for i in idx_range for j in idx_range 
    if i<j and abs(j-i)<=width_max/dl_avg and abs(j-i)>=width_min/dl_avg
])

idx_start, idx_end = distanced_pairs2[np.random.randint(len(distanced_pairs2))]
height = 10*mm 

vertices_tri = indentation(idx_start, idx_end, height=height, denttype="triangle")
v_rail_tri = vertices_tri[vertices_tri[:,1]>100*mm]
psi_rail_tri, psi_ms_tri = field(v_rail_tri)

markersize = 1
fig, ax = plt.subplots(figsize=(2,2), tight_layout=True)
ax.set(xlim=(-125,125), ylim=(50,300))
ax.plot(*vertices[idx_range[0]]/mm, 's', ms=2)
ax.plot(*vertices[idx_range[-1]]/mm, 's', ms=2)
ax.plot(*vertices.T/mm, color="gray", ls='--', zorder=-100, lw=0.5)
ax.plot(*vertices_tri.T/mm, color="black", zorder=-90, lw=0.6)
ax.scatter(*v_rail_tri[1:].T/mm, c=torch.abs(psi_rail_tri)**2, s=markersize, vmin=0, cmap=plt.cm.inferno)
ax.scatter(*v_ms.T/mm, c=torch.abs(psi_ms_tri)**2, vmin=0, s=markersize, cmap=plt.cm.inferno)
ax.set(xlabel=r'$x$ (mm)', ylabel=r'$y$ (mm)')


#%% Data generation
# from tqdm import tqdm

# psi_rail_list = []
# psi_ms_list = []
# defect_range_list = []
# defect_height_list = []

# for ii in tqdm(range(200000)):
#     np.random.seed(ii)
#     if ii%2==0:
#         idx_start, idx_end = distanced_pairs1[np.random.randint(len(distanced_pairs1))]
#         height = 3*mm * (1+np.random.rand())
#         vertices_d = indentation(idx_start, idx_end, height=height, denttype="ellipse")
#     else:
#         idx_start, idx_end = distanced_pairs2[np.random.randint(len(distanced_pairs2))]
#         height = 9*mm * (1+np.random.rand())
#         vertices_d = indentation(idx_start, idx_end, height=height, denttype="triangle")
#     v_rail = vertices_d[vertices_d[:,1]>100*mm]
#     psi_rail, psi_ms = field(v_rail)

#     defect_range_list.append((idx_start, idx_end))
#     defect_height_list.append(height)
#     psi_rail_list.append(psi_rail)
#     psi_ms_list.append(psi_ms)


#%% Data generation : parallel

from tqdm import tqdm
psi_rail_list = []
psi_ms_list = []
defect_range_list = []
defect_height_list = []

v_rail_batch = []

for ii in tqdm(range(100000)):
    np.random.seed(ii)
    # idx_start, idx_end = distanced_pairs[np.random.randint(len(distanced_pairs))]
    # idx_start, idx_end = idx_start.item(), idx_end.item()
    # height = 10*mm * np.random.rand()
    # vertices_d = indentation(idx_start, idx_end, height=height)
    if ii%2==0:
        idx_start, idx_end = distanced_pairs1[np.random.randint(len(distanced_pairs1))]
        height = 2.5*mm * (np.random.rand())
        vertices_d = indentation(idx_start, idx_end, height=height, denttype="ellipse")
    else:
        idx_start, idx_end = distanced_pairs2[np.random.randint(len(distanced_pairs2))]
        height = 10*mm * (np.random.rand())
        vertices_d = indentation(idx_start, idx_end, height=height, denttype="triangle")

    v_rail_batch.append(vertices_d[vertices_d[:,1]>100*mm])
    defect_range_list.append((idx_start, idx_end))
    defect_height_list.append(height)

v_rail_batch = torch.stack(v_rail_batch, dim=0)

for i in tqdm(range(100)):
    v_mini = v_rail_batch[1000*i:1000*(i+1)]
    psi_rail, psi_ms = field(v_mini)
    psi_rail_list.append(psi_rail)
    psi_ms_list.append(psi_ms)


torch.save(torch.tensor(defect_range_list), "range_dist_12mm.pt")
torch.save(torch.tensor(defect_height_list), "depth_dist_12mm.pt")
torch.save(torch.cat(psi_ms_list), "psi_ms_dist_12mm.pt")
torch.save(torch.cat(psi_rail_list), "psi_rail_dist_12mm.pt")
torch.save(v_rail_batch, "v_rail_dist_12mm.pt")

#%%

close_defect_idx = 4505
idx = 90000+close_defect_idx

defect_range_list = torch.load("range_dist_12mm.pt")
defect_height_list = torch.load("depth_dist_12mm.pt")
psi_ms_list = torch.load("psi_ms_dist_12mm.pt")
psi_rail_list = torch.load("psi_rail_dist_12mm.pt")
v_rail_list = torch.load("v_rail_dist_12mm.pt")

test_width = (defect_range_list[idx][1]-defect_range_list[idx][0])*290.0129
test_height = defect_height_list[idx]

v_rail_test = v_rail_batch[idx]
psi_ms_test = psi_ms_list[idx]
psi_rail_test = psi_rail_list[idx]

print(f"width={test_width/1000:.2f} mm, height={test_height/1000:.2f} mm")



fig, ax = plt.subplots(figsize=(2,2), tight_layout=True)
ax.set(xlim=(-125,125), ylim=(50,300))
ax.plot(*vertices[idx_range[0]]/mm, 's', ms=2)
ax.plot(*vertices[idx_range[-1]]/mm, 's', ms=2)
ax.plot(*vertices.T/mm, color="gray", ls='--', zorder=-100, lw=0.5)
# ax.plot(*v_rail_test.T/mm, color="black", zorder=100, lw=0.6)
ax.scatter(*v_rail_test[1:].T/mm, c=torch.abs(psi_rail_test)**2, s=0.5, vmin=0, cmap=plt.cm.inferno)
ax.scatter(*v_ms.T/mm, c=torch.abs(psi_ms_test)**2, vmin=0, s=0.5, cmap=plt.cm.inferno)
ax.set(xlabel=r'$x$ (mm)', ylabel=r'$y$ (mm)')


fig, ax = plt.subplots(figsize=(2,2), tight_layout=True)
ax.set(xlim=(-125,125), ylim=(50,300))
ax.plot(*vertices[idx_range[0]]/mm, 's', ms=2)
ax.plot(*vertices[idx_range[-1]]/mm, 's', ms=2)
ax.plot(*vertices.T/mm, color="gray", ls='--', zorder=-100, lw=0.5)
ax.scatter(*v_rail[1:].T/mm, c=torch.abs(psi_rail_nodefect)**2, s=0.5, vmin=0, cmap=plt.cm.inferno)
ax.scatter(*v_ms.T/mm, c=torch.abs(psi_ms_nodefect)**2, vmin=0, s=0.5, cmap=plt.cm.inferno)
ax.set(xlabel=r'$x$ (mm)', ylabel=r'$y$ (mm)')

