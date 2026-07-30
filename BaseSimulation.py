#%%
import numpy as np 
from scipy import interpolate
import matplotlib.pyplot as plt
import cv2

import tidy3d as td
import tidy3d.web as web

mm = 1e3
height = 180 * mm
wvl = 12 * mm
k0 = 2*np.pi/wvl 

sim_size = (360*mm, 300*mm, 5*wvl)

freq0 = td.C_0/wvl
fwidth = freq0/2
source_time = td.GaussianPulse(freq0=freq0,fwidth=fwidth)

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
cnt = np.array(cnt_sym)[::3]


rail = td.Structure(
    geometry=td.PolySlab(axis=2, slab_bounds=(-10*wvl, 10*wvl), vertices=cnt),
    medium=td.PECMedium()
)


src_center = (-200*mm, 180*mm, 0)

point_src = td.UniformCurrentSource(
    center=src_center, size=(0,0, td.inf if sim_size[2]==0 else 0),
    polarization="Ez",
    source_time=source_time
)


### antenna parameters
thickness = 1 * mm
A_aptr, B_aptr, a_wvg, b_wvg, l_horn= 27.4*mm, 21.9*mm, 9.3*mm, 6.2*mm, 27*mm
R_E = A_aptr * l_horn / (A_aptr-a_wvg)
R_H = B_aptr * l_horn / (B_aptr-b_wvg)
beta_wvg = k0 * np.sqrt(1-(wvl/(2*a_wvg))**2)

dist = 130 * mm
theta = 50 * np.pi/180


ant_wvg_geo = td.ClipOperation(
    geometry_a=td.Box(center=(0, 0, 0), size=(a_wvg+thickness/2, l_horn*2, b_wvg+thickness/2)),
    geometry_b=td.Box(center=(0, 0, 0), size=(a_wvg-thickness/2, l_horn*2, b_wvg-thickness/2)),
    operation="difference"
)
# vertices = np.array([
#     (-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.5, -0.5)
# ])

# horn_geo = td.ClipOperation(
#     geometry_a=td.PolySlab(
#         axis=1, slab_bounds=(-l_horn,0),
#         vertices=vertices*np.array([a_wvg+thickness/2, b_wvg+thickness/2]).reshape(1,-1),
#         reference_plane="top",
#         sidewall_angle=np.arctan((A_aptr-a_wvg)/2/l_horn)
#     ),
#     geometry_b=td.PolySlab(
#         axis=1, slab_bounds=(-l_horn,0),
#         vertices=vertices*np.array([a_wvg-thickness/2, b_wvg-thickness/2]).reshape(1,-1),
#         reference_plane="top",
#         sidewall_angle=np.arctan((A_aptr-a_wvg)/2/l_horn)
#     ),
#     operation="difference"
# )
# ant_geo = td.ClipOperation(
#     geometry_a=ant_wvg_geo, geometry_b=horn_geo, operation="union"
# )
antenna = td.Structure(
    geometry=ant_wvg_geo.rotated(angle=theta, axis=2).translated(
        x=-(dist+l_horn)*np.sin(theta),  
        y=150*mm + (dist+l_horn)*np.cos(theta),
        z=0
    ), 
    medium=td.PECMedium()
)


det_pos = np.linspace(-60*mm, 60*mm, 5)
det_size = (9.1*mm, 5.6*mm)

det_wvgs = [td.Structure(
    geometry=td.ClipOperation(
        geometry_a=td.Box(center=(xc, 300*mm, 0), size=(det_size[0]+thickness/2, 100*mm, det_size[1]+thickness/2)),
        geometry_b=td.Box(center=(xc, 300*mm, 0), size=(det_size[0]-thickness/2, 100*mm, det_size[1]-thickness/2)),
        operation="difference"
    ),
    medium=td.PECMedium()
) for xc in det_pos]


snap = tuple([(x.item(),y.item(),0) for x, y in cnt[cnt[:,1]>100*mm] ])

mnt_surf = [td.FieldMonitor(
    center=(x, y,0), size=(0,0,0),
    name=f"surf {ii}", freqs=[freq0]
) for ii, (x,y, z) in enumerate(snap)]

mnt_plane = td.FieldMonitor(
    center=(0,0,0), size=(td.inf, td.inf, 0),
    name="field_plane", freqs=[freq0]
)

boundary_spec = td.BoundarySpec(
    x=td.Boundary.pml(), 
    y=td.Boundary.pml(), 
    z=td.Boundary.periodic() if sim_size[2]==0 else td.Boundary.pml()
)
grid_spec = td.GridSpec(
    grid_x=td.AutoGrid(min_steps_per_wvl=50, dl_min=0.1*mm),
    grid_y=td.AutoGrid(min_steps_per_wvl=50, dl_min=0.1*mm),
    grid_z=td.AutoGrid(min_steps_per_wvl=50, dl_min=0.1*mm),
    snapping_points=snap,
    wavelength=wvl,
)

mnt_base = [mnt_plane, *mnt_surf]
sim_base = td.Simulation(
    center=(0, sim_size[1]/2, 0),
    size=sim_size,
    structures=[antenna, rail, *det_wvgs],
    sources=[],
    monitors=mnt_base,
    boundary_spec=boundary_spec,
    grid_spec=grid_spec,
    run_time=td.RunTimeSpec(quality_factor=3),
    plot_length_units="mm",
    symmetry=(0,0,-1)
)

fig, ax= plt.subplots()
sim_base.plot_grid(y=275*mm, ax=ax)
sim_base.plot(y=275*mm, ax=ax)
ax.set(xlim=(-det_size[0], det_size[0]), ylim=(-det_size[1],det_size[1]))


fig, ax= plt.subplots()
sim_base.plot(z=0, ax=ax)


#%%  fwd source and monitor 
from tidy3d.plugins.mode import ModeSolver

## source
num_modes = 1
mode_spec=td.ModeSpec(
    num_modes=num_modes, 
    target_neff=8.06568984e-01,
    angle_theta=np.pi/2-theta,
    angle_phi=np.pi,
)

mode_solver = ModeSolver(
    simulation=sim_base,
    plane=td.Box(
        center=(-(dist+l_horn)*np.sin(theta), (dist+l_horn)*np.cos(theta)+150*mm,0),
        size=(0, a_wvg/np.sin(theta)*1, b_wvg*1),
    ),
    mode_spec=mode_spec,
    freqs=[freq0],
)
mode_data = mode_solver.solve()

if num_modes==1:
    fig, ax =plt.subplots(tight_layout=True, figsize=(3,60))
    ax.matshow(np.abs(mode_data.Ez.squeeze())**2)
else:
    fig, ax =plt.subplots(num_modes,1, sharex=True,sharey=True, tight_layout=True, figsize=(3,3*num_modes))
    for i in range(num_modes):
        ax[i].matshow(np.abs(mode_data.Ez.squeeze()[:,:,i])**2)


src_wvg = mode_solver.to_source(
    source_time=source_time,
    direction="+",
    mode_index=1
)

## monitor
num_modes = 1
mode_spec=td.ModeSpec(
    num_modes=num_modes, 
    target_neff=0.8047929
)
mode_solver = ModeSolver(
    simulation=sim_base,
    plane=td.Box(
        center=(0, 275*mm, 0),
        size=(det_size[0], 0, det_size[1]),
    ),
    mode_spec=mode_spec,
    freqs=[freq0],
)
mode_data = mode_solver.solve()


if num_modes>1:
    fig, ax =plt.subplots(1,num_modes, sharex=True,sharey=True, tight_layout=True, figsize=(6,2))
    for i in range(num_modes):
        ax[i].matshow(np.abs(mode_data.Ez.squeeze()[:,:,i])**2)
else:

    fig, ax =plt.subplots(tight_layout=True, figsize=(3,60))
    ax.matshow(np.abs(mode_data.Ez.squeeze())**2)


mnt_wvgs = [mode_solver.to_monitor(
freqs=[freq0], name=f"mode mnt {ii}",
).updated_copy(center=(xc, 275*mm, 0)) for ii, xc in enumerate(det_pos)]


sim_fwd = sim_base.updated_copy(
    sources=[src_wvg],
    monitors=mnt_base+mnt_wvgs
)

sim_fwd.plot(z=0)


#%% fwd run

fwd_data = web.run(
    simulation=sim_fwd,
    task_name="fwd",
    folder_name="Railroad defect"
)

## surface field
x, y = [], []
field_fwd = []
for ii in range(284):
    field = fwd_data[f"surf {ii}"]
    Ez, Hx, Hy = field.Ez.squeeze(), field.Hx.squeeze(), field.Hy.squeeze()
    x.append(Ez.x)
    y.append(Ez.y)
    field_fwd.append([Ez, Hx, Hy])
x = np.array(x)
y = np.array(y)
field_fwd = np.array(field_fwd)

#%% objective function & adjoint sims

sparams = []
sims_adj = dict()
for ii, xc in enumerate(det_pos):
    sparam = fwd_data[f"mode mnt {ii}"].amps.sel(direction="+").squeeze().item()
    coeff = np.array(sparam - 0).conj()/1j
    sparams.append(sparam)

    adj_src = mode_solver.to_source(
        direction="-", mode_index=0, num_freqs=5,
        source_time=td.GaussianPulse(
            freq0=freq0,fwidth=fwidth,
            amplitude=np.abs(coeff), phase=np.angle(coeff),
        )
    )
    adj_src = adj_src.updated_copy(center=(xc, 275*mm, 0))

    sims_adj[f"adj {ii}"] = sim_base.updated_copy(sources=[adj_src])

sparams = np.array(sparams)

adj_data = web.run_async(
    simulations=sims_adj,
    folder_name="Railroad defect"
)

field_adj = []
for jj in range(5):
    data = adj_data[f"adj {jj}"]
    for ii in range(284):
        field = data[f"surf {ii}"]
        Ez, Hx, Hy = field.Ez.squeeze(), field.Hx.squeeze(), field.Hy.squeeze()
        field_adj.append([Ez, Hx, Hy])
field_adj = np.array(field_adj).reshape(5,284, -1)

#%%
Ez_adj = []
for ii in range(5):
    Ez_adj.append(adj_data[f"adj {ii}"]["field_plane"].Ez.squeeze())

#%%

weight = np.array([td.EPSILON_0, td.MU_0, td.MU_0]).reshape(1,1,-1)
sensitivity = (2*np.pi*freq0)**2 * np.sum(np.real(field_fwd * field_adj) * weight, axis=2)


fig, ax = plt.subplots(2, 3, figsize=(6.5, 4.1), sharex=True, sharey=True, tight_layout=True)

ax = ax.reshape(-1)
Ez = fwd_data["field_plane"].Ez.squeeze()
X, Y = np.meshgrid(Ez.x, Ez.y, indexing="ij")
vmax = np.max(np.abs(Ez))/5
ax[0].pcolormesh(X/mm, Y/mm, np.real(Ez), cmap=plt.cm.RdBu, vmin=-vmax,vmax=vmax)
ax[0].plot(cnt[:,0]/mm, cnt[:,1]/mm, color="gray", lw=0.5)
ax[0].set_title("Forward", fontsize=7)

for ii in range(5):
    Ez = Ez_adj[ii]
    X, Y = np.meshgrid(Ez.x, Ez.y, indexing="ij")
    vmax = np.max(np.abs(Ez))/5
    color1 = ax[1+ii].pcolormesh(X/mm, Y/mm, np.real(Ez), cmap=plt.cm.RdBu, vmin=-vmax,vmax=vmax)
    ax[1+ii].plot(cnt[:,0]/mm, cnt[:,1]/mm, color="gray", lw=0.5)
    color2 = ax[1+ii].scatter(x[::2]/mm, y[::2]/mm, c=sensitivity[ii,::2], s=1.5, cmap=plt.cm.viridis, vmin=0, vmax=sensitivity[ii].max())
    ax[1+ii].set_title(f"Adjoint {ii}", fontsize=7)
ax[0].set(xlim=(-100,100), ylim=(50,250))

cax1 = ax[2].inset_axes([1.05, 0, 0.05, 1])
cbar1 = plt.colorbar(color1, cax=cax1)
cbar1.ax.set_yticks([-vmax,0,vmax])
cbar1.ax.set_yticklabels(["-max", 0, "max"])
cbar1.ax.set_ylabel(r"Re($E_z$)")

cax2 = ax[5].inset_axes([1.05, 0, 0.05, 1])
cbar2 = plt.colorbar(color2, cax=cax2)
cbar2.ax.set_yticks([0, sensitivity[-1].max()])
cbar2.ax.set_yticklabels([0, "max"])
cbar2.ax.set_ylabel("Sensitivity")


