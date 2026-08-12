"""Figures for user review: the physical-setup diagram and module-design
visualizations (meshes, envelopes, fields, detector layouts).

Everything here is CPU-only matplotlib; safe to run on the laptop.
"""

from __future__ import annotations

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from . import config, mesh3d, sections


def _horn_geometry():
    """Horn aperture center, corners and boresight ray in the x-z plane (y=0)."""
    th = config.THETA_INC
    D = config.DIST_ANT
    A, B = config.SIZE_ANT[0], config.SIZE_ANT[1]
    center = np.array([D * np.sin(th), 0.0, D * np.cos(th)])
    # aperture long axis (A) lies in the x-z plane: s -> (D sinθ - s cosθ, y, D cosθ + s sinθ)
    a_dir = np.array([-np.cos(th), 0.0, np.sin(th)])
    b_dir = np.array([0.0, 1.0, 0.0])
    corners = [
        center + sa * (A / 2) * a_dir + sb * (B / 2) * b_dir
        for sa, sb in [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    ]
    return center, np.array(corners), a_dir


def setup_diagram(save_path=None, show=False):
    """Annotated 3-panel diagram of the simulated scene (3D, side, top)."""
    section = sections.load_reference_section()
    v, f = mesh3d.sweep_rail_mesh(section)  # intact segment
    vx, vy, vz = v[:, 0].numpy(), v[:, 1].numpy(), v[:, 2].numpy()

    horn_c, horn_corners, _ = _horn_geometry()
    z_ms = config.H_MS
    z_det = config.H_MS + config.LAYER_DISTANCES[-1]
    wx2, wy2 = config.WX / 2, config.WY / 2
    det_c = config.detector_grid_centers().numpy()
    dw, dh = config.DET_SIZE

    fig = plt.figure(figsize=(15, 5.5))

    # --- 3D view -----------------------------------------------------------
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.plot_trisurf(vx, vy, vz, triangles=f.numpy(), color="lightsteelblue",
                    edgecolor="none", alpha=0.9, shade=True)
    for zz, color, label in [(z_ms, "tab:orange", "metasurface z=160"),
                             (z_det, "tab:green", "detector plane z=320")]:
        xs = [-wx2, wx2, wx2, -wx2, -wx2]
        ys = [-wy2, -wy2, wy2, wy2, -wy2]
        ax.plot(xs, ys, [zz] * 5, color=color, lw=1.5, label=label)
    hc = np.vstack([horn_corners, horn_corners[:1]])
    ax.plot(hc[:, 0], hc[:, 1], hc[:, 2], color="tab:red", lw=1.5, label="horn aperture")
    ax.plot([horn_c[0], 0], [0, 0], [horn_c[2], 0], "r--", lw=0.8)
    ax.set(xlabel="x (mm)", ylabel="y (mm)", zlabel="z (mm)",
           title="3D scene (crown at z=0)")
    ax.legend(loc="upper left", fontsize=7)
    ax.view_init(elev=18, azim=-60)

    # --- side view (x-z, y=0) ---------------------------------------------
    ax = fig.add_subplot(1, 3, 2)
    sx = section[:, 0].numpy()
    sz = section[:, 1].numpy() - config.RAIL_HEIGHT
    ax.plot(sx, sz, color="steelblue", lw=1.2, label="rail cross-section")
    ax.axhline(config.Z_CUT, color="gray", ls=":", lw=0.8)
    ax.text(60, config.Z_CUT + 3, f"z_cut = {config.Z_CUT:.0f} (illuminated above)", fontsize=7)
    ax.plot([-wx2, wx2], [z_ms, z_ms], color="tab:orange", lw=2, label="metasurface (240 mm)")
    ax.plot([-wx2, wx2], [z_det, z_det], color="tab:green", lw=2, label="detector plane")
    ax.plot(horn_corners[[0, 1], 0], horn_corners[[0, 1], 2], color="tab:red", lw=3, label="horn (A=27.4 mm)")
    ax.annotate("", xy=(0, 0), xytext=(horn_c[0], horn_c[2]),
                arrowprops=dict(arrowstyle="->", color="tab:red", lw=0.9))
    ax.text(horn_c[0] * 0.55, horn_c[2] * 0.62,
            f"224 mm @ {np.degrees(config.THETA_INC):.0f}°", fontsize=8, color="tab:red")
    ax.annotate("", xy=(-135, z_ms), xytext=(-135, 0),
                arrowprops=dict(arrowstyle="<->", lw=0.8))
    ax.text(-158, z_ms / 2, "160 mm", fontsize=8, rotation=90)
    ax.annotate("", xy=(-135, z_det), xytext=(-135, z_ms),
                arrowprops=dict(arrowstyle="<->", lw=0.8))
    ax.text(-158, (z_ms + z_det) / 2, "160 mm", fontsize=8, rotation=90)
    ax.set(xlabel="x (mm)", ylabel="z (mm)", title="Side view (x-z at y=0)",
           xlim=(-180, 260), ylim=(-200, 360))
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=7)

    # --- top view (x-y) ----------------------------------------------------
    ax = fig.add_subplot(1, 3, 3)
    rail_w = float(section[:, 0].max() - section[:, 0].min())
    ax.add_patch(Rectangle((-rail_w / 2, -config.SEG_LEN / 2), rail_w, config.SEG_LEN,
                           color="steelblue", alpha=0.35, label=f"rail segment ({config.SEG_LEN:.0f} mm)"))
    ax.add_patch(Rectangle((-wx2, -wy2), config.WX, config.WY, fill=False,
                           edgecolor="tab:orange", lw=1.5,
                           label=f"MS aperture {config.WX:.0f}x{config.WY:.0f} mm ({config.NX}x{config.NY} px)"))
    for k, (cx, cy) in enumerate(det_c):
        ax.add_patch(Rectangle((cx - dw / 2, cy - dh / 2), dw, dh, fill=False,
                               edgecolor="tab:green", lw=1.0,
                               label="detectors (6x3 init)" if k == 0 else None))
    ax.plot(horn_c[0], horn_c[1], "r*", ms=12, label="horn center (proj.)")
    y0lo, y0hi = config.DEFECT_CENTER_RANGE
    ax.plot([0, 0], [y0lo, y0hi], "k-", lw=3, alpha=0.4, label="defect-center range y0")
    ax.set(xlabel="x (mm)", ylabel="y (mm)", title="Top view (x-y)",
           xlim=(-160, 260), ylim=(-140, 140))
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=7)

    fig.suptitle(
        f"rail3D simulated setup — λ={config.WVL:.0f} mm (37.5 GHz), dx={config.DX:.0f} mm, "
        f"grid {config.NX}x{config.NY}, horn 224 mm @ 55°, crown→MS 160 mm, MS→det 160 mm",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if save_path:
        fig.savefig(save_path, dpi=200)
    if show:
        plt.show()
    return fig


def mesh_review_figure(save_path=None, seed_base: int = 12345, csv_index: int = 0):
    """One mesh per class (crack/dent/wear + intact) with envelope insets."""
    section = sections.load_reference_section()
    n_arc = mesh3d.default_arc_count(section)

    fig = plt.figure(figsize=(16, 4.6))
    y_grid = np.linspace(-config.SEG_LEN / 2, config.SEG_LEN / 2, 400)

    for i, cls in enumerate(("intact",) + config.CLASS_NAMES):
        if cls == "intact":
            defect = None
        else:
            files = sections.get_dataset_files(cls)
            defect = sections.match_reference_width(
                sections.load_vertices_from_csv(files[csv_index]), section)
        seed = config.sample_seed(cls if cls != "intact" else "intact", csv_index)
        v, f, meta = mesh3d.build_sample_mesh(section, cls, defect, seed,
                                              n_arc=n_arc, augment=False)
        ax = fig.add_subplot(1, 4, i + 1, projection="3d")
        dz = (v[:, 2] - _intact_z_reference(section, v, n_arc)).numpy() if cls != "intact" else np.zeros(len(v))
        ax.plot_trisurf(v[:, 0].numpy(), v[:, 1].numpy(), v[:, 2].numpy(),
                        triangles=f.numpy(), cmap="viridis", edgecolor="none",
                        alpha=0.95)
        title = cls if cls == "intact" else f"{cls} (y0={meta['y0']:.0f} mm, L={meta['L']:.0f} mm)"
        ax.set(title=title, xlabel="x", ylabel="y", zlabel="z")
        ax.view_init(elev=35, azim=-75)

        if cls != "intact":
            gfun, _ = mesh3d.defect_envelope(cls, torch.Generator().manual_seed(seed))
            ax2 = ax.inset_axes([0.02, 0.02, 0.4, 0.22])
            ax2.plot(y_grid, gfun(y_grid), lw=1)
            ax2.set_title("g(y)", fontsize=6)
            ax2.tick_params(labelsize=5)

    fig.suptitle("Swept rail meshes (illuminated region only), one per class", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig


def _intact_z_reference(section, v, n_arc):
    v0, _ = mesh3d.sweep_rail_mesh(section, n_arc=n_arc)
    return v0[:, 2]


def _sample_fields(device="cpu", csv_index: int = 0):
    """psi0 plus per-class (psi1, psi2) for one sample each — small helper for
    the design-review figures. Returns dict[str, tensor] of complex (NX, NY)."""
    import torch as _torch

    from . import field3d

    device = _torch.device(device)
    section = sections.load_reference_section()
    n_arc = mesh3d.default_arc_count(section)
    X, Y = config.plane_grid(device)
    args = (X, Y, config.H_MS, config.WVL, config.THETA_INC,
            config.SIZE_ANT, config.DIST_ANT, config.RESOL_ANT)

    out = {"psi0": field3d.horn_to_plane(*args).cpu()}
    faces = None
    for cls in ("intact",) + config.CLASS_NAMES:
        defect = None
        if cls != "intact":
            files = sections.get_dataset_files(cls)
            defect = sections.match_reference_width(
                sections.load_vertices_from_csv(files[csv_index]), section)
        gen = _torch.Generator().manual_seed(config.sample_seed(cls, csv_index))
        envelope = None
        if cls != "intact":
            envelope, _ = mesh3d.defect_envelope(cls, gen)
        # generation config: λ/4 mesh, ray-cast shadow vs λ/2 occluders
        v, f = mesh3d.sweep_rail_mesh(section, defect, envelope,
                                      slice_ds=config.MESH_DS, arc_ds=config.MESH_DS)
        v_occ, f_occ = mesh3d.sweep_rail_mesh(section, defect, envelope)
        psi1, psi2 = field3d.scattered_fields(
            v.to(device), f.to(device), *args, chunk_faces=2048,
            shadow=config.SHADOW_MODE,
            shadow_occluders=(v_occ.to(device), f_occ.to(device)))
        out[cls] = (psi1.cpu(), psi2.cpu())
    return out


def field_maps_figure(save_path=None, device="cpu", csv_index: int = 0):
    """|psi0|, per-class |psi1|, |psi2|, |psi_tot| maps on the 60x30 grid."""
    fields = _sample_fields(device=device, csv_index=csv_index)
    psi0 = fields["psi0"]
    extent = (-config.WY / 2, config.WY / 2, -config.WX / 2, config.WX / 2)

    rows = ("intact",) + config.CLASS_NAMES
    fig, axes = plt.subplots(len(rows), 4, figsize=(13, 3.1 * len(rows)))
    for r, cls in enumerate(rows):
        psi1, psi2 = fields[cls]
        tot = psi0 + psi1 + psi2
        panels = [(psi0, "|psi0|² (direct horn)"), (psi1, "|psi1|² (single bounce)"),
                  (psi2, "|psi2|² (double bounce)"), (tot, "|psi_tot|²")]
        for c, (psi, title) in enumerate(panels):
            ax = axes[r, c]
            im = ax.imshow((psi.abs() ** 2).numpy(), extent=extent, aspect="equal",
                           cmap="inferno", origin="lower")
            plt.colorbar(im, ax=ax, fraction=0.03)
            if r == 0:
                ax.set_title(title, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{cls}\nx (mm)", fontsize=8)
            ax.set_xlabel("y (mm)", fontsize=7)
    fig.suptitle("Fields at the metasurface plane (z = 160 mm), one sample per class")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig


def propagator_figure(save_path=None):
    """RS kernel, FFT-vs-conv2d agreement, RS-vs-ASM beam comparison."""
    import torch as _torch

    from . import optics3d

    prop = optics3d.PropagatorRSFFT(config.LAYER_DISTANCES[-1])
    asm = optics3d.PropagatorASM2D(config.LAYER_DISTANCES[-1], pad_factor=4)
    X, Y = config.plane_grid()
    beam = _torch.exp(-(X[0] ** 2 + Y[0] ** 2) / (2 * 25.0**2)) + 0j

    out_fft = prop(beam.unsqueeze(0))[0]
    out_dir = prop.forward_direct(beam.unsqueeze(0))[0]
    out_asm = asm(beam.unsqueeze(0))[0]

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    im = axes[0].imshow(prop.kernel.real.numpy(), cmap="RdBu", origin="lower")
    axes[0].set_title(f"RS kernel Re, D={config.LAYER_DISTANCES[-1]:.0f} mm\n"
                      f"({prop.kernel.shape[0]}x{prop.kernel.shape[1]})", fontsize=9)
    plt.colorbar(im, ax=axes[0], fraction=0.04)

    im = axes[1].imshow((out_fft.abs() ** 2).numpy(), cmap="inferno", origin="lower")
    axes[1].set_title("Gaussian beam after 160 mm (RS-FFT)", fontsize=9)
    plt.colorbar(im, ax=axes[1], fraction=0.04)

    diff = (out_fft - out_dir).abs().numpy()
    im = axes[2].imshow(diff, cmap="viridis", origin="lower")
    rel = float(_torch.linalg.norm(out_fft - out_dir) / _torch.linalg.norm(out_dir))
    axes[2].set_title(f"|FFT - conv2d|  (rel L2 = {rel:.1e})", fontsize=9)
    plt.colorbar(im, ax=axes[2], fraction=0.04)

    mid = config.NY // 2
    axes[3].plot((out_fft.abs() ** 2)[:, mid].numpy(), label="RS-FFT")
    axes[3].plot((out_asm.abs() ** 2)[:, mid].numpy(), "--", label="ASM")
    axes[3].set_title("central cut: RS vs angular spectrum", fontsize=9)
    axes[3].legend(fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig


def metaunit_figure(save_path=None):
    """Cropped meta-atom library: response curves + the crop in context."""
    import torch as _torch

    from . import optics3d

    unit = optics3d.MetaUnitSoft()
    w = _torch.linspace(1.0, 3.8, 200)

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    pix = [(0, 0), (30, 15), (59, 29), (15, 7), (45, 22)]
    for (i, j) in pix:
        amp = sum(unit.ampfit[d, i, j] * w**d for d in range(unit.ampfit.shape[0]))
        ph = sum(unit.phasefit[d, i, j] * w**d for d in range(unit.phasefit.shape[0]))
        axes[0].plot(w, amp, lw=0.9, label=f"px {i},{j}")
        axes[1].plot(w, ph, lw=0.9)
    axes[0].set(title="amplitude A(w) at sample pixels", xlabel="pillar width w (mm)")
    axes[0].legend(fontsize=6)
    axes[1].set(title="phase φ(w) at sample pixels (rad)", xlabel="pillar width w (mm)")

    full = np.load(_LIB_DIR_PATH() / "library_amp_fit.npy").reshape(-1, 80, 80)[::-1]
    im = axes[2].imshow(full[0], cmap="viridis", origin="lower")
    from matplotlib.patches import Rectangle as _Rect
    axes[2].add_patch(_Rect((25, 10), 30, 60, fill=False, edgecolor="red", lw=1.5))
    axes[2].set_title("amp-fit coeff (deg 0) on 80x80\nred = 60x30 crop used", fontsize=9)
    plt.colorbar(im, ax=axes[2], fraction=0.04)

    w_map = unit.w_pillar.detach()
    im = axes[3].imshow(w_map.numpy(), cmap="viridis", origin="lower", vmin=1, vmax=3.8)
    axes[3].set_title("initial w_pillar map (soft-clamped, mm)", fontsize=9)
    plt.colorbar(im, ax=axes[3], fraction=0.04)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig


def _LIB_DIR_PATH():
    from pathlib import Path as _P
    return _P(__file__).resolve().parent


def detector_figure(save_path=None):
    """Initial detector layout with soft masks at anneal start and end."""
    from . import optics3d

    det = optics3d.SoftDetector2D()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    extent = (-config.WY / 2, config.WY / 2, -config.WX / 2, config.WX / 2)
    for ax, tau, title in [
        (axes[0], 0.25, "soft masks, τ = w/4 (train start)"),
        (axes[1], 1 / 16, "soft masks, τ = w/16 (train end)"),
        (axes[2], None, "hard masks (evaluation readout)"),
    ]:
        if tau is None:
            masks = det.hard_masks()
        else:
            det.tau_scale = tau
            masks = det.soft_masks()
        im = ax.imshow(masks.sum(dim=0).detach().numpy(), extent=extent,
                       origin="lower", cmap="viridis")
        ax.set(title=title, xlabel="y (mm)", ylabel="x (mm)")
        plt.colorbar(im, ax=ax, fraction=0.03)
    fig.suptitle(f"SoftDetector2D: {det.n_det} windows "
                 f"({config.DET_SIZE[0]}x{config.DET_SIZE[1]} mm), centers trainable, "
                 f"pruned to {config.N_DET_FINAL} during training")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig


def barcode_figure(save_path=None, device="cpu"):
    """Untrained-model barcode space: per-class detector powers + gaps."""
    import torch as _torch

    from . import losses3d, optics3d

    fields = _sample_fields(device=device)
    psi0 = fields["psi0"]
    model = optics3d.ONN3D(noise=False)
    model.eval()

    names = ("intact",) + config.CLASS_NAMES
    dets = {}
    with _torch.no_grad():
        for cls in names:
            psi1, psi2 = fields[cls]
            tot = (psi0 + psi1 + psi2).unsqueeze(0)
            _, det = model(tot, hard=True)
            dets[cls] = det[0]

    ref = dets["intact"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    x = np.arange(len(ref))
    width = 0.2
    for k, cls in enumerate(names):
        axes[0].bar(x + (k - 1.5) * width, dets[cls].numpy(), width, label=cls)
    axes[0].set(title="untrained detector barcodes (hard powers)",
                xlabel="detector index", ylabel="integrated intensity")
    axes[0].legend(fontsize=8)

    gaps = [float(losses3d.relative_l2_gap(dets[c].unsqueeze(0), ref)) for c in config.CLASS_NAMES]
    axes[1].bar(config.CLASS_NAMES, gaps, color=["tab:blue", "tab:orange", "tab:green"])
    axes[1].axhline(losses3d.DETECTION_MARGIN, color="red", ls="--",
                    label=f"detection margin {losses3d.DETECTION_MARGIN}")
    axes[1].set(title="relative L2 gap vs intact (untrained)", ylabel="gap")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig


def cross_section_overlay_figure(save_path=None, csv_index: int = 0):
    """Intact vs defect cross-section loops (the 2D input data), per class."""
    section = sections.load_reference_section()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    for ax, cls in zip(axes, config.CLASS_NAMES):
        files = sections.get_dataset_files(cls)
        defect = sections.match_reference_width(
            sections.load_vertices_from_csv(files[csv_index]), section)
        ax.plot(section[:, 0], section[:, 1] - config.RAIL_HEIGHT, color="gray",
                lw=0.8, ls="--", label="intact")
        ax.plot(defect[:, 0], defect[:, 1] - config.RAIL_HEIGHT, color="black",
                lw=0.9, label=cls)
        ax.axhline(config.Z_CUT, color="tab:red", ls=":", lw=0.7)
        ax.set(title=cls, xlabel="x (mm)")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("z (mm)")
    fig.suptitle("Defect cross-sections (blended in along y by the envelope)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
    return fig
