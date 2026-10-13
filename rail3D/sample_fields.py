"""One sample per class, geometry -> field at the metasurface plane: a 5 x 4 figure.

    python sample_fields.py                         # FRESH: computed now, current geometry
    python sample_fields.py --root data/generated/L5_prelim_689da5   # stored fields
    python sample_fields.py --index 3 --n-intact 8

Rows: intact, crack, dent, wear, shell. Columns:
  1  d(s, y)       the defect's depth field (s across the head, y along the rail)
  2  |psi|^2       intensity arriving at the metasurface plane (z = H_MS)
  3  arg psi       its phase there, wrapped to (-pi, pi]; grey where the
                   intensity is below PHASE_FLOOR of the row's peak (the phase
                   of nothing is noise)
  4  delta |psi|^2 this sample minus the MEAN intact intensity. For the intact
                   row that is the augmentation spread (roll / jitter) -- the
                   floor any defect has to clear.

The field is the metasurface's INPUT, "tot" = psi0 + psi1 + psi2 with the
upward-only rule (README finding 31: = psi1 with the horn above the plane).

FRESH mode builds each sample through generate_dataset_3d.sample_meshes and
solve_fields -- the generator's own code path, augmentation and ray-cast
shadowing included -- so it is the field the dataset WILL hold for that seed.
--root reads what a dataset DID store (same seeds, so the two must agree; a
geometry mismatch is warned about, since the depth maps are re-derived with
the current config). Writes data/figures/sample_fields[_<root>].png.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import generate_dataset_3d as gen
from rail3d import config, data3d, field3d, mesh3d, sections

ROWS = ("intact",) + tuple(config.CLASS_NAMES)
PHASE_FLOOR = 0.01


def fresh_fields(index: int, n_intact: int, profile_name: str):
    """{cls: (tot, meta)} for sample `index` of each class + an intact pool."""
    device = config.get_device(profile_name)
    profile = config.PROFILES[profile_name]
    section = sections.load_reference_section()
    n_fine = mesh3d.default_arc_count(section, config.MESH_DS)
    n_coarse = mesh3d.default_arc_count(section, config.OCCLUDER_DS)
    X, Y = config.plane_grid(device)
    psi0 = field3d.horn_to_plane(X, Y, config.H_MS, config.WVL, config.THETA_INC,
                                 config.SIZE_ANT, config.DIST_ANT, config.RESOL_ANT,
                                 upward_only=True).cpu()
    out, pool = {}, []
    for cls in ROWS:
        files = sections.get_dataset_files(cls) if cls in config.DATASET_DIRS else None
        idx = range(n_intact) if cls == "intact" else [index]
        for i in idx:
            t0 = time.time()
            v, f, vo, fo, meta = gen.sample_meshes(section, cls, i, files, n_fine, n_coarse)
            psi1, psi2 = gen.solve_fields(v[None].to(device), f, vo[None].to(device), fo,
                                          device, profile.chunk_faces)
            tot = (psi0 + psi1[0].cpu() + psi2[0].cpu())
            print(f"  {cls} #{i}: solved in {time.time() - t0:.1f} s")
            if cls == "intact":
                pool.append(tot)
                if i == min(index, n_intact - 1):
                    out[cls] = (tot, meta)
            else:
                out[cls] = (tot, meta)
    return out, torch.stack(pool)


def term_fields(index: int, profile_name: str):
    """{cls: dict} with every term at the metasurface plane, both WITH the
    upward-only rule (what a dataset stores) and WITHOUT it (all light that
    crosses z = H_MS, in any direction -- what the pre-finding-31 figures
    summed). psi1 does not depend on the rule; psi0 and psi2 do."""
    device = config.get_device(profile_name)
    profile = config.PROFILES[profile_name]
    section = sections.load_reference_section()
    n_fine = mesh3d.default_arc_count(section, config.MESH_DS)
    n_coarse = mesh3d.default_arc_count(section, config.OCCLUDER_DS)
    X, Y = config.plane_grid(device)
    hargs = (X, Y, config.H_MS, config.WVL, config.THETA_INC, config.SIZE_ANT,
             config.DIST_ANT, config.RESOL_ANT)
    psi0_up = field3d.horn_to_plane(*hargs, upward_only=True).cpu()
    psi0_all = field3d.horn_to_plane(*hargs, upward_only=False).cpu()
    out = {}
    for cls in ROWS:
        files = sections.get_dataset_files(cls) if cls in config.DATASET_DIRS else None
        t0 = time.time()
        v, f, vo, fo, meta = gen.sample_meshes(section, cls, index, files, n_fine, n_coarse)
        v, vo = v[None].to(device), vo[None].to(device)
        p1, p2_up = gen.solve_fields(v, f, vo, fo, device, profile.chunk_faces)
        p1b, p2_all = gen.solve_fields(v, f, vo, fo, device, profile.chunk_faces,
                                       upward_only=False)
        assert torch.equal(p1, p1b), "psi1 must not depend on the upward-only rule"
        out[cls] = {"meta": meta, "psi1": p1[0].cpu(),
                    "psi0_up": psi0_up, "psi2_up": p2_up[0].cpu(),
                    "psi0_all": psi0_all, "psi2_all": p2_all[0].cpu()}
        print(f"  {cls} #{index}: solved (x2) in {time.time() - t0:.1f} s")
    return out


def _x_period(I: np.ndarray) -> float:
    """Dominant fringe period across the rail (x), mm: FFT of the y-mean.
    Periods above WX/5 (the slow brightness envelope) are not fringes."""
    prof = I.mean(axis=1) - I.mean()
    spec = np.abs(np.fft.rfft(prof))
    spec[:5] = 0
    return config.WX / int(np.argmax(spec))


def _db(a, b):
    return 10 * np.log10(max(a, 1e-30) / max(b, 1e-30))


def draw_terms(fields, out_png):
    X, Y = config.plane_grid("cpu")
    X, Y = X[0].numpy() + config.PLANE_X_CENTER, Y[0].numpy()
    fig, axs = plt.subplots(len(ROWS), 4, figsize=(17, 2.95 * len(ROWS)))
    heads = ["|ψ0|²  direct horn light\n(ALL of it crossing the plane)",
             "|ψ1|²  single bounce off the rail\n= the stored metasurface input",
             "|ψ2|²  rail → horn → plane\n(ALL of it crossing the plane)",
             "|ψ0 + ψ1 + ψ2|²  without the upward-only rule\n"
             "= how the OLD figures defined ψ_tot"]
    report = {}
    for r, cls in enumerate(ROWS):
        F = fields[cls]
        tot_all = F["psi0_all"] + F["psi1"] + F["psi2_all"]
        tot_up = F["psi0_up"] + F["psi1"] + F["psi2_up"]
        I1 = (F["psi1"].abs() ** 2).numpy()
        P1 = I1.sum()
        terms = [F["psi0_all"], F["psi1"], F["psi2_all"], tot_all]
        rep = {"psi0_vs_psi1_dB": _db((F["psi0_all"].abs() ** 2).sum().item(), P1),
               "psi2_vs_psi1_dB": _db((F["psi2_all"].abs() ** 2).sum().item(), P1),
               "stored_input_equals_psi1": bool(torch.equal(tot_up, F["psi1"])),
               "x_period_tot_all_mm": _x_period((tot_all.abs() ** 2).numpy())}
        report[cls] = rep
        notes = [f"{rep['psi0_vs_psi1_dB']:+.1f} dB vs ψ1; upward-only → 0",
                 f"stored input (upward-only ψ0+ψ1+ψ2) == ψ1: {rep['stored_input_equals_psi1']}",
                 f"{rep['psi2_vs_psi1_dB']:+.1f} dB vs ψ1; upward-only → 0",
                 f"fringes ≈ {rep['x_period_tot_all_mm']:.1f} mm apart across x "
                 f"(λ = {config.WVL:g}, pixel {config.DX:g})"]
        for c, (psi, note) in enumerate(zip(terms, notes)):
            ax = axs[r][c]
            im = ax.pcolormesh(X, Y, (psi.abs() ** 2).numpy(), cmap="inferno",
                               shading="nearest")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7)
            ax.set_title((heads[c] + "\n" if r == 0 else "") + note, fontsize=8)
            ax.set_xlabel("x (mm) across the rail", fontsize=7)
        axs[r][0].set_ylabel(f"{_label(cls, F['meta'])}\ny (mm)", fontsize=8)
    fig.suptitle(f"Field terms at the metasurface plane (z = {config.H_MS:.0f} mm, "
                 f"λ = {config.WVL:g} mm, horn {config.DIST_ANT:.1f} mm out, lowest edge "
                 f"at z = {config.HORN_LOWER_EDGE_Z:.0f}). Each panel on its own scale. "
                 "Columns as in the old field_maps.png", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return report


def stored_fields(root: Path, index: int):
    psi0 = torch.load(data3d.psi0_path(root=root), map_location="cpu", weights_only=False)
    out = {}
    for cls in ROWS:
        psi, meta = data3d.load_class_fields(cls, root=root)
        tot = data3d.combine_field(psi, "tot", psi0=psi0)
        k = next((j for j, m in enumerate(meta) if m.get("csv_index") == index), index)
        out[cls] = (tot[k], meta[k])
        if cls == "intact":
            pool = tot
    return out, pool


def _label(cls, m):
    if cls == "intact":
        return f"intact #{m.get('csv_index', 0)}: no defect"
    bits = [f"{cls} #{m.get('csv_index', 0)}: depth {m['depth']:.1f} mm"]
    if cls == "crack":
        bits.append(f"θ {np.degrees(m['theta']):.0f}°, w {m['width']:.1f}")
    if cls == "dent":
        bits.append(f"FWHM {m['fw_s']:.0f}×{m['fw_y']:.0f}")
    if cls == "wear":
        bits.append(f"gauge {m['gauge_angle_deg']:.0f}°"
                    + (f" + crown {m['crown_depth']:.1f}" if m.get("crown") else ""))
    if cls == "shell":
        bits.append(f"{2 * m['r_s']:.0f}×{2 * m['r_y']:.0f} mm")
    return ", ".join(bits)


def draw(fields, pool, out_png, source):
    section = sections.load_reference_section()
    geom = mesh3d.arc_geometry(section, mesh3d.default_arc_count(section, config.MESH_DS))
    y_grid = np.arange(-config.SEG_LEN / 2, config.SEG_LEN / 2 + config.MESH_DS / 2,
                       config.MESH_DS)
    X, Y = config.plane_grid("cpu")
    X, Y = X[0].numpy() + config.PLANE_X_CENTER, Y[0].numpy()
    I_int = (pool.abs() ** 2).mean(0).numpy()
    vmax_I = max(float((t.abs() ** 2).max()) for t, _ in fields.values())

    fig, axs = plt.subplots(len(ROWS), 4, figsize=(17, 3.05 * len(ROWS)),
                            gridspec_kw={"width_ratios": [1.0, 1.25, 1.25, 1.25]})
    for r, cls in enumerate(ROWS):
        tot, m = fields[cls]
        I = (tot.abs() ** 2).numpy()

        ax = axs[r][0]
        if cls == "intact":
            d = np.zeros((len(y_grid), len(geom["s"])))
        else:
            p, _ = mesh3d.defect_params_for_sample(section, cls, None, m["seed"],
                                                   n_arc=geom["n_arc"])
            d = mesh3d.render_depth_field(p, geom, y_grid)
        im = ax.imshow(d, origin="lower", cmap="hot", aspect="auto", vmin=0,
                       vmax=d.max() if d.max() > 0 else 1.0,
                       extent=(geom["s"][0], geom["s"][-1], y_grid[0], y_grid[-1]))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label("mm", fontsize=7)
        ax.set_title(_label(cls, m) + f"\nd(s, y), max {d.max():.1f} mm", fontsize=8)
        ax.set_xlabel("s (mm) across the head", fontsize=7)
        ax.set_ylabel("y (mm) along the rail", fontsize=7)

        ax = axs[r][1]
        im = ax.pcolormesh(X, Y, I, cmap="inferno", vmin=0, vmax=vmax_I, shading="nearest")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        ax.set_title(f"|ψ|² at the metasurface (z = {config.H_MS:.0f} mm)\n"
                     f"shared scale; peak {I.max() / vmax_I:.2f} of the max row", fontsize=8)

        ax = axs[r][2]
        ph = np.angle(tot.numpy())
        ph = np.ma.masked_where(I < PHASE_FLOOR * I.max(), ph)
        cmap = plt.get_cmap("twilight").copy()
        cmap.set_bad("0.75")
        im = ax.pcolormesh(X, Y, ph, cmap=cmap, vmin=-np.pi, vmax=np.pi, shading="nearest")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, ticks=[-np.pi, 0, np.pi])
        cb.ax.set_yticklabels(["−π", "0", "π"])
        ax.set_title(f"phase arg ψ at the metasurface\n(grey: |ψ|² < {PHASE_FLOOR:.0%} of peak)",
                     fontsize=8)

        ax = axs[r][3]
        diff = I - I_int
        lim = float(np.abs(diff).max())
        im = ax.pcolormesh(X, Y, diff, cmap="bwr", vmin=-lim, vmax=lim, shading="nearest")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        what = ("augmentation spread: this intact − mean intact" if cls == "intact"
                else "this sample − mean intact")
        ax.set_title(f"{what}\nmax |Δ| = {100 * lim / I_int.max():.1f}% of the intact peak",
                     fontsize=8)
        for a in axs[r][1:]:
            a.set_aspect("equal")
            a.tick_params(labelsize=7)
            a.set_xlabel("x (mm) across the rail", fontsize=7)
            a.set_ylabel("y (mm)", fontsize=7)
        axs[r][0].tick_params(labelsize=7)

    fig.suptitle(f"One sample per class: geometry → field arriving at the metasurface "
                 f"(λ = {config.WVL:g} mm, plane {config.WX:.0f} × {config.WY:.0f} mm, "
                 f"{len(pool)} intact samples in the mean) — {source}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None,
                    help="dataset root to read; default: compute FRESH for the current geometry")
    ap.add_argument("--index", type=int, default=0, help="sample index per class")
    ap.add_argument("--n-intact", type=int, default=4,
                    help="FRESH mode: intact samples in the mean (each is one solve)")
    ap.add_argument("--profile", default="laptop", choices=list(config.PROFILES))
    ap.add_argument("--terms", action="store_true",
                    help="instead: psi0 | psi1 | psi2 | psi_tot per class, with and "
                         "without the upward-only rule (sample_fields_terms.png)")
    a = ap.parse_args()
    config.ensure_dirs()
    if a.terms:
        print(f"[rail3d] field terms, FRESH (geometry {data3d.geometry_tag()})")
        out = config.FIGURE_DIR / "sample_fields_terms.png"
        rep = draw_terms(term_fields(a.index, a.profile), out)
        for cls, r in rep.items():
            print(f"  {cls:7s} psi0 {r['psi0_vs_psi1_dB']:+6.1f} dB, psi2 "
                  f"{r['psi2_vs_psi1_dB']:+6.1f} dB vs psi1 | stored input == psi1: "
                  f"{r['stored_input_equals_psi1']} | psi0+psi1+psi2 fringes "
                  f"{r['x_period_tot_all_mm']:.1f} mm apart across x")
        print(f"  -> {out}")
        return 0
    if a.root:
        root = Path(a.root)
        print(data3d.describe_dataset(root))
        diffs = data3d.check_dataset_config(root, strict=False)
        if diffs:
            print("!! GEOMETRY MISMATCH vs the current config -- the depth maps are "
                  "re-derived with the current config and will NOT match the stored fields:")
            for d in diffs:
                print(f"   {d}")
        fields, pool = stored_fields(root, a.index)
        source = f"stored fields, {root.name}"
        out = config.FIGURE_DIR / f"sample_fields_{root.name}.png"
    else:
        print(f"[rail3d] computing FRESH (geometry {data3d.geometry_tag()}), "
              f"{len(ROWS) - 1} samples + {a.n_intact} intact")
        fields, pool = fresh_fields(a.index, a.n_intact, a.profile)
        source = f"computed fresh, geometry {data3d.geometry_tag()} (dataset seeds, augmented, shadowed)"
        out = config.FIGURE_DIR / "sample_fields.png"
    draw(fields, pool, out, source)
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
