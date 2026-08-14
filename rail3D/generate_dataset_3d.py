"""Resumable 3D rail-field dataset generator.

Examples (run from rail3D/):
    python generate_dataset_3d.py --profile laptop --smoke     # 20/class + 32 intact -> data/generated/smoke/
    python generate_dataset_3d.py --profile laptop             # full 5000/class + 512 intact
    python generate_dataset_3d.py --status                     # what exists / what remains
    python generate_dataset_3d.py --profile lab                # same on the RTX 5090 (auto-picks the strongest GPU)

Physics per sample (validated in V1-V7): λ/4 swept mesh, ray-cast shadowing
against a λ/2 occluder mesh, exact RS-I surface integral, psi1 + psi2 channels;
the face-independent psi0 is cached once. Shards are written atomically and
skipped on re-run, and every sample is derived from a deterministic seed, so an
interrupted run resumes losslessly on any machine.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from rail3d import config, data3d, field3d, mesh3d, sections

FULL_PER_CLASS = 5000
FULL_INTACT = 512
SMOKE_PER_CLASS = 20
SMOKE_INTACT = 32


def log(msg: str, root: Path) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(root / "generation.log", "a") as fh:
        fh.write(line + "\n")


def plan_counts(args) -> dict[str, int]:
    if args.smoke:
        counts = {cls: SMOKE_PER_CLASS for cls in config.CLASS_NAMES}
        counts["intact"] = SMOKE_INTACT
    else:
        counts = {cls: args.limit for cls in config.CLASS_NAMES}
        counts["intact"] = args.intact
    if args.classes:
        counts = {k: v for k, v in counts.items() if k in args.classes}
    return counts


def generate_class(
    class_name: str,
    n_samples: int,
    device: torch.device,
    profile: config.Profile,
    root: Path,
    shard_size: int,
) -> None:
    section = sections.load_reference_section()
    n_arc_fine = mesh3d.default_arc_count(section, config.MESH_DS)
    n_arc_coarse = mesh3d.default_arc_count(section, config.OCCLUDER_DS)

    # "shell" is fully parametric and "intact" has no defect: neither uses CSVs
    files = (sections.get_dataset_files(class_name)
             if class_name in config.DATASET_DIRS else None)
    if files is not None and n_samples > len(files):
        raise ValueError(f"{class_name}: requested {n_samples} but only {len(files)} CSVs")

    X, Y = config.plane_grid(device)
    solve_args = (X, Y, config.H_MS, config.WVL, config.THETA_INC,
                  config.SIZE_ANT, config.DIST_ANT, config.RESOL_ANT)

    faces_fine = faces_coarse = None
    B = profile.batch_meshes
    n_shards = (n_samples + shard_size - 1) // shard_size
    t_start, n_done_new = time.time(), 0

    for k in range(n_shards):
        path = data3d.shard_path(class_name, k, root=root)
        lo, hi = k * shard_size, min((k + 1) * shard_size, n_samples)
        if path.exists():
            log(f"{class_name} shard {k:03d} exists — skipped", root)
            continue

        psis, metas = [], []
        for start in range(lo, hi, B):
            stop = min(start + B, hi)
            vs, vs_occ = [], []
            batch_meta = []
            for i in range(start, stop):
                seed = config.sample_seed(class_name, i)
                defect = None
                if files is not None:
                    defect = sections.match_reference_width(
                        sections.load_vertices_from_csv(files[i]), section)
                # defect params drawn ONCE at the canonical (fine) resolution,
                # then rendered on both grids -> fine sim mesh and coarse
                # ray-cast occluder describe the same physical defect
                params, aug = mesh3d.defect_params_for_sample(
                    section, class_name, defect, seed, n_arc=n_arc_fine)
                v, f = mesh3d.sweep_rail_mesh(
                    section, defect_params=params,
                    slice_ds=config.MESH_DS, arc_ds=config.MESH_DS, n_arc=n_arc_fine,
                    roll_deg=aug["roll_deg"],
                    jitter_xz=(aug["jitter_x"], aug["jitter_z"]),
                    faces=faces_fine)
                faces_fine = f
                v_occ, f_occ = mesh3d.sweep_rail_mesh(
                    section, defect_params=params,
                    slice_ds=config.OCCLUDER_DS, arc_ds=config.OCCLUDER_DS,
                    n_arc=n_arc_coarse,
                    roll_deg=aug["roll_deg"],
                    jitter_xz=(aug["jitter_x"], aug["jitter_z"]),
                    faces=faces_coarse)
                faces_coarse = f_occ
                meta = {k2: val for k2, val in params.items()
                        if k2 not in ("profile_v", "profile_d", "csv_dev", "csv_s")}
                meta.update(aug)
                meta["seed"] = seed
                meta["csv_index"] = i
                meta["class"] = class_name
                vs.append(v)
                vs_occ.append(v_occ)
                batch_meta.append(meta)

            v_b = torch.stack(vs).to(device)
            v_occ_b = torch.stack(vs_occ).to(device)
            psi1, psi2 = field3d.scattered_fields(
                v_b, faces_fine.to(device), *solve_args,
                chunk_faces=profile.chunk_faces,
                shadow=config.SHADOW_MODE,
                shadow_occluders=(v_occ_b, faces_coarse.to(device)),
            )
            psis.append(torch.stack([psi1, psi2], dim=-1).cpu())
            metas.extend(batch_meta)
            n_done_new += stop - start

            rate = n_done_new / max(1e-9, time.time() - t_start)
            if (stop - lo) % 50 < B:
                log(f"{class_name} {stop}/{n_samples}  ({rate:.2f} samples/s)", root)

        data3d.atomic_save({"psi": torch.cat(psis, dim=0), "meta": metas}, path)
        log(f"{class_name} shard {k:03d} [{lo}:{hi}] written -> {path.name}", root)


def sweep_remaining(counts, root, shard_size):
    remaining = []
    for cls, n in counts.items():
        n_shards = (n + shard_size - 1) // shard_size
        for k in range(n_shards):
            if not data3d.shard_path(cls, k, root=root).exists():
                remaining.append((cls, k))
    return remaining


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="laptop", choices=list(config.PROFILES))
    parser.add_argument("--smoke", action="store_true", help="tiny set into data/generated/smoke/")
    parser.add_argument("--limit", type=int, default=FULL_PER_CLASS, help="samples per defect class")
    parser.add_argument("--intact", type=int, default=FULL_INTACT, help="intact pool size")
    parser.add_argument("--classes", nargs="*", default=None,
                        help="subset of {crack,dent,wear,intact}")
    parser.add_argument("--shard-size", type=int, default=data3d.SHARD_SIZE)
    parser.add_argument("--status", action="store_true", help="report shard status and exit")
    args = parser.parse_args()

    config.ensure_dirs()
    root = config.GENERATED_DIR / "smoke" if args.smoke else config.GENERATED_DIR
    root.mkdir(parents=True, exist_ok=True)
    shard_size = min(args.shard_size, SMOKE_PER_CLASS) if args.smoke else args.shard_size
    counts = plan_counts(args)

    if args.status:
        remaining = sweep_remaining(counts, root, shard_size)
        done = {cls: len(data3d.existing_shards(cls, root=root)) for cls in counts}
        print(f"root: {root}")
        print(f"shards present: {done}")
        print(f"shards remaining: {len(remaining)} -> {remaining[:20]}{'...' if len(remaining) > 20 else ''}")
        return 0

    profile = config.PROFILES[args.profile]
    device = config.get_device(args.profile)
    log(f"start: profile={args.profile} device={device} counts={counts} shard_size={shard_size}", root)

    # psi0 (face independent) — once
    psi0_file = data3d.psi0_path(root=root)
    if not psi0_file.exists():
        X, Y = config.plane_grid(device)
        psi0 = field3d.horn_to_plane(X, Y, config.H_MS, config.WVL, config.THETA_INC,
                                     config.SIZE_ANT, config.DIST_ANT, config.RESOL_ANT)
        data3d.atomic_save(psi0.cpu(), psi0_file)
        log(f"psi0 cached -> {psi0_file.name}", root)

    t0 = time.time()
    for cls, n in counts.items():
        generate_class(cls, n, device, profile, root, shard_size)
    log(f"all requested shards complete in {(time.time() - t0) / 3600:.2f} h", root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
