"""Shard IO, dataset classes, and splits for the 3D rail fields.

Shard layout (written by generate_dataset_3d.py, all under
``rail3D/data/generated/`` or a named subdirectory):
    rail3d_{class}_shard{k:03d}.pt : {"psi": complex64 (n, NX, NY, 2),  # [psi1, psi2]
                                      "meta": list[dict]}
    rail3d_intact_shard{k:03d}.pt  : same layout (augmented intact pool)
    psi0_ms.pt                     : complex64 (NX, NY) direct horn term
    dataset_config.json            : geometry + provenance (see PROVENANCE_KEYS)

Field modes (Face3D convention): "sca" = psi1 only (scattered single bounce),
"tot" = psi0 + psi1 + psi2 (what a real measurement sees).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

from . import config

SHARD_SIZE = 500


def shard_path(class_name: str, k: int, root: Path | None = None) -> Path:
    root = root or config.GENERATED_DIR
    return root / f"rail3d_{class_name}_shard{k:03d}.pt"


def psi0_path(root: Path | None = None) -> Path:
    root = root or config.GENERATED_DIR
    return root / "psi0_ms.pt"


def atomic_save(obj, path: Path) -> None:
    """Write-to-tmp-then-rename so an interrupt never leaves a corrupt file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def existing_shards(class_name: str, root: Path | None = None) -> list[int]:
    root = root or config.GENERATED_DIR
    ks = []
    for p in root.glob(f"rail3d_{class_name}_shard*.pt"):
        try:
            ks.append(int(p.stem.rsplit("shard", 1)[1]))
        except ValueError:
            continue
    return sorted(ks)


def load_class_fields(class_name: str, root: Path | None = None) -> tuple[torch.Tensor, list[dict]]:
    """Concatenate all shards of one class -> (psi (N, NX, NY, 2), meta)."""
    root = root or config.GENERATED_DIR
    ks = existing_shards(class_name, root=root)
    if not ks:
        raise FileNotFoundError(f"No shards for '{class_name}' in {root}")
    psis, metas = [], []
    for k in ks:
        bundle = torch.load(shard_path(class_name, k, root=root), map_location="cpu",
                            weights_only=False)
        psis.append(bundle["psi"])
        metas.extend(bundle["meta"])
    return torch.cat(psis, dim=0), metas


# Geometry that the stored fields depend on. If any of these differ between
# generation and use, the dataset simply does not describe the current setup.
# Expanded 2026-08-17 (λ=5 migration): the horn, the section normalization,
# the augmentation and every defect range also shape the stored fields —
# changing any of them silently invalidated a dataset with no detection.
# Datasets written before the expansion fail the check (missing keys), which
# is correct: they are λ=8 sets and cannot be used at λ=5 anyway.
PROVENANCE_KEYS = ("WVL", "DX", "NX", "NY", "H_MS", "PLANE_X_CENTER", "SEG_LEN",
                   "MESH_DS", "OCCLUDER_DS", "SHADOW_MODE", "SHADOW_MIN_T", "SHADOW_NORMAL_OFFSET", "THETA_INC",
                   "DIST_ANT", "CLASS_NAMES",
                   "SIZE_ANT", "RESOL_ANT", "Z_CUT", "RAIL_HEIGHT",
                   "N_BOUNDARY_VERTICES", "ROLL_DEG_STD", "JITTER_XZ_STD",
                   "SEED", "CROWN_HALF_WIDTH", "GAUGE_X_MIN", "GAUGE_X_MAX",
                   "DEFECT_CENTER_RANGE", "DEFECT_LENGTH_RANGE",
                   "CRACK_LENGTH_RANGE", "CRACK_WIDTH_RANGE",
                   "CRACK_DEPTH_RANGE", "DENT_DEPTH_RANGE",
                   "DENT_FOOTPRINT_Y", "DENT_FOOTPRINT_S",
                   "SHELL_RADIUS_RANGE", "SHELL_DEPTH_RANGE",
                   "WEAR_DEPTH_RANGE",
                   "CRACK_LINE_COUNT", "CRACK_LINE_GAP_FACTOR",
                   "SHELL_GAUGE_X_RANGE")


def stamp(block: dict) -> dict:
    """Tag one verification-report block with when/what computed it.

    verification_report.json is MERGE-loaded (tests_physics_3d, validation_3d
    and v8_smoke_test each write their own gates into it), so a block that was
    not re-run simply survives. Without a stamp a stale block is
    indistinguishable from a fresh one -- which is how the 2026-09-11 bundle
    shipped V6b results computed under the previous defect geometry.

    `geometry` is the same digest that names dataset roots, so a block whose
    geometry differs from the run you are reading was computed for a DIFFERENT
    scene and its numbers do not describe this one.
    """
    block["_stamp"] = {"at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                       "commit": _git_commit(), "geometry": geometry_tag()}
    return block


def geometry_tag(n: int = 6) -> str:
    """Short digest of the CURRENT provenance-tracked geometry.

    Used to disambiguate dataset roots automatically. Two configs that differ
    in any provenance key -- a wavelength, a shadow parameter, a defect range --
    produce different tags, which is exactly the set of changes that make two
    datasets incomparable.
    """
    payload = json.dumps({k: _jsonable(getattr(config, k))
                          for k in PROVENANCE_KEYS}, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:n]


def stage_root(stage: str) -> Path:
    """Where this stage's dataset lives, given the geometry in force NOW.

    `L5_<stage>` while that root is free or already holds THIS geometry;
    `L5_<stage>_<geometry_tag>` once it holds a different one.

    Without this a geometry change strands the run: the generator correctly
    refuses to mix shards into an incompatible root, but every entry point
    derived the root from the stage name alone, so there was no way forward
    except renaming by hand. Old roots stay on disk as the record.
    """
    base = config.GENERATED_DIR / f"L5_{stage}"
    if not (base / "dataset_config.json").exists():
        return base
    return base if not check_dataset_config(base, strict=False) else \
        config.GENERATED_DIR / f"L5_{stage}_{geometry_tag()}"


def run_tag(root: Path) -> str:
    """Checkpoint/run-name suffix for a dataset root.

    Derived from the ROOT, not the stage, so a geometry change gets fresh
    checkpoints too -- otherwise train3d would refuse to resume across the
    geometry stamp and the run would stall for a second reason.
    """
    return root.name.lower()


def _jsonable(v):
    """Recursively convert config values to what json.loads would return.

    JSON has no tuples: a tuple stored via json round-trips as a list, so the
    stored/current comparison must normalize BOTH sides identically or every
    tuple-valued key (SIZE_ANT, the defect ranges) mismatches on every dataset,
    including freshly generated ones. V0c guards this round-trip.
    """
    if isinstance(v, (tuple, list)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    return v


def _values_equal(was, now) -> bool:
    """Compare a stored (json) value against the current config value."""
    if isinstance(now, float) and isinstance(was, (int, float)):
        return abs(float(was) - now) < 1e-9
    if isinstance(now, (list, tuple)):
        return (isinstance(was, list) and len(was) == len(now)
                and all(_values_equal(w, n) for w, n in zip(was, now)))
    if isinstance(now, dict):
        return (isinstance(was, dict) and was.keys() == now.keys()
                and all(_values_equal(was[k], now[k]) for k in now))
    return was == now


def dataset_config_path(root: Path | None = None) -> Path:
    return (root or config.GENERATED_DIR) / "dataset_config.json"


def _git_commit() -> str:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=config.REPO_ROOT, capture_output=True,
                              text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def write_dataset_config(root: Path | None = None, note: str = "") -> dict:
    """Record the geometry AND provenance a dataset was generated with.

    Geometry so training can refuse a mismatch; provenance (when, by which
    commit, on which machine) so you can always tell which dataset you are
    looking at and how old it is.
    """
    import json
    import platform
    import time

    cfg = {k: _jsonable(getattr(config, k)) for k in PROVENANCE_KEYS}
    cfg["_created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    cfg["_created_epoch"] = time.time()
    cfg["_git_commit"] = _git_commit()
    cfg["_host"] = platform.node()
    cfg["_note"] = note
    path = dataset_config_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg


def finalize_dataset_config(root: Path | None = None, **extra) -> None:
    """Append post-generation facts (counts, duration) to dataset_config.json."""
    import json

    path = dataset_config_path(root)
    if not path.exists():
        return
    cfg = json.loads(path.read_text())
    cfg.update({f"_{k}": v for k, v in extra.items()})
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def describe_dataset(root: Path | None = None) -> str:
    """One-block human summary of a dataset: where, when, what, how big."""
    import json
    import time

    root = root or config.GENERATED_DIR
    path = dataset_config_path(root)
    lines = [f"dataset : {root}"]
    if not path.exists():
        lines.append("          (no dataset_config.json - geometry UNVERIFIED, "
                     "predates provenance recording)")
    else:
        c = json.loads(path.read_text())
        age_h = (time.time() - c.get("_created_epoch", time.time())) / 3600
        lines.append(f"created : {c.get('_created', '?')}  ({age_h:.1f} h ago)"
                     f"  commit {c.get('_git_commit', '?')}  host {c.get('_host', '?')}")
        lines.append(f"geometry: H_MS={c['H_MS']} mm  grid {c['NX']}x{c['NY']}  "
                     f"seg={c['SEG_LEN']} mm  mesh=lambda/{c['WVL']/c['MESH_DS']:.0f}  "
                     f"shadow={c['SHADOW_MODE']}")
        lines.append(f"classes : {c['CLASS_NAMES']}")
        if c.get("_note"):
            lines.append(f"note    : {c['_note']}")
    counts = {}
    for cls in list(config.CLASS_NAMES) + ["intact"]:
        ks = existing_shards(cls, root=root)
        counts[cls] = sum(
            len(torch.load(shard_path(cls, k, root=root), map_location="cpu",
                           weights_only=False)["meta"]) for k in ks)
    lines.append(f"samples : {counts}  (total {sum(counts.values())})")
    return "\n".join(lines)


def list_datasets() -> list[Path]:
    """Every dataset directory under data/generated (a dir holding shards)."""
    roots = []
    base = config.GENERATED_DIR
    if any(base.glob("rail3d_*_shard*.pt")):
        roots.append(base)
    for d in sorted(p for p in base.glob("*") if p.is_dir()):
        if any(d.glob("rail3d_*_shard*.pt")):
            roots.append(d)
    return roots


def check_dataset_config(root: Path | None = None, strict: bool = True) -> list[str]:
    """Compare a dataset's recorded geometry against the current config.

    Returns the list of mismatched keys. Raises when strict and any differ —
    silently training on fields generated for a different plane height, grid or
    class list produces results that look fine and mean nothing.
    """
    import json

    path = dataset_config_path(root)
    if not path.exists():
        print(f"[rail3d] WARNING: {path.name} missing — this dataset predates "
              f"provenance recording, so its geometry cannot be verified against "
              f"the current config (H_MS={config.H_MS}, grid {config.NX}x{config.NY}, "
              f"classes {list(config.CLASS_NAMES)}).")
        return []

    stored = json.loads(path.read_text())
    diffs = []
    for k in PROVENANCE_KEYS:
        now = _jsonable(getattr(config, k))
        was = stored.get(k)
        if not _values_equal(was, now):
            diffs.append(f"  {k}: dataset={was!r}  current={now!r}")
    if diffs and strict:
        raise RuntimeError(
            "This dataset was generated with a different geometry:\n"
            + "\n".join(diffs)
            + "\nEither restore the config values above, or generate a NEW "
              "named dataset (old ones stay on disk as the record):\n"
              "  python generate_dataset_3d.py --profile lab --name <name>")
    return [d.strip() for d in diffs]


def check_generation_root(root: Path | None = None) -> None:
    """Refuse to GENERATE into a root holding artifacts of another geometry.

    The generator skips existing shards and overwrites dataset_config.json, so
    without this check a λ change would MIX old and new shards in one root and
    then relabel the whole set as the new geometry — undetectable afterwards.
    Raises RuntimeError; silent on a clean or matching root. V0c covers all
    three cases.
    """
    root = root or config.GENERATED_DIR
    if dataset_config_path(root).exists():
        try:
            check_dataset_config(root, strict=True)
        except RuntimeError as err:
            raise RuntimeError(
                f"REFUSING to generate into {root}: it already holds a dataset "
                f"with a DIFFERENT geometry, and the generator would mix old "
                f"and new shards while relabelling them all as current.\n{err}\n"
                f"Use --name <new_name> for a fresh root, or delete this one.") from err
        return
    leftovers = (list(root.glob("rail3d_*_shard*.pt"))
                 + ([psi0_path(root)] if psi0_path(root).exists() else []))
    if leftovers:
        raise RuntimeError(
            f"REFUSING to generate into {root}: it holds {len(leftovers)} "
            f"artifact(s) but no dataset_config.json, so their geometry cannot "
            f"be verified (they predate provenance recording).\n"
            f"Use --name <new_name> for a fresh root, or delete the old files:\n"
            f"  rm -f {root}/rail3d_*_shard*.pt {root}/psi0_ms.pt")


def load_meta(class_name: str, root: Path | None = None) -> list[dict]:
    """Per-sample metadata for one class WITHOUT loading its fields.

    Same concatenation order as load_class_fields, so indices line up with the
    tensors returned by load_dataset (and therefore with the split indices).
    """
    root = root or config.GENERATED_DIR
    metas: list[dict] = []
    for k in existing_shards(class_name, root=root):
        bundle = torch.load(shard_path(class_name, k, root=root), map_location="cpu",
                            weights_only=False)
        metas.extend(bundle["meta"])
    return metas


def combine_field(psi: torch.Tensor, mode: str, psi0: torch.Tensor | None = None,
                  root: Path | None = None) -> torch.Tensor:
    """(N, NX, NY, 2) [psi1, psi2] -> (N, NX, NY) complex field."""
    if mode == "sca":
        return psi[..., 0]
    if mode == "tot":
        if psi0 is None:
            psi0 = torch.load(psi0_path(root=root), map_location="cpu", weights_only=False)
        if tuple(psi0.shape) != tuple(psi.shape[1:3]):
            # Shape alone cannot catch a wavelength change at the same grid
            # (60x30 at both λ) — dataset_config.json is the real guard; this
            # only stops a stale psi0 from a different NX/NY being broadcast.
            raise RuntimeError(
                f"psi0 shape {tuple(psi0.shape)} does not match the stored "
                f"fields {tuple(psi.shape[1:3])} — stale psi0_ms.pt in this "
                f"root; delete it and regenerate.")
        return psi[..., 0] + psi[..., 1] + psi0.unsqueeze(0)
    raise ValueError(f"Unknown field mode: {mode}")


def load_dataset(mode: str = "tot", class_names=config.CLASS_NAMES,
                 root: Path | None = None):
    """All defect classes -> (fields (N, NX, NY) complex64, labels (N,), meta, rms).

    Fields are globally RMS-normalized (like load_generated_dataset in the 2D
    notebook); the intact pool must be normalized with load_intact using the
    SAME scale, so pass the returned ``rms`` on.
    """
    fields, labels, metas = [], [], []
    for label, name in enumerate(class_names):
        psi, meta = load_class_fields(name, root=root)
        fields.append(combine_field(psi, mode, root=root))
        labels.append(torch.full((psi.shape[0],), label, dtype=torch.long))
        metas.extend(meta)
    fields = torch.cat(fields, dim=0)
    labels = torch.cat(labels, dim=0)
    rms = fields.abs().pow(2).mean().sqrt()
    return fields / rms, labels, metas, rms


def load_intact(mode: str = "tot", rms: torch.Tensor | float | None = None,
                root: Path | None = None):
    """Intact pool (stored as 'intact' shards), normalized with the defect RMS."""
    psi, meta = load_class_fields("intact", root=root)
    fields = combine_field(psi, mode, root=root)
    if rms is None:
        rms = fields.abs().pow(2).mean().sqrt()
    return fields / rms, meta


class Rail3DDataset(Dataset):
    def __init__(self, fields: torch.Tensor, labels: torch.Tensor):
        self.fields = fields
        self.labels = labels

    def __len__(self) -> int:
        return self.fields.shape[0]

    def __getitem__(self, idx):
        return self.fields[idx], self.labels[idx]


def stratified_split(
    labels: torch.Tensor,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = config.SEED,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-class 80/10/10 split, seed 0 — matches the 2D notebooks."""
    gen = torch.Generator().manual_seed(seed)
    train, val, test = [], [], []
    for c in labels.unique(sorted=True):
        idx = torch.nonzero(labels == c, as_tuple=True)[0]
        perm = idx[torch.randperm(len(idx), generator=gen)]
        n_train = int(fractions[0] * len(perm))
        n_val = int(fractions[1] * len(perm))
        train.append(perm[:n_train])
        val.append(perm[n_train:n_train + n_val])
        test.append(perm[n_train + n_val:])
    cat = lambda parts: torch.cat(parts)[torch.randperm(sum(len(p) for p in parts), generator=gen)]
    return cat(train), cat(val), cat(test)
