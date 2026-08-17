"""Shard IO, dataset classes, and splits for the 3D rail fields.

Shard layout (written by generate_dataset_3d.py, all under
``rail3D/data/generated/``):
    rail3d_{class}_shard{k:03d}.pt : {"psi": complex64 (n, NX, NY, 2),  # [psi1, psi2]
                                      "meta": list[dict]}
    rail3d_intact.pt               : same layout (augmented intact pool)
    psi0_ms.pt                     : complex64 (NX, NY) direct horn term

Field modes (Face3D convention): "sca" = psi1 only (scattered single bounce),
"tot" = psi0 + psi1 + psi2 (what a real measurement sees).
"""

from __future__ import annotations

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
PROVENANCE_KEYS = ("WVL", "DX", "NX", "NY", "H_MS", "PLANE_X_CENTER", "SEG_LEN",
                   "MESH_DS", "OCCLUDER_DS", "SHADOW_MODE", "THETA_INC",
                   "DIST_ANT", "CLASS_NAMES")


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

    cfg = {k: getattr(config, k) for k in PROVENANCE_KEYS}
    cfg["CLASS_NAMES"] = list(cfg["CLASS_NAMES"])
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
        now = getattr(config, k)
        if k == "CLASS_NAMES":
            now = list(now)
        was = stored.get(k)
        if isinstance(now, float) and isinstance(was, (int, float)):
            same = abs(float(was) - now) < 1e-9
        else:
            same = was == now
        if not same:
            diffs.append(f"  {k}: dataset={was!r}  current={now!r}")
    if diffs and strict:
        raise RuntimeError(
            "This dataset was generated with a different geometry:\n"
            + "\n".join(diffs)
            + f"\nEither restore the config values above, or regenerate:\n"
              f"  rm -f {(root or config.GENERATED_DIR)}/rail3d_*_shard*.pt\n"
              f"  python generate_dataset_3d.py --profile lab")
    return [d.strip() for d in diffs]


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
