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
