"""Losses and metrics for detection + classification, ported from the 2D rail
notebooks (relative-gap margin, power floor, ROC/AUC) and extended with the
robustness terms from the plan (intact compactness, hardest-k hinge,
class-centroid margin, TV smoothness).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import config

DETECTION_MARGIN = 0.40      # defects must sit outside this relative gap
INTACT_MARGIN = 0.15         # intact samples must stay inside this gap
CLASS_CENTROID_MARGIN = 0.25
TOPK_FRACTION = 0.10
POWER_FLOOR_RATIO = 2.0

W_INTACT = 0.5
W_CLASS = 0.5
W_POWER = 0.2
W_TOPK = 0.3
W_CENTROID = 0.2
W_TV = 1e-4


# ---------------------------------------------------------------------------
# Core quantities (ported from training_ms_notebook)
# ---------------------------------------------------------------------------
def relative_l2_gap(det: torch.Tensor, det_ref: torch.Tensor) -> torch.Tensor:
    """||det - det_ref||_2 / ||det_ref||_2 per sample. det_ref: (N_det,) or (1, N_det)."""
    ref = det_ref.reshape(1, -1)
    return (det - ref).norm(dim=1) / ref.norm()


def linf_gap(det: torch.Tensor, det_ref: torch.Tensor) -> torch.Tensor:
    """max_i |det_i - ref_i| / ref_i — a single anomalous detector suffices."""
    ref = det_ref.reshape(1, -1)
    return ((det - ref).abs() / ref.abs().clamp_min(1e-12)).max(dim=1).values


# ---------------------------------------------------------------------------
# Loss terms
# ---------------------------------------------------------------------------
def detection_margin_loss(gap: torch.Tensor, margin: float = DETECTION_MARGIN) -> torch.Tensor:
    return F.relu(margin - gap).mean()


def topk_margin_loss(gap: torch.Tensor, margin: float = DETECTION_MARGIN,
                     fraction: float = TOPK_FRACTION) -> torch.Tensor:
    """Hinge over the hardest ``fraction`` of the batch (smallest gaps)."""
    k = max(1, int(round(fraction * gap.numel())))
    hardest = torch.topk(gap, k, largest=False).values
    return F.relu(margin - hardest).mean()


def intact_compact_loss(gap0: torch.Tensor, margin: float = INTACT_MARGIN) -> torch.Tensor:
    """Keep the intact cluster tight around its own mean barcode."""
    return F.relu(gap0 - margin).mean()


def class_centroid_loss(det: torch.Tensor, labels: torch.Tensor, ref_norm: torch.Tensor,
                        margin: float = CLASS_CENTROID_MARGIN,
                        n_classes: int = len(config.CLASS_NAMES)) -> torch.Tensor:
    """Push the per-class mean barcodes apart (in units of the intact norm)."""
    centroids = []
    for c in range(n_classes):
        sel = labels == c
        if sel.any():
            centroids.append(det[sel].mean(dim=0))
    if len(centroids) < 2:
        return det.new_zeros(())
    centroids = torch.stack(centroids)
    dist = torch.cdist(centroids, centroids) / ref_norm.clamp_min(1e-12)
    n = centroids.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    return F.relu(margin - dist[iu[0], iu[1]]).mean()


def power_floor_loss(det: torch.Tensor, det0: torch.Tensor, power_floor: float) -> torch.Tensor:
    """Hinge keeping total detected power above the floor (2D notebook port)."""
    p = det.sum(dim=1)
    p0 = det0.sum(dim=1)
    return (F.relu(power_floor - p) / power_floor).mean() + (F.relu(power_floor - p0) / power_floor).mean()


def tv_loss(param_map: torch.Tensor) -> torch.Tensor:
    """Total-variation smoothness on the metasurface map (fabricability)."""
    dx = (param_map[1:, :] - param_map[:-1, :]).abs().mean()
    dy = (param_map[:, 1:] - param_map[:, :-1]).abs().mean()
    return dx + dy


def combined_loss(
    det: torch.Tensor,          # (B, N_det) defect-sample soft powers
    labels: torch.Tensor,       # (B,)
    det0: torch.Tensor,         # (B0, N_det) intact-sample soft powers
    logits: torch.Tensor,       # (B, n_classes)
    surface_map: torch.Tensor | None,  # phase or w_pillar map for the TV term (None: no term)
    power_floor: float,
) -> tuple[torch.Tensor, dict]:
    """Full objective; returns (loss, dict of detached components)."""
    d_ref = det0.mean(dim=0)
    ref_norm = d_ref.norm()
    gap = relative_l2_gap(det, d_ref)
    gap0 = relative_l2_gap(det0, d_ref)

    terms = {
        "detect": detection_margin_loss(gap),
        "intact": W_INTACT * intact_compact_loss(gap0),
        "class": W_CLASS * F.cross_entropy(logits, labels),
        "power": W_POWER * power_floor_loss(det, det0, power_floor),
        "topk": W_TOPK * topk_margin_loss(gap),
        "centroid": W_CENTROID * class_centroid_loss(det, labels, ref_norm),
        "tv": W_TV * tv_loss(surface_map) if surface_map is not None else det.new_zeros(()),
    }
    loss = sum(terms.values())
    logs = {k: float(v.detach()) for k, v in terms.items()}
    logs["gap_mean"] = float(gap.mean().detach())
    logs["gap0_mean"] = float(gap0.mean().detach())
    return loss, logs


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@torch.no_grad()
def detection_metrics(gap: torch.Tensor, gap0: torch.Tensor,
                      margin: float = DETECTION_MARGIN) -> dict:
    return {
        "pass_rate": float((gap > margin).float().mean()),
        "false_alarm": float((gap0 > margin).float().mean()),
        "gap_p5": float(torch.quantile(gap, 0.05)),
        "gap_mean": float(gap.mean()),
        "gap0_mean": float(gap0.mean()),
    }


@torch.no_grad()
def confusion_matrix(pred: torch.Tensor, labels: torch.Tensor,
                     n_classes: int = len(config.CLASS_NAMES)) -> torch.Tensor:
    cm = torch.zeros(n_classes, n_classes, dtype=torch.long)
    for t, p in zip(labels.cpu(), pred.cpu()):
        cm[t, p] += 1
    return cm


@torch.no_grad()
def auc_score(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """All-pairs AUC with ties counted 0.5 (no-MS notebook port)."""
    pos = pos.reshape(-1, 1)
    neg = neg.reshape(1, -1)
    greater = (pos > neg).float().sum()
    ties = (pos == neg).float().sum()
    return float((greater + 0.5 * ties) / (pos.numel() * neg.numel()))


@torch.no_grad()
def tpr_at_fpr(pos: torch.Tensor, neg: torch.Tensor, fpr: float = 0.01) -> float:
    threshold = torch.quantile(neg, 1 - fpr)
    return float((pos > threshold).float().mean())


@torch.no_grad()
def roc_points(pos: torch.Tensor, neg: torch.Tensor, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    scores = torch.cat([pos, neg])
    # q must live on the same device as the input (a bare .to(dtype) leaves it on CPU)
    q = torch.linspace(0, 1, n, device=scores.device, dtype=scores.dtype)
    thresholds = torch.quantile(scores, q)
    tpr = [(pos > t).float().mean().item() for t in thresholds]
    fpr = [(neg > t).float().mean().item() for t in thresholds]
    return np.array(fpr), np.array(tpr)
