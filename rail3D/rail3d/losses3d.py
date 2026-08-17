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

DETECTION_MARGIN = 0.40      # legacy absolute margin (objective="margin" only)
INTACT_MARGIN = 0.15         # legacy intact compactness margin
CLASS_CENTROID_MARGIN = 0.25
TOPK_FRACTION = 0.10
# power floor = this ratio x the mean intact detected power; train3d computes
# the initial floor from the dense start and RECOMPUTES it after every prune
# (a floor frozen at the dense start saturates once most windows are gone).
POWER_FLOOR_RATIO = 2.0

W_INTACT = 0.5
W_CLASS = 0.5
W_POWER = 0.2
W_TOPK = 0.3
W_CENTROID = 0.2
W_TV = 1e-4

# --- rank objective (default since rev.2) ---------------------------------
# The first full lab run showed the fixed 0.40 margin is ill-posed against the
# (deliberately kept) ±4 mm placement augmentation: the intact barcode
# DISTRIBUTION spreads wider than the margin, so pass rate and false alarm rise
# together (FA ≈ 0.5 at AUC 0.75-0.9). The rank objective optimizes what we
# actually report — separation of the defect and intact score distributions
# (a differentiable AUC surrogate) — on per-sample-normalized barcodes, and the
# operating threshold is CALIBRATED from the intact validation spread
# (Face3D's margin_opt approach) instead of being hard-coded.
RANK_MARGIN = 0.1            # pairwise hinge margin (cos-gap units)
W_HARDEST = 0.3              # extra weight on the worst 10% of pairs
W_INTACT_RANK = 1.0          # pull the intact cluster tight (mean cos-gap)
CENTROID_MARGIN_N = 0.1      # class-centroid margin on unit-normalized barcodes
TARGET_FPR = 0.05            # default calibration point
# Reward for concentrating light onto the retained detectors. Set to 0.0 to
# reproduce the pre-2026-08-17 objective exactly (runs are not comparable across
# this change). Secondary to the rank term by design: it should improve SNR
# without buying throughput at the cost of class separation.
W_CAPTURE = 0.2


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


def normalize_barcode(det: torch.Tensor) -> torch.Tensor:
    """Per-sample unit L2 normalization — removes the global-amplitude part of
    placement jitter (Face3D normalized its detection vectors the same way)."""
    return det / det.norm(dim=1, keepdim=True).clamp_min(1e-12)


def cos_gap(det: torch.Tensor, det_ref: torch.Tensor) -> torch.Tensor:
    """1 - cosine(barcode, intact reference direction); in [0, 2]."""
    ref = normalize_barcode(det_ref.reshape(1, -1))
    return 1.0 - (normalize_barcode(det) * ref).sum(dim=1)


def gap_score(det: torch.Tensor, det_ref: torch.Tensor, metric: str = "cos") -> torch.Tensor:
    if metric == "cos":
        return cos_gap(det, det_ref)
    if metric == "l2":
        return relative_l2_gap(det, det_ref)
    raise ValueError(f"Unknown metric: {metric}")


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


def capture_fraction(det: torch.Tensor, total_power: torch.Tensor) -> torch.Tensor:
    """Share of the light on the plane that actually lands on the detectors.

    The power floor below is a hinge: it stops the detectors going dark and then
    goes flat, so above the floor nothing pushes the metasurface to route light
    ONTO the few surviving windows. Receiver SNR in hardware depends exactly on
    that, hence this term. At the final count the windows cover only a few
    percent of the plane (8 windows x window-area/dx^2 pixels of the NX*NY
    total), so there is real headroom without focusing.

    Clamped to <=1 per sample: OVERLAPPING windows double-count shared pixels,
    so the raw sum can exceed the true plane power — unclamped, the loss term
    1 - capture would go negative and reward stacking detectors on top of each
    other (nothing else in the loss repels detector centres).
    """
    frac = det.sum(dim=1) / total_power.clamp_min(1e-12)
    return frac.clamp(max=1.0).mean()


def power_floor_loss(det: torch.Tensor, det0: torch.Tensor, power_floor: float) -> torch.Tensor:
    """Hinge keeping total detected power above the floor (2D notebook port)."""
    p = det.sum(dim=1)
    p0 = det0.sum(dim=1)
    return (F.relu(power_floor - p) / power_floor).mean() + (F.relu(power_floor - p0) / power_floor).mean()


def ranking_loss(gap_defect: torch.Tensor, gap_intact: torch.Tensor,
                 margin: float = RANK_MARGIN) -> torch.Tensor:
    """Pairwise hinge over all (defect, intact) pairs — soft-AUC surrogate.

    Zero when every defect score exceeds every intact score by >= margin;
    minimizing it directly maximizes the reported AUC.
    """
    diff = margin + gap_intact.reshape(1, -1) - gap_defect.reshape(-1, 1)
    return F.relu(diff).mean()


def hardest_ranking_loss(gap_defect: torch.Tensor, gap_intact: torch.Tensor,
                         margin: float = RANK_MARGIN,
                         fraction: float = TOPK_FRACTION) -> torch.Tensor:
    """Same hinge, averaged over only the worst ``fraction`` of pairs."""
    diff = F.relu(margin + gap_intact.reshape(1, -1) - gap_defect.reshape(-1, 1))
    k = max(1, int(round(fraction * diff.numel())))
    return torch.topk(diff.reshape(-1), k).values.mean()


@torch.no_grad()
def calibrated_threshold(gap_intact: torch.Tensor, target_fpr: float = TARGET_FPR) -> float:
    """Operating threshold = (1 - target_fpr) quantile of the intact spread."""
    return float(torch.quantile(gap_intact, 1.0 - target_fpr))


@torch.no_grad()
def best_threshold(gap_defect: torch.Tensor, gap_intact: torch.Tensor,
                   n: int = 251) -> tuple[float, float]:
    """Face3D-style balanced sweep: argmin(FN + FP) over n thresholds.

    Returns (threshold, balanced_accuracy). Port of Face3D's
    compute_distance_statistics / margin_opt selection.
    """
    lo = float(min(gap_defect.min(), gap_intact.min()))
    hi = float(max(gap_defect.max(), gap_intact.max()))
    ts = torch.linspace(lo, hi, n, device=gap_defect.device)
    fn = (gap_defect.reshape(-1, 1) <= ts.reshape(1, -1)).float().mean(dim=0)
    fp = (gap_intact.reshape(-1, 1) > ts.reshape(1, -1)).float().mean(dim=0)
    i = int(torch.argmin(fn + fp))
    return float(ts[i]), float(1.0 - 0.5 * (fn[i] + fp[i]))


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
    objective: str = "rank",    # "rank" (default) | "margin" (legacy, pre-rev.2)
    metric: str = "cos",        # score used by the rank objective / reporting
    total_power: torch.Tensor | None = None,  # plane power per sample, for capture
    w_capture: float = W_CAPTURE,             # TrainConfig.w_capture; 0 = legacy objective
) -> tuple[torch.Tensor, dict]:
    """Full objective; returns (loss, dict of detached components).

    objective="margin" reproduces the original absolute-margin loss exactly
    (L2 gaps, fixed 0.40/0.15 margins) as a regression baseline.
    """
    d_ref = det0.mean(dim=0)
    tv = W_TV * tv_loss(surface_map) if surface_map is not None else det.new_zeros(())

    if objective == "margin":
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
            "tv": tv,
        }
    elif objective == "rank":
        gap = gap_score(det, d_ref, metric)
        gap0 = gap_score(det0, d_ref, metric)
        terms = {
            "rank": ranking_loss(gap, gap0),
            "hardest": W_HARDEST * hardest_ranking_loss(gap, gap0),
            "intact": W_INTACT_RANK * gap0.mean(),
            "class": W_CLASS * F.cross_entropy(logits, labels),
            "power": W_POWER * power_floor_loss(det, det0, power_floor),
            "centroid": W_CENTROID * class_centroid_loss(
                normalize_barcode(det), labels, det.new_ones(()),
                margin=CENTROID_MARGIN_N),
            "tv": tv,
        }
    else:
        raise ValueError(f"Unknown objective: {objective}")

    cap = capture_fraction(det, total_power) if total_power is not None else None
    if objective == "rank" and w_capture > 0 and cap is not None:
        # Near-inactive at the dense start (the tiling windows already catch
        # nearly all of the plane); it becomes the operative term after pruning
        # to a handful of windows, which is exactly when receiver SNR matters.
        terms["capture"] = w_capture * (1.0 - cap)

    loss = sum(terms.values())
    logs = {k: float(v.detach()) for k, v in terms.items()}
    logs["loss"] = float(loss.detach())
    logs["gap_mean"] = float(gap.mean().detach())
    logs["gap0_mean"] = float(gap0.mean().detach())
    if cap is not None:
        # the raw fraction, logged separately from the "capture" LOSS term above
        logs["capture_frac"] = float(cap.detach())
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
