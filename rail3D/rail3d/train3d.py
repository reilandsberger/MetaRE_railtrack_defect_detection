"""Training loop for the 3D rail ONN: detection + classification with
interruption-safe checkpointing (shared-workstation friendly).

Design notes
  * one function ``train()`` drives both the smoke test (V8) and the full
    runs; notebooks stay thin;
  * all randomness flows through the global torch CPU/CUDA RNGs whose states
    are checkpointed, so a killed run resumes bit-identically;
  * soft (differentiable, jittered) detector masks train; hard binary masks
    produce every reported metric;
  * detector pruning (Face3D variance criterion) shrinks 18 -> N_DET_FINAL
    inside a window; the optimizer is rebuilt after each prune.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import config, data3d, losses3d, optics3d


@dataclass
class TrainConfig:
    run_name: str = "run"
    surface: str = "slm"              # "slm" | "metaunit"
    mode: str = "tot"                 # field mode: "tot" | "sca"
    # The tau anneal (0..tau_anneal_end) and the pruning window make the
    # objective NON-STATIONARY for the first ~250 epochs. At the old default of
    # 400 that left only ~150 epochs of stationary fine-tuning at the final
    # detector count and mask sharpness -- too short to settle ~1800 metasurface
    # parameters. 1200 leaves ~950. Epochs are cheap (~0.5 s on the 5090), so
    # check history["best_epoch"]: if it lands in the last 10% of the run, the
    # model was still improving and n_epoch should go up again.
    n_epoch: int = 1200
    batch_size: int = 256
    b0: int = 64                      # intact reference batch per step
    lr: float = 5e-4
    lr_head: float = 1e-3
    n_layer: int = 1
    layer_distances: tuple = config.LAYER_DISTANCES
    # Pruning window (epochs) and schedule. "fraction" keeps a fixed ratio at
    # each prune step, so a dense start (100+ detectors) reaches a handful in a
    # bounded number of steps: n_k = n_0 * prune_keep^k. The legacy "fixed"
    # schedule removes prune_per_step each time (fine from an 18-detector start,
    # but would need 300+ epochs from 130).
    prune_start: int = 60
    prune_end: int = 150
    prune_every: int = 5
    prune_schedule: str = "fraction"  # "fraction" | "fixed"
    prune_keep: float = 0.75          # fraction retained per prune step
    prune_per_step: int = 2           # used only by the "fixed" schedule
    n_det_final: int = config.N_DET_FINAL
    det_grid: tuple | None = None     # override config.DET_GRID (dense start)
    # tau anneal: tau_scale from 1/4 to 1/16 over [0, tau_anneal_end]
    tau_start: float = 0.25
    tau_end: float = 1 / 16
    tau_anneal_end: int = 250
    checkpoint_every: int = 10
    seed: int = config.SEED
    noise: bool = True
    data_root: str | None = None      # None -> full dataset; path -> e.g. smoke dir
    # objective (rev.2): "rank" = pairwise soft-AUC on normalized barcodes with a
    # calibrated operating threshold; "margin" = legacy absolute-margin loss.
    objective: str = "rank"
    metric: str = "cos"               # "cos" | "l2" score for rank/reporting
    target_fpr: float = losses3d.TARGET_FPR


def _ckpt_dir(cfg: TrainConfig) -> Path:
    d = config.CHECKPOINT_DIR / cfg.run_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def tau_for_epoch(cfg: TrainConfig, epoch: int) -> float:
    frac = min(1.0, epoch / max(1, cfg.tau_anneal_end))
    return cfg.tau_start + (cfg.tau_end - cfg.tau_start) * frac


def build_model(cfg: TrainConfig, device: torch.device) -> optics3d.ONN3D:
    detector = None
    if cfg.det_grid is not None:
        detector = optics3d.SoftDetector2D(centers=dense_centers(cfg.det_grid))
    model = optics3d.ONN3D(
        n_layer=cfg.n_layer, layer_distances=cfg.layer_distances,
        surface=cfg.surface, noise=cfg.noise, seed=cfg.seed, detector=detector,
    )
    return model.to(device)


def dense_centers(grid: tuple[int, int]) -> torch.Tensor:
    """Detector centres on a gx x gy lattice spanning the whole aperture.

    Spacing is aperture/(g+1) so windows sit inside the plane rather than on its
    edge; with a dense enough grid the windows tile most of the measurement
    plane, which is the starting point for prune-down experiments.
    """
    gx, gy = grid
    px = config.WX / (gx + 1)
    py = config.WY / (gy + 1)
    cx = (torch.arange(gx) - (gx - 1) / 2) * px
    cy = (torch.arange(gy) - (gy - 1) / 2) * py
    CX, CY = torch.meshgrid(cx, cy, indexing="ij")
    return torch.stack([CX.reshape(-1), CY.reshape(-1)], dim=1)


def prune_count(cfg: TrainConfig, n_now: int) -> int:
    """How many detectors to drop at this prune step."""
    if cfg.prune_schedule == "fixed":
        n_rm = cfg.prune_per_step
    else:
        n_rm = max(1, int(round(n_now * (1.0 - cfg.prune_keep))))
    return min(n_rm, n_now - cfg.n_det_final)


def build_optimizer(cfg: TrainConfig, model: optics3d.ONN3D) -> torch.optim.Adam:
    surf_params = [p for layer in model.layers for p in layer.parameters()]
    return torch.optim.Adam([
        {"params": surf_params + [model.detector.u], "lr": cfg.lr},
        {"params": model.head.parameters(), "lr": cfg.lr_head},
    ])


def surface_map(model: optics3d.ONN3D) -> torch.Tensor | None:
    layer = model.layers[0]
    if hasattr(layer, "phase") and isinstance(layer.phase, nn.Parameter):
        return layer.phase
    if hasattr(layer, "w_pillar"):
        return layer.w_pillar
    return None  # "none" baseline: no TV term


def load_all_data(cfg: TrainConfig, device: torch.device):
    root = Path(cfg.data_root) if cfg.data_root else None
    data3d.check_dataset_config(root)      # refuse fields built for another geometry
    fields, labels, metas, rms = data3d.load_dataset(cfg.mode, root=root)
    intact, _ = data3d.load_intact(cfg.mode, rms=rms, root=root)
    tr, va, te = data3d.stratified_split(labels, seed=cfg.seed)

    gen = torch.Generator().manual_seed(cfg.seed + 1)
    perm0 = torch.randperm(intact.shape[0], generator=gen)
    n0 = intact.shape[0]
    i_tr = perm0[: int(0.8 * n0)]
    i_va = perm0[int(0.8 * n0): int(0.9 * n0)]
    i_te = perm0[int(0.9 * n0):]

    data = {
        "fields": fields.to(device), "labels": labels.to(device),
        "intact": intact.to(device),
        "train": tr.to(device), "val": va.to(device), "test": te.to(device),
        "i_train": i_tr.to(device), "i_val": i_va.to(device), "i_test": i_te.to(device),
    }
    return data


@torch.no_grad()
def _hard_dets(model, fields, idx, chunk=512):
    outs = []
    for s in range(0, len(idx), chunk):
        _, det = model(fields[idx[s: s + chunk]], hard=True)
        outs.append(det)
    return torch.cat(outs, dim=0)


def _classify_input(det: torch.Tensor, d_ref: torch.Tensor, objective: str) -> torch.Tensor:
    if objective == "rank":
        return losses3d.normalize_barcode(det)
    return det / d_ref.norm().clamp_min(1e-12)


@torch.no_grad()
def evaluate(model, data, split="val", cfg: TrainConfig | None = None) -> dict:
    """Split metrics on hard detector powers.

    rank objective (default): score = AUC + class accuracy (both threshold
    independent — checkpoint selection no longer depends on a margin); the
    reported pass_rate/false_alarm use the threshold CALIBRATED at target_fpr
    on this split's intact spread.
    """
    cfg = cfg or TrainConfig()
    model.eval()
    det = _hard_dets(model, data["fields"], data[split])
    det0 = _hard_dets(model, data["intact"], data[f"i_{split}"])
    d_ref = det0.mean(dim=0)
    metric = cfg.metric if cfg.objective == "rank" else "l2"
    gap = losses3d.gap_score(det, d_ref, metric)
    gap0 = losses3d.gap_score(det0, d_ref, metric)

    labels = data["labels"][data[split]]
    logits = model.classify(_classify_input(det, d_ref, cfg.objective))
    class_acc = float((logits.argmax(dim=1) == labels).float().mean())

    auc = losses3d.auc_score(gap, gap0)
    thr = losses3d.calibrated_threshold(gap0, cfg.target_fpr)
    thr_bal, bal_acc = losses3d.best_threshold(gap, gap0)
    metrics = {
        "auc": auc,
        "threshold": thr,
        "pass_rate": float((gap > thr).float().mean()),
        "false_alarm": float((gap0 > thr).float().mean()),
        "balanced_acc": bal_acc,
        "threshold_balanced": thr_bal,
        "gap_p5": float(torch.quantile(gap, 0.05)),
        "gap_mean": float(gap.mean()),
        "gap0_mean": float(gap0.mean()),
        "class_acc": class_acc,
    }
    if cfg.objective == "rank":
        metrics["score"] = auc + class_acc
    else:
        metrics["score"] = metrics["pass_rate"] - metrics["false_alarm"] + class_acc
    return metrics


def _rng_states() -> dict:
    st = {"cpu": torch.get_rng_state(), "numpy": np.random.get_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _restore_rng(st: dict) -> None:
    # checkpoints may be loaded with map_location=cuda; RNG states must be CPU ByteTensors
    torch.set_rng_state(st["cpu"].cpu().to(torch.uint8))
    np.random.set_state(st["numpy"])
    if torch.cuda.is_available() and "cuda" in st:
        torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in st["cuda"]])


def save_checkpoint(path: Path, cfg, model, optimizer, epoch, history, extra) -> None:
    data3d.atomic_save({
        "config": asdict(cfg),
        "epoch": epoch,
        "model": model.state_dict(),
        "n_det": model.detector.n_det,
        "optimizer": optimizer.state_dict(),
        "history": history,
        "rng": _rng_states(),
        **extra,
    }, path)


def _shape_model_to_checkpoint(model: optics3d.ONN3D, state: dict,
                               path: Path | None = None) -> None:
    """Match detector/head shapes to a (possibly pruned) checkpoint state.

    Detector COUNT differences are expected (pruning) and are adopted. A
    different number of CLASSES, or a different metasurface grid, means the
    checkpoint belongs to another experiment entirely — say so plainly instead
    of letting load_state_dict raise a bare size-mismatch.
    """
    sd = state["model"]
    n_cls_ckpt = sd["head.bias"].shape[0]
    n_cls_now = model.head.out_features
    where = f"\n  checkpoint: {path}" if path else ""
    if n_cls_ckpt != n_cls_now:
        raise RuntimeError(
            f"Checkpoint was trained with {n_cls_ckpt} classes, this run has "
            f"{n_cls_now} ({list(config.CLASS_NAMES)}).{where}\n"
            f"  It predates a change to CLASS_NAMES, so it cannot be resumed.\n"
            f"  Delete the run directory to start fresh, or use a new run_name:\n"
            f"    rm -rf {path.parent if path else config.CHECKPOINT_DIR / '<run_name>'}")
    for key in ("layers.0.phase", "layers.0.p"):
        if key in sd and key in model.state_dict():
            want = tuple(model.state_dict()[key].shape)
            got = tuple(sd[key].shape)
            if want != got:
                raise RuntimeError(
                    f"Checkpoint metasurface grid is {got}, this run uses {want}."
                    f"{where}\n  Delete the run directory or use a new run_name.")

    n_det = sd["detector.u"].shape[0]
    if n_det != model.detector.n_det:
        dev = model.detector.u.device
        model.detector.u = nn.Parameter(torch.zeros(n_det, 2, device=dev))
        old = model.head
        model.head = nn.Linear(n_det, old.out_features).to(dev)


def train(cfg: TrainConfig, device: torch.device | None = None,
          verbose: bool = True) -> dict:
    """Train (or resume) a run; returns the final history dict."""
    device = device or config.get_device()
    torch.manual_seed(cfg.seed)

    data = load_all_data(cfg, device)
    model = build_model(cfg, device)
    optimizer = build_optimizer(cfg, model)

    ckpt_dir = _ckpt_dir(cfg)
    latest, best = ckpt_dir / "latest.pt", ckpt_dir / "best.pt"

    history: dict = {"train": [], "val": [], "prune_epochs": [], "best_score": -1e9,
                     "best_epoch": -1}
    start_epoch = 0
    power_floor = None

    if latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        _shape_model_to_checkpoint(model, state, latest)
        model.load_state_dict(state["model"])
        optimizer = build_optimizer(cfg, model)
        optimizer.load_state_dict(state["optimizer"])
        history = state["history"]
        start_epoch = state["epoch"] + 1
        power_floor = state["power_floor"]
        _restore_rng(state["rng"])
        if verbose:
            print(f"[resume] {cfg.run_name} at epoch {start_epoch} (n_det={model.detector.n_det})")

    if power_floor is None:
        model.eval()
        with torch.no_grad():
            det0 = _hard_dets(model, data["intact"], data["i_train"])
        power_floor = float(2.0 * det0.sum(dim=1).mean())

    n_train = len(data["train"])
    for epoch in range(start_epoch, cfg.n_epoch):
        t0 = time.time()
        model.detector.tau_scale = tau_for_epoch(cfg, epoch)
        model.train()

        perm = data["train"][torch.randperm(n_train, device=device)]
        ep_logs = []
        for s in range(0, n_train, cfg.batch_size):
            idx = perm[s: s + cfg.batch_size]
            i0 = data["i_train"][torch.randint(len(data["i_train"]), (cfg.b0,), device=device)]

            psi = torch.cat([data["fields"][idx], data["intact"][i0]], dim=0)
            _, det_all = model(psi)
            det_d, det_0 = det_all[: len(idx)], det_all[len(idx):]
            d_ref = det_0.mean(dim=0)
            logits = model.classify(_classify_input(det_d, d_ref, cfg.objective))

            loss, logs = losses3d.combined_loss(
                det_d, data["labels"][idx], det_0, logits, surface_map(model), power_floor,
                objective=cfg.objective, metric=cfg.metric)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            logs["loss"] = float(loss.detach())
            ep_logs.append(logs)

        history["train"].append({k: float(np.mean([l[k] for l in ep_logs]))
                                 for k in ep_logs[0]})

        # pruning window
        if (cfg.prune_start <= epoch <= cfg.prune_end
                and (epoch - cfg.prune_start) % cfg.prune_every == 0
                and model.detector.n_det > cfg.n_det_final):
            model.eval()
            det_all = torch.cat([
                _hard_dets(model, data["fields"], data["val"]),
                _hard_dets(model, data["intact"], data["i_val"]),
            ], dim=0)
            n_rm = prune_count(cfg, model.detector.n_det)
            model.prune_detectors(det_all, n_rm)
            optimizer = build_optimizer(cfg, model)
            history["prune_epochs"].append({"epoch": epoch, "n_det": model.detector.n_det})

        val = evaluate(model, data, "val", cfg)
        val["epoch"] = epoch
        val["seconds"] = round(time.time() - t0, 2)
        history["val"].append(val)

        if val["score"] > history["best_score"]:
            history["best_score"] = val["score"]
            history["best_epoch"] = epoch
            save_checkpoint(best, cfg, model, optimizer, epoch, history,
                            {"power_floor": power_floor})
        if (epoch + 1) % cfg.checkpoint_every == 0 or epoch == cfg.n_epoch - 1:
            save_checkpoint(latest, cfg, model, optimizer, epoch, history,
                            {"power_floor": power_floor})

        if verbose:
            print(f"ep {epoch:4d}  loss {history['train'][-1]['loss']:.4f}  "
                  f"auc {val['auc']:.3f}  pass@cal {val['pass_rate']:.3f}  "
                  f"FA {val['false_alarm']:.3f}  acc {val['class_acc']:.3f}  "
                  f"thr {val['threshold']:.3f}  Ndet {model.detector.n_det}  "
                  f"tau {model.detector.tau_scale:.3f}  {val['seconds']:.1f}s")

    save_checkpoint(latest, cfg, model, optimizer, cfg.n_epoch - 1, history,
                    {"power_floor": power_floor})

    # convergence hint: was the run still improving when it stopped?
    be, ne = history["best_epoch"], cfg.n_epoch
    stationary = ne - max(cfg.tau_anneal_end, cfg.prune_end)
    if verbose:
        print(f"\nbest epoch {be}/{ne}  ({100 * be / max(ne, 1):.0f}% of the run); "
              f"{stationary} epochs were stationary "
              f"(after tau anneal {cfg.tau_anneal_end} and pruning {cfg.prune_end})")
        if be > 0.9 * ne:
            print("  -> still improving at the end: increase n_epoch")
        elif stationary < 0.4 * ne:
            print("  -> most of the run was non-stationary: increase n_epoch, or "
                  "shorten tau_anneal_end / prune_end")
    return history


def load_trained(cfg: TrainConfig, device: torch.device, which: str = "best"):
    """Rebuild a trained model from a checkpoint. Returns (model, state)."""
    path = _ckpt_dir(cfg) / f"{which}.pt"
    state = torch.load(path, map_location=device, weights_only=False)
    model = build_model(cfg, device)
    _shape_model_to_checkpoint(model, state, path)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state


# ---------------------------------------------------------------------------
# Full evaluation for the notebooks (test split + robustness curves)
# ---------------------------------------------------------------------------
@torch.no_grad()
def full_evaluation(model, data, cfg: TrainConfig | None = None,
                    noise_levels=(0.0, 0.02, 0.05, 0.10, 0.15),
                    shifts_mm=np.linspace(-10, 10, 21)) -> dict:
    """Test metrics + ROC/AUC + noise & alignment robustness curves.

    The operating threshold is calibrated on the VALIDATION intact spread
    (never on test) and applied to every curve.
    """
    cfg = cfg or TrainConfig()
    metric = cfg.metric if cfg.objective == "rank" else "l2"
    model.eval()
    out = {"test": evaluate(model, data, "test", cfg)}

    det = _hard_dets(model, data["fields"], data["test"])
    det0 = _hard_dets(model, data["intact"], data["i_test"])
    det0_val = _hard_dets(model, data["intact"], data["i_val"])
    d_ref = det0.mean(dim=0)
    labels = data["labels"][data["test"]]
    thr = losses3d.calibrated_threshold(
        losses3d.gap_score(det0_val, det0_val.mean(dim=0), metric), cfg.target_fpr)
    out["threshold_from_val"] = thr

    # confusion
    logits = model.classify(_classify_input(det, d_ref, cfg.objective))
    out["confusion"] = losses3d.confusion_matrix(logits.argmax(dim=1), labels)

    # ROC (5% complex noise, negatives = noisy intact) — no-MS notebook port
    def noisy(fields, sigma):
        scale = fields.abs().pow(2).mean().sqrt()
        return fields + sigma * scale * (
            torch.randn_like(fields.real) + 1j * torch.randn_like(fields.real))

    pos_f = noisy(data["fields"][data["test"]], 0.05)
    neg_f = noisy(data["intact"][data["i_test"]], 0.05)
    det_p = torch.cat([model(pos_f[s:s+512], hard=True)[1] for s in range(0, len(pos_f), 512)])
    det_n = torch.cat([model(neg_f[s:s+512], hard=True)[1] for s in range(0, len(neg_f), 512)])
    pos = losses3d.gap_score(det_p, d_ref, metric)
    neg = losses3d.gap_score(det_n, d_ref, metric)
    out["roc"] = {
        "auc": losses3d.auc_score(pos, neg),
        "tpr_at_1pct_fpr": losses3d.tpr_at_fpr(pos, neg, 0.01),
        "points": losses3d.roc_points(pos, neg),
        "auc_linf": losses3d.auc_score(losses3d.linf_gap(det_p, d_ref),
                                       losses3d.linf_gap(det_n, d_ref)),
    }
    per_class_auc = {}
    for c, name in enumerate(config.CLASS_NAMES):
        sel = labels == c
        if sel.any():
            per_class_auc[name] = losses3d.auc_score(pos[sel], neg)
    out["roc"]["per_class_auc"] = per_class_auc

    # noise robustness
    curve = []
    for sig in noise_levels:
        dp = torch.cat([model(noisy(data["fields"][data["test"]], sig)[s:s+512], hard=True)[1]
                        for s in range(0, len(data["test"]), 512)])
        dn = torch.cat([model(noisy(data["intact"][data["i_test"]], sig)[s:s+512], hard=True)[1]
                        for s in range(0, len(data["i_test"]), 512)])
        g = losses3d.gap_score(dp, d_ref, metric)
        g0 = losses3d.gap_score(dn, d_ref, metric)
        lg = model.classify(_classify_input(dp, d_ref, cfg.objective))
        curve.append({"sigma": float(sig),
                      "pass_rate": float((g > thr).float().mean()),
                      "false_alarm": float((g0 > thr).float().mean()),
                      "class_acc": float((lg.argmax(dim=1) == labels).float().mean())})
    out["noise_curve"] = curve

    # alignment robustness (rigid detector-plane shift, x direction)
    align = []
    for dx_mm in shifts_mm:
        det_s = torch.cat([
            model.detector.hard_powers(model.propagate(data["fields"][data["test"]][s:s+512]).abs()**2,
                                       shift=(float(dx_mm), 0.0))
            for s in range(0, len(data["test"]), 512)])
        det0_s = torch.cat([
            model.detector.hard_powers(model.propagate(data["intact"][data["i_test"]][s:s+512]).abs()**2,
                                       shift=(float(dx_mm), 0.0))
            for s in range(0, len(data["i_test"]), 512)])
        d_ref_s = det0_s.mean(dim=0)
        g = losses3d.gap_score(det_s, d_ref_s, metric)
        align.append({"shift_mm": float(dx_mm),
                      "pass_rate": float((g > thr).float().mean())})
    out["alignment_curve"] = align
    return out
