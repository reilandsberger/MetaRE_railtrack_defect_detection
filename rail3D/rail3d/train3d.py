"""Training loop for the 3D rail ONN: detection + classification with
interruption-safe checkpointing (shared-workstation friendly).

Design notes
  * one function ``train()`` drives both the smoke test (V8) and the full
    runs; notebooks stay thin;
  * all randomness flows through the global torch CPU/CUDA RNGs whose states
    are checkpointed, so a killed run resumes bit-identically;
  * soft (differentiable, jittered) detector masks train; hard binary masks
    produce every reported metric;
  * detector pruning (redundancy criterion by default) shrinks the dense
    config.DET_GRID start down to N_DET_FINAL inside a window; the optimizer
    is rebuilt and the power floor recomputed after each prune.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, fields as _dc_fields, asdict
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
    # "redundancy" values each detector by its UNIQUE contribution -- required
    # for a dense start where overlapping windows have near-identical variance.
    # "variance" is the Face3D criterion, valid only for sparse layouts.
    prune_criterion: str = "redundancy"
    prune_per_step: int = 2           # used only by the "fixed" schedule
    n_det_final: int = config.N_DET_FINAL
    det_grid: tuple | None = None     # override config.DET_GRID (dense start)
    # tau anneal: tau_scale from 1/4 to 1/8 over [0, tau_anneal_end]
    tau_start: float = 0.25
    # Do NOT anneal tau below the pixel pitch: below the grid pitch the soft
    # mask cannot represent sub-pixel motion and detector-position gradients
    # stop being meaningful (the old w/16 froze positions mid-anneal). The
    # per-axis floor of 0.5*dx in SoftDetector2D._axis_soft enforces this for
    # BOTH axes — tau_scale*w on the short (y) window axis is below the floor
    # by design, so the floor, not this value, governs y near the anneal end.
    tau_end: float = 1 / 8
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
    # weight of the captured-power reward (recorded in checkpoints so runs
    # remain attributable); 0.0 reproduces the pre-capture objective exactly.
    w_capture: float = losses3d.W_CAPTURE
    # Phase std (rad) at epoch 0 for surface="slm". pi/2 is a random diffuser;
    # 0.0 starts AT the no-metasurface baseline, which is inside this model's
    # hypothesis space exactly (README finding 24). Provenance-free: it changes
    # only the optimizer's starting point, so runs remain comparable in
    # geometry, but NOT in initialisation -- record it when comparing scores.
    slm_init_std: float = float(np.pi / 2)


def _ckpt_dir(cfg: TrainConfig) -> Path:
    d = config.CHECKPOINT_DIR / cfg.run_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def tau_for_epoch(cfg: TrainConfig, epoch: int) -> float:
    frac = min(1.0, epoch / max(1, cfg.tau_anneal_end))
    return cfg.tau_start + (cfg.tau_end - cfg.tau_start) * frac


def build_model(cfg: TrainConfig, device: torch.device) -> optics3d.ONN3D:
    # default layout is the dense lattice from config.DET_GRID; cfg.det_grid
    # overrides it per run
    detector = optics3d.SoftDetector2D(
        centers=dense_centers(cfg.det_grid or config.DET_GRID))
    model = optics3d.ONN3D(
        n_layer=cfg.n_layer, layer_distances=cfg.layer_distances,
        surface=cfg.surface, noise=cfg.noise, seed=cfg.seed, detector=detector,
        slm_init_std=cfg.slm_init_std,
    )
    return model.to(device)


def dense_centers(grid: tuple[int, int]) -> torch.Tensor:
    """Thin wrapper kept for API stability — see config.dense_detector_centers,
    the single source for every detector lattice."""
    return config.dense_detector_centers(grid=grid)


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


def load_all_data(cfg: TrainConfig, device: torch.device, verbose: bool = True):
    root = Path(cfg.data_root) if cfg.data_root else None
    data3d.check_dataset_config(root)      # refuse fields built for another geometry
    if verbose:
        # always state WHICH dataset is being trained on, and how old it is
        print(data3d.describe_dataset(root))
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


def checkpoint_geometry(cfg: TrainConfig) -> dict:
    """The geometry stamp stored in every checkpoint (and verified on load)."""
    return {
        "WVL": config.WVL, "DX": config.DX, "NX": config.NX, "NY": config.NY,
        "H_MS": config.H_MS, "DET_SIZE": tuple(config.DET_SIZE),
        "CLASS_NAMES": list(config.CLASS_NAMES),
        "LAYER_DISTANCES": tuple(cfg.layer_distances),
        "DET_GRID": tuple(cfg.det_grid or config.DET_GRID),
    }


def verify_checkpoint_geometry(state: dict, cfg: TrainConfig,
                               path: Path | None = None) -> None:
    """Refuse a checkpoint stamped with a different geometry.

    Without this, a λ=8 checkpoint resumes silently into a λ=5 run whenever the
    grid shape is unchanged (60x30 at both) — the phase map, head and detector
    u are all shape-compatible while every physical length is wrong. Warn-only
    when the stamp is absent (pre-2026-08-17 checkpoints).
    """
    stored = state.get("geometry")
    where = f"\n  checkpoint: {path}" if path else ""
    if stored is None:
        print("[rail3d] WARNING: checkpoint has no geometry stamp "
              "(pre-2026-08-17) — cannot verify its wavelength/geometry "
              "against the current config" + where)
        return
    now = checkpoint_geometry(cfg)
    diffs = [f"  {k}: checkpoint={stored.get(k)!r}  current={v!r}"
             for k, v in now.items()
             if not data3d._values_equal(data3d._jsonable(stored.get(k)),
                                         data3d._jsonable(v))]
    if diffs:
        raise RuntimeError(
            "Checkpoint was trained with a DIFFERENT geometry:\n"
            + "\n".join(diffs) + where
            + f"\n  Delete the run directory or use a new run_name:\n"
              f"    rm -rf {path.parent if path else config.CHECKPOINT_DIR / '<run_name>'}")


def save_checkpoint(path: Path, cfg, model, optimizer, epoch, history, extra) -> None:
    data3d.atomic_save({
        "config": asdict(cfg),
        "geometry": checkpoint_geometry(cfg),
        "epoch": epoch,
        "model": model.state_dict(),
        "n_det": model.detector.n_det,
        "optimizer": optimizer.state_dict(),
        "history": history,
        "rng": _rng_states(),
        **extra,
    }, path)


def _shape_model_to_checkpoint(model: optics3d.ONN3D, state: dict,
                               path: Path | None = None,
                               cfg: TrainConfig | None = None) -> None:
    """Match detector/head shapes to a (possibly pruned) checkpoint state.

    Detector COUNT differences are expected (pruning) and are adopted. A
    different number of CLASSES, a different metasurface grid, or a different
    geometry stamp means the checkpoint belongs to another experiment
    entirely — say so plainly instead of letting load_state_dict raise a bare
    size-mismatch (or worse, load cleanly with wrong physics).
    """
    if cfg is not None:
        verify_checkpoint_geometry(state, cfg, path)
    sd = state["model"]
    # Checkpoints written before 2026-08-17 persisted the detector's geometry
    # buffers; they are config-derived (persistent=False now), so drop them
    # rather than letting them overwrite the freshly built grid coordinates.
    for legacy in ("detector.grid_x", "detector.grid_y", "detector.half_buf"):
        sd.pop(legacy, None)
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


def _check_resume_config(cfg: TrainConfig, state: dict, path: Path,
                         verbose: bool = True) -> None:
    """Compare the resuming config against the checkpoint's stored one.

    surface / layer_distances / objective silently change the physics or the
    loss shape (the propagator kernels are non-persistent buffers, so nothing
    else would notice) -> raise. Everything except the designed-to-change
    fields (n_epoch extension, run_name, checkpoint_every, verbosity of
    schedule) -> warn, so a drifted flag is at least visible. Prints only —
    no RNG is consumed, so resume bit-identity is unaffected.
    """
    stored = state.get("config")
    if not stored:
        return
    hard = ("surface", "layer_distances", "objective", "n_layer", "mode")
    soft_skip = {"n_epoch", "run_name", "checkpoint_every", "data_root"}
    now = asdict(cfg)
    norm = lambda v: list(v) if isinstance(v, tuple) else v
    bad = [k for k in hard if k in stored and norm(stored[k]) != norm(now[k])]
    if bad:
        raise RuntimeError(
            "Resuming with a DIFFERENT config than the checkpoint was trained "
            "with:\n" + "\n".join(
                f"  {k}: checkpoint={stored[k]!r}  now={now[k]!r}" for k in bad)
            + f"\n  checkpoint: {path}\n  These change the physics/objective "
              f"mid-run. Use a new run_name, or restore the values above.")
    if verbose:
        for k, v in now.items():
            if (k in stored and k not in soft_skip and k not in hard
                    and norm(stored[k]) != norm(v)):
                print(f"[resume] note: {k} differs from the checkpoint "
                      f"({stored[k]!r} -> {v!r})")


def train(cfg: TrainConfig, device: torch.device | None = None,
          verbose: bool = True) -> dict:
    """Train (or resume) a run; returns the final history dict."""
    device = device or config.get_device()
    torch.manual_seed(cfg.seed)

    if verbose:
        print("=" * 72)
        print(f"run     : {cfg.run_name}   surface={cfg.surface}  objective={cfg.objective}"
              f"/{cfg.metric}  epochs={cfg.n_epoch}  batch={cfg.batch_size}")
        print(f"device  : {device}"
              + (f" ({torch.cuda.get_device_name(device.index or 0)})"
                 if device.type == "cuda" else ""))
        print(f"detectors: start {cfg.det_grid or config.DET_GRID} -> "
              f"{cfg.n_det_final} final, {cfg.prune_schedule} schedule "
              f"(prune {cfg.prune_start}-{cfg.prune_end})")
        print(f"optics  : {cfg.n_layer} layer(s), distances {cfg.layer_distances} mm")
    data = load_all_data(cfg, device, verbose=verbose)
    if verbose:
        print(f"splits  : train {len(data['train'])}  val {len(data['val'])}  "
              f"test {len(data['test'])}   intact {len(data['i_train'])}/"
              f"{len(data['i_val'])}/{len(data['i_test'])}")
        # An EPOCH is a full pass over the training split, so its cost scales
        # with dataset size -- but every schedule constant below (tau_anneal_end,
        # prune_start/end) is expressed in epochs and was tuned on the SMOKE set,
        # where train is one batch and 1 epoch == 1 optimizer step. On a real
        # dataset an epoch is many steps, so n_epoch=1200 silently becomes 7-16x
        # the intended optimisation and a 10-hour run. Print the real budget.
        spe = max(1, -(-len(data["train"]) // cfg.batch_size))
        total = spe * cfg.n_epoch
        print(f"budget  : {spe} step(s)/epoch x {cfg.n_epoch} epochs = {total} "
              f"optimizer steps, {cfg.batch_size}+{cfg.b0} samples/step")
        if spe > 1 and total > 3000:
            print(f"          ! the epoch-valued schedule (tau_anneal_end="
                  f"{cfg.tau_anneal_end}, prune {cfg.prune_start}-{cfg.prune_end}) "
                  f"was calibrated at 1 step/epoch, so this is ~{total // 1200}x "
                  f"that budget. Consider scaling n_epoch down by ~{spe}x and "
                  f"compressing the schedule to match -- see README finding 21.")
        print("=" * 72)
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
        _shape_model_to_checkpoint(model, state, latest, cfg=cfg)
        _check_resume_config(cfg, state, latest, verbose=verbose)
        model.load_state_dict(state["model"])
        optimizer = build_optimizer(cfg, model)
        optimizer.load_state_dict(state["optimizer"])
        history = state["history"]
        start_epoch = state["epoch"] + 1
        power_floor = state.get("power_floor")
        if power_floor is None and verbose:
            print("[resume] WARNING: checkpoint has no power_floor (pre-2026-08); "
                  "recomputing from the current model — bit-identity with the "
                  "original run is not guaranteed for this run")
        _restore_rng(state["rng"])
        if verbose:
            print(f"[resume] {cfg.run_name} at epoch {start_epoch} (n_det={model.detector.n_det})")

    if power_floor is None:
        # Initial floor from the TRAIN intact pool at the dense start (the
        # pre-prune legacy behaviour — keeps fresh runs comparable). After each
        # prune the floor is recomputed from the VAL intact pool; do not unify
        # the two without re-baselining V8.
        model.eval()
        with torch.no_grad():
            det0 = _hard_dets(model, data["intact"], data["i_train"])
        power_floor = float(losses3d.POWER_FLOOR_RATIO * det0.sum(dim=1).mean())

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
            intensity, det_all = model(psi)
            # total power on the plane, for the capture (concentration) term
            total_power = (model.detector.dx ** 2) * intensity.sum(dim=(1, 2))[: len(idx)]
            det_d, det_0 = det_all[: len(idx)], det_all[len(idx):]
            d_ref = det_0.mean(dim=0)
            logits = model.classify(_classify_input(det_d, d_ref, cfg.objective))

            loss, logs = losses3d.combined_loss(
                det_d, data["labels"][idx], det_0, logits, surface_map(model), power_floor,
                objective=cfg.objective, metric=cfg.metric, total_power=total_power,
                w_capture=cfg.w_capture)
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
            keep = model.prune_detectors(det_all, n_rm, criterion=cfg.prune_criterion)
            optimizer = build_optimizer(cfg, model)
            # dedupe by epoch: a resume from a pre-prune checkpoint REPLAYS this
            # prune, and a duplicate entry would corrupt the keep-index
            # composition that maps surviving detectors back to lattice sites
            history["prune_epochs"] = [e for e in history["prune_epochs"]
                                       if e["epoch"] != epoch]
            history["prune_epochs"].append(
                {"epoch": epoch, "n_det": model.detector.n_det,
                 "keep": keep.tolist()})
            # The floor was set by the dense start; after pruning to a few
            # windows it would sit permanently saturated (a constant "maximise
            # power" pull that double-counts the capture term). Recompute from
            # the pruned model. Hard masks in eval mode consume no RNG, so a
            # killed-and-resumed run replays this identically (V8 guards it).
            det0_new = _hard_dets(model, data["intact"], data["i_val"])
            power_floor = float(losses3d.POWER_FLOOR_RATIO * det0_new.sum(dim=1).mean())

        val = evaluate(model, data, "val", cfg)
        val["epoch"] = epoch
        val["seconds"] = round(time.time() - t0, 2)
        if verbose and epoch == start_epoch:
            # measured, not estimated: the first completed epoch is the only
            # honest basis for "how long will this take"
            rem = (cfg.n_epoch - epoch - 1) * val["seconds"]
            print(f"        first epoch took {val['seconds']:.1f} s -> "
                  f"~{rem / 3600:.1f} h for the remaining "
                  f"{cfg.n_epoch - epoch - 1} epochs. Ctrl-C is safe: "
                  f"checkpoints every {cfg.checkpoint_every} epochs resume "
                  f"bit-identically.")
        val["n_det"] = model.detector.n_det
        history["val"].append(val)

        # Only a model that MEETS THE DETECTOR BUDGET is a candidate for "best".
        # score = auc + class_acc, and both improve with more detectors, so an
        # unrestricted argmax reliably picks a PRE-PRUNE epoch: measured
        # 2026-09-11, best_epoch 28 of 300 with prune_start=25 saved a
        # 130-detector model, and the whole 130 -> n_det_final exercise was
        # discarded at selection time. Everything downstream (analyze_results
        # --which best, the notebook comparison table) then described the dense
        # array rather than the system being designed.
        # Before any final-count epoch exists, track the best so far so that a
        # short or interrupted run still has a checkpoint; the first final-count
        # epoch resets the baseline so only budget-meeting models compete.
        at_final = val["n_det"] == cfg.n_det_final
        if at_final and not history.get("best_at_final"):
            history["best_at_final"] = True
            history["best_score"] = -1e9
        if at_final or not history.get("best_at_final"):
            if val["score"] > history["best_score"]:
                history["best_score"] = val["score"]
                history["best_epoch"] = epoch
                history["best_n_det"] = val["n_det"]
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
        cap = history["train"][-1].get("capture_frac") if history["train"] else None
        print(f"\ndetectors: {model.detector.n_det} kept "
              f"(criterion={cfg.prune_criterion}), min separation "
              f"{model.detector.min_separation():.1f} mm"
              + (f", captured power {100 * cap:.1f}%" if cap is not None else ""))
        if model.detector.min_separation() < config.DX:
            print("  -> WARNING: centres closer than one pixel; detectors have "
                  "effectively collapsed onto the same spot")
        print(f"best epoch {be}/{ne}  ({100 * be / max(ne, 1):.0f}% of the run); "
              f"{stationary} epochs were stationary "
              f"(after tau anneal {cfg.tau_anneal_end} and pruning {cfg.prune_end})")
        if be > 0.9 * ne:
            print("  -> still improving at the end: increase n_epoch")
        elif stationary < 0.4 * ne:
            print("  -> most of the run was non-stationary: increase n_epoch, or "
                  "shorten tau_anneal_end / prune_end")
    return history


def effective_config(base_cfg: TrainConfig, state: dict) -> TrainConfig:
    """The TrainConfig a checkpoint was actually trained with.

    Rebuilds from ``state["config"]`` filtered to the current dataclass fields
    (fields added since the run take their defaults), keeping run_name and
    data_root from the caller. Use this before scoring a loaded run — scoring a
    margin-objective checkpoint with the default rank/cos config silently
    mis-reports it.
    """
    stored = dict(state.get("config") or {})
    known = {f.name for f in _dc_fields(TrainConfig)}
    kept = {k: v for k, v in stored.items() if k in known}
    if isinstance(kept.get("layer_distances"), list):
        kept["layer_distances"] = tuple(kept["layer_distances"])
    if isinstance(kept.get("det_grid"), list):
        kept["det_grid"] = tuple(kept["det_grid"])
    kept["run_name"] = base_cfg.run_name
    kept["data_root"] = base_cfg.data_root
    return TrainConfig(**kept)


def load_trained(cfg: TrainConfig, device: torch.device, which: str = "best"):
    """Rebuild a trained model from a checkpoint. Returns (model, state).

    The model is built with the checkpoint's own stored config (surface,
    layer distances, ...), not the caller's guess — pass the returned state to
    ``effective_config`` when you also need the right config for scoring.
    """
    path = _ckpt_dir(cfg) / f"{which}.pt"
    state = torch.load(path, map_location=device, weights_only=False)
    eff = effective_config(cfg, state)
    if eff.surface != cfg.surface and state.get("config"):
        print(f"[rail3d] note: checkpoint {path.name} was trained with "
              f"surface={eff.surface!r} (caller asked {cfg.surface!r}); "
              f"using the checkpoint's")
    model = build_model(eff, device)
    _shape_model_to_checkpoint(model, state, path, cfg=eff)
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
