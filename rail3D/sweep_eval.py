"""sweep_eval.py -- score every setup in a setup_sweep.py run, bound what a trained
metasurface could add, and prune the setups that cannot win.

    python sweep_eval.py                        # the newest sweep root
    python sweep_eval.py --root data/generated/setup_sweep/F1_xxxxxxxx

The summary (sweep_eval.json), progress.txt and the figures are also copied into the
day's results folder, data/generated/<YYYY-MM-DD>/sweep_<root>/ (--day, else
$RAIL3D_DAY, else today); the fields stay in the sweep root.

THE BOUND SANDWICH (why no training is needed to discard a setup)
A phase-only metasurface is lossless: it redistributes the light the aperture
receives and adds none. For every setup (placement x metasurface crop):
  FLOOR    L8   no metasurface, 8 detectors chosen from the dense lattice, logistic
               regression on normalised detector powers (what the ONN's head sees).
               Zero phase reproduces this exactly (finding 24), so a trained SLM
               with the same 8-detector budget can always reach it.
  CEILING  U    the ideal matched filter on the full complex field at the aperture,
               given the defect's exact signal (its pose twin, stored by the sweep)
               and the intact rails' pose spread plus a receiver-noise floor. No
               metasurface + detectors can beat it (Gaussian pose-noise model; with
               only ~600 intact poses its covariance is underestimated, which makes
               U HIGHER -- the safe direction for pruning).
A setup whose CEILING is below another setup's FLOOR cannot win, whatever it is
trained to. Setups whose intervals overlap the best floor are the ones that need
full training. Both bounds are reported as recall at 5% false alarms over the
priority arc, per type x position cell, and overall.

MEASURED CAVEAT (laptop smoke, 2026-11-05): the ceiling came out ~1.00 for EVERY
setup. An ideal processor with the full complex field rejects the pose motion (a few
degrees of freedom) almost perfectly, so the rigorous ceiling is too loose to prune.
With 12 intact poses that is partly rank deficiency; 600 may tighten it, but do not
count on it. The working rule is therefore EMPIRICAL (NEXT_PHASE.md section 3): rank
by the floor, fully train a small calibration set (today, the top floors, one
mid-rank) to measure what training adds over the floor, and prune the rest only if
that headroom is smaller than the floor gaps between setups.

Also reported, not used for pruning:
  Ldense  the same classifier on ALL dense detectors (more than the ONN may keep)
  margin  the design study's best-window robust recall (DESIGN_STUDY.md section 5)

Data-centric outputs (for the next step, not the pruning): recall per size stratum
(the sweep's Latin-hypercube bins), per finding-34 trust tag, per crack width vs the
2.5 mm cutoff, and a learning curve for the best setup -- is 100 per cell enough?
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rail3d import config, optics3d   # noqa: E402

OUT = config.GENERATED_DIR / "setup_sweep"
FPR = 0.05
ETAS = (1e-3, 1e-2, 1e-1)          # receiver-noise floor for the ceiling, x mean intact pixel power
ETA_PRUNE = 1e-2
N_DET = config.N_DET_FINAL
FOLDS = 5
NEAR_EDGE = 75.0                   # the aperture's horn-side edge for the "extend" family
FAR_FRACTIONS = (0.35, 0.5, 0.65, 0.8, 0.9, 1.1)
SLIDE_WIDTHS = (150.0, 265.0)
SLIDE_STEP = 50.0
Z_FPR = 1.6448536269514722         # Phi^-1(1 - FPR)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def latest_root() -> Path:
    roots = sorted((p for p in OUT.glob("*") if (p / "sweep.json").exists()),
                   key=lambda p: p.stat().st_mtime)
    if not roots:
        raise SystemExit(f"no sweep roots under {OUT}")
    return roots[-1]


def load_placement(root: Path, name: str, n_shards: int | None = None):
    shards = sorted((root / name).glob("shard_*.npz"))
    if n_shards is not None:
        shards = shards[:n_shards]
    if not shards:
        return None
    psi, dl, meta = [], [], []
    for s in shards:
        z = np.load(s)
        psi.append(z["psi"])
        dl.append(z["delta"])
        meta += json.loads(str(z["meta"]))
        x, y = z["x"], z["y"]
    return {"psi": np.concatenate(psi), "delta": np.concatenate(dl), "meta": meta, "x": x, "y": y,
            "n_shards": len(shards)}


# ---------------------------------------------------------------------------
# crops
# ---------------------------------------------------------------------------
def crops(pl: dict, x: np.ndarray) -> list[dict]:
    """The free dials: metasurface far edge (extend) and position at fixed width (slide)."""
    xg, lo, hi = pl["x_glint"], x[0] - config.DX / 2, x[-1] + config.DX / 2
    out = []

    def add(x0, x1, family, f=None):
        x0, x1 = max(x0, lo), min(x1, hi)
        i0 = int(np.searchsorted(x, x0))
        i1 = int(np.searchsorted(x, x1, side="right"))
        if i1 - i0 < 8:
            return
        key = (i0, i1)
        if any(c["idx"] == key for c in out):
            return
        out.append({"family": family, "x0": float(x[i0] - config.DX / 2),
                    "x1": float(x[i1 - 1] + config.DX / 2), "idx": key,
                    "width": float((i1 - i0) * config.DX), "far_frac": float(x[i0] / xg),
                    "bright": bool(x[i0] <= xg <= x[i1 - 1]),
                    "label": f"{family} x[{x[i0] - config.DX / 2:.0f},{x[i1 - 1] + config.DX / 2:.0f}]"})
    add(-75.0, 75.0, "today")
    for f in FAR_FRACTIONS:
        add(f * xg, NEAR_EDGE, "extend")
    for w in SLIDE_WIDTHS:
        c = NEAR_EDGE - w / 2
        while c - w / 2 >= min(1.1 * xg, lo):
            add(c - w / 2, c + w / 2, f"slide{w:g}")
            c -= SLIDE_STEP
    return out


# ---------------------------------------------------------------------------
# features and classifiers
# ---------------------------------------------------------------------------
def detector_powers(field: np.ndarray, device, batch: int = 256) -> np.ndarray:
    """No metasurface: the crop's field propagated MS -> detector plane by the ONN's
    own RS-FFT propagator, hard-window powers on the crop's dense lattice."""
    n, nx, ny = field.shape
    wx, wy = nx * config.DX, ny * config.DX
    grid = (int(wx // config.DET_SIZE[0]), int(wy // config.DET_SIZE[1]))
    prop = optics3d.PropagatorRSFFT(config.LAYER_DISTANCES[-1], nx=nx, ny=ny).to(device)
    det = optics3d.SoftDetector2D(centers=config.dense_detector_centers(grid, nx=nx, ny=ny),
                                  nx=nx, ny=ny, wx=wx, wy=wy).to(device)
    out = []
    with torch.no_grad():
        for i in range(0, n, batch):
            u = torch.as_tensor(field[i:i + batch]).to(device)
            out.append(det.hard_powers(prop(u).abs() ** 2).cpu().numpy())
    return np.concatenate(out)


def normalise(P: np.ndarray) -> np.ndarray:
    return P / np.linalg.norm(P, axis=1, keepdims=True).clip(1e-30)


def select_detectors(F: np.ndarray, y: np.ndarray, k: int) -> list[int]:
    """Greedy: Fisher separation (defect vs intact) x (1 - max |corr| to those kept),
    the same redundancy logic as prune_criterion='redundancy'."""
    a, b = F[y == 1], F[y == 0]
    fisher = np.abs(a.mean(0) - b.mean(0)) / np.sqrt(a.var(0) + b.var(0) + 1e-30)
    C = np.abs(np.corrcoef(F.T))
    keep: list[int] = []
    for _ in range(min(k, F.shape[1])):
        red = C[:, keep].max(1) if keep else np.zeros(F.shape[1])
        s = fisher * (1 - red)
        s[keep] = -1
        keep.append(int(np.argmax(s)))
    return keep


def logreg(Xtr, ytr, Xte, n_cls: int, l2: float = 1e-3, steps: int = 100, device="cpu"):
    """Multinomial logistic regression (torch, LBFGS). Returns test probabilities."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    xt = torch.as_tensor((Xtr - mu) / sd, dtype=torch.float32, device=device)
    xe = torch.as_tensor((Xte - mu) / sd, dtype=torch.float32, device=device)
    yt = torch.as_tensor(ytr, dtype=torch.long, device=device)
    W = torch.zeros(xt.shape[1], n_cls, device=device, requires_grad=True)
    b = torch.zeros(n_cls, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(xt @ W + b, yt) + l2 * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        return torch.softmax(xe @ W + b, 1).cpu().numpy()


def folds_of(ids: list[str], k: int = FOLDS) -> np.ndarray:
    """Fold by sample id (stable hash): the SAME split for every setup, so the
    comparisons are paired."""
    return np.array([int.from_bytes(i.encode(), "little") % 1000003 % k for i in ids])


def cv_scores(F: np.ndarray, y_det: np.ndarray, y_cls: np.ndarray, n_cls: int, fold: np.ndarray,
              k_det: int | None, device) -> dict:
    """Out-of-fold defect scores (detection) and class predictions; detectors are
    selected inside each training fold."""
    det_score = np.zeros(len(F))
    cls_pred = np.zeros(len(F), dtype=int)
    for f in range(FOLDS):
        tr, te = fold != f, fold == f
        cols = select_detectors(F[tr], y_det[tr], k_det) if k_det else list(range(F.shape[1]))
        p2 = logreg(F[tr][:, cols], y_det[tr], F[te][:, cols], 2, device=device)
        det_score[te] = p2[:, 1]
        pc = logreg(F[tr][:, cols], y_cls[tr], F[te][:, cols], n_cls, device=device)
        cls_pred[te] = pc.argmax(1)
    return {"det_score": det_score, "cls_pred": cls_pred}


def recall_at_fpr(score: np.ndarray, is_def: np.ndarray, fpr: float = FPR) -> tuple[np.ndarray, float]:
    thr = np.quantile(score[~is_def], 1 - fpr)
    return score > thr, float(thr)


def auc(score, is_def) -> float:
    a, b = score[is_def], score[~is_def]
    r = np.argsort(np.argsort(np.concatenate([a, b])))
    return float((r[:len(a)].sum() - len(a) * (len(a) - 1) / 2) / (len(a) * len(b)))


# ---------------------------------------------------------------------------
# the ceiling
# ---------------------------------------------------------------------------
def ceiling(field: np.ndarray, delta: np.ndarray, is_def: np.ndarray, eta: float) -> np.ndarray:
    """Per-sample detection probability at FPR of the ideal matched filter: the defect's
    exact signal (delta) against the intact pose spread (low rank, from the intact
    samples) plus white receiver noise eta x mean intact pixel power, per real component."""
    Zi = field[~is_def].reshape((~is_def).sum(), -1)
    Zi = np.concatenate([Zi.real, Zi.imag], 1).astype(np.float64)
    Zc = Zi - Zi.mean(0)
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    lam = S ** 2 / max(len(Zi) - 1, 1)
    sig2 = eta * float(np.mean(np.abs(field[~is_def]) ** 2)) / 2
    Dall = delta[is_def].reshape(is_def.sum(), -1)
    d2 = []
    for i in range(0, len(Dall), 500):
        D = np.concatenate([Dall[i:i + 500].real, Dall[i:i + 500].imag], 1).astype(np.float64)
        proj = D @ Vt.T
        d2.append(((D ** 2).sum(1) - (proj ** 2).sum(1)) / sig2 + (proj ** 2 / (lam + sig2)).sum(1))
    d = np.sqrt(np.clip(np.concatenate(d2), 0, None))
    from scipy.stats import norm
    return norm.cdf(d - Z_FPR)


# ---------------------------------------------------------------------------
# the design study's field margin, for continuity
# ---------------------------------------------------------------------------
def field_margin(field: np.ndarray, is_def: np.ndarray, noise_abs: float, m: float = 2.0) -> np.ndarray:
    import coverage_scan as cs
    W = cs.window_power(np.abs(field) ** 2)
    Wi = W[~is_def]
    mu, var = Wi.mean(0), Wi.var(0, ddof=1)
    z = (np.abs(W - mu) / np.sqrt(var + noise_abs ** 2)).reshape(len(W), -1).max(1)
    thr = np.quantile(z[~is_def], 1 - FPR)
    return z[is_def] > m * thr


# ---------------------------------------------------------------------------
def evaluate(root: Path, device, n_shards: int | None) -> dict:
    sweep = json.loads((root / "sweep.json").read_text())
    plan = json.loads((root / "plan.json").read_text())
    types = plan["types"]
    res = {"root": str(root), "plan": {k: plan[k] for k in plan if k != "samples"},
           "fpr": FPR, "eta_prune": ETA_PRUNE, "n_det": N_DET, "configs": []}
    ref_window = None
    for pl in sweep["placements"]:
        data = load_placement(root, pl["name"], n_shards)
        if data is None:
            continue
        meta = data["meta"]
        is_def = np.array([m["cls"] != "intact" for m in meta])
        ids = [m["id"] for m in meta]
        y_cls = np.array([0 if m["cls"] == "intact" else 1 + types.index(m["type"]) for m in meta])
        fold = folds_of(ids)
        t0 = time.time()
        for c in crops(pl, data["x"]):
            i0, i1 = c["idx"]
            fld = data["psi"][:, i0:i1]
            dlt = data["delta"][:, i0:i1]
            P = detector_powers(fld, device)
            F = normalise(P)
            if ref_window is None and pl["name"] == "t55_H150" and c["family"] == "today":
                ref_window = float(np.mean(np.abs(fld[~is_def]) ** 2))
            r8 = cv_scores(F, is_def.astype(int), y_cls, len(types) + 1, fold, N_DET, device)
            rd = cv_scores(F, is_def.astype(int), y_cls, len(types) + 1, fold, None, device)
            hit8, _ = recall_at_fpr(r8["det_score"], is_def)
            hitd, _ = recall_at_fpr(rd["det_score"], is_def)
            U = {str(e): ceiling(fld, dlt, is_def, e) for e in ETAS}
            ref = ref_window or float(np.mean(np.abs(fld[~is_def]) ** 2))
            # receiver noise 1% of today's mean 4x3-pixel window power (DESIGN_STUDY section 5)
            marg = field_margin(fld, is_def, 0.01 * ref * 12 * config.DX ** 2)
            dmet, dhits = detection_metrics(detection_scores(P, is_def, fold, r8["det_score"]), is_def)
            cfg = {"placement": pl["name"], "theta": pl["theta"], "H": pl["H"],
                   **{k: v for k, v in c.items() if k != "idx"},
                   "n_samples": len(meta), "n_intact": int((~is_def).sum()),
                   "L8": float(hit8[is_def].mean()), "Ldense": float(hitd[is_def].mean()),
                   "L8_auc": auc(r8["det_score"], is_def), "Ldense_auc": auc(rd["det_score"], is_def),
                   "L8_class_bal_acc": balanced_acc(y_cls, r8["cls_pred"]),
                   "U": {e: float(u.mean()) for e, u in U.items()},
                   "margin_robust": float(marg.mean()),
                   "light_dB": float(10 * np.log10(np.mean(np.abs(fld[~is_def]) ** 2) / ref)),
                   "detect": dmet,
                   "_per_sample": {"detect_hits": dhits, "L8": hit8[is_def].tolist(), "Ldense": hitd[is_def].tolist(),
                                   "U": U[str(ETA_PRUNE)].tolist(), "margin": marg.tolist(),
                                   "cls_pred": r8["cls_pred"].tolist(),
                                   "det_score": r8["det_score"].tolist()}}
            res["configs"].append(cfg)
        print(f"  {pl['name']:10s} {data['n_shards']} shards, {len(meta)} samples, "
              f"{len([c for c in res['configs'] if c['placement'] == pl['name']])} crops "
              f"({time.time() - t0:.0f} s)", flush=True)
        res.setdefault("meta", {})[pl["name"]] = meta
    best = max(res["configs"], key=lambda c: c["L8"])
    res["best_floor"] = {"config": f"{best['placement']} {best['label']}", "L8": best["L8"]}
    for c in res["configs"]:
        c["pruned"] = c["U"][str(ETA_PRUNE)] < best["L8"]
    res["learning_curve"] = learning_curve(root, sweep, best, types, device, n_shards)
    return res


def learning_curve(root, sweep, best, types, device, n_shards, fracs=(0.1, 0.25, 0.5, 0.75, 1.0)) -> dict:
    """Floor recall of the best setup when each training fold keeps only a fraction
    of its samples (stratified by cell): does the floor still climb at 100 per cell?"""
    pl = next(p for p in sweep["placements"] if p["name"] == best["placement"])
    data = load_placement(root, pl["name"], n_shards)
    meta = data["meta"]
    is_def = np.array([m["cls"] != "intact" for m in meta])
    y_cls = np.array([0 if m["cls"] == "intact" else 1 + types.index(m["type"]) for m in meta])
    c = next(c for c in crops(pl, data["x"]) if c["label"] == best["label"])
    F = normalise(detector_powers(data["psi"][:, c["idx"][0]:c["idx"][1]], device))
    fold = folds_of([m["id"] for m in meta])
    rng = np.random.default_rng(0)
    keep_u = rng.random(len(meta))
    out = {"fracs": list(fracs), "L8": [], "per_cell_at_full": best["n_samples"]}
    for fr in fracs:
        score = np.zeros(len(F))
        for f in range(FOLDS):
            tr = (fold != f) & (keep_u < fr)
            te = fold == f
            cols = select_detectors(F[tr], is_def[tr].astype(int), N_DET)
            score[te] = logreg(F[tr][:, cols], is_def[tr].astype(int), F[te][:, cols], 2, device=device)[:, 1]
        hit, _ = recall_at_fpr(score, is_def)
        out["L8"].append(float(hit[is_def].mean()))
    return out


def balanced_acc(y, pred) -> float:
    return float(np.mean([np.mean(pred[y == k] == k) for k in np.unique(y)]))


# ---------------------------------------------------------------------------
# detection only (user focus 2026-11-05): is this rail outside the intact bounds?
# ---------------------------------------------------------------------------
DET_SCORES = {
    "cos8":    "total barcode difference, 8 detectors (1 - cos to the intact mean barcode)",
    "maxz8":   "few-detector out-of-bounds, 8 detectors (largest |z| vs the intact spread)",
    "cosall":  "total barcode difference, all dense detectors",
    "maxzall": "few-detector out-of-bounds, all dense detectors",
    "logreg8": "trained linear detector, 8 detectors (the floor L8)",
}
PRIMARY = "cos8"                   # what the trained detector-only ONN scores with


def _barcode(P: np.ndarray) -> np.ndarray:
    return P / np.linalg.norm(P, axis=1, keepdims=True).clip(1e-30)


def detection_scores(P: np.ndarray, is_def: np.ndarray, fold: np.ndarray,
                     logreg8: np.ndarray) -> dict:
    """Out-of-fold defect scores that need NO defect examples at scoring time: each
    compares a rail's barcode with the intact rails of the training folds only.
    The 8 detectors are chosen inside each training fold (Fisher x redundancy), as a
    stand-in for what training + pruning picks; the 'all' variants use every dense
    detector and are fully one-class."""
    out = {k: np.zeros(len(P)) for k in DET_SCORES}
    out["logreg8"] = logreg8
    F = _barcode(P)
    for f in range(FOLDS):
        tr, te = fold != f, fold == f
        it = tr & ~is_def
        cols = select_detectors(F[tr], is_def[tr].astype(int), N_DET)
        for tag, sub in (("8", P[:, cols]), ("all", P)):
            B = _barcode(sub)
            mu = B[it].mean(0)
            sd = B[it].std(0, ddof=1) + 1e-12
            out["cos" + tag][te] = 1 - B[te] @ mu / np.linalg.norm(mu)
            out["maxz" + tag][te] = (np.abs(B[te] - mu) / sd).max(1)
    return out


def best_balanced_acc(score: np.ndarray, is_def: np.ndarray) -> float:
    """max over thresholds of (recall + specificity) / 2."""
    order = np.argsort(-score)
    d = is_def[order]
    tpr = np.cumsum(d) / max(d.sum(), 1)
    fpr = np.cumsum(~d) / max((~d).sum(), 1)
    return float(np.max((tpr + 1 - fpr) / 2))


def detection_metrics(scores: dict, is_def: np.ndarray) -> tuple[dict, dict]:
    met, hits = {}, {}
    for k, s in scores.items():
        hit, thr = recall_at_fpr(s, is_def)
        fa = float(hit[~is_def].mean())
        met[k] = {"auc": auc(s, is_def), "recall": float(hit[is_def].mean()),
                  "acc_at_fpr": float((hit[is_def].mean() + 1 - fa) / 2),
                  "best_bal_acc": best_balanced_acc(s, is_def)}
        hits[k] = hit[is_def].astype(int).tolist()
    c = scores["cos8"]
    met["distance"] = {"defect_mean": float(c[is_def].mean()), "defect_median": float(np.median(c[is_def])),
                       "intact_mean": float(c[~is_def].mean()),
                       "intact_p95": float(np.quantile(c[~is_def], 1 - FPR)),
                       "ratio": float(np.median(c[is_def]) / max(np.quantile(c[~is_def], 1 - FPR), 1e-30))}
    return met, hits


def strip(res: dict) -> dict:
    out = dict(res)
    out["configs"] = [{k: v for k, v in c.items() if k != "_per_sample"} for c in res["configs"]]
    out.pop("meta", None)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--profile", default="lab", choices=list(config.PROFILES))
    ap.add_argument("--shards", type=int, default=None, help="use only the first N shards per placement")
    ap.add_argument("--day", default=None, help="also copy the summary and figures into "
                    "data/generated/<YYYY-MM-DD>/sweep_<root> (default: $RAIL3D_DAY, else today)")
    a = ap.parse_args()
    root = Path(a.root) if a.root else latest_root()
    device = config.get_device(a.profile)
    print(f"[sweep_eval] {root}")
    res = evaluate(root, device, a.shards)
    (root / "sweep_eval_full.json").write_text(json.dumps(res, default=float))
    (root / "sweep_eval.json").write_text(json.dumps(strip(res), indent=1, default=float))
    rows = sorted(res["configs"], key=lambda c: -c["L8"])
    print(f"\n  best floor: {res['best_floor']['config']}  L8 recall {res['best_floor']['L8']:.3f}")
    print(f"  {'setup':40s} {'L8':>6s} {'Ldense':>7s} {'U(1%)':>6s} {'margin':>7s} {'light':>6s}  verdict")
    for c in rows:
        print(f"  {c['placement'] + ' ' + c['label']:40s} {c['L8']:6.3f} {c['Ldense']:7.3f} "
              f"{c['U'][str(ETA_PRUNE)]:6.3f} {c['margin_robust']:7.3f} {c['light_dB']:+6.1f}  "
              f"{'pruned' if c['pruned'] else 'KEEP'}{'  (bright)' if c['bright'] else ''}")
    print("\n  DETECTION ONLY (defect vs intact), AUC per score; recall/accuracy at "
          f"{100 * FPR:.0f}% false alarms for {PRIMARY}")
    print(f"  {'setup':40s} " + " ".join(f"{k:>7s}" for k in DET_SCORES)
          + f" {'recall':>7s} {'bal.acc':>7s} {'dist/int':>8s}")
    for c in sorted(res["configs"], key=lambda c: -c["detect"][PRIMARY]["auc"]):
        d = c["detect"]
        print(f"  {c['placement'] + ' ' + c['label']:40s} "
              + " ".join(f"{d[k]['auc']:7.3f}" for k in DET_SCORES)
              + f" {d[PRIMARY]['recall']:7.3f} {d[PRIMARY]['best_bal_acc']:7.3f} {d['distance']['ratio']:8.2f}")
    import sweep_figs
    sweep_figs.all_figures(res, root)
    copy_to_day(root, a.day)
    return 0


def copy_to_day(root: Path, day: str | None) -> Path:
    """The day's results folder gets the small outputs (summary, progress, figures);
    the fields and sweep_eval_full.json stay in the sweep root, which sweep_figs.py
    redraws from."""
    import shutil
    dest = config.day_dir(day) / f"sweep_{root.name}"
    dest.mkdir(parents=True, exist_ok=True)
    for f in ["sweep_eval.json", "progress.txt", "plan.json", "sweep.json",
              *sorted(q.name for q in root.glob("sweep_0*.png"))]:
        if (root / f).exists():
            shutil.copy2(root / f, dest / f)
    print(f"copied the summary and figures -> {dest}")
    return dest


if __name__ == "__main__":
    raise SystemExit(main())
