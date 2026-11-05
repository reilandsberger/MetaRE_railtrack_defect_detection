"""train_detect.py -- the detector-only system (no metasurface), trained for DETECTION only.

    python train_detect.py --stage prelim                  # 8 detectors, ~3 min on the 5090
    python train_detect.py --stage prelim --n-det 4 8 12 16   # how many detectors? ~3 min each

User focus (2026-11-05): flag a rail as defective when its detector barcode leaves the
intact bounds; classification is set aside, and the SLM is on hold (README finding 24).

Each run is the no-metasurface ONN (`surface="none"`: propagation + 130 trainable soft
detectors, pruned to --n-det) with `task="detect"`: the rank loss only (no class or
class-centroid term) and the checkpoint chosen on validation AUC alone. It resumes from
its checkpoint if re-run. The detect+classify baseline that run_stage.py trained
(`ms3d_none_<tag>`) is scored alongside when it exists, so the gain from dropping
classification is measured, not assumed.

Every model is scored on the TEST split (never trained on, never used for selection),
with the alarm threshold set on the VALIDATION intact rails, by two detection rules:
  total barcode difference   1 - cos(barcode, mean intact barcode)   (what training optimises)
  few-detector out-of-bounds largest |z| over the detectors vs the intact spread
and reported as AUC, recall at 5% false alarms, test false-alarm rate, best balanced
accuracy, the barcode distance of defects vs intact, recall per class and by arc
position s0 (README finding 25: s0 decides detection).

Writes detect_report.json and detect_compare.png into the day's results folder,
data/generated/<YYYY-MM-DD>/ (--day, else $RAIL3D_DAY, else today).
"""
from __future__ import annotations

import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import torch                      # noqa: E402

from rail3d import config, data3d, train3d   # noqa: E402

FPR = 0.05
S0_BINS = (100, 115, 125, 135, 145, 155, 170)
RULES = {"cos": "total barcode difference", "maxz": "few-detector out-of-bounds"}


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    r = np.argsort(np.argsort(np.concatenate([pos, neg])))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))


def best_bal_acc(pos: np.ndarray, neg: np.ndarray) -> float:
    s = np.concatenate([pos, neg])
    d = np.concatenate([np.ones(len(pos), bool), np.zeros(len(neg), bool)])[np.argsort(-s)]
    return float(np.max((np.cumsum(d) / len(pos) + 1 - np.cumsum(~d) / len(neg)) / 2))


def barcode(P: torch.Tensor) -> np.ndarray:
    P = P.double().cpu().numpy()
    return P / np.linalg.norm(P, axis=1, keepdims=True).clip(1e-30)


def score_model(model, data, metas) -> dict:
    """Test-split detection metrics; reference and threshold from the val intact rails."""
    with torch.no_grad():
        Bd = barcode(train3d._hard_dets(model, data["fields"], data["test"]))
        Bt = barcode(train3d._hard_dets(model, data["intact"], data["i_test"]))
        Bv = barcode(train3d._hard_dets(model, data["intact"], data["i_val"]))
    mu, sd = Bv.mean(0), Bv.std(0, ddof=1) + 1e-12
    rules = {"cos": lambda B: 1 - B @ mu / np.linalg.norm(mu),
             "maxz": lambda B: (np.abs(B - mu) / sd).max(1)}
    labels = data["labels"][data["test"]].cpu().numpy()
    test_idx = data["test"].cpu().numpy()
    s0 = np.array([metas[i].get("s0", np.nan) for i in test_idx], float)
    out = {"n_det": int(model.detector.n_det), "n_test_defects": len(Bd),
           "n_test_intact": len(Bt), "n_val_intact": len(Bv)}
    for name, f in rules.items():
        sd_, st_, sv_ = f(Bd), f(Bt), f(Bv)
        thr = float(np.quantile(sv_, 1 - FPR))
        hit = sd_ > thr
        r = {"auc": auc(sd_, st_), "recall": float(hit.mean()),
             "false_alarm_test": float((st_ > thr).mean()), "threshold": thr,
             "best_bal_acc": best_bal_acc(sd_, st_),
             "per_class": {c: float(hit[labels == k].mean()) for k, c in enumerate(config.CLASS_NAMES)
                           if (labels == k).any()},
             "per_class_auc": {c: auc(sd_[labels == k], st_) for k, c in enumerate(config.CLASS_NAMES)
                               if (labels == k).any()},
             "by_s0": []}
        for a, b in zip(S0_BINS[:-1], S0_BINS[1:]):
            m = (s0 >= a) & (s0 < b)
            r["by_s0"].append({"lo": a, "hi": b, "n": int(m.sum()),
                               "recall": float(hit[m].mean()) if m.any() else None})
        if name == "cos":
            r["distance"] = {"defect_median": float(np.median(sd_)), "defect_mean": float(sd_.mean()),
                             "intact_mean": float(st_.mean()), "alarm_line": thr,
                             "ratio": float(np.median(sd_) / max(thr, 1e-30)),
                             "_defect": sd_.tolist(), "_intact": st_.tolist()}
        out[name] = r
    return out


def figure(rep: dict, path) -> None:
    runs = rep["runs"]
    names = list(runs)
    fig, axs = plt.subplots(1, 4, figsize=(22, 5), constrained_layout=True)
    x = np.arange(len(names))
    cols = {"cos": "#1b9e77", "maxz": "#d95f02"}
    ax = axs[0]
    for j, rule in enumerate(RULES):
        v = [runs[n][rule]["auc"] for n in names]
        ax.bar(x + (j - 0.5) * 0.38, v, 0.38, color=cols[rule], label=RULES[rule])
        for xi, vi in zip(x, v):
            ax.text(xi + (j - 0.5) * 0.38, vi + 0.003, f"{vi:.3f}", ha="center", fontsize=7.5)
    ax.set_ylim(0.5, 1.02)
    ax.set_xticks(x, names, rotation=20, ha="right", fontsize=8)
    ax.set_title("A. Detection AUC, test split", loc="left")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, axis="y")
    ax = axs[1]
    cls = config.CLASS_NAMES
    w = 0.8 / len(names)
    for i, n in enumerate(names):
        v = [runs[n]["cos"]["per_class"].get(c, np.nan) for c in cls]
        ax.bar(np.arange(len(cls)) + (i - (len(names) - 1) / 2) * w, v, w, label=n)
    ax.set_xticks(range(len(cls)), cls)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"B. Defects caught at {100 * FPR:.0f}% false alarms,\nby class (total barcode difference)",
                 loc="left")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3, axis="y")
    ax = axs[2]
    for n in names:
        bs = runs[n]["cos"]["by_s0"]
        ax.plot([(b["lo"] + b["hi"]) / 2 for b in bs], [b["recall"] if b["recall"] is not None else np.nan
                                                       for b in bs], "-o", ms=4, label=n)
    ax.set_xlabel("arc position s0 (mm): crown ← → gauge corner")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("C. Defects caught vs position on the railhead", loc="left")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)
    ax = axs[3]
    best = max(names, key=lambda n: runs[n]["cos"]["auc"])
    d = runs[best]["cos"]["distance"]
    lo = max(min(min(d["_defect"]), min(d["_intact"])), 1e-7)
    bins = np.logspace(np.log10(lo), np.log10(max(d["_defect"]) * 1.1), 40)
    ax.hist(d["_intact"], bins, color="0.6", alpha=0.8, density=True, label="intact (test)")
    ax.hist(d["_defect"], bins, histtype="step", color=cols["cos"], lw=2, density=True, label="defects (test)")
    ax.axvline(d["alarm_line"], color="k", ls="--", lw=1, label=f"alarm line ({100 * FPR:.0f}% on val intact)")
    ax.set_xscale("log")
    ax.set_xlabel("barcode distance from intact, 1 − cos")
    ax.set_title(f"D. Barcode distance, {best}\n(defect median ×{d['ratio']:.1f} the alarm line)", loc="left")
    ax.legend(fontsize=7.5)
    fig.suptitle(f"Detector-only system, detection only: {rep['dataset_root']}  "
                 f"({runs[best]['n_test_defects']} test defects, {runs[best]['n_test_intact']} test intact)",
                 fontsize=13, fontweight="bold")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="prelim", choices=sorted(config.STAGES))
    ap.add_argument("--root", default=None, help="dataset folder under data/generated (default: the stage root)")
    ap.add_argument("--n-det", type=int, nargs="+", default=[config.N_DET_FINAL])
    ap.add_argument("--profile", default="lab")
    ap.add_argument("--day", default=None, help="results folder data/generated/<YYYY-MM-DD> "
                    "(default: $RAIL3D_DAY, else today)")
    ap.add_argument("--quick", action="store_true", help="40-epoch schedule, for a plumbing check only")
    a = ap.parse_args()

    root = data3d.stage_root(a.stage) if a.root is None else config.GENERATED_DIR / a.root
    out = config.day_dir(a.day)
    report = out / ("detect_report_quick.json" if a.quick else "detect_report.json")
    data3d.check_dataset_config(root, strict=True)
    tag = data3d.run_tag(root)
    device = config.get_device(a.profile)
    spec = config.STAGES[a.stage]
    sched = ({"n_epoch": 50, "tau_anneal_end": 15, "prune_start": 5, "prune_end": 40, "prune_keep": 0.5} if a.quick else
             {k: spec[k] for k in ("n_epoch", "tau_anneal_end", "prune_start", "prune_end")})
    base = dict(surface="none", data_root=str(root), batch_size=config.PROFILES[a.profile].train_batch, **sched)
    print(f"\ndataset {root.name} | device {device} | {sched['n_epoch']} epochs/run | n_det {a.n_det}\n", flush=True)

    rep: dict = {"dataset_root": root.name, "fpr": FPR, "schedule": sched, "runs": {}}
    metas: list[dict] = []
    for c in config.CLASS_NAMES:
        metas.extend(data3d.load_meta(c, root=root))
    data = None
    todo = []
    ref = train3d.TrainConfig(run_name=f"ms3d_none_{tag}", **base)
    if (config.CHECKPOINT_DIR / ref.run_name / "best.pt").exists() and not a.quick:
        todo.append(("detect+classify n8 (run_stage)", ref, False))
    for n in a.n_det:
        cfg = train3d.TrainConfig(run_name=f"det_none_n{n}_{tag}" + ("_quick" if a.quick else ""),
                                  task="detect", n_det_final=n, **base)
        todo.append((f"detect-only n{n}", cfg, True))
    for label, cfg, train in todo:
        t0 = time.time()
        if train:
            hist = train3d.train(cfg, device=device, verbose=False)
            print(f"  trained {label}: best epoch {hist.get('best_epoch')} of {cfg.n_epoch}, "
                  f"n_det {hist.get('best_n_det')} ({time.time() - t0:.0f} s)", flush=True)
            if hist.get("best_n_det") != cfg.n_det_final:
                print(f"  !! {label}: pruning never reached {cfg.n_det_final} detectors, so this "
                      f"model is the {hist.get('best_n_det')}-detector one; lengthen the prune "
                      f"window (prune_end) before trusting it", flush=True)
        model, state = train3d.load_trained(cfg, device, "best")
        if data is None:
            data = train3d.load_all_data(train3d.effective_config(cfg, state), device)
        r = score_model(model, data, metas)
        r["run_name"] = cfg.run_name
        r["best_epoch"] = (state.get("history") or {}).get("best_epoch")
        rep["runs"][label] = r
        report.write_text(json.dumps(rep, indent=1), encoding="utf-8")

    print(f"\n  {'model':32s} {'n_det':>5s} | {'AUC cos':>7s} {'maxz':>6s} | {'recall':>6s} "
          f"{'FA test':>7s} {'bal.acc':>7s} {'dist x':>6s} | per-class recall (cos)")
    for label, r in rep["runs"].items():
        c, z = r["cos"], r["maxz"]
        pc = " ".join(f"{k} {v:.2f}" for k, v in c["per_class"].items())
        print(f"  {label:32s} {r['n_det']:5d} | {c['auc']:7.3f} {z['auc']:6.3f} | {c['recall']:6.3f} "
              f"{c['false_alarm_test']:7.3f} {c['best_bal_acc']:7.3f} {c['distance']['ratio']:6.1f} | {pc}")
    figure(rep, report.with_name(report.stem.replace("detect_report", "detect_compare") + ".png"))
    print(f"report -> {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
