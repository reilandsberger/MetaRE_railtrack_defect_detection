"""One command for a whole training stage: generate -> inspect -> train -> analyse.

    python run_stage.py --stage prelim 2>&1 | tee ../prelim.log

Everything is resumable and skipped when already done, so Ctrl-C (or a
disconnect on a shared workstation) costs at most the shard or the 10-epoch
checkpoint window in flight. Re-run the identical command to continue.

It fails FAST rather than asking you to watch: a non-zero step, a geometry
mismatch, an incomplete dataset or a non-finite field stops the run before the
next multi-hour step starts. The checks it makes are the ones documented in
READING_RESULTS.md as historically misread.

Writes data/generated/stage_<stage>_report.md -- paste-able -- and prints the
exact list of files to send back. --bundle zips them.

Sizes and schedule come from config.STAGES, shared with rail3D_pipeline.ipynb.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch

from rail3d import config, data3d, train3d

PY = sys.executable
LOG: list[str] = []


def say(line: str = "") -> None:
    print(line, flush=True)
    LOG.append(line)


def step(title: str) -> None:
    say("")
    say("=" * 72)
    say(f"  {title}")
    say("=" * 72)


def run(*args: str, check: bool = True, capture: bool = True) -> tuple[int, str]:
    """Subprocess, streamed live and captured for the report."""
    say(f"$ python {' '.join(args)}")
    p = subprocess.Popen([PY, *args], cwd=HERE, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    out = []
    for line in p.stdout:
        print(line, end="", flush=True)
        if capture:
            out.append(line.rstrip())
    rc = p.wait()
    if check and rc != 0:
        say(f"\n!! {args[0]} exited {rc} — stopping before the next step.")
        raise SystemExit(rc)
    return rc, "\n".join(out)


def dataset_gate(root: Path, want: dict) -> None:
    """Refuse to spend hours training on a dataset that is not what it claims.

    Cheap checks only -- geometry provenance, completeness, finiteness -- but
    they cover the failure modes that have actually happened here: a partial
    generation looking complete, and a realised parameter range pinned to one
    end of the configured one (depth clipping, the wear normalisation bug).
    """
    cfgp = root / "dataset_config.json"
    if not cfgp.exists():
        say(f"!! {cfgp} missing — the generation did not finish.")
        raise SystemExit(1)

    diffs = data3d.check_dataset_config(root, strict=False)
    if diffs:
        say("!! geometry mismatch between this dataset and the active config:")
        for d in diffs:
            say(f"     {d}")
        raise SystemExit(1)

    meta = json.loads(cfgp.read_text(encoding="utf-8"))
    counts = meta.get("_counts") or {}
    if not counts:
        say("!! dataset_config has no _counts — generation was interrupted "
            "before finalize. Re-run this command to finish it.")
        raise SystemExit(1)

    bad = []
    for cls in config.CLASS_NAMES:
        if counts.get(cls, 0) < want["limit"]:
            bad.append(f"{cls}: {counts.get(cls, 0)} of {want['limit']}")
    if counts.get("intact", 0) < want["intact"]:
        bad.append(f"intact: {counts.get('intact', 0)} of {want['intact']}")
    if bad:
        say("!! dataset is short: " + "; ".join(bad))
        raise SystemExit(1)

    say(f"   provenance OK   commit {meta.get('_git_commit')}  "
        f"host {meta.get('_host')}  created {meta.get('_created')}")
    say(f"   counts OK       {counts}")
    say(f"   generation      {meta.get('_duration_hours')} h at "
        f"{meta.get('_samples_per_second')} samples/s")

    # finiteness + realised ranges, one shard per class (cheap)
    for cls in list(config.CLASS_NAMES) + ["intact"]:
        shard = data3d.shard_path(cls, 0, root=root)
        if not shard.exists():
            say(f"!! {shard.name} missing")
            raise SystemExit(1)
        blob = torch.load(shard, map_location="cpu", weights_only=False)
        psi = blob["psi"]
        if not torch.isfinite(psi).all():
            say(f"!! {cls}: non-finite values in the stored field")
            raise SystemExit(1)
        amp = psi[..., 0].abs().mean()
        say(f"   {cls:7s} shard0 {tuple(psi.shape)}  |psi1| mean {amp:.5f}  finite")


def result_files(stage: str) -> list[Path]:
    """The artifacts worth sending back, whichever way the stage was run.

    Importable, so the notebook produces the SAME bundle as this script --
    training in the kernel and training through run_stage.py must not give
    different review packages.
    """
    root = config.GENERATED_DIR / f"L5_{stage}"
    cand = [config.GENERATED_DIR / f"stage_{stage}_summary.json",
            config.GENERATED_DIR / f"stage_{stage}_report.md",
            config.GENERATED_DIR / "analysis.json",
            config.GENERATED_DIR / "verification_report.json",
            root / "dataset_config.json",
            root / "generation.log"]
    cand += [config.FIGURE_DIR / n for n in (
        "analysis_parameters.png", "analysis_performance.png",
        "analysis_failures.png", "dataset_review.png", "dataset_meta.png",
        "detector_sweep.png", "defect_model_check.png")]
    return [q for q in cand if q.exists()]


def write_summary(stage: str) -> Path:
    """One small JSON that answers the review questions without opening anything.

    Written whichever route ran the stage, so a notebook run is as reviewable
    as a scripted one. The two checks here are the ones that silently invalidated
    the first prelim run: whether the best checkpoint actually met the detector
    budget (README finding 22), and whether the capture term did anything.
    """
    out: dict = {"stage": stage}
    root = config.GENERATED_DIR / f"L5_{stage}"
    cfgp = root / "dataset_config.json"
    if cfgp.exists():
        meta = json.loads(cfgp.read_text(encoding="utf-8"))
        out["dataset"] = {k: meta.get(k) for k in
                          ("_counts", "_created", "_git_commit",
                           "_duration_hours", "_samples_per_second")}
        out["geometry"] = {k: meta.get(k) for k in
                           ("WVL", "H_MS", "SEG_LEN", "MESH_DS", "SHADOW_MIN_T",
                            "SHADOW_NORMAL_OFFSET", "CRACK_DEPTH_RANGE",
                            "CRACK_WIDTH_RANGE", "CRACK_LINE_COUNT",
                            "CRACK_LINE_GAP_FACTOR", "SHELL_GAUGE_X_RANGE")}
    runs = {}
    for surface in ("slm", "none", "metaunit"):
        ck = config.CHECKPOINT_DIR / f"ms3d_{surface}_l5_{stage}" / "best.pt"
        if not ck.exists():
            continue
        st = torch.load(ck, map_location="cpu", weights_only=False)
        h = st.get("history", {})
        tr = h.get("train") or [{}]
        runs[surface] = {
            "best_epoch": h.get("best_epoch"),
            "n_det_at_best": st.get("n_det"),
            "n_det_target": config.N_DET_FINAL,
            "meets_detector_budget": st.get("n_det") == config.N_DET_FINAL,
            "capture_frac_last": tr[-1].get("capture_frac"),
            "capture_floor": config.N_DET_FINAL / 130,
            "best_val": h.get("val", [{}])[h["best_epoch"]]
                        if h.get("best_epoch", -1) >= 0 and h.get("val") else None,
        }
    out["runs"] = runs
    ana = config.GENERATED_DIR / "analysis.json"
    if ana.exists():
        out["analysis"] = json.loads(ana.read_text(encoding="utf-8"))
    q = config.GENERATED_DIR / f"stage_{stage}_summary.json"
    q.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return q


def make_bundle(stage: str) -> Path:
    """Zip the review package. Safe to call from the notebook."""
    write_summary(stage)
    files = result_files(stage)
    z = config.GENERATED_DIR / f"stage_{stage}_bundle.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for q in files:
            zf.write(q, q.name)
    return z


def schedule_of(cfg: train3d.TrainConfig) -> dict:
    keys = ("n_epoch", "tau_anneal_end", "prune_start", "prune_end",
            "prune_keep", "n_det_final")
    return {k: getattr(cfg, k) for k in keys}


def check_checkpoint(cfg: train3d.TrainConfig, fresh: bool) -> None:
    """A checkpoint trained under a DIFFERENT schedule resumes silently.

    train3d only hard-refuses physics/objective drift; the epoch-valued
    schedule is a soft warning. But resuming a 1200-epoch run into a 300-epoch
    config leaves the tau anneal and the prune window half-applied, which is
    neither the old run nor the new one. Refuse, and say how to proceed.
    """
    latest = train3d._ckpt_dir(cfg) / "latest.pt"
    if not latest.exists():
        return
    state = torch.load(latest, map_location="cpu", weights_only=False)
    stored = state.get("config") or {}
    now = schedule_of(cfg)
    drift = {k: (stored[k], v) for k, v in now.items()
             if k in stored and stored[k] != v}
    if not drift:
        say(f"   resuming {cfg.run_name} from epoch {state.get('epoch', '?')}")
        return
    if fresh:
        shutil.rmtree(train3d._ckpt_dir(cfg))
        say(f"   --fresh: removed the stale checkpoint for {cfg.run_name}")
        return
    say(f"!! {cfg.run_name} has a checkpoint trained under a different schedule:")
    for k, (was, now_v) in drift.items():
        say(f"     {k}: checkpoint={was}  now={now_v}")
    say("   Resuming would leave the tau anneal and prune window half-applied.")
    say("   Re-run with --fresh to discard it and start this stage cleanly.")
    raise SystemExit(1)


def train_one(cfg: train3d.TrainConfig, device, fresh: bool) -> dict:
    check_checkpoint(cfg, fresh)
    t0 = time.time()
    hist = train3d.train(cfg, device=device)
    dt = (time.time() - t0) / 3600
    best_ep, n_ep = hist.get("best_epoch", -1), cfg.n_epoch
    best_nd = hist.get("best_n_det")
    say(f"   {cfg.run_name}: {dt:.2f} h ({dt*3600:.0f} s), "
        f"best epoch {best_ep} of {n_ep}, n_det at best {best_nd}")
    if best_nd is not None and best_nd != cfg.n_det_final:
        say(f"   !! the best checkpoint has {best_nd} detectors, not "
            f"{cfg.n_det_final}: every downstream number describes the DENSE "
            f"array, not the system being designed. n_epoch is too short for "
            f"pruning (ends at epoch {cfg.prune_end}) to finish and recover.")
    if best_ep >= 0.9 * n_ep:
        say(f"   ! best epoch is in the last 10% — the model was still "
            f"improving; n_epoch may be too low for this stage.")
    tr = hist.get("train") or [{}]
    cap = tr[-1].get("capture_frac")
    if cap is not None:
        floor = cfg.n_det_final / 130
        say(f"   capture_frac {cap:.4f} vs n_det/130 = {floor:.4f} — "
            + ("concentrating light" if cap > floor * 1.05 else
               "AT the floor: the capture term is idle"))
    return hist


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="prelim", choices=list(config.STAGES))
    ap.add_argument("--profile", default="lab", choices=list(config.PROFILES))
    ap.add_argument("--fresh", action="store_true",
                    help="discard checkpoints whose schedule differs from this stage")
    ap.add_argument("--with-baseline", action="store_true",
                    help="also train the no-metasurface baseline (doubles the time)")
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--bundle", action="store_true",
                    help="zip the files to send back")
    args = ap.parse_args()

    spec = config.STAGES[args.stage]
    tag = f"l5_{args.stage}"
    root = config.GENERATED_DIR / f"L5_{args.stage}"
    device = config.get_device(args.profile)
    config.ensure_dirs()

    step(f"rail3D stage '{args.stage}'")
    say(f"device     {device}" + (f"  ({torch.cuda.get_device_name(device.index or 0)})"
                                  if device.type == "cuda" else ""))
    say(f"geometry   grid {config.NX}x{config.NY} dx={config.DX} mm, "
        f"H_MS={config.H_MS} mm, seg={config.SEG_LEN} mm, "
        f"mesh lambda/{config.WVL / config.MESH_DS:.0f}")
    say(f"shadow     {config.SHADOW_MODE}, min_t={config.SHADOW_MIN_T}, "
        f"normal_offset={config.SHADOW_NORMAL_OFFSET}")
    say(f"stage      {spec['limit']}/class + {spec['intact']} intact  ->  {root.name}")
    say(f"schedule   n_epoch={spec['n_epoch']}, tau_anneal_end="
        f"{spec['tau_anneal_end']}, prune {spec['prune_start']}-{spec['prune_end']}")

    step("1/5  preflight")
    run("preflight.py", "--profile", args.profile, check=False)

    if not args.skip_generate:
        step(f"2/5  generate  ({spec['limit']}/class + {spec['intact']} intact)")
        say("     resumable: existing shards are skipped, Ctrl-C loses at most "
            "the shard in flight")
        run("generate_dataset_3d.py", "--profile", args.profile,
            "--name", root.name, "--limit", str(spec["limit"]),
            "--intact", str(spec["intact"]),
            "--note", f"lambda=5, H=150, 4 classes, {args.stage}")

    step("3/5  dataset gate")
    dataset_gate(root, spec)
    _, inspect_out = run("inspect_dataset.py", "--root", str(root))

    step("4/5  train")
    cfg_slm = train3d.TrainConfig(
        run_name=f"ms3d_slm_{tag}", surface="slm",
        batch_size=config.PROFILES[args.profile].train_batch,
        data_root=str(root),
        **{k: spec[k] for k in ("n_epoch", "tau_anneal_end",
                                "prune_start", "prune_end")})
    train_one(cfg_slm, device, args.fresh)

    if args.with_baseline:
        from dataclasses import replace
        cfg_none = replace(cfg_slm, run_name=f"ms3d_none_{tag}", surface="none")
        step("4b/5  train no-metasurface baseline")
        train_one(cfg_none, device, args.fresh)

    del cfg_slm
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    step("5/5  analyse")
    _, analysis_out = run("analyze_results.py", "--run-name", f"ms3d_slm_{tag}",
                          "--data-root", str(root), "--profile", args.profile)

    # ---- report ---------------------------------------------------------
    report = config.GENERATED_DIR / f"stage_{args.stage}_report.md"
    body = [f"# rail3D stage report — {args.stage}", "",
            f"*{time.strftime('%Y-%m-%d %H:%M:%S')} · {root.name}*", "",
            "## Console", "```", *LOG, "```", ""]
    ana = config.GENERATED_DIR / "analysis.json"
    if ana.exists():
        body += ["## analysis.json", "```json",
                 ana.read_text(encoding="utf-8")[:6000], "```", ""]
    report.write_text("\n".join(body), encoding="utf-8")

    write_summary(args.stage)

    step("done — send these back")
    for q in result_files(args.stage):
        print(f"   {q}")
    if args.bundle:
        print(f"\n   bundled -> {make_bundle(args.stage)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
