"""Run the whole verification chain and emit ONE compact report to paste back.

On the lab GPU this takes roughly 5 minutes end to end:
    V0-V4 (CPU)  ->  V5-V7 (GPU)  ->  smoke generation  ->  dataset stats  ->  V8

Usage (from rail3D/):
    python lab_report.py                 # full chain
    python lab_report.py --skip-smoke    # reuse an existing smoke set
    python lab_report.py --quick         # V0-V4 + smoke + stats + V8, skip V5-V7

Writes data/generated/lab_report.md and prints it. Paste that file's contents
back and it carries everything needed to judge whether the run is healthy:
environment, every gate's pass/fail with its key numbers, per-class geometry
statistics from the STORED shards, class separability, and a measured
samples/second used to extrapolate the full generation.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from rail3d import config, data3d, losses3d, optics3d

PY = sys.executable
HERE = Path(__file__).resolve().parent


def run(cmd: list[str], timeout: int = 7200) -> tuple[str, float, bool]:
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr), time.time() - t0, r.returncode == 0
    except subprocess.TimeoutExpired:
        return "TIMEOUT", time.time() - t0, False


def gate_lines(out: str) -> list[str]:
    return [ln.strip() for ln in out.splitlines() if ln.startswith(("[PASS]", "[FAIL]"))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--quick", action="store_true", help="skip V5-V7 (the slow gates)")
    ap.add_argument("--profile", default="lab")
    ap.add_argument("--smoke-n", type=int, default=config.SMOKE_PER_CLASS,
                    help="smoke samples per class (see generate_dataset_3d.py)")
    args = ap.parse_args()

    L: list[str] = []
    A = L.append
    A("# rail3D lab report")
    A("")

    # ---- environment ----------------------------------------------------
    A("## Environment")
    A("```")
    A(f"host        {platform.node()}  {platform.system()} {platform.release()}")
    A(f"python      {sys.version.split()[0]}")
    A(f"torch       {torch.__version__}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            cap = torch.cuda.get_device_capability(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            A(f"cuda:{i}      {torch.cuda.get_device_name(i)}  sm_{cap[0]}{cap[1]}  {mem:.0f} GB")
    else:
        A("cuda        NOT AVAILABLE")
    A(f"device      {config.get_device(args.profile)}")
    A(f"geometry    grid {config.NX}x{config.NY} dx={config.DX} mm, seg={config.SEG_LEN} mm, "
      f"mesh=lambda/{config.WVL / config.MESH_DS:.0f}, shadow={config.SHADOW_MODE}")
    A(f"classes     {config.CLASS_NAMES}")
    A("```")
    A("")

    # ---- V0-V4 ----------------------------------------------------------
    A("## Gates")
    A("```")
    out, dt, ok = run([PY, "tests_physics_3d.py"])
    for ln in gate_lines(out):
        A(ln)
    A(f"(tests_physics_3d in {dt:.0f}s)")

    # ---- V5-V7 ----------------------------------------------------------
    if not args.quick:
        out, dt, ok = run([PY, "validation_3d.py"])
        for ln in gate_lines(out):
            A(ln)
        A(f"(V5-V7 in {dt:.0f}s)")
    else:
        A("(V5-V7 skipped: --quick)")
    A("```")
    A("")

    # ---- smoke generation ----------------------------------------------
    root = config.GENERATED_DIR / "smoke"
    if not args.skip_smoke:
        if root.exists():
            shutil.rmtree(root)
        out, dt, ok = run([PY, "generate_dataset_3d.py", "--profile", args.profile,
                           "--smoke", "--smoke-n", str(args.smoke_n)])
        n_gen = (len(config.CLASS_NAMES) * args.smoke_n
                 + config.smoke_intact_count(args.smoke_n))
        A("## Smoke generation")
        A("```")
        A(f"{n_gen} samples in {dt:.0f}s  ->  {n_gen / max(dt, 1e-9):.2f} samples/s")
        full = len(config.CLASS_NAMES) * config.FULL_PER_CLASS + config.FULL_INTACT
        A(f"extrapolated full run ({full} samples): {full / max(n_gen / max(dt, 1e-9), 1e-9) / 3600:.2f} h")
        if not ok:
            A("GENERATION FAILED:")
            A(out[-1500:])
        A("```")
        A("")

    if not root.exists():
        A("## Stored shards / separability / V8")
        A("```")
        A(f"smoke set absent ({root}) - sections skipped; rerun without --skip-smoke")
        A("```")
        text = "\n".join(L)
        path = config.GENERATED_DIR / "lab_report.md"
        path.write_text(text, encoding="utf-8")
        print(text)
        print(f"\n[saved to {path}]")
        return 1

    # ---- stored-shard statistics ----------------------------------------
    A("## Stored shards (geometry as generated)")
    A("```")
    rc = subprocess.run([PY, "inspect_dataset.py", "--root", str(root)],
                        cwd=HERE, capture_output=True, text=True)
    for ln in rc.stdout.splitlines():
        if ln.startswith(("===", "    ")) and "wrote" not in ln:
            A(ln.rstrip())
    A("```")
    A("")

    # ---- class separability (untrained optics) --------------------------
    A("## Class separability through UNTRAINED optics")
    A("(cos-gap of each class' mean barcode vs the intact mean; training should widen these)")
    A("```")
    try:
        dev = config.get_device(args.profile)
        psi0 = torch.load(data3d.psi0_path(root=root), map_location="cpu", weights_only=False)
        model = optics3d.ONN3D(noise=False).to(dev).eval()
        bars = {}
        with torch.no_grad():
            for cls in ("intact",) + config.CLASS_NAMES:
                psi, _ = data3d.load_class_fields(cls, root=root)
                tot = data3d.combine_field(psi, "tot", psi0=psi0).to(dev)
                bars[cls] = model(tot, hard=True)[1]
        ref = bars["intact"].mean(dim=0)
        intact_spread = float(losses3d.cos_gap(bars["intact"], ref).mean())
        A(f"intact self-spread     {intact_spread:.4f}   (the noise floor to beat)")
        for cls in config.CLASS_NAMES:
            g = losses3d.cos_gap(bars[cls], ref)
            auc = losses3d.auc_score(g, losses3d.cos_gap(bars["intact"], ref))
            A(f"{cls:8s} cos-gap mean {float(g.mean()):.4f}  p5 {float(torch.quantile(g, .05)):.4f}"
              f"   AUC vs intact {auc:.3f}")
    except Exception as err:  # noqa: BLE001
        A(f"separability check failed: {err!r}")
    A("```")
    A("")

    # ---- wavefront version comparison ------------------------------------
    A("## Wavefront comparison across simulation versions")
    A("(one crack sample solved every way; full numbers in "
      "data/generated/wavefront_comparison.json, figures in data/figures/)")
    A("```")
    out, dt, ok = run([PY, "compare_wavefronts.py", "--profile", args.profile])
    started = False
    for ln in out.splitlines():
        if ln.strip().startswith("version"):
            started = True
        if started and ln.strip():
            A(ln.rstrip())
    if not ok:
        A("comparison FAILED:")
        A(out[-800:])
    A(f"({dt:.0f}s)")
    A("```")
    A("")

    # ---- V8 --------------------------------------------------------------
    A("## V8 end-to-end (rank objective, 30 epochs on the smoke set)")
    A("```")
    out, dt, ok = run([PY, "v8_smoke_test.py"])
    for ln in out.splitlines():
        if ln.startswith(("[PASS]", "[FAIL]", "best val")):
            A(ln.strip())
    A(f"(V8 in {dt:.0f}s)")
    A("```")

    text = "\n".join(L)
    path = config.GENERATED_DIR / "lab_report.md"
    path.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n[saved to {path}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
