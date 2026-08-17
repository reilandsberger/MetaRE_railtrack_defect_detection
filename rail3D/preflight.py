"""Pre-flight check: is this machine's code, config, data and checkpoints consistent?

Run this before generating or training. It answers, in one place, the questions
that have caused every mistake so far:

  - Am I on the latest commit, or about to run stale code?
  - Which GPU will actually be used?
  - Can the defect CSVs be found?
  - What geometry is the config set to right now?
  - Was the dataset on disk generated with THAT geometry, or an older one?
  - Are the shards internally consistent (same run, complete, right classes)?
  - Do any saved checkpoints predate a change that makes them unloadable?

Exit code 0 = safe to proceed, 1 = something needs fixing (each finding prints
the exact command to fix it).

Usage (from rail3D/):
    python preflight.py                      # check the full dataset
    python preflight.py --root data/generated/smoke
    python preflight.py --profile laptop
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from rail3d import config, data3d

OK, WARN, BAD = "  OK  ", " WARN ", " STOP "


class Report:
    def __init__(self) -> None:
        self.fixes: list[str] = []
        self.stop = False

    def ok(self, msg: str, fix: str | None = None) -> None:
        # accepts (and ignores) a fix so callers can dispatch ok/warn/bad uniformly
        print(f"[{OK}] {msg}")

    def warn(self, msg: str, fix: str | None = None) -> None:
        print(f"[{WARN}] {msg}")
        if fix:
            self.fixes.append(fix)

    def bad(self, msg: str, fix: str | None = None) -> None:
        print(f"[{BAD}] {msg}")
        self.stop = True
        if fix:
            self.fixes.append(fix)


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, cwd=Path(__file__).resolve().parent.parent,
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def check_git(r: Report) -> None:
    print("\n--- code ---")
    head = sh(["git", "log", "-1", "--format=%h %s"])
    branch = sh(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if not head:
        r.warn("not a git checkout (or git unavailable)")
        return
    r.ok(f"branch {branch} at {head}")
    sh(["git", "fetch", "--quiet"])
    behind = sh(["git", "rev-list", "--count", "HEAD..@{u}"])
    if behind and behind != "0":
        r.bad(f"{behind} commit(s) behind origin - you would run stale code",
              "git checkout -- rail3D/data/ && git pull")
    else:
        r.ok("up to date with origin")
    dirty = [l for l in sh(["git", "status", "--porcelain"]).splitlines()
             if l and "rail3D/data/" not in l]
    if dirty:
        r.warn(f"{len(dirty)} modified tracked file(s) outside rail3D/data/")


def check_env(r: Report, profile: str) -> None:
    print("\n--- environment ---")
    r.ok(f"torch {torch.__version__}")
    if not torch.cuda.is_available():
        r.bad("CUDA unavailable", "reinstall torch per SETUP_LAB.md section 2")
        return
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        r.ok(f"cuda:{i} {torch.cuda.get_device_name(i)} sm_{cap[0]}{cap[1]} {mem:.0f} GB")
    dev = config.get_device(profile)
    r.ok(f"profile '{profile}' will use {dev} "
         f"({torch.cuda.get_device_name(dev.index or 0) if dev.type == 'cuda' else 'CPU'})")
    if os.environ.get("RAIL3D_DEVICE"):
        r.warn(f"RAIL3D_DEVICE={os.environ['RAIL3D_DEVICE']} overrides the profile")


def check_csvs(r: Report) -> None:
    print("\n--- defect CSVs ---")
    raw = os.environ.get("RAILDEFECT_DATA_DIR")
    if not raw:
        r.warn("RAILDEFECT_DATA_DIR unset - using the built-in default "
               f"({config.RAILDEFECT_DIR})",
               "export RAILDEFECT_DATA_DIR='C:/path/to/RailDefect'")
    if not config.RAILDEFECT_DIR.is_absolute():
        r.bad(f"RAILDEFECT_DATA_DIR is not absolute: {config.RAILDEFECT_DIR}",
              "use 'C:/Users/...' (Git Bash does not translate /c/... for exports)")
        return
    for cls, d in config.DATASET_DIRS.items():
        n = len(list(d.glob("*.csv"))) if d.is_dir() else 0
        (r.ok if n >= 5000 else r.warn if n else r.bad)(
            f"{cls}: {n} CSVs in {d}",
            None if n else f"copy data_defect_{cls}2 to {config.RAILDEFECT_DIR}")


def check_config(r: Report) -> None:
    print("\n--- active geometry ---")
    r.ok(f"lambda={config.WVL} mm, grid {config.NX}x{config.NY} @ dx={config.DX} mm "
         f"(aperture {config.WX:.0f}x{config.WY:.0f} mm)")
    r.ok(f"plane H_MS={config.H_MS} mm, x_center={config.PLANE_X_CENTER} mm, "
         f"MS->det {config.LAYER_DISTANCES[-1]} mm")
    r.ok(f"segment {config.SEG_LEN} mm, mesh lambda/{config.WVL / config.MESH_DS:.0f}, "
         f"shadow {config.SHADOW_MODE}")
    r.ok(f"classes {list(config.CLASS_NAMES)}")


def list_all_datasets(r: Report, chosen: Path) -> None:
    """Show every dataset on disk with its age and geometry, marking the one in use."""
    print("\n--- datasets available ---")
    roots = data3d.list_datasets()
    if not roots:
        r.warn("none found under data/generated",
               "python generate_dataset_3d.py --profile lab")
        return
    for d in roots:
        mark = " <== selected" if d.resolve() == chosen.resolve() else ""
        cfgp = data3d.dataset_config_path(d)
        n = sum(len(data3d.existing_shards(c, root=d))
                for c in list(config.CLASS_NAMES) + ["intact"])
        if cfgp.exists():
            c = json.loads(cfgp.read_text())
            age = (time.time() - c.get("_created_epoch", time.time())) / 3600
            print(f"  {d.name or 'generated':16s} {c.get('_created', '?'):19s} "
                  f"({age:5.1f} h old)  lam={c.get('WVL', '?')}  H={c.get('H_MS')}  "
                  f"{n} shards  commit {c.get('_git_commit', '?')}{mark}")
        else:
            print(f"  {d.name or 'generated':16s} {'(no provenance)':19s} "
                  f"{'':16s} {n} shards{mark}")


def check_dataset(r: Report, root: Path) -> None:
    print(f"\n--- dataset ({root}) ---")
    if not root.exists():
        r.warn("no dataset directory yet",
               "python generate_dataset_3d.py --profile lab")
        return

    cfgp = data3d.dataset_config_path(root)
    if not cfgp.exists():
        r.bad("dataset_config.json missing: these shards predate provenance "
              "recording, so their geometry CANNOT be verified",
              f"rm -f {root}/rail3d_*_shard*.pt   # then regenerate")
    else:
        # strict=False returns the mismatch list instead of raising, so every
        # differing key is reported (the old except-path re-parsed the raised
        # message by string prefix and silently dropped unknown keys)
        diffs = data3d.check_dataset_config(root, strict=False)
        if diffs:
            for d in diffs:
                r.bad(f"geometry mismatch: {d}")
            r.fixes.append(f"generate into a NEW named root: "
                           f"python generate_dataset_3d.py --profile lab --name <name>")
        else:
            stored = json.loads(cfgp.read_text())
            r.ok(f"generated with the current geometry "
                 f"(lam={stored.get('WVL', '?')}, H_MS={stored['H_MS']}, "
                 f"classes {stored['CLASS_NAMES']})")

    # shard inventory + timestamps: mixed mtimes mean shards from different runs
    times, total = [], 0
    for cls in list(config.CLASS_NAMES) + ["intact"]:
        ks = data3d.existing_shards(cls, root=root)
        n = 0
        for k in ks:
            p = data3d.shard_path(cls, k, root=root)
            times.append(p.stat().st_mtime)
            n += len(torch.load(p, map_location="cpu", weights_only=False)["meta"])
        total += n
        (r.ok if ks else r.bad)(f"{cls:7s}: {len(ks):3d} shard(s), {n:5d} samples",
                                None if ks else "regenerate this class")
    if times:
        span = (max(times) - min(times)) / 3600
        oldest = time.strftime("%m-%d %H:%M", time.localtime(min(times)))
        newest = time.strftime("%m-%d %H:%M", time.localtime(max(times)))
        if span > 12:
            r.bad(f"shards written {span:.1f} h apart ({oldest} .. {newest}) - "
                  f"the generator SKIPS existing shards, so this is very likely a "
                  f"mix of runs with different geometry",
                  f"rm -f {root}/rail3d_*_shard*.pt   # then regenerate in one go")
        else:
            r.ok(f"all shards written within {span:.1f} h ({oldest} .. {newest})")
    r.ok(f"total {total} samples")


def check_checkpoints(r: Report) -> None:
    print("\n--- checkpoints ---")
    d = config.CHECKPOINT_DIR
    runs = sorted([p for p in d.glob("*") if p.is_dir()]) if d.exists() else []
    if not runs:
        r.ok("none yet")
        return
    for run in runs:
        latest = run / "latest.pt"
        if not latest.exists():
            continue
        try:
            sd = torch.load(latest, map_location="cpu", weights_only=False)["model"]
            n_cls = sd["head.bias"].shape[0]
            n_det = sd["detector.u"].shape[0]
            if n_cls != len(config.CLASS_NAMES):
                r.bad(f"{run.name}: trained with {n_cls} classes, config has "
                      f"{len(config.CLASS_NAMES)} - cannot resume",
                      f"rm -rf {run}")
            else:
                r.ok(f"{run.name}: {n_cls} classes, {n_det} detectors - loadable")
        except Exception as err:  # noqa: BLE001
            r.warn(f"{run.name}: unreadable ({err!r})", f"rm -rf {run}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="dataset dir (default: data/generated)")
    ap.add_argument("--profile", default="lab")
    args = ap.parse_args()

    root = Path(args.root) if args.root else config.GENERATED_DIR
    r = Report()
    print("=" * 72)
    print("rail3D pre-flight")
    print("=" * 72)
    check_git(r)
    check_env(r, args.profile)
    check_csvs(r)
    check_config(r)
    list_all_datasets(r, root)
    check_dataset(r, root)
    check_checkpoints(r)

    print("\n" + "=" * 72)
    if r.stop:
        print("RESULT: STOP - fix the items above before generating or training")
    elif r.fixes:
        print("RESULT: OK with warnings")
    else:
        print("RESULT: OK - safe to generate and train")
    if r.fixes:
        print("\nsuggested commands:")
        for f in dict.fromkeys(r.fixes):
            print(f"  {f}")
    print("=" * 72)
    return 1 if r.stop else 0


if __name__ == "__main__":
    raise SystemExit(main())
