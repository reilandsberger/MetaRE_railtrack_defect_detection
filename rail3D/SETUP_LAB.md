# rail3D — Lab Workstation Setup (RTX 5090)

Step-by-step replication of the 3D rail-defect pipeline on the lab computer
(VS Code + local filesystem, GPU = **RTX 5090**; its device index varies by
machine, so the code selects it automatically — see §3). Developed and
verified on the laptop (MX250); every command below was designed to run
unchanged on the lab machine except where marked.

---

## 0. Read this first

- **Start the defect-CSV copy now** (§4) — it is ~2 GB and every step from §5
  onward needs it. It can transfer while you do §1–§3.
- **Steps 5–12 are the runbook**, in the order you actually run them. If you
  only want the short version: `lab_report.py` → `generate_dataset_3d.py
  --profile lab` → training notebook → `sweep_detectors.py` →
  `analyze_results.py`.
- **If you cloned earlier in the session, `git pull` before running anything.**
  Fixes land on `3D_railhead_upgrade` between sessions; the `cuda:auto` GPU
  selection in §3 is one of them.
- **Shell syntax matters.** §3's environment variables use `export` in Git Bash,
  `$env:` in PowerShell, `set` in cmd — the wrong one fails *silently* and you
  only find out when a script reports "0 CSVs".

## 1. Get the code

```bash
git clone https://github.com/reilandsberger/MetaRE_railtrack_defect_detection.git
cd MetaRE_railtrack_defect_detection
git checkout 3D_railhead_upgrade
```

(Or `git pull` + `git checkout 3D_railhead_upgrade` if already cloned.)

All 3D code lives in `rail3D/`. Bulk data (`rail3D/data/generated/`,
`rail3D/data/checkpoints/`) is gitignored and regenerated locally — see §8.

## 2. Python environment

The lab computer already has Python (3.11+ required). Create a fresh venv:

```bash
cd rail3D
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux
```

**Install torch with the cu128 build.** The RTX 5090 is Blackwell (sm_120) —
a `+cu118` or `+cu121` wheel will fail with "no kernel image available":

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

**Never add `-U` / `--upgrade` to that second command.** `requirements.txt` only
asks for `torch>=2.7`, which your cu128 build already satisfies; an upgrade would
resolve against PyPI and replace it with a CPU-only wheel, silently killing CUDA.

Sanity check (should print the 5090's compute capability `(12, 0)` without
errors):

```bash
python -c "import torch; print(torch.__version__); [print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]"
```

## 3. Point the code at the right GPU and data

One required environment variable, `RAILDEFECT_DATA_DIR`. **Use your shell's own
syntax** — the wrong form does not error, it just leaves the variable unset:

```bash
# Git Bash (MINGW64) — note the Windows-style C:/ path, NOT /c/...
export RAILDEFECT_DATA_DIR='C:/Users/<you>/Documents/Rei/RailDefect'
```

```powershell
# PowerShell
$env:RAILDEFECT_DATA_DIR = 'C:\Users\<you>\Documents\Rei\RailDefect'
```

```bat
:: cmd.exe
set RAILDEFECT_DATA_DIR=C:\Users\<you>\Documents\Rei\RailDefect
```

In Git Bash an MSYS path like `/c/Users/...` is *not* translated for an exported
variable, and Python on Windows cannot resolve it — always give `C:/...`.

The variable is read at **import time** by `rail3d/config.py`, so it must be set
before Python starts. For notebooks, launch VS Code from the same shell (`code .`)
so the kernel inherits it; otherwise set `os.environ['RAILDEFECT_DATA_DIR'] = ...`
in the first cell, **above** the `from rail3d import ...` line.

Verify it before moving on (from `rail3D/`):

```bash
python -c "from rail3d import config, sections; print(config.RAILDEFECT_DIR, config.RAILDEFECT_DIR.is_absolute()); print({c: len(sections.get_dataset_files(c)) for c in config.CLASS_NAMES})"
```

Expect an absolute path, `True`, and `{'crack': 5000, 'dent': 5000, 'wear': 5000}`
(smaller counts simply mean the §4 copy is still running). Anything wrong raises a
`FileNotFoundError` naming the specific problem.

> You do **not** need to edit the hard-coded fallback path in `config.py`. It is only
> used when `RAILDEFECT_DATA_DIR` is unset; an exported value always wins.

- **GPU selection is automatic** — no `RAIL3D_DEVICE` needed. The `lab` profile
  uses `cuda:auto`:
  `config.best_cuda_device()` ranks the visible CUDA devices by compute
  capability, then VRAM, and takes the strongest — the 5090 regardless of
  whether it is index 0 or 1. It prints which one it chose when more than one
  device is present. Set `RAIL3D_DEVICE=cuda:N` only to override that choice
  (e.g. to leave the 5090 free for someone else). Nothing ever uses a bare
  `"cuda"`.
- `RAILDEFECT_DATA_DIR` — the **parent** folder holding the three defect CSV
  datasets (`data_defect_crack2/`, `data_defect_dent2/`, `data_defect_wear2/`).
  Required by V5–V7, the V8 smoke test, and all dataset generation — see §4.
  Only training-from-existing-shards works without it.

`config.py` fails fast with an actionable message if the torch build cannot
drive the GPU.


## 4. Get the defect CSVs onto this machine (~2 GB — start this first)

`rail3D` reads only these three folders from `RAILDEFECT_DATA_DIR`; nothing else
in the old `RailDefect/` tree is used:

| folder | contents | size |
|---|---|---|
| `data_defect_crack2/` | 5000 CSVs | ~667 MB |
| `data_defect_dent2/`  | 5000 CSVs | ~667 MB |
| `data_defect_wear2/`  | 5000 CSVs | ~667 MB |

(`shell` is fully parametric — it needs no CSV folder.) Copy them under one
parent; that parent is what `RAILDEFECT_DATA_DIR` points at. With the source
mounted as `Z:`:

```bat
robocopy Z:\RailDefect\data_defect_crack2 C:\Users\<you>\Documents\Rei\RailDefect\data_defect_crack2 /E /MT:16
```

Repeat for `data_defect_dent2` and `data_defect_wear2`. Many small files — expect
this to be slower than 2 GB of bulk data suggests, which is why it goes first.

---

# The runbook

Steps 5–12 in the order you actually run them. Steps 5–7 are one-time checks;
8–12 are the experiment.

## 5. Verify the machine — one command

```bash
cd rail3D && python lab_report.py
```

~5 min on the 5090. Runs V0–V4, V5–V7, a smoke generation, stored-shard
statistics, a class-separability check and V8, then writes
`data/generated/lab_report.md`. Every gate must say PASS. `--quick` skips the
slow V5–V7 gates; `--skip-smoke` reuses an existing smoke set.

## 6. Look at the setup

```bash
python setup_diagram.py
```

Writes to `data/figures/`:
- `setup_diagram.png` — 3D scene, side view (x–z) and top view (x–y) with every
  distance annotated, plus a marker showing where the specular lobe lands
  relative to the aperture;
- `cross_sections.png` — intact vs defect CSV cross-sections;
- `mesh_review.png` — one mesh per class with its `d(s,y)` depth-field footprint.

For the physics and module-design figures, run `design_review_notebook.ipynb`
and `validation_3d_notebook.ipynb`.

## 7. Measurement-plane geometry — already decided

```bash
python scan_geometry.py          # only if you want to re-derive it
```

Already run: `H_MS = 240 mm`, plane centred at x = 0, aperture 60×30. The scan
measured H = 80/160/240/320/400/480 and this configuration maximised the
*field-level* information (AUC 0.899). Notably H = 80 puts the specular lobe
**inside** the aperture — 13× the energy and **below-chance** separability — so
this system works precisely because it is dark-field. Full table and reasoning
are in `rail3d/config.py` above `H_MS`.

## 8. Generate the dataset

```bash
python generate_dataset_3d.py --profile lab
```

~36 min on the 5090 for 4 classes × 5000 + 512 intact (measured 9.47 samples/s).
Resumable: shards are written atomically and existing ones are skipped, so an
interrupt just continues on relaunch — safe on a shared workstation.

Flags: `--status` (which shards exist/remain), `--classes crack dent` (subset),
`--limit N` (samples per class), `--smoke` (20/class into `data/generated/smoke/`).

**Shards from before the rev.2 geometry are incompatible** (different defect
model, label order, plane height). Delete them first:

```bash
rm -f data/generated/rail3d_*_shard*.pt
```

## 9. Check the data before training on it

```bash
python inspect_dataset.py --root data/generated
```

Re-derives each stored sample's geometry from its seed and shows it beside the
stored field, plus the mean defect-minus-intact intensity and per-class
parameter statistics. Confirm the depth/length/angle ranges match `README.md` §2.

## 10. Train

Open `training_3d_ms_notebook.ipynb` (select the `.venv` kernel): a phase-only
`SLM2D` run first, then the real meta-atom `MetaUnitSoft`. `training_3d_no_ms_notebook.ipynb`
is the no-metasurface baseline. Headless equivalent:

```bash
python -c "from rail3d import config, train3d; train3d.train(train3d.TrainConfig(run_name='ms3d_slm_v1', surface='slm', n_epoch=1200, batch_size=config.PROFILES['lab'].train_batch))"
```

~10 min at 1200 epochs. Training auto-resumes from
`data/checkpoints/<run_name>/latest.pt`; delete that folder to start over.

### How the detectors are optimized

Positions are **trained, not swept**. `SoftDetector2D` holds the window centres
as an `nn.Parameter` with sigmoid-edged (differentiable) masks, so gradients move
them; variance-based pruning runs *inside* the same training run. One run does:

| epochs | what happens |
|---|---|
| 0–40 | full starting grid, soft masks (large τ), phase + positions training |
| 40–200 | pruning window: geometric reduction to the final count, positions still moving |
| 200–250 | τ anneal completes, masks sharpen toward hard edges |
| 250–1200 | stationary fine-tune at the final count and sharpness |

All reported metrics use **hard** binary windows, never the soft training masks.
Why 1200 epochs: the τ anneal and pruning make the objective non-stationary for
~250 epochs, so a 400-epoch run left only ~150 stationary epochs to settle ~1800
metasurface parameters. `train()` prints where the best epoch landed and warns if
the model was still improving at the end.

## 11. How few detectors can you get away with?

```bash
python sweep_detectors.py --counts 4 6 8 10
```

Starts from a dense 13×10 = 130-detector grid (~92% plane coverage) and prunes
to each final count, one full training run per count. Writes
`data/figures/detector_sweep.png` (AUC and class accuracy vs count, plus the
surviving layout) and `data/generated/detector_sweep.json`.

Add `--dist 120 160 200` to sweep the metasurface→detector distance at the same
time — that is a training-time propagation, so it needs no regeneration.

## 12. Where does it succeed and fail?

```bash
python analyze_results.py --run-name ms3d_slm_v1
```

Joins every test sample's outcome to the defect parameters stored in its shard
metadata, and writes:
- `analysis_parameters.png` — detection rate vs depth, size, orientation and
  across-head position, per class. A downward trend marks the regime the system
  misses (e.g. "cracks shallower than N mm").
- `analysis_performance.png` — per-class ROC, class-confusion matrix, and the
  score distributions with the calibrated threshold drawn on.
- `analysis_failures.png` — the worst missed detections, each rendered as the
  depth field that produced it.
- `analysis.json` — every number behind those figures, plus a per-class ranking
  of which parameter best predicts failure.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `git pull` → "local changes would be overwritten by merge" | You ran the scripts, which rewrite generated output. Everything under `rail3D/data/` is regenerated and safe to discard: `git checkout -- rail3D/data/` then pull. (Keep real edits with `git stash` → `git pull` → `git stash pop`.) |
| "It's running on the Intel GPU / the wrong GPU" | Almost always a misread — **CUDA never uses Intel integrated graphics**, and Task Manager's GPU numbering does not match CUDA's (here the 5090 is Task Manager GPU 1 but CUDA `cuda:0`). Trust the `[rail3d] ... running on cuda:N (NVIDIA ...)` banner each script prints, and `nvidia-smi`. Task Manager also hides CUDA work unless you switch a graph to the **Compute_0** engine. |
| `tests_physics_3d.py` shows no GPU activity | Expected: **V0–V4 run on CPU by design** (tiny meshes, ~5 s, safe on a 2 GB card). Force the GPU with `RAIL3D_TEST_DEVICE=cuda:0`. |
| `ValueError: crack: requested 5000 but only 0 CSVs` | `RAILDEFECT_DATA_DIR` unset in *this* process, wrong shell syntax (§3), an MSYS `/c/...` path, or pointing inside a `data_defect_*2` folder instead of their parent. |
| Reference rail width prints ~94 mm instead of 157.4 mm | Wrong image loader for `crosssection.png` — see `README.md` §6 finding 1. |
| CUDA "no kernel image available" | A non-cu128 torch got installed; re-run §2 and do not use `pip install -U`. |
| Generation looks stalled | It logs every ~50 samples to `data/generated/generation.log`; `--status` lists shards done/remaining. |
| Notebook can't find the data | `RAILDEFECT_DATA_DIR` is read at import time — launch VS Code from the shell where you exported it (`code .`), or set `os.environ[...]` above the `from rail3d import ...` line. |

## Timings (RTX 5090)

| Step | Time |
|---|---|
| `lab_report.py` (V0–V8 + smoke) | ~5 min |
| `setup_diagram.py` | seconds (CPU) |
| `scan_geometry.py` (8 configs) | ~1 min |
| `generate_dataset_3d.py --profile lab` | **~36 min** (9.47 samples/s) |
| `inspect_dataset.py` | ~10 s |
| training, 1200 epochs | ~10 min |
| `sweep_detectors.py` (4 counts) | ~40 min |
| `analyze_results.py` | ~1 min |

Profiles (`rail3d/config.py`): `laptop` = cuda:0, mesh batch 2, chunk 1024,
train batch 256; `lab` = cuda:auto (strongest card), mesh batch 64, chunk 8192,
train batch 1024. `RAIL3D_DEVICE` always wins over the profile's device.

## What you can trust (verification summary)

- Chunked/batched Rayleigh–Sommerfeld solver ≡ verbatim Face3D code to ~1e-7
  (V1); FFT propagator ≡ the verified conv2d kernel to ~1e-6 (V2); angular-
  spectrum cross-check 0.13% (V3); specular/energy/orientation sanity (V4).
- Rev.2 defect geometry: crack orientations, band confinement, seed
  reproducibility and cross-resolution consistency (V0).
- 3D solver vs the established 2D Hankel pipeline on a uniform rail: r = 0.976
  at the 120 mm segment (V5).
- Ray-cast shadowing validated against the 2D line-of-sight ground truth; now a
  real 4.3% effect since the defect ranges widened (V6).
- Generation mesh (λ/8): defect-signal cosine mean 0.9975, min 0.9944 vs λ/16
  (V7).
- End-to-end training: score improves, pruning runs, detector centres move,
  kill-and-resume is bit-identical, `full_evaluation` covered, legacy objective
  guarded (V8).
