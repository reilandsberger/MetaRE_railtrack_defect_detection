# rail3D — Lab Workstation Setup (RTX 5090)

Step-by-step replication of the 3D rail-defect pipeline on the lab computer
(VS Code + local filesystem, GPU = **RTX 5090**; its device index varies by
machine, so the code selects it automatically — see §3). Developed and
verified on the laptop (MX250); every command below was designed to run
unchanged on the lab machine except where marked.

---

## 0. Read this first

- **Start the defect-CSV copy now** (§5a) — it is ~2 GB and everything from §4's
  V5–V7 onward needs it. It can transfer while you do §1–§3.
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
`rail3D/data/checkpoints/`) is gitignored — see §5 for how to get the data.

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
(smaller counts simply mean the §5a copy is still running). Anything wrong raises a
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
  Required by V5–V7, the V8 smoke test, and all dataset generation — see §5a.
  Only training-from-existing-shards works without it.

`config.py` fails fast with an actionable message if the torch build cannot
drive the GPU.

## 4. Run the verification suite (do this once, before anything long)

Ordered so the CSV-free gates run while the §5a copy is still transferring:

```bash
cd rail3D
python tests_physics_3d.py          # V1-V4: solver/propagator equivalence, sanity (~1 min, CPU)
```

V1–V4 need only `crosssection.png`, which is committed — they work immediately.
The rest need the defect CSVs of §5a in place:

```bash
python validation_3d.py             # V5-V7: 2D-vs-3D, shadowing, mesh convergence (~3 min GPU)
python generate_dataset_3d.py --profile lab --smoke   # 20/class + 32 intact (~1 min)
python v8_smoke_test.py             # V8: end-to-end smoke train + kill-and-resume (~5 min GPU)
```

All eight gates must PASS (they do on the laptop). Results are written to
`data/generated/verification_report.json`, which is **untracked** — running the
suite never dirties your working tree, so `git pull` stays conflict-free. The
laptop's reference numbers live in `README.md` §5.

## 5a. Get the defect CSVs onto this machine (~2 GB — start this first)

`rail3D` reads only these three folders from `RAILDEFECT_DATA_DIR`; nothing else
in the old `RailDefect/` tree is used:

| folder | contents | size |
|---|---|---|
| `data_defect_crack2/` | 5000 CSVs | ~667 MB |
| `data_defect_dent2/`  | 5000 CSVs | ~667 MB |
| `data_defect_wear2/`  | 5000 CSVs | ~667 MB |

Copy them under one parent (that parent is what `RAILDEFECT_DATA_DIR` points at).
With the source mounted as `Z:`:

```bat
robocopy Z:\RailDefect\data_defect_crack2 C:\Users\<you>\Documents\Rei\RailDefect\data_defect_crack2 /E /MT:16
```

Repeat for `data_defect_dent2` and `data_defect_wear2`. Many small files — expect
this to be slower than 2 GB of bulk data suggests, which is why it goes first.

## 5. Getting the dataset

**Primary path — generate on the 5090** (≲1 h; only the smoke set was ever
generated on the laptop). Needs `RAILDEFECT_DATA_DIR` pointing at the defect
CSV folders (§3):

```bash
python generate_dataset_3d.py --profile lab
```

Useful flags: `--status` (which shards exist/remain), `--classes crack dent`
(subset), `--limit N` (samples per class, default 5000). Generation is
resumable: shards are written atomically and existing shards are skipped, so
an interrupted run just continues on relaunch — safe on the shared
workstation.

**Rev.2 note (4 classes, finer mesh).** The dataset now has **four** defect
classes — `crack`, `dent`, `wear` and `shell` (shelling, fully parametric: it
needs no CSV folder) — plus the intact pool, and the generation mesh is λ/8
(1 mm facets) so hairline cracks are resolved. Expect **~1–2 h** on the 5090
for 5000/class, and note that **shards from before this change are
incompatible** (different geometry and label order): delete
`data/generated/rail3d_*_shard*.pt` before regenerating.

**Optional fidelity upgrade (lab only):** the laptop dataset uses the λ/4
mesh (defect-signal cosine 0.993 vs λ/12; ~15% systematic magnitude bias
shared across samples). On the 5090 you can afford λ/8: set
`MESH_DS = WVL / 8` in `rail3d/config.py`, delete/rename the old shards, and
regenerate (~4-6× the λ/4 cost — still around an hour).

### Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `git pull` → "Your local changes to the following files would be overwritten by merge" | You ran the scripts, which rewrite generated output. Everything under `rail3D/data/` is regenerated and safe to discard: `git checkout -- rail3D/data/` then pull. (Keep real edits instead with `git stash` → `git pull` → `git stash pop`.) |
| "It's running on the Intel GPU / the wrong GPU" in Task Manager | Almost always a misread — **CUDA never uses Intel integrated graphics**, and Task Manager's GPU numbering does not match CUDA's (on the lab box the 5090 is Task Manager GPU 1 but CUDA `cuda:0`). Trust the banner each script prints (`[rail3d] ... running on cuda:N (NVIDIA ...)`) and `nvidia-smi`, not Task Manager indices. Note Task Manager also hides CUDA work unless you switch a graph to the **Compute_0** engine. |
| `tests_physics_3d.py` shows no GPU activity at all | Expected: **V0–V4 run on CPU by design** (tiny meshes, ~5 s, safe on a 2 GB laptop card). Force the GPU with `RAIL3D_TEST_DEVICE=cuda:0`. V5–V7, generation and training always use the GPU. |
| `ValueError: crack: requested 5000 but only 0 CSVs` | `RAILDEFECT_DATA_DIR` unset in *this* process, wrong shell syntax (§3), an MSYS `/c/...` path, or it points inside a `data_defect_*2` folder instead of their parent |
| Reference rail width prints ~94 mm instead of 157.4 mm | wrong image loader for `crosssection.png` — see `README.md` §6.1 |
| CUDA "no kernel image available" | a non-cu128 torch got installed; re-run §2 and do not use `pip install -U` |
| Generation looks stalled | it logs every ~50 samples to `data/generated/generation.log`; `--status` lists shards done/remaining |

## 6. Training

Open the notebooks in VS Code (select the `.venv` kernel):

- `training_3d_ms_notebook.ipynb` — main run: phase-only SLM first, then the
  real meta-atom (`MetaUnitSoft`) parameterization. 400 epochs; minutes per
  run on the 5090.
- `training_3d_no_ms_notebook.ipynb` — no-metasurface baseline for the
  comparison table.
- `design_review_notebook.ipynb` / `validation_3d_notebook.ipynb` — design
  and physics review figures (already rendered to `data/figures/`).

Or headless from a terminal:

```python
python -c "
from rail3d import config, train3d
cfg = train3d.TrainConfig(run_name='ms3d_slm_v1', surface='slm', n_epoch=400,
                          batch_size=config.PROFILES['lab'].train_batch)
train3d.train(cfg)
"
```

**Interruption safety (shared workstation):** a full resumable checkpoint
(model + optimizer + RNG states + pruning state + metric history) is written
atomically to `data/checkpoints/<run_name>/latest.pt` every 10 epochs and to
`best.pt` on every validation improvement. Re-running the same `TrainConfig`
auto-resumes from `latest.pt` — verified **bit-identical** continuation in V8
(overhead ≪1%). To restart a run from scratch, delete its checkpoint folder.

Trained artifacts to bring back / compare: the checkpoint folder plus the
exported `phase.csv` / `w_pillar.csv` maps from the notebook's final cells.

## 7. What to expect

| Step | Laptop (MX250, 2 GB) | Lab (RTX 5090) |
|---|---|---|
| Physics test suite (V1–V8) | ~10 min | few min |
| Full generation, 15.5k samples (λ/4) | ~4.5 h (not planned — use the 5090) | ≲1 h |
| SLM training, 400 epochs | ~1–2 h | minutes–tens of minutes |

Profiles (`rail3d/config.py`): `laptop` = cuda:0, mesh batch 2, chunk 1024,
train batch 256; `lab` = cuda:auto (strongest card), mesh batch 64, chunk
8192, train batch 1024. `RAIL3D_DEVICE` always wins over the profile's device.

## 8. Physics/verification summary (what you can trust)

- Chunked/batched Rayleigh–Sommerfeld solver ≡ verbatim Face3D code to ~1e-7
  (V1); FFT propagator ≡ verified conv2d kernel to ~1e-6 (V2); angular-
  spectrum cross-check 0.13% (V3); specular/energy/orientation sanity (V4).
- 3D solver vs the established 2D Hankel pipeline on a uniform rail:
  Pearson r = 0.984 (V5).
- Ray-cast shadowing validated against the 2D line-of-sight ground truth;
  grazing-ray false positives eliminated (`min_t`), real crack-crater
  shadowing (~2–14%) retained (V6).
- Generation mesh (λ/4): defect-signal cosine ≥ 0.94 (mean 0.976) vs λ/8,
  intact barcode within 3% (V7).
- End-to-end training: score improves, pruning 18→8 runs, detector centers
  move, kill-and-resume bit-identical (V8).
