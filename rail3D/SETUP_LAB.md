# rail3D — Lab Workstation Setup (RTX 5090)

*Last updated: 2026-09-12 · λ = 5 mm (60 GHz), V0–V8 measured on the 5090 — bump this line in any
commit that changes behaviour this file describes. Timings are now MEASURED
at λ=5 on the lab 5090 (2026-09-04), not extrapolated.*

Step-by-step replication of the 3D rail-defect pipeline on the lab computer
(VS Code + local filesystem, GPU = **RTX 5090**; its device index varies by
machine, so the code selects it automatically — see §3).

## Which document do I use?

| you want to… | use |
|---|---|
| run the experiment day to day, terminal, unattended jobs | **this file** — §A below is the short loop |
| understand *why* each choice was made, see figures inline | **`rail3D_pipeline.ipynb`** — the same steps, narrated, with plots |
| know the conventions and the traps before editing code | `README.md` §4 (conventions) and §6 (hard-won findings) |

They do not duplicate logic: the notebook calls the same scripts documented
here, so neither can drift from the other.

---

## A. The short loop (machine already set up)

If §1–§4 are done, this is the whole cycle. **Every command prints what it is
using**, so read the banners rather than assuming.

```bash
cd ~/Documents/Rei/MetaRE_railtrack_defect_detection && git checkout -- rail3D/data/ && git pull
```

```bash
export RAILDEFECT_DATA_DIR='C:/Users/ct2443/Downloads/RailDefect/RailDefect'
```

```bash
cd rail3D && python preflight.py
```

Pre-flight lists **every dataset on disk with its creation date, geometry and
git commit**, marks the one that would be used, and stops on anything
inconsistent. Then generate (naming it), inspect, train, analyse:

```bash
python generate_dataset_3d.py --profile lab --name L5_H150_v1 --note "lambda=5, H=150, 4 classes"
```

```bash
python inspect_dataset.py --root data/generated/L5_H150_v1
```

```bash
python -c "from rail3d import config, train3d; train3d.train(train3d.TrainConfig(run_name='ms3d_slm_l5_v1', surface='slm', n_epoch=1200, batch_size=config.PROFILES['lab'].train_batch, data_root='data/generated/L5_H150_v1'))"
```

```bash
python analyze_results.py --run-name ms3d_slm_l5_v1 --data-root data/generated/L5_H150_v1
```

### Keeping several datasets

`--name X` writes to `data/generated/X/` instead of the default
`data/generated/`. Use it whenever you change geometry, so old and new datasets
coexist and can be compared instead of overwriting each other:

```bash
python generate_dataset_3d.py --profile lab --name L5_H200_v1 --note "plane at 40*lambda"
```

Point training at one with `TrainConfig(..., data_root='data/generated/L5_H200_v1')`.
Every dataset carries a `dataset_config.json` recording its geometry (36 keys:
wavelength, grid, plane, horn, defect ranges, seed...), creation time, git
commit and host — and **training refuses a dataset whose geometry differs from
the current config**, the **generator refuses to write into a root holding
another geometry's artifacts**, and **checkpoints refuse to resume across a
geometry change**. A stale artifact cannot silently produce numbers.

### Knowing what is feeding training

Training prints a banner before the first epoch:

```
========================================================================
run     : ms3d_slm_l5_v1   surface=slm  objective=rank/cos  epochs=1200  batch=1024
device  : cuda:0 (NVIDIA GeForce RTX 5090)
detectors: start (13, 10) -> 8 final, fraction schedule (prune 60-150)
optics  : 1 layer(s), distances (100.0,) mm
dataset : .../data/generated/L5_H150_v1
created : 2026-08-17 ...  (0.4 h ago)  commit ...  host ...
geometry: H_MS=150.0 mm  grid 60x30  seg=120.0 mm  mesh=lambda/8  shadow=raycast
classes : ['crack', 'dent', 'wear', 'shell']
samples : {'crack': 5000, ...}  (total 20512)
splits  : train 16409  val 2051  test 2052   intact 409/51/52
========================================================================
```

If the dataset line or its date is not what you expect, stop there. You can also
query any dataset directly:

```bash
python -c "from rail3d import data3d; print(data3d.describe_dataset('data/generated/L5_H150_v1'))"
```

---

## B. λ=5 bring-up runbook (one-time, after `git pull` of the 2026-08-17 migration)

The code is now λ=5 mm (60 GHz); every λ=8 dataset and checkpoint on this
machine is stale and will be **refused, not deleted**. Run these in order and
send the listed files back for review before the full generation.

```bash
cd ~/Documents/Rei/MetaRE_railtrack_defect_detection && git checkout -- rail3D/data/ && git pull
export RAILDEFECT_DATA_DIR='C:/Users/<you>/Downloads/RailDefect/RailDefect'
cd rail3D
python preflight.py 2>&1 | tee ../preflight_l5.txt
```

**Expected: exit 1.** The λ=8 datasets are flagged with per-key geometry
mismatches — that is the guard working, not a bug. The fix is generating a λ=5
set (below), never deleting the old ones.

```bash
python generate_dataset_3d.py --profile lab --smoke 2>&1 | tee ../guard_demo.txt
```

**Expected: REFUSAL** — the old λ=8 smoke set occupies `data/generated/smoke/`
and the generator now refuses mixed-geometry roots. Then clear it and run the
whole verification chain (lab_report regenerates the smoke set itself):

```bash
rm -rf data/generated/smoke
python lab_report.py 2>&1 | tee ../lab_report_console.txt
```

~10–15 min: V0/V0b/V0c/V1–V4, **V5–V7 at λ=5 (the pending gates)**, smoke
generation with measured samples/s, stored-shard stats, untrained-optics
separability (through the fixed 130-window lattice), V8 with all new gates.
Then the geometry re-scan and the diagram:

```bash
python scan_geometry.py 2>&1 | tee ../scan_geometry_console.txt
python setup_diagram.py
```

Then the two studies that produce the comparison figures (both are
informational, neither gates):

```bash
python compare_wavefronts.py --profile lab --export-case data/generated/fdtd_case
```

For a **full-wave reference run**, export the plane-wave, closed-body variant
instead — that is the form a MoM solver wants (see §13):

```bash
python compare_wavefronts.py --sample intact --source plane --export-closed --export-only --export-case data/generated/mom_case_intact
python compare_wavefronts.py --sample crack  --source plane --export-closed --export-only --export-case data/generated/mom_case_crack
```

`--export-only` skips the variant solves (exporting is instant; solving is
minutes). Export **both** — the primary comparison is the difference field.
```bash
python validation_3d.py --min-t 0.05 0.5 1.5 3.0 --normal-offset 0 0.05 0.15 0.3 0.6 --crack-idx 0 1 17 2>&1 | tee ../v6b_grid.txt
```

The first solves one sample every way (λ=8 vs λ=5, physics terms, shadow guard,
mesh resolution) and writes `wavefront_cuts.png` / `_maps.png` / `_metrics.png`
plus `wavefront_fields.npz`, and exports an FDTD-ready case bundle.

The second decides `SHADOW_MIN_T` **and** `SHADOW_NORMAL_OFFSET` together
(~10 min, 20 rows). **Do not pick "the smallest artifact-safe min_t"** — that
was the pre-2026-09-04 heuristic and it is meaningless when every benefit is
zero. The script now selects for you: the configuration keeping the **most real
crack shadowing** while the mean intact artifact stays under 0.03, and it prints
`NO tested configuration…` rather than a number when nothing achieves both.
That "None" is a result, not a failure — it authorises `SHADOW_MODE = "none"`,
which halves generation.

Why these values:

| axis | why |
|---|---|
| `--normal-offset 0` | control — reproduces the old single-axis behaviour |
| `0.05` | ≈ the worst-case sagitta `d²/(8R)` = 0.06 mm; the minimum that could work |
| `0.15` | current default, ~2.5× sagitta |
| `0.3` | headroom, still ≪ 1.5 mm (the nearest real crater wall) |
| `0.6` | deliberately too large — the benefit column *should* collapse. If it does not, the knob is not doing what we think |
| `--min-t 0.05` | the token value a working normal offset should permit |
| `0.5, 1.5` | the band where crack shadowing appeared in the old sweep |
| `3.0` | current default, for continuity with every prior result |
| `--crack-idx 0 1 17` | **17 is the deepest crack in the set (9.57 mm).** The benefit column is the whole point, and indices 0 (2.72 mm) and 1 (5.28 mm) sit at or below the mean depth. The old harness probed index 0 alone — a crack that self-shadows at no setting — and concluded the mechanism was inert |

Read the console's `probing cracks: c0 depth … c1 depth …` line to confirm what
was actually tested, and compare `artifact mean` (the gate, field level) against
`barcode` (through the detectors) — if they diverge by orders of magnitude, say
so rather than moving the gate.

**Neither `SHADOW_MIN_T` nor `SHADOW_NORMAL_OFFSET` may change after generation**
— both are provenance keys, so every dataset and checkpoint built from them is
refused when they move. Settle this BEFORE the generation step.

> Interpreting what comes back: **`READING_RESULTS.md`** gives the pass rule
> for every gate, what each field means, and the traps (V6's `worst_rel_l2`
> is a max over 8 samples; V8's `best_val` is degenerate on a smoke split;
> the V6b sweep and `compare_wavefronts` both default to `crack[0]`, which
> is a near-minimum-depth crack that never self-shadows).

**Send back (attach):** `data/generated/lab_report.md`,
`data/generated/wavefront_comparison.json`, `data/figures/wavefront_cuts.png`,
`data/figures/v6b_min_t_sweep.png`, `../v6b_grid.txt`,
`data/generated/verification_report.json`, `data/generated/geometry_scan.json`,
`../preflight_l5.txt`, `../guard_demo.txt`,
`data/generated/smoke/dataset_config.json` + `generation.log`,
`data/figures/setup_diagram.png`. Review criteria: every gate PASS (V6's
`resolved` bound is the one legitimately at-risk gate — 2 mm hairlines are now
0.4λ; if it trips that is a physics finding to discuss, not a bug); the H scan
confirming (or moving) 30λ; measured samples/s → the real full-generation time.
**Only after sign-off:** §8's full generation with `--name L5_H150_v1`.

---

## 0. Read this first

- **Start the defect-CSV copy now** (§4) — it is ~2 GB and every step from §5
  onward needs it. It can transfer while you do §1–§3.
- **Run `python preflight.py` before anything** — it verifies code, GPU, CSVs, dataset geometry and checkpoints in seconds, and prints the exact fix for anything wrong.
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
python -c "from rail3d import config, sections; print(config.RAILDEFECT_DIR, config.RAILDEFECT_DIR.is_absolute()); print({c: len(sections.get_dataset_files(c)) for c in config.DATASET_DIRS})"
```

Expect an absolute path, `True`, and `{'crack': 5000, 'dent': 5000, 'wear': 5000}`
(smaller counts simply mean the §4 copy is still running). Anything wrong raises a
`FileNotFoundError` naming the specific problem. (Iterate `config.DATASET_DIRS`,
not `CLASS_NAMES` — `shell` is parametric, has no CSV folder, and would raise.)

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

## 5. Verify the machine — two commands

**Always run this first.** It is fast (seconds) and catches every mistake that
has bitten so far: stale code, wrong GPU, missing CSVs, a dataset generated with
a different geometry, shards mixed from separate runs, checkpoints that cannot
be loaded. Exit code 0 = safe to proceed.

```bash
cd rail3D && python preflight.py
```

Then the full physics + training verification (~10–15 min estimated on the
5090 at λ=5; was ~5 min at λ=8):

```bash
python lab_report.py
```

Runs V0/V0b/V0c/V1–V4, V5–V7, a smoke generation, stored-shard statistics, a
class-separability check and V8, then writes `data/generated/lab_report.md`.
Every gate must say PASS. `--quick` skips the slow V5–V7 gates; `--skip-smoke`
reuses an existing smoke set; `--smoke-n N` sizes the smoke set.

### Reading the smoke/verification report

The V8 block is the one that needs interpretation — the smoke set is tiny by
design, so know which numbers are signal and which are noise:

| field | meaning | healthy |
|---|---|---|
| `train_separation_first/last` | mean (defect gap − intact gap) on TRAIN through hard windows — exactly what the rank objective optimizes, 64 samples behind it | `last > 1.2 × first` (`separation_improves`) |
| `capture_first/last` | fraction of plane power on the retained windows | starts ~0.9+ (dense tiling), drops as windows are pruned, must stay in (0, 1]; climbs during the stationary phase of REAL runs |
| `min_separation_mm` | closest pair of trained detector centres | `> DX` (2.5 mm) — below one pixel means two windows read the same spot (duplicate barcode entries) |
| `u_max_change` | movement of the SURVIVING detectors vs their original lattice sites (keep-index verified) | `> 1e-4` — gradients genuinely reach the positions |
| `resume_max_metric_mismatch` | run-B-resumed vs run-A-straight, epochs 15–29 | **exactly 0.0** — any nonzero means RNG/checkpoint state is broken |
| `variance_criterion_ok`, `legacy_objective_ok`, `stale_ckpt_refused` | the legacy prune criterion trains, the legacy margin loss trains, a doctored geometry stamp is refused | all true |
| `best_val` AUC / class_acc / FPR | **noise at smoke scale** — 8 defect + 3 intact val samples quantize AUC to 1/24 and pin 4-class accuracy at chance | ignore on smoke; meaningful on the full set |
| raw `loss_first/last` | NOT comparable across the run — the τ anneal and pruning reshape the objective mid-run | judge separation, not loss |

Rule of thumb for "was a change good": train separation up, capture not
collapsing to ~0, min separation above a pixel, resume mismatch exactly zero.
Anything about absolute detection quality waits for the full dataset.

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

## 7. Measurement-plane geometry — RE-SCANNED at λ=5, 30λ confirmed

```bash
python scan_geometry.py 2>&1 | tee ../scan_geometry_console.txt
```

**Re-scanned on the 5090, 2026-09-04: `H_MS = 30λ = 150 mm` is confirmed**
(best on-axis field AUC 0.878, vs 0.840 at 20λ and 0.874 at 40λ), and the
dark-field structure reproduces exactly — H = 10λ puts the specular lobe inside
the aperture and collapses to **0.523 field AUC, near chance, with 20× the
energy**. Keep `x_center = 0`: specular-centred scores higher on *field* AUC
(0.884) but far lower through a fixed readout (det AUC 0.666). The full λ=5
table is in `rail3d/config.py` above `H_MS`.

One result worth knowing: an **80×80 aperture scores best of all** (field 0.889
/ det 0.893) but costs 3.6× the plane points and therefore 3.6× the generation
time — not taken for +0.011 AUC, but it is the first thing to revisit if crack
recall proves limiting.

### Exploring geometries with bigger smoke sets

The default 20/class smoke is a plumbing check, not a geometry discriminator.
For real geometry comparisons on the 5090 use 3–4× sets, one named root per
candidate — each self-describes via its `dataset_config.json`:

```bash
python generate_dataset_3d.py --profile lab --smoke --smoke-n 80 --name geo_H150 --note "H=30*lambda"
# edit config H_MS, then:
python generate_dataset_3d.py --profile lab --smoke --smoke-n 80 --name geo_H200 --note "H=40*lambda"
```

(`--smoke-n 80` → 80/class + 128 intact ≈ 448 samples, a few minutes each on
the 5090.) `scan_geometry.py --n 40` sharpens the field-level AUC the same way
without storing datasets. Compare with `inspect_dataset.py --root ...` and by
training against each root (`TrainConfig(data_root=...)`).

## 7b. The one-command workflow (preferred)

`run_stage.py` runs generate → gate → train → analyse as a single resumable
command, using `config.STAGES` for both dataset size and training schedule:

```bash
python run_stage.py --stage prelim --with-baseline --bundle 2>&1 | tee ../prelim.log
```

| flag | effect |
|---|---|
| `--stage prelim` | 2000/class + 400 intact, ~1.8 h generation, ~3 min training |
| `--stage full` | 5000/class + 512 intact, ~3.5 h generation |
| `--with-baseline` | also trains the no-metasurface control |
| `--fresh` | discard checkpoints whose SCHEDULE differs from this stage |
| `--skip-generate` | dataset already exists |
| `--bundle` | zip the files worth sending for review |

**It refuses rather than warns.** Between generation and training it checks the
things §13 and `READING_RESULTS.md` document as historically misread — geometry
provenance key by key, `_counts` complete against the requested size (an
interrupted generation otherwise *looks* finished), every class's first shard
loadable and finite — and stops before the next long step. A checkpoint whose
schedule differs from the stage is refused too, because resuming across that
leaves the τ anneal and prune window half-applied.

Everything is resumable: shards are atomic and skipped, training resumes from
the 10-epoch checkpoint. Re-run the identical command to continue.

It ends with `data/generated/stage_<stage>_report.md` and an explicit list of
files to send back. Two numbers it reports without being asked: **`n_det` at the
best checkpoint** (if that is not `N_DET_FINAL`, every downstream number
describes the dense array — README finding 22) and **`capture_frac` against
`n_det/130`** (above it the metasurface is concentrating light, at it the
capture term is idle).

`rail3D_pipeline.ipynb` remains the narrated route, and its section 9 reads the
same results from a fresh kernel — see its section 0 for which cells to run.

---

## 8. Generate the dataset

```bash
python generate_dataset_3d.py --profile lab --name L5_H150_v1 --note "lambda=5, H=150"
```

Estimated **~1.5–2 h** on the 5090 for 4 classes × 5000 + 512 intact: the λ=8
run measured 9.47 samples/s (~36 min), and λ/8 facets at λ=5 mean (8/5)² ≈
2.56× the face count. §B's smoke run measures the true samples/s — trust that
extrapolation over this estimate. Resumable: shards are written atomically and
existing ones are skipped, so an interrupt just continues on relaunch — safe
on a shared workstation.

Flags: `--status` (which shards exist/remain), `--classes crack dent` (subset),
`--limit N` (samples per class), `--smoke [--smoke-n N]` (small set into
`data/generated/smoke/`), `--name X` (write to `data/generated/X/`).

**Roots holding another geometry's artifacts are refused automatically**
(`check_generation_root`) — you no longer need to remember to delete stale
shards; use `--name` for a fresh root and keep old sets as the record.

## 9. Check the data before training on it

```bash
python inspect_dataset.py --root data/generated
```

Re-derives each stored sample's geometry from its seed and shows it beside the
stored field, plus the mean defect-minus-intact intensity and per-class
parameter statistics. Confirm the depth/length/angle ranges match `README.md` §2.

## 10. Train

Open `training_3d_ms_notebook.ipynb` (select the `.venv` kernel): the
phase-only `SLM2D` run. (`MetaUnitSoft` is **blocked at λ=5** — the meta-atom
library is an 8 mm-band fit; the notebook's MetaUnit cells raise by design
until a 60 GHz library is fitted.) `training_3d_no_ms_notebook.ipynb` is the
no-metasurface baseline. Headless equivalent:

```bash
python -c "from rail3d import config, train3d; train3d.train(train3d.TrainConfig(run_name='ms3d_slm_l5_v1', surface='slm', n_epoch=1200, batch_size=config.PROFILES['lab'].train_batch, data_root='data/generated/L5_H150_v1'))"
```

~10 min at 1200 epochs (λ=8 measurement; training cost is dominated by the
plane size, which is unchanged at 60×30). Training auto-resumes from
`data/checkpoints/<run_name>/latest.pt` — and now **refuses** a checkpoint
whose geometry stamp or surface/objective/layer-distances differ; delete the
run folder to start over.

### How the detectors are optimized

Positions are **trained, not swept**. `SoftDetector2D` holds the window centres
as an `nn.Parameter` with sigmoid-edged (differentiable) masks, so gradients
move them; redundancy-based pruning (each detector valued by its UNIQUE
contribution, `std × (1 − max|corr| to survivors)`) runs *inside* the same
training run. With the default `TrainConfig` schedule one run does:

| epochs | what happens |
|---|---|
| 0–60 | full 130-window dense start, soft masks (large τ), phase + positions training |
| 60–150 | pruning window: geometric reduction to the final count (keep 0.75/step), positions still moving; power floor recomputed at each prune |
| 150–250 | τ anneal completes, masks sharpen toward the 0.5-px floor |
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

Starts from a dense 13×10 = 130-detector grid and prunes to each final count,
one full training run per count. Writes `data/figures/detector_sweep.png` (AUC,
class accuracy, redundancy/throughput, and the surviving layout) plus
`data/generated/detector_sweep.json`.

**Why start dense.** Detector density relative to the reference implementation
(the λ=8 rows are history; the λ=5 row is the exact 5/8-scaled replica of the
λ=8 dense start — same count, same coverage):

| layout | detectors | pitch | window gap | coverage |
|---|---|---|---|---|
| Face3D 6×6 (λ=8) | 36 | 48 × 48 mm | +29.8 / +36.8 mm | 7.2% |
| rail3D old 6×3 (λ=8) | 18 | 36 × 36 mm | +17.8 / +24.8 mm | 12.7% |
| **rail3D 13×10 (λ=5, now)** | 130 | 10.7 × 6.8 mm | −0.66 / −0.18 mm | **92%** |

The 13×10 is not arbitrary — it is the **tiling bound**
`floor(aperture / window)` (derived in config): windows at window-sized pitch
are the densest USEFUL start, since anything closer only creates duplicates.
The field's speckle grain is **5.0 mm along x, measured** (8.0 mm at λ=8), so
the 11.4 mm window spans ≈2.3 grains — the same ratio as the validated λ=8
design, and the binding scale for spacing. README findings 18–19 work this
through; the aperture holds ~30 independent speckle cells across x, so the 130
overlapping windows oversample on purpose: the dense start exists to give
pruning a rich candidate set, not to add information.
Pruning from the tiling lattice lets training
select from a rich candidate set instead of a handful of fixed spots. It
requires the **redundancy** criterion: overlapping windows have near-identical
variance, so Face3D's variance ranking cannot tell a duplicate from a uniquely
informative detector (README §6 finding 13). Compare the two directly with
`--prune-criterion variance`.

Detector *positions* are trained, not swept — see §10. Watch `mean |corr|` in
the third panel: if it stays high as the count falls, pruning kept duplicates.

Add `--dist 80 100 120` to sweep the metasurface→detector distance at the same
time (the config default is 20λ = 100 mm) — that is a training-time
propagation, so it needs no regeneration.

## 12. Where does it succeed and fail?

```bash
python analyze_results.py --run-name ms3d_slm_l5_v1 --data-root data/generated/L5_H150_v1
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

**All λ=5 rows are ESTIMATES** — measured λ=8 values × the (8/5)² ≈ 2.56× face
count where generation-bound; §B's smoke run gives the real samples/s.

| Step | λ=8 measured | λ=5 estimate |
|---|---|---|
| `lab_report.py` (V0–V8 + smoke) | ~5 min | ~10–15 min |
| `setup_diagram.py` | seconds (CPU) | same |
| `scan_geometry.py` (default configs) | ~1 min | ~3 min |
| `generate_dataset_3d.py --profile lab` | ~36 min (9.47 samples/s) | **~1.5–2 h** |
| `inspect_dataset.py` | ~10 s | ~15 s |
| training, 1200 epochs | ~10 min | ~10 min (plane size unchanged) |
| `sweep_detectors.py` (4 counts) | ~40 min | ~40 min |
| `analyze_results.py` | ~1 min | ~1 min |

Profiles (`rail3d/config.py`): `laptop` = cuda:0, mesh batch 2, chunk 1024,
train batch 256; `lab` = cuda:auto (strongest card), mesh batch 64, chunk 8192,
train batch 1024. `RAIL3D_DEVICE` always wins over the profile's device.

## What you can trust (verification summary)

Verified at **λ=5** on the laptop (2026-08-17):
- Chunked/batched Rayleigh–Sommerfeld solver ≡ verbatim Face3D code to ~5e-7
  (V1); FFT propagator ≡ the verified conv2d kernel to ~1e-6 (V2); angular-
  spectrum cross-check 0.10% — identical to the λ=8 value to 6 significant
  figures, confirming the scaled replica (V3); specular/energy/orientation
  sanity (V4).
- Rev.2 defect geometry: crack orientations, band confinement, seed
  reproducibility and cross-resolution consistency (V0); redundancy pruning
  beats variance on the dense-layout failure case (V0b); every staleness guard
  refuses what it must and passes what it must (V0c).

Measured at **λ=5 on the lab 5090** (2026-09-04):
- 3D solver vs the established 2D Hankel pipeline on a uniform rail:
  **r = 0.958** (V5), down from 0.976 at λ=8 — the figure shows envelope
  agreement with a slight fringe offset (finite 120 mm segment vs infinite
  extrusion), not a solver disagreement.
- Generation mesh (λ/8): defect-signal cosine **min 0.9948, mean 0.9982** vs
  λ/16, all six cases (V7). Mesh converged.
- End-to-end training: separation 0.0290 → 0.0562, 130→8 pruning,
  keep-index-verified detector movement, min separation 6.13 mm, capture
  bounded, kill-and-resume bit-identical (mismatch exactly 0.0), legacy
  objective + variance criterion + stale-checkpoint refusal all guarded (V8).

**What you can NOT yet trust:**
- Ray-cast shadowing is **inert** at `SHADOW_MIN_T = 3.0` — V6 passes but
  `crack_shadow_with_resolved_occluder = 0.0` and 7 of 8 samples are exactly
  0.0. Narrow-crater self-shadowing is absent from the data (README finding 3).
- Anything about TRAINED performance at λ=5: only the 30-epoch smoke run
  exists, whose validation split is 8 defect / 3 intact and therefore noise.

---

## 13. Full-wave (MoM) cross-check — which solver, and how to feed it

**The question this answers:** our solver is physical optics — scalar, PEC
tangent-plane currents, single + double bounce. At λ=5 mm the crack widths are
2–5 mm = **0.4–1λ**, which is precisely where that approximation is expected to
break. V1–V3 only prove we solve *our own* integral correctly; V5 compares
against the 2D code, which shares the assumption. A full-wave run is the only
way to measure what PO misses.

**Use MoM/MLFMM, not Zemax and not full-scene FDTD.**

| tool | verdict |
|---|---|
| **Ansys HFSS-IE** (Integral Equation solver) | **recommended.** Needs the IE licence — an HFSS FEM seat alone will not run it |
| **Altair FEKO** (MLFMM) | equally good; use if HFSS-IE is not licensed |
| Ansys HFSS (FEM) | volumetric; would mesh 150 mm of empty air to the plane |
| Ansys HFSS SBR+ | ray PO + PTD edges — an upgrade on our model, not an independent check |
| Ansys Lumerical FDTD | valid physics, wrong shape: **387 Mcells / ~39 GB** for the full scene. Only viable boxed around the rail (23 Mcells for a 30 mm segment) with a near-field projection to the plane |
| Ansys Zemax OpticStudio | **not valid.** Its POP is scalar Fresnel/Kirchhoff — the same approximation family as our code — so agreement proves nothing |

**Problem size and why the segment length is not free:**

| geometry (closed body, mesh λ/8 as exported) | triangles | RWG unknowns at λ/10 | cut-end illumination |
|---|---|---|---|
| 30 mm segment | 39.5k | 226k | **0.80x mid-span** — truncation-confounded |
| 60 mm segment | 78k | 302k | 0.35x — still hot |
| 80 mm segment | 104k | 379k | 0.16x |
| **120 mm segment (production `SEG_LEN`)** | **156k** | **562k** | **0.08x** |

Closing the body roughly **doubles** the unknowns (the unlit end caps and
underside get meshed too) — that is the price of modelling an opaque rail
instead of an infinitely thin sheet. The dense MoM matrix at 562k unknowns is
**5.05 TB**, so MLFMM is not optional. The exported λ/8 mesh is 233k unknowns
and can be run as-is if λ/10 is too heavy.

**Use the production `SEG_LEN` = 120 mm — do NOT shorten the rail to save
unknowns.** Measured on the exported mesh (`case.json` → `truncation`), the
illuminated power per unit rail length within 1λ of the cut end, relative to
mid-span, is **0.80x at 30 mm** and only **0.08x at 120 mm**. Our PO solver has
**no edge diffraction at all**; a MoM or FDTD reference has plenty. A brightly
lit cut end makes the reference diffract off a truncation the real rail does not
have, and that disagreement gets misread as "PO fails on the defect". The
120 mm truncation study that justified `SEG_LEN` measured the *defect signal*
(a difference, where the common edge contribution cancels) — it does not license
a short segment for an absolute-field comparison.

**So compare the difference field.** Export `intact` and the defect at the same
segment length, and compare `E_defect − E_intact` between solvers as the primary
metric, with absolute fields secondary. That is also the quantity the detector
barcodes actually respond to.

**Ladder — one unknown at a time. Do not start at the bottom.**

1. **Flat PEC plate**, plane wave, 55° incidence. If this disagrees, the setup
   is wrong, not the physics.
2. **Intact rail**, plane wave — tests PO currents on a curved surface.
3. **Cracked rail**, plane wave — the real question.
4. **Real horn**, only after 1–3 agree.

**Setting it up.** `--export-closed` writes a watertight body (MoM puts current
on *both* faces of an open sheet, which is not what an opaque rail does), and
`--source plane` removes the horn aperture model as a confound. Import
`rail_surface.stl` as **millimetres** — STL carries no units. `case.json` has
the source, the cell-centred plane grid (`x_i = -Wx/2 + (i+0.5)*dx` — sample
exactly these points, not a node-centred grid), and a `conventions` block.

**Two traps that look like physics disagreements and are not:**

- **Time convention.** rail3D uses `exp(-iωt)`, so outgoing waves carry
  `exp(+ik₀R)`. HFSS, FEKO and Lumerical use `exp(+jωt)` → `exp(-jk₀R)`, so
  their fields arrive **conjugated**. `--external` detects this and *tells you*
  which convention matched; it does not silently fix it.
- **Polarisation.** rail3D is **scalar**. Export ONE component from the vector
  solver — `E_y` (E along the rail axis, TE) is the cleanest match — and set the
  incident polarisation to match it.

Absolute amplitude does not matter: one complex gain α is fitted over the whole
plane before differencing, so `complex_corr` is the number to quote.

**Returning the result:**

```python
import numpy as np
np.savez("mom_result.npz", field=E.astype(np.complex64), label="HFSS-IE")
```

```bash
python compare_wavefronts.py --profile lab --source plane --external mom_result.npz
```

It joins every figure and metric, with the alignment printed. Expect PO and
full-wave to differ at sub-wavelength features, grazing faces, the cut-end
edges, beyond the second bounce, and anywhere polarisation matters — those are
the interesting comparisons, not failures to hide.
