# rail3D — Lab Workstation Setup (RTX 5090)

Step-by-step replication of the 3D rail-defect pipeline on the lab computer
(VS Code + local filesystem, GPU = **RTX 5090 on device index 1**). Developed
and verified on the laptop (MX250); every command below was designed to run
unchanged on the lab machine except where marked.

---

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

Sanity check (should print the 5090's compute capability `(12, 0)` without
errors):

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(1), torch.cuda.get_device_capability(1))"
```

## 3. Point the code at the right GPU and data

Two environment variables (set them in the VS Code terminal, or a `.env`):

```bash
set RAIL3D_DEVICE=cuda:1
set RAILDEFECT_DATA_DIR=C:\path\to\RailDefect\RailDefect
```

- `RAIL3D_DEVICE=cuda:1` — the 5090 is GPU 1 on the workstation. Every
  device call in `rail3d/config.py` routes through this; nothing uses a bare
  `"cuda"`.
- `RAILDEFECT_DATA_DIR` — folder containing the defect CSV datasets
  (`data_defect_crack2/`, `data_defect_dent2/`, `data_defect_wear2/`).
  **Only needed to (re)generate fields** — training runs entirely from the
  generated `.pt` shards. Copy the three CSV folders (~a few hundred MB) if
  you want to regenerate on the lab machine.

`config.py` fails fast with an actionable message if the torch build cannot
drive the GPU.

## 4. Run the verification suite (do this once, before anything long)

```bash
cd rail3D
python tests_physics_3d.py          # V1-V4: solver/propagator equivalence, sanity  (~1 min, CPU)
python validation_3d.py             # V5-V7: 2D-vs-3D, shadowing, mesh convergence (~3 min GPU)
python v8_smoke_test.py             # V8: end-to-end smoke train + kill-and-resume  (~5 min GPU)
```

All eight gates must PASS (they do on the laptop; results are appended to
`data/generated/verification_report.json`). V8 needs the smoke dataset first:

```bash
python generate_dataset_3d.py --profile lab --smoke
```

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

**Optional fidelity upgrade (lab only):** the laptop dataset uses the λ/4
mesh (defect-signal cosine 0.993 vs λ/12; ~15% systematic magnitude bias
shared across samples). On the 5090 you can afford λ/8: set
`MESH_DS = WVL / 8` in `rail3d/config.py`, delete/rename the old shards, and
regenerate (~4-6× the λ/4 cost — still around an hour).

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
train batch 256; `lab` = cuda:1, mesh batch 64, chunk 8192, train batch 1024.
`RAIL3D_DEVICE` always wins over the profile's device.

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
