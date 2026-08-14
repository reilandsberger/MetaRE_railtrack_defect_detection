# CLAUDE.md — MetaRE railtrack defect detection

Guidance for Claude Code sessions in this repo. Written for cold-start sessions on
smaller models: read this, then `rail3D/README.md` (the full handoff doc), before
touching anything.

## What this repo is

Trainable metasurface + detector "barcode" system for rail defect detection.
- **Legacy 2D pipeline** (repo root): `2Dmesh_from_vertex.py`, `training_ms_notebook.ipynb`,
  `training_no_ms_notebook.ipynb` — 2D Hankel boundary integrals at λ=12 mm. Kept as
  reference/ground truth; do not modify.
- **Active 3D pipeline**: everything under `rail3D/` on branch **`3D_railhead_upgrade`**.
  Physics ported from the experimentally-validated Face3D_clean codebase
  (`C:\Users\Rei\Downloads\Face3D_clean\Face3D_clean`, read-only reference), λ=8 mm.

## Ground rules

- **`rail3D/README.md` §4 (conventions) and §6 (hard-won findings) are binding.**
  Several past bugs look like harmless cleanups (image loader choice, ray-cast min_t,
  meshgrid indexing, RNG-state handling). Do not "simplify" physics code without
  re-running the verification gates.
- Verification gates: `rail3D/tests_physics_3d.py` (V1–V4), `rail3D/validation_3d.py`
  (V5–V7), `rail3D/v8_smoke_test.py` (V8). Any physics/geometry/training change must
  keep them green; they are laptop-safe (small meshes).
- Units: mm everywhere in rail3d (legacy 2D scripts used µm). Axes: x across railhead,
  y along rail, z up, crown at z=0. Label order: crack=0, dent=1, wear=2, shell=3.
- Laptop python: `C:\Users\Rei\Downloads\RailDefect\RailDefect\.venv\Scripts\python.exe`
  (torch 2.7.1+cu118, MX250 2 GB — small batches only; no full generation/training here).
  Lab workstation: RTX 5090, cu128 torch, auto-selected via `cuda:auto`
  (`config.best_cuda_device()`); setup in `rail3D/SETUP_LAB.md`.
- **GPU identity**: never trust Task Manager's GPU indices — they differ from
  CUDA's, and CUDA cannot use Intel integrated graphics at all. Every entry
  point prints a `[rail3d] ... running on cuda:N (NVIDIA ...)` banner; the
  generator also logs the GPU name and PID so it can be matched to `nvidia-smi`.
  `tests_physics_3d.py` (V0–V4) deliberately defaults to **CPU** (laptop-safe);
  override with `RAIL3D_TEST_DEVICE`.
- Defect CSVs live outside the repo: `RAILDEFECT_DATA_DIR` env var (see SETUP_LAB §3;
  read at import time). Generated data/checkpoints are gitignored under `rail3D/data/`.
- Commit on `3D_railhead_upgrade`; bulk `.pt` data never gets committed.

## Current state (2026-08-14 — rev.2 rework complete on the laptop)

Per approved plan (`~/.claude/plans/i-am-noticing-on-imperative-dewdrop.md`):

- **Geometry (done):** `mesh3d.py` rebuilt around per-point depth fields
  `displaced(s,y) = intact(s) − d(s,y)·n̂(s)`; per-class parametric generators
  (oriented hairline cracks incl. transverse/oblique, 2D-Gaussian dents/chains,
  shoulder-band wear, NEW parametric `shell` class — ranges cited from Ye et al.
  2018 Table 1 and Ye et al. 2023 Fig 9, PDFs in `C:\Users\Rei\Downloads\`);
  resolution-independent params (`sample_defect_params` → `render_depth_field`)
  so the fine sim mesh and the coarse ray-cast occluder render the same defect;
  `MESH_DS` now λ/8 (hairlines need it); 4-class plumbing throughout.
- **Objective (done):** default `TrainConfig(objective="rank", metric="cos",
  target_fpr=0.05)` — pairwise soft-AUC hinge on normalized barcodes, threshold
  calibrated on the validation intact spread; `objective="margin"` restores the
  legacy loss and V8 guards it. See README §6 finding #10 for why.
- **Verified on the laptop:** V0–V4 pass; 4-class smoke set generated; V8 passes
  (AUC 1.0 on the tiny val split, resume bit-identical, full_evaluation covered,
  legacy objective runs). V5–V7 rerun at the new resolution was the last step.
- **Remaining (lab 5090):** `git pull` → regenerate the full 4-class dataset
  (`--profile lab`, λ/8 ≈ 1–2 h, resumable) → retrain MS + no-MS notebooks.
  Watch crack recall specifically (hairlines = hardest signal).
- Known quirks to keep: wear CSV depths are shallow (~0.6–1 mm) after the
  20th-percentile baseline correction in `extract_csv_profile` (width-matching
  inflates loops — the correction is required); shell samples may land on the
  vertical head side (x≈38.5) — geometrically intended; on tiny smoke splits the
  calibrated FPR is degenerate (3 intact val samples), fine on the real set.

## Typical commands (laptop)

```bash
cd rail3D
python tests_physics_3d.py                       # V1-V4 (~1 min, CPU)
python validation_3d.py                          # V5-V7 (GPU, minutes)
python generate_dataset_3d.py --profile laptop --smoke
python v8_smoke_test.py                          # V8 end-to-end + resume test
python setup_diagram.py                          # review figures
```
