# CLAUDE.md — MetaRE railtrack defect detection

*Last updated: 2026-09-04 · λ = 5 mm (60 GHz) · V0–V8 measured on the lab 5090.*

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
  (`C:\Users\Rei\Downloads\Face3D_clean\Face3D_clean`, read-only reference, λ=8 mm),
  now running at **λ=5 mm (60 GHz)** as an exact 5/8-scaled replica of that
  validated scene (README §6 finding 18).

## Ground rules

- **`rail3D/README.md` §4 (conventions) and §6 (hard-won findings) are binding.**
  Several past bugs look like harmless cleanups (image loader choice, ray-cast min_t,
  meshgrid indexing, RNG-state handling). Do not "simplify" physics code without
  re-running the verification gates.
- Verification gates: `rail3D/tests_physics_3d.py` (V0, V0b, V1–V4),
  `rail3D/validation_3d.py` (V5–V7), `rail3D/v8_smoke_test.py` (V8). Any
  physics/geometry/training change must keep them green; they are laptop-safe
  (small meshes). `rail3D/lab_report.py` runs the whole chain in one command
  (~5 min on the 5090) and writes a paste-able `data/generated/lab_report.md`.
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
- **Docs carry a "Last updated" line under their title** (this file,
  `rail3D/README.md`, `rail3D/SETUP_LAB.md`). Any commit that changes behaviour a
  doc describes must update the doc AND bump its date line in the same commit —
  doc drift is a recurring failure mode here.

## Current state (2026-09-04 — λ=5 migration VERIFIED end to end on the 5090)

> **Picking this up cold? Read `rail3D/NEXT_SESSION.md` first.** It is the
> short handoff: the two unpushed commits, the ONE open technical question
> (the shadow normal-offset anomaly) with the exact command to resolve it, and
> the pending FDTD-solver decision. Delete it once both are closed.

### Lab results (2026-09-04) — read this first

All eleven gates PASS at λ=5 on the RTX 5090. What the numbers say:

- **V3 = 0.0010273 at λ=8 AND λ=5, on two machines** — agreement to ~5
  significant figures. The scaled replica is confirmed numerically.
- **`scan_geometry.py` re-run at λ=5: H = 30λ CONFIRMED** (field AUC 0.878 vs
  0.840 at 20λ, 0.874 at 40λ) and the dark-field structure reproduces — 10λ
  collapses to 0.523 (near chance) with 20× the energy. An 80×80 aperture
  scores best overall (0.889/0.893) at 3.6× generation cost; not taken.
- **Ray-cast shadowing is INERT.** V6 passes, but
  `crack_shadow_with_resolved_occluder = 0.0` and 7 of 8 samples are exactly
  0.0. The V6b sweep says `SHADOW_MIN_T = 3.0` is the ONLY artifact-safe value
  tested: below 1 mm the augmented-intact artifact pins at 0.0451, over V6's
  0.03 bound. Cost: ~5–6% of real crack self-shadowing discarded. **The
  1.0–3.0 mm gap is untested** — that is the open question. (An earlier
  un-augmented ray-distance probe suggested 0.05–0.3 mm was safe; it was
  misleading, see README finding 3.)
- **Speckle statistics are PRESERVED, not improved** — measured grain 8.0 mm
  (λ=8) → 5.0 mm (λ=5), ratio exactly 5/8, window/grain 2.27 at both. A
  previous claim that λ=5 buys ~2.6× more independent cells was WRONG and is
  corrected in README finding 18. λ=5's gain is defect/λ = 1.6× and trained
  performance, not raw plane information (λ=5 mean field AUC 0.878 vs λ=8's
  0.899 — a wash within n=20 noise).
- **Generation costs 4.62 h, not the 1.5–2 h estimated** (1.30 samples/s). The
  extra factor is the ray-cast: O(rays × triangles) with BOTH scaling ×2.56 =
  6.6×. A large share of that time buys shadowing that is currently inert.
- V5 dropped 0.976 → 0.958 (still passing): the figure shows envelope
  agreement with a fringe offset, i.e. finite-segment vs infinite-extrusion
  registration, not solver disagreement.
- Crack is the limiting class, and for a specific reason: it has a LARGER mean
  signal than shell (10.4% vs 8.3% of peak) but LOWER untrained AUC (0.736 vs
  0.817), because its signature direction varies with orientation (θ spans
  −69°…90°) while shells are consistent blobs. Crack is hard because it is
  VARIABLE, not weak.

### Open decisions before the full generation

1. Refine the V6b sweep in the untested 1.0–3.0 mm gap
   (`--min-t 2.5 2.0 1.75 1.5 1.25`) — a value there may keep the artifact
   under 0.03 while restoring some crack shadowing.
2. If shadowing stays inert, consider `SHADOW_MODE="none"` for generation:
   it would produce a **bit-identical** dataset in roughly half the time.
3. **Which full-wave solver is licensed** (blocks the cross-check run, not the
   code). The answer to "Zemax or Lumerical?" is **neither** — it is a MoM
   problem: **Ansys HFSS-IE** (needs the Integral Equation licence, not just
   FEM) or **Altair FEKO**. Zemax's POP shares this code's scalar-Kirchhoff
   assumptions and would only confirm itself; full-scene FDTD is 387 Mcells /
   ~39 GB. Exporter is built and self-tested (`compare_wavefronts.py
   --export-case DIR --source plane --export-closed`); README finding 20 and
   SETUP_LAB §13 carry the reasoning, sizing and the sign/polarisation traps.

## Previous state (2026-08-17 — λ=5 scaled-replica migration + hazard fixes)

Everything below is committed on `3D_railhead_upgrade` and verified on the laptop.
Read the section top to bottom before changing physics, geometry or the objective.

### λ=5 migration (2026-08-17, newest — three commits)

- **`WVL = 5.0` (60 GHz), scaled replica**: everything Face3D chose in wavelengths
  scales ∝λ — `DX=2.5`, aperture 150×75 (grid stays 60×30), `H_MS = 30λ = 150`,
  MS→det `= 20λ = 100`, `DIST_ANT = 28λ = 140`, horn `SIZE_ANT` ×5/8 (keeps the
  waveguide single-mode), `DET_SIZE` ×5/8 = 11.375×7.0. The rail, `SEG_LEN=120`,
  defect ranges, `DET_JITTER_MM=3`, augmentation, raycast `min_t` stay physical:
  defect/λ grows 1.6× — the migration's purpose. Every Fresnel number is
  preserved exactly (V3's error matches λ=8 to 6 significant figures).
  `DET_GRID` is now DERIVED as the tiling bound `floor(aperture/window)` →
  (13,10) = 130 windows / 92% coverage at both wavelengths.
- **Guards, because a same-grid λ change is invisible to tensor shapes**:
  datasets record 36 provenance keys and training refuses mismatches; the
  GENERATOR refuses roots holding another geometry's artifacts (it used to skip
  existing shards and relabel the mixed set); checkpoints carry a geometry stamp
  and refuse to resume across it (they recorded NO geometry before, and the
  detector's persistent grid buffers would silently restore λ=8 coordinates —
  both fixed); `surface="metaunit"` raises at λ≠8 (`LIBRARY_WVL`) — the 8 mm
  meta-atom fits don't transfer and 3.8 mm pillars can't fit a 2.5 mm cell.
  SLM + no-MS baselines unaffected. **All λ=8 datasets/checkpoints on disk are
  refused, not deleted.**
- **Bug-fix wave (commit 1, before the flip)**: `config.dense_detector_centers()`
  is now the ONLY lattice source — its stale predecessor emitted ±216/±162 mm
  centres that collapsed the default detector to 54 unique spots, corrupting
  V7's probe, lab_report's separability block and the figures (training was
  unaffected). Also fixed: per-axis τ floor 0.5·dx (y-axis was sub-pixel even at
  λ=8), capture-fraction clamp (unclamped, overlap made the loss REWARD stacking
  detectors), power floor recomputed after each prune (was saturated forever),
  V8's vacuous detector-moved gate (now keep-index verified), `effective_config`
  (margin runs were scored as rank/cos), preflight's dead strict-branch.
- **Smoke sets are size-configurable**: `--smoke --smoke-n 60-80 --name <tag>`
  for geometry-exploration smokes on the 5090 (per-user request; intact count
  scales along).

### Defect model (rev.2, 2026-08-14)

- `mesh3d.py` rebuilt around per-point depth fields
  `displaced(s,y) = intact(s) − d(s,y)·n̂(s)`; per-class parametric generators
  (oriented hairline cracks incl. transverse/oblique, 2D-Gaussian dents/chains,
  shoulder-band wear, NEW parametric `shell` class — ranges cited from Ye et al.
  2018 Table 1 and Ye et al. 2023 Fig 9, PDFs in `C:\Users\Rei\Downloads\`);
  resolution-independent params (`sample_defect_params` → `render_depth_field`)
  so the fine sim mesh and the coarse ray-cast occluder render the same defect;
  `MESH_DS` now λ/8 (hairlines need it); 4-class plumbing throughout.
- **Defect parameter ranges are the user's operating ranges** (config.py, one
  block), widened from the paper measurements; README §2 carries the full
  provenance table. Do not "correct" them back to the papers. All depths are
  sampled, never clipped. `y0` is ±10 mm on purpose (the sensor rides the train
  along y, so defects pass the beam center); `s0` stays broadly sampled.
- Known quirks to keep: `extract_csv_profile` needs its 20th-percentile baseline
  correction (width-matching inflates loops); wear's CSV shape is normalized by
  its peak *inside the gauge band*, not globally; shell samples may land on the
  vertical head side (x≈38.5) — geometrically intended; on tiny smoke splits the
  calibrated FPR is degenerate (3 intact val samples), fine on the real set.

### Scene geometry — measured at λ=8, carried by similarity (2026-08-14)

- **`SEG_LEN` 240 → 120 mm.** Truncation study at λ/8 with ray-cast shadowing:
  defect-signal cosine 0.997–0.999 vs the 240 mm reference, 3.1× faster. 80 mm
  put the cut edge inside the illuminated footprint — do not shorten further.
  At λ=5 the footprint shrinks ∝λ, so 120 is SAFER than when measured; it no
  longer equals the y-aperture (75 mm), which is fine.
- **`H_MS` = 30λ** (was measured as 240 mm at λ=8 by `scan_geometry.py`).
  **The system works because it is dark-field**: the horn hits at 55° so the
  specular lobe lands at x = −2.14·H, outside the aperture. H=10λ pulls the
  lobe *inside*, giving 13× the energy and **below-chance** separability. 30λ
  maximized field-level AUC (0.899 at λ=8); beyond it field AUC falls while
  detector AUC through an *untrained* SLM keeps rising — trust the field
  metric. The rig scaled but the defects did not, so **re-run scan_geometry.py
  at λ=5 on the 5090 before the full generation** (SETUP_LAB §7).
- `n_epoch` default 400 → **1200**: the τ anneal + pruning window make the
  objective non-stationary for ~250 epochs. `train()` prints where the best
  epoch landed and warns when the model was still improving at the end.

### Objective (rev.2 + capture term)

- Default `TrainConfig(objective="rank", metric="cos", target_fpr=0.05)` —
  pairwise soft-AUC hinge on normalized barcodes, threshold calibrated on the
  validation intact spread. `objective="margin"` restores the legacy loss and V8
  guards it. README §6 finding #10 explains why.
- **`TrainConfig.w_capture = 0.2`** rewards the fraction of plane power landing
  on the retained detectors (`power_floor_loss` is a hinge that goes flat once
  satisfied, so nothing used to push light *onto* the surviving windows — which
  is what receiver SNR depends on). The fraction is **clamped ≤1 per sample**
  (overlapping windows double-count; unclamped it rewarded stacking detectors)
  and the floor is **recomputed after each prune** (frozen it saturated).
  Near-inactive at the dense start, operative after pruning. `w_capture=0.0`
  reproduces the old objective; **runs are not comparable across this change.**
  `capture_frac` is logged separately from the loss term.

### Detector design (2026-08-17)

- **Dense start is the default**: `config.DET_GRID` is DERIVED as the tiling
  bound `floor(aperture/window)` → (13,10) = 130 detectors, 92% plane coverage,
  pruned to `N_DET_FINAL = 8`. Face3D's 6×6 was 7.2% coverage; the old rail3D
  6×3 was 12.7%. **`config.dense_detector_centers()` is the ONLY lattice
  source** (README finding 17); `SoftDetector2D()` without explicit centres now
  uses it correctly.
- **`prune_criterion="redundancy"` is the default and is required by the dense
  start.** Face3D's variance criterion is only valid for SPARSE layouts: with
  overlapping windows, neighbours see nearly the same light and share a
  variance, so "lowest variance" cannot distinguish a duplicate from a uniquely
  informative detector. The new criterion values each detector by
  `std × (1 − max|corr| to survivors)` and eliminates greedily. **V0b** is the
  regression gate: variance keeps [0,4,5], discarding the unique detector;
  redundancy keeps [3,4,5]. `"variance"` runs as a real V8 training run.
- **Never anneal τ below the pixel pitch** — enforced now by a per-axis
  0.5·dx floor in `_axis_soft` (the old fix reasoned only about the long axis;
  the short axis was already sub-pixel at λ=8). `tau_end = w/8` remains the
  schedule.
- `SoftDetector2D.min_separation()` reports collapsed centres (nothing in the
  loss repels detectors from each other); training prints it and warns below one
  pixel, and V8 asserts it stays above one. `sweep_detectors.py
  --prune-criterion` also records mean |off-diagonal correlation|, min
  separation and capture fraction, with a redundancy panel.
- **V4 measures the intensity-weighted centroid of the specular lobe** (a
  finite plate makes Fresnel fringes; the argmax pixel wanders with distance
  while the centroid stays exactly on axis), on its own plane height
  (`DIST_ANT/2`) and a λ-proportional plate — λ-invariant by construction.

### Verification status (superseded by the lab results above)

- **V0, V0b, V0c, V1–V4 pass at λ=5** (`data/generated/verification_report.json`).
  V3's ASM-vs-RS error is identical to the λ=8 value to 6 significant figures —
  the scaled replica confirmed numerically.
- **V5–V8 are pending the 5090** (per user preference the laptop only runs the
  seconds-scale CPU gates). `lab_report.py` runs everything; SETUP_LAB §B is
  the exact runbook, including two deliberate guard-refusal demos and the list
  of files to send back for review.
- V8 notes that survive the migration: its pass criterion is TRAIN separation
  (`gap_mean − gap0_mean` ≥ 1.2×), not the validation score — the smoke val
  split (8 defect / 3 intact) quantizes AUC to 1/24 and pins 4-class accuracy
  at chance. Raw loss is not comparable across the run (τ anneal + pruning
  reshape the objective). Smoke uses `prune_keep=0.5` so 130 → 8 fits in 30
  epochs; real runs use 0.75. SETUP_LAB §5 has the field-by-field
  interpretation table.

### Remaining (lab 5090 — exact commands in SETUP_LAB §B)

`git pull` → `preflight.py` (expect exit 1: λ=8 datasets flagged — the guard
working) → guard demo (generator refuses the old smoke root) → `rm -rf
data/generated/smoke` → `lab_report.py` (V0–V8 incl. pending V5–V7, ~10–15
min) → `scan_geometry.py` (H table was λ=8-measured; confirm 30λ) →
**send back lab_report.md / verification_report.json / geometry_scan.json /
refusal logs / setup_diagram.png for review** → after sign-off:
`generate_dataset_3d.py --profile lab --name L5_H150_v1` (~1.5–2 h est.) →
`inspect_dataset.py` → train SLM + no-MS (metaunit blocked at λ=5) →
`analyze_results.py` → `sweep_detectors.py`. Watch crack recall (hairlines,
now 0.4–1λ) and the `mean |corr|` sweep panel (high = pruning kept
duplicates). V6's `resolved` bound is the one legitimately at-risk gate — a
trip there is a physics finding to report, not a bug.

### Known open items

- V5–V8 numbers at λ=5 do not exist yet (pending the §B lab run above).
- `surface="metaunit"` is blocked until a 60 GHz meta-atom library is fitted;
  SLM and no-MS are the usable surfaces.
- Loss weights other than `w_capture` remain module constants in losses3d
  (documented in README §8), and `torch.load(weights_only=False)` is used on
  local trusted files only.

## Typical commands (laptop — CPU gates only; everything heavier runs on the 5090)

```bash
cd rail3D
python preflight.py                              # ALWAYS first: code/GPU/CSV/dataset/ckpt
python tests_physics_3d.py                       # V0,V0b,V0c,V1-V4 (~5 s, CPU)
python setup_diagram.py                          # review figures (CPU)
```

Lab entry points (SETUP_LAB §A/§B): `lab_report.py` (whole chain → markdown),
`validation_3d.py` (V5–V7), `generate_dataset_3d.py [--smoke [--smoke-n N]]
[--name X]`, `v8_smoke_test.py`, `scan_geometry.py` (measure the observation
plane), `inspect_dataset.py` (review a dataset), `analyze_results.py` (where it
succeeds and fails, per defect parameter), `sweep_detectors.py` (final detector
count / MS→detector distance), `rail3D_pipeline.ipynb` (narrated end to end).

## Dataset hygiene (2026-08-17)

- Datasets can be named: `generate_dataset_3d.py --name X --note "..."` writes to
  `rail3D/data/generated/X/`; train against it with `TrainConfig(data_root=...)`.
- Every dataset carries `dataset_config.json` (36 geometry keys + creation time
  + git commit + host). Training REFUSES a geometry mismatch
  (`data3d.check_dataset_config`); the GENERATOR refuses to write into a root
  holding another geometry's artifacts (`data3d.check_generation_root`);
  checkpoints carry a geometry stamp and refuse to resume across it;
  `data3d.describe_dataset(root)` summarises any dataset.
- `python preflight.py` is the standing check before any run: code freshness,
  GPU, CSVs, active geometry, all datasets with λ/dates, shard consistency
  (mtime spread catches shards mixed across runs), and checkpoint stamps.
- Training prints a banner naming the run, device, detector schedule, dataset
  and its age, and the split sizes. Read it rather than assuming.
- `rail3D_pipeline.ipynb` is the narrated end-to-end path; it shells out to the
  same scripts (subprocess for long jobs, to keep kernel GPU memory bounded).
