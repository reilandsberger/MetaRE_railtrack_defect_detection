# rail3D — 3D diffraction simulation + metasurface training for rail defect detection

**Handoff document.** This README is written so that a future session (any
model, any context) can pick the project up cold. Read this first, then
`SETUP_LAB.md` for machine setup.

---

## 1. What this is

Upgrade of the repo's 2D rail-defect simulation (`2Dmesh_from_vertex.py` +
`training_ms_notebook.ipynb`, 2D Kirchhoff/Hankel boundary integrals at
λ=12 mm) to a **3D physical-optics simulation** at **λ = 8 mm (37.5 GHz)**,
reusing the verified physics of the Face3D_clean facial-recognition codebase
(`C:\Users\Rei\Downloads\Face3D_clean\Face3D_clean` — exact Rayleigh–
Sommerfeld surface integral, horn antenna source, meta-atom library,
experimentally validated).

Task: a trainable metasurface + trainable detector placement so that raw
detector powers ("barcode") both **detect** rail defects (relative-L2 gap vs
the intact barcode > 0.40 margin) and **classify** them (crack / dent / wear,
tiny linear head).

## 2. Pipeline at a glance

```
2D cross-section loops (crosssection.png + 15k defect CSVs, reused from 2D repo)
        │  sections.py  (mm units; loops arc-length-uniform, width-matched)
        ▼
swept 3D railhead mesh, defect blended in along y by class envelope g(y)
        │  mesh3d.py    (λ/4 facets, fixed topology → batchable)
        ▼
physical-optics scattering: horn (224 mm @ 55°) → rail → 60×30 plane @ z=160mm
        │  field3d.py   (exact RS-I kernel, chunked+batched, ray-cast shadow)
        ▼
dataset shards: psi1, psi2 per sample + cached psi0    [generate_dataset_3d.py]
        │  data3d.py    (mode "tot" = psi0+psi1+psi2, RMS-normalized)
        ▼
trainable optics: [SLM2D | MetaUnitSoft → RS-FFT propagator 160mm] → |·|²
        → SoftDetector2D (18→8 windows, trainable centers) → barcode
        │  optics3d.py, losses3d.py, train3d.py
        ▼
metrics: pass rate / false alarm / gap-p5 / confusion / ROC-AUC / robustness
```

## 3. File map

| File | Role |
|---|---|
| `rail3d/config.py` | ALL constants, paths, device profiles, per-sample seeds. Start here. |
| `rail3d/sections.py` | 2D loop loading (adapted from `2Dmesh_from_vertex.py`, converted to mm, no import-time work) |
| `rail3d/mesh3d.py` | swept mesh builder, per-class defect envelopes, augmentation |
| `rail3d/field3d.py` | PO solver (port of Face3D `FieldCalculation.py`): `horn_to_plane` (psi0, cached once), `scattered_fields` (psi1/psi2, batched+chunked), `raycast_shadow_mask` |
| `rail3d/optics3d.py` | `PropagatorRSFFT` (exact prop3d kernel via FFT), `PropagatorASM2D`, `SLM2D`, `MetaUnitSoft`, `SoftDetector2D`, `ONN3D` |
| `rail3d/losses3d.py` | combined loss (margin + intact-compactness + CE + power floor + top-k + centroid margin + TV) and all metrics |
| `rail3d/data3d.py` | shard IO (atomic writes), dataset assembly, stratified split (seed 0) |
| `rail3d/train3d.py` | `TrainConfig`, `train()` (auto-resume, RNG-state checkpoints, pruning), `full_evaluation()` |
| `rail3d/viz_setup.py` | every review figure (setup diagram, meshes, fields, library, detectors, barcodes) |
| `generate_dataset_3d.py` | CLI generator (`--profile lab`, `--smoke`, `--status`; resumable shards) |
| `tests_physics_3d.py` | V1–V4 automated gates (CPU-safe) |
| `validation_3d.py` | V5–V7 gates + figures (includes the 2D Hankel reference solver) |
| `v8_smoke_test.py` | V8 end-to-end + kill-and-resume bit-identity test |
| `setup_diagram.py` | renders the annotated scene diagram |
| notebooks | `design_review_`, `validation_3d_`, `training_3d_ms_`, `training_3d_no_ms_` — all thin wrappers over the modules |
| `rail3d/library_amp_fit.npy`, `library_phase_fit.npy` | Face3D meta-atom fits, copied byte-for-byte |

## 4. Conventions (do not change silently)

- **Units: mm everywhere** (Face3D convention). The old 2D scripts used µm.
- **Axes**: x across railhead, y along rail, z up; crown at z=0. 2D loop
  point (x2d, y2d) → (x = x2d, z = y2d − 180).
- **Grid**: 60×30 cell-centered, dx = 4 mm, x ∈ ±120, y ∈ ±60, built with
  `meshgrid(..., indexing='ij')`. Metasurface plane z = 160 mm; detector
  plane 160 mm further.
- Horn: Face3D "config 55" verbatim (27.4×21.9 aperture, 224 mm, 55° in x–z).
- Splits: stratified 80/10/10, seed 0 (comparable to the 2D notebooks).
- Label order: crack=0, dent=1, wear=2 (`config.CLASS_NAMES`).
- **Device**: never a bare `"cuda"`. The `lab` profile is `cuda:auto` →
  `config.best_cuda_device()` ranks visible GPUs by (compute capability,
  VRAM) and takes the strongest, because the 5090's index differs per
  machine (cuda:0 on the lab workstation). `RAIL3D_DEVICE` overrides.

## 5. Verification status — ALL 8 GATES PASS (see `data/generated/verification_report.json`)

V1 solver ≡ verbatim Face3D (1e-7) · V2 FFT ≡ conv2d (1e-6) · V3 ASM 0.13% ·
V4 sanity · V5 3D-vs-2D r=0.984 · V6 shadow artifact 0.0 · V7 λ/4 signal
cosine mean 0.976 · V8 resume mismatch 0.0.

## 6. Hard-won findings (READ BEFORE TOUCHING PHYSICS)

1. **`crosssection.png` must be read with `plt.imread`, not PIL** — the PNG is
   RGBA; matplotlib resolves transparent pixels to white, PIL to black, which
   inverts the dark-foreground mask and silently produces a wrong reference
   (width 94 mm instead of 157.4 mm). Handled in `sections.read_binary_image`.
2. **λ/2 meshes are NOT converged** for the PO integral (defect-signal cosine
   0.66 vs fine mesh). Generation uses **λ/4** (`config.MESH_DS`), which
   preserves the defect-signal *direction* (cosine 0.993 vs λ/12) with a
   ~15% systematic magnitude bias shared by all samples. Raw complex-field L2
   does not converge at any practical facet size (glint speckle) — judge
   fidelity at the **detector-barcode level**, not the field level.
3. **Ray-cast shadowing needs `min_t`** (ignore hits < 3 mm along the ray):
   facet chords sag inside the true convex surface, so horizon-grazing rays
   clip their own neighboring facets. Without it, ~4% of faces (terminator
   band) get falsely blocked and the field changes by ~30–60% — the 2D
   line-of-sight ground truth says the real intact-rail shadow effect is ~1%.
   Real crack-crater shadowing (~2–14% on the deepest cracks) is retained.
   Occluders are the **λ/2 coarse mesh** (16× cheaper, verified equivalent).
4. **PO terminator discontinuity**: the Face3D formulation applies no receive-
   cosine at the face, so faces at grazing incidence carry O(1) amplitude and
   binary visibility toggles them discontinuously. This is faithful to the
   verified Face3D/2D codes — do not "fix" it without re-validating V5.
5. **Noise must be gated on `self.training`** (Face3D bug: eval was noisy).
   Same for detector jitter. All reported metrics use **hard** binary
   detector masks (`hard_powers`), never the soft training masks.
6. **Resume bit-identity** relies on: all randomness through global torch
   CPU/CUDA RNGs, RNG states in every checkpoint, no DataLoader workers, and
   restoring RNG states as **CPU ByteTensors** (`_restore_rng`).
7. After **pruning**, `detector.u` and `head` are replaced — the optimizer
   must be rebuilt (train3d does this) and checkpoints store `n_det` so
   `_shape_model_to_checkpoint` can reshape a fresh model before loading.
8. The meta-atom library fits are **per-pixel on 80×80**; the rail grid uses
   the central crop `[:, 10:70, 25:55]` (`optics3d._LIB_CROP`). If the grid
   changes, the crop must change.
9. `chunk_faces` is the total face-slots per chunk **across the batch**
   (per-element chunk = chunk_faces // B) — this is what bounds memory.

## 7. Current state / what remains

Done: all modules, all verification, smoke dataset
(`data/generated/smoke/`), design/validation/training notebooks, SETUP_LAB.md.
Everything committed on branch **`3D_railhead_upgrade`**.

**Remaining (on the lab 5090, in order):**
1. `SETUP_LAB.md` §1–4: env setup + re-run the three verification scripts.
2. `python generate_dataset_3d.py --profile lab` (full 5000/class + 512
   intact, ≲1 h, resumable).
3. Run `training_3d_ms_notebook.ipynb` (SLM run, then MetaUnit run).
4. Run `training_3d_no_ms_notebook.ipynb` (baseline + comparison table).
5. Optional: λ/8 regeneration (`config.MESH_DS = WVL/8`), 2-layer experiment
   (`n_layer=2, layer_distances=(d12, 160)`).

Known accepted limitations: single-frequency (no dispersion); shadowing is
source-side only (rail→MS occlusion not ray-cast — matches the shallow-defect
regime); λ/4 mesh magnitude bias (see 6.2); detector count/positions restart
Adam moments on prune.
