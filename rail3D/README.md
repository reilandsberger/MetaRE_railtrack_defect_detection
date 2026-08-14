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
swept 3D railhead mesh + per-point defect depth field d(s,y)
        │  mesh3d.py    (λ/8 facets, fixed topology → batchable)
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
metrics: AUC / pass@calibrated-threshold / false alarm / confusion / robustness
```

### Defect geometry model (rev. 2)

Defects are **per-point depth fields**, not per-slice cross-section blends:

```
displaced(s, y) = intact(s) − d(s, y) · n̂(s)
```

`s` = arc-length along the illuminated cross-section, `y` = rail axis,
`n̂(s)` = outward section normal, `d ≥ 0` in mm. Each class has a parametric
generator (`sample_defect_params` → `render_depth_field`), with ranges taken
from laser-scan measurements — Ye et al. 2018 (*Proc IMechE F*, Table 1,
Figs 14/23/24) and Ye et al. 2023 (*IEEE TIM*, Figs 7/9):

| class | footprint | depth | region |
|---|---|---|---|
| `crack` | line-divot, 10–50 mm long × 2–5 mm wide; longitudinal / transverse / oblique (20–70°); 30% chance of 2–3 parallel lines | 2–10 mm | running band + gauge corner |
| `dent` | 2D super-Gaussian, 10–30 mm (y) × 10–30 mm (s); 20% chance of a 2–4 pit chain | 1.5–8 mm | running band |
| `wear` | CSV cross-section shape × y-envelope of 300–900 mm — i.e. **the whole 240 mm segment is worn**, only a slight end taper | 2–8 mm | horn-facing shoulder |
| `shell` | **parametric** (no CSVs): Fourier-modulated ellipse 8–20 mm + ragged interior; 30% chance of a second lobe | 1–5 mm | horn-facing shoulder |

All depths are **sampled uniformly** from the ranges above; the CSV supplies the
across-defect profile *shape* only. For wear the shape is normalized by its peak
**inside the gauge band** — normalizing globally left realized depth far below the
sampled value, because many wear CSVs have their deepest point on the crown or
field side, which the band mask removes.

### Where the numbers come from

Baseline measurements — **Ye et al. 2018, Table 1** (journal p. 351): cracks
27–31 mm long × 2.00 mm wide × 3.00–4.33 mm deep (45° cut); squats 16.6–19.5 mm
long × 1.90–2.42 mm deep; a narrow deep notch 10.34 mm × 3.12 mm × 6.85 mm.
Figs 23–24 show three parallel gauge-corner cracks and rounded running-surface
squats. **Ye et al. 2023, Fig. 9**: a shelling patch ≈ 10 × 12 mm, ≈ 2 mm deep,
ragged and multi-lobed; Fig. 7 shows squat chains and two-lobe shelling.

The ranges in the table above are the **operating ranges for this system**,
widened from those measurements to cover the severities it must handle. Neither
paper characterizes gauge-corner wear, so wear's extent and depth are set from
domain knowledge, not measurement.

**`y0` (along-track defect position) is near zero by construction** — ±10 mm,
not spread over the aperture. The horn boresights the crown at y = 0 and the
sensor rides the train along y, so every defect passes through the beam center
at some frame; the residual spread models finite capture rate. This does **not**
apply to `s0`, the across-head position, which the train cannot change and which
stays broadly sampled.

Parameters are resolution independent, so the fine simulation mesh (λ/8) and
the coarse ray-cast occluder (λ/2) render the *same* physical defect.

**Crack and dent depth is sampled, not taken from the CSV.** Raw CSV depths
(crack median 7.8 mm, p95 10.8; dent median 2.5, p95 3.7) overshoot the
measured ranges, so clipping them pinned 66% of cracks at exactly 6.9 mm and
49% of dents at 2.5 mm — destroying depth diversity. The CSV supplies the
across-defect profile *shape*; depth is drawn uniformly from the paper range.
**Wear is deliberately left on its raw CSV depth** (up to ~12.5 mm) and is by
far the strongest signal — mean intensity change ≈ 51% of peak, vs ≈ 15% for
the other three classes. Expect it to be the easiest class to separate.

Check any dataset with `python inspect_dataset.py [--root ...]`: it re-derives
each stored sample's geometry from its seed and shows it next to the stored
field, plus parameter histograms.

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
| `rail3d/viz_setup.py` | every review figure (setup diagram, meshes, fields, library, detectors, barcodes) — output to `data/figures/`, which is **gitignored**: regenerate with `python setup_diagram.py` (+ the validation scripts for V5/V7 plots) |
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
- Label order: crack=0, dent=1, wear=2, **shell=3** (`config.CLASS_NAMES`).
  `shell` is parametric — it has no entry in `config.DATASET_DIRS`; code that
  loads CSVs must branch on `cls in config.DATASET_DIRS`, not on `!= "intact"`.
- **Device**: never a bare `"cuda"`. The `lab` profile is `cuda:auto` →
  `config.best_cuda_device()` ranks visible GPUs by (compute capability,
  VRAM) and takes the strongest, because the 5090's index differs per
  machine (cuda:0 on the lab workstation). `RAIL3D_DEVICE` overrides.

## 5. Verification status (see `data/generated/verification_report.json`)

**V0** rev.2 geometry: crack orientations, band confinement, seed
reproducibility, λ/1 vs λ/4 render consistency 0.11 mm · **V1** solver ≡
verbatim Face3D (1e-7) · **V2** FFT ≡ conv2d (1e-6) · **V3** ASM 0.13% ·
**V4** sanity · **V5** 3D-vs-2D r=0.984 · **V6** shadow artifact 0.0 ·
**V7** λ/8-vs-λ/16 defect-signal cosine · **V8** end-to-end + resume
mismatch 0.0, `full_evaluation` covered, legacy objective guarded.

Run all of them: `python tests_physics_3d.py` (V0–V4, CPU),
`python validation_3d.py` (V5–V7, GPU), `python v8_smoke_test.py` (V8, needs
`--smoke` generation first).

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
10. **A fixed absolute detection margin is ill-posed here.** With the required
    ±4 mm / ±2° placement augmentation, the *intact* barcode distribution has
    a self-spread comparable to the defect signal (first lab run: intact gap
    mean 0.43 vs the hard-coded 0.40 margin → false alarm ≈ 0.5 while pass rate
    and FA rose together). Default objective is therefore **`"rank"`**: a
    pairwise soft-AUC hinge on per-sample-normalized barcodes (`cos_gap`), with
    the operating threshold **calibrated** at `target_fpr` on the *validation*
    intact spread (never on test). Checkpoint selection uses AUC + class
    accuracy (threshold independent). `objective="margin"` restores the old
    loss exactly, and V8 keeps it running as a regression guard.
11. **`extract_csv_profile` must baseline-correct.** `match_reference_width`
    rescales each defect loop by a few percent; without subtracting the 20th
    percentile of the deviation, that inflation cancels the real material
    removal and wear CSVs read as depth ≈ 0.
12. The illuminated arc wraps *around* the head, so an `x >= GAUGE_X_MIN` test
    alone also selects the downward-facing under-head fillet. Region bands
    additionally gate on the normal (`nz`) — see `mesh3d.region_band`.
13. **Ray-cast shadowing is inert for rev.2 hairline cracks.** The occluder
    mesh is λ/2 (4 mm facets) and a crack is 1.5–3 mm wide, so the occluder
    simply has no crack in it: V6's `worst_rel_l2` is exactly 0.0 — that is
    the *occluder resolution*, not physics. Measured against a
    generation-resolution (λ/8) occluder the omitted self-shadowing is
    **0.5%** (V6 records it as `crack_shadow_with_resolved_occluder`), well
    inside other accepted approximations, and capturing it would mean
    ray-casting against 16× more triangles. Do not "conclude" shadowing is
    unnecessary from the 0.0 — re-measure with a resolved occluder if the
    defect geometry ever gets deeper or narrower.

## 6b. Objective & metrics (rev. 2)

`TrainConfig(objective="rank", metric="cos", target_fpr=0.05)` is the default.
Loss terms (`losses3d.combined_loss`):

| term | purpose |
|---|---|
| `rank` | pairwise hinge over all (defect, intact) pairs — soft-AUC surrogate |
| `hardest` | same hinge on the worst 10% of pairs |
| `intact` | mean intact cos-gap → keeps the intact cluster tight |
| `class` | cross-entropy on normalized barcodes (4 classes) |
| `power` | detected-power floor (ported from the 2D notebook) |
| `centroid` | class-centroid separation on unit barcodes |
| `tv` | total variation on the phase / pillar-width map (fabricability) |

Reported: **AUC**, **pass@cal** and **false alarm** at the calibrated
threshold, **balanced accuracy** at the swept optimum
(`losses3d.best_threshold`, Face3D's argmin(FN+FP)), class accuracy, plus
ROC / noise / alignment curves in `full_evaluation`.

## 7. Current state / what remains

Done: all modules, all verification, smoke dataset
(`data/generated/smoke/`), design/validation/training notebooks, SETUP_LAB.md.
Everything committed on branch **`3D_railhead_upgrade`**.

**Remaining (on the lab 5090, in order — full detail in `SETUP_LAB.md`):**
0. Copy the three `data_defect_*2` CSV folders (~2 GB) to the workstation and
   point `RAILDEFECT_DATA_DIR` at their parent — needed by V5–V8 and all
   generation, not just generation. `git pull` before running anything.
1. `SETUP_LAB.md` §1–4: venv (cu128 torch, never `pip -U`), then
   `tests_physics_3d.py` (V1–V4, CSV-free) → `validation_3d.py` (V5–V7) →
   `--smoke` generation → `v8_smoke_test.py`.
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
