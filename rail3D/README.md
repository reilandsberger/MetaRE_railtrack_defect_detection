# rail3D — 3D diffraction simulation + metasurface training for rail defect detection

*Last updated: 2026-08-17 · λ = 5 mm (60 GHz) era — bump this line in any
commit that changes behaviour this file describes.*

**Handoff document.** This README is written so that a future session (any
model, any context) can pick the project up cold. Read this first, then
`SETUP_LAB.md` for machine setup.

---

## 1. What this is

Upgrade of the repo's 2D rail-defect simulation (`2Dmesh_from_vertex.py` +
`training_ms_notebook.ipynb`, 2D Kirchhoff/Hankel boundary integrals at
λ=12 mm) to a **3D physical-optics simulation** at **λ = 5 mm (60 GHz)**,
reusing the verified physics of the Face3D_clean facial-recognition codebase
(`C:\Users\Rei\Downloads\Face3D_clean\Face3D_clean` — exact Rayleigh–
Sommerfeld surface integral, horn antenna source, experimentally validated at
λ = 8 mm). The scene is that validated λ=8 setup **scaled by 5/8** wherever
Face3D chose lengths in wavelengths (distances, horn, detector windows —
see finding 18), so its design conclusions transfer; the rail and its defects
keep their physical sizes, growing 1.6× relative to λ.

Task: a trainable metasurface + trainable detector placement so that raw
detector powers ("barcode") both **detect** rail defects and **classify**
them (crack / dent / wear / shell, tiny linear head). Default objective is
the rev.2 **rank** loss — a pairwise soft-AUC hinge on normalized barcodes
with the operating threshold *calibrated* on the validation intact spread
(finding 10); the legacy fixed-0.40-margin loss survives as
`objective="margin"`.

## 2. Pipeline at a glance

```
2D cross-section loops (crosssection.png + 15k defect CSVs, reused from 2D repo)
        │  sections.py  (mm units; loops arc-length-uniform, width-matched)
        ▼
swept 3D railhead mesh + per-point defect depth field d(s,y)
        │  mesh3d.py    (λ/8 facets = 0.625 mm, fixed topology → batchable)
        ▼
physical-optics scattering: horn (140 mm @ 55°) → rail → 60×30 plane @ z=150 mm
        │  field3d.py   (exact RS-I kernel, chunked+batched, ray-cast shadow)
        ▼
dataset shards: psi1, psi2 per sample + cached psi0    [generate_dataset_3d.py]
        │  data3d.py    (mode "tot" = psi0+psi1+psi2, RMS-normalized;
        │               dataset_config.json provenance, refused on mismatch)
        ▼
trainable optics: [SLM2D → RS-FFT propagator 100 mm] → |·|²
        → SoftDetector2D (dense 130 → 8 windows, trainable centers) → barcode
        │  optics3d.py, losses3d.py, train3d.py   (metaunit blocked at λ≠8)
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
| `wear` | CSV cross-section shape × y-envelope of 300–900 mm — i.e. **the whole modelled segment is worn**, only a slight end taper | 2–8 mm | horn-facing shoulder |
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

**All depths are sampled, never taken raw from the CSV.** Raw CSV depths
(crack median 7.8 mm, p95 10.8; dent median 2.5, p95 3.7) overshoot the
measured ranges, so clipping them pinned 66% of cracks at exactly 6.9 mm and
49% of dents at 2.5 mm — destroying depth diversity. The CSV supplies the
across-defect profile *shape*; depth is drawn uniformly from the operating
range (`config.py`, one block — wear included, `WEAR_DEPTH_RANGE`; its raw
CSV depths reached ~12.5 mm, which only made the easiest class easier).
Wear remains by far the strongest signal; expect it to separate first.

Check any dataset with `python inspect_dataset.py [--root ...]`: it re-derives
each stored sample's geometry from its seed and shows it next to the stored
field, plus parameter histograms.

## 3. File map

| File | Role |
|---|---|
| `rail3d/config.py` | ALL constants, paths, device profiles, per-sample seeds, `dense_detector_centers()` (the ONLY detector-lattice source). Start here. |
| `rail3d/sections.py` | 2D loop loading (adapted from `2Dmesh_from_vertex.py`, converted to mm, no import-time work) |
| `rail3d/mesh3d.py` | swept mesh builder, per-class defect envelopes, augmentation |
| `rail3d/field3d.py` | PO solver (port of Face3D `FieldCalculation.py`): `horn_to_plane` (psi0, cached once), `scattered_fields` (psi1/psi2, batched+chunked), `raycast_shadow_mask` |
| `rail3d/optics3d.py` | `PropagatorRSFFT` (exact prop3d kernel via FFT), `PropagatorASM2D`, `SLM2D`, `MetaUnitSoft` (blocked at λ≠8), `SoftDetector2D`, `ONN3D` |
| `rail3d/losses3d.py` | combined loss (rank objective + capture reward; legacy margin path) and all metrics |
| `rail3d/data3d.py` | shard IO (atomic writes), dataset assembly, stratified split (seed 0), **provenance**: `write/check_dataset_config`, `check_generation_root`, `describe_dataset`, `list_datasets` |
| `rail3d/train3d.py` | `TrainConfig`, `train()` (auto-resume, RNG-state checkpoints, pruning + keep-index history, geometry-stamped checkpoints), `effective_config`, `full_evaluation()` |
| `rail3d/viz_setup.py` | every review figure (setup diagram, meshes, fields, library, detectors, barcodes) — output to `data/figures/`, which is **gitignored**: regenerate with `python setup_diagram.py` (+ the validation scripts for V5/V7 plots) |
| `preflight.py` | **run before anything**: code freshness, GPU, CSVs, active geometry, every dataset with λ/date/commit, shard consistency, checkpoint stamps |
| `generate_dataset_3d.py` | CLI generator (`--profile lab`, `--name`, `--smoke [--smoke-n N]`, `--status`; resumable shards; refuses mixed-geometry roots) |
| `inspect_dataset.py` | review a dataset before/after generation (re-derives geometry from seeds; warns on provenance mismatch) |
| `tests_physics_3d.py` | V0, V0b, V0c, V1–V4 automated gates (CPU-safe) |
| `validation_3d.py` | V5–V7 gates + figures (includes the 2D Hankel reference solver) |
| `v8_smoke_test.py` | V8 end-to-end + kill-and-resume bit-identity + refusal/regression guards |
| `lab_report.py` | the whole verification chain in one command → paste-able `data/generated/lab_report.md` |
| `compare_wavefronts.py` | one sample solved every way — λ=8 vs λ=5, physics terms, shadow guard, mesh — as amplitude/phase figures + metrics; exports an FDTD-ready case and accepts external solver fields back |
| `scan_geometry.py` | measure the observation plane (H, offset, aperture) before committing to a generation |
| `sweep_detectors.py` | final detector count / MS→detector distance sweep (training-time only) |
| `analyze_results.py` | where it succeeds and fails, per defect parameter; failure montage |
| `setup_diagram.py` | renders the annotated scene diagram |
| notebooks | `rail3D_pipeline` (the whole pipeline, narrated), `design_review_`, `validation_3d_`, `training_3d_ms_`, `training_3d_no_ms_` — all thin wrappers over the modules |
| `rail3d/library_amp_fit.npy`, `library_phase_fit.npy` | Face3D meta-atom fits (8 mm band), copied byte-for-byte |

## 4. Conventions (do not change silently)

- **Units: mm everywhere** (Face3D convention). The old 2D scripts used µm.
- **Axes**: x across railhead, y along rail, z up; crown at z=0. 2D loop
  point (x2d, y2d) → (x = x2d, z = y2d − 180).
- **Grid**: 60×30 cell-centered, dx = λ/2 = 2.5 mm, x ∈ ±75, y ∈ ±37.5, built
  with `meshgrid(..., indexing='ij')`. Metasurface plane z = H_MS = 30λ =
  150 mm; detector plane 20λ = 100 mm further (z = 250 mm). All of these are
  DERIVED in config.py — quote config, not this line, if they ever disagree.
- Horn: Face3D "config 55" **scaled ∝λ** (17.1×13.7 mm aperture at 140 mm,
  55° in x–z) — keeps the feeding waveguide single-mode and the far-field
  ratio, see finding 18.
- Splits: stratified 80/10/10, seed 0 (comparable to the 2D notebooks).
- Label order: crack=0, dent=1, wear=2, **shell=3** (`config.CLASS_NAMES`).
  `shell` is parametric — it has no entry in `config.DATASET_DIRS`; code that
  loads CSVs must branch on `cls in config.DATASET_DIRS`, not on `!= "intact"`.
- **Detector lattices come from `config.dense_detector_centers()` and nowhere
  else** (finding 17). A `SoftDetector2D` built for a non-default aperture
  must be passed explicit centres.
- **Device**: never a bare `"cuda"`. The `lab` profile is `cuda:auto` →
  `config.best_cuda_device()` ranks visible GPUs by (compute capability,
  VRAM) and takes the strongest, because the 5090's index differs per
  machine (cuda:0 on the lab workstation). `RAIL3D_DEVICE` overrides.

## 5. Verification status (see `data/generated/verification_report.json`)

Status at λ = 5 mm (the migration wavelength). "laptop" = verified on the
MX250/CPU on 2026-08-17; "**pending 5090**" = must be (re)run on the lab GPU
before the numbers are quotable — `lab_report.py` runs them all.

| gate | what it proves | status at λ=5 |
|---|---|---|
| V0 | rev.2 defect geometry: orientations, bands, seeds, fine↔coarse render consistency (0.045 mm) | PASS (laptop CPU) |
| V0b | redundancy pruning keeps the unique detector where variance keeps duplicates | PASS (laptop CPU) |
| V0c | the staleness guards guard: lattice, capture clamp, dataset/root/checkpoint refusals | PASS (laptop CPU) |
| V1 | chunked/batched solver ≡ verbatim Face3D (≤5e-7) | PASS (laptop CPU) |
| V2 | FFT propagator ≡ conv2d (1.2e-6) | PASS (laptop CPU) |
| V3 | ASM vs RS-FFT 0.10% — **identical to 6 s.f. with the λ=8 value**, confirming the scaled replica | PASS (laptop CPU) |
| V4 | specular centroid on axis, power conservation 0.9998, mesh orientation | PASS (laptop CPU) |
| V5 | 3D PO vs 2D Hankel reference (was r=0.976 at λ=8) | **pending 5090** |
| V6 | ray-cast shadowing real-effect vs artifact bounds (2 mm hairlines are now 0.4λ — if `resolved` trips, that is a physics finding, not a bug) | **pending 5090** |
| V6b | *(sweep, not a gate)* the ray-cast guard's cost/benefit vs `min_t`: artifact suppressed on intact meshes against real crack shadowing kept | run on demand (`--min-t`) |
| V7 | λ/8 vs λ/16 mesh convergence at the barcode level (through the FIXED 130-window probe — pre-fix numbers were measured through 54 collapsed windows) | **pending 5090** |
| V8 | end-to-end training: separation grows, 130→8 pruning, keep-index-verified detector movement, no collapse, capture ∈ (0,1], bit-identical resume, legacy objective + variance criterion + stale-checkpoint refusal | **pending 5090** |

Run them: `python tests_physics_3d.py` (V0–V4, CPU) · `python validation_3d.py`
(V5–V7, GPU) · `python v8_smoke_test.py` (V8, needs `--smoke` generation
first) · or everything at once with `python lab_report.py`.

## 6. Hard-won findings (READ BEFORE TOUCHING PHYSICS)

1. **`crosssection.png` must be read with `plt.imread`, not PIL** — the PNG is
   RGBA; matplotlib resolves transparent pixels to white, PIL to black, which
   inverts the dark-foreground mask and silently produces a wrong reference
   (width 94 mm instead of 157.4 mm). Handled in `sections.read_binary_image`.
2. **λ/2 meshes are NOT converged** for the PO integral (defect-signal cosine
   0.66 vs fine mesh at λ=8). Generation uses **λ/8** (`config.MESH_DS`;
   hairline cracks forced the move from the earlier λ/4, whose measured
   numbers — cosine 0.993 vs λ/12, ~15% shared magnitude bias — are kept here
   as history). V7 re-checks λ/8 vs λ/16 at the active wavelength. Raw
   complex-field L2 does not converge at any practical facet size (glint
   speckle) — judge fidelity at the **detector-barcode level**, not the field
   level.
3. **Ray-cast shadowing needs a `min_t` guard — but a SMALL one.** Facet
   chords sag inside the true convex surface, so horizon-grazing rays clip
   their own neighbouring facets. Without any guard ~4% of faces (terminator
   band) are falsely blocked and the field changes ~30–60%, while the 2D
   line-of-sight ground truth says the real intact-rail shadow effect is ~1%.
   **The two populations were measured directly** (2026-08-17, hit distances
   with the guard disabled):

   | population | where it lives |
   |---|---|
   | artifact (intact rail — a convex rail cannot shadow itself) | `t ≤ 0.021 mm` (λ=5), `≤ 0.037 mm` (λ=8) — the chord sagitta `d²/(8R)`, ~1/100 of a facet |
   | real crater-wall occlusion (crack / dent / shell) | `t ≥ 0.3 mm`, out to ~9 mm |

   They are cleanly separated, so the guard belongs in the gap — but the
   historical `min_t = 3.0 mm` sits far above it and therefore discards real
   self-shadowing too: at λ=5 a deep crack's ray-cast field is **bit-identical
   to no shadowing at all**, while `min_t = 0.125 mm` moves it ~6%. The scale
   is the FACET, not the wavelength, so a finer occluder makes 3.0 mm *worse*
   (better resolution → shorter blocking distances → more of them cut). This
   was already true at λ=8; the migration only sharpened it.
   `config.SHADOW_MIN_T` is now the single source, it is a **provenance key**
   (datasets are incomparable across a change), and
   `python validation_3d.py --min-t ...` (V6b) sweeps it at the field level.
   The default is held at 3.0 until that sweep on the lab GPU justifies moving
   it. Occluders are the **λ/2 coarse mesh** (16× cheaper, verified
   equivalent).
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
13. **Variance-based detector pruning is only valid for SPARSE layouts.** Face3D's
    6×6 grid covered 7.2% of its aperture with 30–37 mm gaps, so windows were
    near-independent and "lowest variance" really did mean "least informative".
    A dense start (13×10 = 92% coverage, windows overlapping by well under a
    mm) breaks that: neighbours see almost the same light and therefore have
    almost the same variance, so the ranking cannot separate "duplicate of my
    neighbour" from "uniquely informative". Demonstrated failure (the **V0b**
    gate): with three bright duplicates and one quiet unique detector, pruning
    to 3 by variance keeps *two duplicates and discards the unique signal*.
    Default is `prune_criterion="redundancy"` — greedy backward elimination
    valuing each detector by `std × (1 − max|corr| to survivors)` — which
    keeps the unique one and drops the duplicates, and empirically ends with
    detectors ~2× further apart. `"variance"` is retained for comparison and
    exercised by a real V8 training run.
14. **Never anneal the detector τ below the pixel pitch** — detector placement
    cannot be learned more finely than the field is sampled. The old
    `tau_end = w/16` froze position gradients mid-anneal, and even `w/8` is
    sub-pixel on the SHORT window axis (w_y/8 < dx on both wavelengths' grids),
    which the original fix missed by reasoning only about x. `_axis_soft` now
    floors τ at **0.5·dx per axis**; `tau_end = w/8` remains the schedule for
    the long axis.
15. **The power term is a floor plus a concentration reward.**
    `power_floor_loss` is a hinge that goes flat once satisfied, so nothing used
    to push the metasurface to route light *onto* the surviving detectors —
    which is what receiver SNR depends on. `TrainConfig.w_capture` (default
    0.2) rewards the captured-power fraction — near-inactive at the dense
    start (the tiling windows already catch nearly everything), operative after
    pruning to a handful. The fraction is **clamped to ≤1 per sample**:
    overlapping windows double-count shared pixels, and unclamped the term
    went negative and *rewarded* stacking detectors on one spot. Set
    `w_capture=0.0` to reproduce the earlier objective; results are **not
    comparable across this change**. The floor itself is **recomputed after
    every prune** — frozen at the dense-start value it saturated permanently
    and double-counted the capture term.
16. **Ray-cast shadowing became active once the defect ranges widened.** With
    the earlier 1.5–3 mm hairline cracks the λ/2 (4 mm) occluder mesh had no
    crack in it at all and V6's `worst_rel_l2` was exactly 0.0 — occluder
    resolution, not physics. With the current ranges (cracks 2–5 mm wide ×
    2–10 mm deep, dents to 8 mm) the occluder resolves them and V6 measured a
    real **4.3%** effect at λ=8. At λ=5 the occluder is finer (2.5 mm) and the
    defects are relatively larger (a hairline is 0.4–1λ), so expect the effect
    to GROW — re-check `crack_shadow_with_resolved_occluder` (the λ/8-occluder
    control) whenever V6's numbers move.
17. **Detector lattices must come from `config.dense_detector_centers()` and
    nowhere else.** Its predecessor (`detector_grid_centers`, a fixed 36 mm
    pitch) silently emitted centres OUTSIDE the aperture once the dense
    DET_GRID landed; `SoftDetector2D`'s clamp then collapsed 130 windows onto
    54 unique spots (min separation 0.0) in every default-constructed model —
    the V7 probe, the setup diagram, the detector/barcode figures and
    lab_report's separability block, while training itself was unaffected.
    Three independent copies of the lattice formula had grown; there is now
    one, and V0c asserts its windows sit inside the aperture at the exact
    aperture/(g+1) pitch.
18. **The λ migration is a scaled replica, and the guards enforce it.**
    Everything Face3D chose in wavelengths scales with λ (DX, aperture, H_MS =
    30λ, MS→det = 20λ, DIST_ANT = 28λ, the horn SIZE_ANT, DET_SIZE); the rail,
    defect ranges, SEG_LEN, mounting tolerances (DET_JITTER_MM, roll/jitter
    augmentation), raycast `min_t` and the 4 mm shell-roughness lattice are
    physical and do NOT scale. Consequence: every Fresnel number **of the rig**
    is preserved exactly — the MS→detector propagation has F = (WX/2)²/(λ·L) =
    11.250 at both wavelengths, which is why V3's error matches the λ=8 value
    to 6 significant figures. Quantities that pair the *fixed* rail against the
    *scaled* rig deliberately do not scale, and that is the gain: defect/λ grows
    8/5 = 1.6×, and because the speckle grain λL/D shrinks as (5/8)² while the
    plane keeps its angular extent, the plane now carries **~2.6× more
    independent speckle cells** (≈182 vs ≈71 across the aperture). Windows,
    scaling as 5/8, go from ≈0.7 grain to ≈1.1 grain — i.e. from slightly
    under-filled to matched (see finding 19).
    A same-grid λ change is INVISIBLE to tensor shapes, so provenance carries
    it instead: datasets record 36 geometry keys (`data3d.PROVENANCE_KEYS`),
    checkpoints carry a geometry stamp, the generator refuses mixed roots, and
    `surface="metaunit"` refuses λ ≠ LIBRARY_WVL outright (the 8 mm meta-atom
    fits do not transfer, and 3.8 mm pillars cannot fit a 2.5 mm cell).
19. **The detector pitch is set by the speckle grain and the window, and the
    dense start is the tiling bound — not "as dense as possible".** Two scales
    govern the readout:
    - *speckle grain* `g ≈ λ·H/D` (D = illuminated rail extent, ~76 mm across
      the head): **≈ 9.9 × 6.2 mm** at λ=5. This is the finest structure the
      field actually has — sampling the plane more finely than g returns
      correlated (duplicate) numbers, no new information.
    - *window* `DET_SIZE` = 11.375 × 7.0 mm ≈ **1.1 grain**. That is the right
      regime: a window much smaller than a grain collects less power for
      readings its neighbours already share, while a window covering N grains
      averages independent speckles and dilutes defect contrast by ~1/√N.
    So the useful pitch is `max(window, grain)` = the window, and the densest
    lattice worth starting from is the **tiling** one,
    `floor(aperture/window)` = 13×10 = 130 (92% coverage) — which is exactly
    how `DET_GRID` is now derived. Denser is genuinely overkill (pure
    duplicates, a longer prune schedule, no extra information); sparser risks
    dead zones, because a detector moves only by local gradient and cannot
    cross a dark region to reach a hotspot it never sees. The aperture holds
    ≈182 independent cells at λ=5, so 130 windows sample near the information
    limit and pruning to 8 selects the most complementary of them.

## 6b. Objective & metrics (rev. 2)

`TrainConfig(objective="rank", metric="cos", target_fpr=0.05)` is the default.
Loss terms (`losses3d.combined_loss`):

| term | purpose |
|---|---|
| `rank` | pairwise hinge over all (defect, intact) pairs — soft-AUC surrogate |
| `hardest` | same hinge on the worst 10% of pairs |
| `intact` | mean intact cos-gap → keeps the intact cluster tight |
| `class` | cross-entropy on normalized barcodes (4 classes) |
| `power` | detected-power floor — a hinge, flat once satisfied (recomputed after each prune) |
| `capture` | `w_capture · (1 − captured fraction)` — routes light ONTO the surviving windows (receiver SNR); clamped, see finding 15 |
| `centroid` | class-centroid separation on unit barcodes |
| `tv` | total variation on the phase / pillar-width map (fabricability) |

Reported: **AUC**, **pass@cal** and **false alarm** at the calibrated
threshold, **balanced accuracy** at the swept optimum
(`losses3d.best_threshold`, Face3D's argmin(FN+FP)), class accuracy, plus
ROC / noise / alignment curves in `full_evaluation`.

## 7. Current state / what remains

Done and committed on **`3D_railhead_upgrade`** (2026-08-17): all modules;
the detector-lattice bug fix + training-hazard fixes; the λ=5 scaled-replica
migration with its provenance/checkpoint/generator/metaunit guards; V0, V0b,
V0c, V1–V4 green at λ=5 on the laptop. All λ=8 datasets and checkpoints on
disk are *refused, not deleted* — they remain the record.

**Remaining (on the lab 5090, in order — the exact commands are SETUP_LAB.md
§B "λ=5 bring-up runbook"):**
1. `git pull` → `python preflight.py` — expect the λ=8 datasets to be flagged
   stale (that is the guard working) and exit 1 until a λ=5 set exists.
2. One guard demo: `generate_dataset_3d.py --smoke` against the old smoke root
   must REFUSE; then delete that root.
3. `python lab_report.py` — V0–V8 including the **pending** V5–V7 at λ=5 plus
   a fresh smoke set with measured samples/s (~10–15 min).
4. `python scan_geometry.py` — the H table in config.py was measured at λ=8;
   the rig scaled but the defects did not, so confirm 30λ still wins.
5. Send back `lab_report.md`, `verification_report.json`,
   `geometry_scan.json`, the refusal logs and `setup_diagram.png` for review
   **before** committing to the full generation.
6. After sign-off: `generate_dataset_3d.py --profile lab --name L5_H150_v1`
   (~1.5–2 h estimated at 2.56× the λ=8 face count; resumable) →
   `inspect_dataset.py` → train (SLM + no-MS; metaunit is blocked until a
   60 GHz library exists) → `analyze_results.py` → `sweep_detectors.py`.
   Geometry exploration: `--smoke --smoke-n 60-80 --name <tag>` per candidate.

## 8. Simplifications & limitations (what this simulation is NOT)

Consolidated 2026-08-17. Each is deliberate; the point is that nobody should
discover them by surprise. Grouped by where they live:

**Electromagnetics / material**
- **Scalar fields** — no polarization, no cross-pol, no vector diffraction.
- **PEC reflection** (r = −1 hard-coded in field3d): no conductivity, loss,
  rust or contamination layer, no Fresnel angle dependence.
- **≤ 2 bounces** (psi1 + psi2): no higher-order multiple scattering, no
  cavity resonance inside a crack.
- **Single tone** — no bandwidth, dispersion or FMCW modelling. (60 GHz sits
  on the O₂ absorption line, ~15 dB/km — negligible at 0.4 m, noted for
  completeness.)
- **Idealized horn**: single-mode cosine aperture field with quadratic phase;
  no edge diffraction, no measured pattern. The λ-scaled SIZE_ANT keeps the
  waveguide's modal content identical to the validated λ=8 model.
- **Binary face visibility, no receive-cosine** (finding 4) — faithful to the
  validated Face3D formulation, discontinuous at the terminator.

**Surface & defect geometry**
- One 2D cross-section **swept uniformly** along y: no joints, welds,
  corrugation, curvature, second rail, sleepers or ballast.
- The intact section is smoothed (31-point moving average) and every defect
  loop is width-matched to it — **no surface roughness model at all**; every
  facet is a specular mirror. Defects are the ONLY texture.
- Depth fields are **normal displacements with d ≥ 0**: no undercut, no
  re-entrant crack walls, no subsurface/internal defects, no material
  build-up (lipping).
- Fixed mesh topology (batching requirement) — no adaptive refinement around
  a defect. Region bands are heuristics (`mesh3d.region_band`); the CSV
  baseline correction assumes ≥20% of the arc is undamaged.
- Single defect per sample; classes are mutually exclusive by construction.

**Shadowing**
- **Source-side only** (rail→horn ray-cast); rail→plane occlusion is not
  tested — matches the shallow-defect regime.
- Occluder mesh is λ/2. `config.SHADOW_MIN_T` is a discretization guard whose
  natural scale is the FACET, not λ; at its historical 3.0 mm it suppresses
  most real self-shadowing as well as the artifact (finding 3) — pending the
  V6b sweep, self-shadowing of narrow craters is effectively absent.

**Sensing & noise**
- Detectors are **ideal rectangular power integrators**: unity quantum
  efficiency, no angular acceptance pattern, no crosstalk or mutual coupling,
  perfectly coplanar. Nothing in the loss forbids overlap (the capture clamp
  removes the *reward* for it; `min_separation` reports collapse).
- The noise model is **relative** (multiplicative 0.5% + additive 1e-5 on the
  RMS-normalized field): absolute received power is discarded by the global
  RMS normalization, so configurations that collect less light are not
  penalized the way real hardware would penalize them (why the capture term
  exists, and why scan_geometry reports absolute energy separately).
- No environmental clutter and **no "unusual intact" negatives** (grease,
  ballast dust, joints) — false-alarm numbers are against clean intact rail
  only.
- The same source CSV cross-section can appear in train and test with
  different sampled depths (split is by sample, not by CSV).

**Training**
- Adam moments restart at every prune (detector/head tensors are rebuilt).
- Loss weights other than `w_capture` are module constants in losses3d, not
  recorded per run.
- `torch.load(weights_only=False)` on local shard/checkpoint files — fine for
  trusted local data, not hardened against a malicious file.

**Meta-atoms**
- The library is an 8 mm-band fit at normal incidence, per-pixel polynomials,
  no inter-pillar coupling — and is **hard-blocked** at λ≠8 until a 60 GHz
  library is fitted. SLM2D is the idealized (lossless, unquantized) upper
  bound; "none" is the no-MS baseline.

Physical tolerances kept in mm (DET_JITTER_MM = 3, ±4 mm placement jitter,
±2° roll) are relatively LARGER at λ=5 — the augmentation is harsher, which
is honest hardware realism, not an oversight.
