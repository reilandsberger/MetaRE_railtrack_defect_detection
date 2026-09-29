# rail3D — 3D diffraction simulation + metasurface training for rail defect detection

*Last updated: 2026-09-29 · λ = 5 mm (60 GHz) · horn = RFspin H-A75-W20 (finding 29) · V0–V8
measured on the 5090 with the PREVIOUS horn; V0–V4, V9 and V5 re-run on the laptop with the new one
— bump this line in any commit that changes behaviour this file describes.*

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
keep their physical sizes, growing 1.6× relative to λ. **The horn is the
exception since 2026-09-24:** it is the physical RFspin H-A75-W20 60 GHz
standard-gain horn, modelled with the textbook aperture method (finding 29).

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
physical-optics scattering: horn (H-A75-W20, 140 mm @ 55°) → rail → 60×30 plane @ z=150 mm
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
| `crack` | **box** divot (flat bottom, steep walls), 10–50 mm long × **1.5–3 mm** wide; longitudinal / transverse / oblique (20–70°); **always a group of 3 parallel lines**, gap = **1.6–2.2 × width** | 4–8 mm | running band + gauge corner |
| `dent` | 2D super-Gaussian, 10–30 mm (y) × 10–30 mm (s); 20% chance of a 2–4 pit chain | 1.5–8 mm | running band |
| `wear` | CSV cross-section shape × y-envelope of 300–900 mm — i.e. **the whole modelled segment is worn**, only a slight end taper | 2–8 mm | horn-facing shoulder |
| `shell` | **parametric** (no CSVs): Fourier-modulated ellipse 8–20 mm + ragged interior; 30% chance of a second lobe | 1–5 mm | **gauge corner only**, centre x ∈ 20–30 mm |

**Revised 2026-09-12** (user operating ranges, and the reason each class is in
the simulation):

- **`crack` is now a BOX divot, not a CSV notch.** Flat bottom, walls ramping
  over a quarter of a mesh cell — as vertical as the λ/8 grid can carry. That
  sharp edge is the feature distinguishing a crack from the smooth
  super-Gaussian `dent`, which is exactly the discrimination the first
  classifier failed (it collapsed 55% of dents into "crack"). Crack no longer
  consumes a CSV profile at all; the CSVs still drive `wear`.
- **`crack` is always a group of three**, not a 30% chance of 2–3. Hairline
  cracks form in clusters in locally worn regions, never singly. The three
  lines share shape, depth, length and orientation and are offset
  **perpendicular to their length**. The previous code offset along *s*, which
  separates a longitudinal crack but leaves a transverse one (θ = π/2) exactly
  on top of itself — the group was invisible for a third of all samples.
- **The gap SCALES WITH THE WIDTH** (`CRACK_LINE_GAP_FACTOR`, 1.6–2.2 ×), it is
  not an absolute range. A fixed gap cannot work at both ends of the width
  range: 2 mm merges 3 mm-wide lines into one ribbed band, while 5 mm spreads
  1.5 mm-wide lines into three unrelated cracks. As a multiple of the width the
  **clear space between adjacent lines is (factor − 1) × width**, i.e. 0.6–1.2
  line-widths — three distinct lines at every width, by construction. Measured
  over 600 samples: gap 2.5–6.6 mm, clear space never below **0.96 mm** (1.5
  mesh cells, so they never merge numerically), and 97% of groups are longer
  than they are wide. The remaining 3% are 10 mm-long cracks with 3 mm lines,
  where three separated parallel lines simply cannot be longer than the group
  is wide — a constraint of the geometry, not a modelling choice.
- **`shell` is confined to the gauge corner**, centre x ∈ [20, 30] mm, a
  truncated normal about the window centre rather than uniform over the whole
  gauge band. This is the realistic initiation site *and* the part of the head
  that the horn illuminates and the metasurface aperture sees — the reason the
  class is being classified at all. Previously shells could land on the
  vertical head side at x ≈ 38.5 mm.
  Implementation note: x is **not monotonic** along the arc (it climbs the
  gauge side, peaks at ≈ 38.9 mm, then falls across the crown), so the window
  is satisfied in two disjoint runs. Taking min/max over both centres the
  sample *between* them, where x ≈ 37 mm — measured, that put only **4%** of
  shells inside the window. The code takes the longest contiguous in-window run
  on the gauge shoulder instead.

`CRACK_LINE_COUNT`, `CRACK_LINE_GAP_RANGE` and `SHELL_GAUGE_X_RANGE` are
**provenance keys**, so every dataset generated before this change is refused.

All depths are **sampled uniformly** from the ranges above; for `wear` the CSV
supplies the across-defect profile *shape* only. For wear the shape is normalized by its peak
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
| `tests_physics_3d.py` | V0, V0b, V0c, V1–V4 and **V9 (horn)** automated gates (CPU-safe) |
| `validation_3d.py` | V5–V7 gates + figures (includes the 2D Hankel reference solver); `--min-t/--normal-offset` runs the V6b guard sweep, which is a study, not a gate |
| `v8_smoke_test.py` | V8 end-to-end + kill-and-resume bit-identity + refusal/regression guards |
| `lab_report.py` | the whole verification chain in one command → paste-able `data/generated/lab_report.md` |
| `compare_wavefronts.py` | one sample solved every way — λ=8 vs λ=5, physics terms, shadow guard, mesh, horn vs plane wave — as amplitude/phase figures + metrics; exports a full-wave case (STL+OBJ, optional closed body, `--source plane`) and accepts external solver fields back, auto-detecting their time convention and amplitude. See finding 20 |
| `scan_geometry.py` | measure the observation plane (H, offset, aperture) before committing to a generation |
| `sweep_detectors.py` | final detector count / MS→detector distance sweep (training-time only) |
| `analyze_results.py` | where it succeeds and fails, per defect parameter; failure montage |
| `setup_diagram.py` | renders the annotated scene diagram |
| `run_stage.py` | generate → gate → train → analyse as ONE resumable command (`--stage prelim/full`, `--with-baseline`, `--fresh`, `--bundle`); refuses short or mismatched datasets |
| `ablate_surface.py` | 2×2 over SLM init × `w_capture` + the no-MS control (finding 24); resumable, writes `surface_ablation.json` |
| `slm_profile.py` | the trained mask: wrapped phase, \|t\|, incident light, change from init |
| `tests_plumbing.py` | P0–P5 non-physics plumbing tests (~25 s CPU): dataset-root resolution, history schema, zero-phase ≡ baseline, slm_profile, FDTD round trip, horn source |
| `check_notebook.py` | fresh-kernel name-ordering check for `rail3D_pipeline.ipynb` (~1 s) |
| `target_figures.py` | what a successful result would look like — labelled TARGET, never a measurement |
| `horn_source.py` | rail3D's horn as a Lumerical **Import source** (E + H .mat + .lsf) on z = 15 mm; `--aperture-from` builds it from the full-wave horn (rung −1) instead |
| `horn_fdtd_case.py` | Lumerical **rung −1**: watertight PEC STL of the H-A75-W20 + WR-15 feed, Mode-source/monitor plan (`horn_case.json`), `export_horn.lsf` |
| `fdtd_agreement.py` | scores Lumerical exports: `--horn` (rung −1), `--injection` (rung 0), `--sample … --external` (rungs 1–3, z = 30 and MS plane), `--target` (synthetic target) |
| `lumerical_mockup.py` | draws the target FDTD setup into a Layout-window screenshot, every box from `case.json` |
| `presentation/` | `deck_figs.py` (horn/result figures computed from the code) + `update_deck.py` (the 2026-09-24 slides) for `data/generated/rail3D_overview.pptx` |
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
- Horn: the **physical RFspin H-A75-W20** — `config.SIZE_ANT = (22.8, 16.8,
  3.7592, 1.8796, 28.0)` mm = inner aperture A × B, WR-15 feed a × b, axial
  flare length L — 140 mm from the crown at 55° in x–z, E along y (s-pol).
  **Not λ-scaled**: it is hardware. Aperture model = textbook
  (`HORN_FLARE_K="k0"`, `HORN_SAMPLING="midpoint"`); Face3D's variants
  (`"beta_wvg"`, `"linspace"`) exist only so V1 can replay Face3D verbatim.
  Finding 29. (Until 2026-09-24 this was Face3D's horn ×5/8 — overmoded.)
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

Status at λ = 5 mm (the migration wavelength). **All gates now measured**:
V0–V4 on the laptop CPU (2026-08-17) and re-confirmed on the 5090, V5–V8 on the
lab 5090 (2026-09-04; V6/V7 re-measured 2026-09-11 with the settled shadow guard).
`lab_report.py` runs them all in one command. **The horn changed on 2026-09-24**
(finding 29): V0–V4, V9 and V5 (r = 0.958) re-ran green on the laptop with the
new horn; V6–V8 on the 5090 are previous-horn numbers until the lab re-run
(SETUP_LAB §7e).

| gate | what it proves | status at λ=5 |
|---|---|---|
| V0 | rev.2 defect geometry: orientations, bands, seeds, fine↔coarse render consistency (0.045 mm) | PASS (laptop CPU) |
| V0b | redundancy pruning keeps the unique detector where variance keeps duplicates | PASS (laptop CPU) |
| V0c | the staleness guards guard: lattice, capture clamp, dataset/root/checkpoint refusals | PASS (laptop CPU) |
| V1 | chunked/batched solver ≡ verbatim Face3D (≤5e-7) | PASS (laptop CPU) |
| V2 | FFT propagator ≡ conv2d (1.2e-6) | PASS (laptop CPU) |
| V3 | ASM vs RS-FFT 0.10% — **identical to 6 s.f. with the λ=8 value**, confirming the scaled replica | PASS (laptop CPU) |
| V4 | specular centroid on axis, power conservation 0.9998, mesh orientation | PASS (laptop CPU) |
| V5 | 3D PO vs 2D Hankel reference | PASS, **r = 0.958** (5090) — was 0.976 at λ=8; the figure shows envelope agreement with slight fringe offset, i.e. finite-segment vs infinite-extrusion registration, not solver disagreement |
| V6 | ray-cast shadowing real-effect vs artifact bounds | PASS at the shipped guard (`min_t` 0.05, `normal_offset` 0.3; 5090, 2026-09-11): `intact_augmented_artifact` **0.0**, `crack_shadow_omitted_by_coarse_occluder` **0.0113** (< 0.02). The earlier "inert at `min_t = 3.0`" reading is retracted — finding 3 |
| V6b | *(sweep, not a gate)* the guard over `min_t` × `normal_offset` | settled 2026-09-11 (finding 3). Ignore V6b blocks without `_stamp`, and its `recommended` field (finding 26) |
| V7 | λ/8 vs λ/16 mesh convergence at the barcode level | PASS, min cosine **0.9969**, mean 0.9988 with shadowing on (5090, 2026-09-11; 0.9948 / 0.9982 before the guard fix) |
| V8 | end-to-end training: separation grows, 130→8 pruning, keep-index-verified detector movement, no collapse, capture ∈ (0,1], bit-identical resume, legacy objective + variance criterion + stale-checkpoint refusal | PASS — separation 0.0290→0.0562, min separation 6.13 mm, resume mismatch exactly 0.0 (5090) |
| V9 | the horn: aperture-integral directivity (Nikolova eq. 18.21) = closed form (18.39) within 0.2 dB; gain inside `HORN_SPEC` 19–21 dBi (±0.3) at 50 / 60 / 75 GHz; `RESOL_ANT` converged on the rail footprint (corr ≥ 0.9995 and amplitude within 1% vs 80×80); feed single-mode | PASS (laptop CPU, 2026-09-24): 20.086 = 20.086 dBi; 19.07 / 20.09 / 21.01 dBi; corr 0.999998; WR-15 TE10-only. Broken on purpose: `beta_wvg` → 20.9 dBi fails, `linspace` fails amplitude (1.022), the previous horn fails band + feed |

Run them: `python tests_physics_3d.py` (V0–V4 + V9, CPU) · `python validation_3d.py`
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
3. **Ray-cast shadowing needs a NORMAL OFFSET, not a `min_t` guard — and it
   is NOT inert.** Facet chords sag inside the true convex surface, so
   horizon-grazing rays clip their own neighbouring facets. Without any guard
   ~4% of faces (terminator band) are falsely blocked and the field changes
   ~30–60%.

   **Why `min_t` was always the wrong knob.** The self-hit is a
   *perpendicular* problem — the neighbour sags by the chord sagitta
   `δ = d²/(8R)`, 0.003 mm on the crown (R≈300) to 0.06 mm at the gauge corner
   (R≈13). But `min_t` biases *along the ray*, and a ray leaving at grazing
   angle θ rises δ only after travelling `δ/sin θ`, which diverges exactly at
   the terminator where self-hits occur (~3.4 mm at 1°). That is why the only
   value that suppressed the artifact was 3.0 mm = 1.2 × the occluder facet —
   it was not clearing a sagitta, it was skipping the whole neighbouring
   facet. And 3.0 mm is **wider than a crack** (2–5 mm), so it deleted the
   physics the guard exists to find.

   **The 2-D sweep that settled it** (2026-09-10, lab 5090; augmented intact
   meshes, three cracks spanning the depth range — idx 0 = 2.72 mm, idx 1 =
   5.28 mm, idx 17 = 9.57 mm, the deepest in the set):

   | `min_t` | offset | intact artifact | crack 2.7 mm | 5.3 mm | 9.6 mm | resolved |
   |---|---|---|---|---|---|---|
   | 3.00 | 0.00 | 0.0000 | 0.0000 | 0.0197 | 0.0950 | 0.1135 |
   | 0.05 | 0.00 | 0.0288 | 0.0562 | 0.0737 | 0.1371 | 0.1488 |
   | 0.05 | 0.05 | **0.0000** | 0.0533 | 0.0676 | 0.1125 | 0.1138 |
   | 0.05 | 0.15 | **0.0000** | 0.0495 | 0.0615 | 0.0954 | 0.1139 |
   | **0.05** | **0.30** | **0.0000** | **0.0485** | **0.0580** | **0.1119** | 0.1141 |
   | 0.05 | 0.60 | **0.0000** | 0.0438 | 0.0573 | 0.1135 | 0.1144 |

   Three results, and they overturn what this finding used to say:

   - **The normal offset is a complete cure.** Every configuration with
     `offset ≥ 0.05` has an intact artifact of **exactly 0.0** — mean *and*
     max, and 1.19e-08 at the barcode level, which is float32 epsilon. Only
     `offset = 0` rows show any artifact at all. The old "pinned at 0.0451"
     figure was the artifact with no offset in play.
   - **Shadowing is not inert.** It is a **5–11% field effect** on cracks at
     and above the mean depth. The earlier "inert" conclusion came entirely
     from probing crack idx 0 — a 2.72 mm crack, *shallower than the 2.5 mm
     occluder facet that would have to represent it*, which shows exactly zero
     at every setting. `SHADOW_MODE = "none"` would therefore **not** be
     bit-identical; it would delete up to 11% of the field on deep cracks.
   - **`min_t` must be a token, and the offset must do the work.** At
     `offset ≥ 0.15` the deep-crack result is min_t-**independent** (spread
     0.3% across all four values tested) — the signature of an offset that is
     actually sufficient. At 0.05 it is not (18% spread), so that row is on a
     knife edge. Shallow cracks still need the small `min_t`: their crater
     walls sit within a few mm of the shading point, and `min_t = 3.0` deleted
     them outright (2.7 mm crack: 0.0000 → 0.0485).

   **Shipped: `SHADOW_MIN_T = 0.05`, `SHADOW_NORMAL_OFFSET = 0.3`.** Offset
   0.05 scores ~6% more total shadowing, but 0.3 is 5× the worst measured
   intact sagitta, sits on the min_t-independent plateau, and agrees with a
   generation-resolution (λ/8) occluder to 98%. The artifact is measured only
   on *intact* meshes, so margin against an unmeasured defect-mesh failure is
   worth more than 6% of signal — **false physics is worse than missing
   physics**, and the historical grazing artifact was ~4% of faces producing a
   30–60% field error.

   Two lessons that generalise beyond this finding:

   - **A benefit column measured on one sample is not a benefit column.**
     Three separate artifacts (the old V6b sweep, V6's resolved-occluder
     control, and `compare_wavefronts`' default) all happened to use crack
     idx 0 and all reported zero shadow effect. Probe a crack at or above the
     mean depth; `--crack-idx` exists for this.
   - **An artifact-contaminated row cannot be compared on benefit.** The
     `offset = 0, min_t = 0.05` row reports both the largest "benefit" (0.1371)
     *and* a resolved-occluder figure 30% above every clean row — both are the
     same false shadowing leaking into the measurement. The selection rule now
     takes the cleanest achievable artifact first.

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
    30λ, MS→det = 20λ, DIST_ANT = 28λ, the horn SIZE_ANT until 2026-09-24 —
    finding 29 — and DET_SIZE); the rail,
    defect ranges, SEG_LEN, mounting tolerances (DET_JITTER_MM, roll/jitter
    augmentation), raycast `min_t` and the 4 mm shell-roughness lattice are
    physical and do NOT scale. Consequence: every Fresnel number **of the rig**
    is preserved exactly — the MS→detector propagation has F = (WX/2)²/(λ·L) =
    11.250 at both wavelengths, which is why V3's error matches the λ=8 value
    to 6 significant figures. Quantities that pair the *fixed* rail against the
    *scaled* rig deliberately do not scale, and that is the gain: **defect/λ
    grows 8/5 = 1.6×** — a fixed-depth defect imposes 1.6× more phase, which is
    the whole point of the migration.
    **The speckle statistics, however, are PRESERVED, not improved.** Measured
    on the lab fields (2026-09-04, envelope removed, `wavefront_fields.npz`):

    | | grain along x | window | window/grain |
    |---|---|---|---|
    | λ=8 | 8.0 mm | 18.2 mm | **2.27** |
    | λ=5 | 5.0 mm | 11.375 mm | **2.27** |

    The grain ratio is 5/8 exactly — it scales as λ¹, so grains-per-window and
    the independent-cell count across the aperture (~30) are unchanged. An
    earlier revision of this finding claimed the grain scales as λ² and that
    λ=5 buys ~2.6× more independent cells; **that was wrong**, because it
    treated the illuminated extent D as fixed. It is not: `SIZE_ANT` and
    `DIST_ANT` both scale with λ, so the horn footprint scales too and
    `g = λH/D ∝ λ`. Consistent with this, the λ=5 geometry scan measures
    essentially the same untrained separability as λ=8 (mean field AUC 0.878
    vs 0.899, within the n=20 noise) — the gain to look for is in TRAINED
    performance and in defect/λ, not in raw plane information.
    A same-grid λ change is INVISIBLE to tensor shapes, so provenance carries
    it instead: datasets record 36 geometry keys (`data3d.PROVENANCE_KEYS`),
    checkpoints carry a geometry stamp, the generator refuses mixed roots, and
    `surface="metaunit"` refuses λ ≠ LIBRARY_WVL outright (the 8 mm meta-atom
    fits do not transfer, and 3.8 mm pillars cannot fit a 2.5 mm cell).
19. **The detector pitch is set by the speckle grain and the window, and the
    dense start is the tiling bound — not "as dense as possible".** Two scales
    govern the readout:
    - *speckle grain*, **measured at 5.0 mm along x at λ=5** (8.0 mm at λ=8;
      `wavefront_fields.npz`, envelope removed). This is the finest structure
      the field actually has — sampling the plane more finely than g returns
      correlated (duplicate) numbers, no new information. Note the naive
      `g = λH/D` with a FIXED D over-predicts the λ-scaling; the horn footprint
      scales with λ too, so g ∝ λ (finding 18).
    - *window* `DET_SIZE` = 11.375 mm ≈ **2.3 grain** (the same ratio as at
      λ=8 — see finding 18). That is a sane regime: a window much smaller than
      a grain collects less power for readings its neighbours already share,
      while a window covering many grains averages independent speckles and
      dilutes defect contrast by ~1/√N. At ~2 grains it is mildly on the
      averaging side, identical to the validated λ=8 design.
    So the useful pitch is `max(window, grain)` = the window, and the densest
    lattice worth starting from is the **tiling** one,
    `floor(aperture/window)` = 13×10 = 130 (92% coverage) — which is exactly
    how `DET_GRID` is now derived. Denser is genuinely overkill (pure
    duplicates, a longer prune schedule, no extra information); sparser risks
    dead zones, because a detector moves only by local gradient and cannot
    cross a dark region to reach a hotspot it never sees. The aperture holds
    ~30 independent speckle cells across x at either wavelength, so 130
    overlapping windows oversample deliberately — the point of the dense start
    is candidate coverage for pruning, not extra information.

20. **The full-wave cross-check must be MoM, not Zemax and not full-scene
    FDTD — and two convention mismatches will masquerade as physics.**
    PO assumes surface radii of curvature ≫ λ. At λ=5 the crack widths are
    2–5 mm = **0.4–1λ**, exactly where the tangent-plane approximation is
    expected to break, and no existing gate can see it: V1–V3 verify we solve
    *our own* integral correctly, and V5 compares against the 2D code, which
    shares the assumption.
    - **Zemax OpticStudio is not a valid reference.** Its Physical Optics
      Propagation is a scalar Fresnel/Kirchhoff propagator — the same
      approximation family as `field3d.py` — so it would agree with us by
      construction. Its non-sequential mode scatters incoherently (no phase)
      and cannot return a complex field at all. Likewise **HFSS SBR+** (ray PO
      + PTD edges) is an upgrade on our model, not an independent check.
    - **FDTD is valid but volumetric.** Boxing the whole scene out to the plane
      at z=150 is **387 Mcells at λ/20 ≈ 39 GB** — over a 32 GB GPU, and mostly
      empty air whose propagation V3 already validates to 0.13%. Viable only
      boxed around the rail (23 Mcells / 2.3 GB for a 30 mm segment) with a
      near-field projection to the plane.
    - **MoM/MLFMM fits**: PEC surface, open region, the mesh is 2-D only.
      Measured on the exported **closed** body at the production 120 mm segment
      (405 cm²): **562k RWG unknowns at λ/10** (233k at the exported λ/8
      density). Closing the body roughly doubles the unknowns — the unlit caps
      and underside get meshed too, which is the price of an opaque rail rather
      than an infinitely thin sheet. Dense MoM would be 5.05 TB, so MLFMM is
      mandatory. Ansys **HFSS-IE** (needs the Integral Equation licence, not
      just FEM) or **Altair FEKO**.
    - **Do not shorten the rail to save unknowns.** Illuminated power per unit
      length within 1λ of the cut end, relative to mid-span, is **0.80x at a
      30 mm segment** and **0.08x at 120 mm** (`case.json` → `truncation`,
      measured by `cut_end_illumination`). PO has *no* edge diffraction; a
      full-wave reference has plenty, so a lit cut end makes the reference
      diffract off a truncation the real rail does not have — read as "PO fails
      on the defect". The truncation study that justified `SEG_LEN` = 120
      measured the *defect signal*, a difference in which the common edge
      contribution cancels; it does not license a short segment for an
      absolute-field comparison. **Compare `E_defect − E_intact` between
      solvers as the primary metric** — export both at the same segment
      length.
    Two traps, both handled by `compare_wavefronts.align_external()`, which
    *reports* what it fitted rather than silently applying it:
    (a) rail3D uses `exp(−iωt)` → outgoing `exp(+ik₀R)` (`field3d.rs_kernel`),
    while HFSS/FEKO/Lumerical use `exp(+jωt)` → `exp(−jk₀R)`, so **external
    fields arrive conjugated** — conjugating the wrong one turns a perfect
    match into an apparent total failure; (b) absolute amplitude is arbitrary,
    so one complex gain α = ⟨cand,ref⟩/⟨cand,cand⟩ is fitted over the plane
    before differencing (the residual is then `sqrt(1 − complex_corr²)` by
    construction, so **quote `complex_corr`**). A third trap the code cannot
    fix: rail3D is **scalar**, so export one component from the vector solver
    (E_y / TE is the cleanest match) and set the incident polarisation to match.
    Export with `--export-closed` — MoM puts current on **both** faces of an
    open sheet, which is not what an opaque rail does — and `--source plane`,
    which removes the horn aperture model as a confound (it forces
    `compute_psi2=False`: psi2 is re-radiation off the horn, and a plane wave
    has no horn). Ladder, one unknown at a time: **flat PEC plate → intact rail
    → cracked rail, all plane-wave; real horn last.** SETUP_LAB §13 is the
    runbook.

21. **An EPOCH is not an optimizer step, and every schedule constant is in
    epochs.** `train()` iterates the whole training split per epoch, so steps
    per epoch scale with dataset size — but `n_epoch`, `tau_anneal_end` and
    `prune_start/end` were all tuned on the **smoke set, where the training
    split is a single batch and 1 epoch is exactly 1 step**:

    | dataset | n_train | steps/epoch | `n_epoch=1200` becomes |
    |---|---|---|---|
    | smoke (20/class) | 64 | **1** | 1,200 steps — as designed |
    | prelim (2000/class) | 6,400 | **7** | 8,400 steps, **~10 h** |
    | full (5000/class) | 16,000 | **16** | 19,200 steps, ~20 h |

    (`lab` uses `train_batch = 1024`, so a step is 1024 + `b0`=64 samples.)

    **Correction, measured 2026-09-11:** the step-count arithmetic above is
    right, but an earlier version of this finding attributed a 10-hour notebook
    cell to it. That was wrong. The prelim run's 300 epochs (2100 steps) took
    **36 seconds** on the 5090 — about 17 ms/step — so `n_epoch = 1200` would
    have been ~2.4 minutes, not hours. Training on this model is effectively
    free and `n_epoch` can be raised freely. The 10-hour cell remains
    unexplained; the most likely cause is that kernel falling back to CPU (the
    banner prints the device for exactly this reason), not the epoch count.

    **Scale `n_epoch` down by the steps-per-epoch factor and compress the
    schedule to match**, keeping pruning at ≥10 events (130 → 8 at
    `prune_keep = 0.75` needs `log(8/130)/log(0.75)` ≈ 10, and events fire every
    `prune_every` **epochs**, so the window cannot shrink below
    `10 × prune_every`). The notebook does this per stage:
    prelim `n_epoch=300, tau_anneal_end=60, prune 25–75`; full
    `n_epoch=150, tau_anneal_end=30, prune 25–75`.

    `train()` prints the real step budget in its banner and a **measured** ETA
    after the first epoch. Keep both: the arithmetic still matters for the
    SCHEDULE (below), even though the wall-clock turned out to be trivial.

22. **"Best" must mean "best model that meets the detector budget."**
    `score = auc + class_acc`, and both improve with MORE detectors, so an
    unrestricted `argmax` over epochs reliably selects a **pre-prune** epoch.
    Measured on the first prelim run: `best_epoch 28` of 300 with
    `prune_start = 25` — and because the epoch-25 prune only *arms* the
    schedule, the saved "best" model still had all **130** detectors.

    Everything downstream then described the dense array: `analyze_results
    --which best`, the notebook comparison table, every per-class recall and
    AUC in `analysis.json`. The 130 → 8 pruning that the whole detector design
    exists to justify was discarded at checkpoint-selection time, silently,
    while the run reported success.

    `train()` now refuses to promote a checkpoint whose `n_det` is not
    `n_det_final`, records `best_n_det`, and `run_stage.py` prints a loud
    warning if the best model missed the budget. Before any budget-meeting
    epoch exists the old behaviour still applies, so a short or interrupted run
    is not left with no checkpoint at all.

    The general lesson: a model-selection criterion that is free to ignore a
    hard design constraint will ignore it. State the constraint in the
    selection rule, not only in the schedule.


23. **A geometry change must move the dataset root, not block the run.**
    The provenance guard is correct to refuse mixing shards of two geometries
    into one root — but every entry point derived the root from the stage name
    alone (`L5_prelim`), so after the rev.3 defect model landed there was no
    way forward at all: the generator refused, and renaming by hand was the
    only escape. A guard with no legal next move is a guard people disable.

    `data3d.stage_root(stage)` now resolves it: `L5_<stage>` while that root is
    free or already holds THIS geometry, `L5_<stage>_<geometry tag>` once it
    holds a different one, where the tag is a 6-hex digest of the 40
    `PROVENANCE_KEYS`. `data3d.run_tag(root)` derives the run/checkpoint name
    from the ROOT rather than the stage, so a geometry change gets fresh
    checkpoints too — otherwise the checkpoint's geometry stamp would refuse to
    resume and the run would stall for a *second* reason a step later.

    `run_stage.py` and the notebook's STAGE cell both call it, so they cannot
    drift. Old roots stay on disk as the record; `--name` overrides. Note that
    only provenance keys move the tag — a training-schedule change does not,
    which is right, because those datasets are comparable.


24. **The no-metasurface baseline BEAT the trained metasurface — and that is a
    provable optimization failure, not a result about metasurfaces.**
    Prelim, 2026-09-11, same dataset / schedule / seed / detector lattice:
    `surface="slm"` val AUC **0.880**, `surface="none"` val AUC **0.976**
    (pass rate 0.757 vs 0.954). The SLM won only on `class_acc` (0.607 vs
    0.541), which is why the `auc + class_acc` selection score still ranked
    them 1.488 vs 1.517 — the baseline wins on the selection criterion too.

    What makes this a bug rather than a finding: `surface="none"` is
    `nn.Identity` with the SAME propagator, detector and noise path, so an SLM
    holding zero phase reproduces it **exactly** — measured at 0.000e+00 on
    the detector powers. The baseline is strictly inside the SLM's hypothesis
    space, so a correctly optimized SLM cannot score below it. It gave up
    0.10 AUC to a point it could have reached by setting a parameter to zero.

    Two suspects, both cheap to test (`ablate_surface.py`, ~3 min/run):
    - **Initialisation.** `SLM2D` starts at phase std **π/2** (measured 1.61 rad
      over 1800 pixels) — a full random diffuser that scrambles the defect
      signature into speckle at epoch 0. `TrainConfig.slm_init_std = 0.0`
      starts the run AT the baseline instead.
    - **`w_capture = 0.2`.** It rewards power on the retained detectors for
      RECEIVER SNR, but `ONN3D.add_noise` is gated on `self.training`, so
      evaluation is noiseless and capture buys nothing at scoring time. The
      shipped run drove `capture_frac` to **0.457** against a 0.062 floor
      (7.4× concentration) while AUC fell. In a noiseless eval that term can
      only trade against contrast.

    The general lesson, and the second time this exact shape has appeared here
    (finding 22 was the first): **when a model scores below a point inside its
    own hypothesis space, stop interpreting the science and debug the
    optimizer.** Do not report "the metasurface underperforms" until an SLM
    seeded at zero phase has been measured.

25. **The system is nearly blind outside the gauge corner — `s0`, not defect
    size, is the dominant variable.** Detection rate vs arc position, prelim:
    crack rises 0.03 → 0.82 across s0 ≈ 107 → 148 mm (r = +0.51); dent 0.17 →
    1.00 (r = +0.68). Depth correlates at r = +0.04 (crack) and +0.14 (dent) —
    essentially nothing. Every one of the eight worst missed detections sits at
    s0 ≈ 100–130 mm, and several are 7 mm+ deep cracks: **a deep defect in the
    wrong place is invisible, a shallow one in the right place is not.**

    This also explains the headline shell improvement (recall 0.42 → **0.975**,
    AUC 0.78 → 0.996). Confining shells to the gauge corner (`SHELL_GAUGE_X_RANGE
    = 20–30 mm`, the physically-motivated change) put them at s0 ≈ 147–154 — the
    only well-seen band. Shells did not become easier to detect; they were moved
    into the region the geometry already saw. Read it as confirmation of the
    field-of-view limit, not as a defect-model win.

    Wear (recall 0.99) and shell are now saturated and cannot show further
    improvement. Crack (0.39) and dent (0.565) are the whole remaining signal,
    and both are limited by *where* the defect sits.

26. **`verification_report.json` is merge-loaded, so a gate that was not re-run
    silently survives and reads as current.** The 2026-09-11 bundle carried
    `V6b_guard_sweep` blocks computed under the PREVIOUS defect geometry —
    `crack_depths_mm {0: 2.72, 1: 5.28, 17: 9.57}` (the old 2–10 mm range) and
    `current_min_t: 3.0, current_normal_offset: 0.15` — sitting beside
    genuinely fresh V0–V8 results, with nothing in the file to tell them apart.
    Anyone reading it would have concluded the shipped shadow guard was 3.0/0.15
    rather than 0.05/0.3.

    The merge itself is correct (three scripts write into one file; a wholesale
    rewrite would erase the others). What was missing is provenance per block.
    Every gate now carries `_stamp` with the time, commit and **geometry
    digest**, so a block whose geometry differs from the dataset root you are
    reading was computed for a different scene. A block with NO `_stamp` was
    not re-run since this landed — treat it as stale.

    Corollary for the V6b blocks specifically: their `recommended` field still
    optimises "most crack shadowing subject to intact artifact < 0.03", which
    selects an `offset = 0` row with artifact 0.0288. We shipped `offset = 0.3`
    because every `offset ≥ 0.05` row has artifact **exactly 0.0**. The
    recommendation contradicts the shipped setting and should not be acted on;
    the shadow question is closed (finding 3).

27. **The FDTD near-field monitor must cover the rail's length, not just the
    comparison plane, and per-pixel agreement is the wrong gate at the
    metasurface plane.** Both were found while building `fdtd_agreement.py`
    (2026-09-18).

    *Monitor.* The MS-plane comparison carries the z = 30 window up to H_MS
    with the ASM, so the window must contain all the light that reaches the
    aperture. The rail runs y = ±60, and cut-end light crosses z = 30 outside
    the ±42.5 mm monitor LUMERICAL.md used to specify. Measured on rail3D's own
    field (intact rail, plane wave, production shadowing; ASM from the window
    vs a direct solve at H_MS): ±80×±42.5 → complex corr **0.943**; ±80×±70 →
    **0.983**; ±110×±80 → 0.989. `fdtd_plan` now sizes y as
    ±(rail half-length + 2λ), +7% cells. That residual 0.983 is the ceiling on
    any MS-plane comparison through this window, and the tool reports it on
    every run.

    *Gates.* Sweeping a synthetic disagreement showed per-pixel tolerance
    collapsing at the MS plane long before anything the detectors see: at
    99.4% complex correlation only 73% of lit MS pixels sit within 5% / 15°,
    while the 130-detector barcode stays at 0.997. That is finding 2 again
    (raw complex error never converges on speckle), now at the solver-comparison
    level. So the z = 30 row is gated per pixel, and the MS row is gated on
    complex correlation and barcode cosine only.

    Also fixed: full-wave runs were scored against rail3D's **no-shadow** field
    (`compare_wavefronts` predates the shadow retraction, finding 3), which
    would have charged our own switched-off physics to the solver comparison.
    Both tools now use production shadowing.

28. **The FDTD cross-check uses rail3D's own horn as a Lumerical Import source —
    no TFSF — and four things about Lumerical were wrong in the earlier plan.**
    (2026-09-18, researched against the Ansys Optics knowledge base; sources in
    LUMERICAL.md.)

    *Design.* `horn_source.py` writes the horn field on a z = 15 mm plane as an
    Import source (E **and** H, as Lumerical advises for non-Gaussian beams),
    injecting down; the z = 30 monitor above it records only the up-going field,
    so incident and scattered separate without TFSF. The horn beam is far wider
    than the rail (−20 dB at y = ±128 mm at crown height), so the plane covers
    the rays that can reach the rail + 6λ with a 2λ taper; propagated in free
    space it reproduces the full horn on the rail footprint at 0.9989 / 1.0000 /
    0.9991 (z = 0 / −40 / −80) with 72% of the power. Vector completion matters:
    forcing E_x = 0 gave Ez/Ey = 0.35 rms for this steep, wide beam; projecting
    ŷ transverse to each component's k gives 0.17 / 0.14 — the least
    cross-polarisation consistent with Maxwell, and closest to the scalar model.
    Region 198 × 172 × 131 mm: 35.9 M cells at λ/10.

    *Corrections.* (1) **Lumerical is exp(−iωt), like rail3D** ("P(ω) = ∫
    e^{iωt} P(t) dt"): exports align as-is. The earlier docs, `align_external`,
    `case.json` and the target figure all said Lumerical arrives conjugated. (2)
    **The GPU solver does not support TFSF** at all, and runs single-frequency
    Import sources only from **2025 R1.1**. (3) **STL is read as µm** unless the
    layout unit is set to mm first (or `stlimport(file, 1e-3)`). (4) The material
    is **"PEC (Perfect Electrical Conductor)"**. Also: the old rung-0 plate
    (x ±60) cut through its TFSF box, which Lumerical's TFSF rules forbid.

    Rung 0 (`fdtd_agreement.py --injection`) checks the injected horn in an
    empty box before any rail — the one place a direction or convention mistake
    in the Import source would show up cleanly.


29. **The horn is the physical RFspin H-A75-W20, modelled with the textbook
    aperture method — and the old "λ-scaled Face3D horn" was neither a real
    part nor single-mode.** (2026-09-24. Equation numbers are N. K. Nikolova,
    *Lecture 18: Rectangular Horn Antennas*, McMaster Univ. —
    https://www.ece.mcmaster.ca/faculty/nikolova/antenna_dload/current_lectures/L18_Horns.pdf ;
    also Balanis, *Antenna Theory*, Ch. 13.)

    *How the horn field is computed — an analytic field AND a grid of point
    sources.* `field3d.aperture_distribution` is eq. 18.38, the pyramidal-horn
    aperture field

        ψ(u, v) = cos(πu/A) · exp[+j (k/2)(u²/R_H0 + v²/R_E0)]

    — the feed's TE10 cosine stretched to the mouth (E along y: s-polarised),
    times the quadratic phase lag of the flare's longer path to the edges
    (eqs. 18.5–18.8). Nikolova writes e^{+jωt}, hence her −j; rail3D and
    Lumerical are e^{−iωt}, so the code carries +j. Apex distances by similar
    triangles (18.43–18.44): R_H0 = A·L/(A − a), R_E0 = B·L/(B − b), with one
    axial flare length L = R_H = R_E for a realizable horn (18.42).
    `aperture_field` samples ψ at N × N = 20 × 20 cell **midpoints** (weight
    ΔS = AB/N², `aperture_weight`) on the tilted aperture 140 mm out at 55°,
    and each sample radiates through the exact RS-I kernel (`rs_kernel`, the
    same one the rail facets use) into ψ0 (plane), ψ0_face (rail) and ψ2.

    *Which part.* 60 GHz must sit inside ONE part's single-mode band;
    f_c(TE_mn) = (c/2)·√((m/a)² + (n/b)²):

    | part | feed a × b (mm) | 60 GHz is … | verdict |
    |---|---|---|---|
    | previous: Face3D horn × 5/8 | 5.81 × 3.88 | above TE10 25.8, TE01 38.7, TE11 46.5, TE20 51.6 GHz — **4 modes** | overmoded, not a real part |
    | RFspin H-A60-W20 (40–60 GHz) | WR-19 4.78 × 2.39 | 4% below TE20/TE01 (62.8 GHz) | band edge |
    | **RFspin H-A75-W20 (50–75 GHz, 19–21 dBi)** | **WR-15 3.7592 × 1.8796** | **1.50 × TE10 cutoff (39.9 GHz); next modes 79.7 GHz** | **chosen** |
    | RFspin H-A90-W20 (60–90 GHz) | WR-12 3.10 × 1.55 | 1.24 × cutoff, β/k = 0.59 | band edge, dispersive |

    *Dimensions — a fit; confirm with RFspin's drawing or calipers.* RFspin
    publish the band, 19–21 dBi, and a 3D model of the OUTER shell
    (`47558_H-A75-W20_simplified-model_2023.glb`, https://www.rfspin.com/product/h-a75-w20/):
    outer mouth 23.8 × 17.8 mm, the WR-15 mouth, a UG-385/U flange, 31 mm
    flange-to-mouth. 0.5 mm walls give inner A × B = 22.8 × 16.8 mm, and
    L = 28.0 mm makes the model's gain **19.07 / 20.09 / 21.01 dBi at 50 / 60 /
    75 GHz** — the spec across the band. Independent checks: within ~0.6 mm of
    the textbook optimum-gain design (18.51), and both planes sit near their
    optimum phase errors (H: t = A²/(8λR_H0) = 0.388 vs 3/8, eq. 18.24;
    E: B²/(8λR_E0) = 0.224 vs 1/4, eq. 18.37). Walls 0.3–1.0 mm and L 26–31 mm
    move the rail illumination by < 0.3 % (complex corr ≥ 0.997).

    Model at 60 GHz: R_H0 = 33.5, R_E0 = 31.5 mm; ε_t = 0.81, ε_ph^H = 0.78,
    ε_ph^E = 0.84 → aperture efficiency 0.53 (optimum ≈ 0.51, eq. 18.41);
    D = 20.09 dBi; HPBW 15.8° (E) / 17.2° (H) (previous horn 18.8 dBi,
    ≈ 19° / 21°); edge phase error 140° (H) / 81° (E). The rail at 140 mm is
    0.44 of 2D²/λ (D = the 28.3 mm aperture diagonal; was 0.73): radiating near
    field — why the RS sum, not a catalogue pattern, is the right model.
    E-plane = y′–z′ (contains E; uniform amplitude, narrower beam, higher
    sidelobes); H-plane = x′–z′ (cosine taper, lower sidelobes) — hence A > B
    for a near-round beam.

    *Two fixes to Face3D's aperture model* (`config.HORN_FLARE_K`,
    `config.HORN_SAMPLING`; both provenance keys):
    1. **Flare phase uses free-space k (`"k0"`), not the feed's guided β_g
       (`"beta_wvg"`, eq. 18.4).** Once the horn widens the wave is
       free-space-like (Nikolova p. 3). With the old feed β_g = 0.903k —
       harmless (rail corr 0.9997); with WR-15 at 60 GHz β_g = 0.747k, a 25 %
       phase underestimate that inflates D to ≈ 20.9 dBi.
    2. **Midpoint sampling (`"midpoint"`)**, weight AB/N². Face3D's
       edge-inclusive linspace grid over-weighted the amplitude by 2.2 %.
    V1 passes `flare_k="beta_wvg", sampling="linspace"` with Face3D's λ=8 horn
    explicitly, so it still tests solver equivalence; `compare_wavefronts.LAM8`
    keeps Face3D's horn as the λ=8 reference.

    *The choices barely matter; the part does.* Rail illumination, complex
    corr: RS-I cosθ vs Huygens (1+cosθ)/2 obliquity 0.99997; midpoint vs
    linspace 0.9986; exact vs quadratic phase 2.8° max; **previous → new horn
    0.977** (a 2.3 % change). On a flat plane at crown height the −3 dB
    footprint along the rail narrows 42 → 36 mm and the rail ends (y = ±60)
    brighten 0.28 → 0.32 of peak — so re-check the SEG_LEN = 120 mm
    truncation criterion with the new beam (SETUP_LAB §7e).

    *Limit, and how it is checked.* The aperture method is itself a PO
    approximation: no reflections inside the horn, no rim diffraction
    (Nikolova p. 15) — though it predicts measured gain well, which is why
    horns are gain standards. **V9** holds the code to the textbook.
    **Lumerical rung −1** (`horn_fdtd_case.py` → `fdtd_agreement.py --horn`,
    LUMERICAL.md §5) measures what the method omits; `horn_source.py
    --aperture-from` then carries the full-wave aperture into the rail runs
    (fits one complex gain so only the SHAPE comes from FDTD; synthetic round
    trip corr 0.99987).

    *Consequences.* `SIZE_ANT`, `HORN_FLARE_K` and `HORN_SAMPLING` are
    provenance keys: every dataset and checkpoint made with the previous horn
    is refused, and the next prelim generates into a fresh auto-suffixed root
    (finding 23; tag `ee8367` for this config). The prelim result of findings
    24–25 was produced with the PREVIOUS horn. The config comment that called
    the scaled horn "single-mode-identical" was wrong and is corrected.


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

## 7. Current state / what remains (2026-09-29)

**Picking this up cold: read `NEXT_SESSION.md` first** — the short handoff with
the exact next commands.

Done and committed on **`3D_railhead_upgrade`**:
- λ=5 migration verified end to end on the 5090 (V0–V8, 2026-09-04); H = 30λ
  re-confirmed by `scan_geometry.py` at λ=5.
- Shadow guard settled: `SHADOW_MIN_T = 0.05`, `SHADOW_NORMAL_OFFSET = 0.3`
  (finding 3).
- Defect model rev.3 (2026-09-12): 3-line box cracks 4–8 mm deep, 1.5–3 mm
  wide; shells confined to the gauge corner.
- First end-to-end prelim run (2026-09-11, `L5_prelim_9d5878`, PREVIOUS
  horn): pipeline clean, optimiser not — the no-MS baseline beat the SLM
  (findings 24–26).
- Lumerical FDTD tooling: horn Import source (no TFSF), rung 0 and rail
  scorers, target figures, Layout mockup (findings 27–28, LUMERICAL.md).
- **Real horn (2026-09-24, finding 29):** RFspin H-A75-W20, textbook aperture
  model, V9 gate, rung −1 full-wave horn case, `horn_source.py
  --aperture-from`.
- `scipy` declared in `requirements.txt` (the lab venv lacked it);
  `preflight.py` names missing packages.
- Overview deck updated with horn / FDTD / prelim slides
  (`data/generated/rail3D_overview.pptx`, gitignored; builders in
  `presentation/`).

**Remaining, in order** (lab 5090 unless noted; commands in SETUP_LAB §7e):
1. `git pull`, `pip install -r requirements.txt` (never `-U`), `preflight.py`,
   `tests_plumbing.py`, `tests_physics_3d.py` (V9 included).
2. `validation_3d.py` — V5–V7 under the new horn.
3. SEG_LEN cut-end re-check with the new beam (not scripted yet).
4. `run_stage.py --stage prelim --fresh --with-baseline --bundle` → fresh
   auto-suffixed root.
5. `ablate_surface.py --stage prelim --long` on that root — the
   baseline-beats-SLM question (finding 24).
6. Optional: `scan_geometry.py` (H = 30λ was chosen under the old beam).
7. Lumerical: rung −1 (horn) → rung 0 (empty box) → plate → intact → crack
   (LUMERICAL.md §5).
8. Confirm the horn's inner dimensions (drawing / calipers). A 60 GHz
   meta-atom library (`surface="metaunit"` is blocked at λ≠8).

Not built yet: the rung-3 **difference-field** scorer in `fdtd_agreement.py`;
plumbing test **P6** for `--horn` / `--aperture-from` (both verified only by
scratch round trips: an ideal export passes and a flat-phase horn fails
`--horn`; aperture-from reproduces the analytic source at corr 0.99987).

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
- **Aperture-model horn**: the H-A75-W20's TE10 cosine aperture field with
  quadratic phase (Nikolova eq. 18.38) — no internal reflections, no rim
  diffraction, no measured pattern; inner dimensions fitted to the gain spec
  (finding 29). Lumerical rung −1 measures what this omits.
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
- Occluder mesh is λ/2, with the shipped guard `SHADOW_MIN_T = 0.05`,
  `SHADOW_NORMAL_OFFSET = 0.3` (finding 3): the intact self-shadow artifact is
  exactly 0.0, and the λ/2 occluder captures 98% of what a λ/8 one sees on a
  9.6 mm crack (`crack_shadow_omitted_by_coarse_occluder = 0.0113`).

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
