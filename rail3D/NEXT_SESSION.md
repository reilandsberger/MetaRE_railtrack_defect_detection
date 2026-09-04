# rail3D — session handoff, 2026-09-04

## Context

The λ=8 → λ=5 mm (60 GHz) migration is complete and verified end to end on the
lab RTX 5090. This note exists because the session ran out of usage mid-way
through one open technical question and one pending decision. Everything below
is committed on `3D_railhead_upgrade`.

**Read first:** `CLAUDE.md` (current state), then `rail3D/README.md` §4
(conventions), §6 (findings 3, 17, 18, 19 are the recent ones) and §8
(limitations). All three carry a `Last updated` line — bump it with any change.

## FIRST ACTION: push

Two commits are **unpushed**. The lab machine cannot see them:

```bash
git push origin 3D_railhead_upgrade
```

- `e171767` — lab V0–V8 results recorded; two of my earlier claims corrected
- `f046554` — Plan A (shadow normal-offset fix) + V6b harness rebuild

## Where things stand

All eleven gates PASS at λ=5 on the 5090 (`lab_report.py`). Highlights:

- **V3 = 0.0010273 at both λ=8 and λ=5, on two machines** — the scaled replica
  confirmed numerically to ~5 significant figures.
- **H = 30λ = 150 mm confirmed** by a λ=5 `scan_geometry.py` re-run; the
  dark-field structure reproduces (10λ collapses to 0.523 field AUC, near
  chance, at 20× the energy).
- V5 r = 0.958 (was 0.976 at λ=8) — envelope agreement with a fringe offset,
  i.e. finite-segment vs infinite-extrusion registration, not solver error.
- V7 min cosine 0.9948 — mesh converged at λ/8.
- V8 all sub-gates pass; resume mismatch exactly 0.0.
- Generation measures **4.62 h** (1.30 samples/s), not the 1.5–2 h I estimated.
  The ray-cast is O(rays × triangles) with BOTH scaling ×2.56 = 6.6×.

**No trained λ=5 result exists yet** — only the 30-epoch smoke run, whose
validation split (8 defect / 3 intact) is noise.

## THE open question — resolve this before generating

`config.SHADOW_MIN_T = 3.0` makes ray-cast shadowing **inert** (V6's
`crack_shadow_with_resolved_occluder = 0.0`; 7 of 8 samples exactly 0.0).

Plan A found and fixed the cause: `raycast_shadow_mask` nudged the ray origin
1 µm *along the ray*, which has no perpendicular component at grazing
incidence — the comment said "lift off the surface", the code lifted along it.
New `config.SHADOW_NORMAL_OFFSET` (0.15 mm) lifts along the face **normal**.

It works on the artifact side. Measured per augmentation (laptop, after
reproducing the lab's 0.0020/0.0451 exactly):

| augmentation | (3.0, off=0) | (0.05, off=0) | (0.05, off=0.15) |
|---|---|---|---|
| roll +2.0, jit (+4,−4) | 0.0000 | 0.0000 | **0.0000** |
| roll −2.0, jit (−4,+4) | 0.0020 | 0.0451 | **0.0000** |
| roll +1.0, jit (0,0) | 0.0000 | 0.0295 | **0.0000** |

The whole 0.0451 came from ONE augmentation (which is why the old max-of-3
statistic was untrustworthy). The offset removes it entirely — better than
min_t = 3.0, which still leaves 0.0020.

**But the crack side is unexplained**, at (min_t 0.05, offset 0.15):

| crack depth | production occluder (λ/2) | resolved occluder (λ/8) |
|---|---|---|
| 2.72 mm | 0.0495 | **0.0000** |
| 5.28 mm | 0.0615 | **0.0000** |
| 9.42 mm | 0.0631 | 0.0617 |

A *finer* occluder should resolve more shadowing, not less. Two candidates
with opposite conclusions:

1. **Offset too large for shallow grooves** — 0.15 mm is ~7% of a 2 mm crack
   width, so the lift may push origins through the near wall. Fix: smaller
   offset; find the knee.
2. **The 0.0495 is itself artifact** — the λ/2 occluder renders a 2 mm
   hairline in 0.8 facets, a crude V that may create a spurious blocker. Then
   the coarse-occluder "signal" was never real.

**Run this to decide:**

```bash
python validation_3d.py --min-t 0.05 --normal-offset 0 0.02 0.05 0.1 0.15 0.3 2>&1 | tee ../offset_sweep.txt
```

Crack effect at the *resolved* occluder rising as the offset shrinks → case 1,
pick the knee. Staying 0.0000 at every offset → case 2.

**Defaults are deliberately unchanged** (`SHADOW_MIN_T = 3.0`,
`SHADOW_NORMAL_OFFSET = 0.15` ≈ current behaviour) until this resolves. Both
are **provenance keys**: changing either refuses every existing dataset and
requires regenerating into a new `--name` root.

Fallback if it stays inert: `SHADOW_MODE = "none"` gives a **bit-identical**
dataset (V6 proves the fields match at min_t=3.0) in roughly a third of the
time — ~1.6 h instead of 4.62 h.

## Full-wave comparison — solver chosen, exporter BUILT (2026-09-04)

The user asked: is Ansys **Zemax OpticStudio** enough for a reflected-wavefront
comparison, or does it need **Lumerical FDTD**? Answer: **neither** — it is a
MoM problem. Numbers below are measured, not estimated.

- **Zemax is not a valid reference.** Its Physical Optics Propagation is a
  scalar Fresnel/Kirchhoff beam propagator — the *same approximation family* as
  `field3d.py`. It would agree with us by construction and prove nothing. Its
  non-sequential mode scatters incoherently (no phase), so it cannot return a
  complex field on the plane at all.
- **Lumerical FDTD is valid physics but the wrong shape.** Boxing the whole
  scene out to the plane at z=150 is **387 Mcells at λ/20 ≈ 39 GB** — over a
  32 GB GPU, and most of it is empty air we already trust (V3 checks that
  propagation against ASM to 0.13%). Only sensible **scoped**: box the rail
  alone (23 Mcells / 2.3 GB for a 30 mm segment, 85 Mcells / 8.5 GB for 120 mm),
  near-field monitor, project analytically to the plane.
- **MoM/MLFMM is the fit**: PEC surface, open region, mesh is 2-D only.
  Measured on the exported **closed** body at the production 120 mm segment
  (405 cm²): **562k RWG unknowns at λ/10**, 233k at the exported λ/8 density.
  Dense matrix would be 5.05 TB, so MLFMM is mandatory, but 562k is routine for
  it. → **Ansys HFSS-IE** (needs the *Integral Equation* solver licensed — an
  HFSS FEM seat alone will not run it) or **Altair FEKO**.
- **HFSS SBR+** is shooting-bounce-ray PO with PTD edge corrections: an upgrade
  on our model, not an independent check. A useful third point, not the
  validator.

**STILL UNANSWERED: which solver the user actually has licensed.** Ask. The
exporter is solver-agnostic, so this blocks the run, not the code.

### What is now implemented (all four planned items, self-tested)

`compare_wavefronts.py --export-case DIR [--source plane] [--export-closed]`

- **STL export** (binary, alongside the OBJ) — the format HFSS/FEKO actually
  import. Units are not carried by STL: import as **mm**.
- **`--export-closed`** caps the swept shell into a watertight body
  (verified: 0 boundary edges, Euler characteristic 2, outward normals, closed
  volume 121,499 mm³ for a 30 mm segment). MoM puts current on **both** faces of
  an open sheet, which is not what an opaque rail does.
- **`--source plane`** (`field3d.scattered_fields(source="plane")`) illuminates
  with a unit plane wave along the same −d direction, removing the horn aperture
  model as a confound. It forces `compute_psi2=False` — psi2 is re-radiation off
  the horn, and a plane wave has no horn. Measured complex corr(plane, horn) =
  0.902 on an intact 20 mm segment: same scatterer, different illumination taper.
  **The horn path is byte-identical** — V1 psi1 error still 1.1621e-07.
- **`align_external()`** fits the two bookkeeping mismatches before any metric
  or figure, and *reports* them rather than silently applying them:
  (a) rail3D uses `exp(−iωt)` → `exp(+ik₀R)`; HFSS/FEKO/Lumerical use
  `exp(+jωt)` → `exp(−jk₀R)`, so **external fields arrive conjugated**;
  (b) one complex gain α = ⟨cand,ref⟩/⟨cand,cand⟩ absorbs source normalisation
  and units. Round-trip verified to <5e-7 for conjugation + 137× gain + 55°
  phase, and uncorrelated noise is flagged `ambiguous` rather than flattered.

`case.json` now also carries mesh stats, MoM unknown counts, the cell-centred
grid formula, and an explicit `conventions` block (time convention, amplitude,
and the warning that rail3D is **scalar** — export one component, E_y/TE is the
cleanest match).

### Segment length is NOT free — measured, and it changed the plan

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

The exporter measures this itself (`cut_end_illumination`, folded into
`case.json` → `truncation` and printed on export), so it stays honest if the
geometry or wavelength changes.

Validation ladder (one unknown at a time), also written into the exported
README: **flat PEC plate → intact rail → cracked rail, all plane-wave; real horn
last.** Do not start at the bottom — if the flat plate disagrees, the setup is
wrong, not the physics.

Bundles already exported and ready to hand to a solver:
`data/generated/mom_case_intact/` and `mom_case_crack/` (120 mm, closed,
watertight, plane wave, STL + OBJ + case.json + README).

Why this is worth doing at λ=5: PO assumes radii of curvature ≫ λ, and crack
widths are 2–5 mm = **0.4–1λ** at 60 GHz. That is exactly where the tangent-plane
approximation is expected to break, and it is the one thing the V-gates cannot
test — V1–V3 verify we solve *our* integral correctly, and V5 compares against
the 2D code, which shares the assumption.

## Landmines for a new session

- **Never lay out detector centres outside `config.dense_detector_centers()`**
  (README finding 17). Its predecessor emitted out-of-aperture centres that
  silently collapsed 130 windows onto 54.
- `surface="metaunit"` **raises by design** at λ≠8 — the meta-atom library is
  an 8 mm fit and 3.8 mm pillars can't fit a 2.5 mm cell. SLM and "none" work.
- The λ=8 full dataset (20,512 samples) is archived, not deleted, in
  `data/generated/legacy_unverified/`. It is the "previous version" for the
  presentation comparison — do not delete it.
- Speckle scales as **λ¹, not λ²** (grain 8.0 → 5.0 mm, window/grain 2.27 at
  both). I got this wrong once and corrected it in finding 18 — the horn
  footprint scales with λ, so D is not fixed.
- Two GPUs on the lab box: 5090 (cuda:0) and 4060 Ti (cuda:1). `RAIL3D_DEVICE=cuda:1`
  can run a sweep alongside a generation.
- The user works in **Git Bash** on Windows; `RAILDEFECT_DATA_DIR` must be
  re-exported per shell with forward slashes.

## Verification

```bash
cd rail3D && python preflight.py          # always first
python tests_physics_3d.py                # V0/V0b/V0c/V1-V4, ~5 s CPU
python lab_report.py                      # whole chain, ~10-15 min on the 5090
```

User preference: **run anything non-trivial on the lab 5090**, not the laptop
(MX250, 2 GB). Laptop is for seconds-scale CPU gates only.
