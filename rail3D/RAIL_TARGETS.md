# Rail-reflection rungs 1–3: approval metrics, targets and per-class predictions

*Last updated: 2026-11-02. Thresholds are `CRITERIA` (rungs 1–2) and `DIFF_CRITERIA` (rung 3) in
`fdtd_agreement.py`. They are proposed; revisit them after the first real rung 3. Figures are in
`data/figures/rail_targets/`; regenerate them with the commands in §6. The companion for rung −1
is `HORN_RUNG-1_TARGETS.md`.*

Every **TARGET** figure here is synthesized: rail3D plus a stated perturbation, i.e. what a
*passing* run looks like. Every **PREDICTION** figure is rail3D re-weighted by an exact 2D
solution, i.e. what we *expect* Lumerical to show. Neither is an FDTD result, and both are titled
and footnoted as such. Keep those labels when they go into a deck.

## 1. The ladder and what “good enough” means (Fig. 00)

![00](data/figures/rail_targets/00_approval_ladder.png)

| rung | compares | gate | why this number |
|---|---|---|---|
| 1 plate | absolute field at z = 30 and the MS plane | complex corr ≥ 0.98; ≥ 85% of lit pixels within 5% / 15°; worst pixel ≤ 15% / 30°; MS barcode cos ≥ 0.98 | both solvers are exact on a plate, so a miss is a setup error |
| 2 intact rail | the same | the same | smooth head, R ≥ 13 mm = 2.6 λ: physical optics should hold |
| 3 each defect | the **difference field** (defect − intact), with ONE gain fitted on the intact run | diff corr ≥ 0.90 (z = 30 and MS) | the same signature: the residual is ≤ 44% of its amplitude |
| | | diff level within ±3 dB | a 2× power error; the design study calls 6 dB "robust", so the model keeps at least half that margin |
| | | detector signature (130 dense detectors, defect − intact) cos ≥ 0.95, size within ±3 dB | what a trained classifier actually consumes |

**Why rung 3 is gated on the difference.** The absolute field of a cracked rail is about 99%
the intact rail's. In the targets, the absolute defect field agrees at 98.5–99.2% while the
difference field is anywhere from 50% to 100%. The absolute number cannot see a wrong crack.

**Sign convention:** a level is FDTD (or exact) relative to rail3D. **+ means rail3D
under-predicts** the defect's signal; − means it over-predicts.

## 2. Is the crack the worst match? Yes, but not because of ray bounces (Fig. 01)

![01](data/figures/rail_targets/01_crack_physics.png)

The intuition is right about the ranking. Cracks are the only class that fails often, and
the mechanism is close to the one guessed. What needs correcting is the picture of the
mechanism.

- **The defect model sets the scale.** rail3D's cracks are box slots 1.5–3 mm wide (0.3–0.6 λ)
  and 4–8 mm deep, in groups of three.
- **There is no room for ray bounces.** The slot is narrower than a wavelength, so there are no
  geometric rays bouncing off its walls. It is a **waveguide**, and which waveguide depends on
  the polarisation relative to the crack. The horn's E is along the rail (y), so:
  - **Longitudinal crack** (slot along y, E ∥ walls): a TE mode with cutoff at width
    λ/2 = 2.5 mm.
    - Narrower slots are **dark below the rim** (top-left map: evanescent). The real signal
      then saturates at ~1 mm depth: deeper cracks look the same.
    - rail3D's floor is shadowed at 55° anyway, so the two agree to −3.7…+2.5 dB for widths
      ≤ 2 mm.
    - Above cutoff (2.5–3 mm) the mode propagates and resonates with depth. rail3D has no
      mode, so it misses by up to **19 dB**, mostly **over-predicting** where the real field
      cancels.
  - **Transverse crack** (slot across the head, E ⊥ walls): a TEM mode with **no cutoff**. The
    light fills the slot to the floor and forms standing waves: a shorted stub with resonances
    near λ_t/4, 3λ_t/4 (λ_t = λ/cos θ = 8.7 mm).
    - rail3D lights the floor and so gets a two-path interference with the right period.
    - It misses the end correction (fringing at the mouth) and the resonant build-up, so the
      error swings **±5 dB with depth** (top-right).
    - The far-field shape also differs: the real slot radiates like a magnetic line source,
      evenly in every direction (bottom-middle). rail3D's floor patch radiates with a cosine
      obliquity. So the all-direction correlation drops to ~0.6–0.8 even where the
      metasurface-window level is right.
- **"Many orders of reflection" is the ray limit of the same thing.** The standing wave in a
  slot wider than λ is exactly the multiple-bounce picture. At 0.3–0.6 λ, the mode picture is
  the one that predicts the numbers.
- **It is a polarisation effect, which a scalar model cannot represent.** The same slot is
  dark or bright depending on its orientation to E. rail3D gives both orientations the same
  scalar treatment.

## 3. Cracks over the dataset's range (Fig. 02)

![02](data/figures/rail_targets/02_crack_maps.png)

The cells are three-line groups. Top row: level, exact − rail3D, in the metasurface window.
Bottom row: correlation over all directions.

| | width 1.5 | 2.0 | 2.5 | 3.0 |
|---|---|---|---|---|
| longitudinal, crown (55°) | −3.5…−3.7 | +0.2…+0.6 | −4.8…+5.5 | −13.9…+4.5 |
| longitudinal, shoulder (35°) | −1.3…−1.4 | +2.3…+2.5 | −2.8…+4.6 | −19.2…+3.6 |
| transverse, crown | −0.4…+4.4 | −0.5…+4.5 | −0.6…+4.9 | −0.7…+5.4 |
| transverse, shoulder | −3.4…+3.9 | −2.9…+4.7 | −2.7…+5.3 | −2.4…+2.4 |

(dB, the range over depths 4 / 6 / 8 mm.)

- **Longitudinal cracks below cutoff:** a steady, modest bias, so a correction factor could
  remove it.
- **Longitudinal cracks above cutoff, and transverse cracks:** the error depends on depth to
  within a millimetre (resonance phase). It cannot be corrected by a factor. The *spread*
  (±5 dB transverse, up to −19 dB longitudinal) is the honest uncertainty rail3D's crack
  training data carries.

## 4. Every class against the rung-3 gates (Fig. 03)

![03](data/figures/rail_targets/03_class_scorecard.png)

The 2D exact solution vs rail3D's physical optics covers 162 cases, each defect cut along both
of its axes:

| class | cases | level, exact − rail3D (median / worst) | MS-window corr (min) | all-direction corr (median / min) | inside the level + MS-corr gates | worst case |
|---|---|---|---|---|---|---|
| **wear** | 12 | 0.0 / 0.0 dB | 1.000 | 1.00 / 1.00 | **100%** | none: R 150–300 mm, PO is exact |
| dent, x-z cut | 18 | 0.0 / +1.9 | 0.892 | 1.00 / 0.95 | 94% | 8 mm deep, FWHM 10 (tip radius ~2.3 mm < λ/2) |
| dent, y-z cut | 18 | 0.0 / **−9.1** | 0.962 | 1.00 / 0.85 | 89% | the same narrow deep dent, over-predicted by rail3D |
| shell, x-z cut | 9 | 0.0 / +1.7 | 0.934 | 0.99 / 0.80 | 100% | 5 mm deep, radius 4 |
| shell, y-z cut | 9 | 0.0 / +2.7 | 0.999 | 1.00 / 0.91 | 100% | 5 mm deep, radius 4 |
| **crack, longitudinal** | 24 | −0.5 / **−19.2** | 0.658 | 0.94 / 0.27 | **46%** | 3 mm wide (above cutoff), 4 or 8 deep |
| **crack, transverse** | 24 | 0.0 / **+5.4** | 0.989 | 0.94 / 0.49 | **67%** | 6 mm deep at the crown (3λ_t/4 resonance) |

**Ranking, best to worst:** wear > shell ≈ dent (except narrow, deep dents) > transverse crack
> longitudinal crack. Single-line cracks behave the same way: 12/24 and 18/24 inside.

## 5. Rung 3 in 3D: targets and predictions (Fig. 04 and one figure per class)

![04](data/figures/rail_targets/04_rung3_summary.png)

Per-class figures: `rung3_<class>_target.png` and `rung3_<class>_predicted.png`. Each shows the
difference field at z = 30 and at the MS plane, both solvers, the residual, a cut through the
peak, the 130-detector scatter and the gate table. The sample is each class's seed-0 draw,
i.e. what `compare_wavefronts --export-case` writes for the lab, with the crack orientation
pinned.

| sample (seed 0) | **target**: loosest passing agreement | **predicted**: MS diff corr / level | detector signature cos / size | verdict |
|---|---|---|---|---|
| crack transverse (2.34 × 4.36 mm, 3 lines) | 96% | 99.6% / −0.8 dB (exact geometry in 2D: **+1.6 dB**) | 0.950 / +0.4 dB | **borderline**: on the signature gate; all-direction corr 0.76 in 2D puts the z = 30 gate at risk |
| crack longitudinal (same, rotated) | 96% | 96.5% / **+4.4 dB** (exact 2D: **+4.7**) | 0.931 / **+7.5 dB** | **expected FAIL**: Lumerical sees more crack than rail3D |
| crack oblique (45°) | 96% | 98.9% / 0.0 dB | 0.994 / −0.2 dB | pass (but see §7: averaged, not solved) |
| dent (3.7 mm, FWHM 16–18) | 96% | 100% / 0.0 dB | 0.994 / +0.1 dB | pass |
| wear (6.5 mm at 40°, R 343) | **99.5%** | 100% / 0.0 dB | 0.987 / −0.5 dB | pass |
| shell (3.4 mm, 6.5 × 6.2) | 96% | 99.8% / +0.2 dB | 0.993 / −0.3 dB | pass |

**Wear needs 99.5% field agreement, every other class 96%.** Wear's difference field is
*stronger* than the intact field (+2.5 dB at the MS plane). There the detector signature is
dominated by |Δ|², so a field residual r costs ~2r in the signature rather than ~r. The target
walks 96 → 98 → 99 → 99.5% and reports the first level that passes (`DIFF_PASS_LADDER`).

**What a real rung 3 will most likely show (the lab exports the transverse crack):** the level
passes, at +1.6 dB. The detector signature sits on its 0.95 gate. The z = 30 correlation is at
risk from the far-field shape difference. A **longitudinal** run would fail on level by ~+4.5 dB,
in the direction "rail3D under-counts" a 2.3 mm crack. That is a physics finding about rail3D's
crack training data, to be reported, not tuned away (LUMERICAL.md §5).

## 6. Reproduce

```bash
python po_canonical_2d.py --validate                 # flat plate exact, PEC cylinder vs Bessel series (0.4-0.5%)
python po_canonical_2d.py                            # 162 cases + depth sweep, ~45 min CPU; then the figures
python fdtd_agreement.py --diff-target all           # rung-3 TARGETS, one figure per class (~3 min laptop GPU)
python fdtd_agreement.py --diff-target all --predicted   # rung-3 PREDICTIONS from po_canonical_2d.json
python rail_targets_figs.py                          # Figs 00-04 from the stored JSONs
```

A real rung 3 (LUMERICAL.md §5): the crack run and an intact run **with the identical mesh
override**, both exported at z = 30:

```bash
python fdtd_agreement.py --sample crack --external data/generated/fdtd_crack/crack_z30.mat --intact-external data/generated/fdtd_crack/intact_z30.mat
```

## 7. What the prediction does and does not cover

- **2D, locally flat.** Each defect is cut across one axis, on a flat PEC surface tilted to the
  local incidence (crown 55°, shoulder 35°, gauge corner 20°). The rail's own curvature and the
  crack's finite length (10–50 mm, tapered ends) are not in it.
- **The transverse cut is exact for an x-invariant slot under the horn's oblique k.** The
  reduced problem is separable at k_t = k cos θ.
- **Oblique cracks were not solved.** Their 3D prediction averages the two cuts' transfer
  functions. Opposite-sign errors partly cancel there, so treat its pass as unverified.
- **3D predictions re-weight rail3D's own difference field, direction by direction**, by the
  nearest 2D case's exact/rail3D ratio. The mapping is stated in
  `fdtd_agreement.predicted_diff`.
- **Neumann (y-z) profiles are truncated at ±160 mm.** The slot's grazing radiation diffracts
  off the ends, which puts a ±10% ripple on the exact all-direction patterns (Fig. 01,
  bottom-middle). The metasurface-window numbers are unaffected.
- **PEC, single frequency, no surface roughness, no rail-steel conductivity.** These match what
  Lumerical will model; whether the real rail matches either is a different question.
