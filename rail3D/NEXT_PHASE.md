# Next phase: the F1 setup sweep, data-centric ML, and the story for the slides

*Last updated: 2026-11-05 (geometry override built; the ceiling is too loose to prune) · 2026-11-04. Decisions so far (user, 2026-11-04): go with F1 (one horn, one
wider metasurface); sweep the setup dials on a controlled grid (6 types × 7 positions ×
~100, plus ~600 intact); keep finding 31's horn rule; overnight budget; score with field
level + a fast classifier where that is provably enough, full training otherwise.*

## 1. Tonight on the lab (5090)

**A. If `L5_prelim_689da5` does not exist yet** (SETUP_LAB §7e not run since the wear
change): the prelim dataset and the ablation come first, ~1.5 h.

```bash
python run_stage.py --stage prelim --fresh --with-baseline --bundle 2>&1 | tee ../prelim_newhorn.log
```
```bash
python ablate_surface.py --stage prelim --long 2>&1 | tee ../ablation_newhorn.log
```

**B. The sweep**, after a 2-minute timing run sets the night's expectations:

```bash
python setup_sweep.py --timing 20 --profile lab
```
```bash
python setup_sweep.py --profile lab --budget-hours 8 2>&1 | tee ../setup_sweep.log
```
```bash
python sweep_eval.py --profile lab 2>&1 | tee ../sweep_eval.log
```

The sweep is resumable and breadth-first: if it stops at the deadline, every placement
has the same balanced subsample. Re-running the same command continues it. Send back
`data/generated/setup_sweep/F1_*/sweep_eval.json`, `progress.txt`, the four
`sweep_0*.png` and both logs. The fields (~0.3 GB per placement) stay on the lab.

## 2. What the sweep measures

| dial | values | cost |
|---|---|---|
| horn angle θ × metasurface height H | θ 45, 55, 65° × H 100, 150, 200 mm (horn distance follows from finding 31) | one solve set per pair: 9 placements |
| metasurface far edge | 0.35, 0.5, 0.65, 0.8, 0.9, 1.1 × the crown mirror point | free (crops) |
| metasurface position at fixed width | 150 or 265 mm wide, slid in 25 mm steps | free (crops) |
| metasurface along the rail | `--y-half` (37.5 = today) | solve-time |

**The defects:** 6 types (transverse, longitudinal and oblique crack; dent; shell; wear)
× 7 positions s = 110…170, ~100 each.
- Sizes are Latin-hypercube stratified over each class's operating ranges.
- Position is a clean experimental factor: bands are overridden, and wear at s is a wheel
  cut tangent at s.
- Every placement sees the same rails in the same poses, so comparisons are paired.

**Every defect also stores its exact signal:** the defect minus the same-pose intact rail.
It comes from the defect's own facets, at ~10% extra cost, and is verified to 10⁻⁷ by
`--self-test`.

**The decision rule.** A phase-only metasurface is lossless, so each setup has two
bounds:
- **Floor:** no metasurface, 8 detectors. The SLM at zero phase reproduces it exactly, so
  a trained SLM can always reach it.
- **Ceiling:** the ideal matched filter on the full aperture field, against the intact
  pose spread.

A setup whose ceiling is below another setup's floor cannot win (`sweep_01_bounds.png`).

**But the laptop smoke run (2026-11-05) put the ceiling at ~1.00 for every setup.** An
ideal processor with the whole complex field rejects the few-degree-of-freedom pose motion
almost perfectly, so this rigorous bound is too loose to prune anything. With 12 intact
poses that is partly rank deficiency; the lab's 600 may tighten it, but don't plan on it.
So, per your rule, **some full training is needed**. It is needed for calibration, not for
every setup:
1. Rank every setup by its floor (cheap, all of them).
2. Fully train a **calibration set** of ~4 setups: today's, the top 2 floors and one
   mid-rank. Train both the SLM and no-MS ONN on each.
3. **Headroom** = trained SLM − floor. If it is small and similar across the calibration
   set (smaller than the floor gaps between ranked setups), the floor ranking is
   validated, and the rest are pruned on the floor, with the headroom as the stated
   margin. If it varies a lot, the floor is not a safe proxy, and the finalists are
   chosen by training.

This needs the ablation to show the SLM can beat the no-MS control at all (finding 24),
and the geometry override (§3.2, built).

## 3. After the sweep

1. **Rank by the floor, and check the ceiling anyway.** At 600 poses the ceiling may prune
   setups that collect too little defect light; if it does not, use the calibration rule
   in §2.
2. **Full training of the calibration set and the finalists** (ONN, SLM and no-MS) runs on
   the **geometry override** (built 2026-11-05). `RAIL3D_GEOMETRY=<json>` is read at
   config import, so the whole pipeline (generation, gates, training, analysis) runs on
   that geometry, in its own auto-suffixed roots.
   - `geometry.py` writes the files from the sweep's dials; `geometries/F1.json` is F1
     (θ 55, H 150, x −190…+75 = 106 × 30 px, 23 × 10 detectors).
   - Every key is a provenance key, so a default-geometry dataset is refused under F1 and
     vice versa (checked).
   - P8 in `tests_plumbing.py` guards the derivations.
   - Not yet: turning the sweep's own fields into a training set (same samples, cropped),
     which would skip the generation. Worth building if the calibration set grows.

   ```bash
   python geometry.py --name F1 --theta 55 --H 150 --x0 -190 --x1 75
   ```
   ```bash
   export RAIL3D_GEOMETRY=geometries/F1.json
   ```
   ```bash
   python geometry.py --show && python run_stage.py --stage prelim --fresh --with-baseline --bundle
   ```

   F1's plane is 1.77× today's, so expect roughly 1.5–1.8× today's generation time.
   `unset RAIL3D_GEOMETRY` returns to the default.
3. **The ablation's answer gates step 2.** Until a trained SLM beats the no-MS control on
   today's geometry, the "trained" numbers are not trustworthy on any geometry
   (finding 24).

## 4. Data-centric ML track

The sweep produces a dataset that is designed rather than drawn. That makes these
cheap:

| # | idea | what it answers | where |
|---|---|---|---|
| D1 | **Stratified evaluation**: recall per type × position cell, per size quartile, per trust tag | where the system fails, not just how often | `sweep_03`, `sweep_04` |
| D2 | **Per-sample difficulty labels**: physical SNR (from the stored exact signal), distance to the glint band, local incidence angle, crack width vs the 2.5 mm cutoff | is a miss the model's fault or the physics'? (low SNR = unfixable by training) | sweep metadata + `delta` |
| D3 | **Learning curves per cell** | is 100 per cell enough? where does more data pay? Allocate the next generation budget to cells still climbing (adaptive sampling) | `sweep_04` D |
| D4 | **Simulation-label fidelity** (finding 34): longitudinal cracks above cutoff and transverse cracks carry ±5…19 dB model error | down-weight, or flag as noisy-label strata; test whether training without them hurts real-crack generalisation; calibrate after rung 3 | `po_trust` tag |
| D5 | **Nuisance coverage**: pose (roll, jitter) and y0 drive the false-alarm threshold | held-out poses outside the training range (out-of-distribution test); is the threshold stable? | intact set |
| D6 | **Interpretability**: which detectors carry which class; map them back to the rail through ray tracing (glint band); SLM phase profile | "the system looks at the gauge corner through detectors 3 and 7", in pictures | `slm_profile.py`, new detector-saliency map |
| D7 | **Leakage-safe splits**: split by sample id (pose seed), never by field; the same split for every setup (paired) | honest, comparable metrics | `sweep_eval.folds_of` |

## 5. The slide story (for later)

1. **Problem.** Defects matter on the crown and the gauge corner (s 110–170); today's
   setup sees ~10 mm of it (DESIGN_STUDY Figs 01–02).
2. **Physics → design.** The tilt-window rule; the ray-traced levers; the field scan
   picks F1 (DESIGN_STUDY Figs 04–07).
3. **Can we trust the simulator?** Approval ladder; cracks are waveguides; wear, dents and
   shells pass (RAIL_TARGETS Figs 00–04).
4. **Choosing the setup without training everything.** The bound sandwich; the dials
   (sweep Figs 01–02).
5. **Data-centric results.** Where it fails (type × position), why (SNR, trust, cutoff),
   and how much data is enough (sweep Figs 03–04).
6. **Trained system on the finalist(s).** SLM vs no-MS, interpretability.

## 6. Open questions for the user

- **Buildability of the wide metasurface.** Is ~265 × 75 mm feasible? If not, what width
  is? The "slide" crops already answer "where should a fixed-width surface sit".
- **Defect sizes.** The sweep uses the full operating ranges. Should the small end
  (hairlines, shallow dents) get its own stratum, given that finding 34 says rail3D is
  least trustworthy exactly there?
- **Which candidates go into the calibration set** once the sweep is back: my default is
  today's setup, F1, the top 2 floors and one mid-rank.
