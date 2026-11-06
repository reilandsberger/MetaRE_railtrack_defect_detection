# Next phase: the F1 setup sweep, data-centric ML, and the story for the slides

*Last updated: 2026-11-06 (sweep_05b slide-size figure) · 2026-11-05 (F2 refined sweep: θ 40–55 × H 125–200, plane out to the horn's beam limit, §1 D; results by day in data/generated/<date>/; detection only: `train_detect.py`, sweep detection figures, a 1-hour sweep slice first; ablation: every SLM loses; SLM on hold, detector-only focus; geometry override built; the ceiling is too loose to prune) · 2026-11-04. Decisions so far (user, 2026-11-04): go with F1 (one horn, one
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

**Ran 2026-11-05: every SLM setting loses (best 0.889 vs control 0.950; README finding
24).** **User decision (2026-11-05): the SLM is on hold.** Work focuses on training and
optimising the detector-only system (`surface="none"`), which scores best. The sweep's
floor IS that system (no metasurface, 8 detectors), so the sweep ranks it directly; the
calibration set in §2 trains only `surface="none"`. `ablate_diagnose.py` stays for
whenever the SLM question is reopened.

**Detection only (user decision 2026-11-05).** Classification is set aside: a rail is
flagged when its detector barcode leaves the intact bounds. Two rules are scored
everywhere: total barcode difference (1 − cos to the mean intact barcode, what training
optimises) and few-detector out-of-bounds (largest |z| of any detector vs the intact
spread). `TrainConfig(task="detect")` trains without the class losses and selects on
AUC (P9).

**Results by day.** `train_detect.py` writes into `data/generated/<YYYY-MM-DD>/`, and
`sweep_eval.py` copies its summary and figures to `data/generated/<YYYY-MM-DD>/sweep_<root>/`
(`config.day_dir`). Set the day once per shell, so an overnight run stays in the day it
started, and keep the logs there too:

```bash
export RAIL3D_DAY=2026-11-05 && mkdir -p data/generated/$RAIL3D_DAY
```

**B. Detector-only training, today's geometry** (~3 min per detector count, on
`L5_prelim_689da5`; also scores run_stage's detect+classify no-MS model beside it):

```bash
python -u train_detect.py --stage prelim --n-det 4 8 12 16 2>&1 | tee data/generated/$RAIL3D_DAY/train_detect.log
```

Send back the day folder (`data/generated/2026-11-05/`), zipped.

**C. The sweep**, after a 2-minute timing run. Run a 1-hour slice first: every placement
gets the same balanced subsample, so `sweep_eval` gives a first comparison tonight, and
re-running the sweep command with the longer budget continues it.

```bash
python -u setup_sweep.py --timing 20 --profile lab
```
```bash
python -u setup_sweep.py --profile lab --budget-hours 1 2>&1 | tee data/generated/$RAIL3D_DAY/setup_sweep.log
```
```bash
python -u sweep_eval.py --profile lab 2>&1 | tee data/generated/$RAIL3D_DAY/sweep_eval.log
```
```bash
python -u setup_sweep.py --profile lab --budget-hours 8 2>&1 | tee -a data/generated/$RAIL3D_DAY/setup_sweep.log
```

Then `sweep_eval.py` again on the full set. Send back the day folder: it holds
`sweep_F1_*/` (sweep_eval.json, progress.txt, the `sweep_0*.png` figures) and the logs. The fields (~0.3 GB per placement) stay on the lab. The
detection comparison is `sweep_05_detection.png` (every setup: AUC, recall and accuracy
at 5% false alarms, barcode distance; `sweep_05b_detection_slide.png` is the same at slide size), `sweep_06_detection_dials.png` (far edge,
θ × H) and `sweep_07_detection_cells.png` (type × position, today vs the best setup).

**D. The refined sweep, F2 (2026-11-05, an 8-hour GPU window).** The F1 slice
(θ 45/55/65 at H 150, 88% of rails) ranked 45° > 55° > 65°. Its best setup was θ 45° with
the plane at x −135..75 (AUC 0.891 → 0.949, untrained, 8 detectors). That setup won the crown
side but lost the gauge face: s 170 shells went from 18% to 6%. The gauge face (s 160–170)
sends its light toward +x, past F1's fixed +80 mm edge. F2 changes two things:

- **θ in 5° steps around the winner, and H 125–200**: θ 40, 45, 50, 55 × H 125, 150, 175,
  200 = 16 placements. 65° is dropped (worst at every width). H 100 is dropped because the
  horn's beam leaves no room on the horn side (next point).
- **The plane reaches toward the horn up to its beam limit** (`--x-hi auto`,
  `setup_sweep.near_edge_limit`). The horn's lowest edge is level with the plane, so its
  beam comes down onto the plane on the horn side. A metasurface there would shadow the
  rail, and the solve cannot show it (`upward_only`). The limit is where a ray 20° off
  the horn axis (its H-plane null), leaving the aperture's upper edge, reaches the plane:

  | H \ θ | 40° | 45° | 50° | 55° |
  |---|---|---|---|---|
  | 100 | +55 | +64 | +72 | +77 |
  | 125 | +76 | +89 | +102 | +113 |
  | 150 | +97 | +114 | +132 | +149 |
  | 175 | +118 | +139 | +162 | +184 |
  | 200 | +139 | +164 | +191 | +220 |

  Every F1 crop sits inside it. 45° has less room on the horn side than 55°, so the
  45° advantage may shrink once each angle gets its full plane: that is what F2 tests.
  `sweep_eval` adds "extend_near" crops (each far edge, near edge at the limit), and
  `sweep_06` gains a panel D: the AUC gain from the wider near edge per θ × H.

At F1's measured rate (~0.40 s per rail on a 135-pixel-wide plane), all 16 placements in
full would take ~9 h. A 6-hour budget gives each placement the same balanced ~65–70%
(~3,200 rails), and F1's learning curve was flat past 50%. `sweep_eval` then takes
~1–1.5 h (F1: 15 min for 3 placements). F2 is a new root because its planes differ;
F1 stays as it is.

```bash
python -u setup_sweep.py --timing 20 --profile lab --name F2 --x-hi auto --thetas 40 45 50 55 --heights 125 150 175 200
```
```bash
python -u setup_sweep.py --profile lab --name F2 --x-hi auto --thetas 40 45 50 55 --heights 125 150 175 200 --budget-hours 6 2>&1 | tee data/generated/$RAIL3D_DAY/setup_sweep_F2.log
```
```bash
python -u sweep_eval.py --profile lab --root data/generated/setup_sweep/F2_f5be4bbb 2>&1 | tee data/generated/$RAIL3D_DAY/sweep_eval_F2.log
```

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
