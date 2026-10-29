# rail3D experimental-design study: horn, metasurface and rail geometry

*Last updated: 2026-10-29 · PRELIMINARY: ray optics + field-level physics only, no metasurface
training · figures in `data/figures/design_2026-10-29/` (regenerate with the commands in §8) ·
tools `glint_band.py` (Tier 0) and `coverage_scan.py` (Tier 1).*

**The question (user, 2026-10-29).** Defects matter on the crown / running band AND the gauge
corner: arc **s = 110–170 mm** (the field side is less defect-prone and is left out for now). How do
we make the system see that whole arc? Options on the table: a wider metasurface (its far edge,
away from the horn, extended), a different horn position / angle, more horns, more metasurfaces.

**TL;DR (field-level scan, 7 horn placements × 4–5 apertures, 52 mid-size defects at
s0 110–170 + 16 intact poses, no metasurface training):**

| system | robust recall (≥ 6 dB) | crown s0 110–140 | gauge corner 145–170 | mean margin |
|---|---|---|---|---|
| **today** (θ 55°, x −75…+75) | **0.67** | 0.50 | 0.88 | 8.6 dB |
| **wider metasurface, far edge to −190** (same horn) | **0.92** | 1.00 | 0.83 | 12.4 dB |
| + a second horn at 65°, time-multiplexed on that same surface | 0.96 | 1.00 | 0.92 | 13.9 dB |
| today's surface + a 2nd surface on the crown glint (one horn) | 0.96 | 1.00 | 0.92 | 13.6 dB |
| 65° horn on today's surface + 55° horn / H 100 surface on the crown glint | **1.00** | 1.00 | 1.00 | 15.9 dB |

1. **The single best move is the one the ray trace pointed to: extend the metasurface's far edge
   to x ≈ −190** (×1.8 plane). Crown recall 0.50 → 1.00 with the horn untouched, still dark
   field (crown glint at −214 lands just outside). Costs: ×1.8 generation, a 265 × 75 mm surface,
   more roll motion (18.7 mm per 2°).
2. **A grazing 65° horn is the best gauge-corner channel** (corner 1.00, 14.3 dB on today's
   surface), which ray optics did NOT predict: it covers less glint, but its intact plane is
   2.5× quieter under pose motion, and the score is signal over pose noise. The price is light:
   −3 dB, and it falls apart at 10% receiver noise (recall 0.81 → 0.58).
3. **Every lever that only moves the horn fails on its own** (θ 45°/35°: 0.67/0.73; H 100: 0.63),
   and **azimuth is worse than today** (φ 30°: 0.46, φ 60°: 0.42): dimmer plane, no dark field.
4. **Bright-field apertures see the crown and lose the corner** (crown 1.00, corner 0.29–0.50):
   bright field is pose-noise-limited; use it as a crown-only second channel, if at all.
5. Differences under ~0.06 (3 of 52 defects) are inside this scan's noise. Next: Tier 2,
   training 2–3 finalists, after `ablate_surface.py` fixes the SLM (§7).

---

## 1. The design problem (Fig. 01)

![01](data/figures/design_2026-10-29/01_priority_region.png)

The production geometry (horn 55° from vertical at 278.5 mm, 150 × 75 mm metasurface at
z = 150) reflects the horn **specularly** into the metasurface from only **s 148–158 mm: 9.7 mm
of the 60 mm priority arc (16%)**, on the gauge shoulder. That "glint band" is where the system
sees best: the intact reflection is strong there and a defect's weak field beats against it,

    |E_i + E_d|² − |E_i|² ≈ 2 Re(E_i* E_d)        (a defect in the band: first order in E_d)
    |E_d|²                                         (a defect off the band: second order)

This is the mechanism behind README finding 25 (prelim recall tracked distance to the band;
crack recall 0.03 → 0.82 across s0 107 → 148) and the 2026-10-19 bottleneck analysis.

## 2. Why only ~10 mm: the tilt window (Fig. 02)

![02](data/figures/design_2026-10-29/02_tilt_window.png)

**One rule explains every number in this study.** A rail point whose outward normal is tilted
τ from vertical reflects a horn at θ from vertical (on the +x, gauge side) into the direction α
from vertical (α > 0 toward the horn) when

    τ = (θ + α) / 2          (the normal bisects the two directions)

An aperture spans a range of α, so it sees a **window of surface tilts**. Today, from the gauge
shoulder, α runs −34° … +17°, so τ = 10° … 33°.

**The rail profile turns that tilt window into arc length** at ds/dτ = R, the local radius:

| region | tilt τ | arc | local radius R | 1° of tilt buys |
|---|---|---|---|---|
| crown / running band (s ~110–145) | within ±5° | ~35 mm | 100–800 mm | 2–14 mm of arc |
| gauge corner (s ~148–164) | 10° → 60° | ~16 mm | 13–35 mm | 0.2–0.6 mm |

So the production aperture spends its whole 23° window on the gauge corner, which packs 50° of
tilt into 16 mm. The crown is the opposite: nearly flat, so a few degrees of view would cover
it. But seeing tilt 0° means α = −θ: **the crown's own mirror reflection**, which lands at
x = −H·tan θ = −214 mm (panel C's dots). An aperture there is "bright field".

Two consequences that hold for every option below:
- **Coverage of the crown and coverage of the bright crown glint are the same thing** to ray
  optics. The question is only how much of that glint to accept (§3, §5).
- **The upper gauge corner / gauge face (s ≳ 160, τ ≳ 45°) is out of specular reach from any
  plane above the rail**: reaching τ = 60–80° needs α beyond +65°, i.e. a receiver beside the
  rail. It is seen only by diffuse scattering (which the field scan in §5 does measure).

## 3. The levers (Fig. 04)

![04](data/figures/design_2026-10-29/04_design_levers.png)

Ray-traced coverage of the priority arc (`glint_band.py`; normals smoothed over 1.5 mm, horn
beam −10 dB as the lit limit, horn's lowest edge always level with the plane as in finding 31):

| setup | priority arc in the glint band | crown glint (x = −H·tan θ_eff) | roll 2° moves the glint | glint strength vs today |
|---|---|---|---|---|
| **today**: θ 55°, H 150, x −75…+75 | **9.7 mm** (s 148–158) | −214, 139 mm outside | 12.3 mm | 0 dB |
| far edge −150 (×1.5 plane) | 20.2 mm (s 137–158) | 64 mm outside | 16.3 mm | +4.3 dB |
| far edge −190 (×1.8 plane) | 28.4 mm (s 129–158) | 24 mm outside (edge of bright) | 18.7 mm | +7.5 dB |
| near edge +150 (toward horn) | 12.3 mm (s 148–160) | 139 mm outside | 12.3 mm | −0.5 dB |
| horn θ 45° | 14.4 mm (s 142–156) | −150, 75 mm outside | 12.6 mm | +2.5 dB |
| horn θ 35° | 20.8 mm (s 134–154) | −105, 30 mm outside (edge of bright) | 12.0 mm | +5.3 dB |
| horn θ 65° | 7.3 mm (s 152–159) | −322 | 11.5 mm | −1.7 dB |
| H 100 (closer plane) | 16.8 mm (s 141–158) | −143, 68 mm outside | 9.8 mm | +2.1 dB |
| H 200 | 6.5 mm | −286 | 15.1 mm | — |

What the sweep (panel A, θ × far edge) says:
- **Every lever moves the same tilt window**, so they trade against each other along the
  contours. Coverage is bought by letting the window reach toward τ = 0, and the hatched region
  (crown glint inside, or horn over the aperture) is where that runs out.
- **Widening the metasurface on the far edge is the cleanest lever.** It is a superset of
  today's view (trained detectors can ignore the new area); horn, pose sensitivity and
  hardware stay; it keeps the most dark-field margin per mm gained. Costs: plane points ×1.5
  (−150) / ×1.8 (−190) in generation time, a larger metasurface, and light reaching the far
  edge at 45–50° incidence (the SLM model ignores a unit cell's angle dependence).
- **Lowering the horn moves the window rather than widening it**, and below θ ≈ 30° the horn
  sits over the metasurface (clearance 25 mm at 35°).
- **A closer plane widens the angle and cuts roll motion** (2H·δ), but brings the crown glint in
  sooner (−143 at H 100).
- **Roll is the price of distance**: 2° of roll moves the glint 10–19 mm on the plane, about
  one detector window (11.4 mm). Wider apertures see more oblique rays, which move more.

## 4. The strategy menu (Fig. 05)

![05](data/figures/design_2026-10-29/05_strategy_menu.png)

| strategy | ray-traced priority coverage | notes |
|---|---|---|
| S0 today | 9.7 mm | dark field |
| S1 wider metasurface | 20 mm (−150) / 28 mm (−190) | superset; cost ×1.5 / ×1.8 |
| S2 lower horn angle | 14 mm (45°) / 21 mm (35°) | moves, does not widen |
| S3 closer metasurface | 17 mm (H 100) | less roll motion |
| S4 second metasurface centred on the crown glint (x −289…−139) | 24 mm (s 115–140) | bright field: +10 dB, but 2° of roll moves it 30 mm (rays arrive at 55°) |
| S5 two horns, time-multiplexed (55° + 35°), one metasurface | union 24 mm | frames alternate between horns |
| S6 horn azimuth (horn beside the metasurface along the rail) | 12 mm (φ 30°) / 20 mm (φ 60°) | see the correction below |
| S7 wide metasurface (−150) + two horns | union 42 mm (70%) | horn B's channel is then bright field |

**A correction to the 2026-10-29 brainstorm: tilting the horn along the rail does NOT give a
dark field.** I had argued that an intact rail, being uniform along y, keeps the incident k_y
and so sends its reflection past the aperture. That holds for a plane wave; for a horn at a
finite distance, some point on an infinite rail always reflects it into the aperture, simply
**between horn and plane along the rail** (y ≈ +26 mm at φ 30°, +72 mm at φ 60°, still inside
the beam at −1 / −6 dB). What azimuth really does is lower the effective incidence angle,
**tan θ_eff = tan θ · cos φ** (φ 60° acts like θ 35°), while letting the horn sit beside the
metasurface along the rail instead of across it. That may matter for clearance from the wheel
flange, which is a hardware question.

## 5. Field-level results (Figs 03, 06, 07)

Ray optics says where the glint lands. It cannot say whether a defect's field beats the rail's
own pose motion, so `coverage_scan.py` solves the real thing (ψ1, upward input, production
shadowing) on a wide plane, x −300…+100 at z = H, for every horn placement, and scores every
aperture as a cut-out of it.

**Method.**
- **Samples:** 13 positions s0 = 110, 115 … 170. At each, 4 defects: transverse, longitudinal
  and 45° oblique cracks, plus a dent; mid-size, at y0 = 0. That makes 52 defects, plus 16
  intact rails, all at production pose randomisation (roll σ 2°, jitter σ 4 mm). The same seeds
  are used for every horn.
- **Score:** for each sample, take the best detector-sized window (10 × 7.5 mm) inside the
  aperture: |ΔP| / sqrt(intact pose variance + receiver noise²). The threshold is the largest
  leave-one-out intact score, so about 1 false alarm in 16.
  - **Robust** means at least 2× threshold (6 dB).
  - Receiver noise is 1% of today's mean window power unless stated.
- **Pairs** are OR-fused: a sample's score is the larger of its two channels' (each over its
  own threshold), then re-calibrated on the intact samples.
  - Two surfaces may share at most 12 mm of x; the same aperture name at another θ or H is
    another surface.
- **Cost:** 7 horns × 68 samples at 11 s each on the laptop MX250. That needed the
  defect-facet-only ray cast, proven bit-identical.

### 5a. The evidence (Fig. 03)

![03](data/figures/design_2026-10-29/03_field_evidence.png)

- **A–B.** The production intact plane is not very dark: it is within ~7 dB everywhere, and the
  crown glint peaks at x ≈ −160 to −210. Today's aperture sits on the dim side, 2–6 dB down.
- **C–E.** These are pose-matched differences: the defect minus the intact rail in the same
  pose.
  - **C:** a crack at s0 150 (inside the glint band) lands on and just left of today's
    aperture.
  - **D:** the same crack moved to the crown (s0 120) puts its main lobe at x −260…−140, outside
    today's aperture. Only weak fringes reach it.
  - **E:** a crown dent's lobe is centred at x ≈ −240.
  - The rule holds: **a defect's signal lands where its own part of the rail reflects to**,
    which for the crown is the crown glint.
- **F.** Today, every defect clears the bare threshold almost everywhere (an oracle window
  layout and mid-size defects), but **the crown sits at 1–2× threshold while the corner sits at
  3–10×**. Transverse and oblique cracks on the crown are the weakest: 0–2 dB at s0 110–130.

### 5b. The full scan (Fig. 06)

![06](data/figures/design_2026-10-29/06_coverage_scan.png)

| horn | aperture | robust recall | crown | corner | margin (crown / corner) | recall at 10% noise | light vs today |
|---|---|---|---|---|---|---|---|
| θ 55 (today) | far edge −190 | **0.92** | 1.00 | 0.83 | 14.1 / 10.3 dB | 0.77 | +2.1 dB |
| θ 45 | far edge −190 ☀ | 0.90 | 1.00 | 0.79 | 13.0 / 9.5 | 0.88 | +5.0 |
| θ 55, H 100 | far edge −190 ☀ | 0.88 | 1.00 | 0.75 | 16.1 / 9.6 | 0.87 | +6.6 |
| θ 65 | far edge −110 | 0.88 | 0.79 | **1.00** | 10.6 / 13.8 | 0.54 | −2.8 |
| θ 35 | far edge −190 ☀ | 0.88 | 1.00 | 0.75 | 13.7 / 8.7 | 0.87 | +6.7 |
| θ 65 | current | 0.81 | 0.64 | **1.00** | 9.1 / **14.3** | 0.58 | −3.1 |
| θ 55 | bright-field crown ☀ | 0.77 | 1.00 | 0.50 | 14.9 / 7.3 | 0.73 | +3.8 |
| θ 55 | far edge −150 | 0.77 | 0.71 | 0.83 | 8.8 / 9.6 | 0.65 | +1.5 |
| θ 55 | far edge −110 | 0.73 | 0.61 | 0.88 | 7.0 / 10.5 | 0.58 | +0.7 |
| θ 35 | current | 0.73 | 0.75 | 0.71 | 8.7 / 8.5 | 0.75 | +5.4 |
| **θ 55** | **current (today)** | **0.67** | **0.50** | **0.88** | **6.8 / 10.8** | 0.62 | 0 |
| θ 45 | current | 0.67 | 0.57 | 0.79 | 6.7 / 9.5 | 0.65 | +2.8 |
| θ 55, H 100 | current | 0.63 | 0.57 | 0.71 | 7.6 / 9.3 | 0.65 | +4.8 |
| θ 55, φ 30 | far edge −190 ☀ | 0.65 | 0.71 | 0.58 | 9.5 / 7.2 | 0.37 | −0.4 |
| θ 55, φ 30 | current | 0.46 | 0.36 | 0.58 | 4.8 / 8.0 | 0.37 | −1.7 |
| θ 55, φ 60 | current | 0.42 | 0.46 | 0.38 | 6.4 / 4.4 | 0.25 | −4.2 |

(☀ = crown glint inside the aperture. All 35 rows are in Fig. 06 and `coverage_scan.json`.)

**What it says.**

1. **The ray trace's main prediction holds.** Widening toward the crown glint is what fixes the
   crown. The far edge at −110 / −150 / −190 takes crown robust recall from 0.50 to 0.61 / 0.71 /
   1.00. The −190 aperture is the best single system of the 35, and it stays dark field (the
   glint at −214 lands outside).
   - Corner recall dips 0.88 → 0.83 (one defect). A wider window search admits more pose-noisy
     windows into the threshold.
2. **Two regimes, and the score sees both.** Pose spread and defect signal, per window,
   relative to today's mean window power (median window / median defect):

   | horn, today's aperture | intact pose σ | defect ΔP, crown | defect ΔP, corner |
   |---|---|---|---|
   | θ 55 (today) | 0.081 | 1.06 | 1.38 |
   | θ 65 (grazing) | **0.032** | 0.39 | 0.48 |
   | θ 35 (steep) | 0.37 | 5.2 | 3.1 |
   | θ 55, H 100 | 0.33 | 5.4 | 4.0 |

   - **Steep horns and close planes are bright.** They make signals 4–5× bigger, but pose
     noise grows just as much, so margins barely move. They are pose-limited and indifferent to
     receiver noise.
   - **The 65° horn is the opposite.** Signals are 3× smaller, but the intact plane is 2.5×
     quieter and has no hot windows (max σ 0.08 vs 0.25), so the corner wins: 1.00 at 14.3 dB.
     It is receiver-noise-limited, though: recall 0.81 → 0.58 at 10% noise.
   - Ray optics counts glint, not pose noise, which is why it ranked θ 65 last.
3. **Moving the horn alone does not help.**
   - θ 45 / 35 on today's surface: 0.67 / 0.73.
   - H 100: 0.63.
   - They move the glint window (§3) but also brighten the pose noise.
4. **Azimuth is the worst option.** φ 30 scores 0.46 and φ 60 scores 0.42. The plane gets
   dimmer (up to −5 dB), the rail's specular point between horn and plane (§4) keeps it from
   being dark, and the crown and corner both drop. Drop S6 unless a hardware constraint forces
   the horn beside the metasurface.
5. **Bright-field crown apertures are crown-only channels.** Crown 1.00 at up to 16 dB, but the
   corner falls to 0.29–0.50. They only make sense as a second surface paired with a
   corner-seeing one.
6. **s0 = 165–170 (upper gauge corner) is the weak column everywhere except θ 65.** That is
   consistent with §2: it is out of specular reach from above at θ ≤ 55.

### 5c. Combinations (Fig. 07)

![07](data/figures/design_2026-10-29/07_combinations.png)

Best of each hardware kind (robust recall, crown / corner, margin):

| kind | system | recall | crown / corner | margin (crown / corner) |
|---|---|---|---|---|
| 1 horn, 1 MS | θ 55, far edge −190 | 0.92 | 1.00 / 0.83 | 14.1 / 10.3 dB |
| 1 horn, 2 MS | θ 55: today's surface + a crown-glint surface (x −289…−139) | 0.96 | 1.00 / 0.92 | 15.4 / 11.6 |
| 2 horns, 1 MS | θ 35 + θ 65 time-multiplexed on far edge −110 | 0.98 | 0.96 / 1.00 | 14.5 / 13.9 |
| 2 horns, 1 MS | θ 55 + θ 65 on far edge −190 (keeps today's horn) | 0.96 | 1.00 / 0.92 | 14.9 / 12.8 |
| 2 horns, 2 MS | θ 65 on today's surface + θ 55 at H 100 on its crown glint | 1.00 | 1.00 / 1.00 | 17.2 / 14.5 |
| 2 horns, 2 MS | θ 65 on today's surface + θ 55 on a crown-glint surface at H 150 | 1.00 | 1.00 / 1.00 | 16.1 / 14.5 |

- **The pattern is the same in every top system: one dark corner channel (θ 65) plus one crown
  channel** (a wide far edge or a crown-glint surface).
  - Panel A shows how the channels split the work: the crown channel lifts s0 110–140 by
    8–12 dB, and θ 65 fills 160–170.
  - It is the ray trace's S7 idea, with the second horn swapped from 35° to 65°.
- **The step from the best single (0.92) to the best pair (1.00) is 4 defects of 52.** That is
  real but small at this defect size. Its value shows in the margins: +3.5 dB on the arc
  overall, +4 dB on the corner. Margins are what small hairlines will need.

## 6. What this does and does not show

- **No metasurface, no training.** The field-level score uses the best detector-sized window
  per sample: an oracle layout, an upper bound for 8 fixed trained windows. It ranks how much
  information reaches each aperture, not what the trained system extracts. The SLM still loses
  to no metasurface (finding 24, `ablate_surface.py` not yet run), so trained comparisons would
  not be trustworthy yet anyway.
- **Mid-size defects only** (passover_big's narrowed ranges: cracks 3–4 mm wide, 20–30 long,
  4–6 deep; dents 4–6 deep, 18–22 mm), at y0 = 0, one per s0 and variant. Small hairlines
  are where coverage will matter most and are not in this scan.
- **16 intact poses** set the threshold (~1/16 false alarms) and the pose spread. Fine for
  ranking; far too few to measure the low false-alarm rates rail inspection needs.
- **Scene deviations** (all configurations alike): 240 mm tapered segment, horn's lowest edge
  level with the plane at every θ, ψ1 only (upward input), defect-facet-only ray casting
  (proven identical: 0.0 difference on two defects; the head is convex where ψ1 looks).
- **Shells and wear were not scanned**: both are gauge-corner classes and already saturated.
- **Scalar physical optics.** No polarisation, so polarisation diversity cannot be evaluated;
  least accurate at grazing angles (θ 65°+) and for edge-diffraction-dominated signals.
- **The horn model**: rung −1 showed the textbook aperture model is off (finding pending:
  D 17.9 vs 20.1 dBi, E-plane split). Ray-optics geometry does not depend on it; the beam
  weighting and field-level numbers do, mildly.

## 7. Recommended next steps

**Candidate designs to carry forward (finalists):**

- **F1:** θ 55, far edge −190. One horn, one 265 × 75 mm surface: the minimum hardware change.
- **F2:** θ 55 + θ 65, time-multiplexed on the F1 surface. Adds one horn.
- **F3:** θ 65 on today's surface + a crown-glint surface. Two horns, two surfaces: the top
  margin.

**In order:**

1. **Fix the SLM first.** Run `python ablate_surface.py --stage prelim --long` (SETUP_LAB §7c).
   Training comparisons between geometries mean nothing while the SLM loses to no metasurface
   (finding 24).
2. **Small-defect scan.** Re-run `coverage_scan.py` on the finalists with the full defect
   ranges (hairlines, shallow dents). Margins of 10–14 dB with mid-size defects say nothing
   about 0.5 mm-wide cracks. This needs a `--defects full` option, and is a 5090 job.
3. **More intact poses (64–256)** on the finalists, so the false-alarm tail is measured rather
   than set by 16 samples. That is cheap: intact samples are the fastest to solve.
4. **Tier 2:** generate prelim-sized datasets for F1–F3 and train SLM + no-MS on each.
   - Generation cost: ×1.8 plane points for F1/F2. F2/F3 also need a two-horn sample format:
     two fields per sample, one per horn, so the pipeline would need a channel axis.
   - That pipeline change is real work. Do F1 first: it is the existing pipeline with `NX = 106`,
     `PLANE_X_CENTER = -57.5` (both provenance keys, so a fresh dataset root).
5. **Hardware questions for the user / lab:**
   - Is a 265 mm-wide surface buildable?
   - Where can a 65° horn sit? It is at D ≈ 380 mm, so 340 mm across from the rail centre,
     160 mm up. Check it against the wheel flange and the vehicle envelope.
   - Is time-multiplexing two horns (switching between them) feasible on the receiver?
   - The 65° channel needs a receiver noise floor of ≲1% of today's mean window power.
6. **Confirm the H-A75-W20 dimensions** (rung −1 open item). The field-level numbers depend
   mildly on the horn model; the geometry conclusions do not.

## 8. Reproduce

```bash
python glint_band.py                                  # today's setup, one line
python glint_band.py --theta 45 --x0 -150             # any other (also --phi, --H, --x1)
python glint_band.py --figures                        # Figs 01, 02, 04, 05 + glint_band.json (~20 s, CPU)
python coverage_scan.py                               # Tier 1 solves (laptop ~12 min/horn; resumable)
python coverage_scan.py --twins                       # pose-matched intact twins for Fig 03
python coverage_scan.py --analyze                     # Figs 03, 06, 07 + coverage_scan.json
```

Outputs: `data/generated/design_scan/` (per-sample fields `h*/`, `glint_band.json`,
`coverage_scan.json`), figures in `data/figures/design_2026-10-29/`.
