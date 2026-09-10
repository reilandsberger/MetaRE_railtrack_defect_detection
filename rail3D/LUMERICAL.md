# Validating rail3D against Ansys Lumerical FDTD

*Last updated: 2026-09-10 · λ = 5 mm (59.9585 GHz) · written for Lumerical FDTD 2025 R1*

The question this answers: `field3d.py` is **physical optics** — scalar, PEC
tangent-plane currents, single + double bounce. Crack widths are 2–5 mm =
**0.4–1λ** at 60 GHz, which is exactly where the tangent-plane approximation is
expected to break. V1–V3 only prove we solve *our own* integral correctly; V5
compares against the 2D code, which shares the assumption. A full-wave run is
the only way to measure what PO misses.

MoM (HFSS-IE / FEKO) would be the natural tool — see README finding 20 — but
with only Lumerical available, FDTD works **if the box is scoped correctly**.
That scoping is the whole design of this document, so read §1 before building
anything.

---

## 1. The one architectural decision: do NOT put the detector plane in the box

The obvious setup — box the whole scene, monitor at `H_MS` = 150 mm — is wrong
twice over.

**It does not fit.** With the horn included the box is 387 Mcells at λ/20,
~39 GB. Over a 32 GB GPU, and most of it is empty air.

**It would not be trustworthy even if it fit.** FDTD waves travel slightly slow
on the grid, and the error *accumulates with distance*. Computed for this exact
geometry (`case.json` → `fdtd.mesh_options`):

| mesh | cells | RAM | phase error to a monitor at z = 30 mm | …if propagated to H_MS = 150 mm |
|---|---|---|---|---|
| λ/10 (0.500 mm) | 26.9 M | 2.9 GB | 25° | **125°** |
| λ/15 (0.333 mm) | 90.9 M | 9.2 GB | 11° | 54° |
| λ/20 (0.250 mm) | 215 M | 23 GB | 6° | 30° |

125° of numerical phase error would swamp every physical effect we are trying
to measure.

**So: record the near field at z = 30 mm (6λ above the crown) and propagate the
rest analytically**, with rail3D's angular-spectrum propagator — which V3
already checks against the Rayleigh–Sommerfeld path to 0.13%. FDTD then only
has to do the part it is uniquely good at: the scattering off the surface,
which is precisely the part PO approximates.

This also shrinks the box to something comfortable: **26.9 Mcells at λ/10**.

### What you will see at z = 30 that you do not see at 150

The rig is **dark-field** by design: at H_MS the specular lobe lands at
x = −214 mm, far outside the ±75 mm aperture. At z = 30 mm it lands at
**x = −42.8 mm — inside the monitor**, and it will dominate the picture.

That is useful, not a problem:

- **Rungs 0–2** (setup, plate, intact rail) are *specular-dominated*. A bright,
  well-placed lobe is an easy, unambiguous check that the source angle,
  polarisation and units are right.
- **Rung 3** compares the **difference** field (defect − intact), where the
  specular lobe is common-mode and cancels — leaving the defect signature,
  which is the actual physics question.

---

## 2. Before you build anything

**Units.** Lumerical is SI internally; every length below is given in mm and
every property field in the GUI has a unit dropdown — set it to `mm`. The one
place this bites is the STL import, which must be told the file is in
millimetres (STL carries no units).

**Frequency.** `59958491600` Hz exactly (= c / 5.000 mm). Do not round to
60 GHz — that is a 0.07% wavelength shift, which over 30λ is 7° of phase.

**Material.** `PEC (Perfect Electric Conductor)` from the material database.
This matches the reflection coefficient −1 assumed in `scattered_fields`.

**Polarisation — this one is physics, not bookkeeping.** rail3D is **scalar**.
The only polarisation that maps cleanly onto it is **s-polarised (TE): E along
ŷ**, perpendicular to the x–z plane of incidence. For s-pol on PEC the
tangential-E reflection coefficient is −1, which is exactly the minus sign in
`field3d.scattered_fields`. So: set the source polarisation angle to put E along
y, and **export `E_y`**. Comparing a p-polarised run, or the field magnitude,
against our scalar field is not a meaningful test.

**Get the case bundles** (run from `rail3D/`):

```bash
python compare_wavefronts.py --sample plate  --source plane --plane-z 30 --export-only --export-case data/generated/fdtd_plate
```
```bash
python compare_wavefronts.py --sample intact --source plane --plane-z 30 --export-closed --export-only --export-case data/generated/fdtd_intact
```
```bash
python compare_wavefronts.py --sample crack  --source plane --plane-z 30 --export-closed --export-only --export-case data/generated/fdtd_crack
```

Each writes `rail_surface.stl` (import this), `rail_surface.obj`, a `README.md`,
and `case.json` — whose `fdtd` block holds every number in §3, derived from the
geometry actually exported, so it cannot drift from this document.

---

## 3. Building the simulation (UI walkthrough)

Menu labels are from FDTD 2025 R1. If a label differs slightly, the action is
the thing to follow. Everything is on the **Objects Tree** (left) and the
**property editor** you get by double-clicking an object.

### 3.1 Simulation region

**Simulation** toolbar dropdown → **Region**. Double-click the new `FDTD` object.

*Geometry* tab — switch each unit dropdown to `mm`:

| field | value |
|---|---|
| x min / x max | **−88 / 88** mm |
| y min / y max | **−73 / 73** mm |
| z min / z max | **−92.9 / 38** mm |
| dimension | 3D |

*General* tab: **simulation time** `10e-9` s (10 ns). This is an upper bound —
auto-shutoff will end it earlier. Leave **auto shutoff min** at `1e-5`.

*Mesh settings* tab: **mesh type** `auto non-uniform`, **mesh accuracy** `2` for
the first runs. Tick **"conformal variant 1"** under mesh refinement if
available — it substantially reduces staircasing on curved PEC and is the single
best accuracy-per-cell setting for this problem.

> To force an exact λ/10 grid instead of the accuracy slider, set mesh type to
> `uniform` and `dx = dy = dz = 0.5` mm. That reproduces the table in §1
> exactly. The auto mesher is usually better per cell; use uniform when you want
> the cell count to be predictable.

*Boundary conditions* tab: **PML on all six** faces. Leave layers at 8,
profile `stabilized`.

### 3.2 Source

**Sources** dropdown → **TFSF** (total-field scattered-field).

TFSF injects the plane wave *inside* its box; outside it, only the **scattered**
field exists. That is exactly what `psi1` is, so the monitor reads a directly
comparable quantity with no subtraction.

*Geometry* tab:

| field | value |
|---|---|
| x min / x max | **−44 / 43.9** mm |
| y min / y max | **−65 / 65** mm |
| z min / z max | **−84.9 / 10** mm |

The box must fully enclose the rail (which spans x ±39, y ±60, z −80…0) with
about 1λ of clearance, and its top must be **below the monitor** so the monitor
sits in the scattered-field region.

*General* tab:

| field | value |
|---|---|
| injection axis | z |
| direction | backward |
| angle theta | **55** ° |
| angle phi | **180** ° |
| polarization angle | **90** ° (puts E along y — see §2) |
| wavelength / frequency | set **frequency**, `59958491600` Hz, single point |

> **Verify φ rather than trusting it.** Conventions for which way θ/φ tilt the
> k-vector differ between tools. Run rung 0 (§4) and look at where the specular
> lobe lands: it must be at **negative x**. If it comes out at positive x,
> change `angle phi` to `0` and re-run. This is the cheapest possible check and
> it is why rung 0 exists.

### 3.3 Monitor

**Monitors** dropdown → **Frequency-domain field and power**.

*Geometry* tab:

| field | value |
|---|---|
| monitor type | 2D Z-normal |
| x min / x max | **−80 / 80** mm |
| y min / y max | **−42.5 / 42.5** mm |
| z | **30** mm |

The monitor is deliberately wider than our comparison plane (±75 / ±37.5 mm) so
resampling never has to extrapolate. If you shrink it, `--external` will tell
you the coverage dropped below 100% rather than silently scoring zeros as
disagreement.

*General* tab: **override global monitor settings** ✓, **frequency points** `1`.
Under *Data to record*, `E` is enough (untick H to save disk).

### 3.4 Geometry

**Structures** dropdown → **Import**. In the import dialog choose
`rail_surface.stl` from the case bundle, and **set file units to millimetres**.
Leave the x/y/z offsets at 0 — the STL is already in our coordinates, crown at
z = 0.

Then in the object's *Material* tab select **PEC (Perfect Electric Conductor)**.

Check it landed right: in the CAD view the crown should sit at z = 0 with the
rail hanging below, and the whole body inside the TFSF box.

### 3.5 Defect refinement (rung 3 only)

A 2 mm crack at a 0.5 mm global mesh is 4 cells across — too coarse to trust.
Refine only where it matters: **Simulation** dropdown → **Mesh** (a mesh
override region).

For the exported `crack` sample, `case.json` → `fdtd.defect_refinement` gives:

| field | value |
|---|---|
| x min / x max | **−17.5 / 41.2** mm |
| y min / y max | **−18.8 / 4.4** mm |
| z min / z max | **−14.4 / 9.9** mm |
| dx = dy = dz | **0.25** mm (λ/20) |

That adds only ~1.9 Mcells to a 26.9 Mcell run. **Use the identical override
box in the intact run too**, even though there is no defect there — matched
meshes are what make the difference field cancel numerical error.

---

## 4. The ladder — run these in order

Each rung isolates one unknown. Skipping to rung 3 means a disagreement has four
possible causes and you cannot tell which.

### Rung 0 — setup sanity, no comparison to our code

Delete the imported rail; add a **Rectangle** structure, PEC, spanning
x ±60, y ±60, z from −5 to 0 mm. Run.

Check three things, all from `Visualize → E` on the monitor:

1. The specular lobe is at **negative x**, near x ≈ −43 mm. If not, fix
   `angle phi` (§3.2).
2. |E| is smooth, with no bright fringe hugging the box walls — a bright edge
   means the PML is too close or the structure touches it.
3. Auto-shutoff fired (check the log) rather than the run hitting 10 ns. If it
   hit the limit, something is resonating; check for PEC touching PML.

Nothing here involves rail3D. If rung 0 misbehaves, no comparison downstream is
worth running.

### Rung 1 — flat plate, first true code-to-code comparison

Import `fdtd_plate/rail_surface.stl` (a 75 × 75 mm PEC square in the z = 0
plane), or keep the rectangle from rung 0 but resize it to x ±37.5, y ±37.5.

Both solvers model this exactly, so **a disagreement here is a setup error, not
physics** — units, angle, polarisation, phase reference, or the monitor grid.

Do not proceed until `complex_corr` is high (≳0.95). Getting rung 1 to agree is
the bulk of the work; rungs 2 and 3 are then mostly re-runs.

> The plate is a *finite* sheet and both solvers see the same finite sheet, so
> its edge diffraction is part of the agreed problem — unlike the rail, where a
> truncation the real object does not have would be a confound (§6).

### Rung 2 — intact rail

Import `fdtd_intact/rail_surface.stl`. Same box, same mesh, same source.

Now the curved PEC surface is in play. This tests PO currents on curvature —
still specular-dominated at z = 30, so expect good agreement. A drop from
rung 1 localises the problem to the surface model.

### Rung 3 — cracked rail, and the difference field

Import `fdtd_crack/rail_surface.stl`, add the mesh override from §3.5, and
**re-run rung 2 with that same override** so the two runs are numerically
matched.

This is the measurement. Compare:

- **absolute fields** — expect them to agree about as well as rung 2, since both
  are specular-dominated
- **the difference field** `E_crack − E_intact` against
  `psi1_crack − psi1_intact` — **this is the real result.** It isolates the
  defect signature, cancels the specular lobe and most of the common-mode
  numerical error, and is the quantity the detector barcodes actually respond to.

A large disagreement *here* while rungs 1–2 agree is exactly the finding worth
having: it is PO failing on a sub-wavelength feature, which is what we set out
to measure. Report it, do not tune it away.

---

## 5. Getting the data back out

In Lumerical's **Script Prompt** (bottom of the window) or a `.lsf` file:

```
mon = "monitor";                              # your monitor's name
E   = getresult(mon, "E");
Ey  = pinch(E.E(:,:,:,:,2));                  # component 2 = y  -> (nx, ny)
x   = E.x;
y   = E.y;
matlabsave("crack_z30.mat", Ey, x, y);
```

`pinch` drops the singleton z/frequency axes. Component index 2 is `E_y` — the
s-polarised component from §2. Save `x` and `y`; they are what lets the
comparison resample the monitor grid onto ours.

Then, back in `rail3D/`:

```bash
python compare_wavefronts.py --sample crack --source plane --plane-z 30 --external crack_z30.mat
```

It reports, before any metric:

```
crack_z30.mat: resampled (321, 171) (m) -> (60,30), coverage 100.0%
aligned EXT crack_z30: conjugated (corr as-is 0.08 / conj 0.99), gain 1.2e-3, phase -14.0 deg
```

- **resampled / coverage** — the monitor grid was interpolated onto ours;
  coverage below 100% means the monitor was too small and the shortfall is being
  scored as disagreement.
- **conjugated / as-is** — rail3D uses `exp(−iωt)`, so outgoing waves carry
  `exp(+ik₀R)`. If Lumerical's convention differs, its fields arrive conjugated.
  The tool *detects and reports* which matched rather than silently fixing it,
  so you know what you are looking at. Conjugating the wrong one turns a perfect
  match into an apparent total failure.
- **gain / phase** — one complex scale fitted over the whole plane, absorbing
  source normalisation and V/m-versus-scalar units. Because of this,
  **`complex_corr` is the number to quote**, not `rel_l2`: the residual after
  the fit is `sqrt(1 − complex_corr²)` by construction.

If it prints `! both conventions score alike -- ambiguous`, the two fields are
essentially uncorrelated. Do not read the alignment as confirmation of anything;
go back to rung 1.

---

## 6. Things that will look like physics and are not

| symptom | cause |
|---|---|
| Total disagreement, `corr` ≈ 0 either way | wrong component exported (magnitude, or E_x/E_z), or the monitor is in the total-field region — it must be **outside** the TFSF box |
| Good amplitude, scrambled phase | frequency rounded to 60 GHz; or the mesh is too coarse for the path (see §1) |
| Specular lobe on the wrong side | `angle phi` — flip 180° ↔ 0° |
| Agreement falls off toward the plane edges | monitor too small; coverage < 100% |
| Rungs 1–2 fine, rung 3 disagrees only near the defect | **this is the actual result**, not a bug |
| Disagreement that grows with segment length | cut-end edge diffraction. Our PO model has *no* edge diffraction; a full-wave solver has plenty. `case.json` → `truncation` measures how brightly lit the cut ends are — 0.08× mid-span at the production 120 mm segment, but **0.80× at 30 mm**, so do not shorten the rail to save cells. Compare the difference field, where it cancels. |

Polarisation coupling is the one thing our scalar model cannot represent at all.
If s-polarised runs agree and the physics you care about involves depolarisation
in a crack, that is a limitation of the model, not something the comparison can
fix.

---

## 7. Cost summary

| run | mesh | cells | RAM | notes |
|---|---|---|---|---|
| rung 0–1 (plate) | λ/10 | ~27 M | ~3 GB | minutes |
| rung 2 (intact) | λ/10 | ~27 M | ~3 GB | |
| rung 3 (crack + override) | λ/10 + λ/20 local | ~29 M | ~3 GB | run intact again with the same override |
| production comparison | λ/15 | ~91 M | ~9 GB | 11° phase error to the monitor |

All well inside a 5090's 32 GB. Lumerical's GPU solver supports a subset of
features — if it refuses the setup, the CPU solver at this size is still
tractable; the constraint here was never total cells, it was the 30λ of empty
air we removed in §1.

---

## 8. What this cannot tell you

- It validates the **surface scattering**, not the propagation to `H_MS` — that
  is V3's job, and V3 already passes at 0.13%.
- It is a **single-frequency, single-polarisation** check. rail3D is scalar and
  monochromatic; that is the model, and this measures the model's error, not its
  scope.
- It says nothing about the metasurface or detector training. Those sit
  downstream of the plane field and are governed by V5–V8.
