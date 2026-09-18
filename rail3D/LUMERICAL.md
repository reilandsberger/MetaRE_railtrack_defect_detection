# Validating rail3D against Ansys Lumerical FDTD — horn Import source

*Last updated: 2026-09-18 · λ = 5 mm (59.9585 GHz) · written for Lumerical FDTD
2025 R1 (modern UI). Rewritten for the horn Import-source design: no TFSF.*

**What this answers.** `field3d.py` is **physical optics**: scalar, PEC
tangent-plane currents, single + double bounce. Crack lines are 1.5–3 mm wide =
**0.3–0.6λ** at 60 GHz, exactly where the tangent-plane approximation is expected
to break. V1–V3 prove we solve *our own* integral correctly; only a full-wave
solver can measure what PO misses. Here that solver is Lumerical FDTD, driven by
**rail3D's own horn wavefront** as an Import source.

Every Lumerical-specific statement below is cited to Ansys's documentation
([Sources](#sources)). Where a step is an inference rather than documented, it
says so, and rung 0 exists to check it.

---

## 0. Before you start — three checks

1. **Version.** File tab → *About*. Import sources run on the **GPU only from
   2025 R1.1**: *"Lumerical FDTD can now run GPU simulations with single-frequency
   import sources … the import source data must be single-frequency"* [R1.1 notes].
   On the base 2025 R1 build, run on CPU, or install the latest R1.x patch.
2. **GPU.** Lumerical's GPU solver needs CUDA 12, driver ≥ 527.41 on Windows and
   compute capability ≥ 5.0 [GPU]. No 2025 R1.x note mentions the RTX 50-series
   (Blackwell) explicitly, so **let the software decide**: FDTD tab → *Check* →
   **GPU** runs the GPU memory/compatibility check [GPU, UI]. If it refuses, use
   CPU. The setup below is identical on either.
3. **Why no TFSF.** The horn is the source (your requirement), and TFSF could not
   run on your GPU anyway: *"FDTD GPU does not support TFSF sources and an error is
   shown if a TFSF source is present"* [GPU].

**Generate the case bundles** (from `rail3D/`, on either machine — ~15 s each):

```bash
python compare_wavefronts.py --sample intact --source horn --plane-z 30 --export-closed --export-only --export-case data/generated/fdtd_intact
```
```bash
python compare_wavefronts.py --sample crack --source horn --plane-z 30 --export-closed --export-only --export-case data/generated/fdtd_crack
```

Each folder holds:

| file | what it is |
|---|---|
| `rail_surface.stl` | the rail, a **watertight closed** body in **mm** (crown at z = 0) |
| `horn_source.mat` | the horn wavefront as plain arrays (SI) — input to the script below |
| `load_horn_source.lsf` | builds the Lumerical dataset and the Import source |
| `horn_source.json` | source plane, window, and the self-check numbers |
| `case.json` → `fdtd` | **every number in §4**, derived from the exported geometry |

---

## 1. The design, and why each piece is where it is

```
 z (mm)
  38 ┬─────────────────────────── FDTD region top (PML above)
  30 ┤ ═══════ mon_z30 ═══════     monitor: records UP-going field only
  15 ┤   ▓▓▓▓▓ horn Import source ▓▓▓▓▓▓▓▓   injects DOWN at 55°  ↙
   0 ┤      ╭──crown──╮                     rail (PEC STL)
 -80 ┤      └─ web ───┘
 -93 ┴─────────────────────────── FDTD region bottom
```

**Near field at z = 30 mm, not the metasurface plane.** FDTD waves travel
slightly slow on the grid, and the error grows with distance. For this geometry
(`case.json` → `fdtd.mesh_options`):

| uniform mesh | phase error rail → z = 30 | …if FDTD itself went to H_MS = 150 |
|---|---|---|
| λ/10 (0.500 mm) | 25° | **125°** |
| λ/15 (0.333 mm) | 11° | 54° |
| λ/20 (0.250 mm) | 6° | 30° |

So FDTD records at z = 30 and rail3D's angular-spectrum propagator (checked by V3
to 0.13%) carries the field up to the metasurface.

**The horn enters as an Import source, not as geometry.** The horn's phase centre
is 140 mm out at 55° — (x, z) = (115, 80) mm. Boxing it with the rail would be
~390 Mcells. Lumerical's Import source instead *"allows the user to specify a
custom spatial field profile for the source injection plane … from an analytic
formula, … or from other simulation tools"* [Import]. `horn_source.py` computes
rail3D's own horn field on a plane and writes it in Lumerical's format.

**The source plane is at z = 15 mm, between the rail and the monitor.** It
injects **down**. The monitor sits on the far side of it, so it sees only what
comes **up**: the reflected and scattered field, every bounce included. No TFSF
and no subtraction run are needed to separate incident from scattered light.

**The source window covers only the rays that can reach the rail.** The horn is
a 3.4λ × 2.7λ aperture, so its beam is wide: at crown height the −20 dB contour
spans y = ±128 mm. `horn_source.py` projects every lit rail facet toward the phase
centre onto z = 15, adds 6λ, and applies a 2λ raised-cosine edge taper. The
result is x −28…102.5, y ±78.25 mm. Checked in free space against rail3D's full
horn on the rail's footprint (`horn_source.json`):

| plane | complex correlation with the full horn |
|---|---|
| crown, z = 0 | 0.9989 |
| z = −40 | 1.0000 |
| z = −80 (web) | 0.9991 |

The window carries 72% of the horn's power. The other 28% never reaches the rail.

**E and H are both supplied.** Lumerical: without H, the source *"makes certain
assumptions … These assumptions hold true for narrow sources such as Gaussian and
plane wave sources, but may lead to significant errors for other sources … When
defining complex beams, it is best to specify both E and H"* [Import].
- **E_y** is rail3D's scalar field, unchanged: s-polarised, E along the rail.
- **Each plane-wave component** carries ŷ projected transverse to its own k.
- **H = (k × E)/(ωμ₀)**.

Self-checks: 99.99% of the flux goes down, |E|/|H| = 376.4 Ω against Z₀ = 376.7 Ω,
and cross-polarisation is Ex/Ey 0.17, Ez/Ey 0.14 rms.

**Time convention — no conjugation expected.** Lumerical's documented transform is
*"P(ω) = ∫ e^{iωt} P(t) dt"*, with J = −iωP [Force]. That is the exp(−iωt)
convention, the same as rail3D, so exports align **as-is**. (HFSS and FEKO use
exp(+jωt) and arrive conjugated. An earlier version of this document wrongly
said Lumerical did too.)

**The monitor is y ±70 mm, not ±42.5.** Its window is what gets carried to the
metasurface plane, and cut-end light from the 120 mm rail crosses z = 30 outside
±42.5 mm (README finding 27).

---

## 2. Finding your way around the 2025 R1 interface

2025 R1 introduced a tabbed toolstrip [UI]. What you see in the Layout window:

| area | what it is for |
|---|---|
| **Objects Tree** (left) | every object; **double-click one to open its property editor** (tabs: General / Geometry / Mesh settings / …) |
| **XY / XZ / YZ / Perspective views** | the CAD panes; selected objects show red handles |
| **Result View** (left, below) | results of the selected object after a run |
| **Script Prompt / Script Workspace** (bottom) | type script commands, see variables. Show/hide under **View → Show** [UI] |
| **Script File Editor** | open and run `.lsf` files (View → Show → Script File Editor) [UI] |

The toolstrip tabs you will use [UI]:

| tab → group → button | what it does |
|---|---|
| **File → Units → Length** | the default length unit. **Set to mm first** (§4 step 1) |
| **File → Program → About / Working Directory** | software version; the folder scripts read and write |
| **Design → Solvers → FDTD** | *"Add an FDTD solver simulation object"* — **the FDTD tab only appears after this** |
| **Design → Import → STL** | import the rail |
| **Design → Structures → Rectangle** | the rung-1 plate |
| **Design → Materials → Database** | material list (PEC lives here) |
| **FDTD → Sources → Import** | *"Import custom source"* — the horn |
| **FDTD → Monitors → Frequency-Domain** | *"Add frequency-domain field profile monitor"* |
| **FDTD → Misc. → Mesh** | *"Add mesh control region"* — the crack override |
| **FDTD → Settings → Global Source / Global Monitor** | frequency and frequency points for all sources / monitors |
| **FDTD → Check → CPU / GPU** | memory + compatibility check before running |
| **FDTD → Run Simulation** | CPU/GPU toggle, resource drop-down, **Run** |

Your blank-project screenshot shows only File / View / Design for exactly this
reason: step 2 below adds the FDTD solver, and the FDTD tab appears.

---

## 3. Preparing each piece (outside Lumerical)

**3.1 The rail — `rail_surface.stl`.** Written by the export command in §0 from
the same mesh rail3D scatters from, capped into a closed body (`--export-closed`,
`case.json` → `mesh.watertight: true`). FDTD needs a closed solid; an open shell
has no inside to fill with PEC. The file is in **millimetres**, and **STL carries
no unit**: *"a 30 (mm) … cube … created in a CAD with a length unit set to mm will
be recognized as 30 (um) … by default"* [STL]. That is why §4 step 1 exists.

**3.2 The horn wavefront — `horn_source.mat` + `load_horn_source.lsf`.** Written
by the same export (or standalone: `python horn_source.py --out <dir>`). The
`.mat` uses the exact layout of Lumerical's own example `usr_custom_source.lsf`
[Equation]:
- x and y as column vectors in metres, z as a scalar
- Ex…Hz as (nx, ny) complex matrices, x varying first
- sampled at λ/20 (0.25 mm), so interpolating the 55° phase ramp onto the FDTD
  mesh costs < 1% amplitude

The `.lsf` turns those arrays into a Lumerical dataset with the documented calls,
`rectilineardataset("EM fields", x, y, z)` and `addattribute("E"…)` /
`addattribute("H"…)` [Import], then loads it with `importdataset` [Equation]. The
GUI's *Import Source* button needs a `.mat` holding a Lumerical **dataset**, not
plain arrays [Import]. That is why the script, not the raw `.mat`, comes first.

**3.3 The plate (rung 1).** Built natively in Lumerical, not from STL. rail3D's
plate is a zero-thickness sheet, which has no volume to import.

---

## 4. Building the simulation — step by step

Each step names the ribbon path, then the property-editor fields. All lengths in
mm once step 1 is done. The numbers are rung 2 (intact rail); rungs 0, 1 and 3
reuse this file with the changes listed in §5.

**Step 1 — units.** File → Units → **Length: mm** (also Frequency: GHz if you
like). *"It is recommended that you choose the right length unit in the layout
editor prior to importing STL files"* [STL].

**Step 2 — the FDTD region.** Design → Solvers → **FDTD**, then double-click
`FDTD` in the Objects Tree.

| tab | field | value |
|---|---|---|
| General | dimension | 3D |
| General | simulation time | 10 ns — an upper bound; *"the actual simulation may be shorter if the autoshutoff criteria are satisfied"* [FDTD] |
| General | auto shutoff min | leave 1e-5 (default) [FDTD] |
| Geometry | x min / max | **−88 / 110.5** |
| Geometry | y min / max | **−86.2 / 86.2** |
| Geometry | z min / max | **−92.9 / 38** |
| Mesh settings | mesh type | **uniform**, dx = dy = dz = **0.5** (λ/10) for rungs 0–2 |
| Mesh settings | mesh refinement | **conformal variant 1** — variant 0 (the default) *"is not applied to interfaces involving metals or PEC"*; variant 1 *"applies the conformal mesh algorithm to the PEC"* [Conformal] |
| Boundary conditions | all six | **PML**, profile stabilized; raise layers if rung 0 shows leakage |

A uniform mesh keeps the cell count predictable (35.9 M at λ/10). It also means
adding or removing the rail does not re-mesh the air, so runs stay directly
comparable. (The auto non-uniform mesher is Lumerical's default and is fine for
exploring; for the comparison runs, use uniform.)

**Step 3 — the rail.** Design → Import → **STL** → `rail_surface.stl`. Then
double-click the new object:

| tab | field | value |
|---|---|---|
| Material | material | **PEC (Perfect Electrical Conductor)** — the exact database name [Materials] |

Check it landed: Geometry tab ≈ x −39…39, y −60…60, z −80…0 (crown at z = 0). If
it reads 1000× too small, step 1 was skipped. Script alternative, independent of
the GUI unit: `stlimport("rail_surface.stl", 1e-3);` — the scaling factor
defaults to 1e-6, micrometres [stlimport].

**Step 4 — the horn.** Set the working directory to the bundle folder (File →
Program → Working Directory). Open `load_horn_source.lsf` in the Script File
Editor and **Run**. It:

1. loads `horn_source.mat` (`matlabload` [matlabload]),
2. builds the EM dataset (E **and** H), and saves `horn_EM_dataset.mat`,
3. adds an Import source named `horn_source`, `importdataset(EM)`, sets
   **Direction = Backward** (−z, toward the rail) and a **single wavelength**
   (`"wavelength span", 0`, per Lumerical's example [Equation]).

GUI route instead of step 3: FDTD → Sources → **Import** → double-click it →
General tab → **Import Source** button → `horn_EM_dataset.mat` (written by the
script with `CREATE_SOURCE = 0`). Then set **Direction: Backward**. *Injection
axis* is set automatically from the data (z) [Import].

Verify: the source's Geometry tab should show x −28…102.5, y −78.25…78.25,
z = 15 (the span *"will be automatically set based on the imported field data"*
[Import]). In the XZ view it is a grey bar at z = 15 with the arrow pointing
**down and to −x**.

> *Inference, checked by rung 0:* the property name `"direction"` in the script
> follows Lumerical's source conventions [Import] but is not spelled out for
> Import sources. If that line errors, set Direction in the GUI.

**Step 5 — frequency, once for everything.** FDTD → Settings → **Global
Source**: frequency **59.9584916 GHz** = 59958491600 Hz, span 0. Do not round to
60 GHz, which is 7° of phase over 30λ. Then FDTD → Settings → **Global Monitor**:
frequency points **1**, use source limits.

**Step 6 — the monitor.** FDTD → Monitors → **Frequency-Domain**; rename it
`mon_z30`.

| tab | field | value |
|---|---|---|
| Geometry | monitor type | 2D Z-normal |
| Geometry | x min / max | **−80 / 80** |
| Geometry | y min / max | **−70 / 70** |
| Geometry | z | **30** |
| Data to record | E | ✓ (H can be unticked) |

**Step 7 — check, then run.** FDTD → Check → **GPU** (or CPU). It reports memory
and flags unsupported objects [GPU]. Then Run Simulation: toggle **GPU**, pick
*Local Host*, **Run**.

**Step 8 — export.** In the Script Prompt:

```
E = getresult("mon_z30", "E");
Ey = pinch(E.Ey);
x = E.x;  y = E.y;
matlabsave("intact_z30.mat", Ey, x, y);
```

`getresult` returns a dataset *"E vs x, y, z, lambda/f"* [getresult], and
`E.Ey` is its y component [Datasets]. `pinch` drops the singleton z and
frequency axes. E_y is the s-polarised component rail3D models. Then, in
`rail3D/`:

```bash
python fdtd_agreement.py --sample intact --external intact_z30.mat
```

---

## 5. The ladder — run these in order

Each rung isolates one unknown. Skipping ahead means a disagreement has several
possible causes and you cannot tell which.

**Rung 0 — empty box: is the horn injected correctly?** Same file as §4, but with
the rail **disabled** (right-click → disable). Add a second Frequency-Domain
monitor `mon_z0`: 2D Z-normal, x −60…60, y −70…70, **z = 0**. Run, then export
**both** monitors (`empty_z0.mat` from `mon_z0`, `empty_z30.mat` from `mon_z30`):

```bash
python fdtd_agreement.py --injection empty_z0.mat --leak empty_z30.mat
```

It passes when:
- FDTD's field at crown height matches rail3D's horn field over the rail
  footprint at complex correlation ≥ 0.98, **as-is**
- the beam centroid is within 5 mm of rail3D's
- `mon_z30` (with nothing to reflect off) reads < 1% of the injected peak

The last number is the noise floor under every later rung. This is Lumerical's own
advice for TFSF, applied here: *"temporarily disable the particle or defect, run
your simulation … determine the noise floor"* [TFSF tips]. It is also what
Lumerical recommends for custom sources: *"compare the field profile recorded by
a monitor just in front of the source with the original"* [Equation].

**Rung 1 — flat plate.** Disable the rail. Design → Structures → **Rectangle**:
x −37.5…37.5, y −37.5…37.5, **z −1…0**, PEC. Delete `mon_z0`.

```bash
python fdtd_agreement.py --sample plate --external plate_z30.mat
```

Both solvers model a plate essentially exactly, so a disagreement here is a
**setup** error — units, source, monitor — not physics. Do not proceed until
agreement is ≥ 98%. (FDTD's plate is 1 mm thick, rail3D's has zero thickness.
Only the edges differ.)

**Rung 2 — intact rail.** Delete the plate, re-enable the rail. This is §4
exactly. It tests PO currents on a curved surface, under the horn.

**Rung 3 — cracked rail, and the difference field.** Replace the STL with
`data/generated/fdtd_crack/rail_surface.stl` (same PEC material). Add FDTD →
Misc. → **Mesh**:

| field | value |
|---|---|
| x min / max | **−17.5 / 41.2** |
| y min / max | **−23.1 / 7.5** |
| z min / max | **−16.1 / 9.9** |
| dx = dy = dz | **0.25** (λ/20) |

Crack lines are 1.5–3 mm wide, i.e. 3–6 cells at λ/10: too coarse to trust. Then
**re-run rung 2 with the identical override**, so the two runs share one mesh and
their common-mode numerical error cancels in the difference.

- **Absolute fields:** `fdtd_agreement.py --sample crack --external crack_z30.mat`.
  Expect roughly rung-2 agreement.
- **The difference field** (crack − intact) against rail3D's is **the real
  result**. It is where the crack signature lives, and README finding 27 /
  READING_RESULTS §9b explain why the absolute field alone is not enough for
  cracks. The difference-field scorer is not built yet. It is the next addition
  to `fdtd_agreement.py`.

A disagreement **here**, while rungs 0–2 agree, is exactly the finding worth
having: PO failing on a sub-wavelength feature. Report it; do not tune it away.

---

## 6. Things that look like physics and are not

| symptom | cause |
|---|---|
| Rail is tiny, or huge | STL has no units; set File → Units → Length = mm **before** importing, or `stlimport(..., 1e-3)` [STL, stlimport] |
| Script error on the material | name is **PEC (Perfect Electrical Conductor)**, not "Electric" [Materials] |
| Rung 0 matches only when conjugated | check Direction = **Backward** first. If it is, rebuild with `python horn_source.py --out DIR --conjugate`: Lumerical conjugates *beam* profiles internally for backward injection [Rotations], and this undoes that if it also applies to imports |
| Rung 0 beam lands in the wrong place | Direction wrong, or the source was imported without H |
| `mon_z30` sees a strong field in the empty box | monitor below the source plane, source injecting both ways (no H), or PML reflection — raise PML layers |
| GPU run refused | a TFSF source is present; build < 2025 R1.1 with an Import source; or Check → GPU names the object [GPU, R1.1] |
| Good amplitude, scrambled phase | frequency rounded to 60 GHz, or the mesh too coarse for the path (§1) |
| Coverage < 100% reported by the scorer | the monitor is smaller than §4 step 6 |
| Disagreement concentrated at the rail ends | cut-end edge diffraction, which PO does not model. Compare the difference field, where it cancels |

Polarisation coupling is the one thing the scalar model cannot represent. If
s-polarised runs agree and the crack depolarises, that is a limit of the model,
not something this comparison can fix.

---

## 7. Cost

From `case.json` → `fdtd.mesh_options` (uniform mesh; RAM ≈ 100 B/cell):

| run | mesh | cells | RAM |
|---|---|---|---|
| rungs 0–2 | λ/10 | 35.9 M | 3.6 GB |
| rung 3 (+ λ/20 override) | λ/10 + local λ/20 | ~38.5 M | ~3.9 GB |
| production check | λ/15 | 121 M | 12.1 GB |
| convergence ceiling | λ/20 | 287 M | 28.7 GB (near a 32 GB GPU's limit) |

---

## 8. What this cannot tell you

- It validates the **surface scattering under the horn**, not the propagation to
  H_MS. That is V3's job, which already passes at 0.13%.
- It is **single-frequency and single-polarisation**. That is the model, and this
  measures the model's error, not its scope.
- It says nothing about the metasurface or detector training, which sit downstream
  and are governed by V5–V8.
- The horn itself is rail3D's aperture model in both codes. FDTD checks what happens
  **after** the horn's field leaves the source plane, not whether the aperture
  model matches a physical horn.

---

## Sources

- [UI] Ansys Lumerical FDTD Modern User Interface — https://optics.ansys.com/hc/en-us/articles/36952912384403-Ansys-Lumerical-FDTD-Modern-User-Interface
- [Import] Import source – Simulation object — https://optics.ansys.com/hc/en-us/articles/360034383014-Import-source-Simulation-object
- [Equation] Using an equation to define the spatial field profile of a source in FDTD (and its `usr_custom_source.lsf`) — https://optics.ansys.com/hc/en-us/articles/360034383054-Using-an-equation-to-define-the-spatial-field-profile-of-a-source-in-FDTD
- [GPU] Getting started with running FDTD on GPU — https://optics.ansys.com/hc/en-us/articles/17518942465811-Getting-started-with-running-FDTD-on-GPU
- [R1.1] 2025 R1.1 Release Notes — https://optics.ansys.com/hc/en-us/articles/38462847569043-2025-R1-1-Release-Notes
- [Force] Methodology for optical force calculations (sign convention) — https://optics.ansys.com/hc/en-us/articles/360042214594-Methodology-for-optical-force-calculations
- [Units] Units and normalization conventions in Lumerical solvers — https://optics.ansys.com/hc/en-us/articles/360034397034-Units-and-normalization-conventions-in-Lumerical-solvers
- [STL] STL import – Simulation object — https://optics.ansys.com/hc/en-us/articles/360034901953-STL-import-Simulation-object
- [stlimport] stlimport – Script command — https://optics.ansys.com/hc/en-us/articles/360034924733-stlimport-Script-command
- [FDTD] FDTD solver – Simulation Object — https://optics.ansys.com/hc/en-us/articles/360034382534-FDTD-solver-Simulation-Object
- [Conformal] Selecting the best mesh refinement option in the FDTD simulation object — https://optics.ansys.com/hc/en-us/articles/360034382614-Selecting-the-best-mesh-refinement-option-in-the-FDTD-simulation-object
- [getresult] getresult – Script command — https://optics.ansys.com/hc/en-us/articles/360034409854-getresult-Script-command
- [Datasets] Introduction to Lumerical datasets — https://optics.ansys.com/hc/en-us/articles/360034409554-Introduction-to-Lumerical-datasets
- [matlabload] matlabload – Script command — https://optics.ansys.com/hc/en-us/articles/360034408034-matlabload-Script-command
- [Materials] Material database in the Lumerical FDTD and MODE products — https://optics.ansys.com/hc/en-us/articles/360034394614-Material-database-in-the-Lumerical-FDTD-and-MODE-products
- [TFSF tips] Tips and best practices when using the FDTD TFSF source — https://optics.ansys.com/hc/en-us/articles/360034382934-Tips-and-best-practices-when-using-the-FDTD-TFSF-source
- [Rotations] Source Rotations in 3D FDTD — https://optics.ansys.com/hc/en-us/articles/1500002383802-Source-Rotations-in-3D-FDTD
