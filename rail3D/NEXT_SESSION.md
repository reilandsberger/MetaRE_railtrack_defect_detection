# rail3D — session handoff, 2026-10-02

*Last updated: 2026-10-02 (horn moved out to 278.5 mm, upward-only metasurface
input, finding 31; Lumerical file I/O 2026-09-29, finding 30). Rewritten for a
cold start on a new Claude account: no chat history or auto-memory is assumed.
Everything needed is in this repo, plus the external paths listed at the end.*

## Read in this order

1. **This file** — state, next commands, open items, landmines.
2. `CLAUDE.md` (repo root) — ground rules and the dated state sections.
3. `rail3D/README.md` — §4 conventions, §6 hard-won findings (**31 = horn
   placement + upward-only input, newest**; 30 = Lumerical file I/O; 29 = the
   horn; 24–26 = the prelim result; 27–28 = Lumerical), §7 current state, §8
   limitations.
4. As needed: `rail3D/SETUP_LAB.md` (lab runbook — **§7e is the next run**),
   `rail3D/LUMERICAL.md` (FDTD, rungs −1…3), `rail3D/READING_RESULTS.md` (what
   every output field means, and how each has been misread).

Every doc has a `Last updated` line under its title — bump it in the same
commit as any behaviour change it describes.

## Project in one paragraph

Trainable metasurface + detector "barcode" system for rail-head defect
detection. A 60 GHz (λ = 5 mm) horn at 55° lights the rail; physical-optics
scattering (exact RS-I surface integral, ported from the experimentally
validated Face3D code) reaches a 60×30 plane 150 mm up (dark-field: the
specular lobe misses the aperture); a trainable phase mask (SLM) and 130→8
trainable detector windows produce 8 numbers; a classifier gives
defect / no-defect and crack / dent / wear / shell. All code is in `rail3D/` on
branch **`3D_railhead_upgrade`** (the λ=12 mm 2D code at the repo root is
legacy reference — do not modify).

## Where things stand (2026-10-02)

- **Horn moved (2026-10-02, finding 31):** slid out along the 55° axis until
  its lowest edge is level with the metasurface (z = 150): `DIST_ANT` = 278.51
  mm, derived from `HORN_LOWER_EDGE_Z = H_MS` (was 140). The metasurface input
  is now **upward-only** (`field3d.upward_only`), so psi0 = psi2 = 0 there and
  "tot" = psi1. `term_budget.py` (laptop, no shadowing) says psi2 was already
  −42 dB before the move; the move instead removes psi0 (−24 dB), costs
  3.5–4.9 dB of signal per unit drive, and creates two unmodelled stray paths:
  horn light skimming the metasurface plane (−4.4 dB of the rail's power) and
  a direct horn → detector line (−17 dB of the rail's light there; was −41).
  Baffles are the rig's answer. The user has these numbers; whether the
  placement stays is their call. New prelim root `L5_prelim_6a5b8f`.

- **Pipeline verified at λ = 5** on the lab RTX 5090 (V0–V8, 2026-09-04),
  shadow guard settled 2026-09-11 (`min_t` 0.05, `normal_offset` 0.3).
- **First prelim run (2026-09-11, `L5_prelim_9d5878`, PREVIOUS horn):**
  8400 samples in 0.57 h, all gates green — but the **no-metasurface baseline
  beat the SLM** (val AUC 0.976 vs 0.880). Zero phase reproduces the baseline
  exactly (0.000e+00), so it is an **optimisation failure**, not physics
  (finding 24). Suspects: SLM init phase std π/2 (a random diffuser) and
  `w_capture = 0.2` in a noiseless eval. Also: arc position `s0`, not depth,
  dominates detection (crack 0.03 → 0.82 across s0 107 → 148 mm; finding 25);
  read the confusion matrix before quoting `class_acc` (the classifier defaults
  to "crack").
- **Horn replaced (2026-09-24, finding 29):** the physical **RFspin
  H-A75-W20** (50–75 GHz, WR-15, 19–21 dBi), `SIZE_ANT = (22.8, 16.8, 3.7592,
  1.8796, 28.0)` mm, not λ-scaled; textbook aperture model (Nikolova L18 eq.
  18.38: free-space k, midpoint sampling); new gate **V9** green on the laptop
  (D 20.09 dBi = closed form; 19.07 / 20.09 / 21.01 dBi across the band).
  Rail illumination vs the old horn: corr 0.977. The old scaled feed was
  overmoded. **Inner dimensions are a fit — confirm with RFspin's drawing or
  calipers.**
- **Lumerical FDTD** is the full-wave solver (no MoM licence): rail3D's horn
  enters as an **Import source** (no TFSF — the user's explicit requirement);
  **rung −1** simulates the horn itself (`horn_fdtd_case.py`,
  `fdtd_agreement.py --horn`); `horn_source.py --aperture-from` feeds the
  full-wave horn into the rail runs. Nothing has been run in Lumerical yet.
- **First Lumerical session (2026-09-29, finding 30):** `load_horn_source.lsf`
  stopped at `matlabload` with "cannot be opened … MATLAB v7 or higher". The
  loader now finds files by absolute path, falls back to a plain-text copy
  (`horn_txt/`, always written, read with `readdata`), and **checks what it
  loaded** (sizes, E/H power, peak E_y sample) before creating the source.
  **Re-run the same day: the file was found and `matlabload` still refused it.
  The text copy loaded, verified (523 × 627), and the Import source was
  created.** So the lab's Lumerical cannot read scipy-written `.mat` at all.
  The `Error:` line it prints on every run is the caught `matlabload` failure
  and is expected. Every export Python reads uses **`matlabsavelegacy`**
  (plain `matlabsave` is v7.3/HDF5, which scipy cannot read).
- **scipy** was missing from `requirements.txt`; the lab venv lacked it and
  `compare_wavefronts.py --export-case` crashed in `horn_source.build`. Fixed
  (commit 8c0c724); `preflight.py` now names missing packages.
- Laptop gates green after the horn change: V0–V4, V9, V5 (r = 0.958),
  P0–P5, `horn_source.py` self-check (rail fidelity 0.9990 / 1.0000 / 0.9991).

## NEXT: the lab re-run (exact commands: SETUP_LAB §7e)

On the 5090, in order: `git pull` → `pip install -r rail3D/requirements.txt`
(never `-U`) → `preflight.py` → `tests_plumbing.py` → `tests_physics_3d.py`
(V9 must pass) → `validation_3d.py` (V5–V7 under the new horn and placement)
→ `term_budget.py --profile lab` (finding-31 table with shadowing, includes the
SEG_LEN 120 → 240 mm check) → `run_stage.py --stage prelim --fresh
--with-baseline --bundle` (must choose `L5_prelim_6a5b8f`; old datasets are
refused by design) → `ablate_surface.py --stage prelim --long` → optional
`scan_geometry.py`. Re-export the Lumerical bundles (`fdtd_intact`,
`fdtd_crack`: the horn moved, so the Import source, window and region all
changed — LUMERICAL.md has the new numbers; `fdtd_horn` is in the horn frame
and unchanged). Then Lumerical: rung −1 → rung 0 → plate → intact → crack.

## Open items (not built / not decided)

- **Rung-3 difference-field scorer** (crack − intact) in `fdtd_agreement.py` —
  the real FDTD result for cracks (LUMERICAL.md §5).
- **Plumbing test P6** for `fdtd_agreement --horn` and `horn_source
  --aperture-from`. Both were verified only by scratch round trips (ideal
  export passes, flat-phase horn fails; aperture-from corr 0.99987 at offset 0,
  0.99975 at 0.5 mm, flat phase 0.909) — worth folding into
  `tests_plumbing.py`.
- **SEG_LEN re-check script** (defect-signal cosine, 120 vs 240 mm) for the
  new beam — rail ends now lit 0.32 vs 0.28 of peak.
- **`HORN_CRITERIA`** (rung −1 thresholds) are proposed; revisit after the
  first real run.
- The optimiser question (finding 24) — `ablate_surface.py` answers it.
- `surface="metaunit"` is blocked at λ≠8 until a 60 GHz meta-atom library
  exists; SLM and "none" are the usable surfaces.

## Presentation deck

`rail3D/data/generated/rail3D_overview.pptx` (gitignored; 25 slides). The
2026-09-04 original is kept as `rail3D_overview_2026-09-04.pptx`. Slides 6–9
(horn: part choice + dimension drawing, formulas with Nikolova eq. numbers,
aperture/beam, rail illumination), 22 (FDTD setup + rung table) and 23 (prelim
result) were added 2026-09-24 with minimal layout QA. Figures + every quoted
number (`horn_facts.json`): `rail3D/data/figures/deck_2026-09-24/`. Builders:
`rail3D/presentation/deck_figs.py`, `update_deck.py` (needs `pip install
python-pptx`; always rebuilds from the 09-04 backup).

**Still stale in the deck (the user said they would clean these up):** slide
19 card 3 still says shadowing is "inert" (retracted — finding 3); slide 12
shows the old crack ranges (now 3 lines, 1.5–3 mm wide, 4–8 mm deep); slide 24
status is from 09-04; slide 20's last paragraph conflicts with finding 25;
slide 5 says "the next slide is the evidence" but the horn block now follows
it (move slides 6–9 after slide 10, or reword).

## Machines and environment

| | laptop (this repo's author machine) | lab workstation |
|---|---|---|
| repo | `C:\Users\Rei\Documents\MetaRE_railtrack_defect_detection` | `C:\Users\ct2443\Documents\Rei\MetaRE_railtrack_defect_detection` |
| GPU | MX250 2 GB — **seconds-scale CPU gates only** | RTX 5090 (cuda:0) + RTX 4060 Ti (cuda:1) |
| python | `C:\Users\Rei\Downloads\RailDefect\RailDefect\.venv\Scripts\python.exe` (torch 2.7.1+cu118) | repo `.venv` with a **cu128** torch (never `pip -U`) |
| shell | PowerShell / Git Bash | **Git Bash (MINGW64)** |
| defect CSVs | `RAILDEFECT_DATA_DIR` | `export RAILDEFECT_DATA_DIR='C:/Users/ct2443/Downloads/RailDefect/RailDefect'` per shell |

Outside the repo (read-only references):
- Face3D (validated λ = 8 mm reference): `C:\Users\Rei\Downloads\Face3D_clean\Face3D_clean`
- Nikolova, *Lecture 18: Rectangular Horn Antennas*: `C:\Users\Rei\Downloads\L18_Horns.pdf`
  (source of every horn equation number)
- Ye et al. 2018 / 2023 defect papers: PDFs in `C:\Users\Rei\Downloads\`
- Prelim result bundle (2026-09-11): `C:\Users\Rei\Downloads\testing_check_files_Sept11\stage_prelim_bundle (1)\`
- RFspin part pages: https://www.rfspin.com/product/h-a75-w20/ (chosen),
  `/h-a60-w20/`, `/h-a90-w/`

## How the user works (preferences learned the hard way)

- **Heavy runs only on the lab 5090.** The laptop runs seconds-scale CPU gates.
  The lab round trip is slow and one-directional: **execute every changed path
  locally before handing over a lab command** (reproduce a reported failure
  locally first). "Please double check code more often" — said twice after
  lab tracebacks. Report measured numbers, never extrapolated estimates.
- **But** for quick deliverables (slides, explanations) the user has asked to
  minimise checking and go fast — match the request.
- `python tests_plumbing.py` (~25 s) after touching anything that addresses,
  reads back or compares results; `python check_notebook.py` (~1 s) after any
  notebook edit.
- Give shell commands one per fenced `bash` block (the user runs them from Git
  Bash). Commit only on `3D_railhead_upgrade`; never commit `.pt` data.
- Claude-side tooling quirk: a Bash heredoc can collapse backslashes. Write
  patch scripts with the file-writing tool (or PowerShell here-strings) when
  they contain `\`.

## Landmines

- **Any new code that builds the metasurface-plane field must pass
  `upward_only=True`** to `field3d.horn_to_plane` / `scattered_fields`
  (finding 31). Without it, the horn (now above the plane) adds a sideways
  field 4.4 dB below the rail's that the upward model would carry to the
  detectors. Do NOT pass it for the horn's own field on a plane below it
  (`horn_source`, V9): that is the downward illumination and must stay.
- **Never lay out detector centres outside `config.dense_detector_centers()`**
  (finding 17).
- **Never hardcode `L5_<stage>`** — use `data3d.stage_root(stage)`; roots are
  geometry-addressed and auto-suffix on a provenance change (finding 23).
- **`verification_report.json` is merge-loaded**: check each block's `_stamp`;
  an unstamped block was not re-run. Ignore V6b's `recommended` field
  (finding 26).
- **Lumerical is exp(−iωt), like rail3D** — exports compare as-is. STL imports
  as µm unless the length unit is mm first. GPU solver: no TFSF; Import sources
  need 2025 R1.1+. Material name: "PEC (Perfect Electrical Conductor)".
- **Lumerical file I/O (finding 30):** export with `matlabsavelegacy`, never
  `matlabsave` (v7.3, unreadable by scipy); `cd` into the bundle folder first, or
  bake an absolute path into the script. `matlabload`'s "cannot be opened … v7"
  does not say whether the file was missing or unreadable. Running a script
  FILE sets the working directory to its folder (Ansys `cd` page), so do not
  assume "wrong working directory" without evidence.
- The laptop's `data/generated/fdtd_horn_agreement.json` is a **synthetic
  self-test** (a deliberately failing flat-phase horn), not a Lumerical result.
- `data/generated/legacy_unverified/` **on the lab machine** holds the λ=8 full
  dataset (20,512 samples) — the "previous version" record. Do not delete it.
- Speckle scales as λ¹, not λ² (finding 18). The horn stopped scaling with λ on
  2026-09-24 (finding 29).

## Verification (laptop)

```bash
cd rail3D && python preflight.py
```
```bash
python tests_physics_3d.py
```
```bash
python tests_plumbing.py
```

`lab_report.py` runs the whole chain on the 5090 (~10–15 min).
