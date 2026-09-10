# Reading rail3D's validation output

*Last updated: 2026-09-10 · λ = 5 mm (60 GHz) — bump this line in any commit that
changes a gate, a threshold, or what a field means.*

What every file the lab run produces actually contains, what its pass rule is in
code, and the specific ways each one has been misread. Written so a cold session
can interpret a returned result set without re-deriving the gate logic.

Companion docs: `README.md` §6 (why the physics is the way it is),
`SETUP_LAB.md` (how to produce these files), `CLAUDE.md` (current state).

---

## 1. Which script writes what

| file | written by | is it a gate? |
|---|---|---|
| `dataset_config.json` | `data3d.write_dataset_config` | no — provenance |
| `generation.log` | `generate_dataset_3d.log()` | no — timing/identity |
| `verification_report.json` | `tests_physics_3d.py` + `validation_3d.py` + `v8_smoke_test.py`, merged | **yes** |
| `lab_report.md` | `lab_report.py` (runs all three, then adds its own analysis) | partly |
| `geometry_scan.json` | `scan_geometry.py` | **no** — a measurement |
| `v6b_min_t_sweep` / `min_t_refine.txt` | `validation_3d.py --min-t …` | **no** — a study |
| `wavefront_comparison.json`, `wavefront_fields.npz` | `compare_wavefronts.py` | **no** — a study |

The three "no" rows are the ones most often quoted as if they were gates. They
have no pass criterion; they exist to *inform a decision*.

---

## 2. First thing to check: which commit produced this?

Results outlive the code that made them. Before reading any number:

1. `dataset_config.json` → `_git_commit` — the commit the **dataset** was
   generated at.
2. `verification_report.json` → `V0c_guards` **key set** — the most reliable
   fingerprint of the code version, because guards get added over time. A report
   with no `shadow_normal_offset_default` key predates `f046554`
   (2026-09-04), which changed how ray-cast self-hits are suppressed.
3. Console captures (`min_t_refine.txt` etc.) → the **print format**. The V6b
   sweep was rebuilt in `f046554`; the old format prints
   `intact artifact … | crack effect … (resolved …)` with no normal-offset
   column and one crack. The new one prints
   `artifact mean … max … barcode … | c0 … c1 … resolved …`.

If any of those say "old", the numbers describe the old mechanism. That is not a
reason to discard them — it is a reason not to carry their *conclusions*
forward.

---

## 3. `dataset_config.json` — the geometry fingerprint

36 keys recording the geometry the fields were computed with, plus `_created`,
`_git_commit`, `_host`, `_counts`, `_duration_hours`, `_samples_per_second`.

Its job is refusal, in three places:

- `data3d.check_dataset_config` — **training** refuses a dataset whose keys
  differ from the active config.
- `data3d.check_generation_root` — the **generator** refuses to write into a root
  already holding another geometry's artifacts (it used to skip existing shards
  and silently relabel the mixed set).
- checkpoints carry the same stamp and refuse to resume across a change.

**Provenance keys that bite.** `SHADOW_MIN_T` and `SHADOW_MODE` are in the
fingerprint. Changing either invalidates every existing dataset — which is
correct, but means a shadow decision must be made *before* the full generation,
not after.

**What to sanity-check:** `WVL`, `H_MS`, `NX`/`NY`, `SEG_LEN`, `MESH_DS`,
`CLASS_NAMES` length, and `_counts`. A `_counts` of ~20/class is a **smoke**
set — every separability number computed from it has n=20 sampling noise of
roughly ±0.05 AUC. Do not compare two smoke runs at three decimal places.

---

## 4. `generation.log`

Per-shard progress with timestamps, plus two things worth reading deliberately:

- **the GPU line** — `device=cuda:0 (NVIDIA GeForce RTX 5090) pid=…`. CUDA
  indices do not match Task Manager's, and CUDA cannot use Intel integrated
  graphics at all, so "it's running on the wrong GPU" is almost always a misread
  of Task Manager. Match the PID against `nvidia-smi` instead.
- **the rate** — `samples/s` per class and overall. Extrapolate the full run from
  this, never from a per-sample estimate: the ray-cast pass, contention and the
  I/O tail are all in the measured rate and absent from a back-of-envelope.

Interrupt safety: shards are written atomically and skipped on re-run, and every
sample derives from a deterministic seed, so Ctrl-C loses at most the shard in
flight.

---

## 5. `verification_report.json` — the gates

Every entry carries `pass` and `seconds`. The pass rules, from the code:

| gate | pass rule (source) | what a failure would mean |
|---|---|---|
| **V0** | `orientations_ok ∧ bands_ok ∧ seed_reproducible ∧ resolution_ok` | the defect geometry is not what the class name says |
| **V0b** | redundancy pruning keeps the unique detector, variance does not | dense-start pruning would retain duplicates |
| **V0c** | 11 booleans: lattice size, capture clamp, dataset/root/checkpoint refusals | a staleness guard has stopped guarding |
| **V1** | all of `psi0, psi1, psi2, chunk_invariance, batch_consistency` < `1e-5` | the fast rewrite changed the physics |
| **V2** | `max_rel_l2 < 1e-4` | FFT propagation ≠ spatial convolution |
| **V3** | `central_rel_l2 < 0.02` | two independent propagators disagree |
| **V4** | specular centroid within one pixel, power conserved, crown up | the scene is geometrically wrong |
| **V5** | `pearson_r > 0.9` | 3D PO disagrees with the validated 2D code |
| **V6** | `artifact_worst < 0.03 ∧ resolved < 0.02` | ray-cast is injecting artifacts |
| **V7** | `mean_cosine > 0.95 ∧ min_cosine > 0.90` | the λ/8 generation mesh is not converged |
| **V8** | 11 booleans ANDed, headed by `sep[-1] > sep[0] * 1.2` | training does not train |

### Fields that mean something specific

**V0 `long_extent_y_mm` / `long_extent_s_mm`** — orientation is asserted
*physically*, not by eye: a longitudinal crack must span >15 mm along y and
<5 mm along s; a transverse one the reverse; an oblique one >10 mm in both.

**V0 `resolution_consistency_mm`** — the same depth field rendered at λ/8 and at
the occluder step, compared. This is what guarantees the simulation mesh and the
ray-cast occluder describe *the same physical defect*.

**V1 `chunk_invariance` / `batch_consistency`** — as important as the field
values. They prove the result does not depend on how the work was divided, which
is what makes the memory-bounded rewrite safe at any scale.

**V3 `central_rel_l2`** — worth checking digit by digit against the λ=8 value
(`0.001027346937917173`). Identical to 6 s.f. across wavelengths *and machines*
is the numerical confirmation of the scaled replica.

**V5** deliberately runs a **λ/16** mesh, so discretisation error cannot
masquerade as solver disagreement. Production-mesh fidelity is V7's job, not
V5's. A fringe offset with envelope agreement is finite-segment vs
infinite-extrusion registration, not solver error.

**V7 `cases` / `min_cosine`** — the criterion is the **defect-signal direction
cosine** (`barcode(defect) − barcode(intact)` at λ/8 vs λ/16), not raw field L2.
Raw complex-field L2 never converges at these facet sizes because of PO glint
speckle; the task consumes barcodes, so fidelity is judged there.

**V6 `worst_rel_l2`** — a **max over 8 samples**, and in practice 7 of them are
exactly 0. Read the `samples` list, not the headline. See §8.

---

## 6. `lab_report.md` — the parts no gate computes

Three sections are analysis, not gates:

### Stored shards
Per class: `|psi1| mean`, `|psi2| mean`, their ratio, finiteness,
sample-to-sample variation, and the realised parameter ranges.

- `psi2/psi1` ≈ 1e-2 — the horn re-radiation term is a ~1% correction. If it
  ever approaches unity, the double-bounce model is being asked to do something
  it was not designed for.
- **realised** `depth`/`L`/`theta` ranges are the check that sampling actually
  covers the configured ranges. Historic bugs (depth clipping, wear normalised
  by a global instead of in-band peak) showed up here first, as a realised range
  pinned to one end of the configured one.

### Class separability through untrained optics
`ref` = intact mean barcode; `cos_gap(det, ref) = 1 − cos(normalised barcode,
normalised ref)`.

- `intact self-spread` is the **noise floor** — the spread of intact against its
  own mean. A class whose `cos-gap mean` is near it is not separable *before
  training*, which is a statement about the optics, not about the trainer.
- `p5` matters more than the mean: it is the 5th-percentile gap, i.e. the
  hardest samples, which is what sets recall at a calibrated threshold.
- `AUC vs intact` here uses `losses3d.auc_score` — all-pairs, ties at 0.5.

### Wavefront comparison
See §9.

---

## 7. `geometry_scan.json` — measuring the plane, not scoring it

One record per configuration. The metrics:

| field | definition |
|---|---|
| `energy` | mean \|ψ\|² over intact samples — how much light lands at all |
| `edge_frac` | share of intensity in the outer ring — high means energy is escaping the aperture |
| `intact_field_spread` | mean ‖I − I_ref‖ / ‖I_ref‖ over intact — the field-level noise floor |
| `intact_det_spread` | the same through the detector chain |
| `<class>_field_auc` | AUC of that class's field distance vs intact's — **no optics**, pure field information |
| `<class>_det_auc` | AUC of `cos_gap` through a **fixed random SLM (seed 0)** + the detector lattice |

`field_auc` is the honest measure of what information reaches the plane.
`det_auc` measures what an *untrained* readout can extract from it — useful, but
it depends on one arbitrary phase seed, so small differences are noise.

**`H` is in millimetres**, not wavelengths. H = 150 is 30λ.

**The x_center rows are the sharpest argument for dark-field.** Sliding the
aperture onto the specular lobe (`x_center = −214`) more than doubles the energy
and marginally *improves* field AUC — while detector AUC collapses. The specular
lobe carries distinguishability the readout cannot use. "More light is worse" is
true, but the reason is that the extra light is not *readable*, not that it is
absent.

---

## 8. The shadow files — the most misread output in the repo

`V6_shadowing` and the V6b `--min-t` sweep. Read §9 of `README.md` and the
`raycast_shadow_mask` docstring alongside these.

**V6's `samples` list is the real content.** In the 2026-09-03 run:

| crack | depth | `rel_l2` |
|---|---|---|
| #0 | 2.72 mm | 0.0 |
| #1 | 5.28 mm | 0.0197 |

Every other sample is 0.0. So `worst_rel_l2: 0.0197` is one crack in eight, and
"shadowing changes the field by 2%" is a misreading.

**`crack_shadow_with_resolved_occluder`** is a separate control: the same crack
re-solved against a λ/8 occluder instead of the production λ/2 one. It isolates
*"the occluder cannot represent the crack"* from *"the guard deleted the hit"* —
two independent reasons a crack shows no shadow.

### Traps

1. **The sweep's benefit column can be a single unrepresentative sample.** The
   pre-`f046554` harness probed `crack` index 0 only — despite naming the
   variable `crack_deep` — and index 0 is a near-minimum-depth crack that shows
   no shadowing at any `min_t`. Its "crack effect 0.0000 everywhere" is therefore
   not evidence that shadowing is inert; V6's own table, in the same report,
   shows index 1 responding at the production setting. The current harness sweeps
   `crack_idxs=(0, 1)` and prints per-index columns for exactly this reason.
2. **"Smallest artifact-safe min_t" is not the objective.** The current code says
   so in a comment and returns `None` instead of a number when nothing has
   benefit. The objective is the configuration keeping the **most real
   shadowing** while staying under the artifact bound.
3. **The two "facet" columns are different facets.** `min_t_over_occluder_facet`
   divides by λ/2; `min_t_over_resolved_facet` divides by λ/8. Comparing the
   production and resolved columns at one "× facet" number was misleading and is
   now reported explicitly.

---

## 9. `wavefront_comparison.json`

One sample solved every way, scored against `lam5 psi1 (no shadow)`.

- `rel_l2` — raw complex-field difference. Sensitive to speckle; a large value
  is not automatically a disagreement of substance.
- `complex_corr` — scale- and offset-invariant; **this is the number to quote.**
  The residual after fitting one complex gain is `sqrt(1 − complex_corr²)` by
  construction.
- `barcode_cos` — what the detectors actually see. Far more robust than the field:
  the λ=8 rows sit at `complex_corr 0.297` but `barcode_cos 0.948`.

**The default sample is `crack[0]`** — the same shallow crack as §8. That is why
`raycast 3.0` and `no shadow` come out at `rel_l2 0.0000` here. Use `--idx 1` to
see a crack that responds.

For an external full-wave field dropped in with `--external`, the tool prints the
detected time convention, the fitted gain and phase, and the resampling coverage
*before* any metric — see `LUMERICAL.md` §5.

---

## 10. V8 — what it does and does not claim

**It is a training smoke test, not a performance measurement.**

- Pass is keyed on **train separation** growing 1.2× (`0.0290 → 0.0562` = 1.94×).
- `best_val` on a smoke split (8 defect / 3 intact) is degenerate:
  `class_acc: 0.25` is exactly chance for four classes, `false_alarm: 0.333` is
  one of three intact samples, `auc` is quantised to 1/24. **Do not read these as
  accuracy.**
- `full_eval_auc` exercises the `full_evaluation` code path (it exists because a
  device-mismatch bug in `roc_points` survived precisely because V8 never called
  it). It is not held-out generalisation.
- `capture_first → capture_last` looks alarming (0.959 → 0.079) and is not.
  Detectors go 130 → 8; the pure count ratio is 8/130 = 0.0615, so 0.0789 is
  **1.28× better than proportional** — the metasurface is concentrating light
  onto the survivors. The gate only asserts `0 < capture ≤ 1`, so this is a
  number to watch, not a pass/fail. Compare it to `n_det_final / 130` each run.
- `resume_max_metric_mismatch: 0.0` is exact, and must stay exact — it is the
  kill-and-resume bit-identity claim.

---

## 11. Triage order for a returned result set

1. `V0c` key set → which code version is this? (§2)
2. Any `pass: false` → stop, read that gate's fields.
3. `V3` digits vs the recorded λ=8 value → is this the geometry we think?
4. `dataset_config` `_counts` → smoke or full? Sets how many digits are real.
5. Shard realised ranges → does sampling cover the configured ranges?
6. `intact self-spread` vs per-class `cos-gap` → is anything separable at all
   before training?
7. Only then, the studies: geometry scan, shadow sweep, wavefront comparison.
