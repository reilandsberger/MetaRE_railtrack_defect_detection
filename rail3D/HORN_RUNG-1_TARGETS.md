# Rung −1 (the horn alone): goal metrics

*Last updated: 2026-10-26. Thresholds are `HORN_CRITERIA` in `fdtd_agreement.py` (proposed;
revisit after the first real run). Target column = `python fdtd_agreement.py --horn-target`
(figure `data/figures/fdtd_horn_target.png`): the model with a STATED set of
imperfections, i.e. what a passing run should look like, not a result.*

**Model reference values** (RFspin H-A75-W20, textbook aperture model): directivity
**20.08 dBi**; half-power beam width **15.8° (E-plane) / 17.2° (H-plane)**; part spec
19–21 dBi; level toward the metasurface (32.5° off boresight, H-plane) **−21.4 dB**.

**Feed check (the waveguide alone):** TE10 effective index
n_eff = √(1 − (λ / 2a)²) = **0.7468** for a = 3.7592 mm, λ = 5 mm
(cutoff c / 2a = 39.9 GHz; group index 1/n_eff = 1.339; guide wavelength 6.70 mm).

| Printed metric | What it tests | Pass at | Good | Target figure | Bad, and what it suggests |
|---|---|---|---|---|---|
| Aperture `complex corr` (z′ = 0.5 mm) | the textbook aperture field vs the real horn's | ≥ 0.95 | 0.97–0.99 | 0.9905 | 0.90–0.95: rim / outer-wall currents matter more than the model assumes. < 0.90: wrong mode, geometry or axes |
| Alignment convention | sign convention of the export | `as-is` | as-is | as-is | `conjugated`: export or source direction flipped |
| Directivity (FDTD) | overall beam concentration | within ±0.5 dB of the model, and in 19–21 dBi (±0.3) | 19.6–20.6 dBi | 20.36 dBi | > 1 dB off: wrong mouth size / STL scale, or a wrong feed |
| Beam width E / H | beam shape in each plane | each within ±1.5° | ≈ 15.8° / 17.2° | 15.6° / 17.9° | consistently wider: effective mouth smaller or less uniform |
| Near-plane `complex corr` (z′ = 15 mm, 3λ out) | the field that travels to the rail | ≥ 0.98 | ≥ 0.99 | 0.9985 | low while the aperture is fine: a propagation or monitor problem |
| z = 150: FDTD vs **our propagation of FDTD's own aperture** | **rail3D's propagation code vs Maxwell**, independent of the aperture model | ≥ 0.98 | ≥ 0.99 | 0.9988 | **the key check**: low means our propagation is wrong near the horn |
| z = 150: FDTD vs our full model (80 × 80) | the aperture model, near the horn | ≥ 0.95 | ≥ 0.97 | 0.9966 | low while the line above is fine: the aperture model, not the propagation |
| Model 20 × 20 vs 80 × 80 | the production quadrature | ≈ 1 | ≥ 0.9999 | ≥ 0.9999 | lower: the 20 × 20 sum is too coarse |
| `level` dB at z = 150 / 140 / 15 | how power falls with distance | — | within ±0.5 dB | +0.09 / +0.12 / +0.15 | > 1 dB: the spreading differs |
| `centroid shift` mm | where the beam points | — | under ~1–2 mm | ≤ 0.5 (z = 150), 1.4 (z = 15) | several mm: the beam points differently |
| Import plane z = 15 `complex corr` | the source injected into the rail runs vs the real horn there | ≥ 0.98 | ≥ 0.99 | 0.9995 | low: the rail rungs inject the wrong field |
| Level toward the metasurface | sideways horn light (finding 31) | informational | ≈ −21 dB in both | model −21.4, FDTD −22.0 | a few dB different: revise the sideways-light estimate |

**Reading them together**
- **All pass:** the horn model and the propagation are both confirmed; the rail rungs can use
  the model source as is.
- **Propagation check passes, model checks fail:** the propagation is fine and the aperture
  model is the limitation. Rebuild the rail source from the full-wave horn:
  `python horn_source.py --aperture-from data/generated/fdtd_horn/horn_aperture.mat`.
- **Propagation check fails:** a problem in rail3D's propagation; stop and investigate.

**The target's imperfection model** (`HORN_TARGET_MODEL`): rim fringing 0.6 mm (Gaussian
blur past the opening), smooth amplitude error 4% RMS (6 mm correlation), smooth phase
error 5° RMS (8 mm), FDTD grid phase error 1°/λ along the boresight, arbitrary complex gain.
