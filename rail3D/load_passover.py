"""Load the rail pass-over detector data into a `sensor_data` dictionary.

Needs only numpy and the Python standard library.

    python load_passover.py                 # summary of the low-noise set in this folder
    python load_passover.py path/to/folder --noise high

    from load_passover import load_passover, to_time, resample
    sensor_data = load_passover("path/to/folder", noise="low")
    p = sensor_data["passes"]["crack_0"]
    p["detectors"]          # (n_frames, 8) array, columns d1..d8
    p["y_mm"]               # (n_frames,) defect position along the rail, mm
    t = to_time(p["y_mm"], speed_m_s=20.0)            # seconds, starts at 0
    t_new, det_new = resample(p, speed_m_s=20.0, rate_hz=50e3)

sensor_data layout:
    {"meta":    everything in passover_labels.json except the pass list
                (geometry, detector positions, units, noise model, ...),
     "channels": ["d1", ..., "d8"],
     "passes":  {pass_id: {"class", "class_index", "visibility", "description",
                            "parameters", "pose", "peak_change_vs_intact",
                            "frame", "y_mm", "depth_under_beam_mm", "detectors"}}}
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

CHANNELS = [f"d{k}" for k in range(1, 9)]


def load_passover(folder: str | Path = ".", noise: str = "low") -> dict:
    """noise: "clean", "low" or "high" -- which passover_<noise>.csv to read."""
    folder = Path(folder)
    labels = json.loads((folder / "passover_labels.json").read_text(encoding="utf-8"))
    info = {p["pass_id"]: p for p in labels["passes"]}
    rows: dict[str, list] = {}
    with open(folder / f"passover_{noise}.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.setdefault(r["pass_id"], []).append(r)
    passes = {}
    for pid, rs in rows.items():
        rs.sort(key=lambda r: int(r["frame"]))
        p = info[pid]
        passes[pid] = {
            "class": p["class"], "class_index": p["class_index"],
            "visibility": p["visibility"], "description": p["description"],
            "parameters": p["parameters"], "pose": p["pose"],
            "peak_change_vs_intact": p["peak_change_vs_intact"],
            "frame": np.array([int(r["frame"]) for r in rs]),
            "y_mm": np.array([float(r["y_mm"]) for r in rs]),
            "depth_under_beam_mm": np.array([float(r["depth_under_beam_mm"]) for r in rs]),
            "detectors": np.array([[float(r[c]) for c in CHANNELS] for r in rs]),
        }
    meta = {k: v for k, v in labels.items() if k != "passes"}
    meta["noise_set"] = noise
    return {"meta": meta, "channels": CHANNELS, "passes": passes}


def to_time(y_mm: np.ndarray, speed_m_s: float) -> np.ndarray:
    """Frame times in seconds for a train at speed_m_s; the first frame is t = 0.
    Every pass moves toward +y, so time increases with y_mm."""
    return (np.asarray(y_mm) - y_mm[0]) / 1000.0 / speed_m_s


def resample(p: dict, speed_m_s: float, rate_hz: float):
    """(t, detectors) on a uniform time grid at rate_hz, linear interpolation.

    The traces are smooth at the stored 1 mm (wear: 2 mm) step, so interpolating
    the CLEAN set is accurate. On a noisy set the noise was drawn per stored
    frame: interpolation smooths it, and its statistics then depend on the new
    rate -- add noise after resampling instead if that matters."""
    t = to_time(p["y_mm"], speed_m_s)
    t_new = np.arange(0.0, t[-1] + 1e-15, 1.0 / rate_hz)
    det = np.stack([np.interp(t_new, t, p["detectors"][:, k]) for k in range(p["detectors"].shape[1])], 1)
    return t_new, det


def load_fields(folder: str | Path = ".", plane: str = "detector") -> dict:
    """Raw complex fields from passover_fields.npz, for choosing your OWN detectors.

    plane: "detector" (z = 250 mm, where the CSV's 8 windows sit; no metasurface,
    just free propagation from the metasurface plane) or "metasurface" (z = 150 mm,
    the light arriving where a metasurface would go).
    Returns {"x_mm": (60,), "y_mm": (30,), "pixel_mm", "power_unit", "z_mm",
             "passes": {pass_id: {"y_mm": (n_frames,), "field": (n_frames, 60, 30) complex}}}
    Frames are the same, in the same order, as the CSV rows of that pass.
    """
    z = np.load(Path(folder) / "passover_fields.npz")
    key = {"detector": "field_detector", "metasurface": "field_metasurface"}[plane]
    pids = sorted({k.split("__")[0] for k in z.files if "__" in k})
    return {"x_mm": z["x_mm"], "y_mm": z["y_mm_grid"], "pixel_mm": float(z["pixel_mm"]),
            "power_unit": float(z["power_unit"]), "plane": plane,
            "z_mm": float(z["z_detector_mm"] if plane == "detector" else z["z_metasurface_mm"]),
            "passes": {p: {"y_mm": z[f"{p}__y_mm"], "field": z[f"{p}__field_{plane}"]} for p in pids}}


def readout(fields: dict, field: np.ndarray, centres_mm, window_mm=(11.375, 7.0)) -> np.ndarray:
    """Detector powers for windows of your choice, in the CSV's units.

    field: (n_frames, 60, 30) from load_fields(...)["passes"][pid]["field"].
    centres_mm: list of (x, y) window centres in mm, on the plane of `fields`.
    A window collects |E|^2 over the pixels whose centres lie inside it, times
    the pixel area -- the same rule that made d1..d8 -- divided by the CSV's
    power unit, so readout(load_fields(...), f, <the 8 CSV centres>) on the
    detector plane reproduces passover_clean.csv.
    Returns (n_frames, n_windows)."""
    x, y, dx = fields["x_mm"], fields["y_mm"], fields["pixel_mm"]
    I = np.abs(field) ** 2
    out = []
    for cx, cy in centres_mm:
        mx = np.abs(x - cx) < window_mm[0] / 2
        my = np.abs(y - cy) < window_mm[1] / 2
        out.append(I[:, mx][:, :, my].sum(axis=(1, 2)) * dx * dx)
    return np.stack(out, 1) / fields["power_unit"]


def main() -> int:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else Path(".")
    noise = sys.argv[sys.argv.index("--noise") + 1] if "--noise" in sys.argv else "low"
    sd = load_passover(folder, noise)
    print(f"{len(sd['passes'])} passes, noise set '{noise}', channels {sd['channels']}")
    print(f"{'pass_id':9s} {'class':7s} {'frames':>6s} {'y_mm':>15s}  {'peak change':>11s}  visibility")
    for pid, p in sd["passes"].items():
        print(f"{pid:9s} {p['class']:7s} {len(p['y_mm']):6d} {p['y_mm'][0]:7.1f}..{p['y_mm'][-1]:<6.1f}"
              f"  {100 * p['peak_change_vs_intact']:10.0f}%  {p['visibility']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
