"""geometry.py -- write a RAIL3D_GEOMETRY override file from the setup dials.

The setup sweep reports each candidate as a horn angle, a metasurface height and the
aperture's x edges (sweep_eval.json: theta, H, x0, x1). This turns one into the file
config.py reads at import (RAIL3D_GEOMETRY), so the WHOLE pipeline -- generation,
gates, training, analysis -- runs on that geometry, in its own auto-suffixed roots.

    python geometry.py --name F1 --theta 55 --H 150 --x0 -190 --x1 75
    python geometry.py --show                     # what the current process resolves to

Then, in the same shell (Git Bash / Linux; PowerShell: $env:RAIL3D_GEOMETRY = "...";
cmd: set RAIL3D_GEOMETRY=...):

    export RAIL3D_GEOMETRY=geometries/F1.json
    python preflight.py && python run_stage.py --stage prelim --fresh --with-baseline --bundle

Unset it (unset RAIL3D_GEOMETRY) to return to the default geometry. Every entry point
prints a "[rail3d] GEOMETRY OVERRIDE ..." line while it is set.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "geometries"


def spec_for(name: str, theta: float, h: float, x0: float, x1: float, y_half: float,
             dx: float) -> dict:
    nx = int(round((x1 - x0) / dx))
    ny = int(round(2 * y_half / dx))
    return {"label": name, "theta_deg": theta, "H_MS": h, "NX": nx, "NY": ny,
            "PLANE_X_CENTER": round(x1 - nx * dx / 2, 4)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name")
    ap.add_argument("--theta", type=float, default=55.0, help="horn angle from vertical (deg)")
    ap.add_argument("--H", type=float, default=150.0, help="metasurface height (mm)")
    ap.add_argument("--x0", type=float, help="aperture far edge (mm, away from the horn)")
    ap.add_argument("--x1", type=float, default=75.0, help="aperture near edge (mm, horn side)")
    ap.add_argument("--y-half", type=float, default=37.5, help="aperture half-length along the rail")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, str(HERE))
    from rail3d import config, data3d
    if a.show:
        print(f"override: {config.GEOMETRY_OVERRIDE}")
        print(f"theta {math.degrees(config.THETA_INC):g} deg, D {config.DIST_ANT:.2f} mm, H {config.H_MS:g}, "
              f"{config.NX}x{config.NY} px at x {config.PLANE_X_CENTER:+g}, detectors {config.DET_GRID}, "
              f"geometry tag {data3d.geometry_tag()}")
        return 0
    if not a.name or a.x0 is None:
        ap.error("--name and --x0 are required (or --show)")
    spec = spec_for(a.name, a.theta, a.H, a.x0, a.x1, a.y_half, config.DX)
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{a.name}.json"
    path.write_text(json.dumps(spec, indent=1) + "\n", encoding="utf-8")
    xg = -a.H * math.tan(math.radians(a.theta))
    lo = spec["PLANE_X_CENTER"] - spec["NX"] * config.DX / 2
    hi = spec["PLANE_X_CENTER"] + spec["NX"] * config.DX / 2
    print(f"wrote {path}: {spec}")
    print(f"  aperture x {lo:+g}..{hi:+g} mm ({spec['NX']}x{spec['NY']} px = "
          f"{spec['NX'] * config.DX:g}x{spec['NY'] * config.DX:g} mm), crown glint at x {xg:.0f} "
          f"({'INSIDE: bright field' if lo <= xg <= hi else 'outside: dark field'})")
    print(f"  use it:  export RAIL3D_GEOMETRY={path.relative_to(HERE).as_posix()}   "
          f"(check: python geometry.py --show)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
