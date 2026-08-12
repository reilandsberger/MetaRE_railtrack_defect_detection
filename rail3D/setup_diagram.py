"""Render the physical-setup diagram and mesh-review figures for approval.

Run from the rail3D folder (CPU-only, laptop-safe):
    python setup_diagram.py
Outputs land in rail3D/data/figures/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")

from rail3d import config, viz_setup  # noqa: E402


def main() -> None:
    config.ensure_dirs()
    out = config.FIGURE_DIR

    fig = viz_setup.setup_diagram(save_path=out / "setup_diagram.png")
    print(f"wrote {out / 'setup_diagram.png'}")

    fig = viz_setup.cross_section_overlay_figure(save_path=out / "cross_sections.png")
    print(f"wrote {out / 'cross_sections.png'}")

    fig = viz_setup.mesh_review_figure(save_path=out / "mesh_review.png")
    print(f"wrote {out / 'mesh_review.png'}")


if __name__ == "__main__":
    main()
