"""Static check of rail3D_pipeline.ipynb: does every cell run in order?

    python check_notebook.py

Jupyter hides cross-cell ordering: a cell that uses a name defined in a LATER
cell looks fine while you are editing and only fails on a fresh kernel, after
you have already scrolled past it. That is a real failure mode here -- a
one-command cell was once placed ahead of the cell defining STAGE, and the first
person to run the notebook top-to-bottom hit NameError.

Checks, cheapest first:
  * every code cell parses
  * every name in WATCH is bound by an earlier cell (or within the same cell --
    Python reports within-cell ordering clearly by itself, so this lint is only
    about the order Jupyter hides)
  * the cells that people most often run out of order carry a STAGE guard, so
    the failure is an instruction rather than a NameError

Run it after editing the notebook. It needs no GPU, no dataset, ~1 second.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB = HERE / "rail3D_pipeline.ipynb"

# Names the setup cell and the pipeline modules provide before any watched
# name is used; not exhaustive, just enough to keep the report signal-only.
AMBIENT = {
    "os", "subprocess", "sys", "json", "Path", "np", "torch", "plt", "Image",
    "display", "Markdown", "HERE", "PROFILE", "device", "run", "free_gpu",
    "show", "config", "data3d", "losses3d", "mesh3d", "optics3d", "sections",
    "train3d", "viz_setup", "pd", "replace",
    "make_bundle", "result_files", "write_summary",
}
# The names whose ordering has actually bitten, or would be expensive to get
# wrong: stage identity, the config factory, and the per-run handles.
WATCH = {"STAGE", "TAG", "DATASET", "_s", "make_cfg", "cfg_slm", "cfg_none",
         "cfg_mu", "hist_slm", "hist_none", "plot_history", "history_of",
         "ana", "summary", "FRESH", "WITH_BASELINE"}
# Entry points people jump straight to; each must fail with an instruction.
GUARDED = ("# The whole chain, as a subprocess",
           "from run_stage import make_bundle")


def bound_names(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            out.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = node.args
                out |= {x.arg for x in a.args + a.kwonlyargs + a.posonlyargs}
                for extra in (a.vararg, a.kwarg):
                    if extra:
                        out.add(extra.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                out.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (ast.For, ast.comprehension)):
            tgt = node.target
            if isinstance(tgt, ast.Name):
                out.add(tgt.id)
    return out


def main() -> int:
    if not NB.exists():
        print(f"!! {NB} not found")
        return 1
    cells = json.loads(NB.read_text(encoding="utf-8"))["cells"]
    seen, problems, n_code = set(AMBIENT), [], 0

    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        n_code += 1
        src = "".join(c["source"])
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            problems.append(f"In[{n_code}] (cell {i}): syntax error: {e}")
            continue
        local = bound_names(tree)
        used = {x.id for x in ast.walk(tree)
                if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load)}
        for name in sorted((used & WATCH) - seen - local):
            problems.append(
                f"In[{n_code}] (cell {i}): uses {name!r} before any earlier "
                f"cell defines it - it would NameError on a fresh kernel")
        seen |= local

    for sub in GUARDED:
        hit = [i for i, c in enumerate(cells) if sub in "".join(c["source"])]
        if not hit:
            problems.append(f"guarded cell not found: {sub!r}")
        elif "'STAGE' not in globals()" not in "".join(cells[hit[0]]["source"]):
            problems.append(
                f"cell {hit[0]} ({sub!r}) has no STAGE guard - running it "
                f"first would NameError instead of saying what to run")

    print(f"{len(cells)} cells ({n_code} code) checked")
    if problems:
        print("\nPROBLEMS")
        for p in problems:
            print("  -", p)
        return 1
    print("ordering OK: every watched name is defined before use, guards present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
