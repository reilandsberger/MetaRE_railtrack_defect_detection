#%%
from __future__ import annotations

from pathlib import Path
import csv

type = "dent"
DATA_PATH = Path(f"data_defect_{type}/data_log_{type}.txt")


def load_rows(path: Path) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []

    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for raw_row in reader:
            row = [cell.strip() for cell in raw_row if cell.strip()]
            if not row or row[0] == "Update" or row[0] == "#":
                continue

            rows.append(
                {
                    "index": int(row[0]),
                    "width": float(row[1]),
                    "depth": float(row[2]),
                    "total": int(row[3]),
                }
            )

    if not rows:
        raise ValueError(f"No numeric rows found in {path}")

    return rows



rows = load_rows(DATA_PATH)

min_width = min(rows, key=lambda row: row["width"])
max_width = max(rows, key=lambda row: row["width"])
min_depth = min(rows, key=lambda row: row["depth"])
max_depth = max(rows, key=lambda row: row["depth"])

print(f"Loaded {len(rows)} rows from {DATA_PATH}")
print(
    f"Minimum width: {min_width['width']*180/1004:.4f} mm"
    f"(row {min_width['index']}, depth {min_width['depth']:.4f})"
)
print(
    f"Maximum width: {max_width['width']*180/1004:.4f} mm"
    f"(row {max_width['index']}, depth {max_width['depth']:.4f})"
)
print(
    f"Minimum depth: {min_depth['depth']*180/1004:.4f} mm"
    f"(row {min_depth['index']}, width {min_depth['width']:.4f})"
)
print(
    f"Maximum depth: {max_depth['depth']*180/1004:.4f} mm"
    f"(row {max_depth['index']}, width {max_depth['width']:.4f})"
)

