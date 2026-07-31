import os
from pathlib import Path

# Folder holding the large simulation datasets (not tracked in this repo).
# Override by setting the RAILDEFECT_DATA_DIR environment variable.
RAILDEFECT_DIR = Path(
    os.environ.get("RAILDEFECT_DATA_DIR", r"C:\Users\Rei\Downloads\RailDefect\RailDefect")
)
