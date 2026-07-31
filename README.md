# MetaRE_railtrack_defect_detection
A metasurface training code (via NN-backprop) for use as a non-invasive, high accuracy, low-power, railhead defect detector 

## Data setup

This repo is a copy of the original `RailDefect` working folder **without the large `.pt` data files** (some are hundreds of MB). All code reads the large datasets from an external data folder via [`raildefect_paths.py`](raildefect_paths.py):

- Default location: `C:\Users\Rei\Downloads\RailDefect\RailDefect`
- Override by setting the `RAILDEFECT_DATA_DIR` environment variable to wherever the data folder lives.

Read from the data folder (never committed; see `.gitignore`): `generated_datasets2/*_fields_limit5000.pt`, `psi_ms*.pt`, `psi_rail*.pt`, `v_rail*.pt`, `depth*.pt`, `range*.pt`, and the `data_defect_*` CSV folders.

Kept in this repo: all code, the notebook (`training_ms_notebook.ipynb`), small reference fields (`psi_ms_nodefect*.pt`), and trained ONN checkpoints (`onn_*_trained.pt`, ~6–130 KB) so experiment results stay versioned. Data-generation scripts (`2Dmesh*.py`) write their large outputs back into the data folder.
