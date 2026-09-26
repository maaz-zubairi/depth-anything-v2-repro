# PROVENANCE

Update this **as you go**, not at the end. Categories from the brief:
`OURS` = written by us - `ADAPTED` = adapted from a repo (say what changed) - `REUSED` = used as-is - `PAPER` = number reported by the authors - `RESULT` = number we produced.

> **AI assistance:** AI tools were used during the build process to help scaffold the repository, draft scripts, tests, and documentation, and to support iterative development. All retained code and files were reviewed, understood, and, where needed, modified by the team before inclusion.

## Code
| Component | Status | Source / notes |
|---|---|---|
| Model weights (V1 & V2, S/B/L) | REUSED | Hugging Face `depth-anything/Depth-Anything-V2-*-hf`, `LiheYoung/depth-anything-*-hf`; unchanged |
| Architecture (DINOv2 + DPT) | REUSED | `transformers` `DepthAnythingForDepthEstimation`; reference: github.com/DepthAnything/Depth-Anything-V2 @ <commit> |
| `src/dav2/models.py` | OURS (AI-assisted) | thin wrapper; interpolation to input size follows the HF model-card example |
| `src/dav2/metrics.py` | OURS (AI-assisted) | alignment protocol follows MiDaS / official eval - confirm vs official code @ <commit> |
| `src/dav2/data.py` | OURS (AI-assisted) | |
| `scripts/*.py`, `tests/` | OURS (AI-assisted) | |
| `app/streamlit_app.py` | OURS (AI-assisted) | |

## Data
| Dataset | Source | Notes |
|---|---|---|
| NYU Depth V2 Eigen test (654) | Ultralytics repackaging of NYU Depth V2 (Silberman et al., 2012) | third-party split file - verify against the original |
| DA-2K | HF `depth-anything/DA-2K` (Yang et al., 2024) | |
| KITTI (optional) | cvlibs.net | state clearly if skipped / subsetted |

## Numbers
| Number | Status |
|---|---|
| Paper-reported values in `results/paper_reported.csv` | PAPER (cite table + page) |
| Everything in `results/*.csv` other than that file | RESULT (our hardware, date, commit) |
