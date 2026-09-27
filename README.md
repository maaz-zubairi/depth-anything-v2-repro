# Depth Anything V2 - reproduction, experiment, deployment

Course project (Intro to Machine Learning). Paper: Yang et al., *Depth Anything V2*, NeurIPS 2024 (arXiv:2406.09414).
Team: Ayyan Sohail, Omar Khan, Maaz Zubairi.

**This project uses the authors' pretrained checkpoints (Small / Base / Large). No model is trained.**
The work is the evaluation pipeline, the metric code, a paper-specific experiment, and a deployed demo.

## Quickstart
```bash
git clone https://github.com/YOUR_USER/depth-anything-v2-repro.git && cd depth-anything-v2-repro
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
pytest -q                                                # metric unit tests (no GPU, no data needed)
python scripts/01_smoke_test.py                          # S/B/L forward pass, shape + finite check
streamlit run app/streamlit_app.py                       # the deployment
```

## Pipeline (run in this order; Colab/Kaggle T4 for 02-04)
| Step | Command | Output |
|---|---|---|
| Smoke test | `python scripts/01_smoke_test.py` | console check |
| Dense-depth eval (NYU, KITTI optional) | `python scripts/02_eval_depth.py --dataset nyu --data-root data/nyu --eigen-crop [--limit 10]` | `results/nyu_*.csv` |
| DA-2K pair accuracy, V1 and V2 | `python scripts/03_eval_da2k.py --data-root data/da2k [--limit 20]` | `results/da2k_pairs.csv` |
| Latency / params / memory | `python scripts/04_benchmark_latency.py` | `results/latency.csv` |
| Experiment analysis | `python scripts/05_analyze_da2k.py` | `results/da2k_*.csv`, `figures/da2k_gain_by_scene.png` |
| Paper-vs-ours + accuracy-vs-latency | `python scripts/06_make_report_tables.py` | `results/paper_vs_ours.csv`, `figures/accuracy_vs_latency.png` |

## Data setup (data/ is git-ignored)
- **NYU Depth V2, Eigen test split (654 images):** `wget -O data/nyu.zip https://github.com/ultralytics/assets/releases/download/v0.0.0/nyu-depth.zip && unzip data/nyu.zip -d data/nyu` (third-party repackaging; cite it, see PROVENANCE).
- **DA-2K:** `huggingface_hub.snapshot_download(repo_id="depth-anything/DA-2K", repo_type="dataset", local_dir="data/da2k")`.
- **KITTI (optional):** needs a free account at cvlibs.net; see data/README.md for the manifest format.

## Deployment documentation checklist
- Start: `streamlit run app/streamlit_app.py`
- Dependencies: `requirements.txt`
- Input: `.jpg` / `.png` upload. Output: input next to colourised depth map (bright = near), latency in ms, model size, PNG downloads.
- Worked example + screenshot: `docs/` (add before submission)

## Licences
Depth-Anything-V2 Small is Apache-2.0; Base and Large are CC-BY-NC-4.0 (non-commercial). Fine for coursework; do not ship them in a commercial product.

## Docs
`docs/DESIGN.md` diagrams + workflow - `docs/EXPERIMENT.md` the experiment - `docs/EVAL_PROTOCOL.md` metric decisions - `PROVENANCE.md` who wrote what
