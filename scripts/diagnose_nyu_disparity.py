"""Measure near-zero and negative raw predictions on five NYU images."""
from pathlib import Path
import os
import sys

repo_root = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(repo_root / "data" / "hf-cache"))
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

import numpy as np
from PIL import Image

sys.path.insert(0, str(repo_root / "src"))
from dav2.metrics import align_scale_shift, valid_mask
from dav2.models import load_model


root = Path("data/nyu")
image_dir = next((p for p in sorted(root.rglob("val")) if p.is_dir() and p.parent.name == "images"), None)
if image_dir is None:
    raise FileNotFoundError(f"no images/val folder under {root}")
images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
if len(images) < 5:
    raise ValueError(f"expected at least five NYU images in {image_dir}, found {len(images)}")
selected = [images[i] for i in np.linspace(0, len(images) - 1, 5).round().astype(int)]
depth_dir = image_dir.parent.parent / "depth" / "val"
ground_truth = {p.stem: np.load(depth_dir / f"{p.stem}.npy") for p in selected}

print(f"Images: {', '.join(p.name for p in selected)}", flush=True)
stats = {}
alignment_rows = []
for key in ("v2-small", "v2-large"):
    model = load_model(key)
    stats[key] = []
    for path in selected:
        with Image.open(path) as image:
            disparity = model.predict(image)
        negative = float(np.mean(disparity < 0))
        near_zero = float(np.mean(disparity < 0.01))
        minimum = float(np.min(disparity))
        stats[key].append((negative, near_zero, minimum))
        print(f"{key:8s} {path.name:32s} negative={negative:.6%}  <0.01={near_zero:.6%}  min={minimum:.6g}", flush=True)
        gt_depth = ground_truth[path.stem]
        mask = valid_mask(gt_depth, 1e-3, 10.0)
        pred_disp = disparity[mask].astype(np.float64)
        gt_disp = 1.0 / gt_depth[mask].astype(np.float64)
        scale, shift = align_scale_shift(pred_disp, gt_disp)
        aligned = scale * pred_disp + shift
        ss_res = float(np.sum((aligned - gt_disp) ** 2))
        ss_tot = float(np.sum((gt_disp - gt_disp.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot
        pred_variance = float(np.var(disparity.astype(np.float64)))
        alignment_rows.append((path.stem, key, ss_res, r2, pred_variance))
    del model

for key, values in stats.items():
    print(f"{key:8s} mean across images: negative={np.mean([v[0] for v in values]):.6%}  "
          f"<0.01={np.mean([v[1] for v in values]):.6%}")
for path, small, large in zip(selected, stats["v2-small"], stats["v2-large"]):
    print(f"Large minus Small {path.name}: negative={large[0] - small[0]:+.6%}  "
          f"<0.01={large[1] - small[1]:+.6%}")

print("image | model | alignment_R2 | pred_variance")
for image, key, ss_res, r2, pred_variance in alignment_rows:
    print(f"{image} | {key} | {r2:.6f} | {pred_variance:.6f}")
