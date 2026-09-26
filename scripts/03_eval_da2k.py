"""DA-2K pair-accuracy for V1 and V2 (the paper-specific experiment).
Run:  python scripts/03_eval_da2k.py --data-root data/da2k --limit 20      # sanity check
      python scripts/03_eval_da2k.py --data-root data/da2k                 # all 1000 images
Writes results/da2k_pairs.csv (one row per annotated pair per model).
"""
import argparse, gc
import pandas as pd, torch
from tqdm import tqdm
from dav2.data import DA2K
from dav2.metrics import score_pairs
from dav2.models import V1_KEYS, V2_KEYS, load_model

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", required=True)
ap.add_argument("--models", nargs="+", default=V2_KEYS + V1_KEYS)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--out", default="results/da2k_pairs.csv")
args = ap.parse_args()

ds = DA2K(args.data_root, args.limit)
print(f"DA-2K: {len(ds)} images x {args.models}")
rows = []
for key in args.models:
    model = load_model(key)
    for i in tqdm(range(len(ds)), desc=key):
        s = ds[i]
        disp = model.predict(s["image"])
        for j, ok in enumerate(score_pairs(disp, s["pairs"])):
            rows.append({"model": key, "scene": s["scene"], "image": s["id"], "pair": j, "correct": int(ok)})
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

df = pd.DataFrame(rows)
df.to_csv(args.out, index=False)
print(df.pivot_table(index="model", columns="scene", values="correct", aggfunc="mean").round(3))
print("overall:\n", df.groupby("model")["correct"].mean().round(4))
