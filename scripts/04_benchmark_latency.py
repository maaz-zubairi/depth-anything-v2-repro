"""Per-model inference cost. Forward pass only, fixed 518x518 input, warm-up + sync.
Run:  python scripts/04_benchmark_latency.py            # V2 S/B/L
Record the hardware string in your report; latency numbers are meaningless without it.
"""
import argparse, gc, statistics as st
import pandas as pd, torch
from dav2.models import V2_KEYS, load_model, pick_device, sync

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=V2_KEYS)
ap.add_argument("--size", type=int, default=518)
ap.add_argument("--warmup", type=int, default=10)
ap.add_argument("--iters", type=int, default=50)
ap.add_argument("--out", default="results/latency.csv")
args = ap.parse_args()

device = pick_device()
hw = torch.cuda.get_device_name(0) if device == "cuda" else device
print("hardware:", hw)
rows = []
for key in args.models:
    m = load_model(key, device)
    x = torch.randn(1, 3, args.size, args.size, device=device)
    if device == "cuda": torch.cuda.reset_peak_memory_stats()
    import time
    for _ in range(args.warmup):
        m.forward_raw(x)
    sync(device)
    ts = []
    for _ in range(args.iters):
        t0 = time.perf_counter(); m.forward_raw(x); sync(device)
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    rows.append({"model": key, "hardware": hw, "params_m": round(m.params_m, 1),
                 "mean_ms": st.mean(ts), "median_ms": st.median(ts), "p95_ms": ts[int(0.95 * len(ts)) - 1],
                 "fps": 1000 / st.mean(ts),
                 "peak_mem_mb": (torch.cuda.max_memory_allocated() / 2**20) if device == "cuda" else None})
    print(rows[-1])
    del m, x; gc.collect()
    if device == "cuda": torch.cuda.empty_cache()
pd.DataFrame(rows).to_csv(args.out, index=False)
