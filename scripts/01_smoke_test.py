"""Assignment step: load S/B/L, forward ONE image, check shape + finite values.
Run:  python scripts/01_smoke_test.py [--image path.jpg] [--models v2-small v2-base v2-large]
"""
import argparse, gc
import numpy as np, torch
from PIL import Image
from dav2.models import V2_KEYS, load_model

ap = argparse.ArgumentParser()
ap.add_argument("--image", default=None, help="any RGB image; a synthetic one is used if omitted")
ap.add_argument("--models", nargs="+", default=V2_KEYS)
args = ap.parse_args()

img = (Image.open(args.image).convert("RGB") if args.image
       else Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)))
print(f"input: {img.size[1]}x{img.size[0]} (HxW)")

for key in args.models:
    m = load_model(key)
    d = m.predict(img)
    ok_shape = d.shape == (img.size[1], img.size[0])
    ok_finite = bool(np.isfinite(d).all())
    print(f"{key:9s} params={m.params_m:6.1f}M device={m.device} out={d.shape} "
          f"shape_ok={ok_shape} finite={ok_finite} range=[{d.min():.2f}, {d.max():.2f}]")
    assert ok_shape and ok_finite, f"{key} failed the sanity check"
    del m; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
print("smoke test passed")
