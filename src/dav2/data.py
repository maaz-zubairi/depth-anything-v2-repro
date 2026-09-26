"""Dataset readers. All return plain dicts so the eval scripts stay tiny.

NYU   : expects the Eigen test split (654 imgs) as images/val/*.jpg|png + depth/val/*.npy (metres).
KITTI : optional stretch. A manifest.txt with lines "rgb_rel_path depth_rel_path";
        depth is 16-bit PNG, metres = value / 256 (KITTI convention).
DA-2K : annotations.json (+ images) from the Hugging Face dataset depth-anything/DA-2K.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png"}
SCENES = [
    "indoor", "outdoor", "non_real", "transparent_reflective",
    "adverse_style", "aerial", "underwater", "object",
]


def subset_indices(n: int, limit: int | None) -> list[int]:
    """Evenly spaced subset (more representative than 'first N')."""
    if not limit or limit >= n:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, limit).round().astype(int).tolist()))


def _load_depth(path: Path, png_scale: float) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)
    return np.array(Image.open(path)).astype(np.float32) / png_scale


class NYUEval:
    name, min_depth, max_depth = "nyu", 1e-3, 10.0

    def __init__(self, root: str | Path, limit: int | None = None):
        root = Path(root)
        img_dir = next(
            (p for p in sorted(root.rglob("val")) if p.is_dir() and p.parent.name == "images"),
            None,
        )
        if img_dir is None:
            raise FileNotFoundError(f"no images/val folder under {root} - see data/README.md")
        depth_dir = img_dir.parent.parent / "depth" / "val"
        files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
        self.items = []
        for f in files:
            d = next((depth_dir / (f.stem + e) for e in (".npy", ".png")
                      if (depth_dir / (f.stem + e)).exists()), None)
            if d is None:
                raise FileNotFoundError(f"no depth file for {f.name} in {depth_dir}")
            self.items.append((f, d))
        self.items = [self.items[i] for i in subset_indices(len(self.items), limit)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        f, d = self.items[i]
        return {"id": f.stem, "image": Image.open(f).convert("RGB"),
                "depth": _load_depth(d, png_scale=1000.0)}


class KITTIEval:
    name, min_depth, max_depth = "kitti", 1e-3, 80.0

    def __init__(self, root: str | Path, limit: int | None = None):
        root = Path(root)
        lines = [l.split() for l in (root / "manifest.txt").read_text().splitlines() if l.strip()]
        lines = [lines[i] for i in subset_indices(len(lines), limit)]
        self.items = [(root / a, root / b) for a, b in lines]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        f, d = self.items[i]
        return {"id": f.stem, "image": Image.open(f).convert("RGB"),
                "depth": _load_depth(d, png_scale=256.0)}


def scene_of(key: str) -> str:
    k = "/" + key.replace("\\", "/")
    return next((s for s in SCENES if f"/{s}/" in k), "unknown")


class DA2K:
    def __init__(self, root: str | Path, limit: int | None = None):
        root = Path(root)
        ann_path = next(root.rglob("annotations.json"), None)
        if ann_path is None:
            raise FileNotFoundError(f"annotations.json not found under {root}")
        self.base = ann_path.parent
        ann = json.loads(ann_path.read_text())
        keys = sorted(ann)
        keys = [keys[i] for i in subset_indices(len(keys), limit)]
        by_name = None
        self.items = []
        for k in keys:
            p = next((c for c in (self.base / k, self.base.parent / k, root / k) if c.exists()), None)
            if p is None:  # fall back to a filename search
                if by_name is None:
                    by_name = {q.name: q for q in root.rglob("*") if q.suffix.lower() in IMG_EXT}
                p = by_name.get(Path(k).name)
            if p is None:
                raise FileNotFoundError(f"image for annotation key '{k}' not found under {root}")
            self.items.append((k, p, ann[k]))
        unknown = sum(scene_of(k) == "unknown" for k, _, _ in self.items)
        if unknown:
            print(f"[DA2K] WARNING: {unknown} images have no recognised scene folder in their path. "
                  f"Inspect annotations.json keys and adapt scene_of() in data.py.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        k, p, pairs = self.items[i]
        return {"id": k, "scene": scene_of(k), "image": Image.open(p).convert("RGB"), "pairs": pairs}


def get_dataset(name: str, root: str | Path, limit: int | None = None):
    return {"nyu": NYUEval, "kitti": KITTIEval, "da2k": DA2K}[name](root, limit)
