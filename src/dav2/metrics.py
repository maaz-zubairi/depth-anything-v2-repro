"""Depth metrics + alignment. Pure numpy, so it is unit-testable without a GPU.

PROTOCOL DECISION (write this in your report):
  The released S/B/L checkpoints predict *relative* inverse depth. The paper's
  zero-shot benchmark table evaluates them MiDaS-style: fit a per-image scale
  and shift in disparity space by least squares against ground truth, clamp,
  convert back to depth, then compute AbsRel and delta1.
  `eval_relative` implements that. RMSE is not in the paper's relative table,
  so we report our aligned RMSE as an extra number with no paper counterpart.
  `eval_metric` is for the separate metric-depth checkpoints (optional track).
  CONFIRM the exact clamp / crop details against the official evaluation code
  and paper appendix and record any deviation in docs/EVAL_PROTOCOL.md.
"""
from __future__ import annotations

import numpy as np


def valid_mask(gt: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    # NYU/KITTI store missing depth as 0, so gt > min_depth also drops holes.
    return np.isfinite(gt) & (gt > min_depth) & (gt < max_depth)


def eigen_crop_mask(h: int, w: int) -> np.ndarray:
    """Standard NYU Eigen crop for 480x640 images (optional)."""
    m = np.zeros((h, w), dtype=bool)
    m[45:471, 41:601] = True
    return m


def depth_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """pred, gt: 1-D arrays of valid, positive depths."""
    ratio = np.maximum(pred / gt, gt / pred)
    return {
        "absrel": float(np.mean(np.abs(pred - gt) / gt)),
        "rmse": float(np.sqrt(np.mean((pred - gt) ** 2))),
        "delta1": float(np.mean(ratio < 1.25)),
    }


def align_scale_shift(pred_disp: np.ndarray, gt_disp: np.ndarray) -> tuple[float, float]:
    """Least-squares (s, t) minimising ||s * pred + t - gt||^2."""
    A = np.stack([pred_disp, np.ones_like(pred_disp)], axis=1).astype(np.float64)
    (s, t), *_ = np.linalg.lstsq(A, gt_disp.astype(np.float64), rcond=None)
    return float(s), float(t)


def eval_relative(
    pred_disp: np.ndarray,
    gt_depth: np.ndarray,
    min_depth: float = 1e-3,
    max_depth: float = 10.0,
    extra_mask: np.ndarray | None = None,
) -> dict | None:
    """Relative-depth protocol: align in disparity space, evaluate in depth space."""
    assert pred_disp.shape == gt_depth.shape, (pred_disp.shape, gt_depth.shape)
    mask = valid_mask(gt_depth, min_depth, max_depth)
    if extra_mask is not None:
        mask &= extra_mask
    if mask.sum() < 10:
        return None
    gt_d = gt_depth[mask].astype(np.float64)
    s, t = align_scale_shift(pred_disp[mask], 1.0 / gt_d)
    aligned = s * pred_disp[mask].astype(np.float64) + t
    aligned = np.clip(aligned, 1.0 / max_depth, None)  # MiDaS-style floor on disparity
    return depth_metrics(1.0 / aligned, gt_d) | {"scale": s, "shift": t}


def eval_metric(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    min_depth: float = 1e-3,
    max_depth: float = 10.0,
    extra_mask: np.ndarray | None = None,
) -> dict | None:
    """Metric-depth protocol (metric checkpoints): no alignment."""
    assert pred_depth.shape == gt_depth.shape
    mask = valid_mask(gt_depth, min_depth, max_depth)
    if extra_mask is not None:
        mask &= extra_mask
    if mask.sum() < 10:
        return None
    p = np.clip(pred_depth[mask].astype(np.float64), min_depth, max_depth)
    return depth_metrics(p, gt_depth[mask].astype(np.float64))


def score_pairs(disp: np.ndarray, pairs: list[dict]) -> list[bool]:
    """DA-2K: point1 is always the closer point; disparity is larger for closer pixels."""
    h, w = disp.shape
    out = []
    for p in pairs:
        (h1, w1), (h2, w2) = p["point1"], p["point2"]
        d1 = disp[min(max(int(h1), 0), h - 1), min(max(int(w1), 0), w - 1)]
        d2 = disp[min(max(int(h2), 0), h - 1), min(max(int(w2), 0), w - 1)]
        out.append(bool(d1 > d2))  # ties count as wrong
    return out
