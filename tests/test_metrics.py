import numpy as np
from dav2.metrics import eval_metric, eval_relative, score_pairs, valid_mask
from dav2.data import subset_indices, scene_of


def _gt(seed=0, shape=(60, 80)):
    return np.random.default_rng(seed).uniform(0.5, 8.0, shape)


def test_relative_is_invariant_to_scale_and_shift():
    gt = _gt()
    pred = 3.7 * (1.0 / gt) + 0.21          # perfect up to an affine map in disparity space
    m = eval_relative(pred, gt, 1e-3, 10.0)
    assert m["absrel"] < 1e-6 and m["delta1"] == 1.0 and m["rmse"] < 1e-6


def test_relative_penalises_a_wrong_ordering():
    gt = _gt()
    m = eval_relative(1.0 / gt, gt)          # right ordering
    bad = eval_relative(gt, gt)              # depth used as disparity: reversed ordering
    assert bad["absrel"] > m["absrel"] + 0.1 and bad["delta1"] < m["delta1"]


def test_holes_and_out_of_range_are_ignored():
    gt = _gt(); gt[:10] = 0.0; gt[10:20] = 50.0     # missing + beyond max_depth
    mask = valid_mask(gt, 1e-3, 10.0)
    assert not mask[:20].any() and mask[20:].all()
    m = eval_metric(gt.copy(), gt)
    assert m["absrel"] == 0 and m["rmse"] == 0 and m["delta1"] == 1


def test_delta1_threshold():
    gt = np.full((20, 20), 2.0)
    assert eval_metric(np.full_like(gt, 2.4), gt)["delta1"] == 1.0   # ratio 1.2 < 1.25
    assert eval_metric(np.full_like(gt, 2.6), gt)["delta1"] == 0.0   # ratio 1.3 > 1.25


def test_score_pairs_point1_is_closer_and_ties_fail():
    disp = np.zeros((10, 10)); disp[2, 2] = 5.0; disp[7, 7] = 1.0
    pairs = [{"point1": [2, 2], "point2": [7, 7], "closer_point": "point1"},   # 5 > 1 -> correct
             {"point1": [7, 7], "point2": [2, 2], "closer_point": "point1"},   # 1 < 5 -> wrong
             {"point1": [0, 0], "point2": [0, 1], "closer_point": "point1"}]   # tie -> wrong
    assert score_pairs(disp, pairs) == [True, False, False]


def test_subset_and_scene_helpers():
    assert subset_indices(100, 10)[0] == 0 and subset_indices(100, 10)[-1] == 99
    assert len(subset_indices(100, 10)) == 10 and subset_indices(5, None) == list(range(5))
    assert scene_of("DA-2K/images/transparent_reflective/a.jpg") == "transparent_reflective"
    assert scene_of("x/y/z.jpg") == "unknown"
