import numpy as np
import pytest

from grpy.diagnostics import local_moran_single, residual_weights


def test_local_moran_single_returns_finite_value():
    resid_local = np.array([1.0, 0.5, -0.5, -1.0], dtype=float)
    w_row = np.array([0.4, 0.3, 0.2, 0.1], dtype=float)
    target_pos = 0

    out = local_moran_single(resid_local, w_row, target_pos)

    assert np.isfinite(out)


def test_local_moran_single_returns_nan_for_length_mismatch():
    resid_local = np.array([1.0, 0.5, -0.5], dtype=float)
    w_row = np.array([0.5, 0.3], dtype=float)

    out = local_moran_single(resid_local, w_row, target_pos=0)

    assert np.isnan(out)


def test_local_moran_single_returns_nan_when_std_is_zero():
    resid_local = np.array([2.0, 2.0, 2.0, 2.0], dtype=float)
    w_row = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)

    out = local_moran_single(resid_local, w_row, target_pos=0)

    assert np.isnan(out)


def test_residual_weights_distance_mode_sums_to_one():
    dist_m = np.array([0.0, 10.0, 20.0, 30.0], dtype=float)

    w = residual_weights(dist_m, h_eff_m=20.0, mode="distance")

    assert np.all(w >= 0.0)
    assert np.isclose(np.sum(w), 1.0)


def test_residual_weights_distance_mode_prefers_near_points():
    dist_m = np.array([0.0, 10.0, 20.0, 30.0], dtype=float)

    w = residual_weights(dist_m, h_eff_m=20.0, mode="distance")

    assert w[0] > w[1] > w[2] > w[3]


def test_residual_weights_distance_mode_falls_back_when_bandwidth_invalid():
    dist_m = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)

    w = residual_weights(dist_m, h_eff_m=np.nan, mode="distance")

    assert np.all(w >= 0.0)
    assert np.isclose(np.sum(w), 1.0)


def test_residual_weights_tempered_gr_sums_to_one():
    dist_m = np.array([5.0, 10.0, 15.0], dtype=float)
    w_gr = np.array([0.6, 0.3, 0.1], dtype=float)

    w = residual_weights(dist_m, h_eff_m=10.0, w_gr=w_gr, mode="tempered_gr", eta=0.5)

    assert np.all(w >= 0.0)
    assert np.isclose(np.sum(w), 1.0)


def test_residual_weights_tempered_gr_preserves_order_for_positive_eta():
    dist_m = np.array([5.0, 10.0, 15.0], dtype=float)
    w_gr = np.array([0.6, 0.3, 0.1], dtype=float)

    w = residual_weights(dist_m, h_eff_m=10.0, w_gr=w_gr, mode="tempered_gr", eta=0.5)

    assert w[0] > w[1] > w[2]


def test_residual_weights_tempered_gr_requires_w_gr():
    dist_m = np.array([5.0, 10.0, 15.0], dtype=float)

    with pytest.raises(ValueError, match="requires w_gr"):
        residual_weights(dist_m, h_eff_m=10.0, w_gr=None, mode="tempered_gr")


def test_residual_weights_tempered_gr_requires_matching_length():
    dist_m = np.array([5.0, 10.0, 15.0], dtype=float)
    w_gr = np.array([0.7, 0.3], dtype=float)

    with pytest.raises(ValueError, match="length must match"):
        residual_weights(dist_m, h_eff_m=10.0, w_gr=w_gr, mode="tempered_gr")


def test_residual_weights_invalid_mode_raises():
    dist_m = np.array([1.0, 2.0, 3.0], dtype=float)

    with pytest.raises(ValueError, match="mode must be"):
        residual_weights(dist_m, h_eff_m=10.0, mode="invalid_mode")


def test_residual_weights_tempered_gr_nonpositive_eta_falls_back():
    dist_m = np.array([5.0, 10.0, 15.0], dtype=float)
    w_gr = np.array([0.6, 0.3, 0.1], dtype=float)

    w1 = residual_weights(dist_m, h_eff_m=10.0, w_gr=w_gr, mode="tempered_gr", eta=0.0)
    w2 = residual_weights(dist_m, h_eff_m=10.0, w_gr=w_gr, mode="tempered_gr", eta=-1.0)

    assert np.isclose(np.sum(w1), 1.0)
    assert np.isclose(np.sum(w2), 1.0)
    assert np.all(w1 >= 0.0)
    assert np.all(w2 >= 0.0)