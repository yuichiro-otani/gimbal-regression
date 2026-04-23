import numpy as np

from grpy.weights import (
    omega_radial,
    estimate_phi_and_r,
    theta_star_unweighted,
    eta_from_geometry,
    metric_elements_from_angles,
    weights_from_metric,
    effective_sample_size,
    normalize_weights_or_uniform,
    one_shot_ess_and_fallback_weights,
)


def test_omega_radial_is_positive_and_leq_one():
    dist = np.array([0.0, 10.0, 100.0], dtype=float)
    w = omega_radial(dist, 50.0)

    assert w.shape == dist.shape
    assert np.all(w > 0.0)
    assert np.all(w <= 1.0)
    assert np.isclose(w[0], 1.0)


def test_estimate_phi_and_r_returns_finite_values():
    dist = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    theta = np.array([0.0, 0.1, -0.1, 0.05], dtype=float)

    phi, r = estimate_phi_and_r(dist, theta, 10.0, 1e-6)

    assert np.isfinite(phi)
    assert np.isfinite(r)
    assert 0.0 <= r <= 1.0


def test_theta_star_unweighted_returns_zero_for_isotropic_case():
    z = np.array([1.0, -1.0, 1.0, -1.0], dtype=float)
    y = np.array([1.0, -1.0, 1.0, -1.0], dtype=float)

    theta, g = theta_star_unweighted(z, y, eps_theta=1e12)

    assert np.isfinite(theta)
    assert np.isfinite(g)
    assert theta == 0.0


def test_eta_from_geometry_is_at_least_one():
    east = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    north = np.array([0.1, 0.1, 0.2, 0.1], dtype=float)
    dist = np.sqrt(east**2 + north**2)

    eta = eta_from_geometry(east, north, dist, h_m=10.0, eps_eta=1e-8, eta_max=50.0)

    assert np.isfinite(eta)
    assert eta >= 1.0
    assert eta <= 50.0


def test_metric_elements_from_angles_return_finite_values():
    m00, m01, m11, alpha = metric_elements_from_angles(
        phi=0.2, theta_z=0.3, eta=2.0, h_m=100.0
    )

    assert np.isfinite(m00)
    assert np.isfinite(m01)
    assert np.isfinite(m11)
    assert np.isfinite(alpha)
    assert m00 > 0.0
    assert m11 > 0.0


def test_weights_from_metric_positive():
    east = np.array([0.0, 1.0, 2.0], dtype=float)
    north = np.array([0.0, 1.0, 2.0], dtype=float)
    m00, m01, m11, _ = metric_elements_from_angles(0.0, 0.0, 1.0, 10.0)

    w = weights_from_metric(east, north, m00, m01, m11)

    assert w.shape == east.shape
    assert np.all(w > 0.0)
    assert np.isclose(w[0], 1.0)


def test_effective_sample_size_from_raw_bounds():
    w = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)
    neff = effective_sample_size(w)

    assert np.isfinite(neff)
    assert np.isclose(neff, 4.0)


def test_normalize_or_uniform_sums_to_one():
    w = np.array([1.0, 2.0, 3.0], dtype=float)
    wn = normalize_weights_or_uniform(w)

    assert np.isclose(np.sum(wn), 1.0)
    assert np.all(wn >= 0.0)


def test_normalize_or_uniform_uniform_when_all_zero():
    w = np.array([0.0, 0.0, 0.0], dtype=float)
    wn = normalize_weights_or_uniform(w)

    assert np.isclose(np.sum(wn), 1.0)
    assert np.allclose(wn, np.array([1 / 3, 1 / 3, 1 / 3]))


def test_one_shot_ess_and_fallback_weights_output_shapes_and_ranges():
    east = np.array([0.0, 10.0, 20.0, 30.0], dtype=float)
    north = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)

    w_final, h_eff, neff_raw, neff_post, used_uniform = one_shot_ess_and_fallback_weights(
        east_m=east,
        north_m=north,
        phi=0.1,
        theta_z=0.2,
        eta=1.5,
        h_m=100.0,
        n0=3.0,
        n_min=2.0,
    )

    assert w_final.shape == east.shape
    assert np.isclose(np.sum(w_final), 1.0)
    assert np.all(w_final >= 0.0)
    assert np.isfinite(h_eff)
    assert np.isfinite(neff_raw)
    assert np.isfinite(neff_post)
    assert used_uniform in (0, 1)