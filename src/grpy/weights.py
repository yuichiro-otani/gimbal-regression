from __future__ import annotations

import numpy as np

from .utils import njit


@njit(cache=True, fastmath=True)
def omega_radial(dist_m, h_m):
    """
    Radial Gaussian-like decay weights exp(-(d / h)^2).
    """
    if h_m <= 0.0:
        return np.zeros(dist_m.shape[0], dtype=np.float64)
    return np.exp(-(dist_m / h_m) ** 2)

@njit(cache=True, fastmath=True)
def estimate_phi_and_r(dist_m, theta_ij, h_m, eps_phi):
    """
    Estimate bearing-based dominant direction phi and resultant length r.

    Parameters
    ----------
    dist_m : array
        Distances from target to neighbors in meters.
    theta_ij : array
        Bearing angles in radians.
    h_m : float
        Radial bandwidth in meters.
    eps_phi : float
        Threshold below which the bearing field is treated as isotropic.

    Returns
    -------
    phi : float
        Dominant bearing angle in radians, or 0 if isotropic.
    r : float
        Normalized resultant length.
    """
    w = omega_radial(dist_m, h_m)
    c = np.sum(w * np.cos(theta_ij))
    s = np.sum(w * np.sin(theta_ij))
    denom = np.sum(w)
    if denom <= 0.0:
        return 0.0, 0.0
    r = np.sqrt(c * c + s * s) / denom
    phi = np.arctan2(s, c) if r > eps_phi else 0.0
    return phi, r

@njit(cache=True, fastmath=True)
def theta_star_unweighted(z, y, eps_theta):
    """
    Compute the value-based orientation theta* from unweighted second moments of (z, y).
    """
    n = z.shape[0]
    if n <= 1:
        return 0.0, 0.0

    mz = np.mean(z)
    my = np.mean(y)

    dz = z - mz
    dy = y - my

    var_z = np.mean(dz * dz)
    var_y = np.mean(dy * dy)
    cov_zy = np.mean(dz * dy)

    g = np.abs(var_y - var_z) + np.abs(2.0 * cov_zy)
    if g <= eps_theta:
        return 0.0, g

    theta = 0.5 * np.arctan2(var_y - var_z, 2.0 * cov_zy)
    return theta, g

@njit(cache=True, fastmath=True)
def eta_from_geometry(east_m, north_m, dist_m, h_m, eps_eta, eta_max):
    """
    Compute geometry-based anisotropy ratio eta from weighted displacement moments.
    """
    w = omega_radial(dist_m, h_m)
    sw = np.sum(w)
    if sw <= 0.0:
        return 1.0

    a00 = 0.0
    a01 = 0.0
    a11 = 0.0
    for k in range(east_m.shape[0]):
        tw = w[k] / sw
        ex = east_m[k]
        no = north_m[k]
        a00 += tw * ex * ex
        a01 += tw * ex * no
        a11 += tw * no * no

    tr = a00 + a11
    disc = (a00 - a11) * (a00 - a11) + 4.0 * a01 * a01
    sdisc = np.sqrt(disc)
    lmax = 0.5 * (tr + sdisc)
    lmin = 0.5 * (tr - sdisc)

    lmin_dag = lmin if lmin > eps_eta else eps_eta
    if lmax <= 0.0:
        return 1.0

    eta_raw = np.sqrt(lmax / lmin_dag)
    if eta_raw < 1.0:
        eta_raw = 1.0
    if eta_raw > eta_max:
        eta_raw = eta_max
    return eta_raw

@njit(cache=True, fastmath=True)
def metric_elements_from_angles(phi, theta_z, eta, h_m):
    """
    Construct metric tensor elements for the anisotropic weight field.
    """
    if h_m <= 0.0:
        h_m = 1e-12
    if eta < 1.0:
        eta = 1.0

    alpha = phi + theta_z
    c = np.cos(alpha)
    s = np.sin(alpha)

    l1 = 1.0 / (h_m * h_m)
    l2 = 1.0 / (h_m * h_m * eta * eta)

    m00 = c * c * l1 + s * s * l2
    m11 = s * s * l1 + c * c * l2
    m01 = c * s * (l1 - l2)
    return m00, m01, m11, alpha

@njit(cache=True, fastmath=True)
def weights_from_metric(east_m, north_m, m00, m01, m11):
    """
    Evaluate directional weights exp(-Δ' M Δ).
    """
    n = east_m.shape[0]
    w = np.empty(n, dtype=np.float64)
    for k in range(n):
        e = east_m[k]
        no = north_m[k]
        q = m00 * e * e + 2.0 * m01 * e * no + m11 * no * no
        w[k] = np.exp(-q)
    return w

@njit(cache=True, fastmath=True)
def effective_sample_size(w):
    """
    Effective sample size of a nonnegative weight vector after normalization.
    """
    s = np.sum(w)
    if s <= 0.0:
        return 0.0

    inv = 0.0
    for k in range(w.shape[0]):
        tw = w[k] / s
        inv += tw * tw

    if inv <= 0.0:
        return 0.0
    return 1.0 / inv

@njit(cache=True, fastmath=True)
def normalize_weights_or_uniform(w_raw):
    """
    Normalize weights, or return uniform weights if the sum is nonpositive.
    """
    s = np.sum(w_raw)
    n = w_raw.shape[0]
    if s <= 0.0:
        return np.ones(n, dtype=np.float64) / n
    return w_raw / s

@njit(cache=True, fastmath=True)
def one_shot_ess_and_fallback_weights(east_m, north_m, phi, theta_z, eta, h_m, n0, n_min,):
    """
    Apply one-shot ESS correction and uniform fallback.

    Returns
    -------
    w_final : array
        Final normalized weights.
    h_eff : float
        One-shot effective bandwidth.
    neff_raw : float
        ESS before bandwidth correction.
    neff_post : float
        ESS after one-shot correction.
    used_uniform : int
        1 if uniform fallback was used, else 0.
    """
    m00, m01, m11, _ = metric_elements_from_angles(phi, theta_z, eta, h_m)
    w_raw = weights_from_metric(east_m, north_m, m00, m01, m11)
    neff_raw = effective_sample_size(w_raw)

    if neff_raw <= 0.0:
        h_eff = h_m
    else:
        h_eff = h_m * np.sqrt(n0 / neff_raw)

    m00e, m01e, m11e, _ = metric_elements_from_angles(phi, theta_z, eta, h_eff)
    w1_raw = weights_from_metric(east_m, north_m, m00e, m01e, m11e)
    w1 = normalize_weights_or_uniform(w1_raw)

    neff_post = effective_sample_size(w1)

    n = w1.shape[0]
    if neff_post < n_min:
        w_final = np.ones(n, dtype=np.float64) / n
        used_uniform = 1
    else:
        w_final = w1
        used_uniform = 0

    return w_final, h_eff, neff_raw, neff_post, used_uniform
