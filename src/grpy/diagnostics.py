from __future__ import annotations

import numpy as np


def local_moran_single(
    resid_local,
    w_row,
    target_pos,
    eps=1e-12,
):
    """
    Compute a simple local Moran-style diagnostic for one target position.

    Parameters
    ----------
    resid_local : array-like of shape (n_neighbors,)
        Local residual vector.
    w_row : array-like of shape (n_neighbors,)
        Row of normalized residual weights.
    target_pos : int
        Position of the target location within the local neighborhood.
    eps : float, default=1e-12
        Small positive threshold used to guard against zero variance.

    Returns
    -------
    float
        Local Moran-style diagnostic value, or np.nan when undefined.
    """
    e = np.asarray(resid_local, dtype=float).reshape(-1)
    w = np.asarray(w_row, dtype=float).reshape(-1)

    if e.size == 0 or w.size == 0:
        return np.nan
    if e.size != w.size:
        return np.nan
    if target_pos < 0 or target_pos >= e.size:
        return np.nan

    mu = float(np.mean(e))
    sd = float(np.std(e))
    if not np.isfinite(sd) or sd <= eps:
        return np.nan

    z = (e - mu) / sd
    zi = float(z[target_pos])
    return float(zi * np.sum(w * z))


def residual_weights(
    dist_m,
    h_eff_m,
    w_gr=None,
    mode="distance",
    eta=0.5,
    eps=1e-12,
):
    """
    Construct normalized downstream residual weights.

    Parameters
    ----------
    dist_m : array-like of shape (n_neighbors,)
        Distances in meters.
    h_eff_m : float
        Effective bandwidth in meters. Used in distance mode.
    w_gr : array-like of shape (n_neighbors,), optional
        GR weights used when `mode="tempered_gr"`.
    mode : {"distance", "tempered_gr"}, default="distance"
        Residual weighting mode.
    eta : float, default=0.5
        Tempering exponent used in `tempered_gr` mode.
    eps : float, default=1e-12
        Small threshold for numerical safety.

    Returns
    -------
    numpy.ndarray
        Normalized residual weights.

    Raises
    ------
    ValueError
        If inputs are invalid for the selected mode.
    """
    d = np.asarray(dist_m, dtype=float).reshape(-1)
    if d.size == 0:
        raise ValueError("dist_m must be non-empty.")

    mode = str(mode).lower()

    if mode == "distance":
        if h_eff_m is not None and np.isfinite(h_eff_m) and float(h_eff_m) > eps:
            tau = float(h_eff_m)
        else:
            tau = float(np.median(d))

        if not np.isfinite(tau) or tau <= eps:
            return np.ones_like(d, dtype=float) / d.size

        w = np.exp(-((d / tau) ** 2))

    elif mode == "tempered_gr":
        if w_gr is None:
            raise ValueError("mode='tempered_gr' requires w_gr.")

        wg = np.asarray(w_gr, dtype=float).reshape(-1)
        if wg.size != d.size:
            raise ValueError("w_gr length must match dist_m length.")
        if not np.isfinite(eta) or float(eta) <= 0.0:
            eta = 0.5
        w = np.power(np.maximum(wg, 0.0), float(eta))

    else:
        raise ValueError("mode must be one of {'distance', 'tempered_gr'}.")

    s = float(np.sum(w))
    if not np.isfinite(s) or s <= eps:
        return np.ones_like(d, dtype=float) / d.size

    return w / s