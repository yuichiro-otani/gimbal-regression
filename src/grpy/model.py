from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

from .diagnostics import local_moran_single, residual_weights
from .neighbors import knn_haversine
from .solver import solve_beta_eq46_numpy
from .utils import bearing_angles_from_deltas, tangent_plane_deltas_m
from .weights import (
    effective_sample_size,
    estimate_phi_and_r,
    eta_from_geometry,
    metric_elements_from_angles,
    one_shot_ess_and_fallback_weights,
    theta_star_unweighted,
    weights_from_metric,
)


def local_fit(
    y,
    x,
    lat,
    lon,
    ix,
    *,
    K=50,
    h_m=3000.0,
    kappa=1.0,
    gamma=1.0,
    n0=15.0,
    min_neff=4.0,
    use_ess=True,
    res_weight_mode="distance",
    res_eta=0.5,
    compute_local_moran=True,
    eps_phi=1e-3,
    eps_theta=1e-8,
    eps_eta=1e-8,
    eta_max=50.0,
    u_scale=None,
):
    """
    Fit a single local GR model at target index `ix`.

    Returns
    -------
    dict
        Dictionary containing local coefficients, diagnostics, fitted value,
        residual, and neighborhood information.
    """
    dist_m, idx = knn_haversine(lat, lon, ix, K)
    if idx.shape[0] < 3:
        return {"ix": int(ix), "error": "too_few_neighbors"}

    si_lat = float(lat[ix])
    si_lon = float(lon[ix])
    s_lat = lat[idx]
    s_lon = lon[idx]

    east_m, north_m = tangent_plane_deltas_m(si_lat, si_lon, s_lat, s_lon)
    theta_ij = bearing_angles_from_deltas(east_m, north_m)

    phi, r_phi = estimate_phi_and_r(dist_m, theta_ij, float(h_m), float(eps_phi))

    if u_scale is None:
        u_scale = float(h_m)
    u_scale = float(u_scale)
    if u_scale <= 0.0:
        u_scale = float(h_m)

    z_loc = dist_m / u_scale
    y_loc = y[idx]

    theta_z, g_ident = theta_star_unweighted(z_loc, y_loc, float(eps_theta))
    eta_i = eta_from_geometry(
        east_m, north_m, dist_m, float(h_m), float(eps_eta), float(eta_max)
    )

    if use_ess:
        w_final, h_eff, neff_raw, neff_post, used_uniform = one_shot_ess_and_fallback_weights(
            east_m,
            north_m,
            float(phi),
            float(theta_z),
            float(eta_i),
            float(h_m),
            float(n0),
            float(min_neff),
        )
    else:
        m00, m01, m11, _ = metric_elements_from_angles(
            float(phi), float(theta_z), float(eta_i), float(h_m)
        )
        w_raw = weights_from_metric(east_m, north_m, m00, m01, m11)
        s = float(np.sum(w_raw))
        w_norm = w_raw / s if s > 0.0 else np.ones_like(w_raw) / len(w_raw)
        neff_raw = float(effective_sample_size_from_raw(w_norm))
        h_eff = float(h_m)
        neff_post = float(neff_raw)

        if neff_post < float(min_neff):
            w_final = np.ones_like(w_norm) / len(w_norm)
            used_uniform = 1
        else:
            w_final = w_norm
            used_uniform = 0

    x_local = np.column_stack(
        [
            np.ones_like(z_loc),
            x[idx],
            z_loc,
        ]
    ).astype(np.float64)

    y_local = y_loc.astype(np.float64)

    beta, yhat_vec, resid_vec, r2, adjr2, rmse, condM = solve_beta_eq46_numpy(
        x_local,
        y_local,
        np.asarray(w_final, dtype=np.float64),
        float(gamma),
    )

    pos = np.where(idx == ix)[0]
    target_pos = int(pos[0]) if pos.size > 0 else 0

    yhat_at_ix = float(yhat_vec[target_pos])
    resid_at_ix = float(y_local[target_pos] - yhat_vec[target_pos])

    local_moran = np.nan
    if compute_local_moran:
        try:
            w_res_row = residual_weights(
                dist_m=dist_m.astype(np.float64),
                h_eff_m=float(h_eff),
                w_gr=np.asarray(w_final, dtype=np.float64),
                mode=res_weight_mode,
                eta=res_eta,
            )
            local_moran = local_moran_single(resid_vec, w_res_row, target_pos)
        except Exception:
            local_moran = np.nan

    _, _, _, alpha = metric_elements_from_angles(
        float(phi), float(theta_z), float(eta_i), float(h_m)
    )

    return {
        "ix": int(ix),
        "lat": si_lat,
        "lon": si_lon,
        "n": int(len(idx)),
        "h_m": float(h_m),
        "h_eff_m": float(h_eff),
        "gamma": float(gamma),
        "phi": float(phi),
        "r_phi": float(r_phi),
        "theta_z": float(theta_z),
        "g_ident": float(g_ident),
        "eta": float(eta_i),
        "alpha": float(alpha),
        "neff_raw": float(neff_raw),
        "neff_post": float(neff_post),
        "R2": float(r2) if np.isfinite(r2) else np.nan,
        "AdjR2": float(adjr2) if np.isfinite(adjr2) else np.nan,
        "RMSE": float(rmse),
        "condM_nor": float(condM) if np.isfinite(condM) else np.nan,
        "B0": float(beta[0]),
        "B1": float(beta[1]),
        "B2": float(beta[2]),
        "y": float(y[ix]),
        "yhat": yhat_at_ix,
        "x": float(x[ix]),
        "resid": resid_at_ix,
        "localMoran": float(local_moran) if np.isfinite(local_moran) else np.nan,
        "nbr_idx": idx.astype(np.int64).tolist(),
        "nbr_dist_m": dist_m.astype(np.float64).tolist(),
        "nbr_z": z_loc.astype(np.float64).tolist(),
        "w_gr": np.asarray(w_final, dtype=np.float64).tolist(),
        "uniform_flag": int(used_uniform),
        "error": None,
    }


class GimbalRegression:
    """
    Geometry-aware deterministic local regression.

    Parameters
    ----------
    K : int, default=50
        Number of nearest neighbors used in each local fit.
    h_m : float, default=3000.0
        Nominal bandwidth in meters.
    gamma : float, default=1.0
        Weighting strength in the GR normal equations.
    n0 : float, default=15.0
        Target effective sample size for one-shot ESS correction.
    min_neff : float, default=4.0
        Minimum post-correction ESS before uniform fallback is used.
    use_ess : bool, default=True
        Whether to apply one-shot ESS correction.
    """

    def __init__(
        self,
        K=50,
        h_m=3000.0,
        kappa=1.0,
        gamma=1.0,
        n0=15.0,
        min_neff=4.0,
        use_ess=True,
        res_weight_mode="distance",
        res_eta=0.5,
        compute_local_moran=True,
        eps_phi=1e-3,
        eps_theta=1e-8,
        eps_eta=1e-8,
        eta_max=50.0,
        u_scale=None,
        fail_fast=False,
    ):
        self.K = K
        self.h_m = h_m
        self.kappa = kappa
        self.gamma = gamma
        self.n0 = n0
        self.min_neff = min_neff
        self.use_ess = use_ess
        self.res_weight_mode = res_weight_mode
        self.res_eta = res_eta
        self.compute_local_moran = compute_local_moran
        self.eps_phi = eps_phi
        self.eps_theta = eps_theta
        self.eps_eta = eps_eta
        self.eta_max = eta_max
        self.u_scale = u_scale
        self.fail_fast = fail_fast
        self.is_fitted_ = False

    def _validate_inputs(self, y, x, lat, lon):
        if not (len(y) == len(x) == len(lat) == len(lon)):
            raise ValueError("y, x, lat, and lon must have the same length.")
        if len(y) == 0:
            raise ValueError("Input arrays must be non-empty.")
        if self.K < 2:
            raise ValueError("K must be at least 2.")
        if self.K > len(y):
            raise ValueError("K cannot exceed the number of observations.")
        if self.h_m <= 0:
            raise ValueError("h_m must be positive.")
        if self.gamma < 0:
            raise ValueError("gamma must be non-negative.")
        if self.n0 <= 0:
            raise ValueError("n0 must be positive.")
        if self.min_neff <= 0:
            raise ValueError("min_neff must be positive.")
        if self.eta_max < 1.0:
            raise ValueError("eta_max must be at least 1.")
        for name, arr in [("y", y), ("x", x), ("lat", lat), ("lon", lon)]:
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} contains non-finite values.")

    def fit(self, y, x, lat, lon):
        """
        Fit GR at all observed locations.

        Returns
        -------
        self
        """
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        lat = np.asarray(lat, dtype=np.float64).reshape(-1)
        lon = np.asarray(lon, dtype=np.float64).reshape(-1)

        self._validate_inputs(y, x, lat, lon)

        results = []
        errors = []

        for ix in range(len(y)):
            try:
                row = local_fit(
                    y=y,
                    x=x,
                    lat=lat,
                    lon=lon,
                    ix=ix,
                    K=self.K,
                    h_m=self.h_m,
                    kappa=self.kappa,
                    gamma=self.gamma,
                    n0=self.n0,
                    min_neff=self.min_neff,
                    use_ess=self.use_ess,
                    res_weight_mode=self.res_weight_mode,
                    res_eta=self.res_eta,
                    compute_local_moran=self.compute_local_moran,
                    eps_phi=self.eps_phi,
                    eps_theta=self.eps_theta,
                    eps_eta=self.eps_eta,
                    eta_max=self.eta_max,
                    u_scale=self.u_scale,
                )
            except Exception as exc:
                if self.fail_fast:
                    raise
                row = {"ix": int(ix), "error": repr(exc)}
                errors.append((ix, repr(exc)))

            results.append(row)

        self.results_ = pd.DataFrame(results)
        self.errors_ = pd.DataFrame(errors, columns=["ix", "error"]) if errors else pd.DataFrame(columns=["ix", "error"])
        self.n_errors_ = len(errors)

        self.y_ = y
        self.x_ = x
        self.lat_ = lat
        self.lon_ = lon
        self.is_fitted_ = True

        if self.n_errors_ > 0:
            warnings.warn(
                f"GimbalRegression.fit completed with {self.n_errors_} failed local fits. "
                "See `errors_` or the `error` column in `results_`.",
                RuntimeWarning,
            )

        return self

    def diagnostics(self):
        """
        Return a reduced diagnostics table.
        """
        self._check_is_fitted()
        cols = [
            "ix",
            "condM_nor",
            "neff_raw",
            "neff_post",
            "uniform_flag",
            "phi",
            "theta_z",
            "eta",
            "alpha",
            "h_eff_m",
            "localMoran",
            "error",
        ]
        keep = [c for c in cols if c in self.results_.columns]
        return self.results_[keep].copy()

    def summary(self):
        """
        Return a compact summary dictionary.
        """
        self._check_is_fitted()
        df = self.results_.copy()

        out = {
            "n_locations": int(len(df)),
            "n_errors": int(self.n_errors_),
            "error_rate": float(self.n_errors_ / len(df)) if len(df) else np.nan,
            "mean_condM_nor": float(df["condM_nor"].dropna().mean()) if "condM_nor" in df else np.nan,
            "median_condM_nor": float(df["condM_nor"].dropna().median()) if "condM_nor" in df else np.nan,
            "mean_neff_raw": float(df["neff_raw"].dropna().mean()) if "neff_raw" in df else np.nan,
            "mean_neff_post": float(df["neff_post"].dropna().mean()) if "neff_post" in df else np.nan,
            "uniform_fallback_rate": float(df["uniform_flag"].dropna().mean()) if "uniform_flag" in df else np.nan,
            "mean_rmse": float(df["RMSE"].dropna().mean()) if "RMSE" in df else np.nan,
            "mean_B1": float(df["B1"].dropna().mean()) if "B1" in df else np.nan,
        }
        return out

    def predict(self):
        """
        Return fitted values at the observed locations.

        Returns
        -------
        numpy.ndarray
            Fitted values corresponding to the training locations passed to `fit`.
        """
        self._check_is_fitted()
        if "yhat" not in self.results_.columns:
            raise RuntimeError("No fitted values available.")
        return self.results_["yhat"].to_numpy(dtype=float)

    def draw_map(
        self,
        column,
        title=None,
        *,
        file_path=None,
        dpi=200,
        show=True,
        cmap="viridis",
        markersize=10,
        alpha=1.0,
        legend=True,
        basemap=True,
        provider="CartoDB.Positron",
        n_ticks=6,
        vmin=None,
        vmax=None,
    ):
        """
        Plot a map of a results column.
        """
        self._check_is_fitted()
        from .plotting import draw_map, results_to_gdf

        gdf = results_to_gdf(
            self.results_,
            lon_col="lon",
            lat_col="lat",
            crs="EPSG:4326",
        )

        return draw_map(
            column=column,
            title=column if title is None else title,
            gdf=gdf,
            file_path=file_path,
            dpi=dpi,
            show=show,
            cmap=cmap,
            markersize=markersize,
            alpha=alpha,
            legend=legend,
            basemap=basemap,
            provider=provider,
            n_ticks=n_ticks,
            vmin=vmin,
            vmax=vmax,
        )

    def _check_is_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("Call fit(...) first.")