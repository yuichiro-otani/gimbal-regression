"""
benchmark.py

Replication utilities for Chapter 8 benchmark experiments.

This module provides:
- cross-validation splitters
- baseline model wrappers (OLS, Local Ridge, GWR, MGWR, UK, SRF)
- GR train/test benchmarking with comparable conditioning diagnostics
- aggregation utilities for benchmark tables

The intended use is from notebooks or replication scripts, e.g.

    import grpy.benchmark as gr_benchmark

    metrics_df, preds_df, diag_df = gr_benchmark.run_benchmark_ch8(
        df,
        cv_mode="spatial",
        n_splits=5,
        block_size_m=5000,
        seed=42,
        gr_K=100,
        gr_h_m=5000.0,
        force_epsg_xy=32652,
    )
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from grpy.solver import solve_beta_eq46_numpy
from grpy.utils import bearing_angles_from_deltas, tangent_plane_deltas_m
from grpy.weights import (
    effective_sample_size,
    estimate_phi_and_r,
    eta_from_geometry,
    metric_elements_from_angles,
    one_shot_ess_and_fallback_weights,
    theta_star_unweighted,
    weights_from_metric,
)

logger = logging.getLogger(__name__)

EARTH_RADIUS_M = 6_371_000.0
EPS = 1e-12


# ---------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------
_HAS_SM = True
try:
    import statsmodels.api as sm
except Exception:  # pragma: no cover
    _HAS_SM = False
    sm = None

_HAS_MGWR = True
try:
    from mgwr.sel_bw import Sel_BW
except Exception:  # pragma: no cover
    _HAS_MGWR = False
    Sel_BW = None

_HAS_PYKRIGE = True
try:
    from pykrige.ok import OrdinaryKriging
except Exception:  # pragma: no cover
    _HAS_PYKRIGE = False
    OrdinaryKriging = None

_HAS_SKLEARN = True
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.neighbors import BallTree
except Exception:  # pragma: no cover
    _HAS_SKLEARN = False
    RandomForestRegressor = None
    LinearRegression = None
    BallTree = None

_HAS_ESDA = True
try:
    from esda import Moran
    from libpysal.weights import KNN
except Exception:  # pragma: no cover
    _HAS_ESDA = False
    Moran = None
    KNN = None

_HAS_PYPROJ = True
try:
    from pyproj import CRS, Transformer
except Exception:  # pragma: no cover
    _HAS_PYPROJ = False
    CRS = None
    Transformer = None


# ---------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class BenchmarkResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    diagnostics: pd.DataFrame


# ---------------------------------------------------------------------
# Basic validation helpers
# ---------------------------------------------------------------------
def _require_columns(df: pd.DataFrame, required: Iterable[str], df_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def _require_dependency(flag: bool, package_name: str) -> None:
    if not flag:
        raise RuntimeError(f"Optional dependency '{package_name}' is required for this operation.")


# ---------------------------------------------------------------------
# Scaling helpers
# ---------------------------------------------------------------------
def _safe_mean_std(x: np.ndarray, eps: float = EPS) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    mu = float(np.nanmean(x))
    sd = float(np.nanstd(x))
    if not np.isfinite(sd) or sd <= eps:
        sd = 1.0
    return mu, sd


def _standardize(x: np.ndarray | float, mu: float, sd: float) -> np.ndarray | float:
    return (np.asarray(x, dtype=float) - float(mu)) / float(sd)


def _scale_design_X2_X3(
    X2: np.ndarray,
    X3: np.ndarray,
    *,
    scale_x: bool = False,
    x_mu: Optional[float] = None,
    x_sd: Optional[float] = None,
    scale_z: str = "none",
    z_mu: Optional[float] = None,
    z_sd: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
    """
    Scale X2=[1, X] and X3=[1, X, z] in a controlled way.
    """
    X2 = np.asarray(X2, dtype=float)
    X3 = np.asarray(X3, dtype=float)

    X2s = X2.copy()
    X3s = X3.copy()

    if scale_x:
        if x_mu is None or x_sd is None:
            raise ValueError("scale_x=True requires x_mu and x_sd.")
        X2s[:, 1] = _standardize(X2s[:, 1], x_mu, x_sd)
        X3s[:, 1] = _standardize(X3s[:, 1], x_mu, x_sd)

    z_mode = scale_z.lower()
    if z_mode == "none":
        z_mu_use, z_sd_use = 0.0, 1.0
    elif z_mode == "local":
        z_mu_use, z_sd_use = _safe_mean_std(X3s[:, 2])
        X3s[:, 2] = _standardize(X3s[:, 2], z_mu_use, z_sd_use)
    elif z_mode == "global":
        if z_mu is None or z_sd is None:
            raise ValueError("scale_z='global' requires z_mu and z_sd.")
        z_mu_use, z_sd_use = float(z_mu), float(z_sd)
        X3s[:, 2] = _standardize(X3s[:, 2], z_mu_use, z_sd_use)
    else:
        raise ValueError("scale_z must be one of {'none', 'local', 'global'}.")

    return X2s, X3s, (z_mu_use, z_sd_use)


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return float(np.mean(np.abs(y - yhat)))


def r2(y: np.ndarray, yhat: np.ndarray, eps: float = EPS) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return np.nan if ss_tot <= eps else float(1.0 - ss_res / ss_tot)


def overlap_count(train_df: pd.DataFrame, test_df: pd.DataFrame) -> int:
    train_keys = set(zip(train_df["lat"].to_numpy(), train_df["lon"].to_numpy()))
    test_keys = set(zip(test_df["lat"].to_numpy(), test_df["lon"].to_numpy()))
    return len(train_keys & test_keys)


def morans_I_residuals_fixed_knn(test_df: pd.DataFrame, resid: np.ndarray, k: int = 8) -> Tuple[float, float]:
    if not _HAS_ESDA:
        return np.nan, np.nan
    if len(test_df) <= k + 2:
        return np.nan, np.nan

    coords = np.column_stack([
        test_df["lon"].to_numpy(dtype=float),
        test_df["lat"].to_numpy(dtype=float),
    ])
    w = KNN.from_array(coords, k=k)
    w.transform = "R"
    mi = Moran(np.asarray(resid, dtype=float), w)
    return float(mi.I), float(mi.p_sim)


# ---------------------------------------------------------------------
# CV splitters
# ---------------------------------------------------------------------
def random_kfold_indices(n: int, n_splits: int = 5, seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, n_splits)

    splits = []
    for f in folds:
        test_idx = np.sort(f)
        train_idx = np.sort(np.setdiff1d(idx, test_idx, assume_unique=False))
        splits.append((train_idx, test_idx))
    return splits


def spatial_block_kfold_indices(
    df: pd.DataFrame,
    n_splits: int = 5,
    block_size_m: float = 5000.0,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    lat = df["lat"].to_numpy(dtype=float)
    lon = df["lon"].to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    lat0 = float(np.mean(lat))

    m_per_deg_lat = 111_000.0
    m_per_deg_lon = 111_000.0 * np.cos(np.radians(lat0))

    dlat = block_size_m / m_per_deg_lat
    dlon = block_size_m / (m_per_deg_lon if m_per_deg_lon > EPS else 111_000.0)

    lat_min = float(np.min(lat))
    lon_min = float(np.min(lon))

    by = np.floor((lat - lat_min) / dlat).astype(int)
    bx = np.floor((lon - lon_min) / dlon).astype(int)
    block_id = bx + 10_000 * by

    blocks = np.unique(block_id)
    rng.shuffle(blocks)
    folds = np.array_split(blocks, n_splits)

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for fblocks in folds:
        test_mask = np.isin(block_id, fblocks)
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(~test_mask)[0]
        splits.append((train_idx, test_idx))
    return splits


def keep_nonempty_folds(splits: Sequence[Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[np.ndarray, np.ndarray]]:
    return [(tr, te) for tr, te in splits if len(tr) > 0 and len(te) > 0]


# ---------------------------------------------------------------------
# Coordinate preparation
# ---------------------------------------------------------------------
def add_xy_m(
    df: pd.DataFrame,
    lon_col: str = "lon",
    lat_col: str = "lat",
    force_epsg: Optional[int] = None,
) -> pd.DataFrame:
    """
    Add projected x_m, y_m coordinates in meters.

    If pyproj is unavailable, fall back to a local planar approximation.
    """
    if "x_m" in df.columns and "y_m" in df.columns:
        return df

    out = df.copy()
    lon = out[lon_col].to_numpy(dtype=float)
    lat = out[lat_col].to_numpy(dtype=float)

    if _HAS_PYPROJ:
        if force_epsg is not None:
            crs_dst = CRS.from_epsg(int(force_epsg))
        else:
            lat0 = float(np.mean(lat))
            lon0 = float(np.mean(lon))
            zone = int((lon0 + 180) // 6) + 1
            epsg = 32600 + zone if lat0 >= 0 else 32700 + zone
            crs_dst = CRS.from_epsg(epsg)

        tf = Transformer.from_crs("EPSG:4326", crs_dst, always_xy=True)
        x_m, y_m = tf.transform(lon, lat)
        out["x_m"] = x_m
        out["y_m"] = y_m
        out["_epsg_xy"] = int(crs_dst.to_epsg())
        return out

    lat0 = float(np.mean(lat))
    m_per_deg_lat = 111_000.0
    m_per_deg_lon = 111_000.0 * np.cos(np.radians(lat0))
    out["x_m"] = (lon - lon.min()) * m_per_deg_lon
    out["y_m"] = (lat - lat.min()) * m_per_deg_lat
    out["_epsg_xy"] = -1
    return out


# ---------------------------------------------------------------------
# Haversine KNN for GR prediction
# ---------------------------------------------------------------------
def build_balltree(train_df: pd.DataFrame) -> Any:
    _require_dependency(_HAS_SKLEARN, "scikit-learn")
    lat = np.radians(train_df["lat"].to_numpy(dtype=float))
    lon = np.radians(train_df["lon"].to_numpy(dtype=float))
    coords = np.column_stack([lat, lon])
    return BallTree(coords, metric="haversine")


def query_knn_balltree(tree: Any, lat_deg: float, lon_deg: float, k: int) -> Tuple[np.ndarray, np.ndarray]:
    q = np.array([[np.radians(lat_deg), np.radians(lon_deg)]], dtype=float)
    dist_rad, idx = tree.query(q, k=k)
    dist_m = dist_rad.reshape(-1) * EARTH_RADIUS_M
    idx = idx.reshape(-1).astype(int)
    return dist_m, idx


# ---------------------------------------------------------------------
# Linear algebra helpers
# ---------------------------------------------------------------------
def cond_spd(A: np.ndarray, eps: float = EPS) -> float:
    A = np.asarray(A, dtype=float)
    A = 0.5 * (A + A.T)
    vals = np.linalg.eigvalsh(A)
    lam_min = float(np.min(vals))
    lam_max = float(np.max(vals))
    if lam_max <= eps or lam_min <= eps:
        return np.inf
    return lam_max / lam_min


def cond_ridge_spd(A: np.ndarray, lam: float, eps: float = EPS) -> float:
    A = np.asarray(A, dtype=float)
    A = 0.5 * (A + A.T)
    p = A.shape[0]
    return cond_spd(A + float(lam) * np.eye(p), eps=eps)


def gram(X: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if w is None:
        return X.T @ X
    w = np.asarray(w, dtype=float).reshape(-1)
    return X.T @ (w[:, None] * X)


# ---------------------------------------------------------------------
# Downstream residual KNN
# ---------------------------------------------------------------------
def _residual_weights_post(
    dist_m: np.ndarray,
    h_eff_m: float,
    w_gr: Optional[np.ndarray] = None,
    mode: str = "distance",
    eta: float = 0.5,
    eps: float = EPS,
) -> np.ndarray:
    d = np.asarray(dist_m, dtype=float).reshape(-1)

    if mode == "distance":
        tau = float(h_eff_m) if np.isfinite(h_eff_m) and h_eff_m > 0 else float(np.median(d))
        if tau <= eps:
            return np.ones_like(d) / len(d)
        w = np.exp(-((d / tau) ** 2))
    elif mode == "tempered_gr":
        if w_gr is None:
            raise ValueError("mode='tempered_gr' requires w_gr.")
        wg = np.asarray(w_gr, dtype=float).reshape(-1)
        if len(wg) != len(d):
            raise ValueError("w_gr length must match dist_m length.")
        eta = float(eta) if eta > 0 else 0.5
        w = np.power(np.maximum(wg, 0.0), eta)
    else:
        raise ValueError("mode must be 'distance' or 'tempered_gr'.")

    s = float(np.sum(w))
    return np.ones_like(d) / len(d) if s <= eps else (w / s)


def apply_residual_knn(local_results: pd.DataFrame, mode: str = "distance", eta: float = 0.5) -> pd.DataFrame:
    df = local_results.copy()
    required = ["ix", "yhat", "h_eff_m", "nbr_idx", "nbr_dist_m"]
    _require_columns(df, required, "local_results")

    if "resid" not in df.columns:
        if "y" not in df.columns:
            raise ValueError("apply_residual_knn requires 'resid' or both 'y' and 'yhat'.")
        df["resid"] = df["y"] - df["yhat"]

    n = int(df["ix"].max()) + 1
    resid_arr = np.full(n, np.nan, dtype=float)

    for _, row in df.iterrows():
        if pd.isna(row.get("resid", np.nan)):
            continue
        resid_arr[int(row["ix"])] = float(row["resid"])

    rhat_list: List[float] = []
    yhat_corr_list: List[float] = []

    for _, row in df.iterrows():
        idx = np.asarray(row["nbr_idx"], dtype=int)
        dist = np.asarray(row["nbr_dist_m"], dtype=float)
        h_eff = float(row["h_eff_m"])

        if mode == "distance":
            w_res = _residual_weights_post(dist, h_eff, w_gr=None, mode="distance")
        elif mode == "tempered_gr":
            w_gr = np.asarray(row["w_gr"], dtype=float)
            w_res = _residual_weights_post(dist, h_eff, w_gr=w_gr, mode="tempered_gr", eta=eta)
        else:
            raise ValueError("mode must be 'distance' or 'tempered_gr'.")

        r_nei = resid_arr[idx]
        mask = np.isfinite(r_nei)

        if not np.any(mask):
            rhat = 0.0
        else:
            w_use = w_res[mask]
            s = float(np.sum(w_use))
            w_use = np.ones_like(w_use) / len(w_use) if s <= EPS else (w_use / s)
            rhat = float(np.sum(w_use * r_nei[mask]))

        rhat_list.append(rhat)
        yhat_corr_list.append(float(row["yhat"]) + rhat)

    df["rhat"] = rhat_list
    df["yhat_corr"] = yhat_corr_list
    return df


# ---------------------------------------------------------------------
# Optional downstream global calibration
# ---------------------------------------------------------------------
def fit_global(local_results: pd.DataFrame, use_corrected: bool = False) -> Any:
    _require_dependency(_HAS_SM, "statsmodels")
    col = "yhat_corr" if use_corrected else "yhat"
    use = local_results.dropna(subset=["y", col]).copy()
    X = sm.add_constant(use[[col]])
    y = use["y"]
    return sm.OLS(y, X).fit()


# ---------------------------------------------------------------------
# GR fit on training set
# ---------------------------------------------------------------------
def fit_train_local_and_residual_knn(
    train_df: pd.DataFrame,
    *,
    K: int = 30,
    h_m: float = 2000.0,
    kappa: float = 2.0,
    gamma: float = 1.0,
    n0: float = 20.0,
    use_ess: bool = True,
    min_neff: float = 4.0,
    res_mode: str = "distance",
    res_eta: float = 0.5,
    compute_local_moran: bool = False,
    do_residual_knn: bool = True,
    do_global_calibration: bool = True,
    eps_phi: float = 1e-3,
    eps_theta: float = 1e-8,
    eps_eta: float = 1e-8,
    eta_max: float = 50.0,
    u_scale: Optional[float] = None,
) -> Tuple[pd.DataFrame, Any, Any]:
    """
    Fit GR on all training locations using grpy.GimbalRegression, then optionally
    apply downstream residual-KNN and global calibration.
    """
    _require_columns(train_df, {"lat", "lon", "X", "Y"}, "train_df")

    from grpy import GimbalRegression

    fit_kwargs: Dict[str, Any] = {
        "K": K,
        "h_m": h_m,
        "gamma": gamma,
        "n0": n0,
        "use_ess": use_ess,
        "res_weight_mode": res_mode,
        "compute_local_moran": compute_local_moran,
    }

    import inspect

    optional_model_kwargs = {
        "kappa": kappa,
        "min_neff": min_neff,
        "res_eta": res_eta,
        "eps_phi": eps_phi,
        "eps_theta": eps_theta,
        "eps_eta": eps_eta,
        "eta_max": eta_max,
        "u_scale": u_scale,
    }

    accepted = set(inspect.signature(GimbalRegression.__init__).parameters.keys())
    for kk, vv in optional_model_kwargs.items():
        if kk in accepted and vv is not None:
            fit_kwargs[kk] = vv

    model = GimbalRegression(**fit_kwargs).fit(
        y=train_df["Y"].to_numpy(dtype=np.float64),
        x=train_df["X"].to_numpy(dtype=np.float64),
        lat=train_df["lat"].to_numpy(dtype=np.float64),
        lon=train_df["lon"].to_numpy(dtype=np.float64),
    )

    train_local = model.results_.copy()

    if "ix" not in train_local.columns:
        train_local["ix"] = np.arange(len(train_local), dtype=int)
    if "y" not in train_local.columns:
        train_local["y"] = train_df["Y"].to_numpy(dtype=float)
    if "x" not in train_local.columns:
        train_local["x"] = train_df["X"].to_numpy(dtype=float)
    if "lat" not in train_local.columns:
        train_local["lat"] = train_df["lat"].to_numpy(dtype=float)
    if "lon" not in train_local.columns:
        train_local["lon"] = train_df["lon"].to_numpy(dtype=float)
    if "resid" not in train_local.columns and {"y", "yhat"}.issubset(train_local.columns):
        train_local["resid"] = train_local["y"] - train_local["yhat"]

    train_local2 = train_local
    if do_residual_knn:
        train_local2 = apply_residual_knn(train_local2, mode=res_mode, eta=res_eta)

    g_uncorr = None
    g_corr = None
    if do_global_calibration:
        try:
            g_uncorr = fit_global(train_local2, use_corrected=False)
        except Exception as exc:
            logger.warning("Global calibration (uncorrected) failed: %s", exc)
        if do_residual_knn:
            try:
                g_corr = fit_global(train_local2, use_corrected=True)
            except Exception as exc:
                logger.warning("Global calibration (corrected) failed: %s", exc)

    return train_local2, g_uncorr, g_corr


# ---------------------------------------------------------------------
# GR test prediction
# ---------------------------------------------------------------------
def predict_test_gr(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    train_local: pd.DataFrame,
    K: int = 30,
    h_m: float = 2000.0,
    gamma: float = 1.0,
    n0: float = 20.0,
    min_neff: float = 4.0,
    use_ess: bool = True,
    eps_phi: float = 1e-3,
    eps_theta: float = 1e-8,
    eps_eta: float = 1e-8,
    eta_max: float = 50.0,
    u_scale: Optional[float] = None,
    scale_x: bool = False,
    x_mu: Optional[float] = None,
    x_sd: Optional[float] = None,
    scale_z: str = "none",
    z_mu: Optional[float] = None,
    z_sd: Optional[float] = None,
    use_residual_knn: bool = False,
    res_mode: str = "distance",
    res_eta: float = 0.5,
    global_model_uncorr: Any = None,
    global_model_corr: Any = None,
) -> pd.DataFrame:
    _require_columns(train_df, {"lat", "lon", "X", "Y"}, "train_df")
    _require_columns(test_df, {"lat", "lon", "X"}, "test_df")

    tlat = train_df["lat"].to_numpy(dtype=float)
    tlon = train_df["lon"].to_numpy(dtype=float)
    tX = train_df["X"].to_numpy(dtype=float)
    tY = train_df["Y"].to_numpy(dtype=float)
    ntr = len(train_df)

    plat = test_df["lat"].to_numpy(dtype=float)
    plon = test_df["lon"].to_numpy(dtype=float)
    pX = test_df["X"].to_numpy(dtype=float)
    pY = test_df["Y"].to_numpy(dtype=float) if "Y" in test_df.columns else None

    resid_train = np.full(ntr, np.nan, dtype=float)
    if "ix" in train_local.columns and "resid" in train_local.columns:
        for _, row in train_local.iterrows():
            j = int(row["ix"])
            if 0 <= j < ntr and pd.notna(row["resid"]):
                resid_train[j] = float(row["resid"])
    else:
        use_residual_knn = False

    tree = build_balltree(train_df)
    u_scale = float(h_m) if u_scale is None else float(u_scale)
    u_scale = u_scale if u_scale > 0 else float(h_m)

    rows: List[Dict[str, Any]] = []

    for i in range(len(test_df)):
        lat0 = float(plat[i])
        lon0 = float(plon[i])
        x0 = float(pX[i])
        y0 = float(pY[i]) if pY is not None else None

        dist_m, idx = query_knn_balltree(tree, lat0, lon0, int(K))
        S_lat = tlat[idx]
        S_lon = tlon[idx]
        y_loc = tY[idx]

        east_m, north_m = tangent_plane_deltas_m(lat0, lon0, S_lat, S_lon)
        theta_ij = bearing_angles_from_deltas(east_m, north_m)

        phi, r_phi = estimate_phi_and_r(
            dist_m.astype(np.float64),
            theta_ij.astype(np.float64),
            float(h_m),
            float(eps_phi),
        )

        z_loc = dist_m / float(u_scale)
        theta_z, g_ident = theta_star_unweighted(
            z_loc.astype(np.float64),
            y_loc.astype(np.float64),
            float(eps_theta),
        )

        eta_i = eta_from_geometry(
            east_m.astype(np.float64),
            north_m.astype(np.float64),
            dist_m.astype(np.float64),
            float(h_m),
            float(eps_eta),
            float(eta_max),
        )

        if use_ess:
            w_final, h_eff, neff_raw, neff_post, used_uniform = one_shot_ess_and_fallback_weights(
                east_m.astype(np.float64),
                north_m.astype(np.float64),
                float(phi),
                float(theta_z),
                float(eta_i),
                float(h_m),
                float(n0),
                float(min_neff),
            )
        else:
            m00, m01, m11, _ = metric_elements_from_angles(
                float(phi),
                float(theta_z),
                float(eta_i),
                float(h_m),
            )
            w_raw = weights_from_metric(
                east_m.astype(np.float64),
                north_m.astype(np.float64),
                m00,
                m01,
                m11,
            )
            s = float(np.sum(w_raw))
            w_norm = w_raw / s if s > 0.0 else np.ones_like(w_raw) / len(w_raw)
            neff_raw = float(effective_sample_size_from_raw(w_norm))
            h_eff = float(h_m)
            neff_post = neff_raw
            if neff_post < float(min_neff):
                w_final = np.ones_like(w_norm) / len(w_norm)
                used_uniform = 1
            else:
                w_final = w_norm
                used_uniform = 0

        w = np.asarray(w_final, dtype=float).reshape(-1)

        X3 = np.column_stack([np.ones_like(z_loc), tX[idx], z_loc]).astype(float)
        X2 = X3[:, :2]

        X2s, X3s, (z_mu_use, z_sd_use) = _scale_design_X2_X3(
            X2,
            X3,
            scale_x=scale_x,
            x_mu=x_mu,
            x_sd=x_sd,
            scale_z=scale_z,
            z_mu=z_mu,
            z_sd=z_sd,
        )

        beta, *_ = solve_beta_eq46_numpy(X3s, y_loc.astype(float), w, float(gamma))
        beta = np.asarray(beta, dtype=float).reshape(-1)

        XtWX2 = gram(X2s, w=w)
        XtWX3 = gram(X3s, w=w)
        condWLS2 = cond_spd(XtWX2)
        condWLS3 = cond_spd(XtWX3)

        G3 = gram(X3s)
        Gw3 = gram(X3s, w=w)
        M3 = G3 + 2.0 * float(gamma) * Gw3
        condM3_nor = cond_spd(M3)

        x0_s = _standardize(x0, x_mu, x_sd) if scale_x else x0
        z0_s = (0.0 - z_mu_use) / z_sd_use if scale_z.lower() != "none" else 0.0
        x_pred = np.array([1.0, x0_s, z0_s], dtype=float)

        yhat = float(x_pred @ beta)
        rhat = 0.0
        yhat_corr = yhat

        if use_residual_knn:
            if res_mode == "distance":
                w_res = _residual_weights_post(dist_m, float(h_eff), w_gr=None, mode="distance")
            elif res_mode == "tempered_gr":
                w_res = _residual_weights_post(dist_m, float(h_eff), w_gr=w, mode="tempered_gr", eta=res_eta)
            else:
                raise ValueError("res_mode must be 'distance' or 'tempered_gr'.")

            r_nei = resid_train[idx]
            mask = np.isfinite(r_nei)
            if np.any(mask):
                w_use = np.asarray(w_res, dtype=float)[mask]
                s = float(np.sum(w_use))
                w_use = np.ones_like(w_use) / len(w_use) if s <= EPS else (w_use / s)
                rhat = float(np.sum(w_use * r_nei[mask]))
            yhat_corr = yhat + rhat

        row = {
            "lat": lat0,
            "lon": lon0,
            "yhat": yhat,
            "rhat": rhat,
            "yhat_corr": yhat_corr,
            "condWLS2": float(condWLS2) if np.isfinite(condWLS2) else np.nan,
            "condWLS3": float(condWLS3) if np.isfinite(condWLS3) else np.nan,
            "condM3_nor": float(condM3_nor) if np.isfinite(condM3_nor) else np.nan,
            "uniform_flag": int(used_uniform),
            "neff_post": float(neff_post),
            "phi": float(phi),
            "r_phi": float(r_phi),
            "theta_z": float(theta_z),
            "g_ident": float(g_ident),
            "eta": float(eta_i),
            "neff_raw": float(neff_raw),
            "h_eff_m": float(h_eff),
        }

        if y0 is not None:
            row["y"] = float(y0)
            row["resid"] = float(y0 - yhat)
            row["resid_corr"] = float(y0 - yhat_corr)

        rows.append(row)

    return pd.DataFrame(rows)


def fit_predict_GR_ch8(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    K: int = 30,
    h_m: float = 2000.0,
    gamma: float = 1.0,
    n0: float = 20.0,
    use_ess: bool = True,
    min_neff: float = 4.0,
    eps_phi: float = 1e-3,
    eps_theta: float = 1e-8,
    eps_eta: float = 1e-8,
    eta_max: float = 50.0,
    u_scale: Optional[float] = None,
    scale_x: bool = False,
    x_mu: Optional[float] = None,
    x_sd: Optional[float] = None,
    scale_z: str = "none",
    z_mu: Optional[float] = None,
    z_sd: Optional[float] = None,
    use_residual_knn: bool = False,
    res_mode: str = "distance",
    res_eta: float = 0.5,
    use_global_calibration: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    train_local2, g_uncorr, g_corr = fit_train_local_and_residual_knn(
        train_df,
        K=K,
        h_m=h_m,
        gamma=gamma,
        n0=n0,
        use_ess=use_ess,
        min_neff=min_neff,
        res_mode=res_mode,
        res_eta=res_eta,
        compute_local_moran=False,
        do_residual_knn=True,
        do_global_calibration=use_global_calibration,
        eps_phi=eps_phi,
        eps_theta=eps_theta,
        eps_eta=eps_eta,
        eta_max=eta_max,
        u_scale=u_scale,
    )

    pred_df = predict_test_gr(
        train_df,
        test_df,
        train_local=train_local2,
        K=K,
        h_m=h_m,
        gamma=gamma,
        n0=n0,
        min_neff=min_neff,
        use_ess=use_ess,
        eps_phi=eps_phi,
        eps_theta=eps_theta,
        eps_eta=eps_eta,
        eta_max=eta_max,
        u_scale=u_scale,
        scale_x=scale_x,
        x_mu=x_mu,
        x_sd=x_sd,
        scale_z=scale_z,
        z_mu=z_mu,
        z_sd=z_sd,
        use_residual_knn=use_residual_knn,
        res_mode=res_mode,
        res_eta=res_eta,
        global_model_uncorr=g_uncorr if use_global_calibration else None,
        global_model_corr=g_corr if use_global_calibration else None,
    )

    yhat = pred_df["yhat_corr"].to_numpy(dtype=float) if use_residual_knn else pred_df["yhat"].to_numpy(dtype=float)
    info = {
        "pred_diag": pred_df[["condWLS2", "condWLS3", "condM3_nor", "uniform_flag", "neff_post"]].copy()
    }
    return yhat, info



# ---------------------------------------------------------------------
# Baseline wrappers
# ---------------------------------------------------------------------
def fit_predict_OLS(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
    _require_dependency(_HAS_SM, "statsmodels")
    Xtr = sm.add_constant(train_df["X"].to_numpy(dtype=float))
    ytr = train_df["Y"].to_numpy(dtype=float)
    model = sm.OLS(ytr, Xtr).fit()

    Xte = sm.add_constant(test_df["X"].to_numpy(dtype=float))
    yhat = model.predict(Xte)
    return np.asarray(yhat, dtype=float), {"pred_diag": None}


def fit_predict_SRF(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_estimators: int = 300,
    max_depth: int = 20,
    random_state: int = 42,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    _require_dependency(_HAS_SKLEARN, "scikit-learn")

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1,
    )
    Xtr = np.column_stack([train_df["X"], train_df["lon"], train_df["lat"]]).astype(float)
    ytr = train_df["Y"].to_numpy(dtype=float)
    rf.fit(Xtr, ytr)

    Xte = np.column_stack([test_df["X"], test_df["lon"], test_df["lat"]]).astype(float)
    yhat = rf.predict(Xte)
    return np.asarray(yhat, dtype=float), {"pred_diag": None}


def fit_predict_UK(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    variogram_model: str = "spherical",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    _require_dependency(_HAS_PYKRIGE and _HAS_SKLEARN, "pykrige + scikit-learn")

    lr = LinearRegression()
    Xtr = train_df["X"].to_numpy(dtype=float).reshape(-1, 1)
    ytr = train_df["Y"].to_numpy(dtype=float)
    lr.fit(Xtr, ytr)

    trend_tr = lr.predict(Xtr)
    resid_tr = ytr - trend_tr

    ok = OrdinaryKriging(
        train_df["lon"].to_numpy(dtype=float),
        train_df["lat"].to_numpy(dtype=float),
        resid_tr,
        variogram_model=variogram_model,
        verbose=False,
        enable_plotting=False,
    )

    Xte = test_df["X"].to_numpy(dtype=float).reshape(-1, 1)
    trend_te = lr.predict(Xte)
    krig_res_te, _ = ok.execute(
        "points",
        test_df["lon"].to_numpy(dtype=float),
        test_df["lat"].to_numpy(dtype=float),
    )
    yhat = trend_te + np.asarray(krig_res_te).reshape(-1)
    return np.asarray(yhat, dtype=float), {"pred_diag": None}


# ---------------------------------------------------------------------
# Local-regression baselines
# ---------------------------------------------------------------------
def _pairwise_dist(coords_tr: np.ndarray, coord_te: np.ndarray) -> np.ndarray:
    dx = coords_tr[:, 0] - coord_te[0]
    dy = coords_tr[:, 1] - coord_te[1]
    return np.sqrt(dx * dx + dy * dy)


def _bisquare_weights(d: np.ndarray, bw: float, *, fixed: bool = False, eps: float = EPS) -> np.ndarray:
    d = np.asarray(d, dtype=float).reshape(-1)

    if fixed:
        dmax = float(bw)
    else:
        k = int(bw)
        k = max(2, min(k, len(d)))
        dmax = float(np.sort(d)[k - 1])

    if dmax <= eps:
        return np.ones_like(d) / len(d)

    u = d / dmax
    w = np.zeros_like(d)
    mask = u < 1.0
    w[mask] = (1.0 - u[mask] ** 2) ** 2

    s = float(np.sum(w))
    return np.ones_like(d) / len(d) if s <= eps else (w / s)


def _neff_from_normalized_weights(w: np.ndarray, eps: float = EPS) -> float:
    w = np.asarray(w, dtype=float).reshape(-1)
    ss = float(np.sum(w * w))
    return np.nan if ss <= eps else float(1.0 / ss)


def _prepare_local_regression_design(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    reproduce_rank_deficiency: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    coords_tr = train_df[["x_m", "y_m"]].to_numpy(dtype=float)
    coords_te = test_df[["x_m", "y_m"]].to_numpy(dtype=float)
    ytr = train_df["Y"].to_numpy(dtype=float).reshape(-1, 1)

    Xtr_raw = train_df["X"].to_numpy(dtype=float).reshape(-1, 1)
    Xte_raw = test_df["X"].to_numpy(dtype=float).reshape(-1, 1)

    if not reproduce_rank_deficiency:
        mu = Xtr_raw.mean(axis=0)
        sd = Xtr_raw.std(axis=0)
        sd[sd == 0] = 1.0
        Xtr_raw = (Xtr_raw - mu) / sd
        Xte_raw = (Xte_raw - mu) / sd
        bw_min = 20
    else:
        bw_min = 10

    Xtr = np.hstack([np.ones((Xtr_raw.shape[0], 1)), Xtr_raw])
    Xte = np.hstack([np.ones((Xte_raw.shape[0], 1)), Xte_raw])

    return coords_tr, coords_te, ytr, Xtr_raw, Xte_raw, Xtr, Xte, bw_min


def _gwr_predict_numpy_with_diag(
    coords_tr: np.ndarray,
    ytr: np.ndarray,
    Xtr: np.ndarray,
    coords_te: np.ndarray,
    Xte: np.ndarray,
    *,
    bw: float,
    fixed: bool = False,
    u_scale_m: float = 2000.0,
    scale_x: bool = False,
    x_mu: Optional[float] = None,
    x_sd: Optional[float] = None,
    scale_z: str = "none",
) -> Tuple[np.ndarray, pd.DataFrame]:
    m = coords_te.shape[0]
    yhat = np.empty(m, dtype=float)
    condWLS2_list = np.empty(m, dtype=float)
    condWLS3_list = np.empty(m, dtype=float)
    neff_list = np.empty(m, dtype=float)

    for t in range(m):
        d = _pairwise_dist(coords_tr, coords_te[t])
        w = _bisquare_weights(d, bw, fixed=fixed)
        neff_list[t] = _neff_from_normalized_weights(w)

        z_loc = d / float(u_scale_m)
        X3 = np.column_stack([Xtr[:, 0], Xtr[:, 1], z_loc]).astype(float)
        X2 = Xtr

        X2s, X3s, _ = _scale_design_X2_X3(
            X2,
            X3,
            scale_x=scale_x,
            x_mu=x_mu,
            x_sd=x_sd,
            scale_z=scale_z,
        )

        XtWX2 = X2s.T @ (w[:, None] * X2s)
        XtWy = X2s.T @ (w * ytr.reshape(-1))
        XtWX3 = X3s.T @ (w[:, None] * X3s)

        condWLS2_list[t] = cond_spd(XtWX2)
        condWLS3_list[t] = cond_spd(XtWX3)

        try:
            beta = np.linalg.solve(XtWX2, XtWy)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(XtWX2) @ XtWy

        x_te = float(Xte[t, 1])
        x_te_s = _standardize(x_te, x_mu, x_sd) if scale_x else x_te
        yhat[t] = float(np.array([1.0, x_te_s]) @ beta)

    pred_diag = pd.DataFrame({
        "condWLS2": condWLS2_list,
        "condWLS3": condWLS3_list,
        "condM3_nor": np.full(m, np.nan),
        "uniform_flag": np.zeros(m, dtype=int),
        "neff_post": neff_list,
    })
    return yhat, pred_diag


def _mgwr_predict_numpy_pointwise_with_diag(
    coords_tr: np.ndarray,
    ytr: np.ndarray,
    Xtr: np.ndarray,
    coords_te: np.ndarray,
    Xte: np.ndarray,
    *,
    bws: Sequence[float],
    fixed: bool = False,
    u_scale_m: float = 2000.0,
    scale_x: bool = False,
    x_mu: Optional[float] = None,
    x_sd: Optional[float] = None,
    scale_z: str = "none",
    max_iter: int = 50,
    tol: float = 1e-8,
    eps: float = EPS,
) -> Tuple[np.ndarray, pd.DataFrame]:
    bws = list(bws)
    p = Xtr.shape[1]
    if len(bws) != p:
        raise ValueError(f"MGWR expected {p} bandwidths, got {len(bws)}.")

    m = coords_te.shape[0]
    yhat = np.empty(m, dtype=float)
    condWLS2_list = np.empty(m, dtype=float)
    condWLS3_list = np.empty(m, dtype=float)
    neff_list = np.empty(m, dtype=float)

    for t in range(m):
        d = _pairwise_dist(coords_tr, coords_te[t])

        Wj = []
        neff_j = []
        for j in range(p):
            wj = _bisquare_weights(d, bws[j], fixed=fixed)
            Wj.append(wj)
            neff_j.append(_neff_from_normalized_weights(wj))
        neff_list[t] = float(np.nanmean(neff_j))

        w_bar = np.mean(np.column_stack(Wj), axis=1)
        s = float(np.sum(w_bar))
        w_bar = np.ones_like(w_bar) / len(w_bar) if s <= eps else (w_bar / s)

        z_loc = d / float(u_scale_m)
        X3 = np.column_stack([Xtr[:, 0], Xtr[:, 1], z_loc]).astype(float)
        X2 = Xtr

        X2s, X3s, _ = _scale_design_X2_X3(
            X2,
            X3,
            scale_x=scale_x,
            x_mu=x_mu,
            x_sd=x_sd,
            scale_z=scale_z,
        )

        XtWX2 = X2s.T @ (w_bar[:, None] * X2s)
        XtWX3 = X3s.T @ (w_bar[:, None] * X3s)

        condWLS2_list[t] = cond_spd(XtWX2)
        condWLS3_list[t] = cond_spd(XtWX3)

        beta = np.zeros(p, dtype=float)
        fitted = X2s @ beta

        for _ in range(max_iter):
            beta_old = beta.copy()
            for j in range(p):
                r = ytr.reshape(-1) - (fitted - X2s[:, j] * beta[j])
                wj = Wj[j]
                xj = X2s[:, j]
                num = float(np.sum(wj * xj * r))
                den = float(np.sum(wj * xj * xj))
                bj = num / den if den > eps else 0.0
                fitted += xj * (bj - beta[j])
                beta[j] = bj
            if float(np.max(np.abs(beta - beta_old))) <= tol:
                break

        x_te = float(Xte[t, 1])
        x_te_s = _standardize(x_te, x_mu, x_sd) if scale_x else x_te
        yhat[t] = float(np.array([1.0, x_te_s]) @ beta)

    pred_diag = pd.DataFrame({
        "condWLS2": condWLS2_list,
        "condWLS3": condWLS3_list,
        "condM3_nor": np.full(m, np.nan),
        "uniform_flag": np.zeros(m, dtype=int),
        "neff_post": neff_list,
    })
    return yhat, pred_diag


def _local_ridge_predict_numpy_with_diag(
    coords_tr: np.ndarray,
    ytr: np.ndarray,
    Xtr: np.ndarray,
    coords_te: np.ndarray,
    Xte: np.ndarray,
    *,
    bw: float,
    fixed: bool = False,
    ridge_alpha: float = 1e-2,
    ridge_mode: str = "trace",
    u_scale_m: float = 2000.0,
    scale_x: bool = False,
    x_mu: Optional[float] = None,
    x_sd: Optional[float] = None,
    scale_z: str = "none",
    eps: float = EPS,
) -> Tuple[np.ndarray, pd.DataFrame]:
    m = coords_te.shape[0]
    yhat = np.empty(m, dtype=float)

    condWLS2_list = np.empty(m, dtype=float)
    condWLS2_ridge_list = np.empty(m, dtype=float)
    condWLS3_list = np.empty(m, dtype=float)
    neff_list = np.empty(m, dtype=float)

    p = Xtr.shape[1]

    for t in range(m):
        d = _pairwise_dist(coords_tr, coords_te[t])
        w = _bisquare_weights(d, bw, fixed=fixed)
        neff_list[t] = _neff_from_normalized_weights(w)

        z_loc = d / float(u_scale_m)
        X3 = np.column_stack([Xtr[:, 0], Xtr[:, 1], z_loc]).astype(float)
        X2 = Xtr

        X2s, X3s, _ = _scale_design_X2_X3(
            X2,
            X3,
            scale_x=scale_x,
            x_mu=x_mu,
            x_sd=x_sd,
            scale_z=scale_z,
        )

        XtWX2 = X2s.T @ (w[:, None] * X2s)
        XtWy = X2s.T @ (w * ytr.reshape(-1))
        XtWX3 = X3s.T @ (w[:, None] * X3s)

        condWLS2_list[t] = cond_spd(XtWX2)
        condWLS3_list[t] = cond_spd(XtWX3)

        if ridge_mode == "trace":
            lam = float(ridge_alpha) * float(np.trace(XtWX2) / max(1, p))
        elif ridge_mode == "diagmean":
            lam = float(ridge_alpha) * float(np.mean(np.diag(XtWX2)))
        elif ridge_mode == "fixed":
            lam = float(ridge_alpha)
        else:
            raise ValueError("ridge_mode must be one of {'trace', 'diagmean', 'fixed'}.")

        condWLS2_ridge_list[t] = cond_ridge_spd(XtWX2, lam)

        try:
            beta = np.linalg.solve(XtWX2 + lam * np.eye(p), XtWy)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(XtWX2 + lam * np.eye(p)) @ XtWy

        x_te = float(Xte[t, 1])
        x_te_s = _standardize(x_te, x_mu, x_sd) if scale_x else x_te
        yhat[t] = float(np.array([1.0, x_te_s]) @ beta)

    pred_diag = pd.DataFrame({
        "condWLS2": condWLS2_list,
        "condWLS2_ridge": condWLS2_ridge_list,
        "condWLS3": condWLS3_list,
        "condM3_nor": np.full(m, np.nan),
        "uniform_flag": np.zeros(m, dtype=int),
        "neff_post": neff_list,
    })
    return yhat, pred_diag


def fit_predict_LocalRidge(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    reproduce_rank_deficiency: bool = False,
    ridge_alpha: float = 1e-2,
    ridge_mode: str = "trace",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    _require_dependency(_HAS_MGWR, "mgwr")

    coords_tr, coords_te, ytr, Xtr_raw, Xte_raw, Xtr, Xte, bw_min = _prepare_local_regression_design(
        train_df,
        test_df,
        reproduce_rank_deficiency=reproduce_rank_deficiency,
    )

    selector = Sel_BW(coords_tr, ytr, Xtr, fixed=False, kernel="bisquare", multi=False, constant=False)
    bw = selector.search(bw_min=bw_min)

    yhat, pred_diag = _local_ridge_predict_numpy_with_diag(
        coords_tr,
        ytr.reshape(-1),
        Xtr,
        coords_te,
        Xte,
        bw=bw,
        fixed=False,
        ridge_alpha=ridge_alpha,
        ridge_mode=ridge_mode,
    )
    return yhat.astype(float), {"bw": bw, "pred_diag": pred_diag}


def fit_predict_GWR(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    reproduce_rank_deficiency: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    _require_dependency(_HAS_MGWR, "mgwr")

    coords_tr, coords_te, ytr, Xtr_raw, Xte_raw, Xtr, Xte, bw_min = _prepare_local_regression_design(
        train_df,
        test_df,
        reproduce_rank_deficiency=reproduce_rank_deficiency,
    )

    selector = Sel_BW(coords_tr, ytr, Xtr, fixed=False, kernel="bisquare", multi=False, constant=False)
    bw = selector.search(bw_min=bw_min)

    yhat, pred_diag = _gwr_predict_numpy_with_diag(
        coords_tr,
        ytr.reshape(-1),
        Xtr,
        coords_te,
        Xte,
        bw=bw,
        fixed=False,
    )
    return yhat.astype(float), {"bw": bw, "pred_diag": pred_diag}


def fit_predict_MGWR(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    reproduce_rank_deficiency: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    _require_dependency(_HAS_MGWR, "mgwr")

    coords_tr, coords_te, ytr, Xtr_raw, Xte_raw, Xtr, Xte, bw_min = _prepare_local_regression_design(
        train_df,
        test_df,
        reproduce_rank_deficiency=reproduce_rank_deficiency,
    )

    selector = Sel_BW(coords_tr, ytr, Xtr, fixed=False, kernel="bisquare", multi=True, constant=False)

    k = int(Xtr.shape[1])
    mbw_min = [bw_min] * k
    try:
        bws = selector.search(multi_bw_min=mbw_min)
    except TypeError:
        bws = selector.search(multi_bw_min=[bw_min])

    if np.isscalar(bws):
        bws = [int(bws)] * k
    else:
        bws = list(bws)
        if len(bws) != k:
            bws = [int(bws[0])] * k

    yhat, pred_diag = _mgwr_predict_numpy_pointwise_with_diag(
        coords_tr,
        ytr.reshape(-1),
        Xtr,
        coords_te,
        Xte,
        bws=bws,
        fixed=False,
    )
    return yhat.astype(float), {"bws": bws, "pred_diag": pred_diag}


# ---------------------------------------------------------------------
# Diagnostics aggregation
# ---------------------------------------------------------------------
def aggregate_local_diagnostics(preds_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if preds_df is None or len(preds_df) == 0:
        return pd.DataFrame(), pd.DataFrame()

    _require_columns(
        preds_df,
        {"fold", "model", "condWLS2", "condWLS3", "uniform_flag", "neff_post"},
        "preds_df",
    )

    df = preds_df.copy()

    agg: Dict[str, Tuple[str, Callable[[pd.Series], float]]] = {
        "n_points": ("condWLS2", "size"),
        "condWLS2_mean": ("condWLS2", lambda x: float(np.nanmean(x.to_numpy(dtype=float)))),
        "condWLS2_median": ("condWLS2", lambda x: float(np.nanmedian(x.to_numpy(dtype=float)))),
        "condWLS3_mean": ("condWLS3", lambda x: float(np.nanmean(x.to_numpy(dtype=float)))),
        "condWLS3_median": ("condWLS3", lambda x: float(np.nanmedian(x.to_numpy(dtype=float)))),
        "neff_post_mean": ("neff_post", lambda x: float(np.nanmean(x.to_numpy(dtype=float)))),
        "uniform_rate": ("uniform_flag", lambda x: float(np.nanmean(x.to_numpy(dtype=float)))),
    }

    if "condWLS2_ridge" in df.columns:
        agg["condWLS2_ridge_mean"] = ("condWLS2_ridge", lambda x: float(np.nanmean(x.to_numpy(dtype=float))))
        agg["condWLS2_ridge_median"] = ("condWLS2_ridge", lambda x: float(np.nanmedian(x.to_numpy(dtype=float))))
    if "condM3_nor" in df.columns:
        agg["condM3_nor_mean"] = ("condM3_nor", lambda x: float(np.nanmean(x.to_numpy(dtype=float))))
        agg["condM3_nor_median"] = ("condM3_nor", lambda x: float(np.nanmedian(x.to_numpy(dtype=float))))

    diag_df = df.groupby(["fold", "model"], as_index=False).agg(**agg)

    num_mask = diag_df["fold"].apply(lambda v: isinstance(v, (int, np.integer)))
    dnum = diag_df[num_mask].copy()

    if len(dnum) == 0:
        return diag_df, pd.DataFrame(columns=diag_df.columns)

    mean_cols = [c for c in diag_df.columns if c not in ("fold", "model", "n_points")]
    diag_df_mean = dnum.groupby("model", as_index=False)[mean_cols].mean(numeric_only=True)
    nsum = dnum.groupby("model", as_index=False)["n_points"].sum()

    diag_df_mean = diag_df_mean.merge(nsum, on="model", how="left")
    diag_df_mean = diag_df_mean.assign(fold="MEAN")
    diag_df_mean = diag_df_mean[["fold", "model", "n_points"] + [c for c in diag_df_mean.columns if c not in ("fold", "model", "n_points")]]

    return diag_df, diag_df_mean


# ---------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------
def run(
    df: pd.DataFrame,
    *,
    cv_mode: str = "random",
    n_splits: int = 5,
    block_size_m: float = 5000.0,
    seed: int = 42,
    fallback_to_random: bool = True,
    moran_k: int = 8,
    reproduce_rank_deficiency: bool = True,
    force_epsg_xy: Optional[int] = None,
    gr_K: int = 30,
    gr_h_m: float = 2000.0,
    gr_gamma: float = 1.0,
    gr_n0: float = 20.0,
    gr_use_ess: bool = True,
    gr_min_neff: float = 4.0,
    gr_eps_phi: float = 1e-3,
    gr_eps_theta: float = 1e-8,
    gr_eps_eta: float = 1e-8,
    gr_eta_max: float = 50.0,
    gr_u_scale: Optional[float] = None,
    gr_res_mode: str = "distance",
    gr_res_eta: float = 0.5,
    gr_use_global_calibration: bool = False,
    srf_n_estimators: int = 300,
    srf_max_depth: int = 20,
    uk_variogram: str = "spherical",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the Chapter 8 benchmark.

    Required input columns:
        lat, lon, X, Y
    """
    _require_columns(df, {"lat", "lon", "X", "Y"}, "df")

    df = df.reset_index(drop=True).copy()
    df = add_xy_m(df, force_epsg=force_epsg_xy)
    n = len(df)

    if cv_mode == "spatial":
        splits = spatial_block_kfold_indices(df, n_splits=n_splits, block_size_m=block_size_m, seed=seed)
        splits = keep_nonempty_folds(splits)
        if fallback_to_random and len(splits) < max(2, n_splits // 2):
            logger.warning("Spatial blocking produced too few non-empty folds; falling back to random K-fold.")
            splits = random_kfold_indices(n, n_splits=n_splits, seed=seed)
    elif cv_mode == "random":
        splits = random_kfold_indices(n, n_splits=n_splits, seed=seed)
    else:
        raise ValueError("cv_mode must be 'spatial' or 'random'.")

    all_metrics: List[Dict[str, Any]] = []
    all_preds: List[pd.DataFrame] = []

    for fold, (train_idx, test_idx) in enumerate(splits):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        ov = overlap_count(train_df, test_df)
        y_test = test_df["Y"].to_numpy(dtype=float)
        x_mu, x_sd = _safe_mean_std(train_df["X"].to_numpy(dtype=float))

        model_specs: List[Tuple[str, Callable[[pd.DataFrame, pd.DataFrame], Tuple[np.ndarray, Dict[str, Any]]]]] = [
            (
                "GR",
                lambda tr, te: fit_predict_GR_ch8(
                    tr,
                    te,
                    K=gr_K,
                    h_m=gr_h_m,
                    gamma=gr_gamma,
                    n0=gr_n0,
                    use_ess=gr_use_ess,
                    min_neff=gr_min_neff,
                    eps_phi=gr_eps_phi,
                    eps_theta=gr_eps_theta,
                    eps_eta=gr_eps_eta,
                    eta_max=gr_eta_max,
                    u_scale=gr_u_scale,
                    scale_x=True,
                    x_mu=x_mu,
                    x_sd=x_sd,
                    scale_z="local",
                    use_residual_knn=False,
                    res_mode=gr_res_mode,
                    res_eta=gr_res_eta,
                    use_global_calibration=gr_use_global_calibration,
                ),
            ),
            (
                "GR+ResidualKNN",
                lambda tr, te: fit_predict_GR_ch8(
                    tr,
                    te,
                    K=gr_K,
                    h_m=gr_h_m,
                    gamma=gr_gamma,
                    n0=gr_n0,
                    use_ess=gr_use_ess,
                    min_neff=gr_min_neff,
                    eps_phi=gr_eps_phi,
                    eps_theta=gr_eps_theta,
                    eps_eta=gr_eps_eta,
                    eta_max=gr_eta_max,
                    u_scale=gr_u_scale,
                    scale_x=True,
                    x_mu=x_mu,
                    x_sd=x_sd,
                    scale_z="local",
                    use_residual_knn=True,
                    res_mode=gr_res_mode,
                    res_eta=gr_res_eta,
                    use_global_calibration=gr_use_global_calibration,
                ),
            ),
            ("OLS", fit_predict_OLS),
            (
                "LocalRidge",
                lambda tr, te: fit_predict_LocalRidge(
                    tr,
                    te,
                    reproduce_rank_deficiency=reproduce_rank_deficiency,
                    ridge_alpha=1e-2,
                    ridge_mode="trace",
                ),
            ),
            (
                "GWR",
                lambda tr, te: fit_predict_GWR(
                    tr,
                    te,
                    reproduce_rank_deficiency=reproduce_rank_deficiency,
                ),
            ),
            (
                "MGWR",
                lambda tr, te: fit_predict_MGWR(
                    tr,
                    te,
                    reproduce_rank_deficiency=reproduce_rank_deficiency,
                ),
            ),
            ("UniversalKriging", lambda tr, te: fit_predict_UK(tr, te, variogram_model=uk_variogram)),
            (
                "SRF",
                lambda tr, te: fit_predict_SRF(
                    tr,
                    te,
                    n_estimators=srf_n_estimators,
                    max_depth=srf_max_depth,
                    random_state=seed,
                ),
            ),
        ]

        for model_name, fitpredict in model_specs:
            logger.info("Fold %s | %s", fold, model_name)
            t0 = time.perf_counter()

            try:
                yhat, info = fitpredict(train_df, test_df)
                elapsed = time.perf_counter() - t0

                resid = y_test - yhat
                I, p_sim = morans_I_residuals_fixed_knn(test_df, resid, k=moran_k)

                pred_diag = info.get("pred_diag", None) if isinstance(info, dict) else None
                extra = None
                if isinstance(info, dict):
                    extra = info.get("bw", info.get("bws", None))

                all_metrics.append({
                    "fold": fold,
                    "model": model_name,
                    "n_train": len(train_df),
                    "n_test": len(test_df),
                    "overlap_latlon": ov,
                    "RMSE": rmse(y_test, yhat),
                    "MAE": mae(y_test, yhat),
                    "R2": r2(y_test, yhat),
                    "Moran_I": I,
                    "Moran_p": p_sim,
                    "time_sec": elapsed,
                    "error": None,
                    "extra": extra,
                })

                base_pred = pd.DataFrame({
                    "fold": fold,
                    "model": model_name,
                    "lat": test_df["lat"].to_numpy(),
                    "lon": test_df["lon"].to_numpy(),
                    "y": y_test,
                    "yhat": yhat,
                    "resid": resid,
                })

                for c in ["condWLS2", "condWLS2_ridge", "condWLS3", "condM3_nor", "uniform_flag", "neff_post"]:
                    base_pred[c] = np.nan

                if isinstance(pred_diag, pd.DataFrame) and len(pred_diag) == len(base_pred):
                    for c in ["condWLS2", "condWLS2_ridge", "condWLS3", "condM3_nor", "uniform_flag", "neff_post"]:
                        if c in pred_diag.columns:
                            base_pred[c] = pred_diag[c].to_numpy()

                all_preds.append(base_pred)

            except Exception as exc:
                elapsed = time.perf_counter() - t0
                logger.exception("Benchmark failed for model=%s, fold=%s", model_name, fold)
                all_metrics.append({
                    "fold": fold,
                    "model": model_name,
                    "n_train": len(train_df),
                    "n_test": len(test_df),
                    "overlap_latlon": ov,
                    "RMSE": np.nan,
                    "MAE": np.nan,
                    "R2": np.nan,
                    "Moran_I": np.nan,
                    "Moran_p": np.nan,
                    "time_sec": elapsed,
                    "error": repr(exc),
                    "extra": None,
                })

    metrics_df = pd.DataFrame(all_metrics)
    preds_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()

    ok = metrics_df[metrics_df["error"].isna()].copy()
    if len(ok):
        summary = ok.groupby("model")[["RMSE", "MAE", "R2", "Moran_I", "time_sec"]].mean().reset_index()
        summary["fold"] = "MEAN"
        summary["n_train"] = np.nan
        summary["n_test"] = np.nan
        summary["overlap_latlon"] = np.nan
        summary["Moran_p"] = np.nan
        summary["error"] = None
        summary["extra"] = None
        metrics_df = pd.concat([metrics_df, summary], ignore_index=True)

    diag_df, diag_df_mean = aggregate_local_diagnostics(preds_df)
    diag_df_full = pd.concat([diag_df, diag_df_mean], ignore_index=True) if len(diag_df_mean) else diag_df

    return metrics_df, preds_df, diag_df_full


__all__ = [
    "BenchmarkResult",
    "rmse",
    "mae",
    "r2",
    "add_xy_m",
    "random_kfold_indices",
    "spatial_block_kfold_indices",
    "keep_nonempty_folds",
    "fit_train_local_and_residual_knn",
    "apply_residual_knn",
    "fit_predict_GR_ch8",
    "fit_predict_OLS",
    "fit_predict_LocalRidge",
    "fit_predict_GWR",
    "fit_predict_MGWR",
    "fit_predict_UK",
    "fit_predict_SRF",
    "aggregate_local_diagnostics",
    "run_benchmark_ch8",
]
