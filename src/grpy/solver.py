from __future__ import annotations

import numpy as np


def _condition_number_symmetric(a: np.ndarray, eps: float = 1e-12) -> float:
    """
    Compute the condition number of a symmetric matrix using eigenvalues.
    Returns np.inf if the matrix is numerically singular.
    """
    a = np.asarray(a, dtype=float)
    a = 0.5 * (a + a.T)
    vals = np.linalg.eigvalsh(a)
    lam_min = float(np.min(vals))
    lam_max = float(np.max(vals))
    if lam_max <= eps or lam_min <= eps:
        return np.inf
    return lam_max / lam_min


def solve_beta_eq46_numpy(X, y, w, gamma):
    """
    Solve the GR local estimator

        beta_hat = (X'X + 2 gamma X'WX)^(-1) (X'y + 2 gamma X'Wy)

    and return fitted values, residuals, and simple fit diagnostics.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Local design matrix.
    y : array-like of shape (n_samples,)
        Local response vector.
    w : array-like of shape (n_samples,)
        Nonnegative local weights.
    gamma : float
        GR tuning parameter. Must be nonnegative.

    Returns
    -------
    beta : ndarray of shape (n_features,)
        Estimated coefficient vector.
    yhat : ndarray of shape (n_samples,)
        Fitted values.
    resid : ndarray of shape (n_samples,)
        Residuals.
    r2 : float
        Coefficient of determination.
    adjr2 : float
        Adjusted R-squared, using p = number of columns of X.
    rmse : float
        Root mean squared error.
    condM : float
        Condition number of the realized normal matrix.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    w = np.asarray(w, dtype=float).reshape(-1)
    gamma = float(gamma)

    if X.ndim != 2:
        raise ValueError("solve_beta_eq46_numpy: X must be a 2D array.")
    if y.ndim != 1:
        raise ValueError("solve_beta_eq46_numpy: y must be a 1D array.")
    if w.ndim != 1:
        raise ValueError("solve_beta_eq46_numpy: w must be a 1D array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("solve_beta_eq46_numpy: X and y must have the same number of rows.")
    if X.shape[0] != w.shape[0]:
        raise ValueError("solve_beta_eq46_numpy: X and w must have the same number of rows.")
    if X.shape[0] == 0:
        raise ValueError("solve_beta_eq46_numpy: empty local design is not allowed.")
    if gamma < 0.0:
        raise ValueError("solve_beta_eq46_numpy: gamma must be nonnegative.")
    if not np.all(np.isfinite(X)):
        raise ValueError("solve_beta_eq46_numpy: X contains non-finite values.")
    if not np.all(np.isfinite(y)):
        raise ValueError("solve_beta_eq46_numpy: y contains non-finite values.")
    if not np.all(np.isfinite(w)):
        raise ValueError("solve_beta_eq46_numpy: w contains non-finite values.")

    XtX = X.T @ X
    XtY = X.T @ y

    WX = w[:, None] * X
    Wy = w * y

    XtWX = X.T @ WX
    XtWY = X.T @ Wy

    M_nor = XtX + 2.0 * gamma * XtWX
    b = XtY + 2.0 * gamma * XtWY

    try:
        condM = float(_condition_number_symmetric(M_nor))
    except Exception:
        condM = np.nan

    try:
        beta = np.linalg.solve(M_nor, b)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "solve_beta_eq46_numpy: realized normal matrix is singular."
        ) from exc

    yhat = X @ beta
    resid = y - yhat

    ss_res = float(resid @ resid)
    yc = y - float(np.mean(y))
    ss_tot = float(yc @ yc)

    if ss_tot <= 1e-12:
        r2 = np.nan
        adjr2 = np.nan
    else:
        r2 = 1.0 - ss_res / ss_tot
        n, p = X.shape
        if n - p <= 0:
            adjr2 = np.nan
        else:
            adjr2 = 1.0 - (1.0 - r2) * (n - 1.0) / (n - p)

    rmse = float(np.sqrt(ss_res / y.shape[0]))
    return beta, yhat, resid, r2, adjr2, rmse, condM