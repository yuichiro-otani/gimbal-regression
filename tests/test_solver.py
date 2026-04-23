import numpy as np

from grpy.solver import solve_beta_eq46_numpy


def test_solve_beta_eq46_numpy_output_shapes():
    X = np.array(
        [
            [1.0, 0.0, 0.1],
            [1.0, 1.0, 0.7],
            [1.0, 2.0, 0.9],
            [1.0, 3.0, 1.8],
            [1.0, 4.0, 2.2],
        ],
        dtype=float,
    )
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    w = np.ones(len(y), dtype=float) / len(y)

    beta, yhat, resid, r2, adjr2, rmse, condM = solve_beta_eq46_numpy(X, y, w, gamma=1.0)

    assert beta.shape == (3,)
    assert yhat.shape == y.shape
    assert resid.shape == y.shape
    assert np.isfinite(r2) or np.isnan(r2)
    assert np.isfinite(adjr2) or np.isnan(adjr2)
    assert np.isfinite(rmse)
    assert np.isfinite(condM) or np.isnan(condM)


def test_solve_beta_eq46_numpy_small_rmse_on_simple_linear_signal():
    x = np.arange(10, dtype=float)
    z = x / 10.0
    X = np.column_stack([np.ones_like(x), x, z])
    y = 2.0 + 3.0 * x + 0.5 * z
    w = np.ones(len(y), dtype=float) / len(y)

    beta, yhat, resid, r2, adjr2, rmse, condM = solve_beta_eq46_numpy(X, y, w, gamma=1.0)

    assert rmse < 1e-8
    assert np.max(np.abs(resid)) < 1e-7
    assert r2 > 0.999999


def test_solve_beta_eq46_numpy_constant_response():
    x = np.arange(6, dtype=float)
    z = np.array([0.0, 0.4, 1.1, 1.7, 2.8, 3.0], dtype=float)
    X = np.column_stack([np.ones_like(x), x, z])
    y = np.ones_like(x) * 5.0
    w = np.ones(len(y), dtype=float) / len(y)

    beta, yhat, resid, r2, adjr2, rmse, condM = solve_beta_eq46_numpy(X, y, w, gamma=1.0)

    assert beta.shape == (3,)
    assert yhat.shape == y.shape
    assert resid.shape == y.shape
    assert np.isfinite(rmse)