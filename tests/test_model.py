import numpy as np
import pandas as pd

from grpy import GimbalRegression


def make_toy_data(n=40, seed=42):
    rng = np.random.default_rng(seed)
    lat = 35.0 + 0.01 * rng.random(n)
    lon = 139.0 + 0.01 * rng.random(n)
    x = rng.normal(size=n)
    y = 1.0 + 2.0 * x + 0.1 * rng.normal(size=n)
    return y, x, lat, lon


def test_model_fit_sets_fitted_state():
    y, x, lat, lon = make_toy_data()

    model = GimbalRegression(K=10, h_m=1000.0, gamma=1.0)
    model.fit(y=y, x=x, lat=lat, lon=lon)

    assert model.is_fitted_ is True
    assert hasattr(model, "results_")
    assert isinstance(model.results_, pd.DataFrame)
    assert len(model.results_) == len(y)


def test_model_predict_length_matches_input():
    y, x, lat, lon = make_toy_data()

    model = GimbalRegression(K=10, h_m=1000.0, gamma=1.0).fit(y=y, x=x, lat=lat, lon=lon)
    pred = model.predict()

    assert isinstance(pred, np.ndarray)
    assert pred.shape == y.shape


def test_model_diagnostics_returns_dataframe():
    y, x, lat, lon = make_toy_data()

    model = GimbalRegression(K=10, h_m=1000.0, gamma=1.0).fit(y=y, x=x, lat=lat, lon=lon)
    diag = model.diagnostics()

    assert isinstance(diag, pd.DataFrame)
    assert len(diag) == len(y)

    expected_cols = {"ix", "condM_nor", "neff_raw", "neff_post", "uniform_flag"}
    assert expected_cols.intersection(set(diag.columns))


def test_model_summary_has_expected_keys():
    y, x, lat, lon = make_toy_data()

    model = GimbalRegression(K=10, h_m=1000.0, gamma=1.0).fit(y=y, x=x, lat=lat, lon=lon)
    s = model.summary()

    assert isinstance(s, dict)

    expected_keys = {
        "n_locations",
        "mean_condM_nor",
        "median_condM_nor",
        "mean_neff_post",
        "uniform_fallback_rate",
        "mean_rmse",
    }
    assert expected_keys.issubset(set(s.keys()))


def test_model_results_contains_core_columns():
    y, x, lat, lon = make_toy_data()

    model = GimbalRegression(K=10, h_m=1000.0, gamma=1.0).fit(y=y, x=x, lat=lat, lon=lon)
    cols = set(model.results_.columns)

    expected = {
        "ix",
        "lat",
        "lon",
        "B0",
        "B1",
        "B2",
        "y",
        "yhat",
        "resid",
        "condM_nor",
        "neff_post",
    }
    assert expected.issubset(cols)


def test_predict_before_fit_raises():
    model = GimbalRegression()

    try:
        model.predict()
        assert False, "predict() should raise before fit()"
    except RuntimeError:
        assert True


def test_fit_with_length_mismatch_raises():
    y, x, lat, lon = make_toy_data()
    model = GimbalRegression()

    try:
        model.fit(y=y[:-1], x=x, lat=lat, lon=lon)
        assert False, "fit() should raise on length mismatch"
    except ValueError:
        assert True