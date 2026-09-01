import numpy as np
import pandas as pd

from grpy import GimbalRegression
from grpy.plotting import results_to_gdf


def make_toy_data(n=20, seed=123):
    rng = np.random.default_rng(seed)
    lat = 35.0 + 0.01 * rng.random(n)
    lon = 139.0 + 0.01 * rng.random(n)
    x = rng.normal(size=n)
    y = 1.0 + 1.5 * x + 0.1 * rng.normal(size=n)
    return y, x, lat, lon


def test_results_to_gdf_returns_geodataframe():
    df = pd.DataFrame(
        {
            "lon": [139.0, 139.01],
            "lat": [35.0, 35.01],
            "value": [1.0, 2.0],
        }
    )

    gdf = results_to_gdf(df)

    assert len(gdf) == len(df)
    assert "geometry" in gdf.columns
    assert gdf.crs is not None


def test_results_to_gdf_missing_columns_raises():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    try:
        results_to_gdf(df)
        assert False, "results_to_gdf() should raise when lon/lat are missing"
    except ValueError:
        assert True


def test_draw_map_runs_without_basemap():
    y, x, lat, lon = make_toy_data()
    model = GimbalRegression(K=8, h_m=1000.0, gamma=1.0).fit(y=y, x=x, lat=lat, lon=lon)

    fig, ax = model.draw_map(
        column="B1",
        title="Test B1 map",
        basemap=False,
        show=False,
    )

    assert fig is not None
    assert ax is not None