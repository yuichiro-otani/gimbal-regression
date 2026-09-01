import numpy as np

from grpy.utils import haversine, tangent_plane_deltas_m, bearing_angles_from_deltas


def test_haversine_zero_distance():
    d = haversine(35.0, 139.0, 35.0, 139.0)
    assert np.isfinite(d)
    assert abs(d) < 1e-9


def test_haversine_positive_distance():
    d = haversine(35.0, 139.0, 35.001, 139.001)
    assert np.isfinite(d)
    assert d > 0.0


def test_tangent_plane_deltas_shapes():
    s_lat = 35.0
    s_lon = 139.0
    lat = np.array([35.0, 35.001, 34.999], dtype=float)
    lon = np.array([139.0, 139.002, 138.998], dtype=float)

    east, north = tangent_plane_deltas_m(s_lat, s_lon, lat, lon)

    assert isinstance(east, np.ndarray)
    assert isinstance(north, np.ndarray)
    assert east.shape == lat.shape
    assert north.shape == lat.shape


def test_tangent_plane_deltas_origin_is_zero():
    s_lat = 35.0
    s_lon = 139.0
    lat = np.array([35.0], dtype=float)
    lon = np.array([139.0], dtype=float)

    east, north = tangent_plane_deltas_m(s_lat, s_lon, lat, lon)

    assert np.allclose(east, 0.0)
    assert np.allclose(north, 0.0)


def test_bearing_angles_from_deltas_basic_axes():
    east = np.array([1.0, 0.0, -1.0, 0.0], dtype=float)
    north = np.array([0.0, 1.0, 0.0, -1.0], dtype=float)

    ang = bearing_angles_from_deltas(east, north)

    assert ang.shape == (4,)
    assert np.allclose(ang[0], 0.0)                 # east
    assert np.allclose(ang[1], np.pi / 2.0)         # north
    assert np.allclose(abs(ang[2]), np.pi)          # west
    assert np.allclose(ang[3], -np.pi / 2.0)        # south