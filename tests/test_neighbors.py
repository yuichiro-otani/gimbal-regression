import numpy as np

from grpy.neighbors import knn_haversine


def make_coords():
    lat = np.array([35.0000, 35.0005, 35.0010, 35.0015, 35.0020], dtype=float)
    lon = np.array([139.0000, 139.0005, 139.0010, 139.0015, 139.0020], dtype=float)
    return lat, lon


def test_knn_haversine_returns_expected_shapes():
    lat, lon = make_coords()

    dist, idx = knn_haversine(lat, lon, ix=0, K=3)

    assert isinstance(dist, np.ndarray)
    assert isinstance(idx, np.ndarray)
    assert dist.shape == (3,)
    assert idx.shape == (3,)


def test_knn_haversine_includes_self():
    lat, lon = make_coords()

    dist, idx = knn_haversine(lat, lon, ix=2, K=3)

    assert 2 in idx
    pos = np.where(idx == 2)[0][0]
    assert np.isclose(dist[pos], 0.0)


def test_knn_haversine_distances_are_nonnegative_and_sorted():
    lat, lon = make_coords()

    dist, idx = knn_haversine(lat, lon, ix=1, K=5)

    assert np.all(dist >= 0.0)
    assert np.all(dist[:-1] <= dist[1:])


def test_knn_haversine_nearest_point_is_self():
    lat, lon = make_coords()

    dist, idx = knn_haversine(lat, lon, ix=4, K=2)

    assert idx[0] == 4
    assert np.isclose(dist[0], 0.0)


def test_knn_haversine_with_k_equals_n():
    lat, lon = make_coords()
    n = len(lat)

    dist, idx = knn_haversine(lat, lon, ix=0, K=n)

    assert len(dist) == n
    assert len(idx) == n
    assert set(idx.tolist()) == set(range(n))


def test_knn_haversine_k_one_returns_only_self():
    lat, lon = make_coords()

    dist, idx = knn_haversine(lat, lon, ix=3, K=1)

    assert idx.shape == (1,)
    assert dist.shape == (1,)
    assert idx[0] == 3
    assert np.isclose(dist[0], 0.0)