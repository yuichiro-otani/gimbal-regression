from __future__ import annotations

import numpy as np

from .utils import haversine, njit


@njit(cache=True, fastmath=True)
def _knn_haversine_numba(lat, lon, ix, K):
    n = lat.shape[0]
    d = np.empty(n, dtype=np.float64)
    lat0 = lat[ix]
    lon0 = lon[ix]

    for j in range(n):
        d[j] = haversine(lat0, lon0, lat[j], lon[j])

    idx = np.argsort(d)[:K]
    return d[idx], idx


def knn_haversine(lat, lon, ix, K):
    """
    Return the K nearest neighbors by Haversine distance, including ix itself.

    Parameters
    ----------
    lat : array-like of shape (n,)
        Latitude values in degrees.
    lon : array-like of shape (n,)
        Longitude values in degrees.
    ix : int
        Target index.
    K : int
        Number of nearest neighbors to return.

    Returns
    -------
    dist_sorted : ndarray of shape (K,)
        Neighbor distances in meters, sorted ascending.
    idx_sorted : ndarray of shape (K,)
        Neighbor indices sorted by distance.

    Notes
    -----
    This function performs exact brute-force nearest-neighbor search.
    It is deterministic and simple, but may be slow for very large datasets.
    """
    lat = np.asarray(lat, dtype=np.float64).reshape(-1)
    lon = np.asarray(lon, dtype=np.float64).reshape(-1)

    n = lat.shape[0]
    if lon.shape[0] != n:
        raise ValueError("knn_haversine: lat and lon must have the same length.")
    if n == 0:
        raise ValueError("knn_haversine: empty coordinate arrays are not allowed.")
    if not (0 <= int(ix) < n):
        raise ValueError("knn_haversine: ix is out of bounds.")
    if int(K) <= 0:
        raise ValueError("knn_haversine: K must be positive.")

    K = min(int(K), n)
    return _knn_haversine_numba(lat, lon, int(ix), K)