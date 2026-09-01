from __future__ import annotations

import numpy as np

try:
    from numba import njit
except Exception:
    def njit(*args, **kwargs):
        def deco(f):
            return f
        return deco


EARTH_RADIUS_M = 6371000.0

__all__ = [
    "njit",
    "haversine",
    "tangent_plane_deltas_m",
    "bearing_angles_from_deltas",
]


@njit(cache=True, fastmath=True)
def haversine(lat1, lon1, lat2, lon2):
    """
    Great-circle distance in meters between two latitude/longitude points.

    Parameters
    ----------
    lat1, lon1 : float
        First point in decimal degrees.
    lat2, lon2 : float
        Second point in decimal degrees.

    Returns
    -------
    float
        Haversine distance in meters.
    """
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2 - lon1)

    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )

    if a < 0.0:
        a = 0.0
    elif a > 1.0:
        a = 1.0

    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


@njit(cache=True, fastmath=True)
def tangent_plane_deltas_m(si_lat_deg, si_lon_deg, S_lat_deg, S_lon_deg):
    """
    Approximate east/north displacements in meters from a target location
    to one or more neighbor locations using a local tangent-plane approximation.

    Parameters
    ----------
    si_lat_deg, si_lon_deg : float
        Target location in decimal degrees.
    S_lat_deg, S_lon_deg : array-like
        Neighbor locations in decimal degrees.

    Returns
    -------
    east : ndarray
        Eastward displacement in meters.
    north : ndarray
        Northward displacement in meters.

    Notes
    -----
    This uses a local first-order approximation:
        north = R * dlat
        east  = R * cos(lat0) * dlon
    It is appropriate for local neighborhoods, not for global-scale projection.
    """
    si_lat = np.radians(si_lat_deg)
    si_lon = np.radians(si_lon_deg)

    lat = np.radians(S_lat_deg)
    lon = np.radians(S_lon_deg)

    dlat = lat - si_lat
    dlon = lon - si_lon

    north = EARTH_RADIUS_M * dlat
    east = EARTH_RADIUS_M * np.cos(si_lat) * dlon
    return east, north


@njit(cache=True, fastmath=True)
def bearing_angles_from_deltas(east_m, north_m):
    """
    Bearing angles in radians from east/north displacements.

    Parameters
    ----------
    east_m, north_m : array-like
        Eastward and northward displacements in meters.

    Returns
    -------
    ndarray
        Angles in radians computed as atan2(north, east).

    Notes
    -----
    Under the east-north convention, angle 0 points east and pi/2 points north.
    When both inputs are zero, NumPy/Numba returns 0.0.
    """
    return np.arctan2(north_m, east_m)