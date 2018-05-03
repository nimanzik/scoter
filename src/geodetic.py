from __future__ import division
import math
import os.path as op

import numpy as np

from shapely.geometry.point import Point

from .util import data_file, load_pickle


KM2M = 1000.
M2KM = 1. / KM2M

EARTH_RADIUS = 6371.0088 * KM2M
EARTH_RADIUS_EQUATOR = 6378.137 * KM2M

DEG2M = EARTH_RADIUS_EQUATOR * math.pi / 180.
M2DEG = 1. / DEG2M

DEG2RAD = math.pi / 180.
RAD2DEG = 1. / DEG2RAD


class WGS84(object):
    """Earth reference ellipsoid parameters for WGS84 geodetic system."""

    def __init__(self):
        self.__a = 6378137.
        self.__f = 1. / 298.257223563

    @property
    def a(self):
        # semi-major axis in [m]
        return self.__a

    @property
    def f(self):
        # earth flattening factor
        return self.__f

    @property
    def b(self):
        # semi-minor axis in [m]
        return self.__a * (1.-self.__f)

    @property
    def e2(self):
        # First eccentricity squared
        return self.__f * (2.-self.__f)

    @property
    def eprime2(self):
        # Second eccentricity squared
        return self.__f * (2.-self.__f) / (1.-self.__f)**2


def cos_gc_distance(lat1, lon1, lat2, lon2):
    """
    Cosine of the great-circle distance using spherical earth trigonometry.

    Parameters
    ----------
    lat1, lon1: float
        Latitude and longitude of the first point in [deg].
    lat2, lon2: float
        Latitude and longitude of the second point in [deg].

    Returns
    -------
    result : float
        The cosine of the great-circle distance.

    References
    ----------
    .. [1] Lay, T. and Wallace, T. (1995). Modern Global Seismology.
    """

    lat1, lon1, lat2, lon2 = np.array([lat1, lon1, lat2, lon2]) * DEG2RAD
    A = lon2 - lon1
    return min(
        1., np.sin(lat1)*np.sin(lat2) + np.cos(lat1)*np.cos(lat2)*np.cos(A))


def gc_distance(lat1, lon1, lat2, lon2):
    """
    Great-circle distance using spherical earth trigonometry.

    Parameters
    ----------
    lat1, lon1: float
        Latitude and longitude of the first point in [deg].
    lat2, lon2: float
        Latitude and longitude of the second point in [deg].

    Returns
    -------
    result : float
        Great-circle distance in [deg].

    References
    ----------
    .. [1] Lay, T. and Wallace, T. (1995). Modern Global Seismology.
    """
    return np.arccos(cos_gc_distance(lat1, lon1, lat2, lon2)) * RAD2DEG


def gc_azimuth(lat1, lon1, lat2, lon2):
    """
    Great-circle azimuth using spherical earth trigonometry.

    Parameters
    ----------
    lat1, lon1: float
        Latitude and longitude of the first point in [deg].
    lat2, lon2: float
        Latitude and longitude of the second point in [deg].

    Returns
    -------
    result : float
        Great-circle azimuth in [deg].

    References
    ----------
    .. [1] Lay, T. and Wallace, T. (1995). Modern Global Seismology.
    """

    numer = (
        np.cos(lat1*DEG2RAD) * np.cos(lat2*DEG2RAD) *
        np.sin((lon2-lon1)*DEG2RAD))

    denom = (
        np.sin(lat2*DEG2RAD) -
        np.sin(lat1*DEG2RAD) * cos_gc_distance(lat1, lon1, lat2, lon2))

    return np.arctan2(numer, denom) * RAD2DEG


def gc_azibazi(lat1, lon1, lat2, lon2):

    azi = gc_azimuth(lat1, lon1, lat2, lon2)

    numer = (
        np.cos(lat2*DEG2RAD) * np.cos(lat1*DEG2RAD) *
        np.sin((lon1-lon2)*DEG2RAD))

    denom = (
        np.sin(lat1*DEG2RAD) -
        np.sin(lat2*DEG2RAD) * cos_gc_distance(lat1, lon1, lat2, lon2))

    bazi = np.arctan2(numer, denom) * RAD2DEG

    return azi, bazi


def ellipsoid_distance(lat1, lon1, lat2, lon2):
    """
    Approximate ellipsoid distance between two points using Vincenty's
    formulae.

    Although it is called approximate, it is actually much more accurate
    than the great circle calculation.

    Parameters
    ----------
    lat1, lon1: float
        Latitude and longitude of the first point in [deg].
    lat2, lon2: float
        Latitude and longitude of the second point in [deg].

    Returns
    -------
    d : float
        Distance between points a and b in [m].

    References
    ----------
    .. [1] http://www.codeguru.com/cpp/cpp/algorithms/article.php/c5115/Geographic-Distance-and-Azimuth-Calculations.htm
    .. [2] Jean Meeus (1998), Astronomical Algorithms, 2nd ed,
        ISBN 0-943396-61-1
    """

    wgs = WGS84()
    a = wgs.a
    f = wgs.f

    lat1, lon1, lat2, lon2 = map(
        math.radians, [lat1, lon1, lat2, lon2])

    F = (lat1+lat2) * 0.5
    G = (lat1-lat2) * 0.5
    L = (lon1-lon2) * 0.5

    sinF, cosF = math.sin(F), math.cos(F)
    sinG, cosG = math.sin(G), math.cos(G)
    sinL, cosL = math.sin(L), math.cos(L)

    S = (sinG*sinG*cosL*cosL) + (cosF*cosF*sinL*sinL)
    C = (cosG*cosG*cosL*cosL) + (sinF*sinF*sinL*sinL)
    W = math.atan2(math.sqrt(S), math.sqrt(C))

    if W == 0.:
        return 0.

    R = math.sqrt(S*C) / W
    H1 = (3.*R - 1.) / (2.*C)
    H2 = (3.*R + 1.) / (2.*S)
    d = 2. * W * a
    d *= (1. + f*H1*sinF*sinF*cosG*cosG - f*H2*cosF*cosF*sinG*sinG)

    return d


def geodetic_to_ecef(lat, lon, alt):
    """
    Convert geodetic coordinates to Earth-Centered, Earth-Fixed (ECEF)
    Cartesian coordinates.

    Parameters
    ----------
    lat : float
        Geodetic latitude in [deg].
    lon : float
        Geodetic longitude in [deg].
    alt : float
        Geodetic altitude (height) in [m] (positive for points outside
        the geoid).

    Returns
    -------
    tuple, float
        ECEF Cartesian coordinates (X, Y, Z) in [m].

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/ECEF
    .. [2] https://en.wikipedia.org/wiki/Geographic_coordinate_conversion
        #From_geodetic_to_ECEF_coordinates
    """

    wgs = WGS84()
    a = wgs.a
    e2 = wgs.e2

    lat, lon = np.radians(lat), np.radians(lon)
    # Normal (plumb line)
    N = a / np.sqrt(1. - (e2*np.sin(lat)**2))

    X = (N+alt) * np.cos(lat) * np.cos(lon)
    Y = (N+alt) * np.cos(lat) * np.sin(lon)
    Z = (N*(1.-e2) + alt) * np.sin(lat)

    return (X, Y, Z)


def ecef_to_geodetic(X, Y, Z):
    """
    Convert Earth-Centered, Earth-Fixed (ECEF) Cartesian coordinates to
    geodetic coordinates (Ferrari's solution).

    Parameters
    ----------
    X, Y, Z : float
        Cartesian coordinates in ECEF system in [m].

    Returns
    -------
    tuple, float
        Geodetic coordinates (lat, lon, alt). Latitude and longitude are
        in [deg] and altitude is in [m] (positive for points outside the
        geoid).

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Geographic_coordinate_conversion
        #The_application_of_Ferrari.27s_solution
    """

    wgs = WGS84()
    a = wgs.a
    b = wgs.b
    e2 = wgs.e2
    eprime2 = wgs.eprime2

    # usefull
    a2 = a**2
    b2 = b**2
    e4 = e2**2
    X2 = X**2
    Y2 = Y**2
    Z2 = Z**2

    r = np.sqrt(X2 + Y2)
    r2 = r**2

    E2 = a2 - b2
    F = 54. * b2 * Z2
    G = r2 + (1.-e2)*Z2 - (e2*E2)
    C = (e4 * F * r2) / (G**3)
    S = np.cbrt(1. + C + np.sqrt(C**2 + 2.*C))
    P = F / (3. * (S+(1./S)+1.)**2 * G**2)
    Q = np.sqrt(1. + (2.*e4*P))

    dummy1 = -(P*e2*r) / (1.+Q)
    dummy2 = 0.5 * a2 * (1.+(1./Q))
    dummy3 = (P*(1.-e2)*Z2) / (Q*(1.+Q))
    dummy4 = 0.5 * P * r2
    r0 = dummy1 + np.sqrt(dummy2-dummy3-dummy4)

    U = np.sqrt((r-(e2*r0))**2 + Z2)
    V = np.sqrt((r-(e2*r0))**2 + (1.-e2)*Z2)
    Z0 = (b2*Z) / (a*V)

    alt = U * (1. - (b2/(a*V)))
    lat = np.arctan((Z+eprime2*Z0) / r)
    lon = np.arctan2(Y, X)

    return (np.degrees(lat), np.degrees(lon), alt)


class FlinnEngdahl(object):
    fe_seismic_regions_filename = data_file('FlinnEngdahl_seismic.pickle')

    def __init__(self):
        self.fe_seismic_regions = load_pickle(self.fe_seismic_regions_filename)

    def get_seismic_region_id(self, lat, lon):
        if lat < -90. or lat > 90.:
            raise ValueError

        if lon < -180. or lon > 180.:
            raise ValueError

        p = Point(lon, lat)
        for srid, srpoly_list in self.fe_seismic_regions.items():
            for srpoly in srpoly_list:
                if p.within(srpoly):
                    return srid


__all__ = """
    cos_gc_distance
    gc_distance
    gc_azimuth
    gc_azibazi
    ellipsoid_distance
    geodetic_to_ecef
    ecef_to_geodetic
""".split()
