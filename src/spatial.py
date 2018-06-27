# -*- coding: utf-8 -*-

from __future__ import division
import math

import numpy as np
from numpy.linalg import norm
from scipy.spatial import cKDTree

from pyrocko.model import Event as PyrockoEvent

from .geodetic import geodetic_to_ecef, DEG2RAD
from .ie.quakeml import Event as QmlEvent
from .meta import ScoterError
from .stats import biweight


# Epsilon value.
FLOAT_EPS = np.finfo(np.float).eps

# KDTree leaf size.
LEAF_SIZE = 40


def build_ecef_kdtree(event_list):
    """
    This function gets a set of geographical points, converts them to
    ECEF Cartesian coordinates, and returns as a KDTree.

    Parameters
    ----------
    event_list : list
        List of geographical data points to be indexed which  are
        instances of one of the followings:

        - :class:`pyrocko.model.Event`
        - :class:`scoter.ie.quakeml.Event`

        Each object should have the following attributes:

        - latitude in [deg],
        - longitude in [deg],
        - depth in [m].

    Returns
    -------
    kdtree : :class:`scipy.spatial.ckdtree.cKDTree`
        The KDTree of the ECEF Cartesian coordinates of the `event`
        locations that can be used for fast high-dimensional
        nearest-neighbor searches.
    """

    npoints = len(event_list)
    data = np.zeros((npoints, 3), dtype=np.float)

    a = event_list[0]

    if isinstance(a, PyrockoEvent):
        for i, event in enumerate(event_list):
            x, y, z = geodetic_to_ecef(event.lat, event.lon, -1.*event.depth)
            data[i, :] = (x, y, z)

    elif isinstance(a, QmlEvent):
        for i, event in enumerate(event_list):
            x, y, z = geodetic_to_ecef(
                event.pyrocko_event.lat,
                event.pyrocko_event.lon,
                event.pyrocko_event.depth * -1.)

            data[i, :] = (x, y, z)
    else:
        raise ScoterError(
            'events are not of type neither pyrocko nor quakeml event')

    kdtree = cKDTree(data, leafsize=LEAF_SIZE)
    return kdtree


def ray_takeoff_direction(OC, OR, takeoff_angle):
    """
    Returns the ray takeoff angle direction (vector).

    Parameters
    ----------
    OC : array-like of shape (3,)
        The ECEF Cartesian coordinate vector of the target event.
    OR : array-like of shape (3,)
        The ECEF Cartesian coordinate vector of the receiver (station).
    takeoff_angle : float
        Ray takeoff angle in [deg] and in 0-180 range.

    Returns
    -------
    CD : array-like of shape (3,)
        The ray takeoff-angle direction in ECEF Cartesian coordinate
        system.

    All coordinates must be in ECEF Cartesian coordinate system.
    In this coordinate system:
        - O is the Earth center.
        - C is the target event.
        - R is the receiver point.
        - D is the intersection of ray takeoff direction and OR.
    """

    OC = np.asarray(OC)
    OR = np.asarray(OR)

    OC_len = norm(OC, ord=2)
    OR_len = norm(OR, ord=2)

    # The angle between target point vector and receiver vector.
    gamma = math.acos(np.dot(OC, OR) / (OC_len*OR_len))

    beta = np.pi - (takeoff_angle*DEG2RAD + gamma)
    OD_len = OC_len * math.sin(takeoff_angle*DEG2RAD) / math.sin(beta)
    frac = OD_len / OR_len
    OD = frac * OR
    CD = OD - OC

    return CD


def opening_angle(CD, CN):
    """
    Get the angle between relative position vector of an event pair and
    ray takeoff direction.

    Parameters
    ----------
    CD : array-like of shape (3,)
        The ray takeoff direction (vector) in ECEF Cartesian coordinate
        system.
    CN : array-like of shape (3,)
        The relative position vector of an event pair in ECEF Cartesian
        coordinate system.

    Returns
    -------
    alpha : float
        The *acute* angle between two vectors in [deg].

    All coordinates must be in ECEF Cartesian coordinate system.
    """

    CD = np.asarray(CD)
    CN = np.asarray(CN)

    if norm(CN) == 0.:
        return 0.

    alpha = math.acos(np.dot(CD, CN) / (norm(CD)*norm(CN)))
    if alpha > np.pi/2.:
        alpha = np.pi - alpha

    return math.degrees(alpha)


def get_w_d(d, cutoff, a=3, b=3):
    """
    Calculates interevent distance weight using a biweight function.

    The weight value is a function of only interevent distance.

    Parameters
    ----------
    d : float or array-like of floats
        Hypocentral separation distance(s) of events.
    cutoff : float
        Cutoff distance value (maximum separation).
    a, b : int
        Exponents defining the shape of the weighting curve (default:
        bicube function, i.e. a=b=3).

    Returns
    -------
    w_d : float or :class:`numpy.ndarray` of floats
        Distance weight value(s).

    Note
    ----
    The units of `d` and `cutoff` should be the same (i.e. both in
    meters, kilometers etc).
    """

    w_d = biweight(d, cutoff, a, b)
    return w_d


def get_w_ed(w_d, alpha):
    """
    Calculate effective interevent distance weight.

    Each weight value is a function of both the distance between
    hypocenters and the angle between relative position vector of the
    event pair and ray takeoff direction computed at one of the events
    (called target event).

    Parameters
    ----------
    w_d : float or array-like of floats
        Distance weight value(s). See :func:`distance_weight`
    alpha : float (min: 0., max: 90.)
        Acute angle between relative position vector of an event pair
        and ray takeoff direction in [deg] computed at one of the events.

    Returns
    -------
    w_ed : float or :class:`numpy.ndarray` of floats
        Effective distance weight value(s).

    Note
    ----
    The units of `d` and `cutoff` should be the same (i.e. both in
    meter, kilometer etc).
    """

    if not 0. <= alpha <= 90.:
        raise ScoterError(
            'inappropriate value for alpha. '
            'Should be in 0-90 degrees range: {}'.format(alpha))

    w_ed = 1. - ((1.-w_d) * (alpha/90.))
    return w_ed


def find_nearest_neighbors(target_event, kdtree, r_max, nparallel=1):
    """
    Find all the nearest neighbors of `target_event` located within
    distance `r_max` by searching against `kdtree`.

    Parameters
    ----------
    target_event : k-tuple or array-like of shape (k,)
        The Cartesian coordinates of the point to search for neighbors
        of (k equals 2 or 3).

    kdtree : :class:`scipy.spatial.ckdtree.cKDTree` instance
        kd-tree of the Cartesian coordinates of the data points used for
        quick nearest-neighbor lookup. This is a set of k-dimensional
        points (k equals 2 or 3) which can be used to rapidly look up
        the nearest neighbors of any point (i.e. event).

    r_max : float
        Search radius. Points located within a circle (2D) or a sphere
        (3D) of radius `r_max` around the `target` are returned.

    nparallel : int, optional (default: 1)
        Number of jobs to schedule for parallel processing. If -1 is
        given all processors are used.

    Returns
    -------
    idxs : :class:`numpy.ndarray` of ints, shape=(n_ngh,)
        The location of the neighbors in `kdtree`

    relpos_vectors : :class:`numpy.ndarray` of floats, shape=(n_ngh, k)
        The position vector of the neighbor events relative to the
        target event (can be used to compute the interevent distance).
    """

    # To include neighbors exactly located at distance `r_max`.
    r_max_dummy = r_max + (3 * FLOAT_EPS)

    # The indices of the neighbor points in data.
    idxs = kdtree.query_ball_point(
        target_event, r_max_dummy, p=2, n_jobs=nparallel)

    if not idxs:
        return (np.array([]), np.array([]))

    idxs = np.asarray(idxs)
    relpos_vectors = kdtree.data[idxs] - target_event

    return (idxs, relpos_vectors)


__all__ = """
    build_ecef_kdtree
    ray_takeoff_direction
    opening_angle
    get_w_d
    get_w_ed
    find_nearest_neighbors
""".split()
