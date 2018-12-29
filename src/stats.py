# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import norm


def mad(a, axis=None):
    """
    Calculate the median absolute deviation (MAD) of an array along
    given axis.

    The MAD is defined as `median(abs(a - median(a)))`.

    Parameters
    ----------
    a : array-like
        Input array or object that can be converted to an array.

    axis : {int, sequence of int, None}, optional
        Axis along which the MADs are computed. The default (`None`) is
        to compute the MAD of the flattened array.

    Returns
    -------
    mad : float or :class:`numpy.ndarray`
        The median absolute deviation of the input array. If `axis` is
        `None` then a scalar will be returned, otherwise a
        :class:`numpy.ndarray` will be returned.
    """

    a = np.asarray(a)

    # Median along given axis, but *keeping* the reduced axis so that
    # result can still broadcast against a.
    med = np.nanmedian(a, axis=axis, keepdims=True)

    # MAD along given axis
    return np.nanmedian(np.absolute(a - med), axis=axis)


def smad(a, axis=None):
    """
    Calculate a robust standard deviation using the median absolute
    deviation (MAD) of an array along given axis.

    It is also called scaled median absolute deviation (SMAD).

    The standard deviation estimator is given by:

    .. math::

        \sigma \approx \frac{\textrm{MAD}}{\Phi^{-1}(3/4)}

    where :math:`\Phi^{-1}(P)` is the constant scale factor, which
    depends on the distribution and evaluated at probability
    :math:`P = 3/4`.

    Parameters
    ----------
    a : array-like
        Input array or object that can be converted to an array.
    axis : {int, sequence of int, None}, optional
        Axis along which the robust standard deviations are computed.
        The default (`None`) is to compute the robust standard deviation
        of the flattened array.

    Returns
    -------
    smad : float or :class:`numpy.ndarray`
        The robust standard deviation of the input data.  If `axis` is
        `None` then a scalar will be returned, otherwise a
        :class:`numpy.ndarray` will be returned.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Median_absolute_deviation
    """

    loc, scale = norm.fit(a)
    kappa = 1. / np.percentile((a-loc)/scale, 75)
    return kappa * mad(a, axis=axis)


def smad_normal(a, axis=None):
    """
    Calculate a robust standard deviation using the median absolute
    deviation (MAD) of a normally distributed data along given axis.

    It is also called scaled median absolute deviation (SMAD).

    The standard deviation estimator is given by:

    .. math::

        \sigma \approx \frac{\textrm{MAD}}{\Phi^{-1}(3/4)}
            \approx 1.4826 \ \textrm{MAD}

    where :math:`\Phi^{-1}(P)` is the normal inverse cumulative
    distribution function evaluated at probability :math:`P = 3/4`.

    Parameters
    ----------
    a : array-like
        Input array or object that can be converted to an array.
    axis : {int, sequence of int, None}, optional
        Axis along which the robust standard deviations are computed.
        The default (`None`) is to compute the robust standard deviation
        of the flattened array.

    Returns
    -------
    smad : float or :class:`numpy.ndarray`
        The robust standard deviation of the input data.  If `axis` is
        `None` then a scalar will be returned, otherwise a
        :class:`numpy.ndarray` will be returned.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Median_absolute_deviation
    """

    # 1.0 / scipy.stats.norm.ppf(3./4.) = 1.482602218505602
    kappa = 1.482602218505602
    return kappa * mad(a, axis=axis)


def biweight(data, cutoff, a, b):
    """
    Biweight function.

    Parameters
    ----------
    data : float or array-like of floats
        Input data.
    cutoff : float
        Cutoff value.
    a, b : int
        Exponents defining the shape of the weighting curve.

    Returns
    -------
    w_d : float or :class:`numpy.ndarray` of floats
        Weight value(s).
    """

    x = np.power(np.true_divide(data, cutoff), int(a))
    y = 1. - x
    w = np.power(np.maximum(y, 0.), int(b))
    return w


def bisquared(data, cutoff):
    """
    Bisquare function.

    To find more about input arguments see :func:`biweight`
    """

    return biweight(data, cutoff, a=2, b=2)


def bicubic(data, cutoff):
    """
    Bicube function.

    To find more about input arguments see :func:`biweight`
    """

    return biweight(data, cutoff, a=3, b=3)


def robust_average(a, epsilon=1.0, axis=None, weights=None, n_iter=10):
    """
    Huber's norm estimator.

    Parameters
    ----------
    a : array-like
        Input array or object that can be converted to an array.
    epsilon : float (optional, default: 1.0)
        Huber's threshold that controls the limit between L1 and L2 norm.
    axis : {int, sequence of int, None}, optional
        Axis along which the averages are computed. The default (`None`) is
        to compute the average of the flattened array.
    weights : array-like (optional)
        An array of weights associated with the values in `a`.
    n_iter : int (optional, default: 10)
        Maximum number of iterations.

    Returns
    -------
    robust_average : array-like or float
    """
    a = np.asarray(a)

    if not np.any(a):
        return np.nan

    if not weights:
        weights = np.ones_like(a)

    for i_iter in range(n_iter-1):
        # Take the average
        avrg = np.average(a, axis=axis, weights=weights, returned=False)
        # Update weights
        weights = 1.0 / (epsilon+np.abs(a-avrg))

    return np.average(a, axis=axis, weights=weights, returned=False)


__all__ = """
    mad
    smad
    biweight
    bisquared
    bicubic
""".split()
