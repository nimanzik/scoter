import cPickle as pickle
import os.path as op
from os.path import join as pjoin

import numpy as np

from pyrocko.util import ensuredirs
import progressbar as pb_mod


WSPACE = ''.ljust(1)

bar_type_A = pb_mod.Bar(marker='-', left='[', right=']')
bar_type_B = pb_mod.Bar(marker=u'\u2588', left='|', right='|')


def loglinspace(start, stop, num):
    """
    Returns evenly spaced values in logarithmic scale.

    The values are evenly spaced over log10(start) and log10(stop)
    interval.
    """
    return np.logspace(
        np.log10(start), np.log10(stop), num=num, endpoint=True, base=10.)


def progressbar(label=None, max_value=None, redirect_stdout=True):
    widgets = [
        WSPACE*2, label or '', WSPACE, pb_mod.Percentage(), WSPACE,
        '(', pb_mod.SimpleProgress(), ')', WSPACE,
        bar_type_A, WSPACE, pb_mod.Timer(), WSPACE]

    pbar = pb_mod.ProgressBar(
        widgets=widgets,
        max_value=max_value,
        redirect_stdout=redirect_stdout)

    return pbar


def data_file(fn):
    return pjoin(op.split(__file__)[0], 'data', fn)


def dump_pickle(obj, fn):
    ensuredirs(fn)
    with open(fn, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(fn):
    with open(fn, 'rb') as f:
        return pickle.load(f)


__all__ = """
    loglinspace
    progressbar
""".split()
