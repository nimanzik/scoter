# -*- coding: utf-8 -*-

from itertools import imap, izip, repeat, starmap
from multiprocessing import Pool, cpu_count

from .util import progressbar


def _func_support_many(x):
    function, args = x[0], x[1:]
    return function(*args)


def _func_support_one(x):
    function, args = x
    return function(*args)


def _get_common_kwargs(kwargs):
    assert all(
        k in ('nparallel', 'chunksize', 'show_progress', 'label')
        for k in kwargs.keys())

    nparallel = kwargs.get('nparallel', None)
    chunksize = kwargs.get('chunksize', None)
    show_progress = kwargs.get('show_progress', False)
    label = kwargs.get('label', None)

    return (nparallel, chunksize, show_progress, label)


def _get_default_nparallel(nparallel):
    if nparallel is None or nparallel <= 0:
        nparallel = cpu_count()
    return nparallel


def _get_default_chunksize(chunksize, pool, ntasks):
    # default from multiprocessing
    if chunksize is None:
        chunksize, extra = divmod(ntasks, len(pool._pool)*4)
        if extra:
            chunksize += 1
    return chunksize


def parmap(function, *iterables, **kwargs):
    """
    Examples
    --------
    >>> a = (2, 3, 10, 5)
    >>> b = (5, 2, 4, 3)
    >>> parmap(pow, a, b, nparallel=4, show_progress=True, label='Test')
    (32, 9, 10000, 125)
    """

    nparallel, chunksize, show_progress, label = _get_common_kwargs(kwargs)

    ntasks = min(len(it) for it in iterables)
    nparallel = _get_default_nparallel(nparallel)

    if nparallel == 1:
        # Single processing
        result = imap(function, *iterables)
        if show_progress:
            output = []
            pbar = progressbar(label, ntasks)
            for x in pbar(result):
                output.append(x)
        else:
            output = list(result)

        return output

    # Parallel processing
    pool = Pool(processes=nparallel)
    tasks = izip(repeat(function), *iterables)

    try:
        if show_progress:
            # `chunksize` default from multiprocessing
            chunksize = _get_default_chunksize(chunksize, pool, ntasks)

            result = pool.map_async(
                _func_support_many, tasks, chunksize=chunksize)
            pool.close()

            try:
                with progressbar(label, ntasks) as pbar:
                    while True:
                        if result.ready():
                            pbar.update(ntasks)
                            break

                        try:
                            nremains = result._number_left * chunksize
                            ndone = ntasks - nremains
                        except:   # noqa
                            break

                        if ndone > 0:
                            pbar.update(ndone)
            finally:
                output = result.get()
        else:
            output = pool.map(_func_support_many, tasks, chunksize=chunksize)

    finally:
        if not show_progress:
            pool.close()
        pool.join()

    return output


def parstarmap(function, iterable, **kwargs):
    """
    Examples
    --------
    >>> a = (2, 3, 10, 5)
    >>> b = (5, 2, 4, 3)
    >>> q = zip(a, b)
    >>> q
    [(2, 5), (3, 2), (10, 4), (5, 3)]
    >>> parstarmap(pow, q, nparallel=4, show_progress=True, label='Test')
    (32, 9, 10000, 125)
    """

    nparallel, chunksize, show_progress, label = _get_common_kwargs(kwargs)

    ntasks = len(iterable)
    nparallel = _get_default_nparallel(nparallel)

    if nparallel == 1:
        # Single processing
        result = starmap(function, iterable)
        if show_progress:
            output = []
            pbar = progressbar(label, ntasks)
            for x in pbar(result):
                output.append(x)
        else:
            output = list(result)

        return output

    # Parallel processing
    pool = Pool(processes=nparallel)
    tasks = izip(repeat(function), iterable)

    try:
        if show_progress:
            # `chunksize` default from multiprocessing
            chunksize = _get_default_chunksize(chunksize, pool, ntasks)

            result = pool.map_async(
                _func_support_one, tasks, chunksize=chunksize)
            pool.close()

            try:
                with progressbar(label, ntasks) as pbar:
                    while True:
                        if result.ready():
                            pbar.update(ntasks)
                            break

                        try:
                            nremains = result._number_left * chunksize
                            ndone = ntasks - nremains
                        except:   # noqa
                            break

                        if ndone > 0:
                            pbar.update(ndone)
            finally:
                output = result.get()
        else:
            output = pool.map(_func_support_one, tasks, chunksize=chunksize)

    finally:
        if not show_progress:
            pool.close()
        pool.join()

    return output


__all__ = ['parmap', 'parstarmap']
