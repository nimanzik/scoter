# -*- coding: utf-8 -*-

from collections import defaultdict, namedtuple, OrderedDict
from itertools import chain
import os.path as op
from os.path import join as pjoin
import shutil
import sys

from matplotlib import pyplot as plt, gridspec, ticker
import numpy as np

from pyrocko import guts, model
from pyrocko.parimap import parimap
from pyrocko.util import ensuredir, ensuredirs, time_to_str

from .core import NAMED_STEPS, HYP_FILE_EXTENSION
from .geodetic import DEG2M, M2KM, KM2M
from .ie.nlloc import load_nlloc_hyp
from .meta import FileNotFound, PathAlreadyExists, ScoterError
from .parmap import parstarmap
from .stats import mad, smad_normal
from .util import data_file, dump_pickle, load_pickle


DELIMITER_STR = '.'


np.set_printoptions(nanstr='NaN')
plt.style.use(['ggplot', data_file('scoter.mplstyle')])

ScoterEvent = namedtuple('ScoterEvent', 'lat lon depth time rms len3 res_dict')


def _load_config(rundir):
    fn = pjoin(rundir, 'used_config.yaml')
    return guts.load(filename=fn)


def _load_one_event(fns, *args):
    es = []
    for fn in fns:
        e = load_nlloc_hyp(fn, *args).event_parameters.event_list[0]
        o = e.preferred_origin
        u = o.origin_uncertainty_list[0]

        res_dict = dict((k, v[::2]) for k, v in e.arrival_maps.iteritems())

        dummy_event = ScoterEvent(
            lat=o.latitude.value, lon=o.longitude.value, depth=o.depth.value,
            time=o.time.value, rms=o.quality.standard_error,
            len3=u.confidence_ellipsoid.semi_major_axis_length,
            res_dict=res_dict)

        es.append((e.name, dummy_event))

    return tuple(es)


def _load_cached_data(rundir, step):
    infile = pjoin(
        rundir, 'harvest', 'results_{}.pickle'.format(NAMED_STEPS[step]))

    if not op.exists(infile):
        raise FileNotFound("No such file or directory: '{}'".format(infile))

    return load_pickle(infile)


# =============================================================================


def harvest(
        rundir, force=False, nparallel=1, show_progress=False, weed=False,
        last_iter=True):

    config = _load_config(rundir)
    sconfig = config.station_terms_config

    dumpdir = pjoin(rundir, 'harvest')

    if op.exists(dumpdir):
        if force:
            shutil.rmtree(dumpdir)
        else:
            raise PathAlreadyExists(
                'directory already exists: "{}"'.format(dumpdir))

    ensuredir(dumpdir)

    pyrocko_events = model.load_events(
        config.expand_path(config.dataset_config.events_path), extra=None)

    event_names = [ev.name for ev in pyrocko_events]
    basename_tmpl = '%(event_name)s%(ext)s'

    niters = {
        'A': 1,
        'B': sconfig.static_config.niter,
        'C': sconfig.ssst_config.niter}

    niter_strlens = {
        'A': 0,
        'B': sconfig.static_config._niter_strlen,
        'C': sconfig.ssst_config._niter_strlen}

    heads = dict((step, pjoin(rundir, NAMED_STEPS[step])) for step in 'ABC')

    tail_templates = {
        'A': '',
        'B': '{iiter:0{strlen:d}d}_2_loc',
        'C': '{iiter:0{strlen:d}d}_2_loc'}

    resultdirs = defaultdict(list)
    for step in 'ABC':
        head = heads[step]
        if not op.exists(head):
            niters[step] = 0
            continue

        niter = niters[step]
        strlen = niter_strlens[step]
        for iiter in range(1, niter+1):
            tail_template = tail_templates[step]
            tail = tail_template.format(
                iiter=iiter, strlen=strlen)
            resultdirs[step].append(pjoin(head, tail))

    # Dictionary sorted by key (A, B, C)
    resultdirs = OrderedDict(sorted(resultdirs.items(), key=lambda t: t[0]))

    task_list = []
    if last_iter is True:
        # Extract only last iteration of iterative steps
        if weed is True:
            # Pick events remained until last iteration
            for event_name in event_names:
                fns = []
                for resultdir_list in resultdirs.values():
                    resultdir = resultdir_list[-1]
                    fn = pjoin(resultdir, basename_tmpl % dict(
                        event_name=event_name,
                        ext=HYP_FILE_EXTENSION))
                    fns.append(fn)

                if not all([op.exists(x) for x in fns]):
                    continue

                task_list.append((
                    tuple(fns),
                    event_name,
                    DELIMITER_STR,
                    True))
        else:
            for event_name in event_names:
                fns = []
                for resultdir_list in resultdirs.values():
                    dummy = []
                    for resultdir in resultdir_list[-1::-1]:
                        fn = op.join(resultdir, basename_tmpl % dict(
                            event_name=event_name,
                            ext=HYP_FILE_EXTENSION))

                        if op.exists(fn):
                            dummy.append(fn)
                            break
                        else:
                            dummy.append(None)

                    dummy = filter(None, dummy)

                    if len(dummy) == 0:
                        fns.append(None)
                    else:
                        fns.extend(dummy)

                if not all(fns):
                    continue

                task_list.append((
                    tuple(fns),
                    event_name,
                    DELIMITER_STR,
                    True))

        # r = [((N1_A,E1_A), (N1_C,E1_C)), ((N2_A,E2_A), (N2_C,E2_C)), ...]
        # N: event_name; E: dummy_event; A,C: location steps
        results = parstarmap(
            _load_one_event,
            task_list,
            nparallel=nparallel,
            show_progress=show_progress,
            label='Loading location files')

        # r = [((N1_A,E1_A),(N2_A,E2_A), ...), ((N1_C,E1_C),(N2_C,E2_C), ...)]
        # N: event_name; E: dummy_event; A,C: location steps
        results = zip(*results)

        for iresult, result in enumerate(results):
            step = resultdirs.keys()[iresult]
            data = {niters[step]: dict(result)}
            fn = pjoin(dumpdir, 'results_{}.pickle'.format(NAMED_STEPS[step]))
            dump_pickle(data, fn)
    else:
        # Extract all iterations in iterative steps
        if weed is True:
            # Pick events remained until last iteration
            for event_name in event_names:
                fns = []
                for resultdir in chain.from_iterable(resultdirs.values()):
                    fn = pjoin(resultdir, basename_tmpl % dict(
                        event_name=event_name,
                        ext=HYP_FILE_EXTENSION))
                    fns.append(fn)

                if not all([op.exists(x) for x in fns]):
                    continue

                task_list.append((
                    tuple(fns),
                    event_name,
                    DELIMITER_STR,
                    True))
        else:
            for event_name in event_names:
                fns = []
                for resultdir_list in resultdirs.values():
                    dummy = []
                    for resultdir in resultdir_list:
                        fn = op.join(resultdir, basename_tmpl % dict(
                            event_name=event_name,
                            ext=HYP_FILE_EXTENSION))

                        if op.exists(fn):
                            dummy.append(fn)
                        else:
                            try:
                                dummy.append(dummy[-1])
                            except IndexError:
                                dummy.append(None)

                    fns.extend(dummy)

                if not all(fns):
                    continue

                task_list.append((
                    tuple(fns),
                    event_name,
                    DELIMITER_STR,
                    True))

        # r = [((N1_A,E1_A),(N1_C1,E1_C1),(N1_C2,E1_C2),...),
        #      ((N2_A,E2_A),(N2_C1,E2_C1),(N2_C2,E2_C2),...), ...]
        results = parstarmap(
            _load_one_event,
            task_list,
            nparallel=nparallel,
            show_progress=show_progress,
            label='Loading location files')

        # r = [((N1_A,E1_A), (N2_A,E2_A),...),
        #      ((N1_C1,E1_C1), (N2_C1,E2_C1),...),
        #      ((N1_C2,E1_C2), (N2_C2,E2_C2),...), ...]
        results = zip(*results)

        i1 = 0
        for step in 'ABC':
            i2 = niters[step]

            if i2 == 0:
                continue

            result_list = results[i1:i1+i2]
            i1 += i2

            data = OrderedDict()
            for i_result, result in enumerate(result_list):
                data[i_result+1] = dict(result)

            fn = pjoin(dumpdir, 'results_{}.pickle'.format(NAMED_STEPS[step]))
            dump_pickle(data, fn)


def export_static(rundir, filename=None):
    config = _load_config(rundir)
    sconfig = config.station_terms_config

    infile = pjoin(rundir, NAMED_STEPS['B'], '{:02}_1_delay.pickle'.format(
        sconfig.static_config.niter))

    if not op.exists(infile):
        raise FileNotFound("No such file or directory: '{}'".format(infile))

    delays = load_pickle(infile)

    if filename is None:
        out = sys.stdout
    else:
        ensuredirs(filename)
        out = open(filename, 'w')

    print >>out, ('# station label, lat, lon; phase label; time correction')

    tmpl = '{slabel:8s} {lat:9.5f} {lon:10.5f} {plabel:6s} {tcor:8.4f}'

    for delay in delays:
        if delay.phase_label not in sconfig.static_config.phase_list:
            continue

        sta = [x for x in config.stations if x.name == delay.station_label][0]

        print >>out, tmpl.format(
            slabel=sta.name, lat=sta.lat, lon=sta.lon,
            plabel=delay.phase_label, tcor=delay.time_correction)

    if out is not sys.stdout:
        out.close()


def export_ssst(rundir, station_label, phase_label, filename=None):
    config = _load_config(rundir)
    sconfig = config.station_terms_config

    infile_d = pjoin(
        rundir,
        NAMED_STEPS['C'],
        '{:02}_1_delay.pickle'.format(sconfig.ssst_config.niter))

    infile_e = pjoin(rundir, 'harvest', 'results_C_ssst.pickle')

    err_msgs = []
    for infile in (infile_d, infile_e):
        if not op.exists(infile_d):
            err_msgs.append(infile)

    if err_msgs:
        raise FileNotFound("No such file or directory: '{}'".format(
            ', '.join(err_msgs)))

    phase_labels = set()
    phase_labels.add(phase_label)
    if phase_label in sconfig.ssst_config.phase_map:
        std_phase = sconfig.ssst_config.phase_map[phase_label]
        phase_labels.update(sconfig.ssst_config.phase_map_rev[std_phase])

    # a dictionary mapping event names to station delays
    event_to_delays = load_pickle(infile_d)

    # a dictionary mapping event names to hypocenter params
    event_to_hypo = load_pickle(infile_e)

    if filename is None:
        out = sys.stdout
    else:
        ensuredirs(filename)
        out = open(filename, 'w')

    print >>out, ('# event lat, lon, depth; phase label; time correction')

    tmpl = '{lat:9.5f} {lon:10.5f} {depth:9.5f} {plabel:6s} {tcor:8.4f}'

    for event_name, delays in event_to_delays.iteritems():
        delays_want = [
            x for x in delays if x.station_label == station_label and
            x.phase_label in phase_labels]

        if not delays_want:
            continue

        event = event_to_hypo[event_name]

        for delay in delays_want:
            print >>out, tmpl.format(
                lat=event.lat, lon=event.lon, depth=event.depth/KM2M,
                plabel=delay.phase_label, tcor=delay.time_correction)

    if out is not sys.stdout:
        out.close()


def export_residuals(rundir, station_label, phase_label, filename=None):
    config = _load_config(rundir)

    # Mapping of phase labels
    phase_labels = set()
    phase_labels.add(phase_label)
    for pid in config.nlloc_config.phaseid_list:
        if ((phase_label == pid.std_phase) or
                (phase_label in pid.phase_code_list)):
            # Get the phase
            phase_labels.update(pid.phase_code_list)

    if filename is None:
        out = sys.stdout
    else:
        ensuredirs(filename)
        out = open(filename, 'w')

    print >>out, '# station label: {}; phase label: {}'.format(
        station_label, phase_label)

    def f(hypo):
        res_one_event = []
        for (sta, pha), (res, _) in hypo.res_dict.iteritems():
            if (sta == station_label) and (pha in phase_labels):
                res_one_event.append(res)
        return res_one_event

    step_to_res = defaultdict(list)
    for step in ['A', 'B', 'C']:
        infile = pjoin(
            rundir, 'harvest', 'results_{}.pickle'.format(NAMED_STEPS[step]))

        if not op.exists(infile):
            continue

        data = load_pickle(infile)

        for event_to_hypo in data.itervalues():
            res_one_iter = []
            for res_one_event in parimap(f, event_to_hypo.itervalues()):
                res_one_iter.extend(res_one_event)
            step_to_res[step].append(res_one_iter)

    # Dictionary sorted by key (A, B, C)
    step_to_res = OrderedDict(sorted(step_to_res.items(), key=lambda t: t[0]))

    ncols = sum([len(x) for x in step_to_res.itervalues()])
    nrows = max([len(y) for x in step_to_res.itervalues() for y in x])

    a = np.empty((nrows, ncols), dtype=np.float)
    a[:] = np.NaN

    j = 0
    for step, res_lists in step_to_res.iteritems():
        for res_list in res_lists:
            i = len(res_list)
            a[:i, j] = res_list
            j += 1

    hdr = 'location steps: {}'.format(
        ', '.join([NAMED_STEPS[x] for x in step_to_res.keys()]))

    np.savetxt(out, a, fmt='%8.4f', header=hdr, comments='# ')

    if out is not sys.stdout:
        out.close()


def export_events(rundir, step, i_iter=-1, fmt='columns', filename=None):

    if fmt not in ['pyrocko', 'columns']:
        raise ScoterError('unsupported events file format: "{}"'.format(fmt))

    step = step.upper()
    data = _load_cached_data(rundir, step)

    if len(data) == 1:
        event_to_hypo = data.values()[0]
    else:
        i_iter = i_iter <= 0 and len(data) or i_iter

        try:
            event_to_hypo = data[i_iter]
        except KeyError:
            raise ScoterError(
                'invalid iteration number for step {}'.format(step))

    event_to_hypo = OrderedDict(
        sorted(event_to_hypo.items(), key=lambda t: t[0]))

    if filename is None:
        out = sys.stdout
    else:
        ensuredirs(filename)
        out = open(filename, 'w')

    if fmt == 'columns':
        print >>out, ('# step {}; event name, origin time, lat, lon, depth, '
                      'rms, semi-major axis length'.format(NAMED_STEPS[step]))

        tmpl = '{name:15s} {time:s} {lat:9.5f} {lon:10.5f} {depth:9.5f} '\
               '{rms:7.4f} {len3:8.4f}'

        for event_name, hypo in event_to_hypo.iteritems():
            print >>out, tmpl.format(
                name=event_name, time=time_to_str(hypo.time),
                lat=hypo.lat, lon=hypo.lon, depth=hypo.depth/KM2M,
                rms=hypo.rms, len3=hypo.len3/KM2M)

    else:
        pyrocko_events = []
        for event_name, hypo in event_to_hypo.iteritems():
            event = model.Event(
                name=event_name, lat=hypo.lat, lon=hypo.lon, depth=hypo.depth,
                time=hypo.time)

            pyrocko_events.append(event)

        model.dump_events(pyrocko_events, stream=out)

    if out is not sys.stdout:
        out.close()


def plot_convergence(
        rundir, step, statistic='SMAD', save=False, fmts=('pdf',), dpi=120.):
    """
    Plot convergence curve for iteerative steps.

    Parameters
    ----------
    rundir : str
        Full or relative path and name for run directory, where the
        output files and cached results are stored.
    step : str {B', 'C'}
        Iterative named location step in SCOTER syntax.
    statistic : str {'SMAD', 'MAD'}
        Measure of statistical dispersion.
    save : bool
        Whether to save figure to file (default: False).
    fmts : list of str
        List of output formats (default: ['pdf']).
    dpi : float
        DPI setting for raster format (default: 120).
    """

    step = step.upper()
    if step.upper() == 'A':
        raise ScoterError(
            'convergence curve should be plotted for iterative steps')

    if statistic.upper() not in ('MAD', 'SMAD'):
        raise ValueError("invalid statistic: '{}'".format(statistic))

    if isinstance(fmts, str):
        fmts = fmts.split(',')

    for fmt in fmts:
        if fmt not in ['pdf', 'png']:
            raise ScoterError("unavailable output format: '{}'".format(fmt))

    data = _load_cached_data(rundir, step)
    conf = _load_config(rundir)

    if step == 'B':
        conf_s = conf.station_terms_config.static_config
    else:
        conf_s = conf.station_terms_config.ssst_config

    phase_labels = set()
    for phase in conf_s.phase_list:
        phase_labels.add(phase)
        std_phase = conf_s.phase_map.get(phase, None)
        phase_labels.update(conf_s.phase_map_rev.get(std_phase, None) or [])

    def f(i_iter):
        event_to_hypo = data[i_iter]
        res_list = []
        for hypo in event_to_hypo.itervalues():
            for (sta, pha), (res, _) in hypo.res_dict.iteritems():
                if pha in phase_labels:
                    res_list.append(res)

        if statistic.upper() == 'MAD':
            return (i_iter, mad(res_list))
        return (i_iter, smad_normal(res_list))

    r = parimap(f, data.keys())
    x, y = zip(*r)

    fig = plt.figure(figsize=(11, 10))
    ax = fig.add_subplot(111)
    ax.plot(x, y, '-o',)
    ax.set_xlabel('Iteration Number')
    ax.set_ylabel('Travel-Time Residual {} [s]'.format(statistic))
    # Set xtick labels to integers
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    if save:
        for fmt in fmts:
            fname = 'convergence_curve_{}.{}'.format(NAMED_STEPS[step], fmt)
            fig.savefig(fname, dpi=dpi)
        plt.close()
    else:
        plt.show()


def plot_residuals(
        rundir, steps, phase_label, dmin, dmax, dd, rmin, rmax, dr,
        interpolation='nearest', cmap='plasma', save=False, fmts=('pdf',),
        dpi=120.):
    """
    Plot travel-time residual heat-map for the given phase.
    For iterative steps, the residuals from the last iteration are used.

    Parameters
    ----------
    rundir : str
        Full or relative path and name for run directory, where the
        output files and cached results are stored.
    steps : list of str
        Named location steps in SCOTER syntax ('A', 'B', 'C').
    phase_label : str
        Phase label for which the residuals are plotted.
    dmin : float
        Minimumin distance either [deg] (GLOBAL mode) or [km]
        (Non-GLOBAL mode).
    dmax : float
        Maximum distance in either [deg] (GLOBAL mode) or [km]
        (Non-GLOBAL mode).
    dd : float
        Distance bins width in either [deg] (GLOBAL mode) or [km]
        (Non-GLOBAL mode).
    rmin : float
        Minimumin travel-time residual in [s].
    rmax : float
        Maximum travel-time residual in [s].
    dr : float
        Residual bins width in [s].
    interpolation : str
        One of :class:`matplotlib.pyplot.imshow` acceptable values for
        interpolation (default: 'nearest').
    cmap : str
        One of :class:`matplotlib.pyplot.imshow` acceptable values for
        cmap (default: 'plasma').
    save : bool
        Whether to save figure to file  (default: False).
    fmts : list of str
        List of output formats (default: ['pdf']).
    dpi : float
        DPI setting for raster format (default: 120).
    """
    config = _load_config(rundir)

    steps = sorted(steps)

    if isinstance(fmts, str):
        fmts = fmts.split(',')

    for fmt in fmts:
        if fmt not in ['pdf', 'png']:
            raise ScoterError("unavailable output format: '{}'".format(fmt))

    # Distance conversion coefficient
    if config.nlloc_config.trans.trans_type == 'GLOBAL':
        convcoef = 1
        xunit = 'deg'
    else:
        convcoef = DEG2M * M2KM
        xunit = 'km'

    # Mapping of phase labels
    phase_labels = set()
    phase_labels.add(phase_label)
    for pid in config.nlloc_config.phaseid_list:
        if ((phase_label == pid.std_phase) or
                (phase_label in pid.phase_code_list)):
            # Get the phase
            phase_labels.update(pid.phase_code_list)

    def f(hypo):
        res_dist_one_event = []
        for (_, pha), (res, dist) in hypo.res_dict.iteritems():
            if pha in phase_labels:
                res_dist_one_event.append((res, dist*convcoef))
        return res_dist_one_event

    step_to_res_dist = {}
    for step in steps:
        # Load data from last iteration
        data = _load_cached_data(rundir, step)
        event_to_hypo = data[len(data)]

        res_dist_one_step = []
        for rd in parimap(f, event_to_hypo.itervalues()):
            res_dist_one_step.extend(rd)

        step_to_res_dist[step] = np.asarray(res_dist_one_step)

    # --- Sort by keys, i.e. A, B, C ---
    step_to_res_dist = OrderedDict(
        sorted(step_to_res_dist.items(), key=lambda t: t[0]))

    # --- Plot the residuals ---
    n_axes = len(step_to_res_dist)

    xbins = np.arange(dmin, dmax, dd)
    ybins = np.arange(rmin, rmax, dr)

    fig = plt.figure(figsize=(n_axes*11, 10))
    gs = gridspec.GridSpec(1, n_axes+1, width_ratios=[10]*n_axes+[0.5])
    gs.update(wspace=0.15)

    for i_ax, (step, res_dist) in enumerate(step_to_res_dist.items()):
        ax = fig.add_subplot(gs[i_ax])

        H, _, _ = np.histogram2d(
            res_dist[:, 0], res_dist[:, 1], bins=(ybins, xbins))

        for i_col in range(H.shape[1]):
            H[:, i_col] /= np.max(H[:, i_col])

        cimg = ax.imshow(
            H, extent=[dmin, dmax, rmin, rmax], interpolation=interpolation,
            origin='lower', cmap=cmap, aspect='auto')

        if i_ax == 0:
            ax.set_ylabel('Travel-Time Residual [s]')

        ax.set_xlabel('Distance [{}]'.format(xunit))
        ax.grid(False)

    cax = fig.add_subplot(gs[-1])
    cbar = fig.colorbar(cimg, cax=cax)
    cbar.set_label('Frequency of Occurance (Normalized)')
    cax.yaxis.set_label_position('left')

    if save:
        for fmt in fmts:
            fname = 'residuals_catalog_{}.{}'.format(phase_label, fmt)
            fig.savefig(fname, dpi=dpi)
        plt.close()
    else:
        plt.show()
