from collections import defaultdict, namedtuple, OrderedDict
import cPickle as pickle
import os.path as op
from os.path import join as pjoin
import shutil
import sys

import numpy as np

from pyrocko import guts, model
from pyrocko.util import ensuredir, ensuredirs, time_to_str

from .core import NAMED_STEPS, HYP_FILE_EXTENSION
from .ie.nlloc import load_nlloc_hyp
from .meta import FileNotFound, PathAlreadyExists, ScoterError
from .parmap import parstarmap


KM2M = 1000.

np.set_printoptions(nanstr='NaN')


def dump_pickle(obj, fn):
    ensuredirs(fn)
    with open(fn, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(fn):
    with open(fn, 'rb') as f:
        return pickle.load(f)


def load_config(dirname):
    fn = pjoin(dirname, 'used_config.yaml')
    return guts.load(filename=fn)


ScoterEvent = namedtuple('ScoterEvent', 'lat lon depth time rms len3 res_dict')


def _load_one_event(fns, *args):
    es = []
    for fn in fns:
        e = load_nlloc_hyp(fn, *args).event_parameters.event_list[0]
        o = e.preferred_origin
        u = o.origin_uncertainty_list[0]

        res_dict = dict((k, v[0]) for k, v in e.arrival_dict.iteritems())

        dummy_event = ScoterEvent(
            lat=o.latitude.value, lon=o.longitude.value, depth=o.depth.value,
            time=o.time.value, rms=o.quality.standard_error,
            len3=u.confidence_ellipsoid.semi_major_axis_length,
            res_dict=res_dict)

        es.append((e.name, dummy_event))

    return tuple(es)


def harvest(rundir, force=False, nparallel=1, show_progress=False):

    config = load_config(rundir)

    dumpdir = pjoin(rundir, 'harvest')

    if op.exists(dumpdir):
        if force:
            shutil.rmtree(dumpdir)
        else:
            raise PathAlreadyExists(
                'directory already exists: "{}"'.format(dumpdir))

    ensuredir(dumpdir)

    pyrocko_events = model.load_events(op.normpath(pjoin(
        rundir, config.path_prefix, config.dataset_config.events_path)))

    event_names = [ev.name for ev in pyrocko_events]

    resultdir_A = pjoin(rundir, NAMED_STEPS['A'])   # noqa
    resultdir_B = pjoin(   # noqa
        rundir,
        NAMED_STEPS['B'],
        '{:02}_2_loc'.format(config.static_config.niter))
    resultdir_C = pjoin(   # noqa
        rundir,
        NAMED_STEPS['C'],
        '{:02}_2_loc'.format(config.ssst_config.niter))

    resultdirs = {}
    for k in 'A B C'.split():
        v = eval('resultdir_{}'.format(k))
        if op.exists(v):
            resultdirs[k] = v

    # Dictionary sorted by key (A, B, C)
    resultdirs = OrderedDict(sorted(resultdirs.items(), key=lambda t: t[0]))

    task_list = []
    for event_name in event_names:
        fns = []
        for resultdir in resultdirs.values():
            fn = pjoin(resultdir, '%(event_name)s%(ext)s' % dict(
                event_name=event_name,
                ext=HYP_FILE_EXTENSION))
            fns.append(fn)

        if not all([op.exists(x) for x in fns]):
            continue

        task_list.append((
            tuple(fns), event_name, config.dataset_config.delimiter_str, True))

    # results = [((N_1A,E_1A), (N_1C,E_1C)), ((N_2A,E_2A), (N_2C,E_2C)), ...]
    # N: event_name; E: dummy_event; A,C: location steps
    results = parstarmap(
        _load_one_event,
        task_list,
        nparallel=nparallel,
        show_progress=show_progress,
        label='Loading location files')

    # results = [((N_1A,E_1A), (N_2A,E_2A),...), ((N_1C,E_1C), (N_2C,E_2C),..)]
    # N: event_name; E: dummy_event; A,C: location steps
    results = zip(*results)

    for iresult, result in enumerate(results):
        d = dict(result)
        s = resultdirs.keys()[iresult]
        fn = pjoin(dumpdir, 'results_{}.pickle'.format(NAMED_STEPS[s]))
        dump_pickle(d, fn)


def export_static(rundir, filename=None):
    config = load_config(rundir)

    infile = pjoin(
        rundir,
        NAMED_STEPS['B'],
        '{:02}_1_delay.pickle'.format(config.static_config.niter))

    if not op.exists(infile):
        raise FileNotFound('cannot access "{}": no such file'.format(infile))

    delays = load_pickle(infile)

    if filename is None:
        out = sys.stdout
    else:
        ensuredirs(filename)
        out = open(filename, 'w')

    print >>out, ('# station label, lat, lon; phase label; time correction')

    tmpl = '{slabel:8s} {lat:9.5f} {lon:10.5f} {plabel:6s} {tcor:8.4f}'

    for delay in delays:
        if delay.phase_label not in config.static_config.phase_list:
            continue

        sta = [x for x in config.stations if x.name == delay.station_label][0]

        print >>out, tmpl.format(
            slabel=sta.name, lat=sta.lat, lon=sta.lon,
            plabel=delay.phase_label, tcor=delay.time_correction)

    if out is not sys.stdout:
        out.close()


def export_ssst(rundir, station_label, phase_label, filename=None):
    config = load_config(rundir)

    infile_d = pjoin(
        rundir,
        NAMED_STEPS['C'],
        '{:02}_1_delay.pickle'.format(config.ssst_config.niter))

    infile_e = pjoin(rundir, 'harvest', 'results_C_ssst.pickle')

    err_msgs = []
    for infile in (infile_d, infile_e):
        if not op.exists(infile_d):
            err_msgs.append(infile)

    if err_msgs:
        raise FileNotFound('cannot access "{}": no such file'.format(
            ', '.join(err_msgs)))

    phase_labels = [phase_label]
    for pid in config.nlloc_config.phaseid_list:
        if (phase_label == pid.std_phase or
                phase_label in pid.phase_code_list):

            phase_labels.extend(pid.phase_code_list)

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
    config = load_config(rundir)

    phase_labels = [phase_label]
    for pid in config.nlloc_config.phaseid_list:
        if (phase_label == pid.std_phase or
                phase_label in pid.phase_code_list):

            phase_labels.extend(pid.phase_code_list)

    if filename is None:
        out = sys.stdout
    else:
        ensuredirs(filename)
        out = open(filename, 'w')

    print >>out, '# station label: {}; phase label: {}'.format(
        station_label, phase_label)

    step_to_res = defaultdict(list)
    for step in ['A', 'B', 'C']:
        infile = pjoin(
            rundir, 'harvest', 'results_{}.pickle'.format(NAMED_STEPS[step]))

        if not op.exists(infile):
            continue

        event_to_hypo = load_pickle(infile)

        for hypo in event_to_hypo.itervalues():
            for (sta, pha), res in hypo.res_dict.iteritems():
                if sta == station_label and pha in phase_labels:
                    step_to_res[step].append(res)

    # Dictionary sorted by key (A, B, C)
    step_to_res = OrderedDict(sorted(step_to_res.items(), key=lambda t: t[0]))

    ncols = len(step_to_res.keys())
    nrows = max([len(v) for v in step_to_res.values()])

    a = np.empty((nrows, ncols), dtype=np.float)
    a[:] = np.NaN

    for j, (step, res_list) in enumerate(step_to_res.iteritems()):
        i = len(res_list)
        a[:i, j] = res_list

    hdr = 'location steps: {}'.format(
        ', '.join([NAMED_STEPS[x] for x in step_to_res.keys()]))

    np.savetxt(out, a, fmt='%8.4f', header=hdr, comments='# ')

    if out is not sys.stdout:
        out.close()


def export_events(rundir, step, fmt, filename=None):

    if fmt not in ['pyrocko', 'columns']:
        raise ScoterError('unsupported events file format: "{}"'.format(fmt))

    infile = pjoin(
        rundir, 'harvest', 'results_{}.pickle'.format(NAMED_STEPS[step]))

    if not op.exists(infile):
        raise FileNotFound('cannot access "{}": no such file'.format(infile))

    event_to_hypo = load_pickle(infile)
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
