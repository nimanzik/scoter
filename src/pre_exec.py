# -*- coding: utf-8 -*-

import os.path as op

import numpy as np

from pyrocko import cake, spit
from pyrocko.model import dump_events
from pyrocko.util import ensuredir

from .ie.geofon import load_geofon_hyp
from .ie.nlloc import dump_nlloc_obs
from .ie.quakeml import QuakeML
from .log_util import custom_logger
from .meta import PathAlreadyExists, ScoterError
from .util import progressbar


KM2M = 1000.

# Set logger
logger = custom_logger(__name__)


def dump_nlloc_obs_all(
        bulletin_files, format, output_dir, events_path, delimiter_str=None,
        prefix='', suffix='.nll', force=False, show_progress=False):

    if not force:
        warn_msgs = []
        for path in (output_dir, events_path):
            if op.exists(path):
                warn_msgs.append(path)

        if warn_msgs:
            raise PathAlreadyExists('file/dir already exists: "{}"'.format(
                ', '.join(warn_msgs)))

    fmt = format.lower()
    if fmt == 'geofon':
        loader = load_geofon_hyp
    elif fmt == 'quakeml':
        loader = QuakeML.load_xml
    else:
        raise ScoterError('unsupported file format: "{}"'.format(fmt))

    ensuredir(output_dir)

    if show_progress:
        pbar = progressbar(max_value=len(bulletin_files)).start()

    pyrocko_events = []
    for ifn, fn in enumerate(bulletin_files):
        try:
            qml_cat = loader(filename=fn)
        except:   # noqa
            logger.warn('Could not load file "{}"'.format(fn))
            pass
        else:
            pyrocko_events.extend(qml_cat.get_pyrocko_events())

        # Reformat quakeml events and dump into NLLOC_OBS files
        for qml_event in qml_cat.event_parameters.event_list:
            out_filename = op.join(
                output_dir, '{prefix}{event_name}{suffix}'.format(
                    prefix=prefix, event_name=qml_event.name, suffix=suffix))

            dump_nlloc_obs(qml_event, out_filename, delimiter_str)

        if show_progress:
            pbar.update(ifn)

    if show_progress:
        pbar.finish()

    if len(pyrocko_events) == 0:
        raise ScoterError('Bulletin files seem corrupt. No event is found')

    # Write pyrocko events file
    dump_events(pyrocko_events, filename=events_path)


def _get_phase_defs(phase):
    """First arriving P and S phases."""

    if phase.lower() == 'p':
        # classic names: p, P, Pdiff, PKiKP, PKIKP, PKP
        phase_defs = [
            cake.PhaseDef('p'),
            cake.PhaseDef('P'),
            cake.PhaseDef('Pv_(cmb)p'),
            cake.PhaseDef('P(cmb)Pv(icb)p(cmb)p'),
            cake.PhaseDef('P(cmb)P(icb)P(icb)p(cmb)p'),
            cake.PhaseDef('P(cmb)P<(icb)(cmb)p')]
    elif phase.lower() == 's':
        # classic names: s, S, Sdiff
        phase_defs = [
            cake.PhaseDef('s'),
            cake.PhaseDef('S'),
            cake.PhaseDef('Sv_(cmb)s')]

    return phase_defs


def build_takeoffangles_sptree(
        model, phase, z_range, x_range, z_tol=1.*KM2M, x_tol=0.1, a_tol=0.01,
        filename=None):
    '''
    Creat a :py:class:`pyrocko.spit.SPTree` interpolator to interpolate
    and query first-arriving P or S waves takeoff angles.

    Parameters
    ----------
    model : str
        1D seismic velocity model. :py:class:`pyrocko.cake` built-in
        model name or user model file. The user model file should have
        TauP style '.nd' (named discontinuity) format.

    phase : str
        Seismic phase ('P', 'S').

    z_range : tuple of 2 float objects
        Minimum and maximum source depth in [m].

    x_range : tuple of 2 float objects
        Minimum and maximum source-receiver surface distance in [deg].

    z_tol : float
        Depth tolerance threshold in [m] defining the accuracy of the
        `:py:class:pyrocko.split.SPTree`.

    x_tol : float
        Distance tolerance threshold in [deg] defining the accuracy of
        the `:py:class:pyrocko.split.SPTree`.

    a_tol : float
        Takeoff angle tolerance threshold in [deg] defining the accuracy
        of the :py:class:`pyrocko.split.SPTree`.

    filename : None or str
        If a file name is provided, it dumps the sptree to the file for
        later reuse (default: None).

    Returns
    -------
    :py:class:`pyrocko.spit.SPTree` object.
    '''

    if filename and op.exists(filename):
        raise PathAlreadyExists('file already exists: "{}"'.format(filename))

    if phase.lower() not in ('p', 's'):
        raise ScoterError('unsupported seismic phase: "{}"'.format(phase))

    mod = cake.load_model(model, format='nd')
    phase_defs = _get_phase_defs(phase)
    xbounds = np.array([z_range, x_range])
    xtols = np.array((z_tol, x_tol))

    def evaluate(args):
        '''
        Calculate ray takeoff angle using source depth, and receiver
        distance defined by *args*. This function is evaluated by the
        `SPTree` instance.

        Parameters
        ----------
        args : tuple of 2 objects
            Tuple of (sdepth, rdist) where `sdepth` is source depth in [m],
            and `rdist` is the source-receiver surface distance in [deg].
        '''

        sdepth, rdist = args

        rays = mod.arrivals(
            phases=phase_defs,
            distances=[rdist],
            zstart=sdepth)

        if rays:
            # Sort rays by arrival times.
            rays.sort(key=lambda ray: ray.t)

            return rays[0].takeoff_angle()

        return None

    sptree = spit.SPTree(f=evaluate, ftol=a_tol, xbounds=xbounds, xtols=xtols)

    if filename:
        sptree.dump(filename)

    return sptree


__all__ = """
    dump_nlloc_obs_all
    build_takeoffangles_sptree
""".split()
