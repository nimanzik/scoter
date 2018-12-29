# -*- coding: utf-8 -*-

from collections import defaultdict
from copy import copy

import numpy as np

from .log_util import custom_logger
from .meta import Delay, Target
from .parmap import parstarmap
from .spatial import *   # noqa
from .stats import smad
from .util import progressbar


# Set logger
logger = custom_logger(__name__)


def _collect_residuals_static(config):
    sconfig = config.station_terms_config.static_config

    # Extract list of events
    event_list = config.get_located_events()

    logger.info('Collecting travel-time residuals')

    stapha_to_res = defaultdict(list)
    stapha_to_tcor = dict()

    if config.show_progress:
        pbar = progressbar(max_value=len(event_list)).start()

    for ievent, event in enumerate(event_list):
        if config.station_terms_config.locqual_config.islowquality(event):
            continue

        for (slabel, plabel), (res, tcor, _) in event.arrival_maps.iteritems():
            plabel = sconfig.phase_map.get(plabel, None)
            if not plabel:
                continue

            stapha_to_res[slabel, plabel].append(res)
            stapha_to_tcor[slabel, plabel] = tcor

        if config.show_progress:
            pbar.update(ievent)

    if config.show_progress:
        pbar.finish()

    return {k: (stapha_to_res[k], stapha_to_tcor[k]) for k in stapha_to_res}


def calc_static(config):
    """
    Calculate static terms for each station-phase pair.

    Parameters
    ----------
    config : :class:`scoter.core.Config` object
        SCOTER main configuration.

    Returns
    -------
    new_delays : list
        List of :class:`scoter.meta.Delay` objects.
    """

    # --- Collect residuals ---
    stapha_to_res_tcor = _collect_residuals_static(config)

    sconfig = config.station_terms_config.static_config
    wconfig = config.station_terms_config.weight_config

    if wconfig.apply_outlier_rejection:
        if wconfig.outlier_rejection_type == 'dynamic':
            # Dynamic outlier rejection
            pha_to_res = defaultdict(list)
            for (_, pha), (reslist, _) in stapha_to_res_tcor.iteritems():
                pha_to_res[pha].extend(reslist)

            pha_to_rescutoff = {
                k: smad(v)*wconfig.outlier_rejection_level for
                k, v in pha_to_res.iteritems()}
        else:
            # Static outlier rejection
            std_plabels = set(sconfig.phase_map.values())
            pha_to_rescutoff = dict.fromkeys(
                std_plabels, wconfig.outlier_rejection_level)

    # --- Analyse residuals ---
    new_delays = []

    logger.info('Computing static terms for each station')

    if config.show_progress:
        pbar = progressbar(max_value=len(stapha_to_res_tcor)).start()

    for i, ((slabel, plabel), (residuals, tcor_old)) in \
            enumerate(stapha_to_res_tcor.iteritems()):

        # Residual outlier rejection
        if wconfig.apply_outlier_rejection:
            rescutoff = pha_to_rescutoff[plabel]
            residuals = [x for x in residuals if (x/rescutoff)**2 < 1]

        nresiduals = len(residuals)

        if nresiduals < sconfig.nresiduals_min:
            if tcor_old == 0.:
                # No need to keep the old correction term
                # Go to the next station-phase pair
                continue

            # Needs to keep the old correction term
            tcor_new = 0. + tcor_old
            std_dev = -1.
        else:
            # Add one dummy zero residual, i.e. increase `nresiduals` by
            # one, to force the mean station term to zero
            # residuals.append(0.)

            tcor_new = np.mean(residuals) + tcor_old
            std_dev = np.std(residuals)

        # New `Delay` object with *standardized* phase label
        new_delay = Delay(
            station_label=slabel,
            phase_label=plabel,
            nresiduals=nresiduals,
            time_correction=float(tcor_new),
            standard_deviation=float(std_dev))

        new_delays.append(new_delay)

        # New `Delay` objects with phase label used in
        # observation/location file
        for p in sconfig.phase_map_rev[plabel]:
            if p == plabel:
                continue

            new_delay_dummy = copy(new_delay)
            setattr(new_delay_dummy, 'phase_label', p)
            new_delays.append(new_delay_dummy)

        if config.show_progress:
            pbar.update(i)

    if config.show_progress:
        pbar.finish()

    return new_delays


g_state = {}


def calc_ssst(config, iiter):
    """
    Calculate source-specific station terms for each source-receiver path.

    Parameters
    ----------
    config : :class:`scoter.core.Config` onject
        SCOTER main configuration.

    iiter : int
        Iteration number in SSST calculation (starting from 0).

    Returns
    -------
    new_targets : list
        List of :class:`core.Target` objects that can be used in the
        next location iteration.
    """

    sconfig = config.station_terms_config.ssst_config
    wconfig = config.station_terms_config.weight_config

    # --- Extract list of all events, low quality events etc ---
    event_list = config.get_located_events()

    lowqualities = []
    pha_to_res = defaultdict(list)

    logger.info('Collecting travel-time residuals')

    if config.show_progress:
        pbar = progressbar(max_value=len(event_list)).start()

    for ievent, event in enumerate(event_list):

        if config.station_terms_config.locqual_config.islowquality(event):
            lowqualities.append(event.name)
            continue

        for (_, plabel), (res, _, _) in event.arrival_maps.iteritems():
            plabel = sconfig.phase_map.get(plabel, None)
            if not plabel:
                continue
            pha_to_res[plabel].append(res)

        if config.show_progress:
            pbar.update(ievent)

    if config.show_progress:
        pbar.finish()

    # --- Build the KDTree ---
    kdtree = build_ecef_kdtree(event_list)

    # Include target event
    nresiduals_min = sconfig.nlinks_min + 1
    nresiduals_max = sconfig.nlinks[iiter] + 1
    r_max = sconfig.radii[iiter]

    if wconfig.apply_outlier_rejection:
        if wconfig.outlier_rejection_type == 'dynamic':
            # Dynamic outlier rejection
            pha_to_rescutoff = {
                k: smad(v)*wconfig.outlier_rejection_level for
                k, v in pha_to_res.iteritems()}
        else:
            # Static outlier rejection
            std_plabels = set(sconfig.phase_map.values())
            pha_to_rescutoff = dict.fromkeys(
                std_plabels, wconfig.outlier_rejection_level)
    else:
        pha_to_rescutoff = dict()

    # --- Calculate the SSST values ---
    g_data = (
        config, event_list, kdtree, r_max, nresiduals_min, nresiduals_max,
        pha_to_rescutoff, lowqualities)

    g_state[id(g_data)] = g_data
    ntargets = len(event_list)
    task_list = zip(xrange(ntargets), [id(g_data)]*ntargets)

    logger.info(
        'Computing SSST values for each source-receiver path '
        '(r_max = {:.0f} km)'.format(r_max/1000.))

    new_targets = parstarmap(
        _calc_single_ssst,
        task_list,
        nparallel=config.nparallel,
        show_progress=config.show_progress,
        label=None)

    del g_state[id(g_data)]

    return filter(None, new_targets)


def _calc_single_ssst(itrg_event, g_data_id):

    config, event_list, kdtree, r_max, nresiduals_min, nresiduals_max,\
        pha_to_rescutoff, lowqualities = g_state[g_data_id]

    trg_event = event_list[itrg_event]
    sconfig = config.station_terms_config.ssst_config
    wconfig = config.station_terms_config.weight_config

    # Find neighboring events; NOTE: Target event is included in the tree
    trg_in_ecef = kdtree.data[itrg_event, :]   # OC vector
    ngh_idxs, ngh_vectors = find_nearest_neighbors(
        trg_in_ecef, kdtree, r_max, nparallel=config.nparallel)

    if len(ngh_idxs) < nresiduals_min:
        # Go to the next target event
        return

    # Sort the neighbors by distance, exclude target event
    ngh_dists = np.linalg.norm(ngh_vectors, ord=2, axis=1)
    arg_sort = np.argsort(ngh_dists, axis=None)
    ngh_dists = ngh_dists[arg_sort][1:]
    ngh_idxs = ngh_idxs[arg_sort][1:]
    ngh_vectors = ngh_vectors[arg_sort][1:]

    # Either uniform or distance weights
    if wconfig.distance_weighting == 'uniform':
        w_d = np.ones_like(ngh_idxs)
    else:
        w_d = get_w_d(ngh_dists, r_max, a=3, b=3)

    # --- For each ray path for the target event ---

    # new adjusted picks counter
    d_counter = 0
    new_delays = []

    for (trg_slabel, trg_plabel_orig), (res, tcor_old, delta_deg) in \
            trg_event.arrival_maps.iteritems():
        # variable `trg_plabel` should be a standardized phase label
        trg_plabel = sconfig.phase_map.get(trg_plabel_orig)
        if trg_plabel is None:
            # Go to the next arrival
            continue

        # residuals counter
        r_counter = 0
        residuals, weights = [], []

        residuals.append(res)
        weights.append(1.0)
        r_counter += 1

        # If needed, find ray takeoff direction vector
        if wconfig.distance_weighting == 'effective_distance':
            takeoffangle = config.takeoffangles[trg_plabel].interpolate(
                [trg_event.preferred_origin.depth.value, delta_deg])

            sta = config.stations_dict[trg_slabel]
            sta_in_ecef = np.asarray([sta.x, sta.y, sta.z])   # OR vector

            # Ray takeoff direction vector
            CD = ray_takeoff_direction(
                trg_in_ecef, sta_in_ecef, takeoffangle)

        # --- For each neighboring event ---

        # Collect the residuals. Neighbors are sorted by distance
        for jngh, ngh_idx in enumerate(ngh_idxs):
            ngh_event = event_list[ngh_idx]

            if ngh_event.name in lowqualities:
                # Low quality location. Go to the next neighbor
                continue

            ngh_res = []
            for dum_plabel in sconfig.phase_map_rev[trg_plabel]:
                rtd = ngh_event.arrival_maps.get((trg_slabel, dum_plabel))
                if rtd:
                    ngh_res.append(rtd[0])

            ngh_weight = w_d[jngh]

            # If needed, apply effective distance weight
            if wconfig.distance_weighting == 'effective_distance':
                CN = ngh_vectors[jngh]
                alpha = opening_angle(CD, CN)
                ngh_weight = get_w_ed(ngh_weight, alpha)

            residuals.extend(ngh_res)
            weights.extend([ngh_weight] * len(ngh_res))
            r_counter += 1

            if r_counter == nresiduals_max:
                break

        # End of loop over neighbors

        # --- Analyse residuals ---

        residuals = np.asarray(residuals)
        weights = np.asarray(weights)

        # Residual outlier rejection
        if wconfig.apply_outlier_rejection:
            rescutoff = pha_to_rescutoff.get(trg_plabel, 1.0)
            b = np.where(np.true_divide(residuals, rescutoff)**2 < 1)[0]
            residuals = residuals[b]
            weights = weights[b]

        nresiduals = residuals.size

        if nresiduals < nresiduals_min:
            if tcor_old == 0.:
                # No need to keep the old correction term
                # Go to the next arrival in target event
                continue

            # Needs to keep the old correction term
            tcor_new = 0. + tcor_old
            std_dev = -1.

        else:
            # Add one dummy zero residual (i.e. increase 'nresiduals'
            # by one) to force the mean station term to zero
            # residuals = np.append(residuals, 0.)
            # weights = np.append(weights, 1.0)

            tcor_new = np.average(residuals, weights=weights) + tcor_old
            std_dev = np.std(residuals)
            d_counter += 1

        # New `Delay` object with *original* phase label, e.g.
        # phase label used in observation/location file
        new_delay = Delay(
            station_label=trg_slabel,
            phase_label=trg_plabel_orig,
            nresiduals=nresiduals,
            time_correction=float(tcor_new),
            standard_deviation=float(std_dev))

        new_delays.append(new_delay)

    # End of loop over arrivals of the target event

    if d_counter >= sconfig.ndelays_min:
        old_target = [
            x for x in config.targets if x.name == trg_event.name][0]

        return Target(
            name=old_target.name,
            station_labels=old_target.station_labels,
            station_delays=new_delays)

    return


__all__ = ['calc_static', 'calc_ssst']
