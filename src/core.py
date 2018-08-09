# -*- coding: utf-8 -*-

from collections import defaultdict
from functools import partial
from glob import glob
from itertools import chain
import os.path as op
from os.path import join as pjoin
import re
import shutil
import sys

import numpy as np

from pyrocko import model
from pyrocko import spit
from pyrocko.util import ensuredir
from pyrocko import guts
from pyrocko.guts import Bool, Choice, Float, Int, List, Object, String,\
    StringChoice

from .delay import calc_static, calc_ssst
from .geodetic import ellipsoid_distance, geodetic_to_ecef, M2DEG
from .grid import read_nll_grid
from .ie.nlloc import load_nlloc_hyp
from .ie.quakeml import QuakeML   # noqa
from .location import nlloc_runner
from .log_util import custom_logger
from .parmap import parstarmap
from .util import dump_pickle, loglinspace
from .meta import expand_template, HasPaths, FileNotFound, Path, PhaseLabel,\
    Station, Target, ScoterError


guts_prefix = 'gp'

WSPACE = ''.ljust(1)
HYP_FILE_EXTENSION = '.loc.hyp'

# Named location steps
NAMED_STEPS = {'A': 'A_single', 'B': 'B_static', 'C': 'C_ssst'}

# Set logger
logger = custom_logger(__name__)


# ----- Dataset configuration (input data) -----

class DatasetConfig(HasPaths):
    events_path = Path.T()
    bulletins_template_path = Path.T()
    stations_path = Path.T()
    delimiter_str = String.T(optional=True)
    traveltimes_path = Path.T()
    takeoffangles_template_path = Path.T(optional=True)
    starting_delays_path = Path.T(optional=True)

    def __init__(
            self, events_path, bulletins_template_path, stations_path,
            traveltimes_path, delimiter_str=None,
            takeoffangles_template_path=None, starting_delays_path=None):

        HasPaths.__init__(
            self,
            events_path=events_path,
            bulletins_template_path=bulletins_template_path,
            stations_path=stations_path,
            delimiter_str=delimiter_str or '',
            traveltimes_path=traveltimes_path,
            takeoffangles_template_path=takeoffangles_template_path,
            starting_delays_path=starting_delays_path)

    def get_pyrocko_events(self):
        """Extracts a list of :class:`pyrocko.model.Event` objects."""

        if not op.exists(self.events_path):
            raise FileNotFound('cannot access "{}": no such file'.format(
                self.events_path))

        logger.debug("Loading events from file '{}'".format(self.events_path))
        return model.load_events(self.events_path)

    def get_stations(self):
        """Returns a list of :class:`meta.Station` objects."""

        if not op.exists(self.stations_path):
            raise FileNotFound('cannot access "{}": no such file'.format(
                self.stations_path))

        logger.debug(
            'Loading stations from file "{}"'.format(self.stations_path))

        f = open(self.stations_path, 'r')
        lines = f.read().splitlines()
        lines = filter(None, lines)
        f.close()

        stations = []
        for line in lines:
            items = line.split()
            try:
                net = items[-5].strip()
            except IndexError:
                net = ''.strip()
            sta = items[-4].strip()
            slabel = self.delimiter_str.join(items[-5:-3])

            lat, lon, elev = map(float, items[-3:])
            x, y, z = geodetic_to_ecef(lat, lon, 0.)

            stations.append(Station(
                network=net, station=sta, lat=lat, lon=lon, depth=0.,
                elevation=elev, x=x, y=y, z=z, name=slabel))

        if len(stations) == 0:
            raise ScoterError(
                'no station information found in "{}"'.format(
                    self.stations_path))

        return stations

    def get_takeoffangles(self, phase_labels):
        """
        Returns a dictionary whose keys are `phase_labels` and values
        are :class:`pyrocko.spit.SPTree` objects.
        """

        def check(fns):
            errmess = []
            for fn in fns.values():
                if not op.exists(fn):
                    errmess.append(
                        'cannot access "{}": no such file').format(fn)

            return errmess

        fns = {}
        for plabel in phase_labels:
            fn = expand_template(
                self.takeoffangles_template_path,
                dict(phase_label=plabel))

            fns[plabel] = fn

        errmess = check(fns)
        if errmess:
            raise FileNotFound("{}".format('\n'.join(errmess)))

        trees = {}
        for plabel, fn in fns.iteritems():
            logger.debug(
                'Loading {0}-wave takeoff angles from file "{1}"'.format(
                    plabel, fn))

            tree = spit.SPTree(filename=fn)
            trees[plabel] = tree
        return trees


# ----- Earthquake location quality configuration -----

class LocationQualityConfig(Object):
    standard_error_max = Float.T(optional=True)
    secondary_azigap_max = Float.T(optional=True)
    largest_uncertainty_max = Float.T(optional=True)

    def __init__(self, **kwargs):
        Object.__init__(
            self,
            standard_error_max=kwargs.get('standard_error_max', 3.),
            secondary_azigap_max=kwargs.get('secondary_azigap_max', 180.),
            largest_uncertainty_max=kwargs.get(
                'largest_uncertainty_max', 1./M2DEG))

    def islowquality(self, event):
        """Check the location quality."""

        o = event.preferred_origin
        q = o.quality
        u = o.origin_uncertainty_list[0]
        c = u.confidence_ellipsoid

        if (q.standard_error > self.standard_error_max or
                q.secondary_azimuthal_gap > self.secondary_azigap_max or
                c.semi_major_axis_length > self.largest_uncertainty_max):
            # Low quality location.
            return True

        return False


# ----- Weighting and re-weighting scheme configuration -----

class WeightConfig(Object):
    distance_weighting = StringChoice.T(
        choices=['uniform', 'distance', 'effective_distance'], optional=True)
    apply_outlier_rejection = Bool.T(optional=True)
    outlier_rejection_type = StringChoice.T(
        choices=['static', 'dynamic'], optional=True)
    outlier_rejection_level = Choice.T(
        choices=[Float.T(), Int.T()], optional=True)

    def __init__(self, **kwargs):

        Object.__init__(
            self,
            distance_weighting=kwargs.get('distance_weighting', 'uniform'),
            apply_outlier_rejection=kwargs.get(
                'apply_outlier_rejection', True),
            outlier_rejection_type=kwargs.get(
                'outlier_rejection_type', 'dynamic'),
            outlier_rejection_level=kwargs.get('outlier_rejection_level', 6))


# ----- Static station terms configuration -----

class StaticConfig(Object):
    niter = Int.T()
    phase_list = List.T(PhaseLabel.T())
    nresiduals_min = Int.T()

    def __init__(self, **kwargs):
        Object.__init__(self, **kwargs)
        self.phase_map = self.get_phase_map()
        self.phase_map_rev = None

    def get_phase_map(self):
        return {x: x for x in self.phase_list}


# ----- Source-specific station terms configuration. -----

class SourceSpecificConfig(Object):
    niter = Int.T()
    phase_list = List.T(PhaseLabel.T())
    start_cutoff_dist = Float.T()
    start_nlinks_max = Int.T()
    end_cutoff_dist = Float.T()
    end_nlinks_max = Int.T()
    nlinks_min = Int.T()
    ndelays_min = Int.T()

    def __init__(self, **kwargs):
        Object.__init__(self, **kwargs)
        self.phase_map = self.get_phase_map()
        self.phase_map_rev = None
        self.radii = self.get_cutoff_distances()
        self.nlinks = self.get_nlinks()

    def get_phase_map(self):
        return {x: x for x in self.phase_list}

    def get_cutoff_distances(self):
        return loglinspace(
            self.start_cutoff_dist, self.end_cutoff_dist, self.niter)

    def get_nlinks(self):
        return np.round(loglinspace(
            self.start_nlinks_max, self.end_nlinks_max, self.niter))


# ----- Station terms config -----

class StationTermsConfig(Object):
    static_config = StaticConfig.T()
    ssst_config = SourceSpecificConfig.T()
    weight_config = WeightConfig.T(optional=True)
    locqual_config = LocationQualityConfig.T(optional=True)


# ----- Seismic network configuration. -----

class NetworkConfig(Object):
    station_selection = Bool.T(optional=True)
    station_dist_min = Float.T(optional=True)
    station_dist_max = Float.T(optional=True)

    def __init__(self, **kwargs):

        Object.__init__(
            self,
            station_selection=kwargs.get('station_selection', False),
            station_dist_min=kwargs.get('station_dist_min', 0.),
            station_dist_max=kwargs.get('station_dist_max', 180.))


# ----- NLLoc configuration (control file statements). -----

class NLLocTrans(Object):
    trans_type = StringChoice.T(choices=[
        'GLOBAL', 'SIMPLE', 'NONE', 'SDC', 'LAMBERT'])
    lat_orig = Float.T(optional=True)
    lon_orig = Float.T(optional=True)
    rot_angle = Float.T(optional=True)
    ref_ellips = StringChoice.T(choices=[
        'WGS-84', 'GRS-80', 'WGS-72', 'Australian', 'Krasovsky',
        'International', 'Hayford-1909', 'Clarke-1880', 'Clarke-1866',
        'Airy', 'Bessel', 'Hayford-1830', 'Sphere'], optional=True)
    first_paral = Float.T(optional=True)
    second_paral = Float.T(optional=True)

    def __init__(self, **kwargs):
        Object.__init__(self, **kwargs)

        if self.trans_type not in ('GLOBAL', 'NONE'):
            try:
                self.lat_orig = kwargs.pop('lat_orig')
                self.lon_orig = kwargs.pop('lon_orig')
                self.rot_angle = kwargs.pop('rot_angle')

                if self.trans_type == 'LAMBERT':
                    self.ref_ellips = kwargs.pop('ref_ellips')
                    self.first_paral = kwargs.pop('first_paral')
                    self.second_paral = kwargs.pop('second_paral')
            except KeyError as e:
                raise ScoterError(
                    'missing argument to NLLocTrans type "%s": %s' %
                    (self.trans_type, e))

    def __str__(self):
        params = ['TRANS', '{self.trans_type}']

        if self.trans_type in ['SIMPLE', 'SDC']:
            params += [
                '', '{self.lat_orig}', '{self.lon_orig}', '{self.rot_angle}']

        elif self.trans_type == 'LAMBERT':
            params += [
                '', '{self.ref_ellips}', '{self.lat_orig}', '{self.lon_orig}',
                '{self.first_paral}', '{self.second_paral}',
                '{self.rot_angle}']

        return WSPACE.join(params).format(self=self)


class NLLocGrid(Object):
    x_num = Int.T()
    y_num = Int.T()
    z_num = Int.T()
    x_orig = Float.T()
    y_orig = Float.T()
    z_orig = Float.T()
    dx = Float.T()
    dy = Float.T()
    dz = Float.T()
    grid_type = StringChoice.T(choices=['MISFIT', 'PROB_DENSITY'])

    def __str__(self):
        params = [
            'LOCGRID', '{self.x_num}', '{self.y_num}', '{self.z_num}',
            '{self.x_orig}', '{self.y_orig}', '{self.z_orig}', '{self.dx}',
            '{self.dy}', '{self.dz}', '{self.grid_type}', 'SAVE']

        return WSPACE.join(params).format(self=self)


class NLLocSearchOcttree(Object):
    init_num_cells_x = Int.T()
    init_num_cells_y = Int.T()
    init_num_cells_z = Int.T()
    min_node_size = Float.T()
    max_num_nodes = Int.T()
    num_scatter = Int.T()
    use_sta_density = Int.T()
    stop_on_what = Int.T(
        help='if 1, stop search when first min_node_size reached, '
             'if 0 stop subdividing a given cell when min_node_size reached')

    def __str__(self):
        params = [
            'LOCSEARCH', 'OCT', '{self.init_num_cells_x}',
            '{self.init_num_cells_y}', '{self.init_num_cells_z}',
            '{self.min_node_size}', '{self.max_num_nodes}',
            '{self.num_scatter}', '{self.use_sta_density}',
            '{self.stop_on_what}']

        return WSPACE.join(params).format(self=self)


class NLLocMeth(Object):
    method = StringChoice.T(
        choices=['GAU_ANALYTIC', 'EDT', 'EDT_OT_WT', 'EDT_OT_WT_ML'])
    max_dist_sta_grid = Float.T()
    min_num_phases = Int.T()
    max_num_phases = Int.T()
    min_num_Sphases = Int.T()
    vp_vs_ratio = Float.T()
    max_num_3dgrid_mem = Int.T()
    min_dist_sta_grid = Float.T()
    reject_duplicate_arrivals = Int.T()

    def __str__(self):
        params = [
            'LOCMETH', '{self.method}', '{self.max_dist_sta_grid}',
            '{self.min_num_phases}', '{self.max_num_phases}',
            '{self.min_num_Sphases}', '{self.vp_vs_ratio}',
            '{self.max_num_3dgrid_mem}', '{self.min_dist_sta_grid}',
            '{self.reject_duplicate_arrivals}']

        return WSPACE.join(params).format(self=self)


class NLLocGau(Object):
    sigma_time = Float.T()
    corr_len = Float.T()

    def __str__(self):
        return 'LOCGAU {self.sigma_time} {self.corr_len}'.format(self=self)


class NLLocGau2(Object):
    sigma_tfraction = Float.T()
    sigma_tmin = Float.T()
    sigma_tmax = Float.T()

    def __str__(self):
        params = [
            'LOCGAU2', '{self.sigma_tfraction}', '{self.sigma_tmin}',
            '{self.sigma_tmax}']

        return WSPACE.join(params).format(self=self)


class NLLocPhaseid(Object):
    std_phase = String.T()
    phase_code_list = List.T(PhaseLabel.T())

    def __str__(self):
        params = ['LOCPHASEID', self.std_phase]
        params.extend(self.phase_code_list)
        return WSPACE.join(params)


class ActivationFlag(Choice):
    choices=[Bool.T(), Int.T()]


class NLLocElevcorr(Object):
    activation_flag = ActivationFlag.T()
    vel_p = Float.T(default=5.80, optional=True)
    vel_s = Float.T(default=3.46, optional=True)

    def __init__(self, activation_flag, vel_p=None, vel_s=None):
        Object.__init__(
            self,
            activation_flag=int(activation_flag > 0),
            vel_p=vel_p or 5.80,
            vel_s=vel_s or 3.46)

    def __str__(self):
        params = [
            'LOCELEVCORR', '{self.activation_flag}', '{self.vel_p}',
            '{self.vel_s}']

        return WSPACE.join(params).format(self=self)


class NLLocStawt(Object):
    activation_flag = ActivationFlag.T()
    cutoff_dist = Float.T(default=-1.0, optional=True)

    def __init__(self, activation_flag, cutoff_dist=None):
        Object.__init__(
            self,
            activation_flag=int(self.activation_flag > 0),
            cutoff_dist=cutoff_dist or -1.0)

    def __str__(self):
        params = ['LOCSTAWT', '{self.activation_flag}', '{self.cutoff_dist}']
        return WSPACE.join(params).format(self=self)


class NLLocConfig(Object):
    trans = NLLocTrans.T()
    grid = NLLocGrid.T()
    search = NLLocSearchOcttree.T()
    meth = NLLocMeth.T()
    gau = NLLocGau.T()
    gau2 = NLLocGau2.T(optional=True)
    phaseid_list = List.T(NLLocPhaseid.T())
    elevcorr = NLLocElevcorr.T(optional=True)
    stawt = NLLocStawt.T(optional=True)

    def __init__(self, **kwargs):
        Object.__init__(self, **kwargs)
        self._swapbytes_flag = 0

    def __str__(self):
        s = """
CONTROL 0 54321
{self.trans}
LOCHYPOUT SAVE_NLLOC_ALL
{self.grid}
{self.search}
{self.meth}
{self.gau}
{self.gau2}
LOCQUAL2ERR 0.1 0.2 0.5 1.0 999.9
{self.elevcorr}
{self.stawt}
"""
        n = len(self.phaseid_list)
        if n != 0:
            for i in range(n):
                s += '{self.phaseid_list[%d]}\n' % i

        s = s.format(self=self)
        s.replace('None\n', '')
        return s


# ----- SCOTER main configuration. -----

def _get_single_target(event, config):
    """
    Utility function to make a `scoter.meta.Target` object from an event.
    This function is used in parallel processing.

    Parameters
    ----------
    event : :class:`pyrocko.model.Event` object
        Pyrocko event.
    config : :class:`scoter.core.Config` object
        SCOTER main configuration.

    Returns
    -------
    result : :class:`scoter.meta.Target` object
    """

    filename = expand_template(
        config.dataset_config.bulletins_template_path,
        dict(event_name=event.name))

    if not op.exists(filename):
        return

    flines = open(filename, 'r').read().splitlines()
    flines = filter(None, flines)

    idx_pha_block = [None, None]
    for i, line in enumerate(flines, 1):
        if line.startswith('PHASE '):
            idx_pha_block[0] = i
        elif line.startswith('END_PHASE'):
            idx_pha_block[1] = i

    if len(filter(None, idx_pha_block)) == 2:
        i1, i2 = idx_pha_block
        pha_lines = flines[i1:i2:1]
    else:
        pattern = re.compile(r'\sGAU\s', re.I)
        pha_lines = [x for x in flines if pattern.findall(x)]

    if len(pha_lines) == 0:
        raise ScoterError(
            'This should not happen. Please check the file "{}"'.format(
                filename))

    if config.network_config.station_selection is True:
        slabels = []
        for line in pha_lines:
            slabel = line.split()[0]
            if slabel in slabels:
                continue

            sta = config.stations_dict.get(slabel, None)
            if not sta:
                continue

            dist_deg = ellipsoid_distance(
                event.lat, event.lon, sta.lat, sta.lon) * M2DEG
            if (
                config.network_config.station_dist_min <= dist_deg <=
                    config.network_config.station_dist_max):
                # station is in the desired surface distance range.
                slabels.append(slabel)

    else:
        # no station selection.
        slabels = set()
        for line in pha_lines:
            slabels.add(line.split()[0])

        slabels = list(slabels)

    if len(slabels) == 0:
        return

    return Target(
        name=event.name,
        station_labels=slabels,
        station_delays=config.dataset_config.starting_delays_path)


class Config(HasPaths):
    rundir = Path.T()
    dataset_config = DatasetConfig.T()
    station_terms_config = StationTermsConfig.T()
    network_config = NetworkConfig.T()
    nlloc_config = NLLocConfig.T()

    def __init__(self, *args, **kwargs):
        HasPaths.__init__(self, *args, **kwargs)
        self._stations = None
        self._stations_stream = None
        self._stations_dict = None
        self._takeoffangles = None
        self._locations_path = None
        self._locdir = None
        self._targets = None
        self._nparallel = None
        self._show_progress = None
        self.update_phase_maps()
        self.update_phase_maps_rev()
        self.update_swapbytes_flag()

    @property
    def stations(self):
        if self._stations is None:
            self._stations = self.dataset_config.get_stations()
        return self._stations

    @property
    def stations_stream(self):
        if self._stations_stream is None:
            stream = ''
            for sta in self.stations:
                stream += '{}\n'.format(sta)
            self._stations_stream = stream

        return self._stations_stream

    @property
    def stations_dict(self):
        if self._stations_dict is None:
            self._stations_dict = {}
            for sta in self.stations:
                self._stations_dict[sta.name] = sta

        return self._stations_dict

    @property
    def takeoffangles(self):
        wconfig = self.station_terms_config.weight_config
        if (wconfig.distance_weighting == 'effective_distance' and
                self._takeoffangles is None):
            # load takeoff-angle trees
            self._takeoffangles = self.dataset_config.get_takeoffangles(
                phase_labels=self.ssst_config.phase_list)

        return self._takeoffangles

    @property
    def locations_path(self):
        return self._locations_path

    @locations_path.setter
    def locations_path(self, new_path):
        self._locations_path = new_path
        self._locdir = op.dirname(new_path)

    @locations_path.getter
    def locations_path(self):
        if self._locations_path is None:
            self.locations_path = self.dataset_config.bulletins_template_path
        return self._locations_path

    @property
    def locdir(self):
        return self._locdir

    @locdir.setter
    def locdir(self, new_locdir):
        self._locdir = new_locdir
        self._locations_path = pjoin(
            new_locdir,
            '${{event_name}}{ext}'.format(ext=HYP_FILE_EXTENSION))

        logger.debug('Set location directory to "{}"'.format(new_locdir))

    @locdir.getter
    def locdir(self):
        if self._locdir is None:
            self.locdir = op.dirname(
                self.dataset_config.bulletins_template_path)
        return self._locdir

    @property
    def targets(self):
        return self._targets

    @targets.setter
    def targets(self, new_targets):
        self._targets = new_targets

    @targets.getter
    def targets(self):
        if self._targets is None:
            pyrocko_events = self.dataset_config.get_pyrocko_events()
            task_list = [(i, self) for i in pyrocko_events]

            logger.info('Getting initial target events')

            x = parstarmap(
                _get_single_target,
                task_list,
                nparallel=self.nparallel,
                show_progress=self.show_progress,
                label=None)

            self.targets = filter(None, x)

        return self._targets

    @property
    def nparallel(self):
        return self._nparallel

    @nparallel.setter
    def nparallel(self, n):
        self._nparallel = n

    @property
    def show_progress(self):
        return bool(self._show_progress)

    @show_progress.setter
    def show_progress(self, boolean):
        self._show_progress = boolean

    def update_phase_maps(self):
        sconfig = self.station_terms_config
        # phase_map should be e.g. {'P': 'P', 'Pb': 'P', 'Pg': 'P', 'Pn': 'P'}
        for phaseid in self.nlloc_config.phaseid_list:
            d = {k: phaseid.std_phase for k in phaseid.phase_code_list}

            if phaseid.std_phase in sconfig.static_config.phase_list:
                sconfig.static_config.phase_map.update(d)

            if phaseid.std_phase in sconfig.ssst_config.phase_list:
                sconfig.ssst_config.phase_map.update(d)

    def update_phase_maps_rev(self):
        # phase_map_rev should be e.g. {'P': ['Pb', 'P', 'Pg', 'Pn']}

        def reverse(d):
            d_rev = defaultdict(list)
            for k, v in d.iteritems():
                d_rev[v].append(k)
            return d_rev

        for s in ['static', 'ssst']:
            obj = eval('self.station_terms_config.{0}_config'.format(s))
            d_rev = reverse(obj.phase_map)
            setattr(obj, 'phase_map_rev', d_rev)

    def update_swapbytes_flag(self):
        fn_grd = glob(self.dataset_config.traveltimes_path+'*.buf')[0]
        # we want to read the binary file in little-endian order
        sys_is_le = sys.byteorder == 'little'
        grd = read_nll_grid(fn_grd, swapbytes=(not sys_is_le))

        if np.any(np.isnan(grd.data_array)):
            # the binary file is stored in big-endian order
            setattr(self.nlloc_config, '_swapbytes_flag', 1)

    def get_located_events(self):
        """
        Extract a list of :class:`scoter.ie.quakeml.Event` objects
        located by the NLLoc program (results from the *last location
        iteration*). It goes through `targets` and `locdir` to access
        the location files.
        """

        assert self.locations_path, 'no locations path has been set'
        assert self.targets, 'found empty list of target events'

        # Make task list for parallel processing.
        dummy_expander = partial(expand_template, self.locations_path)
        task_list = []
        for target in self.targets:
            filename = dummy_expander(dict(event_name=target.name))
            if not op.exists(filename):
                continue
            task_list.append((filename, target.name))

        dummy_loader = partial(
            load_nlloc_hyp,
            delimiter_str=self.dataset_config.delimiter_str,
            add_arrival_maps=True)

        logger.info('Getting located events')

        qmls = parstarmap(
            dummy_loader,
            task_list,
            nparallel=self.nparallel,
            show_progress=self.show_progress,
            label=None)

        # variable `cat` is a list of lists.
        cat = [x.event_parameters.event_list for x in qmls]
        event_list = list(chain.from_iterable(cat))

        return event_list


class ScoterRunner(object):
    def __init__(self, config, nparallel=1, show_progress=False):
        self.config = config
        self.config.nparallel = nparallel
        self.config.show_progress = show_progress
        self.sub_dirs = dict((k, pjoin(self.config.rundir, NAMED_STEPS[k]))
                             for k in 'ABC')

    def _run_single(self):
        logger.info('Started single-event location')
        self.config.locdir = self.sub_dirs['A']
        nlloc_runner(self.config, keep_scat=False, raise_exception=False)
        logger.info('Done with single-event location\n')

    def _run_static(self):
        niter = self.config.station_terms_config.static_config.niter

        if niter <= 0:
            return

        static_dir = self.sub_dirs['B']
        ensuredir(static_dir)

        for iiter in range(niter):
            logger.info(
                'Started static terms location - '
                'iteration {0} / {1}'.format(iiter+1, niter))

            # --- Step 1: get static station terms. ---
            new_delays = calc_static(self.config)

            if not new_delays:
                break

            # stream = ''
            # for delay in new_delays:
            #     stream += '{}\n'.format(delay)

            # fn_delays = pjoin(static_dir, '{:02}_1_delay'.format(iiter+1))
            # with open(fn_delays, 'w') as f:
            #     f.write(stream)

            fn = pjoin(static_dir, '{:02}_1_delay.pickle'.format(iiter+1))
            dump_pickle(new_delays, fn)

            # --- Step 2: set new locdir and new station terms. ---
            self.config.locdir = pjoin(
                static_dir, '{:02}_2_loc'.format(iiter+1))

            for target in self.config.targets:
                trg_new_delays = [
                    x for x in new_delays if x.station_label in
                    target.station_labels]

                setattr(target, 'station_delays', trg_new_delays)

            # --- Step 3: locate events with new station delays. ---
            nlloc_runner(self.config, keep_scat=False, raise_exception=False)

            logger.info(
                'Done with static terms location - '
                'iteration {0} / {1}\n'.format(iiter+1, niter))

    def _run_ssst(self):
        niter = self.config.station_terms_config.ssst_config.niter

        if niter <= 0:
            return

        ssst_dir = self.sub_dirs['C']
        ensuredir(ssst_dir)

        for iiter in range(niter):
            logger.info(
                'Started shrinking-box SSST location - '
                'iteration {0} / {1}'.format(iiter+1, niter))

            # --- Step 1: get source-specific station terms. ---
            new_targets = calc_ssst(self.config, iiter)

            # dn = pjoin(ssst_dir, '{:02}_1_delay'.format(iiter+1))
            # ensuredir(dn)
            # fn_temp = pjoin(dn, '%(event_name)s.scoter')
            # for target in new_targets:
            #     fn = fn_temp % dict(event_name=target.name)
            #     with open(fn, 'w') as f:
            #         for delay in target.station_delays:
            #             f.write('{}\n'.format(delay))

            new_delays = dict((x.name, x.station_delays) for x in new_targets)
            fn = pjoin(ssst_dir, '{:02}_1_delay.pickle'.format(iiter+1))
            dump_pickle(new_delays, fn)

            # --- Step 2: set new locdir and new targets. ---
            self.config.locdir = pjoin(ssst_dir, '{:02}_2_loc'.format(iiter+1))
            self.config.targets = new_targets

            # --- Step 3: locate events with new station delays. ---
            nlloc_runner(self.config, keep_scat=False, raise_exception=False)

            logger.info(
                'Done with shrinking-box SSST location - '
                'iteration {0} / {1}\n'.format(iiter+1, niter))


def read_config(path):
    config = guts.load(filename=path)
    if not isinstance(config, Config):
        raise ScoterError('invalid configuration in file "{}"'.format(path))

    config.set_basepath(op.dirname(path) or ".")
    return config


def go(config, steps, force=False, nparallel=1, show_progress=False):
    """
    Parameters
    ----------
    config : :class:`scoter.core.Config` object
        SCOTER main configuration.
    steps : list of str
        Named location steps in SCOTER syntax ('A', 'B', 'C').
    force : bool (defaulr: False)
        Whether to overwrite existing subdirectories of main rundir.
    nparallel : int (default: 1)
        Set number of events to process in parallel.
    show_progress : bool (default: False)
        Show progress bar to display the progress of running operation.

    Returns
    -------
    None
    """

    steps = sorted(steps)

    if not 1 <= len(steps) <= 3:
        raise ScoterError('invalid location steps: "{}"'.format(
            ','.join(steps)))

    a, b = set(steps), set(['A', 'B', 'C'])
    if not a <= b:
        raise ScoterError('invalid location step: "{}"'.format(
            ','.join(list(a-b))))

    scoter = ScoterRunner(config, nparallel, show_progress)

    warn_msgs = []
    for k in steps:
        sub_dir = scoter.sub_dirs[k]
        if op.exists(sub_dir):
            warn_msgs.append(sub_dir)

    if warn_msgs:
        if not force:
            logger.warn(
                'skipping problem: sub-directories already '
                'exist: "{}"'.format(', '.join(warn_msgs)))
            return
        else:
            for d in warn_msgs:
                shutil.rmtree(d)

    basepath = config.get_basepath()
    config.change_basepath(config.rundir)
    ensuredir(config.rundir)
    guts.dump(config, filename=pjoin(config.rundir, 'used_config.yaml'))
    config.change_basepath(basepath)

    methods = {'A': '_run_single', 'B': '_run_static', 'C': '_run_ssst'}

    for k in steps:
        func = getattr(scoter, methods[k])
        func()


__all__ = """
    DatasetConfig
    LocationQualityConfig
    WeightConfig
    StaticConfig
    SourceSpecificConfig
    StationTermsConfig
    NetworkConfig
    NLLocTrans
    NLLocGrid
    NLLocSearchOcttree
    NLLocMeth
    NLLocGau
    NLLocGau2
    NLLocPhaseid
    NLLocElevcorr
    NLLocStawt
    NLLocConfig
    Config
    read_config
    go
""".split()
