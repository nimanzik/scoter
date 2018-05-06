import calendar
import datetime
import math
import re

import numpy as np

from pyrocko.model import Station
from pyrocko.util import ensuredirs, str_to_time, time_to_str, TimeStrError

from ..geodetic import M2DEG
from .quakeml import Arrival, ConfidenceEllipsoid, Event, EventParameters,\
    Origin, OriginQuality, OriginUncertainty, Pick, Phase, RealQuantity,\
    TimeQuantity, WaveformStreamID, QuakeML


guts_prefix = 'gp'

KM2M = 1000.
M2KM = 1. / KM2M
WSPACE = ''.ljust(1)

ONSETS_LOAD = {'e': 'emergent', 'i': 'impulsive', '?': 'questionable'}
ONSETS_DUMP = {v: k for k, v in ONSETS_LOAD.items()}

POLARITIES_LOAD = {'d': 'negative', 'c': 'positive', '?': 'undecidable'}
POLARITIES_DUMP = {v: k for k, v in POLARITIES_LOAD.items()}

EVALUATION_STATUS = {
    'LOCATED': 'final',
    'REJECTED': 'rejected',
    'ABORTED': 'rejected'}


class NLLocError(Exception):
    pass


def load_nlloc_hyp(
        filename, event_name=None, delimiter_str=None, set_arrival_dict=False):
    """
    Reads a NonLinLoc hypocenter-phase file to a
    :class:`~scoter.ie.quakeml.QuakeML` object.

    Parameters
    ----------
    filename : str
        NonLinLoc hypocenter-phase file name.

    event_name : str
        Event public ID.

    delimiter_str : str or None (default: None)
        If not `None`, it specifies a string to be used as the network
        code and station code separator character, i.e. it is used to
        split the station label in NonLinLoc hypocenter-phase file. If
        it is absent or `None` (default), the entire station label read
        is considered as the `station code` and an empty string is
        assigned to the `network code` for each pick.

    set_arrival_dict : bool (default: False)
        Whether to return arrivals as a dictionary. If `True`,
        :class:`scoter.ie.quakeml.Event` object attribute
        `arrival_dict` is set to a dictionary whose keys are tuples of
        (station_label, phase_label) and values are corresponding
        (time_residual, time_correction, distance_deg).

    Returns
    -------
    :class:`~scoter.ie.quakeml.QuakeML` object.

    Note
    ----
    There is a bug (or simplification!!!) in NonLinLoc, it gives
    negative origin time value for events occurred before midnight!

    Todo
    ----
    Values of -1.0 may be returned for `QML_ConfidenceEllipsoid`
    parameters. Set to None.

    Valid `public_id`s sould be assigned for `event` and `origin` types.
    """

    # Read file contents and remove empty lines
    flines = open(filename, 'r').read().splitlines()
    flines = filter(None, flines)

    # Indices of block start and end
    idx_hyp_block = [None, None]
    idx_pha_block = [None, None]
    for i, line in enumerate(flines):
        if line.startswith("NLLOC "):
            idx_hyp_block[0] = i
        elif line.startswith("END_NLLOC"):
            idx_hyp_block[1] = i
        elif line.startswith("PHASE "):
            idx_pha_block[0] = i
        elif line.startswith("END_PHASE"):
            idx_pha_block[1] = i

    # Skip any other lines around NLLOC block
    flines = flines[idx_hyp_block[0]:idx_hyp_block[1]]
    i1, i2 = idx_pha_block
    hyp_lines, pha_lines = flines[:i1] + flines[i2+1:], flines[i1+1:i2]

    # ------- Read hypocenter block -------

    hyp_lines = dict([line.split(None, 1) for line in hyp_lines])

    # NLLOC line.
    line = hyp_lines['NLLOC']
    stat = re.findall(r'LOCATED|REJECTED|ABORTED', line)
    if stat:
        evaluation_status = EVALUATION_STATUS[stat[0]]
    else:
        evaluation_status = 'NotSet'

    # TRANSFORM line
    line = hyp_lines['TRANSFORM']
    if line.split()[0] == 'GLOBAL':
        locmode = 'GLOBAL'
    else:
        locmode = 'Non-GLOBAL'

    # GEOGRAPHIC line
    line = hyp_lines['GEOGRAPHIC']
    items = line.split()
    torig_str = '-'.join(items[1:4]) + WSPACE + ':'.join(items[4:7])
    try:
        torig = str_to_time(torig_str, '%Y-%m-%d %H:%M:%S.OPTFRAC')
    except (ValueError, TimeStrError):
        # Negative origin time, see note above
        yr, mo, day = map(int, items[1:4])
        msec, sec = np.modf(float(items[6]))
        sec = abs(int(sec))
        msec = abs(int(msec * 1.0e+6))

        midnight = datetime.datetime(yr, mo, day, 0, 0, 0)
        tdelta = datetime.timedelta(days=0, seconds=sec, microseconds=msec)
        ttrue = midnight - tdelta
        torig = calendar.timegm(ttrue.timetuple())

    lat, lon, depth = map(float, items[-5::2])

    # STATISTICS line
    line = hyp_lines['STATISTICS']
    items = line.split(None)
    cov_xx = float(items[7])
    cov_yy = float(items[13])
    cov_zz = float(items[17])
    major_len = float(items[-1])
    inter_len = float(items[-3])
    minor_len = float(items[-9])
    major_plunge, major_azi, major_rot = (-999.9,) * 3

    # QML_OriginQuality line
    line = hyp_lines['QML_OriginQuality']
    items = line.split()
    (assoc_pha_count, used_pha_count, assoc_sta_count, used_sta_count,
        depth_pha_count) = map(int, items[1:11:2])
    stderr, azi_gap, sec_azi_gap = map(float, items[11:17:2])
    gt_level = line[17]
    min_dist, max_dist, med_dist = map(float, items[19:25:2])

    # QML_OriginUncertainty line
    line = hyp_lines['QML_OriginUncertainty']
    items = line.split()
    hor_unc, min_hor_unc, max_hor_unc, azi_hor_unc = map(float, items[1:8:2])

    # QML_ConfidenceEllipsoid line (in NonLinLoc >= v7.00)
    conf_ellips = 'QML_ConfidenceEllipsoid'
    if conf_ellips in hyp_lines:
        line = hyp_lines[conf_ellips]
        items = line.split()
        major_plunge, major_azi, major_rot = map(float, items[-5::2])

    # --- Create QuakeML attributes ---

    torig = TimeQuantity(value=torig)
    lat = RealQuantity(value=lat)
    lon = RealQuantity(value=lon)
    depth = RealQuantity(value=depth*KM2M)   # meters!
    lat.uncertainty = math.sqrt(cov_yy)
    lon.uncertainty = math.sqrt(cov_xx)
    depth.uncertainty = math.sqrt(cov_zz) * KM2M   # meters!
    for field in (lat, lon, depth):
        setattr(field, 'confidence_level', 68.0)

    origin_quality = OriginQuality(
        associated_phase_count=assoc_pha_count,
        used_phase_count=used_pha_count,
        associated_station_count=assoc_sta_count,
        used_station_count=used_sta_count,
        depth_phase_count=depth_pha_count,
        standard_error=stderr,
        azimuthal_gap=azi_gap,
        secondary_azimuthal_gap=sec_azi_gap,
        ground_truth_level=gt_level,
        maximum_distance=max_dist*KM2M*M2DEG,
        minimum_distance=min_dist*KM2M*M2DEG,
        median_distance=med_dist*KM2M*M2DEG)

    confidence_ellipsoid = ConfidenceEllipsoid(
        semi_major_axis_length=major_len*KM2M,   # meters!
        semi_minor_axis_length=minor_len*KM2M,   # meters!
        semi_intermediate_axis_length=inter_len*KM2M,   # meters!
        major_axis_plunge=major_plunge,
        major_axis_azimuth=major_azi,
        major_axis_rotation=major_rot)

    if conf_ellips in hyp_lines:
        preferred_description = 'confidence ellipsoid'
    else:
        preferred_description = 'uncertainty ellipse'

    origin_uncertainty = OriginUncertainty(
        horizontal_uncertainty=hor_unc,
        min_horizontal_uncertainty=min_hor_unc,
        max_horizontal_uncertainty=max_hor_unc,
        azimuth_max_horizontal_uncertainty=azi_hor_unc,
        confidence_ellipsoid=confidence_ellipsoid,
        preferred_description=preferred_description,
        confidence_level=68.0)

    # Values of -1.0, which are used for unset values, should be set to None
    for field in (
            'horizontal_uncertainty',
            'min_horizontal_uncertainty',
            'max_horizontal_uncertainty'):
        val = getattr(origin_uncertainty, field)
        if val == -1:
            setattr(origin_uncertainty, field, None)
        else:
            setattr(origin_uncertainty, field, val*KM2M)   # meters!

    # ------- Read phase block -------

    pick_list = []
    arrival_list = []
    arrival_dict = {}

    for i, line in enumerate(pha_lines):
        items = line.split()

        ttpred, tres, weight = map(float, items[15:18])

        # Does not include picks not used in the location.
        if (ttpred == 0.0) or (weight == 0.0):
            continue

        slabel = items[0].strip()
        if delimiter_str:
            net, sta = slabel.split(delimiter_str)
        else:
            net, sta = '', slabel

        cha = items[2].strip()

        onset = ONSETS_LOAD.get(items[3].lower(), None)
        plabel = items[4].strip()

        tpick_str = WSPACE.join(items[6:9])
        try:
            tpick = str_to_time(tpick_str, format='%Y%m%d %H%M %S.OPTFRAC')
        except TimeStrError:
            mn = items[7][2:]
            hrmn = '00' + mn
            tpick_str = WSPACE.join((items[6], hrmn, items[8]))
            tpick = str_to_time(tpick_str, format='%Y%m%d %H%M %S.OPTFRAC')
            tpick += 24 * 3600   # add one day

        dist = float(items[21])
        if locmode != 'GLOBAL':
            dist *= KM2M * M2DEG
        azi = float(items[22])
        tcor = float(items[26])

        # --- Create QuakeML attributes ---

        waveform_id = WaveformStreamID(
            network_code=net, station_code=sta, channel_code=cha, value='')

        phase = Phase(value=plabel)
        tpick = TimeQuantity(value=tpick)

        dummy_id = 'smi:local/{}'.format(i)

        # Pick attribute
        pick = Pick(
            public_id=dummy_id, time=tpick, onset=onset,
            phase_hint=phase, waveform_id=waveform_id)

        pick_list.append(pick)

        # Arrival attribute
        arrival = Arrival(
            public_id=dummy_id, pick_id=dummy_id, phase=phase,
            time_correction=tcor, azimuth=azi, distance=dist,
            time_residual=tres, time_weight=weight)

        arrival_list.append(arrival)

        # SCOTER arrival
        arrival_dict[slabel, plabel] = (tres, tcor, dist)

    # ------- Create quakeml.Event object -------

    public_id = 'smi:local/{}'.format(event_name or 'NotSet')
    origin = Origin(
        public_id=public_id, time=torig, latitude=lat, longitude=lon,
        depth=depth, arrival_list=arrival_list,
        origin_uncertainty_list=[origin_uncertainty],
        quality=origin_quality, evaluation_status=evaluation_status)

    if not set_arrival_dict:
        arrival_dict = None

    event = Event(
        public_id=public_id,
        origin_list=[origin],
        pick_list=pick_list,
        arrival_dict=arrival_dict)

    event_parameters = EventParameters(
        public_id='smi:local/EventParameters',
        event_list=[event])

    qml = QuakeML(event_parameters=event_parameters)

    return qml


def dump_nlloc_obs(event, filename, delimiter_str=None):
    """
    Write a NonLinLoc phase file (NLLOC_OBS) from
    :class:`~scoter.ie.quakeml.Event`object.

    Parameters
    ----------
    event : :class:`~scoter.ie.quakeml.Event` object
        The quakeml Event object to write.
    filename : str
        File name to write.
    delimiter_str : str or None (default: None)
        If not `None`, it specifies the network code and station code
        separator character, i.e. it is used to join the network code
        and station code to be written as the station label in the
        output NonLinLoc phase file. If it is absent or `None`, the
        written station label would be the concatenation of network code
        and station code without any separator character.

    Warning
    -------
    This function should be called consciously. Since writing NonLinLoc
    phase file is only supported for a single Event, the input argument
    is :class:`~scoter.ie.quakeml.Event` object instead of
    :class:`~scoter.ie.quakeml.QuakeML` object. The latter may contain
    more than one Event object. In such case, use a for loop over
    :attribute:`~scoter.ie.quakeml.QuakeML.event_parameters.event_list`
    and provide an ouput file name for each Event.
    """

    if not isinstance(event, Event):
        raise NLLocError("not a quakeml.Event type: {}".format(type(event)))

    # Header
    hdr = 'PHASE ID Ins Cmp On Pha FM Date HrMn Sec Err ErrMag Coda Amp Per\n'

    p = '{slabel:8s} {inst:4s} {comp:4s} {onset:1s} {pha:6s} {pol:1s} '\
        '{time_str:s} GAU {err_mag:9.2e} {coda:9.2e} {amp:9.2e} {per:9.2e}\n'

    stream = '' + hdr
    orig = event.preferred_origin

    for pick in event.pick_list:

        wid = pick.waveform_id
        if delimiter_str:
            slabel = delimiter_str.join([wid.network_code, wid.station_code])
        else:
            slabel = wid.network_code + wid.station_code

        comp = wid.channel_code and wid.channel_code[-1].upper() or '?'
        onset = ONSETS_DUMP.get(pick.onset, '?')
        pha = pick.phase_hint.value or '?'
        pol = POLARITIES_DUMP.get(pick.polarity, '?')
        time_str = time_to_str(pick.time.value, format='%Y%m%d %H%M %S.4FRAC')

        err_mag = pick.time.uncertainty or -1.0

        if orig is not None:
            arr = [x for x in orig.arrival_list if
                   x.pick_id == pick.public_id][0]

            if arr.time_weight is None and arr.time_used is None:
                continue

            if arr.time_weight == 0.0 or arr.time_used == 0:
                continue

        stream += p.format(
            slabel=slabel, inst='?', comp=comp, onset=onset, pha=pha, pol=pol,
            time_str=time_str, err_mag=err_mag, coda=-1.0, amp=-1.0, per=-1.0)

    # Footer
    stream += 'END_PHASE'

    ensuredirs(filename)

    with open(filename, 'w') as f:
        f.write(stream)


def load_nlloc_sta(filename, delimiter_str=None):
    """
    Read NonLinLoc stations file.

    Currently, it only supports `LATLON` format (see NonLinLoc
    documentation for more details).

    Parameters
    ----------
    filename : str
        NonLinLoc stations file name.
    delimiter_str : str or None (default: None)
        If not `None`, it specifies a string to be used as the network
        code and station code separator character, i.e. it is used to
        split the station label in NonLinLoc hypocenter-phase file. If
        it is absent or `None` (default), the entire station label read
        is considered as the `station code` and an empty string is
        assigned to the `network code` for each pick in the output
        QuakeML event.

    Returns
    -------
    list, :class:`pyrocko.model.Station`
        List of :class:`pyrocko.model.Station` objects.
    """

    stations = []
    lines = open(filename, 'r').read().splitlines()
    lines = [x for x in lines if x.startswith('LOCSRCE')]

    for line in lines:
        items = line.split()

        if delimiter_str:
            net, sta = items[1].split(delimiter_str)
        else:
            net, sta = '', items[1]

        lat, lon, depth, elev = map(float, items[3:])

        stations.append(Station(
            network=net,
            station=sta,
            lat=lat,
            lon=lon,
            elevation=elev,
            depth=depth))

    return stations


def dump_nlloc_sta(stations, filename, delimiter_str=None, zh_unit='m'):
    """
    Write a NonLinLoc station file from a list of
    :class:`pyrocko.model.Station` objects.

    The output station file format is:
        `LOCSRCE label LATLON lat lon depth elev`

    (see http://alomax.free.fr/nlloc/ -> Control File -> NLLoc -> LOCSRCE)

    Parameters
    ----------
    stations : list
        List of :class:`pyrocko.model.Station` objects to write.
    filename : str
        File name to write.
    delimiter_str : str or None (default: None)
        If not `None`, it specifies the network code and station code
        separator character, i.e. it is used to join the network and
        station codes to be written as the station label in the output
        NonLinLoc phase file. If it is absent or `None` (default), the
        written station label would be the concatenation of network code
        and station code without any separator character.
    zh_unit : {'m', 'km'}, optional
        The units of the station depths and elevations, either in meters
        'm' or in kilometers 'km' (default: 'm').
    """

    conv = {'m': M2KM, 'km': 1.0}
    C = conv[zh_unit.lower()]

    p = 'LOCSRCE {slabel:8s} LATLON {lat:8.4f} {lon:9.4f} {z:8.4f} {h:8.4f}\n'

    ensuredirs(filename)

    stations = sorted(stations, key=lambda s: s.name)

    with open(filename, 'w') as ofile:
        for s in stations:
            if delimiter_str:
                slabel = delimiter_str.join((s.network, s.station))
            else:
                slabel = s.network + s.station

            slabel = slabel.ljust(8)

            newline = p.format(
                slabel=slabel, lat=s.lat, lon=s.lon,
                z=s.depth*C, h=s.elevation*C)

            ofile.write(newline)


__all__ = """
    load_nlloc_hyp
    dump_nlloc_obs
    load_nlloc_sta
    dump_nlloc_sta
""".split()
