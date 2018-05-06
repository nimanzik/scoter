import re

import numpy as np

from pyrocko.util import str_to_time

from .quakeml import Arrival, Event, EventParameters, Origin, OriginQuality,\
    Pick, Phase, RealQuantity, TimeQuantity, WaveformStreamID, QuakeML


guts_prefix = 'gp'

KM2M = 1000.

EVALUATION_MODES = {'M': 'manual', 'A': 'automatic'}


def load_geofon_hyp(filename):
    """
    Read GEOFON hypocenter-phase file.

    Parameters
    ----------
    filename: str
        Hypocenter-phase file name.

    Returns
    -------
    :class:`quakeml.QuakeML` instance.
    """

    data = open(filename, 'r').read()

    # ----- Event block. -----

    event_public_id = re.search(
        r'^\s*Public\s+ID\s+(gfz\d{4}[a-z]{4})', data, re.M)
    event_public_id = 'smi:local/{}'.format(event_public_id.group(1))

    origin_public_id = re.search(
        r'^\s*Preferred\s+Origin\s+ID\s+(.+)', data, re.M)
    try:
        origin_public_id = 'smi:local/{}'.format(origin_public_id.group(1))
    except AttributeError:
        origin_public_id = 'smi:local/{}'.format('NotSet')

    # ----- Origin block. -----

    od = re.search(r'^\s*Date\s+(\d\d\d\d-\d\d-\d\d)', data, re.M)
    ot = re.search(r'^\s*Time\s+(\d\d:\d\d:\d\d\.\d+)', data, re.M)
    origin_time_str = ' '.join((od.group(1), ot.group(1)))
    origin_time = str_to_time(
        origin_time_str, format='%Y-%m-%d %H:%M:%S.OPTFRAC')
    origin_time = TimeQuantity(value=origin_time)

    lat = re.search(r'^\s*Latitude\s+(-?\d{1,2}\.\d\d)\sdeg', data, re.M)
    lat = float(lat.group(1))
    lat = RealQuantity(value=lat)

    lon = re.search(r'^\s*Longitude\s+(-?\d{1,3}\.\d\d)\sdeg', data, re.M)
    lon = float(lon.group(1))
    lon = RealQuantity(value=lon)

    depth = re.search(r'^\s*Depth\s+(\d{1,3})\skm', data, re.M)
    try:
        depth = float(depth.group(1)) * KM2M
    except AttributeError:
        pass
    else:
        depth = RealQuantity(value=depth)

    try:
        eval_mode = re.search(
            r'^\s*Mode\s+(manual|automatic)\s*$', data, re.M).group(1)
    except AttributeError:
        eval_mode = None

    try:
        eval_status = re.search(
            r'^\s*Status\s+(preliminary|confirmed|reviewed|final|'
            r'rejected)\s*$', data, re.M).group(1)
    except AttributeError:
        eval_status = None

    try:
        res_rms = re.search(
            r'^\s*Residual\s+RMS\s+(\d+\.\d\d)\ss', data, re.M).group(1)
        res_rms = float(res_rms)
    except AttributeError:
        res_rms = None

    try:
        azi_gap = re.search(
            r'^\s*Azimuthal\s+gap\s+(\d+)\sdeg', data, re.M).group(1)
        azi_gap = float(azi_gap)
    except AttributeError:
        azi_gap = None

    origin_quality = OriginQuality(
        standard_error=res_rms, azimuthal_gap=azi_gap)

    # ----- Magnitudes block. -----

    mag = re.search(r'\s*.(\d.\d\d).+preferred', data, re.M).group(1)
    mag = float(mag)   # noqa

    # ----- Phase arrivals block. -----

    # exclude picks with MX and AX qulity (e.g. picks weighted to zero)!
    p = re.compile(
        r'^\s*([A-Z0-9]+)\s+([A-Z]+)\s+'
        r'(\d+\.\d+)\s+(\d+)\s+([a-zA-Z]+)\s+'
        r'(\d\d:\d\d:\d\d\.\d)\s+'
        r'(-?\d+\.\d|N/A)\s(M|A)\s+'
        r'(\d\.\d)\s+[A-Z0-9]+\s*$', re.M)

    arrivals = p.findall(data)

    pick_list = []
    arrival_list = []

    for iq, q in enumerate(arrivals):

        eval_mode = q[7].strip()
        if eval_mode.endswith('X'):
            continue
        eval_mode = EVALUATION_MODES[eval_mode]

        sta, net = q[0].strip(), q[1].strip()
        wid = WaveformStreamID(network_code=net, station_code=sta, value='')

        dist = float(q[2])   # in degrees
        azi = float(q[3])    # in degrees
        pha = Phase(value=q[4].strip())

        arrtime = ' '.join((od.group(1), q[5]))
        arrtime = str_to_time(arrtime, format='%Y-%m-%d %H:%M:%S.OPTFRAC')
        if arrtime < origin_time.value:
            # After midnight arrival times (next day).
            arrtime += 86400

        arrtime = TimeQuantity(value=arrtime)

        try:
            res = float(q[6])
        except ValueError:
            res = np.nan

        wt = float(q[8])

        dummy_id = 'smi:local/%d' % iq

        # Pick attribute.
        pick = Pick(
            public_id=dummy_id, time=arrtime, evaluation_mode=eval_mode,
            phase_hint=pha, waveform_id=wid)

        pick_list.append(pick)

        # Arrival attribute.
        arrival = Arrival(
            public_id=dummy_id, pick_id=dummy_id, phase=pha, azimuth=azi,
            distance=dist, time_residual=res, time_weight=wt)

        arrival_list.append(arrival)

    # Origin attribute.
    origin = Origin(
        public_id=origin_public_id, arrival_list=arrival_list, latitude=lat,
        longitude=lon, depth=depth, time=origin_time, quality=origin_quality,
        evaluation_mode=eval_mode, evaluation_status=eval_status)

    event = Event(
        public_id=event_public_id, origin_list=[origin], pick_list=pick_list)

    event_parameters = EventParameters(
        public_id='smi:local/EventParameters',
        event_list=[event])

    qml = QuakeML(event_parameters=event_parameters)

    return qml


__all__ = ['load_geofon_hyp']
