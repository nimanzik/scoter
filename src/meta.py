import os.path as op
from string import Template

from pyrocko.model import Station as PyrockoStation
from pyrocko.guts import Choice, Float, Int, List, Object, String


guts_prefix = 'gp'

KM2M = 1000.
M2KM = 1. / KM2M
WSPACE = ''.ljust(1)
LABEL_LENGTH = 27


class ScoterError(Exception):
    pass


class FileNotFound(Exception):
    pass


class PathAlreadyExists(Exception):
    pass


class Path(String):
    pass


class PhaseLabel(String):
    pass


class StationLabel(String):
    pass


class Station(PyrockoStation):
    """Base class for station."""

    def __init__(self, x=None, y=None, z=None, **kwargs):
        PyrockoStation.__init__(self, **kwargs)
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        params = [
            'LOCSRCE', '{self.name:8s}', 'LATLON', '{self.lat:.4f}',
            '{self.lon:.4f}', '{depth_in_km:.2f} {elevation_in_km:.4f}']

        return WSPACE.join(params).format(
            self=self,
            depth_in_km=self.depth * M2KM,
            elevation_in_km=self.elevation * M2KM)


class Delay(Object):
    """Base class for station time delay."""

    station_label = StationLabel.T()
    phase_label = PhaseLabel.T()
    nresiduals = Int.T(help='number of residuals used')
    time_correction = Float.T()
    standard_deviation = Float.T()

    def __str__(self):
        params = [
            'LOCDELAY', '{self.station_label:8s}', '{self.phase_label:6s}',
            '{self.nresiduals:6d}', '{self.time_correction:8.4f}',
            '{self.standard_deviation:7.4f}']

        return WSPACE.join(params).format(self=self)


class Target(Object):
    """Base class for target (individual) event."""

    name = String.T()
    station_labels = List.T(StationLabel.T())
    station_delays = Choice.T(
        choices=[List.T(Delay.T()), Path.T()], optional=True)

    def set_station_delays(self, delays_or_path):
        self.station_delays = delays_or_path


def xjoin(basepath, path):
    if path is None and basepath is not None:
        return basepath
    elif op.isabs(path) or basepath is None:
        return path
    else:
        return op.join(basepath, path)


def xrelpath(path, start):
    if op.isabs(path):
        return path
    else:
        return op.relpath(path, start)


def expand_template(template, d):
    try:
        return Template(template).substitute(d)
    except KeyError as e:
        raise ScoterError(
            'invalid placeholder "{0}" in template: "{1}"'.format(e, template))
    except ValueError:
        raise ScoterError(
            'malformed placeholder in template: "{}"'.format(template))


class HasPaths(Object):
    path_prefix = Path.T(optional=True)

    def __init__(self, **kwargs):
        Object.__init__(self, **kwargs)
        self._basepath = None
        self._parent_path_prefix = None

    def set_basepath(self, basepath, parent_path_prefix=None):
        self._basepath = basepath
        self._parent_path_prefix = parent_path_prefix

        for (prop, val) in self.T.ipropvals(self):
            if isinstance(val, HasPaths):
                val.set_basepath(
                    basepath, self.path_prefix or self._parent_path_prefix)

    def get_basepath(self):
        assert self._basepath is not None
        return self._basepath

    def change_basepath(self, new_basepath, parent_path_prefix=None):
        assert self._basepath is not None
        self._parent_path_prefix = parent_path_prefix
        if self.path_prefix or not self._parent_path_prefix:
            self.path_prefix = op.normpath(xjoin(xrelpath(
                self._basepath, new_basepath), self.path_prefix))
        for val in self.T.ivals(self):
            if isinstance(val, HasPaths):
                val.change_basepath(
                    new_basepath, self.path_prefix or self._parent_path_prefix)
        self._basepath = new_basepath

    def expand_path(self, path, extra=None):
        assert self._basepath is not None
        if extra is None:
            def extra(path):
                return path
        path_prefix = self.path_prefix or self._parent_path_prefix
        if path is None:
            return None
        elif isinstance(path, basestring):
            return extra(
                op.normpath(xjoin(self._basepath, xjoin(path_prefix, path))))
        else:
            return [
                extra(
                    op.normpath(xjoin(self._basepath, xjoin(path_prefix, p))))
                for p in path]


__all__ = """
    ScoterError
    FileNotFound
    Path
    PhaseLabel
    StationLabel
    Station
    Delay
    Target
    expand_template
    HasPaths
""".split()
