# -*- coding: utf-8 -*-

import os.path as op
import sys

import numpy as np

from pyrocko.guts import Object, Int, Float, StringChoice, String


g_fstem_templates = {
    'time': '%(basename)s.%(wave_type)s.%(slabel)s.%(grid_type)s',
    'vel': '%(basename)s.%(wave_type)s.mod'}

g_float_types = {'FLOAT': 'f4', 'DOUBLE': 'f8'}

sys_is_le = sys.byteorder == 'little'


def istimegrid(grid_type):
    if grid_type.startswith('TIME') or grid_type.startswith('ANGLE'):
        return True
    return False


def native_byte_order():
    native_code = sys_is_le and '<' or '>'
    return native_code


def swapped_byte_order():
    swapped_code = sys_is_le and '>' or '<'
    return swapped_code


class Station(Object):
    name = String.T(default='')
    lat = Float.T(default=0.0)
    lon = Float.T(default=0.0)
    depth = Float.T(default=0.0, help='Unit: [m]')
    elevation = Float.T(default=0.0, help='Unit: [m]')


class FloatType(StringChoice):
    choices = ['FLOAT', 'DOUBLE']


class GridType(StringChoice):
    choices = [
        'VELOCITY',
        'VELOCITY_METERS',
        'SLOWNESS',
        'VEL2',
        'SLOW2',
        'SLOW2_METERS',
        'SLOW_LEN',
        'PROB_DENSITY',
        'MISFIT',
        'TIME',
        'TIME2D',
        'ANGLE',
        'ANGLE2D']


class GridShape(Object):
    nx = Int.T(help='number of grid nodes in the x or lon')
    ny = Int.T(help='number of grid nodes in the y or lat')
    nz = Int.T(help='number of grid nodes in the z or depth')


class GridOrigin(Object):
    x = Float.T(
        help='x location of the grid origin in [km] relative to the '
             'geographic origin (Non-GLOBAL) or longitude in [deg] of '
             'the location of the south-west corner of the grid (GLOBAL)')
    y = Float.T(
        help='y location of the grid origin in [km] relative to the '
             'geographic origin (Non-GLOBAL) or latitude in [deg] of '
             'the location of the south-west corner of the grid (GLOBAL)')
    z = Float.T(
        help='z location of the grid origin in [km] relative to the '
             'geographic origin (Non-GLOBAL) or z in [km] of the '
             'location of the south-west corner of the grid (GLOBAL)')


class GridSpacing(Object):
    dx = Float.T(
        help='grid node spacing in [km] (Non-GLOBAL) or in [deg] (GLOBAL)'
             'along the x')
    dy = Float.T(
        help='grid node spacing in [km] (Non-GLOBAL) or in [deg] (GLOBAL)'
             'along the y')
    dz = Float.T(
        help='grid node spacing in [km] along the z')


class NLLGrid(Object):
    basename = String.T()
    float_type = FloatType.T()
    grid_type = GridType.T()
    wave_type = String.T()
    shape = GridShape.T()
    origin = GridOrigin.T()
    spacing = GridSpacing.T()
    station = Station.T(optional=True)

    def __init__(self, data_array=None, **kwargs):
        Object.__init__(self, **kwargs)
        self.station = kwargs.get('station') or Station(name='DEFAULT')
        self.data_array = data_array
        self._nodes_x = None
        self._nodes_y = None
        self._nodes_z = None

    def __get_nodes_dim(self, dim):
        u0 = getattr(self.origin, dim)
        nu = getattr(self.shape, 'n{}'.format(dim))
        du = getattr(self.spacing, 'd{}'.format(dim))
        return np.linspace(u0, u0 + (nu-1)*du, nu)

    @property
    def nodes_x(self):
        if self._nodes_x is None:
            self._nodes_x = self.__get_nodes_dim('x')
        return self._nodes_x

    @property
    def nodes_y(self):
        if self._nodes_y is None:
            self._nodes_y = self.__get_nodes_dim('y')
        return self._nodes_y

    @property
    def nodes_z(self):
        if self._nodes_z is None:
            self._nodes_z = self.__get_nodes_dim('z')
        return self._nodes_z

    def __get_fn_template(self):
        if istimegrid(self.grid_type):
            tmpl = g_fstem_templates['time']
        else:
            tmpl = g_fstem_templates['vel']

        return tmpl % dict(
            basename=self.basename,
            wave_type=self.wave_type,
            grid_type=self.grid_type.startswith('TIME') and 'time' or 'angle',
            slabel=self.station.name)

    def write_hdr(self):
        lines = []
        lines.append(
            '{nx} {ny} {nz} {x0} {y0} {z0} {dx} {dy} {dz} {gtyp} '
            '{ftype}\n'.format(
                nx=self.shape.nx,
                ny=self.shape.ny,
                nz=self.shape.nz,
                x0=self.origin.x,
                y0=self.origin.y,
                z0=self.origin.z,
                dx=self.spacing.dx,
                dy=self.spacing.dy,
                dz=self.spacing.dz,
                gtyp=self.grid_type,
                ftype=self.float_type))

        lines.append('{label} {x_s} {y_s} {z_s}\n'.format(
            label=self.station.name,
            x_s=self.station.lon,
            y_s=self.station.lat,
            z_s=self.station.elevation*(-0.001)+0.0))   # z must be in [km]

        fstem = self.__get_fn_template()
        fname = fstem + '.hdr'
        with open(fname, 'wb') as fid:
            for line in lines:
                fid.write(line)

    def write_buf(self):
        dt = np.dtype(native_byte_order() + g_float_types[self.float_type])
        if sys_is_le is False:
            # (most commonly) programs written in 'C' use little-endian order
            dt = dt.newbyteorder(swapped_byte_order())

        fstem = self.__get_fn_template()
        fname = fstem + '.buf'
        with open(fname, 'wb') as fid:
            # NOTE - ndarray.tofile() always writes data in 'C' order
            self.data_array.astype(dt).tofile(fid)


def read_nll_grid(path, swapbytes=False):
    dname, fstem = op.split(path)
    if fstem.endswith('.hdr') or fstem.endswith('.buf'):
        fstem = op.splitext(fstem)[0]

    fn_hdr = op.join(dname, fstem+'.hdr')
    fn_buf = op.join(dname, fstem+'.buf')

    assert op.exists(fn_hdr), 'no such file or directory: {}'.format(fn_hdr)
    assert op.exists(fn_buf), 'no such file or directory: {}'.format(fn_buf)

    basename_tail, wave_type = fstem.split('.')[:2]
    basename = op.join(dname, basename_tail)

    # --- read header file ---
    with open(fn_hdr, 'r') as fid:
        lines = fid.read().splitlines()
        toks0 = lines[0].split()

        nx, ny, nz = map(int, toks0[0:3])
        shape = GridShape(nx=nx, ny=ny, nz=nz)

        x0, y0, z0 = map(float, toks0[3:6])
        origin = GridOrigin(x=x0, y=y0, z=z0)

        dx, dy, dz = map(float, toks0[6:9])
        spacing = GridSpacing(dx=dx, dy=dy, dz=dz)

        grid_type = toks0[9].strip()

        try:
            float_type = toks0[10].strip()
        except IndexError:
            float_type = 'FLOAT'

        if istimegrid(grid_type):
            toks1 = lines[1].split()
            label_s = toks1[0].strip()
            x_or_lon_s, y_or_lat_s, z_s = map(float, toks1[1:])   # z in [km]
            z_s *= 1000.
            elev_s = -1 * z_s
            station = Station(
                name=label_s,
                lat=y_or_lat_s,
                lon=x_or_lon_s,
                elevation=elev_s)
        else:
            station = None

    # --- read buffer file ---
    with open(fn_buf, 'rb') as fid:
        dt = np.dtype(native_byte_order() + g_float_types[float_type])
        if swapbytes:
            dt = dt.newbyteorder(swapped_byte_order())

        data_array = np.fromfile(fid, dtype=dt, count=nx*ny*nz)
        data_array = data_array.reshape(nx, ny, nz)

    return NLLGrid(
        basename=basename,
        float_type=float_type,
        grid_type=grid_type,
        wave_type=wave_type,
        shape=shape,
        origin=origin,
        spacing=spacing,
        station=station,
        data_array=data_array)


__all__ = [
    'GridShape',
    'GridOrigin',
    'GridSpacing',
    'NLLGrid',
    'read_nll_grid']
