# -*- coding: utf-8 -*-

import unittest
import itertools

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa
import numpy as np

from pyrocko import cake
from pyrocko.guts import Object, Float

from scoter import spatial, geodetic


KM2M = 1000.
M2KM = 1. / KM2M


class CartesianLocation(Object):
    x = Float.T()
    y = Float.T()
    z = Float.T()

    @property
    def coords(self):
        return np.array([self.x, self.y, self.z])


class GeographicLocation(Object):
    lat = Float.T()
    lon = Float.T()
    depth = Float.T(help='Unit: m')

    def __init__(self, **kwargs):
        Object.__init__(self, **kwargs)
        self.cartesian_location = self.get_cartesian_location()

    def get_cartesian_location(self):
        x, y, z = geodetic.geodetic_to_ecef(self.lat, self.lon, -1.*self.depth)
        return CartesianLocation(x=x, y=y, z=z)

    @property
    def coords(self):
        return np.array([self.lat, self.lon, self.depth])


class SpatialTestCase(unittest.TestCase):

    def test_ray_takeoff_direction(self):
        OC = np.array([-2, 7])
        OR = np.array([12, 9])
        takeoff = 40.4
        CD = spatial.ray_takeoff_direction(OC, OR, takeoff)
        OD = OC + CD
        np.testing.assert_almost_equal(OD, np.array([4., 3.]), decimal=0)

    def test_opening_angle(cls):
        target = GeographicLocation(lat=0., lon=0., depth=300.*KM2M)
        receiver = GeographicLocation(lat=0., lon=-5., depth=0.)
        radius = 100. * KM2M
        npoints = 90

        # Load velocity model and seismic phase type.
        model = cake.load_model('ak135-f-average.m')
        phases = [cake.PhaseDef('p'), cake.PhaseDef('P')]

        # Get the takeoff angle for the `target` event recorded
        # at the `receiver`.
        distance = geodetic.ellipsoid_distance(
            target.lat,
            target.lon,
            receiver.lat,
            receiver.lon) * geodetic.M2DEG

        arrivals = model.arrivals(
            phases=phases,
            distances=[distance],
            zstart=target.depth,
            zstop=receiver.depth)

        arrivals.sort(key=lambda arr: arr.t)

        if arrivals:
            # Takeoff angle in [deg].
            takeoff_deg = arrivals[0].takeoff_angle()
        else:
            raise Exception(
                "No travel time for given source-receiver geometry.")

        # Get ray takeoff direction.
        CD = spatial.ray_takeoff_direction(
            target.cartesian_location.coords,
            receiver.cartesian_location.coords,
            takeoff_deg)

        # Neighbors
        thetas = np.linspace(0., np.pi, npoints)
        phis = np.linspace(0., 2*np.pi, 2*npoints)
        neighbors = np.zeros((thetas.size*phis.size, 4))

        for i, (theta, phi) in enumerate(itertools.product(thetas, phis)):
            x = radius * np.sin(theta) * np.cos(phi) + \
                target.cartesian_location.x

            y = radius * np.sin(theta) * np.sin(phi) + \
                target.cartesian_location.y

            z = radius * np.cos(theta) + target.cartesian_location.z

            lat, lon, alt = geodetic.ecef_to_geodetic(x, y, z)
            depth = -1. * alt

            CN = np.asarray([x, y, z]) - \
                np.asarray(target.cartesian_location.coords)

            alpha = spatial.opening_angle(CD, CN)
            w_ed = spatial.get_w_ed(0., alpha)

            neighbors[i, :] = (lat, lon, depth, w_ed)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d', aspect='equal')
        p = ax.scatter(
            neighbors[:, 0],
            neighbors[:, 1],
            neighbors[:, 2]*M2KM,
            c=neighbors[:, 3],
            vmin=0.,
            vmax=1.,
            marker='o',
            s=10,
            cmap='plasma_r')

        cbar = fig.colorbar(   # noqa
            p,
            label='Weight',
            shrink=0.5,
            pad=0.07,
            ticks=np.arange(0., 1.2, 0.2))

        ax.set_xlabel('Lat [deg]')
        ax.set_ylabel('Lon [deg]')
        ax.set_zlabel('Depth [km]')

        dh = 1.5 * geodetic.M2DEG * radius
        dv = 1.5 * radius   # in meter
        ax.set_xlim(target.lat-dh, target.lat+dh)
        ax.set_ylim(target.lon-dh, target.lon+dh)
        ax.set_zlim((target.depth-dv)*M2KM, (target.depth+dv)*M2KM)

        ax.invert_xaxis()
        ax.invert_zaxis()

        ax.view_init(azim=-40., elev=15)

        plt.show()


if __name__ == '__main__':
    unittest.main()
