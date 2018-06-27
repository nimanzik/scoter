# -*- coding: utf-8 -*-

import unittest

import numpy as np

from pyrocko import orthodrome

from scoter import geodetic


KM2M = 1000.


class GeodeticTestCase(unittest.TestCase):

    def test_gc_distance(self):
        self.assertAlmostEqual(
            geodetic.gc_distance(0., 0., 0., 120.), 120.)

        self.assertAlmostEqual(
            geodetic.gc_distance(0., 0., 45., 0.), 45.)

        p = (40., 10., 75., 162.)
        self.assertEqual(
            geodetic.gc_distance(*p),
            np.arccos(orthodrome.cosdelta(*p)) * orthodrome.r2d)

    def test_gc_azibazi(self):
        self.assertEqual(geodetic.gc_azibazi(0., 0., 0., 10.), (90., -90.))
        self.assertEqual(geodetic.gc_azibazi(0., 0., 30., 0.), (0., 180.))

        p = (40., 10., 75., 162.)
        self.assertEqual(
            geodetic.gc_azibazi(*p), orthodrome.azibazi(*p))

    def test_ellipsoid_distance(self):
        # Berlin-Tokyo
        p = (52.51666666666667, 13.4, 35.7, 139.76666666666667)
        self.assertEqual(
            geodetic.ellipsoid_distance(*p)/KM2M, 8941.20250458698)

    def test_geodetic_to_ecef(self):
        wgs = geodetic.WGS84()
        a = wgs.a
        b = wgs.b

        coords = [
            ((90., 0., 0.), (0., 0., b)),
            ((-90., 0., 10.*KM2M), (0., 0., -b-10.*KM2M)),
            ((0., 0., 0.), (a, 0., 0.))]

        for coord in coords:
            np.testing.assert_almost_equal(
                geodetic.geodetic_to_ecef(*coord[0]), coord[1])

    def test_ecef_to_geodetic(self):
        npoints = 10
        lats = np.random.uniform(-90., 90., size=npoints)
        lons = np.random.uniform(-180., 180., size=npoints)
        alts = np.random.uniform(-50., 50., size=npoints) * KM2M

        points = np.array([lats, lons, alts]).T

        for ip in xrange(points.shape[0]):
            xyz = geodetic.geodetic_to_ecef(*points[ip, :])
            lla = geodetic.ecef_to_geodetic(*xyz)

            np.testing.assert_almost_equal(lla, points[ip, :])


if __name__ == '__main__':
    unittest.main()
