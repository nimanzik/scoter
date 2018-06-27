# -*- coding: utf-8 -*-

import unittest

import numpy as np

from scoter import stats


class StatsTestCase(unittest.TestCase):

    def test_median_absolute_deviation(self):
        data = [
            (3, 8, 8, 8, 8, 9, 9, 9, 9),
            (1, 1, 2, 2, 2, 4, 4, 6, 9)]

        for d in data:
            self.assertEqual(stats.median_absolute_deviation(d), 1)

        np.testing.assert_equal(
            stats.median_absolute_deviation(np.array(data), axis=1), [1., 1.])

    def test_bisquared(self):
        data = np.array([0., 0.5, 1., 2., 4., 11])
        cutoff = 2.
        np.testing.assert_array_equal(
            stats.bisquared(data, cutoff),
            np.array([1., 0.87890625, .5625, 0., 0., 0.]))


if __name__ == '__main__':
    unittest.main()
