# -*- coding: utf-8 -*-

import unittest

import numpy as np

from scoter import util


class UtilTestCase(unittest.TestCase):

    def test_loglinspace(self):
        a, b, n = 1, 20, 50
        x = np.power(10, np.linspace(np.log10(a), np.log10(b), n))
        y = util.loglinspace(a, b, n)
        np.testing.assert_equal(y, x)

        yy = np.log10(y)
        self.assertTrue(np.all((np.ediff1d(yy) - yy[0]) < 0.1))


if __name__ == '__main__':
    unittest.main()
