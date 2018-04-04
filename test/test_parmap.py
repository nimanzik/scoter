import unittest
import itertools

import numpy as np

from scoter import parmap


class ParmapTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.a = np.random.randint(low=10, high=100, size=10000)
        cls.b = np.random.randint(low=2, high=10, size=10000)
        cls.benchmark = list(itertools.imap(pow, cls.a, cls.b))

    def test_parmap(self):
        result = parmap.parmap(
            pow, self.a, self.b, nworkers=4, show_progress=True,
            label='TEST parmap')

        np.testing.assert_equal(result, self.benchmark)

    def test_parstarmap(self):
        result = parmap.parstarmap(
            pow, zip(self.a, self.b), nworkers=4, show_progress=True,
            label='TEST parstarmap')

        np.testing.assert_equal(result, self.benchmark)


if __name__ == '__main__':
    unittest.main()
