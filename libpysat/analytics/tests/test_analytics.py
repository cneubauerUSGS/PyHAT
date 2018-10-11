import unittest

import numpy as np
import pandas as pd

from libpysat.analytics import analytics


class Test_Analytics(unittest.TestCase):
    np.random.seed(12345)

    def setUp(self):
        self.series = pd.Series(np.random.random(25))

    def test_band_minima(self):
        print(self.series)
        minidx, minvalue = analytics.band_minima(self.series)
        self.assertEqual(minidx, 24)
        self.assertAlmostEqual(minvalue, 0.04271530433633308)

        minidx, minvalue = analytics.band_minima(self.series, 0, 10)
        self.assertEqual(minidx, 0)
        self.assertAlmostEqual(minvalue, 0.22563760652063947)

        with self.assertRaises(ValueError):
            minidx, minvalue = analytics.band_minima(self.series, 6, 1)

    def test_band_area(self):
        x = np.arange(-2, 2, 0.1)
        y = x ** 2
        parabola = pd.Series(y[y <= 1], index=x[y <= 1])
        area = analytics.band_area(parabola)
        self.assertAlmostEqual(area, -5.7950)

    def test_band_center(self):
        pass

    def test_band_asymmetry(self):
        pass

    def test_get_noise(self):
        pass

    def test_sigma_clip(self):
        pass

    def test_meancenter(self):
        pass
