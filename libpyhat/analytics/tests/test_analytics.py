import unittest

import numpy as np

from libpyhat.analytics import analytics


class Test_Analytics(unittest.TestCase):
    np.random.seed(12345)

    def setUp(self):
        self.series = np.random.random(25)

    def test_band_minima(self):
        minidx, minvalue = analytics.band_minima(self.series)
        self.assertEqual(minidx, 19)
        self.assertAlmostEqual(minvalue, 0.107322912)

        minidx, minvalue = analytics.band_minima(self.series, 0, 7)
        self.assertEqual(minidx, 0)
        self.assertAlmostEqual(minvalue, 0.225637606)

        with self.assertRaises(ValueError):
            minidx, minvalue = analytics.band_minima(self.series, 6, 1)

    def test_band_center(self):
        center, center_fit = analytics.band_center(self.series)
        self.assertEqual(center[0], 6)
        self.assertAlmostEqual(center[1], 0.51505549)
        self.assertAlmostEqual(center_fit[0], 0.553834612)
        self.assertAlmostEqual(center_fit[12], 0.542132017)
        self.assertAlmostEqual(center_fit[23], 0.652578626)
        self.assertAlmostEqual(center_fit.mean(), 0.561734550)
        self.assertAlmostEqual(np.median(center_fit), 0.54642353)
        self.assertAlmostEqual(center_fit.std(), 0.043854163)

    def test_band_area(self):
        x = np.arange(-2, 2, 0.1)
        y = x ** 2
        parabola = y
        area = analytics.band_area(parabola)
        self.assertEqual(area, [370.5])

    def test_band_asymmetry(self):
        assymetry = analytics.band_asymmetry(self.series)
        self.assertAlmostEqual(assymetry, 0.99447513)

        assymetry = analytics.band_asymmetry(np.ones(24))
        self.assertEqual(assymetry, 1.0)
