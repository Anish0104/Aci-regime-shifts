"""Run offline checks with: python -m unittest data.test_data."""
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np
import pandas as pd
from .shifts import inject_shift
from . import uci, entsoe


class DataTests(unittest.TestCase):
    def test_mean_and_ramp(self):
        x = np.arange(100., dtype=float)
        for speed in ("abrupt", "gradual"):
            y, s = inject_shift(x, "mean", speed, 2, ramp=4)
            self.assertEqual(s, 85)
            np.testing.assert_array_equal(y[:s], x[:s])
            self.assertAlmostEqual(y[s+4]-x[s+4], 2*x[:60].std())
            self.assertAlmostEqual(y[s]-x[s], 2*x[:60].std() if speed == "abrupt" else 0)
        np.testing.assert_array_equal(x, np.arange(100.))

    def test_variance_and_seed(self):
        x = np.random.default_rng(1).normal(size=200000)
        y, s = inject_shift(x, "variance", "abrupt", 3)
        self.assertAlmostEqual(y[s:].std()/x[:120000].std(), 3, delta=.08)
        np.testing.assert_array_equal(y[:s], x[:s])
        np.testing.assert_array_equal(y, inject_shift(x, "variance", "abrupt", 3)[0])
        np.testing.assert_array_equal(x, inject_shift(x, "variance", "gradual", 1)[0])

    def test_invalid_inputs(self):
        for x in ([], [[1, 2]], [1, np.nan]):
            with self.assertRaises(ValueError):
                inject_shift(x, "mean", "abrupt", 1)
        with self.assertRaises(AssertionError):
            inject_shift(np.ones(100), "mean", "abrupt", 1, shift_at=79)
        with self.assertRaises(ValueError):
            inject_shift(np.ones(100), "variance", "abrupt", .5)

    def test_uci_decimal_sum_and_resample(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            hours = pd.date_range("2011-12-31 23:00", periods=12, freq="15min")
            pd.DataFrame({"A": np.full(12, 1.5), "B": np.full(12, 2.5)}, index=hours).to_csv(
                path / "LD2011_2014.txt", sep=";", decimal=",")
            with patch.object(uci, "RAW_DIR", path):
                np.testing.assert_array_equal(uci.load_uci(), [4., 4.])

    def test_entsoe_gaps_and_shift(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "load.csv"
            hours = pd.date_range("2019-01-01", "2020-05-31 23:00", freq="h", tz="UTC")
            frame = pd.DataFrame({"utc_timestamp": hours, "IT_load_actual_entsoe_transparency": np.arange(len(hours), dtype=float)})
            frame.loc[100, "IT_load_actual_entsoe_transparency"] = np.nan
            frame = frame.drop(index=200)
            frame.to_csv(path, index=False)
            with patch.object(entsoe, "download", return_value=path):
                with self.assertLogs(entsoe.logger, level="INFO") as logs:
                    x, s = entsoe.load_entsoe()
                self.assertIn("2 missing hours", logs.output[0])
                np.testing.assert_array_equal(x, np.arange(len(hours)))
                self.assertEqual(hours[s], pd.Timestamp("2020-03-09", tz="UTC"))
                self.assertGreaterEqual(s, int(.8*len(x)))


if __name__ == "__main__":
    unittest.main()
