"""
Test the basic functionality of the package.
"""

import os
import sys

current_directory = os.path.dirname(__file__)
sibling_directory = os.path.join(current_directory, "..", "src")
sys.path.append(sibling_directory)

import unittest
from package.cusum import CUSUM
from package.ARLTheoretical import get_ARL_1_h_mu1_k
import tomli
import pandas as pd


class TestCUSUM(unittest.TestCase):
    """
    Test class for CUSUM to check functionality.
    """

    def test_cusum(self):
        obj_cusum = CUSUM()
        # obj_cusum.initialize()

        path_file_config = os.path.abspath("../config/config.toml")
        with open(os.path.abspath(path_file_config), "rb") as file_config:
            obj_cusum.config = tomli.load(file_config)

        ref_value = 0.5
        normalized_threshold = 4

        obj_cusum.df_metric = pd.read_csv(os.path.abspath("../config/spec-60-60.csv"))
        obj_cusum.data = obj_cusum.df_metric[obj_cusum.df_metric.columns[1]].to_numpy()
        obj_cusum.set_timeline(obj_cusum.data)

        # Set initial days and get in-control mean and standard deviation
        obj_cusum.set_init_stats(init_days=30)

        # Detects a change in the process
        obj_cusum.change_detection(
            normalized_ref_value=ref_value,
            normalized_threshold=normalized_threshold,
        )

        self.assertEqual(
            obj_cusum.S_lo[-1], 2.6, "Cumulative (negative) sum does not match."
        )

    def test_rpy2(self):
        arl_1 = get_ARL_1_h_mu1_k(h=4, k=0.2996, mu1=1.2)

        self.assertEqual(arl_1, 4.43, "Package rpy2 is not working properly")


if __name__ == "__main__":
    unittest.main()
