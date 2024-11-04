import unittest
from time import time

import numpy as np

from mirror.util import log
from mirror.io import load_spectrum_from_mzML, oms
from mirror.preprocessing import *

TEST_DATA = "./tests/data/BY_04_1.mzML"

N_PROCESSES = 12

class Test002_Preprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print(f"\n{cls.__name__}")
    
    def test(self, path_to_data = TEST_DATA, max_mz = 3000, resolution = 0.02, max_err = 15):
        print("\ntest3_comp")
        log("loading spectra")
        load_t = time()
        exp = load_spectrum_from_mzML(path_to_data)
        load_t = time() - load_t
        print(load_t, "s")

        log("binning spectra")
        bin_t = time()
        bins = create_bins(exp, max_mz, resolution, parallelize = False)
        bin_t = time() - bin_t
        print(bin_t, "s")

        log("binning spectra (parallel)")
        bin_t_p = time()
        bins_p = create_bins(exp, max_mz, resolution, parallelize = True, n_processes = N_PROCESSES)
        bin_t_p = time() - bin_t_p
        print(bin_t_p, "s")

        tot_err = 0
        tot_pct_err = 0
        num_errs = 0
        for i in range(len(bins)):
            err_i = abs(bins[i] - bins_p[i])
            if err_i > max_err:
                print(i, bins[i], bins_p[i])
            if err_i > 0:
                num_errs += 1
                tot_err += err_i
                tot_pct_err += err_i / bins_p[i]
        self.assertTrue(np.all(np.abs(bins - bins_p) <= max_err))
        print(f"total err:\t{tot_err}")
        print(f"avg err:\t{round(tot_err / num_errs, 3)}")
        print(f"avg pct err:\t{100 * round(tot_pct_err / num_errs, 3)}%")