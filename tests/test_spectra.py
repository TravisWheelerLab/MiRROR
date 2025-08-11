from time import time
from tqdm import tqdm
from itertools import pairwise
from random import shuffle

from mirror.io import *
from mirror.spectra.types import *
from mirror.presets import VALIDATION_PEPTIDES

import numpy as np

import unittest

class TestSpectra(unittest.TestCase):

    @staticmethod
    def read_test_9species_mzlib():
        return read_mzlib("data/spectra/Apis-mellifera.mzlib.txt")
    
    @staticmethod
    def read_test_9species_mgf():
        return read_mgf("data/spectra/Apis-mellifera.mgf")
    
    def test_benchmark_from_simulation(self):
        for peptide in VALIDATION_PEPTIDES:
            for mode, charges in (("simple", 1), ("simple", 3), ("complex", 3)):
                sim_bpl = BenchmarkPeakList.from_simulation(peptide, mode, charges)
                if charges > 1:
                    self.assertTrue(any(charge > 1 for charge in sim_bpl.charge))
                if mode == "simple":
                    self.assertTrue(all(loss == '' for loss in sim_bpl.losses))
                elif mode == "complex":
                    self.assertTrue(any(loss != '' for loss in sim_bpl.losses))
                
