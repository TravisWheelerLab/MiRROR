from time import time
from tqdm import tqdm

from mirror.spectra.simulation import simulate_simple_peaks
from mirror.spectra.types import PeakList
from mirror.residues.types import *
from mirror.residues.presets import *
from mirror.residues.transformations import *

import unittest

class TestResidues(unittest.TestCase):

    def test_default_params(self):
        default_params = DEFAULT_RESIDUE_PARAMS
        print(default_params)

    def test_transformation_solver(self):
        # initialize a transformation solver with the default params
        ## make peaks
        peptide = "AEEHANR"
        mz = simulate_simple_peaks(peptide)
        intensity = [1. for _ in mz]
        peaks = PeakList(mz, intensity)
        # run bisect strategy
        params = DEFAULT_RESIDUE_PARAMS
        bisect_solutions = list(solve_peak_list(peaks, params))
        # run tensor strategy
        params.strategy = TensorMassTransformationSolver
        tensor_solutions = list(solve_peak_list(peaks, params))
        self.assertEqual(bisect_solutions, tensor_solutions)
        print([(params.residue_symbols[mt.residue], mt.peaks) for mt in tensor_solutions])
        print([(params.residue_symbols[mt.residue], mt.peaks) for mt in bisect_solutions])
    
    def test_mass_query(self):
        # given a k-mer mass, find the solution sequence(s).
        pass
