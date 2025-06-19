from time import time
from tqdm import tqdm

from mirror.spectra.simulation import simulate_simple_peaks, simulate_complex_peaks
from mirror.spectra.types import PeakList, BenchmarkPeakList
from mirror.residues.types import *
from mirror.residues.presets import *
from mirror.residues.transformations import *
from mirror.io import read_mzlib

import unittest

class TestResidues(unittest.TestCase):

    def test_default_params(self):
        default_params = DEFAULT_RESIDUE_PARAMS
        print(default_params)

    def _test_transformation_solver(self, peptide, params = DEFAULT_RESIDUE_PARAMS, simulator = simulate_simple_peaks, verbose = False):
        # make peaks
        mz = simulator(peptide)
        intensity = [1. for _ in mz]
        peaks = PeakList(mz, intensity)
        # run bisect strategy
        params = DEFAULT_RESIDUE_PARAMS
        bisect_solutions = sorted(list(solve_peak_list(peaks, params)), key = lambda x: x.mass_error)
        # run tensor strategy
        params.strategy = TensorMassTransformationSolver
        tensor_solutions = sorted(list(solve_peak_list(peaks, params)), key = lambda x: x.mass_error)
        # run checks
        self.assertEqual(bisect_solutions, tensor_solutions)
        expected_residues = set(peptide[1:-1])
        observed_residues = set(map(lambda mt: mt.residue_symbol, bisect_solutions))
        self.assertEqual(expected_residues, observed_residues.intersection(expected_residues))
        if verbose:
            print("Solutions:")
            for mt in bisect_solutions:
                print(f"MassTransformation: res {mt.residue_symbol} charge {mt.charges_symbol} losses {mt.losses_symbol} modification {mt.modification_symbol} error {mt.mass_error}")
    
    def test_transformation_solver(self):
        samples = ["NREQSTK",
            "AEEHANR","GNAGGLHHHR","HHVLHHQTVDK","HHVLHHQTVDK",
            "HHSTIPQK","FTHQHKPDER","FTHQHKPDER","CEACPKPGTHAHK",
            "HHTIAHYK","KPGVHQPQR","AAHLAAHEAAK","GHSCYRPR",
            "HHNIIR","HLAEHEVK","HGLTNTASHTR","INPDNHNEK",
            "HGATVVNHVK","HLNGHGSPPATNSSHR","HASNIHVEK","ELHVHPK"]
        for peptide in samples:
            self._test_transformation_solver(peptide, verbose = True)
            input("end simple results")
            self._test_transformation_solver(peptide, simulator = simulate_complex_peaks, verbose = True)
            input("end complex results")

    def test_mass_query(self):
        # given a k-mer mass, find the solution sequence(s).
        pass

    def _benchmark(self, path_to_dataset, num_samples, sample_stride = 1, sample_offset = 0, params = DEFAULT_RESIDUE_PARAMS):
        print(f"reading dataset: {path_to_dataset}")
        dataset = read_mzlib(path_to_dataset)
        print(f"iterating {num_samples} samples from position {sample_offset} with stride {sample_stride}")
        peptide_lengths = []
        solution_times = []
        for i in range(sample_offset, num_samples * sample_stride, sample_stride):
            # construct the benchmark peak list
            bpl = BenchmarkPeakList.from_mzlib(dataset, i)
            peptide_lengths.append(len(bpl.peptide))
            # solve the transformations
            time_start = time()
            solutions = list(solve_peak_list(bpl, params))
            time_elapsed = time() - time_start
            solution_times.append(time_elapsed)
            #print(f"spectrum[{i}]\n- peptide: {bpl.get_peptide()}\n- time: {time_elapsed}")
        avg_time = sum(solution_times) / num_samples
        avg_len = sum(peptide_lengths) / num_samples
        print(f"average time: {avg_time}\naverage peptide length: {avg_len}")

    def benchmark_bisect(self):
        self._benchmark("data/spectra/Apis-mellifera.mzlib.txt", 100, 100, 5)

    def benchmark_tensor(self):
        tensor_params = DEFAULT_RESIDUE_PARAMS
        tensor_params.strategy = TensorMassTransformationSolver
        self._benchmark("data/spectra/Apis-mellifera.mzlib.txt", 100, 100, 5, params = tensor_params)
