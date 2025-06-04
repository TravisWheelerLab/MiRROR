from time import time
from tqdm import tqdm

from mirror.io import *
from mirror.residues.types import *
from mirror.residues.presets import *
from mirror.residues.transformations import *
from mirror.spectra.types import *
from mirror.spectra.preprocessing import *
from mirror.spectra.simulation import simulate_simple_peaks

import unittest

class TestSpectra(unittest.TestCase):

    @staticmethod
    def read_test_9species_mzlib():
        return read_mzlib("data/spectra/Apis-mellifera.mzlib.txt")
    
    @staticmethod
    def read_test_9species_mgf():
        return read_mgf("data/spectra/Apis-mellifera.mgf")
    
    def _test_BenchmarkPeakList_9species(self, name: str, dataset_constructor, benchmark_constructor, sampling_interval: int = 100):
        print(f"{name} start\n-reading spectra")
        time_start = time()
        dataset = dataset_constructor()
        print(f"-creating BenchmarkPeakList objects (sampling one of every {sampling_interval} spectra)")
        for spectrum_index in tqdm(range(0, len(dataset), sampling_interval)):
            bpl = benchmark_constructor(
                dataset = dataset,
                i = spectrum_index)
        elapsed = time() - time_start
        print(f"{name} done\n-elapsed: {elapsed}")
    
    def test_BenchmarkPeakList_9species(self):
        self._test_BenchmarkPeakList_9species(
            name = "mgf",
            dataset_constructor = self.read_test_9species_mgf,
            benchmark_constructor = BenchmarkPeakList.from_mgf,
            sampling_interval = 24)

        self._test_BenchmarkPeakList_9species(
            name = "mzlib",
            dataset_constructor = self.read_test_9species_mzlib,
            benchmark_constructor = BenchmarkPeakList.from_mzlib,
            sampling_interval = 50)
    
    def test_BenchmarkPeakList_annotation(self):
        benchmark_peak_list = BenchmarkPeakList.from_mzlib(
            dataset = self.read_test_9species_mzlib(),
            i = 0)
        print(benchmark_peak_list)

    def test_annotate_peaks(self):
        peaks = PeakList(
            mz = [0,2,5,7,10],
            intensity = [1,1,1,1,1])
        transformations = [
            MassTransformation.dummy(index_pair = (0,1)),
            MassTransformation.dummy(index_pair = (1,2)),
            MassTransformation.dummy(index_pair = (2,3)),
            MassTransformation.dummy(index_pair = (3,4)),
        ]
        annotated_peaks = annotate_peaks(peaks, transformations)
        self.assertEqual(
            annotated_peaks.metadata['consistency'],
            [False, True, True, True, False])
        print(annotated_peaks)
    
    def _benchmark(self, path_to_dataset, num_samples, sample_stride = 1, sample_offset = 0, params = DEFAULT_RESIDUE_PARAMS):
        print(f"reading dataset: {path_to_dataset}")
        dataset = read_mzlib(path_to_dataset)
        print(f"iterating {num_samples} samples from position {sample_offset} with stride {sample_stride}")
        peptide_lengths = []
        solution_times = []
        annotation_times = []
        for i in range(sample_offset, num_samples * sample_stride, sample_stride):
            # construct the benchmark peak list
            bpl = BenchmarkPeakList.from_mzlib(dataset, i)
            peptide_lengths.append(len(bpl.peptide))
            # solve the transformations
            time_start = time()
            solutions = list(solve_peak_list(bpl, params))
            time_elapsed = time() - time_start
            solution_times.append(time_elapsed)
            # annotate the spectrum
            time_start = time()
            apl = annotate_peaks(bpl, solutions)
            time_elapsed = time() - time_start
            annotation_times.append(time_elapsed)
            
            input(f"bpl {bpl}\napl {apl}")
        avg_len = sum(peptide_lengths) / num_samples
        print(f"soln time: {sum(solution_times)}\nanno time: {sum(annotation_times)}\naverage peptide length: {avg_len}")
    
    def benchmark(self):
        self._benchmark("data/spectra/Apis-mellifera.mzlib.txt", 10, 100, 5)