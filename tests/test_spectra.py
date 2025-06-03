from time import time
from tqdm import tqdm

from mirror.residues.types import *
from mirror.io import *
from mirror.spectra.types import *
from mirror.spectra.preprocessing import *

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