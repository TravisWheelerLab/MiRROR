from time import time
from tqdm import tqdm

from mirror.spectra.types import *
from mirror.io import *

import unittest

class TestSpectra(unittest.TestCase):

    @staticmethod
    def read_test_mzlib():
        return read_mzlib("data/spectra/Apis-mellifera.mzlib.txt")
    
    @staticmethod
    def read_test_mgf():
        return read_mgf("data/spectra/Apis-mellifera.mgf")
    
    def _test_BenchmarkPeakList(self, name: str, dataset_constructor, benchmark_constructor, sampling_interval: int = 100):
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
    
    def test_BenchmarkPeakList(self):
        self._test_BenchmarkPeakList(
            name = "mzlib",
            dataset_constructor = self.read_test_mzlib,
            benchmark_constructor = BenchmarkPeakList.from_mzlib,
            sampling_interval = 1
        )

        self._test_BenchmarkPeakList(
            name = "mgf",
            dataset_constructor = self.read_test_mgf,
            benchmark_constructor = BenchmarkPeakList.from_mgf,
            sampling_interval = 1
        )
        
        # print("mzlib start\n-reading spectra")
        # time_start = time()
        # mzlib = self.read_test_mzlib()
        # print('-creating BenchmarkPeakList objects (every 100th spectrum)')
        # for spectrum_index in tqdm(range(0, len(mzlib), sampling_interval)):
        #     mzlib_spectrum_benchmark = BenchmarkPeakList.from_mzlib(
        #         dataset = mzlib,
        #         i = spectrum_index)
        # elapsed = time() - time_start
        # print(f"mzlib done\n-elapsed: {elapsed}")
        # 
        # print("mgf start\n-reading spectra")
        # time_start = time()
        # mgf = self.read_test_mgf()
        # print('-creating BenchmarkPeakList objects')
        # for spectrum_index in tqdm(0, range(len(mgf), 100)):
        #     mzlib_spectrum_benchmark = BenchmarkPeakList.from_mzlib(
        #         dataset = mgf,
        #         i = spectrum_index)
        # elapsed = time() - time_start
        # print(f"mgf done\n-elapsed: {elapsed}")