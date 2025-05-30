from time import time
from tqdm import tqdm

from mirror.residues.types import *
from mirror.io import *
from mirror.spectra.types import *
from mirror.spectra.preprocessing import *

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
            name = "mgf",
            dataset_constructor = self.read_test_mgf,
            benchmark_constructor = BenchmarkPeakList.from_mgf,
            sampling_interval = 24)

        self._test_BenchmarkPeakList(
            name = "mzlib",
            dataset_constructor = self.read_test_mzlib,
            benchmark_constructor = BenchmarkPeakList.from_mzlib,
            sampling_interval = 50)

    def test_annotate_peaks(self):
        peaks = PeakList(
            mz = [0,2,5,7,10],
            intensity = [1,1,1,1,1])
        transformations = [
            MassTransformation(
                residue = 0,
                modification = 0,
                inner_index = (0, 1),
                peaks = (0, 1),
                losses = (0, 0),
                charges = (0, 0),
            ),
            MassTransformation(
                residue = 0,
                modification = 0,
                inner_index = (1, 2),
                peaks = (1, 2),
                losses = (0, 0),
                charges = (0, 0),
            ),
            MassTransformation(
                residue = 0,
                modification = 0,
                inner_index = (2, 3),
                peaks = (2, 3),
                losses = (0, 0),
                charges = (0, 0),
            ),
            MassTransformation(
                residue = 0,
                modification = 0,
                inner_index = (3, 4),
                peaks = (3, 4),
                losses = (0, 0),
                charges = (0, 0),
            )]
        annotated_peaks = annotate_peaks(peaks, transformations)
        self.assertEqual(
            annotated_peaks.metadata['consistency'],
            [False, True, True, True, False])
        print(f"AnnotatedPeakList:\n{annotated_peaks.mz}\n{annotated_peaks.intensity}\n{annotated_peaks.charge}\n{annotated_peaks.losses}\n{annotated_peaks.metadata['consistency']}")