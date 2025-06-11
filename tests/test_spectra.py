from time import time
from tqdm import tqdm
from itertools import pairwise
from random import shuffle

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
    
    def test_BenchmarkPeakList_sample(self):
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
        dataset = self.read_test_9species_mzlib()
        tally = 0
        total = 0
        indices = list(range(len(dataset)))
        shuffle(indices)
        for spectrum_idx in indices:
            benchmark_peak_list = BenchmarkPeakList.from_mzlib(
                dataset = dataset,
                i = spectrum_idx)
            states = [benchmark_peak_list.get_states(i, metadata_keys=benchmark_peak_list.get_metadata_keys()) for i in range(len(benchmark_peak_list))]
            peptide = benchmark_peak_list.peptide
            num_exp_ions = len(peptide) - 1
            coverage = {
                'b': [0 for _ in range(num_exp_ions)],
                'y': [0 for _ in range(num_exp_ions)],
            }
            for state in sorted(states, key = lambda x: x[0][-2:]):
                pos, series, err, losses, charge = state[0][::-1]
                if pos == -1:
                    continue
                else:
                    print(state[0][::-1])
                    coverage[series][pos - 1] = 1
            b_continuity = [(l == r == 1) for (l, r) in pairwise(coverage['b'])]
            y_continuity = [(l == r == 1) for (l, r) in pairwise(coverage['y'])]
            fragment_recoverability = [b or y for (b, y) in zip(b_continuity, y_continuity)]
            can_be_sequenced = all(fragment_recoverability)
            tally += can_be_sequenced
            total += 1
            print(f"------------------------------------\n[{spectrum_idx}] peptide = {peptide}, expected # ions = {num_exp_ions}\nb: {coverage['b']}\ny: {coverage['y']}\ncontinuity:\nb: {b_continuity}\ny: {y_continuity}\nrecoverability: {fragment_recoverability}\ncan be sequenced: {can_be_sequenced}")
            print(f"tally: {tally} / {total} = {tally / total}")

    def test_NineSpeciesBenchmarkPeakList(self):
        dataset = self.read_test_9species_mzlib()
        for i in range(len(dataset)):
            bpl = NineSpeciesBenchmarkPeakList.from_mzlib(dataset, i)
            y_series = sorted([(bpl.get_position(i)[0], i) for i in bpl.get_y_series_peaks()])
            y_pos, y_peaks = zip(*y_series)
            b_series = sorted([(bpl.get_position(i)[0], i) for i in bpl.get_b_series_peaks()])
            b_pos, b_peaks = zip(*b_series)
            input(f"9-species benchmark peak list:\npeptide-\t{bpl.peptide}\nb peaks-\t{b_peaks}\nb pos  -\t{b_pos}\ny peaks-\t{y_peaks}\ny pos  -\t{y_pos}\n")

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
    
    def test_annotation_comparison(self):
        apl_1 = AnnotatedPeakList(
            mz = [1,2,3],
            intensity = [1,1,1],
            charge = [[1],[2,1],[2]],
            losses = [['q'],['w','e'],[]],
            metadata = {
                "x": [[0],[0,1],[0]],
                "y": [['a'],['b','a'],[]]
            }
        )
        apl_0 = AnnotatedPeakList(
            mz = [1,2,3],
            intensity = [1,1,1],
            charge = [[],[],[]],
            losses= [[],[],[]],
            metadata = {
                "x": [[],[],[]]
            }
        )
        matches, misses = apl_1.compare(apl_0)
        print(f"matched states: {matches}\nmissed states: {misses}")
    
    def _benchmark(self, path_to_dataset, num_samples, params = DEFAULT_RESIDUE_PARAMS):
        print(f"reading dataset: {path_to_dataset}")
        dataset = read_mzlib(path_to_dataset)
        indices = list(range(len(dataset)))
        shuffle(indices)
        sampled_indices = indices[:num_samples]
        print(f"iterating {num_samples} random samples:\n{sampled_indices}")
        peptide_lengths = []
        solution_times = []
        annotation_times = []
        for i in sampled_indices:
            # construct the benchmark peak list
            bpl = BenchmarkPeakList.from_mzlib(dataset, i)
            bpl_keys = bpl.get_metadata_keys()
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

            matches, misses = bpl.compare(apl)
            n_matches = sum(len(x) for x in matches)
            n_misses = sum(len(x) for x in misses)
            for peak_idx in range(len(bpl)):
                true_state = bpl.get_states(peak_idx, metadata_keys = bpl_keys)[0]
                observable_true_state = true_state[:2]
                charge, losses, mass_error, series, position = true_state
                if series == '':
                    continue
                annotated_states = list(set(apl.get_states(peak_idx)))
                print(f"peak[{peak_idx}] {observable_true_state} {true_state[2:]}\n\tannotations: {annotated_states}\n\tmatch? {observable_true_state in annotated_states}\n")
            input(f"spectrum[{i}] | peaks: {len(bpl)}, matches: {n_matches}, misses: {n_misses}\n")
        avg_len = sum(peptide_lengths) / num_samples
        print(f"soln time: {sum(solution_times)}\nanno time: {sum(annotation_times)}\naverage peptide length: {avg_len}")
    
    def benchmark(self):
        self._benchmark("data/spectra/Apis-mellifera.mzlib.txt", 10)