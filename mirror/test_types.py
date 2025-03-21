from dataclasses import dataclass
import pickle
from types import NoneType
from time import time
from pathlib import Path

import networkx as nx
import numpy as np
import numpy.typing as npt
from scipy.stats import zscore
from tabulate import tabulate

from . import *

@dataclass
class TestSpectrum:
    # true/target data:
    residue_sequence: np.ndarray
    mass_sequence: np.ndarray
    losses: np.ndarray
    charges: np.ndarray
    modifications: np.ndarray
    noise: np.ndarray
    mz: np.ndarray
    target_gaps: dict[str, GapResult]
    target_pivot: Pivot
    
    # params:
    gap_search_parameters: GapSearchParameters
    intergap_tolerance: float
    symmetry_factor: float
    terminal_residues: list[str]
    boundary_padding: int
    gap_key: str
    occurrence_threshold: int
    suffix_array: SuffixArray

    # test params:
    autorun: bool = True
    
    # output:
    _ran: bool = False
    # a (potentially empty) list of candidates for each viable pair of affixes.
    _candidates: list[list[list[list[Candidate]]]] = None
    # a list of index pairs into each array of affixes
    _affix_pairs: list[list[npt.NDArray[int]]] = None
    # an array of affixes for each spectrum graph pair.
    _affixes: list[list[npt.NDArray[Affix]]] = None
    # a spectrum graph pair for each boundary.
    _spectrum_graphs: list[list[tuple[nx.DiGraph,nx.DiGraph]]] = None
    # an augmented spectrum, pivot, gap indices, and boundary peak indices for each boundary.
    _augmented_data: list[list[tuple[np.ndarray,Pivot,list[tuple[int,int]],list[int]]]] = None
    # a list of boundaries for each pivot.
    _boundaries: list[list[Boundary]] = None
    # list of pivots.
    _pivots: list[Pivot] = None
    # potential terminal y ions
    _y_terminii: list[tuple[int, float]] = None
    # peaks annotated by the set of GapResult objects
    _annotated_peaks: np.ndarray = None
    # a gap result for each amino acid, plus an unrecognized category.
    _gaps: list[GapResult] = None

    # stitches the output stack together by indexing into pivots, boundaries (and augmented data and graph pairs), affixes, and candidates.
    # there is an OutputTrace for every Candidate.
    _indices: list[OutputIndex] = None

    _time: dict[str, float] = None
    _crash: list = None
    
    #   TODO: comparison for numpy types. lift from the serialization unit test.
    #def __eq__(self, other):
    #    if type(other) == TestSpectrum:
    #        return (
    #            self.get_target_data() == other.get_target_data() and
    #            self.get_params() == other.get_params() and
    #            self.get_state() == other.get)
    #    else:
    #        return False
    
    def __len__(self):
        return len(self._indices)
    
    def __getitem__(self, i: int) -> OutputIndex:
        return self._indices[i]
    
    def __iter__(self) -> list[OutputIndex]:
        return iter(self._indices)

    def _check_state(self, *args):
        values = [self.__dict__[k] for k in args]
        valtypes = [type(v) for v in values]
        if NoneType in valtypes:
            invalid_states = [(k, v, t) for (k, v, t) in zip(args, values, valtypes) if t == type(None)]
            state_str = '\n\t'.join(f"{k} = {v} {t}" for (k, v, t) in invalid_states)
            raise ValueError(f"Invalid TestSpectrum State:{invalid_states}")

    def _record_complexity(self, tag, fn, *args, **kwargs):
        t = time()
        self._size[tag] = fn(*args, **kwargs)
        self._time[tag] = time() - t

    def _optimize(self):
        if self.n_indices > 0:
            self._edit_distances = np.array([self.get_candidate(index).edit_distance(self.residue_sequence)[0] for index in self])
            self._optimizer = self._edit_distances.argmin()
        else:
            self._edit_distances = np.inf
            self._optimizer = -1

    def get_peptide(self):
        return ''.join(self.residue_sequence)
    
    def get_gaps(self):
        return self._gaps

    def get_annotated_peaks(self):
        return self._annotated_peaks
    
    def get_y_terminii(self):
        return self._y_terminii

    def get_pivot(self, p: int):
        return self._pivots[p]
    
    def get_boundary(self, p: int, b: int):
        return self._boundaries[p][b]

    def get_augmented_data(self, p: int, b: int):
        return self._augmented_data[p][b]
    
    def get_spectrum_graph_pair(self, p: int, b: int):
        return self._spectrum_graphs[p][b]

    def get_affixes(self, p: int, b: int):
        return self._affixes[p][b]

    def get_affix_pair(self, p: int, b: int, a: int):
        return self._affix_pairs[p][b][a]

    def get_candidate(self, index: OutputIndex):
        p, b, a, c = index()
        return self._candidates[p][b][a][c]
    
    def get_output_stack(self, index: OutputIndex
    ) -> tuple[
        Pivot, 
        Boundary, 
        tuple[np.ndarray,Pivot,list[tuple[int,int]],list[int]], 
        GraphPair, 
        Affix, 
        tuple[int, int], 
        Candidate
    ]:
        p, b, a, c = index()
        return (
            self.get_pivot(p),
            self.get_boundary(p, b),
            self.get_augmented_data(p, b),
            self.get_spectrum_graph_pair(p, b),
            self.get_affixes(p, b),
            self.get_affix_pair(p, b, a),
            self.get_candidate(index),
        )

    def get_target_data(self):
        """Lists of the groundtruth and target data for the TestSpectrum:
        
            [residue_sequence, mass_sequence, losses, charges, modifications, noise, mz, target_gaps, target_pivot]"""
        return [
            self.residue_sequence, 
            self.mass_sequence, 
            self.losses, 
            self.charges, 
            self.modifications, 
            self.noise, 
            self.mz, 
            self.target_gaps, 
            self.target_pivot]
    
    def get_params(self):
        """Lists the parameters passed to the TestSpectrum:

            [gap_search_parameters, intergap_tolerance, symmetry_factor, terminal_residues, boundary_padding, gap_key, suffix_array_file, occurrence_threshold]""" 
        return [
            self.gap_search_parameters, 
            self.intergap_tolerance, 
            self.symmetry_factor, 
            self.terminal_residues, 
            self.boundary_padding, 
            self.gap_key, 
            self.suffix_array_file, 
            self.occurrence_threshold]
        
    def get_times_as_vec(self):
        tvals = list(self._time.values())
        return np.array(tvals + [-np.inf for _ in range(len(self.step_names()) - len(tvals))])
    
    def get_sizes_as_vec(self):
        svals = list(self._size.values())
        return np.array(svals + [-np.inf for _ in range(len(self.step_names()) - len(svals))])
    
    def get_time(self, tag: str):
        return self._time[tag]
    
    def get_size(self, tag: str):
        return self._size[tag]

    def get_optimizer(self):
        return self._optimizer

    def get_optimum(self):
        if self._optimizer >= 0:
            return self._edit_distances[self._optimizer]
        elif self._ran:
            return self._edit_distances
        else:
            return np.inf
    
    def run_gaps(self):
        self._check_state("gap_search_parameters", "mz")
        self._annotated_peaks, self._gaps = find_gaps(
            self.gap_search_parameters,
            self.mz
        )
        return len(self._gaps)
    
    def run_y_terminii(self):
        self._check_state("_annotated_peaks")
        self._y_terminii = list(util.find_terminal_y_ion(self._annotated_peaks, len(self._annotated_peaks)))
        self.n_y_terminii = len(self._y_terminii)
        return self.n_y_terminii

    def run_pivots(self):
        self._check_state("_gaps", "_annotated_peaks", "intergap_tolerance", "symmetry_factor")
        self._pivots = find_all_pivots(
            self._annotated_peaks, 
            self.symmetry_factor * util.expected_num_mirror_symmetries(self._annotated_peaks), 
            self._gaps, 
            self.intergap_tolerance
        )
        self.n_pivots = len(self._pivots)
        return self.n_pivots

    def run_boundaries(self):
        self._check_state("_annotated_peaks", "_pivots", "_y_terminii")
        self._boundaries = [None for _ in range(len(self._pivots))]
        self.n_boundaries = [-1 for _ in range(len(self._pivots))]
        for p_idx, pivot in enumerate(self._pivots):
            boundaries, _ = find_and_create_boundaries(
                self._annotated_peaks, 
                self._y_terminii,
                pivot,
                self.gap_search_parameters,
                self.terminal_residues,
                self.boundary_padding
            )
            self._boundaries[p_idx] = boundaries
            self.n_boundaries[p_idx] = len(boundaries)
        return sum(self.n_boundaries)

    def run_augment(self):
        self._check_state("_pivots", "_boundaries")
        self._augmented_data = [[None 
            for _ in range(len(self._boundaries[p_idx]))] 
            for p_idx in range(len(self._pivots))]
        for p_idx in range(self.n_pivots):
            for b_idx, boundary in enumerate(self._boundaries[p_idx]):
                augmented_peaks = boundary.get_augmented_peaks()
                augmented_pivot = boundary.get_augmented_pivot()
                augmented_gaps = boundary.get_augmented_gaps()
                boundary_indices = boundary.get_boundary_indices()
                self._augmented_data[p_idx][b_idx] = (
                    augmented_peaks, 
                    augmented_pivot, 
                    augmented_gaps, 
                    boundary_indices
                )
        return sum(len(peaks) for peaks, _, __, ___ in self._augmented_data[p_idx] for p_idx in range(self.n_pivots))
    
    def run_spectrum_graphs(self):
        self._check_state("_pivots", "_boundaries", "_augmented_data", "gap_key")
        self._spectrum_graphs = [[None for _ in range(len(self._boundaries[p_idx]))] for p_idx in range(self.n_pivots)]
        for p_idx in range(self.n_pivots):
            for b_idx in range(self.n_boundaries[p_idx]):
                augmented_peaks, augmented_pivot, augmented_gaps, boundary_indices = self._augmented_data[p_idx][b_idx]
                self._spectrum_graphs[p_idx][b_idx] = create_spectrum_graph_pair(
                    augmented_peaks, 
                    augmented_gaps, 
                    augmented_pivot, 
                    boundary_indices, 
                    gap_key = self.gap_key
                )
        return sum(asc.size() + desc.size() for (asc, desc) in self._spectrum_graphs[p_idx] for p_idx in range(self.n_pivots))
    
    def run_affixes(self):
        self._check_state("_pivots", "_boundaries", "_spectrum_graphs", "gap_key")
        self._affixes = [[None  for _ in range(self.n_boundaries[p_idx])] for p_idx in range(self.n_pivots)]
        self.n_affixes = [[-1 for _ in range(self.n_boundaries[p_idx])] for p_idx in range(self.n_pivots)]
        for p_idx in range(self.n_pivots):
            for b_idx in range(self.n_boundaries[p_idx]):
                graph_pair = self._spectrum_graphs[p_idx][b_idx]
                gap_comparator = lambda x, y: (abs(x - y) < self.intergap_tolerance) and (x != -1) and (y != -1)
                dual_paths = find_dual_paths(
                    *graph_pair,
                    self.gap_key,
                    gap_comparator)
                affixes = np.array([create_affix(dp, graph_pair) for dp in dual_paths])
                self._affixes[p_idx][b_idx] = affixes
                self.n_affixes[p_idx][b_idx] = len(affixes)
        return sum(sum(self.n_affixes[i]) for i in range(self.n_pivots))

    """def run_affixes_filter(self):
        if self.suffix_array != None:
            self._check_state("_pivots", "_boundaries", "_affixes")
            self.n_unfiltered_affixes = [[-1 for _ in range(self.n_boundaries[p_idx])] for p_idx in range(self.n_pivots)]
            for p_idx in range(self.n_pivots):
                for b_idx in range(self.n_boundaries[p_idx]):
                    self.n_unfiltered_affixes[p_idx][b_idx] = self.n_affixes[p_idx][b_idx]
                    self._affixes[p_idx][b_idx] = filter_affixes(
                        self._affixes[p_idx][b_idx], 
                        self.suffix_array, 
                        self.occurrence_threshold)
                    self.n_affixes[p_idx][b_idx] = len(self._affixes[p_idx][b_idx])
        return sum(sum(self.n_affixes[i]) for i in range(self.n_pivots))"""
    
    def run_affixes_filter(self):
        if self.suffix_array != None:
            self._check_state("_pivots", "_boundaries", "_affixes")
            self.n_unfiltered_affixes = [[-1 for _ in range(self.n_boundaries[p_idx])] for p_idx in range(self.n_pivots)]

            flat_affixes, index = util.recursive_collapse(self._affixes)
            mask = mask_nonoccurring_affixes(flat_affixes, self.suffix_array, self.occurrence_threshold)
            structured_mask = util.recursive_uncollapse(mask, index)

            for p_idx in range(self.n_pivots):
                for b_idx in range(self.n_boundaries[p_idx]):
                    self.n_unfiltered_affixes[p_idx][b_idx] = self.n_affixes[p_idx][b_idx]
                    local_mask = structured_mask[p_idx][b_idx]
                    self._affixes[p_idx][b_idx] = self._affixes[p_idx][b_idx][local_mask]
                    self.n_affixes[p_idx][b_idx] = len(self._affixes[p_idx][b_idx])
        return sum(sum(self.n_affixes[i]) for i in range(self.n_pivots))

    def run_affixes_pair(self):
        self._check_state("_pivots", "_boundaries", "_affixes")
        self._affix_pairs = [[None  for _ in range(self.n_boundaries[p_idx])] for p_idx in range(self.n_pivots)]
        self.n_affix_pairs = [[-1 for _ in range(self.n_boundaries[p_idx])] for p_idx in range(self.n_pivots)]
        for p_idx in range(self.n_pivots):
            for b_idx in range(self.n_boundaries[p_idx]):
                affixes = self._affixes[p_idx][b_idx]
                affix_pairs = find_affix_pairs(affixes)
                self._affix_pairs[p_idx][b_idx] = affix_pairs
                self.n_affix_pairs[p_idx][b_idx] = len(affix_pairs)
        return sum(sum(self.n_affix_pairs[i]) for i in range(self.n_pivots))

    def run_candidates(self):
        self._check_state("_pivots", "_boundaries", "_affixes", "_affix_pairs")
        self._candidates = [[[None for _ in range(self.n_affix_pairs[p_idx][b_idx])] for b_idx in range(self.n_boundaries[p_idx])] for p_idx in range(self.n_pivots)]
        self.n_candidates = [[[-1 for _ in range(self.n_affix_pairs[p_idx][b_idx])] for b_idx in range(self.n_boundaries[p_idx])] for p_idx in range(self.n_pivots)]
        for p_idx, pivot in enumerate(self._pivots):
            pivot_residue = pivot.residue()
            for b_idx, boundary in enumerate(self._boundaries[p_idx]):
                boundary_residues = boundary.get_residues()
                augmented_peaks = self._augmented_data[p_idx][b_idx][0]
                graph_pair = self._spectrum_graphs[p_idx][b_idx]
                affixes = self._affixes[p_idx][b_idx]
                for a_idx, affix_pair in enumerate(self._affix_pairs[p_idx][b_idx]):
                    self._candidates[p_idx][b_idx][a_idx] = create_candidates(
                        augmented_peaks,
                        graph_pair,
                        affixes[affix_pair],
                        boundary_residues,
                        pivot_residue
                    )
                    self.n_candidates[p_idx][b_idx][a_idx] = len(self._candidates[p_idx][b_idx][a_idx])
        return sum(sum(self.n_candidates[p][b]) for p in range(self.n_pivots) for b in range(self.n_boundaries[p]))
   
    def run_indices(self):
        self._check_state("_pivots", "_boundaries", "_affix_pairs", "_candidates")
        self._indices = []
        for p_idx in range(self.n_pivots):
            for b_idx in range(self.n_boundaries[p_idx]):
                for a_idx in range(self.n_affix_pairs[p_idx][b_idx]):
                    for c_idx in range(self.n_candidates[p_idx][b_idx][a_idx]):
                        self._indices.append(OutputIndex(
                            pivot_index = p_idx, 
                            boundary_index = b_idx, 
                            affixes_index = a_idx, 
                            candidate_index = c_idx
                        ))
        self.n_indices = len(self._indices)
        return self.n_indices
    
    @classmethod
    def run_sequence(cls):
        RUN_SEQUENCE = [
            ("gaps", cls.run_gaps),
            ("y-term", cls.run_y_terminii),
            ("pivots", cls.run_pivots),
            ("boundaries", cls.run_boundaries),
            ("augment", cls.run_augment),
            ("topology", cls.run_spectrum_graphs),
            ("affix", cls.run_affixes),
            ("afx-filter", cls.run_affixes_filter),
            ("afx-pair", cls.run_affixes_pair),
            ("candidates", cls.run_candidates),
            ("index", cls.run_indices),
        ]
        return RUN_SEQUENCE

    @classmethod
    def step_names(cls):
        names, _ = zip(*cls.run_sequence())
        return names

    def run(self):
        """Generate candidates, associated data structures, and output traces."""
        self._size = dict()
        self._time = dict()
        for (tag, fn) in self.run_sequence():
            try:
                self._record_complexity(tag, fn, self)
            except Exception as e:
                print(f"[Warning] step {tag} crashed:\n{e}")
                self.n_indices = -1
                self._crash = [tag]
                break
        self._ran = True

    def set_suffix_array(self, suffix_array: SuffixArray):
        self.suffix_array = suffix_array
    
    def delete_suffix_array(self):
        self.suffix_array = None

    @classmethod
    def _read(cls, handle):
        return pickle.load(handle)
    
    @classmethod
    def read(cls, ts_file, suffix_array_file = None):
        with open(ts_file, 'rb') as handle:
            ts = cls._read(handle)
            if suffix_array_file != None:
                ts.set_suffix_array(SuffixArray.read(suffix_array_file))

    def _write(self, handle):
        pickle.dump(self, handle)

    def write(self, filepath):
        self.delete_suffix_array()
        self.autorun = False
        with open(filepath, 'wb') as handle:
            self._write(handle)

    def __post_init__(self):
        # setup
        self._edit_distances = None
        self._optimizer = None
        # construct candidates
        if self.autorun:
            self.run()
        # compare to target
        self._optimize()

class TestRecord:

    def __init__(self, identifier: str):
        self._id = identifier
        self._test_spectra = []
        self._times = []
        self._sizes = []
        self._optimizers = []
        self._optima = []
        self._n = 0
        self._finalized = False
    
    def _check_finalized(self):
        if not self._finalized:
            raise ValueError("this method must be run after `finalize(`")
    
    def __len__(self):
        return self._n

    def add(self, ts: TestSpectrum):
        if not self._finalized:
            self._test_spectra.append(ts)
            self._times.append(ts.get_times_as_vec())
            self._sizes.append(ts.get_sizes_as_vec())
            #optimum, optimizer = ts.optimize()
            optimum = ts.get_optimum()
            optimizer = ts.get_optimizer()
            self._optima.append(optimum)
            self._optimizers.append(optimizer)
            self._n += 1
        else:
            raise ValueError("this method cannot be run after `finalize(`")

    def finalize(self):
        # convert to numpy arrays=
        self._test_spectra = np.array(self._test_spectra, dtype=TestSpectrum)
        self._times = np.vstack(self._times)
        self._sizes = np.vstack(self._sizes)
        self._optimizers = np.array(self._optimizers)
        self._optima = np.array(self._optima)
        # construct masks for each class of TestSpectrum 
        self._matches_mask = self._optima == 0
        self._misses_mask = self._optima > 0
        self._crashes_mask = self._optima == -1
        self._time_outliers_mask = np.absolute(zscore(self._times.sum(axis = 1))) > 2
        self._size_outliers_mask = np.absolute(zscore(self._sizes.sum(axis = 1))) > 2
        self._size_outliers_mask = self._size_outliers_mask & ~self._time_outliers_mask
        # store indices for class membership
        ind = np.arange(self._n)
        self._matches = ind[self._matches_mask]
        self._misses = ind[self._misses_mask]
        self._crashes = ind[self._crashes_mask]
        self._time_outliers = ind[self._time_outliers_mask]
        self._size_outliers = ind[self._size_outliers_mask]
        # store class sizes
        self.n_matches = len(self._matches)
        self.n_misses = len(self._misses)
        self.n_crashes = len(self._crashes)
        self.n_temporal_outliers = len(self._time_outliers)
        self.n_spatial_outliers = len(self._size_outliers)
        # activate the getter methods
        self._finalized = True

    def get_matches(self):
        self._check_finalized()
        return list(self._test_spectra[self._matches])
        
    def get_misses(self):
        self._check_finalized()
        return list(self._test_spectra[self._misses])

    def get_crashes(self):
        self._check_finalized()
        return list(self._test_spectra[self._crashes])

    def get_temporal_outliers(self):
        self._check_finalized()
        return list(self._test_spectra[self._time_outliers])

    def get_spatial_outliers(self):
        self._check_finalized()
        return list(self._test_spectra[self._size_outliers])
    
    def save_matches(self, output_dir):
        self._save_test_spectra(output_dir, "matches", self.get_matches())
    
    def save_misses(self, output_dir):
        self._save_test_spectra(output_dir, "misses", self.get_misses())

    def save_crashes(self, output_dir):
        self._save_test_spectra(output_dir, "crashes", self.get_crashes())

    def save_temporal_outliers(self, output_dir):
        self._save_test_spectra(output_dir, "temporal-outliers", self.get_temporal_outliers())

    def save_spatial_outliers(self, output_dir):
        self._save_test_spectra(output_dir, "spatial-outliers", self.get_spatial_outliers())

    def _save_test_spectra(self, output_dir: Path, tag: str, test_spectra: list[TestSpectrum]):
        test_spectra.sort(key = lambda x: x.get_optimum())
        for (i, test_spectrum) in enumerate(test_spectra):
            output_path = output_dir / f"{self._id}_{tag}_{i}.ts"
            test_spectrum.write(output_path)

    def print_summary(self):
        self._check_finalized()
        n = len(self)
        n_matches = self.n_matches
        n_misses = self.n_misses
        n_crashes = self.n_crashes
        n_time_outliers = self.n_temporal_outliers
        n_size_outliers = self.n_spatial_outliers
        print(f"\nsummary:\n>total\n\t{n}\n>matches\n\t{n_matches} ({n_matches / n})\n>misses\n\t{n_misses} ({n_misses / n})\n>crashes\n\t{n_crashes} ({n_crashes / n})\n>temporal outliers\n\t{n_time_outliers} ({n_time_outliers / n})\n>spatial outliers (without temporal outliers)\n\t{n_size_outliers} ({n_size_outliers / n})\n")
    
    def print_miss_distances(self):
        self._check_finalized()
        miss_scores = self._optima[self._misses]
        if len(miss_scores) > 0:
            util.plot_hist(miss_scores, "miss distance distribution")
        else:
            print("no misses!")

    def _construct_complexity_table(self, mask):
        self._check_finalized()
        times = self._times[mask].sum(axis = 0)
        sizes = self._sizes[mask].sum(axis = 0)
        pct_times = list((100 * times / times.sum()).round(2))
        pct_sum = sum(pct_times)
        raw_times = list(times.round(4))
        step_sizes = list(sizes)
        step_names = TestSpectrum.step_names()
        table = [
            ["step name", *step_names, "total"],
            ["size", *step_sizes, sum(step_sizes)],
            ["time", *raw_times, round(sum(raw_times), 4)],
            ["time (pct)", *pct_times, f"100 (err: {round(pct_sum - 100, 4)})"],]
        return tabulate(table)

    def print_complexity_table(self):
        self._check_finalized()
        time_mask = self._time_outliers_mask
        size_mask = self._size_outliers_mask
        normal_mask = ~(time_mask | size_mask)

        complexity_table = self._construct_complexity_table(normal_mask)
        complexity_title = "complexity" + (" (without outliers)" if ((self.n_spatial_outliers + self.n_temporal_outliers) > 0) else '')
        print(f"\n{complexity_title}:\n{complexity_table}")
        
        if self.n_temporal_outliers > 0:
            temporal_outlier_table = self._construct_complexity_table(time_mask)
            temporal_outlier_title = "temporal outlier complexity" + (" (without spatial outliers)" if self.n_spatial_outliers > 0 else '')
            print(f"{temporal_outlier_title}:\n{temporal_outlier_table}")
        
        if self.n_spatial_outliers > 0:
            spatial_outlier_table = self._construct_complexity_table(size_mask)
            spatial_outlier_title = "spatial outlier complexity" + (" (without temporal outliers)" if self.n_temporal_outliers > 0 else '')
            print(f"{spatial_outlier_title}:\n{spatial_outlier_table}")