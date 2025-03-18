from dataclasses import dataclass
import pickle
from types import NoneType
from time import time

import networkx as nx
import numpy as np
import numpy.typing as npt
from scipy.stats import zscore

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
    suffix_array_file: str
    occurrence_threshold: int

    # test params:
    autorun: bool = True
    
    # output:
    # a (potentially empty) list of candidates for each viable pair of affixes.
    _candidates: list[list[list[list[Candidate]]]] = None
    # a list of index pairs into each array of affixes
    _affix_pairs: list[list[npt.NDArray[int]]] = None
    # an array of affixes for each spectrum graph pair.
    _affixes: list[list[npt.NDArray[Affix]]] = None
    # a spectrum graph pair for each boundary.
    _graph_pairs: list[list[tuple[nx.DiGraph,nx.DiGraph]]] = None
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
    
    def _check_state(self, *args):
        values = [self.__dict__[k] for k in args]
        valtypes = [type(v) for v in values]
        if NoneType in valtypes:
            state = '\n\t'.join(f"{k} = {v} {t}" for (k, v, t) in zip(args, values, valtypes) if t == type(None))
            raise ValueError(f"Invalid TestSpectrum State:{state}")
    
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
        self._n_y_terminii = len(self._y_terminii)
        return self._n_y_terminii

    def run_pivots(self):
        self._check_state("_gaps", "_annotated_peaks", "intergap_tolerance", "symmetry_factor")
        self._pivots = find_all_pivots(
            self._annotated_peaks, 
            self.symmetry_factor * util.expected_num_mirror_symmetries(self._annotated_peaks), 
            self._gaps, 
            self.intergap_tolerance
        )
        self._n_pivots = len(self._pivots)
        return self._n_pivots

    def run_boundaries(self):
        self._check_state("_annotated_peaks", "_pivots", "_y_terminii")
        self._boundaries = [None for _ in range(len(self._pivots))]
        self._n_boundaries = [-1 for _ in range(len(self._pivots))]
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
            self._n_boundaries[p_idx] = len(boundaries)
        return sum(self._n_boundaries)

    def run_augment(self):
        self._check_state("_pivots", "_boundaries")
        self._augmented_data = [[None 
            for _ in range(len(self._boundaries[p_idx]))] 
            for p_idx in range(len(self._pivots))]
        for p_idx in range(self._n_pivots):
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
        return sum(len(gaps) for _, __, gaps, ___ in self._augmented_data[p_idx] for p_idx in range(self._n_pivots))
    
    def run_spectrum_graphs(self):
        self._check_state("_pivots", "_boundaries", "_augmented_data", "gap_key")
        self._spectrum_graphs = [[None for _ in range(len(self._boundaries[p_idx]))] for p_idx in range(self._n_pivots)]
        for p_idx in range(self._n_pivots):
            for b_idx in range(self._n_boundaries[p_idx]):
                augmented_peaks, augmented_pivot, augmented_gaps, boundary_indices = self._augmented_data[p_idx][b_idx]
                self._spectrum_graphs[p_idx][b_idx] = create_spectrum_graph_pair(
                    augmented_peaks, 
                    augmented_gaps, 
                    augmented_pivot, 
                    boundary_indices, 
                    gap_key = self.gap_key
                )
        return -1
    
    def run_affixes(self):
        self._check_state("_pivots", "_boundaries", "_spectrum_graphs", "gap_key")
        self._affixes = [[None  for _ in range(self._n_boundaries[p_idx])] for p_idx in range(self._n_pivots)]
        self._n_affixes = [[-1 for _ in range(self._n_boundaries[p_idx])] for p_idx in range(self._n_pivots)]
        for p_idx in range(self._n_pivots):
            for b_idx in range(self._n_boundaries[p_idx]):
                graph_pair = self._spectrum_graphs[p_idx][b_idx]
                gap_comparator = lambda x, y: (abs(x - y) < self.intergap_tolerance) and (x != -1) and (y != -1)
                dual_paths = find_dual_paths(
                    *graph_pair,
                    self.gap_key,
                    gap_comparator)
                affixes = np.array([create_affix(dp, graph_pair) for dp in dual_paths])
                self._affixes[p_idx][b_idx] = affixes
                self._n_affixes[p_idx][b_idx] = len(affixes)
        return sum(sum(self._n_affixes[i]) for i in range(self._n_pivots))
    
    def run_affixes_filter(self):
        self._check_state("_pivots", "_boundaries", "_affixes", "_suffix_array")
        self._unfiltered_n_affixes = [[-1 for _ in range(self._n_boundaries[p_idx])] for p_idx in range(self._n_pivots)]
        for p_idx in range(self._n_pivots):
            for b_idx in range(self._n_boundaries[p_idx]):
                self._unfiltered_n_affixes[p_idx][b_idx] = self._n_affixes[p_idx][b_idx]
                self._affixes[p_idx][b_idx] = filter_affixes(
                    self._affixes[p_idx][b_idx], 
                    self._suffix_array, 
                    self.occurrence_threshold)
                self._n_affixes[p_idx][b_idx] = len(self._affixes[p_idx][b_idx])
                #print(f"unfiltered {self._unfiltered_n_affixes[p_idx][b_idx]} → filtered {self._n_affixes[p_idx][b_idx]}")
        #print(self._n_affixes)
        return sum(sum(self._n_affixes[i]) for i in range(self._n_pivots))

    def run_affixes_pair(self):
        self._check_state("_pivots", "_boundaries", "_affixes")
        self._affix_pairs = [[None  for _ in range(self._n_boundaries[p_idx])] for p_idx in range(self._n_pivots)]
        self._n_affix_pairs = [[-1 for _ in range(self._n_boundaries[p_idx])] for p_idx in range(self._n_pivots)]
        for p_idx in range(self._n_pivots):
            for b_idx in range(self._n_boundaries[p_idx]):
                affixes = self._affixes[p_idx][b_idx]
                affix_pairs = find_affix_pairs(affixes)
                self._affix_pairs[p_idx][b_idx] = affix_pairs
                self._n_affix_pairs[p_idx][b_idx] = len(affix_pairs)
        return sum(sum(self._n_affix_pairs[i]) for i in range(self._n_pivots))

    def run_candidates(self):
        self._check_state("_pivots", "_boundaries", "_affixes", "_affix_pairs")
        self._candidates = [[[None for _ in range(self._n_affix_pairs[p_idx][b_idx])] for b_idx in range(self._n_boundaries[p_idx])] for p_idx in range(self._n_pivots)]
        self._n_candidates = [[[-1 for _ in range(self._n_affix_pairs[p_idx][b_idx])] for b_idx in range(self._n_boundaries[p_idx])] for p_idx in range(self._n_pivots)]
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
                    self._n_candidates[p_idx][b_idx][a_idx] = len(self._candidates[p_idx][b_idx][a_idx])
        return sum(sum(self._n_candidates[p][b]) for p in range(self._n_pivots) for b in range(self._n_boundaries[p]))
   
    def run_indices(self):
        self._check_state("_pivots", "_boundaries", "_affix_pairs", "_candidates")
        self._indices = []
        for p_idx in range(self._n_pivots):
            for b_idx in range(self._n_boundaries[p_idx]):
                for a_idx in range(self._n_affix_pairs[p_idx][b_idx]):
                    for c_idx in range(self._n_candidates[p_idx][b_idx][a_idx]):
                        self._indices.append(OutputIndex(
                            pivot_index = p_idx, 
                            boundary_index = b_idx, 
                            affixes_index = a_idx, 
                            candidate_index = c_idx
                        ))
        self._n_indices = len(self._indices)
        return self._n_indices

    def _record_complexity(self, tag, fn, *args, **kwargs):
        t = time()
        self._size[tag] = fn(*args, **kwargs)
        self._time[tag] = time() - t
    
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

    def run(self):
        """Generate candidates, associated data structures, and output traces."""
        self._size = dict()
        self._time = dict()
        for (tag, fn) in self.run_sequence():
            try:
                self._record_complexity(tag, fn, self)
            except Exception as e:
                print(f"[Warning] step {tag} crashed:\n{e}")
                self._n_indices = -1
                self._crash = [tag]
                break

    @classmethod
    def step_names(cls):
        names, _ = zip(*cls.run_sequence())
        return names

    def times_as_vec(self):
        tvals = list(self._time.values())
        return np.array(tvals + [-np.inf for _ in range(len(self.step_names()) - len(tvals))])
    
    def sizes_as_vec(self):
        svals = list(self._size.values())
        return np.array(svals + [-np.inf for _ in range(len(self.step_names()) - len(svals))])
    
    def get_time(self, tag: str):
        return self._time[tag]
    
    def get_size(self, tag: str):
        return self._size[tag]

    def load_suffix_array(self):
        self._suffix_array = SuffixArray.read(self.suffix_array_file)
    
    def unload_suffix_array(self):
        self._suffix_array = None

    def __post_init__(self):
        # setup
        self._edit_distances = None
        self._optimizer = None
        self._optimum = None
        self.load_suffix_array()
        # construct candidates
        if self.autorun:
            self.run()
            # compare to target
            self._optimize()

    def __len__(self):
        return len(self._indices)
    
    def __getitem__(self, _i: OutputIndex) -> Candidate:
        i = self._indices[_i]
        return self._candidates[i.pivot_index][i.boundary_index][i.affixes_index][i.candidate_index]
    
    def __iter__(self) -> list[Candidate]:
        return (self[i] for i in self._indices)
    
    def _optimize(self):
        if self._n_indices > 0:
            candidate_indices = [
                (trace.pivot_index, trace.boundary_index, trace.affixes_index, trace.candidate_index) 
                for trace in self._indices]
            self._edit_distances = np.array([
                self._candidates[p][b][a][c].edit_distance(self.residue_sequence)[0]
                for (p, b, a, c) in candidate_indices
            ])
            self._optimizer = self._edit_distances.argmin()
            self._optimum = self._edit_distances[self._optimizer]
    
    def optimize(self) -> tuple[int, Candidate]:
        if self._n_indices == 0:
            return 2 * len(self.peptide()), None
        elif self._n_indices > 0:
            if self._optimum == None:
                self._optimize()
            return self._optimum, self[self._optimizer]
        else:
            return -1, None

    def peptide(self):
        return ''.join(self.residue_sequence)

    @classmethod
    def _read(cls, handle):
        return pickle.load(handle)
    
    @classmethod
    def read(cls, filepath):
        with open(filepath, 'rb') as handle:
            ts = cls._read(handle)
            ts.load_suffix_array()
            return ts

    def _write(self, handle):
        pickle.dump(self, handle)

    def write(self, filepath):
        self.unload_suffix_array()
        self.autorun = False
        with open(filepath, 'wb') as handle:
            self._write(handle)

class TestRecord:

    def __init__(self):
        self._test_spectra = []
        self._times = []
        self._sizes = []
        self._optimizers = []
        self._optima = []
        self._n = 0
        self._finalized = False
    
    def __len__(self):
        return self._n

    def add_test_spectrum(self, ts: TestSpectrum):
        self._test_spectra.append(ts)
        self._times.append(ts.times_as_vec())
        self._sizes.append(ts.sizes_as_vec())
        optimum, optimizer = ts.optimize()
        self._optima.append(optimum)
        self._optimizers.append(optimizer)
        self._n += 1

    def finalize(self):
        # convert to numpy arrays
        self._test_spectra = np.array(self._test_spectra)
        self._times = np.vstack(self._times)
        self._sizes = np.vstack(self._sizes)
        self._optimizers = np.array(self._optimizers)
        self._optima = np.array(self._optima)
        # construct masks for each class of TestSpectrum 
        matches = self._optima == 0
        misses = self._optima > 0
        crashes = self._optima == -1
        time_outliers = np.absolute(zscore(self._times.sum(axis = 1))) > 2
        size_outliers = np.absolute(zscore(self._sizes.sum(axis = 1))) > 2
        # store indices for class membership
        ind = np.arange(self._n)
        self._matches = ind[matches]
        self._misses = ind[misses]
        self._crashes = ind[crashes]
        self._time_outliers = ind[time_outliers]
        self._size_outliers = ind[size_outliers]

        self._finalized = True

    def get_matches(self):
        if self._finalized:
            return list(self._test_spectra[self._matches])
        else:
            raise ValueError("this method must be run after `finalize(`")

    def get_misses(self):
        if self._finalized:
            return list(self._test_spectra[self._misses])
        else:
            raise ValueError("this method must be run after `finalize(`")

    def get_crashes(self):
        if self._finalized:
            return list(self._test_spectra[self._crashes])
        else:
            raise ValueError("this method must be run after `finalize(`")

    def get_time_outliers(self):
        if self._finalized:
            return list(self._test_spectra[self._time_outliers])
        else:
            raise ValueError("this method must be run after `finalize(`")

    def get_size_outliers(self):
        if self._finalized:
            return list(self._test_spectra[self._size_outliers])
        else:
            raise ValueError("this method must be run after `finalize(`")