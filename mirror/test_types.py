from dataclasses import dataclass
import pickle
from types import NoneType

import networkx as nx
import numpy as np
import numpy.typing as npt

from . import *

#from .io import *
#from .preprocessing import create_spectrum_bins, filter_spectrum_bins
#
#from .gaps import GapMatch, GapResult, GapTensorTransformationSolver, GapBisectTransformationSolver, duplicate_inverse_charges, find_gaps, find_gaps_old
#from .pivots import find_all_pivots, find_all_pivots_gap_agnostic
#from .boundaries import find_and_create_boundaries, Boundary
#
#from .spectrum_graphs import create_spectrum_graph_pair
#from .graph_utils import get_sinks, get_sources, find_dual_paths, find_extended_paths, find_edge_disjoint_dual_path_pairs
#
#from .affixes import create_affix, filter_affixes, find_affix_pairs
#from .candidates import create_candidates, filter_candidate_sequences

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
    gaps: dict[str, GapResult]
    pivot: Pivot
    
    # params:
    gap_search_parameters: GapSearchParameters
    intergap_tolerance: float
    symmetry_factor: float
    terminal_residues: list[str]
    boundary_padding: int
    gap_key: str
    
    # output:
    # a (potentially empty) list of candidates for each viable pair of affixes.
    _output_candidates: list[list[list[list[Candidate]]]] = None
    # a list of index pairs into each array of affixes
    _output_affix_pairs: list[list[npt.NDArray[int]]] = None
    # an array of affixes for each spectrum graph pair.
    _output_affixes: list[list[npt.NDArray[Affix]]] = None
    # a spectrum graph pair for each boundary.
    _output_graph_pairs: list[list[tuple[nx.DiGraph,nx.DiGraph]]] = None
    # an augmented spectrum, pivot, gap indices, and boundary peak indices for each boundary.
    _output_augmented_data: list[list[tuple[np.ndarray,Pivot,list[tuple[int,int]],list[int]]]] = None
    # a list of boundaries for each pivot.
    _output_boundaries: list[list[Boundary]] = None
    # some number of pivots.
    _output_pivots: list[Pivot] = None
    # peaks annotated by the set of GapResult objects
    _output_annotated_peaks: np.ndarray = None
    # a gap result for each amino acid, plus an unrecognized category.
    _output_gaps: list[GapResult] = None

    # stitches the output stack together by indexing into pivots, boundaries, affixes, and candidates.
    # there is an OutputTrace for every Candidate.
    _output_indices: list[OutputIndex] = None

    #@classmethod
    #def random(cls, params: TestParameters):
    #    run_params = params.run_params
    #    mass_sequence, residue_sequence, losses, modifications, charges, noise, mz, true_gaps, position_lookup = gap_simulate.random_data(
    #        params.num_residues,
    #        params.num_losses,
    #        params.num_modifications,
    #        params.num_charged,
    #        params.num_noise,
    #        masses = params.masses,
    #        residues = params.residues,
    #        losses = params.losses,
    #        modifications = params.modifications,
    #        charges = params.charges
    #    )
    #    
    #    return cls(
    #        residue_sequence,
    #        mass_sequence,
    #        losses,
    #        charges,
    #        modifications,
    #        noise,
    #        mz,
    #        gaps,
    #        pivot,
    #        gap_search_parameters = run_params.gap_search_parameters,
    #        intergap_tolerance = run_params.pivot_tolerance,
    #        terminal_residues = run_params.terminal_residues,
    #        boundary_padding = run_params.boundary_padding,
    #        gap_key = run_params.gap_key
    #    )

    @classmethod
    def read(cls, handle):
        return pickle.load(handle)
    
    def _check_state(self, *args):
        values = [self.__dict__[k] for k in args]
        valtypes = [type(v) for v in values]
        if NoneType in valtypes:
            state = '\n\t'.join(f"{k} = {v} (t)" for (k, v, t) in zip(args, values, valtypes))
            raise ValueError(f"Invalid TestSpectrum State:{state}")
    
    def run_gaps(self):
        self._check_state("gap_search_parameters", "mz")
        self._output_annotated_peaks, self._output_gaps = find_gaps(
            self.gap_search_parameters,
            self.mz
        )

    def run_pivots(self):
        self._check_state("_output_gaps", "_output_annotated_peaks", "intergap_tolerance", "symmetry_factor")
        symmetry_threshold = self.symmetry_factor * util.expected_num_mirror_symmetries(self._output_annotated_peaks)
        self._output_pivots = find_all_pivots(
            self._output_annotated_peaks, 
            symmetry_threshold, 
            self._output_gaps, 
            self.intergap_tolerance
        )
        self._n_pivots = len(self._output_pivots)
        #print(f"pivots:\n\t{self._n_pivots}\n\t{self._output_pivots}")
    
    def run_boundaries(self):
        self._check_state("_output_annotated_peaks", "_output_pivots")
        self._output_boundaries = [None for _ in range(len(self._output_pivots))]
        self._n_boundaries = [-1 for _ in range(len(self._output_pivots))]
        for p_idx, pivot in enumerate(self._output_pivots):
            boundaries, _, __ = find_and_create_boundaries(
                self._output_annotated_peaks, 
                pivot,
                self.gap_search_parameters,
                self.terminal_residues,
                self.boundary_padding
            )
            self._output_boundaries[p_idx] = boundaries
            self._n_boundaries[p_idx] = len(boundaries)
        #print(f"boundaries:\n\t{self._n_boundaries}\n\t{self._output_boundaries}")
    
    def run_augment(self):
        self._check_state("_output_pivots", "_output_boundaries")
        self._output_augmented_data = [[None 
            for _ in range(len(self._output_boundaries[p_idx]))] 
            for p_idx in range(len(self._output_pivots))]
        for p_idx in range(self._n_pivots):
            for b_idx, boundary in enumerate(self._output_boundaries[p_idx]):
                augmented_peaks = boundary.get_augmented_peaks()
                augmented_pivot = boundary.get_augmented_pivot()
                augmented_gaps = boundary.get_augmented_gaps()
                boundary_indices = boundary.get_boundary_indices()
                self._output_augmented_data[p_idx][b_idx] = (
                    augmented_peaks, 
                    augmented_pivot, 
                    augmented_gaps, 
                    boundary_indices
                )
    
    def run_spectrum_graphs(self):
        self._check_state("_output_pivots", "_output_boundaries", "_output_augmented_data", "gap_key")
        self._output_spectrum_graphs = [[None for _ in range(len(self._output_boundaries[p_idx]))] for p_idx in range(self._n_pivots)]
        for p_idx in range(self._n_pivots):
            for b_idx in range(self._n_boundaries[p_idx]):
                augmented_peaks, augmented_pivot, augmented_gaps, boundary_indices = self._output_augmented_data[p_idx][b_idx]
                self._output_spectrum_graphs[p_idx][b_idx] = create_spectrum_graph_pair(
                    augmented_peaks, 
                    augmented_gaps, 
                    augmented_pivot, 
                    boundary_indices, 
                    gap_key = self.gap_key
                )
    
    def run_affixes_and_affix_pairs(self):
        self._check_state("_output_pivots", "_output_boundaries", "_output_spectrum_graphs", "gap_key")
        self._output_affixes = [[None  for _ in range(self._n_boundaries[p_idx])] for p_idx in range(self._n_pivots)]
        self._n_affixes = [[-1  for _ in range(self._n_boundaries[p_idx])] for p_idx in range(self._n_pivots)]
        self._output_affix_pairs = [[None  for _ in range(self._n_boundaries[p_idx])] for p_idx in range(self._n_pivots)]
        self._n_affix_pairs = [[-1 for _ in range(self._n_boundaries[p_idx])] for p_idx in range(self._n_pivots)]
        for p_idx in range(self._n_pivots):
            for b_idx in range(self._n_boundaries[p_idx]):
                graph_pair = self._output_spectrum_graphs[p_idx][b_idx]
                gap_comparator = lambda x, y: (abs(x - y) < self.intergap_tolerance) and (x != -1) and (y != -1)
                dual_paths = find_dual_paths(
                    *graph_pair,
                    self.gap_key,
                    gap_comparator)
                affixes = np.array([create_affix(dp, graph_pair) for dp in dual_paths])
                affix_pairs = np.array(find_edge_disjoint_dual_path_pairs(dual_paths))
                self._output_affixes[p_idx][b_idx] = affixes
                self._n_affixes[p_idx][b_idx] = len(affixes)
                self._output_affix_pairs[p_idx][b_idx] = affix_pairs
                self._n_affix_pairs[p_idx][b_idx] = len(affix_pairs)
        #print(f"affixes:\n\t{self._n_affixes}\n\t{[list(aa) for a in self._output_affixes for aa in a]}")
        #print(f"affix pairs:\n\t{self._n_affix_pairs}\n\t{[list(aa) for a in self._output_affix_pairs for aa in a]}")
    
    def run_candidates(self):
        self._check_state("_output_pivots", "_output_boundaries", "_output_affixes", "_output_affix_pairs")
        self._output_candidates = [[[None for _ in range(self._n_affix_pairs[p_idx][b_idx])] for b_idx in range(self._n_boundaries[p_idx])] for p_idx in range(self._n_pivots)]
        self._n_candidates = [[[-1 for _ in range(self._n_affix_pairs[p_idx][b_idx])] for b_idx in range(self._n_boundaries[p_idx])] for p_idx in range(self._n_pivots)]
        for p_idx, pivot in enumerate(self._output_pivots):
            pivot_residue = pivot.residue()
            for b_idx, boundary in enumerate(self._output_boundaries[p_idx]):
                boundary_residues = boundary.get_residues()
                augmented_peaks = self._output_augmented_data[p_idx][b_idx][0]
                graph_pair = self._output_spectrum_graphs[p_idx][b_idx]
                affixes = self._output_affixes[p_idx][b_idx]
                for a_idx, affix_pair in enumerate(self._output_affix_pairs[p_idx][b_idx]):
                    self._output_candidates[p_idx][b_idx][a_idx] = create_candidates(
                        augmented_peaks,
                        graph_pair,
                        affixes[affix_pair],
                        boundary_residues,
                        pivot_residue
                    )
                    self._n_candidates[p_idx][b_idx][a_idx] = len(self._output_candidates[p_idx][b_idx][a_idx])
        #print(f"candidates:\n\t{self._n_candidates}\n\t{self._output_candidates}")
    
    def run_indices(self):
        self._check_state("_output_pivots", "_output_boundaries", "_output_affix_pairs", "_output_candidates")
        self._output_indices = []
        for p_idx in range(self._n_pivots):
            for b_idx in range(self._n_boundaries[p_idx]):
                for a_idx in range(self._n_affix_pairs[p_idx][b_idx]):
                    for c_idx in range(self._n_candidates[p_idx][b_idx][a_idx]):
                        self._output_indices.append(OutputIndex(
                            pivot_index = p_idx, 
                            boundary_index = b_idx, 
                            affixes_index = a_idx, 
                            candidate_index = c_idx
                        ))
        self._n_indices = len(self._output_indices)
        #print(f"indices (all candidates):\n\t{self._n_indices}\n\t{self._output_indices}")

    def run(self):
        """Generate candidates, associated data structures, and output traces."""
        # gaps
        self.run_gaps()
        
        # pivots
        self.run_pivots()
        
        # boundaries
        self.run_boundaries()

        # augmented datad
        self.run_augment()
        
        # spectrum graphs
        self.run_spectrum_graphs()
        
        # affixes and affix pairs
        self.run_affixes_and_affix_pairs()

        # candidates
        self.run_candidates()
        
        # indices
        self.run_indices()

    def __post_init__(self):
        self.run()
        if self._n_indices > 0:
            candidate_indices = [(trace.pivot_index, trace.boundary_index, trace.affixes_index, trace.candidate_index) for trace in self._output_indices]
            self._edit_distances = np.array([
                self._output_candidates[p][b][a][c].edit_distance(self.residue_sequence)[0]
                for (p, b, a, c) in candidate_indices
            ])
            self._optimizer = self._edit_distances.argmin()
            self._optimum = self._edit_distances[self._optimizer]

    def __len__(self):
        return len(self._output_indices)
    
    def __getitem__(self, _i: OutputIndex) -> Candidate:
        i = self._output_indices[_i]
        return self._output_candidates[i.pivot_index][i.boundary_index][i.affixes_index][i.candidate_index]
    
    def __iter__(self) -> list[Candidate]:
        return (self[i] for i in self._output_indices)
      
    def optimize(self) -> tuple[int, Candidate]:
        if self._n_indices == 0:
            return 2 * len(self.peptide()), None
        else:
            return self._optimum, self[self._optimizer]
    
    def peptide(self):
        return ''.join(self.residue_sequence)

    def write(self, handle):
        pickle.dump(self, handle)
