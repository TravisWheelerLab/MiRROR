from dataclasses import dataclass
import pickle

import networkx as nx
import numpy as np
import numpy.typing as npt

@dataclass
class OutputTrace:
    pivot_index: int
    boundary_index: int
    affixes_index: int
    candidate_index: int

@dataclass
class TestSpectrum:
    # true/target data:
    mass_sequence: np.ndarray
    residue_sequence: np.ndarray
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
    terminal_residues: list[str]
    boundary_padding: int
    gap_key: str
    
    # output:
    # a list of candidates for each viable pair of affixes.
    _output_candidates: list[list[list[list[Candidate]]]] = None
    # a list of index pairs into each array of affixes
    _output_affix_pairs: list[list[npt.NDArray[int]]]
    # an array of affixes for each spectrum graph pair.
    _output_affixes: list[list[npt.NDArray[Affix]]] = None
    # a spectrum graph pair for each boundary.
    _output_graph_pairs: list[list[tuple[nx.DiGraph,nx.DiGraph]]] = None
    # an augmented spectrum, pivot, and gap indices for each boundary.
    _output_augmented_data: list[list[tuple[np.ndarray,Pivot,list[tuple[int,int]]]]] = None
    # a list of boundaries for each pivot.
    _output_boundaries: list[list[Boundary]] = None
    # some number of pivots.
    _output_pivots: list[Pivot] = None
    # peaks annotated by the set of GapResult objects
    _output_annotated_peaks: np.ndarray 
    # a gap result for each amino acid, plus an unrecognized category.
    _output_gaps: list[GapResult] = None

    # stitches the output stack together by indexing into pivots, boundaries, affixes, and candidates.
    # there is an OutputTrace for every Candidate:
    # len(_output_traces) == sum(sum(len(x) for x in X) for X in self._output_candidates)
    _output_traces: list[OutputTrace] = None
    
    def run(self):
        """Generate candidates, associated data structures, and output traces."""
        # gaps
        self._output_annotated_peaks, self._output_gaps = mirror.find_gaps(
            self.gap_search_parameters,
            self.mz
        )
        
        # pivots
        symmetry_threshold = mirror.util.expected_num_mirror_symmetries(annotated_peaks)
        self._output_pivots = mirror.find_all_pivots(
            self._output_annotated_peaks, 
            symmetry_threshold, 
            self._output_gaps, 
            self.intergap_tolerance
        )
        
        # boundaries
        self._output_boundaries = [None for _ in range(len(self._output_pivots))]
        for p_idx, pivot in enumerate(self._output_pivots):
            boundaries, _, __ = mirror.find_and_create_boundaries(
                self._output_annotated_peaks, 
                pivot,
                self.gap_search_parameters,
                self.terminal_residues,
                self.boundary_padding
            )
            self._output_boundaries[p_idx] = boundaries
        
        # augmented data
        self._output_augmented_data = [[None for _ in range(len(self._output_boundaries[p_idx]))] for p_idx in range(len(self._output_pivots))]
        for p_idx in range(len(self._output_pivots)):
            for b_idx, boundary in enumerate(self._output_boundaries[p_idx]):
                augmented_peaks = boundary.get_augmented_peaks()
                augmented_pivot = boundary.get_augmented_pivot()
                augmented_gaps = boundary.get_augmented_gaps()
                self._output_augmented_data[p_idx][b_idx] = (augmented_peaks, augmented_pivot, augmented_gaps)
        
        # spectrum graphs
        self._output_spectrum_graphs = [[None for _ in range(len(self._output_boundaries[p_idx]))] for p_idx in range(len(self._output_pivots))]
        for p_idx in range(len(self._output_pivots)):
            for b_idx in range(len(self._output_boundaries[p_idx])):
                augmented_peaks, augmented_pivot, augmented_gaps = self._output_augmented_data[p_idx][b_idx]
                self._output_spectrum_graphs[p_idx][b_idx] = mirror.create_spectrum_graph_pair(augmented_peaks, augmented_pivot, augmented_gaps, gap_key = self.gap_key)
        
        # affixes and affix pairs
        self._output_affixes = [[None for _ in range(len(self._output_boundaries[p_idx]))] for p_idx in range(len(self._output_pivots))]
        self._output_affix_pairs = [[None for _ in range(len(self._output_boundaries[p_idx]))] for p_idx in range(len(self._output_pivots))]
        for p_idx in range(len(self._output_pivots)):
            for b_idx in range(len(self._output_boundaries[p_idx])):
                graph_pair = self._output_spectrum_graphs[p_idx][b_idx]
                gap_comparator = lambda x, y: (abs(x - y) < args.intergap_tolerance) and (x != -1) and (y != -1)
                dual_paths = mirror.find_dual_paths(
                    *graph_pair,
                    args.gap_key,
                    gap_comparator)
                self._output_affix_pairs = np.array(mirror.find_edge_disjoint_dual_path_pairs(dual_paths))
                self._output_affixes[p_idx][b_idx] = np.array([mirror.create_affix(dp, graph_pair) for dp in dual_paths])
        
        # candidates
        self._output_candidates = [[[None for _ in range(len(self._output_affix_pairs[p_idx][b_idx]))] 
                                        for b_idx in range(len(self._output_boundaries[p_idx]))] for p_idx in range(len(self._output_pivots))]
        for p_idx, pivot in enumerate(self._output_pivots):
            pivot_residue = mirror.util.residue_lookup(pivot.gap())
            for b_idx, boundary in enumerate(self._output_boundaries[p_idx]):
                boundary_residues = boundary.get_residues()
                augmented_peaks = self._output_augmented_data[p_idx][b_idx][0]
                graph_pair = self._output_spectrum_graphs[p_idx][b_idx]
                affixes = self._output_affixes[p_idx][b_idx]
                for a_idx, affix_pair in enumerate(self._output_affix_pairs[p_idx][b_idx]):
                    self._output_candidates[p_idx][b_idx][a_idx] = mirror.create_candidates(
                        augmented_peaks,
                        graph_pair,
                        affixes[index_pair],
                        boundary_residues,
                        pivot_residue
                    )
        
        # traces
        self._output_traces = []
        for p_idx in range(len(self._output_pivots)):
            for b_idx, boundary in enumerate(self._output_boundaries[p_idx]):
                for a_idx in range(len(self._output_affix_pairs[p_idx][b_idx])):
                    for c_idx in range(len(self._output_candidates[p_idx][b_idx][a_idx])):
                        self._output_traces.append(OutputTrace(
                            pivot_index = p_idx, 
                            boundary_index = b_idx, 
                            affixes_index = a_idx, 
                            candidate_index = c_idx
                        ))

    def __post_init__(self):
        self.call()
        candidate_indices = [(trace.pivot_index, trace.boundary_index, trace.affixes_index, trace.candidate_index) for trace in self._output_traces]
        self._edit_distances = np.array([
            self._output_candidates[p_idx][b_idx][a_idx][c_idx].edit_distance(self.residue_sequence) 
            for (p_idx, b_idx, a_idx, c_idx) in candidate_indices
        ])

    def __len__(self):
        return len(self._output_traces)
    
    def __getitem__(self, i) -> OutputTrace:
        return self._output_traces[i]
    
    def __iter__(self) -> list[OutputTrace]:
        return self._output_traces
    
    def peptide(self):
        return ''.join(self.residue_sequence)

def write_test_spectrum(handle, test_spectrum: TestSpectrum):
    pickle.dump(test_spectrum, handle)

def read_test_spectrum(handle):
    return pickle.load(handle)