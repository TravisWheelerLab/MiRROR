from .util import residue_lookup, collapse_second_order_list, reflect, AMINOS, AMINO_MASS_MONO, ION_OFFSET_LOOKUP, TOLERANCE
from .scan import ScanConstraint, constrained_pair_scan
from .pivot import Pivot
from .paired_paths import * 

from networkx import DiGraph

import itertools

        
def recover_initial_residue(
    mz: float,
    pivot: float,
    amino_names = AMINOS,
    amino_masses = AMINO_MASS_MONO,
    ion_offset = ION_OFFSET_LOOKUP
):
    residue_mass = reflect(mz, pivot) - ion_offset['b']
    return residue_lookup(residue_mass, amino_names, amino_masses)

def recover_final_residue(
    mz: float,
    amino_names = AMINOS,
    amino_masses = AMINO_MASS_MONO,
    ion_offset = ION_OFFSET_LOOKUP
):
    residue_mass = mz - ion_offset['y']
    return residue_lookup(residue_mass, amino_names, amino_masses)

class SpectrumGraphConstraint(ScanConstraint):

    def __init__(self, 
        tolerance = TOLERANCE,
        amino_masses = AMINO_MASS_MONO,
    ):
        self.tolerance = tolerance
        self.amino_masses = amino_masses
        self.max_amino_mass = max(amino_masses)
        self.min_amino_mass = min(amino_masses)
    
    def evaluate(self, state):
        gap = abs(state[0] - state[1])
        return gap
    
    def stop(self, gap):
        return gap > self.max_amino_mass + self.tolerance

    def match(self, gap):
        return min(abs(mass - gap) for mass in self.amino_masses) < self.tolerance

def construct_spectrum_graphs(
    spectrum,
    pivot: Pivot,
):
    outer_left, inner_left, inner_right, outer_right = pivot.index_data
    # build the descending graph
    desc_graph = DiGraph()
    desc_outer_loop_range = lambda size: (0, inner_left + 1)
    desc_inner_loop_range = lambda size,idx: (idx + 1, inner_left + 1)
    desc_constraint = SpectrumGraphConstraint()
    ## this step could be a linear scan of the precomputed gaps.
    desc_edges = constrained_pair_scan(
        spectrum,
        desc_constraint,
        desc_outer_loop_range,
        desc_inner_loop_range
    )
    for (i,j) in desc_edges:
        v = abs(spectrum[i] - spectrum[j])
        desc_graph.add_edge(j, i, gap = v, peaks = (spectrum[j], spectrum[i]))
    # build the ascending graph
    asc_graph = DiGraph()
    asc_outer_loop_range = lambda size: (inner_right, size)
    asc_inner_loop_range = lambda size,idx: (idx + 1, size)
    asc_constraint = SpectrumGraphConstraint()
    ## this step could be a linear scan of the precomputed gaps.
    asc_edges = constrained_pair_scan(
        spectrum,
        asc_constraint,
        asc_outer_loop_range,
        asc_inner_loop_range
    )
    for (i,j) in asc_edges:
        v = abs(spectrum[i] - spectrum[j])
        asc_graph.add_edge(i, j, gap = v, peaks = (spectrum[j], spectrum[i]))
    # 
    return asc_graph, desc_graph

class PartialSequence:

    def __init__(self,
        desc_indices,
        desc_pairs,
        desc_gaps,
        desc_residues,
        asc_indices,
        asc_pairs,
        asc_gaps,
        asc_residues
    ):
        self._len = len(desc_indices)
        assert len(asc_indices) == self._len
        assert desc_residues == asc_residues
        self.residues = desc_residues
        self.desc_indices = desc_indices
        self.desc_pairs = desc_pairs
        self.desc_gaps = desc_gaps
        self.asc_indices = asc_indices
        self.asc_pairs = asc_pairs
        self.asc_gaps = asc_gaps
    
    def __len__(self):
        return self._len
    
    def __repr__(self):
        precision = 3
        round_gaps = lambda gaps: list(map(lambda x: round(x, precision), gaps))
        round_pairs = lambda pairs: list(map(lambda x: (round(x[0], precision),round(x[1], precision)), pairs))
        make_str = lambda idcs, gaps, pairs: f"path\t{idcs}\ngaps\t{round_gaps(gaps)}\npairs\t{round_pairs(pairs)}"
        desc = make_str(self.desc_indices, self.desc_gaps, self.desc_pairs)
        asc = make_str(self.asc_indices, self.asc_gaps, self.asc_pairs)
        return f"\n{self.residues}\nd {desc}\na {asc}\n"
    
    def get_edges(self):
        di = self.desc_indices
        ai = self.asc_indices
        return [((di[i], di[i + 1]), (ai[i], ai[i + 1])) for i in range(len(self) - 1)]

def edge_disjoint(a: PartialSequence, b: PartialSequence):
    return len(set(a.get_edges()) & set(b.get_edges())) == 0

def get_sources(D: DiGraph):
    return [i for i in D.nodes if D.in_degree(i) == 0]

def get_sinks(D: DiGraph):
    return [i for i in D.nodes if D.out_degree(i) == 0]

def get_weights(graph, path, key):
    return [graph[path[i]][path[i + 1]][key] for i in range(len(path) - 1)]

def construct_partial_sequences(
    asc_graph: DiGraph,
    desc_graph: DiGraph,
    tolerance = TOLERANCE,
    amino_names = AMINOS,
    amino_masses = AMINO_MASS_MONO,
    gap_key = "gap",
    peak_pair_key = "peaks",
):
    weight_transform = lambda mass: residue_lookup(mass, amino_names, amino_masses, tolerance)
    #asc_graph, desc_graph = construct_spectrum_graphs(spectrum, pivot)
    asc_sources = get_sources(asc_graph)
    asc_sinks = get_sinks(asc_graph)
    desc_sources = get_sources(desc_graph)
    desc_sinks = get_sinks(desc_graph)
    # iterate all the shared path spaces for all combinations of source nodes
    # in the product space
    mirror_paths_iterator = itertools.chain.from_iterable(
        weighted_paired_simple_paths(
            desc_graph, d_source, set(desc_sinks),
            asc_graph, a_source, set(asc_sinks),
            gap_key, weight_transform) 
        for d_source in desc_sources for a_source in asc_sources
    )
    # iterate paths; use path weights to generate amino sequences
    for mirror_path_pair in mirror_paths_iterator:
        desc_path, asc_path = list(zip(*mirror_path_pair))
        desc_gaps = get_weights(desc_graph, desc_path, gap_key)
        desc_residues = [residue_lookup(w, amino_names, amino_masses, tolerance) for w in desc_gaps]
        desc_pairs = get_weights(desc_graph, desc_path, peak_pair_key)
        asc_gaps = get_weights(asc_graph, asc_path, gap_key)
        asc_residues = [residue_lookup(w, amino_names, amino_masses, tolerance) for w in asc_gaps]
        asc_pairs = get_weights(asc_graph, asc_path, peak_pair_key)
        yield PartialSequence(desc_path, desc_pairs, desc_gaps, desc_residues, asc_path, asc_pairs, asc_gaps, asc_residues)

class JoinedSequence:

    def __init__(self, sequence_a: PartialSequence, sequence_b: PartialSequence, pivot: Pivot):
        self.sequence_a = sequence_a
        self.sequence_b = sequence_b
        self.pivot = pivot

    def call(self):
        # TODO
        # recover initial and final residues
        # using initial and final residues, order the partial sequence pair
        # reverse and concatenate residue strings around pivot char
        pivot = [residue_lookup(self.pivot.gap())]
        seq_a = self.sequence_a.residues
        seq_b = self.sequence_b.residues
        return [
            seq_a + pivot + seq_b[::-1],
            seq_b + pivot + seq_a[::-1],
            seq_a[::-1] + pivot + seq_b,
            seq_b[::-1] + pivot + seq_a,
        ]
    
    def __repr__(self):
        calls = self.call()
        call_abr, call_bar, call_arb, call_bra = map(lambda x: ' '.join(x), calls)
        return f"ab̅:\t{call_abr}\nba̅:\t{call_bar}\na̅b:\t{call_arb}\nb̅a:\t{call_bra}\n"


def construct_candidates(partial_seqs: list[PartialSequence], pivot: Pivot):
    # if the size of partial_seqs becomes unruly, implement the methods from ../2024_1101-disjoint_pairs.
    n = len(partial_seqs)
    return [JoinedSequence(partial_seqs[i], partial_seqs[j], pivot) for i in range(n) for j in range(i + 1, n) if edge_disjoint(partial_seqs[i], partial_seqs[j])]