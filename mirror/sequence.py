from .util import collapse_second_order_list, reflect, AMINOS, AMINO_MASS_MONO, ION_OFFSET_LOOKUP
from .scan import ScanConstraint, constrained_pair_scan
from .pivot import Pivot
from numpy import argmin
from networkx import DiGraph
import itertools

TOLERANCE = min(abs(m1 - m2) for m1 in AMINO_MASS_MONO for m2 in AMINO_MASS_MONO if abs(m1 - m2) > 0)

def amino_lookup(
    mass: float,
    letters: list[str],
    mass_table: list[float],
    tolerance: float = TOLERANCE,
):
    dif = [abs(m - mass) for m in mass_table]
    i = argmin(dif)
    optimum = dif[i]
    if optimum > err:
        return 'X' 
    else:
        l = letters[i]
        if l == "L" or l == "I": # these have the same mass
            return "I/L"
        else:
            return l

def recover_initial_residue(
    mz: float,
    pivot: float,
    amino_names = AMINOS,
    amino_masses = AMINO_MASS_MONO,
    ion_offset = ION_OFFSET_LOOKUP
):
    residue_mass = reflect(mz, pivot) - ion_offset['b']
    return amino_lookup(residue_mass, amino_names, amino_masses)

def recover_final_residue(
    mz: float,
    amino_names = AMINOS,
    amino_masses = AMINO_MASS_MONO,
    ion_offset = ION_OFFSET_LOOKUP
):
    residue_mass = mz - ion_offset['y']
    return amino_lookup(residue_mass, amino_names, amino_masses)

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
    desc_outer_loop_range = lambda size: (0, inner_l + 1)
    desc_inner_loop_range = lambda size,idx: (idx + 1, inner_l + 1)
    desc_constraint = SpectrumGraphConstraint()
    desc_edges = constrained_pair_scan(
        spectrum,
        constraint,
        outer_loop_range,
        inner_loop_range
    )
    for (i,j,v) in desc_edges:
        desc_graph.add_edge(j, i, gap = v, peaks = (spectrum[j], spectrum[i]))
    # build the ascending graph
    asc_graph = DiGraph()
    asc_outer_loop_range = lambda size: (inner_r, size)
    asc_inner_loop_range = lambda size,idx: (idx + 1, size)
    asc_constraint = SpectrumGraphConstraint()
    asc_edges = constrained_pair_scan(
        spectrum,
        constraint,
        outer_loop_range,
        inner_loop_range
    )
    for (i,j,v) in asc_edges:
        asc_graph.add_edge(j, i, gap = v, peaks = (spectrum[j], spectrum[i]))
    # 
    return asc_graph, desc_graph

def get_sources(D: DiGraph):
    return [i for i in D.nodes if D.in_degree(i) == 0]

def get_sinks(D: DiGraph):
    return [i for i in D.nodes if D.out_degree(i) == 0]

def get_weights(graph, path, key):
    return [graph[path[i]][path[i + 1]][key] for i in range(len(path) - 1)]

class PartialSequence:

    def __init__(self,
        desc_pairs,
        desc_gaps,
        desc_aminos,
        asc_pairs,
        asc_gaps,
        asc_aminos
    ):
        self.desc_pairs = desc_pairs
        self.desc_gaps = desc_gaps
        self.desc_aminos = desc_aminos
        self.asc_pairs = asc_pairs
        self.asc_gaps = asc_gaps
        self.asc_aminos = asc_aminos

    # todo instance methods
    # - determine whether two partial sequences can be concatenated
    # - reverse the sequence

def partial_sequences(
    asc_graph: DiGraph,
    desc_graph: DiGraph,
    tolerance = TOLERANCE,
    amino_names = AMINOS,
    amino_masses = AMINO_MASS_MONO,
    gap_key = "gap",
    peak_pair_key = "peaks",
):
    weight_transform = lambda mass: amino_lookup(mass, amino_names, amino_masses, tolerance)
    #asc_graph, desc_graph = construct_spectrum_graphs(spectrum, pivot)
    asc_sources = get_sources(asc_graph)
    asc_sinks = get_sinks(asc_graph)
    desc_sources = get_sources(desc_graph)
    desc_sinks = get_sinks(desc_graph)
    # iterate all the shared path spaces for all combinations of source nodes in the product space
    mirror_paths_iterator = itertools.chain.from_iterable(
        weighted_paired_simple_paths(
            desc_graph, d_source, set(desc_sinks),
            asc_graph, a_source, set(asc_sinks),
            gap_key, weight_transform) 
        for d_source in desc_sources for a_source in asc_sources
    )
    # iterate paths and process path metadata to generate amino sequences
    for mirror_path_pair in mirror_paths_iterator:
        desc_path, asc_path = mirror_path_pair
        desc_gaps = get_weights(desc_graph, desc_path, gap_key)
        desc_aminos = [amino_lookup(w, amino_names, amino_masses, tolerance) for w in desc_weights]
        desc_pairs = get_weights(desc_graph, desc_path, peak_pair_key)
        asc_gaps = get_weights(asc_graph, asc_path, gap_key)
        asc_aminos = [amino_lookup(w, amino_names, amino_masses, tolerance) for w in asc_weights]
        yield PartialSequence(desc_pairs, desc_gaps, desc_aminos, asc_pairs, asc_gaps, asc_aminos)