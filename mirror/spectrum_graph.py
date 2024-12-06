import itertools
from enum import Enum

import networkx as nx

from .util import AMINO_MASS_MONO, GAP_TOLERANCE, INTERGAP_TOLERANCE
from .scan import ScanConstraint, constrained_pair_scan
from .pivot import Pivot
from .graph_util import *

GAP_KEY = "gap"

GAP_COMPARATOR = lambda x,y: (abs(x - y) < INTERGAP_TOLERANCE) and (x != -1) and (y != -1)

#=============================================================================#
# spectrum graph constructors

class SpectrumGraphOrientation(Enum):
    ASCENDING = 1
    DESCENDING = 2

def _half_graph_edges(
    gaps,
    pivot,
    orientation,
):
    n_gaps = len(gaps)
    if orientation == SpectrumGraphOrientation.ASCENDING:
        reverse_edges = True
        gap_range = range(n_gaps - 1, -1, -1)
        gap_constraint = lambda i: i >= pivot.inner_right()
    elif orientation == SpectrumGraphOrientation.DESCENDING:
        reverse_edges = False
        gap_range = range(0, n_gaps)
        gap_constraint = lambda i: i <= pivot.inner_left()
    else:
        raise ValueError(f"Unrecognized orientation: {orientation}.\nAre you sure it's a SpectrumGraphOrientation object?")
    for (i,j) in gaps:
        if gap_constraint(i) and gap_constraint(j):
            if reverse_edges:
                yield (j, i)
            else:
                yield (i, j)
            
def _construct_half_graph_from_gaps(
    spectrum,
    gaps,
    pivot,
    gap_key,
    orientation,
):
    edges = list(_half_graph_edges(gaps, pivot, orientation))
    half_graph = nx.DiGraph()
    for (i, j) in edges:
        v = abs(spectrum[j] - spectrum[i])
        half_graph.add_edge(i, j)
        half_graph[i][j][gap_key] = v
    
    for i in list(half_graph.nodes):
        half_graph.add_edge(i, -1)
        half_graph[i][-1][gap_key] = -1
    
    return half_graph

def construct_spectrum_graphs(
    spectrum,
    gaps: list[tuple[int,int]],
    pivot: Pivot,
    gap_key = GAP_KEY,
):
    # build the descending graph on the lower half of the spectrum
    desc_graph = _construct_half_graph_from_gaps(
        spectrum,
        gaps,
        pivot,
        gap_key,
        SpectrumGraphOrientation.DESCENDING
    )
    
    # build the ascending graph on the upper half of the spectrum
    asc_graph = _construct_half_graph_from_gaps(
        spectrum,
        gaps,
        pivot,
        gap_key,
        SpectrumGraphOrientation.ASCENDING
    )
    
    return asc_graph, desc_graph

#=============================================================================#
# old constructors, just here for the sanity check.

class SpectrumGraphConstraint(ScanConstraint):

    def __init__(self, 
        tolerance,
        amino_masses,
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

def _construct_half_graph_from_spectrum(
    spectrum,
    gap_key,
    outer_loop_range,
    inner_loop_range,
    constraint,
    reverse_edges = False
):
    half_graph = nx.DiGraph()
    edges = list(constrained_pair_scan(
        spectrum,
        constraint,
        outer_loop_range,
        inner_loop_range
    ))
    for (i, j) in edges:
        v = spectrum[j] - spectrum[i]
        if reverse_edges:
            i, j = j, i
        half_graph.add_edge(i, j)
        half_graph[i][j][gap_key] = v
    for i in list(half_graph.nodes):
        half_graph.add_edge(i, -1)
        half_graph[i][-1][gap_key] = -1
    return half_graph

def construct_spectrum_graphs_from_spectrum(
    spectrum,
    pivot: Pivot,
    gap_key = GAP_KEY,
):
    outer_left, inner_left, inner_right, outer_right = pivot.index_data
    
    # build the descending graph on the lower half of the spectrum
    desc_outer_loop_range = lambda size: (0, inner_left + 1)
    desc_inner_loop_range = lambda size,idx: (idx + 1, inner_left + 1)
    desc_constraint = SpectrumGraphConstraint(GAP_TOLERANCE, AMINO_MASS_MONO)
    desc_graph = _construct_half_graph_from_spectrum(
        spectrum,
        gap_key,
        desc_outer_loop_range,
        desc_inner_loop_range,
        desc_constraint,
        reverse_edges = True
    )
    
    # build the ascending graph on the upper half of the spectrum
    asc_outer_loop_range = lambda size: (inner_right, size)
    asc_inner_loop_range = lambda size,idx: (idx + 1, size)
    asc_constraint = SpectrumGraphConstraint(GAP_TOLERANCE, AMINO_MASS_MONO)
    asc_graph = _construct_half_graph_from_spectrum(
        spectrum,
        gap_key,
        asc_outer_loop_range,
        asc_inner_loop_range,
        asc_constraint,
        reverse_edges = False
    )
    
    return asc_graph, desc_graph

#=============================================================================#

def find_mirrored_paths(
    spectrum,
    gaps,
    pivot: Pivot,
    gap_key = GAP_KEY,
    gap_comparator = GAP_COMPARATOR
):
    ascending_graph, descending_graph = construct_spectrum_graphs(spectrum, gaps, pivot, gap_key)
    
    mirrored_paths = all_weighted_paired_simple_paths(ascending_graph, descending_graph, gap_key, gap_comparator)
    
    #extended_paths = extend_truncated_paths(mirrored_paths, ascending_graph, descending_graph)

    return list(mirrored_paths)