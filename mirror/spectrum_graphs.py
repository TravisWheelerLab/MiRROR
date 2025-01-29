import itertools
from enum import Enum

import networkx as nx

from .graph_utils import GraphPair
from .util import AMINO_MASS_MONO, GAP_TOLERANCE, INTERGAP_TOLERANCE
from .scan import ScanConstraint, constrained_pair_scan
from .pivots import Pivot
from .graph_utils import *

#=============================================================================#

GAP_KEY = "gap"

GAP_COMPARATOR = lambda x,y: (abs(x - y) < INTERGAP_TOLERANCE) and (x != -1) and (y != -1)

class SpectrumGraphOrientation(Enum):
    ASCENDING = 1
    DESCENDING = 2

#=============================================================================#

def create_spectrum_graph_pair(
    spectrum,
    gaps: list[tuple[int,int]],
    pivot: Pivot,
    gap_key = GAP_KEY,
) -> GraphPair:
    # build the descending graph on the lower half of the spectrum
    desc_graph = _create_half_graph_from_gaps(
        spectrum,
        gaps,
        pivot,
        gap_key,
        SpectrumGraphOrientation.DESCENDING
    )
    
    # build the ascending graph on the upper half of the spectrum
    asc_graph = _create_half_graph_from_gaps(
        spectrum,
        gaps,
        pivot,
        gap_key,
        SpectrumGraphOrientation.ASCENDING
    )
    
    return (asc_graph, desc_graph)
            
def _create_half_graph_from_gaps(
    spectrum,
    gaps,
    pivot,
    gap_key,
    orientation,
) -> nx.Digraph:
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

def _half_graph_edges(
    gaps,
    pivot,
    orientation,
) -> nx.Digraph:
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