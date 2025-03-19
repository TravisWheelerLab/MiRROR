import itertools
from enum import Enum

import networkx as nx

from .util import GAP_TOLERANCE, INTERGAP_TOLERANCE
from .pivots import Pivot
from .graph_utils import *

#=============================================================================#

GAP_KEY = "gap"

# todo - update for new gap annotations
GAP_COMPARATOR = lambda x,y: (abs(x - y) < INTERGAP_TOLERANCE) and (x != -1) and (y != -1)

class SpectrumGraphOrientation(Enum):
    ASCENDING = 1
    DESCENDING = 2

#=============================================================================#

def _half_graph_edges(
    gaps,
    pivot,
    orientation,
) -> nx.DiGraph:
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
            
def _create_half_graph_from_gaps(
    spectrum,
    gaps,
    pivot,
    boundaries,
    gap_key,
    orientation,
) -> nx.DiGraph:
    edges = list(_half_graph_edges(gaps, pivot, orientation))
    half_graph = nx.DiGraph()
    for (i, j) in edges:
        v = abs(spectrum[j] - spectrum[i])
        half_graph.add_edge(i, j)
        half_graph[i][j][gap_key] = v
    
    for i in list(half_graph.nodes):
        half_graph.add_edge(i, -1)
        half_graph[i][-1][gap_key] = -1
    
    for i in boundaries:
        if (i in half_graph):
            inbound_edges = list(half_graph.in_edges(i))
            for (j, _) in inbound_edges:
                half_graph.remove_edge(j, i)
    
    return half_graph

def create_spectrum_graph_pair(
    spectrum,
    gaps: list[tuple[int,int]],
    pivot: Pivot,
    boundaries: list[int],
    gap_key = GAP_KEY,
) -> GraphPair:
    """Constructs a pair of spectrum graphs. The ascending graph contains all gaps
    with indices larger than the pivot indices, with edge pairs in ascending order.
    The descending graph contains all gaps with indices smaller than the pivot indices,
    with edge pairs in descending order.
    
    :spectrum: a sorted array of floats (peak mz values).
    :gap_indices: a list of integer 2-tuples which index into `spectrum`.
    :pivot: a Pivot object indexing into `spectrum`.
    :gap_key: the key to retrieve edge weights from the spectrum graph pair."""
    # build the descending graph on the lower half of the spectrum
    desc_graph = _create_half_graph_from_gaps(
        spectrum,
        gaps,
        pivot,
        boundaries,
        gap_key,
        SpectrumGraphOrientation.DESCENDING
    )
    
    # build the ascending graph on the upper half of the spectrum
    asc_graph = _create_half_graph_from_gaps(
        spectrum,
        gaps,
        pivot,
        boundaries,
        gap_key,
        SpectrumGraphOrientation.ASCENDING
    )
    
    return (asc_graph, desc_graph)

#=============================================================================#
import subprocess
from pathlib import Path

import networkx as nx
import graphviz as gvz

def _draw_graph(graph, title, gap_key):
    graph.remove_node(-1)
    graph.graph['graph'] = {'rankdir':'LR'}
    graph.graph['node'] = {'shape':'circle'}
    graph.graph['edges'] = {'arrowsize':'4.0'}
    A = to_agraph(graph)
    for (i,j) in graph.edges:
        truncated_weight = round(graph[i][j][gap_key],4)
        res = mirror.util.residue_lookup(truncated_weight)
        A.get_edge(i,j).attr['label'] = f"{res} [ {str(truncated_weight)} ]" 
    A.layout('dot')
    A.draw(title)

def draw_graph_ascii(graph: nx.DiGraph, label, output_dir = Path("./data/output/plots/")):
    # create the graph .dot file
    dot_path = output_dir / f"{label}.dot"
    nx.drawing.nx_pydot.write_dot(graph, dot_path)
    # render the graph ascii with graph-easy
    plot_path = output_dir / f"{label}.txt"
    subprocess.run(["graph-easy", dot_path, plot_path])
    # read and return the ascii string
    with open(plot_path, 'r') as f:
        return f.read()

def draw_graph_pair_ascii(graph_pair: GraphPair):
    labels = (".asc", ".desc")
    asc_ascii, desc_ascii = [draw_graph_ascii(g, l) for (g, l) in zip(graph_pair, labels)]
    asc_lines = asc_ascii.split('\n')
    asc_max = max([len(l) for l in asc_lines])
    desc_lines = desc_ascii.split('\n')
    desc_max = max([len(l) for l in desc_lines])
    max_lines = max(len(asc_lines), len(desc_lines))
    for lines in (asc_lines, desc_lines):
        for i in range(max_lines - len(lines)):
            lines.append("")
    return '\n'.join([f"{asc_line.ljust(asc_max)}\t{desc_line.ljust(desc_max)}" for (asc_line, desc_line) in zip(asc_lines, desc_lines)])

def _test_graph_pair_ascii():
    g = nx.DiGraph()
    g.add_edge(1,2)
    g.add_edge(2,3)
    g.add_edge(1,3)
    h = nx.DiGraph()
    h.add_edge(-1,-2)
    h.add_edge(-2,-3)
    h.add_edge(-3,-4)
    h.add_edge(-1,-4)
    print(draw_graph_pair_ascii((g,h)))