import itertools
import subprocess
from pathlib import Path
from enum import Enum
from copy import deepcopy

import graphviz as gvz
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph

from ..util import horizontal_panes, residue_lookup, GAP_TOLERANCE, INTERGAP_TOLERANCE
from ..pivots import Pivot
from ..boundaries import AugmentedData
from .graph_types import DAG, NodeLabeledDAG, StrongProductDAG
from .align_types import CostModel, FuzzyLocalCostModel, LocalCostModel
from .align import align
from .fragment_types import FragmentChain
from .fragment import collate_fragments
from .graph_utils import *

#=============================================================================#

GAP_KEY = "gap"

# todo - update for new gap annotations
GAP_COMPARATOR = lambda x,y,i,j: (abs(x - y) < INTERGAP_TOLERANCE) and (x != -1) and (y != -1)

def _compare_gaps(x, y, i, j):
    is_eq = (abs(x - y) < INTERGAP_TOLERANCE) and (x != -1) and (y != -1)
    #if is_eq:
    #    print((i, x), "→", (j, y))
    return is_eq
    
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
    
    for i in boundaries:
        if (i in half_graph):
            inbound_edges = list(half_graph.in_edges(i))
            for (j, _) in inbound_edges:
                half_graph.remove_edge(j, i)
    
    for i in pivot.indices():
        if (i in half_graph):
            outbound_edges = list(half_graph.out_edges(i))
            for (_, j) in outbound_edges:
                half_graph.remove_edge(i, j)
    
    # if len(get_sources(half_graph)) == 0 or len(get_sinks(half_graph)) == 0:
    #     node_in_degrees = [len(half_graph.in_edges(i)) for i in half_graph.nodes]
    #     node_out_degrees = [len(half_graph.out_edges(i)) for i in half_graph.nodes]
    #     node_degrees = list(zip(node_in_degrees, node_out_degrees))
    #     print(f"\nno sinks / no sources!\nedges {edges}\nhalf graph {half_graph.nodes}\ndegrees {node_degrees}\nboundaries {boundaries}\npivots {pivot.indices()}")
        
    # for i in list(half_graph.nodes):
    #     half_graph.add_edge(i, -1)
    #     half_graph[i][-1][gap_key] = -1
    
    return half_graph

def create_spectrum_graph_pair(
    augmented_data: AugmentedData,
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
        augmented_data.spectrum,
        augmented_data.gaps,
        augmented_data.pivot,
        augmented_data.boundary,
        gap_key,
        SpectrumGraphOrientation.DESCENDING
    )
    
    # build the ascending graph on the upper half of the spectrum
    asc_graph = _create_half_graph_from_gaps(
        augmented_data.spectrum,
        augmented_data.gaps,
        augmented_data.pivot,
        augmented_data.boundary,
        gap_key,
        SpectrumGraphOrientation.ASCENDING
    )
    
    return (asc_graph, desc_graph)

def align_spectrum_graphs(
    spectrum_graph_pair: tuple[nx.DiGraph, nx.DiGraph],
    threshold: float,
    cost_model: CostModel = None,
    gap_key = GAP_KEY,
) -> list[FragmentChain]:
    # no sufr interface, yet.

    asc_graph, desc_graph = spectrum_graph_pair

    # construct the topology
    asc_dag, desc_dag = map(
        lambda g: NodeLabeledDAG(
            graph = g, 
            weight_key = gap_key), 
        spectrum_graph_pair)
    
    spectrum_product = StrongProductDAG(
        first_graph = asc_dag, 
        second_graph = desc_dag)
    
    def digraph_data(graph):
        nodes = list(graph.nodes)
        degrees = [(graph.in_degree(i), graph.out_degree(i)) for i in nodes]
        sources = [i for i in nodes if graph.in_degree(i) == 0]
        sinks = [i for i in nodes if graph.out_degree(i) == 0]
        return nodes, degrees, sources, sinks
    asc_nodes, asc_degrees, asc_sources, asc_sinks = digraph_data(asc_graph)
    desc_nodes, desc_degrees, desc_sources, desc_sinks = digraph_data(desc_graph)
    #print(f"- asc graph\nnodes {asc_nodes}\ndegrees {asc_degrees}\nsources {asc_sources}\nsinks {asc_sinks}")
    #print(f"- desc graph\nnodes {desc_nodes}\ndegrees {desc_degrees}\nsources {desc_sources}\nsinks {desc_sinks}")
    
    def dag_data(dag):
        nodes = [(dag.get_node_label(i), i) for i in range(dag.order())]
        degrees = [(len(list(dag.adj_in(i[1]))), len(list(dag.adj_out(i[1])))) for i in nodes]
        sources = list(dag.sources())
        sinks = list(dag.sinks())
        return nodes, degrees, sources, sinks
    dag_asc_nodes, dag_asc_degrees, dag_asc_sources, dag_asc_sinks = dag_data(asc_dag)
    dag_desc_nodes, dag_desc_degrees, dag_desc_sources, dag_desc_sinks = dag_data(desc_dag)
    #print(f"- asc DAG\nnodes {dag_asc_nodes}\ndegrees {dag_asc_degrees}\nsources {dag_asc_sources}\nsinks {dag_asc_sinks}")
    #print(f"- desc DAG\nnodes {dag_desc_nodes}\ndegrees {dag_desc_degrees}\nsources {dag_desc_sources}\nsinks {dag_desc_sinks}")

    #print("\n\n\n")
    
    # if list(spectrum_product.sources()) == []:
    #     print("no sources!")
    #     return []

    # generate preliminary alignment
    if cost_model == None:
        cost_model = FuzzyLocalCostModel(tolerance = INTERGAP_TOLERANCE)
    
    aligned_paths = list(align(
        product_graph = spectrum_product,
        cost_model = cost_model,
        threshold = threshold))

    #print(aligned_paths[0].aligned_weights, aligned_paths[0].alignment)

    # collate fragments
    #fragment_chains = list(collate_fragments(
    #    alignments = aligned_paths,
    #    cost_model = cost_model,
    #    threshold = 2 * threshold))
    
    return aligned_paths#, fragment_chains

#=============================================================================#

def draw_graph_simple(graph: nx.DiGraph, gap_key = "gap"):
    graph = deepcopy(graph)
    graph.remove_node(-1)
    graph_repr = []
    for (i, j) in graph.edges:
        truncated_weight = round(graph[i][j][gap_key],4)
        res = residue_lookup(truncated_weight)
        graph_repr.append(f"{i} → {j} : res = {res}, mass = {truncated_weight}")
    return '\n'.join(graph_repr)

def _construct_agraph(graph: nx.DiGraph, gap_key = "gap"):
    graph = deepcopy(graph)
    graph.remove_node(-1)
    #graph.graph['graph'] = {'rankdir':'LR'}
    #graph.graph['node'] = {'shape':'circle'}
    #graph.graph['edges'] = {'arrowsize':'4.0'}
    A = to_agraph(graph)
    for (i,j) in graph.edges:
        truncated_weight = round(graph[i][j][gap_key],4)
        res = residue_lookup(truncated_weight)
        A.get_edge(i,j).attr['label'] = f"{j} {res} {str(truncated_weight)}" 
    return A

def draw_graph_ascii(graph: nx.DiGraph, label, output_dir = Path("./data/output/plots/"), gap_key = "gap"):
    # create agraph with node and edge weights
    agraph = _construct_agraph(graph, gap_key = gap_key)
    # create the graph .dot file
    dot_path = output_dir / f"{label}.dot"
    agraph.layout('dot')
    agraph.write(dot_path)
    # render the graph ascii with graph-easy
    plot_path = output_dir / f"{label}.txt"
    subprocess.run(["graph-easy", dot_path, plot_path], stdout = subprocess.DEVNULL, stderr = subprocess.STDOUT)
    # read and return the ascii string
    with open(plot_path, 'r') as f:
        return f.read()

def draw_graph_pair(graph_pair: GraphPair, mode: str, gap_key = "gap"):
    labels = (".asc", ".desc")
    if mode == "ascii":
        asc_repr, desc_repr = [draw_graph_ascii(g, l, gap_key = gap_key) for (g, l) in zip(graph_pair, labels)]
    else:
        if mode != "simple":
            print(f"Warning: unrecognized graph drawing mode {mode}; defaulting to 'simple'.")
        asc_repr, desc_repr = [draw_graph_simple(g, gap_key = gap_key) for (g, l) in zip(graph_pair, labels)]
    
    return horizontal_panes(asc_repr, desc_repr)
    
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