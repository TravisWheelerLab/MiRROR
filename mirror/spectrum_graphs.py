# based on the implementation and interface in `networkx.algorithms.simple_paths`
# https://github.com/networkx/networkx/blob/main/networkx/algorithms/simple_paths.py
import networkx as nx
from .util import residue_lookup, AMINOS, AMINO_MASS_MONO, TOLERANCE
from .scan import ScanConstraint, constrained_pair_scan
from .pivot import Pivot

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

GAP_KEY = "gap"
RES_KEY = "res"

def construct_spectrum_graphs(
    spectrum,
    pivot: Pivot,
    gap_key = GAP_KEY,
    res_key = RES_KEY
):
    outer_left, inner_left, inner_right, outer_right = pivot.index_data
    # build the descending graph
    desc_graph = nx.DiGraph()
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
        desc_graph.add_edge(j, i)
        desc_graph[j][i][gap_key] = v
        desc_graph[j][i][res_key] = residue_lookup(v)
    # build the ascending graph
    asc_graph = nx.DiGraph()
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
        asc_graph.add_edge(i, j, gap = v)
        asc_graph[i][j][gap_key] = v
        asc_graph[i][j][res_key] = residue_lookup(v)
    # 
    return asc_graph, desc_graph

def get_sources(D: nx.DiGraph):
    return [i for i in D.nodes if D.in_degree(i) == 0]

def get_sinks(D: nx.DiGraph):
    return [i for i in D.nodes if D.out_degree(i) == 0]

def get_weights(graph, path, key):
    return [graph[path[i]][path[i + 1]][key] for i in range(len(path) - 1)]

def get_edges(graph,node):
    return graph.edges(node)

def weighted_paired_simple_paths(
    G: nx.DiGraph,
    g_source,
    g_target,
    H: nx.DiGraph,
    h_source,
    h_target,
    weight_key
):
    edge_paths = weighted_paired_simple_edge_paths(
        G,
        g_source,
        g_target,
        H,
        h_source,
        h_target,
        weight_key
    )
    for edge_path in edge_paths:
        yield [(g_source,h_source)] + [edge[1] for edge in edge_path]

def weighted_paired_simple_edge_paths(
    G: nx.DiGraph,
    g_source,
    g_target,
    H: nx.DiGraph,
    h_source,
    h_target,
    weight_key
):
    if g_source not in G:
        raise nx.NodeNotFound(f"primary source node {g_source} not in primary graph")
    if h_source not in H:
        raise nx.NodeNotFound(f"secondary source node {h_source} not in secondary graph")
    
    if g_target in G:
        g_targets = {g_target}
    else:
        try:
            g_targets = set(g_target)
        except TypeError as err:
            raise nx.NodeNotFound(f"primary target node {g_target} not in primary graph") from err
    if h_target in H:
        h_targets = {h_target}
    else:
        try:
            h_targets = set(h_target)
        except TypeError as err:
            raise nx.NodeNotFound(f"secondary target node {h_target} not in secondary graph") from err

    yield from _weighted_paired_simple_edge_paths(
        G,
        g_source,
        g_targets,
        H,
        h_source,
        h_targets,
        weight_key
    )

def _weighted_paired_simple_edge_paths(
    G: nx.DiGraph,
    g_source,
    g_targets: set,
    H: nx.DiGraph,
    h_source,
    h_targets: set,
    weight_key,
):
    def get_edges(graph,node):
        return graph.edges(node)

    def get_weights(graph,edges):
        return [graph[s][t][weight_key] for (s,t) in edges]
    
    current_path = {(None,None): (None,None)}
    stack = [iter([((None,None),(g_source,h_source))])]
    while stack:
        next_edge = next((e for e in stack[-1] if e[1] not in current_path),None)
        if next_edge is None:
            stack.pop()
            current_path.popitem()
            continue
        prev_node, next_node = next_edge

        if next_node[0] in g_targets or next_node[1] in h_targets:
            yield (list(current_path.values()) + [next_edge])[2:]
            
        tup_vtx = current_path.keys()
        g_vtx = set([x[0] for x in tup_vtx])
        h_vtx = set([x[1] for x in tup_vtx])
        if g_targets - g_vtx - {next_node[0]} and h_targets - h_vtx - {next_node[1]}:
            current_path[next_node] = next_edge
            g_edges = get_edges(G, next_node[0])
            g_weights = get_weights(G, g_edges)
            h_edges = get_edges(H, next_node[1])
            h_weights = get_weights(H, h_edges)
            edge_itx = [((g_e[0], h_e[0]), (g_e[1], h_e[1])) 
                for (g_e, g_w) in zip(g_edges, g_weights) 
                for (h_e, h_w) in zip(h_edges, h_weights) 
                if g_w == h_w]
            stack.append(iter(edge_itx))
