import networkx as nx
import itertools
from .util import AMINO_MASS_MONO, GAP_TOLERANCE, INTERGAP_TOLERANCE
from .scan import ScanConstraint, constrained_pair_scan
from .pivot import Pivot

#=============================================================================#

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

GAP_KEY = "gap"

def _construct_half_graph(
    spectrum,
    gap_key,
    outer_loop_range,
    inner_loop_range,
    constraint,
    reverse_edges = False
):
    half_graph = nx.DiGraph()
    edges = constrained_pair_scan(
        spectrum,
        constraint,
        outer_loop_range,
        inner_loop_range
    )
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

def construct_spectrum_graphs(
    spectrum,
    pivot: Pivot,
    gap_key = GAP_KEY
):
    outer_left, inner_left, inner_right, outer_right = pivot.index_data
    
    # build the descending graph on the lower half of the spectrum
    desc_outer_loop_range = lambda size: (0, inner_left + 1)
    desc_inner_loop_range = lambda size,idx: (idx + 1, inner_left + 1)
    desc_constraint = SpectrumGraphConstraint(GAP_TOLERANCE, AMINO_MASS_MONO)
    desc_graph = _construct_half_graph(
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
    asc_graph = _construct_half_graph(
        spectrum,
        gap_key,
        asc_outer_loop_range,
        asc_inner_loop_range,
        asc_constraint,
        reverse_edges = False
    )
    
    return asc_graph, desc_graph

#=============================================================================#
# based on the implementation in networkx.algorithms.simple_paths
# https://github.com/networkx/networkx/blob/main/networkx/algorithms/simple_paths.py

def get_sources(D: nx.DiGraph):
    return [i for i in D.nodes if D.in_degree(i) == 0 if i != -1]

def get_sinks(D: nx.DiGraph):
    return [i for i in D.nodes if D.out_degree(i) == 1 if i != -1]

def get_edges(graph,node):
    return [i for i in graph.edges(node) if i != -1]

#def get_weights(graph, path, key):
#    return [graph[path[i]][path[i + 1]][key] for i in range(len(path) - 1)]

GAP_COMPARATOR = lambda x,y: (abs(x - y) < INTERGAP_TOLERANCE) and (x != -1) and (y != -1)

#def GAP_COMPARATOR(x, y):
#    result = (abs(x - y) < INTERGAP_TOLERANCE) and (x != -1) and (y != -1)
#    print(f"{x} {y} {result}")
#    return result

def all_weighted_paired_simple_paths(
    G: nx.DiGraph,
    H: nx.DiGraph,
    weight_key,
    weight_comparator
):
    G_sinks = get_sinks(G)
    G_sources = get_sources(G)
    H_sinks = get_sinks(H_graph)
    H_sources = get_sources(H_graph)
    return itertools.chain.from_iterable(
            mirror.spectrum_graphs.weighted_paired_simple_paths(
                G, 
                g_source, 
                set(G_sinks),
                H, 
                h_source, 
                set(H_sinks),
                mirror.spectrum_graphs.GAP_KEY,
                mirror.spectrum_graphs.GAP_COMPARATOR) 
            for h_source in H_sources 
            for g_source in G_sources)

def weighted_paired_simple_paths(
    G: nx.DiGraph,
    g_source,
    g_target,
    H: nx.DiGraph,
    h_source,
    h_target,
    weight_key,
    weight_comparator
):
    edge_paths = weighted_paired_simple_edge_paths(
        G,
        g_source,
        g_target,
        H,
        h_source,
        h_target,
        weight_key,
        weight_comparator
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
    weight_key,
    weight_comparator
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
        weight_key,
        weight_comparator
    )

def _weighted_paired_simple_edge_paths(
    G: nx.DiGraph,
    g_source,
    g_targets: set,
    H: nx.DiGraph,
    h_source,
    h_targets: set,
    weight_key,
    weight_comparator
):
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
                if weight_comparator(g_w, h_w)]
            stack.append(iter(edge_itx))

#=============================================================================#

def extend_truncated_paths(
    paired_paths,
    G: nx.DiGraph,
    H: nx.DiGraph,
):
    G_sinks = get_sinks(G)
    H_sinks = get_sinks(H)
    for paired_path in paired_paths:
        G_target, H_target = paired_path[-1]
        G_terminated = (G_target in G_sinks)
        H_terminated = (H_target in H_sinks)
        if G_terminated and H_terminated:
            yield paired_path
        if not G_terminated:
            extensions = nx.algorithms.simple_paths.all_simple_paths(G, G_target, G_sinks)
            for path_extension in extensions:
                yield paired_path + [(node, -1) for node in path_extension[1:]]
        elif not H_terminated:
            extensions = nx.algorithms.simple_paths.all_simple_paths(H, H_target, H_sinks)
            for path_extension in extensions:
                yield paired_path + [(-1, node) for node in path_extension[1:]]

#=============================================================================#

def find_mirrored_paths(
    spectrum,
    pivot: Pivot
):
    ascending_graph, descending_graph = construct_spectrum_graphs(spectrum, pivot, GAP_KEY)
    
    mirrored_paths = all_weighted_paired_simple_paths(ascending_graph, descending_graph, GAP_KEY, GAP_COMPARATOR)
    
    extended_paths = extend_truncated_paths(mirrored_paths, ascending_graph, descending_graph)

    return list(extended_paths)