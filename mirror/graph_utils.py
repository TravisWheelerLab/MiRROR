import itertools

import networkx as nx
import numpy as np

from .util import disjoint_pairs

#=============================================================================#

Edge = tuple[int,int]
SingularPath = list[int]
DualPath = list[tuple[int,int]]
GraphPair = tuple[nx.DiGraph, nx.DiGraph]

def get_sources(D: nx.DiGraph):
    return [i for i in D.nodes if D.in_degree(i) == 0 if i != -1]

def get_sinks(D: nx.DiGraph):
    return [i for i in D.nodes if D.out_degree(i) == 1 if i != -1]

def get_nontrivial_sinks_and_sources(D: nx.DiGraph):
    sinks = set(get_sinks(D))
    sources = set(get_sources(D))
    return sinks.difference(sources), sources.difference(sinks)

def get_edges(graph,node):
    return [i for i in graph.edges(node) if i != -1]

#=============================================================================#
# methods for iterating the mutual path space of a pair of graphs.
# based on the implementation in networkx.algorithms.simple_paths
# https://github.com/networkx/networkx/blob/main/networkx/algorithms/simple_paths.py

def find_dual_paths(
    G: nx.DiGraph,
    H: nx.DiGraph,
    weight_key,
    weight_metric,
) -> list[DualPath]:
    return list(all_weighted_dual_simple_paths(G,H,weight_key,weight_metric))

def all_nontrivial_weighted_dual_simple_paths(
    G: nx.DiGraph,
    H: nx.DiGraph,
    weight_key,
    weight_metric
):
    G_sinks, G_sources = get_nontrivial_sinks_and_sources(G)
    H_sinks, H_sources = get_nontrivial_sinks_and_sources(H)
    return itertools.chain.from_iterable(
            weighted_dual_simple_paths(
                G, 
                g_source, 
                G_sinks,
                H, 
                h_source, 
                H_sinks,
                weight_key,
                weight_metric
            ) 
            for h_source in H_sources 
            for g_source in G_sources)

def all_weighted_dual_simple_paths(
    G: nx.DiGraph,
    H: nx.DiGraph,
    weight_key,
    weight_metric
):
    G_sinks = set(get_sinks(G))
    G_sources = get_sources(G)
    H_sinks = set(get_sinks(H))
    H_sources = get_sources(H)
    return itertools.chain.from_iterable(
            weighted_dual_simple_paths(
                G, 
                g_source, 
                G_sinks,
                H, 
                h_source, 
                H_sinks,
                weight_key,
                weight_metric
            ) 
            for h_source in H_sources 
            for g_source in G_sources)

def weighted_dual_simple_paths(
    G: nx.DiGraph,
    g_source,
    g_target,
    H: nx.DiGraph,
    h_source,
    h_target,
    weight_key,
    weight_metric
):
    edge_paths = weighted_dual_simple_edge_paths(
        G,
        g_source,
        g_target,
        H,
        h_source,
        h_target,
        weight_key,
        weight_metric
    )
    for edge_path in edge_paths:
        yield [(g_source,h_source)] + [edge[1] for edge in edge_path]

def weighted_dual_simple_edge_paths(
    G: nx.DiGraph,
    g_source,
    g_target,
    H: nx.DiGraph,
    h_source,
    h_target,
    weight_key,
    weight_metric
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

    yield from _weighted_dual_simple_edge_paths(
        G,
        g_source,
        g_targets,
        H,
        h_source,
        h_targets,
        weight_key,
        weight_metric
    )

def _weighted_dual_simple_edge_paths(
    G: nx.DiGraph,
    g_source,
    g_targets: set,
    H: nx.DiGraph,
    h_source,
    h_targets: set,
    weight_key,
    weight_metric
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
                if weight_metric(g_w, h_w)]
            stack.append(iter(edge_itx))

#=============================================================================#
# find and repair prematurely-truncated dual paths. 
# a path is said to be "extended" if its suffix takes 
# the form [..., (vₙ₋ₖ, -1), (vₙ₋ₖ₊₁, -1), (vₙ₋ₖ₊₂, -1), (vₙ₋ₖ₊₃, -1), ... , (vₙ, -1)]
# or [..., (-1, wₙ₋ₖ), (-1, wₙ₋ₖ₊₁), (-1, wₙ₋ₖ₊₂), (-1, wₙ₋ₖ₊₃), ... , (-1, wₙ)]
# indicating that the path is supported by one spectrum graph but not the other.

def find_extended_paths(
    dual_paths,
    G: nx.DiGraph,
    H: nx.DiGraph,
) -> list[DualPath]:
    return list(extend_truncated_paths(dual_paths, G, H))

def extend_truncated_paths(
    dual_paths,
    G: nx.DiGraph,
    H: nx.DiGraph,
):
    "identifies and extends dual paths that end on a non-sink vertex in one of their supporting graphs."
    G_sinks = get_sinks(G)
    H_sinks = get_sinks(H)
    for dual_path in dual_paths:
        G_target, H_target = dual_path[-1]
        G_terminated = (G_target in G_sinks)
        H_terminated = (H_target in H_sinks)
        if G_terminated and H_terminated:
            continue
        if not G_terminated:
            extensions = list(nx.algorithms.simple_paths.all_simple_paths(G, G_target, G_sinks))
            for path_extension in extensions:
                yield dual_path + [(node, -1) for node in path_extension[1:]]
        elif not H_terminated:
            extensions = list(nx.algorithms.simple_paths.all_simple_paths(H, H_target, H_sinks))
            for path_extension in extensions:
                yield dual_path + [(-1, node) for node in path_extension[1:]]
        else:
            raise ValueError("dually-truncated paths cannot be passed to this function.")

#=============================================================================#
# find pairs of paths that do not share edges, but may share vertices.

def find_edge_disjoint_dual_path_pairs(
    dual_paths: list[DualPath],
) -> list[tuple[int,int]]:
    "associates between paths that do not share any edges."
    path_edge_sets = list(map(dual_path_to_edge_set, dual_paths))
    return disjoint_pairs(path_edge_sets, mode)

def dual_path_to_edge_set(
    dual_path: DualPath,
) -> set[Edge]:
    "transforms a dual path into the set containing all its edges in both graphs."
    path1, path2 = unzip_dual_path(dual_path)
    return set(path_to_edges(path1)).union(set(path_to_edges(path2)))

def unzip_dual_path(
    dual_path: DualPath,
) -> tuple[SingularPath,SingularPath]:
    "transforms a dual path [(x1,y1),(x2,y2),...] into a pair of paths ([x1,x2,...],[y1,y2,...])"
    return list(zip(*dual_path))

def path_to_edges(
    path: SingularPath,
) -> list[Edge]:
    n = len(path)
    return [(path[i], path[i + 1]) for i in range(n - 1)]

def dual_path_to_matrix(
    dual_path: DualPath,
) -> np.ndarray:
    "transforms a dual path of length n into an (n,2) numpy array"
    return np.array(dual_path)