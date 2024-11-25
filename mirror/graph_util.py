import itertools

import networkx as nx

#=============================================================================#
# based on the implementation in networkx.algorithms.simple_paths
# https://github.com/networkx/networkx/blob/main/networkx/algorithms/simple_paths.py

def get_sources(D: nx.DiGraph):
    return [i for i in D.nodes if D.in_degree(i) == 0 if i != -1]

def get_sinks(D: nx.DiGraph):
    return [i for i in D.nodes if D.out_degree(i) == 1 if i != -1]

def get_edges(graph,node):
    return [i for i in graph.edges(node) if i != -1]

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
                weight_key, #mirror.spectrum_graphs.GAP_KEY,
                weight_comparator #mirror.spectrum_graphs.GAP_COMPARATOR
            ) 
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

def edge_disjoint_paths_naiive(
    paths
):
    pass

def edge_disjoint_paths_hashed(
    paths
):
    pass
