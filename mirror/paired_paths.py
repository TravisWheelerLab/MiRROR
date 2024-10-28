# based on the implementation and interface in `networkx.algorithms.simple_paths`
# https://github.com/networkx/networkx/blob/main/networkx/algorithms/simple_paths.py
import networkx as nx

def weighted_paired_simple_paths(
    G: nx.DiGraph,
    g_source,
    g_target,
    H: nx.DiGraph,
    h_source,
    h_target,
    weight_key,
    weight_transform
):
    edge_paths = weighted_paired_simple_edge_paths(
        G,
        g_source,
        g_target,
        H,
        h_source,
        h_target,
        weight_key,
        weight_transform
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
    weight_transform
):
    def get_edges(graph,node):
        return graph.edges(node)

    def get_weights(graph,edges):
        return [weight_transform(graph[s][t][weight_key]) for (s,t) in edges]
    
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
        get_edges,
        get_weights
    )


def _weighted_paired_simple_edge_paths(
    G: nx.DiGraph,
    g_source,
    g_targets: set,
    H: nx.DiGraph,
    h_source,
    h_targets: set,
    get_edges,
    get_weights,
):
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

