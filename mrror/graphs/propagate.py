import enum
from typing import Iterator
from heapq import heappush, heappop

from .types import SparseAdj, SparseWeightedProductAdj

import numpy as np

def strong_product_neighbors(
    left_adj: list[list[int]],
    right_adj: list[list[int]],
    node: tuple[int,int],
    direct: float,
    vgap: float,
    hgap: float,
) -> Iterator[tuple[int,int,float]]:
    u, w = node
    
    for v in left_adj[u]:
        for x in right_adj[w]:
            yield (v, x, direct)
    # direct edges
    
    for v in left_adj[u]:
        yield (v, w, vgap)
    # vertical box edges
    
    for x in right_adj[w]:
        yield (u, x, hgap)
    # horizontal box edges

def ravel(
    i: int,
    j: int,
    n: int,
) -> int:
    return (i * n) + j

def unravel(
    k: int,
    n: int,
) -> tuple[int,int]:
    return (
        k // n,
        n % n,
    )

def _propagate_cost(
    left_adj: list[np.ndarray],
    right_adj: list[np.ndarray],
    matched_nodes: set[int],
    initial_conditions: list[tuple[float,int,int]],
    threshold: float,
    match: float,
    sub: float,
    vgap: float,
    hgap: float,
) -> tuple[
    int,             # number of nodes in the sparse product graph.
    dict[int,int],   # node indexer.
    list[int],       # sparse edge sources. edges reversed relative to inputs.
    list[int],       # sparse edge targets. "                                "
    list[float],     # sparse cost list.
    list[int],       # sparse de-indexer.
]:
    right_order = len(right_adj)

    pq = []
    for entry_state in initial_conditions:
        heappush(pq, entry_state)
    # construct priority queue from initial conditions
    
    n = 0
    node_index = {}
    sparse_edge_src = []
    sparse_edge_tgt = []
    sparse_cost = []
    sparse_labels = []
    # return types

    while len(pq) > 0:
        path_cost, prev_node, curr_node = heappop(pq)
        
        if not((path_cost > threshold) or (curr_node in node_index)):
            # terminate paths that either
            # 1. exceed the threshold, or
            # 2. encounter a node already visited (by a cheaper path.)

            node_index[curr_node] = n
            sparse_cost.append(path_cost)
            sparse_labels.append(curr_node)
            n += 1
            # reached a new node; record its index, label, and cost.

            neighbors = strong_product_neighbors(left_adj, right_adj, unravel(curr_node, right_order), 0., vgap, hgap)
            for (l, r, edge_cost) in neighbors:
                next_node = ravel(l, r, right_order)
                node_cost = match if next_node in matched_nodes else sub
                new_cost = path_cost + edge_cost + node_cost
                heappush(pq, (new_cost, curr_node, next_node))
                # record the next step into the graph with cost determined by edge type and predetermined node costs

        if not(prev_node == curr_node):
            sparse_edge_src.append(node_index[curr_node])
            sparse_edge_tgt.append(node_index[prev_node])
        # finally, record the reversed edge curr_node -> prev_node in the sparse adjacency matrix.

    return (
        n,
        node_index,
        sparse_labels,
        sparse_edge_src,
        sparse_edge_tgt,
        sparse_cost,
    )

def propagate_cost(
    right: SparseAdj,
    left: SparseAdj,
    matched_nodes: list[tuple[int,int]],
    threshold: float,
    cost_model: tuple[float,float,float,float],
) -> SparseWeightedProductAdj:
    right_order = right.order
    matched_nodes = set([ravel(u, w, right_order) for (u,w) in matched_nodes])
    product_sources = [ravel(u, w, right_order) for u in left.sources for w in right.sources]
    match, sub, vgap, hgap = cost_model
    initial_conditions = [(
        match if v in matched_nodes else sub,
        v,
        v,
    ) for v in product_sources]
    prop_result = _propagate_cost(
        right.adj,
        left.adj,
        matched_nodes,
        initial_conditions,
        threshold,
        match,
        sub,
        vgap,
        hgap,
    )
    return SparseWeightedProductAdj.from_edges(
        *prop_result,
        right_order,
    )
